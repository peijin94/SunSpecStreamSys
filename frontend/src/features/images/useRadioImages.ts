import * as React from 'react';
import { fetchRadioImages, radioImageUrl, type RadioImage, type RadioImagesResponse } from '../../api/client';

interface LoadedImage extends RadioImage { src: string }
interface ImageState {
  metadata: RadioImagesResponse | null;
  images: LoadedImage[];
  error: string | null;
}

async function decodeImage(url: string) {
  const image = new Image();
  image.src = url;
  await image.decode();
}

export function useRadioImages(intervalMs = 60000) {
  const [state, setState] = React.useState<ImageState>({ metadata: null, images: [], error: null });
  React.useEffect(() => {
    const controller = new AbortController();
    let inFlight = false;
    let version: string | null = null;
    let loaded: LoadedImage[] = [];
    let lastPoll = -Infinity;

    async function tick() {
      if (inFlight || document.hidden || controller.signal.aborted) return;
      inFlight = true;
      lastPoll = Date.now();
      const request = new AbortController();
      const abort = () => request.abort();
      controller.signal.addEventListener('abort', abort);
      const timeout = window.setTimeout(abort, 15000);
      const pending: string[] = [];
      try {
        const metadata = await fetchRadioImages(request.signal);
        if (metadata.images.length && metadata.version !== version) {
          if (metadata.images.length !== 4 || metadata.images.map(x => x.target_mhz).sort((a, b) => a - b).join(',') !== '30,45,60,75') {
            throw new Error('Incomplete image observation');
          }
          const downloads = await Promise.allSettled(metadata.images.map(async item => {
            const response = await fetch(radioImageUrl(item.image_url), { signal: request.signal });
            if (!response.ok) throw new Error('Image download failed');
            const src = URL.createObjectURL(await response.blob());
            pending.push(src);
            await decodeImage(src);
            return { ...item, src };
          }));
          const images = downloads.map(result => {
            if (result.status !== 'fulfilled') throw new Error('Incomplete image download');
            return result.value;
          });
          if (request.signal.aborted) throw new Error('Image load cancelled');
          loaded = images;
          version = metadata.version;
        }
        if (!controller.signal.aborted) {
          setState(previous => metadata.images.length || !loaded.length
            ? { metadata, images: loaded, error: metadata.message }
            : { ...previous, error: metadata.message || 'Waiting for new images. Last available observation is shown.' });
          pending.length = 0;
        }
      } catch {
        if (!controller.signal.aborted) setState(previous => ({ ...previous,
          error: 'Images temporarily unavailable. Retrying every minute.' }));
      } finally {
        pending.forEach(url => URL.revokeObjectURL(url));
        controller.signal.removeEventListener('abort', abort);
        window.clearTimeout(timeout);
        inFlight = false;
      }
    }

    function onVisibilityChange() {
      if (!document.hidden && Date.now() - lastPoll >= intervalMs) void tick();
    }
    void tick();
    const timer = window.setInterval(tick, intervalMs);
    document.addEventListener('visibilitychange', onVisibilityChange);
    return () => {
      controller.abort();
      window.clearInterval(timer);
      document.removeEventListener('visibilitychange', onVisibilityChange);
    };
  }, [intervalMs]);

  React.useEffect(() => () => {
    state.images.forEach(image => URL.revokeObjectURL(image.src));
  }, [state.images]);
  return state;
}
