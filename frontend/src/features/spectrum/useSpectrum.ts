import * as React from 'react';
import { fetchSpectrum, type SpectrumBatch, type SpectrumCursor } from '../../api/client';

export function useSpectrumPolling(intervalMs = 512) {
  const [status, setStatus] = React.useState<'loading' | 'error' | 'ok'>('loading');
  const [frameCount, setFrameCount] = React.useState(0);
  const [lastUpdate, setLastUpdate] = React.useState<string | null>(null);
  const [batch, setBatch] = React.useState<SpectrumBatch | null>(null);
  const [refreshVersion, requestRefresh] = React.useReducer((n: number) => n + 1, 0);

  React.useEffect(() => {
    const controller = new AbortController();
    let cursor: SpectrumCursor | undefined;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let inFlight = false;

    async function tick() {
      if (controller.signal.aborted || inFlight || document.hidden) return;
      inFlight = true;
      try {
        const data = await fetchSpectrum(cursor, controller.signal);
        if (controller.signal.aborted) return;
        cursor = { sequence: data.sequence, boot_id: data.boot_id };
        setStatus('ok');
        if (data.reset || data.I.length) setBatch(data);
        if (data.I.length) {
          setFrameCount((n) => n + data.I.length);
          const latest = data.timestamps[data.timestamps.length - 1];
          setLastUpdate(new Date(latest * 1000).toISOString().slice(11, 19) + ' (UTC)');
        }
      } catch {
        if (!controller.signal.aborted) setStatus('error');
      } finally {
        inFlight = false;
        if (!controller.signal.aborted) timer = setTimeout(tick, intervalMs);
      }
    }

    function onVisibilityChange() {
      if (timer) clearTimeout(timer);
      if (!document.hidden) void tick();
    }

    void tick();
    document.addEventListener('visibilitychange', onVisibilityChange);
    return () => {
      controller.abort();
      if (timer) clearTimeout(timer);
      document.removeEventListener('visibilitychange', onVisibilityChange);
    };
  }, [intervalMs, refreshVersion]);

  return {
    status, frameCount, lastUpdate, batch,
    refresh: () => requestRefresh(),
  };
}
