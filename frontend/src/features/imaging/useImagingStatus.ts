import * as React from 'react';
import { fetchImagingStatus } from '../../api/client';
import type { ImagingStatusResponse } from '../../api/imagingProtocol';

export function useImagingStatus(intervalMs = 60000) {
  const [state, setState] = React.useState<{data: ImagingStatusResponse | null; error: string | null}>({data: null, error: null});
  React.useEffect(() => {
    const controller = new AbortController();
    let inFlight = false, lastPoll = -Infinity;
    async function tick() {
      if (inFlight || document.hidden || controller.signal.aborted) return;
      inFlight = true;
      lastPoll = Date.now();
      const request = new AbortController();
      const abort = () => request.abort();
      controller.signal.addEventListener('abort', abort);
      const timeout = window.setTimeout(abort, 15000);
      try {
        const data = await fetchImagingStatus(request.signal);
        if (!controller.signal.aborted) setState(previous => data.bands.length || !previous.data?.bands.length
          ? {data, error: data.message}
          : {...previous, error: data.message || 'Imaging QA unavailable; last observation shown.'});
      } catch {
        if (!controller.signal.aborted) setState(previous => ({...previous, error: 'Imaging QA temporarily unavailable; retrying every minute.'}));
      } finally {
        window.clearTimeout(timeout);
        controller.signal.removeEventListener('abort', abort);
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
      controller.abort(); window.clearInterval(timer);
      document.removeEventListener('visibilitychange', onVisibilityChange);
    };
  }, [intervalMs]);
  return state;
}
