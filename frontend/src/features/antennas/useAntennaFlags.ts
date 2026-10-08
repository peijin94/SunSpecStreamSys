import * as React from 'react';
import { fetchAntennaFlags } from '../../api/client';
import type { AntennaFlagsResponse } from '../../api/antennaProtocol';

export function useAntennaFlags(intervalMs = 60000) {
  const [state, setState] = React.useState<{data: AntennaFlagsResponse | null; error: string | null}>({data: null, error: null});
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
        const data = await fetchAntennaFlags(request.signal);
        if (!controller.signal.aborted) setState(previous => data.bands.length || !previous.data?.bands.length
          ? {data, error: data.message}
          : {...previous, error: data.message || 'Flagging unavailable; last observation shown.'});
      } catch {
        if (!controller.signal.aborted) setState(previous => ({...previous, error: 'Flagging temporarily unavailable; retrying every minute.'}));
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
