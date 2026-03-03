import * as React from 'react';
import {
  fetchLatestData,
  refreshSpectrum,
  type SpectrumFrame,
} from '../../api/client';

interface UseSpectrumResult {
  status: 'idle' | 'loading' | 'error' | 'ok';
  frameCount: number;
  lastUpdate: string | null;
  latestFrame: SpectrumFrame | null;
  refresh: () => Promise<SpectrumFrame[] | null>;
}

export function useSpectrumPolling(
  intervalMs = 512,
): UseSpectrumResult {
  const [status, setStatus] = React.useState<
    UseSpectrumResult['status']
  >('idle');
  const [frameCount, setFrameCount] =
    React.useState(0);
  const [lastUpdate, setLastUpdate] =
    React.useState<string | null>(null);
  const [latestFrame, setLatestFrame] =
    React.useState<SpectrumFrame | null>(null);

  React.useEffect(() => {
    let cancelled = false;

    async function tick() {
      try {
        const data = await fetchLatestData();
        if (cancelled) return;
        if (data && data.length > 0) {
          setLatestFrame(data);
          setFrameCount((c) => c + 1);
          const now = new Date();
          const timeStr = `${now
            .getUTCHours()
            .toString()
            .padStart(2, '0')}:${now
            .getUTCMinutes()
            .toString()
            .padStart(2, '0')}:${now
            .getUTCSeconds()
            .toString()
            .padStart(2, '0')} (UTC)`;
          setLastUpdate(timeStr);
          setStatus('ok');
        }
      } catch {
        if (!cancelled) {
          setStatus('error');
        }
      }
    }

    setStatus('loading');
    tick();
    const id = window.setInterval(tick, intervalMs);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [intervalMs]);

  const refresh = React.useCallback(async () => {
    try {
      const data = await refreshSpectrum();
      if (!data.data || !data.data.length) {
        return null;
      }
      return data.data;
    } catch {
      return null;
    }
  }, []);

  return {
    status,
    frameCount,
    lastUpdate,
    latestFrame,
    refresh,
  };
}

