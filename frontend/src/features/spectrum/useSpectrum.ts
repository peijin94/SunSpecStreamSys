import * as React from 'react';
import { fetchSpectrum, fetchSpectrumHistory, type SpectrumBatch, type SpectrumCursor } from '../../api/client';

export function useSpectrumPolling(onBatch: (data: SpectrumBatch) => void, intervalMs = 512) {
  const [status, setStatus] = React.useState<'loading' | 'error' | 'ok'>('loading');
  const [frameCount, setFrameCount] = React.useState(0);
  const [lastUpdate, setLastUpdate] = React.useState<string | null>(null);
  const [historyLoading, setHistoryLoading] = React.useState(false);
  const [refreshing, setRefreshing] = React.useState(false);
  const refreshRef = React.useRef<() => void>(() => {});

  React.useEffect(() => {
    const controller = new AbortController();
    let historyController = new AbortController();
    let cursor: SpectrumCursor | undefined;
    let older: { boot: string; before: number } | null = null;
    let generation = 0;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let inFlight = false;
    let historyInFlight = false;
    let urgent = false;

    async function loadHistory() {
      if (!older || historyInFlight || document.hidden || controller.signal.aborted) return;
      historyInFlight = true;
      const session = generation;
      const signal = historyController.signal;
      setHistoryLoading(true);
      try {
        while (older && !signal.aborted && !document.hidden && session === generation) {
          const page = await fetchSpectrumHistory(older.boot, older.before, signal);
          if (signal.aborted || session !== generation || document.hidden) break;
          if (page.boot_id !== older.boot || page.reset) { older = null; break; }
          if (page.I.length) onBatch(page);
          older = page.has_more && page.next_before !== null && page.next_before !== undefined
            ? { boot: page.boot_id, before: page.next_before } : null;
          // Give live requests and the renderer priority between small history pages.
          await new Promise(resolve => setTimeout(resolve, 20));
        }
      } catch {
        // Retain the page cursor for retry; live updates remain independent.
      } finally {
        if (session === generation) {
          historyInFlight = false;
          if (!controller.signal.aborted) setHistoryLoading(false);
        }
      }
    }

    async function tick() {
      if (controller.signal.aborted || inFlight || document.hidden) return;
      inFlight = true;
      urgent = false;
      try {
        const data = await fetchSpectrum(cursor, controller.signal);
        if (controller.signal.aborted) return;
        cursor = { sequence: data.sequence, boot_id: data.boot_id };
        if (data.reset) {
          generation++;
          historyController.abort();
          historyController = new AbortController();
          historyInFlight = false;
          older = data.has_more && data.next_before !== null && data.next_before !== undefined
            ? { boot: data.boot_id, before: data.next_before } : null;
          setHistoryLoading(!!older);
        }
        setStatus('ok');
        if (data.reset || data.I.length) onBatch(data);
        if (data.I.length) {
          setFrameCount(n => n + data.I.length);
          const latest = data.timestamps[data.timestamps.length - 1];
          setLastUpdate(new Date(latest * 1000).toISOString().slice(11, 19) + ' (UTC)');
        }
        void loadHistory();
      } catch {
        if (!controller.signal.aborted) setStatus('error');
      } finally {
        inFlight = false;
        if (!controller.signal.aborted) {
          setRefreshing(false);
          timer = setTimeout(tick, urgent ? 0 : intervalMs);
        }
      }
    }

    refreshRef.current = () => {
      setRefreshing(true);
      if (timer) clearTimeout(timer);
      if (inFlight) urgent = true;
      else void tick();
    };
    function onVisibilityChange() {
      if (timer) clearTimeout(timer);
      if (!document.hidden) { void tick(); void loadHistory(); }
    }
    void tick();
    document.addEventListener('visibilitychange', onVisibilityChange);
    return () => {
      controller.abort(); historyController.abort();
      if (timer) clearTimeout(timer);
      document.removeEventListener('visibilitychange', onVisibilityChange);
      refreshRef.current = () => {};
    };
  }, [intervalMs, onBatch]);

  return { status, frameCount, lastUpdate, historyLoading, refreshing,
    refresh: () => refreshRef.current() };
}
