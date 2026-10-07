import * as React from 'react';
import {
  fetchDetections,
  type Detection,
} from '../../api/client';

interface UseDetectionsResult {
  detections: Detection[];
  latestCount: number;
  currentType3: number;
  currentType3b: number;
}

export function useDetectionsPolling(
  intervalMs = 2000,
): UseDetectionsResult {
  const [detections, setDetections] = React.useState<
    Detection[]
  >([]);
  const [currentType3, setCurrentType3] =
    React.useState(0);
  const [currentType3b, setCurrentType3b] =
    React.useState(0);

  React.useEffect(() => {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    let inFlight = false;

    async function tick() {
      if (controller.signal.aborted || document.hidden || inFlight) return;
      inFlight = true;
      try {
        const data = await fetchDetections(controller.signal);
        if (controller.signal.aborted) return;
        const newDetections = data.detections ?? [];
        setDetections(newDetections);
        const type3 = newDetections.filter(
          (d) => d.class === 'type3' || d.class_id === 0,
        ).length;
        const type3b = newDetections.filter(
          (d) => d.class === 'type3b' || d.class_id === 1,
        ).length;
        setCurrentType3(type3);
        setCurrentType3b(type3b);
      } catch {
        // Keep the last valid source-anchored detections.
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
  }, [intervalMs]);

  return {
    detections,
    latestCount: detections.length,
    currentType3,
    currentType3b,
  };
}
