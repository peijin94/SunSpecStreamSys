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
  intervalMs = 8000,
): UseDetectionsResult {
  const [detections, setDetections] = React.useState<
    Detection[]
  >([]);
  const [currentType3, setCurrentType3] =
    React.useState(0);
  const [currentType3b, setCurrentType3b] =
    React.useState(0);

  React.useEffect(() => {
    let cancelled = false;

    async function tick() {
      try {
        const data = await fetchDetections();
        if (cancelled) return;
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
        // ignore errors for now
      }
    }

    tick();
    const id = window.setInterval(tick, intervalMs);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [intervalMs]);

  return {
    detections,
    latestCount: detections.length,
    currentType3,
    currentType3b,
  };
}

