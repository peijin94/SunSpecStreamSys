import type { SpectrumBatch } from '../../api/spectrumProtocol';
import { SpectrumRaster, type RasterUpdate } from './spectrumRaster';

const listeners = new Set<(update: RasterUpdate) => void>();
let worker: Worker | null = null;
let fallback: SpectrumRaster | null = null;
let fallbackEpoch = 0;
const publish = (update: RasterUpdate | null) => {
  if (update) listeners.forEach(listener => listener(update));
};

export function subscribeSpectrum(listener: (update: RasterUpdate) => void) {
  listeners.add(listener);
  if (!worker && !fallback) {
    if (typeof Worker !== 'undefined') {
      worker = new Worker(new URL('./spectrum.worker.ts', import.meta.url), { type: 'module' });
      worker.onmessage = (event: MessageEvent<RasterUpdate>) => publish(event.data);
    } else fallback = new SpectrumRaster();
  }
  return () => {
    listeners.delete(listener);
    if (!listeners.size) { worker?.terminate(); worker = null; fallback = null; }
  };
}

export function submitSpectrumBatch(batch: SpectrumBatch) {
  if (worker) {
    const buffers = new Set<ArrayBuffer>();
    for (const frame of [...batch.I, ...batch.V]) {
      if (ArrayBuffer.isView(frame) && frame.buffer instanceof ArrayBuffer) buffers.add(frame.buffer);
    }
    worker.postMessage(batch, [...buffers]);
  } else if (fallback) {
    // Yield between short chunks on browsers without Web Workers.
    const renderer = fallback;
    if (batch.reset) fallbackEpoch++;
    const epoch = fallbackEpoch;
    let begin = 0;
    const next = () => {
      if (renderer !== fallback || epoch !== fallbackEpoch) return;
      const end = Math.min(begin + 16, batch.I.length);
      publish(renderer.update({ ...batch, reset: batch.reset && begin === 0,
        I: batch.I.slice(begin, end), V: batch.V.slice(begin, end), timestamps: batch.timestamps.slice(begin, end),
        first_sequence: batch.first_sequence === null ? null : batch.first_sequence + begin,
        previous_timestamp: begin ? batch.timestamps[begin - 1] : batch.previous_timestamp }));
      begin = end;
      if (begin < batch.I.length) setTimeout(next, 0);
    };
    setTimeout(next, 0);
  }
}
