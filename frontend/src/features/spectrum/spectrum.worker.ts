import { SpectrumRaster } from './spectrumRaster';
import type { SpectrumBatch } from '../../api/spectrumProtocol';

const raster = new SpectrumRaster();
self.onmessage = (event: MessageEvent<SpectrumBatch>) => {
  const update = raster.update(event.data);
  if (!update) return;
  self.postMessage(update, { transfer: [...update.I, ...update.V].map(span => span.pixels.buffer) });
};
