import type { Detection } from '../../api/detectionProtocol.js';
import { TIME_WINDOW_SECONDS } from './spectrumRendering.js';

export function detectionGeometry(detection: Detection, displayEnd: number, bootId: string) {
  if (detection.boot_id && detection.boot_id !== bootId) return null;
  let [x, y, width, height] = detection.bbox;
  if (detection.time_start !== undefined && detection.time_end !== undefined) {
    x = (detection.time_start - (displayEnd - TIME_WINDOW_SECONDS)) / TIME_WINDOW_SECONDS;
    width = (detection.time_end - detection.time_start) / TIME_WINDOW_SECONDS;
  } else {
    x -= (displayEnd - detection.timestamp) / TIME_WINDOW_SECONDS;
  }
  if (height < 0) { y += height; height = -height; }
  if (![x, y, width, height].every(Number.isFinite) || width <= 0 || height <= 0 ||
      x + width <= 0 || x >= 1) return null;
  return { x, y, width, height };
}
