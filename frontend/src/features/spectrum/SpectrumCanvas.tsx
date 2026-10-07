import * as React from 'react';
import type { Detection, PolarizedFrame } from '../../api/client';
import {
  COLOR_TABLES, colorIndex, timeSlot, firstCoveredSlot,
  FREQUENCY_BINS, TIME_COLUMNS, TIME_WINDOW_SECONDS,
  type Polarization,
} from './spectrumRendering';

export interface SpectrumCanvasHandle {
  updateFrames: (frames: PolarizedFrame[], timestamps: number[], reset: boolean) => void;
  setDetections: (detections: Detection[]) => void;
}

interface SpectrumCanvasProps {
  className?: string;
  polarization: Polarization;
}

interface Engine extends SpectrumCanvasHandle { dispose: () => void }

// One animation callback for all dirty panels, and no redraw loop when idle.
const dirtyPanels = new Set<() => void>();
let animationFrame: number | null = null;
function requestPaint(paint: () => void) {
  dirtyPanels.add(paint);
  if (animationFrame !== null) return;
  animationFrame = requestAnimationFrame(() => {
    animationFrame = null;
    const panels = Array.from(dirtyPanels);
    dirtyPanels.clear();
    panels.forEach((draw) => draw());
  });
}

function createEngine(canvas: HTMLCanvasElement, polarization: Polarization): Engine {
  const ctx = canvas.getContext('2d', { alpha: false })!;
  const texture = document.createElement('canvas');
  texture.width = TIME_COLUMNS;
  texture.height = FREQUENCY_BINS;
  const textureContext = texture.getContext('2d', { alpha: false })!;
  const image = textureContext.createImageData(TIME_COLUMNS, FREQUENCY_BINS);
  const colors = COLOR_TABLES[polarization];
  const dirtyColumns = new Set<number>();
  let fullUpload = true;
  let latestSlot: number | null = null;
  let latestTimestamp: number | null = null;
  let detections: Detection[] = [];
  let disposed = false;
  const columnFor = (slot: number) => ((slot % TIME_COLUMNS) + TIME_COLUMNS) % TIME_COLUMNS;

  function clearColumn(column: number) {
    for (let y = 0; y < FREQUENCY_BINS; y += 1) {
      const offset = (y * TIME_COLUMNS + column) * 4;
      image.data[offset] = 0;
      image.data[offset + 1] = 0;
      image.data[offset + 2] = 0;
      image.data[offset + 3] = 255;
    }
    dirtyColumns.add(column);
  }

  function flushTexture() {
    if (fullUpload || dirtyColumns.size > TIME_COLUMNS / 2) {
      textureContext.putImageData(image, 0, 0);
    } else {
      // Upload adjacent columns together; a wrapped batch needs two short spans.
      const columns = Array.from(dirtyColumns).sort((a, b) => a - b);
      for (let i = 0; i < columns.length;) {
        const first = columns[i];
        let last = first;
        while (i + 1 < columns.length && columns[i + 1] === last + 1) {
          i += 1;
          last = columns[i];
        }
        textureContext.putImageData(image, 0, 0, first, 0, last - first + 1, FREQUENCY_BINS);
        i += 1;
      }
    }
    dirtyColumns.clear();
    fullUpload = false;
  }

  function paint() {
    if (disposed) return;
    flushTexture();
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const width = Math.max(1, Math.round(rect.width * dpr));
    const height = Math.max(1, Math.round(rect.height * dpr));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const w = rect.width;
    const h = rect.height;
    const left = 64;
    const top = 16;
    const plotWidth = Math.max(1, w - left - 16);
    const plotHeight = Math.max(1, h - top - 36);
    ctx.fillStyle = '#020617';
    ctx.fillRect(0, 0, w, h);
    ctx.imageSmoothingEnabled = false;

    const first = latestSlot === null ? 0 : columnFor(latestSlot + 1);
    const leftColumns = TIME_COLUMNS - first;
    ctx.drawImage(texture, first, 0, leftColumns, FREQUENCY_BINS,
      left, top, plotWidth * leftColumns / TIME_COLUMNS, plotHeight);
    if (first) {
      ctx.drawImage(texture, 0, 0, first, FREQUENCY_BINS,
        left + plotWidth * leftColumns / TIME_COLUMNS, top,
        plotWidth * first / TIME_COLUMNS, plotHeight);
    }

    ctx.font = '11px system-ui, sans-serif';
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.15)';
    ctx.fillStyle = '#94a3b8';
    ctx.textAlign = 'right';
    for (const frequency of [15, 25, 35, 45, 55, 65, 75, 85]) {
      const y = top + (85 - frequency) / 70 * plotHeight;
      ctx.fillText(`${frequency}`, left - 10, y + 4);
      ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(left + plotWidth, y); ctx.stroke();
    }
    ctx.save();
    ctx.translate(15, top + plotHeight / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillText('Frequency (MHz)', 0, 0);
    ctx.restore();

    for (let tick = 0; tick <= 4; tick += 1) {
      const x = left + plotWidth * tick / 4;
      ctx.textAlign = tick === 0 ? 'left' : tick === 4 ? 'right' : 'center';
      ctx.fillText(tick === 4 ? 'Latest' : `−${TIME_WINDOW_SECONDS * (1 - tick / 4)}s`, x, h - 13);
      ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, top + plotHeight); ctx.stroke();
    }

    if (polarization === 'I' && latestTimestamp !== null) {
      ctx.save();
      ctx.beginPath(); ctx.rect(left, top, plotWidth, plotHeight); ctx.clip();
      for (const detection of detections) {
        const age = latestTimestamp - detection.timestamp;
        if (age > TIME_WINDOW_SECONDS) continue;
        const [x, y, boxWidth, boxHeight] = detection.bbox;
        const boxX = left + (x - age / TIME_WINDOW_SECONDS) * plotWidth;
        const boxY = top + y * plotHeight;
        ctx.strokeStyle = detection.class_id === 0 ? '#f87171' : '#f8fafc';
        ctx.fillStyle = ctx.strokeStyle;
        ctx.lineWidth = 1.5;
        ctx.strokeRect(boxX, boxY, boxWidth * plotWidth, boxHeight * plotHeight);
        ctx.textAlign = 'left';
        ctx.fillText(`${detection.class} ${Math.round(detection.confidence * 100)}%`, boxX + 3, Math.max(top + 12, boxY - 4));
      }
      ctx.restore();
    }
  }

  const observer = new ResizeObserver(() => requestPaint(paint));
  observer.observe(canvas);
  requestPaint(paint);

  return {
    updateFrames(frames, timestamps, reset) {
      if (reset) {
        image.data.fill(0);
        latestSlot = null;
        latestTimestamp = null;
        fullUpload = true;
      }
      frames.forEach((frame, index) => {
        if (frame.length !== FREQUENCY_BINS || !Number.isFinite(timestamps[index])) return;
        const slot = timeSlot(timestamps[index]);
        if (latestSlot !== null && slot < latestSlot) return;
        const firstCovered = firstCoveredSlot(latestTimestamp, timestamps[index]);
        if (latestSlot !== null && slot - latestSlot >= TIME_COLUMNS) {
          image.data.fill(0);
          fullUpload = true;
        } else if (latestSlot !== null) {
          // Preserve true elapsed time: missing integrations stay black.
          for (let gap = latestSlot + 1; gap < firstCovered; gap += 1) clearColumn(columnFor(gap));
        }
        for (let covered = firstCovered; covered <= slot; covered += 1) {
          const column = columnFor(covered);
          for (let channel = 0; channel < FREQUENCY_BINS; channel += 1) {
            const color = colorIndex(frame[channel], polarization);
            const offset = ((FREQUENCY_BINS - channel - 1) * TIME_COLUMNS + column) * 4;
            image.data[offset] = color < 0 ? 0 : colors[color * 4];
            image.data[offset + 1] = color < 0 ? 0 : colors[color * 4 + 1];
            image.data[offset + 2] = color < 0 ? 0 : colors[color * 4 + 2];
            image.data[offset + 3] = 255;
          }
          dirtyColumns.add(column);
        }
        latestSlot = slot;
        latestTimestamp = timestamps[index];
      });
      if (reset || frames.length) requestPaint(paint);
    },
    setDetections(next) {
      if (JSON.stringify(next) === JSON.stringify(detections)) return;
      detections = next.slice();
      requestPaint(paint);
    },
    dispose() {
      disposed = true;
      observer.disconnect();
      dirtyPanels.delete(paint);
    },
  };
}

export const SpectrumCanvas = React.forwardRef<SpectrumCanvasHandle, SpectrumCanvasProps>(
  ({ className, polarization }, ref) => {
    const canvasRef = React.useRef<HTMLCanvasElement | null>(null);
    const engineRef = React.useRef<Engine | null>(null);
    React.useEffect(() => {
      if (!canvasRef.current) return;
      const engine = createEngine(canvasRef.current, polarization);
      engineRef.current = engine;
      return () => { engine.dispose(); engineRef.current = null; };
    }, [polarization]);
    React.useImperativeHandle(ref, () => ({
      updateFrames: (frames, times, reset) => engineRef.current?.updateFrames(frames, times, reset),
      setDetections: (detections) => engineRef.current?.setDetections(detections),
    }), []);
    return (
      <canvas ref={canvasRef} className={className} role="img"
        aria-label={`Stokes ${polarization} spectrum: frequency against the last five minutes`} />
    );
  },
);
SpectrumCanvas.displayName = 'SpectrumCanvas';
