import * as React from 'react';
import type { Detection } from '../../api/client';
import { subscribeSpectrum } from './spectrumRenderer';
import { detectionGeometry } from './detectionGeometry';
import {
  FREQUENCY_BINS, TIME_COLUMNS, TIME_WINDOW_SECONDS, COLUMN_SECONDS,
  type Polarization,
} from './spectrumRendering';

export interface SpectrumCanvasHandle {
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
  let latestSlot: number | null = null;
  let bootId = '';
  let detections: Detection[] = [];
  let disposed = false;
  const columnFor = (slot: number) => ((slot % TIME_COLUMNS) + TIME_COLUMNS) % TIME_COLUMNS;

  function paint() {
    if (disposed) return;
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

    const tickCount = plotWidth < 480 ? 2 : 4;
    const displayEnd = latestSlot === null ? null : (latestSlot + 1) * COLUMN_SECONDS;
    for (let tick = 0; tick <= tickCount; tick += 1) {
      const x = left + plotWidth * tick / tickCount;
      const ago = TIME_WINDOW_SECONDS * (1 - tick / tickCount);
      const label = displayEnd === null ? `−${ago}s`
        : new Date((displayEnd - ago) * 1000).toISOString().slice(11, 19) + (tick === tickCount ? ' UTC' : '');
      ctx.textAlign = tick === 0 ? 'left' : tick === tickCount ? 'right' : 'center';
      ctx.fillText(label, x, h - 13);
      ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, top + plotHeight); ctx.stroke();
    }

    if (polarization === 'I' && displayEnd !== null) {
      ctx.save();
      ctx.beginPath(); ctx.rect(left, top, plotWidth, plotHeight); ctx.clip();
      for (const detection of detections) {
        const geometry = detectionGeometry(detection, displayEnd, bootId);
        if (!geometry) continue;
        const boxX = left + geometry.x * plotWidth;
        const boxY = top + geometry.y * plotHeight;
        ctx.strokeStyle = detection.class_id === 0 ? '#f87171' : '#f8fafc';
        ctx.fillStyle = ctx.strokeStyle;
        ctx.lineWidth = 1.5;
        ctx.strokeRect(boxX, boxY, geometry.width * plotWidth, geometry.height * plotHeight);
        ctx.textAlign = 'left';
        ctx.fillText(`${detection.class} ${Math.round(detection.confidence * 100)}%`, boxX + 3, Math.max(top + 12, boxY - 4));
      }
      ctx.restore();
    }
  }

  const unsubscribe = subscribeSpectrum((update) => {
    if (disposed) return;
    if (update.reset) {
      textureContext.fillStyle = '#000';
      textureContext.fillRect(0, 0, TIME_COLUMNS, FREQUENCY_BINS);
    }
    for (const span of update[polarization]) {
      textureContext.putImageData(new ImageData(span.pixels, span.width, FREQUENCY_BINS), span.left, 0);
    }
    latestSlot = update.latestSlot;
    bootId = update.bootId;
    if (update.reset || update[polarization].length) requestPaint(paint);
  });
  const observer = new ResizeObserver(() => requestPaint(paint));
  observer.observe(canvas);
  requestPaint(paint);
  return {
    setDetections(next) {
      if (JSON.stringify(next) === JSON.stringify(detections)) return;
      detections = next.slice();
      requestPaint(paint);
    },
    dispose() {
      disposed = true;
      unsubscribe();
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
      setDetections: (detections) => engineRef.current?.setDetections(detections),
    }), []);
    return (
      <canvas ref={canvasRef} className={className} role="img"
        aria-label={`Stokes ${polarization === 'I' ? 'I in s.f.u.' : 'V / I fraction'} spectrum: frequency against the last five minutes`} />
    );
  },
);
SpectrumCanvas.displayName = 'SpectrumCanvas';
