import * as React from 'react';
import type { SpectrumFrame } from '../../api/client';
import type { Detection } from '../../api/client';

export interface SpectrumCanvasHandle {
  pushFrame: (frame: SpectrumFrame) => void;
  loadFrames: (frames: SpectrumFrame[]) => void;
  setDetections: (detections: Detection[]) => void;
}

interface SpectrumCanvasProps {
  className?: string;
}

interface Engine {
  pushFrame: (frame: SpectrumFrame) => void;
  loadFrames: (frames: SpectrumFrame[]) => void;
  setDetections: (detections: Detection[]) => void;
  dispose: () => void;
}

function createEngine(canvas: HTMLCanvasElement): Engine {
  const ctx = canvas.getContext('2d', {
    alpha: false,
    desynchronized: true,
  })!;

  const tex = document.createElement('canvas');
  const tctx = tex.getContext('2d', {
    alpha: false,
    desynchronized: true,
  })!;

  const bins = 768;
  const heightLines = 600;
  let writeX = heightLines - 1;
  let frameCount = 0;
  let running = true;

  let detectionHistory: Detection[] = [];

  let columnImage: ImageData | null = null;

  let dataMin = 1e4;
  let dataMax = 1e8;
  const freqMin = 15;
  const freqMax = 85;

  function recreateTexture() {
    tex.width = heightLines;
    tex.height = bins;
    tctx.fillStyle = '#000';
    tctx.fillRect(0, 0, tex.width, tex.height);
    columnImage = tctx.createImageData(1, bins);
    writeX = heightLines - 1;
  }

  function toIdx(v: number) {
    if (v < dataMin) v = dataMin;
    if (v > dataMax) v = dataMax;
    const logV = Math.log(v) / Math.LN10;
    const t =
      (logV - Math.log(dataMin) / Math.LN10) /
      ((Math.log(dataMax) / Math.LN10) -
        (Math.log(dataMin) / Math.LN10));
    return t;
  }

  function infernoColormap(intensity: number) {
    const t = intensity;

    if (t < 0.25) {
      const s = t / 0.25;
      return {
        r: Math.round(4 * s * (1 - s) * 255),
        g: 0,
        b: Math.round((1 - 4 * s * (1 - s)) * 255),
      };
    }
    if (t < 0.5) {
      const s = (t - 0.25) / 0.25;
      return {
        r: 0,
        g: Math.round(s * 255),
        b: Math.round((1 - s) * 255),
      };
    }
    if (t < 0.75) {
      const s = (t - 0.5) / 0.25;
      return {
        r: Math.round(s * 255),
        g: Math.round(255),
        b: Math.round((1 - s) * 255),
      };
    }
    const s = (t - 0.75) / 0.25;
    return {
      r: Math.round(255),
      g: Math.round(255),
      b: Math.round(s * 255),
    };
  }

  function writeRow(dataArray: SpectrumFrame) {
    if (!columnImage) return;
    if (!dataArray || dataArray.length !== bins) return;

    const columnData = columnImage.data;

    for (let y = 0; y < bins; y += 1) {
      const flippedY = bins - 1 - y;
      const intensity = toIdx(dataArray[y]);
      const colors = infernoColormap(intensity);
      const q = flippedY * 4;
      columnData[q] = colors.r;
      columnData[q + 1] = colors.g;
      columnData[q + 2] = colors.b;
      columnData[q + 3] = 255;
    }

    tctx.putImageData(columnImage, writeX, 0);
    writeX = (writeX + 1) % heightLines;
    frameCount += 1;
  }

  function drawFrequencyScale(
    canvasWidth: number,
    canvasHeight: number,
  ) {
    ctx.save();
    ctx.fillStyle = '#9fb0c2';
    ctx.font = '18px monospace';
    ctx.textAlign = 'right';

    const frequencies = [15, 25, 35, 45, 55, 65, 75, 85];
    frequencies.forEach((freq) => {
      const freqRatio =
        (freqMax - freq) / (freqMax - freqMin);
      const y =
        50 + freqRatio * (canvasHeight - 120);
      ctx.fillText(`${freq}MHz`, 145, y + 4);

      ctx.strokeStyle = '#222';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(200, y);
      ctx.lineTo(canvasWidth - 40, y);
      ctx.stroke();
    });

    ctx.restore();
  }

  function drawTimeScale(
    canvasWidth: number,
    canvasHeight: number,
  ) {
    ctx.save();
    ctx.fillStyle = '#9fb0c2';
    ctx.font = '18px monospace';
    ctx.textAlign = 'center';

    const timeLabels = ['Now', '-75s', '-150s', '-225s', '-300s'];
    timeLabels.forEach((label, index) => {
      const timeRatio =
        (timeLabels.length - 1 - index) /
        (timeLabels.length - 1);
      const x =
        200 + timeRatio * (canvasWidth - 240);
      ctx.fillText(label, x, canvasHeight - 25);

      ctx.strokeStyle = '#222';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, 50);
      ctx.lineTo(x, canvasHeight - 70);
      ctx.stroke();
    });

    ctx.restore();
  }

  function drawDetections(
    canvasWidth: number,
    canvasHeight: number,
  ) {
    if (!detectionHistory.length) return;

    ctx.save();

    const plotWidth = canvasWidth - 240;
    const plotHeight = canvasHeight - 120;
    const plotX = 200;
    const plotY = 50;

    const timeWindow = 300;
    const currentTime = Date.now() / 1000;

    detectionHistory.forEach((detection) => {
      const [x, y, width, height] = detection.bbox;
      const confidence = detection.confidence;
      const classId = detection.class_id;
      const className = detection.class;
      const detectionTime =
        // prefer last_detection_time style timestamp if present
        (detection as any).detectionTime ??
        detection.timestamp;

      const timeOffset = currentTime - detectionTime;
      if (timeOffset > timeWindow) return;

      const timeRatio = timeOffset / timeWindow;
      const timeX = -timeRatio * plotWidth;

      const canvasX =
        plotX + timeX + x * plotWidth;
      const canvasY =
        plotY + y * plotHeight;
      const canvasWidthBox = width * plotWidth;
      const canvasHeightBox = height * plotHeight;

      if (
        canvasX < plotX ||
        canvasX > plotX + plotWidth
      )
        return;

      let strokeColor: string;
      if (classId === 0) strokeColor = '#ff0000';
      else if (classId === 1) strokeColor = '#ffffff';
      else strokeColor = '#ffff00';

      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = 2;
      ctx.strokeRect(
        canvasX,
        canvasY,
        canvasWidthBox,
        canvasHeightBox,
      );

      ctx.fillStyle = strokeColor;
      ctx.font = '16px monospace';
      ctx.textAlign = 'left';
      ctx.fillText(
        `${className} ${(confidence * 100).toFixed(1)}%`,
        canvasX + 2,
        canvasY - 4,
      );
    });

    ctx.restore();
  }

  function blitToView() {
    const W = canvas.width;
    const H = canvas.height;

    ctx.clearRect(0, 0, W, H);

    const plotWidth = W - 240;
    const plotHeight = H - 120;
    const scaleX = plotWidth / heightLines;

    const leftPortion = heightLines - writeX;
    if (leftPortion > 0) {
      const leftWidth = leftPortion * scaleX;
      ctx.drawImage(
        tex,
        writeX,
        0,
        leftPortion,
        bins,
        200,
        50,
        leftWidth,
        plotHeight,
      );
    }

    if (writeX > 0) {
      const rightWidth = writeX * scaleX;
      const rightX = 200 + leftPortion * scaleX;
      ctx.drawImage(
        tex,
        0,
        0,
        writeX,
        bins,
        rightX,
        50,
        rightWidth,
        plotHeight,
      );
    }

    drawFrequencyScale(W, H);
    drawTimeScale(W, H);
    drawDetections(W, H);
  }

  function resize() {
    const dpr = Math.min(
      2,
      window.devicePixelRatio || 1,
    );
    const rect = canvas.getBoundingClientRect();
    const w = Math.floor(rect.width * dpr);
    const h = Math.floor(rect.height * dpr);
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
  }

  function setDetections(detections: Detection[]) {
    detectionHistory = detections.slice();
  }

  let lastT = performance.now();
  let acc = 0;
  const updateRate = 1 / 0.512;

  function loop(now: number) {
    if (!running) return;
    const dt = Math.min(0.2, (now - lastT) / 1000);
    lastT = now;
    acc += dt * updateRate;

    if (acc >= 1) {
      blitToView();
      acc -= 1;
    }

    requestAnimationFrame(loop);
  }

  recreateTexture();
  resize();
  window.addEventListener('resize', resize);
  requestAnimationFrame(loop);

  return {
    pushFrame: (frame: SpectrumFrame) => {
      writeRow(frame);
    },
    loadFrames: (frames: SpectrumFrame[]) => {
      recreateTexture();
      frames.forEach((f) => writeRow(f));
    },
    setDetections,
    dispose: () => {
      running = false;
      window.removeEventListener('resize', resize);
    },
  };
}

export const SpectrumCanvas = React.forwardRef<
  SpectrumCanvasHandle,
  SpectrumCanvasProps
>(({ className }, ref) => {
  const canvasRef =
    React.useRef<HTMLCanvasElement | null>(null);
  const engineRef = React.useRef<Engine | null>(null);

  React.useEffect(() => {
    if (!canvasRef.current) return;
    const engine = createEngine(canvasRef.current);
    engineRef.current = engine;
    return () => {
      engine.dispose();
      engineRef.current = null;
    };
  }, []);

  React.useImperativeHandle(ref, () => ({
    pushFrame: (frame) => {
      engineRef.current?.pushFrame(frame);
    },
    loadFrames: (frames) => {
      engineRef.current?.loadFrames(frames);
    },
    setDetections: (detections) => {
      engineRef.current?.setDetections(detections);
    },
  }));

  return (
    <div
      className={className}
    >
      <canvas
        ref={canvasRef}
        className="h-full w-full"
      />
    </div>
  );
});

SpectrumCanvas.displayName = 'SpectrumCanvas';

