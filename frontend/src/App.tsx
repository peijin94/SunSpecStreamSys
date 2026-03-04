import './App.css';
import { SpectrumCanvas } from './features/spectrum/SpectrumCanvas';
import { useSpectrumPolling } from './features/spectrum/useSpectrum';
import { useDetectionsPolling } from './features/detections/useDetections';
import { Button } from './components/ui/button';
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from './components/ui/card';
import * as React from 'react';
import type { SpectrumCanvasHandle } from './features/spectrum/SpectrumCanvas';
import { fetchAiSummary, fetchVisitorCount } from './api/client';
import { Gemini } from '@lobehub/icons';

function App() {
  const spectrum = useSpectrumPolling();
  const detections = useDetectionsPolling();
  const canvasRef =
    React.useRef<SpectrumCanvasHandle | null>(null);
  const [aiSummary, setAiSummary] = React.useState('');
  const [aiSummaryTime, setAiSummaryTime] =
    React.useState<number | null>(null);
  const [eventsMode, setEventsMode] =
    React.useState<'count' | 'list'>('count');
  const [visitorCount, setVisitorCount] =
    React.useState<number | null>(null);

  React.useEffect(() => {
    if (!spectrum.latestFrame) return;
    canvasRef.current?.pushFrame(
      spectrum.latestFrame,
    );
  }, [spectrum.latestFrame]);

  React.useEffect(() => {
    canvasRef.current?.setDetections(
      detections.detections,
    );
  }, [detections.detections]);

  React.useEffect(() => {
    let cancelled = false;

    async function tick() {
      try {
        const data = await fetchAiSummary();
        if (cancelled) return;
        if (data.summary) {
          setAiSummary(data.summary);
          setAiSummaryTime(
            data.timestamp || null,
          );
        }
      } catch {
        // ignore errors for now
      }
    }

    tick();
    const id = window.setInterval(tick, 60000);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, []);

  React.useEffect(() => {
    let cancelled = false;

    async function loadCount() {
      try {
        const data = await fetchVisitorCount();
        if (cancelled) return;
        if (typeof data.count === 'number') {
          setVisitorCount(data.count);
        }
      } catch {
        // ignore for now
      }
    }

    loadCount();

    return () => {
      cancelled = true;
    };
  }, []);

  const handleRefresh = async () => {
    const frames = await spectrum.refresh();
    if (frames && frames.length) {
      canvasRef.current?.loadFrames(frames);
    }
  };

  return (
    <div className="min-h-screen bg-black text-slate-50">
      <main className="mx-auto flex min-h-screen max-w-6xl flex-col gap-6 px-4 py-6">
        <header className="flex items-baseline justify-between">
          <h1 className="text-2xl font-semibold tracking-tight">
            OVRO LWA Live Spectrum
          </h1>
          <div className="text-xs text-slate-400">
            <span
              role="img"
              aria-label="wave"
              className="mr-1"
            >
              👋
            </span>
            {visitorCount !== null
              ? `Welcome as ${visitorCount.toLocaleString()} visitor`
              : 'Welcome'}
          </div>
        </header>

        <section className="overflow-hidden rounded-2xl border border-slate-800 bg-slate-900/80 shadow-lg">
          <div className="aspect-[5/2] w-full">
            <SpectrumCanvas
              ref={canvasRef}
              className="h-full w-full"
            />
          </div>
        </section>

        <section className="grid gap-6 md:grid-cols-3">
          <Card className="bg-slate-900/80 border-slate-800">
            <CardHeader>
              <CardTitle>Obs status and control</CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 text-sm">
              <div className="flex justify-between">
                <span className="text-slate-400">
                  Connection
                </span>
                <span
                  className={
                    spectrum.status === 'ok'
                      ? 'text-emerald-400'
                      : spectrum.status === 'error'
                        ? 'text-red-400'
                        : 'text-slate-400'
                  }
                >
                  {spectrum.status === 'ok'
                    ? 'Connected'
                    : spectrum.status === 'error'
                      ? 'Error'
                      : 'Connecting...'}
                </span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">
                  Frames received
                </span>
                <span>{spectrum.frameCount}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-slate-400">
                  Last update
                </span>
                <span>
                  {spectrum.lastUpdate ?? 'Never'}
                </span>
              </div>
              <div className="pt-3">
                <Button
                  className="w-full"
                  onClick={handleRefresh}
                  variant="outline"
                >
                  Refresh Spectrum
                </Button>
              </div>
            </CardContent>
          </Card>

          <Card className="bg-slate-900/80 border-slate-800">
            <CardHeader>
              <div className="flex items-center justify-between">
                <div className="flex flex-col gap-1">
                  <CardTitle>Events</CardTitle>
                  <span className="text-[10px] tracking-wide text-slate-500">
                    Powered by YOLOv8
                  </span>
                </div>
                <div className="inline-flex rounded-full border border-slate-700 bg-slate-900 text-xs">
                  <button
                    type="button"
                    className={`px-3 py-1 rounded-full ${
                      eventsMode === 'count'
                        ? 'bg-slate-100 text-slate-900'
                        : 'text-slate-300'
                    }`}
                    onClick={() => setEventsMode('count')}
                  >
                    Counts
                  </button>
                  <button
                    type="button"
                    className={`px-3 py-1 rounded-full ${
                      eventsMode === 'list'
                        ? 'bg-slate-100 text-slate-900'
                        : 'text-slate-300'
                    }`}
                    onClick={() => setEventsMode('list')}
                  >
                    List
                  </button>
                </div>
              </div>
            </CardHeader>
            <CardContent className="space-y-3 text-sm">
              {eventsMode === 'count' ? (
                <>
                  <div className="flex justify-between">
                    <span className="text-slate-400">
                      Current detections
                    </span>
                    <span>{detections.latestCount}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">
                      Type III (current)
                    </span>
                    <span>{detections.currentType3}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-400">
                      Type IIIb (current)
                    </span>
                    <span>{detections.currentType3b}</span>
                  </div>
                </>
              ) : (
                <div className="space-y-2 max-h-64 overflow-auto pr-1">
                  {detections.detections.length === 0 ? (
                    <p className="text-slate-500">
                      No detections in the latest poll.
                    </p>
                  ) : (
                    detections.detections.map((d) => (
                      <div
                        key={d.id}
                        className="rounded-md border border-slate-700 bg-slate-900 px-2 py-1.5 text-xs space-y-1"
                      >
                        <div className="flex justify-between">
                          <span className="font-medium">
                            {d.class}
                          </span>
                          <span className="text-slate-400">
                            {(d.confidence * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="flex justify-between text-slate-500">
                          <span>ID {d.id}</span>
                          <span>
                            x={d.bbox[0].toFixed(2)}, y=
                            {d.bbox[1].toFixed(2)}
                          </span>
                        </div>
                      </div>
                    ))
                  )}
                </div>
              )}
            </CardContent>
          </Card>

          <Card className="bg-slate-900/80 border-slate-800">
            <CardHeader>
              <div className="flex flex-col gap-1">
                <div className="flex items-center gap-2">
                  <Gemini.Color size={18} />
                  <CardTitle>AI summarize</CardTitle>
                </div>
                <span className="text-[10px] tracking-wide text-slate-500">
                  Powered by gemini-2.5-flash-lite
                </span>
              </div>
            </CardHeader>
            <CardContent className="space-y-3 text-sm">
              {aiSummary ? (
                <>
                  <p className="text-slate-200">
                    {aiSummary}
                  </p>
                  {aiSummaryTime && (
                    <p className="text-xs text-slate-500">
                      Last updated:{' '}
                      {new Date(
                        aiSummaryTime * 1000,
                      ).toUTCString()}
                    </p>
                  )}
                </>
              ) : (
                <p className="text-slate-500">
                  Waiting for AI summary...
                </p>
              )}
            </CardContent>
          </Card>
        </section>
      </main>
    </div>
  );
}

export default App;
