import './App.css';
import { SpectrumCanvas } from './features/spectrum/SpectrumCanvas';
import { useSpectrumPolling } from './features/spectrum/useSpectrum';
import { SpectrumControls } from './features/spectrum/SpectrumControls';
import {MeasurementPanel} from './features/spectrum/MeasurementPanel';
import {useMeasurement} from './features/spectrum/useMeasurement';
import { useDetectionsPolling } from './features/detections/useDetections';
import { RadioImagesPanel } from './features/images/RadioImagesPanel';
import { ObservationInfo } from './components/ObservationInfo';
import { AntennaFlagsCard } from './features/antennas/AntennaFlagsCard';
import { ImagingStatusCard } from './features/imaging/ImagingStatusCard';
import { Button } from './components/ui/button';
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from './components/ui/card';
import * as React from 'react';
import OpenAI from '@lobehub/icons/es/OpenAI/components/Mono';
import type { SpectrumCanvasHandle } from './features/spectrum/SpectrumCanvas';
import {
  fetchAiSummary,
  fetchVisitorCount,
  fetchSunEphemeris,
  type SunEphemeris,
  type AiSummaryResponse,
} from './api/client';
import { IS_PUBLIC } from './lib/appMode';

function App() {
  const spectrum = useSpectrumPolling();
  // Public fixed page never measures: force the tool inactive even if a stale
  // measuring flag ever arrived.
  const measurement=useMeasurement(IS_PUBLIC ? false : spectrum.measuring);
  const closeMeasurement=()=>{measurement.clear();spectrum.endMeasurement();};
  const detections = useDetectionsPolling();
  const intensityRef =
    React.useRef<SpectrumCanvasHandle | null>(null);
  const [aiResult, setAiResult] = React.useState<AiSummaryResponse | null>(null);
  const aiSummary = aiResult?.summary ?? '';
  const aiSummaryTime = aiResult?.timestamp ?? null;
  const [eventsMode, setEventsMode] =
    React.useState<'count' | 'list'>('count');
  const [visitorCount, setVisitorCount] =
    React.useState<number | null>(null);
  const [sunEph, setSunEph] =
    React.useState<SunEphemeris | null>(null);

  React.useEffect(() => {
    if(!spectrum.measuring)intensityRef.current?.setDetections(detections.detections);
  }, [detections.detections,spectrum.measuring]);

  React.useEffect(() => {
    let cancelled = false;

    async function tick() {
      try {
        const data = await fetchAiSummary();
        if (cancelled) return;
        setAiResult(data);
      } catch {
        if (cancelled) return;
        setAiResult(previous => ({
          ...(previous ?? { summary: '', timestamp: 0, model: 'gpt-6-luna', provider: 'openai-codex' }),
          status: previous?.summary ? 'stale' : 'error', error: 'AI summary temporarily unavailable',
        }));
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

  React.useEffect(() => {
    let cancelled = false;

    async function loadEphemeris() {
      try {
        const data = await fetchSunEphemeris();
        if (cancelled) return;
        setSunEph(data);
      } catch {
        if (cancelled) return;
        setSunEph(null);
      }
    }

    loadEphemeris();
    const id = window.setInterval(loadEphemeris, 60000);

    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, []);

  const formatEphemTime = (value: string | undefined) => {
    if (!value) return '—';
    const d = new Date(value);
    if (Number.isNaN(d.getTime())) return value;
    return d.toLocaleTimeString(undefined, {
      hour: '2-digit',
      minute: '2-digit',
      timeZoneName: 'short',
    });
  };

  return (
    <div className="min-h-screen bg-black text-slate-50">
      <main className="flex min-h-screen w-full flex-col gap-4 px-4 py-4 xl:px-6">
        <header className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-2">
          <h1 className="text-2xl font-semibold tracking-tight">
            OVRO LWA solar observation live
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

        <ObservationInfo elevation={sunEph?.el ?? null} />

        {IS_PUBLIC ? (
          <section aria-label="Public live status" data-testid="public-fixed-status" data-playing="true" data-seconds={spectrum.seconds}
            className="rounded-xl border border-emerald-800 bg-slate-900/80 p-3">
            <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-sm">
              <span className="font-semibold text-emerald-300">● Live · Fixed last 5 min</span>
              <span className="text-slate-300">Full band 15–85 MHz · Rolling latest 300 s · No pause or zoom</span>
            </div>
            <p className="mt-1 text-xs text-slate-400" data-testid="public-source-timestamp">
              Source data through {spectrum.lastUpdate ?? 'waiting for data…'}
              {spectrum.data ? ` · Showing ${new Date(spectrum.view.start*1000).toISOString().slice(11,19)} → ${new Date(spectrum.view.end*1000).toISOString().slice(11,19)} UTC` : ''}
            </p>
          </section>
        ) : (
          <SpectrumControls spectrum={spectrum} />
        )}

        <section aria-label="Live spectra and solar images" className={`live-monitor-grid ${spectrum.measuring?'measure-open':''}`}>
          <section aria-label="Live Stokes I and V spectra" className="live-spectrum-stack grid min-w-0 gap-4">
            <section aria-labelledby="intensity-title" className="live-spectrum-panel overflow-hidden rounded-2xl border border-slate-800 bg-slate-900/80 shadow-lg">
              <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 px-4 py-3">
                <h2 id="intensity-title" className="text-sm font-semibold">Stokes I · Total intensity</h2>
                <div className="flex items-center gap-2 text-[11px] text-slate-400">
                  <span>1</span>
                  <span className="h-2 w-24 rounded-full" style={{ background: 'linear-gradient(to right, #440154, #3b528b, #21918c, #5ec962, #fde725)' }} />
                  <span>3×10³</span><span>Log · s.f.u.</span>
                </div>
              </div>
              <SpectrumCanvas ref={intensityRef} data={spectrum.data} view={spectrum.view} onPause={IS_PUBLIC ? ()=>{} : spectrum.pause} onZoom={IS_PUBLIC ? ()=>{} : spectrum.zoom} measurePoints={IS_PUBLIC ? [] : measurement.points} onMeasurePoint={IS_PUBLIC ? undefined : (spectrum.measuring&&!spectrum.snapshotLoading?measurement.addPoint:undefined)} interactive={!IS_PUBLIC} polarization="I" className="live-spectrum-canvas h-[230px] w-full sm:h-[280px]" />
            </section>
            <section aria-labelledby="circular-title" className="live-spectrum-panel overflow-hidden rounded-2xl border border-slate-800 bg-slate-900/80 shadow-lg">
              <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 px-4 py-3">
                <h2 id="circular-title" className="text-sm font-semibold">Stokes V / I · Circular polarization</h2>
                <div className="flex items-center gap-2 text-[11px] text-slate-400">
                  <span>−0.6</span>
                  <span className="relative h-2 w-24 rounded-full bg-gradient-to-r from-blue-600 via-slate-100 to-red-500" aria-label="Blue negative, white zero, red positive" />
                  <span>+0.6</span><span>Linear · V / I</span>
                </div>
              </div>
              <SpectrumCanvas data={spectrum.data} view={spectrum.view} onPause={IS_PUBLIC ? ()=>{} : spectrum.pause} onZoom={IS_PUBLIC ? ()=>{} : spectrum.zoom} measurePoints={IS_PUBLIC ? [] : measurement.points} onMeasurePoint={IS_PUBLIC ? undefined : (spectrum.measuring&&!spectrum.snapshotLoading?measurement.addPoint:undefined)} interactive={!IS_PUBLIC} polarization="V" className="live-spectrum-canvas h-[230px] w-full sm:h-[280px]" />
            </section>
          </section>

          <div className="live-image-stack">
            {!IS_PUBLIC && <MeasurementPanel spectrum={spectrum} tool={measurement} onClose={closeMeasurement}/>}
            <RadioImagesPanel />
          </div>
        </section>



        <section aria-label="Observation status, flagging map, imaging status, events, and AI summary" className="grid min-w-0 items-start gap-4 md:grid-cols-2 xl:grid-cols-4 2xl:grid-cols-5">
          <AntennaFlagsCard />
          <ImagingStatusCard />
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
                  Samples in selected range
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
              <div className="flex items-center justify-between pt-1 text-xs">
                <span className="text-slate-400">
                  Sun ephemeris
                </span>
                <div className="flex items-center gap-2">
                  <span className="text-slate-200">
                    EL:{' '}
                    {sunEph && Number.isFinite(sunEph.el)
                      ? sunEph.el.toFixed(1)
                      : '–'}
                    °, AZ:{' '}
                    {sunEph && Number.isFinite(sunEph.az)
                      ? sunEph.az.toFixed(1)
                      : '–'}
                    °
                  </span>
                  <div className="relative group">
                    <button
                      type="button"
                      className="rounded-full border border-slate-600 px-2 py-0.5 text-[10px] text-slate-200 hover:bg-slate-700"
                    >
                      Detail
                    </button>
                    <div className="pointer-events-none absolute right-0 z-10 mt-1 w-60 rounded-md border border-slate-700 bg-slate-900/95 p-2 text-[11px] text-slate-200 opacity-0 shadow-lg transition-opacity group-hover:opacity-100">
                      {sunEph ? (
                        <>
                          <div>
                            Sunrise (OVRO):{' '}
                            {formatEphemTime(
                              sunEph.sunrise,
                            )}
                          </div>
                          <div>
                            Sunset (OVRO):{' '}
                            {formatEphemTime(
                              sunEph.sunset,
                            )}
                          </div>
                          {typeof sunEph.sunup === 'boolean' && (
                            <div className="mt-1 flex items-center gap-1">
                              <span>Sun up:</span>
                              <span
                                className={
                                  sunEph.sunup
                                    ? 'inline-flex items-center rounded-full border border-emerald-500/60 bg-emerald-500/15 px-2 py-0.5 text-[10px] font-medium text-emerald-300'
                                    : 'inline-flex items-center rounded-full border border-slate-500/60 bg-slate-700/60 px-2 py-0.5 text-[10px] font-medium text-slate-200'
                                }
                              >
                                {sunEph.sunup ? 'Yes' : 'No'}
                              </span>
                            </div>
                          )}
                        </>
                      ) : (
                        <div>Loading ephemeris…</div>
                      )}
                    </div>
                  </div>
                </div>
              </div>
              <div className="pt-3">
                <Button
                  className="w-full"
                  onClick={spectrum.refresh}
                  variant="outline"
                >
                  {spectrum.refreshing ? 'Updating…' : 'Refresh I / V'}
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

          <Card data-testid="ai-summary-card" className="bg-slate-900/80 border-slate-800">
            <CardHeader>
              <div className="flex flex-col gap-1">
                <div className="flex items-center gap-2">
                  <OpenAI size={18} role="img" aria-label="OpenAI" className="shrink-0 text-slate-200" />
                  <CardTitle>AI summary</CardTitle>
                </div>
                <span className="text-[10px] tracking-wide text-slate-500">
                  Powered by {aiResult?.model && aiResult.model !== 'gpt-6-luna' ? aiResult.model : 'GPT-6 Luna'}
                </span>
                <span className="text-[10px] text-slate-500">
                  Latest five-minute spectrum · UTC
                </span>
              </div>
            </CardHeader>
            <CardContent className="space-y-3 text-sm">
              {aiResult?.status === 'stale' && (
                <p role="status" className="text-xs text-amber-400">Stale interpretation · {aiResult.error || 'Waiting for an updated interpretation'}</p>
              )}
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
                  {aiResult?.error || 'Waiting for AI summary...'}
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
