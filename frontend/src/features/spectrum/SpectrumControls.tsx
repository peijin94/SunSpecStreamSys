import * as React from 'react';
import type {useSpectrumPolling} from './useSpectrum';

type Props={spectrum:ReturnType<typeof useSpectrumPolling>};
const time=(value:number)=>new Date(value*1000).toISOString().replace('T',' ').slice(0,19);
export function SpectrumControls({spectrum:s}:Props) {
 const [from,setFrom]=React.useState(''),[to,setTo]=React.useState(''),[error,setError]=React.useState('');
 const [minimum,setMinimum]=React.useState('15'),[maximum,setMaximum]=React.useState('85'),[frequencyError,setFrequencyError]=React.useState('');
 const data=s.data;
 const available=data?.coverage_start!==null && data?.coverage_start!==undefined && data.coverage_end!==null
   ? Math.max(0,data.coverage_end-data.coverage_start):0;
 const stale=data?.coverage_end?data.timestamp-data.coverage_end>10:false;
 function apply(event:React.FormEvent) {
  event.preventDefault();
  const start=Date.parse(from+'Z')/1000,end=Date.parse(to+'Z')/1000;
  if(!Number.isFinite(start)||!Number.isFinite(end)||end-start<1||end-start>3600) {
   setError('Choose a valid UTC interval between 1 second and 1 hour.');return;
  }
  setError('');s.zoom({start,end,fmin:s.view.fmin,fmax:s.view.fmax});
 }
 const button='rounded-md border border-slate-600 px-3 py-1.5 text-xs hover:bg-slate-700 focus-visible:outline-2 focus-visible:outline-sky-400';
 return <section aria-label="Spectrum time range and playback" className="rounded-xl border border-slate-800 bg-slate-900/80 p-3 space-y-3" data-testid="spectrum-controls" data-playing={String(s.playing)} data-seconds={s.seconds}>
  <div className="flex flex-wrap items-center gap-2">
   <span className="text-xs text-slate-400 mr-1">Show last</span>
   {[[180,'3 min'],[300,'5 min'],[600,'10 min'],[1800,'30 min'],[3600,'1 hr']].map(([seconds,label])=>
    <button key={seconds} type="button" className={`${button} ${Math.abs(s.seconds-Number(seconds))<.01?'bg-sky-950 border-sky-500 text-sky-200':''}`} data-testid={`range-${seconds}`} aria-pressed={Math.abs(s.seconds-Number(seconds))<.01} onClick={()=>{setError('');s.preset(Number(seconds));}}>{label}</button>)}
   <div className="flex flex-wrap gap-2 ml-auto">
    <button type="button" data-testid="spectrum-pause" className={`${button} disabled:opacity-40`} disabled={s.measuring} onClick={s.playing?s.pause:s.play}>{s.playing?'❚❚ Pause':'▶ Play'}</button>
    <button type="button" data-testid="spectrum-live" disabled={s.measuring} className={`${button} border-emerald-600 text-emerald-300 disabled:opacity-40`} onClick={()=>{setError('');s.live();}}>● Live · 3 min</button>
    <button type="button" className={button} onClick={s.zoomOut}>Zoom out</button>
    <button type="button" className={button} onClick={s.resetZoom}>Reset zoom</button>
   </div>
  </div>
  <div className="flex flex-wrap justify-between gap-2 text-xs text-slate-400" role="status">
   <span>{s.measuring?'Paused · measurement lock':s.playing?(stale?'Stream stale · waiting for data':'Playing'):'Paused'} · {s.refreshing?'Loading selected range…':s.status==='error'?'Request failed — last display retained':data?.bin_seconds?`${data.bin_seconds.toFixed(2)} s time bins`:'Native time cadence'} · Drag either spectrum to zoom time + frequency and pause.</span>
   <span>{Math.min(60,available/60).toFixed(1)} / 60 min retained{available<3590?' · buffer warming':''}</span>
  </div>
  {data && <p className="text-[11px] text-slate-400 break-words" data-testid="spectrum-selection">{time(s.view.start)} → {time(s.view.end)} UTC · {s.view.fmin.toFixed(1)}–{s.view.fmax.toFixed(1)} MHz{available>0 && data.coverage_start!==null && s.view.start<data.coverage_start?' · unavailable older data remains blank':''}</p>}
  <details className="text-xs text-slate-400">
   <summary className="cursor-pointer">Select UTC time range</summary>
   <form className="mt-2 flex flex-wrap items-end gap-2" onSubmit={apply}>
    <label className="grid min-w-0 gap-1">From (UTC)<input aria-label="Range from UTC" type="datetime-local" step="1" value={from} onChange={e=>setFrom(e.target.value)} required className="rounded border border-slate-600 bg-slate-950 px-2 py-1 text-slate-100 min-w-0 max-w-full"/></label>
    <label className="grid min-w-0 gap-1">To (UTC)<input aria-label="Range to UTC" type="datetime-local" step="1" value={to} onChange={e=>setTo(e.target.value)} required className="rounded border border-slate-600 bg-slate-950 px-2 py-1 text-slate-100 min-w-0 max-w-full"/></label>
    <button type="submit" className={button}>Apply range & pause</button>
   </form>
   {data?.coverage_start!==null && data?.coverage_start!==undefined && data.coverage_end!==null && <p className="mt-2">Available: {time(data.coverage_start)} → {time(data.coverage_end)} UTC. Empty intervals stay blank.</p>}
   {error && <p role="alert" className="mt-2 text-red-300">{error}</p>}
  </details>
  <details className="text-xs text-slate-400">
   <summary className="cursor-pointer">Select frequency range</summary>
   <form className="mt-2 flex flex-wrap items-end gap-2" onSubmit={event=>{
    event.preventDefault();const fmin=Number(minimum),fmax=Number(maximum);
    if(!Number.isFinite(fmin)||!Number.isFinite(fmax)||fmin<15||fmax>85||fmax-fmin<.1){setFrequencyError('Choose a frequency range within 15–85 MHz, at least 0.1 MHz wide.');return;}
    setFrequencyError('');s.zoom({...s.view,fmin,fmax});
   }}>
    <label className="grid gap-1">Minimum (MHz)<input aria-label="Minimum frequency MHz" className="w-28 rounded border border-slate-600 bg-slate-950 px-2 py-1 text-slate-100" type="number" step="0.1" min="15" max="85" required value={minimum} onChange={e=>setMinimum(e.target.value)}/></label>
    <label className="grid gap-1">Maximum (MHz)<input aria-label="Maximum frequency MHz" className="w-28 rounded border border-slate-600 bg-slate-950 px-2 py-1 text-slate-100" type="number" step="0.1" min="15" max="85" required value={maximum} onChange={e=>setMaximum(e.target.value)}/></label>
    <button type="submit" className={button}>Apply frequency range</button>
   </form>
   {frequencyError&&<p role="alert" className="mt-2 text-red-300">{frequencyError}</p>}
  </details>
 </section>;
}
