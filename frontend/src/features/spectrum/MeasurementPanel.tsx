import type {useSpectrumPolling} from './useSpectrum';
import type {useMeasurement} from './useMeasurement';
import type {DensityModel} from './measureModel';

type Series={x:number;y:number}[];
const utc=(t:number)=>new Date(t*1000).toISOString().slice(11,23);
const number=(n:number)=>Math.abs(n)>=10000||Math.abs(n)>0&&Math.abs(n)<.01?n.toExponential(1):Number(n.toPrecision(3)).toString();
function MeasurePlot({series,time,title,xlabel,ylabel,id}:{series:Series;time?:boolean;title:string;xlabel:string;ylabel:string;id:string}) {
 const w=420,h=185,left=64,top=14,pw=340,ph=120;
 const xs=series.map(p=>p.x),ys=series.map(p=>p.y);
 const low=xs.length?Math.min(...xs):0,high=xs.length?Math.max(...xs):1;
 const pad=Math.max((high-low)*.05,time?.5:.01),xmin=low-pad,xmax=high+pad;
 const ymin=Math.min(0,...ys),ymax=Math.max(0,...ys),yp=Math.max((ymax-ymin)*.1,.01);
 const bottom=ymin-yp,upper=ymax+yp;
 const x=(v:number)=>left+(v-xmin)/(xmax-xmin)*pw;
 const y=(v:number)=>top+ph-(v-bottom)/(upper-bottom)*ph;
 return <figure className="min-w-0 rounded-lg border border-slate-700 bg-slate-950 p-2">
  <figcaption className="mb-1 text-xs font-semibold text-sky-100">{title}</figcaption>
  <svg data-testid={id} viewBox={`0 0 ${w} ${h}`} className="block w-full" role="img" aria-label={`${title}. ${xlabel}; ${ylabel}. ${series.length} interval estimates.`}>
   {Array.from({length:4},(_,i)=>{const yy=top+ph*i/3,value=upper-(upper-bottom)*i/3;return <g key={'y'+i}>
    <line x1={left} y1={yy} x2={left+pw} y2={yy} stroke="#334155"/>
    <text x={left-7} y={yy+4} textAnchor="end" fill="#cbd5e1" fontSize="11">{number(value)}</text>
   </g>;})}
   {Array.from({length:3},(_,i)=>{const xx=left+pw*i/2,value=xmin+(xmax-xmin)*i/2;return <text key={'x'+i} x={xx} y={top+ph+17} textAnchor={i===0?'start':i===2?'end':'middle'} fill="#cbd5e1" fontSize="11">{time?utc(value).slice(0,8):number(value)}</text>;})}
   <text x={left+pw/2} y={h-7} textAnchor="middle" fill="#94a3b8" fontSize="11">{xlabel}</text>
   <text transform={`translate(13 ${top+ph/2}) rotate(-90)`} textAnchor="middle" fill="#94a3b8" fontSize="11">{ylabel}</text>
   {series.length>1&&<polyline points={series.map(p=>`${x(p.x)},${y(p.y)}`).join(' ')} fill="none" stroke="#38bdf8" strokeWidth="1.5"/>}
   {series.map((p,i)=><circle key={i} cx={x(p.x)} cy={y(p.y)} r="3.5" fill="#fbbf24" data-x={p.x} data-y={p.y}><title>{`${time?utc(p.x)+' UTC':p.x.toFixed(4)+' R☉'}: ${number(p.y)}`}</title></circle>)}
   {!series.length&&<text x={left+pw/2} y={top+ph/2} textAnchor="middle" fill="#64748b" fontSize="12">Pick at least two points</text>}
  </svg>
 </figure>;
}

type Props={spectrum:ReturnType<typeof useSpectrumPolling>;tool:ReturnType<typeof useMeasurement>;onClose:()=>void};
export function MeasurementPanel({spectrum:s,tool:t,onClose}:Props) {
 const button='rounded border border-slate-600 px-2 py-1 hover:bg-slate-800 disabled:opacity-40';
 const field='min-w-0 rounded border border-slate-600 bg-slate-950 px-2 py-1 text-slate-100';
 const segments=t.result?.segments??[];
 return <section className="rounded-xl border border-sky-800 bg-slate-900/95 p-3">
  <button type="button" className="w-full text-left text-sm font-semibold text-sky-200 disabled:opacity-40" disabled={!s.data} aria-expanded={s.measuring} aria-controls="spectrum-measure-panel" onClick={()=>s.measuring?onClose():s.beginMeasurement()}>Interactive spectrum measure</button>
  {s.measuring&&<div id="spectrum-measure-panel" data-testid="spectrum-measure-panel" data-ready={String(!s.snapshotLoading)} className="mt-3 space-y-3 text-xs" role="region" aria-label="Interactive spectrum measure">
   <div className="flex items-center justify-between gap-2"><span className="font-medium text-amber-200">❚❚ Spectrum frozen</span><button type="button" className={button} onClick={onClose}>Close measurement</button></div>
   <p role="status" className="text-slate-300">{s.snapshotLoading?'Capturing pinned native history…':'Click points along a feature in either spectrum. Drag to zoom; ranges and frequency limits remain adjustable without going live.'}</p>
   {s.snapshotError&&<p role="alert" className="text-amber-300">{s.snapshotError}</p>}
   <div className="grid grid-cols-2 gap-2">
    <label className="grid gap-1">Density model<select aria-label="Density model" className={field} value={t.model} onChange={e=>t.setModel(e.target.value as DensityModel)}><option value="saito77">Saito (1977)</option><option value="leblanc98">Leblanc (1998)</option><option value="newkirk">Newkirk (1961)</option></select></label>
    <label className="grid gap-1">Emission<select aria-label="Emission harmonic" className={field} value={t.harmonic} onChange={e=>t.setHarmonic(Number(e.target.value) as 1|2)}><option value="1">Fundamental · f = fₚ</option><option value="2">Harmonic · f = 2 fₚ</option></select></label>
    <label className="grid gap-1">Density multiplier<input aria-label="Density multiplier" className={field} type="number" min="0.01" max="100" step="0.1" value={t.densityScale} onChange={e=>t.setDensityScale(e.target.value)}/></label>
    <div className="flex flex-wrap items-end justify-end gap-2"><button type="button" className={button} disabled={!t.points.length} onClick={t.undo}>Undo point</button><button type="button" className={button} disabled={!t.points.length} onClick={t.clear}>Clear points</button></div>
   </div>
   <div className="flex justify-between gap-2 text-slate-400"><span data-testid="measure-point-count">{t.points.length} points · max 128</span><span role="status">{t.calculating?'Calculating…':segments.length?`${segments.length} interval estimates`:''}</span></div>
   {t.error&&<p role="alert" className="text-red-300">{t.error}</p>}
   <MeasurePlot series={segments.map(v=>({x:v.time,y:v.drift_mhz_s}))} time title="Frequency drift rate" xlabel="Time (UTC)" ylabel="df/dt (MHz/s)" id="measure-drift"/>
   <MeasurePlot series={segments.map(v=>({x:v.r_sun,y:v.speed_km_s}))} title="Inferred source speed" xlabel="Heliocentric radius (R☉)" ylabel="Speed (km/s)" id="measure-speed"/>
   {t.result?.warnings.map(w=><p role="alert" className="text-amber-300" key={w}>{w}</p>)}
   {t.points.length>0&&<div className="max-h-36 overflow-auto rounded border border-slate-700"><table className="w-full text-left text-[11px]"><caption className="sr-only">Selected spectrum points</caption><thead className="text-slate-400"><tr><th className="p-1">#</th><th>Time (UTC)</th><th>MHz</th><th>R☉</th><th><span className="sr-only">Remove</span></th></tr></thead><tbody>
    {t.points.map((p,i)=><tr key={p.time} className="border-t border-slate-800"><td className="p-1 text-amber-300">{i+1}</td><td>{utc(p.time)}</td><td>{p.frequency_mhz.toFixed(2)}</td><td>{t.result?.points.find(v=>v.time===p.time)?.r_sun.toFixed(3)??'—'}</td><td><button type="button" className="p-1 text-slate-400 hover:text-red-300" aria-label={`Remove point ${i+1}`} onClick={()=>t.remove(i)}>×</button></td></tr>)}
   </tbody></table></div>}
   <p className="text-[11px] leading-relaxed text-slate-400">Adjacent-point estimates: signed df/dt; speed = |Δr/Δt|. R☉ is radius from solar centre, not height. Assumes radial plasma emission in the selected density model, without propagation, projection or light-travel corrections. Blank integrations cannot be picked. Closing keeps the spectrum paused until you press Play.</p>
  </div>}
 </section>;
}
