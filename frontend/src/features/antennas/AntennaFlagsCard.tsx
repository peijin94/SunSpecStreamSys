import * as React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../../components/ui/card';
import { selectAntennaFlags } from '../../api/antennaProtocol';
import { useAntennaFlags } from './useAntennaFlags';
import layout from './layout.json';
import licenseUrl from './lwa-antpos-LICENSE.txt?url';

const antennas = layout.antennas;
const fullExtent = Math.ceil(Math.max(...antennas.flatMap(a => [Math.abs(a.east_m), Math.abs(a.north_m)])) / 100) * 100;

export function AntennaFlagsCard() {
  const {data, error} = useAntennaFlags();
  const [frequency, setFrequency] = React.useState<number | null>(null);
  const [core, setCore] = React.useState(false);
  const [selected, setSelected] = React.useState<number | null>(null);
  const bands = data?.bands ?? [];
  const {badIds, complete} = selectAntennaFlags(bands, frequency);
  const extent = core ? 200 : fullExtent;
  const visible = antennas.filter(a => Math.abs(a.east_m) <= extent && Math.abs(a.north_m) <= extent);
  // A square map preserves distance and orientation at every viewport size.
  const x = (east: number) => 184 + east / extent * 132;
  const y = (north: number) => 150 - north / extent * 132;
  const status = (id: number) => badIds.has(id) ? 'flagged' : complete ? 'unflagged' : 'unknown';
  const observed = data?.observation_time
    ? new Date(data.observation_time).toISOString().slice(0, 19).replace('T', ' ') + ' UTC' : null;
  const active = antennas.find(a => a.id === selected);

  return (
    <Card aria-labelledby="antenna-flags-title" className="min-w-0 bg-slate-900/80 border-slate-800">
      <CardHeader>
        <CardTitle id="antenna-flags-title">Antenna flagging</CardTitle>
        <p className="text-[11px] text-slate-400">Updates every minute</p>
      </CardHeader>
      <CardContent className="space-y-3 text-xs">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <label className="flex items-center gap-2 text-slate-400">
            Band
            <select aria-label="Antenna flagging band" value={frequency ?? 'any'} onChange={event => setFrequency(event.target.value === 'any' ? null : Number(event.target.value))}
              className="rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-slate-200">
              <option value="any">Any band</option>
              {bands.map(band => <option key={band.freq_mhz} value={band.freq_mhz}>{band.freq_mhz} MHz</option>)}
            </select>
          </label>
          <div className="flex rounded-md border border-slate-700 p-0.5" role="group" aria-label="Antenna map view">
            {[false, true].map(value => <button key={String(value)} type="button" aria-pressed={core === value} onClick={() => setCore(value)}
              className={`rounded px-2 py-1 ${core === value ? 'bg-slate-200 text-slate-950' : 'text-slate-400 hover:text-slate-100'}`}>
              {value ? 'Core zoom' : 'Full array'}
            </button>)}
          </div>
        </div>
        <div className="flex flex-wrap gap-x-3 gap-y-1" aria-label="Antenna flagging legend">
          <span className="text-red-400">● {badIds.size} flagged</span>
          <span className="text-emerald-400">● {complete ? antennas.length - badIds.size : 0} not flagged</span>
          {!complete && <span className="text-slate-400">● {antennas.length - badIds.size} unknown</span>}
        </div>
        <svg viewBox="0 0 350 316" role="img" aria-label={`OVRO LWA antenna layout, ${core ? 'core zoom' : 'full array'}, ${badIds.size} flagged antennas`}
          className="block w-full rounded-lg border border-slate-800 bg-slate-950">
          <defs><clipPath id="antenna-map-clip"><rect x="52" y="18" width="264" height="264" /></clipPath></defs>
          {[-1, -.5, 0, .5, 1].map(t => <g key={t} fill="#94a3b8" fontSize="10">
            <path d={`M ${x(t * extent)} 18 V 282 M 52 ${y(t * extent)} H 316`} stroke="#1e293b" strokeWidth="1" />
            <text x={x(t * extent)} y="296" textAnchor="middle">{Math.round(t * extent)}</text>
            <text x="45" y={y(t * extent) + 3} textAnchor="end">{Math.round(t * extent)}</text>
          </g>)}
          <text x="184" y="311" textAnchor="middle" fill="#94a3b8" fontSize="10">East (m)</text>
          <text x="12" y="150" textAnchor="middle" transform="rotate(-90 12 150)" fill="#94a3b8" fontSize="10">North (m)</text>
          <g clipPath="url(#antenna-map-clip)">
            {[...visible].sort((a,b) => Number(badIds.has(a.id)) - Number(badIds.has(b.id))).map(antenna => {
              const flagged = status(antenna.id);
              return <circle key={antenna.id} data-antenna-id={antenna.id} data-antenna-name={antenna.name} data-status={flagged}
                cx={x(antenna.east_m)} cy={y(antenna.north_m)} r={core ? 2.7 : 2.4}
                fill={flagged === 'flagged' ? '#f87171' : flagged === 'unflagged' ? '#34d399' : '#64748b'}
                stroke={selected === antenna.id ? '#f8fafc' : '#020617'} strokeWidth={selected === antenna.id ? 1.5 : .4}
                onClick={() => setSelected(antenna.id)} className="cursor-pointer">
                <title>{antenna.name} · index {antenna.id} · {flagged === 'unflagged' ? 'not flagged' : flagged} · E {antenna.east_m.toFixed(1)} m, N {antenna.north_m.toFixed(1)} m</title>
              </circle>;
            })}
          </g>
        </svg>
        <p className="text-slate-400">{core ? `${visible.length} of ${antennas.length} antennas in core view` : `${antennas.length} antennas · north up`}
          {frequency === null ? ' · red = flagged in any band' : ` · ${frequency} MHz`}</p>
        {active && <p className="text-slate-200">{active.name} · index {active.id} · {status(active.id) === 'unflagged' ? 'not flagged' : status(active.id)}</p>}
        <p className="text-slate-400">{observed ? `Observed ${observed}` : 'Waiting for antenna flagging'}</p>
        {(error || data?.status === 'stale') && <p role="status" className="text-amber-300">{error || 'Last available observation shown.'}</p>}
        {!complete && bands.length > 0 && <p className="text-slate-400">Some antenna lists are unavailable; gray antennas have unknown status.</p>}
        <p className="text-[10px] text-slate-500">Coordinates: <a href={layout.source} target="_blank" rel="noreferrer" className="underline hover:text-slate-300">lwa-antpos</a>
          {' · '}<a href={licenseUrl} download="lwa-antpos-LICENSE.txt" className="underline hover:text-slate-300">License</a></p>
      </CardContent>
    </Card>
  );
}
