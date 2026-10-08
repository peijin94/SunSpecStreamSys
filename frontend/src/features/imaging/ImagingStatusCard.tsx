import * as React from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '../../components/ui/card';
import { selectImagingBand } from '../../api/imagingProtocol';
import { useImagingStatus } from './useImagingStatus';

const sources = ['CasA', 'CygA', 'TauA', 'VirA'] as const;
const fluxFormat = new Intl.NumberFormat('en-US', {maximumFractionDigits: 0});
const flux = (value: number | null | undefined) => value == null ? '—' : fluxFormat.format(value);

export function ImagingStatusCard() {
  const {data, error} = useImagingStatus();
  const [frequency, setFrequency] = React.useState<number | null>(null);
  const band = selectImagingBand(data?.bands ?? [], frequency);
  const fraction = band?.flagged_frac ?? null;
  const observed = data?.observation_time
    ? new Date(data.observation_time).toISOString().slice(0, 19).replace('T', ' ') + ' UTC' : null;

  return (
    <Card aria-labelledby="imaging-status-title" className="min-w-0 bg-slate-900/80 border-slate-800">
      <CardHeader>
        <CardTitle id="imaging-status-title">Imaging status info</CardTitle>
        <p className="text-[11px] text-slate-400">Updates every minute</p>
      </CardHeader>
      <CardContent className="space-y-4 text-xs">
        <label className="flex items-center gap-2 text-slate-400">
          Band
          <select aria-label="Imaging status band" value={band?.freq_mhz ?? ''} disabled={!band} onChange={e => setFrequency(Number(e.target.value))}
            className="rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-slate-200 disabled:opacity-50">
            {!band && <option value="">Waiting for QA</option>}
            {data?.bands.map(b => <option key={b.freq_mhz} value={b.freq_mhz}>{b.freq_mhz} MHz</option>)}
          </select>
        </label>
        <section aria-label="Baseline flagging ratio">
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <h4 className="font-medium text-slate-300">Baseline flagging ratio:</h4>
            <span data-testid="baseline-flagging-ratio" className="text-base font-semibold tabular-nums">
              {fraction === null ? '—' : `${(fraction * 100).toFixed(1)}%`}
            </span>
          </div>
          <div className="mt-2 h-2 overflow-hidden rounded-full bg-slate-800" role="meter" aria-label="Flagged visibility samples"
            aria-valuemin={0} aria-valuemax={100} aria-valuenow={fraction === null ? undefined : fraction * 100}>
            {fraction !== null && <div className="h-full rounded-full bg-amber-400" style={{width: `${fraction * 100}%`}} />}
          </div>
          <p className="mt-1 text-[10px] text-slate-500">Flagged visibility samples across baselines</p>
        </section>
        <section aria-label="A-team flux check">
          <h4 className="mb-2 font-medium text-slate-300">A-team flux check:</h4>
          <div className="overflow-x-auto">
            <table className="w-full text-right text-[11px] tabular-nums" aria-label="Measured and expected A-team flux in Jy">
              <thead className="text-slate-400"><tr>
                <th scope="col" className="py-1 pr-2 text-left font-normal">Source</th>
                <th scope="col" className="px-1 font-normal">Measured</th>
                <th scope="col" className="px-1 font-normal">Expected</th>
                <th scope="col" className="pl-1 font-normal">M/E</th>
              </tr></thead>
              <tbody>{sources.map(source => {
                const row = data?.flux_rows.find(r => r.freq_mhz === band?.freq_mhz && r.source === source);
                return <tr key={source} data-source={source} className="border-t border-slate-800">
                  <th scope="row" className="py-2 pr-2 text-left font-medium">{source}</th>
                  <td className="px-1">{flux(row?.measured_jy)}</td>
                  <td className="px-1">{flux(row?.expected_jy)}</td>
                  <td className="pl-1">{row?.ratio == null ? '—' : `${row.ratio.toFixed(2)}×`}</td>
                </tr>;
              })}</tbody>
            </table>
          </div>
          <p className="mt-2 text-[10px] text-slate-500">Flux in Jy · M/E = measured / expected<br />Expected flux includes the beam model; — means no measurement.</p>
          {data?.flux_message && <p className="mt-2 text-slate-400">{data.flux_message}</p>}
        </section>
        <p className="text-slate-400">{observed ? `Observed ${observed}` : 'Waiting for imaging QA'}</p>
        {(error || data?.status === 'stale') && <p role="status" className="text-amber-300">{error || 'Last available observation shown.'}</p>}
      </CardContent>
    </Card>
  );
}
