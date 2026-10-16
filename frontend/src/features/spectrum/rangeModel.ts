import type { SpectrumBatch, SpectrumCursor } from '../../api/spectrumProtocol';

export interface RangeBatch extends SpectrumBatch {
  range: true;
  display_start: number;
  display_end: number;
  coverage_start: number | null;
  coverage_end: number | null;
  bin_seconds: number;
  native_cadence_seconds: number;
  integration_seconds?: number;
  raw_rows: number;
}
export interface QueryRange { seconds: number; start?: number; end?: number }
export interface ViewBounds { start: number; end: number; fmin: number; fmax: number }
export interface Point { x: number; y: number }
export const FULL_BAND = { fmin: 15, fmax: 85 };

export function rangeQuery(view: QueryRange, cursor?: SpectrumCursor) {
  const query = new URLSearchParams({format:'binary'});
  if (view.start !== undefined && view.end !== undefined) {
    query.set('start', String(view.start)); query.set('end', String(view.end));
  } else query.set('seconds', String(view.seconds));
  query.set('max_rows', view.seconds <= 300 ? '1200' : '600');
  if (cursor && view.seconds <= 300 && view.start === undefined) {
    query.set('since', String(cursor.sequence)); query.set('boot_id', cursor.boot_id);
  }
  return query.toString();
}

export function mergeRange(previous: RangeBatch | null, next: RangeBatch): RangeBatch {
  if (!previous || next.reset || next.boot_id !== previous.boot_id) return next;
  const I = [...previous.I], V = [...previous.V], timestamps = [...previous.timestamps];
  next.timestamps.forEach((timestamp,index)=>{
    if (!timestamps.length || timestamp > timestamps[timestamps.length-1]) {
      I.push(next.I[index]); V.push(next.V[index]); timestamps.push(timestamp);
    }
  });
  const first = timestamps.findIndex(t=>t >= next.display_start);
  return {...next, I:first<0?[]:I.slice(first), V:first<0?[]:V.slice(first), timestamps:first<0?[]:timestamps.slice(first)};
}

export function zoomRange(a: Point, b: Point, view: ViewBounds): ViewBounds | null {
  const clamp = (x:number)=>Math.min(1,Math.max(0,x));
  const x0=Math.min(clamp(a.x),clamp(b.x)), x1=Math.max(clamp(a.x),clamp(b.x));
  const y0=Math.min(clamp(a.y),clamp(b.y)), y1=Math.max(clamp(a.y),clamp(b.y));
  if (x1-x0 < .005 || y1-y0 < .01) return null;
  return {start:view.start+x0*(view.end-view.start), end:view.start+x1*(view.end-view.start),
    fmin:view.fmax-y1*(view.fmax-view.fmin), fmax:view.fmax-y0*(view.fmax-view.fmin)};
}
