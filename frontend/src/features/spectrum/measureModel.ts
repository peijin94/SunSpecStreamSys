import type {Point,ViewBounds,RangeBatch} from './rangeModel';
import type {SpectrumBatch} from '../../api/spectrumProtocol';
import {frequencyChannel} from './frequencyAxis.js';
export interface MeasurePoint {time:number;frequency_mhz:number}
export type DensityModel='saito77'|'leblanc98'|'newkirk';
export interface MeasureOptions {model:DensityModel;harmonic:1|2;density_scale:number}
export interface MeasureSegment extends MeasurePoint {duration_s:number;drift_mhz_s:number;r_sun:number;radial_velocity_km_s:number;speed_km_s:number}
export interface MeasureResult extends MeasureOptions {points:(MeasurePoint&{r_sun:number})[];segments:MeasureSegment[];warnings:string[];assumptions:string[];method:string}
export function canMeasurePoint(data:RangeBatch,point:MeasurePoint):boolean {
 const duration=data.bin_seconds||data.integration_seconds||data.native_cadence_seconds||.5;
 const channel=frequencyChannel(point.frequency_mhz);
 return data.timestamps.some((time,index)=>{
  const end=time+(data.bin_seconds?duration/2:0),begin=end-duration;
  const power=data.I[index]?.[channel];
  return begin<=point.time&&point.time<=end&&power!==null&&power!==undefined&&Number.isFinite(power)&&power>0;
 });
}
export function measurePoint(p:Point,view:ViewBounds):MeasurePoint|null {
 if(p.x<0||p.x>1||p.y<0||p.y>1)return null;
 return {time:view.start+p.x*(view.end-view.start),frequency_mhz:view.fmax-p.y*(view.fmax-view.fmin)};
}

// Pin native pages to the sequence and receiver boot displayed when the tool opened.
// Never use averaged long-range data as the frozen measurement source.
export async function loadFrozenHistory(anchor:RangeBatch,fetchPage:(before:number,boot:string)=>Promise<SpectrumBatch>):Promise<RangeBatch> {
 const pages:SpectrumBatch[]=[];let before=anchor.sequence;
 for(let count=0;count<15;count++) {
  const page=await fetchPage(before,anchor.boot_id);
  if(page.boot_id!==anchor.boot_id||page.reset||!page.history)throw new Error('Receiver changed or snapshot cursor expired; visible range remains frozen.');
  pages.push(page);
  if(!page.has_more)break;
  if(!Number.isSafeInteger(page.next_before)||page.next_before!>=before||page.next_before!<0)
   throw new Error('Invalid frozen history cursor');
  before=page.next_before!;
  if(count===14)throw new Error('Frozen history exceeds bounded receiver capacity');
 }
 const end=anchor.coverage_end??anchor.display_end;
 const rows=new Map<number,{i:SpectrumBatch['I'][number];v:SpectrumBatch['V'][number]}>();
 for(const page of pages)page.timestamps.forEach((time,index)=>{
  if(time>=end-3600&&time<=end)rows.set(time,{i:page.I[index],v:page.V[index]});
 });
 const timestamps=[...rows.keys()].sort((a,b)=>a-b);
 if(!timestamps.length)throw new Error('Pinned history is no longer retained; visible range remains frozen.');
 if(timestamps.length>7201)throw new Error('Frozen history exceeds bounded receiver capacity');
 return {...anchor,timestamps,I:timestamps.map(t=>rows.get(t)!.i),V:timestamps.map(t=>rows.get(t)!.v),
  coverage_start:timestamps[0],coverage_end:timestamps[timestamps.length-1],bin_seconds:0,raw_rows:timestamps.length};
}
export function sliceFrozen(frozen:RangeBatch,view:ViewBounds):RangeBatch {
 const indices=frozen.timestamps.flatMap((time,index)=>time>=view.start&&time<=view.end?[index]:[]);
 return {...frozen,display_start:view.start,display_end:view.end,raw_rows:indices.length,
  timestamps:indices.map(i=>frozen.timestamps[i]),I:indices.map(i=>frozen.I[i]),V:indices.map(i=>frozen.V[i])};
}
