import * as React from 'react';
import {fetchSpectrumMeasurement} from '../../api/client';
import type {DensityModel,MeasurePoint,MeasureResult} from './measureModel';
import { IS_PUBLIC } from '../../lib/appMode';

export function useMeasurement(active:boolean) {
 const [points,setPoints]=React.useState<MeasurePoint[]>([]);
 const [model,setModel]=React.useState<DensityModel>('saito77');
 const [harmonic,setHarmonic]=React.useState<1|2>(1);
 const [densityScale,setDensityScale]=React.useState('1');
 const [pickError,setPickError]=React.useState('');
 const [response,setResponse]=React.useState<{key:string;result?:MeasureResult;error?:string}|null>(null);
 const scale=Number(densityScale);
 const invalid=!Number.isFinite(scale)||scale<.01||scale>100?'Density multiplier must be between 0.01 and 100.':'';
 const key=JSON.stringify({points,model,harmonic,scale});
 React.useEffect(()=>{
  if(IS_PUBLIC||!active||points.length<2||invalid)return;
  const controller=new AbortController();
  const timer=setTimeout(()=>{
   void fetchSpectrumMeasurement(points,{model,harmonic,density_scale:scale},controller.signal).then(result=>{
    if(!controller.signal.aborted)setResponse({key,result});
   }).catch(error=>{if(!controller.signal.aborted)setResponse({key,error:error instanceof Error?error.message:'Measurement failed'});});
  },180);
  return ()=>{clearTimeout(timer);controller.abort();};
 },[active,points,model,harmonic,scale,invalid,key]);
 const result=response?.key===key?response.result:null;
 const error=invalid||pickError||(response?.key===key?response.error:'')||'';
 function addPoint(point:MeasurePoint) {
  if(IS_PUBLIC||!active)return;
  if(points.length>=128){setPickError('Maximum 128 points. Undo or clear points to continue.');return;}
  if(points.some(p=>Math.abs(p.time-point.time)<1e-6)){setPickError('Choose a different time; two frequencies at the same time have undefined drift.');return;}
  setPickError('');setPoints(previous=>[...previous,point]);
 }
 function clear(){setPoints([]);setPickError('');}
 function undo(){setPoints(previous=>previous.slice(0,-1));setPickError('');}
 function remove(index:number){setPoints(previous=>previous.filter((_,i)=>i!==index));setPickError('');}
 return {points,addPoint,clear,undo,remove,model,setModel,harmonic,setHarmonic,densityScale,setDensityScale,
  result,error,calculating:active&&points.length>=2&&!invalid&&!result&&response?.key!==key};
}
