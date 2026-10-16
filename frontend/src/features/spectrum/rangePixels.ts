import { COLOR_TABLES, colorIndex, displayValue, type Polarization } from './spectrumRendering.js';
import type { RangeBatch, ViewBounds } from './rangeModel';

type Samples = Pick<RangeBatch,'I'|'V'|'timestamps'|'bin_seconds'|'native_cadence_seconds'|'integration_seconds'>;
export function renderRange(data:Samples, view:ViewBounds, width:number, polarization:Polarization) {
  const height=768;
  const pixels=new Uint8ClampedArray(width*height*4);
  for(let i=3;i<pixels.length;i+=4)pixels[i]=255;
  const span=view.end-view.start;
  if(span<=0)return {pixels,width,height};
  const column=(time:number)=>Math.floor((time-view.start)/span*width);
  const table=COLOR_TABLES[polarization];
  data.timestamps.forEach((stamp,row)=>{
    const duration=data.bin_seconds || data.integration_seconds || data.native_cadence_seconds || .5;
    const lo=data.bin_seconds?stamp-duration/2:stamp-duration;
    const hi=data.bin_seconds?stamp+duration/2:stamp;
    const a=Math.max(0,column(lo)), b=Math.min(width,Math.max(column(hi),column(lo)+1));
    if(a>=b)return;
    for(let frequency=0;frequency<height;frequency++) {
      const i=data.I[row]?.[frequency]??null;
      const v=polarization==='I'?i:(data.V[row]?.[frequency]??null);
      const color=colorIndex(displayValue(v,polarization,i),polarization);
      if(color<0)continue;
      const y=height-1-frequency;
      for(let x=a;x<b;x++) {
        const offset=(y*width+x)*4;
        pixels[offset]=table[color*4];pixels[offset+1]=table[color*4+1];pixels[offset+2]=table[color*4+2];
      }
    }
  });
  return {pixels,width,height};
}
