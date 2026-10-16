import {renderRange} from './rangePixels';
import type {RangeBatch,ViewBounds} from './rangeModel';
import type {Polarization} from './spectrumRendering';
self.onmessage=(event:MessageEvent<{id:number;data:RangeBatch;view:ViewBounds;width:number;polarization:Polarization}>)=>{
 const {id,data,view,width,polarization}=event.data;
 const result=renderRange(data,view,width,polarization);
 self.postMessage({id,...result}, {transfer:[result.pixels.buffer]});
};
