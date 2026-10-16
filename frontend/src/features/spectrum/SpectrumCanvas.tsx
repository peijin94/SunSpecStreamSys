import * as React from 'react';
import type { Detection } from '../../api/client';
import { renderRange } from './rangePixels';
import { zoomRange, type Point, type RangeBatch, type ViewBounds } from './rangeModel';
import type { Polarization } from './spectrumRendering';
import {measurePoint,canMeasurePoint,type MeasurePoint} from './measureModel';
import {frequencyCrop,FREQUENCY_MAX,FREQUENCY_SPAN} from './frequencyAxis';

export interface SpectrumCanvasHandle { setDetections:(detections:Detection[])=>void }
interface Props {
  className?:string; polarization:Polarization; data:RangeBatch|null; view:ViewBounds;
  onPause:()=>void; onZoom:(bounds:ViewBounds)=>void;
  measurePoints?:MeasurePoint[];onMeasurePoint?:(point:MeasurePoint)=>void;
}
interface Engine extends SpectrumCanvasHandle {
  update:(data:RangeBatch|null,view:ViewBounds,onPause:()=>void,onZoom:(bounds:ViewBounds)=>void,points:MeasurePoint[],pick:((p:MeasurePoint)=>void)|undefined)=>void;
  dispose:()=>void;
}

function createEngine(canvas:HTMLCanvasElement,polarization:Polarization):Engine {
  const ctx=canvas.getContext('2d',{alpha:false})!;
  const texture=document.createElement('canvas');
  const tctx=texture.getContext('2d',{alpha:false})!;
  let data:RangeBatch|null=null;
  let view:ViewBounds={start:0,end:300,fmin:15,fmax:85};
  let onPause=()=>{};let onZoom=(_bounds:ViewBounds)=>{void _bounds;};
  let detections:Detection[]=[];
  let picked:MeasurePoint[]=[];let onPick:((p:MeasurePoint)=>void)|undefined;
  let disposed=false;let nextId=0;let frame:number|null=null;let textureValid=false;
  let drag:{a:Point;b:Point;view:ViewBounds;pointer:number}|null=null;
  let worker:Worker|null=null;
  const rect=()=>{
    const r=canvas.getBoundingClientRect();
    return {w:r.width,h:r.height,left:58,top:14,pw:Math.max(1,r.width-74),ph:Math.max(1,r.height-50)};
  };
  function requestPaint() {if(frame===null)frame=requestAnimationFrame(()=>{frame=null;paint();});}
  function paint() {
    if(disposed)return;
    const {w,h,left,top,pw,ph}=rect(),dpr=Math.min(2,window.devicePixelRatio||1);
    if(canvas.width!==Math.round(w*dpr)||canvas.height!==Math.round(h*dpr)) {canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);}
    ctx.setTransform(dpr,0,0,dpr,0,0);ctx.fillStyle='#020617';ctx.fillRect(0,0,w,h);
    ctx.imageSmoothingEnabled=false;
    const {top:sourceY,height:sourceH}=frequencyCrop(view.fmin,view.fmax);
    if(textureValid&&texture.width>0&&texture.height>0)ctx.drawImage(texture,0,sourceY,texture.width,sourceH,left,top,pw,ph);
    ctx.font='11px system-ui, sans-serif';ctx.strokeStyle='rgba(148,163,184,.15)';ctx.fillStyle='#94a3b8';ctx.lineWidth=1;
    const steps=4;
    for(let i=0;i<=steps;i++) {
      const y=top+ph*i/steps;
      ctx.textAlign='right';ctx.fillText((view.fmax-(view.fmax-view.fmin)*i/steps).toFixed(1),left-8,y+4);
      ctx.beginPath();ctx.moveTo(left,y);ctx.lineTo(left+pw,y);ctx.stroke();
    }
    ctx.save();ctx.translate(13,top+ph/2);ctx.rotate(-Math.PI/2);ctx.textAlign='center';ctx.fillText('Frequency (MHz)',0,0);ctx.restore();
    const ticks=pw<420?2:4;
    for(let i=0;i<=ticks;i++) {
      const x=left+pw*i/ticks,time=view.start+(view.end-view.start)*i/ticks;
      ctx.textAlign=i===0?'left':i===ticks?'right':'center';
      ctx.fillText(time>0?new Date(time*1000).toISOString().slice(11,19)+(i===ticks?' UTC':''):'—',x,h-13);
      ctx.beginPath();ctx.moveTo(x,top);ctx.lineTo(x,top+ph);ctx.stroke();
    }
    ctx.save();ctx.beginPath();ctx.rect(left,top,pw,ph);ctx.clip();
    if(polarization==='I' && data)for(const d of detections) {
      if(d.boot_id!==data.boot_id||!Number.isFinite(d.time_start)||!Number.isFinite(d.time_end))continue;
      const [start,end]=[d.time_start!,d.time_end!];
      if(end<view.start||start>view.end)continue;
      const x=left+(start-view.start)/(view.end-view.start)*pw;
      const fmax=FREQUENCY_MAX-d.bbox[1]*FREQUENCY_SPAN, fmin=FREQUENCY_MAX-(d.bbox[1]+d.bbox[3])*FREQUENCY_SPAN;
      const y=top+(view.fmax-fmax)/(view.fmax-view.fmin)*ph;
      ctx.strokeStyle=d.class_id===0?'#f87171':'#f8fafc';ctx.fillStyle=ctx.strokeStyle;ctx.lineWidth=1.5;
      ctx.strokeRect(x,y,(end-start)/(view.end-view.start)*pw,(fmax-fmin)/(view.fmax-view.fmin)*ph);
      ctx.textAlign='left';ctx.fillText(`${d.class} ${Math.round(d.confidence*100)}%`,x+3,Math.max(top+12,y-4));
    }
    if(picked.length) {
      const ordered=[...picked].sort((a,b)=>a.time-b.time);
      ctx.strokeStyle='#fbbf24';ctx.fillStyle='#fbbf24';ctx.lineWidth=1.5;ctx.beginPath();
      ordered.forEach((p,i)=>{const x=left+(p.time-view.start)/(view.end-view.start)*pw,y=top+(view.fmax-p.frequency_mhz)/(view.fmax-view.fmin)*ph;if(i)ctx.lineTo(x,y);else ctx.moveTo(x,y);});ctx.stroke();
      picked.forEach((p,i)=>{const x=left+(p.time-view.start)/(view.end-view.start)*pw,y=top+(view.fmax-p.frequency_mhz)/(view.fmax-view.fmin)*ph;
        ctx.beginPath();ctx.arc(x,y,4,0,Math.PI*2);ctx.fill();ctx.fillText(String(i+1),x+7,y-6);});
    }
    if(drag) {
      const x=left+Math.min(drag.a.x,drag.b.x)*pw,y=top+Math.min(drag.a.y,drag.b.y)*ph;
      const width=Math.abs(drag.a.x-drag.b.x)*pw,height=Math.abs(drag.a.y-drag.b.y)*ph;
      ctx.fillStyle='rgba(56,189,248,.16)';ctx.fillRect(x,y,width,height);
      ctx.strokeStyle='#38bdf8';ctx.lineWidth=1.5;ctx.setLineDash([5,3]);ctx.strokeRect(x,y,width,height);
    }
    ctx.restore();
    if(!data || !data.I.length) {ctx.fillStyle='#cbd5e1';ctx.textAlign='center';ctx.fillText(data?'No retained data in this range':'Loading spectrum…',left+pw/2,top+ph/2);}
    canvas.dataset.measureCount=String(picked.length);
    canvas.dataset.start=String(view.start);canvas.dataset.end=String(view.end);
    canvas.dataset.fmin=String(view.fmin);canvas.dataset.fmax=String(view.fmax);
  }
  function accept(result:{id:number;pixels:Uint8ClampedArray;width:number;height:number}) {
    if(disposed||result.id!==nextId)return;
    textureValid=true;texture.width=result.width;texture.height=result.height;
    tctx.putImageData(new ImageData(new Uint8ClampedArray(result.pixels),result.width,result.height),0,0);
    requestPaint();
  }
  function render() {
    if(!data)return requestPaint();
    const width=Math.min(1200,Math.max(2,Math.ceil((view.end-view.start)/(data.bin_seconds||data.native_cadence_seconds||.5))*2));
    const id=++nextId;
    if(worker)worker.postMessage({id,data,view,width,polarization});
    else accept({id,...renderRange(data,view,width,polarization)});
  }
  try {
    worker=new Worker(new URL('./range.worker.ts',import.meta.url),{type:'module'});
    worker.onmessage=e=>accept(e.data);
    worker.onerror=()=>{worker?.terminate();worker=null;render();};
  } catch {worker=null;}
  const position=(e:PointerEvent):Point=>{
    const r=canvas.getBoundingClientRect(),p=rect();
    return {x:(e.clientX-r.left-p.left)/p.pw,y:(e.clientY-r.top-p.top)/p.ph};
  };
  function down(e:PointerEvent) {
    if(!data||e.button!==0||drag)return;
    const p=position(e);if(p.x<0||p.x>1||p.y<0||p.y>1)return;
    e.preventDefault();canvas.setPointerCapture(e.pointerId);
    drag={a:p,b:p,view:{...view},pointer:e.pointerId};
    onPause();requestPaint();
  }
  function move(e:PointerEvent) {
    if(!drag||drag.pointer!==e.pointerId)return;
    const p=position(e);drag.b={x:Math.max(0,Math.min(1,p.x)),y:Math.max(0,Math.min(1,p.y))};requestPaint();
  }
  function up(e:PointerEvent) {
    if(!drag||drag.pointer!==e.pointerId)return;
    move(e);const finished=drag;
    const geometry=rect();const click=Math.hypot((finished.a.x-finished.b.x)*geometry.pw,(finished.a.y-finished.b.y)*geometry.ph)<=5;
    const selected=click&&onPick?null:zoomRange(finished.a,finished.b,finished.view);drag=null;
    if(click&&onPick&&data){const point=measurePoint(finished.a,finished.view);if(point&&canMeasurePoint(data,point))onPick(point);}
    if(canvas.hasPointerCapture(e.pointerId))canvas.releasePointerCapture(e.pointerId);
    if(selected && selected.end-selected.start>=1 && selected.fmax-selected.fmin>=.1)onZoom(selected);
    requestPaint();
  }
  function cancel(){drag=null;requestPaint();}
  function key(e:KeyboardEvent){if(e.key==='Escape')cancel();}
  canvas.addEventListener('pointerdown',down);canvas.addEventListener('pointermove',move);
  canvas.addEventListener('pointerup',up);canvas.addEventListener('pointercancel',cancel);canvas.addEventListener('keydown',key);
  const observer=new ResizeObserver(requestPaint);observer.observe(canvas);requestPaint();
  return {
    update(next,nextView,pause,zoom,points,pick) {
      onPause=pause;onZoom=zoom;onPick=pick;
      const overlayChanged=picked!==points;picked=points;
      const changed=data!==next || JSON.stringify(view)!==JSON.stringify(nextView);
      if(JSON.stringify(view)!==JSON.stringify(nextView))textureValid=false;
      data=next;view=nextView;
      if(changed)render();else if(overlayChanged)requestPaint();
    },
    setDetections(next) {detections=next;requestPaint();},
    dispose() {
      disposed=true;worker?.terminate();observer.disconnect();if(frame!==null)cancelAnimationFrame(frame);
      canvas.removeEventListener('pointerdown',down);canvas.removeEventListener('pointermove',move);
      canvas.removeEventListener('pointerup',up);canvas.removeEventListener('pointercancel',cancel);canvas.removeEventListener('keydown',key);
    },
  };
}

export const SpectrumCanvas=React.forwardRef<SpectrumCanvasHandle,Props>(({className,polarization,data,view,onPause,onZoom,measurePoints=[],onMeasurePoint},ref)=>{
  const canvasRef=React.useRef<HTMLCanvasElement|null>(null),engine=React.useRef<Engine|null>(null);
  React.useEffect(()=>{if(!canvasRef.current)return;engine.current=createEngine(canvasRef.current,polarization);return()=>{engine.current?.dispose();engine.current=null;};},[polarization]);
  React.useEffect(()=>{engine.current?.update(data,view,onPause,onZoom,measurePoints,onMeasurePoint);},[data,view,onPause,onZoom,measurePoints,onMeasurePoint]);
  React.useImperativeHandle(ref,()=>({setDetections:(items)=>engine.current?.setDetections(items)}),[]);
  return <canvas ref={canvasRef} className={className} data-testid={`spectrum-canvas-${polarization}`}
    style={{touchAction:'none',cursor:'crosshair'}} tabIndex={0} role="img"
    aria-label={`Stokes ${polarization==='I'?'I intensity':'V / I fraction'} spectrum. ${onMeasurePoint?'Click to add measurement points. ':''}Drag to select time and frequency and pause. Escape cancels selection.`}/>;
});
SpectrumCanvas.displayName='SpectrumCanvas';
