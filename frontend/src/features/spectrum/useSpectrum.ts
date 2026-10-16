import * as React from 'react';
import { fetchSpectrumRange, type SpectrumCursor } from '../../api/client';
import { FULL_BAND, mergeRange, rangeQuery, type QueryRange, type RangeBatch, type ViewBounds } from './rangeModel';

interface Selection extends QueryRange { fmin: number; fmax: number; reason: 'view' | 'pause' }
const initial: Selection = {seconds:300,...FULL_BAND,reason:'view'};

export function useSpectrumPolling() {
  const [selection,setSelection] = React.useState<Selection>(initial);
  const [playing,setPlaying] = React.useState(true);
  const [presetSeconds,setPresetSeconds] = React.useState(300);
  const [data,setData] = React.useState<RangeBatch|null>(null);
  const [status,setStatus] = React.useState<'loading'|'error'|'ok'>('loading');
  const [refreshing,setRefreshing] = React.useState(false);
  const [revision,setRevision] = React.useState(0);
  const controllerRef = React.useRef<AbortController|null>(null);
  const generationRef = React.useRef(0);
  const dataRef = React.useRef<RangeBatch|null>(null);

  const stopRequests = React.useCallback(()=>{
    generationRef.current++;
    controllerRef.current?.abort();
  },[]);

  React.useEffect(()=>{
    if (!playing && selection.reason === 'pause') return;
    const controller = new AbortController(); controllerRef.current = controller;
    const generation = ++generationRef.current;
    let timer: ReturnType<typeof setTimeout>|undefined;
    let cursor: SpectrumCursor|undefined;
    let current: RangeBatch|null = null;
    let busy = false;
    async function tick() {
      if (controller.signal.aborted || busy || document.hidden) return;
      busy = true;
      try {
        const next = await fetchSpectrumRange(rangeQuery(selection,cursor),controller.signal);
        if (controller.signal.aborted || generation !== generationRef.current) return;
        current = mergeRange(current,next);
        dataRef.current = current;
        setData(current); setStatus('ok'); setRefreshing(false);
        cursor = {sequence:next.sequence,boot_id:next.boot_id};
      } catch {
        if (!controller.signal.aborted && generation === generationRef.current) {setStatus('error');setRefreshing(false);}
      } finally {
        busy = false;
        if (playing && !controller.signal.aborted && generation === generationRef.current)
          timer = setTimeout(tick,selection.seconds<=300 ? 512 : 5000);
      }
    }
    function visible() {if(timer)clearTimeout(timer);if(!document.hidden && (playing || !current))void tick();}
    void tick(); document.addEventListener('visibilitychange',visible);
    return ()=>{controller.abort();if(timer)clearTimeout(timer);document.removeEventListener('visibilitychange',visible);};
  },[selection,playing,revision]);

  const pause = React.useCallback(()=>{
    stopRequests();
    const current = dataRef.current;
    setPlaying(false); setRefreshing(false);
    setSelection(s=>{const end=s.end??current?.display_end;return {...s, start:s.start??(end!==undefined?end-s.seconds:undefined),end,reason:'pause'};});
  },[stopRequests]);

  function preset(seconds:number) {
    setPresetSeconds(seconds);
    stopRequests(); setRefreshing(true);
    const end = selection.end ?? dataRef.current?.display_end;
    setSelection({...FULL_BAND,seconds,...(!playing && end!==undefined?{start:end-seconds,end}:{}),reason:'view'});
  }
  function zoom(bounds:ViewBounds) {
    stopRequests();setPlaying(false);setRefreshing(true);
    setSelection({...bounds,seconds:bounds.end-bounds.start,reason:'view'});
  }
  function play() {
    stopRequests();setPlaying(true);setRefreshing(true);
    setSelection(s=>({seconds:s.seconds,fmin:s.fmin,fmax:s.fmax,reason:'view'}));
  }
  function live() {
    setPresetSeconds(180);
    stopRequests();setPlaying(true);setRefreshing(true);
    setSelection({...FULL_BAND,seconds:180,reason:'view'});
  }
  function refresh() {stopRequests();setRefreshing(true);setSelection(s=>({...s,reason:'view'}));setRevision(n=>n+1);}
  const end = selection.end ?? data?.display_end ?? 0;
  const view = {start:selection.start ?? end-selection.seconds,end,fmin:selection.fmin,fmax:selection.fmax};
  const lastStamp = data?.coverage_end;
  return {data,view,playing,seconds:selection.seconds,status,refreshing,
    frameCount:data?.raw_rows??0, historyLoading:refreshing,
    lastUpdate:lastStamp?new Date(lastStamp*1000).toISOString().slice(11,19)+' (UTC)':null,
    pause,play,live,preset,zoom,refresh,resetZoom:()=>preset(presetSeconds)};
}
