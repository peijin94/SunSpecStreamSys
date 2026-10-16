import test from 'node:test';
import assert from 'node:assert/strict';
import {pathToFileURL} from 'node:url';
import path from 'node:path';
const modulePath=path.resolve(process.env.MEASURE_TEST_BUILD||'.test-build','features/spectrum/measureModel.js');
let model;
try {model=await import(pathToFileURL(modulePath).href);}catch{}
test('measurement point maps pointer coordinates to shared UTC and frequency view',()=>{
 assert.ok(model?.measurePoint,'measurement coordinate model must exist');
 assert.deepEqual(model.measurePoint({x:.25,y:.2},{start:1000,end:1100,fmin:20,fmax:70}),{time:1025,frequency_mhz:60});
 assert.equal(model.measurePoint({x:-.1,y:.2},{start:1000,end:1100,fmin:20,fmax:70}),null);
});
const anchor={range:true,display_start:1007,display_end:1010,coverage_start:1007,coverage_end:1010,
 native_cadence_seconds:1,integration_seconds:1,bin_seconds:2,raw_rows:4,sequence:8,first_sequence:5,
 boot_id:'boot',reset:true,buffer_length:7201,buffer_index:8,timestamp:1010,
 timestamps:[1008,1010],I:[[99],[99]],V:[[9],[9]]};
test('frozen native history uses pinned sequence and boot, not averaged/future integrations',async()=>{
 assert.ok(model?.loadFrozenHistory,'frozen native snapshot loader must exist');
 const calls=[];
 const frozen=await model.loadFrozenHistory(anchor,async(before,boot)=>{
  calls.push([before,boot]);const latest=before===8;
  return {...anchor,range:undefined,history:true,reset:false,has_more:latest,next_before:latest?6:null,
   timestamps:latest?[1009,1010]:[1007,1008],I:latest?[[3],[4]]:[[1],[2]],V:latest?[[-3],[-4]]:[[-1],[-2]]};
 });
 assert.deepEqual(calls,[[8,'boot'],[6,'boot']]);
 assert.deepEqual(frozen.timestamps,[1007,1008,1009,1010]);assert.equal(frozen.bin_seconds,0);
 assert.deepEqual(frozen.I.map(r=>r[0]),[1,2,3,4]);
 const selected=model.sliceFrozen(frozen,{start:1008.5,end:1009.5,fmin:20,fmax:70});
 assert.deepEqual(selected.timestamps,[1009]);assert.equal(selected.coverage_end,1010);
 assert.deepEqual(frozen.timestamps,[1007,1008,1009,1010],'changing ranges cannot mutate frozen data');
});
test('measurement frequencies follow actual four-channel averaged correlator bins, not a stretched 15–85 raster',()=>{
 const frequency=(600+4*300+1.5)*196/8192;
 const row=new Float32Array(768);row[300]=1;
 const data={...anchor,bin_seconds:0,integration_seconds:.5,timestamps:[1000],I:[row]};
 assert.equal(model.canMeasurePoint(data,{time:999.75,frequency_mhz:frequency}),true);
});
test('picking rejects actual gaps and non-finite integrations',()=>{
 assert.ok(model?.canMeasurePoint,'Gap-aware point picking must exist');
 const data={...anchor,bin_seconds:0,integration_seconds:.5,timestamps:[1000,1002],I:[new Float32Array(768).fill(1),new Float32Array(768).fill(1)]};
 assert.equal(model.canMeasurePoint(data,{time:1001,frequency_mhz:30}),false);
 assert.equal(model.canMeasurePoint(data,{time:1001.75,frequency_mhz:30}),true);
 data.I[1].fill(NaN);assert.equal(model.canMeasurePoint(data,{time:1001.75,frequency_mhz:30}),false);
});
test('snapshot acquisition rejects a restarted receiver or ignored cursor',async()=>{
 await assert.rejects(()=>model.loadFrozenHistory(anchor,async()=>({...anchor,boot_id:'new',history:false,reset:true})),/changed|restart|cursor/i);
});
