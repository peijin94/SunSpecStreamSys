import test from 'node:test';
import assert from 'node:assert/strict';
import {renderRange} from '../.test-build/features/spectrum/rangePixels.js';
import {mergeRange, zoomRange, rangeQuery} from '../.test-build/features/spectrum/rangeModel.js';

test('native deltas retain selected real-time span and matched I/V',()=>{
 const base={I:[[1],[2]],V:[[-1],[-2]],timestamps:[100,101],boot_id:'a',reset:true,display_start:99,display_end:101};
 const next={...base,I:[[3]],V:[[-3]],timestamps:[102],reset:false,display_start:101,display_end:102};
 const result=mergeRange(base,next);
 assert.deepEqual(result.timestamps,[101,102]);
 assert.deepEqual(result.V,[[-2],[-3]]);
 assert.deepEqual(mergeRange(base,{...next,boot_id:'b'}).timestamps,[102]);
});
test('zoom is bidirectional, clamped to plot and preserves time/frequency coordinates',()=>{
 const bounds={start:100,end:400,fmin:15,fmax:85};
 assert.deepEqual(zoomRange({x:.8,y:.8},{x:.2,y:.2},bounds),{start:160,end:340,fmin:29,fmax:71});
 assert.equal(zoomRange({x:.2,y:.2},{x:.2,y:.8},bounds),null);
 assert.deepEqual(zoomRange({x:-2,y:-3},{x:2,y:2},bounds),bounds);
});
test('raster keeps real gaps blank and uses matched circular fraction',()=>{
 const frame=Array(768).fill(10000), v=Array(768).fill(-6000);
 const data={I:[frame,frame],V:[v,v],timestamps:[1,9],native_cadence_seconds:1,bin_seconds:0};
 const result=renderRange(data,{start:0,end:10,fmin:15,fmax:85},100,'V');
 assert.equal(result.pixels[(40*100+50)*4],0,'five-second gap is black');
 assert.equal(result.pixels[(40*100+5)*4+2],235,'negative V/I is blue');
});

test('native integration duration does not bridge repeated dropped frames',()=>{
 const values=Array(768).fill(10000);
 const data={I:[values,values,values],V:[values,values,values],timestamps:[1.5,3,4.5],
   native_cadence_seconds:1.5,integration_seconds:.5,bin_seconds:0};
 const result=renderRange(data,{start:0,end:5,fmin:15,fmax:85},100,'I');
 assert.equal(result.pixels[(40*100+40)*4],0,'missing integration at two seconds stays blank');
 assert.notEqual(result.pixels[(40*100+27)*4],0,'received integration remains visible');
});

test('range queries request only selected window,3min native and1hour reduced',()=>{
 assert.equal(rangeQuery({seconds:180}, {sequence:4,boot_id:'a'}),'format=binary&seconds=180&max_rows=1200&since=4&boot_id=a');
 assert.equal(rangeQuery({seconds:3600}, {sequence:4,boot_id:'a'}),'format=binary&seconds=3600&max_rows=600');
 assert.equal(rangeQuery({seconds:30,start:100,end:130}),'format=binary&start=100&end=130&max_rows=1200');
});
