import test from 'node:test';
import assert from 'node:assert/strict';
import {parseImagingStatus,selectImagingBand} from '../.test-build/api/imagingProtocol.js';

const payload=()=>({schema:'lwa-imaging-status-v1',status:'ready',observation_time:'2026-10-08T12:00:00Z',
  bands:[{freq_mhz:18,flagged_frac:0},{freq_mhz:59,flagged_frac:.25},{freq_mhz:64,flagged_frac:null}],
  flux_rows:[{freq_mhz:59,source:'CasA',measured_jy:200,expected_jy:100,ratio:2}],
  flux_status:'ready',flux_message:null,message:null,poll_interval_seconds:60});

test('default band is nearest 60 MHz and user selection remains exact',()=>{
 const bands=parseImagingStatus(payload()).bands;
 assert.equal(selectImagingBand(bands,null).freq_mhz,59);
 assert.equal(selectImagingBand(bands,18).flagged_frac,0);
 assert.equal(selectImagingBand(bands,64).flagged_frac,null);
 assert.equal(selectImagingBand([],null),null);
 assert.equal(selectImagingBand(bands,75).freq_mhz,59);
});
test('missing flux measurements stay absent while baseline information remains',()=>{
 const data=payload();data.flux_rows=[];data.flux_status='unavailable';
 assert.equal(parseImagingStatus(data).bands[1].flagged_frac,.25);
 assert.deepEqual(parseImagingStatus(data).flux_rows,[]);
});
test('invalid fractions, nonfinite fluxes, duplicate rows, and missing observation time are rejected',()=>{
 for(const change of [d=>d.bands[0].flagged_frac=1.1,d=>d.bands[0].flagged_frac=NaN,
   d=>d.flux_rows[0].measured_jy=Infinity,d=>d.flux_rows.push({...d.flux_rows[0]}),
   d=>d.observation_time=null,d=>d.schema='unknown']){
  const data=payload();change(data);assert.throws(()=>parseImagingStatus(data));
 }
});
