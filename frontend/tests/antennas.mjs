import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {parseAntennaFlags, selectAntennaFlags} from '../.test-build/api/antennaProtocol.js';

const layout = JSON.parse(await readFile(new URL('../src/features/antennas/layout.json', import.meta.url), 'utf8'));
const payload = () => ({status:'ready', antenna_count:352, id_system:'ms_antenna_index',
  observation_time:'2026-10-08T12:00:00Z', poll_interval_seconds:60, message:null,
  bands:[{freq_mhz:30,ant_list:[0],n_bad_ant:1,flagged_frac:.1},
    {freq_mhz:60,ant_list:[0,351],n_bad_ant:2,flagged_frac:.2}]});

test('template preserves correlator mapping rather than stand-number ordering', () => {
  assert.deepEqual(layout.antennas.map(a=>a.id),Array.from({length:352},(_,i)=>i));
  assert.equal(layout.antennas[0].name,'LWA-266');
  assert.equal(layout.antennas[65].name,'LWA-001');
  assert.equal(layout.antennas[351].name,'LWA-304');
  assert.ok(layout.antennas.every(a=>Number.isFinite(a.east_m)&&Number.isFinite(a.north_m)));
});

test('any-band union and chosen band keep zero-based IDs', () => {
  const data=parseAntennaFlags(payload());
  assert.deepEqual([...selectAntennaFlags(data.bands,null).badIds],[0,351]);
  assert.deepEqual([...selectAntennaFlags(data.bands,30).badIds],[0]);
  assert.equal(selectAntennaFlags(data.bands,null).complete,true);
});

test('empty list means green; missing list or missing band means unknown', () => {
  const data=payload(); data.bands[0].ant_list=[]; data.bands[0].n_bad_ant=0;
  assert.equal(selectAntennaFlags(parseAntennaFlags(data).bands,30).complete,true);
  data.bands[0].ant_list=null;
  assert.equal(selectAntennaFlags(parseAntennaFlags(data).bands,null).complete,false);
  assert.deepEqual([...selectAntennaFlags(data.bands,null).badIds],[0,351]);
  assert.equal(selectAntennaFlags(data.bands,45).complete,false);
});

test('wrong schema, out-of-range IDs, and malformed lists are rejected', () => {
  for(const change of [d=>d.id_system='stand_number',d=>d.antenna_count=256,
    d=>d.bands[0].ant_list=[352],d=>d.bands[0].ant_list=[-1],d=>d.bands[0].ant_list=[true],
    d=>d.bands[0].ant_list=[0,0],d=>d.bands[0].ant_list='0',d=>d.observation_time=null]) {
    const data=payload();change(data);assert.throws(()=>parseAntennaFlags(data));
  }
});
