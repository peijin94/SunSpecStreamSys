import assert from 'node:assert/strict';
import { resolveConfig } from 'vite';

// Exercise Vite itself: .env.production is not in process.env while config is
// evaluated. A build must work using the tracked branch defaults alone.
const original=process.env.VITE_BASE_PATH;
try {
  delete process.env.VITE_BASE_PATH;
  const branch=await resolveConfig({},'build','production');
  assert.ok(branch.env.VITE_BASE_PATH,'branch must declare its nonsecret base path');
  assert.equal(branch.base,branch.env.VITE_BASE_PATH,'Vite base must load the branch .env.production before config evaluation');
  process.env.VITE_BASE_PATH='/live/';
  const publicOverride=await resolveConfig({},'build','production');
  assert.equal(publicOverride.base,'/live/','explicit build environment overrides branch defaults');
  process.env.VITE_BASE_PATH='/live-dev/';
  const devOverride=await resolveConfig({},'build','production');
  assert.equal(devOverride.base,'/live-dev/');
  console.log('PASS Vite branch defaults and explicit deployment base overrides');
} finally {
  if(original===undefined)delete process.env.VITE_BASE_PATH;
  else process.env.VITE_BASE_PATH=original;
}
