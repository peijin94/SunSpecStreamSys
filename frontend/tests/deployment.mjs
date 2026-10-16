import test from 'node:test';
import assert from 'node:assert/strict';
import { pathToFileURL } from 'node:url';
import path from 'node:path';

const buildDir = path.resolve(process.env.DEPLOYMENT_TEST_BUILD || '.test-build');
const appMode = await import(pathToFileURL(path.join(buildDir, 'lib/appMode.js')).href);
const basePath = await import(pathToFileURL(path.join(buildDir, 'lib/basePath.js')).href);

test('app mode defaults to dev and only public opts into the fixed page', () => {
  assert.equal(appMode.resolveAppMode('public'), 'public');
  assert.equal(appMode.resolveAppMode('dev'), 'dev');
  assert.equal(appMode.resolveAppMode(undefined), 'dev');
  assert.equal(appMode.resolveAppMode(null), 'dev');
  assert.equal(appMode.resolveAppMode(''), 'dev');
  assert.equal(appMode.resolveAppMode('PUBLIC'), 'dev');
  // Default build (no VITE_APP_MODE) preserves the full dev UI and old tests.
  assert.equal(appMode.APP_MODE, 'dev');
  assert.equal(appMode.IS_PUBLIC, false);
});

test('public contract is a fixed rolling 300 s full-band window', () => {
  assert.equal(appMode.PUBLIC_SECONDS, 300);
  assert.equal(appMode.PUBLIC_FMIN, 15);
  assert.equal(appMode.PUBLIC_FMAX, 85);
});

test('interaction gates forbid view/measurement changes in public mode', () => {
  assert.equal(appMode.isInteractiveMode('public'), false);
  assert.equal(appMode.isInteractiveMode('dev'), true);
  assert.equal(appMode.canChangeView('public'), false);
  assert.equal(appMode.canChangeView('dev'), true);
  assert.equal(appMode.canMeasure('public'), false);
  assert.equal(appMode.canMeasure('dev'), true);
});

test('production base defaults to /live/ and accepts /live-dev/', () => {
  assert.equal(basePath.normalizeBasePath(undefined, '/live/'), '/live/');
  assert.equal(basePath.normalizeBasePath('/live/', '/live/'), '/live/');
  assert.equal(basePath.normalizeBasePath('/live-dev/', '/live/'), '/live-dev/');
  assert.equal(basePath.normalizeBasePath('/live-dev', '/live/'), '/live-dev/');
  assert.equal(basePath.normalizeBasePath('live-dev/', '/live/'), '/live-dev/');
  assert.equal(basePath.normalizeBasePath('/', '/live/'), '/');
});

test('every API/asset/image URL carries the deployment base', () => {
  assert.equal(
    basePath.joinBasePath('/live/', '/spectrum-range?format=binary&seconds=300'),
    '/live/spectrum-range?format=binary&seconds=300',
  );
  assert.equal(
    basePath.joinBasePath('/live-dev/', '/spectrum-range?format=binary&seconds=300'),
    '/live-dev/spectrum-range?format=binary&seconds=300',
  );
  assert.equal(basePath.joinBasePath('/', '/data'), '/data');
  assert.equal(
    basePath.joinBasePath('/live-dev/', '/radio-images/2026-01-01/30mhz.png'),
    '/live-dev/radio-images/2026-01-01/30mhz.png',
  );
  assert.equal(
    basePath.buildApiUrl('/spectrum-range?format=binary', '/live-dev/', ''),
    '/live-dev/spectrum-range?format=binary',
  );
  assert.equal(
    basePath.buildApiUrl('/data', '/live/', 'https://ovsa.njit.edu'),
    'https://ovsa.njit.edu/live/data',
  );
});
