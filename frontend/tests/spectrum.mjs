import test from 'node:test';
import assert from 'node:assert/strict';
import { decodeSpectrum } from '../.test-build/api/spectrumProtocol.js';
import { SpectrumRaster } from '../.test-build/features/spectrum/spectrumRaster.js';
import { detectionGeometry } from '../.test-build/features/spectrum/detectionGeometry.js';
import { colorIndex, displayValue, COLOR_TABLES, timeSlot, firstCoveredSlot, TIME_COLUMNS, TIME_WINDOW_SECONDS } from '../.test-build/features/spectrum/spectrumRendering.js';

function packet() {
  const metadata = { schema: 'lwa-spectrum-iv-v1', rows: 2, bins: 768,
    dtype: 'float32', byteorder: 'little', order: ['I', 'V'], sequence: 42,
    first_sequence: 41, boot_id: 'receiver-one', reset: true,
    buffer_length: 600, buffer_index: 42, timestamp: 1720000000 };
  const header = new TextEncoder().encode(JSON.stringify(metadata));
  const offset = Math.ceil((4 + header.length) / 8) * 8;
  const buffer = new ArrayBuffer(offset + 2 * (8 + 768 * 8));
  new DataView(buffer).setUint32(0, offset - 4, true);
  new Uint8Array(buffer, 4, offset - 4).fill(32);
  new Uint8Array(buffer, 4, header.length).set(header);
  new Float64Array(buffer, offset, 2).set([1720000000, 1720000000.5]);
  new Float32Array(buffer, offset + 16, 1536).fill(3e6);
  const v = new Float32Array(buffer, offset + 16 + 1536 * 4, 1536);
  v.fill(-1e6, 0, 768); v.fill(1e6, 768);
  return buffer;
}

test('binary snapshots keep timestamps, signed V, and zero-copy rows together', () => {
  const buffer = packet();
  const data = decodeSpectrum(buffer);
  assert.deepEqual(data.timestamps, [1720000000, 1720000000.5]);
  assert.equal(data.I[0][767], 3e6);
  assert.equal(data.V[0][767], -1e6);
  assert.equal(data.V[1][0], 1e6);
  assert.equal(data.V[0].buffer, buffer);
  assert.equal(data.I[1].buffer, buffer);
  assert.equal(data.sequence, 42);
  assert.equal(data.reset, true);
});

test('truncated, misaligned, and incompatible packets are rejected', () => {
  const buffer = packet();
  assert.throws(() => decodeSpectrum(buffer.slice(0, -1)));
  assert.throws(() => decodeSpectrum(new ArrayBuffer(3)));
  const invalid = packet();
  new DataView(invalid).setUint32(0, 1, true);
  assert.throws(() => decodeSpectrum(invalid));
  const wrong = packet();
  const bytes = new Uint8Array(wrong);
  const text = new TextDecoder().decode(bytes);
  bytes[text.indexOf('float32') + 5] = '6'.charCodeAt(0);
  assert.throws(() => decodeSpectrum(wrong));
});

test('V/I has a linear signed scale clipped to minus and plus 0.6', () => {
  const negative = colorIndex(-0.3, 'V');
  const positive = colorIndex(0.3, 'V');
  assert.equal(negative + positive, 255);
  assert.ok(negative < colorIndex(0, 'V'));
  assert.ok(positive > colorIndex(0, 'V'));
  const lut = COLOR_TABLES.V;
  assert.ok(lut[negative * 4 + 2] > lut[negative * 4]);
  assert.ok(lut[positive * 4] > lut[positive * 4 + 2]);
  assert.equal(colorIndex(-0.6, 'V'), 0);
  assert.equal(colorIndex(0.6, 'V'), 255);
  assert.equal(colorIndex(-2, 'V'), 0);
  assert.equal(colorIndex(2, 'V'), 255);
  assert.equal(colorIndex(null, 'V'), -1);
  assert.equal(colorIndex(NaN, 'V'), -1);
});

test('intensity retains logarithmic range in s.f.u. and missing data remains distinct', () => {
  assert.equal(colorIndex(1, 'I'), 0);
  assert.equal(colorIndex(3000, 'I'), 255);
  assert.equal(colorIndex(Math.sqrt(3000), 'I'), 128);
  assert.equal(colorIndex(Infinity, 'I'), -1);
});

test('intensity uses the exact Viridis color table', () => {
  assert.deepEqual(Array.from(COLOR_TABLES.I.slice(0, 3)), [68, 1, 84]);
  assert.deepEqual(Array.from(COLOR_TABLES.I.slice(128 * 4, 128 * 4 + 3)), [32, 144, 140]);
  assert.deepEqual(Array.from(COLOR_TABLES.I.slice(255 * 4, 255 * 4 + 3)), [253, 231, 36]);
});

test('display converts Jy to s.f.u. and divides V by matching raw I', () => {
  assert.equal(displayValue(1e6, 'I'), 100);
  assert.equal(displayValue(-3e5, 'V', 1e6), -0.3);
  assert.equal(displayValue(3e5, 'V', 1e6), 0.3);
  assert.equal(displayValue(0, 'V', 1e6), 0);
  assert.equal(displayValue(3e5, 'V', 0), null);
  assert.equal(displayValue(3e5, 'V', -1), null);
  assert.equal(displayValue(3e5, 'V', NaN), null);
  assert.equal(displayValue(3e5, 'V'), null);
  assert.equal(displayValue(0, 'I'), null);
  assert.equal(displayValue(NaN, 'I'), null);
});

test('source timestamps preserve gaps across the five-minute ring boundary', () => {
  assert.equal(TIME_WINDOW_SECONDS, 300);
  const slot = timeSlot(1720000000);
  assert.equal(timeSlot(1720000000.5), slot + 1);
  assert.equal(timeSlot(1720000002.5), slot + 5);
  assert.equal(timeSlot(1720000300) % TIME_COLUMNS, slot % TIME_COLUMNS);
});

test('ordinary integrations fill their duration without hiding a missed frame', () => {
  assert.equal(firstCoveredSlot(100.49, 101.0036), 201);
  assert.equal(timeSlot(101.0036), 202);
  assert.equal(firstCoveredSlot(100.49, 101.5172), 203);
  assert.equal(firstCoveredSlot(null, 101.0036), 202);
});

function batch(times, { first = 1, reset = false, history = false, boot = 'source', intensity = 1e6 } = {}) {
  return { I: times.map(() => new Float32Array(768).fill(intensity)),
    V: times.map(() => new Float32Array(768).fill(-.3 * intensity)), timestamps: times,
    sequence: first + times.length - 1, first_sequence: first, boot_id: boot,
    reset, history, buffer_length: 600, buffer_index: 0, timestamp: Date.now() / 1000 };
}

test('older history fills the past without moving the latest live time', () => {
  const raster = new SpectrumRaster();
  const live = raster.update(batch([101, 101.5], { first: 3, reset: true }));
  const older = raster.update(batch([100, 100.5], { history: true }));
  assert.equal(older.latestSlot, live.latestSlot);
  assert.equal(older.latestTimestamp, 101.5);
  assert.ok(older.I.some(span => span.left <= 200 && span.left + span.width > 200));
  assert.equal(raster.update(batch([100], { history: true, boot: 'previous-process' })), null);
});

test('late backfill cannot replace a newer integration in the same time column', () => {
  const raster = new SpectrumRaster();
  raster.update(batch([100.2], { first: 3, reset: true, intensity: 3e7 }));
  const late = raster.update(batch([100.1], { first: 2, history: true, intensity: 1e4 }));
  assert.equal(late.I.length, 0);
  assert.equal(late.V.length, 0);
});

test('a five-minute gap expires the old image and a restart rejects old history', () => {
  const raster = new SpectrumRaster();
  raster.update(batch([100], { reset: true }));
  const jumped = raster.update(batch([401], { first: 2 }));
  assert.equal(jumped.latestSlot, 802);
  assert.equal(jumped.I.reduce((total, span) => total + span.width, 0), 600);
  raster.update(batch([402], { reset: true, boot: 'new-process' }));
  assert.equal(raster.update(batch([100], { history: true })), null);
});

test('detections follow source UTC independently of inference and polling delay', () => {
  const detection = { id: 1, class_id: 0, class: 'type3', confidence: .57,
    bbox: [.9, .2, .1, .3], timestamp: 9000, boot_id: 'source',
    time_start: 1030, time_end: 1060 };
  const original = detectionGeometry(detection, 1300, 'source');
  assert.ok(Math.abs(original.x - .1) < 1e-12);
  assert.ok(Math.abs(original.width - .1) < 1e-12);
  const later = detectionGeometry(detection, 1330, 'source');
  assert.equal(later.x, 0);
  assert.equal(later.width, original.width);
  assert.equal(detectionGeometry(detection, 1330, 'different-process'), null);
  assert.equal(detectionGeometry(detection, 1361, 'source'), null);
});

test('I clips at 3000 s.f.u. while preserving a logarithmic midpoint', () => {
  assert.equal(colorIndex(3000, 'I'), 255);
  assert.equal(colorIndex(10000, 'I'), 255);
  assert.equal(colorIndex(Math.sqrt(3000), 'I'), 128);
  assert.equal(firstCoveredSlot(101.5, 100), 200);
});
