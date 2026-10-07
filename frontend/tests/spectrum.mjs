import test from 'node:test';
import assert from 'node:assert/strict';
import { decodeSpectrum } from '../.test-build/api/spectrumProtocol.js';
import { colorIndex, COLOR_TABLES, timeSlot, firstCoveredSlot, TIME_COLUMNS, TIME_WINDOW_SECONDS } from '../.test-build/features/spectrum/spectrumRendering.js';

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

test('V distinguishes both signs and zero while clipping extreme magnitudes', () => {
  const negative = colorIndex(-1e6, 'V');
  const positive = colorIndex(1e6, 'V');
  assert.equal(negative + positive, 255);
  assert.ok(negative < colorIndex(0, 'V'));
  assert.ok(positive > colorIndex(0, 'V'));
  const lut = COLOR_TABLES.V;
  assert.ok(lut[negative * 4 + 2] > lut[negative * 4]);
  assert.ok(lut[positive * 4] > lut[positive * 4 + 2]);
  assert.equal(colorIndex(-1e20, 'V'), 0);
  assert.equal(colorIndex(1e20, 'V'), 255);
  assert.equal(colorIndex(null, 'V'), -1);
  assert.equal(colorIndex(NaN, 'V'), -1);
});

test('intensity retains logarithmic range and missing data remains distinct', () => {
  assert.equal(colorIndex(1e4, 'I'), 0);
  assert.equal(colorIndex(1e8, 'I'), 255);
  assert.equal(colorIndex(1e6, 'I'), 128);
  assert.equal(colorIndex(Infinity, 'I'), -1);
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
