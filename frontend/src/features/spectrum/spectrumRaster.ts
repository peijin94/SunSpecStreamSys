import type { SpectrumBatch } from '../../api/spectrumProtocol.js';
import { COLOR_TABLES, colorIndex, displayValue, timeSlot, firstCoveredSlot, FREQUENCY_BINS, TIME_COLUMNS } from './spectrumRendering.js';

export interface PixelSpan { left: number; width: number; pixels: Uint8ClampedArray<ArrayBuffer> }
export interface RasterUpdate {
  I: PixelSpan[];
  V: PixelSpan[];
  reset: boolean;
  bootId: string;
  latestSlot: number | null;
  latestTimestamp: number | null;
}

export class SpectrumRaster {
  private I = new Uint8ClampedArray(TIME_COLUMNS * FREQUENCY_BINS * 4);
  private V = new Uint8ClampedArray(TIME_COLUMNS * FREQUENCY_BINS * 4);
  private slots = new Float64Array(TIME_COLUMNS).fill(NaN);
  private sequences = new Float64Array(TIME_COLUMNS).fill(-1);
  private bootId = '';
  private latestSlot: number | null = null;
  private latestTimestamp: number | null = null;

  update(batch: SpectrumBatch): RasterUpdate | null {
    if (batch.history && batch.boot_id !== this.bootId) return null;
    const reset = batch.reset || batch.boot_id !== this.bootId;
    if (reset) {
      this.I.fill(0); this.V.fill(0); this.slots.fill(NaN); this.sequences.fill(-1);
      this.latestSlot = null; this.latestTimestamp = null; this.bootId = batch.boot_id;
    }
    const dirty = new Set<number>();
    const columnFor = (slot: number) => ((slot % TIME_COLUMNS) + TIME_COLUMNS) % TIME_COLUMNS;
    const clear = (slot: number) => {
      const column = columnFor(slot);
      for (let y = 0; y < FREQUENCY_BINS; y++) {
        const offset = (y * TIME_COLUMNS + column) * 4;
        this.I.fill(0, offset, offset + 4); this.V.fill(0, offset, offset + 4);
      }
      this.slots[column] = slot; this.sequences[column] = -1; dirty.add(column);
    };
    let previous = batch.previous_timestamp ?? null;
    batch.timestamps.forEach((timestamp, index) => {
      const intensity = batch.I[index]; const circular = batch.V[index];
      if (!Number.isFinite(timestamp) || intensity?.length !== FREQUENCY_BINS || circular?.length !== FREQUENCY_BINS) return;
      const slot = timeSlot(timestamp);
      if (!batch.history && (this.latestSlot === null || slot > this.latestSlot)) {
        if (this.latestSlot !== null) {
          if (slot - this.latestSlot >= TIME_COLUMNS) {
            this.I.fill(0); this.V.fill(0); this.slots.fill(NaN); this.sequences.fill(-1);
            for (let c = 0; c < TIME_COLUMNS; c++) dirty.add(c);
          } else {
            for (let s = this.latestSlot + 1; s <= slot; s++) clear(s);
          }
        }
        this.latestSlot = slot;
      }
      if (!batch.history && (this.latestTimestamp === null || timestamp > this.latestTimestamp)) this.latestTimestamp = timestamp;
      if (this.latestSlot === null || slot < this.latestSlot - TIME_COLUMNS + 1 || slot > this.latestSlot) return;
      const first = Math.max(this.latestSlot - TIME_COLUMNS + 1, firstCoveredSlot(previous, timestamp));
      const sequence = (batch.first_sequence ?? batch.sequence - batch.I.length + 1) + index;
      for (let s = first; s <= slot; s++) {
        const column = columnFor(s);
        if (this.slots[column] === s && this.sequences[column] > sequence) continue;
        for (let channel = 0; channel < FREQUENCY_BINS; channel++) {
          const offset = ((FREQUENCY_BINS - channel - 1) * TIME_COLUMNS + column) * 4;
          const iColor = colorIndex(displayValue(intensity[channel], 'I'), 'I');
          const vColor = colorIndex(displayValue(circular[channel], 'V', intensity[channel]), 'V');
          for (let component = 0; component < 3; component++) {
            this.I[offset + component] = iColor < 0 ? 0 : COLOR_TABLES.I[iColor * 4 + component];
            this.V[offset + component] = vColor < 0 ? 0 : COLOR_TABLES.V[vColor * 4 + component];
          }
          this.I[offset + 3] = 255; this.V[offset + 3] = 255;
        }
        this.slots[column] = s; this.sequences[column] = sequence; dirty.add(column);
      }
      previous = timestamp;
    });
    const columns = [...dirty].sort((a, b) => a - b);
    const intensitySpans: PixelSpan[] = []; const circularSpans: PixelSpan[] = [];
    for (let i = 0; i < columns.length;) {
      const left = columns[i]; let last = left;
      while (i + 1 < columns.length && columns[i + 1] === last + 1) last = columns[++i];
      const width = last - left + 1;
      const I = new Uint8ClampedArray(width * FREQUENCY_BINS * 4);
      const V = new Uint8ClampedArray(I.length);
      for (let y = 0; y < FREQUENCY_BINS; y++) {
        const begin = (y * TIME_COLUMNS + left) * 4;
        I.set(this.I.subarray(begin, begin + width * 4), y * width * 4);
        V.set(this.V.subarray(begin, begin + width * 4), y * width * 4);
      }
      intensitySpans.push({ left, width, pixels: I }); circularSpans.push({ left, width, pixels: V });
      i++;
    }
    return { I: intensitySpans, V: circularSpans, reset, bootId: this.bootId,
      latestSlot: this.latestSlot, latestTimestamp: this.latestTimestamp };
  }
}
