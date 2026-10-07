export type Polarization = 'I' | 'V';
export const FREQUENCY_BINS = 768;
export const TIME_COLUMNS = 600;
export const COLUMN_SECONDS = 0.5;
export const TIME_WINDOW_SECONDS = TIME_COLUMNS * COLUMN_SECONDS;

type ColorStop = readonly [number, number, number, number];
const intensityStops: ColorStop[] = [
  [0, 0, 0, 4], [0.125, 31, 12, 72], [0.25, 85, 15, 109],
  [0.375, 136, 34, 106], [0.5, 187, 55, 84], [0.625, 227, 89, 51],
  [0.75, 249, 140, 10], [0.875, 249, 201, 50], [1, 252, 255, 164],
];
const circularStops: ColorStop[] = [
  [0, 37, 99, 235], [0.5, 241, 245, 249], [1, 239, 68, 68],
];

function makeColorTable(stops: ColorStop[]) {
  const table = new Uint8ClampedArray(256 * 4);
  for (let index = 0; index < 256; index += 1) {
    const t = index / 255;
    const upper = stops.findIndex((stop) => stop[0] >= t);
    const b = stops[Math.max(1, upper)];
    const a = stops[Math.max(1, upper) - 1];
    const weight = (t - a[0]) / (b[0] - a[0]);
    for (let channel = 0; channel < 3; channel += 1) {
      table[index * 4 + channel] = Math.round(a[channel + 1] + weight * (b[channel + 1] - a[channel + 1]));
    }
    table[index * 4 + 3] = 255;
  }
  return table;
}

export const COLOR_TABLES = {
  I: makeColorTable(intensityStops),
  V: makeColorTable(circularStops),
};

const SIGNED_LOG_RANGE = Math.log1p(1e8 / 1e4);

export function colorIndex(value: number | null, polarization: Polarization) {
  if (value === null || !Number.isFinite(value)) return -1;
  const scaled = polarization === 'I'
    ? (Math.log10(Math.max(1e4, Math.min(1e8, value))) - 4) / 4
    : 0.5 + 0.5 * Math.sign(value) * Math.log1p(Math.min(1e8, Math.abs(value)) / 1e4) / SIGNED_LOG_RANGE;
  return Math.round(Math.max(0, Math.min(1, scaled)) * 255);
}

export function timeSlot(timestamp: number) {
  return Math.floor(timestamp / COLUMN_SECONDS);
}

// The source integrates for approximately 0.514 s, so a normal frame can
// cover two 0.5 s raster columns. Longer gaps remain unfilled.
export function firstCoveredSlot(previous: number | null, current: number) {
  const slot = timeSlot(current);
  return previous !== null && current - previous <= COLUMN_SECONDS * 1.5
    ? Math.min(slot, timeSlot(previous) + 1)
    : slot;
}
