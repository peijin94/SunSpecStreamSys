export type PolarizedFrame = ArrayLike<number | null>;

export interface SpectrumBatch {
  I: PolarizedFrame[];
  V: PolarizedFrame[];
  timestamps: number[];
  sequence: number;
  first_sequence: number | null;
  boot_id: string;
  reset: boolean;
  buffer_length: number;
  buffer_index: number;
  timestamp: number;
}

export interface SpectrumCursor {
  sequence: number;
  boot_id: string;
}

export function decodeSpectrum(buffer: ArrayBuffer): SpectrumBatch {
  if (buffer.byteLength < 4) throw new Error('Truncated spectrum header');
  const headerBytes = new DataView(buffer).getUint32(0, true);
  const offset = 4 + headerBytes;
  if (offset > buffer.byteLength || offset % 8 !== 0) throw new Error('Invalid spectrum header');
  const metadata = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 4, headerBytes)));
  const { rows, bins } = metadata;
  if (
    metadata.schema !== 'lwa-spectrum-iv-v1' || bins !== 768 ||
    !Number.isInteger(rows) || rows < 0 || rows > 600 ||
    metadata.byteorder !== 'little' || metadata.dtype !== 'float32' ||
    metadata.order?.join(',') !== 'I,V' ||
    !Number.isSafeInteger(metadata.sequence) || typeof metadata.boot_id !== 'string' ||
    buffer.byteLength !== offset + rows * (8 + bins * 4 * 2)
  ) throw new Error('Invalid I/V spectrum response');
  const times = new Float64Array(buffer, offset, rows);
  if (!times.every(Number.isFinite)) throw new Error('Invalid spectrum timestamp');
  const intensity = new Float32Array(buffer, offset + rows * 8, rows * bins);
  const circular = new Float32Array(buffer, offset + rows * 8 + rows * bins * 4, rows * bins);
  return {
    ...metadata,
    I: Array.from({ length: rows }, (_, i) => intensity.subarray(i * bins, (i + 1) * bins)),
    V: Array.from({ length: rows }, (_, i) => circular.subarray(i * bins, (i + 1) * bins)),
    timestamps: Array.from(times),
  };
}

