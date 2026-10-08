export interface ImagingBand { freq_mhz: number; flagged_frac: number | null }
export interface FluxCheck {
  freq_mhz: number;
  source: 'CasA' | 'CygA' | 'TauA' | 'VirA';
  measured_jy: number | null;
  expected_jy: number | null;
  ratio: number | null;
}
export interface ImagingStatusResponse {
  schema: 'lwa-imaging-status-v1';
  status: 'loading' | 'unavailable' | 'ready' | 'stale';
  observation_time: string | null;
  message: string | null;
  bands: ImagingBand[];
  flux_rows: FluxCheck[];
  flux_status: 'ready' | 'unavailable';
  flux_message: string | null;
  poll_interval_seconds: number;
}

export function parseImagingStatus(input: unknown): ImagingStatusResponse {
  const data = input as ImagingStatusResponse;
  if (!data || data.schema !== 'lwa-imaging-status-v1'
      || !['loading', 'unavailable', 'ready', 'stale'].includes(data.status)
      || !['ready', 'unavailable'].includes(data.flux_status)
      || !Array.isArray(data.bands) || !Array.isArray(data.flux_rows)
      || data.observation_time !== null && !Number.isFinite(Date.parse(data.observation_time))
      || (data.bands.length > 0 || data.flux_rows.length > 0) && data.observation_time === null) {
    throw new Error('Invalid imaging QA format');
  }
  for (const band of data.bands) {
    if (!band || !Number.isFinite(band.freq_mhz) || band.freq_mhz <= 0
        || band.flagged_frac !== null && (!Number.isFinite(band.flagged_frac) || band.flagged_frac < 0 || band.flagged_frac > 1)) {
      throw new Error('Invalid flagging fraction');
    }
  }
  for (const row of data.flux_rows) {
    if (!row || !Number.isFinite(row.freq_mhz) || row.freq_mhz <= 0
        || !['CasA', 'CygA', 'TauA', 'VirA'].includes(row.source)
        || [row.measured_jy, row.expected_jy, row.ratio].some(v => v !== null && !Number.isFinite(v))) {
      throw new Error('Invalid A-team flux measurement');
    }
  }
  if (new Set(data.bands.map(b => b.freq_mhz)).size !== data.bands.length
      || new Set(data.flux_rows.map(r => `${r.freq_mhz}/${r.source}`)).size !== data.flux_rows.length) {
    throw new Error('Duplicate imaging QA row');
  }
  return data;
}

export function selectImagingBand(bands: ImagingBand[], requested: number | null) {
  return bands.find(b => b.freq_mhz === requested)
    ?? [...bands].sort((a, b) => Math.abs(a.freq_mhz - 60) - Math.abs(b.freq_mhz - 60) || a.freq_mhz - b.freq_mhz)[0]
    ?? null;
}
