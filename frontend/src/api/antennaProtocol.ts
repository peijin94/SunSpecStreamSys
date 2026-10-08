export interface AntennaFlagBand {
  freq_mhz: number;
  ant_list: number[] | null;
  n_bad_ant: number | null;
  flagged_frac: number | null;
}

export interface AntennaFlagsResponse {
  status: 'loading' | 'unavailable' | 'ready' | 'stale';
  antenna_count: number;
  id_system: 'ms_antenna_index';
  observation_time: string | null;
  message: string | null;
  poll_interval_seconds: number;
  bands: AntennaFlagBand[];
}

export function parseAntennaFlags(input: unknown): AntennaFlagsResponse {
  const data = input as AntennaFlagsResponse;
  if (!data || data.antenna_count !== 352 || data.id_system !== 'ms_antenna_index'
      || !['loading', 'unavailable', 'ready', 'stale'].includes(data.status)
      || !Array.isArray(data.bands)
      || data.observation_time !== null && !Number.isFinite(Date.parse(data.observation_time))
      || data.bands.length > 0 && data.observation_time === null) {
    throw new Error('Unknown antenna flag format');
  }
  for (const band of data.bands) {
    if (!band || !Number.isFinite(band.freq_mhz) || band.freq_mhz <= 0
        || band.ant_list !== null && (!Array.isArray(band.ant_list)
          || band.ant_list.some(id => !Number.isInteger(id) || id < 0 || id >= 352)
          || new Set(band.ant_list).size !== band.ant_list.length
          || band.n_bad_ant !== band.ant_list.length)) {
      throw new Error('Invalid antenna flag list');
    }
  }
  if (new Set(data.bands.map(band => band.freq_mhz)).size !== data.bands.length) {
    throw new Error('Duplicate antenna band');
  }
  return data;
}

export function selectAntennaFlags(bands: AntennaFlagBand[], frequency: number | null) {
  const selected = frequency === null ? bands : bands.filter(band => band.freq_mhz === frequency);
  return { badIds: new Set(selected.flatMap(band => band.ant_list ?? [])),
    complete: selected.length > 0 && selected.every(band => band.ant_list !== null) };
}
