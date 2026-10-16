import type { Detection } from './detectionProtocol';
import { parseAntennaFlags } from './antennaProtocol';
import { parseImagingStatus } from './imagingProtocol';
export type { Detection } from './detectionProtocol';
import { decodeSpectrum, type SpectrumBatch, type SpectrumCursor } from './spectrumProtocol';
export type { PolarizedFrame, SpectrumBatch, SpectrumCursor } from './spectrumProtocol';

const DEFAULT_API_BASE =
  import.meta.env.VITE_API_BASE_URL ?? '';

// Use Vite's BASE_URL so that in production we automatically
// prefix with /live/, while in dev it stays at /
const deploymentBase =
  import.meta.env.BASE_URL ?? '/';

function buildUrl(path: string): string {
  const base = DEFAULT_API_BASE.replace(/\/$/, '');
  const prefix = deploymentBase.replace(/\/$/, '');
  const p = `${prefix}${path}`.replace(/\/{2,}/g, '/');
  if (!base) return p;
  return `${base}${p.startsWith('/') ? p : `/${p}`}`;
}

export async function fetchJson<T>(path: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(buildUrl(path), { signal });
  if (!res.ok) {
    throw new Error(`Request failed: ${res.status} ${res.statusText}`);
  }
  return (await res.json()) as T;
}

async function fetchText(path: string): Promise<string> {
  const res = await fetch(path);
  if (!res.ok) {
    throw new Error(`Request failed: ${res.status} ${res.statusText}`);
  }
  return await res.text();
}

export type SpectrumFrame = number[];

export async function fetchSpectrum(
  cursor?: SpectrumCursor,
  signal?: AbortSignal,
): Promise<SpectrumBatch> {
  const query = cursor
    ? `?format=binary&limit=64&since=${cursor.sequence}&boot_id=${encodeURIComponent(cursor.boot_id)}`
    : '?format=binary&limit=64';
  const res = await fetch(buildUrl(`/spectrum${query}`), { signal });
  if (!res.ok) throw new Error(`Spectrum request failed: ${res.status}`);
  return decodeSpectrum(await res.arrayBuffer());
}

export async function fetchSpectrumHistory(bootId: string, before: number, signal?: AbortSignal): Promise<SpectrumBatch> {
  const res = await fetch(buildUrl(`/spectrum?format=binary&limit=64&before=${before}&boot_id=${encodeURIComponent(bootId)}`), { signal });
  if (!res.ok) throw new Error(`History request failed: ${res.status}`);
  return decodeSpectrum(await res.arrayBuffer());
}

export async function fetchSpectrumRange(query: string, signal?: AbortSignal) {
  const res = await fetch(buildUrl(`/spectrum-range?${query}`), { signal });
  if (!res.ok) throw new Error(`Spectrum range failed: ${res.status}`);
  const data = decodeSpectrum(await res.arrayBuffer());
  const range = data as import('../features/spectrum/rangeModel').RangeBatch;
  if (range.range !== true || !Number.isFinite(range.display_start) ||
      !Number.isFinite(range.display_end) || range.display_end <= range.display_start ||
      range.display_end-range.display_start > 3600.001) throw new Error('Invalid selected spectrum range');
  return range;
}

export interface RefreshResponse {
  data: SpectrumFrame[];
  buffer_length: number;
  buffer_index: number;
  timestamp: number;
}

export interface DetectionResponse {
  detections: Detection[];
  timestamp: number;
  time_anchor: number;
  count: number;
  last_detection_time: number;
}

export interface AiSummaryResponse {
  summary: string;
  timestamp: number;
}

export interface VisitorCountResponse {
  count: number;
}

export interface SunEphemeris {
  el: number;
  az: number;
  sunrise?: string;
  sunset?: string;
  sunup?: boolean;
  raw: string;
}

export function fetchLatestData() {
  return fetchJson<SpectrumFrame>('/data');
}

export function refreshSpectrum() {
  return fetchJson<RefreshResponse>('/refresh');
}

export function fetchDetections(signal?: AbortSignal) {
  return fetchJson<DetectionResponse>('/type3detect', signal);
}

export function fetchAiSummary() {
  return fetchJson<AiSummaryResponse>('/ai-summary');
}

export interface RadioImage {
  target_mhz: number;
  freq_mhz: number;
  date_obs: string;
  unit: string;
  image_url: string;
}

export interface RadioImagesResponse {
  status: 'loading' | 'unavailable' | 'ready' | 'stale';
  observation_time: string | null;
  version: string | null;
  checked_at: number | null;
  fetched_at: number | null;
  poll_interval_seconds: number;
  message: string | null;
  images: RadioImage[];
}

export function fetchRadioImages(signal?: AbortSignal) {
  return fetchJson<RadioImagesResponse>('/radio-images', signal);
}

export async function fetchAntennaFlags(signal?: AbortSignal) {
  return parseAntennaFlags(await fetchJson<unknown>('/antenna-flags', signal));
}

export async function fetchImagingStatus(signal?: AbortSignal) {
  return parseImagingStatus(await fetchJson<unknown>('/imaging-status', signal));
}

export function radioImageUrl(path: string) {
  return buildUrl(path);
}

export function fetchVisitorCount() {
  return fetchJson<VisitorCountResponse>('/visitors/count');
}

export async function fetchSunEphemeris(): Promise<SunEphemeris> {
  // This hits the separate OVSA ephemeris API handled by Apache at /api/ephm/info
  const txt = await fetchText('/api/ephm/info');

  const altMatch = /alt=([+-]?\d+(?:\.\d+)?)deg/.exec(txt);
  const azMatch = /az=([+-]?\d+(?:\.\d+)?)deg/.exec(txt);
  const sunupMatch = /sunup=(\d+)/.exec(txt);
  const sunriseMatch = /sunrise=([^\s]+)/.exec(txt);
  const sunsetMatch = /sunset=([^\s]+)/.exec(txt);

  const el = altMatch ? Number(altMatch[1]) : NaN;
  const az = azMatch ? Number(azMatch[1]) : NaN;

  return {
    el,
    az,
    sunup: sunupMatch ? sunupMatch[1] === '1' : undefined,
    sunrise: sunriseMatch?.[1],
    sunset: sunsetMatch?.[1],
    raw: txt,
  };
}
