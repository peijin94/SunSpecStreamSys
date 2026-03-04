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

export async function fetchJson<T>(path: string): Promise<T> {
  const res = await fetch(buildUrl(path));
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

export interface RefreshResponse {
  data: SpectrumFrame[];
  buffer_length: number;
  buffer_index: number;
  timestamp: number;
}

export interface Detection {
  id: number;
  class_id: number;
  class: string;
  confidence: number;
  bbox: [number, number, number, number];
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

export function fetchDetections() {
  return fetchJson<DetectionResponse>('/type3detect');
}

export function fetchAiSummary() {
  return fetchJson<AiSummaryResponse>('/ai-summary');
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

