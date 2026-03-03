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

