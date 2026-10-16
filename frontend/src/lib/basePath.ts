// Shared base-path helpers so production builds work under either
// VITE_BASE_PATH (/live/ for public, /live-dev/ for dev) without hardcoding.
// Pure functions (no import.meta) so they are unit-testable with node:test.

/** Normalize a Vite base to a leading/trailing-slash path (e.g. "/live/"). */
export function normalizeBasePath(value: string | undefined | null, fallback: string): string {
  let base = (value ?? fallback) || '/';
  if (!base.startsWith('/')) base = `/${base}`;
  if (!base.endsWith('/')) base += '/';
  return base;
}

/**
 * Join a deployment base ("/live/" or "/live-dev/" or "/") with an
 * absolute API/image path ("/spectrum-range?...", "/radio-images/...").
 * Collapses duplicate slashes; always returns a leading-slash path.
 */
export function joinBasePath(base: string, path: string): string {
  const prefix = base.replace(/\/$/, '');
  const joined = `${prefix}${path}`.replace(/\/{2,}/g, '/');
  return joined.startsWith('/') ? joined : `/${joined}`;
}

/**
 * Build the final request URL from an optional API origin override
 * (VITE_API_BASE_URL) plus the deployment base path.
 */
export function buildApiUrl(path: string, deploymentBase: string, apiBaseUrl: string): string {
  const base = (apiBaseUrl ?? '').replace(/\/$/, '');
  const prefixed = joinBasePath(deploymentBase, path);
  if (!base) return prefixed;
  return `${base}${prefixed.startsWith('/') ? prefixed : `/${prefixed}`}`;
}
