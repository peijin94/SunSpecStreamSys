// Deployment variant for the live spectrum UI.
// VITE_APP_MODE=public builds the fixed public page (latest 300 s, full band,
// no interactive controls or measurement). Anything else (including unset)
// builds the full dev/operator UI. Default is dev so existing tests keep passing.
export type AppMode = 'public' | 'dev';

export function resolveAppMode(value: unknown): AppMode {
  return value === 'public' ? 'public' : 'dev';
}

const rawMode: unknown =
  (import.meta as unknown as { env?: Record<string, unknown> }).env?.VITE_APP_MODE ?? undefined;

export const APP_MODE: AppMode = resolveAppMode(rawMode);
export const IS_PUBLIC: boolean = APP_MODE === 'public';

// Public page contract: continuous fixed rolling 300 s, full 15-85 MHz band.
export const PUBLIC_SECONDS = 300;
export const PUBLIC_FMIN = 15;
export const PUBLIC_FMAX = 85;

/** True when the mode permits interactive view/measurement controls. */
export function isInteractiveMode(mode: AppMode): boolean {
  return mode !== 'public';
}

/** True when programmatic view changes (zoom/preset/pause/range) are allowed. */
export function canChangeView(mode: AppMode): boolean {
  return mode !== 'public';
}

/** True when the measurement tool is allowed. */
export function canMeasure(mode: AppMode): boolean {
  return mode !== 'public';
}
