import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

// https://vite.dev/config/
function resolveBase(value: string | undefined, fallback: string): string {
  let base = value ?? fallback;
  if (!base.startsWith('/')) base = `/${base}`;
  if (!base.endsWith('/')) base += '/';
  return base;
}

export default defineConfig(({ mode }) => ({
  plugins: [react()],
  // Production base comes from VITE_BASE_PATH so the same build works under
  // /live/ (public, VITE_APP_MODE=public) or /live-dev/ (dev, VITE_APP_MODE=dev).
  // The parent owns frontend/.env.production and sets VITE_BASE_PATH there;
  // default stays /live/ for backwards compatibility. Local dev keeps root.
  base:
    mode === 'production'
      ? resolveBase(loadEnv(mode, process.cwd()).VITE_BASE_PATH, '/live/')
      : '/',
  server: {
    proxy: {
      '/data': 'http://localhost:9527',
      '/refresh': 'http://localhost:9527',
      '/spectrum': 'http://localhost:9527',
      '/type3detect': 'http://localhost:9527',
    },
  },
}));
