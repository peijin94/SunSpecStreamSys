import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vite.dev/config/
export default defineConfig(({ mode }) => ({
  plugins: [react()],
  // In production we are served under /live/, locally we keep root.
  base: mode === 'production' ? '/live/' : '/',
  server: {
    proxy: {
      '/data': 'http://localhost:9527',
      '/refresh': 'http://localhost:9527',
      '/type3detect': 'http://localhost:9527',
    },
  },
}));
