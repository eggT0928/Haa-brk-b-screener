import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  build: { rollupOptions: { output: { manualChunks: { firebase: ['firebase/app', 'firebase/auth', 'firebase/firestore', 'firebase/app-check'] } } } },
  server: { proxy: { '/api': 'http://127.0.0.1:5000' } },
  test: { environment: 'jsdom' },
});
