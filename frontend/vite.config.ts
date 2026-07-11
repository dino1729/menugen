import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    host: '127.0.0.1',
    port: 3000,
    strictPort: true,
  },
  test: {
    environment: 'jsdom',
    setupFiles: './src/setupTests.ts',
    environmentOptions: {
      jsdom: { url: 'http://localhost:3000' },
    },
  },
});
