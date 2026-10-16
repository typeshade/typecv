import { defineConfig } from 'astro/config';
import react from '@astrojs/react';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  site: 'https://cv.typeshade.dev',
  integrations: [react()],
  vite: { plugins: [tailwindcss()] },
  output: 'static',
  trailingSlash: 'always',
  build: { format: 'directory' },
});
