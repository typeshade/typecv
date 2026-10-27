import { defineConfig } from 'vite';
import { typeshade } from 'typeshade/vite';
export default defineConfig({
  plugins: [
    typeshade(),
    {
      name: 'typecv-runtime-boundary',
      generateBundle(_options, output) {
        for (const chunk of Object.values(output)) {
          if (chunk.type !== 'chunk') continue;
          const forbidden = Object.keys(chunk.modules).filter(
            (id) =>
              /\/node_modules\/typescript\//.test(id) ||
              /\/typeshade\/src\/(compiler|vite)(\/|\.)/.test(id),
          );
          if (forbidden.length)
            throw new Error(`Build-time compiler entered browser bundle: ${forbidden.join(', ')}`);
        }
      },
    },
  ],
  build: {
    outDir: 'dist/gpu',
    emptyOutDir: true,
    lib: { entry: 'src/gpu/index.ts', formats: ['es'], fileName: () => 'index.js' },
    minify: false,
  },
});
