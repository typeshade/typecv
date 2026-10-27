import { defineConfig } from 'vite';
import { typeshade } from 'typeshade/vite';

// The package root (Vite's root), with `/` separators: the prefix of the module ids.
let root = process.cwd().replace(/\\/g, '/');

export default defineConfig({
  plugins: [
    typeshade(),
    {
      // The TypeShade plugin at the pin writes a kernel function's fuse record with the module id
      // it was handed, the build machine's absolute path (`"fuse": { "origin": "/…/x.shade.ts" }`,
      // issue #266). The record's origin is a key, compared only with the origins of other
      // records, so the path relative to the package root keeps every comparison and drops the
      // machine's path from dist. TypeShade's own shipped faces use the same relative form.
      name: 'typecv-fuse-origin',
      enforce: 'pre',
      configResolved(config: { readonly root: string }) {
        root = config.root.replace(/\\/g, '/');
      },
      transform(code: string, id: string) {
        const path = id.split('?')[0]!.replace(/\\/g, '/');
        if (!path.endsWith('.shade.ts') || !path.startsWith(`${root}/`)) return null;
        const absolute = JSON.stringify(path);
        if (!code.includes(absolute)) return null;
        const relative = JSON.stringify(path.slice(root.length + 1));
        return { code: code.split(absolute).join(relative), map: null };
      },
    },
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
