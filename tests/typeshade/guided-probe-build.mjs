// Bundle `guided-probe.mjs` with the TypeShade plugin into `.typecv-build/guided-probe/`, the way
// `scripts/build-gpu.mjs` builds the package, and remove the host views the plugin writes.
import { readdir, unlink } from 'node:fs/promises';
import { resolve } from 'node:path';
import { build as bundle } from 'esbuild';

const root = resolve(import.meta.dirname, '../..');
export const guidedProbeFile = resolve(root, '.typecv-build/guided-probe/guided-probe.js');

async function clean(directory) {
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const file = `${directory}/${entry.name}`;
    if (entry.isDirectory()) await clean(file);
    else if (entry.name.endsWith('.shade.typeshade.ts')) await unlink(file);
  }
}

export async function buildGuidedProbe() {
  const plugin = resolve(root, '.typecv-build/guided-probe-plugin.mjs');
  await bundle({
    stdin: { contents: "export { typeshade } from 'typeshade/vite';", resolveDir: root },
    outfile: plugin,
    bundle: true,
    platform: 'node',
    format: 'esm',
    external: ['vite', 'typescript'],
    logLevel: 'error',
  });
  const [{ typeshade }, { build }] = await Promise.all([import(plugin), import('vite')]);
  try {
    await build({
      root,
      configFile: false,
      logLevel: 'error',
      plugins: [typeshade({ fuse: true })],
      build: {
        outDir: resolve(root, '.typecv-build/guided-probe'),
        emptyOutDir: true,
        lib: {
          entry: resolve(root, 'tests/typeshade/guided-probe.mjs'),
          formats: ['es'],
          fileName: () => 'guided-probe.js',
        },
        minify: false,
      },
    });
  } finally {
    await clean(resolve(root, 'src/gpu'));
  }
  return guidedProbeFile;
}
