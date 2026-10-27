// Bundle a compute probe with the TypeShade plugin into `.typecv-build/`, the way
// `scripts/build-gpu.mjs` builds the package, and remove the host views the plugin writes.
import { readdir, unlink } from 'node:fs/promises';
import { resolve } from 'node:path';
import { build as bundle } from 'esbuild';
import { withComputeProbeLock } from './compute-probe-lock.mjs';

const root = resolve(import.meta.dirname, '../..');

async function clean(directory) {
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const file = `${directory}/${entry.name}`;
    if (entry.isDirectory()) await clean(file);
    else if (entry.name.endsWith('.shade.typeshade.ts')) await unlink(file);
  }
}

// The directories of GPU code under `root`, `gpu`, `*/gpu` and `*/*/gpu`, as scripts/build-gpu.mjs
// finds them (docs/library-structure.md, section 3.1): the plugin writes a host view beside each
// kernel there.
async function gpuDirectories(root, depth = 0) {
  const found = [];
  for (const entry of await readdir(root, { withFileTypes: true })) {
    if (!entry.isDirectory()) continue;
    const directory = `${root}/${entry.name}`;
    if (entry.name === 'gpu') found.push(directory);
    else if (depth < 2) found.push(...(await gpuDirectories(directory, depth + 1)));
  }
  return found;
}

export async function buildComputeProbe(name) {
  return withComputeProbeLock(root, () => buildLockedProbe(name));
}

async function buildLockedProbe(name) {
  const plugin = resolve(root, `.typecv-build/${name}-plugin.mjs`);
  const file = resolve(root, `.typecv-build/${name}/${name}.js`);
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
      plugins: [
        typeshade(),
        {
          name: `${name}-runtime-boundary`,
          generateBundle(_options, output) {
            for (const chunk of Object.values(output)) {
              if (chunk.type !== 'chunk') continue;
              const forbidden = Object.keys(chunk.modules).filter(
                (id) =>
                  /\/node_modules\/typescript\//.test(id) ||
                  /\/typeshade\/src\/(compiler|vite)(\/|\.)/.test(id),
              );
              if (forbidden.length)
                throw new Error(`Compiler entered the browser probe: ${forbidden.join(', ')}`);
            }
          },
        },
      ],
      build: {
        outDir: resolve(root, `.typecv-build/${name}`),
        emptyOutDir: true,
        lib: {
          entry: resolve(root, `tests/typeshade/${name}.mjs`),
          formats: ['es'],
          fileName: () => `${name}.js`,
        },
        minify: false,
        rollupOptions: { output: { inlineDynamicImports: true } },
      },
    });
  } finally {
    for (const directory of await gpuDirectories(resolve(root, 'src'))) await clean(directory);
  }
  return file;
}
