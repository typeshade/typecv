import { mkdir, readdir, unlink, rm, writeFile } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import { build as bundle } from 'esbuild';
import { build as viteBuild } from 'vite';

// TypeNN's consumer pattern: bundle the build-time TypeShade plugin, ship only its runtime.
async function clean(directory, suffix = '.shade.typeshade.ts') {
  for (const entry of await readdir(directory, { withFileTypes: true })) {
    const file = `${directory}/${entry.name}`;
    if (entry.isDirectory()) await clean(file, suffix);
    else if (entry.name.endsWith(suffix)) await unlink(file);
  }
}

// The directories of GPU code under `root`: `gpu`, `*/gpu` and `*/*/gpu`, the globs of
// docs/library-structure.md, section 3.1. The host views of the kernels sit there under `src`, and
// their declarations at the same paths under `dist/types`. A glob that matches no directory adds
// none. The walk does not enter a `gpu` directory, since `clean` reaches everything below it.
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

async function cleanGpu(root, suffix) {
  for (const directory of await gpuDirectories(root)) await clean(directory, suffix);
}

await cleanGpu('src');
await mkdir('.typecv-build', { recursive: true });
await bundle({
  entryPoints: ['vite.gpu.config.ts'],
  outfile: '.typecv-build/vite.config.mjs',
  bundle: true,
  platform: 'node',
  format: 'esm',
  external: ['vite', 'typescript'],
});
try {
  await viteBuild({ configFile: '.typecv-build/vite.config.mjs' });
  // The test entry of the modules that `src/gpu/index.ts` does not export yet: a bundle of its
  // own, outside `dist/gpu`, so the public bundle stays as it is (`src/gpu/internal.ts`).
  await viteBuild({
    configFile: '.typecv-build/vite.config.mjs',
    logLevel: 'warn',
    build: {
      outDir: 'dist/internal',
      lib: { entry: 'src/gpu/internal.ts', formats: ['es'], fileName: () => 'gpu.js' },
    },
  });
  await writeFile('dist/gpu.js', "export * from './gpu/index.js';\n");
  await rm('dist/types', { recursive: true, force: true });
  execFileSync(process.execPath, ['node_modules/typescript/bin/tsc', '-p', 'tsconfig.gpu.json'], {
    stdio: 'inherit',
  });
  await cleanGpu('dist/types', '.shade.typeshade.d.ts');
} finally {
  await cleanGpu('src');
}
