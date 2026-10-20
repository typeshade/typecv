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
await clean('src/gpu');
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
  await clean('dist/types/gpu', '.shade.typeshade.d.ts');
} finally {
  await clean('src/gpu');
}
