// The host faces of TypeCV's kernel modules, compiled as `npm run build` compiles them: TypeShade's
// `hostFace` with the options of the build's Vite plugin (`typeshade()` in vite.gpu.config.ts:
// fusion facts by default since TypeShade change 0085; a production build records no console
// calls). A face holds the host view (`name.shade.typeshade.ts`, which `tsc -p tsconfig.gpu.json`
// reads for a kernel import), the generated module and the module's exports.
// scripts/api-surface.mjs reads the views and scripts/kernel-digests.mjs the emitted WGSL and GLSL.
//
// TypeShade ships its compiler as TypeScript source, so this bundles the host-face module with
// esbuild into .typecv-build/ first, as scripts/audit-typeshade.mjs bundles its compiler harness.
// Nothing is written under src/: the Vite plugin writes each view beside its kernel, and this
// keeps the views in memory.
import { mkdirSync, readdirSync, readFileSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { build } from 'esbuild';

const ROOT = join(import.meta.dirname, '..');

/** The kernel modules of `src/` (`*.shade.ts`), relative to `root`, with `/`, sorted. */
export function kernelFiles(root = ROOT) {
  const files = [];
  const walk = (directory) => {
    for (const entry of readdirSync(join(root, directory), { withFileTypes: true })) {
      const path = `${directory}/${entry.name}`;
      if (entry.isDirectory()) walk(path);
      else if (entry.name.endsWith('.shade.ts')) files.push(path);
    }
  };
  walk('src');
  return files.sort();
}

/** TypeShade's `hostFace`, from the pinned package, bundled once per process. */
let compiler;
function hostFaceOf(root) {
  compiler ??= (async () => {
    // One file per process: two test files that bundle at once do not share a half-written file.
    const outfile = join(root, '.typecv-build', `kernel-faces-${process.pid}.mjs`);
    mkdirSync(join(root, '.typecv-build'), { recursive: true });
    await build({
      stdin: {
        contents:
          "export { hostFace } from './node_modules/typeshade/src/compiler/ts/host-face.ts';",
        resolveDir: root,
        loader: 'js',
      },
      outfile,
      bundle: true,
      platform: 'node',
      format: 'esm',
      external: ['typescript'],
      logLevel: 'warning',
    });
    try {
      return (await import(pathToFileURL(outfile).href)).hostFace;
    } finally {
      rmSync(outfile, { force: true });
    }
  })();
  return compiler;
}

/**
 * The face of every kernel module: `path` (relative to `root`), `fileName` (absolute, as Vite
 * names the module), `view`, `code` and `exports`. Throws when a module does not compile, as the
 * build does.
 */
export async function kernelFaces(root = ROOT) {
  const hostFace = await hostFaceOf(root);
  const readDocument = (fileName) => {
    try {
      return readFileSync(fileName, 'utf8');
    } catch {
      return undefined;
    }
  };
  return kernelFiles(root).map((path) => {
    const fileName = join(root, path).replaceAll('\\', '/');
    const face = hostFace(readFileSync(fileName, 'utf8'), { fileName, readDocument });
    if (face.code === undefined || face.view === undefined) {
      const errors = face.diagnostics
        .filter((d) => d.category === 'error')
        .map((d) => `${d.fileName ?? path}: ${d.code} ${d.message}`);
      throw new Error(`${path} does not compile:\n${errors.join('\n')}`);
    }
    return { path, fileName, view: face.view, code: face.code, exports: face.exports };
  });
}
