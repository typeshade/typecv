// Build-time TypeShade analysis stays outside the browser/runtime bundle.
// node scripts/profile-kernel-ir.mjs [--out directory] [--kernel file.shade.ts#entry ...]
import { mkdir } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import { build } from 'esbuild';

await mkdir('.typecv-build', { recursive: true });
await build({
  entryPoints: ['scripts/profile-kernel-ir-compiler.mjs'],
  outfile: '.typecv-build/profile-kernel-ir.mjs',
  bundle: true,
  platform: 'node',
  format: 'esm',
  external: ['typescript'],
});
execFileSync(process.execPath, ['.typecv-build/profile-kernel-ir.mjs', ...process.argv.slice(2)], {
  stdio: 'inherit',
});
