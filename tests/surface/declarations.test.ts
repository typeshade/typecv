import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { join } from 'node:path';

// The declarations that `npm run build` emits type-check for a consumer whose `skipLibCheck` is
// off (typeshade/typecv#251). `stripInternal` of tsconfig.gpu.json removes every `@internal`
// declaration from dist/types, so a public declaration that names one leaves a dangling
// reference, which only such a consumer sees. tsconfig.declarations.json names the two entry
// points, dist/index.d.ts and dist/types/gpu/index.d.ts, with `skipLibCheck: false`.
// `npm run test:surface` runs this file; the check job of CI runs it after `npm run build`.

const root = join(import.meta.dirname, '../..');

test('the emitted declarations type-check with skipLibCheck off', (t) => {
  if (
    !existsSync(join(root, 'dist/types/gpu/index.d.ts')) ||
    !existsSync(join(root, 'dist/index.d.ts'))
  ) {
    t.skip('no build in dist/: run npm run build first');
    return;
  }
  const tsc = join(root, 'node_modules/typescript/bin/tsc');
  const result = spawnSync(
    process.execPath,
    [tsc, '-p', join(root, 'tsconfig.declarations.json'), '--pretty', 'false'],
    { cwd: root, encoding: 'utf8' },
  );
  assert.equal(
    result.status,
    0,
    `tsc -p tsconfig.declarations.json reports errors in the emitted declarations:\n` +
      `${result.stdout}${result.stderr}`,
  );
});
