import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { resolve } from 'node:path';
import { build } from 'esbuild';
import { W, H, P, z, fixture, bind } from './mvs-boundary-probe-fixture.mjs';

async function compiler() {
  const b = await build({
    stdin: {
      contents: "export {compile,compileModuleJs} from 'typeshade';",
      resolveDir: resolve(import.meta.dirname, '../..'),
    },
    bundle: true,
    platform: 'node',
    format: 'cjs',
    write: false,
    external: ['typescript'],
  });
  const c = { exports: {} };
  new Function('require', 'module', 'exports', b.outputFiles[0].text)(
    createRequire(import.meta.url),
    c,
    c.exports,
  );
  return c.exports;
}

test('sample-centered NCC resolves aligned bright low-contrast texture in f32', async () => {
  const c = await compiler(),
    source = await readFile(new URL('../../src/gpu/mvs.shade.ts', import.meta.url), 'utf8'),
    parsed = c.compile(source),
    rows = [];
  for (const precision of ['f64', 'f32']) {
    const m = c.compileModuleJs(parsed.module, { precision });
    for (const [policy, boundaryAware, minFiniteGain] of [
      ['default', false, 0],
      ['boundary', true, 0],
      ['finite-gain', false, 0.02],
    ]) {
      let tested = 0,
        maxError = 0;
      for (const base of [0.7, 0.9, 0.98])
        for (const contrast of [0.012, 0.018, 0.025]) {
          const b = fixture('finite', boundaryAware, minFiniteGain);
          for (let i = 0; i < b.mvsGray.length; i++)
            b.mvsGray[i] = base + contrast * (b.mvsGray[i] - 0.45);
          bind(m, b);
          for (const [x, y] of [
            [24, 12],
            [32, 16],
            [40, 20],
          ]) {
            const stats = m.fns.referenceWindow(x, y);
            if (!stats.ok) continue;
            const X = m.fns.referenceRay(x, y).map((v) => v * z),
              cost = m.fns.sourceCost(1, x, y, X, [0, 0, -1], stats);
            // Integer-disparity copies have NCC=1 independently of brightness and window policy.
            assert.ok(
              cost < 0.001,
              `${precision}/${policy}: aligned low-contrast cost ${cost}, base ${base}`,
            );
            maxError = Math.max(maxError, cost);
            tested++;
          }
        }
      assert.ok(tested >= 20, `${precision}/${policy}: at least 20 valid samples`);
      rows.push({ precision, policy, tested, maxError });
    }
  }
  console.log(JSON.stringify({ lowContrastNcc: rows }));
});
