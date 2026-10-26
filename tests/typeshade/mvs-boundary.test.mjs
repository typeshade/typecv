import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import path from 'node:path';
import { build } from 'esbuild';

async function compiler() {
  const b = await build({
    stdin: {
      contents: "export { compile, compileModuleJs } from 'typeshade';",
      resolveDir: path.resolve(import.meta.dirname, '../..'),
    },
    bundle: true,
    platform: 'node',
    format: 'cjs',
    write: false,
    external: ['typescript'],
  });
  const m = { exports: {} };
  new Function('require', 'module', 'exports', b.outputFiles[0].text)(
    createRequire(import.meta.url),
    m,
    m.exports,
  );
  return m.exports;
}
import { W, P, z, fixture, bind } from './mvs-boundary-probe-fixture.mjs';

test('centered support measures a one-pixel foreground line; rotation-only ambiguity withholds depth', async () => {
  const c = await compiler(),
    source = await readFile(new URL('../../src/mvs/gpu/mvs.shade.ts', import.meta.url), 'utf8'),
    parsed = c.compile(source);
  assert.deepEqual(
    parsed.diagnostics.filter((d) => d.category === 'error'),
    [],
  );
  const rows = [];
  for (const precision of ['f64', 'f32']) {
    const m = c.compileModuleJs(parsed.module, { precision });
    const cost = (kind, aware, x = 32) => {
      const b = fixture(kind, aware, 0.02);
      bind(m, b);
      const stats = m.fns.referenceWindow(x, 16),
        X = m.fns.referenceRay(x, 16).map((v) => v * z);
      return {
        stats,
        finite: m.fns.sourceCost(1, x, 16, X, [0, 0, -1], stats),
        directional: m.fns.warpedSourceCost(1, x, 16, X, [0, 0, -1], stats, 1),
      };
    };
    const oldLine = cost('line', false),
      newLine = cost('line', true),
      finite = cost('finite', true),
      direction = cost('direction', true);
    assert.ok(newLine.finite < 0.01, `${precision}: aligned one-pixel line ${newLine.finite}`);
    assert.ok(
      oldLine.finite > newLine.finite + 0.1,
      `${precision}: skipped center must lose line evidence`,
    );
    assert.ok(
      newLine.directional > newLine.finite + 0.1,
      `${precision}: line has measurable finite depth`,
    );
    assert.ok(
      finite.finite < 0.001 && finite.directional > finite.finite + 0.1,
      `${precision}: finite texture is observable`,
    );
    assert.ok(
      Math.abs(direction.finite - direction.directional) < 1e-4,
      `${precision}: periodic directional texture must be ambiguous`,
    );
    const centered = fixture('line', true);
    bind(m, centered);
    assert.deepEqual(
      Array.from({ length: 5 }, (_, j) => m.fns.windowOffset(j)),
      [-4, -2, 0, 2, 4],
    );
    // An invalid source comparison is unknown, not evidence of direction-only structure.
    // The independent best-source NCC rule must still reject an entirely textureless source set.
    const unknown = fixture('finite', true, 0.02);
    unknown.mvsGray.fill(0.5, P);
    const unknownPixel = 16 * W + 32;
    unknown.mvsRange = { first: unknownPixel, end: unknownPixel + 1 };
    bind(m, unknown);
    const stats = m.fns.referenceWindow(32, 16);
    const X = m.fns.referenceRay(32, 16).map((v) => v * z);
    assert.equal(m.fns.warpedSourceCost(1, 32, 16, X, [0, 0, -1], stats, 1), 2);
    assert.equal(m.fns.observableSupport(1, 32, 16, X, [0, 0, -1], stats, 2), true);
    m.fns.mvsFilterOracle([0, 0, 0]);
    assert.equal(unknown.mvsGeoDepth[unknownPixel], 0);
    for (const [kind, x, expected] of [
      ['finite', 32, z],
      ['direction', 32, 0],
      ['line', 32, z],
      ['line', 35, 0],
    ]) {
      const outputs = [];
      for (const streamed of [false, true]) {
        const b = fixture(kind, true, 0.02),
          i = 16 * W + x;
        b.mvsRange = { first: i, end: i + 1 };
        b.streamControl.first = i;
        bind(m, b);
        if (streamed) {
          for (const s of [1, 2]) {
            b.mvsSources.set([1, s, 0, 0, 0, 0, 0]);
            b.streamControl.start = Number(s === 1);
            bind(m, b);
            m.fns.mvsStreamCostsOracle([0, 0, 0]);
          }
          m.fns.mvsStreamSelect([0, 0, 0]);
        } else m.fns.mvsFilterOracle([0, 0, 0]);
        outputs.push([b.mvsGeoDepth[i], b.mvsCost[i]]);
        assert.equal(
          b.mvsGeoDepth[i],
          expected,
          `${precision} ${kind}@${x} ${streamed ? 'streamed' : 'window'}`,
        );
      }
      assert.deepEqual(outputs[0], outputs[1], `${precision}: grouping changes observability`);
    }
    // The withheld direction remains accepted under the legacy filter, proving the extra rule
    // executes rather than a fixture failing the existing NCC, angle or reprojection conditions.
    const legacy = fixture('direction', true, 0),
      i = 16 * W + 32;
    legacy.mvsRange = { first: i, end: i + 1 };
    bind(m, legacy);
    m.fns.mvsFilterOracle([0, 0, 0]);
    assert.equal(legacy.mvsGeoDepth[i], z);
    rows.push({
      precision,
      oldLineCost: oldLine.finite,
      newLineCost: newLine.finite,
      lineDirectionCost: newLine.directional,
      finiteCost: finite.finite,
      directionCost: direction.directional,
    });
  }
  console.log(JSON.stringify({ boundary: rows }));
});
