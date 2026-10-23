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
const W = 64,
  H = 32,
  P = W * H,
  z = 4,
  f = 80,
  cx = 31.5,
  cy = 15.5;
// Parallel calibrated cameras. Finite objects move by exactly ten pixels; world directions do
// not move. No ground-truth depths are estimated from the implementation under test.
function fixture(kind, boundaryAware = false, minFiniteGain = 0) {
  const centers = [0, 0.5, -0.5],
    cameras = new Float32Array(72),
    gray = new Float32Array(3 * P);
  const texture = (x, y) =>
    0.45 +
    0.15 * Math.sin(x * 0.51) +
    0.12 * Math.cos(y * 0.37) +
    0.1 * Math.sin(x * 0.23 + y * 0.31);
  for (let s = 0; s < 3; s++) {
    const c = centers[s];
    cameras.set(
      [1, 0, 0, 0, 1, 0, 0, 0, 1, -c, 0, 0, f, f, cx, cy, W, H, 1, 40, c, 0, 0, s],
      24 * s,
    );
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++) {
        let value;
        if (kind === 'finite') value = texture(x + (f * c) / z, y);
        else if (kind === 'direction')
          value = 0.5 + 0.17 * Math.sin((x * 2 * Math.PI) / 10) + 0.13 * Math.cos(y * 0.37);
        else {
          const rod = 32 - (f * c) / z;
          value =
            x === rod
              ? 0.2 + 0.04 * Math.sin(y * 0.7)
              : 0.9 + 0.015 * Math.sin(x * 0.4) + 0.015 * Math.cos(y * 0.55);
        }
        gray[s * P + y * W + x] = value;
      }
  }
  const axis = boundaryAware ? 5 : 6;
  return {
    mvsControl: {
      width: W,
      height: H,
      reference: 0,
      state: 0,
      color: 0,
      iteration: 1,
      seed: 0,
      bestSources: 2,
      windowRadius: 5,
      windowStep: 2,
      axisSamples: axis,
      minSamples: Math.ceil((axis * axis) / 4),
      bilateral: Number(boundaryAware),
      boundaryAware: Number(boundaryAware),
      minFiniteGain,
      spatialFalloff: 1 / 50,
      colorFalloff: 50,
      minAngleCos: Math.cos(Math.PI / 180),
      perturbDepth: 0.2,
      perturbAngle: 0.2,
      geometric: 1,
      geometricWeight: 0.3,
      geometricMaxCost: 3,
      view: 0,
      relativePoseCache: 0,
    },
    mvsCameras: cameras,
    mvsGray: gray,
    mvsSources: Uint32Array.from([2, 1, 2, 0, 0, 0, 0]),
    mvsDepth: new Float32Array(3 * P).fill(z),
    mvsGeoDepth: new Float32Array(P).fill(z),
    mvsNormal: new Uint32Array(P).fill(0x7fff7fff),
    mvsCost: new Float32Array(P).fill(2),
    mvsPartial: new Float32Array(3),
    mvsRange: { first: 0, end: 1 },
    filterControl: {
      maxCost: 0.9,
      minConsistent: 2,
      maxError: 1,
      minAngleCos: Math.cos((3 * Math.PI) / 180),
      minConfidence: 0,
      writeConfidence: 1,
    },
    streamControl: { phase: 5, start: 1, first: 0, stride: 3, costs: 2, sources: 2 },
  };
}
const bind = (m, b) => {
  for (const [k, v] of Object.entries(b)) m.setBinding(k, v);
};

test('centered support measures a one-pixel foreground line; rotation-only ambiguity withholds depth', async () => {
  const c = await compiler(),
    source = await readFile(new URL('../../src/gpu/mvs.shade.ts', import.meta.url), 'utf8'),
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
