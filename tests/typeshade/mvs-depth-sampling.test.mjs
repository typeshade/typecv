import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import path from 'node:path';
import { build } from 'esbuild';

test('MVS reprojection preserves a slanted plane and rejects interpolation across edges', async () => {
  // Probe the actual production helper with TypeShade's f64 and f32 CPU generators.
  const bundled = await build({
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
  const compiler = { exports: {} };
  new Function('require', 'module', 'exports', bundled.outputFiles[0].text)(
    createRequire(import.meta.url),
    compiler,
    compiler.exports,
  );
  const source = await readFile(new URL('../../src/gpu/mvs.shade.ts', import.meta.url), 'utf8');
  const parsed = compiler.exports.compile(
    source +
      `
export function depthSamplingProbe(x: f32, y: f32, z: f32): f32 {
  return reprojectionError(1, x, y, referenceRay(x, y) * z, 3.);
}
`,
  );
  assert.deepEqual(
    parsed.diagnostics.filter((d) => d.category === 'error'),
    [],
  );
  const W = 64,
    H = 48,
    P = W * H,
    f = 80,
    cx = 31.5,
    cy = 23.5,
    baseline = 0.23;
  // World plane z = 3 + 0.8 x + 0.2 y; its inverse depth is affine in each camera's pixels.
  const planeDepth = (x, y, centerX = 0) =>
    (3 + 0.8 * centerX) / (1 - (0.8 * (x - cx)) / f - (0.2 * (y - cy)) / f);
  const cameras = new Float32Array(48);
  for (let s = 0; s < 2; s++)
    cameras.set(
      [
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
        -s * baseline,
        0,
        0,
        f,
        f,
        cx,
        cy,
        W,
        H,
        1,
        8,
        s * baseline,
        0,
        0,
        s,
      ],
      s * 24,
    );
  const depth = new Float32Array(2 * P);
  for (let y = 0; y < H; y++)
    for (let x = 0; x < W; x++) depth[P + y * W + x] = planeDepth(x, y, baseline);
  const x = 27.3,
    y = 20.7,
    z = planeDepth(x, y),
    qx = x - (f * baseline) / z;
  const ix = Math.round(qx),
    iy = Math.round(y),
    nearest = P + iy * W + ix;
  const x0 = Math.floor(qx),
    y0 = Math.floor(y),
    cell = P + y0 * W + x0;
  const legacyError = (sx, sy, sz, ds) => {
    const sourceX = sx - (f * baseline) / sz;
    const backX = ((sourceX - cx) / f) * ds + baseline;
    return Math.abs((f * backX) / ds + cx - sx);
  };
  for (const precision of ['f64', 'f32']) {
    const module = compiler.exports.compileModuleJs(parsed.module, { precision });
    module.setBinding('mvsCameras', cameras);
    module.setBinding('mvsControl', { width: W, height: H, reference: 0, relativePoseCache: 0 });
    const run = (values, px = x, py = y, pz = z) => {
      module.setBinding('mvsDepth', values);
      return module.fns.depthSamplingProbe(px, py, pz);
    };
    const exact = run(depth);
    assert.ok(exact < 2e-4, `${precision}: planar reprojection ${exact}`);
    assert.ok(
      legacyError(x, y, z, depth[nearest]) > 1e-3,
      'fixture must expose rounded-depth error',
    );
    // A non-nearest corner may be missing or on a different surface. Both keep nearest depth.
    const other = [cell, cell + 1, cell + W, cell + W + 1].find((i) => i !== nearest);
    for (const replacement of [0, depth[other] * 2]) {
      const changed = depth.slice();
      changed[other] = replacement;
      assert.ok(
        Math.abs(run(changed) - legacyError(x, y, z, depth[nearest])) < 2e-4,
        `${precision}: depth edge ${replacement}`,
      );
    }
    const missing = depth.slice();
    missing[nearest] = 0;
    assert.equal(run(missing), 3, `${precision}: missing nearest remains invalid`);
    const stripY = 10.2;
    const stripX =
      (-0.25 +
        (f * baseline) / 3 +
        (baseline * 0.8 * cx) / 3 -
        (baseline * 0.2 * (stripY - cy)) / 3) /
      (1 + (baseline * 0.8) / 3);
    const stripZ = planeDepth(stripX, stripY);
    const stripDepth = depth[P + Math.round(stripY) * W];
    assert.ok(
      Math.abs(
        run(depth, stripX, stripY, stripZ) - legacyError(stripX, stripY, stripZ, stripDepth),
      ) < 2e-4,
      `${precision}: valid border strip keeps nearest depth`,
    );
    const borderX = 0.1,
      borderY = 10.2,
      borderZ = planeDepth(borderX, borderY);
    assert.equal(
      run(depth, borderX, borderY, borderZ),
      3,
      `${precision}: projection outside source`,
    );
    assert.equal(run(depth, x, y, -z), 3, `${precision}: point behind camera`);
    console.log(
      JSON.stringify({
        precision,
        planarError: exact,
        roundedError: legacyError(x, y, z, depth[nearest]),
      }),
    );
  }
});
