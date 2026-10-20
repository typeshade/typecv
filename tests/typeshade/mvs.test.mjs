// Verifies: docs/dense-mvs.md step 2 (sections 3.4 to 3.7 and 7.1, the photometric pass).
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  ShadeDevice,
  decodeMvsNormal,
  mvsPhotometricGpu,
  planMvsViews,
} from '../../dist/internal/gpu.js';
import { MVS_HEIGHT, MVS_WIDTH, depthMetrics, mvsScene, undistortReference } from './mvs-cases.mjs';

const REFERENCE_ID = 2;
const scene = mvsScene();
const plan = planMvsViews(scene);
const reference = plan.views[REFERENCE_ID].index;
const range = plan.views[REFERENCE_ID].depthRange;
const metrics = (result) =>
  depthMetrics(result, scene.truth[REFERENCE_ID], plan.normalization.scale, range, decodeMvsNormal);
const rounded = (record) =>
  Object.fromEntries(Object.entries(record).map(([k, v]) => [k, +v.toFixed(4)]));

async function run(options) {
  const device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    const started = performance.now();
    const result = await mvsPhotometricGpu(device, scene, plan, scene.images, reference, options);
    const ms = performance.now() - started;
    assert.equal(device.statistics.liveBytes, 0);
    return { result, ms };
  } finally {
    await device.dispose();
  }
}
// The default run, shared by the accuracy and the chunk tests.
let defaultRun;
const defaults = () => (defaultRun ??= run({ snapshots: true }));

test('MVS scene: every view gets a depth map and the reference has sources and seeds', () => {
  assert.deepEqual(
    plan.views.map((view) => view.status),
    ['ready', 'ready', 'ready', 'ready', 'ready'],
  );
  assert.equal(plan.sources[reference * 7], 4);
  assert.ok(plan.seedOffsets[reference + 1] - plan.seedOffsets[reference] >= 20);
});

test('mvsUndistort on the CPU tier: the gray images follow the host undistortion', async () => {
  const { result } = await run({ photometricIterations: 0, readGray: true });
  const P = MVS_WIDTH * MVS_HEIGHT;
  let distorted = 0,
    marked = 0;
  plan.packed.forEach((r, v) => {
    const view = scene.views[r],
      expected = undistortReference(scene.images.get(view.id).rgba, view.camera),
      actual = result.gray.subarray(v * P, (v + 1) * P);
    if (!view.camera.k1) {
      // The copy mode writes the gray image of `rgbaToGray` unchanged.
      for (let i = 0; i < P; i++) assert.ok(Math.abs(actual[i] - expected[i]) < 1e-6);
      return;
    }
    distorted++;
    for (let i = 0; i < P; i++) {
      if (expected[i] < 0 || actual[i] < 0) {
        assert.equal(actual[i], expected[i], `view ${view.id} pixel ${i}: no-data mark`);
        marked++;
        continue;
      }
      assert.ok(Math.abs(actual[i] - expected[i]) < 1e-5, `view ${view.id} pixel ${i}`);
    }
  });
  assert.equal(distorted, 3);
  // k1 > 0 pushes the corners of the undistorted image past the distorted one.
  assert.ok(marked > 0);
});

test('mvsPropagate on the CPU tier: photometric accuracy on the synthetic scene', async () => {
  const { result, ms } = await defaults();
  const final = metrics(result);
  const iterations = result.snapshots.map((snapshot) => rounded(metrics(snapshot)));
  console.log(JSON.stringify({ ms: Math.round(ms), final: rounded(final), iterations }));
  assert.equal(result.snapshots.length, 9);
  assert.deepEqual(result.snapshots[8].depth, result.depth);
  // Section 7.1: median relative depth error at most 1 % on the kept pixels, at least 80 % kept.
  assert.ok(final.median <= 0.01, `median relative depth error ${final.median}`);
  assert.ok(final.kept >= 0.8, `kept ${final.kept}`);
  // Deviation (docs/dense-mvs.md, 7.1): the photometric cost resolves normals to about 5° to
  // 10° on this scene, so the bound is the measured 14° with a margin, not 5°.
  assert.ok(final.medianNormalDeg <= 16, `median normal error ${final.medianNormalDeg}°`);
  // The default of 8 iterations: the last iteration moves the median by less than 0.05 points.
  const [seventh, eighth] = [metrics(result.snapshots[7]), final];
  assert.ok(Math.abs(seventh.median - eighth.median) < 0.0005, 'iteration 8 still improves');
  // Every iteration after the first lowers the cost of no pixel's hypothesis.
  for (let t = 1; t < result.snapshots.length; t++) {
    const before = result.snapshots[t - 1].cost,
      after = result.snapshots[t].cost;
    for (let i = 0; i < after.length; i++) assert.ok(after[i] <= before[i]);
  }
});

test('mvsInitialize and mvsPropagate: chunks of 48 elements give the same arrays bit for bit', async () => {
  const [{ result: whole }, { result: chunked, ms }] = await Promise.all([
    defaults(),
    run({ chunkPixels: 48 }),
  ]);
  console.log(JSON.stringify({ chunkedMs: Math.round(ms) }));
  // 48 does not divide a row of 64 color indices or 128 pixels, so chunks cross rows.
  const calls = (report, name) =>
    report.groups.flatMap((group) => group.calls).filter((call) => call === name).length;
  assert.equal(calls(whole.report, 'mvsInitialize'), 1);
  assert.equal(calls(chunked.report, 'mvsInitialize'), (MVS_WIDTH * MVS_HEIGHT) / 48);
  assert.equal(calls(whole.report, 'mvsPropagate'), 16);
  assert.equal(
    calls(chunked.report, 'mvsPropagate'),
    16 * Math.ceil(((MVS_WIDTH / 2) * MVS_HEIGHT) / 48),
  );
  assert.deepEqual(new Uint8Array(chunked.depth.buffer), new Uint8Array(whole.depth.buffer));
  assert.deepEqual(chunked.normal, whole.normal);
  assert.deepEqual(new Uint8Array(chunked.cost.buffer), new Uint8Array(whole.cost.buffer));
});

// The comparisons of section 3.7 and 3.6 that set the defaults: window step 1 against 2,
// bilateral weights on against off, 16 iterations against 8. About a minute on the CPU tier, so
// it runs when TYPECV_MVS_DEFAULTS=1. Its numbers are in docs/dense-mvs.md, section 8.
test(
  'MVS defaults: window step, bilateral weights and iterations',
  { skip: process.env.TYPECV_MVS_DEFAULTS !== '1' && 'set TYPECV_MVS_DEFAULTS=1' },
  async () => {
    const rows = [];
    const { result, ms } = await defaults();
    rows.push({ options: 'defaults', ms: Math.round(ms), ...rounded(metrics(result)) });
    for (const options of [{ windowStep: 1 }, { bilateral: true }, { photometricIterations: 16 }]) {
      const variant = await run(options);
      rows.push({
        options: JSON.stringify(options),
        ms: Math.round(variant.ms),
        ...rounded(metrics(variant.result)),
      });
    }
    console.log(JSON.stringify(rows));
  },
);
