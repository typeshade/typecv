// Verifies: docs/dense-mvs.md section 10.5 (step 4, the visibility labeling) on the CPU tier: the
// tally, `minViews` and `carveSpan` of the TSDF kernels against the CPU reference, the labeling of
// an uncarved floater, passes of a quarter of the blocks against one pass (the points, their
// attributes and their labels), and `reconstructDense` with `labeling`.
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  ShadeDevice,
  fuseTsdf,
  fuseTsdfGpu,
  labelDenseCloud,
  labelLandmarks,
  planMvsViews,
  reconstructDense,
  tsdfSamples,
  tsdfViews,
} from '../../dist/internal/gpu.js';
import { mvsScene } from './mvs-cases.mjs';
import { TSDF_SYNTHETIC, pairedPoints, surfaceMetrics, tsdfInput } from './tsdf-cases.mjs';

const scene = mvsScene();

async function onCpuTier(run) {
  const device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    const result = await run(device);
    assert.equal(device.statistics.liveBytes, 0);
    return result;
  } finally {
    await device.dispose();
  }
}

/** The points of `points` that `keep` marks. */
function kept(points, keep) {
  const index = [];
  keep.forEach((k, i) => k && index.push(i));
  return {
    count: index.length,
    positions: Float32Array.from(
      index.flatMap((i) => [...points.positions.subarray(3 * i, 3 * i + 3)]),
    ),
  };
}

test('the tally, minViews and carveSpan of the kernels on the CPU tier agree with the CPU reference', async () => {
  const options = { ...TSDF_SYNTHETIC, minViews: 2, carveSpan: 1 };
  const input = tsdfInput({ scene });
  const reference = fuseTsdf(input, options);
  const tier = await onCpuTier((device) => fuseTsdfGpu(device, input, options));
  const pairs = pairedPoints(tier, reference, 1e-4);
  let views = 0,
    samples = 0,
    weights = 0,
    free = 0;
  for (const [a, b] of pairs) {
    if (tsdfViews(tier.tallies[a]) === tsdfViews(reference.tallies[b])) views++;
    if (tsdfSamples(tier.tallies[a]) === tsdfSamples(reference.tallies[b])) samples++;
    if (Math.abs(tier.weights[a] - reference.weights[b]) <= 0.01 * reference.weights[b]) weights++;
    if (Math.abs(tier.free[a] - reference.free[b]) <= Math.max(1, 0.01 * reference.free[b])) free++;
  }
  const share = (n) => n / pairs.length;
  console.log(
    JSON.stringify({
      tier: tier.count,
      reference: reference.count,
      pairs: pairs.length,
      views: share(views),
      samples: share(samples),
      weights: share(weights),
      free: share(free),
    }),
  );
  assert.ok(Math.abs(tier.count - reference.count) <= 0.005 * reference.count);
  assert.ok(pairs.length >= 0.995 * reference.count);
  // f32 walks and f64 walks can differ by a voxel at a cell's edge: 99 % of the pairs agree.
  for (const n of [views, samples, weights, free]) assert.ok(share(n) >= 0.99, `${share(n)}`);
  // minViews 2: every point's voxel was integrated by at least 2 views.
  for (let k = 0; k < tier.count; k++) assert.ok(tsdfViews(tier.tallies[k]) >= 2);
});

test('the labeling removes a floater that carving does not delete, and keeps the surface', () => {
  const input = tsdfInput({ scene });
  // Without the deletion of tsdfCarve: the floater stays in the TSDF's points.
  const points = fuseTsdf(input, { ...TSDF_SYNTHETIC, carveRatio: 1e9 });
  const { keep, report } = labelDenseCloud(points, [], {});
  const before = surfaceMetrics(points, 0.01, 0.05),
    after = surfaceMetrics(kept(points, keep), 0.01, 0.05);
  console.log(JSON.stringify({ before, after, report }));
  assert.ok(before.farther >= 150, `the floater gives ${before.farther} far points`);
  assert.ok(after.farther <= 0.3 * before.farther, `${after.farther} of ${before.farther} left`);
  assert.ok(report.cutByVisibility > report.cutByConfidence);
  // The surface stays: at most 4 % of the points go, and the kept points are within 1 cm.
  assert.ok(report.kept >= 0.96 * points.count, `${report.kept} of ${points.count}`);
  assert.ok(after.within >= 0.97);
});

test('passes of a quarter of the blocks give the points, the attributes and the labels of one pass', async () => {
  const input = tsdfInput({ scene });
  const options = { ...TSDF_SYNTHETIC, carveRatio: 1e9 };
  const [whole, boxes] = await onCpuTier(async (device) => {
    const whole = await fuseTsdfGpu(device, input, options);
    const boxes = await fuseTsdfGpu(device, input, {
      ...options,
      maxBlocks: Math.ceil(whole.tsdf.blocks / 4),
    });
    return [whole, boxes];
  });
  assert.ok(boxes.tsdf.passes >= 4, `${boxes.tsdf.passes} passes`);
  for (const key of ['positions', 'normals', 'colors', 'weights', 'free', 'tallies'])
    assert.deepEqual(boxes[key], whole[key], key);
  const a = labelDenseCloud(whole, [], {}),
    b = labelDenseCloud(boxes, [], {});
  assert.deepEqual(b.keep, a.keep);
  assert.deepEqual(b.report, a.report);
  assert.ok(a.report.cutByVisibility > 0);
});

test('reconstructDense with labeling: the labeled points of the TSDF, and the refusals', async () => {
  // The exact maps of the scene in the packed order: the fusion alone, no depth stage.
  const plan = planMvsViews(scene);
  const input = tsdfInput({ scene });
  const at = (r) => input.views[plan.packed[r]];
  const depthMaps = {
    depth: plan.packed.map((_, r) => at(r).depth),
    normal: plan.packed.map((_, r) => at(r).normal),
    confidence: plan.packed.map((_, r) => at(r).confidence),
  };
  const tsdf = { ...TSDF_SYNTHETIC, carveRatio: 1e9 };
  let raw;
  const [plain, labeled] = await onCpuTier(async (device) => [
    await reconstructDense(device, scene, scene.images, {
      fusion: 'tsdf',
      tsdf,
      depthMaps,
      onTsdfPoints: (p) => (raw = p),
    }),
    await reconstructDense(device, scene, scene.images, {
      fusion: 'tsdf',
      tsdf,
      depthMaps,
      labeling: {},
    }),
  ]);
  assert.equal(plain.labeling, undefined);
  assert.equal(raw.count, plain.points.count);
  const report = labeled.labeling;
  console.log(JSON.stringify(report));
  assert.equal(report.points, plain.points.count);
  assert.equal(labeled.points.count, report.kept);
  assert.ok(report.kept < report.points);
  // The kept points are the labels of the TSDF's points, in their order.
  const { keep } = labelDenseCloud(raw, labelLandmarks(scene), {});
  assert.deepEqual(labeled.points.positions, kept(raw, keep).positions);
  await onCpuTier(async (device) => {
    await assert.rejects(
      reconstructDense(device, scene, scene.images, { depthMaps, labeling: {} }),
      RangeError,
    );
    await assert.rejects(
      reconstructDense(device, scene, scene.images, {
        fusion: 'tsdf',
        depthMaps,
        labeling: { minComponent: 0 },
      }),
      RangeError,
    );
  });
});
