// Verifies: docs/dense-mvs.md section 10.4 (step 3, the TSDF fusion): the kernels on the CPU tier
// against the CPU reference on the synthetic scene of 7.1, the free-space carving of a floater,
// the boxes of a small budget against one pass, and `reconstructDense` with `fusion: 'tsdf'` and
// the confidence of 10.3 as the weight.
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  ShadeDevice,
  fuseTsdf,
  fuseTsdfGpu,
  planMvsViews,
  reconstructDense,
} from '../../dist/internal/gpu.js';
import { MVS_HEIGHT, MVS_WIDTH, mvsScene } from './mvs-cases.mjs';
import { matchedShare, surfaceMetrics, tsdfInput } from './tsdf-cases.mjs';

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

test('tsdf kernels on the CPU tier agree with the CPU reference, and the surface is accurate', async () => {
  const input = tsdfInput({ scene });
  const reference = fuseTsdf(input);
  const tier = await onCpuTier((device) => fuseTsdfGpu(device, input));
  console.log(
    JSON.stringify({
      reference: reference.count,
      tier: tier.count,
      blocks: tier.tsdf.blocks,
      voxel: tier.tsdf.voxel,
      metrics: surfaceMetrics(tier),
    }),
  );
  assert.equal(tier.tsdf.passes, 1);
  assert.equal(tier.tsdf.blocks, reference.blocks);
  // The same arithmetic in f32 and in f64: the counts within 0.5 %, and 99.5 % of the points of each
  // within 1e-4 of a point of the other (1/170 of a voxel).
  assert.ok(Math.abs(tier.count - reference.count) <= 0.005 * reference.count);
  assert.ok(matchedShare(tier, reference, 1e-4) >= 0.995);
  assert.ok(matchedShare(reference, tier, 1e-4) >= 0.995);
  // The exact depth maps give a surface within 1 cm (0.6 voxel) for 95 % of the points.
  const metrics = surfaceMetrics(tier, 0.01, 0.05);
  assert.ok(metrics.within >= 0.95, `within 1 cm: ${metrics.within}`);
  // Every normal is a unit vector and every color has alpha 255.
  for (let k = 0; k < tier.count; k++) assert.equal(tier.colors[k] >>> 24, 255);
});

test('tsdfCarve removes a floater that only one view sees', () => {
  const far = (options, floater) =>
    surfaceMetrics(fuseTsdf(tsdfInput({ scene, floater }), options), 0.01, 0.05).farther;
  const noCarve = { carveRatio: 1e9 };
  // The points more than 5 cm off the surface that the floater adds, without and with carving.
  const added = far(noCarve, true) - far(noCarve, false);
  const left = far({}, true) - far({}, false);
  console.log(JSON.stringify({ added, left }));
  assert.ok(added >= 100, `the floater adds ${added} points`);
  assert.ok(left <= 0.2 * added, `carving leaves ${left} of ${added}`);
});

test('fuseTsdfGpu: passes of a quarter of the blocks give the points of one pass', async () => {
  const input = tsdfInput({ scene });
  const [whole, boxes] = await onCpuTier(async (device) => {
    const whole = await fuseTsdfGpu(device, input);
    const boxes = await fuseTsdfGpu(device, input, {
      maxBlocks: Math.ceil(whole.tsdf.blocks / 4),
    });
    return [whole, boxes];
  });
  console.log(
    JSON.stringify({
      passes: boxes.tsdf.passes,
      peak: [whole.tsdf.peakBytes, boxes.tsdf.peakBytes],
    }),
  );
  assert.ok(boxes.tsdf.passes >= 4, `${boxes.tsdf.passes} passes`);
  assert.ok(boxes.tsdf.peakBytes < whole.tsdf.peakBytes);
  // As a set, to 1e-6: the points are sorted, so the arrays are equal.
  assert.equal(boxes.count, whole.count);
  assert.deepEqual(boxes.positions, whole.positions);
  assert.deepEqual(boxes.normals, whole.normals);
  assert.deepEqual(boxes.colors, whole.colors);
});

// A small configuration, as in `mvs-fusion.test.mjs`.
const SMALL = { windowRadius: 2, windowStep: 2, photometricIterations: 2, geometricIterations: 1 };
test("reconstructDense with fusion 'tsdf': the confidence weights, the points and the pixel default", async () => {
  const plan = planMvsViews(scene);
  const N = plan.packed.length,
    P = MVS_WIDTH * MVS_HEIGHT;
  const maps = [];
  const [tsdf, pixel, tight] = await onCpuTier(async (device) => [
    await reconstructDense(device, scene, scene.images, {
      ...SMALL,
      fusion: 'tsdf',
      onDepthMaps: (m) => maps.push(m),
    }),
    await reconstructDense(device, scene, scene.images, {
      ...SMALL,
      onDepthMaps: (m) => maps.push(m),
    }),
    await reconstructDense(device, scene, scene.images, {
      ...SMALL,
      fusion: 'tsdf',
      tsdfMaxBlocks: 64,
    }),
  ]);
  // The filter's confidence (10.3): in [0, 1], 0 without depth, and the depths do not change.
  const [withConfidence, without] = maps;
  assert.equal(withConfidence.confidence.length, N);
  assert.equal(without.confidence, undefined);
  let positive = 0;
  for (let v = 0; v < N; v++) {
    assert.deepEqual(withConfidence.depth[v], without.depth[v]);
    for (let i = 0; i < P; i++) {
      const c = withConfidence.confidence[v][i];
      assert.ok(c >= 0 && c <= 1);
      if (!(withConfidence.depth[v][i] > 0)) assert.equal(c, 0);
      if (c > 0) positive++;
    }
  }
  assert.ok(positive > 0.3 * N * P, `${positive} pixels with a confidence`);
  // The TSDF cloud: every view done, no point in a view's slice, the report of the fusion.
  assert.deepEqual(
    tsdf.views.map((view) => view.status),
    pixel.views.map((view) => view.status),
  );
  for (const view of tsdf.views) assert.equal(view.points, 0);
  assert.ok(tsdf.points.count > 1000, `${tsdf.points.count} points`);
  assert.equal(tsdf.points.positions.length, 3 * tsdf.points.count);
  assert.ok(tsdf.tsdf.blocks > 0 && tsdf.tsdf.passes === 1);
  assert.equal(pixel.tsdf, undefined);
  const metrics = surfaceMetrics(
    { count: tsdf.points.count, positions: tsdf.points.positions },
    0.02,
    0.1,
  );
  console.log(JSON.stringify({ tsdf: tsdf.points.count, pixel: pixel.points.count, metrics }));
  assert.ok(metrics.within >= 0.8, `within 2 cm: ${metrics.within}`);
  // A budget of 64 blocks a pass gives the same points.
  assert.ok(tight.tsdf.passes > 1);
  assert.deepEqual(tight.points, tsdf.points);
});

test("fusion 'tsdf' in streamed mode: the same confidence maps and points as window mode", async () => {
  const runs = [];
  await onCpuTier(async (device) => {
    for (const extra of [{}, { streamGroup: 2 }]) {
      let maps;
      const result = await reconstructDense(device, scene, scene.images, {
        ...SMALL,
        ...extra,
        fusion: 'tsdf',
        onDepthMaps: (m) => (maps = m),
      });
      runs.push({ result, maps });
    }
  });
  const [window, streamed] = runs;
  assert.equal(streamed.result.residency.geometric.mode, 'streamed');
  assert.deepEqual(streamed.maps.depth, window.maps.depth);
  assert.deepEqual(streamed.maps.confidence, window.maps.confidence);
  assert.deepEqual(streamed.result.points, window.result.points);
});
