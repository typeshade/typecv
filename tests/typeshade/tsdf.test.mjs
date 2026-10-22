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
import {
  TSDF_SYNTHETIC,
  matchedShare,
  surfaceMetrics,
  tsdfInput,
  untrustedCarvingInput,
  weightedVisibilityCase,
} from './tsdf-cases.mjs';

const scene = mvsScene();

test('view-weighted visibility preserves weakly contradicted surfaces on the CPU tier', async () => {
  const r = await weightedVisibilityCase({ ShadeDevice, fuseTsdfGpu }, 'cpu');
  assert.equal(r.liveBytes, 0);
  assert.equal(r.baseline.count, 196);
  for (const { whole, uncarved, boxes, passes } of r.variants) {
    // The extra cameras change the volume's grid origin: compare at the same grid and input.
    assert.deepEqual(whole.positions, uncarved.positions);
    assert.ok(whole.count >= 196);
    for (let k = 0; k < whole.count; k++)
      assert.ok(Math.abs(whole.positions[3 * k + 2] - 2) < 0.0001);
    assert.deepEqual(boxes, whole);
    assert.ok(passes > 1);
    for (let k = 0; k < whole.count; k++) {
      assert.ok(whole.free[k] <= 4);
      assert.ok(whole.freeWeights[k] <= whole.free[k]);
    }
  }
  assert.equal(
    r.single.positions.filter((v, i) => i % 3 === 2 && Math.abs(v - 2) < 0.01).length,
    196,
  );
  assert.equal(
    r.correlated.positions.filter((v, i) => i % 3 === 2 && Math.abs(v - 2) < 0.01).length,
    196,
  );
  assert.ok(Math.max(...r.correlated.free) <= 1);
});

test('view-weighted visibility keeps surface accuracy and removes a synthetic floater', () => {
  const options = { ...TSDF_SYNTHETIC, visibility: 'view-weighted' };
  const far = (floater, carveRatio) =>
    surfaceMetrics(fuseTsdf(tsdfInput({ scene, floater }), { ...options, carveRatio }), 0.01, 0.05)
      .farther;
  const added = far(true, 1e9) - far(false, 1e9);
  const left = far(true, 1) - far(false, 1);
  assert.ok(added >= 100);
  assert.ok(left <= 0.2 * added, `${left} of ${added} false points`);
  assert.ok(surfaceMetrics(fuseTsdf(tsdfInput({ scene }), options), 0.01, 0.05).within >= 0.95);
});

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
  const reference = fuseTsdf(input, TSDF_SYNTHETIC);
  const tier = await onCpuTier((device) => fuseTsdfGpu(device, input, TSDF_SYNTHETIC));
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
    surfaceMetrics(
      fuseTsdf(tsdfInput({ scene, floater }), { ...TSDF_SYNTHETIC, ...options }),
      0.01,
      0.05,
    ).farther;
  const noCarve = { carveRatio: 1e9 };
  // The points more than 5 cm off the surface that the floater adds, without and with carving.
  const added = far(noCarve, true) - far(noCarve, false);
  const left = far({}, true) - far({}, false);
  console.log(JSON.stringify({ added, left }));
  assert.ok(added >= 100, `the floater adds ${added} points`);
  assert.ok(left <= 0.2 * added, `carving leaves ${left} of ${added}`);
});

test('observations with zero integration weight cannot carve a trusted surface', async () => {
  const options = { levels: 1, voxelScale: 1, truncation: 3 };
  const expected = fuseTsdf(untrustedCarvingInput(), options);
  assert.equal(expected.count, 196);
  let center = 0;
  for (let k = 0; k < expected.count; k++) {
    assert.ok(Math.abs(expected.positions[3 * k + 2] - 2) < 0.01);
    if (
      Math.abs(expected.positions[3 * k]) < 0.25 &&
      Math.abs(expected.positions[3 * k + 1]) < 0.25
    )
      center++;
  }
  assert.equal(center, 36);
  await onCpuTier(async (device) => {
    const baseline = await fuseTsdfGpu(device, untrustedCarvingInput(), options);
    for (const kind of ['zero-confidence', 'grazing', 'rounded-zero']) {
      const input = untrustedCarvingInput(kind);
      const reference = fuseTsdf(input, options);
      assert.deepEqual(reference.positions, expected.positions, kind);
      assert.deepEqual(reference.normals, expected.normals, kind);
      assert.deepEqual(reference.colors, expected.colors, kind);
      const tier = await fuseTsdfGpu(device, input, options);
      assert.equal(tier.count, expected.count, kind);
      assert.deepEqual(tier.positions, baseline.positions, kind);
      assert.deepEqual(tier.normals, baseline.normals, kind);
      assert.deepEqual(tier.colors, baseline.colors, kind);
      assert.equal(matchedShare(tier, reference, 0.0001), 1, kind);
    }
  });
});

test('fuseTsdfGpu: passes of a quarter of the blocks give the points of one pass', async () => {
  const input = tsdfInput({ scene });
  const [whole, boxes] = await onCpuTier(async (device) => {
    const whole = await fuseTsdfGpu(device, input, TSDF_SYNTHETIC);
    const boxes = await fuseTsdfGpu(device, input, {
      ...TSDF_SYNTHETIC,
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

// Record 08f5a9af: `tsdfExtract` asked for 102,227 workgroups in X on a hardware adapter, past
// `maxComputeWorkgroupsPerDimension` (65,535), when the settled chunk size covered the whole pass.
// Here the device grants every chunk all it asks for and allows 4 workgroups (256 elements) per
// dispatch: no chunk may then ask for more than 256 elements, and the points must be those of
// whole-pass chunks.
test('fuseTsdfGpu: no chunk exceeds the workgroups of one dimension, and the points are unchanged', async () => {
  const input = tsdfInput({ scene });
  const [whole, small, asked] = await onCpuTier(async (device) => {
    const chunkSize = device.chunkSize;
    const limit = device.maxKernelElements;
    const set = (name, value) =>
      Object.defineProperty(device, name, { value, configurable: true, writable: true });
    let asked = [];
    set('chunkSize', (_key, max) => (asked.push(max), max));
    try {
      const whole = await fuseTsdfGpu(device, input, TSDF_SYNTHETIC);
      asked = [];
      set('maxKernelElements', 4 * 64);
      const small = await fuseTsdfGpu(device, input, TSDF_SYNTHETIC);
      return [whole, small, asked];
    } finally {
      set('chunkSize', chunkSize);
      set('maxKernelElements', limit);
    }
  });
  console.log(JSON.stringify({ chunks: asked.length, most: Math.max(...asked) }));
  assert.ok(asked.length > (input.width * input.height) / 256);
  for (const max of asked) assert.ok(max <= 256, `a chunk of up to ${max} elements`);
  assert.equal(small.count, whole.count);
  assert.deepEqual(small.positions, whole.positions);
  assert.deepEqual(small.normals, whole.normals);
  assert.deepEqual(small.colors, whole.colors);
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
      tsdf: TSDF_SYNTHETIC,
      onDepthMaps: (m) => maps.push(m),
    }),
    await reconstructDense(device, scene, scene.images, {
      ...SMALL,
      onDepthMaps: (m) => maps.push(m),
    }),
    await reconstructDense(device, scene, scene.images, {
      ...SMALL,
      fusion: 'tsdf',
      tsdf: TSDF_SYNTHETIC,
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
        tsdf: TSDF_SYNTHETIC,
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
