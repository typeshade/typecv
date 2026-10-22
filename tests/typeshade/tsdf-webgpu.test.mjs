// Verifies: docs/dense-mvs.md section 10.4 (step 3) on WebGPU: the TSDF kernels agree with the
// CPU reference on the synthetic scene of 7.1, and passes of a quarter of the blocks give the
// points of one pass.
import test from 'node:test';
import assert from 'node:assert/strict';
import { fuseTsdf } from '../../src/reconstruction/tsdf.ts';
import {
  TSDF_SYNTHETIC,
  matchedShare,
  surfaceMetrics,
  tsdfInput,
  untrustedCarvingInput,
} from './tsdf-cases.mjs';
import { inBrowser } from './browser.mjs';

test('view-weighted visibility preserves weakly contradicted surfaces on WebGPU', async () => {
  const { result: r, errors } = await inBrowser(async () => {
    const api = await import('/dist/internal/gpu.js');
    const { weightedVisibilityCase } = await import('/tests/typeshade/tsdf-cases.mjs');
    return weightedVisibilityCase(api, 'webgpu');
  });
  assert.deepEqual(errors, []);
  assert.equal(r.liveBytes, 0);
  assert.equal(r.baseline.count, 196);
  for (const { whole, uncarved, boxes, passes } of r.variants) {
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
  const reference = fuseTsdf(untrustedCarvingInput(), {
    levels: 1,
    voxelScale: 1,
    truncation: 3,
    visibility: 'view-weighted',
  });
  assert.equal(matchedShare(r.baseline, reference, 0.0001), 1);
});

test('zero-weight observations cannot carve a trusted surface on WebGPU', async () => {
  const { result, errors } = await inBrowser(async () => {
    const { ShadeDevice, fuseTsdfGpu } = await import('/dist/internal/gpu.js');
    const { untrustedCarvingInput } = await import('/tests/typeshade/tsdf-cases.mjs');
    const device = await ShadeDevice.open({ backend: 'webgpu', poolBytes: 0 });
    const options = { levels: 1, voxelScale: 1, truncation: 3 };
    try {
      const plain = (r) => ({
        count: r.count,
        positions: Array.from(r.positions),
        normals: Array.from(r.normals),
        colors: Array.from(r.colors),
      });
      const baseline = await fuseTsdfGpu(device, untrustedCarvingInput(), options);
      const variants = [];
      for (const kind of ['zero-confidence', 'grazing', 'rounded-zero']) {
        const input = untrustedCarvingInput(kind);
        const whole = await fuseTsdfGpu(device, input, options);
        const boxes = await fuseTsdfGpu(device, input, {
          ...options,
          maxBlocks: Math.ceil(whole.tsdf.blocks / 4),
        });
        variants.push({
          kind,
          whole: plain(whole),
          boxes: plain(boxes),
          passes: boxes.tsdf.passes,
        });
      }
      return { baseline: plain(baseline), variants, liveBytes: device.statistics.liveBytes };
    } finally {
      await device.dispose();
    }
  });
  assert.deepEqual(errors, []);
  assert.equal(result.liveBytes, 0);
  const reference = fuseTsdf(untrustedCarvingInput(), { levels: 1, voxelScale: 1, truncation: 3 });
  assert.equal(result.baseline.count, 196);
  assert.equal(matchedShare(result.baseline, reference, 0.0001), 1);
  for (const { kind, whole, boxes, passes } of result.variants) {
    assert.deepEqual(whole, result.baseline, kind);
    assert.deepEqual(boxes, whole, kind);
    assert.ok(passes > 1, kind);
  }
});

test('view-weighted TSDF on a 3D scene matches the reference and the block budget changes no points', async () => {
  const { result: r, errors } = await inBrowser(
    async () => {
      const { ShadeDevice, fuseTsdfGpu } = await import('/dist/internal/gpu.js');
      const { tsdfInput, TSDF_SYNTHETIC } = await import('/tests/typeshade/tsdf-cases.mjs');
      const device = await ShadeDevice.open({ backend: 'webgpu', poolBytes: 0 });
      const options = { ...TSDF_SYNTHETIC, visibility: 'view-weighted' };
      const plain = (r) => ({
        count: r.count,
        positions: [...r.positions],
        colors: [...r.colors],
        normals: [...r.normals],
        weights: [...r.weights],
        free: [...r.free],
        freeWeights: [...r.freeWeights],
        tallies: [...r.tallies],
      });
      try {
        const input = tsdfInput();
        const whole = await fuseTsdfGpu(device, input, options);
        const boxes = await fuseTsdfGpu(device, input, {
          ...options,
          maxBlocks: Math.ceil(whole.tsdf.blocks / 4),
        });
        return {
          whole: plain(whole),
          boxes: plain(boxes),
          passes: boxes.tsdf.passes,
          liveBytes: device.statistics.liveBytes,
        };
      } finally {
        await device.dispose();
      }
    },
    undefined,
    { timeoutMs: 120000 },
  );
  assert.deepEqual(errors, []);
  assert.equal(r.liveBytes, 0);
  assert.ok(r.passes > 1);
  assert.deepEqual(r.boxes, r.whole);
  const reference = fuseTsdf(tsdfInput(), { ...TSDF_SYNTHETIC, visibility: 'view-weighted' });
  assert.ok(Math.abs(r.whole.count - reference.count) <= 0.005 * reference.count);
  assert.ok(matchedShare(r.whole, reference, 0.0001) >= 0.995);
  assert.ok(matchedShare(reference, r.whole, 0.0001) >= 0.995);
  assert.ok(surfaceMetrics(r.whole, 0.01, 0.05).within >= 0.95);
});

test('MVS fusion previews on WebGPU keep world coordinates and original view ids', async () => {
  const { result: r, errors } = await inBrowser(
    async () => {
      const api = await import('/dist/internal/gpu.js');
      const { mvsFeedbackCase } = await import('/tests/typeshade/tsdf-feedback-cases.mjs');
      return mvsFeedbackCase(api, 'webgpu');
    },
    undefined,
    { timeoutMs: 120000 },
  );
  assert.deepEqual(errors, []);
  assert.ok(r.observed.count > 0);
  assert.deepEqual(r.observed, r.baseline);
  assert.deepEqual(r.previewPoints, r.finalPoints);
  assert.deepEqual(r.finalEvents, ['points']);
  assert.deepEqual(r.viewIds, [10, 11, 12]);
  assert.equal(r.liveBytes, 0);
});

test('TSDF live feedback preserves WebGPU results and submissions and owns preview arrays', async () => {
  const { result, errors } = await inBrowser(
    async () => {
      const api = await import('/dist/internal/gpu.js');
      const { feedbackCase } = await import('/tests/typeshade/tsdf-feedback-cases.mjs');
      return feedbackCase(api, 'webgpu');
    },
    undefined,
    { timeoutMs: 120000 },
  );
  assert.deepEqual(errors, []);
  assert.ok(result.baseline.count > 0);
  assert.deepEqual(result.observed, result.baseline);
  assert.equal(
    result.partCounts.reduce((a, b) => a + b, 0),
    result.baseline.count,
  );
  assert.ok(result.progress.some((e) => e.phase === 'integrate' && e.done > 0));
  assert.equal(result.cancelled, true);
  assert.equal(result.liveBytes, 0);
});

test('the TSDF kernels on WebGPU agree with the CPU reference', async () => {
  const { result, errors } = await inBrowser(
    async () => {
      const { ShadeDevice, fuseTsdfGpu } = await import('/dist/internal/gpu.js');
      const { TSDF_SYNTHETIC, tsdfInput } = await import('/tests/typeshade/tsdf-cases.mjs');
      const input = tsdfInput();
      const device = await ShadeDevice.open({ backend: 'webgpu', poolBytes: 0 });
      const plain = (r) => ({
        count: r.count,
        positions: Array.from(r.positions),
        normals: Array.from(r.normals),
        colors: Array.from(r.colors),
        tsdf: { ...r.tsdf, report: undefined },
      });
      try {
        const started = performance.now();
        const whole = await fuseTsdfGpu(device, input, TSDF_SYNTHETIC);
        const ms = performance.now() - started;
        const boxes = await fuseTsdfGpu(device, input, {
          ...TSDF_SYNTHETIC,
          maxBlocks: Math.ceil(whole.tsdf.blocks / 4),
        });
        return {
          ms,
          whole: plain(whole),
          boxes: plain(boxes),
          liveBytes: device.statistics.liveBytes,
        };
      } finally {
        await device.dispose();
      }
    },
    undefined,
    { timeoutMs: 600000 },
  );
  assert.deepEqual(errors, []);
  assert.equal(result.liveBytes, 0);
  const typed = (r) => ({ ...r, positions: Float32Array.from(r.positions) });
  const gpu = typed(result.whole),
    boxes = typed(result.boxes);
  const reference = fuseTsdf(tsdfInput(), TSDF_SYNTHETIC);
  console.log(
    JSON.stringify({
      ms: Math.round(result.ms),
      gpu: gpu.count,
      reference: reference.count,
      tsdf: result.whole.tsdf,
      passes: result.boxes.tsdf.passes,
    }),
  );
  assert.equal(result.whole.tsdf.blocks, reference.blocks);
  assert.ok(Math.abs(gpu.count - reference.count) <= 0.005 * reference.count);
  assert.ok(matchedShare(gpu, reference, 1e-4) >= 0.995);
  assert.ok(matchedShare(reference, gpu, 1e-4) >= 0.995);
  assert.ok(surfaceMetrics(gpu, 0.01, 0.05).within >= 0.95);
  // The passes: the same points as a set, to 1e-6.
  assert.ok(result.boxes.tsdf.passes >= 4);
  assert.equal(boxes.count, gpu.count);
  assert.equal(matchedShare(boxes, gpu, 1e-6), 1);
  assert.equal(matchedShare(gpu, boxes, 1e-6), 1);
});
