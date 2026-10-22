// Verifies: docs/dense-mvs.md section 10.5 (step 4) on WebGPU: the TSDF kernels' tally, `minViews`
// and point attributes agree with the CPU reference on the synthetic scene of 7.1, the labels of
// the WebGPU points agree with the labels of the reference's points, and passes of a quarter of
// the blocks give the attributes and the labels of one pass.
import test from 'node:test';
import assert from 'node:assert/strict';
import { fuseTsdf, tsdfSamples, tsdfViews } from '../../src/reconstruction/tsdf.ts';
import { labelDenseCloud } from '../../src/reconstruction/dense-label.ts';
import { TSDF_SYNTHETIC, pairedPoints, tsdfInput } from './tsdf-cases.mjs';
import { inBrowser } from './browser.mjs';

const OPTIONS = { ...TSDF_SYNTHETIC, carveRatio: 1e9, minViews: 2 };

test('the TSDF attributes and the labels on WebGPU agree with the CPU reference, and with passes', async () => {
  const { result, errors } = await inBrowser(
    async (options) => {
      const { ShadeDevice, fuseTsdfGpu } = await import('/dist/internal/gpu.js');
      const { tsdfInput } = await import('/tests/typeshade/tsdf-cases.mjs');
      const input = tsdfInput();
      const device = await ShadeDevice.open({ backend: 'webgpu', poolBytes: 0 });
      const plain = (r) => ({
        count: r.count,
        positions: Array.from(r.positions),
        normals: Array.from(r.normals),
        colors: Array.from(r.colors),
        weights: Array.from(r.weights),
        free: Array.from(r.free),
        tallies: Array.from(r.tallies),
        passes: r.tsdf.passes,
      });
      try {
        const whole = await fuseTsdfGpu(device, input, options);
        const boxes = await fuseTsdfGpu(device, input, {
          ...options,
          maxBlocks: Math.ceil(whole.tsdf.blocks / 4),
        });
        return { whole: plain(whole), boxes: plain(boxes), liveBytes: device.statistics.liveBytes };
      } finally {
        await device.dispose();
      }
    },
    OPTIONS,
    { timeoutMs: 600000 },
  );
  assert.deepEqual(errors, []);
  assert.equal(result.liveBytes, 0);
  const typed = (r) => ({
    count: r.count,
    positions: Float32Array.from(r.positions),
    normals: Uint32Array.from(r.normals),
    colors: Uint32Array.from(r.colors),
    weights: Uint32Array.from(r.weights),
    free: Uint32Array.from(r.free),
    tallies: Uint32Array.from(r.tallies),
  });
  const gpu = typed(result.whole),
    boxes = typed(result.boxes);
  const reference = fuseTsdf(tsdfInput(), OPTIONS);
  // The attributes of the pairs of points: 99 % agree (f32 walks against f64 walks).
  const pairs = pairedPoints(gpu, reference, 1e-4);
  let same = 0;
  for (const [a, b] of pairs)
    if (
      tsdfViews(gpu.tallies[a]) === tsdfViews(reference.tallies[b]) &&
      tsdfSamples(gpu.tallies[a]) === tsdfSamples(reference.tallies[b]) &&
      Math.abs(gpu.weights[a] - reference.weights[b]) <= 0.01 * reference.weights[b] &&
      Math.abs(gpu.free[a] - reference.free[b]) <= Math.max(1, 0.01 * reference.free[b])
    )
      same++;
  for (let k = 0; k < gpu.count; k++) assert.ok(tsdfViews(gpu.tallies[k]) >= 2);
  // The labels: the kept points of each tier have a kept point of the other within 1e-4.
  const a = labelDenseCloud(gpu, [], {}),
    b = labelDenseCloud(reference, [], {});
  let agree = 0;
  for (const [i, j] of pairs) if (a.keep[i] === b.keep[j]) agree++;
  console.log(
    JSON.stringify({
      gpu: gpu.count,
      reference: reference.count,
      pairs: pairs.length,
      attributes: same / pairs.length,
      labels: agree / pairs.length,
      gpuReport: a.report,
      referenceKept: b.report.kept,
    }),
  );
  assert.ok(pairs.length >= 0.995 * reference.count);
  assert.ok(same >= 0.99 * pairs.length, `${same} of ${pairs.length}`);
  assert.ok(agree >= 0.99 * pairs.length, `${agree} of ${pairs.length}`);
  assert.ok(Math.abs(a.report.kept - b.report.kept) <= 0.01 * b.report.kept);
  // The passes: the same points, attributes and labels, as arrays (both are sorted).
  assert.ok(result.boxes.passes >= 4);
  for (const key of ['positions', 'normals', 'colors', 'weights', 'free', 'tallies'])
    assert.deepEqual(boxes[key], gpu[key], key);
  assert.deepEqual(labelDenseCloud(boxes, [], {}).keep, a.keep);
});
