// Verifies: docs/dense-mvs.md section 10.4 (step 3) on WebGPU: the TSDF kernels agree with the
// CPU reference on the synthetic scene of 7.1, and passes of a quarter of the blocks give the
// points of one pass.
import test from 'node:test';
import assert from 'node:assert/strict';
import { fuseTsdf } from '../../src/reconstruction/tsdf.ts';
import { matchedShare, surfaceMetrics, tsdfInput } from './tsdf-cases.mjs';
import { inBrowser } from './browser.mjs';

test('the TSDF kernels on WebGPU agree with the CPU reference', async () => {
  const { result, errors } = await inBrowser(
    async () => {
      const { ShadeDevice, fuseTsdfGpu } = await import('/dist/internal/gpu.js');
      const { tsdfInput } = await import('/tests/typeshade/tsdf-cases.mjs');
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
        const whole = await fuseTsdfGpu(device, input);
        const ms = performance.now() - started;
        const boxes = await fuseTsdfGpu(device, input, {
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
  const reference = fuseTsdf(tsdfInput());
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
