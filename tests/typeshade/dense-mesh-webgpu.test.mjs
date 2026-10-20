// Verifies: docs/dense-mvs.md section 3.13 (the mesh kernels on WebGPU agree with the CPU
// reference).
import test from 'node:test';
import assert from 'node:assert/strict';
import { meshDenseCloud } from '../../src/reconstruction/dense-mesh.ts';
import { sphereCloud } from './dense-mesh-cases.mjs';
import { inBrowser } from './browser.mjs';

test('the mesh kernels on WebGPU agree with the CPU reference', async () => {
  const cloud = sphereCloud(4000);
  const { result, errors } = await inBrowser(
    async () => {
      const { ShadeDevice, meshDenseCloudGpu } = await import('/dist/internal/gpu.js');
      const { sphereCloud } = await import('/tests/typeshade/dense-mesh-cases.mjs');
      const device = await ShadeDevice.open({ backend: 'webgpu', poolBytes: 0 });
      try {
        const got = await meshDenseCloudGpu(device, sphereCloud(4000), { voxelSize: 0.08 });
        return {
          report: got.report,
          indices: Array.from(got.mesh.indices),
          positions: Array.from(got.mesh.positions),
          colors: Array.from(got.mesh.colors),
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
  assert.equal(result.liveBytes, 0);
  const reference = meshDenseCloud(cloud, { voxelSize: 0.08 }).mesh;
  assert.equal(result.report.tier, 'webgpu');
  assert.deepEqual(result.indices, Array.from(reference.indices));
  let worst = 0;
  for (let i = 0; i < reference.positions.length; i++)
    worst = Math.max(worst, Math.abs(result.positions[i] - reference.positions[i]));
  assert.ok(worst <= 1e-5, `worst position difference ${worst}`);
  for (let i = 0; i < reference.colors.length; i++)
    assert.ok(Math.abs(result.colors[i] - reference.colors[i]) <= 1);
  console.log(JSON.stringify({ worstPositionDifference: worst, report: result.report }));
});
