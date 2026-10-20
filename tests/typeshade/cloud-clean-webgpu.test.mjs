// Verifies: docs/dense-mvs.md section 3.12 (the cleanup kernels on WebGPU agree with the CPU
// reference, and the time of each stage on SwiftShader is recorded).
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  cleanCloudWith,
  cpuPrimitives,
  meanNeighbourDistances,
  voxelFrame,
  voxelRepresentatives,
} from '../../src/reconstruction/cloud-clean.ts';
import { cleanCloud } from './cloud-clean-cases.mjs';
import { partition } from './cloud-clean-node.mjs';
import { inBrowser } from './browser.mjs';

test('the cleanup kernels on WebGPU agree with the CPU reference', async () => {
  const { cloud } = cleanCloud();
  const { result, errors } = await inBrowser(
    async () => {
      const { ShadeDevice, gpuCleanPrimitives, cleanCloudWith, voxelFrame } =
        await import('/dist/internal/gpu.js');
      const { cleanCloud } = await import('/tests/typeshade/cloud-clean-cases.mjs');
      const { cloud } = cleanCloud();
      const device = await ShadeDevice.open({ backend: 'webgpu', poolBytes: 0 });
      try {
        const primitives = gpuCleanPrimitives(device);
        const distances = Array.from(await primitives.meanDistances(cloud.positions, 16));
        const frame = voxelFrame(cloud.positions, 0.05);
        const rep = Array.from(await primitives.voxelRepresentatives(cloud.positions, frame));
        const pipeline = await cleanCloudWith(primitives, cloud);
        return {
          distances,
          rep,
          report: pipeline.report,
          after: pipeline.cloud.count,
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
  const reference = meanNeighbourDistances(cloud.positions, 16);
  let worst = 0;
  for (let i = 0; i < cloud.count; i++)
    worst = Math.max(worst, Math.abs(result.distances[i] - reference[i]) / reference[i]);
  assert.ok(worst < 1e-6, `worst relative difference ${worst}`);
  const frame = voxelFrame(cloud.positions, 0.05);
  assert.deepEqual(partition(result.rep), partition(voxelRepresentatives(cloud.positions, frame)));
  const cpu = await cleanCloudWith(cpuPrimitives, cloud);
  assert.equal(result.report.tier, 'webgpu');
  for (const key of ['before', 'afterOutliers', 'afterVoxel', 'after', 'clusters'])
    assert.equal(result.report[key], cpu.report[key], key);
  console.log(JSON.stringify({ worstRelativeDifference: worst, report: result.report }));
});
