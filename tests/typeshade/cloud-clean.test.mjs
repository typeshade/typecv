// Verifies: docs/dense-mvs.md section 3.12 (the GPU kernels of the dense cloud cleanup, on the CPU
// tier of the device, against the CPU reference of `src/reconstruction/cloud-clean.ts`).
import test from 'node:test';
import assert from 'node:assert/strict';
import { gpuCleanPrimitives } from '../../dist/internal/gpu.js';
import {
  cellVisitOrder,
  cleanCloudWith,
  cpuPrimitives,
  meanNeighbourDistances,
  voxelFrame,
  voxelRepresentatives,
} from '../../src/reconstruction/cloud-clean.ts';
import { cleanCloud } from './cloud-clean-cases.mjs';
import { partition, withDevice } from './cloud-clean-node.mjs';

const { cloud, planeCount } = cleanCloud();

test('kNN mean distances on the device match the CPU reference', async () => {
  const reference = meanNeighbourDistances(cloud.positions, 16);
  await withDevice(async (device) => {
    const { meanDistances } = gpuCleanPrimitives(device);
    const got = await meanDistances(cloud.positions, 16);
    assert.equal(got.length, cloud.count);
    let worst = 0;
    for (let i = 0; i < cloud.count; i++)
      worst = Math.max(worst, Math.abs(got[i] - reference[i]) / reference[i]);
    console.log(JSON.stringify({ points: cloud.count, worstRelativeDifference: worst }));
    assert.ok(worst < 1e-6, `worst relative difference ${worst}`);
  });
});

test('the visit order is a permutation sorted by z, then y, then x of the cell', () => {
  const rnd = (i) => ((i * 2654435761) >>> 0) % 7;
  const n = 500;
  const cells = new Uint32Array(3 * n);
  for (let i = 0; i < 3 * n; i++) cells[i] = 1 + rnd(i + 11);
  const order = cellVisitOrder(cells, n);
  assert.equal(new Set(order).size, n);
  const key = (i) => (cells[3 * i + 2] * 8 + cells[3 * i + 1]) * 8 + cells[3 * i];
  for (let s = 1; s < n; s++) assert.ok(key(order[s - 1]) <= key(order[s]), `position ${s}`);
});

test('voxel binning on the device gives the CPU reference partition', async () => {
  for (const voxelSize of [0.02, 0.05, 3]) {
    const frame = voxelFrame(cloud.positions, voxelSize);
    const reference = voxelRepresentatives(cloud.positions, frame);
    await withDevice(async (device) => {
      const got = await gpuCleanPrimitives(device).voxelRepresentatives(cloud.positions, frame);
      assert.deepEqual(partition(got), partition(reference), `voxel size ${voxelSize}`);
    });
  }
});

test('the pipeline on the device equals the CPU pipeline', async () => {
  const cpu = await cleanCloudWith(cpuPrimitives, cloud);
  await withDevice(async (device) => {
    const gpu = await cleanCloudWith(gpuCleanPrimitives(device), cloud);
    const strip = ({ ms, tier, outlierThreshold, ...rest }) => rest;
    assert.ok(Math.abs(gpu.report.outlierThreshold / cpu.report.outlierThreshold - 1) < 1e-5);
    assert.deepEqual(strip(gpu.report), strip(cpu.report));
    assert.deepEqual(gpu.cloud.positions, cpu.cloud.positions);
    assert.deepEqual(gpu.cloud.colors, cpu.cloud.colors);
    // The plane stays; the outliers, the far singletons and the small blob go.
    assert.equal(gpu.report.before, cloud.count);
    assert.ok(gpu.report.after > 0 && gpu.report.after < planeCount);
    for (let i = 0; i < gpu.cloud.count; i++) assert.ok(gpu.cloud.positions[3 * i + 2] < 0.05);
    assert.equal(gpu.report.clusters >= 1, true);
  });
});

test('a cloud that exceeds the binding limit falls back to the CPU reference', async () => {
  await withDevice(async (device) => {
    const small = Object.create(device, { bindingBytes: { value: 64 } });
    const got = await gpuCleanPrimitives(small).meanDistances(cloud.positions, 8);
    const reference = meanNeighbourDistances(cloud.positions, 8);
    assert.deepEqual(Array.from(got), Array.from(reference));
  });
});
