// Verifies: docs/dense-mvs.md section 3.13 (the mesh kernels on the CPU tier of the device, against
// the CPU reference of `src/reconstruction/dense-mesh.ts`).
import test from 'node:test';
import assert from 'node:assert/strict';
import { meshDenseCloudGpu } from '../../dist/internal/gpu.js';
import { meshDenseCloud } from '../../src/reconstruction/dense-mesh.ts';
import { planeCloud, sphereCloud } from './dense-mesh-cases.mjs';
import { withDevice } from './cloud-clean-node.mjs';

/** The two meshes agree: the same faces, the positions within `tolerance`, the colors within 1. */
export function assertSameMesh(got, reference, tolerance = 1e-5) {
  assert.equal(got.faceCount, reference.faceCount);
  assert.equal(got.vertexCount, reference.vertexCount);
  assert.deepEqual(Array.from(got.indices), Array.from(reference.indices));
  let worst = 0;
  for (let i = 0; i < got.positions.length; i++)
    worst = Math.max(worst, Math.abs(got.positions[i] - reference.positions[i]));
  assert.ok(worst <= tolerance, `worst position difference ${worst}`);
  for (let i = 0; i < got.colors.length; i++)
    assert.ok(Math.abs(got.colors[i] - reference.colors[i]) <= 1, `color ${i}`);
  return worst;
}

for (const [name, cloud, options] of [
  ['a sphere', sphereCloud(4000), { voxelSize: 0.08 }],
  ['a sphere at the default options', sphereCloud(6000), {}],
  ['a plane', planeCloud(40, 1), { voxelSize: 0.04 }],
]) {
  test(`the mesh of ${name} on the device equals the CPU reference`, async () => {
    const reference = meshDenseCloud(cloud, options);
    await withDevice(async (device) => {
      const got = await meshDenseCloudGpu(device, cloud, options);
      const worst = assertSameMesh(got.mesh, reference.mesh);
      assert.ok(got.mesh.faceCount > 500);
      assert.equal(got.report.cubes, reference.report.cubes);
      assert.equal(got.report.nodes, reference.report.nodes);
      console.log(
        JSON.stringify({ name, faces: got.mesh.faceCount, worstPositionDifference: worst }),
      );
    });
  });
}

test('an empty result and a cloud over the binding limit', async () => {
  const cloud = sphereCloud(3000);
  const reference = meshDenseCloud(cloud, { voxelSize: 0.08 });
  await withDevice(async (device) => {
    // Every node lies beyond the points: the minimum weight no node reaches gives no cube.
    const none = await meshDenseCloudGpu(device, cloud, { voxelSize: 0.08, minWeight: 1e9 });
    assert.equal(none.mesh.faceCount, 0);
    const small = Object.create(device, { bindingBytes: { value: 64 } });
    const got = await meshDenseCloudGpu(small, cloud, { voxelSize: 0.08 });
    assert.equal(got.report.tier, 'cpu');
    assert.deepEqual(Array.from(got.mesh.indices), Array.from(reference.mesh.indices));
  });
});
