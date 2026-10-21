// Verifies: docs/dense-mvs.md section 10.3 (depth confidence before fusion): `minConfidence`
// drops low-confidence depths in the filter, window mode and streamed mode agree bit for bit, and
// the default 0 changes nothing.
import test from 'node:test';
import assert from 'node:assert/strict';
import { ShadeDevice, reconstructDense } from '../../dist/internal/gpu.js';
import { mvsScene } from './mvs-cases.mjs';

const SMALL = { windowRadius: 2, windowStep: 2, photometricIterations: 1, geometricIterations: 1 };

async function dense(scene, options) {
  const device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    return await reconstructDense(device, scene, scene.images, { ...SMALL, ...options });
  } finally {
    await device.dispose();
  }
}

const depthPixels = (result) => result.views.reduce((sum, v) => sum + v.depthPixels, 0);

test('minConfidence drops depths, and streamed mode keeps the same ones as window mode', async () => {
  const scene = mvsScene();
  const off = await dense(scene, {});
  const zero = await dense(scene, { minConfidence: 0 });
  assert.deepEqual(Array.from(zero.points.positions), Array.from(off.points.positions));
  const on = await dense(scene, { minConfidence: 0.5 });
  const streamed = await dense(scene, { minConfidence: 0.5, streamGroup: 2 });
  console.log(JSON.stringify({ off: depthPixels(off), on: depthPixels(on) }));
  assert.ok(depthPixels(on) < depthPixels(off), 'a confidence of 0.5 removes depths');
  assert.ok(depthPixels(on) > 0);
  assert.deepEqual(
    streamed.views.map((v) => v.depthPixels),
    on.views.map((v) => v.depthPixels),
  );
  assert.deepEqual(Array.from(streamed.points.positions), Array.from(on.points.positions));
});

test('minConfidence outside [0, 1] is refused', async () => {
  await assert.rejects(dense(mvsScene(), { minConfidence: 1.5 }), /minConfidence/);
});
