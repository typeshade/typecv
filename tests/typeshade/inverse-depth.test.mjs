import test from 'node:test';
import assert from 'node:assert/strict';
import * as api from '../../dist/internal/gpu.js';
import { inverseDepthGpuCases, inverseDepthSessionCase } from './inverse-depth-cases.mjs';
import { inverseDepthScene } from './inverse-depth-cases.mjs';
import { InverseDepthTracker as CoreTracker } from '../../dist/index.js';
test('TypeShade inverse-depth linearization and joint covariance match CPU without live arrays', async () => {
  const r = await inverseDepthGpuCases(api, 'cpu');
  assert.equal(r.liveBytes, 0);
  console.log(JSON.stringify(r));
});

test('inverse-depth GPU admission rejects budget errors and drops arrays after device-step failure', async () => {
  const model = inverseDepthScene(),
    device = await api.ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    const point = api.initializeInverseDepth(
        { ...model.views[0], pixel: model.landmarks[0].observations[0].pixel },
        model.landmarks[0].xyz,
      ),
      job = { measurements: [{ view: model.views[1], point }] },
      small = Object.create(device, { workingBytes: { value: 1 } });
    await assert.rejects(api.linearizeInverseDepthGpu(small, job), /budget/);
    assert.equal(device.statistics.liveBytes, 0);
    const failing = Object.create(device, {
      planSteps: {
        value: async () => {
          throw new Error('injected shader failure');
        },
      },
    });
    await assert.rejects(api.linearizeInverseDepthGpu(failing, job), /injected shader failure/);
    assert.equal(device.statistics.liveBytes, 0);
  } finally {
    await device.dispose();
  }
});
test('GPU mapper session accepts a core-bundle tracker and preserves unique depth observations', async () => {
  const r = await inverseDepthSessionCase(api, 'cpu', CoreTracker);
  assert.equal(r.registered, 4);
  assert.equal(r.rawObservations, 100);
  console.log(JSON.stringify(r));
});

test('sliding depth history survives camera-window eviction and finish', async () => {
  const r = await inverseDepthSessionCase(api, 'cpu', CoreTracker, 10);
  assert.equal(r.registered, 10);
  assert.equal(r.rawObservations, 250);
  console.log(JSON.stringify(r));
});
