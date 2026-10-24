import test from 'node:test';
import assert from 'node:assert/strict';
import * as api from '../../dist/internal/gpu.js';
import { inverseDepthGpuCases, inverseDepthSessionCase } from './inverse-depth-cases.mjs';
import { InverseDepthTracker as CoreTracker } from '../../dist/index.js';
test('TypeShade inverse-depth linearization and joint covariance match CPU without live arrays', async () => {
  const r = await inverseDepthGpuCases(api, 'cpu');
  assert.equal(r.liveBytes, 0);
  console.log(JSON.stringify(r));
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
