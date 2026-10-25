import test from 'node:test';
import assert from 'node:assert/strict';
import { inBrowser } from './browser.mjs';

test('owned planar maps match frozen immutable evidence and preserve caller data on WebGPU', async () => {
  const { result, errors } = await inBrowser(async () => {
    const api = await import('/dist/internal/gpu.js');
    const { ownedPlanarCase } = await import('/tests/typeshade/planar-depth-owned-cases.mjs');
    return ownedPlanarCase(api, 'webgpu');
  });
  assert.deepEqual(errors, []);
  assert.equal(result.rows.length, 18);
  assert.equal(result.maskedSlots.length, 3);
  assert.ok(result.bootstrapExtra > 0);
  assert.equal(result.callerMapsPreserved, true);
  assert.equal(result.mvs.filled, 36);
  assert.equal(result.mvs.remainingHoles, 0);
  assert.ok(result.mvs.points > 100);
  assert.ok(result.mvs.maxSurfaceError < 0.01);
  assert.equal(result.mvs.liveBytes, 0);
  assert.equal(result.publicCancellation.aborted, true);
  assert.equal(result.ownedCancellation.aborted, true);
  assert.ok(result.publicCancellation.plans >= 6);
  assert.ok(result.ownedCancellation.plans >= 6);
});
