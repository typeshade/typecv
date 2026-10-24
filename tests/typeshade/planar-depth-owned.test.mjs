import test from 'node:test';
import assert from 'node:assert/strict';
import * as api from '../../dist/internal/gpu.js';
import { ownedPlanarCase } from './planar-depth-owned-cases.mjs';

test('owned planar maps match frozen immutable evidence and preserve caller data on the CPU tier', async () => {
  const result = await ownedPlanarCase(api, 'cpu');
  assert.equal(result.rows.length, 15);
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
  console.log(
    JSON.stringify({
      cases: result.rows.length,
      bootstrapExtra: result.bootstrapExtra,
      hostMaps: result.rows.map(({ name, hostMaps }) => ({ name, ...hostMaps })),
    }),
  );
});
