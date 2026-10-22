import test from 'node:test';
import assert from 'node:assert/strict';
import * as api from '../../dist/internal/gpu.js';
import { planarDepthCase, planarMvsCase } from './planar-depth-cases.mjs';

test('planar depth completion on the CPU tier fills verified planes and preserves boundaries', async () => {
  const r = await planarDepthCase(api, 'cpu');
  assert.equal(r.liveBytes, 0);
  assert.equal(r.budgetRejected, true);
  assert.equal(r.cancelled, true);
  assert.equal(r.selectedFilled, 0);
  assert.equal(r.boundedFilled, 36);
  for (const row of r.results) {
    assert.equal(row.changedInput, 0, row.mode);
    assert.equal(row.changedExisting, 0, row.mode);
    assert.equal(
      row.filled,
      row.mode === 'plane' || row.mode === 'weak-support' || row.mode === 'rotated'
        ? 36
        : row.mode === 'limited-overlap'
          ? 24
          : 0,
      row.mode,
    );
    assert.ok(row.worstRelativeError < 0.0001, row.mode);
    assert.ok(row.worstNormalError < 0.0001, row.mode);
  }
});

test('reconstruction completes verified depth holes before TSDF fusion on the CPU tier', async () => {
  const r = await planarMvsCase(api, 'cpu');
  assert.equal(r.liveBytes, 0);
  assert.equal(r.filled, 36);
  assert.equal(r.remainingHoles, 0);
  assert.ok(r.points > 100);
  assert.ok(r.maxSurfaceError < 0.01);
});
