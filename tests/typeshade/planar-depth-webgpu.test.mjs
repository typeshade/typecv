import test from 'node:test';
import assert from 'node:assert/strict';
import * as api from '../../dist/internal/gpu.js';
import { planarDepthCase } from './planar-depth-cases.mjs';
import { inBrowser } from './browser.mjs';

test('planar depth completion on WebGPU matches the CPU tier and rejects occlusions', async () => {
  const { result: r, errors } = await inBrowser(async () => {
    const api = await import('/dist/internal/gpu.js');
    const { planarDepthCase } = await import('/tests/typeshade/planar-depth-cases.mjs');
    return planarDepthCase(api, 'webgpu');
  });
  assert.deepEqual(errors, []);
  assert.equal(r.liveBytes, 0);
  assert.equal(r.budgetRejected, true);
  assert.equal(r.cancelled, true);
  assert.equal(r.selectedFilled, 0);
  assert.equal(r.boundedFilled, 36);
  const cpu = await planarDepthCase(api, 'cpu');
  for (const [i, row] of r.results.entries()) {
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
    assert.equal(row.depth.length, cpu.results[i].depth.length);
    for (let k = 0; k < row.depth.length; k++) {
      assert.equal(row.depth[k] > 0, cpu.results[i].depth[k] > 0);
      assert.ok(Math.abs(row.depth[k] - cpu.results[i].depth[k]) < 0.0001);
    }
  }
});

test('reconstruction completes verified depth holes before TSDF fusion on WebGPU', async () => {
  const { result: r, errors } = await inBrowser(async () => {
    const api = await import('/dist/internal/gpu.js');
    const { planarMvsCase } = await import('/tests/typeshade/planar-depth-cases.mjs');
    return planarMvsCase(api, 'webgpu');
  });
  assert.deepEqual(errors, []);
  assert.equal(r.liveBytes, 0);
  assert.equal(r.filled, 36);
  assert.equal(r.remainingHoles, 0);
  assert.ok(r.points > 100);
  assert.ok(r.maxSurfaceError < 0.01);
});
