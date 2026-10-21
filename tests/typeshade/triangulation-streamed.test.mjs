// docs/gpu-budget.md, section 4: the triangulation runs in chunks of tracks that fit the budget.
// The CPU tier gives the same results, bit for bit, in one chunk and in many.
import test from 'node:test';
import assert from 'node:assert/strict';
import { ShadeDevice, triangulateTracksGpu } from '../../dist/gpu.js';
import { triangulationFixture, triangulationDigest } from './geometry-cases.mjs';

test('triangulation in chunks of tracks gives the results of one chunk, bit for bit', async () => {
  const { tracks } = triangulationFixture(100);
  const observations = tracks.reduce((n, t) => n + t.length, 0);
  const whole = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  let expected, wholeReads;
  try {
    const r = await triangulateTracksGpu(whole, tracks);
    expected = await triangulationDigest(r);
    wholeReads = whole.statistics.reads;
    assert.ok(r.status.some((s) => s === 0));
  } finally {
    await whole.dispose();
  }
  // Budgets: a few tracks, 7 observations' worth, and one track at a time.
  for (const workingBytes of [212 * 30 + 48 * 6 + 4, 212 * 7 + 4096, 4096]) {
    const device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0, workingBytes });
    try {
      const r = await triangulateTracksGpu(device, tracks);
      assert.equal(await triangulationDigest(r), expected, `workingBytes ${workingBytes}`);
      // Each chunk reads its results back in three reads.
      assert.ok(device.statistics.reads > wholeReads, 'more than one chunk ran');
      assert.equal(device.statistics.liveBytes, 0);
      assert.equal(r.status.length, tracks.length);
      assert.ok(observations > 100);
    } finally {
      await device.dispose();
    }
  }
});

test('triangulation of a batch larger than the working budget no longer throws', async () => {
  const { tracks } = triangulationFixture(400);
  const need = tracks.reduce((n, t) => n + 212 * t.length + 48, 4);
  const device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0, workingBytes: 8192 });
  try {
    assert.ok(need > 8192 * 10);
    const r = await triangulateTracksGpu(device, tracks);
    assert.equal(r.tracks.length, tracks.length);
    assert.ok(r.status.filter((s) => s === 0).length > 100);
  } finally {
    await device.dispose();
  }
});
