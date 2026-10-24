import test from 'node:test';
import assert from 'node:assert/strict';
import { pathToFileURL } from 'node:url';
import { buildComputeProbe } from './compute-probe-build.mjs';

test('in-place pair QR preserves every scalar factor bit and releases its resident storage', async () => {
  const file = await buildComputeProbe('pair-row-probe');
  const { checkPairRows, checkPairRowWindows, pairRefitRowChunk } = await import(
    pathToFileURL(file).href
  );
  const gpu = {
    backend: 'webgpu',
    fp64Flavor: 'float',
    boundedSubmissions: false,
    timeBudgetMs: 250,
  };
  assert.equal(pairRefitRowChunk(gpu, 1024), 512);
  assert.equal(pairRefitRowChunk(gpu, 1023), 128);
  for (const change of [
    { backend: 'cpu' },
    { fp64Flavor: 'integer' },
    { boundedSubmissions: true },
    { timeBudgetMs: 249 },
    { timeBudgetMs: 1 },
  ])
    assert.equal(pairRefitRowChunk({ ...gpu, ...change }, 2049), 128);
  const result = await checkPairRows();
  assert.equal(result.rows.length, 14);
  for (const row of result.rows) {
    assert.equal(row.exact, true, JSON.stringify(row.input));
    assert.deepEqual(row.second.bits, row.first.bits);
  }
  assert.equal(result.statistics.liveBuffers, 0);
  assert.equal(result.statistics.liveBytes, 0);
  assert.equal(result.statistics.reads, 28);
  const windows = await checkPairRowWindows();
  for (const row of windows.rows) assert.deepEqual(row.second.bits, row.first.bits);
  assert.equal(windows.statistics.liveBuffers, 0);
});
