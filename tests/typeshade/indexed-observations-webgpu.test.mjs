import test from 'node:test';
import assert from 'node:assert/strict';
import { inBrowser } from './browser.mjs';
import { buildComputeProbe } from './compute-probe-build.mjs';

test('indexed scene observations match expanded kernels and clean partial allocations on WebGPU', async () => {
  const file = await buildComputeProbe('indexed-observations-probe');
  const { result, errors } = await inBrowser(
    async () => {
      const { checkIndexedObservations } = await import('/probe/indexed-observations.js');
      return checkIndexedObservations('webgpu');
    },
    null,
    { timeoutMs: 60000, serve: { '/probe/indexed-observations.js': file } },
  );
  assert.deepEqual(errors, []);
  for (const row of result.cases) {
    assert.deepEqual(row.indexed, row.old, row.name);
    assert.equal(row.inputUnchanged, true, row.name);
  }
  assert.deepEqual(result.snapshotted, result.expected);
  assert.equal(result.failed, true);
  assert.equal(result.afterFailure, 0);
  assert.equal(result.liveBytes, 0);
  assert.deepEqual(
    result.tight,
    result.cases.find((row) => row.name === 'radial-tail-129').indexed,
  );
});
