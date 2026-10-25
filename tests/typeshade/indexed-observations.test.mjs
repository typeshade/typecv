import test from 'node:test';
import assert from 'node:assert/strict';
import { pathToFileURL } from 'node:url';
import { buildComputeProbe } from './compute-probe-build.mjs';

test('indexed scene observations preserve expanded projection/depth results and async ownership on CPU', async () => {
  const file = await buildComputeProbe('indexed-observations-probe');
  const { checkIndexedObservations } = await import(pathToFileURL(file).href);
  const result = await checkIndexedObservations();
  assert.equal(result.cases.length, 10);
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
