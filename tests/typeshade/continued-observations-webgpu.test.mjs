import test from 'node:test';
import assert from 'node:assert/strict';
import { buildComputeProbe } from './compute-probe-build.mjs';
import { inBrowser } from './browser.mjs';
import { assertContinuedObservations } from './continued-observations-assertions.mjs';

test('WebGPU continuation keeps appended masks and full-context uncertainty and input validation', async () => {
  const file = await buildComputeProbe('continued-observations-probe');
  const { result, errors } = await inBrowser(
    async () => {
      const { checkContinuedObservations } = await import('/probe/continued-observations.js');
      return checkContinuedObservations('webgpu');
    },
    null,
    { timeoutMs: 30000, serve: { '/probe/continued-observations.js': file } },
  );
  assert.deepEqual(errors, []);
  assertContinuedObservations(result);
});
