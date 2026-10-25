import test from 'node:test';
import assert from 'node:assert/strict';
import { inBrowser } from './browser.mjs';

test('WebGPU optical-path exclusions survive grouped depth and planar completion', async () => {
  const { result, errors } = await inBrowser(
    async () => {
      const api = await import('/dist/internal/gpu.js');
      const { observationPathCases } = await import('/tests/typeshade/observation-path-cases.mjs');
      return observationPathCases(api, 'webgpu');
    },
    null,
    { timeoutMs: 600000 },
  );
  assert.deepEqual(errors, []);
  assert.equal(result.backend, 'webgpu');
  assert.equal(result.liveBytes, 0);
  assert.equal(result.filled, 18);
  console.log(JSON.stringify(result));
});
