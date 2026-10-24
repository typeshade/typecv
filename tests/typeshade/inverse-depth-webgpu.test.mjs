import test from 'node:test';
import assert from 'node:assert/strict';
import { inBrowser } from './browser.mjs';
test('WebGPU inverse-depth factors preserve pure-direction derivatives and joint uncertainty', async () => {
  const { result, errors } = await inBrowser(
    async () => {
      const api = await import('/dist/internal/gpu.js'),
        { inverseDepthGpuCases } = await import('/tests/typeshade/inverse-depth-cases.mjs');
      return inverseDepthGpuCases(api, 'webgpu');
    },
    null,
    { timeoutMs: 600000 },
  );
  assert.deepEqual(errors, []);
  assert.equal(result.backend, 'webgpu');
  assert.equal(result.liveBytes, 0);
  console.log(JSON.stringify(result));
});
test('WebGPU keyframe session updates persistent depth through finish without duplicate evidence', async () => {
  const { result, errors } = await inBrowser(
    async () => {
      const api = await import('/dist/internal/gpu.js'),
        { inverseDepthSessionCase } = await import('/tests/typeshade/inverse-depth-cases.mjs');
      return inverseDepthSessionCase(api, 'webgpu', api.InverseDepthTracker, 10);
    },
    null,
    { timeoutMs: 600000 },
  );
  assert.deepEqual(errors, []);
  assert.equal(result.rawObservations, 250);
  assert.equal(result.liveBytes, 0);
  console.log(JSON.stringify(result));
});
test('WebGPU sparse joint uncertainty keeps 24 camera views and bounded measurement batches', async () => {
  const { result, errors } = await inBrowser(
    async () => {
      const api = await import('/dist/internal/gpu.js'),
        { inverseDepthLargeGpuCase } = await import('/tests/typeshade/inverse-depth-cases.mjs');
      return inverseDepthLargeGpuCase(api, 'webgpu');
    },
    null,
    { timeoutMs: 600000 },
  );
  assert.deepEqual(errors, []);
  assert.equal(result.views, 24);
  assert.equal(result.liveBytes, 0);
  console.log(JSON.stringify(result));
});
