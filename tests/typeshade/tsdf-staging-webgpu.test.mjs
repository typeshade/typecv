import test from 'node:test';
import assert from 'node:assert/strict';
import { inBrowser } from './browser.mjs';
import { assertStagingCase } from './tsdf-staging-assertions.mjs';

test('TSDF bounded staging and reused discovery keys preserve all output channels on WebGPU', async () => {
  const { result, errors } = await inBrowser(async () => {
    const api = await import('/dist/internal/gpu.js');
    const { stagingCase } = await import('/tests/typeshade/tsdf-staging-cases.mjs');
    return stagingCase(api, 'webgpu');
  });
  assert.deepEqual(errors, []);
  assertStagingCase(result);
});
