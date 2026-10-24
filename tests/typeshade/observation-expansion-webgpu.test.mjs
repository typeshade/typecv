import test from 'node:test';
import assert from 'node:assert/strict';
import { inBrowser } from './browser.mjs';
import { expansionFixture, assertExpansion } from './observation-expansion-fixture.mjs';

test('image-based observation expansion on WebGPU preserves evidence and rejects ambiguous/missing pixels', async () => {
  const input = expansionFixture();
  const { result, errors } = await inBrowser(async (input) => {
    const api = await import('/dist/gpu.js');
    const { runExpansionCases } = await import('/tests/typeshade/observation-expansion-cases.mjs');
    const adapter = await navigator.gpu.requestAdapter();
    const gpu = await adapter.requestDevice();
    const device = await api.ShadeDevice.open({ gpuDevice: gpu, poolBytes: 0 });
    try {
      const result = await runExpansionCases(api, device, input);
      return {
        ...result,
        adapter: {
          vendor: adapter.info.vendor,
          architecture: adapter.info.architecture,
          description: adapter.info.description,
          fallback: adapter.info.isFallbackAdapter,
        },
      };
    } finally {
      await device.dispose();
    }
  }, input);
  assert.deepEqual(errors, []);
  assertExpansion(result, input);
  console.log(JSON.stringify({ ...result.expanded.diagnostics, adapter: result.adapter }));
});
