import test from 'node:test';
import assert from 'node:assert/strict';
import { inBrowser } from './browser.mjs';
import { residencyFixture, assertResidency } from './observation-residency-fixture.mjs';

test('WebGPU automatic observation residency preserves evidence, caller buffers and bounded active memory', async () => {
  const input = residencyFixture();
  const { result, errors } = await inBrowser(
    async (input) => {
      const api = await import('/dist/internal/gpu.js'),
        { runResidencyCases } = await import('/tests/typeshade/observation-residency-cases.mjs'),
        adapter = await navigator.gpu.requestAdapter(),
        gpuDevice = await adapter.requestDevice(),
        rows = [];
      try {
        for (const poolBytes of [0, 32 * 1024 ** 2])
          for (const tightBudget of [false, true])
            rows.push({
              poolBytes,
              tightBudget,
              output: await runResidencyCases(api, input, {
                backend: 'webgpu',
                gpuDevice,
                poolBytes,
                tightBudget,
              }),
            });
        return {
          rows,
          adapter: {
            vendor: adapter.info.vendor,
            architecture: adapter.info.architecture,
            fallback: adapter.info.isFallbackAdapter,
          },
        };
      } finally {
        gpuDevice.destroy();
      }
    },
    input,
    { timeoutMs: 180000 },
  );
  assert.deepEqual(errors, []);
  for (const row of result.rows) assertResidency(row.output, input, row);
  console.log(JSON.stringify({ adapter: result.adapter, configurations: result.rows.length }));
});
