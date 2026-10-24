import test from 'node:test';
import assert from 'node:assert/strict';
import { inBrowser } from './browser.mjs';
import { compactInput, assertCompact } from './klt-compact-fixture.mjs';

test('compact WebGPU KLT preserves affine chunk tails, NCC, optional back/pyramid outputs and exact counts', async () => {
  const input = compactInput(),
    { result, errors } = await inBrowser(
      async (input) => {
        const api = await import('/dist/internal/gpu.js'),
          { runCompactCases } = await import('/tests/typeshade/klt-compact-cases.mjs'),
          device = await api.ShadeDevice.open({ backend: 'webgpu', poolBytes: 0 });
        try {
          return { ...(await runCompactCases(api, device, input)), adapter: device.info.webgpu };
        } finally {
          await device.dispose();
        }
      },
      input,
      { timeoutMs: 120000 },
    );
  assert.deepEqual(errors, []);
  assertCompact(result, input);
  console.log(
    JSON.stringify({
      adapter: result.adapter,
      rows: result.rows.map(({ name, legacy, compact }) => ({
        name,
        legacy: legacy.counts,
        compact: compact.counts,
      })),
    }),
  );
});
