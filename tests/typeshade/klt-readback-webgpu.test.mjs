import test from 'node:test';
import assert from 'node:assert/strict';
import { inBrowser } from './browser.mjs';
import { assertKltReadbacks, kltReadbackInput } from './klt-readback-fixture.mjs';

test('KLT WebGPU result and pyramid reads share a wave with identical outputs and transfer bytes', async () => {
  const { result, errors } = await inBrowser(
    async (input) => {
      const api = await import('/dist/internal/gpu.js'),
        { runKltReadbackCases } = await import('/tests/typeshade/klt-readback-cases.mjs'),
        device = await api.ShadeDevice.open({ backend: 'webgpu', poolBytes: 0 });
      try {
        const result = await runKltReadbackCases(api, device, input);
        return { ...result, adapter: device.info.webgpu };
      } finally {
        await device.dispose();
      }
    },
    kltReadbackInput(),
    { timeoutMs: 120000 },
  );
  assert.deepEqual(errors, []);
  assertKltReadbacks(result);
  console.log(
    JSON.stringify({
      adapter: result.adapter,
      rows: result.rows.map(({ name, serial, concurrent }) => ({
        name,
        serial: serial.counts,
        concurrent: concurrent.counts,
      })),
    }),
  );
});
