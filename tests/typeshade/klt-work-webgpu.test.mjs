import test from 'node:test';
import assert from 'node:assert/strict';
import * as cpu from '../../dist/internal/gpu.js';
import { workFixture, workReferences, assertWork } from './klt-work-fixture.mjs';
import { inBrowser } from './browser.mjs';

test('WebGPU KLT work elision preserves chunk tails, initial affine guesses and dead backward seeds', async () => {
  const input = workFixture(),
    references = workReferences(cpu, input);
  const { result, errors } = await inBrowser(
    async (input) => {
      const api = await import('/dist/internal/gpu.js'),
        { runWork } = await import('/tests/typeshade/klt-work-cases.mjs'),
        device = await api.ShadeDevice.open({ backend: 'webgpu', poolBytes: 0 });
      try {
        return await runWork(api, device, input);
      } finally {
        await device.dispose();
      }
    },
    input,
    { timeoutMs: 120000 },
  );
  assert.deepEqual(errors, []);
  assertWork(result, input, references, 1e-3);
});
