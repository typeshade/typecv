import test from 'node:test';
import assert from 'node:assert/strict';
import * as cpu from '../../dist/internal/gpu.js';
import { inBrowser } from './browser.mjs';
import {
  roundtripFixture,
  roundtripReferences,
  assertRoundtrips,
} from './klt-roundtrip-fixture.mjs';

test('WebGPU opt-in KLT roundtrips preserve outputs and the separate-pass default', async () => {
  const input = roundtripFixture(),
    reference = roundtripReferences(cpu, input);
  const { result, errors } = await inBrowser(
    async (input) => {
      const api = await import('/dist/internal/gpu.js'),
        { runRoundtrips } = await import('/tests/typeshade/klt-roundtrip-cases.mjs'),
        device = await api.ShadeDevice.open({ backend: 'webgpu', poolBytes: 0 });
      try {
        return await runRoundtrips(api, device, input);
      } finally {
        await device.dispose();
      }
    },
    input,
    { timeoutMs: 120000 },
  );
  assert.deepEqual(errors, []);
  assertRoundtrips(result, input, reference, 1e-3);
});
