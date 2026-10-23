import test from 'node:test';
import assert from 'node:assert/strict';
import { inBrowser } from './browser.mjs';
import { assertTracking, rotationFixtures } from './rotation-tracking-fixture.mjs';

test('WebGPU session agrees with the f64 rotation oracle and never uses rotation as a depth seed', async () => {
  const { result, errors } = await inBrowser(
    async (fixtures) => {
      const api = await import('/dist/gpu.js'),
        { runRotationTrackingCases } = await import('/tests/typeshade/rotation-tracking-cases.mjs');
      const device = await api.ShadeDevice.open({ poolBytes: 0 });
      try {
        return {
          ...(await runRotationTrackingCases(api, device, fixtures)),
          backend: device.backend,
        };
      } finally {
        await device.dispose();
      }
    },
    rotationFixtures(),
    { timeoutMs: 180000 },
  );
  assert.deepEqual(errors, []);
  assert.equal(result.backend, 'webgpu');
  assertTracking(result);
});
