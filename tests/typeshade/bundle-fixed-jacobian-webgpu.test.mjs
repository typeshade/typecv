import test from 'node:test';
import assert from 'node:assert/strict';
import { inBrowser } from './browser.mjs';
import { assertFixedJacobianCases } from './bundle-fixed-jacobian-cases.mjs';

test('fixed camera observations retain point and shared-focal constraints on WebGPU', async () => {
  const { result, errors } = await inBrowser(async () => {
    const api = await import('/dist/gpu.js');
    const { checkFixedJacobianCases } =
      await import('/tests/typeshade/bundle-fixed-jacobian-cases.mjs');
    return checkFixedJacobianCases(api, 'webgpu');
  });
  assert.deepEqual(errors, []);
  assertFixedJacobianCases(assert, result);
});
