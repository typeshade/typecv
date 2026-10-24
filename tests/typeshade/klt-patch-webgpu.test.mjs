import test from 'node:test';
import assert from 'node:assert/strict';
import * as cpu from '../../dist/internal/gpu.js';
import { inBrowser } from './browser.mjs';
import { patchFixture, assertPatches } from './klt-patch-fixture.mjs';
test('affine KLT on WebGPU corrects patch bias and preserves chunk-tail measurements', async () => {
  const input = patchFixture();
  const { result, errors } = await inBrowser(
    async (input) => {
      const api = await import('/dist/internal/gpu.js'),
        { runPatches } = await import('/tests/typeshade/klt-patch-run.mjs');
      const device = await api.ShadeDevice.open({ backend: 'webgpu', poolBytes: 0 });
      try {
        return await runPatches(api, device, input);
      } finally {
        await device.dispose();
      }
    },
    input,
    { timeoutMs: 120000 },
  );
  assert.deepEqual(errors, []);
  assertPatches(result, input);
  for (let j = 0; j < input.cases.length; j++) {
    const c = input.cases[j],
      opts = cpu.kltOptionsFor(input.width, {
        levels: 2,
        halfWindow: 5,
        epsilonPx: 0.005,
        maxIterations: 30,
      }),
      ref = cpu.trackKlt(
        cpu.buildKltPyramid(input.first, input.width, input.height, 2),
        cpu.buildKltPyramid(c.second, input.width, input.height, 2),
        input.points,
        opts,
        c.guesses,
        c.affine,
      );
    for (let i = 0; i < ref.records.length; i++)
      if (i % 4 < 2) assert.ok(Math.abs(result.cases[j].records[i] - ref.records[i]) < 1e-3);
  }
});
