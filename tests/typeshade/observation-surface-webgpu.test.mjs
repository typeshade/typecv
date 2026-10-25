import test from 'node:test';
import assert from 'node:assert/strict';
import { inBrowser } from './browser.mjs';
import {
  surfaceFixture,
  consensusFixture,
  assertSurface,
  assertConsensus,
} from './observation-surface-fixture.mjs';

test('WebGPU measures local surfaces and corroborates only original image evidence', async () => {
  const surface = surfaceFixture(),
    consensus = consensusFixture();
  const { result, errors } = await inBrowser(
    async (input) => {
      const api = await import('/dist/gpu.js'),
        { runSurfaceCases, runConsensusCases } =
          await import('/tests/typeshade/observation-surface-cases.mjs'),
        adapter = await navigator.gpu.requestAdapter(),
        gpu = await adapter.requestDevice(),
        device = await api.ShadeDevice.open({ gpuDevice: gpu, poolBytes: 0 });
      try {
        return {
          surface: await runSurfaceCases(api, device, input.surface),
          consensus: await runConsensusCases(api, device, input.consensus),
          adapter: {
            description: adapter.info.description,
            fallback: adapter.info.isFallbackAdapter,
          },
        };
      } finally {
        await device.dispose();
      }
    },
    { surface: surface.surface, consensus },
    { timeoutMs: 120000 },
  );
  assert.deepEqual(errors, []);
  console.log(
    JSON.stringify({ ...assertSurface(result.surface, surface), adapter: result.adapter }),
  );
  assertConsensus(result.consensus, consensus);
});
