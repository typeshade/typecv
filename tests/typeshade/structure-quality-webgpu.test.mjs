import test from 'node:test';
import assert from 'node:assert/strict';
import { inBrowser } from './browser.mjs';
import { assertStructureQuality, structureQualityFixtures } from './structure-quality-fixture.mjs';

test('structure quality uses WebGPU depth filtering and seed refinement without geometry leakage', async () => {
  const { result, errors } = await inBrowser(
    async (fixtures) => {
      const api = await import('/dist/gpu.js'),
        { runStructureQualityCases } = await import('/tests/typeshade/structure-quality-cases.mjs');
      const device = await api.ShadeDevice.open({ poolBytes: 0 });
      try {
        return await runStructureQualityCases(api, device, fixtures);
      } finally {
        await device.dispose();
      }
    },
    structureQualityFixtures(),
    { timeoutMs: 240000 },
  );
  assert.deepEqual(errors, []);
  assertStructureQuality(result);
});
