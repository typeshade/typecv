import test from 'node:test';
import { ShadeDevice } from '../../dist/gpu.js';
import * as api from '../../dist/gpu.js';
import { runStructureQualityCases } from './structure-quality-cases.mjs';
import { assertStructureQuality, structureQualityFixtures } from './structure-quality-fixture.mjs';

test('structure quality on TypeShade CPU defers depth, preserves masks and refines the seed', async () => {
  const device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    assertStructureQuality(await runStructureQualityCases(api, device, structureQualityFixtures()));
  } finally {
    await device.dispose();
  }
});
