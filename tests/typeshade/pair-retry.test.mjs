import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { ShadeDevice, verifyImagePairGpu } from '../../dist/gpu.js';

const fixture = JSON.parse(
  readFileSync(new URL('../fixtures/pyramid-pair-4-5.json', import.meta.url), 'utf8'),
);

test('GPU pair verification retries a cheirality failure on a near-planar pair (CPU tier)', async () => {
  const device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    const pixels = Float64Array.from(fixture.pixels);
    for (const seed of [...fixture.seedsFailingBeforeRetry.gpu, fixture.options.seed]) {
      const result = await verifyImagePairGpu(device, pixels, fixture.first, fixture.second, {
        ...fixture.options,
        seed,
      });
      assert.equal(result.reason, undefined, `seed ${seed}: ${result.reason}`);
      assert.ok(result.geometry, `seed ${seed}`);
      assert.ok(result.inlierRows.length >= 60, `seed ${seed}: ${result.inlierRows.length}`);
      assert.ok(result.homographyInlierCount >= 55);
    }
  } finally {
    await device.dispose();
  }
});
