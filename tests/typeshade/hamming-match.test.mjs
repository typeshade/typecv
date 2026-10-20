import test from 'node:test';
import assert from 'node:assert/strict';
import { ShadeDevice, matchFeaturesGpu } from '../../dist/gpu.js';
import { descriptorPair, hammingOracle } from './hamming-oracle.mjs';

// Verifies the CPU tier of the Hamming matcher: it runs the untiled kernel, the oracle of the
// tiled kernel that WebGPU devices run (tests/typeshade/hamming-match-webgpu.test.mjs).
test('the Hamming matcher on the CPU tier equals the JavaScript oracle', async () => {
  const device = await ShadeDevice.open({ backend: 'cpu' });
  try {
    let matched = 0;
    for (const [nf, ns, ratio, maxDistance] of [
      [1, 1, 0.8, 80],
      [63, 65, 0.8, 256],
      [130, 200, 0.9, 256],
      [300, 129, 0.8, 100],
    ]) {
      const { first, second } = descriptorPair(nf, ns, 11 + nf),
        got = await matchFeaturesGpu(device, first, second, { ratio, maxDistance }),
        want = hammingOracle(first, second, ratio, maxDistance);
      matched += want.matches.length / 2;
      assert.deepEqual(got.matches, want.matches, `${nf} x ${ns} matches`);
      assert.deepEqual(got.distances, want.distances, `${nf} x ${ns} distances`);
    }
    assert.ok(matched >= 100, `${matched} matches in all`);
  } finally {
    await device.dispose();
  }
});

test('the Hamming matcher has no pair cap', async () => {
  const device = await ShadeDevice.open({ backend: 'cpu' });
  try {
    // 2100 x 2100 is 4.4 million pairs, above the former cap of 4,000,000.
    const { first, second } = descriptorPair(2100, 2100, 1),
      got = await matchFeaturesGpu(device, first, second, { ratio: 0.9, maxDistance: 256 });
    assert.deepEqual(got.matches, hammingOracle(first, second, 0.9, 256).matches);
  } finally {
    await device.dispose();
  }
});
