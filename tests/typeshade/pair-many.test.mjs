import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { ShadeDevice, verifyImagePairGpu, verifyImagePairsGpu } from '../../dist/internal/gpu.js';
import { radialPairFixture, compactedPairFixture } from './initialization-cases.mjs';

// Verifies: docs/multi-video.md Amendment 8 (pairs share plans and reads, and each pair gets the
// result of a single verification). The CPU tier runs the same kernels in Node.

const retry = JSON.parse(
  readFileSync(new URL('../fixtures/pyramid-pair-4-5.json', import.meta.url), 'utf8'),
);

const strip = ({ execution: _execution, ...rest }) => rest;

test('verifying pairs together gives the result of verifying each alone', async () => {
  const one = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  const many = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    const options = { iterations: 256, threshold: 0.001, homographyIterations: 128 };
    const requests = [];
    for (const [n, seed] of [
      [321, 71],
      [150, 72],
      [260, 73],
    ]) {
      const f = radialPairFixture({ n });
      requests.push({
        matches: f.pixels,
        first: f.first,
        second: f.second,
        options: { ...options, seed },
      });
    }
    const c = compactedPairFixture();
    requests.push({
      matches: c.pixels,
      first: c.first,
      second: c.second,
      options: { ...options, seed: 74 },
    });
    // A near-planar pair whose first seeds fail the cheirality test and retry.
    const pixels = Float64Array.from(retry.pixels);
    for (const seed of retry.seedsFailingBeforeRetry.gpu)
      requests.push({
        matches: pixels,
        first: retry.first,
        second: retry.second,
        options: { ...retry.options, seed },
      });
    requests.push({
      matches: new Float64Array(0),
      first: c.first,
      second: c.second,
      options: { ...options, seed: 75 },
    });
    const expected = [];
    for (const r of requests)
      expected.push(await verifyImagePairGpu(one, r.matches, r.first, r.second, r.options));
    for (const maxPairs of [1, 3, 16]) {
      const got = await verifyImagePairsGpu(many, requests, maxPairs);
      assert.equal(got.length, expected.length);
      got.forEach((g, i) =>
        assert.deepEqual(strip(g), strip(expected[i]), `pair ${i}, ${maxPairs}`),
      );
    }
    assert.ok(
      expected.some((e) => e.geometry),
      'some pair verifies',
    );
    assert.equal(one.statistics.liveBuffers, 0);
    assert.equal(many.statistics.liveBuffers, 0);
  } finally {
    await one.dispose();
    await many.dispose();
  }
});

test('a pair that fails stops the job and frees every buffer', async () => {
  const device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    const f = radialPairFixture({ n: 120 });
    const good = { matches: f.pixels, first: f.first, second: f.second, options: { seed: 3 } };
    const bad = { ...good, matches: Float64Array.of(1, 2, 3) };
    await assert.rejects(() => verifyImagePairsGpu(device, [good, bad], 4));
    assert.equal(device.statistics.liveBuffers, 0);
    const again = await verifyImagePairsGpu(device, [good], 4);
    assert.equal(again.length, 1);
  } finally {
    await device.dispose();
  }
});

test('a device with bounded submissions verifies the pairs one by one, with the same results', async () => {
  const plain = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  const bounded = await ShadeDevice.open({
    backend: 'cpu',
    poolBytes: 0,
    boundedSubmissions: true,
    maxDispatchesPerSubmission: 8,
  });
  try {
    const requests = [201, 144].map((n, i) => {
      const f = radialPairFixture({ n });
      return {
        matches: f.pixels,
        first: f.first,
        second: f.second,
        options: { iterations: 128, threshold: 0.001, homographyIterations: 64, seed: 90 + i },
      };
    });
    const expected = await verifyImagePairsGpu(plain, requests, 4);
    const got = await verifyImagePairsGpu(bounded, requests, 4);
    got.forEach((g, i) => assert.deepEqual(strip(g), strip(expected[i])));
    assert.equal(bounded.statistics.liveBuffers, 0);
  } finally {
    await plain.dispose();
    await bounded.dispose();
  }
});
