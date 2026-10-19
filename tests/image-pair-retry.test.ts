import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { CHEIRALITY_RETRIES, verifyImagePair } from '../src/geometry/image-pair.ts';

const fixture = JSON.parse(
  readFileSync(new URL('./fixtures/pyramid-pair-4-5.json', import.meta.url), 'utf8'),
) as {
  first: { fx: number; fy: number; cx: number; cy: number; k1: number; k2: number };
  second: { fx: number; fy: number; cx: number; cy: number; k1: number; k2: number };
  options: Record<string, number>;
  seedsFailingBeforeRetry: { cpu: number[]; gpu: number[] };
  pixels: number[];
};

test('a near-planar pair that fails cheirality on one seed verifies after a bounded retry', () => {
  const pixels = Float64Array.from(fixture.pixels);
  assert.equal(CHEIRALITY_RETRIES, 2);
  // The seeds that failed before the retry, and the seed the lab's mapper used for this pair.
  for (const seed of [...fixture.seedsFailingBeforeRetry.cpu, fixture.options.seed!]) {
    const result = verifyImagePair(pixels, fixture.first, fixture.second, {
      ...fixture.options,
      seed,
    });
    assert.equal(result.reason, undefined, `seed ${seed}: ${result.reason}`);
    assert.ok(result.geometry, `seed ${seed}`);
    assert.ok(result.inlierRows.length >= 60, `seed ${seed}: ${result.inlierRows.length} inliers`);
    assert.ok(result.homographyInlierCount >= 55, 'the pair is near-planar');
  }
});

test('a pair with no valid geometry still reports its reason after the retries', () => {
  // Random correspondences: no essential matrix explains them.
  let state = 7;
  const next = () => (state = (Math.imul(state, 1664525) + 1013904223) >>> 0) / 2 ** 32;
  const pixels = new Float64Array(4 * 40);
  for (let i = 0; i < pixels.length; i++) pixels[i] = 20 + next() * 600;
  const result = verifyImagePair(pixels, fixture.first, fixture.second, {
    ...fixture.options,
    seed: 3,
  });
  assert.equal(result.geometry, undefined);
  assert.ok(typeof result.reason === 'string' && result.reason.length > 0);
});
