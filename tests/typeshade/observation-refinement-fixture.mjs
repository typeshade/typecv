import assert from 'node:assert/strict';
import { refinementFixture } from '../fixtures/refinement.ts';
export function refinementInput() {
  const f = refinementFixture();
  const pack = (images) => images.map((v) => ({ ...v, keypoints: Array.from(v.keypoints) }));
  const pairs = (input) => input.map((p) => ({ ...p, matches: Array.from(p.matches) }));
  return {
    images: pack(f.inputs),
    limited: pack(f.limited.inputs),
    pairs: pairs(f.pairs),
    oldPairs: pairs(f.limited.pairs),
    controls: f.controls,
  };
}
export function assertRefinement(result) {
  assert.deepEqual(result.before, { views: 4, points: 30, observations: 110 });
  assert.deepEqual(result.after, { views: 4, points: 60, observations: 240 });
  assert.deepEqual(result.seed, result.originalSeed);
  assert.ok(result.unchanged);
  assert.equal(result.liveBytes, 0);
  assert.ok(result.error < 0.01);
}
