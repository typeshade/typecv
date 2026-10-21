import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { verifyImagePair } from '../src/geometry/image-pair.ts';
import type { ImageFeatures, PairMatches } from '../src/index.ts';
import { pairBase, pairRows, verificationFromBase } from '../src/reconstruction/focal-pairs.ts';
import { pairSeed } from '../src/reconstruction/identity.ts';
import { cameraAtRatio } from '../src/reconstruction/focal-search.ts';

// docs/auto-focal.md, section 8: a pair verified from its base at a focal gives the verification
// that a full search gives at that focal. The pyramid sample (tests/fixtures/pyramid-sift-cpu.json)
// has 8 frames of 640 × 360.
const pyramid = JSON.parse(
  readFileSync(new URL('./fixtures/pyramid-sift-cpu.json', import.meta.url), 'utf8'),
) as {
  width: number;
  height: number;
  frames: { id: number; keypoints: number[] }[];
  pairs: { first: number; second: number; matches: number[] }[];
};
const images: ImageFeatures[] = pyramid.frames.map((f) => ({
  id: f.id,
  camera: { fx: 0.8 * pyramid.width, fy: 0.8 * pyramid.width, cx: 319.5, cy: 179.5 },
  keypoints: Float64Array.from(f.keypoints),
}));
const controls = { seed: 17, iterations: 500, maxError: 3, pointAngle: 0.5 };

test('a pair verified from its base matches the full verification at every focal', () => {
  const pairs: PairMatches[] = pyramid.pairs
    .filter((p) => p.first < 4 && p.second < 4)
    .map((p) => ({ first: p.first, second: p.second, matches: Uint32Array.from(p.matches) }));
  assert.ok(pairs.length >= 5);
  for (const pair of pairs) {
    const a = images[pair.first]!,
      b = images[pair.second]!;
    const base = pairBase(
      a,
      b,
      pair,
      cameraAtRatio(a.camera, 1),
      cameraAtRatio(b.camera, 1),
      controls,
    );
    for (const ratio of [0.3, 0.55, 0.8, 1.3, 2.4]) {
      const first = cameraAtRatio(a.camera, ratio),
        second = cameraAtRatio(b.camera, ratio);
      const full = verifyImagePair(pairRows(a, b, pair), first, second, {
        iterations: controls.iterations,
        seed: pairSeed(controls.seed, pair.first, pair.second),
        threshold: controls.maxError / first.fx,
        refinementIterations: 0,
        minParallaxDeg: Math.min(controls.pointAngle, 0.1),
        maxReprojectionErrorPx: controls.maxError,
        homographyThresholdPx: controls.maxError,
        homographyIterations: Math.min(controls.iterations, 500),
      });
      const derived = verificationFromBase(base, first, second, controls);
      const label = `pair ${pair.first}-${pair.second} at ${ratio}`;
      assert.equal(derived.invertibleCount, full.invertibleCount, label);
      assert.equal(derived.homographyInlierCount, full.homographyInlierCount, label);
      assert.deepEqual(derived.inlierRows, full.inlierRows, label);
      assert.equal(derived.reason, full.reason, label);
      assert.equal(derived.geometry?.inlierCount, full.geometry?.inlierCount, label);
      assert.equal(derived.geometry?.positiveDepthCount, full.geometry?.positiveDepthCount, label);
      if (full.geometry)
        for (let i = 0; i < 3; i++)
          assert.ok(Math.abs(derived.geometry!.t[i]! - full.geometry.t[i]!) < 1e-9, label);
    }
  }
});

test('a pair with too few rows has no geometry at any focal', () => {
  const pair: PairMatches = { first: 0, second: 1, matches: Uint32Array.from([0, 0, 1, 1, 2, 2]) };
  const a = images[0]!,
    b = images[1]!;
  const base = pairBase(a, b, pair, a.camera, b.camera, controls);
  const result = verificationFromBase(base, a.camera, b.camera, controls);
  assert.equal(result.geometry, undefined);
  assert.equal(result.inlierRows.length, 0);
});
