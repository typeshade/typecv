import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { verifyImagePair, verifyImagePairWithHomography } from '../src/geometry/image-pair.ts';
import type { Camera } from '../src/geometry/camera.ts';
import { projectPoint, type ImageFeatures, type PairMatches } from '../src/index.ts';
import {
  pairBase,
  pairRows,
  verificationFromBase,
  type PairBase,
  type PairBaseControls,
} from '../src/reconstruction/focal-pairs.ts';
import { pairSeed } from '../src/reconstruction/identity.ts';
import { cameraAtRatio } from '../src/reconstruction/focal-search.ts';
import { projectEssential } from '../src/sfm/two-view.ts';
import { scene } from './scene.ts';

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

// The previous focal path fitted a one-trial H, then discarded its support. Keep that full
// verification as an exact oracle for every essential/pose output and diagnostic below.
function fullVerificationFromBase(
  base: PairBase,
  first: Camera,
  second: Camera,
  controls: PairBaseControls,
) {
  const [ref1, ref2] = base.reference;
  const left = [first.fx / ref1.fx, first.fy / ref1.fy, 1],
    right = [second.fx / ref2.fx, second.fy / ref2.fy, 1];
  const result = verifyImagePair(base.raw, first, second, {
    iterations: controls.iterations,
    seed: pairSeed(controls.seed, base.pair.first, base.pair.second),
    threshold: controls.maxError / Math.min(first.fx, first.fy, second.fx, second.fy),
    refinementIterations: 0,
    minParallaxDeg: Math.min(controls.pointAngle, 0.1),
    maxReprojectionErrorPx: controls.maxError,
    homographyThresholdPx: controls.maxError,
    homographyIterations: 1,
    hypotheses: (k) => {
      const F = base.trial(k);
      if (!F) return undefined;
      const scaled = new Float64Array(9);
      for (let i = 0; i < 3; i++)
        for (let j = 0; j < 3; j++) scaled[i * 3 + j] = F[i * 3 + j]! * right[i]! * left[j]!;
      return projectEssential(scaled);
    },
  });
  return { ...result, homographyInlierCount: base.homographyInliers };
}

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
      assert.deepEqual(derived, fullVerificationFromBase(base, first, second, controls), label);
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

test('cached pair verification preserves full rejection results and homography evidence', () => {
  const sample = scene({ count: 48, views: 2, noise: 0 });
  const camera = sample.inputs[0]!.camera;
  const planar = Float64Array.from(
    sample.points.flatMap((point) => {
      const X = Float64Array.of(point[0]!, point[1]!, 6);
      return [
        ...projectPoint(camera, sample.registered[0]!.pose, X)!,
        ...projectPoint(camera, sample.registered[1]!.pose, X)!,
      ];
    }),
  );
  let state = 7;
  const randomRows = Float64Array.from({ length: 4 * 40 }, () => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return 20 + (state / 2 ** 32) * 600;
  });
  const degenerate = Float64Array.from({ length: 4 * 12 }, () => 100);
  const cases = [
    { name: 'small', raw: planar.slice(0, 12) },
    { name: 'degenerate', raw: degenerate },
    { name: 'no essential model', raw: randomRows },
    { name: 'homography dominant', raw: planar },
  ];
  const limited = { ...controls, iterations: 64 };
  for (const { name, raw } of cases) {
    const first: ImageFeatures = {
      id: 10,
      camera,
      keypoints: Float64Array.from(
        Array.from({ length: raw.length / 4 }, (_, i) => [raw[i * 4]!, raw[i * 4 + 1]!]).flat(),
      ),
    };
    const second: ImageFeatures = {
      id: 20,
      camera,
      keypoints: Float64Array.from(
        Array.from({ length: raw.length / 4 }, (_, i) => [raw[i * 4 + 2]!, raw[i * 4 + 3]!]).flat(),
      ),
    };
    const pair: PairMatches = {
      first: 10,
      second: 20,
      matches: Uint32Array.from(Array.from({ length: raw.length / 4 }, (_, i) => [i, i]).flat()),
    };
    const base = pairBase(first, second, pair, camera, camera, limited);
    const inputBefore = raw.slice();
    const result = verificationFromBase(base, camera, camera, limited);
    assert.deepEqual(result, fullVerificationFromBase(base, camera, camera, limited), name);
    assert.equal(result.geometry, undefined, name);
    assert.ok(result.reason, name);
    assert.deepEqual(raw, inputBefore, name);
    if (name.startsWith('homography')) {
      assert.equal(result.homographyInlierCount, raw.length / 4, name);
      assert.match(result.reason!, /valid two-view|Cheirality/, name);
    }
  }
});

test('cached pair verification preserves a failed pose after all cheirality retries', () => {
  const fixture = pyramid.pairs.find((pair) => pair.first === 0 && pair.second === 2)!;
  const pair: PairMatches = { ...fixture, matches: Uint32Array.from(fixture.matches) };
  const a = images[0]!,
    b = images[2]!;
  const limited = { ...controls, iterations: 64 };
  const base = pairBase(
    a,
    b,
    pair,
    cameraAtRatio(a.camera, 1),
    cameraAtRatio(b.camera, 1),
    limited,
  );
  const first = cameraAtRatio(a.camera, 0.1),
    second = cameraAtRatio(b.camera, 0.1);
  const result = verificationFromBase(base, first, second, limited);
  assert.match(result.reason!, /^Cheirality failed/);
  assert.equal(result.geometry, undefined);
  assert.deepEqual(result, fullVerificationFromBase(base, first, second, limited));
});

test('cached homography support is internal and validates its raw-row count', () => {
  const a = images[0]!,
    b = images[1]!;
  const pair: PairMatches = { first: 0, second: 1, matches: Uint32Array.from([0, 0, 1, 1, 2, 2]) };
  const raw = pairRows(a, b, pair);
  for (const count of [-1, 0.5, 4, NaN, Infinity])
    assert.throws(() => verifyImagePairWithHomography(raw, a.camera, b.camera, count), RangeError);
  for (const options of [
    { homographyIterations: 0 },
    { homographyThresholdPx: 0 },
    { iterations: 0 },
  ])
    assert.throws(
      () => verifyImagePairWithHomography(raw, a.camera, b.camera, 0, options),
      RangeError,
    );
  // The previous derived path overwrote H support even when ideal rows were insufficient.
  assert.equal(verifyImagePairWithHomography(raw, a.camera, b.camera, 3).homographyInlierCount, 3);
  assert.deepEqual(
    verifyImagePairWithHomography(raw, a.camera, b.camera, 0),
    verifyImagePair(raw, a.camera, b.camera),
  );
  const entry = readFileSync(new URL('../src/index.ts', import.meta.url), 'utf8');
  assert.doesNotMatch(entry, /verifyImagePairWithHomography/);
});
