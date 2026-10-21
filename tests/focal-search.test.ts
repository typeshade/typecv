import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { reconstructIncremental } from '../src/index.ts';
import type { ImageFeatures, PairMatches, SparseReconstruction } from '../src/index.ts';
import {
  bestCandidate,
  cameraAtRatio,
  focalGrid,
  focalSearchControls,
  parabolaVertex,
  scoreCandidates,
  type FocalCandidate,
} from '../src/reconstruction/focal-search.ts';

// docs/auto-focal.md: the automatic shared-focal estimate.
const lab = {
  ransacIterations: 500,
  maxErrorPx: 3,
  minInitialAngleDeg: 1,
  minTriangulationAngleDeg: 0.5,
  bundleIterations: 4,
  finalBundleIterations: 8,
  refineSharedFocal: true,
};

/** Every view at `ratio × width`; the principal point and the distortion stay. */
function at(images: readonly ImageFeatures[], ratio: number): ImageFeatures[] {
  return images.map((image) => ({ ...image, camera: cameraAtRatio(image.camera, ratio) }));
}
function medianRatio(model: SparseReconstruction): number {
  const r = model.views.map((v) => v.camera.fx / (2 * v.camera.cx + 1)).sort((a, b) => a - b);
  return r[Math.floor(r.length / 2)]!;
}

test('the grid, the controls and the parabola', () => {
  const grid = focalGrid(focalSearchControls(true)!);
  assert.equal(grid.length, 17);
  assert.equal(grid[0], 0.3);
  assert.ok(Math.abs(grid.at(-1)! - 0.3 * 1.15 ** 16) < 1e-12 && grid.at(-1)! <= 3);
  assert.equal(focalSearchControls(false), undefined);
  assert.equal(focalSearchControls(undefined), undefined);
  for (const bad of [{ views: 1 }, { step: 1 }, { minRatio: 0 }, { minRatio: 2, maxRatio: 1 }])
    assert.throws(() => focalSearchControls(bad), RangeError);
  // y = (ln r − ln 0.7)² + 0.2 sampled at three grid points: the vertex is 0.7.
  const points: FocalCandidate[] = [0.6, 0.69, 0.7935].map((ratio) => ({
    ratio,
    views: 5,
    rmsePx: 0,
    observations: 100,
    score: Math.log(ratio / 0.7) ** 2 + 0.2,
  }));
  assert.ok(Math.abs(parabolaVertex(points, 1) - 0.7) < 1e-9);
  // A neighbour with fewer views, or none, leaves the grid point.
  assert.equal(parabolaVertex([{ ...points[0]!, views: 4 }, points[1]!, points[2]!], 1), 0.69);
  assert.equal(parabolaVertex(points.slice(1), 0), 0.69);
  // A run that drops observations pays the Huber threshold for each one it dropped. The owner's
  // record 4a8aa2ff, first five views: 0.60 × width kept 3734 observations at 0.263 px, 0.92 kept
  // 5035 at 0.347 px. The RMS error alone prefers 0.60.
  const scored = scoreCandidates(
    [
      { ratio: 0.6, views: 5, rmsePx: 0.263, observations: 3734 },
      { ratio: 0.92, views: 5, rmsePx: 0.347, observations: 5035 },
      { ratio: 1, views: 2, rmsePx: 0.1, observations: 300 },
      { ratio: 2, views: 0, rmsePx: Infinity, observations: 0 },
    ],
    1.5,
  );
  assert.ok(Math.abs(scored[0]!.score - (3734 * 0.263 ** 2 + 1301 * 2.25) / 5035) < 1e-12);
  assert.ok(Math.abs(scored[1]!.score - 0.347 ** 2) < 1e-12);
  assert.ok(Math.abs(scored[2]!.score - 0.01) < 1e-12);
  assert.equal(scored[3]!.score, Infinity);
  // More views win before a smaller score.
  assert.equal(bestCandidate(scored), 1);
  assert.equal(bestCandidate([scored[3]!]), -1);
});

// The pyramid sample (8 frames, 640 × 360): SIFT on the CPU tier, mutual ratio-0.8 matches of
// every pair. Its focal is unknown. A start of 1.2 × width reconstructed the relief upside down
// (docs/lab-run-records.md). The relief is read in the last frame: the median depth of the
// points seen near the apex against the points seen far from it.
const pyramid = JSON.parse(
  readFileSync(new URL('./fixtures/pyramid-sift-cpu.json', import.meta.url), 'utf8'),
) as {
  width: number;
  height: number;
  frames: { id: number; keypoints: number[] }[];
  pairs: { first: number; second: number; matches: number[] }[];
};
const pyramidImages: ImageFeatures[] = pyramid.frames.map((f) => ({
  id: f.id,
  camera: { fx: 1, fy: 1, cx: (pyramid.width - 1) / 2, cy: (pyramid.height - 1) / 2 },
  keypoints: Float64Array.from(f.keypoints),
}));
const pyramidPairs: PairMatches[] = pyramid.pairs.map((p) => ({
  first: p.first,
  second: p.second,
  matches: Uint32Array.from(p.matches),
}));
function relief(model: SparseReconstruction, viewId = 7): number {
  const view = model.views.find((v) => v.id === viewId)!;
  const { R, t } = view.pose;
  const near: number[] = [],
    far: number[] = [];
  for (const point of model.landmarks) {
    const o = point.observations.find((x) => x.viewId === viewId);
    if (!o) continue;
    const [X, Y, Z] = point.xyz as unknown as [number, number, number];
    const depth = R[6]! * X + R[7]! * Y + R[8]! * Z + t[2]!;
    const d = Math.hypot(o.pixel[0] - 320, o.pixel[1] - 170);
    if (d < 90) near.push(depth);
    else if (d > 220) far.push(depth);
  }
  const median = (v: number[]) => v.sort((a, b) => a - b)[Math.floor(v.length / 2)]!;
  assert.ok(near.length >= 10 && far.length >= 10, `${near.length} near, ${far.length} far`);
  return median(far) / median(near) - 1;
}

test('pyramid: the estimate does not depend on the start, and the relief is the right way up', () => {
  const fixed = reconstructIncremental(at(pyramidImages, 1.2), pyramidPairs, lab);
  assert.ok(relief(fixed.reconstruction) < 0, 'the 1.2 start of 2026-10-20 inverts the relief');
  const results = [0.5, 1.2].map((start) =>
    reconstructIncremental(at(pyramidImages, start), pyramidPairs, { ...lab, focalSearch: true }),
  );
  const [low, high] = results;
  assert.deepEqual(low!.diagnostics.focalSearch, high!.diagnostics.focalSearch);
  assert.equal(medianRatio(low!.reconstruction), medianRatio(high!.reconstruction));
  const found = low!.diagnostics.focalSearch!;
  assert.deepEqual(found.views, [0, 1, 2, 3, 4]);
  assert.equal(found.candidates.length, 17);
  // The pair verifications of the grid come from one base for each pair (docs/auto-focal.md,
  // section 8). Their scores are those that a full verification at each focal gave before that
  // change (measured 2026-10-21, `scripts/measure-focal.mjs --search=1`).
  const before: [number, number][] = [
    [0.456, 0.482],
    [0.525, 0.3688],
    [0.603, 0.2257],
    [0.694, 0.201],
    [0.798, 0.2228],
    [0.918, 0.2258],
    [1.055, 0.3687],
    [1.846, 0.3923],
  ];
  for (const [ratio, score] of before) {
    const c = found.candidates.find((x) => Math.abs(x.ratio - ratio) < 5e-4)!;
    assert.ok(Math.abs(c.score - score) < 5e-4, `${ratio}: ${c.score} against ${score}`);
  }
  // Measured 2026-10-21: 0.78 (docs/auto-focal.md, Measurements).
  assert.ok(found.ratio > 0.65 && found.ratio < 0.95, `estimate ${found.ratio}`);
  assert.equal(low!.reconstruction.views.length, 8);
  assert.ok(relief(low!.reconstruction) > 0.2, `relief ${relief(low!.reconstruction)}`);
});

// graham-hall with the lab's own front end (tests/fixtures/graham-hall-lab-matches.json, see
// weak-side.test.ts), treated as uncalibrated: zero distortion, and a start of 1.2 × width, which
// the final refinement alone leaves at 1.09. COLMAP's focal is 0.692 × width (the median of the
// 12 photos).
test('graham-hall from the lab front end: the estimate lands within 5 % of COLMAP', () => {
  const fixture = JSON.parse(
    readFileSync(new URL('./fixtures/graham-hall-lab-matches.json', import.meta.url), 'utf8'),
  ) as {
    frames: { id: number; camera: ImageFeatures['camera']; keypoints: number[] }[];
    pairs: { first: number; second: number; matches: number[] }[];
  };
  const truth = 0.6916;
  const images = fixture.frames.map((f) => ({
    id: f.id,
    camera: cameraAtRatio({ ...f.camera, k1: 0, k2: 0 }, 1.2),
    keypoints: Float64Array.from(f.keypoints),
  }));
  const pairs = fixture.pairs.map((p) => ({
    first: p.first,
    second: p.second,
    matches: Uint32Array.from(p.matches),
  }));
  const result = reconstructIncremental(images, pairs, { ...lab, focalSearch: true });
  const estimate = medianRatio(result.reconstruction);
  // Measured 2026-10-21: 0.680 (−1.7 %); the subset's estimate 0.675.
  assert.ok(Math.abs(estimate / truth - 1) < 0.05, `estimate ${estimate}`);
  assert.equal(result.reconstruction.views.length, 12);
});
