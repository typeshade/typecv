import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { reconstructIncremental, ReconstructionInitializationError } from '../src/index.ts';
import type { ImageFeatures, PairMatches, SparseReconstruction } from '../src/index.ts';
import {
  bestCandidate,
  cameraAtRatio,
  focalGrid,
  focalCandidateSteps,
  focalSearchControls,
  focalSearchSteps,
  focalSearchComponents,
  parabolaVertex,
  scoreCandidates,
  type FocalCandidate,
} from '../src/reconstruction/focal-search.ts';
import { executeReconstructionJob } from '../src/reconstruction/mapper.ts';
import { focalComponentsFixture } from './fixtures/focal-components.ts';
import { generalVideoFixture } from './fixtures/general-video.ts';

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

test('adaptive focal search brackets minima and exhausts uncertain grids without duplicate runs', () => {
  const grid = Array.from({ length: 17 }, (_, i) => 0.3 * 1.15 ** i);
  const evaluate = (values: { views: number; error: number }[], exhaustive = false) => {
    const machine = focalCandidateSteps(grid, 5, 1.5, exhaustive),
      visited: number[] = [];
    let next = machine.next();
    while (!next.done) {
      const i = grid.indexOf(next.value);
      visited.push(i);
      const value = values[i]!;
      next = machine.next({
        ratio: next.value,
        views: value.views,
        rmsePx: value.error,
        observations: value.views ? 100 : 0,
      });
    }
    assert.equal(new Set(visited).size, visited.length);
    assert.deepEqual(
      next.value.map((c) => c.ratio),
      [...visited].sort((a, b) => a - b).map((i) => grid[i]),
    );
    return { candidates: next.value, visited };
  };
  const bowl = grid.map((_, i) => ({ views: 5, error: 0.1 + (i - 7.2) ** 2 / 100 }));
  const adaptive = evaluate(bowl),
    full = evaluate(bowl, true);
  assert.ok(adaptive.visited.length < 17);
  assert.equal(full.visited.length, 17);
  assert.equal(
    parabolaVertex(adaptive.candidates, bestCandidate(adaptive.candidates)),
    parabolaVertex(full.candidates, bestCandidate(full.candidates)),
  );
  // Multiple basins are refined, including the smaller one between coarse points.
  const basins = bowl.map((v, i) => ({
    ...v,
    error: Math.min(v.error, 0.11 + (i - 12) ** 2 / 100),
  }));
  assert.ok(evaluate(basins).visited.includes(11));
  assert.ok(evaluate(basins).visited.includes(13));
  for (const uncertain of [
    bowl.map((v) => ({ ...v, views: 4 })),
    bowl.map((v) => ({ ...v, error: 1 })),
    bowl.map((v, i) => ({ ...v, error: i / 100 })),
    // Only an odd grid point initializes: a failed coarse pass must still find it.
    bowl.map((v, i) => ({ ...v, views: i === 7 ? 5 : 0 })),
  ])
    assert.equal(evaluate(uncertain).visited.length, 17);
});

function search(
  images: ImageFeatures[],
  pairs: PairMatches[],
  exhaustive = false,
  options = lab,
  controls = focalSearchControls(true)!,
  minimumViews = 2,
) {
  const machine = focalSearchSteps(images, pairs, options, controls, exhaustive, minimumViews);
  let next = machine.next();
  while (!next.done) {
    let value: unknown;
    try {
      value = executeReconstructionJob(next.value);
    } catch (error) {
      next = machine.throw(error);
      continue;
    }
    next = machine.next(value);
  }
  return next.value;
}

function exhaustiveSearch(images: ImageFeatures[], pairs: PairMatches[]) {
  return search(images, pairs, true);
}

test('focal calibration uses a connected subset instead of a disconnected chronological prefix', () => {
  const fixture = focalComponentsFixture();
  const found = search(
    fixture.images,
    fixture.pairs,
    false,
    { ...lab, ransacIterations: 64 },
    focalSearchControls({ refineIterations: 12 })!,
  );
  assert.deepEqual(found.views, [10, 11, 12, 13, 14]);
  assert.ok(Math.abs(found.ratio / fixture.ratio - 1) < 0.02, `estimate ${found.ratio}`);
  assert.ok(found.candidates.some((candidate) => candidate.views === 5));
  const fromLow = focalComponentsFixture(0.5);
  assert.deepEqual(
    search(
      fromLow.images,
      fromLow.pairs,
      false,
      { ...lab, ransacIterations: 64 },
      focalSearchControls({ refineIterations: 12 })!,
    ),
    found,
    'the supplied start does not influence component selection or calibration',
  );
});

test('an invalid interpolated refinement falls back to a grid candidate that built the map', () => {
  const { images, pairs } = focalComponentsFixture();
  const machine = focalSearchSteps(
    images,
    pairs,
    { ...lab, ransacIterations: 64 },
    focalSearchControls({ refineIterations: 12 })!,
  );
  let next = machine.next();
  let rejected = false;
  while (!next.done) {
    const job = next.value;
    if (job.kind === 'bundle' && job.args[1]?.refineSharedFocal) {
      rejected = true;
      next = machine.throw(
        new ReconstructionInitializationError([], 'Injected invalid joint refinement'),
      );
      continue;
    }
    try {
      next = machine.next(executeReconstructionJob(job));
    } catch (error) {
      next = machine.throw(error);
    }
  }
  assert.equal(rejected, true);
  const best = next.value.candidates[bestCandidate(next.value.candidates)]!;
  assert.equal(next.value.ratio, best.ratio);
  assert.notEqual(next.value.ratio, next.value.gridRatio);
  assert.ok(best.views >= 5);
});

test('component selection keeps ties and explicit seeds stable and ignores edges below eight rows', () => {
  const { images, pairs } = focalComponentsFixture();
  const before = structuredClone({ images, pairs });
  const selected = focalSearchComponents(images, pairs);
  assert.deepEqual(
    selected.map((group) => group.map((view) => view.id)),
    [
      [10, 11, 12, 13, 14, 15],
      [0, 1, 2],
    ],
  );
  const tie = [images[3]!, images[4]!, images[0]!, images[1]!];
  const edges = pairs.filter(
    (pair) => tie.some((v) => v.id === pair.first) && tie.some((v) => v.id === pair.second),
  );
  edges.push({ first: 0, second: 10, matches: new Uint32Array(14) });
  assert.deepEqual(
    focalSearchComponents(tie, edges).map((group) => group.map((v) => v.id)),
    [
      [10, 11],
      [0, 1],
    ],
  );
  assert.deepEqual(focalSearchComponents(images, pairs, [0, 1]), [images]);
  assert.deepEqual({ images, pairs }, before);
});

test('streaming focal readiness requires registered calibration views and finish can use short input', () => {
  const fixture = focalComponentsFixture();
  const images = fixture.images.slice(0, 5);
  const ids = new Set(images.map((view) => view.id));
  const pairs = fixture.pairs.filter((p) => ids.has(p.first) && ids.has(p.second));
  const options = { ...lab, ransacIterations: 32 };
  const controls = focalSearchControls({ minRatio: 0.7, maxRatio: 0.7, refineIterations: 0 })!;
  assert.throws(
    () => search(images, pairs, false, options, controls, 5),
    ReconstructionInitializationError,
  );
  // A raw edge can join components without providing enough geometry to register five views.
  const bridge = {
    first: 2,
    second: 10,
    matches: Uint32Array.from(Array.from({ length: 8 }, (_, i) => [i, i]).flat()),
  };
  assert.throws(
    () => search(images, [...pairs, bridge], false, options, controls, 5),
    (error: unknown) => {
      assert.ok(error instanceof ReconstructionInitializationError);
      assert.ok(error.focalSearch!.attempts.some((attempt) => attempt.failure.includes('need 5')));
      return true;
    },
  );
  const final = search(images, pairs, false, options, controls);
  assert.deepEqual(final.views, [10, 11]);
});

test('focal search tries a smaller component when the largest is rotation-only', () => {
  const fixture = focalComponentsFixture();
  const rotation = generalVideoFixture('rotation').input;
  const images = [
    ...rotation.images,
    ...fixture.images.filter((view) => view.id >= 10 && view.id <= 13),
  ];
  const pairs = [
    ...rotation.pairs,
    ...fixture.pairs.filter((pair) => pair.first >= 10 && pair.second <= 13),
  ];
  const found = search(
    images,
    pairs,
    false,
    { ...lab, ransacIterations: 32 },
    focalSearchControls({ minRatio: 0.7, maxRatio: 0.7, refineIterations: 0 })!,
  );
  assert.deepEqual(found.views, [10, 11, 12, 13]);
});

test('failed focal searches preserve geometric rejection evidence without forcing degenerate structure', () => {
  for (const scenario of ['rotation', 'planar', 'far'] as const) {
    const fixture = generalVideoFixture(scenario).input;
    assert.throws(
      () =>
        search(
          fixture.images,
          fixture.pairs,
          false,
          { ...lab, ransacIterations: 32 },
          focalSearchControls({ minRatio: 0.7, maxRatio: 0.7, refineIterations: 0 })!,
        ),
      (error: unknown) => {
        assert.ok(error instanceof ReconstructionInitializationError);
        assert.ok(error.pairVerification.length > 0);
        assert.ok(error.pairVerification.every((pair) => !pair.seedEligible));
        const attempt = error.focalSearch!.attempts[0]!;
        assert.equal(attempt.ratio, 0.7);
        assert.equal(attempt.views, 5);
        assert.equal(attempt.eligiblePairs, 0);
        assert.ok(attempt.bestPair);
        assert.equal(
          attempt.reasons.reduce((sum, r) => sum + r.pairs, 0),
          attempt.pairs,
        );
        assert.ok(attempt.reasons.every((reason) => reason.reason.length > 0));
        return true;
      },
      scenario,
    );
  }
});

test('failure evidence stays bounded when many subsets and focal candidates cannot initialize', () => {
  const base = generalVideoFixture('rotation').input.images[0]!;
  const images = Array.from({ length: 47 }, (_, id) => ({ ...base, id }));
  assert.throws(
    () => search(images, []),
    (error: unknown) => {
      assert.ok(error instanceof ReconstructionInitializationError);
      assert.equal(error.focalSearch!.attempts.length, 64);
      assert.equal(error.focalSearch!.omittedAttempts, 21);
      return true;
    },
  );
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
  const exhaustive = exhaustiveSearch(at(pyramidImages, 0.8), pyramidPairs);
  assert.ok(found.candidates.length < exhaustive.candidates.length);
  assert.ok(Math.abs(found.ratio - exhaustive.ratio) < 1e-6);
  assert.ok(Math.abs(found.gridRatio - exhaustive.gridRatio) < 1e-10);
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
    const c = exhaustive.candidates.find((x) => Math.abs(x.ratio - ratio) < 5e-4)!;
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
