import test from 'node:test';
import assert from 'node:assert/strict';
import {
  estimateRelativeRotation,
  cameraCenter,
  projectPoint,
  reconstructIncremental,
  ReconstructionInitializationError,
  type Camera,
  type PairMatches,
  type IncrementalResult,
} from '../src/index.ts';
import { multiply, transpose, determinant3 } from '../src/sfm/linear.ts';
import { reconstructIncrementalScene } from '../src/reconstruction/scene-mapper.ts';
import { generalVideoFixture } from './fixtures/general-video.ts';

const controls = { ransacIterations: 128, bundleIterations: 1, finalBundleIterations: 2 };
// One global gauge fixed by the seed; no frame-by-frame or shot-by-shot alignment.
function accurateMap(result: IncrementalResult, fixture: ReturnType<typeof generalVideoFixture>) {
  const [first, second] = result.diagnostics.initialPair,
    anchor = fixture.truth.views[first]!,
    center = cameraCenter(anchor.pose),
    other = cameraCenter(fixture.truth.views[second]!.pose),
    scale = Math.hypot(...other.map((x, i) => x - center[i]!));
  const world = (point: Float64Array) =>
    [0, 1, 2].map(
      (r) =>
        center[r]! +
        scale *
          (anchor.pose.R[r]! * point[0]! +
            anchor.pose.R[3 + r]! * point[1]! +
            anchor.pose.R[6 + r]! * point[2]!),
    );
  for (const view of result.reconstruction.views) {
    const truth = fixture.truth.views[view.id]!,
      expected = multiply(truth.pose.R, transpose(anchor.pose.R, 3), 3);
    assert.ok(Math.hypot(...view.pose.R.map((x, i) => x - expected[i]!)) < 1e-6);
    assert.ok(
      Math.hypot(
        ...world(cameraCenter(view.pose)).map((x, i) => x - cameraCenter(truth.pose)[i]!),
      ) < 1e-5,
    );
  }
  for (const point of result.reconstruction.landmarks) {
    const truth = fixture.truth.points[point.observations[0]!.featureId]!;
    assert.ok(Math.hypot(...world(point.xyz).map((x, i) => x - truth[i]!)) < 1e-4);
  }
}
function pixels(fixture: ReturnType<typeof generalVideoFixture>, pair: PairMatches) {
  const a = fixture.input.images.find((v) => v.id === pair.first)!,
    b = fixture.input.images.find((v) => v.id === pair.second)!;
  return Float64Array.from(
    Array.from({ length: pair.matches.length / 2 }, (_, i) => [
      a.keypoints[2 * pair.matches[2 * i]!]!,
      a.keypoints[2 * pair.matches[2 * i]! + 1]!,
      b.keypoints[2 * pair.matches[2 * i + 1]!]!,
      b.keypoints[2 * pair.matches[2 * i + 1]! + 1]!,
    ]).flat(),
  );
}

test('calibrated rotation tolerates outliers, radial inversion and known zoom without inventing translation', () => {
  const fixture = generalVideoFixture('rotation-zoom'),
    a = fixture.truth.views[0]!,
    b = fixture.truth.views[3]!;
  const first: Camera = { ...a.camera, k1: -0.15 },
    second: Camera = { ...b.camera, k1: -0.15 };
  const raw = Float64Array.from(
    fixture.truth.points.flatMap((p) => [
      ...projectPoint(first, a.pose, p)!,
      ...projectPoint(second, b.pose, p)!,
    ]),
  );
  const outliers = new Set<number>();
  for (let i = 0; i < raw.length / 4; i++) {
    if (i % 5 === 0) {
      outliers.add(i);
      raw[4 * i + 2] += i % 2 ? 110 : -110;
    } else {
      raw[4 * i + 2] += Math.sin(i) * 0.12;
      raw[4 * i + 3] += Math.cos(i) * 0.12;
    }
  }
  raw[0] = 1e6; // outside the injective radial domain, retaining original raw row ownership
  const before = raw.slice(),
    result = estimateRelativeRotation(raw, first, second)!;
  assert.ok(result);
  assert.equal(result.invertibleCount, 59);
  assert.equal(result.inlierRows.length, 48);
  assert.ok([...result.inlierRows].every((row) => !outliers.has(row)));
  const expected = multiply(b.pose.R, transpose(a.pose.R, 3), 3);
  assert.ok(Math.hypot(...result.R.map((x, i) => x - expected[i]!)) < 0.001);
  assert.ok(Math.abs(determinant3(result.R) - 1) < 1e-12);
  assert.ok(result.meanReprojectionErrorPx < 0.3);
  assert.ok(Math.min(...result.spreadDeg) > 0.5);
  assert.equal(result.translation, 'unobservable');
  assert.equal(result.depth, 'not-estimated');
  assert.ok(!('t' in result) && !('points' in result));
  assert.deepEqual(raw, before);
  const reverse = Float64Array.from(
    Array.from({ length: raw.length / 4 }, (_, i) => [
      raw[4 * i + 2]!,
      raw[4 * i + 3]!,
      raw[4 * i]!,
      raw[4 * i + 1]!,
    ]).flat(),
  );
  const reversed = estimateRelativeRotation(reverse, second, first)!;
  assert.deepEqual(reversed.inlierRows, result.inlierRows);
  assert.ok(Math.hypot(...reversed.R.map((x, i) => x - transpose(result.R, 3)[i]!)) < 1e-10);
});

test('coverage, planar translation and incorrect zoom calibration cannot masquerade as an accepted rotation', () => {
  const planar = generalVideoFixture('planar'),
    pair = planar.input.pairs.find((p) => p.first === 0 && p.second === 4)!;
  assert.equal(
    estimateRelativeRotation(
      pixels(planar, pair),
      planar.input.images[0]!.camera,
      planar.input.images[4]!.camera,
      { maxErrorPx: 0.1, minInlierRatio: 0.9 },
    ),
    undefined,
  );
  const zoom = generalVideoFixture('zoom'),
    zoomPair = zoom.input.pairs.find((p) => p.first === 0 && p.second === 4)!;
  const raw = pixels(zoom, zoomPair),
    first = zoom.input.images[0]!.camera,
    second = zoom.input.images[4]!.camera;
  assert.ok(estimateRelativeRotation(raw, first, second));
  assert.equal(estimateRelativeRotation(raw, first, first, { maxErrorPx: 1 }), undefined);
  const line = Float64Array.from(
    Array.from({ length: 40 }, (_, i) => [
      first.cx - 100 + 5 * i,
      first.cy,
      first.cx - 100 + 5 * i,
      first.cy,
    ]).flat(),
  );
  assert.equal(estimateRelativeRotation(line, first, first), undefined);
  const cluster = line.map((x, i) => (i % 2 === 0 ? first.cx : first.cy) + (x - first.cx) * 1e-6);
  assert.equal(estimateRelativeRotation(cluster, first, first), undefined);
});

test('a rotation-compatible distant scene retains translation and depth ambiguity', () => {
  const f = generalVideoFixture('translation'),
    a = f.truth.views[0]!,
    b = f.truth.views[1]!;
  // Scale all world coordinates, retaining broad image coverage rather than a tiny central patch.
  const raw = Float64Array.from(
    f.truth.points.flatMap((point) => {
      const distant = point.map((x) => 1e6 * x);
      return [
        ...projectPoint(a.camera, a.pose, distant)!,
        ...projectPoint(b.camera, b.pose, distant)!,
      ];
    }),
  );
  const result = estimateRelativeRotation(raw, a.camera, b.camera)!;
  assert.ok(result);
  assert.equal(result.translation, 'unobservable');
  assert.equal(result.depth, 'not-estimated');
  // The actual fixture centers move: acceptance is compatibility, not a pure-rotation classifier.
  assert.notDeepEqual(f.truth.views[0]!.pose.t, f.truth.views[1]!.pose.t);
});

test('both CPU mappers retain relative orientation and 2D tracks while refusing a rotation-only map', () => {
  const f = generalVideoFixture('rotation-start'),
    images = f.input.images.slice(0, 3),
    pairs = f.input.pairs.filter((p) => p.second < 3 && p.second - p.first === 1);
  for (const run of [reconstructIncremental, reconstructIncrementalScene]) {
    assert.throws(
      () => run(images, pairs, { ...controls, rotationTracking: true }),
      (e: unknown) => {
        assert.ok(e instanceof ReconstructionInitializationError);
        assert.ok(e.pairVerification.every((p) => !p.seedEligible));
        const tracking = e.tracking!;
        assert.equal(tracking.rotations.length, 2);
        assert.equal(tracking.tracks.length, 60);
        assert.equal(tracking.untriangulatedTrackIds.length, 60);
        for (const p of tracking.rotations) {
          const expected = multiply(
            f.truth.views[p.second]!.pose.R,
            transpose(f.truth.views[p.first]!.pose.R, 3),
            3,
          );
          assert.ok(Math.hypot(...p.rotation.R.map((x, i) => x - expected[i]!)) < 1e-8);
          assert.ok(!('t' in p.rotation));
        }
        assert.ok(!('reconstruction' in e));
        return true;
      },
    );
  }
});

test('rotation-prefix observations connect to a later baseline; disabled mode preserves the existing result', () => {
  const f = generalVideoFixture('rotation-start'),
    pairs = f.input.pairs.filter((p) => p.second - p.first === 1);
  for (const run of [reconstructIncremental, reconstructIncrementalScene]) {
    const result = run(f.input.images, pairs, { ...controls, rotationTracking: true });
    assert.equal(result.reconstruction.views.length, 5);
    assert.equal(result.reconstruction.landmarks.length, 60);
    accurateMap(result, f);
    assert.equal(result.tracking!.untriangulatedTrackIds.length, 0);
    assert.equal(result.tracking!.rotations.length, 2);
    assert.ok(
      result.diagnostics.pairVerification
        .filter((p) => p.rotationInliers)
        .every((p) => !p.seedEligible),
    );
    const baseline = run(f.input.images, pairs, controls),
      disabled = run(f.input.images, pairs, { ...controls, rotationTracking: false });
    assert.deepEqual(disabled, baseline);
    assert.ok(!('tracking' in baseline));
  }
});

test('depth stays pending without reobservation and becomes available only with a later view of that point', () => {
  for (const [scenario, points, pending] of [
    ['mapped-rotation', 40, 20],
    ['reobserved', 60, 0],
  ] as const) {
    const f = generalVideoFixture(scenario);
    for (const run of [reconstructIncremental, reconstructIncrementalScene]) {
      const result = run(f.input.images, f.input.pairs, {
        ...controls,
        initialPair: [0, 1],
        rotationTracking: true,
      });
      assert.equal(result.reconstruction.landmarks.length, points, scenario);
      accurateMap(result, f);
      assert.equal(result.tracking!.untriangulatedTrackIds.length, pending, scenario);
      for (const id of result.tracking!.untriangulatedTrackIds)
        assert.ok(
          result.tracking!.tracks[id]!.observations.every(
            (o) => o.featureId >= 40 && (o.viewId === 2 || o.viewId === 3),
          ),
        );
    }
  }
});

test('rotation evidence reuses a pre-verified rejection without promoting its seed eligibility', () => {
  const f = generalVideoFixture('rotation'),
    images = f.input.images.slice(0, 2),
    pairs = f.input.pairs.slice(0, 1),
    p = pairs[0]!;
  assert.throws(
    () =>
      reconstructIncremental(images, pairs, {
        ...controls,
        rotationTracking: true,
        preVerified: [
          {
            rejected: true,
            verified: { ...p, matches: new Uint32Array() },
            verification: {
              first: p.first,
              second: p.second,
              matches: 60,
              inliers: 0,
              homographyInliers: 60,
              medianAngleDeg: 0,
              seedEligible: false,
              reason: 'Earlier H/E rejection',
            },
          },
        ],
      }),
    (e: unknown) => {
      assert.ok(e instanceof ReconstructionInitializationError);
      assert.equal(e.pairVerification[0]!.reason, 'Earlier H/E rejection');
      assert.equal(e.pairVerification[0]!.seedEligible, false);
      assert.equal(e.tracking!.tracks.length, 60);
      return true;
    },
  );
});

test('rotation option validation occurs without observations and unknown focal search is explicit', () => {
  const f = generalVideoFixture('rotation'),
    k = f.input.images[0]!.camera;
  for (const options of [
    { iterations: 0 },
    { minInliers: 2 },
    { minInlierRatio: 0 },
    { minSpreadDeg: 0 },
    { confidence: 1 },
    { seed: NaN },
  ])
    assert.throws(() => estimateRelativeRotation(new Float64Array(), k, k, options), RangeError);
  assert.equal(estimateRelativeRotation(new Float64Array(), k, k), undefined);
  assert.throws(() => estimateRelativeRotation(Float64Array.of(1, 2, 3), k, k), RangeError);
  assert.throws(
    () =>
      reconstructIncremental(f.input.images, f.input.pairs, {
        rotationTracking: true,
        focalSearch: true,
      }),
    /supplied calibration/,
  );
  assert.throws(
    () => reconstructIncremental(f.input.images, f.input.pairs, { rotationTracking: false }),
    (error: unknown) => {
      assert.ok(error instanceof ReconstructionInitializationError);
      assert.ok(!('tracking' in error));
      return true;
    },
  );
});
