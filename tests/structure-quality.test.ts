import test from 'node:test';
import assert from 'node:assert/strict';
import {
  assessTrackDepth,
  triangulateTrack,
  triangulateTrackDetailed,
  reconstructIncremental,
  ReconstructionInitializationError,
  cameraMaskFromRegions,
} from '../src/index.ts';
import { identityPose, projectPoint, cameraCenter } from '../src/geometry/pose.ts';
import { triangulateLinear } from '../src/geometry/multiview.ts';
import { seededRandom } from '../src/geometry/numerics.ts';
import { reconstructIncrementalScene } from '../src/reconstruction/scene-mapper.ts';
import { selectCameraObservations } from '../src/reconstruction/observation-selection.ts';
import { generalVideoFixture } from './fixtures/general-video.ts';
import { foregroundFixture } from './fixtures/foreground.ts';
import { multiply, transpose } from '../src/sfm/linear.ts';

const camera = { fx: 800, fy: 800, cx: 320, cy: 240 },
  point = Float64Array.of(0.1, -0.15, 10);
const fixed = (centers: number[], scale = 1) =>
  centers.map((x) => {
    const pose = { ...identityPose(), t: Float64Array.of(-x * scale, 0, 0) };
    return {
      camera,
      pose,
      pixel: projectPoint(
        camera,
        pose,
        point.map((v) => v * scale),
      )!,
    };
  });
const controls = { ransacIterations: 128, bundleIterations: 1, finalBundleIterations: 2 };

test('conditional depth sensitivity matches rectified-stereo propagation and is scale invariant', () => {
  const expected = (Math.SQRT2 * 10) / (800 * 0.1),
    result = assessTrackDepth(point, fixed([0, 0.1]))!;
  assert.equal(result.conditionalOn, 'fixed-poses-and-intrinsics');
  assert.ok(Math.abs(result.maxRelativeDepthStdDev / expected - 1) < 1e-9);
  assert.ok(
    Math.abs(
      assessTrackDepth(
        point.map((x) => 100 * x),
        fixed([0, 0.1], 100),
      )!.maxRelativeDepthStdDev /
        expected -
        1,
    ) < 1e-9,
  );
  assert.equal(assessTrackDepth(point, fixed([0, 0])), undefined);
  assert.equal(assessTrackDepth(point, []), undefined);
});

test('conditional depth bound defers a residual-perfect point and admits later baseline observations', () => {
  const observations = fixed([0, 0.1]);
  assert.ok(triangulateTrack(observations));
  assert.deepEqual(triangulateTrackDetailed(observations, { maxRelativeDepthStdDev: 0.1 }), {
    status: 5,
  });
  assert.equal(triangulateTrack(observations, { maxRelativeDepthStdDev: 0.1 }), undefined);
  const later = fixed([0, 0.1, 0.4]);
  assert.equal(triangulateTrackDetailed(later, { maxRelativeDepthStdDev: 0.1 }).status, 0);
  assert.deepEqual(
    triangulateTrack(later),
    triangulateTrack(later, { maxRelativeDepthStdDev: 0.1 }),
  );
  for (const options of [
    { pixelNoiseStdDevPx: 0 },
    { pixelNoiseStdDevPx: Infinity },
    { maxRelativeDepthStdDev: 0 },
    { maxRelativeDepthStdDev: NaN },
  ])
    assert.throws(() => triangulateTrackDetailed([], options), RangeError);
});

test('fixed-pose noise experiment validates the linearized depth spread without truth in triangulation', () => {
  const rng = seededRandom(391),
    sigma = 0.25,
    observations = fixed([0, 0.6]),
    samples: number[] = [];
  const gaussian = () =>
    Math.sqrt(-2 * Math.log(Math.max(1e-12, rng()))) * Math.cos(2 * Math.PI * rng());
  for (let i = 0; i < 600; i++) {
    const noisy = observations.map((o) => ({
      ...o,
      pixel: [o.pixel[0] + sigma * gaussian(), o.pixel[1] + sigma * gaussian()] as const,
    }));
    samples.push(triangulateLinear(noisy)![2]!);
  }
  const mean = samples.reduce((n, x) => n + x, 0) / samples.length,
    spread = Math.sqrt(samples.reduce((n, x) => n + (x - mean) ** 2, 0) / (samples.length - 1)),
    predicted = assessTrackDepth(point, observations, sigma)!.maxRelativeDepthStdDev * point[2]!;
  assert.ok(Math.abs(spread / predicted - 1) < 0.12, `${spread} / ${predicted}`);
});

test('both mappers retain depth-rejected tracks and recover only after useful observations arrive', () => {
  const f = generalVideoFixture('translation');
  for (const run of [reconstructIncremental, reconstructIncrementalScene]) {
    assert.throws(
      () =>
        run(
          f.input.images.slice(0, 2),
          f.input.pairs.filter((p) => p.second < 2),
          { ...controls, maxRelativeDepthStdDev: 0.03 },
        ),
      (e) => {
        assert.ok(e instanceof ReconstructionInitializationError);
        assert.equal(e.tracking!.depthDeferredTrackIds!.length, 60);
        assert.equal(e.tracking!.untriangulatedTrackIds.length, 60);
        assert.ok(!('reconstruction' in e));
        return true;
      },
    );
    const later = run(f.input.images, f.input.pairs, { ...controls, maxRelativeDepthStdDev: 0.03 });
    assert.equal(later.reconstruction.views.length, 5);
    assert.equal(later.reconstruction.landmarks.length, 60);
    assert.deepEqual(later.tracking!.depthDeferredTrackIds, []);
  }
});

test('existing two-view refinement removes the reproduced noisy seed error on both mappers', () => {
  for (const run of [reconstructIncremental, reconstructIncrementalScene]) {
    const f = generalVideoFixture('rotation-start'),
      noise = 0.7;
    const images = f.input.images.map((v) => ({
        ...v,
        keypoints: v.keypoints.map((x, i) => x + noise * Math.sin((i + 1) * (v.id + 3) * 1.37)),
      })),
      pairs = f.input.pairs.filter((p) => p.second - p.first === 1);
    const r = run(images, pairs, {
        ...controls,
        pairRefinementIterations: 10,
        maxRelativeDepthStdDev: 0.1,
      }),
      [first, second] = r.diagnostics.initialPair,
      anchor = f.truth.views[first]!,
      C = cameraCenter(anchor.pose),
      B = cameraCenter(f.truth.views[second]!.pose),
      scale = Math.hypot(...B.map((x, i) => x - C[i]!));
    assert.equal(r.reconstruction.views.length, 5);
    assert.equal(r.reconstruction.landmarks.length, 60);
    for (const v of r.reconstruction.views) {
      const e = multiply(f.truth.views[v.id]!.pose.R, transpose(anchor.pose.R, 3), 3),
        trace = v.pose.R.reduce((n, x, i) => n + x * e[i]!, 0);
      assert.ok((Math.acos(Math.max(-1, Math.min(1, (trace - 1) / 2))) * 180) / Math.PI < 0.4);
    }
    let error = 0;
    for (const p of r.reconstruction.landmarks) {
      const truth = f.truth.points[p.observations[0]!.featureId]!;
      for (let i = 0; i < 3; i++) {
        const world =
          C[i]! +
          scale *
            (anchor.pose.R[i]! * p.xyz[0]! +
              anchor.pose.R[3 + i]! * p.xyz[1]! +
              anchor.pose.R[6 + i]! * p.xyz[2]!);
        error += (world - truth[i]!) ** 2;
      }
    }
    assert.ok(Math.sqrt(error / r.reconstruction.landmarks.length) < 0.15);
  }
});

test('foreground/HUD rows are separated before consensus while original observations remain intact', () => {
  for (const run of [reconstructIncremental, reconstructIncrementalScene]) {
    const f = foregroundFixture(),
      before = structuredClone({ images: f.images, pairs: f.pairs, masks: f.masks });
    const r = run(f.images, f.pairs, { ...controls, cameraObservationMasks: f.masks });
    assert.equal(r.reconstruction.views.length, 5);
    assert.equal(r.reconstruction.landmarks.length, 60);
    assert.equal(r.observationSelection!.inputMatchCount, 3000);
    assert.equal(r.observationSelection!.backgroundMatchCount, 600);
    assert.equal(
      r.observationSelection!.separatedPairs.reduce((n, p) => n + p.matches.length / 2, 0),
      2400,
    );
    assert.ok(
      r.reconstruction.landmarks.every((p) => p.observations.every((o) => o.featureId < 60)),
    );
    assert.deepEqual({ images: f.images, pairs: f.pairs, masks: f.masks }, before);
  }
});

test('quality checks local BA publications without dropping observations of stable points', () => {
  const f = generalVideoFixture('translation');
  for (const run of [reconstructIncremental, reconstructIncrementalScene]) {
    let publications = 0;
    const r = run(f.input.images, f.input.pairs, {
      ...controls,
      maxRelativeDepthStdDev: 0.03,
      globalBundleViewRatio: Infinity,
      globalBundlePointRatio: Infinity,
      onReconstruction(model) {
        publications++;
        const views = new Map(model.views.map((v) => [v.id, v]));
        for (const p of model.landmarks) {
          const observations = p.observations.map((o) => ({
            ...views.get(o.viewId)!,
            pixel: o.pixel,
          }));
          const sensitivity = assessTrackDepth(p.xyz, observations)!;
          assert.ok(sensitivity.maxRelativeDepthStdDev <= 0.03 + 1e-9);
        }
      },
    });
    assert.ok(publications >= 4);
    assert.equal(r.reconstruction.landmarks.length, 60);
    assert.ok(r.reconstruction.landmarks.every((p) => p.observations.length === 5));
  }
});

test('changed masks invalidate preverified verdicts rather than reusing foreground consensus', () => {
  const f = foregroundFixture(),
    p = f.pairs.find((p) => p.first === 0 && p.second === 4)!;
  const r = reconstructIncremental(f.images, [p], {
    ...controls,
    cameraObservationMasks: f.masks,
    preVerified: [
      {
        verified: p,
        rejected: true,
        verification: {
          first: 0,
          second: 4,
          matches: 300,
          inliers: 0,
          homographyInliers: 0,
          medianAngleDeg: 0,
          seedEligible: false,
          reason: 'old foreground verdict',
        },
      },
    ],
  });
  assert.equal(r.reconstruction.views.length, 2);
  assert.equal(r.reconstruction.landmarks.length, 60);
  assert.ok(r.diagnostics.pairVerification[0]!.seedEligible);
});

test('no admitted background defers the map and retains all separated source correspondences', () => {
  const f = foregroundFixture();
  for (const mask of f.masks) mask.values.fill(0);
  for (const run of [reconstructIncremental, reconstructIncrementalScene])
    assert.throws(
      () => run(f.images, f.pairs, { ...controls, cameraObservationMasks: f.masks }),
      (error) => {
        assert.ok(error instanceof ReconstructionInitializationError);
        assert.equal(error.observationSelection!.backgroundMatchCount, 0);
        assert.equal(error.observationSelection!.inputMatchCount, 3000);
        assert.equal(
          error.observationSelection!.separatedPairs.reduce((n, p) => n + p.matches.length / 2, 0),
          3000,
        );
        assert.equal(error.tracking!.rotations.length, 0);
        assert.ok(!('reconstruction' in error));
        return true;
      },
    );
});

test('region masks use image-edge coordinates, preserve feature ids and record pending masks', () => {
  const f = generalVideoFixture('translation'),
    image = { ...f.input.images[0]!, keypoints: Float64Array.of(0, 0, 49, 49, 50, 50, 99, 99) };
  const mask = cameraMaskFromRegions(
    image,
    100,
    100,
    [{ left: 0, top: 0, right: 0.5, bottom: 0.5 }],
    'manual',
  );
  assert.deepEqual(mask.values, Uint8Array.of(0, 0, 1, 1));
  const p = {
    first: 0,
    second: 1,
    matches: Uint32Array.of(0, 0, 1, 1, 2, 2, 3, 3),
    distances: Float64Array.of(8, 7, 6, 5),
  };
  const selection = selectCameraObservations(
    [image, f.input.images[1]!],
    [p],
    [mask, { viewId: 9, values: Uint8Array.of(1), source: 'manual' }],
  );
  assert.deepEqual(selection.pairs[0]!.matches, Uint32Array.of(2, 2, 3, 3));
  assert.deepEqual(selection.pairs[0]!.distances, Float64Array.of(6, 5));
  assert.deepEqual(selection.selection!.unappliedViewIds, [9]);
  assert.throws(
    () =>
      selectCameraObservations(
        [image, f.input.images[1]!],
        [p],
        [{ ...mask, values: Uint8Array.of(1) }],
      ),
    RangeError,
  );
});
