import test from 'node:test';
import assert from 'node:assert/strict';
import {
  reconstructIncremental,
  ReconstructionInitializationError,
} from '../src/reconstruction/mapper.ts';
import { reconstructIncrementalScene } from '../src/reconstruction/scene-mapper.ts';
import {
  directionObservationFixture,
  directionPairPixels,
} from './fixtures/direction-observations.ts';
import { reconstructTwoView } from '../src/sfm/two-view.ts';
import { cameraCenter } from '../src/geometry/pose.ts';
import { GeometryEstimationError } from '../src/geometry/errors.ts';

// An infinite direction field follows orientation only, even while the physical camera translates.
// Known calibration and supplied point identities isolate geometry; no semantic classifier.
function directionalInput(mixed: boolean) {
  return directionObservationFixture({ finiteCount: mixed ? 60 : 0 }).input;
}

test('infinite directions cannot initialize finite structure despite physical camera movement', () => {
  const { images, pairs } = directionalInput(false);
  for (const mapper of [reconstructIncremental, reconstructIncrementalScene])
    for (const separateDirectionalObservations of [undefined, true]) {
      assert.throws(
        () => mapper(images, pairs, { ransacIterations: 128, separateDirectionalObservations }),
        (error: unknown) => {
          assert.ok(error instanceof ReconstructionInitializationError);
          assert.ok(error.tracking!.rotations.length > 0);
          assert.equal(error.tracking!.tracks.length, 60);
          assert.equal(error.tracking!.untriangulatedTrackIds.length, 60);
          assert.ok(error.tracking!.rotations.every((r) => r.rotation.depth === 'not-estimated'));
          return true;
        },
      );
    }
  // The uncalibrated mode must also defer; calibrated orientation tracking is not enabled there.
  assert.throws(
    () =>
      reconstructIncremental(images, pairs, {
        focalSearch: true,
        ransacIterations: 128,
      }),
    ReconstructionInitializationError,
  );
});

test('finite structure and infinite directions retain separate depth outcomes', () => {
  const { images, pairs } = directionalInput(true);
  for (const mapper of [reconstructIncremental, reconstructIncrementalScene]) {
    // The opt-in stays separate from the established default until noisy/uncalibrated
    // geometry is reliable enough; adding a mode must not silently change that path.
    assert.throws(
      () => mapper(images, pairs, { ransacIterations: 128 }),
      ReconstructionInitializationError,
    );
    const result = mapper(images, pairs, {
      separateDirectionalObservations: true,
      ransacIterations: 128,
      bundleIterations: 2,
      finalBundleIterations: 2,
    });
    assert.equal(result.reconstruction.views.length, 5);
    assert.equal(result.reconstruction.landmarks.length, 60);
    assert.ok(
      result.reconstruction.landmarks.every((p) => p.observations.every((o) => o.featureId < 60)),
    );
    assert.equal(result.tracking!.untriangulatedTrackIds.length, 60);
    assert.ok(
      result.tracking!.untriangulatedTrackIds.every((id) =>
        result.tracking!.tracks[id]!.observations.every((o) => o.featureId >= 60),
      ),
    );
  }
});

test('directional mode rechecks cached legacy rejections and validates its option', () => {
  const { images, pairs } = directionalInput(true);
  let failure: ReconstructionInitializationError | undefined;
  try {
    reconstructIncremental(images, pairs, { ransacIterations: 128 });
  } catch (e) {
    assert.ok(e instanceof ReconstructionInitializationError);
    failure = e;
  }
  assert.ok(failure);
  const preVerified = pairs.map((p, i) => ({
    verified: { ...p, matches: new Uint32Array() },
    verification: failure!.pairVerification[i]!,
    rejected: true,
  }));
  const r = reconstructIncremental(images, pairs, {
    separateDirectionalObservations: true,
    preVerified,
    ransacIterations: 128,
    bundleIterations: 2,
    finalBundleIterations: 2,
  });
  assert.equal(r.reconstruction.views.length, 5);
  assert.equal(r.reconstruction.landmarks.length, 60);
  const invalid = { separateDirectionalObservations: 'enabled' as unknown as boolean };
  assert.throws(() => reconstructIncremental(images, pairs, invalid), RangeError);
  const fixture = directionObservationFixture();
  assert.throws(
    () =>
      reconstructTwoView(
        directionPairPixels(fixture),
        images[0]!.camera,
        images[4]!.camera,
        invalid,
      ),
    RangeError,
  );
});

test('two-view recovery keeps directional consensus out of finite depth and refinement', () => {
  for (const directionalCount of [0, 30, 60, 180])
    for (const noisePx of [0, 0.2])
      for (const zoom of [false, true]) {
        const f = directionObservationFixture({ directionalCount, noisePx, zoom }),
          a = f.input.images[0]!.camera,
          b = f.input.images[4]!.camera;
        for (const refinementIterations of [0, 5]) {
          const r = reconstructTwoView(directionPairPixels(f), a, b, {
            separateDirectionalObservations: true,
            iterations: 128,
            refinementIterations,
          });
          if (directionalCount || !noisePx) assert.equal(r.positiveDepthCount, 60);
          else assert.ok(r.positiveDepthCount >= 40);
          assert.ok((r.diagnostics.directionCompatibleCount ?? 0) >= directionalCount * 0.8);
          assert.ok(r.diagnostics.medianParallaxDeg > 1);
          for (let i = 60; i < 60 + directionalCount; i++) {
            assert.equal(r.inliers[i], 1, 'direction observations remain in the 2D consensus');
            assert.equal(r.validPoints[i], 0);
            assert.ok(r.points.slice(i * 3, i * 3 + 3).every(Number.isNaN));
          }
        }
      }
});

test('direction support cannot replace finite cheirality or the minimum point count', () => {
  const f = directionObservationFixture(),
    a = f.input.images[0]!,
    b = f.input.images[4]!,
    pixels = directionPairPixels(f),
    { R, t } = f.truth.views[4]!.pose;
  // Half of the finite points are behind both cameras; even perfect directional
  // support must not choose one sign of translation from a 50/50 finite split.
  for (let i = 30; i < 60; i++) {
    const X = f.truth.finitePoints[i]!.map((x) => -x),
      q = [0, 1, 2].map(
        (r) => R[r * 3]! * X[0]! + R[r * 3 + 1]! * X[1]! + R[r * 3 + 2]! * X[2]! + t[r]!,
      );
    pixels[i * 4 + 2] = (b.camera.fx * q[0]!) / q[2]! + b.camera.cx;
    pixels[i * 4 + 3] = (b.camera.fy * q[1]!) / q[2]! + b.camera.cy;
  }
  assert.throws(
    () =>
      reconstructTwoView(pixels, a.camera, b.camera, {
        iterations: 128,
        separateDirectionalObservations: true,
      }),
    /Cheirality failed/,
  );
  const few = directionObservationFixture({ finiteCount: 5, directionalCount: 180 });
  assert.throws(
    () =>
      reconstructTwoView(directionPairPixels(few), a.camera, b.camera, {
        iterations: 128,
        separateDirectionalObservations: true,
      }),
    GeometryEstimationError,
  );
});

test('far finite tracks acquire depth only after another observed viewpoint supplies parallax', () => {
  const f = directionObservationFixture({ distantDepth: 100 }),
    controls = {
      separateDirectionalObservations: true,
      initialPair: [0, 1] as const,
      ransacIterations: 128,
      bundleIterations: 2,
      finalBundleIterations: 2,
    };
  for (const mapper of [reconstructIncremental, reconstructIncrementalScene]) {
    const prefix = mapper(
      f.input.images.slice(0, 2),
      f.input.pairs.filter((p) => p.second < 2),
      controls,
    );
    assert.equal(prefix.reconstruction.landmarks.length, 60);
    assert.equal(prefix.tracking!.untriangulatedTrackIds.length, 60);
    const full = mapper(f.input.images, f.input.pairs, controls);
    assert.equal(full.reconstruction.landmarks.length, 120);
    assert.equal(full.tracking!.untriangulatedTrackIds.length, 0);
    const anchor = cameraCenter(f.truth.views[0]!.pose),
      other = cameraCenter(f.truth.views[1]!.pose),
      scale = Math.hypot(...other.map((x, k) => x - anchor[k]!));
    for (const point of full.reconstruction.landmarks) {
      const id = point.observations[0]!.featureId;
      if (id >= 60)
        assert.ok(Math.abs(point.xyz[2]! * scale - 100) < 0.01, 'one seed gauge for all views');
    }
    const unseen = mapper(
      f.input.images,
      f.input.pairs.map((p) => ({
        ...p,
        matches: p.second < 2 ? p.matches : p.matches.filter((id) => id < 60),
      })),
      controls,
    );
    assert.equal(unseen.reconstruction.landmarks.length, 60);
    assert.equal(unseen.tracking!.untriangulatedTrackIds.length, 60);
  }
});

test('featureless input defers without filling cameras or depths', () => {
  const { images } = directionalInput(false),
    empty = images.map((v) => ({ ...v, keypoints: new Float64Array() }));
  assert.throws(
    () => reconstructIncremental(empty, []),
    (error: unknown) => {
      assert.ok(error instanceof ReconstructionInitializationError);
      assert.equal(error.tracking!.tracks.length, 0);
      assert.equal(error.tracking!.rotations.length, 0);
      return true;
    },
  );
});
