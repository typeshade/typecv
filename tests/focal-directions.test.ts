import test from 'node:test';
import assert from 'node:assert/strict';
import {
  reconstructIncremental,
  ReconstructionInitializationError,
} from '../src/reconstruction/mapper.ts';
import {
  bestCandidate,
  parabolaVertex,
  scoreCandidates,
} from '../src/reconstruction/focal-search.ts';
import { cameraCenter } from '../src/geometry/pose.ts';
import { multiply, transpose } from '../src/sfm/linear.ts';
import { directionObservationFixture } from './fixtures/direction-observations.ts';
import { planMvsViews } from '../src/gpu/mvs-views.ts';
import { supportedBundleDirections } from '../src/reconstruction/directional-tracks.ts';
import { bundleDirectionsFixture } from './fixtures/bundle-directions.ts';

const controls = {
  separateDirectionalObservations: true,
  focalSearch: true,
  ransacIterations: 128,
  bundleIterations: 2,
  finalBundleIterations: 2,
};

test('automatic focal calibration keeps supported direction depth unresolved and recovers one map gauge', () => {
  for (const noisePx of [0, 0.2]) {
    const f = directionObservationFixture({ noisePx }),
      images = f.input.images.map((v) => ({ ...v, camera: { ...v.camera, fx: 512, fy: 512 } })),
      before = structuredClone({ images, pairs: f.input.pairs }),
      r = reconstructIncremental(images, f.input.pairs, controls),
      model = r.reconstruction;
    assert.deepEqual({ images, pairs: f.input.pairs }, before);
    assert.equal(model.views.length, 5);
    assert.equal(model.landmarks.length, 60);
    assert.ok(model.landmarks.every((p) => p.observations.every((o) => o.featureId < 60)));
    assert.equal(r.tracking!.untriangulatedTrackIds.length, 60);
    for (const id of r.tracking!.untriangulatedTrackIds)
      assert.ok(r.tracking!.tracks[id]!.observations.every((o) => o.featureId >= 60));
    assert.equal(r.diagnostics.focalSearch!.scoring, 'directional-bic');
    assert.equal(r.diagnostics.focalSearch!.refinements!.length, 2);
    for (const v of model.views) assert.ok(Math.abs(v.camera.fx / 480 - 1) < 0.01);
    const [a, b] = r.diagnostics.initialPair,
      anchor = f.truth.views[a]!,
      origin = cameraCenter(anchor.pose),
      other = cameraCenter(f.truth.views[b]!.pose),
      scale = Math.hypot(...other.map((x, i) => x - origin[i]!)),
      world = (p: Float64Array) =>
        [0, 1, 2].map(
          (i) =>
            origin[i]! +
            scale *
              (anchor.pose.R[i]! * p[0]! +
                anchor.pose.R[3 + i]! * p[1]! +
                anchor.pose.R[6 + i]! * p[2]!),
        );
    for (const v of model.views) {
      const expectedR = multiply(f.truth.views[v.id]!.pose.R, transpose(anchor.pose.R, 3), 3),
        trace = v.pose.R.reduce((s, x, i) => s + x * expectedR[i]!, 0),
        angle = (Math.acos(Math.max(-1, Math.min(1, (trace - 1) / 2))) * 180) / Math.PI;
      assert.ok(angle < 0.2, `rotation error ${angle} degrees`);
    }
    const pointRmse = Math.sqrt(
      model.landmarks.reduce((sum, p) => {
        const expected = f.truth.finitePoints[p.observations[0]!.featureId]!;
        return sum + world(p.xyz).reduce((s, x, i) => s + (x - expected[i]!) ** 2, 0);
      }, 0) / 60,
    );
    assert.ok(pointRmse < 0.1, `point error ${pointRmse} in one seed gauge`);
    // Existing MVS consumes the same calibrated finite scene. Pending direction hypotheses
    // must not enter its normalization, finite depth range or sparse depth seeds.
    const densePlan = planMvsViews({
      views: model.views.map((v) => ({ ...v, width: 640, height: 480 })),
      landmarks: model.landmarks,
    });
    assert.equal(densePlan.packed.length, 5);
    assert.ok(densePlan.views.every((v) => v.landmarks === 60 && v.status === 'ready'));
    assert.ok(densePlan.cameras.every(Number.isFinite));
    assert.ok(densePlan.seeds.every(Number.isFinite));
    assert.ok(densePlan.seeds.length / 3 <= 300);
  }
});

test('directional scoring accounts for depth freedom and does not interpolate across model changes', () => {
  const runs = [
    {
      ratio: 0.8,
      views: 5,
      rmsePx: 0.142,
      observations: 600,
      modelParameters: 323,
      directionalObservations: 300,
    },
    {
      ratio: 0.95,
      views: 5,
      rmsePx: 0.14,
      observations: 600,
      modelParameters: 383,
      directionalObservations: 0,
    },
    {
      ratio: 1.1,
      views: 5,
      rmsePx: 0.143,
      observations: 600,
      modelParameters: 383,
      directionalObservations: 0,
    },
  ];
  const c = scoreCandidates(runs, 1.5);
  assert.equal(
    bestCandidate(c),
    0,
    'extra finite depth coordinates do not win by overfitting noise',
  );
  assert.equal(
    parabolaVertex(
      c.map((x, i) => ({ ...x, score: i === 1 ? -10 : 0 })),
      1,
    ),
    0.95,
  );
  // No supported bearing model at the largest view count: preserve the established score.
  const finite = runs.map((c) => ({ ...c, directionalObservations: 0 }));
  for (const c of scoreCandidates(finite, 1.5)) assert.equal(c.score, c.rmsePx ** 2);
  const betterCoverage = scoreCandidates([...runs, { ...runs[1]!, views: 6 }], 1.5);
  assert.equal(
    betterCoverage[0]!.score,
    runs[0]!.rmsePx ** 2,
    'partial bearing support cannot change the objective of a better-covered finite map',
  );
  assert.equal(bestCandidate([{ ...c[0]!, score: Infinity }]), -1);
});

test('unknown calibration with only directions or rotation still defers finite initialization', () => {
  for (const f of [
    directionObservationFixture({ finiteCount: 0 }),
    directionObservationFixture({ scenario: 'rotation' }),
  ]) {
    const images = f.input.images.map((v) => ({ ...v, camera: { ...v.camera, fx: 512, fy: 512 } }));
    assert.throws(
      () => reconstructIncremental(images, f.input.pairs, controls),
      ReconstructionInitializationError,
    );
  }
});

test('automatic bearing constraints require repeat consensus and spatial support', () => {
  const f = bundleDirectionsFixture();
  assert.equal(supportedBundleDirections(f.directions, f.truth.views).length, 60);
  assert.equal(supportedBundleDirections(f.directions.slice(0, 3), f.truth.views).length, 0);
  const clustered = f.directions.map((d) => ({
    ...d,
    observations: d.observations.map((o, i) => ({
      ...o,
      pixel: f.directions[0]!.observations[i]!.pixel,
    })),
  }));
  assert.equal(supportedBundleDirections(clustered, f.truth.views).length, 0);
  const single = directionObservationFixture({ directionalCount: 1 }),
    r = reconstructIncremental(single.input.images, single.input.pairs, {
      ...controls,
      focalSearch: false,
    });
  assert.equal(r.reconstruction.landmarks.length, 60);
  assert.equal(r.tracking!.untriangulatedTrackIds.length, 1);
  assert.equal(r.diagnostics.bundleAdjustments.at(-1)!.directionalObservations, undefined);
});
