import test from 'node:test';
import assert from 'node:assert/strict';
import { reconstructIncremental, ReconstructionInitializationError } from '../src/index.ts';
import { cameraCenter, projectPoint } from '../src/geometry/pose.ts';
import { multiply, transpose } from '../src/sfm/linear.ts';
import {
  generalVideoFixture,
  renderGeneralVideoFrame,
  type GeneralVideoCase,
} from './fixtures/general-video.ts';

const controls = { ransacIterations: 256, bundleIterations: 2, finalBundleIterations: 4 };

test('known-calibration geometry fixtures preserve global camera/point accuracy under one map gauge', () => {
  for (const scenario of ['translation', 'rotation-start', 'orbit'] as const) {
    const fixture = generalVideoFixture(scenario);
    const before = structuredClone(fixture.input);
    const result = reconstructIncremental(fixture.input.images, fixture.input.pairs, controls);
    assert.deepEqual(fixture.input, before);
    assert.equal(result.reconstruction.views.length, 5, scenario);
    assert.equal(result.reconstruction.landmarks.length, 60, scenario);
    const [a, b] = result.diagnostics.initialPair;
    const anchor = fixture.truth.views[a]!,
      anchorCenter = cameraCenter(anchor.pose);
    const secondCenter = cameraCenter(fixture.truth.views[b]!.pose);
    const scale = Math.hypot(...secondCenter.map((value, axis) => value - anchorCenter[axis]!));
    const toWorld = (point: Float64Array) =>
      [0, 1, 2].map(
        (axis) =>
          scale *
            (anchor.pose.R[axis]! * point[0]! +
              anchor.pose.R[3 + axis]! * point[1]! +
              anchor.pose.R[6 + axis]! * point[2]!) +
          anchorCenter[axis]!,
      );
    for (const view of result.reconstruction.views) {
      const expected = fixture.truth.views[view.id]!;
      const expectedR = multiply(expected.pose.R, transpose(anchor.pose.R, 3), 3);
      assert.ok(
        Math.hypot(...view.pose.R.map((value, axis) => value - expectedR[axis]!)) < 1e-6,
        scenario,
      );
      const center = toWorld(cameraCenter(view.pose)),
        truth = cameraCenter(expected.pose);
      assert.ok(Math.hypot(...center.map((value, axis) => value - truth[axis]!)) < 1e-5, scenario);
    }
    for (const point of result.reconstruction.landmarks) {
      const truth = fixture.truth.points[point.observations[0]!.featureId]!;
      assert.ok(
        Math.hypot(...toWorld(point.xyz).map((value, axis) => value - truth[axis]!)) < 1e-4,
        scenario,
      );
    }
  }
});

test('rotation, distant structure, planar scenes and calibrated zoom do not force structure initialization', () => {
  for (const scenario of ['rotation', 'far', 'planar', 'zoom', 'rotation-zoom'] as const) {
    const fixture = generalVideoFixture(scenario);
    assert.throws(
      () => reconstructIncremental(fixture.input.images, fixture.input.pairs, controls),
      (error: unknown) =>
        error instanceof ReconstructionInitializationError &&
        error.pairVerification.length > 0 &&
        error.pairVerification.every((pair) => !pair.seedEligible),
      scenario,
    );
  }
});

test('registered rotation frames do not determine unseen depths; reobservation after motion can', () => {
  const run = (scenario: GeneralVideoCase) => {
    const fixture = generalVideoFixture(scenario);
    return reconstructIncremental(fixture.input.images, fixture.input.pairs, {
      ...controls,
      initialPair: [0, 1],
    });
  };
  const unseen = run('mapped-rotation');
  assert.equal(unseen.reconstruction.views.length, 5);
  assert.equal(unseen.reconstruction.landmarks.length, 40);
  assert.ok(
    unseen.reconstruction.landmarks.every((point) =>
      point.observations.every((o) => o.featureId < 40),
    ),
  );
  const reseen = run('reobserved');
  assert.equal(reseen.reconstruction.views.length, 6);
  assert.equal(reseen.reconstruction.landmarks.length, 60);
  assert.ok(
    reseen.reconstruction.landmarks.some(
      (point) =>
        point.observations[0]!.featureId >= 40 && point.observations.some((o) => o.viewId >= 4),
    ),
  );
});

test('supplied cross-shot matches allow joint initialization while disconnected evidence stays unregistered', () => {
  const together = generalVideoFixture('same-place-shots');
  const combined = reconstructIncremental(together.input.images, together.input.pairs, controls);
  const [a, b] = combined.diagnostics.initialPair;
  assert.notEqual(together.truth.shotOfView[a], together.truth.shotOfView[b]);
  assert.equal(combined.reconstruction.views.length, 4);
  const separate = generalVideoFixture('disconnected');
  const partial = reconstructIncremental(separate.input.images, separate.input.pairs, {
    ...controls,
    initialPair: [0, 2],
  });
  assert.deepEqual([...partial.reconstruction.views.map((v) => v.id)].sort(), [0, 1, 2]);
  assert.deepEqual(
    partial.unregistered.map((v) => v.viewId),
    [3, 4, 5],
  );
  assert.ok(
    partial.reconstruction.landmarks.every((point) =>
      point.observations.every((o) => o.viewId < 3),
    ),
  );
});

test('RGB fixtures have deterministic pixels, camera-Z depth and a distinct non-oracle input contract', () => {
  const fixture = generalVideoFixture('translation');
  const frame = renderGeneralVideoFrame(fixture.truth.views[1]!, 96, 72);
  const repeated = renderGeneralVideoFrame(fixture.truth.views[1]!, 96, 72);
  assert.deepEqual(frame, repeated);
  assert.ok(frame.rgba.some((value, i) => i % 4 !== 3 && value > 100));
  assert.ok(frame.depthZ.every((value) => value > 0));
  const center = cameraCenter(fixture.truth.views[1]!.pose),
    R = fixture.truth.views[1]!.pose.R;
  for (const [x, y] of [
    [9, 10],
    [48, 36],
    [80, 60],
  ] as const) {
    const depth = frame.depthZ[y * frame.width + x]!;
    assert.ok(Number.isFinite(depth));
    const ray = [
      (x - frame.camera.cx) / frame.camera.fx,
      (y - frame.camera.cy) / frame.camera.fy,
      1,
    ];
    const world = Float64Array.from(
      [0, 1, 2].map(
        (axis) =>
          center[axis]! + depth * (R[axis]! * ray[0]! + R[3 + axis]! * ray[1]! + R[6 + axis]!),
      ),
    );
    const projected = projectPoint(frame.camera, fixture.truth.views[1]!.pose, world)!;
    assert.ok(Math.hypot(projected[0] - x, projected[1] - y) < 1e-10);
  }
  // The consumer receives only RGB and explicitly known calibration, never evaluator depth/pose.
  const input = {
    id: frame.id,
    width: frame.width,
    height: frame.height,
    rgba: frame.rgba,
    camera: frame.camera,
  };
  assert.ok(!('pose' in input) && !('depthZ' in input) && !('visiblePointIds' in input));
  assert.equal(fixture.provenance.mode, 'geometry-only');
});
