import test from 'node:test';
import assert from 'node:assert/strict';
import {
  cameraProjection,
  unprojectPixel,
  type Camera,
  type Pixel,
} from '../src/geometry/camera.ts';
import { projectPoint, rotationExp, transformPoint } from '../src/geometry/pose.ts';
import { multiply, transpose } from '../src/sfm/linear.ts';
import { createObservationSurfacePredictor } from '../src/gpu/observation-surface.ts';
import type { Landmark, RegisteredView } from '../src/reconstruction/types.ts';

function fixture(
  offsets = Array.from({ length: 12 }, (_, i) => {
    const angle = (2 * Math.PI * i) / 12;
    return [24 * Math.cos(angle), 24 * Math.sin(angle)];
  }),
) {
  const camera: Camera = { fx: 400, fy: 410, cx: 320, cy: 240, k1: -0.07, k2: 0.01 },
    from: RegisteredView = {
      id: 3,
      camera,
      pose: { R: rotationExp([0.05, 0.03, -0.02]), t: Float64Array.of(0.1, -0.2, 0.3) },
    },
    relative = rotationExp([0.04, -0.06, 0.02]),
    translation = [0.5, -0.1, -1],
    to: RegisteredView = {
      id: 8,
      camera: { ...camera, fx: 500, fy: 510, k1: -0.03 },
      pose: {
        R: multiply(relative, from.pose.R, 3),
        t: Float64Array.from(
          { length: 3 },
          (_, i) =>
            relative[3 * i]! * from.pose.t[0]! +
            relative[3 * i + 1]! * from.pose.t[1]! +
            relative[3 * i + 2]! * from.pose.t[2]! +
            translation[i]!,
        ),
      },
    },
    pixel: Pixel = [326, 236],
    world = (p: Pixel, depthMultiplier = 1): Float64Array => {
      const ray = unprojectPixel(camera, p)!,
        z = depthMultiplier / (0.2 + 0.07 * ray[0] - 0.04 * ray[1]),
        q = [ray[0] * z - from.pose.t[0]!, ray[1] * z - from.pose.t[1]!, z - from.pose.t[2]!],
        inverseR = transpose(from.pose.R, 3);
      return Float64Array.from(
        { length: 3 },
        (_, j) =>
          inverseR[3 * j]! * q[0]! + inverseR[3 * j + 1]! * q[1]! + inverseR[3 * j + 2]! * q[2]!,
      );
    },
    landmarks: Landmark[] = offsets.map(([dx, dy], featureId) => {
      const p: Pixel = [pixel[0] + dx!, pixel[1] + dy!];
      return {
        trackId: featureId,
        xyz: world(p),
        observations: [{ viewId: from.id, featureId, pixel: p }],
      };
    });
  return { from, to, relative, pixel, world, landmarks };
}

test('surface proposals include translated slanted-plane scale and shear with radial cameras', () => {
  const f = fixture(),
    predicted = createObservationSurfacePredictor(
      f.from,
      f.to,
      f.landmarks,
    )(f.pixel, f.world(f.pixel));
  assert.ok(predicted);
  assert.equal(predicted.support, 12);
  assert.ok(predicted.maxRelativeDepthResidual < 1e-10);
  const truth = (x: number, y: number) => projectPoint(f.to.camera, f.to.pose, f.world([x, y]))!,
    centre = truth(...f.pixel),
    left = truth(f.pixel[0] - 0.5, f.pixel[1]),
    right = truth(f.pixel[0] + 0.5, f.pixel[1]),
    top = truth(f.pixel[0], f.pixel[1] - 0.5),
    bottom = truth(f.pixel[0], f.pixel[1] + 0.5),
    expected = [right[0] - left[0], bottom[0] - top[0], right[1] - left[1], bottom[1] - top[1]];
  assert.ok(Math.hypot(predicted.pixel[0] - centre[0], predicted.pixel[1] - centre[1]) < 1e-9);
  assert.ok(Math.max(...predicted.affine.map((v, i) => Math.abs(v - expected[i]!))) < 1e-9);
  const rotated = (x: number, y: number) => {
    const ray = unprojectPixel(f.from.camera, [x, y])!;
    return cameraProjection(
      f.to.camera,
      [0, 1, 2].map(
        (i) =>
          f.relative[3 * i]! * ray[0] + f.relative[3 * i + 1]! * ray[1] + f.relative[3 * i + 2]!,
      ),
    )!.pixel;
  };
  const oldLeft = rotated(f.pixel[0] - 0.5, f.pixel[1]),
    oldRight = rotated(f.pixel[0] + 0.5, f.pixel[1]);
  assert.ok(
    Math.abs(oldRight[0] - oldLeft[0] - predicted.affine[0]!) > 0.2,
    'rotation-only warp should miss the translated surface magnification',
  );
});

test('a plane proposal can guide an untriangulated image seed without claiming a depth observation', () => {
  const f = fixture(),
    prediction = createObservationSurfacePredictor(f.from, f.to, f.landmarks)(f.pixel),
    truth = projectPoint(f.to.camera, f.to.pose, f.world(f.pixel))!;
  assert.ok(prediction);
  assert.ok(Math.hypot(prediction.pixel[0] - truth[0], prediction.pixel[1] - truth[1]) < 1e-9);
  assert.deepEqual(Object.keys(prediction).sort(), [
    'affine',
    'maxRelativeDepthResidual',
    'pixel',
    'support',
  ]);
});

test('surface fits defer sparse, collinear, one-sided and surrounding-but-ill-conditioned support', () => {
  const cases = [
    [
      [-20, -20],
      [20, -20],
      [-20, 20],
      [20, 20],
    ],
    Array.from({ length: 8 }, (_, i) => [-28 + 8 * i, -28 + 8 * i]),
    Array.from({ length: 8 }, (_, i) => [12 + (i % 2) * 8, -24 + 8 * Math.floor(i / 2)]),
    Array.from({ length: 8 }, (_, i) => [-28 + 8 * i, -28 + 8 * i + (i % 2 ? 0.05 : -0.05)]),
  ];
  for (const offsets of cases) {
    const f = fixture(offsets);
    assert.equal(createObservationSurfacePredictor(f.from, f.to, f.landmarks)(f.pixel), undefined);
  }
});

test('surface support must contain the full source patch and cannot project behind the target', () => {
  const f = fixture();
  assert.ok(createObservationSurfacePredictor(f.from, f.to, f.landmarks, 5)(f.pixel));
  assert.equal(
    createObservationSurfacePredictor(f.from, f.to, f.landmarks, 18)(f.pixel),
    undefined,
  );
  const behind = {
    ...f.to,
    pose: { ...f.to.pose, t: f.to.pose.t.map((v, i) => v - (i === 2 ? 10 : 0)) },
  };
  assert.equal(createObservationSurfacePredictor(f.from, behind, f.landmarks)(f.pixel), undefined);
  assert.throws(
    () => createObservationSurfacePredictor(f.from, f.to, f.landmarks, NaN),
    RangeError,
  );
});

test('surface fitting rejects depth boundaries, a single off-plane neighbour and a contradictory finite seed', () => {
  const f = fixture(),
    changed = (indices: number[]) =>
      f.landmarks.map((l, i) => ({
        ...l,
        xyz: indices.includes(i) ? f.world(l.observations[0]!.pixel, 1.3) : l.xyz,
      }));
  assert.equal(
    createObservationSurfacePredictor(f.from, f.to, changed([0, 1, 2, 3, 4, 5]))(f.pixel),
    undefined,
  );
  assert.equal(createObservationSurfacePredictor(f.from, f.to, changed([0]))(f.pixel), undefined);
  assert.equal(
    createObservationSurfacePredictor(f.from, f.to, f.landmarks)(f.pixel, f.world(f.pixel, 1.3)),
    undefined,
  );
});

test('query landmark is held out of the fit and source-map reprojection failures are ignored', () => {
  const f = fixture(),
    badCentre: Landmark = {
      trackId: 100,
      xyz: f.world(f.pixel, 1.4),
      observations: [{ viewId: f.from.id, featureId: 100, pixel: f.pixel }],
    },
    held = createObservationSurfacePredictor(f.from, f.to, [...f.landmarks, badCentre])(f.pixel);
  assert.ok(held);
  assert.equal(held.support, 12);
  const incorrectPixels = f.landmarks.map((l) => ({
    ...l,
    observations: l.observations.map((o) => ({
      ...o,
      pixel: [o.pixel[0] + 10, o.pixel[1]] as Pixel,
    })),
  }));
  assert.equal(
    createObservationSurfacePredictor(f.from, f.to, incorrectPixels)(f.pixel),
    undefined,
  );
});

test('surface prediction uses the map gauge consistently and never mutates its accepted input', () => {
  const f = fixture(),
    before = JSON.stringify(f),
    expected = createObservationSurfacePredictor(f.from, f.to, f.landmarks)(f.pixel)!;
  for (const scale of [0.001, 1000]) {
    const scaledView = (view: RegisteredView): RegisteredView => ({
        ...view,
        pose: { ...view.pose, t: view.pose.t.map((v) => v * scale) },
      }),
      points = f.landmarks.map((l) => ({ ...l, xyz: l.xyz.map((v) => v * scale) })),
      actual = createObservationSurfacePredictor(
        scaledView(f.from),
        scaledView(f.to),
        points,
      )(
        f.pixel,
        f.world(f.pixel).map((v) => v * scale),
      );
    assert.ok(actual);
    assert.ok(
      Math.hypot(actual.pixel[0] - expected.pixel[0], actual.pixel[1] - expected.pixel[1]) < 1e-8,
    );
    assert.ok(Math.max(...actual.affine.map((v, i) => Math.abs(v - expected.affine[i]!))) < 1e-8);
    assert.ok(transformPoint(scaledView(f.from).pose, points[0]!.xyz)[2]! > 0);
  }
  assert.equal(JSON.stringify(f), before);
});
