import test from 'node:test';
import assert from 'node:assert/strict';
import {
  ALIVE,
  HostScene,
  MODEL_STRIDE,
  POSE_STRIDE,
  PIXEL_STRIDE,
} from '../src/reconstruction/scene.ts';
import { validateReconstruction } from '../src/reconstruction/bundle-adjustment.ts';
import { scene } from './scene.ts';

test('a scene round-trips a reconstruction: same views, landmarks and observations in order', () => {
  const { reconstruction } = scene({
    count: 50,
    views: 4,
    camera: { fx: 850, fy: 870, cx: 480, cy: 270, k1: -0.03, k2: 0.004 },
  });
  const host = HostScene.fromReconstruction(reconstruction);
  assert.equal(host.cameraCount, 4);
  assert.equal(host.pointCount, 50);
  assert.equal(host.observationCount, 200);
  assert.equal(host.cameraPoses.length, host.cameraCapacity * POSE_STRIDE);
  assert.equal(host.cameraModels.length, host.cameraCapacity * MODEL_STRIDE);
  assert.equal(host.obsPixel.length, host.observationCapacity * PIXEL_STRIDE);
  const copy = host.snapshot();
  assert.deepEqual(copy, reconstruction);
  validateReconstruction(copy);
  // The snapshot is a copy: writing the scene does not reach it, and the input is untouched.
  host.points[0] = 99;
  assert.equal(copy.landmarks[0]!.xyz[0], reconstruction.landmarks[0]!.xyz[0]);
  assert.notEqual(copy.landmarks[0]!.xyz[0], 99);
  for (const [index, view] of reconstruction.views.entries()) {
    assert.equal(host.cameraIndex(view.id), index);
    assert.deepEqual(host.pose(index), view.pose);
    assert.deepEqual(host.camera(index), view.camera);
  }
  for (const [index, point] of reconstruction.landmarks.entries())
    assert.equal(host.pointIndex(point.trackId), index);
});

test('cameras without distortion read back without k1 and k2, as the inputs spell them', () => {
  const host = new HostScene();
  const index = host.registerCamera(
    7,
    { fx: 500, fy: 500, cx: 320, cy: 240 },
    {
      R: Float64Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1),
      t: Float64Array.of(0, 0, 0),
    },
  );
  assert.deepEqual(host.camera(index), { fx: 500, fy: 500, cx: 320, cy: 240 });
  host.setCamera(index, { fx: 500, fy: 500, cx: 320, cy: 240, k1: -0.1 });
  assert.deepEqual(host.camera(index), { fx: 500, fy: 500, cx: 320, cy: 240, k1: -0.1 });
});

test('the arrays grow by doubling and keep their contents', () => {
  const host = new HostScene({ cameras: 1, points: 1, observations: 1 });
  const identity = { R: Float64Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1), t: new Float64Array(3) };
  for (let v = 0; v < 5; v++)
    host.registerCamera(
      v,
      { fx: 100 + v, fy: 100, cx: 50, cy: 50 },
      {
        R: identity.R,
        t: Float64Array.of(v, 0, 0),
      },
    );
  assert.equal(host.cameraCount, 5);
  assert.equal(host.cameraCapacity, 8);
  for (let p = 0; p < 9; p++) host.addPoint(p, [p, p + 0.5, 3]);
  assert.equal(host.pointCapacity, 16);
  let n = 0;
  for (let p = 0; p < 9; p++)
    for (let v = 0; v < 5; v++) host.addObservation(v, p, n++, p + v, p - v);
  assert.equal(host.observationCount, 45);
  assert.equal(host.observationCapacity, 64);
  assert.equal(host.cameraModels[4 * MODEL_STRIDE], 104);
  assert.equal(host.cameraPoses[4 * POSE_STRIDE + 9], 4);
  assert.deepEqual(host.point(8), Float64Array.of(8, 8.5, 3));
  assert.equal(host.obsPixel[44 * PIXEL_STRIDE], 8 + 4);
  assert.equal(host.obsFeature[44], 44);
  assert.ok(host.obsFlags.subarray(0, 45).every((f) => f & ALIVE));
  assert.equal(host.pointObservations[3], 5);
});

test('orders group the alive observations by point and by camera in arrival order', () => {
  const { reconstruction } = scene({ count: 20, views: 3 });
  const host = HostScene.fromReconstruction(reconstruction);
  host.removeObservation(4); // point 1, camera 1
  const byPoint = host.pointOrder(),
    byCamera = host.cameraOrder();
  assert.equal(byPoint.start.length, 21);
  assert.equal(byPoint.order.length, 59);
  assert.deepEqual([...byPoint.order.subarray(byPoint.start[1]!, byPoint.start[2]!)], [3, 5]);
  assert.equal(byCamera.start.length, 4);
  assert.equal(byCamera.start[2]! - byCamera.start[1]!, 19);
  assert.ok(!byCamera.order.includes(4));
  for (let k = 1; k < byCamera.order.length; k++) {
    const a = byCamera.order[k - 1]!,
      b = byCamera.order[k]!;
    if (host.obsCamera[a] === host.obsCamera[b]) assert.ok(a < b, 'arrival order within a camera');
  }
});

test('removing and compacting drops dead points and observations and renumbers the rest', () => {
  const { reconstruction } = scene({ count: 10, views: 3 });
  const host = HostScene.fromReconstruction(reconstruction);
  host.removeObservation(0); // point 0, camera 0
  host.removeObservation(0); // idempotent
  assert.equal(host.pointObservations[0], 2);
  host.removePoint(4);
  assert.equal(host.pointIndex(4), undefined);
  assert.equal(host.pointAlive(4), false);
  assert.throws(() => host.addObservation(0, 4, 1, 0, 0), RangeError);
  // The track of a dead point may get a new point before compaction.
  const replacement = host.addPoint(4, [1, 2, 3]);
  assert.equal(replacement, 10);
  host.addObservation(2, replacement, 4, 7, 8);
  const removed = host.compact();
  assert.deepEqual(removed, { pointsRemoved: 1, observationsRemoved: 4 });
  assert.equal(host.pointCount, 10);
  assert.equal(host.observationCount, 27);
  assert.equal(host.pointIndex(4), 9);
  assert.equal(host.pointIndex(5), 4);
  assert.equal(host.pointIndex(9), 8);
  const model = host.snapshot();
  assert.deepEqual(
    model.landmarks.map((p) => p.trackId),
    [0, 1, 2, 3, 5, 6, 7, 8, 9, 4],
  );
  assert.equal(model.landmarks[0]!.observations.length, 2);
  assert.deepEqual(model.landmarks[9], {
    trackId: 4,
    xyz: Float64Array.of(1, 2, 3),
    observations: [{ viewId: 30, featureId: 4, pixel: [7, 8] }],
  });
  for (let i = 0; i < host.observationCount; i++) {
    assert.ok(host.obsFlags[i]! & ALIVE);
    assert.ok(host.pointAlive(host.obsPoint[i]!));
  }
  assert.ok(host.pointFlags.subarray(0, 10).every((f) => f & ALIVE));
  // Compacting a clean scene changes nothing.
  assert.deepEqual(host.compact(), { pointsRemoved: 0, observationsRemoved: 0 });
  assert.deepEqual(host.snapshot(), model);
});

test('a scene refuses invalid cameras, poses, points, observations and indices', () => {
  const host = new HostScene();
  const identity = { R: Float64Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1), t: new Float64Array(3) };
  assert.throws(() => new HostScene({ points: 0 }), RangeError);
  assert.throws(
    () => host.registerCamera(-1, { fx: 1, fy: 1, cx: 0, cy: 0 }, identity),
    RangeError,
  );
  assert.throws(() => host.registerCamera(0, { fx: 0, fy: 1, cx: 0, cy: 0 }, identity), RangeError);
  assert.throws(
    () =>
      host.registerCamera(
        0,
        { fx: 1, fy: 1, cx: 0, cy: 0 },
        { R: identity.R.map((v) => 2 * v), t: identity.t },
      ),
    RangeError,
  );
  host.registerCamera(0, { fx: 1, fy: 1, cx: 0, cy: 0 }, identity);
  assert.throws(() => host.registerCamera(0, { fx: 1, fy: 1, cx: 0, cy: 0 }, identity), RangeError);
  assert.throws(() => host.pose(1), RangeError);
  assert.throws(() => host.addPoint(0, [1, 2]), RangeError);
  assert.throws(() => host.addPoint(0, [1, 2, NaN]), RangeError);
  host.addPoint(0, [1, 2, 3]);
  assert.throws(() => host.addPoint(0, [1, 2, 3]), RangeError);
  assert.throws(() => host.addObservation(1, 0, 0, 0, 0), RangeError);
  assert.throws(() => host.addObservation(0, 1, 0, 0, 0), RangeError);
  assert.throws(() => host.addObservation(0, 0, -1, 0, 0), RangeError);
  assert.throws(() => host.addObservation(0, 0, 0, Infinity, 0), RangeError);
  assert.throws(() => host.removeObservation(0), RangeError);
  assert.throws(
    () =>
      HostScene.fromReconstruction({
        views: [],
        landmarks: [
          {
            trackId: 0,
            xyz: Float64Array.of(0, 0, 1),
            observations: [{ viewId: 3, featureId: 0, pixel: [0, 0] }],
          },
        ],
      }),
    RangeError,
  );
});
