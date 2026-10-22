import test from 'node:test';
import assert from 'node:assert/strict';
import {
  cameraAxes,
  cameraPosition,
  fitGrid,
  gridChanged,
  niceStep,
  previewSeeker,
  previousCamera,
  packLandmarks,
  unpackScene,
} from './src/lib/sfm-scene.ts';

/** Rotation about an axis by an angle in degrees, row-major. */
function rotation([x, y, z], degrees) {
  const a = (degrees * Math.PI) / 180,
    c = Math.cos(a),
    s = Math.sin(a),
    t = 1 - c;
  return Float64Array.from([
    t * x * x + c,
    t * x * y - s * z,
    t * x * z + s * y,
    t * x * y + s * z,
    t * y * y + c,
    t * y * z - s * x,
    t * x * z - s * y,
    t * y * z + s * x,
    t * z * z + c,
  ]);
}
const apply = (R, v) =>
  [0, 1, 2].map((i) => R[3 * i] * v[0] + R[3 * i + 1] * v[1] + R[3 * i + 2] * v[2]);
const close = (a, b) => a.every((value, i) => Math.abs(value - b[i]) < 1e-12);

test('camera axes are the camera frame in world coordinates: R maps each one to a unit axis', () => {
  // A camera turned 90 degrees about y and tilted 30 degrees about x, placed away from the origin.
  const R = rotation([0, 1, 0], 90);
  const tilt = rotation([1, 0, 0], 30);
  const combined = Float64Array.from(
    [0, 1, 2].flatMap((i) =>
      [0, 1, 2].map((j) => [0, 1, 2].reduce((s, k) => s + tilt[3 * i + k] * R[3 * k + j], 0)),
    ),
  );
  const center = [2, -1, 3];
  const t = apply(combined, center).map((v) => -v);
  const view = {
    id: 0,
    pose: { R: combined, t: Float64Array.from(t) },
    camera: { fx: 1, fy: 1, cx: 0, cy: 0 },
  };
  assert.ok(close(cameraPosition(view), center));
  const axes = cameraAxes(view);
  for (let k = 0; k < 3; k++) {
    const unit = [0, 0, 0];
    unit[k] = 1;
    assert.ok(close(apply(combined, axes[k]), unit), 'axis ' + k);
  }
  // The forward axis of a camera turned 90 degrees about y looks along world -x, not +z.
  const turned = { ...view, pose: { R, t: Float64Array.from([0, 0, 0]) } };
  assert.ok(close(cameraAxes(turned)[2], [-1, 0, 0]));
});

test('the preview seeker skips a seek while one is in flight and within its interval', () => {
  let clock = 0;
  const video = { seeking: false, currentTime: 0 };
  const seek = previewSeeker(video, 500, () => clock);
  assert.equal(seek(1), true);
  assert.equal(video.currentTime, 1);
  clock = 100;
  assert.equal(seek(2), false, 'inside the interval');
  assert.equal(video.currentTime, 1);
  clock = 600;
  video.seeking = true;
  assert.equal(seek(3), false, 'the previous seek is in flight');
  video.seeking = false;
  assert.equal(seek(3), true);
  assert.equal(video.currentTime, 3);
});

test('niceStep rounds up to 1, 2 or 5 times a power of ten', () => {
  assert.equal(niceStep(0.7), 1);
  assert.equal(niceStep(1.1), 2);
  assert.equal(niceStep(3), 5);
  assert.equal(niceStep(5.5), 10);
  assert.ok(Math.abs(niceStep(0.013) - 0.02) < 1e-12);
  assert.equal(niceStep(200), 200);
});

test('the grid fits the points, ignores outliers and uses a round cell', () => {
  const points = [];
  for (let i = 0; i < 1000; i++) points.push(((i % 40) - 20) * 0.5 + 100, Math.floor(i / 40) * 0.4);
  const base = fitGrid(points);
  assert.ok(base.divisions >= 8 && base.divisions <= 20, 'divisions ' + base.divisions);
  assert.equal(base.size, base.divisions * base.cell);
  const mantissa = base.cell / Math.pow(10, Math.floor(Math.log10(base.cell)));
  assert.ok(
    [1, 2, 5].some((m) => Math.abs(m - mantissa) < 1e-9),
    'cell ' + base.cell,
  );
  assert.ok(base.size >= 20 * 1.1, 'covers the points with a margin');
  assert.ok(Math.abs(base.x - 100) <= base.cell, 'centered on the points');
  // Ten far outliers (1 %) leave the fit unchanged.
  const withOutliers = points.concat(Array.from({ length: 20 }, (_, i) => (i % 2 ? 1e6 : -1e6)));
  assert.deepEqual(fitGrid(withOutliers), base);
  assert.equal(fitGrid([]), undefined);
  assert.equal(fitGrid([NaN, 1]), undefined);
});

test('the grid rebuilds only when the extent changes by more than a fifth', () => {
  const shown = { x: 0, z: 0, size: 10, cell: 1, divisions: 10 };
  assert.equal(gridChanged(undefined, shown), true);
  assert.equal(gridChanged(shown, { ...shown, size: 11 }), false);
  assert.equal(gridChanged(shown, { ...shown, size: 13 }), true);
  assert.equal(gridChanged(shown, { ...shown, size: 8 }), true);
  assert.equal(gridChanged(shown, { ...shown, x: 1 }), false);
  assert.equal(gridChanged(shown, { ...shown, x: 3 }), true);
});

test('a growing map packs its landmarks into flat arrays and unpacks to the viewer scene', () => {
  const landmarks = [
    {
      trackId: 7,
      xyz: new Float64Array([1, 2, 3]),
      color: [1, 0.5, 0],
      observations: [
        { viewId: 0, featureId: 1, pixel: [1, 2] },
        { viewId: 1, featureId: 2, pixel: [3, 4] },
      ],
    },
    { trackId: 9, xyz: new Float64Array([-1, 0, 4.5]), observations: [] },
  ];
  const packed = packLandmarks(landmarks);
  assert.equal(packed.count, 2);
  assert.equal(packed.observations, 2);
  assert.equal(packed.xyz.byteLength, 2 * 3 * 8);
  const views = [{ id: 0 }];
  const scene = unpackScene({ views, packed });
  assert.equal(scene.views, views);
  assert.equal(scene.observationCount, 2);
  assert.deepEqual(
    scene.landmarks.map((p) => p.trackId),
    [7, 9],
  );
  assert.deepEqual([...scene.landmarks[0].xyz], [1, 2, 3]);
  // The viewer rounds a color to a byte, so the byte survives the round trip.
  assert.equal(Math.round(scene.landmarks[0].color[1] * 255), 128);
  assert.deepEqual(
    scene.landmarks[1].color.map((c) => Math.round(c * 255)),
    [179, 179, 179],
  );
});

test('a final map splits into parts and the page assembles the same map', async () => {
  const { splitModel, ModelAssembler } = await import('./src/lib/model-chunks.ts');
  const landmarks = Array.from({ length: 10 }, (_, trackId) => ({
    trackId,
    xyz: [trackId, 0, 1],
    observations: [],
  }));
  const message = {
    type: 'model',
    component: 2,
    meanReprojectionErrorPx: 0.4,
    reconstruction: { views: [{ id: 5 }], landmarks },
  };
  assert.deepEqual(splitModel(message, 10), [message], 'a small map goes unchanged');
  assert.deepEqual(splitModel({ ...message, partial: true }, 10), [{ ...message, partial: true }]);
  const parts = splitModel(message, 4);
  assert.deepEqual(
    parts.map((p) => p.type),
    ['modelPart', 'modelPart', 'modelPart', 'model'],
  );
  assert.ok(parts[0].views && !parts[1].views);
  assert.equal(parts.at(-1).parts, 3);
  const assembler = new ModelAssembler();
  const seen = parts.map((p) => assembler.accept(p));
  assert.deepEqual(seen.slice(0, 3), [undefined, undefined, undefined]);
  assert.equal(seen[3].component, 2);
  assert.equal(seen[3].meanReprojectionErrorPx, 0.4);
  assert.deepEqual(seen[3].reconstruction, message.reconstruction);
  // A message of any other kind passes through.
  const other = { type: 'progress', detail: 'x' };
  assert.equal(assembler.accept(other), other);
});

test('a growing map with packed landmarks is never split', async () => {
  const { splitModel } = await import('./src/lib/model-chunks.ts');
  const message = {
    type: 'model',
    partial: true,
    component: 0,
    reconstruction: { views: [], packed: { count: 9 } },
  };
  assert.deepEqual(splitModel(message, 2), [message]);
});

test('camera motion stays within the selected video and never crosses a missing pose', () => {
  const a0 = { id: 10, sequence: 0, videoId: 'a' };
  const b0 = { id: 20, sequence: 0, videoId: 'b' };
  const b1 = { id: 21, sequence: 1, videoId: 'b' };
  const b3 = { id: 23, sequence: 3, videoId: 'b' };
  const scene = { views: [a0, b0, b1, b3], landmarks: [] };
  assert.equal(previousCamera(scene, b1), b0);
  assert.equal(previousCamera(scene, b0), undefined);
  assert.equal(previousCamera(scene, b3), undefined);
  assert.equal(previousCamera(scene, undefined), undefined);
  assert.equal(
    previousCamera({ views: [{ id: 0 }, { id: 2 }], landmarks: [] }, { id: 2 }),
    undefined,
  );
});
