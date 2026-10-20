import test from 'node:test';
import assert from 'node:assert/strict';
import { cameraAxes, cameraPosition } from './src/lib/sfm-scene.ts';

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
