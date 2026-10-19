import test from 'node:test';
import assert from 'node:assert/strict';
import { alignPoses, cameraCenter } from './src/lib/pose-alignment.ts';

/** Rotation about an axis by an angle in degrees, row-major. */
function rotation(axis, degrees) {
  const [x, y, z] = axis,
    a = (degrees * Math.PI) / 180,
    c = Math.cos(a),
    s = Math.sin(a),
    t = 1 - c;
  return [
    t * x * x + c,
    t * x * y - s * z,
    t * x * z + s * y,
    t * x * y + s * z,
    t * y * y + c,
    t * y * z - s * x,
    t * x * z - s * y,
    t * y * z + s * x,
    t * z * z + c,
  ];
}
const multiply = (a, b) => {
  const out = new Array(9);
  for (let r = 0; r < 3; r++)
    for (let c = 0; c < 3; c++)
      out[r * 3 + c] = a[r * 3] * b[c] + a[r * 3 + 1] * b[3 + c] + a[r * 3 + 2] * b[6 + c];
  return out;
};
const apply = (m, v) => [
  m[0] * v[0] + m[1] * v[1] + m[2] * v[2],
  m[3] * v[0] + m[4] * v[1] + m[5] * v[2],
  m[6] * v[0] + m[7] * v[1] + m[8] * v[2],
];
/** A world-to-camera pose from a camera orientation (camera-to-world) and a center. */
function poseAt(id, orientation, center) {
  const R = [
    orientation[0],
    orientation[3],
    orientation[6],
    orientation[1],
    orientation[4],
    orientation[7],
    orientation[2],
    orientation[5],
    orientation[8],
  ];
  const rc = apply(R, center);
  return { id, R, t: [-rc[0], -rc[1], -rc[2]] };
}
/** Six cameras on an arc looking inward: a trajectory with extent in three directions. */
function referencePoses() {
  const poses = [];
  for (let i = 0; i < 6; i++) {
    const angle = i * 25;
    const center = [
      4 * Math.sin((angle * Math.PI) / 180),
      0.3 * i,
      4 * Math.cos((angle * Math.PI) / 180),
    ];
    poses.push(
      poseAt(i, multiply(rotation([0, 1, 0], angle), rotation([1, 0, 0], 10 * (i % 2))), center),
    );
  }
  return poses;
}
/** The same cameras in another world frame: rotated, scaled and shifted. */
function transformed(poses, Q, scale, shift) {
  return poses.map((pose) => {
    const center = cameraCenter(pose);
    const moved = apply(Q, center).map((value, i) => scale * value + shift[i]);
    const orientation = multiply(Q, [
      pose.R[0],
      pose.R[3],
      pose.R[6],
      pose.R[1],
      pose.R[4],
      pose.R[7],
      pose.R[2],
      pose.R[5],
      pose.R[8],
    ]);
    return poseAt(pose.id, orientation, moved);
  });
}

test('cameraCenter inverts the world-to-camera pose', () => {
  const pose = poseAt(0, rotation([0, 1, 0], 40), [1, 2, 3]);
  const center = cameraCenter(pose);
  assert.ok(Math.hypot(center[0] - 1, center[1] - 2, center[2] - 3) < 1e-12);
});

test('a rotated, scaled and shifted copy aligns with zero error', () => {
  const reference = referencePoses();
  const source = transformed(
    reference,
    rotation(
      [0.3, 1, -0.2].map((v) => v / Math.hypot(0.3, 1, 0.2)),
      70,
    ),
    0.35,
    [2, -1, 5],
  );
  const result = alignPoses(source, reference);
  assert.equal(result.status, 'aligned');
  assert.equal(result.commonViews, 6);
  assert.ok(Math.abs(result.scale - 1 / 0.35) < 1e-9, String(result.scale));
  assert.ok(result.rotationDeg.max < 1e-5, String(result.rotationDeg.max));
  assert.ok(result.centerOfRadius.max < 1e-9, String(result.centerOfRadius.max));
});

test('a perturbed view is measured and only common views count', () => {
  const reference = referencePoses();
  const source = transformed(reference, rotation([0, 0, 1], 15), 2, [0, 0, 0]);
  const wrong = poseAt(
    3,
    multiply(rotation([1, 0, 0], 5), [
      source[3].R[0],
      source[3].R[3],
      source[3].R[6],
      source[3].R[1],
      source[3].R[4],
      source[3].R[7],
      source[3].R[2],
      source[3].R[5],
      source[3].R[8],
    ]),
    cameraCenter(source[3]),
  );
  const extra = poseAt(99, rotation([0, 1, 0], 1), [0, 0, 0]);
  const result = alignPoses([...source.slice(0, 3), wrong, ...source.slice(4), extra], reference);
  assert.equal(result.status, 'aligned');
  assert.equal(result.commonViews, 6);
  const view = result.perView.find((entry) => entry.id === 3);
  assert.ok(Math.abs(view.rotationDeg - 5) < 1.5, String(view.rotationDeg));
  assert.ok(result.rotationDeg.median < 1.5, String(result.rotationDeg.median));
  assert.ok(result.perView.every((entry) => entry.id !== 99));
});

test('fewer than three common views or a degenerate trajectory is reported, not thrown', () => {
  const reference = referencePoses();
  assert.equal(
    alignPoses(reference.slice(0, 2), reference).status,
    'fewer than three common cameras',
  );
  assert.equal(alignPoses([], reference).commonViews, 0);
  const stacked = reference.map((pose) => poseAt(pose.id, rotation([0, 1, 0], 0), [1, 1, 1]));
  assert.equal(alignPoses(stacked, reference).status, 'source trajectory has no extent');
  assert.ok(Number.isNaN(alignPoses(stacked, reference).rotationDeg.median));
});
