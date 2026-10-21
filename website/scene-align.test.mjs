// The ground plane helpers of the viewer's placement on the grid (website/src/lib/scene-align.ts).
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  dominantPlane,
  orientTowards,
  planeFromPoints,
  quantile,
  smallestEigenvector,
} from './src/lib/scene-align.ts';

const near = (actual, expected, tolerance, label) =>
  actual.forEach((value, i) =>
    assert.ok(
      Math.abs(value - expected[i]) <= tolerance,
      `${label}[${i}] ${value} vs ${expected[i]}`,
    ),
  );

test('three points give their plane; points on one line give none', () => {
  const plane = planeFromPoints([0, 0, 0], [1, 0, 0], [0, 0, 1]);
  near(plane.normal, [0, -1, 0], 1e-12, 'normal');
  assert.equal(planeFromPoints([0, 0, 0], [1, 1, 1], [2, 2, 2]), undefined);
  assert.equal(planeFromPoints([0, 0, 0], [0, 0, 0], [1, 0, 0]), undefined);
});

test('the normal turns toward the side that holds most cameras', () => {
  const plane = { normal: [0, -1, 0], point: [0, 0, 0] };
  const centers = [0, 5, 0, 1, 4, 1, 2, -1, 0];
  near(orientTowards(plane, centers).normal, [0, 1, 0], 0, 'normal');
  const kept = { normal: [0, 1, 0], point: [0, 0, 0] };
  assert.equal(orientTowards(kept, centers), kept);
});

test('the smallest eigenvector of a diagonal and of a rotated matrix', () => {
  near(smallestEigenvector([3, 0, 0, 0, 1, 0, 0, 0, 2]).map(Math.abs), [0, 1, 0], 1e-12, 'diag');
  // diag(1, 4, 9) turned 30 degrees about z: the smallest axis is (cos 30, sin 30, 0).
  const c = Math.cos(Math.PI / 6),
    s = Math.sin(Math.PI / 6);
  const R = [c, -s, 0, s, c, 0, 0, 0, 1];
  const D = [1, 4, 9];
  const m = Array.from({ length: 9 }, (_, k) => {
    const r = Math.floor(k / 3),
      col = k % 3;
    let sum = 0;
    for (let j = 0; j < 3; j++) sum += R[3 * r + j] * D[j] * R[3 * col + j];
    return sum;
  });
  near(smallestEigenvector(m).map(Math.abs), [c, s, 0], 1e-9, 'rotated');
});

test('a tilted ground with a pyramid and noise gives the ground plane', () => {
  // The ground z = 0.2 x + 0.1 y (20 x 20, 6400 points) with a pyramid of 1600 points on it and
  // 400 points of noise above: the dominant plane is the ground, not a face of the pyramid.
  let state = 7;
  const random = () => ((state = (state * 1103515245 + 12345) >>> 0) / 4294967296) * 2 - 1;
  const points = [];
  const ground = (x, y) => 0.2 * x + 0.1 * y;
  for (let i = 0; i < 6400; i++) {
    const x = 10 * random(),
      y = 10 * random();
    points.push(x, y, ground(x, y) + 0.005 * random());
  }
  for (let i = 0; i < 1600; i++) {
    const x = 2 * random(),
      y = 2 * random(),
      h = 3 - 1.5 * Math.max(Math.abs(x), Math.abs(y));
    points.push(x, y, ground(x, y) + h);
  }
  for (let i = 0; i < 400; i++) points.push(10 * random(), 10 * random(), 5 + 5 * random());
  const found = dominantPlane(points, points.length / 3);
  const expected = [-0.2, -0.1, 1].map((v) => v / Math.hypot(0.2, 0.1, 1));
  const sign = Math.sign(found.normal[2]);
  near(
    found.normal.map((v) => v * sign),
    expected,
    2e-3,
    'normal',
  );
  assert.ok(found.inlierShare > 0.7, 'inliers ' + found.inlierShare);
  // The same input gives the same plane (seeded).
  assert.deepEqual(dominantPlane(points, points.length / 3), found);
  assert.equal(dominantPlane([0, 0, 0, 1, 1, 1], 2), undefined);
});

test('quantile picks from a sorted copy', () => {
  assert.equal(quantile([5, 1, 3, 2, 4], 0), 1);
  assert.equal(quantile([5, 1, 3, 2, 4], 0.5), 3);
  assert.equal(quantile([5, 1, 3, 2, 4], 1), 5);
  assert.ok(Number.isNaN(quantile([], 0.5)));
});
