import test from 'node:test';
import assert from 'node:assert/strict';
import { p3p } from '../src/geometry/p3p.ts';
import { transformPoint, rotationExp } from '../src/geometry/pose.ts';
import { seededRandom } from '../src/geometry/numerics.ts';

test('Grunert P3P includes the true positive-depth pose across scales and scene samples', () => {
  const rng = seededRandom(8642);
  for (let i = 0; i < 100; i++) {
    const scale = [0.01, 1, 100][i % 3]!;
    const world = Array.from({ length: 3 }, () =>
      Float64Array.of(
        (rng() - 0.5) * 3 * scale,
        (rng() - 0.5) * 2 * scale,
        (4 + rng() * 5) * scale,
      ),
    );
    const truth = {
      R: rotationExp([0.03, -0.09, 0.02]),
      t: Float64Array.of(0.3 * scale, -0.02 * scale, 0.1 * scale),
    };
    const rays = world.map((X) => transformPoint(truth, X));
    const poses = p3p(world, rays);
    assert.ok(poses.length >= 1, `no pose ${i}`);
    const best = Math.min(
      ...poses.map((p) => Math.hypot(...p.t.map((v, j) => (v - truth.t[j]!) / scale))),
    );
    assert.ok(best < 1e-4, `sample ${i} error ${best}`);
  }
});
test('P3P rejects collinear world samples and repeated rays safely', () => {
  assert.deepEqual(
    p3p(
      [Float64Array.of(0, 0, 5), Float64Array.of(1, 0, 5), Float64Array.of(2, 0, 5)],
      Array.from({ length: 3 }, () => Float64Array.of(0, 0, 1)),
    ),
    [],
  );
});

test('symmetric P3P samples recover the true branch without splitting repeated roots', () => {
  const world = [
    [-1, 0, 5],
    [1, 0, 5],
    [0, Math.sqrt(3), 5],
  ].map((p) => Float64Array.from(p));
  const poses = p3p(world, world);
  assert.ok(poses.length >= 1 && poses.length <= 4);
  assert.ok(poses.some((p) => Math.hypot(...p.t) < 1e-8));
  const narrow = [
    [0.01, 0, 100],
    [-0.01, 0, 100],
    [0, 0.01, 100],
  ].map((p) => Float64Array.from(p));
  assert.deepEqual(p3p(narrow, narrow), []);
});
