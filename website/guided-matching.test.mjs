import test from 'node:test';
import assert from 'node:assert/strict';
import {
  GUIDE_RANSAC_ITERATIONS,
  essentialFromPose,
  guideVerificationOptions,
  preferGuided,
} from './src/lib/guided-matching.ts';

test('the guide is a short RANSAC with no homography test, under the run’s own budget', () => {
  const a = { fx: 500, fy: 520 },
    b = { fx: 480, fy: 500 };
  const options = guideVerificationOptions(a, b, 4, 500, 108);
  assert.ok(GUIDE_RANSAC_ITERATIONS <= 50);
  assert.equal(options.iterations, GUIDE_RANSAC_ITERATIONS);
  assert.equal(options.homographyIterations, 1);
  assert.equal(options.refinementIterations, 0);
  assert.equal(options.seed, 108);
  assert.equal(options.threshold, 4 / 480);
  assert.equal(options.maxReprojectionErrorPx, 4);
  assert.equal(options.homographyThresholdPx, 4);
  // A run with fewer RANSAC iterations than the guide's default caps the guide too.
  assert.equal(guideVerificationOptions(a, b, 4, 20, 1).iterations, 20);
});

test('the guide’s essential matrix holds x2ᵀ E x1 = 0 for X2 = R X1 + t', () => {
  const angle = 0.2,
    c = Math.cos(angle),
    s = Math.sin(angle);
  const R = [c, 0, s, 0, 1, 0, -s, 0, c],
    t = [0.3, -0.1, 0.05];
  const E = essentialFromPose(R, t);
  for (const X1 of [
    [0.1, 0.2, 4],
    [-1, 0.5, 6],
    [0.7, -0.4, 3],
  ]) {
    const X2 = [0, 1, 2].map(
      (r) => R[r * 3] * X1[0] + R[r * 3 + 1] * X1[1] + R[r * 3 + 2] * X1[2] + t[r],
    );
    const x1 = [X1[0] / X1[2], X1[1] / X1[2], 1],
      x2 = [X2[0] / X2[2], X2[1] / X2[2], 1];
    let residual = 0;
    for (let r = 0; r < 3; r++)
      for (let k = 0; k < 3; k++) residual += x2[r] * E[r * 3 + k] * x1[k];
    assert.ok(Math.abs(residual) < 1e-12, String(residual));
  }
});

test('the guided matches replace the plain ones only when they are more', () => {
  assert.equal(preferGuided(200, 260), true);
  assert.equal(preferGuided(200, 200), false);
  assert.equal(preferGuided(200, 150), false);
  assert.equal(preferGuided(200, undefined), false);
});
