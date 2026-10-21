import test from 'node:test';
import assert from 'node:assert/strict';
import {
  GUIDE_RANSAC_ITERATIONS,
  guideEssential,
  guideEssentialOptions,
  preferGuided,
} from './src/lib/guided-matching.ts';

test('the guide is a short RANSAC on E with fixed draws, under the run’s own budget', () => {
  const a = { fx: 500, fy: 520 },
    b = { fx: 480, fy: 500 };
  const options = guideEssentialOptions(a, b, 4, 500, 108);
  assert.ok(GUIDE_RANSAC_ITERATIONS <= 50);
  assert.equal(options.iterations, GUIDE_RANSAC_ITERATIONS);
  assert.equal(options.minIterations, GUIDE_RANSAC_ITERATIONS);
  assert.equal(options.seed, 108);
  assert.equal(options.threshold, 4 / 480);
  // The guide asks for E only: no pose, homography or refinement option is set.
  assert.deepEqual(Object.keys(options).sort(), [
    'iterations',
    'minIterations',
    'seed',
    'threshold',
  ]);
  // A run with fewer RANSAC iterations than the guide's default caps the guide too.
  const capped = guideEssentialOptions(a, b, 4, 20, 1);
  assert.equal(capped.iterations, 20);
  assert.equal(capped.minIterations, 20);
});

test('the guide takes E from the estimate and passes the options through', async () => {
  const options = guideEssentialOptions({ fx: 500, fy: 500 }, { fx: 500, fy: 500 }, 4, 500, 7);
  const E = new Float64Array([0, 0, 0, 0, 0, -1, 0, 1, 0]);
  let seen;
  const got = await guideEssential(async (o) => ((seen = o), { E }), options);
  assert.equal(got, E);
  assert.equal(seen, options);
});

test('the guide gives no E when the RANSAC finds no model, and rethrows other errors', async () => {
  const options = guideEssentialOptions({ fx: 500, fy: 500 }, { fx: 500, fy: 500 }, 4, 500, 7);
  const none = new Error('Could not find a valid two-view geometry');
  none.name = 'GeometryEstimationError';
  assert.equal(await guideEssential(async () => Promise.reject(none), options), undefined);
  await assert.rejects(
    guideEssential(async () => Promise.reject(new RangeError('bad')), options),
    RangeError,
  );
});

test('the guided matches replace the plain ones only when they are more', () => {
  assert.equal(preferGuided(200, 260), true);
  assert.equal(preferGuided(200, 200), false);
  assert.equal(preferGuided(200, 150), false);
  assert.equal(preferGuided(200, undefined), false);
});
