import test from 'node:test';
import assert from 'node:assert/strict';
import {
  Change,
  QUIET_MS,
  addedViews,
  freshness,
  landmarkChanges,
  pulseTarget,
} from './src/lib/live-view.ts';

const point = (trackId, x, y, z) => ({
  trackId,
  xyz: Float64Array.of(x, y, z),
  observations: [],
});

test('landmarkChanges sorts each track into added, moved and kept, and lists the dropped', () => {
  const first = landmarkChanges(new Map(), [point(1, 0, 0, 1), point(2, 1, 0, 1)], 1e-4);
  assert.deepEqual([...first.changes], [Change.Added, Change.Added]);
  assert.equal(first.added, 2);
  assert.equal(first.removed.length, 0);
  const second = landmarkChanges(
    first.index,
    [point(1, 0, 0, 1.00001), point(3, 2, 0, 1), point(4, NaN, 0, 1), point(2, 1, 0.5, 1)],
    1e-4,
  );
  assert.deepEqual([...second.changes], [Change.Kept, Change.Added, Change.Skipped, Change.Moved]);
  assert.equal(second.added, 1);
  assert.equal(second.moved, 1);
  assert.deepEqual([...second.removed], []);
  const third = landmarkChanges(second.index, [point(3, 2, 0, 1)], 1e-4);
  assert.deepEqual([...third.changes], [Change.Kept]);
  // Tracks 1 and 2 are gone; a skipped track is not in the index, so it is not dropped again.
  assert.deepEqual([...third.removed], [0, 0, 1.00001, 1, 0.5, 1].map(Math.fround));
});

test('addedViews names the views the previous map did not hold', () => {
  const view = (id) => ({ id });
  assert.deepEqual(
    [...addedViews(new Set([1, 2]), { views: [1, 2, 5].map(view), landmarks: [] })],
    [5],
  );
  assert.deepEqual([...addedViews(new Set(), { views: [view(7)], landmarks: [] })], [7]);
});

test('pulseTarget maps each worker stage to the cameras its work touches', () => {
  assert.equal(pulseTarget('bundle · bundleStep (start)'), 'all');
  assert.equal(pulseTarget('final #3'), 'all');
  assert.equal(pulseTarget('pnp · p3p (end)'), 'latest');
  assert.equal(pulseTarget('verifyPair'), 'latest');
  assert.equal(pulseTarget('triangulation'), 'latest');
  assert.equal(pulseTarget('features · sift (start)'), 'none');
  assert.equal(pulseTarget(''), 'none');
});

test('freshness says when the last update came, and how long the GPU has been quiet', () => {
  assert.deepEqual(freshness(1000, 3400, false), {
    quiet: false,
    text: 'Working · updated 2 s ago',
  });
  assert.equal(freshness(1000, 3400, true).text, '작업 중 · 2초 전 갱신');
  const quiet = freshness(0, QUIET_MS + 2500, true);
  assert.equal(quiet.quiet, true);
  assert.equal(quiet.text, 'GPU 작업 중 · 7초째 새 소식 없음');
  // A clock read before the event never shows a negative age.
  assert.equal(freshness(5000, 4000, false).text, 'Working · updated 0 s ago');
});
