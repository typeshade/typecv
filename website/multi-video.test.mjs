// Verifies: docs/multi-video.md 2.1 (duplicates), 2.2 (the cap), 6.3 (tags, matrix, colors).
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  COMPONENT_COLORS,
  componentColor,
  inlierMatrix,
  markDuplicates,
  matrixShade,
  multiVideoOptions,
  multiVideoQuery,
  tagsOf,
  thinnedPositions,
} from './src/lib/multi-video.ts';

test('the query: ?canonical=0, ?maxKeyframes= and ?maxFeatures=', () => {
  assert.deepEqual(multiVideoQuery(''), {
    canonical: true,
    maxKeyframesPerVideo: 60,
    maxFeatures: undefined,
  });
  assert.equal(multiVideoQuery('?canonical=0').canonical, false);
  assert.equal(multiVideoQuery('?canonical=1').canonical, true);
  assert.equal(multiVideoQuery('?maxKeyframes=8&maxFeatures=1024').maxKeyframesPerVideo, 8);
  assert.equal(multiVideoQuery('?maxKeyframes=8&maxFeatures=1024').maxFeatures, 1024);
  // A value out of range or not an integer falls back to the default.
  assert.equal(multiVideoQuery('?maxKeyframes=1').maxKeyframesPerVideo, 60);
  assert.equal(multiVideoQuery('?maxKeyframes=x').maxKeyframesPerVideo, 60);
  assert.equal(multiVideoQuery('?maxFeatures=3').maxFeatures, undefined);
});

test('the cap keeps the first and the last keyframe and thins the rest evenly', () => {
  assert.deepEqual(thinnedPositions(5, 60), [0, 1, 2, 3, 4]);
  assert.deepEqual(thinnedPositions(11, 3), [0, 5, 10]);
  const kept = thinnedPositions(100, 60);
  assert.equal(kept.length, 60);
  assert.equal(kept[0], 0);
  assert.equal(kept.at(-1), 99);
  assert.deepEqual(
    kept,
    [...new Set(kept)].sort((a, b) => a - b),
  );
  assert.throws(() => thinnedPositions(5, 1));
});

test('a second file with one videoId is a duplicate of the first', () => {
  const rows = markDuplicates([{ videoId: 'a' }, { videoId: 'b' }, { videoId: 'a' }]);
  assert.deepEqual(
    rows.map((r) => r.duplicateOf),
    [undefined, undefined, 'a'],
  );
});

test('tags: connected, alone, not connected and rotation', () => {
  const v = (tag) => ({ videoId: 'x', tag, component: 0, share: 1 });
  const none = { edges: 0, seedEligible: 0 };
  assert.deepEqual(tagsOf(v('connected'), none), ['component']);
  assert.deepEqual(tagsOf(v('alone'), none), ['alone']);
  assert.deepEqual(tagsOf(v('not connected'), none), ['not connected']);
  assert.deepEqual(tagsOf(v('connected'), { edges: 5, seedEligible: 0 }), [
    'component',
    'rotation',
  ]);
  assert.deepEqual(tagsOf(v('connected'), { edges: 5, seedEligible: 1 }), ['component']);
  assert.deepEqual(tagsOf(v('connected'), { edges: 2, seedEligible: 0 }), ['component']);
  assert.deepEqual(tagsOf(undefined, none), []);
});

test('the matrix is symmetric, has a zero diagonal and shades by the largest cell', () => {
  const m = inlierMatrix(
    ['a', 'b', 'c'],
    [
      { a: 'a', b: 'b', edges: 2, inliers: 70 },
      { a: 'b', b: 'a', edges: 1, inliers: 30 },
      { a: 'a', b: 'z', edges: 1, inliers: 9 },
    ],
  );
  assert.deepEqual(m, [
    [0, 100, 0],
    [100, 0, 0],
    [0, 0, 0],
  ]);
  assert.equal(matrixShade(100, 100), 1);
  assert.equal(matrixShade(0, 100), 0);
  assert.equal(matrixShade(1, 100), 0.12);
  assert.equal(matrixShade(5, 0), 0);
});

test('component colors repeat after the palette and are never undefined', () => {
  assert.equal(componentColor(0), COMPONENT_COLORS[0]);
  assert.equal(componentColor(COMPONENT_COLORS.length), COMPONENT_COLORS[0]);
  assert.ok(componentColor(-1));
});

test('options.multiVideo names the defaults, the switch and the shared focal', () => {
  const o = multiVideoOptions({ query: multiVideoQuery('?canonical=0&maxKeyframes=8') });
  assert.deepEqual(o, {
    descriptor: 'vlad-32-v1',
    k: 10,
    minMatches: 30,
    minEdgeInliers: 25,
    maxKeyframesPerVideo: 8,
    growth: 'incremental',
    canonical: false,
    focalPerVideo: false,
  });
});
