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
  partialVideos,
  tagsOf,
  thinnedPositions,
} from './src/lib/multi-video.ts';
import { runRecordProblem } from './src/lib/run-record.ts';
import { stageTimings, newBoard, advance, tick } from './src/lib/multi-video-stages.ts';
import { sampleRecord } from './record-fixture.mjs';

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

test('a failed run lists its videos, the views of its maps and its stage times', () => {
  const rows = [
    { videoId: '0123456789abcdef', sampleId: 'a', durationSeconds: 60, focal: 0.8, keyframes: 12 },
    { videoId: 'fedcba9876543210', durationSeconds: 30, focal: 0.7, keyframes: 3 },
    { sampleId: 'not started', durationSeconds: 5, focal: 0.8 },
  ];
  const facts = new Map([
    ['0123456789abcdef', { frames: 100, width: 640, height: 360, keyframes: 10 }],
    ['fedcba9876543210', { frames: 20, width: 640, height: 360 }],
  ]);
  const scenes = new Map([
    [0, { views: [{ videoId: '0123456789abcdef' }, { videoId: '0123456789abcdef' }] }],
    [1, { views: [{ videoId: 'fedcba9876543210' }] }],
  ]);
  const videos = partialVideos({ rows, facts, scenes });
  assert.equal(videos.length, 2);
  assert.deepEqual(
    { ...videos[0], tags: undefined },
    {
      videoId: '0123456789abcdef',
      label: 'a',
      durationSeconds: 60,
      keyframes: 10,
      frames: 100,
      width: 640,
      height: 360,
      registeredViews: 2,
      focalRatio: 0.8,
      component: 0,
      tags: undefined,
    },
  );
  assert.equal(videos[1].label, null);
  assert.equal(videos[1].keyframes, 3);
  assert.equal(videos[1].component, 1);

  let board = newBoard(0);
  board = advance(board, { stage: 'frames', done: 5, total: 40 }, 100);
  board = advance(board, { stage: 'reconstruction', done: 1, total: 9 }, 5100);
  board = tick(board, 9100);
  const stages = stageTimings(board);
  assert.equal(stages.frames.elapsedMs, 5000);
  assert.equal(stages.reconstruction.elapsedMs, 4000);
  // A stage that the run passed with no event of its own shows 0 ms.
  assert.equal(stages.matching.elapsedMs, 0);

  // The record stays valid for the site and the worker.
  const base = sampleRecord();
  const record = sampleRecord({
    input: { ...base.input, videos },
    result: { ...base.result, registeredViews: 3, landmarks: 40, observations: 120 },
    timing: { ...base.timing, totalMs: 9100, stages },
    failure: {
      message: 'x',
      operation: 'prepareTriangulation (start)',
      kernel: '',
      reason: '',
      framesProcessed: 120,
    },
  });
  assert.equal(runRecordProblem(record), undefined);
});
