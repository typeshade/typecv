// Phase 5a of the refactoring programme (`docs/library-structure.md`, section 5, row 5): the table
// tests of the lab's pure helpers. Each helper was inline in `src/workers/sfm.worker.ts`,
// `src/workers/multi-video.ts` or `src/lib/multi-video-job.ts` on `main` at db7216a, the parent of
// phase 5a. Each table holds the helper's values and units, and rows on both sides of each branch.
// `inline` is the code of a site as it stood there, copied with its literals, and every row holds
// the helper to it. Where two sites differ, each variant has its own copy and its own rows.
import test from 'node:test';
import assert from 'node:assert/strict';
import {
  BUDGET_REFERENCE_EDGE,
  REFERENCE_POOL_BYTES,
  REFERENCE_WORKING_BYTES,
  analysisScale,
  deviceBudgets,
  workingBytesFromAdapter,
} from './src/lib/working-budget.ts';
import { denseWorkingBytes } from './src/lib/dense-run.ts';
import { MAX_FEATURE_CAP, featureCap } from './src/lib/lab-features.ts';
import { LAB_OPTIONS } from './src/lib/lab-options.ts';
import { medianMotionPx } from './src/lib/keyframe-gate.ts';
import { trimStages } from './src/lib/lab-ledger.ts';

const MiB = 1024 ** 2;

test('the budgets: 64 and 32 MiB at 640 px, 4 times at octave -1, half of maxBufferSize (start, runDenseStage)', () => {
  assert.equal(BUDGET_REFERENCE_EDGE, 640);
  assert.equal(REFERENCE_WORKING_BYTES, 64 * MiB);
  assert.equal(REFERENCE_POOL_BYTES, 32 * MiB);
  // `start` of both workers: the scale.
  const inlineScale = (analysisEdge, firstOctave) =>
    ((analysisEdge * analysisEdge) / (640 * 640)) * (firstOctave === -1 ? 4 : 1);
  // `start` of both workers: the budgets of the SfM device.
  const inlineSfm = (scale, maxBufferSize) => ({
    workingBytes: workingBytesFromAdapter(Math.round(64 * 1024 ** 2 * scale), maxBufferSize),
    poolBytes: Math.round(32 * 1024 ** 2 * scale),
  });
  // `runDenseStage` of the multi-video worker: the budgets of the dense device.
  const inlineDense = (scale, maxBufferSize) => ({
    workingBytes: denseWorkingBytes(Math.round(64 * 1024 ** 2 * scale), maxBufferSize),
    poolBytes: Math.round(32 * 1024 ** 2 * scale),
  });
  for (const [edge, octave, maxBufferSize, scale, workingBytes, poolBytes] of [
    [640, 0, undefined, 1, 64 * MiB, 32 * MiB],
    [640, -1, undefined, 4, 256 * MiB, 128 * MiB],
    [1280, 0, undefined, 4, 256 * MiB, 128 * MiB],
    [1280, -1, undefined, 16, 1024 * MiB, 512 * MiB],
    [320, 0, undefined, 0.25, 16 * MiB, 8 * MiB],
    [720, 0, undefined, 1.265625, 84934656, 42467328],
    [853, 0, undefined, 1.77638916015625, 119211459, 59605729],
    [640, 0, 0, 1, 64 * MiB, 32 * MiB],
    [640, 0, 128 * MiB, 1, 64 * MiB, 32 * MiB],
    [640, 0, 129 * MiB + 1, 1, 67633152, 32 * MiB],
    [640, 0, 256 * MiB, 1, 128 * MiB, 32 * MiB],
    [960, 0, 256 * MiB, 2.25, 144 * MiB, 72 * MiB],
    [640, 0, 2048 * MiB, 1, 1024 * MiB, 32 * MiB],
    [1280, 0, 2048 * MiB, 4, 1024 * MiB, 128 * MiB],
    [1280, -1, 2048 * MiB, 16, 1024 * MiB, 512 * MiB],
  ]) {
    const row = `${edge} px, octave ${octave}, maxBufferSize ${maxBufferSize}`;
    assert.equal(analysisScale(edge, octave), scale, row);
    assert.equal(inlineScale(edge, octave), scale, row + ', inline');
    const expected = { workingBytes, poolBytes };
    assert.deepEqual(deviceBudgets(scale, maxBufferSize), expected, row);
    assert.deepEqual(inlineSfm(scale, maxBufferSize), expected, row + ', inline');
    assert.deepEqual(
      deviceBudgets(scale, maxBufferSize, denseWorkingBytes),
      expected,
      row + ', dense',
    );
    assert.deepEqual(inlineDense(scale, maxBufferSize), expected, row + ', dense, inline');
  }
});

test('the feature cap: 4096 features at 640 px, at most 16384 (start of both workers)', () => {
  assert.equal(MAX_FEATURE_CAP, 16384);
  assert.equal(LAB_OPTIONS.maxFeatures, 4096);
  // sfm.worker.ts `start`: the base is the `maxFeatures` of the run's controls.
  const inlineSfm = (base, scale) => Math.min(16384, Math.round(base.maxFeatures * scale));
  // multi-video.ts `start`, when the query sets no cap: the base is 4096.
  const inlineMulti = (scale) => Math.min(16384, Math.round(4096 * scale));
  for (const [scale, cap] of [
    [0.25, 1024],
    [1, 4096],
    [1.265625, 5184],
    [1.77638916015625, 7276],
    [2.25, 9216],
    [3.99, 16343],
    [4, 16384],
    [4.01, 16384],
    [16, 16384],
  ]) {
    assert.equal(featureCap(4096, scale), cap, `scale ${scale}`);
    assert.equal(featureCap(LAB_OPTIONS.maxFeatures, scale), cap, `scale ${scale}, controls`);
    assert.equal(inlineSfm(LAB_OPTIONS, scale), cap, `scale ${scale}, inline`);
    assert.equal(inlineMulti(scale), cap, `scale ${scale}, multi-video, inline`);
  }
});

test('the median motion of the legacy rule: the median displacement of the matches, in the pixels of this frame (processFrame)', () => {
  // sfm.worker.ts `medianMotionPx`, as it stood.
  const inline = (matches, previousPoints, previous, points, frame) => {
    const distances = [];
    for (let i = 0; i < matches.length; i += 2) {
      const a = matches[i] * 2,
        b = matches[i + 1] * 2;
      distances.push(
        Math.hypot(
          points[b] - (previousPoints[a] * frame.width) / previous.width,
          points[b + 1] - (previousPoints[a + 1] * frame.height) / previous.height,
        ),
      );
    }
    if (!distances.length) return Infinity;
    distances.sort((x, y) => x - y);
    return distances[distances.length >> 1];
  };
  const size = (width, height) => ({ width, height });
  const f64 = (...values) => Float64Array.from(values);
  for (const [what, matches, previousPoints, previous, points, frame, motion] of [
    ['no match', [], f64(1, 1), size(128, 96), f64(1, 1), size(128, 96), Infinity],
    ['one match', [0, 0], f64(10, 10), size(128, 96), f64(13, 14), size(128, 96), 5],
    [
      'an odd count',
      [0, 0, 1, 1, 2, 2],
      f64(0, 0, 0, 0, 0, 0),
      size(9, 9),
      f64(3, 0, 1, 0, 2, 0),
      size(9, 9),
      2,
    ],
    [
      'an even count: the upper middle',
      [0, 0, 1, 1, 2, 2, 3, 3],
      f64(0, 0, 0, 0, 0, 0, 0, 0),
      size(9, 9),
      f64(4, 0, 1, 0, 3, 0, 2, 0),
      size(9, 9),
      3,
    ],
    [
      'the previous points scaled to this frame',
      [0, 0],
      f64(100, 200),
      size(1280, 960),
      f64(53, 104),
      size(640, 480),
      5,
    ],
    ['a scale in x only', [0, 0], f64(100, 40), size(200, 100), f64(50, 43), size(100, 100), 3],
    [
      'the indices of each frame',
      [1, 0, 0, 1],
      f64(0, 0, 7, 7),
      size(16, 16),
      f64(7, 11, 3, 4),
      size(16, 16),
      5,
    ],
  ]) {
    const args = [Uint32Array.from(matches), previousPoints, previous, points, frame];
    assert.equal(medianMotionPx(...args), motion, what);
    assert.equal(inline(...args), motion, what + ', inline');
  }
});

test('the detailed stage list: 400 entries, keyframe entries dropped last (record, registerFrame)', () => {
  assert.equal(LAB_OPTIONS.detailedStageLimit, 400);
  // sfm.worker.ts `trimStages`, as it stood, with `LAB_OPTIONS.detailedStageLimit` written out.
  const inline = (stages) => {
    let excess = stages.length - 400;
    for (let i = 0; excess > 0 && i < stages.length;)
      if (stages[i].operation === 'keyframe') i++;
      else {
        stages.splice(i, 1);
        excess--;
      }
    if (excess > 0) stages.splice(0, excess);
  };
  // A list of `n` entries; the entries at the indices of `keyframes` are keyframe entries.
  const list = (n, keyframes = () => false) =>
    Array.from({ length: n }, (_, id) => ({
      operation: keyframes(id) ? 'keyframe' : 'matches',
      id,
    }));
  const ids = (stages) => stages.map((stage) => stage.id);
  const range = (from, to) => Array.from({ length: to - from }, (_, i) => from + i);
  for (const [what, stages, kept] of [
    ['an empty list', list(0), []],
    ['400 entries', list(400), range(0, 400)],
    ['401 entries: the oldest goes', list(401), range(1, 401)],
    ['the oldest keyframe entries stay', list(401, (id) => id < 3), [0, 1, 2, ...range(4, 401)]],
    ['403 entries: three go', list(403, (id) => id % 2 === 0), [0, 2, 4, ...range(6, 403)]],
    ['only keyframe entries: the oldest go', list(402, () => true), range(2, 402)],
    ['keyframe entries go when the others are gone', list(405, (id) => id < 403), range(3, 403)],
  ]) {
    const helper = stages.map((stage) => ({ ...stage })),
      copy = stages.map((stage) => ({ ...stage }));
    trimStages(helper);
    inline(copy);
    assert.deepEqual(ids(helper), kept, what);
    assert.deepEqual(ids(copy), kept, what + ', inline');
    assert.deepEqual(helper, copy, what + ', the entries');
  }
});
