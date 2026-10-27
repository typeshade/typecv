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
