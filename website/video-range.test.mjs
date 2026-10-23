import test from 'node:test';
import assert from 'node:assert/strict';
import { formatVideoTime, videoRangeBounds, videoRangeProgress } from './src/lib/video-range.ts';
import { runRecordProblem } from './src/lib/run-record.ts';
import { sampleRecord } from './record-fixture.mjs';

test('selection bounds clip to real track timestamps and never widen an invalid selection', () => {
  assert.deepEqual(videoRangeBounds(0.5, 8, { start: 0, end: 4 }), { start: 0.5, end: 4 });
  assert.deepEqual(videoRangeBounds(0, 8, { start: 6, end: 10 }), { start: 6, end: 8 });
  assert.deepEqual(videoRangeBounds(0.5, 8), { start: 0.5, end: 8 });
  for (const range of [
    { start: -1, end: 3 },
    { start: 3, end: 3 },
    { start: 5, end: 3 },
    { start: NaN, end: 5 },
    { start: 1, end: Infinity },
    { start: 9, end: 10 },
  ])
    assert.throws(() => videoRangeBounds(0, 8, range), RangeError);
});

test('run progress is relative to the selected interval and source timestamps remain absolute', () => {
  const range = { start: 30, end: 40 };
  assert.equal(videoRangeProgress(0, 60, range), 0);
  assert.equal(videoRangeProgress(30, 60, range), 0);
  assert.equal(videoRangeProgress(35, 60, range), 0.5);
  assert.equal(videoRangeProgress(60, 60, range), 1);
  assert.equal(videoRangeProgress(30, 60), 0.5);
  assert.equal(formatVideoTime(59.999), '1:00.00');
  assert.equal(formatVideoTime(61.25), '1:01.25');
});

test('run records accept selected source intervals and reject invalid single and multi-video bounds', () => {
  const record = sampleRecord();
  record.input.range = { start: 2, end: 5 };
  assert.equal(runRecordProblem(record), undefined);
  record.input.videos = [
    {
      videoId: 'a'.repeat(16),
      label: null,
      durationSeconds: 8,
      keyframes: 3,
      registeredViews: 2,
      focalRatio: 0.8,
      component: 0,
      tags: [],
      range: { start: 3, end: 6 },
    },
  ];
  assert.equal(runRecordProblem(record), undefined);
  for (const range of [
    { start: -1, end: 4 },
    { start: 5, end: 5 },
    { start: 5, end: 4 },
    { start: 0, end: 25 },
    { start: NaN, end: 6 },
  ]) {
    const single = structuredClone(record);
    single.input.range = range;
    assert.equal(runRecordProblem(single), 'input');
    const multi = structuredClone(record);
    multi.input.videos[0].range = range;
    assert.equal(runRecordProblem(multi), 'input');
  }
});
