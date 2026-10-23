import test from 'node:test';
import assert from 'node:assert/strict';
import {
  appendFusionPreview,
  denseActivityProgress,
  denseActivityText,
} from './src/lib/dense-activity.ts';

test('free-space progress spans cameras; result waits and final processing have no guessed percentage', () => {
  assert.deepEqual(
    denseActivityProgress({ phase: 'carve', done: 0, total: 100, phaseDone: 100, phaseTotal: 400 }),
    {
      done: 100,
      total: 400,
      percent: 25,
      submitted: true,
    },
  );
  assert.equal(
    denseActivityProgress({ phase: 'carve', done: 10, total: 100, phaseDone: 110, phaseTotal: 400 })
      .percent,
    27,
  );
  for (const phase of ['readback', 'sort', 'clean', 'mesh', 'waiting'])
    assert.equal(denseActivityProgress({ phase, done: 0, total: 1 }), undefined);
  assert.equal(denseActivityProgress({ phase: 'count', done: 7, total: 10 }).percent, 70);
  assert.equal(denseActivityProgress({ phase: 'photometric', done: 2, total: 4 }).submitted, false);
});

test('fusion activity names the actual phase, region and scene camera in both languages', () => {
  const activity = { phase: 'integrate', pass: 2, passes: 4, view: 3, views: 8, viewId: 105 };
  assert.equal(denseActivityText(activity, true), '깊이 합치기 · 구역 2/4 · 카메라 #105 (4/8)');
  assert.equal(
    denseActivityText(activity, false),
    'Combining depths · Region 2/4 · Camera #105 (4/8)',
  );
  assert.equal(denseActivityText({ phase: 'clean' }, true), '노이즈 정리');
  assert.ok(!denseActivityText({ phase: 'allocate', pass: 0, passes: 0 }, true).includes('0/0'));
});

test('fusion preview chunks append within one run and reset for the next', () => {
  const chunk = {
    viewId: -1,
    positions: Float32Array.of(0, 0, 1),
    colors: Uint8Array.of(255, 0, 0),
  };
  const first = appendFusionPreview(undefined, 1, chunk);
  const second = appendFusionPreview(first, 1, chunk);
  assert.equal(second.count, 2);
  assert.equal(first.count, 1);
  assert.equal(appendFusionPreview(second, 2, chunk).count, 1);
});
