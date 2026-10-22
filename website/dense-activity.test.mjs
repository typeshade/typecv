import test from 'node:test';
import assert from 'node:assert/strict';
import { appendFusionPreview, denseActivityText } from './src/lib/dense-activity.ts';

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
