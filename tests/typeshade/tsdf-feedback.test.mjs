import test from 'node:test';
import assert from 'node:assert/strict';
import * as api from '../../dist/internal/gpu.js';
import { feedbackCase, mvsFeedbackCase } from './tsdf-feedback-cases.mjs';

test('TSDF live feedback preserves CPU results and submissions, owns preview arrays and cancels', async () => {
  const result = await feedbackCase(api, 'cpu');
  assert.ok(result.baseline.count > 0);
  assert.deepEqual(result.observed, result.baseline);
  assert.equal(
    result.partCounts.reduce((a, b) => a + b, 0),
    result.baseline.count,
  );
  const phases = new Set(result.progress.map((e) => e.phase));
  for (const phase of [
    'allocate',
    'assign',
    'integrate',
    'carve',
    'count',
    'extract',
    'readback',
    'sort',
  ])
    assert.ok(phases.has(phase), phase);
  for (const event of result.progress) assert.ok(event.done >= 0 && event.done <= event.total);
  assert.equal(result.cancelled, true);
  assert.equal(result.liveBytes, 0);
});

test('MVS fusion previews use world coordinates and original view ids without changing final events', async () => {
  const r = await mvsFeedbackCase(api, 'cpu');
  assert.ok(r.observed.count > 0);
  assert.deepEqual(r.observed, r.baseline);
  assert.deepEqual(r.previewPoints, r.finalPoints);
  assert.deepEqual(r.finalEvents, ['points']);
  assert.deepEqual(r.viewIds, [10, 11, 12]);
  assert.equal(r.liveBytes, 0);
});
