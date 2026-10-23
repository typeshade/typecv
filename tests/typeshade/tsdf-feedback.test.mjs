import test from 'node:test';
import assert from 'node:assert/strict';
import * as api from '../../dist/internal/gpu.js';
import { feedbackCase, mvsFeedbackCase } from './tsdf-feedback-cases.mjs';

test('weighted free-space feedback advances across all cameras and vote resets within each region', async () => {
  const result = await feedbackCase(api, 'cpu', { visibility: 'view-weighted', maxBlocks: 1 });
  assert.deepEqual(
    result.observed,
    result.baseline,
    'feedback preserves arrays and GPU work counts',
  );
  const regions = new Map();
  for (const event of result.progress.filter((event) => event.phase === 'carve')) {
    const previous = regions.get(event.pass);
    assert.ok(event.phaseDone >= (previous?.phaseDone ?? 0));
    if (previous) assert.equal(event.phaseTotal, previous.phaseTotal);
    assert.ok(event.phaseDone <= event.phaseTotal);
    regions.set(event.pass, event);
  }
  assert.ok(regions.size > 1, 'progress covers repeated regions');
  for (const event of regions.values()) {
    assert.equal(event.phaseDone, event.phaseTotal);
    assert.ok(event.phaseTotal > event.total * event.views, 'vote resets are included');
  }
});

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
