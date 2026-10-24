import test from 'node:test';
import assert from 'node:assert/strict';
import { inBrowser } from './browser.mjs';

test('unbounded CG scheduling skips guarded iterations while retaining numerical budgets and fallbacks', async () => {
  const { result, errors } = await inBrowser(
    async () => {
      const api = await import('/dist/gpu.js');
      const { cgScheduleFixtures, runCgScheduleFixture } =
        await import('/tests/typeshade/bundle-cg-schedule-cases.mjs');
      const rows = [];
      for (const fixture of cgScheduleFixtures()) {
        if (fixture.options.schurCache) {
          rows.push({ cached: await runCgScheduleFixture(api, fixture) });
        } else {
          rows.push({
            reference: await runCgScheduleFixture(api, fixture, true),
            scheduled: await runCgScheduleFixture(api, fixture),
          });
        }
      }
      return rows;
    },
    undefined,
    { timeoutMs: 90000 },
  );
  assert.deepEqual(errors, []);
  for (const row of result) {
    if (row.cached) {
      assert.ok(row.cached.certificates > 0, 'cached solves retain their mandatory certificate');
      assert.equal(row.cached.liveBytes, 0);
      assert.equal(row.cached.sourceBefore, row.cached.sourceAfter);
      continue;
    }
    const { reference, scheduled } = row;
    assert.equal(scheduled.output, reference.output, scheduled.name);
    assert.equal(scheduled.cgIterations, reference.cgIterations, scheduled.name);
    assert.equal(scheduled.linearSolverFailures, reference.linearSolverFailures, scheduled.name);
    assert.equal(scheduled.sourceBefore, scheduled.sourceAfter, scheduled.name);
    assert.equal(scheduled.liveBytes, 0, scheduled.name);
    if (scheduled.name === 'local-streamed') {
      assert.equal(scheduled.streamed, true);
      assert.equal(reference.streamed, true);
    }
    assert.ok(scheduled.advanceCalls <= reference.advanceCalls, scheduled.name);
    if (scheduled.name === 'all-fixed')
      assert.ok(
        scheduled.advanceCalls < reference.advanceCalls,
        'zero-CG convergence warms the hint',
      );
  }
});
