/** Compare complete BA outputs and submitted work against an immutable public GPU bundle.
 * No video data or required solver iterations are removed. Checkpoint wave counts are reported
 * because an underestimated hint can require another normal solver checkpoint. */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { inBrowser } from '../tests/typeshade/browser.mjs';

const { values } = parseArgs({
  options: {
    baseline: { type: 'string' },
    current: { type: 'string', default: 'dist/gpu/index.js' },
    out: { type: 'string' },
  },
});
if (!values.baseline) throw new Error('--baseline must name a separately built public GPU bundle');
const { result, errors } = await inBrowser(
  async () => {
    const variants = {
      baseline: await import('/cg-schedule-baseline.mjs'),
      current: await import('/cg-schedule-current.mjs'),
    };
    const { cgScheduleFixtures, runCgScheduleFixture } =
      await import('/tests/typeshade/bundle-cg-schedule-cases.mjs');
    const rows = [];
    for (const [index, fixture] of cgScheduleFixtures().entries()) {
      const runs = {};
      for (const variant of index % 2 ? ['current', 'baseline'] : ['baseline', 'current'])
        runs[variant] = await runCgScheduleFixture(variants[variant], fixture);
      rows.push({ name: fixture.name, runs });
    }
    return rows;
  },
  undefined,
  {
    timeoutMs: 90000,
    serve: {
      '/cg-schedule-baseline.mjs': resolve(values.baseline),
      '/cg-schedule-current.mjs': resolve(values.current),
    },
  },
);
assert.deepEqual(errors, []);
for (const { name, runs } of result) {
  assert.equal(runs.current.output, runs.baseline.output, `${name}: terminal bits changed`);
  assert.equal(runs.current.cgIterations, runs.baseline.cgIterations, name);
  for (const run of Object.values(runs)) {
    assert.equal(run.sourceBefore, run.sourceAfter, name);
    assert.equal(run.liveBytes, 0, name);
  }
  if (name.endsWith('cache')) {
    assert.equal(runs.current.certificates, runs.baseline.certificates, name);
    assert.equal(runs.current.cachedAdvanceCalls, runs.baseline.cachedAdvanceCalls, name);
    assert.equal(runs.current.readbackBytes, runs.baseline.readbackBytes, name);
  } else {
    assert.ok(runs.current.advanceCalls <= runs.baseline.advanceCalls, name);
    if (name === 'all-fixed')
      assert.ok(runs.current.advanceCalls < runs.baseline.advanceCalls, name);
  }
}
const sha = async (file) =>
  createHash('sha256')
    .update(await readFile(file))
    .digest('hex');
const receipt = {
  scope:
    'Generated fixtures with device chunks forced to their full numerical cap. Only guarded submitted work after convergence is reduced; no timing percentage or whole-video speed is claimed.',
  bundles: {
    baseline: await sha(resolve(values.baseline)),
    current: await sha(resolve(values.current)),
  },
  rows: result,
};
const json = JSON.stringify(receipt, null, 2) + '\n';
if (values.out) await writeFile(resolve(values.out), json);
process.stdout.write(json);
