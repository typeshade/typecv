/**
 * Compare fixed-camera pose-derivative pruning against a separately built public GPU bundle.
 * TYPESHADE_CHROMIUM=/usr/bin/chromium node scripts/measure-fixed-camera-jacobians.mjs \
 *   --baseline /absolute/path/to/baseline-public.mjs --pairs 3 --out /tmp/bundle.json
 *
 * Generated geometry only. Structural counts are per active linearization, not GPU timestamps
 * or a whole-video speed prediction. Both variants must produce identical terminal bits.
 */
import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { inBrowser } from '../tests/typeshade/browser.mjs';

const { values } = parseArgs({
  options: {
    baseline: { type: 'string' },
    current: { type: 'string', default: 'dist/gpu/index.js' },
    backend: { type: 'string', default: 'webgpu' },
    pairs: { type: 'string', default: '3' },
    count: { type: 'string', default: '64' },
    views: { type: 'string', default: '8' },
    out: { type: 'string' },
  },
});
const pairs = Number(values.pairs),
  count = Number(values.count),
  views = Number(values.views);
if (!values.baseline) throw new Error('--baseline must name a separately built public GPU bundle');
if (!['cpu', 'webgpu'].includes(values.backend)) throw new Error('--backend must be cpu or webgpu');
if (!Number.isSafeInteger(pairs) || pairs < 1 || pairs > 5) throw new Error('--pairs must be 1..5');
if (!Number.isSafeInteger(count) || count < 8 || count > 256)
  throw new Error('--count must be 8..256');
if (!Number.isSafeInteger(views) || views < 4 || views > 16)
  throw new Error('--views must be 4..16');

const { result, errors } = await inBrowser(
  async ({ backend, pairs, count, views }) => {
    const variants = {
      baseline: await import('/fixed-jacobian-baseline.mjs'),
      current: await import('/fixed-jacobian-current.mjs'),
    };
    const { fixedJacobianCases, fixedJacobianDigest } =
      await import('/tests/typeshade/bundle-fixed-jacobian-cases.mjs');
    const devices = {},
      rows = [];
    try {
      for (const [variant, api] of Object.entries(variants))
        devices[variant] = await api.ShadeDevice.open({ backend, poolBytes: 0 });
      for (const fixture of fixedJacobianCases({ count, views })) {
        const sourceBefore = await fixedJacobianDigest(fixture.model);
        // Compile and warm each complete function before timing either variant.
        for (const [variant, api] of Object.entries(variants))
          await api.bundleAdjustGpu(devices[variant], fixture.model, fixture.options);
        const runs = [];
        for (let pair = 0; pair < pairs; pair++)
          for (const variant of pair % 2 ? ['current', 'baseline'] : ['baseline', 'current']) {
            const api = variants[variant],
              device = devices[variant];
            const start = performance.now();
            const output = await api.bundleAdjustGpu(device, fixture.model, fixture.options);
            const elapsedMs = performance.now() - start;
            runs.push({
              pair,
              variant,
              elapsedMs,
              output: await fixedJacobianDigest(output.reconstruction, output),
              diagnostics: output.diagnostics,
              cgIterations: output.cgIterations,
              linearSolverFailures: output.linearSolverFailures,
              dispatches: output.execution.report.dispatches,
              submissions: output.execution.report.submissions,
              liveBytes: device.statistics.liveBytes,
            });
          }
        rows.push({
          name: fixture.name,
          views,
          points: count,
          observations: count * views,
          variableCameras: fixture.variable.size,
          fixedObservations: fixture.fixedObservations,
          unusedPoseRowsPerActiveLinearization: 2 * fixture.fixedObservations,
          unusedPoseEntriesPerActiveLinearization: 12 * fixture.fixedObservations,
          sourceBefore,
          sourceAfter: await fixedJacobianDigest(fixture.model),
          runs,
        });
      }
      return {
        backend,
        devices: Object.fromEntries(
          Object.entries(devices).map(([variant, device]) => [variant, device.info]),
        ),
        rows,
      };
    } finally {
      for (const device of Object.values(devices)) await device.dispose();
    }
  },
  { backend: values.backend, pairs, count, views },
  {
    timeoutMs: 60000,
    serve: {
      '/fixed-jacobian-baseline.mjs': resolve(values.baseline),
      '/fixed-jacobian-current.mjs': resolve(values.current),
    },
  },
);
assert.deepEqual(errors, []);
const median = (numbers) => {
  const sorted = [...numbers].sort((a, b) => a - b);
  return (sorted[Math.floor((sorted.length - 1) / 2)] + sorted[Math.floor(sorted.length / 2)]) / 2;
};
for (const row of result.rows) {
  assert.equal(row.sourceBefore, row.sourceAfter, `${row.name}: caller input changed`);
  for (const run of row.runs) {
    assert.equal(run.liveBytes, 0, `${row.name}: active GPU allocation leak`);
    assert.equal(run.output, row.runs[0].output, `${row.name}: terminal bits differ`);
  }
  row.medianMs = Object.fromEntries(
    ['baseline', 'current'].map((variant) => [
      variant,
      median(row.runs.filter((run) => run.variant === variant).map((run) => run.elapsedMs)),
    ]),
  );
}
const sha = async (filename) =>
  createHash('sha256')
    .update(await readFile(filename))
    .digest('hex');
result.bundleSha256 = {
  baseline: await sha(resolve(values.baseline)),
  current: await sha(resolve(values.current)),
};
result.scope =
  'Generated BA fixtures, identical input resolution/observations/iterations. Fixed observations still constrain landmarks and shared focal. Counts describe removed unused pose-derivative work per active linearization; timing is synthetic and adapter-specific.';
const json = JSON.stringify(result, null, 2) + '\n';
if (values.out) await writeFile(resolve(values.out), json);
process.stdout.write(json);
