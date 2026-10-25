/** Exact old/new mapper comparison for long tracks and guarded continuation; no wall-time claim. */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { inBrowser } from '../tests/typeshade/browser.mjs';

const { values } = parseArgs({
  options: {
    baseline: { type: 'string' },
    candidate: { type: 'string', default: 'dist/gpu/index.js' },
    backend: { type: 'string', default: 'webgpu' },
    out: { type: 'string' },
  },
});
if (!values.baseline || !['cpu', 'webgpu'].includes(values.backend))
  throw new Error(
    'Use --baseline <standalone-gpu.js> [--candidate <standalone-gpu.js>] [--backend cpu|webgpu]',
  );
const baseline = resolve(values.baseline),
  candidate = resolve(values.candidate);
const { result, errors } = await inBrowser(
  async (backend) => {
    const variants = {
      baseline: await import('/continued-baseline.mjs'),
      continued: await import('/continued-candidate.mjs'),
    };
    const { continuedMapperFixture, continuedMapperSignature } =
      await import('/tests/typeshade/continued-mapper-cases.mjs');
    const { incrementalControls, verifyPairs, executeReconstructionJob } =
      await import('/dist/reconstruction/mapper.js');
    // Like the multi-video component mapper, reuse real verified graph verdicts. CPU geometry
    // is computed once so this continuation check does not repeat unrelated GPU pair RANSAC.
    const graphInput = continuedMapperFixture();
    const controls = { ransacIterations: 16, seed: 17, rotationTracking: false };
    const checked = incrementalControls(graphInput.images, graphInput.pairs, controls);
    const machine = verifyPairs(checked.byId, graphInput.pairs, checked.controls);
    let step = machine.next();
    while (!step.done) step = machine.next(executeReconstructionJob(step.value));
    const graph = step.value;
    const key = (pair) => [pair.first, pair.second].sort((a, b) => a - b).join(':');
    const diagnostics = new Map(graph.diagnostics.map((row) => [key(row), row]));
    const candidates = new Map(graph.candidates.map((row) => [key(row.pair), row.result]));
    const preVerified = graph.verified.map((pair) => ({
      verified: pair,
      verification: diagnostics.get(key(pair)),
      geometry: candidates.get(key(pair)),
    }));
    console.log(JSON.stringify({ phase: 'graph-verified', pairs: preVerified.length }));
    const rows = [];
    const digest = async (value) =>
      [
        ...new Uint8Array(
          await crypto.subtle.digest('SHA-256', new TextEncoder().encode(JSON.stringify(value))),
        ),
      ]
        .map((b) => b.toString(16).padStart(2, '0'))
        .join('');
    const devices = {};
    try {
      for (const [variant, api] of Object.entries(variants))
        devices[variant] = await api.ShadeDevice.open({
          backend,
          fp64Flavor: 'float',
          poolBytes: 0,
        });
      for (const guarded of [false, true])
        for (const [variant, api] of Object.entries(variants)) {
          const device = devices[variant];
          const fixture = continuedMapperFixture(),
            before = JSON.stringify(fixture);
          const events = [],
            published = [];
          const mapped = await api.reconstructIncrementalGpu(
            device,
            fixture.images,
            preVerified.map((row) => row.verified),
            {
              ransacIterations: 16,
              seed: 17,
              bundleIterations: 0,
              finalBundleIterations: 0,
              rotationTracking: false,
              preVerified,
              ...(guarded ? { maxRelativeDepthStdDev: 0.1 } : {}),
              onProgress(event) {
                events.push({ ...event });
              },
              onStage(stage) {
                if (
                  !stage.execution?.report ||
                  !Number.isFinite(stage.execution.applicationUploadBytes)
                )
                  throw new Error('Numerical stage callback lost its execution contract');
                events.push({
                  stage: stage.operation,
                  backend: stage.execution.backend,
                  problem: stage.problem ?? null,
                });
              },
              onReconstruction(model) {
                published.push({ views: model.views.length, points: model.landmarks.length });
                // Public observers must keep their existing independent snapshot contract.
                model.views[0].pose.R.fill(999);
                model.landmarks[0]?.xyz.fill(999);
              },
            },
          );
          const observations = mapped.execution.stages.filter(
            (s) => s.operation === 'observations',
          );
          if (mapped.execution.stages.some((stage) => stage.operation === 'pairVerification'))
            throw new Error('A component repeated its already verified graph RANSAC');
          rows.push({
            guarded,
            variant,
            digest: await digest(continuedMapperSignature(mapped, events, published)),
            views: mapped.reconstruction.views.length,
            points: mapped.reconstruction.landmarks.length,
            longestTrack: Math.max(
              ...mapped.reconstruction.landmarks.map((p) => p.observations.length),
            ),
            inputUnchanged: before === JSON.stringify(fixture),
            observationUploadBytes: observations.reduce(
              (sum, s) => sum + s.execution.applicationUploadBytes,
              0,
            ),
            observationReadbackBytes: observations.reduce(
              (sum, s) => sum + s.execution.applicationReadbackBytes,
              0,
            ),
            stages: observations.length,
            liveBytes: device.statistics.liveBytes,
          });
          console.log(JSON.stringify({ phase: 'mapper-case-done', variant, guarded }));
        }
      return rows;
    } finally {
      for (const device of Object.values(devices)) await device.dispose();
    }
  },
  values.backend,
  {
    timeoutMs: 60000,
    serve: { '/continued-baseline.mjs': baseline, '/continued-candidate.mjs': candidate },
  },
);
assert.deepEqual(errors, []);
for (const guarded of [false, true]) {
  const rows = result.filter((r) => r.guarded === guarded);
  assert.equal(rows.length, 2);
  assert.equal(
    rows[0].digest,
    rows[1].digest,
    'mapper output, mean, registration order or public callbacks changed',
  );
  for (const row of rows) {
    assert.equal(row.inputUnchanged, true);
    assert.equal(row.liveBytes, 0);
    assert.equal(row.views, 8);
    assert.ok(row.longestTrack >= 6);
  }
}
const receipt = {
  backend: values.backend,
  scope:
    'Generated eight-view/24-point tracks with inconsistent measurements; no whole-run runtime or RSS claim.',
  rows: result,
  bundleSha256: Object.fromEntries(
    await Promise.all(
      [
        ['baseline', baseline],
        ['continued', candidate],
      ].map(async ([name, file]) => [
        name,
        createHash('sha256')
          .update(await readFile(file))
          .digest('hex'),
      ]),
    ),
  ),
};
const json = JSON.stringify(receipt, null, 2) + '\n';
if (values.out) await writeFile(resolve(values.out), json);
process.stdout.write(json);
