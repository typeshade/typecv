/**
 * Combined generated-track view-graph → verification → incremental SfM/BA → observation check.
 * Pass separately built standalone INTERNAL GPU bundles; each is snapshotted before launch.
 * This excludes decode/SIFT, Dense MVS and rendering, and does not measure browser RSS.
 */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { quantizeDescriptors } from '../src/reconstruction/view-graph.ts';
import { pathScene } from '../tests/view-graph-scene.ts';
import { inBrowser } from '../tests/typeshade/browser.mjs';
import { summarizeCombinedReceipt } from '../tests/typeshade/multi-video-combined-cases.mjs';

const { values } = parseArgs({
  options: {
    baseline: { type: 'string' },
    candidate: { type: 'string', default: 'dist/internal/gpu.js' },
    'baseline-public': { type: 'string' },
    'candidate-public': { type: 'string', default: 'dist/gpu/index.js' },
    backend: { type: 'string', default: 'webgpu' },
    videos: { type: 'string', default: '3' },
    views: { type: 'string', default: '3' },
    points: { type: 'string', default: '40' },
    pairs: { type: 'string', default: '1' },
    warmup: { type: 'boolean', default: false },
    timeout: { type: 'string', default: '120' },
    out: { type: 'string' },
    'summary-out': { type: 'string' },
  },
});
if (!values.baseline) throw new Error('--baseline must name a standalone internal GPU bundle');
if (!values['baseline-public'])
  throw new Error("--baseline-public must name the same baseline revision's public GPU bundle");
if (!['cpu', 'webgpu'].includes(values.backend)) throw new Error('--backend must be cpu or webgpu');
const videos = Number(values.videos),
  views = Number(values.views),
  points = Number(values.points),
  pairs = Number(values.pairs),
  timeout = Number(values.timeout);
if (
  ![videos, views, points, pairs, timeout].every(Number.isSafeInteger) ||
  videos < 2 ||
  videos > 15 ||
  views < 2 ||
  views > 6 ||
  videos * views > 30 ||
  points < 32 ||
  points > 80 ||
  pairs < 1 ||
  pairs > 3 ||
  timeout < 10 ||
  timeout > 120
)
  throw new Error(
    'Use 2..15 videos, 2..6 views each, ≤30 total views, 32..80 points, 1..3 pairs, 10..120 second timeout',
  );

const { keyframes } = pathScene(
  Array.from({ length: videos }, (_, video) => {
    const from = -30 + (video % 5) * 8;
    return {
      label: `generated-video-${video}`,
      cloud: 0,
      views,
      from,
      to: from + 40,
      radius: 9,
      height: -1 - Math.floor(video / 5) * 0.6 - (video % 3) * 0.1,
    };
  }),
  { points, seed: 41 },
);
const serialized = keyframes.map(
  ({ viewId: _viewId, keypoints, descriptors, global, ...frame }) => ({
    ...frame,
    keypoints: [...keypoints],
    descriptors: [...quantizeDescriptors(descriptors)],
    global: [...global],
  }),
);
const options = {
  graph: {
    retrievalK: 3,
    temporalNeighbours: 2,
    exclusionWindow: 2,
    reservePerVideoPair: 1,
    minComponentViews: 3,
    verification: { seed: 7, ransacIterations: 64 },
  },
  mapper: { seed: 7, ransacIterations: 64, bundleIterations: 2, finalBundleIterations: 2 },
};
const temporary = await mkdtemp(join(tmpdir(), 'typecv-combined-'));
try {
  const modules = {};
  for (const [variant, path] of Object.entries({
    baseline: values.baseline,
    candidate: values.candidate,
    baselinePublic: values['baseline-public'],
    candidatePublic: values['candidate-public'],
  })) {
    const bytes = await readFile(resolve(path));
    const snapshot = join(temporary, `${variant}.mjs`);
    await writeFile(snapshot, bytes);
    modules[variant] = { path: snapshot, sha256: createHash('sha256').update(bytes).digest('hex') };
  }
  let browserResult;
  try {
    browserResult = await inBrowser(
      async ({ backend, serialized, options, pairs, warmup }) => {
        const variants = {
          baseline: await import('/combined-baseline.mjs'),
          candidate: await import('/combined-candidate.mjs'),
        };
        const observationApis = {
          baseline: await import('/combined-baseline-public.mjs'),
          candidate: await import('/combined-candidate-public.mjs'),
        };
        const { runCombinedPipeline } =
          await import('/tests/typeshade/multi-video-combined-cases.mjs');
        if (typeof variants.candidate.reconstructMultiVideoOwnedGpu !== 'function')
          throw new Error('Candidate must export the owned multi-video pipeline');
        const devices = {},
          observationDevices = {},
          rows = [];
        try {
          for (const [variant, api] of Object.entries(variants))
            devices[variant] = await api.ShadeDevice.open({ backend, poolBytes: 0 });
          for (const [variant, api] of Object.entries(observationApis))
            observationDevices[variant] = await api.ShadeDevice.open({ backend, poolBytes: 0 });
          if (warmup)
            for (const [variant, api] of Object.entries(variants))
              await runCombinedPipeline(api, devices[variant], serialized, options, false, {
                api: observationApis[variant],
                device: observationDevices[variant],
              });
          for (let pair = 0; pair < pairs; pair++)
            for (const variant of pair % 2
              ? ['candidate', 'baseline']
              : ['baseline', 'candidate']) {
              rows.push({
                pair,
                variant,
                ...(await runCombinedPipeline(
                  variants[variant],
                  devices[variant],
                  serialized,
                  options,
                  false,
                  { api: observationApis[variant], device: observationDevices[variant] },
                )),
              });
              await fetch('/upload/progress.json', {
                method: 'PUT',
                body: JSON.stringify({ rows, owned: null, complete: false }),
              });
            }
          // The consuming worker route has extra lifetime evidence, separately from paired timing.
          const owned = await runCombinedPipeline(
            variants.candidate,
            devices.candidate,
            serialized,
            options,
            true,
            { api: observationApis.candidate, device: observationDevices.candidate },
          );
          await fetch('/upload/progress.json', {
            method: 'PUT',
            body: JSON.stringify({ rows, owned, complete: true }),
          });
          return {
            backend,
            devices: Object.fromEntries(
              Object.entries(devices).map(([name, device]) => [name, device.info]),
            ),
            observationDevices: Object.fromEntries(
              Object.entries(observationDevices).map(([name, device]) => [name, device.info]),
            ),
            rows,
            owned,
          };
        } finally {
          for (const device of Object.values(devices)) await device.dispose();
          for (const device of Object.values(observationDevices)) await device.dispose();
        }
      },
      { backend: values.backend, serialized, options, pairs, warmup: values.warmup },
      {
        timeoutMs: timeout * 1000,
        uploads: temporary,
        serve: {
          '/combined-baseline.mjs': modules.baseline.path,
          '/combined-candidate.mjs': modules.candidate.path,
          '/combined-baseline-public.mjs': modules.baselinePublic.path,
          '/combined-candidate-public.mjs': modules.candidatePublic.path,
        },
      },
    );
  } catch (error) {
    let partial;
    try {
      partial = JSON.parse(await readFile(join(temporary, 'progress.json'), 'utf8'));
    } catch {}
    if (values.out)
      await writeFile(
        resolve(values.out),
        JSON.stringify(
          {
            exactEvidenceEqual: false,
            incomplete: true,
            failure: String(error),
            fixture: { videos, viewsPerVideo: views, pointsPerCloud: points, seed: 41, options },
            modules: Object.fromEntries(
              Object.entries(modules).map(([variant, { sha256 }]) => [variant, sha256]),
            ),
            ...(partial ?? {}),
          },
          null,
          2,
        ) + '\n',
      );
    throw error;
  }
  const { result, errors } = browserResult;
  let validationFailure;
  try {
    assert.deepEqual(errors, []);
    const reference = result.rows[0];
    for (const row of result.rows) {
      assert.equal(row.sourceContainerLength, serialized.length, 'public input array changed');
      assert.equal(row.digests.inputBefore, row.digests.inputAfter, 'source feature bytes changed');
      assert.deepEqual(row.digests, reference.digests, 'pipeline output bits differ');
      assert.equal(row.counters.liveBytes, 0, 'active GPU bytes leaked');
      assert.equal(row.counters.liveBuffers, 0, 'active GPU handles leaked');
      assert.equal(row.observationCounters.liveBytes, 0, 'observation GPU bytes leaked');
      assert.equal(row.observationCounters.liveBuffers, 0, 'observation GPU handles leaked');
      assert.equal(row.counts.registered, serialized.length, 'not all generated frames registered');
      assert.ok(
        row.counts.landmarks > 0 && row.counts.bundles > 0,
        'fixture did not exercise mapping and BA',
      );
    }
    const owned = result.owned;
    assert.equal(owned.sourceContainerLength, 0, 'owned input container was not consumed');
    assert.equal(
      owned.digests.inputBefore,
      owned.digests.inputAfter,
      'owned route changed feature bytes',
    );
    for (const key of [
      'identities',
      'graphSummary',
      'components',
      'callbacks',
      'observations',
      'inputBefore',
    ])
      assert.equal(owned.digests[key], reference.digests[key], `owned ${key} output bits differ`);
    assert.equal(owned.counters.liveBytes, 0);
    assert.equal(owned.counters.liveBuffers, 0);
    assert.equal(owned.observationCounters.liveBytes, 0);
    assert.equal(owned.observationCounters.liveBuffers, 0);
    assert.equal(owned.hostRetention.featuresAfterMappingBytes, 0);
    assert.equal(owned.hostRetention.graphMatchesAfterMappingBytes, 0);
    assert.ok(
      owned.hostRetention.featuresBeforeGraphBytes > owned.hostRetention.featuresAfterGraphBytes,
    );
  } catch (error) {
    validationFailure = error;
  }
  const median = (items) => {
    const ordered = [...items].sort((a, b) => a - b);
    return (
      (ordered[Math.floor((ordered.length - 1) / 2)] + ordered[Math.floor(ordered.length / 2)]) / 2
    );
  };
  const medianMs = Object.fromEntries(
    ['baseline', 'candidate'].map((variant) => [
      variant,
      median(result.rows.filter((row) => row.variant === variant).map((row) => row.combinedMs)),
    ]),
  );
  const receipt = {
    scope:
      'Generated RootSIFT tracks: graph/retrieval/matching/verification → incremental SfM/BA → explicit observation evaluation',
    excludes: [
      'video decode',
      'SIFT extraction',
      'Dense MVS',
      'rendering',
      'browser/process RSS',
      'actual 15-video runtime',
    ],
    timing: values.warmup
      ? 'One complete warmup per public variant, then alternating pairs'
      : 'Includes first-use shader compilation; one-pair differences are not steady-state speed evidence',
    ownedTiming: 'Separate validation run, not included in public paired median',
    observationDevice:
      "Explicit final observation check uses each revision's public bundle and a separate device/runtime; mapper observation stages stay in its internal runtime",
    hostLifetime:
      'Logical unique owned feature and graph-match backing allocations; external test references and browser/runtime/GPU memory excluded',
    fixture: {
      videos,
      viewsPerVideo: views,
      pointsPerCloud: points,
      keyframes: serialized.length,
      seed: 41,
      options,
    },
    modules: Object.fromEntries(
      Object.entries(modules).map(([variant, { sha256 }]) => [variant, sha256]),
    ),
    pairs,
    timeoutSeconds: timeout,
    medianMs,
    observedChangePercent: 100 * (medianMs.candidate / medianMs.baseline - 1),
    exactEvidenceEqual: !validationFailure,
    ...(validationFailure ? { validationFailure: String(validationFailure) } : {}),
    ...result,
  };
  const json = JSON.stringify(receipt, null, 2) + '\n';
  if (values.out) await writeFile(resolve(values.out), json);
  else process.stdout.write(json);
  if (values['summary-out'] && !validationFailure)
    await writeFile(
      resolve(values['summary-out']),
      JSON.stringify(summarizeCombinedReceipt(receipt), null, 2) + '\n',
    );
  if (validationFailure) throw validationFailure;
} finally {
  await rm(temporary, { recursive: true, force: true });
}
