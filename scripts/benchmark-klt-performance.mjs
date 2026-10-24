import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { basename, dirname, join, resolve } from 'node:path';
import { promisify } from 'node:util';

const option = (key, fallback) =>
  process.argv.find((value) => value.startsWith(`--${key}=`))?.slice(key.length + 3) ?? fallback;
const root = resolve(import.meta.dirname, '..'),
  baselineOption = option('baseline-bundle', '');
if (!baselineOption) throw new Error('--baseline-bundle=<compiled local bundle> is required');
const baselineBundle = resolve(baselineOption),
  currentBundle = join(root, 'dist/gpu/index.js'),
  video = option('video', ''),
  repeats = Number(option('repeats', '3')),
  cache = option('pyramid-cache', 'auto'),
  baselineCache = option('baseline-pyramid-cache', cache),
  batchPassesPolicy = option('batch-passes', 'auto'),
  poolBytes = Number(option('pool-bytes', '0')),
  output = resolve(option('output', 'docs/experiments/general-video/klt-performance.json'));
if (!Number.isSafeInteger(repeats) || repeats < 1 || repeats > 5)
  throw new Error('repeats must be an integer from 1 to 5');
for (const [name, policy] of [
  ['pyramid-cache', cache],
  ['baseline-pyramid-cache', baselineCache],
])
  if (!['0', '1', '2', 'auto'].includes(policy)) throw new Error(`${name} must be 0, 1, 2 or auto`);
if (!Number.isSafeInteger(poolBytes) || poolBytes < 0)
  throw new Error('pool-bytes must be a nonnegative integer');
if (!['0', '1', 'auto'].includes(batchPassesPolicy))
  throw new Error('batch-passes must be 0, 1 or auto');
const knownCases = [
    'render-translation',
    'render-translation-omission',
    'render-rotation',
    'render-low-parallax',
    'graham-hall-cached-rgb-features',
    ...(video ? ['user-video-window'] : []),
  ],
  selected = option('cases', knownCases.join(','))
    .split(',')
    .map((name) => name.trim());
if (
  !selected.length ||
  new Set(selected).size !== selected.length ||
  selected.some((name) => !knownCases.includes(name))
)
  throw new Error('cases must be distinct supported inputs; user-video-window requires --video');
const sha = (bytes) => createHash('sha256').update(bytes).digest('hex'),
  bundleHashes = Object.fromEntries(
    await Promise.all(
      [
        ['baseline', baselineBundle],
        ['current', currentBundle],
      ].map(async ([mode, path]) => [mode, sha(await readFile(path))]),
    ),
  ),
  execute = promisify(execFile),
  temporary = await mkdtemp(join(tmpdir(), 'typecv-klt-performance-')),
  runs = [];
let reference;
const inputsWithoutTimings = (inputs) =>
  inputs.map((input) => {
    const copy = structuredClone(input);
    if (copy.videoWindow) delete copy.videoWindow.preparationMs;
    return copy;
  });
const trackingWithoutCachePolicy = ({ pyramidCachePolicy, pyramidCache, ...options }) => options;
const modelSummary = (model) => ({
  modelSha256: model.modelSha256 ?? null,
  registeredFrames: model.registeredFrames,
  landmarks: model.landmarks,
  observations: model.observations,
  tracks: model.tracks,
  trackedObservations: model.trackedObservations,
  pendingTracks: model.pendingTracks,
  initialPair: model.initialPair ?? null,
  initialization: model.initialization ?? 'initialized',
});
const quality = (row) => ({
  measurementSha256: row.measurementSha256,
  baseline: modelSummary(row.baseline),
  expanded: modelSummary(row.expanded),
  inputFeatureCounts: row.inputFeatureCounts,
  outputFeatureCounts: row.outputFeatureCounts,
  cohortComparison: row.cohortComparison,
});
const compactRow = (row) => ({
  name: row.name,
  ...quality(row),
  cache: row.expansion.cache ?? null,
  kernels: row.expansion.kernels,
  timings: {
    expansionMs: row.expansion.elapsedMs,
    totalMs: row.totalMs,
    trackingMs: row.expansion.trackingMs,
    geometryPlanningMs: row.expansion.geometryPlanningMs,
    hostValidationMs: row.expansion.hostValidationMs,
    baselineTotalMs: row.baselineTotalMs,
    refinementMs: row.expanded.elapsedMs,
    videoPreparationMs: row.videoWindow?.preparationMs ?? 0,
    decodeMs: row.frontend.decodeMs,
    featureMs: row.frontend.featureMs,
    matchingMs: row.frontend.matchingMs,
  },
  counts: {
    attempts: row.expansion.attempts,
    newCoordinates: row.expansion.newCoordinates,
    reads: row.expansion.reads,
    readWaves: row.expansion.readWaves,
    planCalls: row.expansion.planCalls,
    runtimeSubmissions: row.expansion.runtimeSubmissions,
    uploadBytes: row.expansion.uploadBytes,
    readbackBytes: row.expansion.readbackBytes,
    residentSnapshotBytes: row.expansion.residentSnapshotBytes,
  },
});
const statistics = (values) => {
  assert.ok(values.every(Number.isFinite), 'performance counters must be finite');
  const sorted = values.slice().sort((a, b) => a - b),
    middle = Math.floor(sorted.length / 2);
  return {
    median: sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2,
    min: sorted[0],
    max: sorted.at(-1),
  };
};
try {
  for (let repeat = 0; repeat < repeats; repeat++) {
    const order = repeat % 2 ? ['current', 'baseline'] : ['baseline', 'current'];
    for (const [position, mode] of order.entries()) {
      const reportPath = join(temporary, `${repeat}-${mode}.json`),
        args = [
          'scripts/benchmark-observation-expansion.mjs',
          '--backend=webgpu',
          '--refinement=resume',
          '--patch-warp=1',
          '--surface-warp=1',
          '--multi-view=1',
          `--batch-passes=${batchPassesPolicy}`,
          `--pyramid-cache=${mode === 'baseline' ? baselineCache : cache}`,
          `--pool-bytes=${poolBytes}`,
          `--cases=${selected.join(',')}`,
          `--output=${reportPath}`,
          ...(mode === 'baseline' ? [`--gpu-bundle=${baselineBundle}`] : []),
          ...(video ? [`--video=${resolve(video)}`] : []),
        ],
        started = performance.now();
      await execute(process.execPath, args, { cwd: root, timeout: 1000000, maxBuffer: 1 << 24 });
      const processMs = performance.now() - started,
        report = JSON.parse(await readFile(reportPath, 'utf8')),
        inputs = inputsWithoutTimings(report.inputs);
      assert.equal(
        report.gpuBundle.sha256,
        bundleHashes[mode],
        'compiled bundle changed during benchmark',
      );
      assert.equal(
        report.trackingOptions.pyramidCachePolicy,
        mode === 'baseline' ? baselineCache : cache,
        'requested cache policy was not recorded',
      );
      assert.equal(
        report.trackingOptions.batchPassesPolicy,
        batchPassesPolicy,
        'requested batch-pass policy was not recorded',
      );
      assert.equal(
        report.cases.length,
        selected.length,
        'benchmark did not return every requested case',
      );
      assert.deepEqual(new Set(report.cases.map((row) => row.name)), new Set(selected));
      assert.equal(
        report.diagnostics.memory.liveBytes - report.diagnostics.memory.pooledBytes,
        0,
        'active buffers must be released',
      );
      assert.equal(report.afterDisposeMemory.liveBytes, 0, 'disposal frees pooled buffers too');
      assert.equal(report.afterDisposeMemory.pendingReads, 0, 'disposal settles reads');
      assert.equal(report.diagnostics.memory.pendingReads, 0, 'readbacks must be settled');
      for (const row of report.cases) {
        assert.match(row.measurementSha256, /^[a-f0-9]{64}$/, row.name + ' measurement hash');
        for (const model of [row.baseline, row.expanded])
          if (model.registeredFrames || model.landmarks)
            assert.match(model.modelSha256, /^[a-f0-9]{64}$/, row.name + ' finite model hash');
      }
      if (!reference) {
        assert.equal(mode, 'baseline');
        reference = { ...report, inputs };
      } else {
        for (const field of [
          'controls',
          'executionOptions',
          'adapter',
          'calibration',
          'sourceHashes',
        ])
          assert.deepEqual(
            report[field],
            reference[field],
            `${mode} repeat${repeat + 1}: ${field} changed`,
          );
        assert.deepEqual(
          trackingWithoutCachePolicy(report.trackingOptions),
          trackingWithoutCachePolicy(reference.trackingOptions),
          'geometry/tracking controls changed',
        );
        assert.deepEqual(
          inputs,
          reference.inputs,
          `${mode} repeat${repeat + 1}: input evidence changed`,
        );
        for (const row of report.cases) {
          const expected = reference.cases.find((candidate) => candidate.name === row.name);
          assert.deepEqual(
            quality(row),
            quality(expected),
            `${mode} repeat${repeat + 1}: ${row.name} measurement/model/cohort changed`,
          );
        }
      }
      const run = {
        repeat: repeat + 1,
        mode,
        position: position + 1,
        processMs,
        adapter: report.adapter,
        trackingOptions: report.trackingOptions,
        memory: report.diagnostics.memory,
        afterDisposeMemory: report.afterDisposeMemory,
        cases: report.cases.map(compactRow),
      };
      runs.push(run);
      console.log(
        JSON.stringify({
          repeat: run.repeat,
          mode,
          position: run.position,
          cases: run.cases.map((row) => ({
            name: row.name,
            expansionMs: row.timings.expansionMs,
            totalMs: row.timings.totalMs,
            readWaves: row.counts.readWaves,
          })),
        }),
      );
    }
  }
  for (const [mode, path] of [
    ['baseline', baselineBundle],
    ['current', currentBundle],
  ])
    assert.equal(
      sha(await readFile(path)),
      bundleHashes[mode],
      'compiled bundle changed during benchmark',
    );
  const summaries = reference.cases.map(({ name }) => {
    const modes = Object.fromEntries(
      ['baseline', 'current'].map((mode) => {
        const rows = runs
          .filter((run) => run.mode === mode)
          .map((run) => run.cases.find((row) => row.name === name));
        return [
          mode,
          {
            timings: Object.fromEntries(
              Object.keys(rows[0].timings).map((key) => [
                key,
                statistics(rows.map((row) => row.timings[key])),
              ]),
            ),
            counts: Object.fromEntries(
              Object.keys(rows[0].counts).map((key) => [
                key,
                statistics(rows.map((row) => row.counts[key])),
              ]),
            ),
          },
        ];
      }),
    );
    return {
      name,
      ...modes,
      speedup: Object.fromEntries(
        ['expansionMs', 'totalMs', 'trackingMs'].map((key) => [
          key,
          {
            ratio: modes.baseline.timings[key].median / modes.current.timings[key].median,
            timeReductionPercent:
              100 * (1 - modes.current.timings[key].median / modes.baseline.timings[key].median),
          },
        ]),
      ),
    };
  });
  const result = {
    recordedAt: new Date().toISOString(),
    repeats,
    backend: 'webgpu',
    methodology:
      'Sequential fresh browser per mode; baseline/current order alternates by repeat; median and min/max use same-mode runs. All runs must exactly match the first baseline inputs, geometry/tracking controls except the explicitly recorded cache policy, measured-observation hash, finite-model hashes and cohort evaluation.',
    baselineBundle: { file: basename(baselineBundle), sha256: bundleHashes.baseline },
    currentBundle: { file: 'dist/gpu/index.js', sha256: bundleHashes.current },
    benchmarkHashes: {
      wrapper: sha(await readFile(new URL(import.meta.url))),
      observationBenchmark: sha(
        await readFile(join(root, 'scripts/benchmark-observation-expansion.mjs')),
      ),
    },
    sourceHashes: reference.sourceHashes,
    implementationCommit: reference.implementationCommit,
    controls: reference.controls,
    trackingOptions: Object.fromEntries(
      ['baseline', 'current'].map((mode) => [
        mode,
        runs.find((run) => run.mode === mode).trackingOptions,
      ]),
    ),
    executionOptions: reference.executionOptions,
    inputs: reference.inputs,
    calibration: reference.calibration,
    adapter: reference.adapter,
    exactQualityParity: true,
    limitations:
      'Adapter metadata is reported without independent physical-device verification. Browser stage totals exclude browser startup/transport and synthetic fixture generation; video preparation is separate. Cached Graham Hall detector/matcher costs are excluded. The same requested batch-pass policy is forwarded to both modes; the historical baseline bundle may not implement that option and keeps its original pass scheduling, as reflected in direct plan counters. Plan counters wrap direct device.plan calls and sum their returned runtime submission reports, not composite reports or queue fences. Read/upload counts are application/Resident counters, not physical bus bytes. Faster execution does not establish accuracy beyond the declared input conditions.',
    summaries,
    runs,
  };
  await mkdir(dirname(output), { recursive: true });
  await writeFile(output, JSON.stringify(result, null, 2) + '\n');
  console.log(
    JSON.stringify({
      output,
      exactQualityParity: true,
      speedups: summaries.map(({ name, speedup }) => ({ name, ...speedup })),
    }),
  );
} finally {
  await rm(temporary, { recursive: true, force: true });
}
