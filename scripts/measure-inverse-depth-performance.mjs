// Paired full-observation replay. Only aggregate statistics go into summary.json.
// Worker files contain private map geometry and must not be committed or published.
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { resolve, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import assert from 'node:assert/strict';

const args = process.argv.slice(2),
  flag = (name, fallback) => (args.includes(name) ? args[args.indexOf(name) + 1] : fallback),
  repetitions = Number(flag('--repetitions', '5')),
  syntheticViews = flag('--synthetic-views'),
  repo = fileURLToPath(new URL('..', import.meta.url)),
  worker = fileURLToPath(new URL('./measure-inverse-depth-scaling.mjs', import.meta.url)),
  hash = (value) => createHash('sha256').update(value).digest('hex');
if (!flag('--records') || !flag('--out') || !flag('--baseline'))
  throw new Error(
    'Usage: --records <private-record-root> --out <private-result-directory> --baseline <frozen-bundle> [--repetitions 5] [--case <record>] [--synthetic-views 128]',
  );
if (!Number.isSafeInteger(repetitions) || repetitions < 1)
  throw new RangeError('repetitions must be a positive safe integer');
if (
  syntheticViews !== undefined &&
  (!Number.isSafeInteger(Number(syntheticViews)) || Number(syntheticViews) < 4)
)
  throw new RangeError('synthetic-views must be a safe integer of at least four');
const recordsRoot = resolve(flag('--records')),
  out = resolve(flag('--out')),
  baseline = resolve(flag('--baseline')),
  manifest = JSON.parse(
    await readFile(
      new URL(
        '../docs/experiments/general-video/inverse-depth-scaling-inputs.json',
        import.meta.url,
      ),
      'utf8',
    ),
  ),
  available = [
    ...manifest.records,
    ...(syntheticViews ? [{ record: `synthetic-local-${syntheticViews}`, synthetic: true }] : []),
  ],
  selected = available.filter((r) => !flag('--case') || r.record === flag('--case'));
if (!selected.length) throw new Error('No matching record was selected');
await mkdir(out, { recursive: true });
const { build } = await import('esbuild'),
  candidate = join(out, '.candidate.mjs');
await build({
  entryPoints: [new URL('../src/reconstruction/inverse-depth-bundle.ts', import.meta.url).pathname],
  outfile: candidate,
  bundle: true,
  platform: 'node',
  format: 'esm',
});
const sourceRevision = execFileSync('git', ['rev-parse', 'HEAD'], {
    cwd: repo,
    encoding: 'utf8',
  }).trim(),
  sourceTrackedChanges = execFileSync('git', ['diff', '--name-only', 'HEAD', '--', 'src'], {
    cwd: repo,
    encoding: 'utf8',
  })
    .trim()
    .split('\n')
    .filter(Boolean),
  sourceFiles = [
    'src/reconstruction/inverse-depth-bundle.ts',
    'src/reconstruction/sparse-schur.ts',
    'src/geometry/inverse-depth.ts',
  ],
  sourceHashes = Object.fromEntries(
    await Promise.all(
      sourceFiles.map(async (name) => [name, hash(await readFile(join(repo, name)))]),
    ),
  ),
  baselineHash = hash(await readFile(baseline)),
  candidateHash = hash(await readFile(candidate)),
  absoluteTolerance = 1e-10,
  relativeTolerance = 1e-8,
  results = [],
  failures = [];

function scalarStatistics() {
  return {
    comparedScalars: 0,
    differingScalars: 0,
    maxAbsoluteDifference: 0,
    maxRelativeDifference: 0,
    toleranceViolations: 0,
  };
}
function compareNumeric(a, b, stats) {
  assert.equal(typeof a, 'number', 'Numeric comparison received a missing baseline value');
  assert.equal(typeof b, 'number', 'Numeric comparison received a missing optimized value');
  assert.ok(Number.isFinite(a) && Number.isFinite(b), 'Non-finite comparison value');
  const absolute = Math.abs(a - b),
    scale = Math.max(Math.abs(a), Math.abs(b));
  stats.comparedScalars++;
  if (a !== b) stats.differingScalars++;
  stats.maxAbsoluteDifference = Math.max(stats.maxAbsoluteDifference, absolute);
  stats.maxRelativeDifference = Math.max(
    stats.maxRelativeDifference,
    absolute / Math.max(scale, Number.MIN_VALUE),
  );
  if (absolute > absoluteTolerance + relativeTolerance * scale) stats.toleranceViolations++;
}
function compareArray(a, b, stats) {
  if (a === null || b === null) {
    assert.equal(a, b, 'Covariance availability changed');
    return;
  }
  assert.ok(Array.isArray(a) && Array.isArray(b), 'Missing full-value array');
  assert.equal(a.length, b.length, 'Full-value array length changed');
  for (let i = 0; i < a.length; i++) compareNumeric(a[i], b[i], stats);
}
function comparePair(a, b) {
  const statistics = Object.fromEntries(
    [
      'cameraParameters',
      'pointParameters',
      'pointCovariances',
      'bearingCovariances',
      'referenceCharts',
      'costs',
    ].map((name) => [name, scalarStatistics()]),
  );
  for (const name of ['record', 'sha256', 'views', 'points', 'admitted', 'iterations'])
    assert.equal(a[name], b[name], `Worker metadata changed: ${name}`);
  for (const name of [
    'acceptedSteps',
    'observations',
    'inputObservations',
    'cameraParameters',
    'linearSolver',
    'deferredTrackIds',
  ])
    assert.deepEqual(a.diagnostics[name], b.diagnostics[name], `Solver state changed: ${name}`);
  assert.deepEqual(a.uncertainty, b.uncertainty, 'Uncertainty conditioning changed');
  compareNumeric(a.diagnostics.initialCost, b.diagnostics.initialCost, statistics.costs);
  compareNumeric(a.diagnostics.finalCost, b.diagnostics.finalCost, statistics.costs);
  assert.ok(Array.isArray(a.cameraValues) && Array.isArray(b.cameraValues));
  assert.equal(a.cameraValues.length, b.cameraValues.length, 'Camera count changed');
  const cameras = new Map(a.cameraValues.map((v) => [v.id, v]));
  assert.equal(cameras.size, a.cameraValues.length, 'Duplicate baseline camera id');
  assert.equal(new Set(b.cameraValues.map((v) => v.id)).size, b.cameraValues.length);
  for (const right of b.cameraValues) {
    const left = cameras.get(right.id);
    assert.ok(left, 'Camera identity changed');
    assert.deepEqual(Object.keys(left.camera).sort(), Object.keys(right.camera).sort());
    for (const key of Object.keys(left.camera))
      compareNumeric(left.camera[key], right.camera[key], statistics.cameraParameters);
    compareArray(left.R, right.R, statistics.cameraParameters);
    compareArray(left.t, right.t, statistics.cameraParameters);
  }
  assert.equal(a.pointValues.length, b.pointValues.length, 'Point count changed');
  const points = new Map(a.pointValues.map((p) => [p.id, p]));
  assert.equal(points.size, a.pointValues.length, 'Duplicate baseline point id');
  assert.equal(new Set(b.pointValues.map((p) => p.id)).size, b.pointValues.length);
  for (const right of b.pointValues) {
    const left = points.get(right.id);
    assert.ok(left, 'Point identity changed');
    for (const key of [
      'referenceViewId',
      'directionOnly',
      'informationDeferred',
      'observationCount',
    ])
      assert.equal(left[key], right[key], `Point state changed: ${key}`);
    compareArray(left.parameters, right.parameters, statistics.pointParameters);
    compareArray(left.covariance, right.covariance, statistics.pointCovariances);
    compareArray(left.bearingCovariance, right.bearingCovariance, statistics.bearingCovariances);
    compareArray(left.reference.R, right.reference.R, statistics.referenceCharts);
    compareArray(left.reference.t, right.reference.t, statistics.referenceCharts);
  }
  return {
    passed: Object.values(statistics).every((s) => s.toleranceViolations === 0),
    statistics,
  };
}
function summarize(samples) {
  const values = [...samples].sort((a, b) => a - b),
    middle = Math.floor(values.length / 2);
  return {
    samples,
    median: values.length % 2 ? values[middle] : (values[middle - 1] + values[middle]) / 2,
    min: values[0],
    max: values.at(-1),
  };
}
async function run(record, version, round) {
  const command = [
    '--expose-gc',
    worker,
    '--worker',
    record.synthetic
      ? 'synthetic'
      : join(recordsRoot, 'record-' + record.record, 'typecv-reconstruction.json'),
    '--record',
    record.record,
    '--solver',
    'auto',
    '--iterations',
    '1',
    '--full-values',
    '--candidate',
    version === 'baseline' ? baseline : candidate,
  ];
  if (record.synthetic) command.push('--synthetic-views', syntheticViews);
  console.error(`Starting ${record.record} / pair ${round + 1} / ${version}`);
  const value = JSON.parse(
    execFileSync(process.execPath, command, {
      cwd: repo,
      encoding: 'utf8',
      maxBuffer: 64 * 1024 ** 2,
      timeout: 300000,
    }),
  );
  assert.ok(Number.isFinite(value.elapsedMs) && value.elapsedMs >= 0, 'Invalid worker duration');
  assert.ok(Number.isFinite(value.maxRssKiB) && value.maxRssKiB > 0, 'Invalid worker RSS');
  if (!record.synthetic) assert.equal(value.sha256, record.sha256, 'Frozen input hash changed');
  await writeFile(
    join(out, `${record.record}-${round + 1}-${version}.private.json`),
    JSON.stringify(value),
  );
  return value;
}
for (const record of selected) {
  const timings = { baseline: [], optimized: [] },
    memory = { baseline: [], optimized: [] },
    pairs = [];
  let metadata;
  for (let round = 0; round < repetitions; round++) {
    const order = round % 2 ? ['optimized', 'baseline'] : ['baseline', 'optimized'],
      pair = {};
    try {
      for (const version of order) {
        pair[version] = await run(record, version, round);
        timings[version].push(pair[version].elapsedMs);
        memory[version].push(pair[version].maxRssKiB);
      }
      const numerical = comparePair(pair.baseline, pair.optimized);
      if (!numerical.passed) failures.push(`${record.record}: pair ${round + 1} numeric tolerance`);
      const currentMetadata = {
        record: record.record,
        sha256: pair.baseline.sha256,
        views: pair.baseline.views,
        points: pair.baseline.points,
        admitted: pair.baseline.admitted,
        observations: pair.baseline.diagnostics.observations,
        inputObservations: pair.baseline.diagnostics.inputObservations,
        cameraParameters: pair.baseline.diagnostics.cameraParameters,
        acceptedSteps: pair.baseline.diagnostics.acceptedSteps,
        linearSolver: pair.baseline.diagnostics.linearSolver,
        uncertainty: pair.baseline.uncertainty,
      };
      if (metadata)
        assert.deepEqual(currentMetadata, metadata, 'Input/solver state varied across repetitions');
      metadata = currentMetadata;
      pairs.push({
        round: round + 1,
        order,
        baselineMs: pair.baseline.elapsedMs,
        optimizedMs: pair.optimized.elapsedMs,
        baselineMaxRssKiB: pair.baseline.maxRssKiB,
        optimizedMaxRssKiB: pair.optimized.maxRssKiB,
        numerical,
      });
    } catch (error) {
      failures.push(`${record.record}: pair ${round + 1}: ${error.message}`);
      pairs.push({ round: round + 1, order, passed: false, error: error.message });
      break;
    }
  }
  const timingMs = Object.fromEntries(
      Object.entries(timings)
        .filter(([, s]) => s.length)
        .map(([v, s]) => [v, summarize(s)]),
    ),
    peakRssKiB = Object.fromEntries(
      Object.entries(memory)
        .filter(([, s]) => s.length)
        .map(([v, s]) => [v, summarize(s)]),
    );
  results.push({
    ...(metadata ?? { record: record.record, sha256: record.sha256 }),
    timingMs,
    peakRssKiB,
    medianTimeReductionFraction:
      timingMs.baseline && timingMs.optimized
        ? 1 - timingMs.optimized.median / timingMs.baseline.median
        : null,
    medianPeakRssReductionFraction:
      peakRssKiB.baseline && peakRssKiB.optimized
        ? 1 - peakRssKiB.optimized.median / peakRssKiB.baseline.median
        : null,
    pairs,
  });
}
const summary = {
  createdAt: new Date().toISOString(),
  protocol:
    'Fresh Node process for every version/round; paired serial workers with alternating baseline/optimized order. Entire frozen finite sparse observations, fixed K, sigma=1px, Huber=3px, auto solver, one permitted joint iteration, camera budget 1536. No RGB rematching, observation pruning, gate changes or covariance approximation.',
  timingMeaning:
    'Worker elapsedMs covers the solver only, after explicit GC. maxRSS covers the whole worker including module/parse/full-value extraction. These are fresh-process samples, not steady-state/UI/hardware-GPU throughput.',
  provenance:
    'Already-triangulated sparse exports and synthetic supplied correspondences. No source RGB/Dense MVS/ground-truth accuracy comparison. Only summary.json is suitable for repository publication; *.private.json contain map geometry.',
  command:
    'node scripts/measure-inverse-depth-performance.mjs --records <private-record-root> --out <private-result-directory> --baseline <frozen-b036bd8-bundle> --repetitions ' +
    repetitions +
    (flag('--case') ? ' --case ' + flag('--case') : '') +
    (syntheticViews ? ' --synthetic-views ' + syntheticViews : ''),
  sourceRevision,
  sourceTrackedChanges,
  sourceHashes,
  candidateHash,
  baselineHash,
  runtime: process.version,
  platform: process.platform,
  repetitions,
  absoluteTolerance,
  relativeTolerance,
  passed: failures.length === 0,
  failures,
  results,
};
await writeFile(join(out, 'summary.json'), JSON.stringify(summary, null, 2) + '\n');
console.error(JSON.stringify({ passed: summary.passed, records: results.length, failures }));
if (failures.length) process.exitCode = 1;
