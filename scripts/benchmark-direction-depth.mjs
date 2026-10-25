import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { cpus } from 'node:os';
import { dirname, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { directionObservationFixture } from '../tests/fixtures/direction-observations.ts';
import { generalVideoFixture } from '../tests/fixtures/general-video.ts';
import { compareSeedGauge } from './geometry-comparison.mjs';

const arg = (name, fallback) =>
  process.argv.find((v) => v.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;
const baseline = arg('baseline'),
  baselineDirectional = arg('baseline-directional', 'false') === 'true',
  repeats = Number(arg('repeats', '2')),
  output = resolve(arg('output', 'docs/experiments/general-video/direction-depth-cpu.json'));
if (!baseline || !Number.isInteger(repeats) || repeats < 2 || repeats > 12 || repeats % 2)
  throw new RangeError('--baseline=/path/to/worktree and even --repeats=2..12 required');
const baselineRoot = resolve(baseline),
  root = resolve(import.meta.dirname, '..'),
  implementations = await Promise.all(
    [baselineRoot, root].map(
      (dir) => import(pathToFileURL(resolve(dir, 'src/reconstruction/mapper.ts')).href),
    ),
  );
const git = (dir, ...args) =>
    execFileSync('git', ['-C', dir, ...args], { encoding: 'utf8' }).trim(),
  sha = (v) => createHash('sha256').update(v).digest('hex'),
  digest = (v) => sha(JSON.stringify(v, (_, x) => (ArrayBuffer.isView(x) ? [...x] : x))),
  median = (a) => {
    const s = [...a].sort((x, y) => x - y);
    return (s[(s.length - 1) >> 1] + s[s.length >> 1]) / 2;
  };
const cases = [];
const add = (name, options, controls = {}) =>
  cases.push({ name, fixture: directionObservationFixture(options), options, controls });
for (const noisePx of [0, 0.2, 1])
  for (const directionalCount of [0, 30, 60, 180])
    add(`finite60-direction${directionalCount}-noise${noisePx}`, { noisePx, directionalCount });
for (const noisePx of [0, 0.2, 1]) {
  add(`direction-only-noise${noisePx}`, { noisePx, finiteCount: 0 });
  add(`estimated-focal-noise${noisePx}`, { noisePx }, { focalSearch: true });
}
add('rotation-with-finite-and-direction', { scenario: 'rotation' });
add('rotation-and-zoom', { scenario: 'rotation', zoom: true });
add('translation-and-known-zoom', { zoom: true, noisePx: 0.2 });
add('distant-finite', { finiteCount: 0, distantDepth: 1000000 });
add('screen-fixed-unmasked', { screenFixed: true });
add('screen-fixed-masked', { screenFixed: true });
const masked = cases.at(-1);
masked.controls.cameraObservationMasks = masked.fixture.input.images.map((image) => ({
  viewId: image.id,
  source: 'ground-truth',
  values: Uint8Array.from({ length: 120 }, (_, i) => (i < 60 ? 1 : 0)),
}));
add('unrelated-correspondences', {});
const unrelated = cases.at(-1);
for (const p of unrelated.fixture.input.pairs)
  p.matches = p.matches.map((x, i) => (i % 2 ? (13 * x + 7) % 120 : x));
const legacy = generalVideoFixture('rotation-start');
legacy.input.pairs = legacy.input.pairs.filter((p) => p.second - p.first === 1);
for (const image of legacy.input.images)
  image.keypoints = image.keypoints.map(
    (x, i) => x + 0.7 * Math.sin((i + 1) * (image.id + 3) * 1.37),
  );
cases.push({
  name: 'legacy-noisy-rotation-prefix-refinement',
  fixture: {
    input: legacy.input,
    truth: { views: legacy.truth.views, finitePoints: legacy.truth.points },
  },
  options: { directionalCount: 0, noise: '0.7px deterministic sinusoid' },
  controls: {
    pairRefinementIterations: 10,
    maxRelativeDepthStdDev: 0.1,
    bundleIterations: 1,
    finalBundleIterations: 2,
  },
});

function quality(result, failure, f, options) {
  const tracking = result?.tracking ?? failure?.tracking,
    model = result?.reconstruction,
    finiteCount = options.finiteCount ?? 60;
  const falseFinite =
    model?.landmarks.filter((p) => p.observations.some((o) => o.featureId >= finiteCount)).length ??
    0;
  return {
    status: model ? 'map' : 'initialization-deferred',
    registered: model?.views.length ?? 0,
    totalViews: f.input.images.length,
    points: model?.landmarks.length ?? 0,
    finitePoints: (model?.landmarks.length ?? 0) - falseFinite,
    falseFinite,
    falseFiniteRate:
      options.directionalCount === 0 ? null : falseFinite / (options.directionalCount ?? 60),
    pending: tracking?.untriangulatedTrackIds.length ?? null,
    pendingDirectional:
      tracking?.untriangulatedTrackIds.filter((id) =>
        tracking.tracks[id].observations.every((o) => o.featureId >= finiteCount),
      ).length ?? null,
    focalMaxRelativeError: model
      ? Math.max(
          ...model.views.map((v) =>
            Math.abs(v.camera.fx / f.truth.views.find((t) => t.id === v.id).camera.fx - 1),
          ),
        )
      : null,
    geometry: model
      ? compareSeedGauge(
          { views: f.truth.views, points: f.truth.finitePoints },
          model,
          result.diagnostics.initialPair,
        )
      : null,
    meanReprojectionErrorPx: result?.diagnostics.meanReprojectionErrorPx ?? null,
    initializationFailure: failure?.message ?? null,
  };
}
const report = {
  createdAt: new Date().toISOString(),
  baseline: {
    commit: git(baselineRoot, 'rev-parse', 'HEAD'),
    dirty: !!git(baselineRoot, 'status', '--porcelain'),
  },
  current: { commit: git(root, 'rev-parse', 'HEAD'), dirty: !!git(root, 'status', '--porcelain') },
  environment: {
    node: process.version,
    cpu: cpus()[0]?.model,
    backend: 'f64 CPU',
    physicalGpuMeasured: false,
  },
  protocol: {
    currentOption: { separateDirectionalObservations: true },
    baselineOption: { separateDirectionalObservations: baselineDirectional },
    shippedDefault: false,
    input:
      'Synthetic calibrated keypoints with supplied identities. No RGB detection, decode or upload is timed.',
    calibration:
      'Known synthetic K except estimated-focal cases: input focal replaced by 0.8*width; shared focal searched from images.',
    truth:
      'Only the evaluator reads true poses, points and source labels. No ground-truth depth/pose enters reconstruction.',
    masks: 'Only screen-fixed-masked uses explicitly ground-truth-sourced feature masks.',
    alignment:
      'One fixed seed pose/baseline gauge per whole connected map, never per-frame alignment.',
    timing:
      'One warmup per version then alternating paired runs, reconstruction only; includes CPU geometry, tracking and BA. Other checks may run concurrently: timings are diagnostic, not acceleration evidence.',
    repeats,
  },
  expectedCases: cases.length,
  completed: false,
  sourceHashes: Object.fromEntries(
    await Promise.all(
      [
        'src/sfm/two-view.ts',
        'src/sfm/refinement.ts',
        'src/gpu/recovery.shade.ts',
        'src/reconstruction/mapper.ts',
        'src/reconstruction/directional-tracks.ts',
        'src/reconstruction/bundle-directions.ts',
        'src/reconstruction/focal-search.ts',
        'src/geometry/rotation.ts',
        'tests/fixtures/direction-observations.ts',
        'scripts/benchmark-direction-depth.mjs',
      ].map(async (path) => [path, sha(await readFile(resolve(root, path)))]),
    ),
  ),
  cases: [],
};
for (const entry of cases) {
  const { name, fixture: f, options } = entry,
    original = digest(f.input),
    controls = {
      ransacIterations: 128,
      bundleIterations: 2,
      finalBundleIterations: 2,
      ...entry.controls,
    };
  if (controls.focalSearch) {
    // The estimate is not seeded with the true focal (480); only cx/cy and aspect are supplied.
    for (const image of f.input.images) image.camera = { ...image.camera, fx: 512, fy: 512 };
  }
  const actualDigest = digest(f.input),
    runs = [];
  for (let round = -1; round < repeats; round++)
    for (const mode of round % 2 === 0 ? [0, 1] : [1, 0]) {
      let result, failure;
      const started = performance.now();
      try {
        result = implementations[mode].reconstructIncremental(f.input.images, f.input.pairs, {
          ...controls,
          ...(mode || baselineDirectional ? { separateDirectionalObservations: true } : {}),
        });
      } catch (e) {
        if (e.name !== 'ReconstructionInitializationError') throw e;
        failure = e;
      }
      const elapsedMs = performance.now() - started;
      if (round >= 0)
        runs.push({
          mode: mode ? 'current' : 'baseline',
          elapsedMs,
          ...quality(result, failure, f, options),
        });
    }
  if (digest(f.input) !== actualDigest) throw new Error(`Input mutated: ${name}`);
  const summary = Object.fromEntries(
    ['baseline', 'current'].map((mode) => {
      const rows = runs.filter((r) => r.mode === mode),
        first = rows[0];
      return [
        mode,
        {
          ...first,
          mode: undefined,
          elapsedMs: undefined,
          medianMs: median(rows.map((r) => r.elapsedMs)),
          stableCounts: rows.every(
            (r) =>
              r.registered === first.registered &&
              r.points === first.points &&
              r.falseFinite === first.falseFinite,
          ),
        },
      ];
    }),
  );
  report.cases.push({
    name,
    options,
    inputSha256: actualDigest,
    originalInputSha256: original,
    controls: {
      ...controls,
      cameraObservationMasks: controls.cameraObservationMasks?.map((m) => ({
        viewId: m.viewId,
        source: m.source,
        sha256: sha(m.values),
      })),
    },
    summary,
    runs,
  });
  process.stdout.write(
    `${name}: ${summary.baseline.registered}/${summary.baseline.falseFinite} -> ${summary.current.registered}/${summary.current.falseFinite}\n`,
  );
  await mkdir(dirname(output), { recursive: true });
  await writeFile(output, JSON.stringify(report, null, 2) + '\n');
}
report.completed = true;
await writeFile(output, JSON.stringify(report, null, 2) + '\n');
process.stdout.write(`Wrote ${output}\n`);
