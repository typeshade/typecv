import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { reconstructIncremental, ReconstructionInitializationError } from '../src/index.ts';
import { generalVideoFixture, GENERAL_VIDEO_CASES } from '../tests/fixtures/general-video.ts';
import { foregroundFixture } from '../tests/fixtures/foreground.ts';
import { compareSeedGauge } from './geometry-comparison.mjs';

const arg = (name, fallback) =>
  process.argv.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;
const repeats = Number(arg('repeats', '6')),
  output = resolve(
    arg('output', 'docs/experiments/general-video/structure-quality-cpu-comparison.json'),
  );
if (!Number.isSafeInteger(repeats) || repeats < 2 || repeats > 12 || repeats % 2)
  throw new RangeError('Even --repeats=2..12 required');
const median = (v) => {
  const a = [...v].sort((x, y) => x - y);
  return (a[(a.length - 1) >> 1] + a[a.length >> 1]) / 2;
};
const sha = (v) => createHash('sha256').update(v).digest('hex');
const digest = (v) => sha(JSON.stringify(v, (_, x) => (ArrayBuffer.isView(x) ? [...x] : x)));
const cases = GENERAL_VIDEO_CASES.map((name) => ({
  name,
  fixture: generalVideoFixture(name),
  added: { pairRefinementIterations: 10, maxRelativeDepthStdDev: 0.1 },
}));
for (const noise of [0, 0.1, 0.3, 0.7, 1]) {
  const f = generalVideoFixture('rotation-start');
  f.input.pairs = f.input.pairs.filter((p) => p.second - p.first === 1);
  for (const image of f.input.images)
    image.keypoints = image.keypoints.map(
      (x, i) => x + noise * Math.sin((i + 1) * (image.id + 3) * 1.37),
    );
  cases.push({
    name: `rotation-chain-noise-${noise}`,
    fixture: f,
    added: { pairRefinementIterations: 10, maxRelativeDepthStdDev: 0.1 },
  });
}
for (const refined of [false, true]) {
  const f = foregroundFixture();
  cases.push({
    name: refined ? 'foreground-mask-and-quality' : 'foreground-mask-only',
    fixture: { ...f, input: { images: f.images, pairs: f.pairs } },
    added: {
      cameraObservationMasks: f.masks,
      ...(refined ? { pairRefinementIterations: 10, maxRelativeDepthStdDev: 0.1 } : {}),
    },
  });
}
const low = generalVideoFixture('translation');
low.input.images = low.input.images.slice(0, 2);
low.input.pairs = low.input.pairs.filter((p) => p.second < 2);
cases.push({
  name: 'conditional-depth-two-views',
  fixture: low,
  added: { maxRelativeDepthStdDev: 0.03 },
});
const report = {
  createdAt: new Date().toISOString(),
  head: execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).trim(),
  workingTreeDirty: true,
  node: process.version,
  backend: 'f64 CPU',
  nativeGpuMeasured: false,
  protocol: {
    repeats,
    warmupsPerMode: 1,
    order: 'alternating paired on/off order, no other validation jobs during timing',
    input:
      'Known synthetic K and supplied identity correspondences; not RGB accuracy; foreground masks are explicitly ground-truth sourced',
    evaluation:
      'truth stays in evaluator; one fixed seed pose/baseline gauge for each complete connected map',
    baseline: 'current rotation tracking default, pair refinement 0 and depth guard disabled',
    uncertainty:
      'Depth sensitivity conditions on fixed poses/K and 1px independent coordinate noise, not total SfM confidence',
  },
  cases: [],
};
for (const { name, fixture: f, added } of cases) {
  const original = digest(f.input),
    controls = {
      ransacIterations: 128,
      bundleIterations: 1,
      finalBundleIterations: 2,
      ...(['mapped-rotation', 'reobserved'].includes(f.scenario) ? { initialPair: [0, 1] } : {}),
      ...(f.scenario === 'disconnected' ? { initialPair: [0, 2] } : {}),
    };
  const row = {
    name,
    inputSha256: original,
    controls,
    added: {
      ...added,
      cameraObservationMasks: added.cameraObservationMasks?.map((m) => ({
        viewId: m.viewId,
        source: m.source,
        sha256: sha(m.values),
      })),
    },
    runs: [],
  };
  for (let round = -1; round < repeats; round++)
    for (const enabled of round % 2 === 0 ? [true, false] : [false, true]) {
      let r, failure;
      const started = performance.now();
      try {
        r = reconstructIncremental(f.input.images, f.input.pairs, {
          ...controls,
          ...(enabled ? added : {}),
        });
      } catch (e) {
        if (!(e instanceof ReconstructionInitializationError)) throw e;
        failure = e;
      }
      const elapsedMs = performance.now() - started,
        tracking = r?.tracking ?? failure?.tracking;
      row.runs.push({
        round,
        warmup: round < 0,
        enabled,
        elapsedMs,
        status: r ? 'map' : 'initialization-rejected',
        registered: r?.reconstruction.views.length ?? 0,
        points: r?.reconstruction.landmarks.length ?? 0,
        geometry: r ? compareSeedGauge(f.truth, r.reconstruction, r.diagnostics.initialPair) : null,
        foregroundLandmarks:
          r?.reconstruction.landmarks.filter((p) => p.observations.some((o) => o.featureId >= 60))
            .length ?? 0,
        pending: tracking?.untriangulatedTrackIds.length ?? null,
        depthDeferred: tracking?.depthDeferredTrackIds?.length ?? null,
        separatedRows:
          r?.observationSelection?.separatedPairs.reduce((n, p) => n + p.matches.length / 2, 0) ??
          0,
        falseStructure:
          f.expected.initialization === 'deferred' && (r?.reconstruction.landmarks.length ?? 0) > 0,
        falseMerge: r
          ? new Set(r.reconstruction.views.map((v) => f.truth.componentOfView[v.id])).size > 1
          : false,
      });
    }
  if (digest(f.input) !== original) throw new Error('Mutated source input');
  row.summary = Object.fromEntries(
    [false, true].map((enabled) => {
      const runs = row.runs.filter((r) => !r.warmup && r.enabled === enabled),
        r = runs[0];
      return [
        enabled ? 'on' : 'off',
        {
          ...r,
          round: undefined,
          warmup: undefined,
          enabled: undefined,
          elapsedMs: undefined,
          medianMs: median(runs.map((r) => r.elapsedMs)),
          minMs: Math.min(...runs.map((r) => r.elapsedMs)),
          maxMs: Math.max(...runs.map((r) => r.elapsedMs)),
          allRunsSameCounts: runs.every(
            (x) => x.registered === r.registered && x.points === r.points,
          ),
          falseStructure: runs.some((x) => x.falseStructure),
          falseMerge: runs.some((x) => x.falseMerge),
        },
      ];
    }),
  );
  row.summary.timeChangePercent = 100 * (row.summary.on.medianMs / row.summary.off.medianMs - 1);
  report.cases.push(row);
  console.log(JSON.stringify({ case: name, summary: row.summary }));
}
report.sourceSha256 = Object.fromEntries(
  await Promise.all(
    [
      'src/sfm/mapper/mapper.ts',
      'src/sfm/tracks/observation-selection.ts',
      'src/geometry/multiview.ts',
      'tests/fixtures/foreground.ts',
      'scripts/geometry-comparison.mjs',
      'scripts/benchmark-structure-quality.mjs',
    ].map(async (p) => [p, sha(await readFile(p))]),
  ),
);
report.processPeakRssBytes = process.resourceUsage().maxRSS * 1024;
await mkdir(dirname(output), { recursive: true });
await writeFile(output, JSON.stringify(report, null, 2) + '\n');
