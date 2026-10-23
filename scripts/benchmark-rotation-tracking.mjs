import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { cpus } from 'node:os';
import {
  cameraCenter,
  reconstructIncremental,
  ReconstructionInitializationError,
} from '../src/index.ts';
import { multiply, transpose } from '../src/sfm/linear.ts';
import {
  incrementalControls,
  verifyPairs,
  executeReconstructionJob,
} from '../src/reconstruction/mapper.ts';
import { buildTracks } from '../src/reconstruction/tracks.ts';
import { generalVideoFixture, GENERAL_VIDEO_CASES } from '../tests/fixtures/general-video.ts';

const option = (name, fallback) =>
  process.argv.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;
const repeats = Number(option('repeats', '6'));
if (!Number.isSafeInteger(repeats) || repeats < 2 || repeats > 20 || repeats % 2)
  throw new RangeError('--repeats must be an even integer from 2 through 20');
const output = resolve(
  option('output', 'docs/experiments/general-video/rotation-tracking-cpu-comparison.json'),
);
const median = (values) => {
  const sorted = [...values].sort((a, b) => a - b);
  return (sorted[(sorted.length - 1) >> 1] + sorted[sorted.length >> 1]) / 2;
};
const sha = (value) => createHash('sha256').update(value).digest('hex');
const maxOrNull = (values) => {
  const finite = values.filter(Number.isFinite);
  return finite.length ? Math.max(...finite) : null;
};
const digest = (input) =>
  sha(JSON.stringify(input, (_, value) => (ArrayBuffer.isView(value) ? [...value] : value)));
const cases = GENERAL_VIDEO_CASES.map((name) => ({ name, fixture: generalVideoFixture(name) }));
for (const noise of [0, 0.1, 0.3, 0.7, 1]) {
  const fixture = generalVideoFixture('rotation-start');
  fixture.input.pairs = fixture.input.pairs.filter((p) => p.second - p.first === 1);
  for (const view of fixture.input.images)
    view.keypoints = view.keypoints.map(
      (x, i) => x + noise * Math.sin((i + 1) * (view.id + 3) * 1.37),
    );
  cases.push({ name: `rotation-chain-noise-${noise}`, fixture });
}
const mismatch = generalVideoFixture('translation');
mismatch.expected.initialization = 'deferred';
for (const pair of mismatch.input.pairs)
  pair.matches = pair.matches.map((x, i) => (i % 2 ? (13 * x + 7) % 60 : x));
cases.push({ name: 'unrelated-correspondences', fixture: mismatch });

function quality(result, failure, fixture) {
  const tracking = result?.tracking ?? failure?.tracking;
  if (!result)
    return {
      status: 'initialization-rejected',
      registered: 0,
      points: 0,
      tracking: tracking
        ? {
            tracks: tracking.tracks.length,
            observations: tracking.tracks.reduce((n, t) => n + t.observations.length, 0),
            pending: tracking.untriangulatedTrackIds.length,
            rotationPairs: tracking.rotations.length,
          }
        : null,
      falseStructure: false,
    };
  const model = result.reconstruction,
    components = new Set(model.views.map((v) => fixture.truth.componentOfView[v.id]));
  // The seed gauge is fixed once for the entire map, using evaluator-only truth. A
  // camera-center fit is ill-conditioned for a rotation prefix or two-center shots.
  const [first, second] = result.diagnostics.initialPair,
    anchor = fixture.truth.views.find((v) => v.id === first),
    center = cameraCenter(anchor.pose),
    other = cameraCenter(fixture.truth.views.find((v) => v.id === second).pose),
    scale = Math.hypot(...other.map((x, i) => x - center[i])),
    world = (point) =>
      [0, 1, 2].map(
        (r) =>
          center[r] +
          scale *
            (anchor.pose.R[r] * point[0] +
              anchor.pose.R[3 + r] * point[1] +
              anchor.pose.R[6 + r] * point[2]),
      ),
    cameras = model.views.map((v) => {
      const truth = fixture.truth.views.find((t) => t.id === v.id),
        expectedR = multiply(truth.pose.R, transpose(anchor.pose.R, 3), 3),
        trace = v.pose.R.reduce((n, x, i) => n + x * expectedR[i], 0),
        target = cameraCenter(truth.pose);
      return {
        id: v.id,
        rotationErrorDeg: (Math.acos(Math.max(-1, Math.min(1, (trace - 1) / 2))) * 180) / Math.PI,
        centerError: Math.hypot(...world(cameraCenter(v.pose)).map((x, i) => x - target[i])),
      };
    }),
    alignment = {
      status: components.size === 1 ? 'aligned' : 'unavailable',
      method: 'One global seed gauge: truth anchor orientation/center and seed baseline length',
      initialPair: [first, second],
      scale,
      centerRmse: Math.sqrt(cameras.reduce((n, v) => n + v.centerError ** 2, 0) / cameras.length),
      maxRotationErrorDeg: Math.max(...cameras.map((v) => v.rotationErrorDeg)),
    };
  const errors = [];
  if (alignment.status === 'aligned')
    for (const p of model.landmarks) {
      const expected = fixture.truth.points[p.observations[0].featureId];
      errors.push(Math.hypot(...world(p.xyz).map((v, i) => v - expected[i])));
    }
  return {
    status: 'map',
    registered: model.views.length,
    viewIds: model.views.map((v) => v.id),
    points: model.landmarks.length,
    observations: model.landmarks.reduce((n, p) => n + p.observations.length, 0),
    tracks: result.tracks.length,
    trackObservations: result.tracks.reduce((n, t) => n + t.observations.length, 0),
    pending: tracking?.untriangulatedTrackIds.length ?? null,
    rotationPairs: tracking?.rotations.length ?? 0,
    meanReprojectionErrorPx: result.diagnostics.meanReprojectionErrorPx,
    cameraAlignment: alignment,
    cameraErrors: cameras,
    pointRmse: errors.length
      ? Math.sqrt(errors.reduce((n, e) => n + e * e, 0) / errors.length)
      : null,
    inconsistentPointIdentities: model.landmarks.filter(
      (p) => new Set(p.observations.map((o) => o.featureId)).size > 1,
    ).length,
    falseStructure: fixture.expected.initialization === 'deferred' && model.landmarks.length > 0,
    falseComponentMerge: components.size > 1,
  };
}

const report = {
  createdAt: new Date().toISOString(),
  head: execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).trim(),
  workingTreeDirty:
    execFileSync('git', ['status', '--porcelain'], { encoding: 'utf8' }).trim() !== '',
  environment: {
    node: process.version,
    platform: process.platform,
    cpu: cpus()[0]?.model,
    logicalProcessors: cpus().length,
    backend: 'synchronous f64 CPU',
    nativeGpuMeasured: false,
  },
  protocol: {
    repeats,
    warmupsPerMode: 1,
    order: 'off/on then on/off in alternating paired rounds',
    timing:
      'reconstructIncremental only; evaluator, hashes and separate epipolar-graph inspection excluded',
    input:
      'known synthetic K and supplied synthetic correspondences; no ground-truth pose/depth enters reconstruction',
    alignment:
      'One complete-map seed gauge using evaluator-only truth anchor pose and baseline length, as in rotation-tracking.test.ts. Never per-frame fits; avoids camera-center alignment ambiguity for rotation prefixes and two-center shots.',
    baseline:
      'rotationTracking:false on the same feature commit, not an older runtime or an unexposed 2D graph assumed empty',
  },
  sourceHashes: Object.fromEntries(
    await Promise.all(
      [
        'src/geometry/rotation.ts',
        'src/reconstruction/tracking.ts',
        'src/reconstruction/mapper.ts',
        'src/reconstruction/focal-search.ts',
        'tests/fixtures/general-video.ts',
        'scripts/benchmark-rotation-tracking.mjs',
      ].map(async (p) => [p, sha(await readFile(p))]),
    ),
  ),
  cases: [],
};
for (const { name, fixture } of cases) {
  const { images, pairs } = fixture.input,
    before = digest(fixture.input);
  const controls = {
    ransacIterations: 128,
    bundleIterations: 1,
    finalBundleIterations: 2,
    ...(['mapped-rotation', 'reobserved'].includes(fixture.scenario)
      ? { initialPair: [0, 1] }
      : {}),
    ...(fixture.scenario === 'disconnected' ? { initialPair: [0, 2] } : {}),
  };
  const { byId, controls: checked } = incrementalControls(images, pairs, {
    ...controls,
    rotationTracking: false,
  });
  const machine = verifyPairs(byId, pairs, checked);
  let step = machine.next();
  while (!step.done) step = machine.next(executeReconstructionJob(step.value));
  const internal = buildTracks(images, step.value.verified).tracks;
  const row = {
    name,
    inputSha256: before,
    frames: images.length,
    featureCount: images[0].keypoints.length / 2,
    baselineEpipolarGraph: {
      tracks: internal.length,
      observations: internal.reduce((n, t) => n + t.observations.length, 0),
      coveredFrames: new Set(internal.flatMap((t) => t.observations.map((o) => o.viewId))).size,
    },
    runs: [],
    controls,
  };
  for (let round = -1; round < repeats; round++) {
    for (const enabled of round % 2 === 0 ? [true, false] : [false, true]) {
      let result, failure;
      const started = performance.now();
      try {
        result = reconstructIncremental(images, pairs, { ...controls, rotationTracking: enabled });
      } catch (error) {
        if (!(error instanceof ReconstructionInitializationError)) throw error;
        failure = error;
      }
      const elapsedMs = performance.now() - started;
      row.runs.push({
        round,
        warmup: round < 0,
        enabled,
        elapsedMs,
        ...quality(result, failure, fixture),
      });
    }
  }
  if (digest(fixture.input) !== before) throw new Error(`Input mutation: ${name}`);
  row.summary = Object.fromEntries(
    [false, true].map((enabled) => {
      const runs = row.runs.filter((r) => !r.warmup && r.enabled === enabled),
        first = runs[0];
      return [
        enabled ? 'on' : 'off',
        {
          registered: first.registered,
          points: first.points,
          medianMs: median(runs.map((r) => r.elapsedMs)),
          minMs: Math.min(...runs.map((r) => r.elapsedMs)),
          maxMs: Math.max(...runs.map((r) => r.elapsedMs)),
          tracking: first.tracking ?? {
            tracks: first.tracks,
            observations: first.trackObservations,
            pending: first.pending,
            rotationPairs: first.rotationPairs,
          },
          maxCameraCenterRmse: maxOrNull(runs.map((r) => r.cameraAlignment?.centerRmse)),
          maxCameraRotationErrorDeg: maxOrNull(
            runs.map((r) => r.cameraAlignment?.maxRotationErrorDeg),
          ),
          maxPointRmse: maxOrNull(runs.map((r) => r.pointRmse)),
          falseStructure: runs.some((r) => r.falseStructure),
          falseMerge: runs.some((r) => r.falseComponentMerge),
          inconsistentPointIdentities: Math.max(
            ...runs.map((r) => r.inconsistentPointIdentities ?? 0),
          ),
        },
      ];
    }),
  );
  row.summary.registrationGain = row.summary.on.registered - row.summary.off.registered;
  row.summary.registrationGainPercentagePoints =
    (100 * row.summary.registrationGain) / images.length;
  row.summary.medianTimeChangePercent =
    100 * (row.summary.on.medianMs / row.summary.off.medianMs - 1);
  report.cases.push(row);
  console.log(JSON.stringify({ case: name, ...row.summary }));
}
// Compatibility is reported separately, never counted as a speed improvement from early rejection.
const pyramid = JSON.parse(await readFile('tests/fixtures/pyramid-sift-cpu.json', 'utf8')),
  focalImages = pyramid.frames.map((f) => ({
    id: f.id,
    camera: {
      fx: pyramid.width * 0.5,
      fy: pyramid.width * 0.5,
      cx: (pyramid.width - 1) / 2,
      cy: (pyramid.height - 1) / 2,
    },
    keypoints: Float64Array.from(f.keypoints),
  })),
  focalPairs = pyramid.pairs.map((p) => ({ ...p, matches: Uint32Array.from(p.matches) }));
report.focalSearchFixture =
  'tests/fixtures/pyramid-sift-cpu.json, same controls as focal-search.test.ts';
report.focalSearchCompatibility = [];
for (const rotationTracking of [undefined, false, true]) {
  try {
    const r = reconstructIncremental(focalImages, focalPairs, {
      rotationTracking,
      focalSearch: true,
      ransacIterations: 500,
      maxErrorPx: 3,
      minInitialAngleDeg: 1,
      minTriangulationAngleDeg: 0.5,
      bundleIterations: 4,
      finalBundleIterations: 8,
      refineSharedFocal: true,
    });
    report.focalSearchCompatibility.push({
      rotationTracking: rotationTracking ?? 'default',
      status: 'map',
      registered: r.reconstruction.views.length,
      trackingPresent: 'tracking' in r,
    });
  } catch (e) {
    report.focalSearchCompatibility.push({
      rotationTracking: rotationTracking ?? 'default',
      status: 'error',
      name: e.name,
      message: e.message,
    });
  }
}
report.processPeakRssBytes = process.resourceUsage().maxRSS * 1024;
await mkdir(dirname(output), { recursive: true });
await writeFile(output, JSON.stringify(report, null, 2) + '\n');
