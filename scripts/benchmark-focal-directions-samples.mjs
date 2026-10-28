import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { LAB, loadSample, loadPyramid, camerasAt, pyramidRelief } from './measure-focal.mjs';
import { alignPoses } from './pose-alignment.ts';

const arg = (name, fallback) =>
    process.argv.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback,
  root = resolve(import.meta.dirname, '..'),
  baseline = arg('baseline'),
  output = resolve(arg('output', '/tmp/focal-direction-samples.json')),
  assertDefaultParity = arg('assert-default-parity', '0');
if (!baseline) throw new RangeError('--baseline=/path/to/checkout is required');
if (!['0', '1'].includes(assertDefaultParity))
  throw new RangeError('--assert-default-parity must be 0 or 1');
const baselineRoot = resolve(baseline);
const implementations = await Promise.all(
    [baselineRoot, root].map(
      (dir) => import(pathToFileURL(resolve(dir, 'src/reconstruction/mapper.ts')).href),
    ),
  ),
  git = (dir, ...args) => execFileSync('git', ['-C', dir, ...args], { encoding: 'utf8' }).trim(),
  digest = (v) =>
    createHash('sha256')
      .update(JSON.stringify(v, (_, x) => (ArrayBuffer.isView(x) ? [...x] : x)))
      .digest('hex'),
  cases = [];
const sampleNames = ['south-building', 'gerrard-hall', 'person-hall', 'graham-hall'];
for (const name of sampleNames) {
  const f = loadSample(name),
    sample = JSON.parse(
      readFileSync(resolve(root, `tests/fixtures/lab-images/${name}/sample.json`)),
    );
  cases.push({
    name,
    source: 'cached COLMAP SIFT/raw putative matches',
    calibration: 'provided COLMAP K/distortion, held fixed',
    calibrationSource: 'supplied COLMAP estimate; this is a known-intrinsics experiment',
    images: f.frames.map((v) => ({ id: v.id, camera: { ...v.truth }, keypoints: v.keypoints })),
    pairs: f.pairs,
    reference: sample.frames.map((v) => ({ id: v.id, ...v.reference })),
    controls: { ...LAB },
  });
}
const g = JSON.parse(readFileSync(resolve(root, 'tests/fixtures/graham-hall-lab-matches.json'))),
  gr = JSON.parse(readFileSync(resolve(root, 'tests/fixtures/lab-images/graham-hall/sample.json')));
cases.push({
  name: 'graham-hall-lab',
  source: 'cached TypeCV SIFT/guided matches',
  calibration: 'provided COLMAP K/distortion, held fixed',
  calibrationSource: 'supplied COLMAP estimate; this is a known-intrinsics experiment',
  images: g.frames.map((v) => ({ ...v, keypoints: Float64Array.from(v.keypoints) })),
  pairs: g.pairs.map((p) => ({ ...p, matches: Uint32Array.from(p.matches) })),
  reference: gr.frames.map((v) => ({ id: v.id, ...v.reference })),
  controls: { ...LAB },
});
const p = loadPyramid();
cases.push({
  name: 'pyramid-8',
  source: 'cached CPU SIFT/mutual ratio-0.8 matches, 8 frames',
  calibration:
    'estimated shared focal; initial guess 0.8*width, centered principal point, square pixels, zero distortion',
  calibrationSource: 'image dimensions and explicit pinhole assumptions; no reference calibration',
  images: camerasAt(p.frames, 0.8, false),
  pairs: p.pairs,
  controls: { ...LAB, focalSearch: true, refineSharedFocal: true },
});
// Construct every calibration value from dimensions, independently of COLMAP's cameras.
// Those cameras and poses remain outside reconstruction input for this extra experiment.
const grahamDimensions = new Map(gr.frames.map((v) => [v.id, [v.width, v.height]]));
cases.push({
  name: 'graham-hall-lab-estimated-focal',
  source: 'cached TypeCV SIFT/guided matches; same coordinates and edges as graham-hall-lab',
  calibration:
    'estimated shared focal; initial guess 0.8*width, centered principal point, square pixels, zero distortion',
  calibrationSource:
    'image dimensions and explicit pinhole assumptions; COLMAP K/distortion is not supplied',
  images: g.frames.map((v) => {
    const [width, height] = grahamDimensions.get(v.id);
    return {
      id: v.id,
      keypoints: Float64Array.from(v.keypoints),
      camera: {
        fx: 0.8 * width,
        fy: 0.8 * width,
        cx: (width - 1) / 2,
        cy: (height - 1) / 2,
        k1: 0,
        k2: 0,
      },
    };
  }),
  pairs: g.pairs.map((p) => ({ ...p, matches: Uint32Array.from(p.matches) })),
  reference: gr.frames.map((v) => ({ id: v.id, ...v.reference })),
  controls: { ...LAB, focalSearch: true, refineSharedFocal: true },
});
const sourcePaths = [
    'src/sfm/mapper/mapper.ts',
    'src/sfm/focal/focal-search.ts',
    'src/sfm/bundle/directional-tracks.ts',
    'src/sfm/bundle/bundle-directions.ts',
    'src/sfm/bundle/bundle-adjustment.ts',
    'src/sfm/bundle/joint-focal.ts',
    'src/sfm/tracks/tracking.ts',
    'src/geometry/rotation.ts',
    'src/geometry/two-view.ts',
    'src/geometry/refinement.ts',
  ],
  hashes = (dir, paths) =>
    Object.fromEntries(
      paths.map((path) => [
        path,
        existsSync(resolve(dir, path))
          ? createHash('sha256')
              .update(readFileSync(resolve(dir, path)))
              .digest('hex')
          : null,
      ]),
    );

function focalSupport(search) {
  if (!search) return undefined;
  const compact = (c) => ({
    ratio: c.ratio,
    initialRatio: c.initialRatio,
    views: c.views,
    observations: c.observations,
    directionalObservations: c.directionalObservations ?? 0,
    modelParameters: c.modelParameters,
  });
  return {
    selectedRatio: search.ratio,
    scoring: search.scoring ?? 'truncated-reprojection-px2',
    candidates: search.candidates.map(compact),
    refinements: (search.refinements ?? []).map(compact),
  };
}

function compareModes(before, after) {
  const delta = (key) =>
    Number.isFinite(before[key]) && Number.isFinite(after[key]) ? after[key] - before[key] : null;
  return {
    beforeStatus: before.status,
    afterStatus: after.status,
    sameStatus: before.status === after.status,
    sameGeometry:
      before.status === 'map' && after.status === 'map'
        ? before.geometrySha256 === after.geometrySha256
        : null,
    sameTracking: before.trackingSha256 === after.trackingSha256,
    registeredDelta: delta('registered'),
    pointsDelta: delta('points'),
    observationsDelta: delta('observations'),
    pendingDelta: delta('pending'),
    focalDeltaPx: delta('focal'),
  };
}

const report = {
  createdAt: new Date().toISOString(),
  baseline: git(baselineRoot, 'rev-parse', 'HEAD'),
  baselineDirty: !!git(baselineRoot, 'status', '--porcelain'),
  current: git(root, 'rev-parse', 'HEAD'),
  currentDirty: !!git(root, 'status', '--porcelain'),
  node: process.version,
  backend: 'CPU f64',
  physicalGpuMeasured: false,
  defaults: {
    separateDirectionalObservations: false,
    assertDefaultParity: assertDefaultParity === '1',
  },
  protocol:
    'Four modes: baseline/current, default-off/opt-in. Both versions receive the same current-workspace cached RGB-derived features and matches; no decode, extraction, GPU or dense stage. Cached guided matches may contain prior geometric selection: no claim of a fresh calibration-independent RGB front end. Five known-K rows supply COLMAP K/distortion explicitly; the two shared-focal rows use centered principal point, square pixels and zero distortion assumptions. No reference poses/depths enter reconstruction, and the added estimated-focal row does not receive COLMAP K. No masks are used. One sequential pass per mode; elapsedMs times reconstruction only, excluding hashing/alignment; diagnostic, not a speed comparison. COLMAP reference poses are estimates, used only in evaluation. One similarity alignment per whole connected map, with missing registrations retained in the report. Reference agreement and Pyramid relief do not establish ground-truth accuracy. This is not the full current Pyramid catalogue video. Across-version comparisons measure integration changes; within-version off/on comparisons isolate the directional flag under that version.',
  sourceHashes: hashes(root, sourcePaths),
  baselineSourceHashes: hashes(baselineRoot, sourcePaths),
  benchmarkHashes: hashes(root, [
    'scripts/benchmark-focal-directions-samples.mjs',
    'scripts/measure-focal.mjs',
    'scripts/pose-alignment.ts',
  ]),
  inputFileHashes: hashes(root, [
    ...sampleNames.flatMap((name) => [
      `tests/fixtures/lab-images/${name}/sample.json`,
      `tests/fixtures/lab-images/${name}/colmap-sift.bin`,
    ]),
    'tests/fixtures/graham-hall-lab-matches.json',
    'tests/fixtures/pyramid-sift-cpu.json',
  ]),
  completed: false,
  expectedCases: cases.length,
  cases: [],
};
for (const c of cases) {
  const input = { images: c.images, pairs: c.pairs },
    sha = digest(input),
    row = {
      name: c.name,
      source: c.source,
      calibration: c.calibration,
      calibrationSource: c.calibrationSource,
      masks: 'none',
      frames: c.images.length,
      pairs: c.pairs.length,
      inputSha256: sha,
      controls: c.controls,
      modes: [],
    };
  for (const enabled of [false, true])
    for (const [i, label] of ['baseline', 'current'].entries()) {
      const start = performance.now();
      try {
        const r = implementations[i].reconstructIncremental(c.images, c.pairs, {
            ...c.controls,
            separateDirectionalObservations: enabled,
          }),
          elapsedMs = performance.now() - start,
          model = r.reconstruction,
          agreement = c.reference
            ? alignPoses(
                model.views.map((v) => ({ id: v.id, R: v.pose.R, t: v.pose.t })),
                c.reference,
              )
            : undefined;
        row.modes.push({
          label,
          enabled,
          elapsedMs,
          status: 'map',
          geometrySha256: digest(model),
          trackingSha256: digest(r.tracking ?? null),
          registered: model.views.length,
          unregisteredViewIds: c.images
            .filter((v) => !model.views.some((registered) => registered.id === v.id))
            .map((v) => v.id),
          points: model.landmarks.length,
          observations: model.landmarks.reduce((n, p) => n + p.observations.length, 0),
          meanReprojectionErrorPx: r.diagnostics.meanReprojectionErrorPx,
          pending: r.tracking?.untriangulatedTrackIds.length,
          bearingObservationsInFinalBundle:
            r.diagnostics.bundleAdjustments.at(-1)?.directionalObservations ?? 0,
          focal: model.views[0]?.camera.fx,
          focalScoring: r.diagnostics.focalSearch?.scoring,
          focalDirectionalSupport: focalSupport(r.diagnostics.focalSearch),
          poseAgreement: agreement && {
            status: agreement.status,
            commonViews: agreement.commonViews,
            rotationDeg: agreement.rotationDeg,
            centerOfRadius: agreement.centerOfRadius,
          },
          pyramidRelief: c.name === 'pyramid-8' ? pyramidRelief(model) : undefined,
        });
      } catch (e) {
        if (e.name !== 'ReconstructionInitializationError') throw e;
        row.modes.push({
          label,
          enabled,
          status: 'deferred',
          reason: e.message,
          elapsedMs: performance.now() - start,
          trackingSha256: digest(e.tracking ?? null),
          pending: e.tracking?.untriangulatedTrackIds.length,
          unregisteredViewIds: c.images.map((v) => v.id),
        });
      }
      if (digest(input) !== sha) throw new Error(`Input mutated: ${c.name}`);
    }
  row.defaultGeometryUnchanged =
    row.modes[0].status === 'map' && row.modes[0].geometrySha256 === row.modes[1].geometrySha256;
  const [baselineOff, currentOff, baselineOn, currentOn] = row.modes;
  row.comparisons = {
    defaultAcrossVersions: compareModes(baselineOff, currentOff),
    enabledAcrossVersions: compareModes(baselineOn, currentOn),
    baselineFlagEffect: compareModes(baselineOff, baselineOn),
    currentFlagEffect: compareModes(currentOff, currentOn),
  };
  row.defaultParity =
    baselineOff.status === currentOff.status &&
    baselineOff.trackingSha256 === currentOff.trackingSha256 &&
    (baselineOff.status === 'map'
      ? baselineOff.geometrySha256 === currentOff.geometrySha256
      : baselineOff.reason === currentOff.reason);
  report.cases.push(row);
  writeFileSync(output, JSON.stringify(report, null, 2) + '\n');
  process.stdout.write(JSON.stringify(row) + '\n');
}
report.completed = true;
report.validation = {
  expectedCases: 7,
  expectedModesPerCase: 4,
  complete:
    report.cases.length === 7 &&
    report.cases.every((c) =>
      ['baseline:false', 'current:false', 'baseline:true', 'current:true'].every(
        (key, i) => `${c.modes[i]?.label}:${c.modes[i]?.enabled}` === key,
      ),
    ),
  defaultParity: report.cases.every((c) => c.defaultParity),
  defaultParityFailures: report.cases.filter((c) => !c.defaultParity).map((c) => c.name),
};
writeFileSync(output, JSON.stringify(report, null, 2) + '\n');
if (!report.validation.complete) throw new Error('Incomplete seven-case/four-mode comparison');
if (assertDefaultParity === '1' && !report.validation.defaultParity)
  throw new Error(
    `Default-off geometry/tracking changed: ${report.validation.defaultParityFailures.join(', ')}`,
  );
