import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { LAB, loadSample, loadPyramid, camerasAt, pyramidRelief } from './measure-focal.mjs';
import { alignPoses } from '../website/src/lib/pose-alignment.ts';

const arg = (name, fallback) =>
    process.argv.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback,
  root = resolve(import.meta.dirname, '..'),
  baseline = arg('baseline'),
  output = resolve(arg('output', '/tmp/focal-direction-samples.json'));
if (!baseline) throw new RangeError('--baseline=/path/to/checkout is required');
const implementations = await Promise.all(
    [resolve(baseline), root].map(
      (dir) => import(pathToFileURL(resolve(dir, 'src/reconstruction/mapper.ts')).href),
    ),
  ),
  git = (dir, ...args) => execFileSync('git', ['-C', dir, ...args], { encoding: 'utf8' }).trim(),
  digest = (v) =>
    createHash('sha256')
      .update(JSON.stringify(v, (_, x) => (ArrayBuffer.isView(x) ? [...x] : x)))
      .digest('hex'),
  cases = [];
for (const name of ['south-building', 'gerrard-hall', 'person-hall', 'graham-hall']) {
  const f = loadSample(name),
    sample = JSON.parse(
      readFileSync(resolve(root, `tests/fixtures/lab-images/${name}/sample.json`)),
    );
  cases.push({
    name,
    source: 'cached COLMAP SIFT/raw putative matches',
    calibration: 'provided COLMAP K/distortion, held fixed',
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
  images: g.frames.map((v) => ({ ...v, keypoints: Float64Array.from(v.keypoints) })),
  pairs: g.pairs.map((p) => ({ ...p, matches: Uint32Array.from(p.matches) })),
  reference: gr.frames.map((v) => ({ id: v.id, ...v.reference })),
  controls: { ...LAB },
});
const p = loadPyramid();
cases.push({
  name: 'pyramid-8',
  source: 'cached CPU SIFT/mutual ratio-0.8 matches, 8 frames',
  calibration: 'estimated shared focal; initial guess 0.8*width',
  images: camerasAt(p.frames, 0.8, false),
  pairs: p.pairs,
  controls: { ...LAB, focalSearch: true, refineSharedFocal: true },
});
const report = {
  createdAt: new Date().toISOString(),
  baseline: git(baseline, 'rev-parse', 'HEAD'),
  current: git(root, 'rev-parse', 'HEAD'),
  currentDirty: !!git(root, 'status', '--porcelain'),
  node: process.version,
  backend: 'CPU f64',
  physicalGpuMeasured: false,
  protocol:
    'Four modes: baseline/current, default-off/opt-in. Same cached RGB-derived features and matches; no decode, extraction, GPU or dense stage. One sequential pass per mode; times are diagnostic, not a speed comparison. COLMAP poses are estimates, only evaluator inputs. One similarity alignment per whole connected map. Pyramid relief is qualitative, not ground-truth accuracy. This is not the full current 47-frame Pyramid video.',
  sourceHashes: Object.fromEntries(
    [
      'src/reconstruction/mapper.ts',
      'src/reconstruction/focal-search.ts',
      'src/reconstruction/directional-tracks.ts',
      'src/reconstruction/bundle-adjustment.ts',
      'src/reconstruction/joint-focal.ts',
    ].map((path) => [
      path,
      createHash('sha256')
        .update(readFileSync(resolve(root, path)))
        .digest('hex'),
    ]),
  ),
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
          elapsedMs: performance.now() - start,
          status: 'map',
          geometrySha256: digest(model),
          registered: model.views.length,
          points: model.landmarks.length,
          observations: model.landmarks.reduce((n, p) => n + p.observations.length, 0),
          meanReprojectionErrorPx: r.diagnostics.meanReprojectionErrorPx,
          pending: r.tracking?.untriangulatedTrackIds.length,
          bearingObservationsInFinalBundle:
            r.diagnostics.bundleAdjustments.at(-1)?.directionalObservations ?? 0,
          focal: model.views[0]?.camera.fx,
          focalScoring: r.diagnostics.focalSearch?.scoring,
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
        });
      }
      if (digest(input) !== sha) throw new Error(`Input mutated: ${c.name}`);
    }
  row.defaultGeometryUnchanged =
    row.modes[0].status === 'map' && row.modes[0].geometrySha256 === row.modes[1].geometrySha256;
  report.cases.push(row);
  writeFileSync(output, JSON.stringify(report, null, 2) + '\n');
  process.stdout.write(JSON.stringify(row) + '\n');
}
report.completed = true;
writeFileSync(output, JSON.stringify(report, null, 2) + '\n');
