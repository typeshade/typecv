import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { focalComponentsFixture } from '../tests/fixtures/focal-components.ts';
import { generalVideoFixture } from '../tests/fixtures/general-video.ts';
import { orbitFixture } from '../tests/typeshade/focal-search-cases.mjs';
import { compareCameraModels } from './geometry-comparison.mjs';

// CPU geometry evaluation with supplied correspondences. Truth enters only the evaluator;
// this is neither an RGB front-end benchmark nor a native GPU performance measurement.
const currentRoot = fileURLToPath(new URL('..', import.meta.url));
const at = process.argv.indexOf('--baseline');
const roots = [
  ...(at < 0 ? [] : [{ label: 'baseline', root: resolve(process.argv[at + 1]) }]),
  { label: 'candidate', root: currentRoot },
];
const controls = {
  ransacIterations: 64,
  bundleIterations: 4,
  finalBundleIterations: 8,
  refineSharedFocal: true,
  focalSearch: { refineIterations: 12 },
};
const cases = [
  ...[48, 64, 80].map((points) => ({
    name: `connected-${points}`,
    fixture: orbitFixture({ points }),
  })),
  ...[48, 64, 80].map((points) => ({
    name: `disconnected-prefix-${points}`,
    fixture: focalComponentsFixture(1.2, { points }),
  })),
  { name: 'disconnected-prefix-four', fixture: focalComponentsFixture(1.2, { prefixViews: 4 }) },
];
const digest = (value) => createHash('sha256').update(value).digest('hex');
const quantile = (values, fraction) => {
  const sorted = [...values].sort((a, b) => a - b),
    at = (values.length - 1) * fraction;
  return (
    sorted[Math.floor(at)] +
    (at - Math.floor(at)) * (sorted[Math.ceil(at)] - sorted[Math.floor(at)])
  );
};
const runs = [];
for (const { label, root } of roots) {
  const { reconstructIncremental } = await import(pathToFileURL(resolve(root, 'dist/index.js')));
  const sourceFiles = ['src/sfm/focal/focal-search.ts', 'src/sfm/mapper/mapper.ts'];
  const source = {
    commit: execFileSync('git', ['rev-parse', 'HEAD'], { cwd: root, encoding: 'utf8' }).trim(),
    modified:
      execFileSync('git', ['status', '--porcelain', '--', ...sourceFiles], {
        cwd: root,
        encoding: 'utf8',
      }).trim().length > 0,
    mapperSourceSha256: digest(
      sourceFiles.map((path) => path + '\n' + readFileSync(resolve(root, path), 'utf8')).join('\n'),
    ),
  };
  const results = [];
  for (const { name, fixture } of cases) {
    const run = reconstructIncremental(fixture.images, fixture.pairs, controls);
    const { reconstruction: model } = run;
    const alignment = compareCameraModels({ views: fixture.truth.views }, model).alignment;
    if (alignment.status !== 'aligned') throw new Error(`${name}: no common similarity gauge`);
    const { scale, rotation: R, translation: t } = alignment;
    const pointErrors = model.landmarks.map((point) => {
      const transformed = [0, 1, 2].map(
        (r) =>
          scale *
            (R[3 * r] * point.xyz[0] + R[3 * r + 1] * point.xyz[1] + R[3 * r + 2] * point.xyz[2]) +
          t[r],
      );
      const truth = fixture.truth.points[point.observations[0].featureId];
      return Math.hypot(...transformed.map((value, i) => value - truth[i]));
    });
    const finalRatio = model.views[0].camera.fx / fixture.width;
    results.push({
      case: name,
      registeredViews: model.views.length,
      expectedViews: fixture.truth.views.length,
      unexpectedViews: model.views.filter(
        (v) => !fixture.truth.views.some((truth) => truth.id === v.id),
      ).length,
      landmarks: model.landmarks.length,
      observations: model.landmarks.reduce((sum, p) => sum + p.observations.length, 0),
      calibrationSubset: run.diagnostics.focalSearch.views,
      searchRelativeFocalError: Math.abs(run.diagnostics.focalSearch.ratio / fixture.ratio - 1),
      finalRelativeFocalError: Math.abs(finalRatio / fixture.ratio - 1),
      meanReprojectionErrorPx: run.diagnostics.meanReprojectionErrorPx,
      centerRmse: alignment.centerRmse,
      meanRotationErrorDeg: alignment.meanRotationDifferenceDeg,
      medianPointError: quantile(pointErrors, 0.5),
      p95PointError: quantile(pointErrors, 0.95),
      finalGeometrySha256: digest(JSON.stringify(model)),
    });
  }
  const degenerate = ['rotation', 'planar', 'far'].map((scenario) => {
    const fixture = generalVideoFixture(scenario).input;
    try {
      const result = reconstructIncremental(fixture.images, fixture.pairs, controls);
      return { scenario, initialized: true, landmarks: result.reconstruction.landmarks.length };
    } catch (error) {
      if (error.name !== 'ReconstructionInitializationError') throw error;
      return {
        scenario,
        initialized: false,
        diagnosticPairs: error.pairVerification.length,
        focalFailureAttempts: error.focalSearch?.attempts.length ?? 0,
      };
    }
  });
  runs.push({ label, source, results, degenerate });
}
console.log(
  JSON.stringify(
    {
      mode: 'CPU geometry with supplied identities; one camera-center Sim(3) alignment per map',
      units: 'Camera centers and points are in synthetic fixture units, not meters',
      truthIsEvaluatorOnly: true,
      limitations:
        'Small deterministic cases with one shared focal and nonplanar background. No native GPU timing, real-video quality claim, dense surface evaluation or general success-rate estimate.',
      controls,
      runs,
    },
    null,
    2,
  ),
);
