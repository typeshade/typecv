// Synthetic numerical regression only. Output contains no landmark geometry or observations.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { bundleAdjustInverseDepth as current } from '../src/reconstruction/inverse-depth-bundle.ts';
import { scene } from '../tests/scene.ts';
import { initializeInverseDepth, inverseDepthJacobian } from '../src/geometry/inverse-depth.ts';
import { projectPoint } from '../src/geometry/pose.ts';

const usage =
  'Usage: node scripts/compare-inverse-depth-variants.mjs --baseline <frozen-esm-bundle> [--out <aggregate-json>]';
const args = process.argv.slice(2),
  options = new Map();
if (args.length === 1 && args[0] === '--help') {
  console.log(usage);
  process.exit(0);
}
for (let i = 0; i < args.length; i += 2) {
  const name = args[i],
    value = args[i + 1];
  if (
    !['--baseline', '--out'].includes(name) ||
    !value ||
    value.startsWith('--') ||
    options.has(name)
  )
    throw new Error(usage);
  options.set(name, value);
}
if (!options.has('--baseline')) throw new Error(usage);

const baseline = resolve(options.get('--baseline')),
  sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex'),
  sourceFiles = [
    'src/reconstruction/inverse-depth-bundle.ts',
    'src/reconstruction/sparse-schur.ts',
    'src/reconstruction/bundle-adjustment.ts',
    'src/geometry/inverse-depth.ts',
    'src/geometry/pose.ts',
    'src/geometry/camera.ts',
    'src/geometry/numerics.ts',
    'tests/scene.ts',
  ],
  [baselineBytes, harnessBytes, sourceHashes] = await Promise.all([
    readFile(baseline),
    readFile(new URL(import.meta.url)),
    Promise.all(
      sourceFiles.map(async (name) => [
        name,
        sha256(await readFile(new URL('../' + name, import.meta.url))),
      ]),
    ),
  ]),
  { bundleAdjustInverseDepth: reference } = await import(pathToFileURL(baseline).href);
assert.equal(typeof reference, 'function', 'Baseline must export bundleAdjustInverseDepth');

const { reconstruction } = scene({ count: 36, views: 6, noise: 0.12 }),
  base = reconstruction.views[0],
  repeated = { ...structuredClone(base), id: 100 },
  views = [...reconstruction.views, repeated],
  good = reconstruction.landmarks.map((p) => ({
    ...p,
    observations: [
      ...p.observations,
      {
        viewId: 100,
        featureId: p.trackId,
        pixel: projectPoint(base.camera, base.pose, p.xyz),
      },
    ],
  })),
  xyz = Float64Array.of(0.2, 0.1, 8),
  pixel = projectPoint(base.camera, base.pose, xyz),
  deferred = {
    trackId: 1000,
    xyz,
    observations: [
      { viewId: base.id, featureId: 1000, pixel },
      { viewId: 100, featureId: 1000, pixel },
    ],
  },
  landmarks = [...good.slice(0, 7), deferred, ...good.slice(7, 18)],
  bearings = Array.from({ length: 4 }, (_, i) => {
    const point = {
      reference: structuredClone(base.pose),
      parameters: Float64Array.of((i - 1.5) * 0.12, 0.1 - 0.08 * i, 0),
    };
    return {
      trackId: 2000 + i,
      point,
      directionOnly: true,
      referenceViewId: base.id,
      observations: views.map((v) => ({
        viewId: v.id,
        featureId: 2000 + i,
        pixel: inverseDepthJacobian(v.camera, v.pose, point).pixel,
      })),
    };
  }),
  constraints = [];
for (const [i, p] of good.slice(18).entries()) {
  if (i < bearings.length) constraints.push(bearings[i]);
  constraints.push({
    trackId: p.trackId,
    observations: p.observations,
    referenceViewId: p.observations[0].viewId,
    point: initializeInverseDepth({ ...base, pixel: p.observations[0].pixel }, p.xyz),
  });
}
const model = { views, landmarks },
  focalGroups = new Map(views.map((v, i) => [v.id, i % 2])),
  variableViewIds = views.slice(2, 5).map((v) => v.id),
  scenarios = [
    ['joint-fixedK', {}],
    ['joint-two-focals', { refineSharedFocal: true, focalGroups }],
    ['fixed-poses-fixedK-m0', { variableViewIds: [] }],
    ['fixed-poses-focalOnly', { variableViewIds: [], refineSharedFocal: true, focalGroups }],
    ['fixed-exterior-fixedK', { variableViewIds }],
    ['fixed-exterior-two-focals', { variableViewIds, refineSharedFocal: true, focalGroups }],
  ],
  records = [];
let numericComparisons = 0;
function compare(a, b, path) {
  if (typeof a === 'number' && typeof b === 'number') numericComparisons++;
  if (Object.is(a, b)) return;
  if (
    a === undefined ||
    b === undefined ||
    a === null ||
    b === null ||
    typeof a !== 'object' ||
    typeof b !== 'object'
  )
    throw new Error(`Mismatch ${path}: ${String(a)} vs ${String(b)}`);
  assert.equal(a.constructor, b.constructor, `constructor ${path}`);
  const keys = Object.keys(a);
  assert.deepEqual(keys, Object.keys(b), `keys ${path}`);
  for (const key of keys) compare(a[key], b[key], path + '.' + key);
}
for (const [name, controls] of scenarios)
  for (const solver of ['dense', 'sparse'])
    for (const iterations of [0, 2]) {
      const opt = { ...controls, solver, iterations },
        expected = reference(model, constraints, opt),
        actual = current(model, constraints, opt);
      compare(expected, actual, name + '/' + solver + '/' + iterations);
      if (iterations === 0) assert.deepEqual(actual.diagnostics.deferredTrackIds, [1000]);
      assert.ok(
        actual.estimates
          .filter((p) => p.trackId !== 1000 && !p.directionOnly)
          .every((p) => p.covariance),
      );
      assert.ok(
        actual.estimates
          .filter((p) => p.directionOnly)
          .every((p) => p.bearingCovariance && !p.covariance),
      );
      records.push({
        name,
        solver,
        iterations,
        bitwiseIdentical: true,
        observable: actual.uncertainty.observable,
        cameraParameters: actual.diagnostics.cameraParameters,
        finiteCovariances: actual.estimates.filter((p) => p.covariance).length,
        bearingOnlyCovariances: actual.estimates.filter(
          (p) => p.directionOnly && p.bearingCovariance,
        ).length,
        deferred: actual.diagnostics.deferredTrackIds,
      });
    }
const result = {
    provenance: {
      baselineSha256: sha256(baselineBytes),
      harnessSha256: sha256(harnessBytes),
      sourceSha256: Object.fromEntries(sourceHashes),
      runtime: process.version,
      protocol: 'Synthetic numerical comparison, not a timing or RGB quality benchmark.',
    },
    cases: records.length,
    numericComparisons,
    allOutputsBitwiseIdentical: true,
    records,
  },
  output = JSON.stringify(result, null, 2) + '\n';
if (options.has('--out')) {
  const file = resolve(options.get('--out'));
  await mkdir(dirname(file), { recursive: true });
  await writeFile(file, output);
}
process.stdout.write(output);
