/** Typed-data boundary and output adapter; all reconstruction lives in the TypeCV package. */
import fs from 'node:fs';
import path from 'node:path';
import { performance } from 'node:perf_hooks';
import { reconstructIncremental, ReconstructionInitializationError } from 'typecv';
const [input, output, optionsPath] = process.argv.slice(2);
if (!input || !output)
  throw new Error('Usage: node scripts/reconstruct-sequence.mjs sequence.json output');
const data = JSON.parse(fs.readFileSync(input, 'utf8'));
if (
  data.views.some(
    (v) =>
      !Array.isArray(v.keypoints) ||
      v.keypoints.some((x) => typeof x !== 'number' || !Number.isFinite(x)),
  )
)
  throw new RangeError('Invalid JSON keypoints');
if (
  data.pairs.some(
    (p) =>
      !Array.isArray(p.matches) ||
      p.matches.some((x) => !Number.isSafeInteger(x) || x < 0 || x > 0xffffffff),
  )
)
  throw new RangeError('Invalid JSON feature indices');
const views = data.views.map((v) => ({ ...v, keypoints: Float64Array.from(v.keypoints) }));
const pairs = data.pairs.map((p) => ({
  first: p.first,
  second: p.second,
  matches: Uint32Array.from(p.matches),
  ...(p.distances ? { distances: Float64Array.from(p.distances) } : {}),
}));
if (fs.existsSync(output) && fs.readdirSync(output).length)
  throw new Error('Output directory must be empty to avoid stale model files');
fs.mkdirSync(output, { recursive: true });
const start = performance.now();
try {
  const options = {
    seed: 42,
    ransacIterations: 750,
    ...(optionsPath ? JSON.parse(fs.readFileSync(optionsPath, 'utf8')) : {}),
  };
  const result = reconstructIncremental(views, pairs, options);
  const model = {
    views: result.reconstruction.views.map((v) => ({
      ...v,
      pose: { R: [...v.pose.R], t: [...v.pose.t] },
    })),
    landmarks: result.reconstruction.landmarks.map((p) => ({ ...p, xyz: [...p.xyz] })),
  };
  const summary = {
    source: data.metadata,
    options,
    elapsedMs: performance.now() - start,
    registeredViews: model.views.length,
    landmarks: model.landmarks.length,
    observations: model.landmarks.reduce((sum, p) => sum + p.observations.length, 0),
    tracks: result.tracks.length,
    unregistered: result.unregistered,
    diagnostics: result.diagnostics,
  };
  fs.writeFileSync(path.join(output, 'summary.json'), JSON.stringify(summary, null, 2) + '\n');
  fs.writeFileSync(path.join(output, 'model.json'), JSON.stringify(model) + '\n');
  fs.writeFileSync(
    path.join(output, 'cloud.ply'),
    `ply\nformat ascii 1.0\nelement vertex ${model.landmarks.length}\nproperty double x\nproperty double y\nproperty double z\nend_header\n${model.landmarks.map((p) => p.xyz.join(' ')).join('\n')}\n`,
  );
  console.log(
    JSON.stringify({
      registered: summary.registeredViews,
      points: summary.landmarks,
      errorPx: summary.diagnostics.meanReprojectionErrorPx,
      elapsedMs: summary.elapsedMs,
    }),
  );
} catch (error) {
  if (!(error instanceof ReconstructionInitializationError)) throw error;
  fs.writeFileSync(
    path.join(output, 'initialization-failure.json'),
    JSON.stringify(
      { source: data.metadata, reason: error.message, pairs: error.pairVerification },
      null,
      2,
    ) + '\n',
  );
  console.error(error.message);
  process.exitCode = 1;
}
