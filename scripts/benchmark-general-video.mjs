import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { crc32, deflateSync } from 'node:zlib';
import { reconstructIncremental, ReconstructionInitializationError } from '../src/index.ts';
import {
  generalVideoFixture,
  GENERAL_VIDEO_CASES,
  renderGeneralVideoFrame,
} from '../tests/fixtures/general-video.ts';
import { compareCameraModels } from './geometry-comparison.mjs';

const option = (name, fallback) =>
  process.argv.find((v) => v.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;
const output = resolve(option('output', 'docs/experiments/general-video/geometry.json'));
const renderDirectory = option('render-dir', '');
const sha256 = (value) => createHash('sha256').update(value).digest('hex');
const git = (...args) => execFileSync('git', args, { encoding: 'utf8' }).trim();
const sourcePaths = (await import('node:fs/promises')).readdir;
async function sources(directory) {
  const entries = await sourcePaths(directory, { withFileTypes: true });
  return (
    await Promise.all(
      entries.map((entry) =>
        entry.isDirectory()
          ? sources(`${directory}/${entry.name}`)
          : [`${directory}/${entry.name}`],
      ),
    )
  ).flat();
}
const files = [
  ...(await sources('src')),
  'tests/fixtures/general-video.ts',
  'scripts/benchmark-general-video.mjs',
  'scripts/geometry-comparison.mjs',
];
const sourceHashes = Object.fromEntries(
  await Promise.all(
    files
      .filter((file) => file.endsWith('.ts') || file.endsWith('.mjs'))
      .sort()
      .map(async (file) => [file, sha256(await readFile(file))]),
  ),
);
const report = {
  version: 'typecv-general-video-audit-v1',
  createdAt: new Date().toISOString(),
  head: git('rev-parse', 'HEAD'),
  workingTreeDirty: git('status', '--porcelain') !== '',
  sourceHashes,
  sourceManifestSha256: sha256(JSON.stringify(sourceHashes)),
  environment: {
    node: process.version,
    platform: process.platform,
    arch: process.arch,
    backend: 'independent-f64-cpu-reference',
    nativeGpuMeasured: false,
  },
  inputContract: {
    mode: 'calibrated-correspondence-geometry',
    calibration: 'known-synthetic',
    correspondences: 'supplied-synthetic-identities',
    rgbOnlyAccuracyClaim: false,
    truthPassedToReconstruction: false,
    executionMode: 'offline-all-views',
    trackingRetentionMeasured: false,
    rotationTracking: false,
    cutsDetected: false,
  },
  alignment:
    'One existing compareCameraModels Sim(3) per returned connected map; never per frame or shot. Unobservable alignment stays unavailable.',
  memory: {
    processPeakRssBytes: null,
    gpuPeakBytes: null,
    scope:
      'CPU process high-water RSS includes Node/runtime and all preceding scenarios; not device memory.',
  },
  scenarios: [],
  rgbFixtures: [],
};

for (const scenario of GENERAL_VIDEO_CASES) {
  const fixture = generalVideoFixture(scenario);
  const { images, pairs } = fixture.input;
  const controls = {
    // Keep the original audit's epipolar-only baseline reproducible after default changes.
    rotationTracking: false,
    ransacIterations: 256,
    bundleIterations: 2,
    finalBundleIterations: 4,
    ...(['mapped-rotation', 'reobserved'].includes(scenario) ? { initialPair: [0, 1] } : {}),
    ...(scenario === 'disconnected' ? { initialPair: [0, 2] } : {}),
  };
  const inputDigest = sha256(
    JSON.stringify({
      images: images.map((v) => ({ ...v, keypoints: [...v.keypoints] })),
      pairs: pairs.map((p) => ({ ...p, matches: [...p.matches] })),
      controls,
    }),
  );
  const started = performance.now();
  const row = {
    id: fixture.scenarioId,
    scenario,
    controls,
    inputSha256: inputDigest,
    inputFrames: images.length,
    inputCorrespondencePairs: pairs.length,
    expected: fixture.expected,
    backend: 'cpu',
    runtimeMs: 0,
    orientationOnlyTracking: 'disabled-in-this-epipolar-baseline',
    newTrackUncertainty: 'not-represented-by-current-result-contract',
    groundTruthDepthUsedAsInput: false,
    registrationRatio: 0,
    cameraAlignment: null,
    falseGeometryCommit: false,
    scopeGate: null,
  };
  try {
    const result = reconstructIncremental(images, pairs, controls);
    const model = result.reconstruction;
    const components = [...new Set(model.views.map((v) => fixture.truth.componentOfView[v.id]))];
    const cameraComparison =
      components.length === 1
        ? compareCameraModels({ views: fixture.truth.views }, model)
        : {
            alignment: {
              status: 'unavailable',
              reason: 'Returned map mixes ground-truth components; no alignment allowed',
            },
          };
    const alignment = cameraComparison.alignment;
    const errors = [];
    if (alignment.status === 'aligned')
      for (const point of model.landmarks) {
        const id = point.observations[0].featureId;
        const truth = fixture.truth.points[id];
        const transformed = [0, 1, 2].map(
          (r) =>
            alignment.scale *
              (alignment.rotation[r * 3] * point.xyz[0] +
                alignment.rotation[r * 3 + 1] * point.xyz[1] +
                alignment.rotation[r * 3 + 2] * point.xyz[2]) +
            alignment.translation[r],
        );
        errors.push(Math.hypot(...transformed.map((v, axis) => v - truth[axis])));
      }
    const observedPointIds = new Set(
      model.landmarks.flatMap((p) => p.observations.map((o) => o.featureId)),
    );
    const incorrectlyCreated = fixture.expected.unobservedAfterMotion.filter((id) =>
      observedPointIds.has(id),
    );
    row.status = 'map';
    row.registeredViewIds = model.views.map((v) => v.id);
    row.registrationRatio = model.views.length / images.length;
    row.landmarks = model.landmarks.length;
    row.verifiedTracks = result.tracks.length;
    row.unregistered = result.unregistered;
    row.initialPair = result.diagnostics.initialPair;
    row.initialPairCrossesShot =
      fixture.truth.shotOfView[row.initialPair[0]] !== fixture.truth.shotOfView[row.initialPair[1]];
    row.meanReprojectionErrorPx = result.diagnostics.meanReprojectionErrorPx;
    row.cameraAlignment = alignment;
    row.pointRmseAfterMapAlignment = errors.length
      ? Math.sqrt(errors.reduce((sum, e) => sum + e * e, 0) / errors.length)
      : null;
    row.meanRelativeFocalError =
      model.views.reduce(
        (sum, view) => sum + Math.abs(view.camera.fx / fixture.truth.views[view.id].camera.fx - 1),
        0,
      ) / model.views.length;
    row.focalErrorScope = 'Known intrinsics were supplied; this does not measure focal estimation.';
    row.groundTruthComponentsInReturnedMap = components;
    row.falseMapMerges = Math.max(0, components.length - 1);
    row.expectedUnresolvedPoints = fixture.expected.unobservedAfterMotion.length;
    row.incorrectlyCreatedUnobservedPoints = incorrectlyCreated.length;
    row.explicitDeferredTrackRecords = 0;
    row.falseGeometryCommit =
      fixture.expected.initialization === 'deferred' ||
      incorrectlyCreated.length > 0 ||
      components.length > 1;
    row.scopeGate = !row.falseGeometryCommit && model.views.length >= 2;
  } catch (error) {
    row.status =
      error instanceof ReconstructionInitializationError
        ? 'initialization-deferred-by-rejection'
        : 'error';
    row.message = error.message;
    row.pairs = error.pairVerification ?? null;
    row.landmarks = 0;
    row.explicitDeferredTrackRecords = 0;
    row.scopeGate =
      error instanceof ReconstructionInitializationError &&
      fixture.expected.initialization === 'deferred';
  }
  row.runtimeMs = performance.now() - started;
  row.scopeGateMeaning =
    'Only the stated geometry/non-fabrication condition; not full scenario, RGB tracking, uncertainty or cinematic support.';
  report.scenarios.push(row);
}

function png(width, height, pixels) {
  const signature = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
  const chunk = (name, bytes) => {
    const tag = Buffer.from(name),
      size = Buffer.alloc(4),
      checksum = Buffer.alloc(4);
    size.writeUInt32BE(bytes.length);
    checksum.writeUInt32BE(crc32(Buffer.concat([tag, bytes])));
    return Buffer.concat([size, tag, bytes, checksum]);
  };
  const header = Buffer.alloc(13);
  header.writeUInt32BE(width, 0);
  header.writeUInt32BE(height, 4);
  header[8] = 8;
  header[9] = 6;
  const rows = Buffer.alloc(height * (1 + width * 4));
  for (let y = 0; y < height; y++)
    rows.set(pixels.subarray(y * width * 4, (y + 1) * width * 4), y * (width * 4 + 1) + 1);
  return Buffer.concat([
    signature,
    chunk('IHDR', header),
    chunk('IDAT', deflateSync(rows)),
    chunk('IEND', Buffer.alloc(0)),
  ]);
}

if (renderDirectory)
  for (const scenario of ['translation', 'rotation', 'rotation-zoom']) {
    const fixture = generalVideoFixture(scenario);
    const location = resolve(renderDirectory, scenario);
    await mkdir(resolve(location, 'input/images'), { recursive: true });
    await mkdir(resolve(location, 'truth'), { recursive: true });
    const input = {
      version: 'typecv-analytic-rgb-fixture-v1',
      scene: 'static-analytic-room',
      frames: [],
    };
    const calibration = { provenance: 'known-synthetic', frames: [] };
    const truth = {
      provenance: 'ground-truth-evaluation-only',
      depth: 'camera-positive-Z-f32-little-endian',
      coordinates: 'TypeCV row-major world-to-camera, +Z forward, integer pixel centers',
      frames: [],
    };
    for (const view of fixture.truth.views) {
      const rendered = renderGeneralVideoFrame(view);
      const name = `frame-${view.id}.png`,
        bytes = png(rendered.width, rendered.height, rendered.rgba);
      const depth = Buffer.alloc(rendered.depthZ.length * 4);
      rendered.depthZ.forEach((value, i) => depth.writeFloatLE(value, i * 4));
      await writeFile(resolve(location, 'input/images', name), bytes);
      await writeFile(resolve(location, 'truth', `depth-${view.id}.f32`), depth);
      input.frames.push({
        id: view.id,
        image: `images/${name}`,
        width: rendered.width,
        height: rendered.height,
        editTime: fixture.truth.editTimes[view.id],
        sha256: sha256(bytes),
      });
      calibration.frames.push({ id: view.id, camera: rendered.camera });
      truth.frames.push({
        id: view.id,
        pose: { R: [...view.pose.R], t: [...view.pose.t] },
        depth: `depth-${view.id}.f32`,
        depthSha256: sha256(depth),
        shotId: fixture.truth.shotOfView[view.id],
      });
    }
    await writeFile(resolve(location, 'input/frames.json'), JSON.stringify(input, null, 2) + '\n');
    await writeFile(
      resolve(location, 'known-calibration.json'),
      JSON.stringify(calibration, null, 2) + '\n',
    );
    await writeFile(
      resolve(location, 'truth/evaluation-only.json'),
      JSON.stringify(truth, null, 2) + '\n',
    );
    report.rgbFixtures.push({
      scenario,
      directory: location,
      inputManifestSha256: sha256(JSON.stringify(input)),
      truthProvidedToReconstruction: false,
      rgbReconstructionExecuted: false,
      distinction:
        'Analytical room RGB/depth fixture, separate from the supplied-identity point geometry fixtures above.',
    });
  }
report.memory.processPeakRssBytes = process.resourceUsage().maxRSS * 1024;
report.summary = {
  scopeGatesPassed: report.scenarios.filter((r) => r.scopeGate).length,
  scopeGatesTotal: report.scenarios.length,
  falseGeometryCommits: report.scenarios.filter((r) => r.falseGeometryCommit).length,
  fullGeneralVideoSupport: false,
  nativeGpuMeasured: false,
};
await mkdir(dirname(output), { recursive: true });
await writeFile(output, JSON.stringify(report, null, 2) + '\n');
console.log(
  JSON.stringify(
    {
      output,
      summary: report.summary,
      scenarios: report.scenarios.map((r) => ({
        id: r.id,
        status: r.status,
        registered: r.registeredViewIds?.length ?? 0,
        points: r.landmarks,
        scopeGate: r.scopeGate,
        runtimeMs: Math.round(r.runtimeMs),
      })),
    },
    null,
    2,
  ),
);
if (report.scenarios.some((r) => !r.scopeGate)) process.exitCode = 1;
