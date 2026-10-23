import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { generalVideoFixture } from '../tests/fixtures/general-video.ts';
import { compareCameraModels } from './geometry-comparison.mjs';
import { inBrowser } from '../tests/typeshade/browser.mjs';

const option = (name, fallback) =>
  process.argv.find((a) => a.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;
const repeats = Number(option('repeats', '6'));
if (!Number.isSafeInteger(repeats) || repeats < 2 || repeats > 12 || repeats % 2)
  throw new RangeError('--repeats must be an even integer from 2 through 12');
const output = resolve(
  option('output', 'docs/experiments/general-video/rotation-tracking-rgb-comparison.json'),
);
const sha = (b) => createHash('sha256').update(b).digest('hex');
const median = (v) => {
  const a = [...v].sort((x, y) => x - y);
  return (a[(a.length - 1) >> 1] + a[a.length >> 1]) / 2;
};
const cases = [],
  labels = [];
for (const scenario of ['translation', 'rotation', 'rotation-zoom']) {
  const root = resolve('data/general-video-fixtures', scenario);
  const input = JSON.parse(await readFile(resolve(root, 'input/frames.json'), 'utf8')),
    calibration = JSON.parse(await readFile(resolve(root, 'known-calibration.json'), 'utf8'));
  const frames = [];
  for (const f of input.frames) {
    const bytes = await readFile(resolve(root, 'input', f.image));
    assert.equal(sha(bytes), f.sha256);
    frames.push({
      id: f.id,
      width: f.width,
      height: f.height,
      camera: calibration.frames.find((v) => v.id === f.id).camera,
      png: bytes.toString('base64'),
      sha256: f.sha256,
      mime: 'image/png',
    });
  }
  labels.push({
    scenario,
    calibration: 'known-synthetic',
    truth: generalVideoFixture(scenario).truth.views,
  });
  cases.push({ frames, maxFeatures: 1000 });
}
const pyramid = [];
for (let id = 0; id < 8; id++) {
  const bytes = await readFile(
    resolve('website/public/sample/pyramid', `frame-${String(id).padStart(2, '0')}.jpg`),
  );
  pyramid.push({ id, png: bytes.toString('base64'), sha256: sha(bytes), mime: 'image/jpeg' });
}
labels.push({
  scenario: 'pyramid-video-frames',
  calibration: 'assumed-0.8-times-width',
  truth: null,
});
cases.push({ frames: pyramid, maxFeatures: 400 });

const { result, errors } = await inBrowser(
  async ({ cases, repeats }) => {
    const api = await import('/dist/gpu.js');
    const core = await import('/dist/index.js');
    const { judgePair, incrementalControls } = await import('/dist/reconstruction/mapper.js');
    const { pairSeed } = await import('/dist/reconstruction/identity.js');
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('WebGPU adapter unavailable');
    const description = [
      adapter.info?.vendor,
      adapter.info?.architecture,
      adapter.info?.device,
      adapter.info?.description,
    ]
      .filter(Boolean)
      .join(' ');
    const gpu = await adapter.requestDevice(),
      gpuErrors = [];
    gpu.addEventListener('uncapturederror', (e) => gpuErrors.push(e.error.message));
    const device = await api.ShadeDevice.open({
      gpuDevice: gpu,
      solverCheckpoints: true,
      boundedSubmissions: false,
      maxModelsPerBatch: 32,
    });
    let peak = 0,
      peakActive = 0;
    for (const key of ['allocate', 'allocateU32', 'allocateF64']) {
      const original = device[key].bind(device);
      device[key] = (...args) => {
        const value = original(...args);
        peak = Math.max(peak, device.statistics.liveBytes);
        peakActive = Math.max(
          peakActive,
          device.statistics.liveBytes - device.statistics.pooledBytes,
        );
        return value;
      };
    }
    const casesOut = [],
      controls = {
        ransacIterations: 512,
        bundleIterations: 2,
        finalBundleIterations: 4,
        refineSharedFocal: false,
        maxErrorPx: 3,
        minInitialAngleDeg: 1,
        minTriangulationAngleDeg: 0.5,
      };
    try {
      for (const entry of cases) {
        const beforeFrontend = device.statistics,
          frontStart = performance.now(),
          handles = [],
          images = [],
          pairs = [],
          featureCounts = [];
        peak = 0;
        peakActive = 0;
        let decodeMs = 0,
          featureMs = 0,
          matchingMs = 0,
          releaseMs = 0;
        try {
          for (const source of entry.frames) {
            let start = performance.now();
            const bitmap = await createImageBitmap(
              new Blob([Uint8Array.from(atob(source.png), (c) => c.charCodeAt(0))], {
                type: source.mime,
              }),
            );
            let pixels, width, height;
            try {
              width = bitmap.width;
              height = bitmap.height;
              const canvas = document.createElement('canvas');
              canvas.width = width;
              canvas.height = height;
              const ctx = canvas.getContext('2d', { willReadFrequently: true });
              ctx.drawImage(bitmap, 0, 0);
              pixels = ctx.getImageData(0, 0, width, height).data;
            } finally {
              bitmap.close();
            }
            decodeMs += performance.now() - start;
            start = performance.now();
            const features = await api.extractFeaturesGpuPyramidResident(
              device,
              pixels,
              width,
              height,
              { maxFeatures: entry.maxFeatures, threshold: 0.045, smoothDescriptors: true },
            );
            handles.push(features);
            featureCounts.push(features.count);
            images.push({
              id: source.id,
              camera: source.camera ?? {
                fx: width * 0.8,
                fy: width * 0.8,
                cx: (width - 1) / 2,
                cy: (height - 1) / 2,
                k1: 0,
                k2: 0,
              },
              keypoints: await features.readKeypoints(),
            });
            featureMs += performance.now() - start;
          }
          for (let second = 1; second < images.length; second++)
            for (let first = Math.max(0, second - 3); first < second; first++) {
              const start = performance.now();
              const m = await api.matchResidentFeaturesGpu(
                device,
                handles[first],
                handles[second],
                { ratio: 0.82, maxDistance: 80 },
              );
              matchingMs += performance.now() - start;
              if (m.matches.length >= 16)
                pairs.push({
                  first: images[first].id,
                  second: images[second].id,
                  matches: m.matches,
                  distances: m.distances,
                });
            }
        } finally {
          const start = performance.now();
          for (const h of handles) await h.dispose();
          releaseMs = performance.now() - start;
        }
        const frontendMs = performance.now() - frontStart,
          afterFrontend = device.statistics;
        const row = {
          featureCounts,
          pairs: pairs.length,
          frontend: {
            elapsedMs: frontendMs,
            decodeMs,
            featureMs,
            matchingMs,
            releaseMs,
            uploadBytes:
              afterFrontend.applicationUploadBytes - beforeFrontend.applicationUploadBytes,
            readbackBytes:
              afterFrontend.applicationReadbackBytes - beforeFrontend.applicationReadbackBytes,
            readCalls: afterFrontend.reads - beforeFrontend.reads,
            readWaves: afterFrontend.readWaves - beforeFrontend.readWaves,
            peakManagedBytes: peak,
            peakActiveBytes: peakActive,
          },
          runs: [],
        };
        const inputText = JSON.stringify({ images, pairs }),
          inputDigest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(inputText));
        row.mapperInputSha256 = [...new Uint8Array(inputDigest)]
          .map((n) => n.toString(16).padStart(2, '0'))
          .join('');
        for (let round = -1; round < repeats; round++)
          for (const enabled of round % 2 === 0 ? [true, false] : [false, true]) {
            const before = device.statistics,
              stages = [];
            peak = before.liveBytes;
            peakActive = before.liveBytes - before.pooledBytes;
            let mapped, failure;
            const start = performance.now();
            try {
              mapped = await api.reconstructIncrementalGpu(device, images, pairs, {
                ...controls,
                rotationTracking: enabled,
                onStage: (s) =>
                  stages.push({
                    operation: s.operation,
                    elapsedMs: s.elapsedMs,
                    uploadBytes: s.execution.applicationUploadBytes,
                    readbackBytes: s.execution.applicationReadbackBytes,
                  }),
              });
            } catch (error) {
              if (error.name !== 'ReconstructionInitializationError') throw error;
              failure = error;
            }
            const mappingMs = performance.now() - start,
              after = device.statistics,
              tracking = mapped?.tracking ?? failure?.tracking;
            const trackingSummary = tracking
              ? {
                  tracks: tracking.tracks.length,
                  observations: tracking.tracks.reduce((n, t) => n + t.observations.length, 0),
                  pending: tracking.untriangulatedTrackIds.length,
                  rotations: tracking.rotations.map((p) => ({
                    first: p.first,
                    second: p.second,
                    R: [...p.rotation.R],
                    inliers: p.rotation.inlierRows.length,
                    meanReprojectionErrorPx: p.rotation.meanReprojectionErrorPx,
                  })),
                }
              : null;
            row.runs.push({
              round,
              warmup: round < 0,
              enabled,
              status: mapped ? 'map' : 'initialization-rejected',
              mappingMs,
              composedPipelineMs: frontendMs + mappingMs,
              uploadBytes: after.applicationUploadBytes - before.applicationUploadBytes,
              readbackBytes: after.applicationReadbackBytes - before.applicationReadbackBytes,
              readCalls: after.reads - before.reads,
              readWaves: after.readWaves - before.readWaves,
              peakManagedMapperBytes: peak,
              peakActiveMapperBytes: peakActive,
              liveBuffersAfter: after.liveBuffers,
              pooledBytesAfter: after.pooledBytes,
              activeBytesAfter: after.liveBytes - after.pooledBytes,
              registered: mapped?.reconstruction.views.length ?? 0,
              points: mapped?.reconstruction.landmarks.length ?? 0,
              tracking: trackingSummary,
              tracks: mapped?.tracks.length ?? null,
              trackObservations:
                mapped?.tracks.reduce((n, t) => n + t.observations.length, 0) ?? null,
              meanReprojectionErrorPx: mapped?.diagnostics.meanReprojectionErrorPx ?? null,
              stages,
              model: mapped
                ? {
                    views: mapped.reconstruction.views.map((v) => ({
                      ...v,
                      pose: { R: [...v.pose.R], t: [...v.pose.t] },
                    })),
                    landmarks: mapped.reconstruction.landmarks.map((p) => ({
                      xyz: [...p.xyz],
                      observations: p.observations.map((o) => ({
                        viewId: o.viewId,
                        featureId: o.featureId,
                      })),
                    })),
                  }
                : null,
            });
            if (after.liveBytes !== after.pooledBytes)
              throw new Error('Mapper retained active GPU buffers');
          }
        if (JSON.stringify({ images, pairs }) !== inputText) throw new Error('A/B input mutation');
        // Inspect the actual GPU epipolar graph once, outside every timed sample. A failed
        // initialization must not be described as having zero existing 2D tracks by assumption.
        const inspectStarted = performance.now(),
          verified = [],
          { controls: c } = incrementalControls(images, pairs, {
            ...controls,
            rotationTracking: false,
          });
        for (const pair of pairs) {
          const a = images.find((v) => v.id === pair.first),
            b = images.find((v) => v.id === pair.second),
            pixels = new Float64Array(pair.matches.length * 2);
          for (let k = 0; k < pair.matches.length; k += 2)
            pixels.set(
              [
                a.keypoints[2 * pair.matches[k]],
                a.keypoints[2 * pair.matches[k] + 1],
                b.keypoints[2 * pair.matches[k + 1]],
                b.keypoints[2 * pair.matches[k + 1] + 1],
              ],
              2 * k,
            );
          const v = await api.verifyImagePairGpu(device, pixels, a.camera, b.camera, {
            iterations: c.iterations,
            seed: pairSeed(c.seed, pair.first, pair.second),
            threshold: c.maxError / Math.min(a.camera.fx, a.camera.fy, b.camera.fx, b.camera.fy),
            refinementIterations: 0,
            minParallaxDeg: Math.min(c.pointAngle, 0.1),
            maxReprojectionErrorPx: c.maxError,
            homographyThresholdPx: c.maxError,
            homographyIterations: Math.min(c.iterations, 500),
          });
          const judged = judgePair(pair, v, c);
          if (judged.accepted) verified.push(judged.accepted);
        }
        const tracks = core.buildTracks(images, verified).tracks;
        row.baselineEpipolarGraph = {
          tracks: tracks.length,
          observations: tracks.reduce((n, t) => n + t.observations.length, 0),
          coveredFrames: new Set(tracks.flatMap((t) => t.observations.map((o) => o.viewId))).size,
          inspectionMs: performance.now() - inspectStarted,
        };
        casesOut.push(row);
      }
    } finally {
      await device.dispose();
      gpu.destroy();
    }
    return {
      description,
      userAgent: navigator.userAgent,
      backend: device.backend,
      software:
        /swiftshader|software|llvmpipe/i.test(description) || adapter.isFallbackAdapter === true,
      gpuErrors,
      controls,
      cases: casesOut,
      liveBuffersAfterDisposal: device.statistics.liveBuffers,
    };
  },
  { cases, repeats },
  { timeoutMs: 600000 },
);
assert.deepEqual(errors, []);
assert.deepEqual(result.gpuErrors, []);
assert.equal(result.liveBuffersAfterDisposal, 0);

function pointAgreement(reference, candidate, alignment) {
  if (alignment.status !== 'aligned') return null;
  const byObservation = new Map(
      reference.landmarks.flatMap((p) =>
        p.observations.map((o) => [`${o.viewId}:${o.featureId}`, p]),
      ),
    ),
    errors = [];
  for (const point of candidate.landmarks) {
    const other = point.observations
      .map((o) => byObservation.get(`${o.viewId}:${o.featureId}`))
      .find(Boolean);
    if (!other) continue;
    const xyz = [0, 1, 2].map(
      (r) =>
        alignment.scale *
          (alignment.rotation[3 * r] * point.xyz[0] +
            alignment.rotation[3 * r + 1] * point.xyz[1] +
            alignment.rotation[3 * r + 2] * point.xyz[2]) +
        alignment.translation[r],
    );
    errors.push(Math.hypot(...xyz.map((x, i) => x - other.xyz[i])));
  }
  return {
    matchedPoints: errors.length,
    rmse: errors.length ? Math.sqrt(errors.reduce((n, x) => n + x * x, 0) / errors.length) : null,
    max: errors.length ? Math.max(...errors) : null,
  };
}
for (const [i, row] of result.cases.entries()) {
  const label = labels[i];
  row.scenario = label.scenario;
  row.calibration = label.calibration;
  row.inputs = cases[i].frames.map(({ png, ...rest }) => rest);
  for (const run of row.runs) {
    run.cameraTruthAlignment =
      run.model && label.truth
        ? compareCameraModels({ views: label.truth }, run.model).alignment
        : null;
    run.falseGeometry = label.truth && label.scenario !== 'translation' && run.points > 0;
    for (const pair of run.tracking?.rotations ?? [])
      if (label.truth) {
        const a = label.truth[pair.first].pose.R,
          b = label.truth[pair.second].pose.R;
        let trace = 0;
        for (let r = 0; r < 3; r++)
          for (let c = 0; c < 3; c++) {
            let x = 0;
            for (let k = 0; k < 3; k++) x += b[3 * r + k] * a[3 * c + k];
            trace += x * pair.R[3 * r + c];
          }
        pair.rotationErrorDeg =
          (Math.acos(Math.max(-1, Math.min(1, (trace - 1) / 2))) * 180) / Math.PI;
      }
  }
  for (let round = 0; round < repeats; round++) {
    const a = row.runs.find((r) => r.round === round && !r.enabled),
      b = row.runs.find((r) => r.round === round && r.enabled);
    if (a.model && b.model) {
      const alignment = compareCameraModels(a.model, b.model).alignment;
      b.agreementWithOff = {
        modelExactlyEqual: JSON.stringify(a.model) === JSON.stringify(b.model),
        alignment,
        points: pointAgreement(a.model, b.model, alignment),
        note: 'Agreement only, not ground-truth accuracy; one alignment for the complete map.',
      };
    }
  }
  row.summary = Object.fromEntries(
    [false, true].map((enabled) => {
      const runs = row.runs.filter((r) => !r.warmup && r.enabled === enabled),
        first = runs[0];
      return [
        enabled ? 'on' : 'off',
        {
          registered: first.registered,
          points: first.points,
          tracking: first.tracking ?? {
            tracks: first.tracks,
            observations: first.trackObservations,
          },
          medianMappingMs: median(runs.map((r) => r.mappingMs)),
          medianComposedPipelineMs: median(runs.map((r) => r.composedPipelineMs)),
          minMappingMs: Math.min(...runs.map((r) => r.mappingMs)),
          maxMappingMs: Math.max(...runs.map((r) => r.mappingMs)),
          peakManagedMapperBytes: Math.max(...runs.map((r) => r.peakManagedMapperBytes)),
          peakActiveMapperBytes: Math.max(...runs.map((r) => r.peakActiveMapperBytes)),
          medianUploadBytes: median(runs.map((r) => r.uploadBytes)),
          medianReadbackBytes: median(runs.map((r) => r.readbackBytes)),
          falseGeometry: runs.some((r) => r.falseGeometry),
          allRunsSameRegistration: runs.every(
            (r) => r.registered === first.registered && r.points === first.points,
          ),
          allMappedPairsExactlyEqual:
            enabled && first.status === 'map'
              ? runs.every((r) => r.agreementWithOff?.modelExactlyEqual)
              : null,
        },
      ];
    }),
  );
  row.summary.registrationGain = row.summary.on.registered - row.summary.off.registered;
  row.summary.mappingTimeChangePercent =
    100 * (row.summary.on.medianMappingMs / row.summary.off.medianMappingMs - 1);
  row.summary.composedPipelineTimeChangePercent =
    100 * (row.summary.on.medianComposedPipelineMs / row.summary.off.medianComposedPipelineMs - 1);
  for (const run of row.runs) delete run.model;
  console.log(
    JSON.stringify({
      scenario: row.scenario,
      epipolarGraph: row.baselineEpipolarGraph,
      summary: row.summary,
    }),
  );
}
const report = {
  createdAt: new Date().toISOString(),
  head: execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).trim(),
  workingTreeDirty:
    execFileSync('git', ['status', '--porcelain'], { encoding: 'utf8' }).trim() !== '',
  node: process.version,
  platform: process.platform,
  protocol: {
    repeats,
    warmupsPerMode: 1,
    order: 'off/on then on/off in alternating rounds; one persistent device',
    input:
      'Identical RGB-derived keypoints/matches shared by both modes; no truth poses/depth/masks/identity correspondences enter inference',
    frontend:
      'Decode, features, upload, matching, readback and descriptor disposal measured once per case. Composed pipeline time adds that same measured frontend to each mapper run.',
    limits:
      'No raw-video codec, native GPU, GS/MVS or real gameplay benchmark. Six warm samples are not a statistical performance guarantee. Epipolar-graph inspection is separate from timings.',
    calibration:
      'Synthetic cases use supplied known K. Pexels pyramid uses assumed 0.8*width K and has no ground-truth camera/depth accuracy claim.',
    memory:
      'Managed device allocation peaks include the reusable pool; active peaks subtract pooled bytes. Both exclude driver, decoded pixels and host graph memory. Released pooled buffers are not leaks; active bytes must return to zero after every mapping run.',
  },
  nativeGpuMeasured: false,
  sourceHashes: Object.fromEntries(
    await Promise.all(
      [
        'scripts/benchmark-rotation-tracking-rgb.mjs',
        'src/geometry/rotation.ts',
        'src/reconstruction/mapper.ts',
        'src/gpu/mapper.ts',
        'src/reconstruction/tracking.ts',
        'src/reconstruction/focal-search.ts',
        'dist/gpu.js',
        'dist/gpu/index.js',
      ].map(async (p) => [p, sha(await readFile(p))]),
    ),
  ),
  measurements: result,
};
await mkdir(dirname(output), { recursive: true });
await writeFile(output, JSON.stringify(report, null, 2) + '\n');
