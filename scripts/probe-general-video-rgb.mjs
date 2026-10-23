import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { generalVideoFixture } from '../tests/fixtures/general-video.ts';
import { compareCameraModels } from './geometry-comparison.mjs';
import { inBrowser } from '../tests/typeshade/browser.mjs';

const arg = (name, fallback) =>
  process.argv.find((value) => value.startsWith(`--${name}=`))?.slice(name.length + 3) ?? fallback;
const directory = resolve(arg('input', 'data/general-video-fixtures'));
const output = resolve(arg('output', 'docs/experiments/general-video/rgb-software.json'));
const cases = [];
for (const scenario of ['translation', 'rotation', 'rotation-zoom']) {
  const input = JSON.parse(
    await readFile(resolve(directory, scenario, 'input/frames.json'), 'utf8'),
  );
  const calibration = JSON.parse(
    await readFile(resolve(directory, scenario, 'known-calibration.json'), 'utf8'),
  );
  const frames = await Promise.all(
    input.frames.map(async (frame) => {
      const bytes = await readFile(resolve(directory, scenario, 'input', frame.image));
      assert.equal(createHash('sha256').update(bytes).digest('hex'), frame.sha256);
      return {
        id: frame.id,
        width: frame.width,
        height: frame.height,
        camera: calibration.frames.find((row) => row.id === frame.id).camera,
        png: bytes.toString('base64'),
        sha256: frame.sha256,
      };
    }),
  );
  // Only pixels and explicitly known calibration cross the inference boundary.
  // Scenario labels, GT poses/depth, identity matches and expected verdicts stay in this process.
  cases.push({ frames });
}
const { result, errors } = await inBrowser(
  async ({ cases }) => {
    const api = await import('/dist/gpu.js');
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) throw new Error('WebGPU adapter unavailable');
    const description = [
      adapter.info?.vendor,
      adapter.info?.architecture,
      adapter.info?.device,
      adapter.info?.description,
    ]
      .filter(Boolean)
      .join(' ');
    const software =
      /swiftshader|software|llvmpipe/i.test(description) || adapter.isFallbackAdapter === true;
    const deviceStarted = performance.now();
    const gpu = await adapter.requestDevice();
    const gpuErrors = [];
    gpu.addEventListener('uncapturederror', (event) => gpuErrors.push(event.error.message));
    const device = await api.ShadeDevice.open({
      gpuDevice: gpu,
      solverCheckpoints: true,
      boundedSubmissions: false,
      maxModelsPerBatch: 32,
    });
    const deviceOpenMs = performance.now() - deviceStarted;
    let peakManagedLiveBytes = 0;
    for (const name of ['allocate', 'allocateU32', 'allocateF64']) {
      const original = device[name].bind(device);
      device[name] = (...args) => {
        const value = original(...args);
        peakManagedLiveBytes = Math.max(peakManagedLiveBytes, device.statistics.liveBytes);
        return value;
      };
    }
    const results = [];
    try {
      for (const entry of cases) {
        const handles = [];
        const before = device.statistics,
          start = performance.now();
        const frames = [];
        for (const source of entry.frames) {
          const bytes = Uint8Array.from(atob(source.png), (c) => c.charCodeAt(0));
          const bitmap = await createImageBitmap(new Blob([bytes], { type: 'image/png' }));
          try {
            const canvas = document.createElement('canvas');
            canvas.width = source.width;
            canvas.height = source.height;
            const context = canvas.getContext('2d');
            context.drawImage(bitmap, 0, 0);
            frames.push({
              id: source.id,
              camera: source.camera,
              width: source.width,
              height: source.height,
              rgba: context.getImageData(0, 0, source.width, source.height).data,
            });
          } finally {
            bitmap.close();
          }
        }
        const decodeMs = performance.now() - start;
        const images = [],
          pairs = [],
          counts = [];
        let featureMs = 0,
          matchingMs = 0,
          mappingMs = 0,
          stages = [],
          model,
          rejection;
        try {
          for (const frame of frames) {
            const at = performance.now();
            const features = await api.extractFeaturesGpuPyramidResident(
              device,
              frame.rgba,
              frame.width,
              frame.height,
              { maxFeatures: 1000, threshold: 0.045, smoothDescriptors: true },
            );
            handles.push(features);
            counts.push(features.count);
            images.push({
              id: frame.id,
              camera: frame.camera,
              keypoints: await features.readKeypoints(),
            });
            featureMs += performance.now() - at;
          }
          for (let second = 1; second < frames.length; second++)
            for (let first = Math.max(0, second - 3); first < second; first++) {
              const at = performance.now();
              const found = await api.matchResidentFeaturesGpu(
                device,
                handles[first],
                handles[second],
                { ratio: 0.82, maxDistance: 80 },
              );
              matchingMs += performance.now() - at;
              if (found.matches.length >= 16)
                pairs.push({
                  first: frames[first].id,
                  second: frames[second].id,
                  matches: found.matches,
                  distances: found.distances,
                });
            }
          const at = performance.now();
          try {
            const mapped = await api.reconstructIncrementalGpu(device, images, pairs, {
              ransacIterations: 512,
              minInitialAngleDeg: 1,
              minTriangulationAngleDeg: 0.5,
              maxErrorPx: 3,
              bundleIterations: 2,
              finalBundleIterations: 4,
              refineSharedFocal: false,
            });
            model = {
              views: mapped.reconstruction.views.map((view) => ({
                ...view,
                pose: { R: [...view.pose.R], t: [...view.pose.t] },
              })),
              landmarks: mapped.reconstruction.landmarks.map((point) => ({
                ...point,
                xyz: [...point.xyz],
              })),
            };
            stages = mapped.execution.stages.map((stage) => ({
              operation: stage.operation,
              elapsedMs: stage.elapsedMs,
              uploadBytes: stage.execution.applicationUploadBytes,
              readbackBytes: stage.execution.applicationReadbackBytes,
              dispatches: stage.execution.report.dispatches,
              submissions: stage.execution.report.submissions,
            }));
          } catch (error) {
            rejection = {
              name: error.name,
              message: error.message,
              pairs: error.pairVerification ?? null,
            };
          }
          mappingMs = performance.now() - at;
        } finally {
          for (const handle of handles) await handle.dispose();
        }
        const after = device.statistics;
        results.push({
          decodeMs,
          featureMs,
          matchingMs,
          mappingMs,
          elapsedMs: performance.now() - start,
          featureCounts: counts,
          matchedPairs: pairs.length,
          model,
          rejection,
          stages,
          uploadBytes: after.applicationUploadBytes - before.applicationUploadBytes,
          readbackBytes: after.applicationReadbackBytes - before.applicationReadbackBytes,
          residentSnapshotBytes: after.residentSnapshotBytes - before.residentSnapshotBytes,
          allocations: after.allocations - before.allocations,
          liveBytesAfterFeatureRelease: after.liveBytes,
        });
      }
    } finally {
      await device.dispose();
      gpu.destroy();
    }
    return {
      description,
      software,
      deviceOpenMs,
      gpuErrors,
      results,
      peakManagedLiveBytes,
      liveBuffersAfterDisposal: device.statistics.liveBuffers,
    };
  },
  { cases },
  { timeoutMs: 300000 },
);
assert.deepEqual(errors, []);
assert.deepEqual(result.gpuErrors, []);
assert.equal(result.liveBuffersAfterDisposal, 0);
for (const [i, scenario] of ['translation', 'rotation', 'rotation-zoom'].entries()) {
  const row = result.results[i];
  row.status = row.model
    ? 'map'
    : row.rejection?.name === 'ReconstructionInitializationError'
      ? 'initialization-rejected'
      : 'execution-error';
  row.scenario = scenario;
  row.inputs = cases[i].frames.map(({ id, sha256, camera, width, height }) => ({
    id,
    sha256,
    camera,
    width,
    height,
  }));
  const fixture = generalVideoFixture(scenario);
  row.registrationRatio = (row.model?.views.length ?? 0) / cases[i].frames.length;
  row.unexpectedGeometryCommit =
    scenario !== 'translation' && (row.model?.landmarks.length ?? 0) > 0;
  row.cameraAlignment = row.model
    ? compareCameraModels({ views: fixture.truth.views }, row.model).alignment
    : null;
  row.landmarks = row.model?.landmarks.length ?? 0;
  row.cameraPoses = row.model?.views;
  delete row.model;
}
const report = {
  version: 'typecv-general-video-rgb-probe-v1',
  createdAt: new Date().toISOString(),
  head: execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).trim(),
  inferenceInput:
    'PNG RGB frames and explicitly known synthetic intrinsics only; no GT pose/depth/matches/masks/shot labels',
  inputScope:
    'Analytic static room; no video codec, gameplay, dynamic objects, cuts or inferred intrinsics',
  controls: {
    features: 'pyramid FAST/BRIEF, smooth',
    maxFeatures: 1000,
    temporalWindow: 3,
    ransacIterations: 512,
    bundleIterations: 2,
    finalBundleIterations: 4,
    refineSharedFocal: false,
  },
  calibration: 'known-synthetic',
  correspondences: 'estimated-from-RGB',
  mask: 'none',
  nativeGpuMeasured: false,
  backend: result.software ? 'software-webgpu' : 'webgpu-hardware-unverified',
  timingScope:
    'PNG decode + features/upload + matching + geometry/readbacks + feature release. Includes compilation and JS overhead; not device kernel time. Cases share one device in listed order.',
  memoryScope:
    'Peak live bytes managed by ShadeDevice allocations; excludes pool, decoded host pixels, browser/driver allocations.',
  measurements: result,
  limitations: [
    'These are diagnostic baselines, not all-scenario acceptance.',
    'No per-frame alignment.',
    'Geometry rejection is not orientation tracking or resumable deferred-depth state.',
  ],
};
await mkdir(dirname(output), { recursive: true });
await writeFile(output, JSON.stringify(report, null, 2) + '\n');
console.log(
  JSON.stringify(
    {
      output,
      backend: report.backend,
      adapter: result.description,
      results: result.results.map((row) => ({
        scenario: row.scenario,
        registrationRatio: row.registrationRatio,
        landmarks: row.landmarks,
        rejection: row.rejection?.message,
        unexpectedGeometryCommit: row.unexpectedGeometryCommit,
        elapsedMs: row.elapsedMs,
        uploadBytes: row.uploadBytes,
        readbackBytes: row.readbackBytes,
      })),
    },
    null,
    2,
  ),
);
