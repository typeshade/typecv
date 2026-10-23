import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { inBrowser } from '../tests/typeshade/browser.mjs';
import { reconstructIncremental } from '../dist/index.js';

const args = process.argv.slice(2);
const arg = (name, fallback) => {
  const prefix = `--${name}=`;
  const value = args.find((item) => item.startsWith(prefix));
  return value ? value.slice(prefix.length) : fallback;
};
const warmRuns = Number(arg('warm-runs', '2'));
const refineFocalArg = arg('refine-focal', '1');
if (!['0', '1'].includes(refineFocalArg)) throw new RangeError('--refine-focal must be 0 or 1');
const refineSharedFocal = refineFocalArg === '1';
const featureMode = arg('feature-mode', 'single');
const maxFeatures = Number(arg('max-features', '400'));
const smoothDescriptorsArg = arg('smooth-descriptors', '0');
if (!['0', '1'].includes(smoothDescriptorsArg))
  throw new RangeError('--smooth-descriptors must be 0 or 1');
const smoothDescriptors = smoothDescriptorsArg === '1';
const timestampsArg = arg('timestamps', '0');
if (!['0', '1'].includes(timestampsArg)) throw new RangeError('--timestamps must be 0 or 1');
const timestamps = timestampsArg === '1';
const maxModelsPerBatch = Number(arg('max-models-per-batch', '32'));
if (!Number.isSafeInteger(maxModelsPerBatch) || maxModelsPerBatch < 1 || maxModelsPerBatch > 4096)
  throw new RangeError('--max-models-per-batch must be an integer from 1 through 4096');
const occupancyArg = arg('occupancy-floors', 'auto');
if (!['auto', '0', '1'].includes(occupancyArg))
  throw new RangeError('--occupancy-floors must be auto, 0 or 1');
const occupancyFloors = occupancyArg === 'auto' ? undefined : occupancyArg === '1';
const matchRatio = Number(arg('match-ratio', '0.82'));
const maxDistance = Number(arg('max-distance', '80'));
const windowSize = Number(arg('window-size', '3'));
const output = resolve(arg('output', 'work/benchmark-results/sfm-pyramid-e2e.json'));
const runtimeArg = arg('runtime', '/dist/gpu.js');
const runtimeUrl = runtimeArg.startsWith('/') ? runtimeArg : `/dist/${runtimeArg}`;
if (!runtimeUrl.startsWith('/dist/') || runtimeUrl.includes('..'))
  throw new RangeError('--runtime must identify a browser module under /dist');
const runtimeFile = resolve(import.meta.dirname, `..${runtimeUrl}`);
const runtimeSha256 = createHash('sha256')
  .update(await readFile(runtimeFile))
  .digest('hex');
if (!Number.isSafeInteger(warmRuns) || warmRuns < 1 || warmRuns > 10)
  throw new RangeError('--warm-runs must be an integer from 1 through 10');
if (!['single', 'pyramid'].includes(featureMode))
  throw new RangeError('--feature-mode must be single or pyramid');
if (!Number.isSafeInteger(maxFeatures) || maxFeatures < 1 || maxFeatures > 2000)
  throw new RangeError('--max-features must be an integer from 1 through 2000');
if (!Number.isFinite(matchRatio) || matchRatio <= 0.1 || matchRatio >= 1)
  throw new RangeError('--match-ratio must be greater than 0.1 and less than 1');
if (!Number.isSafeInteger(maxDistance) || maxDistance < 1 || maxDistance > 256)
  throw new RangeError('--max-distance must be an integer from 1 through 256');
if (!Number.isSafeInteger(windowSize) || windowSize < 1 || windowSize > 7)
  throw new RangeError('--window-size must be an integer from 1 through 7');

const framePaths = Array.from(
  { length: 8 },
  (_, index) =>
    new URL(
      `../tests/fixtures/lab-images/pyramid/frame-${String(index).padStart(2, '0')}.jpg`,
      import.meta.url,
    ),
);
const frameBuffers = await Promise.all(framePaths.map((path) => readFile(path)));
const fixtureSha256 = createHash('sha256')
  .update(frameBuffers.map((buffer) => buffer.toString('base64')).join('\n'))
  .digest('hex');
const browserFrames = frameBuffers.map((buffer) => buffer.toString('base64'));

const { result, errors } = await inBrowser(
  async ({
    encodedFrames,
    warmRuns,
    hardware,
    runtimeUrl,
    featureMode,
    maxFeatures,
    smoothDescriptors,
    timestamps,
    maxModelsPerBatch,
    occupancyFloors,
    matchRatio,
    maxDistance,
    windowSize,
    refineSharedFocal,
  }) => {
    const api = await import(runtimeUrl);
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) throw new Error('WebGPU adapter unavailable');
    const adapterInfo = adapter.info;
    const adapterDescription = [
      adapterInfo?.vendor,
      adapterInfo?.architecture,
      adapterInfo?.device,
      adapterInfo?.description,
    ]
      .filter(Boolean)
      .join(' ');
    const software =
      /swiftshader|software|llvmpipe/i.test(adapterDescription) ||
      adapter.isFallbackAdapter === true;
    if (hardware && software)
      throw new Error('Hardware requested, but a software adapter was selected');
    if (timestamps && !adapter.features.has('timestamp-query'))
      throw new Error('Timestamp queries requested, but the adapter does not support them');

    const decodeStarted = performance.now();
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d', { willReadFrequently: true });
    if (!context) throw new Error('2D canvas context unavailable');
    const frames = [];
    for (let index = 0; index < encodedFrames.length; index++) {
      const bytes = Uint8Array.from(atob(encodedFrames[index]), (character) =>
        character.charCodeAt(0),
      );
      const bitmap = await createImageBitmap(new Blob([bytes], { type: 'image/jpeg' }));
      try {
        canvas.width = bitmap.width;
        canvas.height = bitmap.height;
        context.drawImage(bitmap, 0, 0);
        const pixels = context.getImageData(0, 0, bitmap.width, bitmap.height).data.slice();
        frames.push({
          id: index,
          width: bitmap.width,
          height: bitmap.height,
          pixels,
          time: [0, 1, 2, 4, 8, 12, 16, 20][index],
        });
        // Match the website's thumbnail conversion as part of sample preparation.
        canvas.toDataURL('image/jpeg', 0.78);
      } finally {
        bitmap.close();
      }
    }
    const preparationMs = performance.now() - decodeStarted;

    const openStarted = performance.now();
    const gpu = await adapter.requestDevice({
      requiredFeatures: timestamps ? ['timestamp-query'] : [],
    });
    const validationErrors = [];
    gpu.addEventListener('uncapturederror', (event) => validationErrors.push(event.error.message));
    const device = await api.ShadeDevice.open({
      gpuDevice: gpu,
      backend: 'webgpu',
      boundedSubmissions: false,
      solverCheckpoints: true,
      maxModelsPerBatch,
      ...(occupancyFloors === undefined ? {} : { occupancyFloors }),
    });
    const deviceOpenMs = performance.now() - openStarted;
    const mapOptions = {
      ransacIterations: 500,
      maxErrorPx: 3,
      minInitialAngleDeg: 1,
      minTriangulationAngleDeg: 0.5,
      bundleIterations: 4,
      finalBundleIterations: 8,
      refineSharedFocal,
    };
    const runs = [];
    let comparisonInput;
    let reconstructionModel;
    const gpuTimeNs = (...reports) => {
      if (!timestamps) return null;
      const groups = reports.flatMap((report) => report.groups);
      if (!groups.length || groups.some((group) => !Number.isFinite(group.gpuTime))) return null;
      return groups.reduce((sum, group) => sum + group.gpuTime, 0);
    };
    try {
      for (let runIndex = 0; runIndex <= warmRuns; runIndex++) {
        const started = performance.now();
        const copyStarted = performance.now();
        const jobFrames = frames.map((frame) => ({ ...frame, pixels: frame.pixels.slice() }));
        const payloadCopyMs = performance.now() - copyStarted;
        const before = device.statistics;
        const images = [];
        const pairs = [];
        const live = new Map();
        const featureStages = [];
        const matchStages = [];
        const featureCounts = [];
        let matchedCandidates = 0;
        try {
          for (let index = 0; index < jobFrames.length; index++) {
            const frame = jobFrames[index];
            const phaseStarted = performance.now();
            const beforeFeature = device.statistics;
            const extractFeatures =
              featureMode === 'pyramid'
                ? api.extractFeaturesGpuPyramidResident
                : api.extractFeaturesGpuResident;
            const features = await extractFeatures(
              device,
              frame.pixels,
              frame.width,
              frame.height,
              { threshold: 0.045, maxFeatures, smoothDescriptors },
            );
            live.set(index, features);
            if (features.count < 8)
              throw new Error(`Frame ${index + 1} produced only ${features.count} features`);
            images.push({
              id: frame.id,
              camera: {
                fx: frame.width * 0.8,
                // The website derives one square-pixel focal length from source width (0.8 × width by default).
                fy: frame.width * 0.8,
                cx: (frame.width - 1) / 2,
                cy: (frame.height - 1) / 2,
                k1: 0,
                k2: 0,
              },
              keypoints: await features.readKeypoints(),
            });
            featureCounts.push(features.count);
            featureStages.push({
              frame: frame.id,
              elapsedMs: performance.now() - phaseStarted,
              count: features.count,
              dispatches: features.report.dispatches,
              submissions: features.report.submissions,
              applicationUploadBytes:
                device.statistics.applicationUploadBytes - beforeFeature.applicationUploadBytes,
              applicationReadbackBytes:
                device.statistics.applicationReadbackBytes - beforeFeature.applicationReadbackBytes,
              gpuTimeNs: gpuTimeNs(features.report),
            });
            for (let previous = Math.max(0, index - windowSize); previous < index; previous++) {
              const matchStarted = performance.now();
              const beforeMatch = device.statistics;
              const match = await api.matchResidentFeaturesGpu(
                device,
                live.get(previous),
                features,
                { ratio: matchRatio, maxDistance },
              );
              matchedCandidates++;
              const accepted = match.matches.length >= 8;
              if (accepted)
                pairs.push({
                  first: jobFrames[previous].id,
                  second: frame.id,
                  matches: match.matches,
                  distances: match.distances,
                });
              matchStages.push({
                first: jobFrames[previous].id,
                second: frame.id,
                elapsedMs: performance.now() - matchStarted,
                matches: match.matches.length / 2,
                acceptedForMapping: accepted,
                dispatches: match.report.dispatches,
                submissions: match.report.submissions,
                applicationUploadBytes:
                  device.statistics.applicationUploadBytes - beforeMatch.applicationUploadBytes,
                applicationReadbackBytes:
                  device.statistics.applicationReadbackBytes - beforeMatch.applicationReadbackBytes,
                gpuTimeNs: gpuTimeNs(match.report),
              });
            }
            const expired = live.get(index - windowSize);
            if (expired) {
              await expired.dispose();
              live.delete(index - windowSize);
            }
          }
          for (const features of live.values()) await features.dispose();
          live.clear();
          const frameOrder = new Map(jobFrames.map((frame, index) => [frame.id, index]));
          pairs.sort(
            (a, b) =>
              frameOrder.get(a.second) -
                frameOrder.get(a.first) -
                (frameOrder.get(b.second) - frameOrder.get(b.first)) ||
              frameOrder.get(a.first) - frameOrder.get(b.first),
          );
          if (!pairs.length) throw new Error('Pyramid sample produced no usable feature matches');

          const mapperStarted = performance.now();
          const mapping = await api.reconstructIncrementalGpu(device, images, pairs, mapOptions);
          const mapperMs = performance.now() - mapperStarted;
          const reconstruction = mapping.reconstruction;
          reconstructionModel = {
            views: reconstruction.views.map((view) => ({
              ...view,
              pose: { R: Array.from(view.pose.R), t: Array.from(view.pose.t) },
            })),
            landmarks: reconstruction.landmarks.map((point) => ({
              ...point,
              xyz: Array.from(point.xyz),
            })),
          };
          const output = {
            registeredViews: reconstruction.views.length,
            viewIds: reconstruction.views.map((view) => view.id),
            registrationOrder: mapping.diagnostics.registrationOrder,
            landmarkCount: reconstruction.landmarks.length,
            observationCount: reconstruction.landmarks.reduce(
              (count, point) => count + point.observations.length,
              0,
            ),
            unregistered: mapping.unregistered,
            initialPair: mapping.diagnostics.initialPair,
            meanReprojectionErrorPx: mapping.diagnostics.meanReprojectionErrorPx,
            featureCounts,
            candidatePairs: matchedCandidates,
            acceptedPairs: pairs.length,
            mapperStages: mapping.execution.stages.map(({ operation, elapsedMs, execution }) => ({
              operation,
              elapsedMs,
              dispatches: execution.report.dispatches,
              submissions: execution.report.submissions,
              gpuTimeNs: gpuTimeNs(execution.report),
              hypothesesScored: execution.hypothesesScored,
            })),
          };
          if (!comparisonInput)
            comparisonInput = {
              images: images.map((image) => ({ ...image, keypoints: Array.from(image.keypoints) })),
              pairs: pairs.map((pair) => ({
                ...pair,
                matches: Array.from(pair.matches),
                distances: Array.from(pair.distances),
              })),
            };
          runs.push({
            kind: runIndex === 0 ? 'cold-first-call' : 'warm-steady-state',
            elapsedMs: performance.now() - started,
            payloadCopyMs,
            featureMs: featureStages.reduce((sum, stage) => sum + stage.elapsedMs, 0),
            matchMs: matchStages.reduce((sum, stage) => sum + stage.elapsedMs, 0),
            mapperMs,
            featureStages,
            matchStages,
            statisticsDelta: Object.fromEntries(
              Object.keys(device.statistics)
                .filter((key) => typeof device.statistics[key] === 'number')
                .map((key) => [key, device.statistics[key] - before[key]]),
            ),
            output,
          });
        } finally {
          for (const features of live.values()) await features.dispose();
        }
      }
    } finally {
      await device.dispose();
      gpu.destroy();
    }
    return {
      adapter: { description: adapterDescription, software },
      backend: 'webgpu',
      preparationMs,
      deviceOpenMs,
      preparation: frames.map(({ id, width, height, time }) => ({ id, width, height, time })),
      runs,
      comparisonInput,
      reconstruction: reconstructionModel,
      validationErrors,
    };
  },
  {
    encodedFrames: browserFrames,
    warmRuns,
    hardware: process.env.TYPECV_HARDWARE_GPU === '1',
    runtimeUrl,
    featureMode,
    maxFeatures,
    smoothDescriptors,
    timestamps,
    maxModelsPerBatch,
    occupancyFloors,
    matchRatio,
    maxDistance,
    windowSize,
    refineSharedFocal,
  },
  { timeoutMs: 300000 },
);
assert.deepEqual(errors, []);
assert.deepEqual(result.validationErrors, []);
assert.equal(result.backend, 'webgpu');
assert.equal(result.preparation.length, 8);
assert.equal(result.runs.length, warmRuns + 1);

const cpuStarted = performance.now();
const cpu = reconstructIncremental(
  result.comparisonInput.images.map((image) => ({
    ...image,
    keypoints: Float64Array.from(image.keypoints),
  })),
  result.comparisonInput.pairs.map((pair) => ({
    ...pair,
    matches: Uint32Array.from(pair.matches),
    distances: Float64Array.from(pair.distances),
  })),
  {
    ransacIterations: 500,
    maxErrorPx: 3,
    minInitialAngleDeg: 1,
    minTriangulationAngleDeg: 0.5,
    bundleIterations: 4,
    finalBundleIterations: 8,
    refineSharedFocal,
  },
);
const cpuElapsedMs = performance.now() - cpuStarted;
const summarizeCpu = {
  elapsedMs: cpuElapsedMs,
  registeredViews: cpu.reconstruction.views.length,
  viewIds: cpu.reconstruction.views.map((view) => view.id),
  registrationOrder: cpu.diagnostics.registrationOrder,
  landmarkCount: cpu.reconstruction.landmarks.length,
  observationCount: cpu.reconstruction.landmarks.reduce(
    (count, point) => count + point.observations.length,
    0,
  ),
  unregistered: cpu.unregistered,
  initialPair: cpu.diagnostics.initialPair,
  meanReprojectionErrorPx: cpu.diagnostics.meanReprojectionErrorPx,
};
const gpuSteady = result.runs.at(-1).output;
const comparison = {
  sameRegisteredViewIds:
    JSON.stringify(gpuSteady.viewIds.toSorted((a, b) => a - b)) ===
    JSON.stringify(summarizeCpu.viewIds.toSorted((a, b) => a - b)),
  sameRegistrationOrder:
    JSON.stringify(gpuSteady.registrationOrder) === JSON.stringify(summarizeCpu.registrationOrder),
  sameInitialPair:
    JSON.stringify(gpuSteady.initialPair) === JSON.stringify(summarizeCpu.initialPair),
  registeredViewDelta: gpuSteady.registeredViews - summarizeCpu.registeredViews,
  landmarkDelta: gpuSteady.landmarkCount - summarizeCpu.landmarkCount,
  observationDelta: gpuSteady.observationCount - summarizeCpu.observationCount,
  gpuMeanReprojectionErrorPx: gpuSteady.meanReprojectionErrorPx,
  cpuMeanReprojectionErrorPx: summarizeCpu.meanReprojectionErrorPx,
};
const report = {
  schemaVersion: 1,
  benchmark: 'website-pyramid-eight-frame-end-to-end',
  featureMode,
  maxFeatures,
  smoothDescriptors,
  timestamps,
  maxModelsPerBatch,
  refineSharedFocal,
  matching: { ratio: matchRatio, maxDistance, windowSize },
  runtime: { url: runtimeUrl, entrySha256: runtimeSha256 },
  fixture: {
    frameCount: frameBuffers.length,
    sha256: fixtureSha256,
    files: framePaths.map((path) => path.pathname.split('/').at(-1)),
  },
  mode: process.env.TYPECV_HARDWARE_GPU === '1' ? 'hardware-requested' : 'software-webgpu',
  source: result,
  cpuReference: summarizeCpu,
  comparison,
  overallColdSessionMs: result.preparationMs + result.deviceOpenMs + result.runs[0].elapsedMs,
};
await mkdir(dirname(output), { recursive: true });
await writeFile(output, `${JSON.stringify(report, null, 2)}\n`);
console.log(
  JSON.stringify({
    output,
    fixture: report.fixture,
    adapter: result.adapter,
    preparationMs: result.preparationMs,
    deviceOpenMs: result.deviceOpenMs,
    runs: result.runs.map(({ kind, elapsedMs, featureMs, matchMs, mapperMs, output: summary }) => ({
      kind,
      elapsedMs,
      featureMs,
      matchMs,
      mapperMs,
      views: summary.registeredViews,
      landmarks: summary.landmarkCount,
      reprojectionPx: summary.meanReprojectionErrorPx,
    })),
    cpuReference: summarizeCpu,
    comparison,
  }),
);
