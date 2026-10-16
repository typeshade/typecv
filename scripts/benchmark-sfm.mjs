import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { inBrowser } from '../tests/typeshade/browser.mjs';
import { reconstructIncremental } from '../dist/index.js';
import { mappingFixture } from '../tests/typeshade/geometry-cases.mjs';

const args = process.argv.slice(2);
const arg = (name, fallback) => {
  const prefix = `--${name}=`;
  const found = args.find((value) => value.startsWith(prefix));
  return found ? found.slice(prefix.length) : fallback;
};
const parseSizes = (value) =>
  value.split(',').map((item) => {
    const match = /^(\d+)x(\d+)$/i.exec(item.trim());
    if (!match) throw new Error(`Invalid analysis size ${item}; expected WIDTHxHEIGHT`);
    const width = Number(match[1]),
      height = Number(match[2]);
    if (width < 48 || height < 48 || width > 4096 || height > 4096)
      throw new RangeError(`Analysis size ${item} is outside the GPU API's 48..4096 bounds`);
    return { width, height, label: `${width}x${height}` };
  });
const sizes = parseSizes(arg('sizes', '640x360,1280x720,1920x1080'));
const configurationPresets = [
  {
    name: 'checkpoint-32',
    boundedSubmissions: false,
    solverCheckpoints: true,
    maxModelsPerBatch: 32,
  },
  { name: 'unbounded-default', boundedSubmissions: false, maxModelsPerBatch: 64 },
  { name: 'bounded-8', boundedSubmissions: true, maxModelsPerBatch: 8 },
  {
    name: 'bounded-8-dispatch-16',
    boundedSubmissions: true,
    maxModelsPerBatch: 8,
    maxDispatchesPerSubmission: 16,
  },
  {
    name: 'bounded-32-dispatch-64',
    boundedSubmissions: true,
    maxModelsPerBatch: 32,
    maxDispatchesPerSubmission: 64,
  },
  {
    name: 'bounded-64-dispatch-64',
    boundedSubmissions: true,
    maxModelsPerBatch: 64,
    maxDispatchesPerSubmission: 64,
  },
];
const requestedConfigs = arg(
  'configs',
  configurationPresets
    .filter((item) => ['unbounded-default', 'bounded-8', 'checkpoint-32'].includes(item.name))
    .map((item) => item.name)
    .join(','),
)
  .split(',')
  .map((item) => item.trim());
if (!requestedConfigs.length || requestedConfigs.some((name) => !name))
  throw new RangeError('At least one --configs entry is required');
const configurations = requestedConfigs.map((name) => {
  const preset = configurationPresets.find((item) => item.name === name);
  if (!preset) throw new RangeError(`Unknown SfM configuration ${name}`);
  return preset;
});
const runFeatures = arg('features', '1') !== '0';
const sourceWidth = Number(arg('source-width', '18000')),
  sourceHeight = Number(arg('source-height', '9000')),
  warmRuns = Number(arg('warm-runs', '3')),
  outputPath = arg('output', 'work/benchmark-results/sfm-benchmark.json');
if (
  !Number.isSafeInteger(sourceWidth) ||
  sourceWidth < 48 ||
  !Number.isSafeInteger(sourceHeight) ||
  sourceHeight < 48 ||
  !Number.isSafeInteger(warmRuns) ||
  warmRuns < 1 ||
  warmRuns > 20
)
  throw new RangeError('Invalid source dimensions or warm run count');

const mapping = mappingFixture(),
  cpuStart = performance.now(),
  cpu = reconstructIncremental(mapping.images, mapping.pairs, {
    seed: 42,
    ransacIterations: 128,
    bundleIterations: 2,
    finalBundleIterations: 4,
  }),
  cpuElapsedMs = performance.now() - cpuStart;
const cpuSummary = {
  elapsedMs: cpuElapsedMs,
  registeredViews: cpu.reconstruction.views.length,
  landmarks: cpu.reconstruction.landmarks.length,
  unregistered: cpu.unregistered,
  meanReprojectionErrorPx: cpu.diagnostics.meanReprojectionErrorPx,
  initialPair: cpu.diagnostics.initialPair,
  registrationOrder: cpu.diagnostics.registrationOrder,
};

const { result, errors } = await inBrowser(
  async ({
    sizes,
    sourceWidth,
    sourceHeight,
    warmRuns,
    mapping,
    hardware,
    configurations,
    runFeatures,
  }) => {
    const {
      ShadeDevice,
      extractFeaturesGpu,
      matchFeaturesGpu,
      extractFeaturesGpuResident,
      matchResidentFeaturesGpu,
      reconstructIncrementalGpu,
    } = await import('/dist/gpu.js');
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('WebGPU adapter is unavailable');
    const info = adapter.info,
      description = [info?.vendor, info?.architecture, info?.device, info?.description]
        .filter(Boolean)
        .join(' '),
      software =
        /swiftshader|software|llvmpipe/i.test(description) || adapter.isFallbackAdapter === true;
    if (hardware && software)
      throw new Error('Hardware run requested but a software adapter was selected');
    const validationErrors = [];
    const summary = (run) => ({
      registeredViews: run.reconstruction.views.length,
      landmarks: run.reconstruction.landmarks.length,
      observationCount: run.reconstruction.landmarks.reduce(
        (count, point) => count + point.observations.length,
        0,
      ),
      unregistered: run.unregistered,
      meanReprojectionErrorPx: run.diagnostics.meanReprojectionErrorPx,
      initialPair: run.diagnostics.initialPair,
      registrationOrder: run.diagnostics.registrationOrder,
      gpuBundleAdjustments: run.execution.gpuBundleAdjustments,
      cpuStages: run.execution.cpuStages,
      pairConsensus: run.execution.pairConsensus.map((consensus) => ({
        hypothesisBatches: consensus.hypothesisBatches,
        hypothesesScored: consensus.hypothesesScored,
        scoreDispatches: consensus.scoreDispatches,
        scoreSubmissions: consensus.scoreSubmissions,
        applicationReadbackBytes: consensus.applicationReadbackBytes,
      })),
      stages: run.execution.stages.map(({ operation, elapsedMs, execution }) => ({
        operation,
        elapsedMs,
        tier: execution.report.tier,
        dispatches: execution.report.dispatches,
        submissions: execution.report.submissions,
        hypothesisBatches: execution.hypothesisBatches,
        hypothesesScored: execution.hypothesesScored,
        scoreDispatches: execution.scoreDispatches,
        scoreSubmissions: execution.scoreSubmissions,
        homographyIterations: execution.homographyIterations,
        uploadBytes: execution.applicationUploadBytes,
        readbackBytes: execution.applicationReadbackBytes,
      })),
    });
    const delta = (before, after) =>
      Object.fromEntries(
        Object.keys(after)
          .filter((key) => typeof after[key] === 'number' && typeof before[key] === 'number')
          .map((key) => [key, after[key] - before[key]]),
      );
    const installPipelineProfiler = (gpu) => {
      const stats = {
          shaderModules: 0,
          wgslChars: 0,
          maxWgslChars: 0,
          computePipelines: 0,
          computePipelineCallMs: 0,
          asyncComputePipelines: 0,
          asyncComputePipelineWaitMs: 0,
          entries: {},
        },
        originals = new Map();
      const patch = (name, wrap) => {
        const original = gpu[name];
        if (typeof original !== 'function') return false;
        const descriptor = Object.getOwnPropertyDescriptor(gpu, name);
        originals.set(name, descriptor);
        try {
          Object.defineProperty(gpu, name, {
            configurable: true,
            writable: true,
            value: wrap(original),
          });
          return true;
        } catch {
          originals.delete(name);
          return false;
        }
      };
      const supported = [
        patch(
          'createShaderModule',
          (original) =>
            function (descriptor) {
              const code = descriptor?.code;
              stats.shaderModules++;
              if (typeof code === 'string') {
                stats.wgslChars += code.length;
                stats.maxWgslChars = Math.max(stats.maxWgslChars, code.length);
              }
              return Reflect.apply(original, gpu, [descriptor]);
            },
        ),
        patch(
          'createComputePipeline',
          (original) =>
            function (descriptor) {
              const entry = descriptor?.compute?.entryPoint ?? '(unknown)',
                started = performance.now();
              stats.computePipelines++;
              stats.entries[entry] = (stats.entries[entry] ?? 0) + 1;
              try {
                return Reflect.apply(original, gpu, [descriptor]);
              } finally {
                stats.computePipelineCallMs += performance.now() - started;
              }
            },
        ),
        patch(
          'createComputePipelineAsync',
          (original) =>
            function (descriptor) {
              const entry = descriptor?.compute?.entryPoint ?? '(unknown)',
                started = performance.now();
              stats.asyncComputePipelines++;
              stats.entries[entry] = (stats.entries[entry] ?? 0) + 1;
              return Reflect.apply(original, gpu, [descriptor]).finally(() => {
                stats.asyncComputePipelineWaitMs += performance.now() - started;
              });
            },
        ),
      ].every(Boolean);
      const snapshot = () => structuredClone(stats);
      const since = (before) => {
        const after = snapshot(),
          entries = Object.fromEntries(
            Object.entries(after.entries)
              .map(([key, value]) => [key, value - (before.entries[key] ?? 0)])
              .filter(([, value]) => value !== 0),
          );
        return {
          shaderModules: after.shaderModules - before.shaderModules,
          wgslChars: after.wgslChars - before.wgslChars,
          maxWgslChars: after.maxWgslChars,
          computePipelines: after.computePipelines - before.computePipelines,
          computePipelineCallMs: after.computePipelineCallMs - before.computePipelineCallMs,
          asyncComputePipelines: after.asyncComputePipelines - before.asyncComputePipelines,
          asyncComputePipelineWaitMs:
            after.asyncComputePipelineWaitMs - before.asyncComputePipelineWaitMs,
          entries,
        };
      };
      const restore = () => {
        for (const [name, descriptor] of originals)
          if (descriptor) Object.defineProperty(gpu, name, descriptor);
          else delete gpu[name];
      };
      return { supported, snapshot, since, restore };
    };
    const imageArgument = {
      images: mapping.images.map((view) => ({
        ...view,
        keypoints: Float64Array.from(view.keypoints),
      })),
      pairs: mapping.pairs.map((pair) => ({
        ...pair,
        matches: Uint32Array.from(pair.matches),
      })),
    };
    const sfmConfigurations = [];
    for (const config of configurations) {
      const adapterStarted = performance.now(),
        configAdapter = await navigator.gpu.requestAdapter(),
        adapterRequestMs = performance.now() - adapterStarted;
      if (!configAdapter) throw new Error(`WebGPU adapter unavailable for ${config.name}`);
      const configDescription = [
        configAdapter.info?.vendor,
        configAdapter.info?.architecture,
        configAdapter.info?.device,
        configAdapter.info?.description,
      ].join(' ');
      if (
        hardware &&
        (/swiftshader|software|llvmpipe/i.test(configDescription) ||
          configAdapter.isFallbackAdapter === true)
      )
        throw new Error(`${config.name} selected a software adapter`);
      const gpuStarted = performance.now(),
        gpu = await configAdapter.requestDevice(),
        gpuRequestMs = performance.now() - gpuStarted;
      gpu.addEventListener('uncapturederror', (event) =>
        validationErrors.push(event.error.message),
      );
      const pipelineProfiler = installPipelineProfiler(gpu),
        shadeRuntimeStarted = performance.now(),
        device = await ShadeDevice.open({ gpuDevice: gpu, ...config }),
        shadeRuntimeOpenMs = performance.now() - shadeRuntimeStarted;
      try {
        const runs = [];
        for (let i = 0; i <= warmRuns; i++) {
          const before = device.statistics,
            pipelineBefore = pipelineProfiler.snapshot(),
            started = performance.now(),
            run = await reconstructIncrementalGpu(
              device,
              imageArgument.images,
              imageArgument.pairs,
              {
                seed: 42,
                ransacIterations: 128,
                bundleIterations: 2,
                finalBundleIterations: 4,
              },
            );
          runs.push({
            kind: i === 0 ? 'cold-first-call' : 'warm-steady-state',
            elapsedMs: performance.now() - started,
            pipelineApiDelta: pipelineProfiler.since(pipelineBefore),
            statisticsDelta: delta(before, device.statistics),
            output: summary(run),
          });
        }
        sfmConfigurations.push({
          configuration: config,
          startup: {
            adapterRequestMs,
            gpuRequestMs,
            shadeRuntimeOpenMs,
            pipelineApiInstrumentationSupported: pipelineProfiler.supported,
          },
          deviceBudget: {
            maxKernelElements: device.maxKernelElements,
            workingBytes: device.workingBytes,
            bindingBytes: device.bindingBytes,
          },
          runs,
        });
      } finally {
        pipelineProfiler.restore();
        await device.dispose();
        gpu.destroy();
      }
    }

    const featureRuns = [];
    const residentApiAvailable =
      typeof extractFeaturesGpuResident === 'function' &&
      typeof matchResidentFeaturesGpu === 'function';
    for (const size of runFeatures ? sizes : []) {
      // Generate at analysis resolution only; the configured 18K source is never allocated.
      const pixelCount = size.width * size.height,
        tiles = Math.ceil(size.width / 24) * Math.ceil(size.height / 24),
        options = { maxFeatures: Math.min(2000, tiles), threshold: 0.045 },
        shifts = [0, 1, 2, 3].map((step) => ({ x: step * 8, y: step * 8 })),
        makeFrame = ({ x: dx, y: dy }) => {
          const rgba = new Uint8Array(pixelCount * 4),
            tile = 16,
            hash = (x, y) => {
              let value = Math.imul(x + 1, 0x45d9f3b) ^ Math.imul(y + 1, 0x119de1f3);
              value = Math.imul(value ^ (value >>> 16), 0x45d9f3b);
              value = Math.imul(value ^ (value >>> 16), 0x45d9f3b);
              return (value ^ (value >>> 16)) & 255;
            };
          for (let y = 0; y < size.height; y++)
            for (let x = 0; x < size.width; x++) {
              const sourceX = x - dx,
                sourceY = y - dy,
                at = (y * size.width + x) * 4,
                value =
                  sourceX < 0 || sourceY < 0
                    ? 0
                    : hash(Math.floor(sourceX / tile), Math.floor(sourceY / tile));
              rgba[at] = value;
              rgba[at + 1] = value;
              rgba[at + 2] = value;
              rgba[at + 3] = 255;
            }
          return rgba;
        };
      if (pixelCount > 4_194_240 || pixelCount * 20 > 64 * 1024 ** 2) {
        featureRuns.push({
          ...size,
          inputGenerationMs: null,
          skipped: 'frame exceeds TypeShade default feature working/dispatch budget',
          residentApiAvailable,
        });
        continue;
      }
      const generationStart = performance.now(),
        frames = shifts.map(makeFrame),
        inputGenerationMs = performance.now() - generationStart;

      const matchQuality = (matches, keypoints, targetShift) => {
        const errors = [];
        for (let i = 0; i < matches.length; i += 2) {
          const a = matches[i] * 2,
            b = matches[i + 1] * 2,
            dx = keypoints[1][b] - keypoints[0][a],
            dy = keypoints[1][b + 1] - keypoints[0][a + 1];
          errors.push(Math.hypot(dx - targetShift.x, dy - targetShift.y));
        }
        const accurate = errors.filter((error) => error <= 1).length;
        return {
          matches: matches.length / 2,
          accurateMatchesAt1px: accurate,
          accuracyFraction: errors.length ? accurate / errors.length : 0,
          medianPixelError: errors.sort((a, b) => a - b)[Math.floor(errors.length / 2)] ?? null,
        };
      };
      const pipeline = async (kind) => {
        const adapterForRun = await navigator.gpu.requestAdapter();
        if (!adapterForRun) throw new Error(`WebGPU adapter unavailable for ${kind} feature run`);
        const runDescription = [
          adapterForRun.info?.vendor,
          adapterForRun.info?.architecture,
          adapterForRun.info?.device,
          adapterForRun.info?.description,
        ].join(' ');
        if (
          hardware &&
          (/swiftshader|software|llvmpipe/i.test(runDescription) ||
            adapterForRun.isFallbackAdapter === true)
        )
          throw new Error(`${kind} selected a software adapter`);
        const gpuForRun = await adapterForRun.requestDevice();
        gpuForRun.addEventListener('uncapturederror', (event) =>
          validationErrors.push(event.error.message),
        );
        const featureDevice = await ShadeDevice.open({ gpuDevice: gpuForRun });
        const runOnce = async () => {
          const before = featureDevice.statistics,
            started = performance.now(),
            features = [],
            extractionReports = [],
            extractionTimes = [];
          const handles = [];
          try {
            for (const frame of frames) {
              const t = performance.now();
              const feature =
                kind === 'legacy'
                  ? await extractFeaturesGpu(featureDevice, frame, size.width, size.height, options)
                  : await extractFeaturesGpuResident(
                      featureDevice,
                      frame,
                      size.width,
                      size.height,
                      options,
                    );
              extractionTimes.push(performance.now() - t);
              extractionReports.push(feature.report);
              features.push(feature);
              if (kind === 'resident') handles.push(feature);
            }
            const keypoints =
              kind === 'legacy'
                ? features.map((feature) => feature.keypoints)
                : await Promise.all(features.map((feature) => feature.readKeypoints()));
            const current = 3,
              pairs = [],
              matchReports = [],
              pairQuality = [];
            let matchedPairs = 0;
            const maxPair = Math.max(
              ...features.map((feature) =>
                kind === 'legacy' ? feature.scores.length : feature.count,
              ),
            );
            if (maxPair * maxPair <= 4_000_000) {
              for (const previous of [2, 1, 0]) {
                const t = performance.now(),
                  match =
                    kind === 'legacy'
                      ? await matchFeaturesGpu(featureDevice, features[current], features[previous])
                      : await matchResidentFeaturesGpu(
                          featureDevice,
                          features[current],
                          features[previous],
                        );
                pairs.push({ previousFrame: previous, elapsedMs: performance.now() - t });
                matchReports.push(match.report);
                const quality = matchQuality(
                  match.matches,
                  [keypoints[current], keypoints[previous]],
                  {
                    x: shifts[previous].x - shifts[current].x,
                    y: shifts[previous].y - shifts[current].y,
                  },
                );
                matchedPairs += quality.matches;
                pairQuality.push({ previousFrame: previous, ...quality });
              }
            }
            return {
              elapsedMs: performance.now() - started,
              extractionMs: extractionTimes,
              features: features.map((feature) =>
                kind === 'legacy' ? feature.scores.length : feature.count,
              ),
              extractionReports,
              matches: pairs,
              matchReports,
              matchedPairs,
              pairQuality,
              pairLimitSkipped: maxPair * maxPair > 4_000_000,
              statisticsDelta: delta(before, featureDevice.statistics),
            };
          } finally {
            for (const handle of handles) await handle.dispose();
          }
        };
        try {
          const samples = [];
          for (let i = 0; i <= warmRuns; i++)
            samples.push({
              kind: i === 0 ? 'cold-first-call' : 'warm-steady-state',
              ...(await runOnce()),
            });
          return {
            mode: kind,
            deviceBudget: {
              maxKernelElements: featureDevice.maxKernelElements,
              workingBytes: featureDevice.workingBytes,
              bindingBytes: featureDevice.bindingBytes,
            },
            samples,
            warmMedianMs: samples
              .slice(1)
              .map((sample) => sample.elapsedMs)
              .sort((a, b) => a - b)[Math.floor(warmRuns / 2)],
          };
        } finally {
          await featureDevice.dispose();
          gpuForRun.destroy();
        }
      };
      const sourceScale = {
        x: sourceWidth / size.width,
        y: sourceHeight / size.height,
      };
      const legacy = await pipeline('legacy'),
        resident = residentApiAvailable ? await pipeline('resident') : null;
      const legacySteady = legacy.samples.at(-1),
        residentSteady = resident?.samples.at(-1);
      const qualityComparison = residentSteady
        ? {
            legacyFeatures: legacySteady.features,
            residentFeatures: residentSteady.features,
            featureCountDelta: residentSteady.features.map(
              (count, i) => count - legacySteady.features[i],
            ),
            pairs: legacySteady.pairQuality.map((baselinePair, i) => {
              const residentPair = residentSteady.pairQuality[i];
              return {
                previousFrame: baselinePair.previousFrame,
                matchCountDelta: residentPair.matches - baselinePair.matches,
                accuracyFractionDelta:
                  residentPair.accuracyFraction - baselinePair.accuracyFraction,
                legacy: baselinePair,
                resident: residentPair,
              };
            }),
          }
        : null;
      featureRuns.push({
        ...size,
        sourceScale,
        inputMode: 'four deterministic translated analysis-size frames; no 18K source allocation',
        inputGenerationMs,
        temporalWindow: 3,
        residentApiAvailable,
        legacy,
        resident,
        qualityComparison,
      });
    }
    return {
      adapter: { description: description.trim(), software },
      backend: 'webgpu',
      sfmConfigurations,
      featureRuns,
      validationErrors,
    };
  },
  {
    sizes,
    sourceWidth,
    sourceHeight,
    warmRuns,
    configurations,
    runFeatures,
    mapping: {
      images: mapping.images.map((view) => ({
        ...view,
        keypoints: Array.from(view.keypoints),
      })),
      pairs: mapping.pairs.map((pair) => ({
        ...pair,
        matches: Array.from(pair.matches),
      })),
    },
    hardware: process.env.TYPECV_HARDWARE_GPU === '1',
  },
  { timeoutMs: 300000 },
);
assert.deepEqual(errors, []);
assert.deepEqual(result.validationErrors, []);
assert.equal(result.backend, 'webgpu');
assert.equal(result.sfmConfigurations.length, configurations.length);
assert.ok(
  result.sfmConfigurations.every((configuration) => configuration.runs.length === warmRuns + 1),
);
const gpuSfm = result.sfmConfigurations[0].runs[0].output;
assert.equal(gpuSfm.registeredViews, cpuSummary.registeredViews);
assert.equal(gpuSfm.landmarks, cpuSummary.landmarks);
assert.deepEqual(gpuSfm.unregistered, cpuSummary.unregistered);
for (const configuration of result.sfmConfigurations)
  for (const run of configuration.runs) {
    assert.deepEqual(run.output.initialPair, cpuSummary.initialPair);
    assert.deepEqual(run.output.registrationOrder, cpuSummary.registrationOrder);
    assert.equal(run.output.observationCount, 160);
    assert.ok(run.output.meanReprojectionErrorPx < 0.01);
  }
const report = {
  schemaVersion: 1,
  gpuEntrySha256: createHash('sha256')
    .update(await readFile(new URL('../dist/gpu/index.js', import.meta.url)))
    .digest('hex'),
  mode: process.env.TYPECV_HARDWARE_GPU === '1' ? 'hardware-requested' : 'software-webgpu',
  sourceResolution: { width: sourceWidth, height: sourceHeight },
  sourcePixelsAllocated: false,
  analysisSizes: sizes,
  warmRuns,
  sourceFixture:
    'TypeCV tests/typeshade/geometry-cases.mjs mappingFixture(): 4 views, 40 points, 6 pairs',
  fixtureSha256: createHash('sha256').update(JSON.stringify(mapping.images)).digest('hex'),
  pairFixtureSha256: createHash('sha256').update(JSON.stringify(mapping.pairs)).digest('hex'),
  options: { seed: 42, ransacIterations: 128, bundleIterations: 2, finalBundleIterations: 4 },
  cpu: cpuSummary,
  ...result,
};
const bytes = `${JSON.stringify(report, null, 2)}\n`;
const target = resolve(outputPath);
await mkdir(dirname(target), { recursive: true });
await writeFile(target, bytes);
console.log(
  JSON.stringify({
    backend: report.backend,
    sourceResolution: report.sourceResolution,
    sfm: report.sfmConfigurations.map((configuration) => ({
      name: configuration.configuration.name,
      coldMs: configuration.runs[0].elapsedMs,
      warmAvgMs:
        configuration.runs.slice(1).reduce((sum, run) => sum + run.elapsedMs, 0) / warmRuns,
      dispatches: configuration.runs[0].output.stages.reduce(
        (sum, stage) => sum + stage.dispatches,
        0,
      ),
      submissions: configuration.runs[0].output.stages.reduce(
        (sum, stage) => sum + stage.submissions,
        0,
      ),
    })),
    features: report.featureRuns.map((entry) => ({
      size: `${entry.width}x${entry.height}`,
      legacyWarmMs:
        entry.legacy.samples.slice(1).reduce((sum, sample) => sum + sample.elapsedMs, 0) / warmRuns,
      residentWarmMs:
        entry.resident.samples.slice(1).reduce((sum, sample) => sum + sample.elapsedMs, 0) /
        warmRuns,
      quality: entry.qualityComparison,
    })),
  }),
);
console.error(`Saved benchmark JSON to ${target}`);
