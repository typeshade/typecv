import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { bundleAdjust } from '../dist/index.js';
import { inBrowser } from '../tests/typeshade/browser.mjs';

const args = process.argv.slice(2);
const arg = (name, fallback) => {
  const prefix = `--${name}=`;
  const value = args.find((item) => item.startsWith(prefix));
  return value ? value.slice(prefix.length) : fallback;
};
const inputPath = resolve(arg('input', 'work/benchmark-results/sfm-colmap-input-profile.json'));
const outputPath = resolve(arg('output', 'work/benchmark-results/sfm-bundle-profile.json'));
const runs = Number(arg('runs', '3'));
const solverCheckpointsArg = arg('solver-checkpoints', '1');
if (!['0', '1'].includes(solverCheckpointsArg))
  throw new RangeError('--solver-checkpoints must be 0 or 1');
const solverCheckpoints = solverCheckpointsArg === '1';
if (!Number.isSafeInteger(runs) || runs < 2 || runs > 5)
  throw new RangeError('--runs must be an integer from 2 through 5');
if (process.env.TYPECV_HARDWARE_GPU !== '1')
  throw new Error('Set TYPECV_HARDWARE_GPU=1 to profile on a hardware GPU');

const inputFile = await readFile(inputPath);
const profile = JSON.parse(inputFile);
const reconstruction = profile?.source?.reconstruction;
if (
  !reconstruction ||
  reconstruction.views?.length !== 8 ||
  reconstruction.landmarks?.length !== 726
)
  throw new Error('Expected source.reconstruction with 8 views and 726 landmarks');
const observationCount = reconstruction.landmarks.reduce(
  (sum, point) => sum + point.observations.length,
  0,
);
if (observationCount !== 1895)
  throw new Error(`Expected 1895 observations, received ${observationCount}`);

const perturbedInput = {
  views: reconstruction.views.map((view, index) => ({
    ...view,
    camera: { ...view.camera },
    pose: {
      R: [...view.pose.R],
      // Keep the first two views fixed as BA gauge constraints; lightly perturb the rest.
      t: view.pose.t.map((value, axis) =>
        index < 2 ? value : value + 0.005 * Math.sin((view.id + 1) * 0.73 + axis * 1.17),
      ),
    },
  })),
  landmarks: reconstruction.landmarks.map((point) => ({
    ...point,
    xyz: point.xyz.map(
      (value, axis) => value + 0.01 * Math.sin((point.trackId + 1) * 0.73 + axis * 1.17),
    ),
    observations: point.observations.map((observation) => ({
      ...observation,
      pixel: [...observation.pixel],
    })),
  })),
};
const inputSha256 = createHash('sha256').update(JSON.stringify(perturbedInput)).digest('hex');
const options = { iterations: 8 };

const { result, errors } = await inBrowser(
  async ({ reconstruction: input, runs, options, solverCheckpoints }) => {
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter) throw new Error('WebGPU adapter unavailable');
    const info = adapter.info;
    const adapterDescription = [info?.vendor, info?.architecture, info?.device, info?.description]
      .filter(Boolean)
      .join(' ');
    const software =
      /swiftshader|software|llvmpipe/i.test(adapterDescription) ||
      adapter.isFallbackAdapter === true;
    if (software)
      throw new Error(`Hardware adapter required; selected ${adapterDescription || 'unknown'}`);
    if (!adapter.features.has('timestamp-query'))
      throw new Error('Hardware adapter lacks timestamp-query');
    const gpu = await adapter.requestDevice({ requiredFeatures: ['timestamp-query'] });
    const validationErrors = [];
    gpu.addEventListener('uncapturederror', (event) => validationErrors.push(event.error.message));
    const { ShadeDevice, bundleAdjustGpu } = await import('/dist/gpu.js');
    const device = await ShadeDevice.open({
      gpuDevice: gpu,
      backend: 'webgpu',
      boundedSubmissions: false,
      solverCheckpoints,
    });
    const measuredRuns = [];
    try {
      for (let index = 0; index < runs; index++) {
        const started = performance.now();
        const before = device.statistics;
        const output = await bundleAdjustGpu(device, input, options);
        const wallMs = performance.now() - started;
        const groups = output.execution.report.groups.map((group) => ({
          calls: [...group.calls],
          fused: group.fused,
          why: group.why,
          workgroupShapes: group.workgroupShapes?.map((shape) => [...shape]),
          gpuTimeNs: group.gpuTime ?? null,
        }));
        measuredRuns.push({
          run: index + 1,
          wallMs,
          actualGpuTimeNs: groups.every((group) => Number.isFinite(group.gpuTimeNs))
            ? groups.reduce((sum, group) => sum + group.gpuTimeNs, 0)
            : null,
          applicationReadbackBytes: output.execution.applicationReadbackBytes,
          applicationUploadBytes: output.execution.applicationUploadBytes,
          dispatches: output.execution.report.dispatches,
          submissions: output.execution.report.submissions,
          cgIterations: output.cgIterations,
          linearSolverFailures: output.linearSolverFailures,
          diagnostics: output.diagnostics,
          groupStats: groups,
          reconstruction: {
            views: output.reconstruction.views.map((view) => ({
              ...view,
              camera: { ...view.camera },
              pose: { R: Array.from(view.pose.R), t: Array.from(view.pose.t) },
            })),
            landmarks: output.reconstruction.landmarks.map((point) => ({
              ...point,
              xyz: Array.from(point.xyz),
              observations: point.observations.map((observation) => ({
                ...observation,
                pixel: [...observation.pixel],
              })),
            })),
          },
        });
        if (
          device.statistics.applicationReadbackBytes - before.applicationReadbackBytes !==
          output.execution.applicationReadbackBytes
        )
          throw new Error('Readback counters disagree with the bundle execution report');
      }
      return {
        adapter: adapterDescription || 'unknown hardware adapter',
        validationErrors,
        backend: device.backend,
        limits: {
          workingBytes: device.workingBytes,
          bindingBytes: device.bindingBytes,
          maxKernelElements: device.maxKernelElements,
        },
        runs: measuredRuns,
        finalDeviceStatistics: device.statistics,
      };
    } finally {
      await device.dispose();
    }
  },
  { reconstruction: perturbedInput, runs, options, solverCheckpoints },
  { timeoutMs: 300000 },
);
assert.deepEqual(errors, []);
assert.deepEqual(result.validationErrors, []);
assert.equal(result.backend, 'webgpu');
assert.equal(result.runs.length, runs);

const cpuStarted = performance.now();
const cpu = bundleAdjust(perturbedInput, { iterations: options.iterations });
const cpuWallMs = performance.now() - cpuStarted;

const cpuById = new Map(cpu.reconstruction.views.map((view) => [view.id, view]));
const runQuality = result.runs.map((run) => {
  const gpuById = new Map(run.reconstruction.views.map((view) => [view.id, view]));
  let poseSum = 0,
    poseCount = 0,
    pointSum = 0,
    maxCoordinateDelta = 0;
  for (const cpuView of cpu.reconstruction.views) {
    const gpuView = gpuById.get(cpuView.id);
    for (const key of ['R', 't']) {
      for (let axis = 0; axis < cpuView.pose[key].length; axis++) {
        const delta = Math.abs(cpuView.pose[key][axis] - gpuView.pose[key][axis]);
        maxCoordinateDelta = Math.max(maxCoordinateDelta, delta);
        if (key === 't') {
          poseSum += delta * delta;
          poseCount++;
        }
      }
    }
  }
  for (let index = 0; index < cpu.reconstruction.landmarks.length; index++) {
    const cpuX = cpu.reconstruction.landmarks[index].xyz;
    const gpuX = run.reconstruction.landmarks[index].xyz;
    for (let axis = 0; axis < 3; axis++) {
      const delta = Math.abs(cpuX[axis] - gpuX[axis]);
      maxCoordinateDelta = Math.max(maxCoordinateDelta, delta);
      pointSum += delta * delta;
    }
  }
  const costDelta = Math.abs(cpu.diagnostics.finalCost - run.diagnostics.finalCost);
  const passed =
    maxCoordinateDelta <= 1e-7 &&
    costDelta <= 1e-6 * Math.max(1, Math.abs(cpu.diagnostics.finalCost)) &&
    cpu.diagnostics.acceptedSteps === run.diagnostics.acceptedSteps &&
    cpu.diagnostics.rejectedSteps === run.diagnostics.rejectedSteps;
  return {
    run: run.run,
    passed,
    maxCoordinateDelta,
    costDelta,
    poseTranslationRms: Math.sqrt(poseSum / poseCount),
    landmarkCoordinateRms: Math.sqrt(pointSum / (cpu.reconstruction.landmarks.length * 3)),
  };
});
const cpuComparison = {
  wallMs: cpuWallMs,
  diagnostics: cpu.diagnostics,
  gpuFinalDiagnostics: result.runs.at(-1).diagnostics,
  poseTranslationRms: runQuality.at(-1).poseTranslationRms,
  landmarkCoordinateRms: runQuality.at(-1).landmarkCoordinateRms,
  qualityPassed: runQuality.every((run) => run.passed),
  runQuality,
};

const groupTotals = new Map();
for (const run of result.runs.slice(1)) {
  for (const group of run.groupStats) {
    const key = group.calls.join('+');
    const value = groupTotals.get(key) ?? {
      calls: group.calls,
      count: 0,
      gpuTimeNs: 0,
      hasGpuTime: true,
      workgroupShapes: new Map(),
    };
    value.count++;
    if (Number.isFinite(group.gpuTimeNs)) value.gpuTimeNs += group.gpuTimeNs;
    else value.hasGpuTime = false;
    for (const shape of group.workgroupShapes ?? []) {
      const shapeKey = shape.join('x');
      value.workgroupShapes.set(shapeKey, (value.workgroupShapes.get(shapeKey) ?? 0) + 1);
    }
    groupTotals.set(key, value);
  }
}
const kernelSummary = [...groupTotals.values()]
  .map((value) => ({
    calls: value.calls,
    count: value.count,
    totalGpuTimeNs: value.hasGpuTime ? value.gpuTimeNs : null,
    meanGpuTimeNs: value.hasGpuTime ? value.gpuTimeNs / value.count : null,
    workgroupShapes: Object.fromEntries(value.workgroupShapes),
  }))
  .sort((a, b) => (b.totalGpuTimeNs ?? 0) - (a.totalGpuTimeNs ?? 0));

const report = {
  schemaVersion: 1,
  benchmark: 'sfm-bundle-kernel-profile',
  input: {
    path: inputPath,
    sourceSha256: createHash('sha256').update(inputFile).digest('hex'),
    perturbedReconstructionSha256: inputSha256,
    views: perturbedInput.views.length,
    points: perturbedInput.landmarks.length,
    observations: observationCount,
    perturbation: {
      pointXYZSinusoidAmplitude: 0.01,
      nonAnchorTranslationSinusoidAmplitude: 0.005,
      firstTwoViewsFixed: true,
    },
  },
  options,
  solverCheckpoints,
  mode: 'hardware-timestamp-query',
  adapter: result.adapter,
  backend: result.backend,
  limits: result.limits,
  runs: result.runs.map(({ reconstruction: _reconstruction, ...run }) => run),
  kernelSummary,
  cpuIndependentComparison: cpuComparison,
  validationErrors: result.validationErrors,
  finalDeviceStatistics: result.finalDeviceStatistics,
};
await mkdir(dirname(outputPath), { recursive: true });
await writeFile(outputPath, `${JSON.stringify(report, null, 2)}\n`);
console.log(
  JSON.stringify({
    output: outputPath,
    adapter: report.adapter,
    input: report.input,
    solverCheckpoints: report.solverCheckpoints,
    runs: report.runs.map((run) => ({
      run: run.run,
      wallMs: run.wallMs,
      actualGpuTimeNs: run.actualGpuTimeNs,
      readbackBytes: run.applicationReadbackBytes,
      cgIterations: run.cgIterations,
      diagnostics: run.diagnostics,
    })),
    cpuIndependentComparison: report.cpuIndependentComparison,
    kernelSummary: report.kernelSummary,
  }),
);

assert.equal(cpuComparison.qualityPassed, true, `BA quality gate failed; inspect ${outputPath}`);
