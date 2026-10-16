import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { inBrowser } from '../tests/typeshade/browser.mjs';
import { reconstructIncremental } from '../dist/index.js';
import { compareCameraModels } from './geometry-comparison.mjs';

const [input, modelOutput] = process.argv.slice(2);
if (!input)
  throw new Error('Usage: node scripts/benchmark-geometry.mjs sequence.json [gpu-model.json]');
const bytes = await readFile(input),
  data = JSON.parse(bytes.toString('utf8'));
const options = { seed: 42, ransacIterations: 750 };
const views = data.views.map((v) => ({ ...v, keypoints: Float64Array.from(v.keypoints) }));
const pairs = data.pairs.map((p) => ({
  ...p,
  matches: Uint32Array.from(p.matches),
  distances: Float64Array.from(p.distances),
}));
const started = performance.now(),
  reference = reconstructIncremental(views, pairs, options),
  cpuElapsedMs = performance.now() - started;
const { result, errors } = await inBrowser(
  async ({ data, options, hardware }) => {
    const { ShadeDevice, reconstructIncrementalGpu } = await import('/dist/gpu.js');
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('WebGPU unavailable');
    const info = adapter.info,
      description = [info.vendor, info.architecture, info.device, info.description].join(' ');
    const software =
      /swiftshader|software|llvmpipe/i.test(description) || adapter.isFallbackAdapter === true;
    if (hardware && software) throw new Error('Hardware requested but software adapter selected');
    const gpu = await adapter.requestDevice(),
      validationErrors = [];
    gpu.addEventListener('uncapturederror', (e) => validationErrors.push(e.error.message));
    const d = await ShadeDevice.open({ gpuDevice: gpu }),
      original = d.plan;
    d.plan = async function (body) {
      const report = await original.call(this, body);
      console.debug(
        'completed ' +
          report.groups[0]?.calls.join(',') +
          ' (' +
          report.dispatches +
          ' dispatches)',
      );
      return report;
    };
    const oldMap = GPUBuffer.prototype.mapAsync,
      oldSubmit = GPUQueue.prototype.submit,
      oldDispatch = GPUComputePassEncoder.prototype.dispatchWorkgroups;
    const calls = { maps: 0, submissions: 0, dispatches: 0 };
    GPUBuffer.prototype.mapAsync = function (...args) {
      calls.maps++;
      return oldMap.apply(this, args);
    };
    GPUQueue.prototype.submit = function (...args) {
      calls.submissions++;
      return oldSubmit.apply(this, args);
    };
    GPUComputePassEncoder.prototype.dispatchWorkgroups = function (...args) {
      calls.dispatches++;
      return oldDispatch.apply(this, args);
    };
    try {
      const views = data.views.map((v) => ({ ...v, keypoints: Float64Array.from(v.keypoints) }));
      const pairs = data.pairs.map((p) => ({
        ...p,
        matches: Uint32Array.from(p.matches),
        distances: Float64Array.from(p.distances),
      }));
      const start = performance.now(),
        r = await reconstructIncrementalGpu(d, views, pairs, options);
      return {
        elapsedMs: performance.now() - start,
        model: {
          views: r.reconstruction.views.map((v) => ({
            ...v,
            pose: { R: [...v.pose.R], t: [...v.pose.t] },
          })),
          landmarks: r.reconstruction.landmarks.map((p) => ({ ...p, xyz: [...p.xyz] })),
        },
        diagnostics: r.diagnostics,
        unregistered: r.unregistered,
        execution: r.execution,
        adapter: { description, software },
        validationErrors,
        calls,
        statistics: d.statistics,
      };
    } finally {
      GPUBuffer.prototype.mapAsync = oldMap;
      GPUQueue.prototype.submit = oldSubmit;
      GPUComputePassEncoder.prototype.dispatchWorkgroups = oldDispatch;
      await d.dispose();
      gpu.destroy();
    }
  },
  { data, options, hardware: process.env.TYPECV_HARDWARE_GPU === '1' },
  { timeoutMs: 180000 },
);
assert.deepEqual(errors, []);
assert.deepEqual(result.validationErrors, []);
if (modelOutput) await writeFile(modelOutput, JSON.stringify(result.model) + '\n');
const sameGauge = reference.diagnostics.initialPair.every(
  (id, i) => id === result.diagnostics.initialPair[i],
);
const cameras = result.model.views.map((v) => {
  const expected = reference.reconstruction.views.find((p) => p.id === v.id);
  if (!expected) return { id: v.id, shared: false };
  const trace = v.pose.R.reduce((s, x, i) => s + x * expected.pose.R[i], 0);
  const rotationDifferenceDeg =
    (Math.acos(Math.max(-1, Math.min(1, (trace - 1) / 2))) * 180) / Math.PI;
  const center = (p) =>
    [0, 1, 2].map((i) => -p.R[i] * p.t[0] - p.R[i + 3] * p.t[1] - p.R[i + 6] * p.t[2]);
  return {
    id: v.id,
    shared: true,
    rotationDifferenceDeg: sameGauge ? rotationDifferenceDeg : null,
    centerDifference: sameGauge
      ? Math.hypot(...center(v.pose).map((x, i) => x - center(expected.pose)[i]))
      : null,
  };
});
const summarize = (model, diagnostics) => ({
  registeredViews: model.views.length,
  landmarks: model.landmarks.length,
  observations: model.landmarks.reduce((n, p) => n + p.observations.length, 0),
  meanEuclideanErrorPx: diagnostics.meanReprojectionErrorPx,
  initialPair: diagnostics.initialPair,
  registrationOrder: diagnostics.registrationOrder,
});
const aligned = compareCameraModels(reference.reconstruction, result.model);
const cpuModel = {
  views: reference.reconstruction.views.map((v) => ({
    ...v,
    pose: { R: [...v.pose.R], t: [...v.pose.t] },
  })),
  landmarks: reference.reconstruction.landmarks.map((p) => ({ ...p, xyz: [...p.xyz] })),
};
const modelHash = (model) =>
  createHash('sha256')
    .update(JSON.stringify(model) + '\n')
    .digest('hex');
const report = {
  source: data.metadata,
  sequenceSha256: createHash('sha256').update(bytes).digest('hex'),
  options,
  adapter: result.adapter,
  cpu: summarize(reference.reconstruction, reference.diagnostics),
  gpu: summarize(result.model, result.diagnostics),
  elapsedMs: { cpu: cpuElapsedMs, gpu: result.elapsedMs },
  sameGauge,
  cameraReferenceAgreement: cameras,
  similarityAlignment: aligned.alignment,
  alignedCameraReferenceAgreement: aligned.cameras,
  modelSha256: { cpu: modelHash(cpuModel), gpu: modelHash(result.model) },
  unregistered: result.unregistered,
  pairVerification: result.diagnostics.pairVerification,
  deviceStages: result.execution.stages.map((s) => ({
    operation: s.operation,
    tier: s.execution.report.tier,
    dispatches: s.execution.report.dispatches,
    submissions: s.execution.report.submissions,
    applicationUploadBytes: s.execution.applicationUploadBytes,
    applicationReadbackBytes: s.execution.applicationReadbackBytes,
  })),
  gpuBundleAdjustments: result.execution.gpuBundleAdjustments,
  cpuStages: result.execution.cpuStages,
  actualCalls: result.calls,
  applicationTransfers: result.statistics,
  limitations:
    'Implemented numerical geometry uses TypeShade throughout initialization, PnP, triangulation, filtering and joint BA. Host graph/control, row compaction and packing remain; stages repack geometry. Fixed assumed calibration, no ground truth, different accepted observations. Software WebGPU timings do not establish physical-GPU speedup.',
};
console.log(JSON.stringify(report, null, 2));
assert.equal(
  report.gpu.registeredViews,
  report.cpu.registeredViews,
  'Registration regression; all outcomes printed above',
);
assert.ok(report.deviceStages.every((s) => s.tier === 'webgpu'));
