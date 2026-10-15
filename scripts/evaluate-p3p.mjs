import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { inBrowser } from '../tests/typeshade/browser.mjs';
import {
  p3pCases,
  packP3pCases,
  p3pMetrics,
  poseDifference,
} from '../tests/typeshade/p3p-cases.mjs';
import { p3p } from '../dist/geometry/p3p.js';
import { unprojectPixel } from '../dist/index.js';

const modelPath = process.argv[2];
let cases = p3pCases(),
  source = { kind: 'synthetic-known-pose', problems: cases.length };
if (modelPath) {
  const bytes = await readFile(modelPath),
    model = JSON.parse(bytes.toString('utf8'));
  source = {
    kind: 'reconstructed-landmarks-and-observed-pixels',
    modelSha256: createHash('sha256').update(bytes).digest('hex'),
    samplesPerView: 32,
  };
  cases = [];
  let state = 713;
  const random = (n) => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;
    return (state >>> 0) % n;
  };
  for (const view of model.views) {
    const observations = model.landmarks.flatMap((p) =>
      p.observations
        .filter((o) => o.viewId === view.id)
        .map((o) => ({ world: p.xyz, pixel: o.pixel, trackId: p.trackId })),
    );
    if (observations.length < 3)
      throw new Error(`View ${view.id} has fewer than three observations`);
    for (let sample = 0; sample < 32; sample++) {
      const indices = new Set();
      while (indices.size < 3) indices.add(random(observations.length));
      const points = [...indices].map((i) => observations[i]);
      const rays = points.map((p) => {
        const xy = unprojectPixel(view.camera, p.pixel);
        if (!xy) throw new Error('Invalid observed camera ray');
        return [...xy, 1];
      });
      cases.push({
        name: `view-${view.id}-sample-${sample}`,
        world: points.map((p) => p.world),
        rays,
        trackIds: points.map((p) => p.trackId),
      });
    }
  }
}
const { result, errors } = await inBrowser(
  async ({ packed, hardware }) => {
    const { ShadeDevice, solveP3pBatchGpu } = await import('/dist/gpu.js');
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('WebGPU adapter unavailable');
    const info = adapter.info,
      description = [info.vendor, info.architecture, info.device, info.description].join(' ');
    const software =
      /swiftshader|software|llvmpipe/i.test(description) || adapter.isFallbackAdapter === true;
    if (hardware && software) throw new Error('Hardware requested but software adapter selected');
    const gpu = await adapter.requestDevice(),
      d = await ShadeDevice.open({ gpuDevice: gpu }),
      validationErrors = [];
    gpu.addEventListener('uncapturederror', (e) => validationErrors.push(e.error.message));
    const originalDispatch = GPUComputePassEncoder.prototype.dispatchWorkgroups,
      originalSubmit = GPUQueue.prototype.submit,
      originalMap = GPUBuffer.prototype.mapAsync;
    const calls = { dispatches: 0, submissions: 0, maps: 0 };
    GPUComputePassEncoder.prototype.dispatchWorkgroups = function (...args) {
      calls.dispatches++;
      return originalDispatch.apply(this, args);
    };
    GPUQueue.prototype.submit = function (...args) {
      calls.submissions++;
      return originalSubmit.apply(this, args);
    };
    GPUBuffer.prototype.mapAsync = function (...args) {
      calls.maps++;
      return originalMap.apply(this, args);
    };
    try {
      const batch = await solveP3pBatchGpu(d, Float64Array.from(packed)),
        solved = { ...calls },
        beforeRead = d.statistics;
      const values = await batch.read(),
        data = Object.fromEntries(Object.entries(values).map(([key, array]) => [key, [...array]]));
      await batch.dispose();
      return {
        data,
        solved,
        beforeRead,
        afterRead: d.statistics,
        report: batch.report,
        adapter: { description, software },
        validationErrors,
      };
    } finally {
      GPUComputePassEncoder.prototype.dispatchWorkgroups = originalDispatch;
      GPUQueue.prototype.submit = originalSubmit;
      GPUBuffer.prototype.mapAsync = originalMap;
      await d.dispose();
      gpu.destroy();
    }
  },
  { packed: [...packP3pCases(cases)], hardware: process.env.TYPECV_HARDWARE_GPU === '1' },
);
assert.deepEqual(errors, []);
assert.deepEqual(result.validationErrors, []);
const data = Object.fromEntries(
  Object.entries(result.data).map(([key, array]) => [
    key,
    key === 'poses' ? Float64Array.from(array) : Uint32Array.from(array),
  ]),
);
const metrics = p3pMetrics(cases, data),
  outcomes = [];
for (let i = 0; i < cases.length; i++) {
  const item = cases[i],
    actual = metrics.output[i];
  // The GPU intentionally has an explicit side-ratio condition guard absent from the CPU oracle.
  const reference =
    item.status === 6 || item.name === 'zero-ray'
      ? []
      : p3p(
          item.world.map((p) => Float64Array.from(p)),
          item.rays.map((p) => Float64Array.from(p)),
        );
  const expected = reference.map((p) => [...p.R, ...p.t]);
  const directed = (a, b) =>
    a.length
      ? Math.max(
          ...a.map((p) => Math.min(...b.map((q) => poseDifference(p, q, actual.sceneScale)))),
        )
      : 0;
  const difference = Math.max(
    directed(expected, actual.candidates),
    directed(actual.candidates, expected),
  );
  outcomes.push({
    name: item.name,
    ...(item.trackIds ? { trackIds: item.trackIds } : {}),
    status: actual.status,
    cpuCount: expected.length,
    gpuCount: actual.candidates.length,
    branchDifference: Number.isFinite(difference) ? difference : null,
    agreement: expected.length === actual.candidates.length && difference < 1e-4,
  });
}
const report = {
  source,
  adapter: result.adapter,
  precision: 'TypeShade float-flavor emulated f64 (~48 significand bits, f32 exponent range)',
  observations:
    'Oracle agreement and residual checks only; real model coordinates and assumed calibration are not ground truth. P3P is not connected to GPU PnP RANSAC or the mapper yet.',
  problems: cases.length,
  agreeing: outcomes.filter((o) => o.agreement).length,
  maxRayError: metrics.maxRayError,
  maxOrthogonalityError: metrics.maxOrthogonalityError,
  maxTruthError: modelPath ? null : metrics.maxTruthError,
  validationFailures: metrics.failures,
  actualCallsBeforeRead: result.solved,
  applicationReadbackBytesBeforeRead: result.beforeRead.applicationReadbackBytes,
  applicationUploadBytes: result.beforeRead.applicationUploadBytes,
  explicitReadbackBytes: result.afterRead.applicationReadbackBytes,
  resultReport: result.report,
  outcomes,
};
console.log(JSON.stringify(report, null, 2));
assert.equal(
  report.agreeing,
  report.problems,
  'P3P branch disagreement; all outcomes printed above',
);
assert.deepEqual(report.validationFailures, []);
assert.ok(report.maxRayError < 1e-7);
assert.ok(report.maxOrthogonalityError < 1e-7);
assert.equal(result.solved.dispatches, 1);
assert.equal(result.solved.maps, 0);
