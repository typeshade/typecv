import test from 'node:test';
import assert from 'node:assert/strict';
import { inBrowser } from './browser.mjs';
import { p3p } from '../../dist/geometry/p3p.js';
import { p3pCases, poseDifference } from './p3p-cases.mjs';

test('P3P generates all pose branches in one WGSL dispatch with no candidate readback', async () => {
  const { result, errors } = await inBrowser(async () => {
    const { ShadeDevice, solveP3pBatchGpu } = await import('/dist/gpu.js');
    const { p3pCases, packP3pCases, p3pMetrics } = await import('/tests/typeshade/p3p-cases.mjs');
    const cases = p3pCases(),
      adapter = await navigator.gpu.requestAdapter(),
      gpu = await adapter.requestDevice();
    const d = await ShadeDevice.open({ gpuDevice: gpu }),
      gpuErrors = [];
    gpu.addEventListener('uncapturederror', (e) => gpuErrors.push(e.error.message));
    const originalDispatch = GPUComputePassEncoder.prototype.dispatchWorkgroups,
      originalSubmit = GPUQueue.prototype.submit,
      originalMap = GPUBuffer.prototype.mapAsync;
    const work = { dispatches: 0, submissions: 0, maps: 0 };
    GPUComputePassEncoder.prototype.dispatchWorkgroups = function (...args) {
      work.dispatches++;
      return originalDispatch.apply(this, args);
    };
    GPUQueue.prototype.submit = function (...args) {
      work.submissions++;
      return originalSubmit.apply(this, args);
    };
    GPUBuffer.prototype.mapAsync = function (...args) {
      work.maps++;
      return originalMap.apply(this, args);
    };
    try {
      console.debug('P3P dispatch');
      const batch = await solveP3pBatchGpu(d, packP3pCases(cases)),
        solved = { ...work },
        before = d.statistics;
      console.debug('P3P read');
      const data = await batch.read(),
        metrics = p3pMetrics(cases, data);
      const readbackBytes = d.statistics.applicationReadbackBytes - before.applicationReadbackBytes;
      await batch.dispose();
      // Reuse must overwrite previous valid slots when the next problems reject.
      const rejected = await solveP3pBatchGpu(
        d,
        packP3pCases(cases.map(() => cases[cases.length - 2])),
      );
      const empty = await rejected.read();
      const stale = [...empty.poses, ...empty.valid, ...empty.counts].some((x) => x !== 0);
      await rejected.dispose();
      return { metrics, solved, before, readbackBytes, report: batch.report, gpuErrors, stale };
    } finally {
      GPUComputePassEncoder.prototype.dispatchWorkgroups = originalDispatch;
      GPUQueue.prototype.submit = originalSubmit;
      GPUBuffer.prototype.mapAsync = originalMap;
      await d.dispose();
      gpu.destroy();
    }
  });
  assert.deepEqual(errors, []);
  assert.deepEqual(result.gpuErrors, []);
  assert.deepEqual(result.metrics.failures, []);
  assert.ok(result.metrics.maxTruthError < 1e-4, JSON.stringify(result.metrics));
  assert.ok(result.metrics.maxRayError < 1e-7);
  assert.ok(result.metrics.maxOrthogonalityError < 1e-7);
  assert.equal(result.report.tier, 'webgpu');
  assert.equal(result.report.dispatches, 1);
  assert.equal(result.solved.dispatches, 1);
  assert.equal(result.solved.submissions, 1);
  assert.equal(result.solved.maps, 0);
  assert.equal(result.before.applicationReadbackBytes, 0);
  assert.equal(result.readbackBytes, p3pCases().length * 408);
  assert.equal(result.stale, false);
  const cases = p3pCases();
  let fourBranches = 0;
  for (let i = 0; i < cases.length; i++) {
    if (cases[i].name === 'zero-ray' || cases[i].status === 6) continue;
    const expected = p3p(
      cases[i].world.map((p) => Float64Array.from(p)),
      cases[i].rays.map((p) => Float64Array.from(p)),
    );
    const actual = result.metrics.output[i];
    assert.equal(actual.candidates.length, expected.length, cases[i].name);
    if (expected.length === 4) fourBranches++;
    for (const p of expected)
      assert.ok(
        actual.candidates.some(
          (q) => poseDifference(q, [...p.R, ...p.t], actual.sceneScale) < 1e-4,
        ),
        cases[i].name,
      );
  }
  assert.ok(fourBranches > 0);
  console.log(
    JSON.stringify({
      p3pProblems: cases.length,
      fourBranches,
      work: result.solved,
      maxTruthError: result.metrics.maxTruthError,
      maxRayError: result.metrics.maxRayError,
    }),
  );
});

test('real WGSL dispatches match f64 and report the actual work, without hidden CPU fallback', async () => {
  const { result, errors } = await inBrowser(async () => {
    const { ShadeDevice, EpipolarSession } = await import('/dist/gpu.js');
    const { fixture, reference } = await import('/tests/typeshade/cases.mjs');
    const data = fixture(513),
      expected = reference(data),
      adapter = await navigator.gpu.requestAdapter(),
      gpu = await adapter.requestDevice(),
      d = await ShadeDevice.open({ gpuDevice: gpu });
    const gpuErrors = [];
    gpu.addEventListener('uncapturederror', (e) => gpuErrors.push(e.error.message));
    const originalSubmit = GPUQueue.prototype.submit,
      originalDispatch = GPUComputePassEncoder.prototype.dispatchWorkgroups,
      originalMap = GPUBuffer.prototype.mapAsync,
      originalWrite = GPUQueue.prototype.writeBuffer;
    const counts = { submissions: 0, dispatches: 0, maps: 0, writes: 0 };
    GPUQueue.prototype.submit = function (...args) {
      counts.submissions++;
      return originalSubmit.apply(this, args);
    };
    GPUComputePassEncoder.prototype.dispatchWorkgroups = function (...args) {
      counts.dispatches++;
      return originalDispatch.apply(this, args);
    };
    GPUBuffer.prototype.mapAsync = function (...args) {
      counts.maps++;
      return originalMap.apply(this, args);
    };
    GPUQueue.prototype.writeBuffer = function (...args) {
      counts.writes++;
      return originalWrite.apply(this, args);
    };
    try {
      const s = await EpipolarSession.create(d, data.pixels, data.first, data.second);
      for (const k in counts) counts[k] = 0;
      const scored = await s.score(data.models, data.threshold, {
        readMask: true,
        readStatistics: true,
      });
      const seen = { ...counts };
      const before = d.statistics;
      for (const k in counts) counts[k] = 0;
      const again = await s.score(data.models, data.threshold);
      const repeat = { ...counts };
      const after = d.statistics;
      const changedModels = data.models.slice();
      changedModels.set(data.models.subarray(18, 27), 0);
      const changed = await s.score(changedModels, data.threshold);
      await s.dispose();
      await d.dispose();
      // A borrowed GPUDevice must remain usable after TypeCV is disposed.
      gpu.pushErrorScope('validation');
      const probe = gpu.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_DST });
      gpu.queue.writeBuffer(probe, 0, new Float32Array([1]));
      await gpu.queue.onSubmittedWorkDone();
      probe.destroy();
      const ownershipError = await gpu.popErrorScope();
      return {
        backend: d.backend,
        expected,
        scored: { ...scored, mask: [...scored.mask], modelStatistics: [...scored.modelStatistics] },
        again,
        seen,
        repeat,
        before,
        after,
        changed,
        disposed: d.statistics,
        gpuErrors,
        ownershipError: ownershipError?.message ?? null,
      };
    } finally {
      GPUQueue.prototype.submit = originalSubmit;
      GPUComputePassEncoder.prototype.dispatchWorkgroups = originalDispatch;
      GPUBuffer.prototype.mapAsync = originalMap;
      GPUQueue.prototype.writeBuffer = originalWrite;
      await d.dispose();
      gpu.destroy();
    }
  });
  console.log(
    JSON.stringify({
      backend: result.backend,
      maskDispatches: result.seen.dispatches,
      summaryDispatches: result.repeat.dispatches,
      summaryReadbackBytes:
        result.after.applicationReadbackBytes - result.before.applicationReadbackBytes,
      fused: result.again.report,
    }),
  );
  assert.deepEqual(errors, []);
  assert.deepEqual(result.gpuErrors, []);
  assert.equal(result.ownershipError, null);
  assert.equal(result.backend, 'webgpu');
  assert.equal(result.scored.report.tier, 'webgpu');
  assert.equal(result.scored.bestIndex, 2);
  assert.deepEqual(result.scored.mask, result.expected.masks[2]);
  const h = result.expected.counts.length;
  for (let i = 0; i < h; i++) {
    assert.equal(result.scored.modelStatistics[i], result.expected.counts[i]);
    assert.ok(Math.abs(result.scored.modelStatistics[h + i] - result.expected.costs[i]) < 0.002);
  }
  assert.equal(result.seen.dispatches, result.scored.report.dispatches);
  assert.equal(result.seen.submissions, result.scored.report.submissions + 3);
  assert.equal(result.seen.maps, 3);
  assert.equal(result.repeat.dispatches, result.again.report.dispatches);
  assert.equal(result.repeat.submissions, result.again.report.submissions + 1);
  assert.equal(result.repeat.maps, 1);
  assert.equal(result.scored.report.dispatches, 5);
  assert.equal(result.again.report.dispatches, 4);
  assert.equal(result.again.report.submissions, 1);
  assert.equal(result.again.report.groups[0].fused, true);
  assert.deepEqual(result.again.report.groups[0].calls, [
    'sampsonBatch',
    'inlierFlags',
    'msacCosts',
  ]);
  assert.equal(result.after.applicationReadbackBytes - result.before.applicationReadbackBytes, 12);
  assert.equal(
    result.after.applicationUploadBytes - result.before.applicationUploadBytes,
    5 * 9 * 4,
  );
  assert.equal(result.after.allocations, result.before.allocations);
  assert.equal(result.changed.bestIndex, 0);
  assert.equal(result.changed.inlierCount, result.again.inlierCount);
  assert.equal(result.disposed.liveBuffers, 0);
});

test('actual WebGPU RANSAC feeds pose recovery and sparse reconstruction', async () => {
  const { result, errors } = await inBrowser(async () => {
    const { ShadeDevice, reconstructTwoViewGpu } = await import('/dist/gpu.js');
    const { reconstructionFixture, poseMetrics } = await import('/tests/typeshade/cases.mjs');
    const data = reconstructionFixture(),
      d = await ShadeDevice.open();
    try {
      const r = await reconstructTwoViewGpu(d, data.pixels, data.first, data.second, {
        iterations: 500,
        minIterations: 32,
        threshold: 0.001,
        batchSize: 16,
        seed: 71,
      });
      return { ...poseMetrics(r, data), reports: d.reports.map((r) => r.tier) };
    } finally {
      await d.dispose();
    }
  });
  console.log(JSON.stringify(result));
  assert.deepEqual(errors, []);
  assert.ok(result.valid >= 250);
  assert.ok(result.rotationErrorDeg < 0.1);
  assert.ok(result.translationCosine > 0.999);
  assert.ok(result.errorPx < 0.15);
  assert.equal(result.execution.backend, 'webgpu');
  assert.ok(result.execution.hypothesisBatches > 1);
  assert.ok(result.execution.scoreDispatches > 4);
  assert.ok(result.reports.length > 1 && result.reports.every((t) => t === 'webgpu'));
});
