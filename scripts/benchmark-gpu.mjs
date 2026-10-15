import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { inBrowser } from '../tests/typeshade/browser.mjs';
import { reconstructTwoView } from '../dist/index.js';

// Reuses the exact browser/toolchain exercised by CI. Default is software WebGPU,
// useful for correctness and transfer accounting, never a hardware speed claim.
const sequencePath = process.argv[2];
let sequence, sequenceSha256;
if (sequencePath) {
  const bytes = await readFile(sequencePath);
  sequence = JSON.parse(bytes.toString('utf8'));
  sequenceSha256 = createHash('sha256').update(bytes).digest('hex');
}
const pairs = sequence
  ? sequence.pairs
      .filter((p) => p.matches.length >= 32)
      .map((p) => {
        const a = sequence.views.find((v) => v.id === p.first),
          b = sequence.views.find((v) => v.id === p.second);
        const pixels = [];
        for (let i = 0; i < p.matches.length; i += 2) {
          const u = p.matches[i] * 2,
            v = p.matches[i + 1] * 2;
          pixels.push(a.keypoints[u], a.keypoints[u + 1], b.keypoints[v], b.keypoints[v + 1]);
        }
        return { first: a.camera, second: b.camera, pixels, ids: [p.first, p.second] };
      })
  : [];
const options = { iterations: 1000, minIterations: 64, threshold: 0.0015, batchSize: 64, seed: 42 };
const cpuPairs = pairs.map((p) => {
  try {
    const r = reconstructTwoView(Float64Array.from(p.pixels), p.first, p.second, options);
    return {
      ids: p.ids,
      inliers: r.inlierCount,
      valid: r.positiveDepthCount,
      errorPx: r.meanReprojectionErrorPx,
    };
  } catch (e) {
    return { ids: p.ids, failure: e.message };
  }
});
const { result, errors } = await inBrowser(
  async ({ pairs, options, hardware }) => {
    const { ShadeDevice, EpipolarSession, reconstructTwoViewGpu } = await import('/dist/gpu.js');
    const { fixture, reference, reconstructionFixture, poseMetrics } =
      await import('/tests/typeshade/cases.mjs');
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error('WebGPU adapter is unavailable');
    const info = adapter.info,
      description = [info?.vendor, info?.architecture, info?.device, info?.description].join(' ');
    const software =
      /swiftshader|software|llvmpipe/i.test(description) || adapter.isFallbackAdapter === true;
    if (hardware && software)
      throw new Error('Hardware run requested but a software adapter was selected');
    const gpu = await adapter.requestDevice(),
      validationErrors = [];
    gpu.addEventListener('uncapturederror', (e) => validationErrors.push(e.error.message));
    const d = await ShadeDevice.open({ gpuDevice: gpu });
    try {
      const data = fixture(8193),
        base = data.models.subarray(0, 27),
        models = new Float64Array(64 * 9);
      for (let i = 0; i < 64; i++) models.set(base.subarray((i % 3) * 9, (i % 3) * 9 + 9), i * 9);
      data.models = models;
      const start = performance.now(),
        session = await EpipolarSession.create(d, data.pixels, data.first, data.second);
      const initializationMs = performance.now() - start;
      await session.score(models, data.threshold); // Pipeline compilation/warmup is outside steady-state samples.
      const before = d.statistics,
        times = [];
      let last;
      for (let i = 0; i < 5; i++) {
        const start = performance.now();
        last = await session.score(models, data.threshold);
        times.push(performance.now() - start);
      }
      const after = d.statistics,
        expected = reference(data);
      const scoring = {
        matches: session.count,
        hypotheses: 64,
        initializationMs,
        sampleMs: times,
        bestIndex: last.bestIndex,
        inliers: last.inlierCount,
        expectedBestIndex: expected.best,
        expectedInliers: expected.counts[expected.best],
        dispatches: last.report.dispatches,
        submissions: last.report.submissions,
        applicationUploadBytesPerBatch:
          (after.applicationUploadBytes - before.applicationUploadBytes) / 5,
        applicationReadbackBytesPerBatch:
          (after.applicationReadbackBytes - before.applicationReadbackBytes) / 5,
        newAllocationsAfterWarmup: after.allocations - before.allocations,
      };
      await session.dispose();
      const truth = reconstructionFixture(),
        reconstructed = await reconstructTwoViewGpu(
          d,
          truth.pixels,
          truth.first,
          truth.second,
          options,
        );
      const realPairs = [];
      for (const p of pairs) {
        const start = performance.now();
        try {
          const r = await reconstructTwoViewGpu(
            d,
            Float64Array.from(p.pixels),
            p.first,
            p.second,
            options,
          );
          realPairs.push({
            ids: p.ids,
            matches: p.pixels.length / 4,
            inliers: r.inlierCount,
            valid: r.positiveDepthCount,
            errorPx: r.meanReprojectionErrorPx,
            elapsedMs: performance.now() - start,
            execution: r.execution,
          });
        } catch (e) {
          realPairs.push({ ids: p.ids, failure: e.message });
        }
      }
      await d.dispose();
      return {
        adapter: description.trim(),
        software,
        backend: d.backend,
        scoring,
        synthetic: poseMetrics(reconstructed, truth),
        realPairs,
        validationErrors,
        disposedLiveBuffers: d.statistics.liveBuffers,
      };
    } finally {
      await d.dispose();
      gpu.destroy();
    }
  },
  { pairs, options, hardware: process.env.TYPECV_HARDWARE_GPU === '1' },
);
assert.deepEqual(errors, []);
assert.deepEqual(result.validationErrors, []);
assert.equal(result.scoring.bestIndex, result.scoring.expectedBestIndex);
assert.equal(result.scoring.inliers, result.scoring.expectedInliers);
assert.equal(result.scoring.applicationReadbackBytesPerBatch, 12);
assert.equal(result.scoring.newAllocationsAfterWarmup, 0);
assert.equal(result.disposedLiveBuffers, 0);
console.log(
  JSON.stringify(
    {
      mode: process.env.TYPECV_HARDWARE_GPU === '1' ? 'hardware-requested' : 'software-webgpu',
      sequenceMetadata: sequence?.metadata,
      sequenceSha256,
      options,
      ...result,
      cpuPairs,
    },
    null,
    2,
  ),
);
