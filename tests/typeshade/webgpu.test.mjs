import test from 'node:test';
import assert from 'node:assert/strict';
import { inBrowser } from './browser.mjs';
import { p3p } from '../../dist/geometry/p3p.js';
import { p3pCases, poseDifference } from './p3p-cases.mjs';
import { bundleAdjust } from '../../dist/reconstruction/bundle-adjustment.js';
import { bundleFixture } from './geometry-cases.mjs';

test('joint GPU BA reduces the fixed robust objective and preserves local boundaries and gauge', async () => {
  const { result, errors } = await inBrowser(async () => {
    const { ShadeDevice, bundleAdjustGpu } = await import('/dist/gpu.js');
    const { bundleFixture, bundleMetrics } = await import('/tests/typeshade/geometry-cases.mjs');
    const adapter = await navigator.gpu.requestAdapter(),
      gpu = await adapter.requestDevice();
    const d = await ShadeDevice.open({ gpuDevice: gpu }),
      outputs = [];
    const oldMap = GPUBuffer.prototype.mapAsync,
      oldDispatch = GPUComputePassEncoder.prototype.dispatchWorkgroups,
      oldSubmit = GPUQueue.prototype.submit;
    let work = { maps: 0, dispatches: 0, submissions: 0, firstMapAfter: 0 };
    GPUBuffer.prototype.mapAsync = function (...args) {
      if (!work.maps) work.firstMapAfter = work.dispatches;
      work.maps++;
      return oldMap.apply(this, args);
    };
    GPUComputePassEncoder.prototype.dispatchWorkgroups = function (...args) {
      work.dispatches++;
      return oldDispatch.apply(this, args);
    };
    GPUQueue.prototype.submit = function (...args) {
      work.submissions++;
      return oldSubmit.apply(this, args);
    };
    try {
      for (const settings of [{}, { outlier: true }, { shift: [7, -3, 2] }]) {
        console.debug('GPU BA ' + JSON.stringify(settings));
        const input = bundleFixture(settings),
          snapshot = JSON.stringify(input),
          before = bundleMetrics(input, 1);
        const shifted = !!settings.shift,
          anchor = shifted ? 1 : 0,
          scaleView = shifted ? 3 : 1;
        work = { maps: 0, dispatches: 0, submissions: 0, firstMapAfter: 0 };
        const r = await bundleAdjustGpu(d, input, {
            iterations: 12,
            huberPx: 1,
            ...(shifted ? { anchorViewId: 20, scaleViewId: 40 } : {}),
          }),
          after = bundleMetrics(r.reconstruction, 1);
        const center = (p) =>
          [0, 1, 2].map((i) => -p.R[i] * p.t[0] - p.R[i + 3] * p.t[1] - p.R[i + 6] * p.t[2]);
        const a = center(r.reconstruction.views[anchor].pose),
          b = center(r.reconstruction.views[scaleView].pose);
        outputs.push({
          before,
          after,
          diagnostics: r.diagnostics,
          work: { ...work },
          reportedDispatches: r.execution.report.dispatches,
          reportedSubmissions: r.execution.report.submissions,
          // The activity flags the host reads: once after initializeBundleCg and after each chunk
          // of the PCG loop (key `bundleCg`), on every device.
          activityReads: r.execution.report.groups
            .flatMap((group) => group.calls)
            .filter((call) => call === 'snapshotBundleActivity').length,
          chunks: r.execution.report.chunks,
          cg: r.cgIterations,
          linearFailures: r.linearSolverFailures,
          tier: r.execution.report.tier,
          readback: r.execution.applicationReadbackBytes,
          inputPreserved: snapshot === JSON.stringify(input),
          anchorPreserved:
            JSON.stringify(r.reconstruction.views[anchor].pose) ===
            JSON.stringify(input.views[anchor].pose),
          baseline: Math.hypot(...a.map((x, i) => x - b[i])),
        });
      }
      const input = bundleFixture(),
        r = await bundleAdjustGpu(d, input, { iterations: 8, variableViewIds: [30, 40] });
      const fixed = r.reconstruction.views
        .slice(0, 2)
        .every((v, i) => JSON.stringify(v.pose) === JSON.stringify(input.views[i].pose));
      const rejected = await bundleAdjustGpu(d, input, {
        iterations: 1,
        maxCgIterations: 1,
        cgRelativeTolerance: 1e-12,
      });
      const focalInput = bundleFixture();
      focalInput.views = focalInput.views.map((view) => ({
        ...view,
        camera: { ...view.camera, fx: view.camera.fx * 0.75, fy: view.camera.fy * 0.75 },
      }));
      const focalFixed = await bundleAdjustGpu(d, focalInput, { iterations: 0 }),
        focalRefined = await bundleAdjustGpu(d, focalInput, {
          iterations: 40,
          refineSharedFocal: true,
        });
      return {
        outputs,
        local: { fixed, initial: r.diagnostics.initialCost, final: r.diagnostics.finalCost },
        rollback: {
          accepted: rejected.diagnostics.acceptedSteps,
          rejected: rejected.diagnostics.rejectedSteps,
          unchanged: JSON.stringify(rejected.reconstruction) === JSON.stringify(input),
        },
        focalCalibration: {
          fixedCost: focalFixed.diagnostics.initialCost,
          refinedCost: focalRefined.diagnostics.finalCost,
          inputFocalScale: 0.75,
          outputFocalScale:
            focalRefined.reconstruction.views[0].camera.fx / input.views[0].camera.fx,
          dispatches: focalRefined.execution.report.dispatches,
          tier: focalRefined.execution.report.tier,
        },
      };
    } finally {
      GPUBuffer.prototype.mapAsync = oldMap;
      GPUComputePassEncoder.prototype.dispatchWorkgroups = oldDispatch;
      GPUQueue.prototype.submit = oldSubmit;
      await d.dispose();
      gpu.destroy();
    }
  });
  assert.deepEqual(errors, []);
  for (const [i, r] of result.outputs.entries()) {
    assert.equal(r.tier, 'webgpu');
    assert.ok(r.after.cost < r.before.cost * 0.05, JSON.stringify(r));
    assert.ok(Math.abs(r.after.cost - r.diagnostics.finalCost) < 1e-6);
    assert.ok(r.inputPreserved && r.anchorPreserved);
    assert.ok(Math.abs(r.baseline - r.diagnostics.baseline) < 1e-9);
    assert.equal(r.after.observations, 160);
    // The PCG loop runs as chunks of the time budget: each chunk is a submission of its own and
    // ends with the activity flags, which the host reads (8 bytes each) before the next chunk.
    // The final output is the only other map. The first map is the first activity read, after
    // the ten dispatches of the setup plan and the first LM iteration up to initializeBundleCg
    // and its snapshot.
    assert.ok(r.activityReads > 1);
    assert.ok(r.chunks.bundleCg.chunks >= 1);
    assert.equal(r.work.maps, 1 + r.activityReads);
    assert.equal(r.work.firstMapAfter, 10);
    assert.equal(r.work.dispatches, r.reportedDispatches);
    assert.equal(r.work.submissions, r.reportedSubmissions + r.work.maps);
    assert.equal(r.readback, 8 * (12 * 4 + 3 * 40 + 26) + 8 * r.activityReads);
    const settings = [{}, { outlier: true }, { shift: [7, -3, 2] }][i];
    const reference = bundleAdjust(bundleFixture(settings), {
      iterations: 12,
      huberPx: 1,
      ...(i === 2 ? { anchorViewId: 20, scaleViewId: 40 } : {}),
    });
    assert.ok(Math.abs(reference.diagnostics.finalCost - r.after.cost) < 1e-5);
  }
  assert.ok(result.local.fixed && result.local.final < result.local.initial);
  assert.deepEqual(result.rollback, { accepted: 0, rejected: 1, unchanged: true });
  assert.equal(result.focalCalibration.tier, 'webgpu');
  assert.ok(result.focalCalibration.refinedCost < result.focalCalibration.fixedCost * 0.3);
  const focalReferenceInput = bundleFixture();
  focalReferenceInput.views = focalReferenceInput.views.map((view) => ({
    ...view,
    camera: { ...view.camera, fx: view.camera.fx * 0.75, fy: view.camera.fy * 0.75 },
  }));
  const focalReference = bundleAdjust(focalReferenceInput, {
    iterations: 40,
    refineSharedFocal: true,
  });
  const referenceScale =
    focalReference.reconstruction.views[0].camera.fx / bundleFixture().views[0].camera.fx;
  assert.ok(Math.abs(result.focalCalibration.outputFocalScale - 1) < 0.25);
  assert.ok(Math.abs(result.focalCalibration.outputFocalScale - referenceScale) < 1e-6);
  assert.ok(
    Math.abs(result.focalCalibration.refinedCost - focalReference.diagnostics.finalCost) < 1e-5,
  );
  console.log(JSON.stringify(result));
});

test('incremental reconstruction consumes GPU PnP/triangulation/filtering and joint BA', async () => {
  const { result, errors } = await inBrowser(
    async () => {
      const { ShadeDevice, reconstructIncrementalGpu } = await import('/dist/gpu.js');
      const { mappingFixture } = await import('/tests/typeshade/geometry-cases.mjs');
      const d = await ShadeDevice.open(),
        data = mappingFixture();
      try {
        console.debug('GPU incremental mapper');
        const start = performance.now();
        const r = await reconstructIncrementalGpu(d, data.images, data.pairs, {
          ransacIterations: 128,
          bundleIterations: 2,
          finalBundleIterations: 4,
        });
        const elapsedMs = performance.now() - start;
        const totals = new Map();
        for (const { operation, execution } of r.execution.stages) {
          const report = execution.report;
          const total = totals.get(operation) ?? {
            operation,
            calls: 0,
            tier: report.tier,
            dispatches: 0,
            submissions: 0,
            uploadBytes: 0,
            readbackBytes: 0,
            reportGroups: 0,
            fusedGroups: 0,
          };
          total.calls++;
          total.dispatches += report.dispatches;
          total.submissions += report.submissions;
          total.uploadBytes += execution.applicationUploadBytes;
          total.readbackBytes += execution.applicationReadbackBytes;
          total.reportGroups += report.groups.length;
          total.fusedGroups += report.groups.filter((group) => group.fused).length;
          totals.set(operation, total);
        }
        return {
          views: r.reconstruction.views.length,
          points: r.reconstruction.landmarks.length,
          unregistered: r.unregistered,
          error: r.diagnostics.meanReprojectionErrorPx,
          backend: r.execution.backend,
          elapsedMs,
          cpuStages: r.execution.cpuStages,
          gpuBA: r.execution.gpuBundleAdjustments,
          stages: [...totals.values()],
        };
      } finally {
        await d.dispose();
      }
    },
    undefined,
    { timeoutMs: 180000 },
  );
  assert.deepEqual(errors, []);
  assert.equal(result.views, 4);
  assert.equal(result.points, 40);
  assert.deepEqual(result.unregistered, []);
  assert.ok(result.error < 1e-6, JSON.stringify(result));
  assert.equal(result.backend, 'webgpu');
  assert.ok(result.gpuBA > 0);
  assert.ok(!result.cpuStages.includes('joint bundle adjustment'));
  assert.equal(result.stages.find((s) => s.operation === 'pnp')?.calls, 2);
  assert.equal(result.stages.find((s) => s.operation === 'pairVerification')?.calls, 6);
  assert.ok(result.stages.every((s) => s.tier === 'webgpu' && s.dispatches > 0));
  console.log(JSON.stringify(result));
});

test('GPU FAST/BRIEF extraction and mutual Hamming matching recover frame translation', async () => {
  const { result, errors } = await inBrowser(async () => {
    const { ShadeDevice, extractFeaturesGpu, matchFeaturesGpu } = await import('/dist/gpu.js');
    const width = 192,
      height = 160,
      pixels = new Uint8Array(width * height * 4),
      moved = new Uint8Array(width * height * 4);
    let seed = 0x1234abcd;
    const random = () => {
      seed ^= seed << 13;
      seed ^= seed >>> 17;
      seed ^= seed << 5;
      return seed >>> 0;
    };
    const gray = new Uint8Array(width * height);
    for (let i = 0; i < gray.length; i++) gray[i] = random() & 255;
    for (let y = 0; y < height; y++)
      for (let x = 0; x < width; x++) {
        const at = (y * width + x) * 4,
          a = gray[y * width + x],
          sx = x - 5,
          sy = y - 3,
          b = sx >= 0 && sy >= 0 ? gray[sy * width + sx] : 0;
        for (let c = 0; c < 3; c++) {
          pixels[at + c] = a;
          moved[at + c] = b;
        }
        pixels[at + 3] = 255;
        moved[at + 3] = 255;
      }
    const d = await ShadeDevice.open();
    try {
      const a = await extractFeaturesGpu(d, pixels, width, height),
        b = await extractFeaturesGpu(d, moved, width, height),
        p = await matchFeaturesGpu(d, a, b);
      const dx = [],
        dy = [];
      for (let i = 0; i < p.matches.length; i += 2) {
        const x = p.matches[i],
          y = p.matches[i + 1];
        dx.push(b.keypoints[y * 2] - a.keypoints[x * 2]);
        dy.push(b.keypoints[y * 2 + 1] - a.keypoints[x * 2 + 1]);
      }
      const median = (v) => v.sort((x, y) => x - y)[Math.floor(v.length / 2)];
      const result = {
        tier: a.report.tier,
        features: a.scores.length,
        matches: p.matches.length / 2,
        dx: median(dx),
        dy: median(dy),
        dispatches: a.report.dispatches + p.report.dispatches,
        submissions: a.report.submissions + p.report.submissions,
      };
      await d.dispose();
      return { ...result, live: d.statistics.liveBuffers };
    } finally {
      await d.dispose();
    }
  });
  assert.deepEqual(errors, []);
  assert.equal(result.tier, 'webgpu');
  assert.ok(result.features > 20 && result.matches >= 20, JSON.stringify(result));
  assert.equal(result.dx, 5);
  assert.equal(result.dy, 3);
  assert.ok(result.dispatches > 0 && result.submissions > 0);
  assert.equal(result.live, 0);
  console.log(JSON.stringify(result));
});

test('robust track triangulation runs direct SVD, outlier selection and point LM on WebGPU', async () => {
  const { result, errors } = await inBrowser(async () => {
    const { ShadeDevice, triangulateTracksGpu } = await import('/dist/gpu.js');
    const { triangulationFixture } = await import('/tests/typeshade/geometry-cases.mjs');
    const data = triangulationFixture(),
      d = await ShadeDevice.open();
    try {
      console.debug('GPU triangulation');
      const r = await triangulateTracksGpu(d, data.tracks);
      return {
        status: [...r.status],
        execution: r.execution,
        points: r.tracks.map((p) =>
          p
            ? {
                xyz: [...p.xyz],
                inliers: [...p.inliers],
                angle: p.maxAngleDeg,
                error: p.meanErrorPx,
              }
            : null,
        ),
        truth: data.points,
        count: data.count,
      };
    } finally {
      await d.dispose();
    }
  });
  assert.deepEqual(errors, []);
  for (let i = 0; i < result.count; i++) {
    const p = result.points[i];
    assert.ok(p, `track ${i}, status ${result.status[i]}`);
    assert.ok(Math.hypot(...p.xyz.map((v, j) => v - result.truth[i][j])) < 1e-7, `track ${i}`);
    assert.deepEqual(p.inliers, i % 2 === 0 ? [1, 1, 0, 1, 1] : [1, 1, 1, 1, 1]);
    assert.ok(p.error < 1e-7);
    assert.ok(p.angle > 5);
  }
  assert.deepEqual(result.points.slice(result.count), [null, null, null]);
  assert.equal(result.execution.report.tier, 'webgpu');
  // `prepareTriangulation`, then `triangulateTracks` as chunks of tracks under the time budget,
  // one dispatch and one plan each. A fresh device's first chunk is the least, one workgroup of
  // 16 tracks, so the 36 tracks take at least two chunks; how many more depends on the rate the
  // first chunk measured (at most three, all timed). Before the first chunk the fresh device
  // runs the warm-up, `triangulateTracks` over zero workgroups in a plan of its own, so that the
  // timed chunk excludes the creation of its pipeline: one more dispatch and one more submission.
  const { report } = result.execution,
    chunks = report.chunks?.triangulateTracks?.chunks;
  assert.ok(chunks >= 2, JSON.stringify(report.chunks));
  assert.deepEqual(
    report.groups.flatMap((group) => group.calls),
    ['prepareTriangulation', ...Array.from({ length: 1 + chunks }, () => 'triangulateTracks')],
  );
  assert.equal(report.dispatches, 2 + chunks);
  assert.equal(report.submissions, 2 + chunks);
  assert.equal(result.status[result.count + 2], 4);
  console.log(
    JSON.stringify({
      triangulated: result.count,
      rejected: 3,
      dispatches: report.dispatches,
      chunks,
    }),
  );
});

test('PnP sampling, cooperative scores/normals, device stopping and LM read only the scheduling word', async () => {
  const { result, errors } = await inBrowser(async () => {
    const { ShadeDevice, estimatePosePnPGpu } = await import('/dist/gpu.js');
    const { poseFixture, poseErrors } = await import('/tests/typeshade/geometry-cases.mjs');
    const adapter = await navigator.gpu.requestAdapter(),
      gpu = await adapter.requestDevice();
    const d = await ShadeDevice.open({ gpuDevice: gpu }),
      originalMap = GPUBuffer.prototype.mapAsync,
      originalDispatch = GPUComputePassEncoder.prototype.dispatchWorkgroups,
      originalIndirect = GPUComputePassEncoder.prototype.dispatchWorkgroupsIndirect,
      originalSubmit = GPUQueue.prototype.submit;
    let work = { maps: 0, dispatches: 0, submissions: 0, firstMapAfterDispatches: 0 };
    GPUBuffer.prototype.mapAsync = function (...args) {
      if (!work.maps) work.firstMapAfterDispatches = work.dispatches;
      work.maps++;
      return originalMap.apply(this, args);
    };
    GPUComputePassEncoder.prototype.dispatchWorkgroups = function (...args) {
      work.dispatches++;
      return originalDispatch.apply(this, args);
    };
    // The entries of a search batch dispatch with the counts the GPU wrote (TypeShade change
    // 0075), and the report counts each such dispatch once.
    GPUComputePassEncoder.prototype.dispatchWorkgroupsIndirect = function (...args) {
      work.dispatches++;
      return originalIndirect.apply(this, args);
    };
    GPUQueue.prototype.submit = function (...args) {
      work.submissions++;
      return originalSubmit.apply(this, args);
    };
    const outputs = [];
    try {
      for (const settings of [
        {},
        { count: 41, planar: true, noise: 0, outliers: 0 },
        { count: 4, noise: 0, outliers: 0 },
      ]) {
        const data = poseFixture(settings);
        work = { maps: 0, dispatches: 0, submissions: 0, firstMapAfterDispatches: 0 };
        console.debug('GPU PnP ' + JSON.stringify(settings));
        const r = await estimatePosePnPGpu(d, data.world, data.pixels, data.camera, {
          iterations: 128,
          seed: 91,
          refinementIterations: 12,
        });
        const calls = r.execution.report.groups.flatMap((group) => group.calls);
        outputs.push({
          metric: poseErrors(r, data),
          n: data.pixels.length / 2,
          // The scheduling words the search reads, one after every fourth batch, and the
          // batches it submits, one `planPnPBatch` each.
          controlWords: calls.filter((call) => call === 'snapshotPnPBudget').length,
          searchBatches: calls.filter((call) => call === 'planPnPBatch').length,
          batches: r.hypothesisBatches,
          readback: r.execution.applicationReadbackBytes,
          expected: data.expectedInliers,
          work: { ...work },
          execution: r.execution,
          accepted: r.refinementAcceptedSteps,
          settings,
        });
      }
      return outputs;
    } finally {
      GPUBuffer.prototype.mapAsync = originalMap;
      GPUComputePassEncoder.prototype.dispatchWorkgroups = originalDispatch;
      GPUComputePassEncoder.prototype.dispatchWorkgroupsIndirect = originalIndirect;
      GPUQueue.prototype.submit = originalSubmit;
      await d.dispose();
      gpu.destroy();
    }
  });
  assert.deepEqual(errors, []);
  for (const r of result) {
    const e = r.metric;
    assert.ok(e.inliers >= r.expected, JSON.stringify(e));
    assert.ok(e.rotationFrobenius < 0.005, JSON.stringify(e));
    assert.ok(e.translation < 0.03, JSON.stringify(e));
    if (r.settings.noise === 0) {
      assert.ok(e.translation < 1e-6, JSON.stringify(e));
      assert.ok(e.errorPx < 1e-7, JSON.stringify(e));
    }
    assert.equal(r.execution.report.tier, 'webgpu');
    assert.deepEqual(r.execution.cpuStages, []);
    // The three final readbacks (pose and summary, flags, counters) and one scheduling word
    // after every fourth batch of the search are the only maps. Each scoring of a batch is a
    // plan of its own (key `scorePnP`), and each map a submission of its own.
    assert.ok(r.searchBatches > 0 && 4 * r.controlWords <= r.searchBatches);
    assert.ok(r.batches >= 1 && r.batches <= r.searchBatches);
    assert.equal(r.work.maps, 3 + r.controlWords);
    assert.equal(r.work.submissions, r.execution.report.submissions + r.work.maps);
    assert.ok(r.execution.report.submissions >= 1 + 2 * r.searchBatches);
    assert.equal(r.work.dispatches, r.execution.report.dispatches);
    if (r.controlWords === 0) assert.equal(r.work.firstMapAfterDispatches, r.work.dispatches);
    else assert.ok(r.work.firstMapAfterDispatches < r.work.dispatches);
    assert.equal(r.readback, 164 + 4 * r.n + 4 * r.controlWords);
    assert.ok(e.trials >= 64 && e.trials <= 128);
  }
  console.log(
    JSON.stringify(
      result.map((r) => ({
        metric: r.metric,
        work: r.work,
        accepted: r.accepted,
        batches: r.batches,
        searchBatches: r.searchBatches,
        controlWords: r.controlWords,
      })),
    ),
  );
});

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

test('resident WebGPU pair verification eliminates intermediate maps and preserves composed results', async () => {
  const { result, errors } = await inBrowser(async () => {
    const {
      ShadeDevice,
      verifyImagePairGpu,
      undistortCorrespondencesGpu,
      estimateHomographyGpu,
      reconstructTwoViewGpu,
    } = await import('/dist/gpu.js');
    const { radialPairFixture, compactedPairFixture } =
      await import('/tests/typeshade/initialization-cases.mjs');
    const adapter = await navigator.gpu.requestAdapter(),
      gpu = await adapter.requestDevice(),
      d = await ShadeDevice.open({ gpuDevice: gpu }),
      oldMap = GPUBuffer.prototype.mapAsync,
      oldDispatch = GPUComputePassEncoder.prototype.dispatchWorkgroups,
      oldIndirect = GPUComputePassEncoder.prototype.dispatchWorkgroupsIndirect,
      oldSubmit = GPUQueue.prototype.submit;
    let work = { maps: 0, dispatches: 0, submissions: 0, firstMapAfter: 0 };
    GPUBuffer.prototype.mapAsync = function (...args) {
      if (!work.maps) work.firstMapAfter = work.dispatches;
      work.maps++;
      return oldMap.apply(this, args);
    };
    GPUComputePassEncoder.prototype.dispatchWorkgroups = function (...args) {
      work.dispatches++;
      return oldDispatch.apply(this, args);
    };
    // The batches of model generation dispatch with the counts the GPU wrote (TypeShade change
    // 0075), and the report counts each such dispatch once.
    GPUComputePassEncoder.prototype.dispatchWorkgroupsIndirect = function (...args) {
      work.dispatches++;
      return oldIndirect.apply(this, args);
    };
    GPUQueue.prototype.submit = function (...args) {
      work.submissions++;
      return oldSubmit.apply(this, args);
    };
    const options = {
      iterations: 256,
      threshold: 0.001,
      seed: 71,
      homographyIterations: 128,
      refinementIterations: 10,
    };
    const composed = async (f) => {
      const p = await undistortCorrespondencesGpu(d, f.pixels, f.first, f.second),
        H = await estimateHomographyGpu(d, p.pixels, {
          iterations: options.homographyIterations,
          thresholdPx: 4,
          seed: options.seed,
        }),
        r = await reconstructTwoViewGpu(
          d,
          p.pixels,
          { ...f.first, k1: 0, k2: 0 },
          { ...f.second, k1: 0, k2: 0 },
          options,
        );
      return {
        rows: [...p.indices].filter((_, i) => r.inliers[i]),
        H: H.inlierCount,
        invertible: p.indices.length,
        geometry: r,
      };
    };
    const difference = (a, b) => Math.max(...a.map((v, i) => Math.abs(v - b[i])));
    try {
      const f = radialPairFixture({ n: 321 }),
        holes = compactedPairFixture(),
        outputs = [];
      for (const input of [f, holes]) {
        // Warm both paths so uploads measure input data rather than pool initialization.
        await composed(input);
        await verifyImagePairGpu(d, input.pixels, input.first, input.second, options);
        work = { maps: 0, dispatches: 0, submissions: 0, firstMapAfter: 0 };
        const beforeOld = d.statistics,
          reportCount = d.reports.length,
          old = await composed(input),
          oldWork = { ...work },
          afterOld = d.statistics,
          oldReports = d.reports.slice(reportCount);
        work = { maps: 0, dispatches: 0, submissions: 0, firstMapAfter: 0 };
        const before = d.statistics,
          pending = verifyImagePairGpu(d, input.pixels, input.first, input.second, options),
          r = await pending,
          after = d.statistics;
        outputs.push({
          n: input.pixels.length / 4,
          invertible: r.invertibleCount,
          H: r.homographyInlierCount,
          expectedH: old.H,
          rowsMatch: JSON.stringify([...r.inlierRows]) === JSON.stringify(old.rows),
          inliers: r.geometry.inlierCount,
          positive: r.geometry.positiveDepthCount,
          expectedPositive: old.geometry.positiveDepthCount,
          rotationDifference: difference(r.geometry.R, old.geometry.R),
          translationDifference: difference(r.geometry.t, old.geometry.t),
          errorDifference: Math.abs(
            r.geometry.meanReprojectionErrorPx - old.geometry.meanReprojectionErrorPx,
          ),
          execution: { ...r.execution, report: { ...r.execution.report, groups: [] } },
          // The scheduling words the chunked model generation reads, one after every fourth
          // batch of a search, and the batches, one `planPairModelBatch` each.
          controlWords: r.execution.report.groups
            .flatMap((group) => group.calls)
            .filter((call) => call === 'snapshotPairModelBudget').length,
          batches: r.execution.report.groups
            .flatMap((group) => group.calls)
            .filter((call) => call === 'planPairModelBatch').length,
          work: { ...work },
          allocations: after.allocations - before.allocations,
          upload: after.applicationUploadBytes - before.applicationUploadBytes,
          read: after.applicationReadbackBytes - before.applicationReadbackBytes,
          old: {
            work: oldWork,
            controlWords: oldReports
              .flatMap((report) => report.groups)
              .flatMap((group) => group.calls)
              .filter((call) => call === 'snapshotPairModelBudget').length,
            batches: oldReports
              .flatMap((report) => report.groups)
              .flatMap((group) => group.calls)
              .filter((call) => call === 'planPairModelBatch').length,
            upload: afterOld.applicationUploadBytes - beforeOld.applicationUploadBytes,
            read: afterOld.applicationReadbackBytes - beforeOld.applicationReadbackBytes,
            submissions: oldReports.reduce((n, r) => n + r.submissions, 0),
          },
        });
        const snapshot = [...r.inlierRows];
        await verifyImagePairGpu(d, input.pixels, input.first, input.second, options);
        outputs.at(-1).snapshotPreserved =
          JSON.stringify(snapshot) === JSON.stringify([...r.inlierRows]);
      }
      return outputs;
    } finally {
      GPUBuffer.prototype.mapAsync = oldMap;
      GPUComputePassEncoder.prototype.dispatchWorkgroups = oldDispatch;
      GPUComputePassEncoder.prototype.dispatchWorkgroupsIndirect = oldIndirect;
      GPUQueue.prototype.submit = oldSubmit;
      await d.dispose();
      gpu.destroy();
    }
  });
  assert.deepEqual(errors, []);
  for (const r of result) {
    assert.ok(r.rowsMatch && r.snapshotPreserved);
    assert.equal(r.H, r.expectedH);
    assert.equal(r.positive, r.expectedPositive);
    assert.ok(
      r.rotationDifference < 1e-6 && r.translationDifference < 1e-6 && r.errorDifference < 1e-6,
    );
    // The model generation runs as chunks under the time budget: each chunk is a submission.
    // The GPU decides each batch's work (TypeShade change 0075), and the scheduling word is read
    // after every fourth batch of a search, so at most once per four batches. The three terminal
    // snapshots are the only other maps, and the first map comes after the first chunk, not
    // before any dispatch.
    assert.ok(r.batches > 0 && 4 * r.controlWords <= r.batches);
    assert.equal(r.work.maps, 3 + r.controlWords);
    assert.ok(r.work.firstMapAfter > 0 && r.work.firstMapAfter <= r.execution.report.dispatches);
    assert.equal(r.work.dispatches, r.execution.report.dispatches);
    assert.ok(r.execution.report.submissions > 1);
    assert.ok(r.execution.report.chunks.scorePairModels.chunks >= 1);
    assert.equal(r.work.submissions, r.execution.report.submissions + r.work.maps);
    assert.equal(r.read, 624 + 4 * r.n + 4 * r.controlWords);
    assert.equal(r.upload, 32 * r.n);
    assert.equal(r.allocations, 0);
    assert.deepEqual(r.execution.cpuStages, []);
    // The composed path's homography and two-view searches run as chunks too: its ten
    // readbacks and three plans gain the scheduling words and the chunk submissions.
    assert.ok(r.old.batches > 0 && 4 * r.old.controlWords <= r.old.batches);
    assert.equal(r.old.work.maps, 10 + r.old.controlWords);
    assert.equal(r.old.work.submissions, r.old.submissions + r.old.work.maps);
    assert.ok(r.old.submissions > 3);
    assert.equal(r.old.read, 808 + 36 * r.n + 36 * r.invertible + 4 * r.old.controlWords);
    assert.equal(r.old.upload, 32 * r.n + 64 * r.invertible);
  }
  console.log(JSON.stringify(result));
});

test('WebGPU stable compaction preserves tails and rejects empty/folded/planar pairs', async () => {
  const { result, errors } = await inBrowser(async () => {
    const { ShadeDevice, verifyImagePairGpu } = await import('/dist/gpu.js');
    const { compactedPairFixture, radialPairFixture } =
      await import('/tests/typeshade/initialization-cases.mjs');
    const d = await ShadeDevice.open(),
      outputs = [],
      options = { iterations: 16, minIterations: 16, homographyIterations: 16, threshold: 1e-7 };
    try {
      for (const n of [0, 3, 63, 64, 65, 127, 128, 129, 197, 257, 65]) {
        const f = compactedPairFixture(n, { noise: 0, outliers: false }),
          r = await verifyImagePairGpu(d, f.pixels, f.first, f.second, options);
        outputs.push({
          n,
          expected: f.validRows,
          rows: [...r.inlierRows],
          count: r.invertibleCount,
          positive: r.geometry?.positiveDepthCount,
          reason: r.reason,
          error: r.geometry?.meanReprojectionErrorPx,
        });
      }
      const f = compactedPairFixture(65, { allInvalid: true }),
        invalid = await verifyImagePairGpu(d, f.pixels, f.first, f.second, options),
        p = radialPairFixture({ planar: true, noise: 0, outliers: false }),
        planar = await verifyImagePairGpu(d, p.pixels, p.first, p.second, options);
      return {
        outputs,
        invalid: {
          count: invalid.invertibleCount,
          H: invalid.homographyInlierCount,
          rows: invalid.inlierRows.length,
          reason: invalid.reason,
        },
        planar: {
          H: planar.homographyInlierCount,
          rows: planar.inlierRows.length,
          reason: planar.reason,
        },
      };
    } finally {
      await d.dispose();
    }
  });
  assert.deepEqual(errors, []);
  for (const r of result.outputs) {
    assert.equal(r.count, r.expected.length, `n=${r.n}`);
    if (r.count >= 8) {
      assert.equal(r.reason, undefined, `n=${r.n}`);
      assert.deepEqual(r.rows, r.expected);
      assert.equal(r.positive, r.count);
      assert.ok(r.error < 1e-5);
    } else {
      assert.match(r.reason, /Insufficient invertible/);
      assert.deepEqual(r.rows, []);
    }
  }
  assert.equal(result.invalid.count, 0);
  assert.equal(result.invalid.H, 0);
  assert.equal(result.invalid.rows, 0);
  assert.ok(result.invalid.reason);
  assert.equal(result.planar.H, 97);
  assert.equal(result.planar.rows, 0);
  assert.ok(result.planar.reason);
});

test('resident WebGPU two-view initialization uses one plan and only final readbacks', async () => {
  const { result, errors } = await inBrowser(async () => {
    const { ShadeDevice, reconstructTwoViewGpu, estimateEssentialRansacGpu } =
      await import('/dist/gpu.js');
    const { reconstructionFixture, poseMetrics } = await import('/tests/typeshade/cases.mjs');
    const { radialPairFixture, recoveredMetrics } =
      await import('/tests/typeshade/initialization-cases.mjs');
    const adapter = await navigator.gpu.requestAdapter(),
      gpu = await adapter.requestDevice(),
      d = await ShadeDevice.open({ gpuDevice: gpu }),
      data = reconstructionFixture(),
      oldMap = GPUBuffer.prototype.mapAsync,
      oldDispatch = GPUComputePassEncoder.prototype.dispatchWorkgroups,
      oldIndirect = GPUComputePassEncoder.prototype.dispatchWorkgroupsIndirect,
      oldSubmit = GPUQueue.prototype.submit;
    let work = { maps: 0, dispatches: 0, submissions: 0, firstMapAfter: 0 };
    GPUBuffer.prototype.mapAsync = function (...args) {
      if (!work.maps) work.firstMapAfter = work.dispatches;
      work.maps++;
      return oldMap.apply(this, args);
    };
    GPUComputePassEncoder.prototype.dispatchWorkgroups = function (...args) {
      work.dispatches++;
      return oldDispatch.apply(this, args);
    };
    // The batches of model generation dispatch with the counts the GPU wrote (TypeShade change
    // 0075), and the report counts each such dispatch once.
    GPUComputePassEncoder.prototype.dispatchWorkgroupsIndirect = function (...args) {
      work.dispatches++;
      return oldIndirect.apply(this, args);
    };
    GPUQueue.prototype.submit = function (...args) {
      work.submissions++;
      return oldSubmit.apply(this, args);
    };
    const summarize = (r) => ({
      work: { ...work },
      tier: r.execution.report.tier,
      dispatches: r.execution.report.dispatches,
      submissions: r.execution.report.submissions,
      // The scheduling words the chunked model generation reads, one after every fourth batch
      // of a search, and the batches, one `planPairModelBatch` each.
      controlWords: r.execution.report.groups
        .flatMap((group) => group.calls)
        .filter((call) => call === 'snapshotPairModelBudget').length,
      modelBatches: r.execution.report.groups
        .flatMap((group) => group.calls)
        .filter((call) => call === 'planPairModelBatch').length,
      readback: r.execution.applicationReadbackBytes,
      cpuStages: r.execution.cpuStages,
      batches: r.execution.hypothesisBatches,
    });
    try {
      const r = await reconstructTwoViewGpu(d, data.pixels, data.first, data.second, {
          iterations: 500,
          minIterations: 32,
          threshold: 0.001,
          batchSize: 16,
          seed: 71,
        }),
        metrics = poseMetrics(r, data),
        ordinary = {
          ...summarize(r),
          valid: metrics.valid,
          rotation: metrics.rotationErrorDeg,
          translation: metrics.translationCosine,
          error: metrics.errorPx,
          diagnostics: r.diagnostics,
        };
      work = { maps: 0, dispatches: 0, submissions: 0, firstMapAfter: 0 };
      const f = radialPairFixture(),
        radial = await reconstructTwoViewGpu(d, f.pixels, f.first, f.second, {
          iterations: 500,
          threshold: 0.001,
          seed: 71,
        }),
        radialOutput = {
          ...summarize(radial),
          ...recoveredMetrics(radial, f),
          diagnostics: radial.diagnostics,
          reportedMean: radial.meanReprojectionErrorPx,
        };
      work = { maps: 0, dispatches: 0, submissions: 0, firstMapAfter: 0 };
      const regression = await estimateEssentialRansacGpu(d, data.pixels, data.first, data.second, {
        iterations: 32,
        minIterations: 1,
        threshold: 0.001,
        batchSize: 16,
        seed: 71,
      });
      return {
        ordinary,
        radial: radialOutput,
        regression: {
          ...summarize(regression),
          trials: regression.ransacIterations,
          count: regression.inlierCount,
        },
        reports: d.reports.length,
      };
    } finally {
      GPUBuffer.prototype.mapAsync = oldMap;
      GPUComputePassEncoder.prototype.dispatchWorkgroups = oldDispatch;
      GPUComputePassEncoder.prototype.dispatchWorkgroupsIndirect = oldIndirect;
      GPUQueue.prototype.submit = oldSubmit;
      await d.dispose();
      gpu.destroy();
    }
  });
  assert.deepEqual(errors, []);
  const r = result.ordinary;
  assert.ok(
    r.valid >= 250 && r.rotation < 0.1 && r.translation > 0.999 && r.error < 0.15,
    JSON.stringify(r),
  );
  assert.ok(r.batches > 1);
  const radial = result.radial;
  assert.ok(
    radial.rotationErrorDeg < 0.1 && radial.translationCosine > 0.999 && radial.count >= 75,
    JSON.stringify(radial),
  );
  assert.ok(radial.minDepth > 0 && radial.meanError < 0.15 && Math.abs(radial.tNorm - 1) < 1e-8);
  assert.ok(Math.abs(radial.meanError - radial.reportedMean) < 1e-7);
  assert.ok(
    radial.diagnostics.meanReprojectionErrorAfterPx <=
      radial.diagnostics.meanReprojectionErrorBeforePx,
  );
  for (const [entry, n, maps] of [
    [r, 321, 5],
    [radial, 97, 5],
    [result.regression, 321, 3],
  ]) {
    assert.equal(entry.tier, 'webgpu');
    assert.deepEqual(entry.cpuStages, []);
    // The chunked model generation reads its scheduling word after every fourth batch of a
    // search (TypeShade change 0075); the final readbacks are the only other maps, and each
    // chunk is a submission of its own.
    assert.ok(entry.modelBatches > 0 && 4 * entry.controlWords <= entry.modelBatches);
    assert.equal(entry.work.maps, maps + entry.controlWords);
    assert.ok(entry.work.firstMapAfter > 0 && entry.work.firstMapAfter <= entry.dispatches);
    assert.equal(entry.work.dispatches, entry.dispatches);
    assert.equal(entry.work.submissions, entry.submissions + entry.work.maps);
    assert.ok(entry.submissions > 1);
    assert.equal(
      entry.readback,
      (maps === 5 ? 608 + 32 * n : 200 + 4 * n) + 4 * entry.controlWords,
    );
  }
  assert.equal(result.regression.trials, 32);
  assert.ok(result.regression.count < 50);
  assert.equal(result.reports, 3);
  console.log(JSON.stringify(result));
});

test('WebGPU homography and radial inversion preserve independently computed geometry', async () => {
  const { result, errors } = await inBrowser(async () => {
    const {
      ShadeDevice,
      estimateHomographyGpu,
      undistortCorrespondencesGpu,
      reconstructTwoViewGpu,
    } = await import('/dist/gpu.js');
    const { homographyFixture, symmetricTransferErrors, radialPairFixture } =
      await import('/tests/typeshade/initialization-cases.mjs');
    const d = await ShadeDevice.open(),
      outputs = [];
    try {
      for (const settings of [{}, { offset: 10000, scale: 3 }, { scale: 0.001 }]) {
        const f = homographyFixture(settings),
          threshold = 0.8 * f.scale,
          r = await estimateHomographyGpu(d, f.pixels, { iterations: 256, thresholdPx: threshold });
        outputs.push({
          mask: [...r.inliers],
          expected: [...f.clean],
          errors: symmetricTransferErrors(r.H, f.pixels),
          threshold,
          backend: r.execution.backend,
          cpuStages: r.execution.cpuStages,
        });
      }
      const bad = homographyFixture({ collinear: true }),
        rejected = await estimateHomographyGpu(d, bad.pixels, { iterations: 32 }),
        k = { fx: 100, fy: 120, cx: 50, cy: 40, k1: -0.5 },
        pixels = Float64Array.from([
          50, 40, 52, 43, 250, 40, 50, 40, 65, 55, 72, 61, 50, 40, 270, 40,
        ]),
        undistorted = await undistortCorrespondencesGpu(d, pixels, k, k),
        failures = [];
      for (const f of [
        radialPairFixture({ baseline: 1e-4, noise: 0, outliers: false }),
        radialPairFixture({ planar: true, noise: 0, outliers: false }),
      ]) {
        try {
          await reconstructTwoViewGpu(d, f.pixels, f.first, f.second, {
            iterations: 64,
            refinementIterations: 0,
          });
          failures.push('accepted');
        } catch (e) {
          failures.push(e.name + ': ' + e.message);
        }
      }
      return {
        outputs,
        rankRejected: rejected === undefined,
        indices: [...undistorted.indices],
        pixels: [...undistorted.pixels],
        failures,
      };
    } finally {
      await d.dispose();
    }
  });
  assert.deepEqual(errors, []);
  for (const r of result.outputs) {
    assert.equal(r.backend, 'webgpu');
    assert.deepEqual(r.cpuStages, []);
    assert.deepEqual(r.mask, r.expected);
    assert.deepEqual(
      r.errors.map((e) => Number(e <= r.threshold * r.threshold)),
      r.mask,
    );
  }
  assert.ok(result.rankRejected);
  assert.deepEqual(result.indices, [0, 2]);
  assert.ok(result.failures.every((e) => e.startsWith('GeometryEstimationError:')));
  console.log(JSON.stringify({ rankRejected: result.rankRejected, failures: result.failures }));
});

test('WebGPU relative-pose branches recover twelve known general motions and structures', async () => {
  const { result, errors } = await inBrowser(async () => {
    const { ShadeDevice, reconstructTwoViewGpu } = await import('/dist/gpu.js');
    const { relativePoseCases, recoveredMetrics } =
      await import('/tests/typeshade/initialization-cases.mjs');
    const d = await ShadeDevice.open(),
      out = [];
    try {
      for (const f of relativePoseCases()) {
        const r = await reconstructTwoViewGpu(d, f.pixels, f.first, f.second, {
            iterations: 16,
            minIterations: 16,
            refinementIterations: 0,
            threshold: 1e-7,
            minParallaxDeg: 0.001,
          }),
          scale = Math.hypot(...f.t);
        out.push({
          id: f.id,
          ...recoveredMetrics(r, f),
          structureError: Math.max(...r.points.map((x, i) => Math.abs(x - f.world[i] / scale))),
        });
      }
      return out;
    } finally {
      await d.dispose();
    }
  });
  assert.deepEqual(errors, []);
  for (const r of result) {
    assert.equal(r.count, 24, JSON.stringify(r));
    assert.ok(
      r.rotationErrorDeg < 1e-4 &&
        r.translationCosine > 1 - 1e-10 &&
        r.maxError < 1e-4 &&
        r.structureError < 1e-5,
      JSON.stringify(r),
    );
  }
});

test('bounded geometry fences every dispatch and preserves fused QR/SVD, PnP and BA results', async () => {
  const { result, errors } = await inBrowser(
    async () => {
      const { ShadeDevice, verifyImagePairGpu, estimatePosePnPGpu, bundleAdjustGpu } =
        await import('/dist/gpu.js');
      const { radialPairFixture } = await import('/tests/typeshade/initialization-cases.mjs');
      const { poseFixture, bundleFixture } = await import('/tests/typeshade/geometry-cases.mjs');
      const pair = radialPairFixture({ n: 65 });
      const pose = poseFixture();
      const bundle = bundleFixture();
      const adapter = await navigator.gpu.requestAdapter();
      const gpu = await adapter.requestDevice();
      const options = {
        iterations: 32,
        minIterations: 32,
        homographyIterations: 32,
        threshold: 0.001,
        seed: 71,
        refinementIterations: 4,
      };
      const output = [];
      let pendingDispatches = 0,
        maxDispatches = 0,
        fences = 0;
      const originalDispatch = GPUComputePassEncoder.prototype.dispatchWorkgroups;
      const originalIndirect = GPUComputePassEncoder.prototype.dispatchWorkgroupsIndirect;
      const originalSubmit = GPUQueue.prototype.submit;
      const originalFence = GPUQueue.prototype.onSubmittedWorkDone;
      GPUComputePassEncoder.prototype.dispatchWorkgroups = function (...args) {
        pendingDispatches++;
        return originalDispatch.apply(this, args);
      };
      GPUComputePassEncoder.prototype.dispatchWorkgroupsIndirect = function (...args) {
        pendingDispatches++;
        return originalIndirect.apply(this, args);
      };
      GPUQueue.prototype.submit = function (...args) {
        maxDispatches = Math.max(maxDispatches, pendingDispatches);
        pendingDispatches = 0;
        return originalSubmit.apply(this, args);
      };
      GPUQueue.prototype.onSubmittedWorkDone = function (...args) {
        fences++;
        return originalFence.apply(this, args);
      };
      const events = [];
      try {
        for (const boundedSubmissions of [false, true]) {
          const device = await ShadeDevice.open({
            gpuDevice: gpu,
            maxModelsPerBatch: 8,
            boundedSubmissions,
            onProgress: (event) => events.push(event),
          });
          maxDispatches = 0;
          fences = 0;
          try {
            const pairResult = await verifyImagePairGpu(
              device,
              pair.pixels,
              pair.first,
              pair.second,
              options,
            );
            const poseResult = await estimatePosePnPGpu(
              device,
              pose.world,
              pose.pixels,
              pose.camera,
              { iterations: 32, minIterations: 32, refinementIterations: 2 },
            );
            const bundleResult = await bundleAdjustGpu(device, bundle, {
              iterations: 4,
              maxCgIterations: 32,
            });
            output.push({
              rows: [...pairResult.inlierRows],
              reason: pairResult.reason,
              R: [...pairResult.geometry.R],
              t: [...pairResult.geometry.t],
              pairReadback: pairResult.execution.applicationReadbackBytes,
              // The scheduling words the searches read, and the refinement's activity word that
              // the bounded path's solver checkpoints read.
              pairControlWords: pairResult.execution.report.groups
                .flatMap((group) => group.calls)
                .filter(
                  (call) =>
                    call === 'snapshotPairModelBudget' || call === 'snapshotTwoViewActivity',
                ).length,
              poseR: [...poseResult.pose.R],
              poseT: [...poseResult.pose.t],
              cost: bundleResult.diagnostics.finalCost,
              maxDispatches,
              fences,
            });
          } finally {
            await device.dispose();
          }
        }
      } finally {
        GPUComputePassEncoder.prototype.dispatchWorkgroups = originalDispatch;
        GPUComputePassEncoder.prototype.dispatchWorkgroupsIndirect = originalIndirect;
        GPUQueue.prototype.submit = originalSubmit;
        GPUQueue.prototype.onSubmittedWorkDone = originalFence;
        gpu.destroy();
      }
      return { output, kernels: [...new Set(events.map((e) => e.kernel))] };
    },
    undefined,
    { timeoutMs: 180000 },
  );
  assert.deepEqual(errors, []);
  const [fused, bounded] = result.output;
  assert.deepEqual(bounded.rows, fused.rows);
  assert.equal(bounded.reason, fused.reason);
  for (const key of ['R', 't', 'poseR', 'poseT'])
    bounded[key].forEach((v, i) => assert.ok(Math.abs(v - fused[key][i]) < 1e-6, `${key}[${i}]`));
  assert.ok(Math.abs(bounded.cost - fused.cost) < 1e-6);
  assert.ok(bounded.cost < 1000, 'BA must accept useful steps');
  assert.equal(bounded.maxDispatches, 1);
  assert.ok(bounded.fences > 20);
  assert.ok(fused.maxDispatches > 20);
  // Each search of 32 trials runs in four batches of 8, which reach the requested maximum before
  // the host would read the scheduling word: the GPU decides each batch's work (TypeShade change
  // 0075), so the fused path reads no control word. The bounded path reads the refinement's
  // activity word; the terminal readbacks are the same.
  assert.equal(fused.pairControlWords, 0);
  assert.ok(bounded.pairControlWords > 0);
  assert.equal(
    bounded.pairReadback - 4 * bounded.pairControlWords,
    fused.pairReadback - 4 * fused.pairControlWords,
  );
  for (const kernel of [
    'appendPairModelRows',
    'solvePairModelNull',
    'solveP3p',
    'bundleSchurPoint',
  ])
    assert.ok(result.kernels.includes(kernel), kernel);
  console.log(JSON.stringify(result));
});

test('bounded geometry propagates native device loss and releases resident buffers', async () => {
  const { result, errors } = await inBrowser(async () => {
    const { ShadeDevice, verifyImagePairGpu } = await import('/dist/gpu.js');
    const { radialPairFixture } = await import('/tests/typeshade/initialization-cases.mjs');
    const adapter = await navigator.gpu.requestAdapter();
    const gpu = await adapter.requestDevice();
    let destroyed = false;
    const device = await ShadeDevice.open({
      gpuDevice: gpu,
      boundedSubmissions: true,
      onProgress: (event) => {
        if (!destroyed && event.kernel === 'preparePairModels' && event.phase === 'start') {
          destroyed = true;
          gpu.destroy();
        }
      },
    });
    const input = radialPairFixture({ n: 65 });
    let failure;
    try {
      await verifyImagePairGpu(device, input.pixels, input.first, input.second, {
        iterations: 32,
        homographyIterations: 32,
      });
    } catch (error) {
      failure = String(error);
    } finally {
      await device.dispose();
    }
    return { destroyed, failure, lost: await device.lost, live: device.statistics.liveBuffers };
  });
  assert.deepEqual(errors, []);
  assert.equal(result.destroyed, true);
  assert.ok(result.failure, 'device loss must reject instead of yielding a numerical result');
  assert.equal(result.lost.reason, 'destroyed');
  assert.equal(result.live, 0);
});
