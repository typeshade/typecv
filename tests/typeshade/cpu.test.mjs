import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  ShadeDevice,
  EpipolarSession,
  estimateEssentialRansacGpu,
  reconstructTwoViewGpu,
  solveP3pBatchGpu,
  estimatePosePnPGpu,
  triangulateTracksGpu,
  reconstructIncrementalGpu,
  bundleAdjustGpu,
  estimateHomographyGpu,
  undistortCorrespondencesGpu,
  verifyImagePairGpu,
  extractFeaturesGpu,
  extractFeaturesGpuResident,
  matchFeaturesGpu,
  matchResidentFeaturesGpu,
} from '../../dist/gpu.js';

/**
 * The bytes of the control words the chunked model generation reads back: one `u32` after each
 * `snapshotPairModelBudget` step of the plan, which runs after every fourth batch of a search.
 * `summarizePairModelWork` writes the batch's indirect workgroup counts (TypeShade change 0075),
 * which the host does not read.
 */
function controlWordBytes(report) {
  return (
    4 *
    report.groups
      .flatMap((group) => group.calls)
      .filter((call) => call === 'snapshotPairModelBudget').length
  );
}
import { fixture, reference, reconstructionFixture, poseMetrics } from './cases.mjs';
import { p3p } from '../../dist/geometry/p3p.js';
import { p3pCases, packP3pCases, p3pMetrics, poseDifference } from './p3p-cases.mjs';
import {
  poseFixture,
  poseErrors,
  triangulationFixture,
  mappingFixture,
  bundleFixture,
  bundleMetrics,
} from './geometry-cases.mjs';
import { cameraCenter } from '../../dist/geometry/pose.js';
import { triangulateTrack } from '../../dist/geometry/multiview.js';
import { compareCameraModels } from '../../scripts/geometry-comparison.mjs';
import { unprojectPixel } from '../../dist/geometry/camera.js';
import { fitHomography } from '../../dist/geometry/homography.js';
import {
  homographyFixture,
  symmetricTransferErrors,
  radialPairFixture,
  recoveredMetrics,
  relativePoseCases,
  compactedPairFixture,
} from './initialization-cases.mjs';

test('resident pair verification matches composed GPU stages with stable original row ownership', async () => {
  const d = await ShadeDevice.open({ backend: 'cpu' });
  try {
    const f = compactedPairFixture(),
      options = { iterations: 256, threshold: 0.001, seed: 71, homographyIterations: 128 },
      prepared = await undistortCorrespondencesGpu(d, f.pixels, f.first, f.second),
      H = await estimateHomographyGpu(d, prepared.pixels, {
        iterations: options.homographyIterations,
        thresholdPx: 4,
        seed: options.seed,
      });
    assert.deepEqual([...prepared.indices], f.validRows);
    for (const refinementIterations of [0, 10]) {
      const old = await reconstructTwoViewGpu(
          d,
          prepared.pixels,
          { ...f.first, k1: 0, k2: 0 },
          { ...f.second, k1: 0, k2: 0 },
          { ...options, refinementIterations },
        ),
        r = await verifyImagePairGpu(d, f.pixels, f.first, f.second, {
          ...options,
          refinementIterations,
        });
      assert.equal(r.reason, undefined);
      assert.equal(r.invertibleCount, f.validRows.length);
      assert.equal(r.homographyInlierCount, H.inlierCount);
      assert.equal(r.execution.homographyIterations, H.iterations);
      assert.deepEqual(
        [...r.inlierRows],
        [...prepared.indices].filter((_, i) => old.inliers[i]),
      );
      assert.equal(r.geometry.positiveDepthCount, old.positiveDepthCount);
      assert.equal(r.geometry.diagnostics.refinementApplied, old.diagnostics.refinementApplied);
      assert.ok(Math.abs(r.geometry.meanReprojectionErrorPx - old.meanReprojectionErrorPx) < 1e-6);
      for (const key of ['R', 't'])
        r.geometry[key].forEach((v, i) => assert.ok(Math.abs(v - old[key][i]) < 1e-6));
      assert.deepEqual(r.execution.cpuStages, []);
      assert.equal(
        r.execution.applicationReadbackBytes,
        624 + 4 * (f.pixels.length / 4) + controlWordBytes(r.execution.report),
      );
      assert.equal(r.execution.report.submissions, 0); // Explicit CPU oracle does not submit GPU work.
    }
  } finally {
    await d.dispose();
  }
});

test('resident pair scan handles tails, empty blocks, invalid lenses and buffer reuse', async () => {
  const d = await ShadeDevice.open({ backend: 'cpu' });
  const options = {
    iterations: 16,
    minIterations: 16,
    homographyIterations: 16,
    threshold: 1e-7,
    refinementIterations: 0,
  };
  try {
    for (const n of [0, 1, 7, 63, 64, 65, 127, 128, 129, 197, 257, 65]) {
      const f = compactedPairFixture(n, { noise: 0, outliers: false }),
        pixels = f.pixels.slice(),
        a = { ...f.first },
        controls = { ...options },
        pending = verifyImagePairGpu(d, pixels, a, f.second, controls);
      pixels.fill(0);
      a.fx = 1;
      controls.threshold = 1;
      const r = await pending;
      assert.equal(r.invertibleCount, f.validRows.length, `n=${n}`);
      if (f.validRows.length < 8) {
        assert.equal(r.geometry, undefined);
        assert.match(r.reason, /Insufficient invertible/);
        assert.equal(r.homographyInlierCount, 0);
        assert.equal(r.inlierRows.length, 0);
      } else {
        assert.equal(r.reason, undefined, `n=${n}: ${r.reason}`);
        assert.deepEqual([...r.inlierRows], f.validRows);
        assert.equal(r.geometry.positiveDepthCount, f.validRows.length);
        assert.ok(Math.abs(Math.hypot(...r.geometry.t) - 1) < 1e-9);
        assert.ok(r.geometry.meanReprojectionErrorPx < 1e-5);
      }
      if (!n) {
        assert.equal(r.execution.report.dispatches, 0);
        assert.equal(r.execution.applicationReadbackBytes, 0);
      }
    }
    const invalid = compactedPairFixture(65, { allInvalid: true }),
      rejected = await verifyImagePairGpu(
        d,
        invalid.pixels,
        invalid.first,
        invalid.second,
        options,
      );
    assert.equal(rejected.invertibleCount, 0);
    assert.equal(rejected.homographyInlierCount, 0);
    assert.equal(rejected.inlierRows.length, 0);
    assert.equal(rejected.geometry, undefined);
    const planar = radialPairFixture({ planar: true, noise: 0, outliers: false }),
      r = await verifyImagePairGpu(d, planar.pixels, planar.first, planar.second, options);
    assert.equal(r.homographyInlierCount, 97);
    assert.equal(r.geometry, undefined);
    assert.ok(r.reason);
  } finally {
    await d.dispose();
  }
  await assert.rejects(verifyImagePairGpu(d, new Float64Array(), {}, {}), RangeError);
  const f = radialPairFixture();
  await assert.rejects(
    verifyImagePairGpu(d, new Float64Array(), f.first, f.second),
    /disposed|closed/,
  );
});

test('resident pair failures release every buffer without CPU retry', async () => {
  const f = radialPairFixture(),
    options = { iterations: 16, homographyIterations: 16 };
  let d = await ShadeDevice.open({ backend: 'cpu', workingBytes: 4096 });
  try {
    await assert.rejects(verifyImagePairGpu(d, f.pixels, f.first, f.second, options), /budget/);
    assert.equal(d.statistics.allocations, 0);
  } finally {
    await d.dispose();
  }
  d = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    for (const stage of ['allocateF64', 'plan', 'read']) {
      const original = d[stage];
      let calls = 0;
      d[stage] = function (...args) {
        if (++calls === (stage === 'plan' ? 1 : 2))
          throw new Error('injected resident pair failure');
        return original.apply(this, args);
      };
      await assert.rejects(
        verifyImagePairGpu(d, f.pixels, f.first, f.second, options),
        /injected resident/,
      );
      d[stage] = original;
      assert.equal(d.statistics.liveBuffers, 0, stage);
    }
    const k = { fx: 1e6, fy: 1e6, cx: 0, cy: 0, k1: -1e-9 };
    await assert.rejects(
      verifyImagePairGpu(d, Float64Array.of(999900000, 0, 1, 1), k, k, options),
      /Undistorted pixel/,
    );
    assert.equal(d.statistics.liveBuffers, 0);
    for (const options of [{ threshold: 0 }, { homographyIterations: 0 }, { batchSize: 0 }])
      await assert.rejects(
        verifyImagePairGpu(d, new Float64Array(), f.first, f.second, options),
        RangeError,
      );
  } finally {
    await d.dispose();
  }
});

test('camera-reference evaluation aligns different SfM gauges and rejects unobservable alignment', () => {
  const theta = 0.3,
    c = Math.cos(theta),
    s = Math.sin(theta),
    Q = [c, 0, s, 0, 1, 0, -s, 0, c],
    Qt = [c, 0, -s, 0, 1, 0, s, 0, c],
    scale = 1.7,
    shift = [0.3, -0.6, 1.5],
    centers = [
      [0, 0, 0],
      [1, 0, 0],
      [0, 1, 0],
      [0.2, 0.3, 1],
    ],
    candidate = {
      views: centers.map((C, id) => ({
        id,
        pose: { R: [1, 0, 0, 0, 1, 0, 0, 0, 1], t: C.map((v) => -v) },
      })),
    },
    reference = {
      views: centers.map((C, id) => ({
        id,
        pose: {
          R: Qt,
          t: C.map(
            (v, j) =>
              -scale * v -
              Qt[j * 3] * shift[0] -
              Qt[j * 3 + 1] * shift[1] -
              Qt[j * 3 + 2] * shift[2],
          ),
        },
      })),
    },
    comparison = compareCameraModels(reference, candidate);
  assert.equal(comparison.alignment.status, 'aligned');
  assert.ok(Math.abs(comparison.alignment.scale - scale) < 1e-10);
  Q.forEach((v, i) => assert.ok(Math.abs(v - comparison.alignment.rotation[i]) < 1e-10));
  shift.forEach((v, i) => assert.ok(Math.abs(v - comparison.alignment.translation[i]) < 1e-10));
  assert.ok(comparison.alignment.maxCenterDifference < 1e-10);
  assert.ok(comparison.alignment.maxRotationDifferenceDeg < 1e-5);
  assert.equal(compareCameraModels({ views: [] }, candidate).alignment.status, 'unavailable');
  const line = {
    views: candidate.views.map((v, i) => ({ ...v, pose: { ...v.pose, t: [-i, 0, 0] } })),
  };
  assert.equal(compareCameraModels(line, line).alignment.status, 'unavailable');
});

test('pair geometry validates resources and releases buffers after failed readbacks', async () => {
  const f = radialPairFixture();
  let d = await ShadeDevice.open({ backend: 'cpu', workingBytes: 4096 });
  try {
    await assert.rejects(reconstructTwoViewGpu(d, f.pixels, f.first, f.second), /budget/);
    await assert.rejects(estimateHomographyGpu(d, f.pixels), /budget/);
    assert.throws(() => undistortCorrespondencesGpu(d, f.pixels, f.first, f.second), /budget/);
    assert.equal(d.statistics.allocations, 0);
  } finally {
    await d.dispose();
  }
  d = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    const original = d.read;
    d.read = async () => {
      throw new Error('injected initialization read failure');
    };
    for (const operation of [
      () => reconstructTwoViewGpu(d, f.pixels, f.first, f.second, { iterations: 16 }),
      () => estimateHomographyGpu(d, f.pixels, { iterations: 16 }),
      () => undistortCorrespondencesGpu(d, f.pixels, f.first, f.second),
    ]) {
      await assert.rejects(operation(), /injected initialization/);
      assert.equal(d.statistics.liveBuffers, 0);
    }
    d.read = original;
    await assert.rejects(
      reconstructTwoViewGpu(d, f.pixels, { ...f.first, k1: NaN }, f.second),
      RangeError,
    );
  } finally {
    await d.dispose();
  }
});

test('GPU-authored relative pose resolves signed forward/lateral motion and known structure', async () => {
  const d = await ShadeDevice.open({ backend: 'cpu' });
  try {
    for (const f of relativePoseCases()) {
      const r = await reconstructTwoViewGpu(d, f.pixels, f.first, f.second, {
          iterations: 16,
          minIterations: 16,
          refinementIterations: 0,
          threshold: 1e-7,
          minParallaxDeg: 0.001,
        }),
        m = recoveredMetrics(r, f),
        scale = Math.hypot(...f.t);
      assert.equal(m.count, 24, JSON.stringify({ id: f.id, ...m }));
      assert.ok(
        m.rotationErrorDeg < 1e-4 && m.translationCosine > 1 - 1e-10 && m.maxError < 1e-4,
        JSON.stringify({ id: f.id, ...m }),
      );
      assert.ok(Math.max(...r.points.map((x, i) => Math.abs(x - f.world[i] / scale))) < 1e-5);
    }
  } finally {
    await d.dispose();
  }
});

test('GPU-authored normalized homography QR/SVD matches independent direct DLT and rejects rank loss', async () => {
  const d = await ShadeDevice.open({ backend: 'cpu' });
  try {
    for (const settings of [{}, { offset: 10000, scale: 3 }, { scale: 0.001 }]) {
      const f = homographyFixture(settings),
        threshold = 0.8 * f.scale,
        result = await estimateHomographyGpu(d, f.pixels, {
          thresholdPx: threshold,
          iterations: 256,
        });
      assert.ok(result);
      assert.deepEqual(result.inliers, f.clean);
      const errors = symmetricTransferErrors(result.H, f.pixels);
      assert.deepEqual(
        errors.map((e) => Number(e <= threshold * threshold)),
        [...result.inliers],
      );
      const reference = fitHomography(
        Array.from({ length: f.clean.length }, (_, i) => i)
          .filter((i) => f.clean[i])
          .map((i) => [...f.pixels.subarray(4 * i, 4 * i + 4)]),
      );
      const expected = symmetricTransferErrors(reference, f.pixels);
      for (let i = 0; i < errors.length; i++)
        if (f.clean[i]) assert.ok(Math.abs(errors[i] - expected[i]) < 1e-6 * f.scale * f.scale);
      assert.equal(
        result.execution.applicationReadbackBytes,
        200 + 4 * f.clean.length + controlWordBytes(result.execution.report),
      );
      assert.deepEqual(result.execution.cpuStages, []);
    }
    const degenerate = homographyFixture({ collinear: true });
    assert.equal(await estimateHomographyGpu(d, degenerate.pixels, { iterations: 32 }), undefined);
    await assert.rejects(
      estimateHomographyGpu(d, new Float64Array(16), { iterations: 100001 }),
      RangeError,
    );
    const constrained = await estimateHomographyGpu(d, homographyFixture().pixels, {
      iterations: 17,
      minIterations: 17,
      batchSize: 64,
    });
    assert.equal(constrained.iterations, 17);
  } finally {
    await d.dispose();
  }
});

test('GPU radial pair inversion preserves source indices and rejects the folded lens branch', async () => {
  const d = await ShadeDevice.open({ backend: 'cpu' });
  try {
    const k = { fx: 100, fy: 120, cx: 50, cy: 40, k1: -0.5 },
      pixels = Float64Array.from([
        50, 40, 52, 43, 250, 40, 50, 40, 65, 55, 72, 61, 50, 40, 270, 40,
      ]),
      result = await undistortCorrespondencesGpu(d, pixels, k, k),
      expected = [],
      indices = [];
    for (let i = 0; i < 4; i++) {
      const a = unprojectPixel(k, [pixels[4 * i], pixels[4 * i + 1]]),
        b = unprojectPixel(k, [pixels[4 * i + 2], pixels[4 * i + 3]]);
      if (a && b) {
        indices.push(i);
        expected.push(
          a[0] * k.fx + k.cx,
          a[1] * k.fy + k.cy,
          b[0] * k.fx + k.cx,
          b[1] * k.fy + k.cy,
        );
      }
    }
    assert.deepEqual([...result.indices], indices);
    assert.equal(indices.length, 2);
    expected.forEach((x, i) => assert.ok(Math.abs(x - result.pixels[i]) < 1e-8));
    assert.equal(result.execution.applicationReadbackBytes, 36 * 4);
    assert.deepEqual(result.execution.cpuStages, []);
  } finally {
    await d.dispose();
  }
});

test('resident two-view refinement handles radial cameras and rejects low-parallax and planar seeds', async () => {
  const d = await ShadeDevice.open({ backend: 'cpu' });
  try {
    const f = radialPairFixture(),
      input = f.pixels.slice(),
      options = { iterations: 500, threshold: 0.001, seed: 71 };
    const pending = reconstructTwoViewGpu(d, f.pixels, f.first, f.second, options);
    f.pixels.fill(0);
    const r = await pending;
    f.pixels = input;
    const m = recoveredMetrics(r, f);
    assert.ok(m.rotationErrorDeg < 0.1 && m.translationCosine > 0.999, JSON.stringify(m));
    assert.ok(m.count >= 75 && m.meanError < 0.15 && m.minDepth > 0, JSON.stringify(m));
    assert.ok(Math.abs(m.tNorm - 1) < 1e-9);
    assert.ok(Math.abs(m.meanError - r.meanReprojectionErrorPx) < 1e-7);
    assert.ok(
      r.diagnostics.meanReprojectionErrorAfterPx <= r.diagnostics.meanReprojectionErrorBeforePx,
    );
    assert.equal(
      r.execution.applicationReadbackBytes,
      608 + 32 * f.clean.length + controlWordBytes(r.execution.report),
    );
    assert.deepEqual(r.execution.cpuStages, []);
    assert.equal(d.reports.length, 1);
    for (const bad of [
      radialPairFixture({ baseline: 1e-4, noise: 0, outliers: false }),
      radialPairFixture({ planar: true, noise: 0, outliers: false }),
    ])
      await assert.rejects(
        reconstructTwoViewGpu(d, bad.pixels, bad.first, bad.second, {
          iterations: 64,
          refinementIterations: 0,
        }),
        /parallax|valid two-view|reliable|Cheirality/,
      );
    await assert.rejects(
      reconstructTwoViewGpu(d, input, f.first, f.second, { refinementIterations: 101 }),
      RangeError,
    );
  } finally {
    await d.dispose();
  }
});

test('small sample probabilities do not turn GPU adaptive RANSAC into a premature stop', async () => {
  const d = await ShadeDevice.open({ backend: 'cpu' });
  try {
    const f = reconstructionFixture(),
      r = await estimateEssentialRansacGpu(d, f.pixels, f.first, f.second, {
        iterations: 32,
        minIterations: 1,
        batchSize: 16,
        seed: 71,
        threshold: 0.001,
      });
    assert.ok(r.inlierCount < 50);
    assert.equal(r.ransacIterations, 32);
  } finally {
    await d.dispose();
  }
});

test('device geometry captures inputs, rejects unsupported work and releases failed readbacks', async () => {
  const d = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    const data = poseFixture({ count: 4, noise: 0, outliers: 0 }),
      pending = estimatePosePnPGpu(d, data.world, data.pixels, data.camera, {
        iterations: 17,
        minIterations: 17,
        refinementIterations: 0,
      });
    data.world.fill(0);
    data.pixels.fill(0);
    assert.equal((await pending).inlierCount, 4);
    const fresh = poseFixture({ count: 4, noise: 0, outliers: 0 });
    assert.throws(
      () => estimatePosePnPGpu(d, fresh.world, fresh.pixels, fresh.camera, { thresholdPx: 1e-100 }),
      RangeError,
    );
    assert.throws(
      () => triangulateTracksGpu(d, [Array(65).fill(triangulationFixture().tracks[0][0])]),
      RangeError,
    );
    assert.throws(() => bundleAdjustGpu(d, bundleFixture(), { variableViewIds: [10] }), /window/);
    const original = d.read;
    d.read = async () => {
      throw new Error('injected geometry read failure');
    };
    await assert.rejects(
      estimatePosePnPGpu(d, fresh.world, fresh.pixels, fresh.camera, {
        iterations: 17,
        minIterations: 17,
      }),
      /injected/,
    );
    assert.equal(d.statistics.liveBuffers, 0);
    await assert.rejects(triangulateTracksGpu(d, triangulationFixture(2).tracks), /injected/);
    assert.equal(d.statistics.liveBuffers, 0);
    await assert.rejects(
      bundleAdjustGpu(d, bundleFixture({ count: 4 }), { iterations: 0 }),
      /injected/,
    );
    assert.equal(d.statistics.liveBuffers, 0);
    d.read = original;
  } finally {
    await d.dispose();
  }
});

test('GPU-authored joint BA uses matrix-free Schur PCG, preserves gauge and rolls back failed solves', async () => {
  const d = await ShadeDevice.open({ backend: 'cpu' });
  try {
    for (const settings of [{}, { outlier: true }, { shift: [7, -3, 2] }]) {
      const input = bundleFixture(settings),
        snapshot = structuredClone(input),
        initial = bundleMetrics(input, 1);
      const shifted = !!settings.shift,
        anchor = shifted ? 1 : 0,
        scaleView = shifted ? 3 : 1;
      const r = await bundleAdjustGpu(d, input, {
          iterations: 15,
          huberPx: 1,
          ...(shifted ? { anchorViewId: 20, scaleViewId: 40 } : {}),
        }),
        actual = bundleMetrics(r.reconstruction, 1);
      assert.ok(r.diagnostics.acceptedSteps > 0, JSON.stringify(r.diagnostics));
      assert.ok(
        actual.cost < initial.cost * 0.05,
        JSON.stringify({ initial, actual, diagnostics: r.diagnostics }),
      );
      assert.ok(Math.abs(actual.cost - r.diagnostics.finalCost) < 1e-6);
      assert.deepEqual(input, snapshot);
      assert.deepEqual(r.reconstruction.views[anchor].pose, input.views[anchor].pose);
      const a = cameraCenter(r.reconstruction.views[anchor].pose),
        b = cameraCenter(r.reconstruction.views[scaleView].pose);
      assert.ok(Math.abs(Math.hypot(...a.map((x, i) => x - b[i])) - r.diagnostics.baseline) < 1e-9);
      assert.equal(actual.observations, 160);
      assert.deepEqual(r.execution.cpuStages, []);
      console.log(
        JSON.stringify({
          gpuBaOracle: r.diagnostics,
          cg: r.cgIterations,
          linearFailures: r.linearSolverFailures,
        }),
      );
    }
    const input = bundleFixture(),
      local = await bundleAdjustGpu(d, input, { iterations: 10, variableViewIds: [30, 40] });
    assert.ok(local.diagnostics.finalCost < local.diagnostics.initialCost);
    assert.deepEqual(local.reconstruction.views[0].pose, input.views[0].pose);
    assert.deepEqual(local.reconstruction.views[1].pose, input.views[1].pose);
    const unchanged = await bundleAdjustGpu(d, input, { iterations: 0 });
    assert.deepEqual(unchanged.reconstruction, input);
    assert.notEqual(unchanged.reconstruction.landmarks[0].xyz, input.landmarks[0].xyz);
    const pointsOnly = await bundleAdjustGpu(d, input, { iterations: 5, variableViewIds: [] });
    assert.ok(pointsOnly.diagnostics.finalCost < pointsOnly.diagnostics.initialCost);
    assert.deepEqual(
      pointsOnly.reconstruction.views.map((v) => v.pose),
      input.views.map((v) => v.pose),
    );
    const rejected = await bundleAdjustGpu(d, input, {
      iterations: 1,
      maxCgIterations: 1,
      cgRelativeTolerance: 1e-12,
    });
    assert.equal(rejected.diagnostics.acceptedSteps, 0);
    assert.ok(rejected.linearSolverFailures > 0);
    assert.deepEqual(rejected.reconstruction, input);
    const zero = {
      ...input,
      views: input.views.map((v, i) => (i === 1 ? { ...v, pose: input.views[0].pose } : v)),
    };
    await assert.rejects(bundleAdjustGpu(d, zero, { iterations: 0 }), /baseline/);
    const behind = {
      ...input,
      landmarks: input.landmarks.map((p, i) =>
        i === 0 ? { ...p, xyz: Float64Array.of(0, 0, -5) } : p,
      ),
    };
    await assert.rejects(bundleAdjustGpu(d, behind, { iterations: 0 }), /in front/);
  } finally {
    await d.dispose();
  }
});

test('incremental device driver shares orchestration, snapshots caller input and propagates device failures', async () => {
  const data = mappingFixture(),
    d = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    const pending = reconstructIncrementalGpu(d, data.images, data.pairs, {
      ransacIterations: 128,
      bundleIterations: 2,
      finalBundleIterations: 4,
    });
    data.images[0].keypoints.fill(0);
    data.pairs[0].matches.fill(0);
    const r = await pending;
    assert.equal(r.reconstruction.views.length, 4);
    assert.equal(r.reconstruction.landmarks.length, 40);
    assert.deepEqual(r.unregistered, []);
    assert.ok(r.diagnostics.meanReprojectionErrorPx < 1e-6);
    assert.equal(r.execution.stages.filter((s) => s.operation === 'pnp').length, 2);
    assert.equal(r.execution.stages.filter((s) => s.operation === 'pairVerification').length, 6);
    assert.ok(!r.execution.cpuStages.includes('joint bundle adjustment'));
    assert.ok(r.execution.gpuBundleAdjustments > 0);
    const fresh = mappingFixture(),
      original = d.plan;
    d.plan = async () => {
      throw new Error('injected device plan failure');
    };
    await assert.rejects(
      reconstructIncrementalGpu(d, fresh.images, fresh.pairs, { ransacIterations: 17 }),
      /injected device/,
    );
    d.plan = original;
    assert.equal(d.statistics.liveBuffers, 0);
  } finally {
    await d.dispose();
  }
});

test('GPU-authored triangulation preserves direct-SVD consensus and rejects repeated/low-baseline rays', async () => {
  const data = triangulationFixture(),
    d = await ShadeDevice.open({ backend: 'cpu' });
  try {
    const r = await triangulateTracksGpu(d, data.tracks);
    for (let i = 0; i < data.count; i++) {
      const p = r.tracks[i],
        reference = triangulateTrack(data.tracks[i]);
      assert.ok(p, `track ${i}`);
      assert.ok(reference);
      assert.ok(Math.hypot(...p.xyz.map((v, j) => v - data.points[i][j])) < 1e-7);
      assert.deepEqual(p.inliers, reference.inliers);
      assert.ok(Math.abs(p.maxAngleDeg - reference.maxAngleDeg) < 1e-7);
    }
    assert.equal(r.tracks[data.count], undefined);
    assert.equal(r.tracks[data.count + 1], undefined);
    assert.equal(r.tracks[data.count + 2], undefined);
    assert.equal(r.status[data.count], 1);
    assert.equal(r.status[data.count + 2], 4);
    assert.deepEqual(r.execution.cpuStages, []);
  } finally {
    await d.dispose();
  }
});

/** The tracks of the lab job that lost a device in its first triangulation (1672 tracks of two
 *  observations, `tests/fixtures/south-building-pair-triangulation.json`) and its options. */
function southBuildingTriangulation() {
  const fixture = JSON.parse(
    readFileSync(
      new URL('../fixtures/south-building-pair-triangulation.json', import.meta.url),
      'utf8',
    ),
  );
  const tracks = fixture.pixels.map((pixels) =>
    fixture.views.map((view, k) => ({
      camera: view.camera,
      pose: { R: Float64Array.from(view.pose.R), t: Float64Array.from(view.pose.t) },
      pixel: [pixels[2 * k], pixels[2 * k + 1]],
    })),
  );
  return { tracks, options: fixture.options };
}

test('track triangulation runs as chunks of tracks under the time budget with unchanged results', async () => {
  // The least chunk is one workgroup of `triangulateTracks`: 16 tracks.
  const least = 16;
  const synthetic = triangulationFixture(200);
  for (const { tracks, options } of [
    { tracks: triangulationFixture().tracks, options: {} },
    { tracks: synthetic.tracks, options: { maxPairHypotheses: 3, refinementIterations: 3 } },
    southBuildingTriangulation(),
  ]) {
    const T = tracks.length;
    const runs = [];
    for (const settings of [
      // The reference: the CPU tier's `chunkSize` returns `max`, so one chunk dispatches the
      // kernel over every track, as the single dispatch before the time budget did.
      { name: 'whole' },
      // Forced least chunks: what WebGPU does on a slow device, one workgroup of tracks a chunk.
      { name: 'least', force: true },
      { name: 'least, bounded 1', force: true, boundedSubmissions: true },
      {
        name: 'least, bounded 16',
        force: true,
        boundedSubmissions: true,
        maxDispatchesPerSubmission: 16,
      },
      { name: 'whole, bounded 16', boundedSubmissions: true, maxDispatchesPerSubmission: 16 },
    ]) {
      const { name, force, ...open } = settings;
      const events = [];
      // Every chunk is timed, so each chunk's plan names its key and its units. By default a
      // device stops timing a key after eight chunks (`chunkTimingSamples`,
      // `tests/typeshade/plan-steps.test.mjs`), and the forced chunks of the larger fixtures
      // are more than eight.
      const device = await ShadeDevice.open({
        backend: 'cpu',
        chunkTimingSamples: Infinity,
        ...open,
        onProgress: (event) => events.push(event),
      });
      const plans = [],
        sizes = [],
        plan = device.plan.bind(device);
      device.plan = async (body, chunk) => {
        const report = await plan(body, chunk);
        plans.push({ calls: report.groups.flatMap((group) => group.calls), chunk });
        return report;
      };
      if (force)
        device.chunkSize = (key, max, min) => {
          sizes.push({ key, max, min });
          return min;
        };
      try {
        runs.push({
          name,
          settings,
          plans,
          sizes,
          events,
          result: await triangulateTracksGpu(device, tracks, options),
        });
      } finally {
        await device.dispose();
      }
      assert.equal(device.statistics.liveBuffers, 0);
    }
    const [reference, ...chunked] = runs;
    for (const run of runs) {
      const { name, settings, plans, sizes, events, result } = run;
      // Bit for bit: a track's work reads the prepared geometry of its own observations and
      // writes only its own slots, so the chunk boundaries change no result.
      assert.deepEqual(result.status, reference.result.status, name);
      assert.deepEqual(result.tracks, reference.result.tracks, name);
      // `prepareTriangulation` is a plan before the first chunk; each chunk is a plan of its own
      // with one `triangulateTracks` call over a contiguous range of tracks.
      assert.deepEqual(plans[0], { calls: ['prepareTriangulation'], chunk: undefined }, name);
      const chunks = plans.slice(1);
      assert.ok(
        chunks.every((p) => p.chunk?.key === 'triangulateTracks'),
        name,
      );
      assert.ok(
        chunks.every((p) => p.calls.join() === 'triangulateTracks'),
        name,
      );
      assert.equal(
        chunks.reduce((n, p) => n + p.chunk.units, 0),
        T,
        name,
      );
      if (settings.force) {
        // Chunks of one workgroup each, the last one holding what remains. The host asks for
        // each chunk after the previous one ran, with what remains as the maximum.
        assert.equal(chunks.length, Math.ceil(T / least), name);
        assert.ok(
          chunks.slice(0, -1).every((p) => p.chunk.units === least),
          name,
        );
        assert.equal(chunks.at(-1).chunk.units, T - least * (chunks.length - 1), name);
        assert.deepEqual(
          sizes,
          chunks.map((_, i) => ({ key: 'triangulateTracks', max: T - least * i, min: least })),
          name,
        );
      } else
        assert.deepEqual(
          chunks.map((p) => p.chunk.units),
          [T],
          name,
        );
      // Each chunk reports `triangulateTracks` before it starts and after it completes, so a device
      // lost in a chunk names the kernel (the lab's `failure.kernel`).
      assert.deepEqual(
        events,
        ['prepareTriangulation', ...chunks.map(() => 'triangulateTracks')].flatMap((kernel) => [
          { kernel, phase: 'start' },
          { kernel, phase: 'complete' },
        ]),
        name,
      );
      // The report carries the rate the TypeShade device learned of the key.
      const rate = result.execution.report.chunks?.triangulateTracks;
      assert.ok(rate, JSON.stringify(result.execution.report.chunks));
      assert.equal(rate.chunks, chunks.length, name);
      assert.equal(rate.lastUnits, chunks.at(-1).chunk.units, name);
      // Bounded submissions keep at most `maxDispatchesPerSubmission` steps in a plan.
      if (settings.boundedSubmissions)
        assert.ok(
          plans.every((p) => p.calls.length <= (settings.maxDispatchesPerSubmission ?? 1)),
          name,
        );
      // The readback does not depend on the chunks: the results, the masks and the statuses.
      assert.equal(
        result.execution.applicationReadbackBytes,
        reference.result.execution.applicationReadbackBytes,
        name,
      );
    }
    assert.ok(chunked.length > 0);
  }
});

test('GPU-authored PnP CPU oracle handles distortion/outliers, planar scenes and four-point input', async () => {
  const d = await ShadeDevice.open({ backend: 'cpu' });
  try {
    for (const settings of [
      {},
      { count: 41, planar: true, noise: 0, outliers: 0 },
      { count: 4, noise: 0, outliers: 0 },
      { scale: 0.01 },
      { scale: 100 },
    ]) {
      const data = poseFixture(settings),
        options = { iterations: 128, seed: 91, refinementIterations: 12 };
      const r = await estimatePosePnPGpu(d, data.world, data.pixels, data.camera, options),
        e = poseErrors(r, data);
      assert.ok(e.inliers >= data.expectedInliers, JSON.stringify(e));
      assert.ok(e.rotationFrobenius < 0.005, JSON.stringify(e));
      assert.ok(e.translation < 0.03, JSON.stringify(e));
      if (settings.noise === 0) {
        assert.ok(e.translation < 1e-6, JSON.stringify(e));
        assert.ok(e.errorPx < 1e-7);
      }
      assert.ok(r.iterations >= 64 && r.iterations <= 128);
      assert.equal(r.execution.applicationReadbackBytes, 160 + (4 * data.pixels.length) / 2);
      assert.deepEqual(r.execution.cpuStages, []);
    }
    const data = poseFixture({ count: 4, noise: 0, outliers: 0 });
    const opts = {
      iterations: 17,
      minIterations: 17,
      batchSize: 5,
      seed: 42,
      refinementIterations: 0,
    };
    const first = await estimatePosePnPGpu(d, data.world, data.pixels, data.camera, opts);
    const again = await estimatePosePnPGpu(d, data.world, data.pixels, data.camera, opts);
    assert.deepEqual([...first.pose.R, ...first.pose.t], [...again.pose.R, ...again.pose.t]);
    assert.equal(first.iterations, 17);
    const bad = Float64Array.from(Array.from({ length: 12 }, (_, i) => [i * 0.1, 0, 4]).flat());
    await assert.rejects(
      estimatePosePnPGpu(d, bad, new Float64Array(24).fill(100), data.camera, {
        iterations: 17,
        minIterations: 17,
      }),
      /could not register/,
    );
    assert.throws(
      () => estimatePosePnPGpu(d, data.world, data.pixels, data.camera, { minInliers: 9 }),
      RangeError,
    );
  } finally {
    await d.dispose();
  }
});

test('compiled P3P isolates every CPU reference branch and validates poses across scales', async () => {
  const d = await ShadeDevice.open({ backend: 'cpu' });
  try {
    const cases = p3pCases(),
      packed = packP3pCases(cases);
    const pending = solveP3pBatchGpu(d, packed);
    packed.fill(0);
    const batch = await pending;
    assert.equal(d.statistics.applicationReadbackBytes, 0);
    assert.equal(batch.report.tier, 'cpu');
    const result = await batch.read(),
      metrics = p3pMetrics(cases, result);
    assert.deepEqual(metrics.failures, []);
    assert.ok(metrics.maxTruthError < 1e-4, JSON.stringify(metrics));
    assert.ok(metrics.maxRayError < 1e-7);
    assert.ok(metrics.maxOrthogonalityError < 1e-7);
    let fourBranches = 0;
    for (let h = 0; h < cases.length; h++) {
      if (cases[h].name === 'zero-ray' || cases[h].status === 6) continue;
      const expected = p3p(
        cases[h].world.map((p) => Float64Array.from(p)),
        cases[h].rays.map((p) => Float64Array.from(p)),
      );
      const actual = metrics.output[h];
      assert.equal(actual.candidates.length, expected.length, cases[h].name);
      if (expected.length === 4) fourBranches++;
      for (const p of expected)
        assert.ok(
          actual.candidates.some(
            (q) => poseDifference(q, [...p.R, ...p.t], actual.sceneScale) < 1e-4,
          ),
          cases[h].name,
        );
    }
    assert.ok(fourBranches > 0);
    const reading = batch.read(),
      disposing = batch.dispose();
    await reading;
    await disposing;
    assert.equal(batch.dispose(), disposing);
    await assert.rejects(batch.read(), /disposed/);
    const snapshot = result.poses.slice();
    const rejectedCase = cases.find((c) => c.name === 'narrow-cone');
    const reused = await solveP3pBatchGpu(d, packP3pCases(cases.map(() => rejectedCase)));
    assert.ok((await reused.read()).poses.every((x) => x === 0));
    assert.deepEqual(result.poses, snapshot, 'readback must outlive buffer reuse');
    await reused.dispose();
    console.log(
      JSON.stringify({
        p3pProblems: cases.length,
        fourBranches,
        maxRayError: metrics.maxRayError,
        maxTruthError: metrics.maxTruthError,
      }),
    );
  } finally {
    await d.dispose();
  }
});

test('P3P rejects invalid budgets and destroys buffers on submission or readback failure', async () => {
  const d = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0, workingBytes: 4096 });
  const packed = packP3pCases(p3pCases().slice(0, 1));
  try {
    assert.throws(() => solveP3pBatchGpu(d, new Float64Array(0)), RangeError);
    assert.throws(() => solveP3pBatchGpu(d, new Float64Array(19)), RangeError);
    assert.throws(() => solveP3pBatchGpu(d, new Float64Array(8 * 18)), /budget/);
    for (const invalid of [NaN, Infinity, 1e10, 1e-40]) {
      const input = packed.slice();
      input[0] = invalid;
      assert.throws(() => solveP3pBatchGpu(d, input), RangeError);
    }
    const originalPlan = d.plan;
    d.plan = async () => {
      throw new Error('injected submission failure');
    };
    await assert.rejects(solveP3pBatchGpu(d, packed), /injected submission/);
    assert.equal(d.statistics.liveBuffers, 0);
    d.plan = originalPlan;
    const batch = await solveP3pBatchGpu(d, packed);
    d.read = async () => {
      throw new Error('injected readback failure');
    };
    await assert.rejects(batch.read(), /injected readback/);
    await batch.dispose();
    assert.equal(d.statistics.liveBuffers, 0);
    await assert.rejects(batch.read(), /disposed/);
  } finally {
    await d.dispose();
  }
});

test('resident allocation pools distinguish f32, f64 and u32 layouts and count f64 bytes', async () => {
  const d = await ShadeDevice.open({ backend: 'cpu' });
  try {
    const f = d.allocate(4, Float32Array.of(1, 2, 3, 4));
    d.recycle(f);
    const wide = d.allocateF64(4, Float64Array.of(2 ** 25 + 0.125, 2, 3, 4));
    const integer = d.allocateU32(4, Uint32Array.of(1, 2, 3, 4));
    assert.notEqual(wide, f);
    assert.notEqual(integer, f);
    assert.notEqual(integer, wide);
    assert.equal((await d.read(wide, 4))[0], 2 ** 25 + 0.125);
    assert.equal(d.statistics.applicationReadbackBytes, 32);
    d.recycle(wide);
    d.recycle(integer);
    // The pool is keyed by element and length (TypeShade change 0071): each comes back for its own
    // kind, as the reuse count says. A reused array is a new handle over the same storage.
    assert.deepEqual(await d.read(d.allocateF64(4), 4), new Float64Array(4));
    assert.deepEqual(await d.read(d.allocateU32(4), 4), new Uint32Array(4));
    assert.deepEqual(await d.read(d.allocate(4), 4), new Float32Array(4));
    assert.equal(d.statistics.reuses, 3);
    assert.equal(d.statistics.liveBuffers, 3);
  } finally {
    await d.dispose();
  }
});

test('compiled TypeShade CPU tier agrees with independent f64 scoring and tie order', async () => {
  const data = fixture(),
    expected = reference(data),
    d = await ShadeDevice.open({ backend: 'cpu' });
  try {
    const s = await EpipolarSession.create(d, data.pixels, data.first, data.second);
    const r = await s.score(data.models, data.threshold, { readMask: true, readStatistics: true });
    assert.equal(r.report.tier, 'cpu');
    assert.equal(r.inlierCount, expected.counts[2]);
    assert.equal(r.bestIndex, 2);
    assert.deepEqual(Array.from(r.mask), expected.masks[2]);
    const h = data.models.length / 9;
    for (let i = 0; i < h; i++) {
      assert.equal(r.modelStatistics[i], expected.counts[i]);
      assert.ok(Math.abs(r.modelStatistics[h + i] - expected.costs[i]) < 0.002);
    }
    await s.dispose();
  } finally {
    await d.dispose();
  }
});

test('resident observations, reuse, serialized queries and concurrent disposal have defined lifetimes', async () => {
  const data = fixture(31),
    d = await ShadeDevice.open({ backend: 'cpu' });
  const s = await EpipolarSession.create(d, data.pixels, data.first, data.second);
  const before = d.statistics;
  const a = await s.score(data.models, data.threshold),
    mid = d.statistics;
  const [b, c] = await Promise.all([
    s.score(data.models, data.threshold),
    s.score(data.models, data.threshold),
  ]);
  assert.deepEqual([a.bestIndex, b.bestIndex, c.bestIndex], [2, 2, 2]);
  assert.equal(
    d.statistics.applicationUploadBytes - before.applicationUploadBytes,
    3 * data.models.length * 4,
  );
  assert.equal(d.statistics.applicationReadbackBytes - before.applicationReadbackBytes, 3 * 12);
  assert.equal(d.statistics.allocations, mid.allocations);
  assert.ok(d.statistics.reuses > 0);
  const pending = s.score(data.models, data.threshold);
  const closing = s.dispose();
  await pending;
  await closing;
  await assert.rejects(s.score(data.models, data.threshold), /disposed/);
  const dispose = d.dispose();
  assert.equal(d.dispose(), dispose);
  await dispose;
  assert.equal(d.statistics.liveBuffers, 0);
  await assert.rejects(
    d.enqueue(async () => 0),
    /disposed/,
  );
});

test('strict tier selection and f32/storage bounds reject invalid work', async () => {
  const data = fixture(10),
    d = await ShadeDevice.open({ backend: 'cpu', maxModelsPerBatch: 2 });
  try {
    // Two devices in one context are independent (TypeShade change 0071).
    await (await ShadeDevice.open({ backend: 'cpu' })).dispose();
    const s = await EpipolarSession.create(d, data.pixels, data.first, data.second);
    assert.equal(s.capacity, 2);
    assert.throws(() => s.score(data.models, data.threshold), RangeError);
    assert.throws(() => s.score(data.models.slice(0, 9), 1e-30), RangeError);
    assert.throws(() => s.score(data.models.slice(0, 9), 1e-20), RangeError);
    assert.throws(
      () => EpipolarSession.create(d, data.pixels, { ...data.first, k1: 0.1 }, data.second),
      /undistorted/,
    );
    assert.throws(
      () => EpipolarSession.create(d, Float64Array.of(Infinity, 0, 0, 0), data.first, data.second),
      RangeError,
    );
    const huge = { fx: 1e38, fy: 1e38, cx: -3e38, cy: 0 };
    assert.throws(
      () => EpipolarSession.create(d, Float64Array.of(3e38, 0, 0, 0), huge, data.second),
      /f32 range/,
    );
    await s.dispose();
  } finally {
    await d.dispose();
  }
});

test('failed readback destroys temporary buffers and leaves a reusable session', async () => {
  const d = await ShadeDevice.open({ backend: 'cpu' }),
    data = fixture(257);
  try {
    const s = await EpipolarSession.create(d, data.pixels, data.first, data.second);
    const original = d.read;
    d.read = async () => {
      throw new Error('injected readback failure');
    };
    await assert.rejects(s.score(data.models, data.threshold), /injected/);
    assert.equal(d.statistics.liveBuffers, 1);
    assert.equal(d.statistics.pooledBytes, 0);
    d.read = original;
    assert.equal((await s.score(data.models, data.threshold)).bestIndex, 2);
    await s.dispose();
  } finally {
    await d.dispose();
  }
});

test('reduction block tails, zero-pool budgets and caller snapshots are deterministic', async () => {
  // n257 normalization holds two4112-byte arrays; scoring with its mask and
  // scratch needs8296bytes. Three pages admit the intended cases while n400
  // normalization still needs12800bytes and must reject below.
  const d = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0, workingBytes: 12288 });
  try {
    for (const n of [1, 255, 256, 257]) {
      const data = fixture(n),
        expected = reference(data),
        a = { ...data.first };
      const creating = EpipolarSession.create(d, data.pixels, a, data.second);
      a.fx = 1;
      data.pixels.fill(0);
      const s = await creating;
      const options = { readMask: true };
      const pending = s.score(data.models.subarray(18, 27), data.threshold, options);
      options.readMask = false;
      const r = await pending;
      assert.deepEqual([...r.mask], expected.masks[2]);
      assert.equal(r.inlierCount, expected.counts[2]);
      assert.equal(d.statistics.pooledBytes, 0);
      await s.dispose();
      assert.equal(d.statistics.liveBuffers, 0);
    }
    const data = fixture(400);
    await assert.rejects(EpipolarSession.create(d, data.pixels, data.first, data.second), /budget/);
    assert.equal(d.statistics.liveBuffers, 0);
  } finally {
    await d.dispose();
  }
});

test('missing WebGPU rejects explicitly, naming the tier, and a CPU device still opens', async () => {
  await assert.rejects(ShadeDevice.open(), /webgpu/);
  const d = await ShadeDevice.open({ backend: 'cpu' });
  await d.dispose();
});

test('resident RANSAC reconstructs a known noisy pose and obeys adaptive batch/trial caps', async () => {
  const d = await ShadeDevice.open({ backend: 'cpu' }),
    data = reconstructionFixture();
  try {
    const options = {
      iterations: 500,
      minIterations: 32,
      threshold: 0.001,
      batchSize: 16,
      seed: 71,
    };
    const result = await reconstructTwoViewGpu(d, data.pixels, data.first, data.second, options);
    const metric = poseMetrics(result, data);
    assert.ok(metric.valid >= 250, JSON.stringify(metric));
    assert.ok(metric.rotationErrorDeg < 0.1, JSON.stringify(metric));
    assert.ok(metric.translationCosine > 0.999, JSON.stringify(metric));
    assert.ok(metric.errorPx < 0.15);
    assert.equal(metric.execution.backend, 'cpu');
    assert.ok(metric.execution.hypothesisBatches > 1);
    assert.deepEqual(metric.execution.cpuStages, []);
    assert.ok(metric.trials >= 32 && metric.trials <= 500);
    const capped = await estimateEssentialRansacGpu(d, data.pixels, data.first, data.second, {
      ...options,
      iterations: 17,
      minIterations: 17,
      batchSize: 64,
    });
    assert.equal(capped.ransacIterations, 17);
    assert.equal(capped.execution.hypothesisBatches, 1);
    await assert.rejects(
      estimateEssentialRansacGpu(d, data.pixels, data.first, data.second, { batchSize: 0 }),
      RangeError,
    );
    const coincident = new Float64Array(40 * 4).fill(200);
    await assert.rejects(
      estimateEssentialRansacGpu(d, coincident, data.first, data.second, { iterations: 4 }),
      /valid two-view/,
    );
  } finally {
    await d.dispose();
  }
});

test('TypeShade FAST and oriented BRIEF features match a translated textured frame', async () => {
  const width = 192,
    height = 160,
    firstPixels = new Uint8Array(width * height * 4),
    secondPixels = new Uint8Array(width * height * 4);
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
      const at = (y * width + x) * 4;
      const a = gray[y * width + x];
      firstPixels[at] = a;
      firstPixels[at + 1] = a;
      firstPixels[at + 2] = a;
      firstPixels[at + 3] = 255;
      const sx = x - 5,
        sy = y - 3,
        b = sx >= 0 && sy >= 0 ? gray[sy * width + sx] : 0;
      secondPixels[at] = b;
      secondPixels[at + 1] = b;
      secondPixels[at + 2] = b;
      secondPixels[at + 3] = 255;
    }
  const d = await ShadeDevice.open({ backend: 'cpu' });
  try {
    await assert.rejects(
      extractFeaturesGpu(d, new Uint8Array(48 * 48 * 4), 48, 48, { maxFeatures: 2001 }),
      RangeError,
    );
    const first = await extractFeaturesGpu(d, firstPixels, width, height);
    const second = await extractFeaturesGpu(d, secondPixels, width, height);
    assert.ok(first.scores.length > 20, `features=${first.scores.length}`);
    assert.equal(first.descriptors.length, first.scores.length * 8);
    const paired = await matchFeaturesGpu(d, first, second);
    assert.ok(paired.matches.length >= 20, `matches=${paired.matches.length / 2}`);
    const displacement = [];
    for (let i = 0; i < paired.matches.length; i += 2) {
      const a = paired.matches[i],
        b = paired.matches[i + 1];
      displacement.push([
        second.keypoints[b * 2] - first.keypoints[a * 2],
        second.keypoints[b * 2 + 1] - first.keypoints[a * 2 + 1],
      ]);
    }
    const median = (values) => values.sort((a, b) => a - b)[Math.floor(values.length / 2)];
    assert.equal(median(displacement.map((v) => v[0])), 5);
    assert.equal(median(displacement.map((v) => v[1])), 3);
    const singleton = {
      keypoints: Float64Array.of(10, 10),
      scores: Float32Array.of(1),
      descriptors: new Uint32Array(8),
      capacity: 1,
    };
    assert.equal(
      (await matchFeaturesGpu(d, singleton, singleton)).matches.length,
      0,
      'ratio test needs two candidates',
    );
    assert.equal(first.report.tier, 'cpu');

    const budgetWidth = 512,
      budgetHeight = 512,
      budgetPixels = new Uint8Array(budgetWidth * budgetHeight * 4);
    for (let i = 0; i < budgetWidth * budgetHeight; i++) {
      const value = random() & 255,
        at = i * 4;
      budgetPixels[at] = value;
      budgetPixels[at + 1] = value;
      budgetPixels[at + 2] = value;
      budgetPixels[at + 3] = 255;
    }
    const tileCount = Math.ceil(budgetWidth / 24) * Math.ceil(budgetHeight / 24);
    for (const limit of [1, 3, 400]) {
      const budgeted = await extractFeaturesGpu(d, budgetPixels, budgetWidth, budgetHeight, {
          maxFeatures: limit,
        }),
        tilesX = Math.ceil(budgetWidth / 24),
        slotsPerTile = Math.max(1, Math.min(4, Math.ceil(limit / tileCount))),
        selectedTiles = Math.ceil(Math.min(tileCount * slotsPerTile, limit) / slotsPerTile),
        expectedTiles = Array.from({ length: selectedTiles }, (_, selectedTile) =>
          Math.floor(((2 * selectedTile + 1) * tileCount) / (2 * selectedTiles)),
        ),
        actualTiles = Array.from(
          { length: budgeted.scores.length },
          (_, i) =>
            Math.floor(budgeted.keypoints[2 * i + 1] / 24) * tilesX +
            Math.floor(budgeted.keypoints[2 * i] / 24),
        );
      assert.ok(budgeted.scores.length <= limit, `count=${budgeted.scores.length} limit=${limit}`);
      assert.deepEqual(actualTiles, expectedTiles.slice(0, budgeted.scores.length));
      assert.equal(budgeted.scores.length, limit, `spatially sampled features for limit=${limit}`);
      if (limit === 1) assert.equal(budgeted.capacity, 1);
      if (limit === 3) assert.equal(budgeted.capacity, 3);
      if (limit === 400) assert.equal(budgeted.capacity, 400);
    }

    const residentFirst = await extractFeaturesGpuResident(d, firstPixels, width, height),
      residentSecond = await extractFeaturesGpuResident(d, secondPixels, width, height);
    try {
      assert.equal(residentFirst.count, first.scores.length);
      assert.equal(residentSecond.count, second.scores.length);
      const keypoints = await residentFirst.readKeypoints();
      assert.equal(keypoints.length, residentFirst.count * 2);
      assert.strictEqual(
        await residentFirst.readKeypoints(),
        keypoints,
        'keypoint readback is cached',
      );
      const residentPairs = await matchResidentFeaturesGpu(d, residentFirst, residentSecond);
      assert.deepEqual(residentPairs.matches, paired.matches);
      assert.deepEqual(residentPairs.distances, paired.distances);
      assert.ok(residentPairs.matches.length >= 40);
      for (let i = 2; i < residentPairs.matches.length; i += 2)
        assert.ok(
          residentPairs.matches[i] > residentPairs.matches[i - 2],
          'stable source-index ordering',
        );
      assert.equal(
        residentPairs.report.dispatches,
        d.backend === 'cpu' ? 0 : 6,
        'CPU reports no GPU kernel dispatches; GPU matching uses six dispatches',
      );
      assert.equal(
        residentPairs.execution.applicationUploadBytes,
        0,
        'resident matching uploads no feature data',
      );
      assert.equal(
        residentPairs.execution.applicationReadbackBytes,
        residentPairs.matches.byteLength + residentPairs.distances.length * 4 + 4,
      );
      await residentFirst.dispose();
      await assert.rejects(matchResidentFeaturesGpu(d, residentFirst, residentSecond), /disposed/);
    } finally {
      await residentFirst.dispose();
      await residentSecond.dispose();
    }
  } finally {
    await d.dispose();
  }
});

test('bounded QR/SVD preserves pair geometry and rejection while reading scheduling metadata only', async () => {
  const options = {
    iterations: 32,
    minIterations: 32,
    homographyIterations: 32,
    threshold: 0.001,
    seed: 71,
    refinementIterations: 4,
  };
  for (const input of [
    radialPairFixture({ n: 65 }),
    compactedPairFixture(97),
    radialPairFixture({ n: 65, planar: true }),
  ]) {
    const results = [];
    const events = [];
    for (const boundedSubmissions of [false, true]) {
      const device = await ShadeDevice.open({
        backend: 'cpu',
        maxModelsPerBatch: 8,
        boundedSubmissions,
        onProgress: (event) => events.push(event),
      });
      try {
        results.push(
          await verifyImagePairGpu(device, input.pixels, input.first, input.second, options),
        );
      } finally {
        await device.dispose();
      }
      assert.equal(device.statistics.liveBuffers, 0);
    }
    const [original, bounded] = results;
    assert.equal(bounded.reason, original.reason);
    assert.deepEqual(bounded.inlierRows, original.inlierRows);
    assert.equal(bounded.homographyInlierCount, original.homographyInlierCount);
    assert.equal(bounded.execution.hypothesisBatches, original.execution.hypothesisBatches);
    assert.equal(bounded.execution.hypothesesScored, original.execution.hypothesesScored);
    assert.equal(bounded.execution.homographyIterations, original.execution.homographyIterations);
    if (original.geometry) {
      for (const key of ['R', 't'])
        bounded.geometry[key].forEach((v, i) =>
          assert.ok(Math.abs(v - original.geometry[key][i]) < 1e-8, key),
        );
      assert.ok(
        Math.abs(
          bounded.geometry.meanReprojectionErrorPx - original.geometry.meanReprojectionErrorPx,
        ) < 1e-8,
      );
    }
    assert.ok(events.some((event) => event.kernel === 'appendPairModelRows'));
    assert.ok(events.some((event) => event.kernel === 'solvePairModelNull'));
    assert.ok(
      bounded.execution.applicationReadbackBytes > original.execution.applicationReadbackBytes,
    );
    assert.deepEqual(bounded.execution.cpuStages, []);
  }
});
