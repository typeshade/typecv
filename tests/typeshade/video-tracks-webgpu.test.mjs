import test from 'node:test';
import assert from 'node:assert/strict';
import { reconstructIncremental } from '../../dist/reconstruction/mapper.js';
import { bundleAdjust } from '../../dist/reconstruction/bundle-adjustment.js';
import { extractLocalBundle } from '../../dist/reconstruction/local-bundle.js';
import { inBrowser } from './browser.mjs';
import { poseFixture, bundleMetrics } from './geometry-cases.mjs';

function sequenceFixture(grouped = false) {
  const count = grouped ? 64 : 40,
    frames = grouped ? 6 : 4,
    data = poseFixture({ count, noise: 0, outliers: 0, radial: false }),
    points = Array.from({ length: count }, (_, i) => [...data.world.subarray(i * 3, i * 3 + 3)]),
    visible = (frame, point) => !grouped || (point < 32 ? frame <= 2 : frame >= 1),
    images = Array.from({ length: frames }, (_, frame) => ({
      id: 10 + frame * 10,
      camera: data.camera,
      keypoints: Float64Array.from(
        points.flatMap((point) => [
          (data.camera.fx * (point[0] - 0.3 * frame)) / point[2] + data.camera.cx,
          (data.camera.fy * point[1]) / point[2] + data.camera.cy,
        ]),
      ),
    })),
    pairs = [];
  for (let first = 0; first < frames; first++)
    for (let second = first + 1; second < frames; second++) {
      const common = points.flatMap((_, i) =>
        visible(first, i) && visible(second, i) ? [i, i] : [],
      );
      if (common.length)
        pairs.push({
          first: images[first].id,
          second: images[second].id,
          matches: Uint32Array.from(common),
        });
    }
  return { images, pairs, points, visible };
}

function packedModel(model) {
  return {
    views: model.views.map((v) => ({ ...v, pose: { R: [...v.pose.R], t: [...v.pose.t] } })),
    landmarks: model.landmarks.map((p) => ({ ...p, xyz: [...p.xyz] })),
  };
}

const options = {
  initialPair: [10, 20],
  ransacIterations: 128,
  bundleIterations: 2,
  finalBundleIterations: 4,
  localBundleSize: 1,
  // Every registration takes a local BA: no growth ratio or frequency makes one global.
  globalBundleViewRatio: Infinity,
  globalBundlePointRatio: Infinity,
};

async function runGpu(data, controls, localInput) {
  return inBrowser(
    async ({ images, pairs, controls, localInput, hardware }) => {
      const { ShadeDevice, reconstructIncrementalGpu, bundleAdjustGpu } =
          await import('/dist/gpu.js'),
        { extractLocalBundle, mergeLocalBundle } =
          await import('/dist/reconstruction/local-bundle.js');
      const adapter = await navigator.gpu?.requestAdapter();
      if (!adapter) throw new Error('WebGPU adapter unavailable');
      const adapterDescription = [
          adapter.info?.vendor,
          adapter.info?.architecture,
          adapter.info?.device,
          adapter.info?.description,
        ]
          .filter(Boolean)
          .join(' '),
        software =
          /swiftshader|software|llvmpipe/i.test(adapterDescription) ||
          adapter.isFallbackAdapter === true;
      if (hardware && software) throw new Error('Hardware requested but software adapter selected');
      const gpu = await adapter.requestDevice(),
        gpuErrors = [];
      gpu.addEventListener('uncapturederror', (event) => gpuErrors.push(event.error.message));
      const device = await ShadeDevice.open({ gpuDevice: gpu });
      let output;
      try {
        images = images.map((v) => ({ ...v, keypoints: Float64Array.from(v.keypoints) }));
        pairs = pairs.map((p) => ({ ...p, matches: Uint32Array.from(p.matches) }));
        const snapshot = JSON.stringify({ images, pairs }),
          result = await reconstructIncrementalGpu(device, images, pairs, controls);
        output = {
          model: {
            views: result.reconstruction.views.map((v) => ({
              ...v,
              pose: { R: [...v.pose.R], t: [...v.pose.t] },
            })),
            landmarks: result.reconstruction.landmarks.map((p) => ({ ...p, xyz: [...p.xyz] })),
          },
          diagnostics: result.diagnostics,
          unregistered: result.unregistered,
          backend: result.execution.backend,
          cpuStages: result.execution.cpuStages,
          gpuBundleAdjustments: result.execution.gpuBundleAdjustments,
          stages: result.execution.stages.map((stage) => ({
            operation: stage.operation,
            problem: stage.problem,
            tier: stage.execution.report.tier,
            dispatches: stage.execution.report.dispatches,
          })),
          inputPreserved: snapshot === JSON.stringify({ images, pairs }),
        };
        if (localInput) {
          const input = {
              views: localInput.views.map((v) => ({
                ...v,
                pose: { R: Float64Array.from(v.pose.R), t: Float64Array.from(v.pose.t) },
              })),
              landmarks: localInput.landmarks.map((p) => ({ ...p, xyz: Float64Array.from(p.xyz) })),
            },
            snapshot = JSON.stringify(input),
            local = extractLocalBundle(input, [60]);
          if (!local)
            throw new Error('Expected connected local problem with fixed boundary cameras');
          const refined = await bundleAdjustGpu(device, local.reconstruction, {
              iterations: 5,
              variableViewIds: local.variableViewIds,
              anchorViewId: local.anchorViewId,
              scaleViewId: local.scaleViewId,
            }),
            merged = mergeLocalBundle(input, local, refined.reconstruction),
            selected = new Set(local.reconstruction.landmarks.map((p) => p.trackId));
          output.local = {
            diagnostics: refined.diagnostics,
            tier: refined.execution.report.tier,
            dispatches: refined.execution.report.dispatches,
            inputPreserved: snapshot === JSON.stringify(input),
            fixedCamerasPreserved: merged.views.every(
              (v, i) => v.id === 60 || v === input.views[i],
            ),
            intrinsicsPreserved: merged.views.every((v, i) => v.camera === input.views[i].camera),
            unrelatedPreserved: merged.landmarks.every(
              (p, i) => selected.has(p.trackId) || p === input.landmarks[i],
            ),
            observationsPreserved: merged.landmarks.every(
              (p, i) => p.observations === input.landmarks[i].observations,
            ),
            solverObservationsPreserved: refined.reconstruction.landmarks.every(
              (p, i) =>
                JSON.stringify(p.observations) ===
                JSON.stringify(local.reconstruction.landmarks[i].observations),
            ),
          };
        }
      } finally {
        await device.dispose();
        gpu.destroy();
      }
      return {
        ...output,
        gpuErrors,
        liveBuffers: device.statistics.liveBuffers,
        adapterDescription,
        software,
      };
    },
    {
      images: data.images.map((v) => ({ ...v, keypoints: [...v.keypoints] })),
      pairs: data.pairs.map((p) => ({ ...p, matches: [...p.matches] })),
      controls,
      localInput: localInput ? packedModel(localInput) : undefined,
      hardware: process.env.TYPECV_HARDWARE_GPU === '1',
    },
    { timeoutMs: 300000 },
  );
}

function assertQuality(model, data) {
  assert.deepEqual(
    model.views.map((v) => v.id),
    data.images.map((v) => v.id),
  );
  assert.equal(model.landmarks.length, data.points.length);
  assert.ok(bundleMetrics(model).rmse < 1e-6);
  for (const view of model.views) {
    const frame = view.id / 10 - 1,
      center = [0, 1, 2].map(
        (i) =>
          -view.pose.R[i] * view.pose.t[0] -
          view.pose.R[i + 3] * view.pose.t[1] -
          view.pose.R[i + 6] * view.pose.t[2],
      );
    assert.ok(Math.hypot(...view.pose.R.map((x, i) => x - Number(i % 4 === 0))) < 1e-6);
    assert.ok(Math.hypot(center[0] - frame, center[1], center[2]) < 1e-6);
  }
  for (const point of model.landmarks) {
    const feature = point.observations[0].featureId,
      visible = data.images.filter((_, frame) => data.visible(frame, feature));
    assert.ok(Math.hypot(...point.xyz.map((x, i) => x - data.points[feature][i] / 0.3)) < 1e-5);
    assert.deepEqual(
      point.observations.map((o) => o.viewId),
      visible.map((v) => v.id),
    );
    for (const observation of point.observations) {
      const image = data.images.find((v) => v.id === observation.viewId);
      assert.equal(observation.featureId, feature);
      assert.deepEqual(observation.pixel, [
        ...image.keypoints.subarray(feature * 2, feature * 2 + 2),
      ]);
    }
  }
}

function assertGpuResult(result, errors, data, reference) {
  assert.deepEqual(errors, []);
  assert.deepEqual(result.gpuErrors, []);
  assert.equal(result.liveBuffers, 0);
  assert.equal(result.backend, 'webgpu');
  assert.equal(result.inputPreserved, true);
  assert.deepEqual(result.unregistered, []);
  assert.deepEqual(result.cpuStages, [
    'track graph, result indexing, data packing and mapper control',
  ]);
  assert.ok(result.stages.every((stage) => stage.tier === 'webgpu' && stage.dispatches > 0));
  assert.equal(
    result.stages.filter((stage) => stage.operation === 'pairVerification').length,
    data.pairs.length,
  );
  assert.equal(
    result.stages.filter((stage) => stage.operation === 'pnp').length,
    data.images.length - 2,
  );
  assert.equal(result.diagnostics.pairVerification.length, data.pairs.length);
  assert.ok(result.diagnostics.meanReprojectionErrorPx < 1e-6);
  assert.ok(result.gpuBundleAdjustments > 0);
  assert.ok(result.diagnostics.bundleAdjustments.every((d) => d.finalCost <= d.initialCost));
  assert.deepEqual(result.diagnostics.registrationOrder, reference.diagnostics.registrationOrder);
  assertQuality(reference.reconstruction, data);
  assertQuality(result.model, data);
  for (let i = 0; i < result.model.views.length; i++) {
    const actual = result.model.views[i].pose,
      expected = reference.reconstruction.views[i].pose;
    for (const key of ['R', 't'])
      assert.ok(Math.hypot(...actual[key].map((x, j) => x - expected[key][j])) < 1e-6);
  }
  const expectedPoints = new Map(
    reference.reconstruction.landmarks.map((p) => [p.observations[0].featureId, p]),
  );
  for (const point of result.model.landmarks) {
    const expected = expectedPoints.get(point.observations[0].featureId);
    assert.ok(Math.hypot(...point.xyz.map((x, j) => x - expected.xyz[j])) < 1e-5);
  }
}

test('WebGPU local subsets reduce actual point work and preserve boundaries, observations and final global refinement', async () => {
  const data = sequenceFixture(true),
    reference = reconstructIncremental(data.images, data.pairs, options),
    localInput = {
      views: reference.reconstruction.views.map((v) =>
        v.id === 60
          ? {
              ...v,
              pose: { R: v.pose.R, t: v.pose.t.map((x, i) => x + [0.01, -0.005, 0.005][i]) },
            }
          : v,
      ),
      landmarks: reference.reconstruction.landmarks.map((p) =>
        p.observations.some((o) => o.viewId === 60)
          ? {
              ...p,
              xyz: p.xyz.map((x, i) => x + 0.005 * Math.sin(p.trackId + i)),
            }
          : p,
      ),
    },
    local = extractLocalBundle(localInput, [60]),
    localReference = bundleAdjust(local.reconstruction, {
      iterations: 5,
      variableViewIds: local.variableViewIds,
      anchorViewId: local.anchorViewId,
      scaleViewId: local.scaleViewId,
    }),
    { result, errors } = await runGpu(
      data,
      {
        ...options,
        localBundleSubset: true,
      },
      localInput,
    );
  assertGpuResult(result, errors, data, reference);
  const bundles = result.stages.filter((stage) => stage.operation === 'bundle'),
    subset = bundles.find((stage) => stage.problem.variableViewIds?.[0] === 40),
    final = bundles.at(-1).problem;
  assert.ok(subset);
  assert.deepEqual(subset.problem, {
    views: 3,
    variableViewIds: [40],
    landmarks: 32,
    observations: 96,
  });
  assert.equal(final.variableViewIds, undefined);
  assert.deepEqual([final.views, final.landmarks, final.observations], [6, 64, 256]);
  assert.ok(subset.problem.landmarks < final.landmarks);
  assert.ok(subset.problem.observations < final.observations);
  assert.equal(result.local.tier, 'webgpu');
  assert.ok(result.local.dispatches > 0);
  for (const key of [
    'inputPreserved',
    'fixedCamerasPreserved',
    'intrinsicsPreserved',
    'unrelatedPreserved',
    'observationsPreserved',
    'solverObservationsPreserved',
  ])
    assert.equal(result.local[key], true, key);
  assert.equal(result.local.diagnostics.observations, 160);
  assert.ok(result.local.diagnostics.finalCost < result.local.diagnostics.initialCost);
  assert.ok(
    Math.abs(result.local.diagnostics.finalCost - localReference.diagnostics.finalCost) < 1e-5,
  );
});
