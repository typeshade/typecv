import test from 'node:test';
import assert from 'node:assert/strict';
import { measureHistory, refineHistory } from '../src/reconstruction/history-refinement.ts';
import { bundleAdjust } from '../src/reconstruction/bundle-adjustment.ts';
import { cameraWidth } from '../src/reconstruction/focal-search.ts';
import { refinedHistoryWindow } from '../src/reconstruction/history-refinement.ts';
import { scene } from './scene.ts';
import { cameraCenter, rotationExp, transformPoint } from '../src/geometry/pose.ts';
import { cameraProjection } from '../src/geometry/camera.ts';
import { seededRandom } from '../src/geometry/numerics.ts';
import {
  focalGrid,
  focalSearchControls,
  historyFocalSeeds,
} from '../src/reconstruction/focal-search.ts';

test('later rotations recover true depth after an exactly ambiguous parallel-camera prefix', async () => {
  const rng = seededRandom(49217),
    width = 800,
    height = 600,
    trueRatio = 0.8,
    earlyRatio = 0.3,
    compression = earlyRatio / trueRatio,
    camera = { fx: trueRatio * width, fy: trueRatio * width, cx: 399.5, cy: 299.5 },
    truthPoints = Array.from({ length: 64 }, () =>
      Float64Array.of((rng() - 0.5) * 3, (rng() - 0.5) * 2, 6 + 3 * rng()),
    ),
    truthViews = Array.from({ length: 12 }, (_, id) => {
      const turn = Math.max(0, id - 4),
        R = rotationExp([-0.04 * turn, 0.03 * turn, 0.01 * turn]),
        C = Float64Array.of(0.13 * id, 0.16 * id, 0.1 * id);
      return {
        id,
        camera,
        pose: { R, t: transformPoint({ R, t: new Float64Array(3) }, C).map((x) => -x) },
        C,
      };
    }),
    input = {
      views: truthViews.map((v) => {
        const C = Float64Array.of(v.C[0]!, v.C[1]!, v.C[2]! * compression);
        return {
          id: v.id,
          camera: { ...camera, fx: earlyRatio * width, fy: earlyRatio * width },
          pose: {
            R: v.pose.R.slice(),
            t: transformPoint({ R: v.pose.R, t: new Float64Array(3) }, C).map((x) => -x),
          },
        };
      }),
      landmarks: truthPoints.map((p, trackId) => ({
        trackId,
        xyz: Float64Array.of(p[0]!, p[1]!, p[2]! * compression),
        observations: truthViews.map((v) => ({
          viewId: v.id,
          featureId: trackId,
          pixel: cameraProjection(camera, transformPoint(v.pose, p))!.pixel,
        })),
      })),
    };
  for (const point of input.landmarks)
    for (const o of point.observations)
      assert.ok(o.pixel[0] >= 0 && o.pixel[0] < width && o.pixel[1] >= 0 && o.pixel[1] < height);
  const prefix = {
    views: input.views.slice(0, 5),
    landmarks: input.landmarks.map((p) => ({ ...p, observations: p.observations.slice(0, 5) })),
  };
  // Parallel axes permit a wrong focal and compressed Z with perfect image agreement.
  assert.ok(measureHistory(prefix).meanErrorPx < 1e-10);
  const grid = focalGrid(focalSearchControls(true)!),
    ratios = historyFocalSeeds(
      {
        ratio: earlyRatio,
        gridRatio: earlyRatio,
        views: [0, 1, 2, 3, 4],
        candidates: grid.map((ratio) => ({
          ratio,
          views: 5,
          observations: 320,
          rmsePx: 0,
          score: 0,
        })),
      },
      earlyRatio,
    ),
    result = await refineHistory(
      input,
      { iterations: 32, refineSharedFocal: true, commonFocalRatios: ratios },
      async (model, options) => bundleAdjust(model, options),
    );
  // One fixed seed gauge, identical before/after; no per-point or per-frame alignment.
  const baseline = (
      a: Pick<(typeof truthViews)[number], 'pose'>,
      b: Pick<(typeof truthViews)[number], 'pose'>,
    ) => Math.hypot(...cameraCenter(b.pose).map((x, i) => x - cameraCenter(a.pose)[i]!)),
    scale = baseline(truthViews[0]!, truthViews[11]!) / baseline(input.views[0]!, input.views[11]!);
  const pointRmse = (points: readonly { readonly xyz: ArrayLike<number> }[]) =>
    Math.sqrt(
      points.reduce(
        (sum, p, i) =>
          sum +
          Array.from(p.xyz).reduce((s, x, j) => s + (scale * x - truthPoints[i]![j]!) ** 2, 0),
        0,
      ) / points.length,
    );
  assert.ok(pointRmse(input.landmarks) > 1);
  assert.ok(pointRmse(result.reconstruction.landmarks) < 1e-5);
  assert.ok(Math.abs(result.reconstruction.views[0]!.camera.fx / width - trueRatio) < 1e-6);
  assert.equal(result.after.observations, 768);
  assert.deepEqual(
    result.reconstruction.landmarks.map((p) => p.observations),
    input.landmarks.map((p) => p.observations),
  );
});

test('archive measurement includes outliers outside the final local window', () => {
  const { reconstruction } = scene({ count: 4, views: 10 });
  const point = reconstruction.landmarks[0]!,
    observation = point.observations[0]!,
    model = {
      ...reconstruction,
      landmarks: [
        {
          ...point,
          observations: [
            { ...observation, pixel: [observation.pixel[0] + 10, observation.pixel[1]] as const },
            ...point.observations.slice(1),
          ],
        },
        ...reconstruction.landmarks.slice(1),
      ],
    };
  const measured = measureHistory(model);
  assert.equal(measured.observations, 40);
  assert.ok(Math.abs(measured.meanErrorPx - 0.25) < 1e-12);
  assert.ok(Math.abs(measured.cost - (3 * 10 - 2.25)) < 1e-10);
});

test('a complete camera/point/focal solve repairs a mixed archive without removing observations', async () => {
  const { reconstruction } = scene({ count: 40, views: 10, noise: 0.05 }),
    model = {
      views: reconstruction.views.map((v, i) => ({
        ...v,
        camera: {
          ...v.camera,
          fx: v.camera.fx * (i < 2 ? 0.97 : 1),
          fy: v.camera.fy * (i < 2 ? 0.97 : 1),
        },
      })),
      landmarks: reconstruction.landmarks.map((p, i) => ({
        ...p,
        xyz: p.xyz.map((x, k) => x + (k === 2 ? 0.08 * Math.sin(i) : 0.01)),
      })),
    },
    snapshot = structuredClone(model),
    result = await refineHistory(
      model,
      {
        iterations: 20,
        refineSharedFocal: true,
        commonFocalRatios: [
          reconstruction.views[0]!.camera.fx / cameraWidth(reconstruction.views[0]!.camera),
        ],
      },
      async (input, options) => bundleAdjust(input, options),
    );
  assert.ok(result.after.cost < result.before.cost * 0.1);
  assert.equal(result.after.observations, result.before.observations);
  assert.deepEqual(
    result.reconstruction.landmarks.map((p) => p.observations),
    model.landmarks.map((p) => p.observations),
  );
  assert.deepEqual(model, snapshot);
  assert.equal(new Set(result.reconstruction.views.map((v) => v.camera.fx)).size, 1);
});

test('an unrefined focal overwrite is rolled back against the original full observation objective', async () => {
  const { reconstruction } = scene({ count: 8, views: 10 }),
    result = await refineHistory(
      reconstruction,
      { iterations: 1, commonFocalRatios: [0.3] },
      async (input) => ({
        reconstruction: input,
        diagnostics: {
          initialCost: 0,
          finalCost: 0,
          initialRmsePx: 0,
          finalRmsePx: 0,
          acceptedSteps: 0,
          rejectedSteps: 0,
          observations: 80,
          baseline: 1,
        },
      }),
    );
  assert.deepEqual(result.reconstruction, reconstruction);
  assert.equal(result.after.cost, result.before.cost);
});

test('a resident-floor refusal preserves geometry, observations and the real whole-output metric', async () => {
  const { reconstruction: base } = scene({ count: 6, views: 7 });
  const reconstruction = {
    views: Array.from({ length: 70 }, (_, i) => ({ ...base.views[i % 7]!, id: i })),
    landmarks: base.landmarks.map((p) => ({
      ...p,
      observations: Array.from({ length: 70 }, (_, i) => ({
        ...p.observations[i % 7]!,
        viewId: i,
      })),
    })),
  };
  let suppliedViews = 0;
  const result = await refineHistory(
    reconstruction,
    { iterations: 2, commonFocalRatios: [0.3] },
    async (input) => {
      suppliedViews = input.views.length;
      throw new RangeError('GPU BA needs 100 bytes, more than the working budget of 50 bytes');
    },
  );
  assert.equal(suppliedViews, 70);
  assert.equal(result.deferred, true);
  assert.deepEqual(result.reconstruction, reconstruction);
  assert.equal(result.after.observations, 420);
});

test('disconnected components retain their calibration and defer shared focal restarts', async () => {
  const { reconstruction } = scene({ count: 8, views: 4 }),
    model = {
      ...reconstruction,
      landmarks: reconstruction.landmarks.map((p, i) => ({
        ...p,
        observations: i < 4 ? p.observations.slice(0, 2) : p.observations.slice(2),
      })),
    };
  let calls = 0;
  const result = await refineHistory(
    model,
    { iterations: 1, refineSharedFocal: true, commonFocalRatios: [0.3, 0.8] },
    async (input, options) => {
      calls++;
      assert.equal(options.refineSharedFocal, false);
      for (const view of input.views) assert.equal(view.camera.fx, 850);
      return bundleAdjust(input, { ...options, iterations: 0 });
    },
  );
  assert.equal(calls, 2);
  assert.equal(result.deferred, true);
  assert.deepEqual(result.reconstruction, model);
});

test('a rotation-only first camera pair uses a later valid gauge and device failures propagate', async () => {
  const { reconstruction } = scene({ count: 8, views: 3 }),
    model = {
      ...reconstruction,
      views: reconstruction.views.map((v, i) =>
        i === 1 ? { ...v, pose: reconstruction.views[0]!.pose } : v,
      ),
    };
  let gauge: number | undefined;
  await refineHistory(model, { iterations: 1 }, async (input, options) => {
    gauge = options.scaleViewId;
    return bundleAdjust(input, { ...options, iterations: 0 });
  });
  assert.equal(gauge, 30);
  for (const error of [
    new DOMException('Cancelled', 'AbortError'),
    new Error('WebGPU device lost'),
  ])
    await assert.rejects(
      refineHistory(model, { iterations: 1 }, async () => {
        throw error;
      }),
      (caught) => caught === error,
    );
});

test('window synchronization keeps native track ids while carrying solved archive coordinates', () => {
  const { reconstruction } = scene({ count: 4, views: 10 }),
    local = {
      views: reconstruction.views.slice(-8),
      landmarks: reconstruction.landmarks.map((p) => ({
        ...p,
        observations: p.observations.slice(-8),
      })),
    },
    full = {
      ...reconstruction,
      landmarks: reconstruction.landmarks.map((p) => ({
        ...p,
        trackId: p.trackId + 100,
        xyz: p.xyz.map((x) => x + 0.001),
      })),
    },
    synced = refinedHistoryWindow(local, full);
  assert.deepEqual(
    synced.landmarks.map((p) => p.trackId),
    local.landmarks.map((p) => p.trackId),
  );
  assert.deepEqual(
    synced.landmarks.map((p) => p.xyz),
    full.landmarks.map((p) => p.xyz),
  );
  assert.deepEqual(
    synced.landmarks.map((p) => p.observations),
    local.landmarks.map((p) => p.observations),
  );
  synced.landmarks[0]!.xyz.fill(999);
  assert.notEqual(full.landmarks[0]!.xyz[0], 999);
});

test('invalid archived cheirality is reported without silently pruning or entering BA', async () => {
  const { reconstruction } = scene({ count: 4, views: 10 }),
    model = {
      ...reconstruction,
      landmarks: reconstruction.landmarks.map((p) => ({
        ...p,
        xyz: Float64Array.of(p.xyz[0]!, p.xyz[1]!, -100),
      })),
    };
  let calls = 0;
  const result = await refineHistory(model, { iterations: 8 }, async () => {
    calls++;
    throw new Error('Must not solve invalid input');
  });
  assert.equal(calls, 0);
  assert.equal(result.deferred, true);
  assert.equal(result.after.meanErrorPx, Infinity);
  assert.deepEqual(result.reconstruction, model);
});
