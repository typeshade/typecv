import test from 'node:test';
import assert from 'node:assert/strict';
import * as api from '../../dist/gpu.js';
import { runExpansionCases } from './observation-expansion-cases.mjs';
import { expansionFixture, assertExpansion } from './observation-expansion-fixture.mjs';

test('image-based observation expansion on TypeShade CPU recovers missed coordinates in both directions', async () => {
  const device = await api.ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    const input = expansionFixture();
    const result = await runExpansionCases(api, device, input);
    assertExpansion(result, input);
    console.log(JSON.stringify(result.expanded.diagnostics));
    await assert.rejects(
      api.expandObservationsGpu(
        device,
        [{ ...input.frames[0], gray: Float32Array.of(1), keypoints: new Float64Array() }],
        [],
      ),
      /grayscale/,
    );
  } finally {
    await device.dispose();
  }
});

test('expansion refines finite-map predictions without storing their projected coordinates', async () => {
  // Supplied synthetic model for API validation only, not an RGB-only reconstruction benchmark.
  const { renderScene, gridPoints } = await import('../klt-cases.ts');
  const { identityPose } = await import('../../dist/index.js');
  const device = await api.ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    const width = 192,
      height = 144,
      camera = { fx: 180, fy: 180, cx: 95.5, cy: 71.5 },
      points = Float64Array.from(gridPoints(width, height, 45, 40, 4, 3)),
      a = { id: 0, camera, width, height, keypoints: points, gray: renderScene({ width, height }) },
      b = {
        id: 1,
        camera,
        width,
        height,
        keypoints: new Float64Array(),
        gray: renderScene({ width, height }, { dx: 24, dy: -17 }),
      },
      pose = identityPose(),
      model = {
        views: [
          { id: 0, camera, pose },
          {
            id: 1,
            camera,
            pose: { R: pose.R, t: Float64Array.of((23 * 5) / 180, (-16 * 5) / 180, 0) },
          },
        ],
        landmarks: Array.from({ length: points.length / 2 }, (_, k) => ({
          trackId: k,
          xyz: Float64Array.of(
            ((points[2 * k] - camera.cx) * 5) / camera.fx,
            ((points[2 * k + 1] - camera.cy) * 5) / camera.fy,
            5,
          ),
          observations: [{ viewId: 0, featureId: k, pixel: [points[2 * k], points[2 * k + 1]] }],
        })),
      };
    const result = await api.expandObservationsGpu(device, [a, b], [], {
      initialModel: model,
      klt: { levels: 1, halfWindow: 5 },
    });
    assert.equal(result.newObservations.length, points.length / 2);
    for (const o of result.newObservations) {
      assert.equal(o.prediction, 'finite');
      assert.ok(
        Math.hypot(
          o.measuredPixel[0] - points[2 * o.sourceFeatureId] - 24,
          o.measuredPixel[1] - points[2 * o.sourceFeatureId + 1] + 17,
        ) < 0.1,
      );
    }
    const wrong = {
      ...model,
      views: [
        model.views[0],
        { ...model.views[1], pose: { R: pose.R, t: Float64Array.of(0, 0, -10) } },
      ],
    };
    const withheld = await api.expandObservationsGpu(device, [a, b], [], {
      initialModel: wrong,
      klt: { levels: 1, halfWindow: 5 },
    });
    assert.equal(withheld.newObservations.length, 0);
    assert.equal(withheld.diagnostics.rejected.invalidPrediction, points.length / 2);
    assert.equal(device.statistics.liveBytes, 0);
  } finally {
    await device.dispose();
  }
});

test('expansion defers conflicting established owners and enforces an append budget', async () => {
  const { renderScene } = await import('../klt-cases.ts');
  const device = await api.ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    const input = expansionFixture(),
      f = input.frames[0],
      camera = f.camera;
    const frames = [
      { ...f, camera, keypoints: Float64Array.of(50, 50), gray: renderScene(f) },
      { ...f, id: 1, keypoints: Float64Array.of(52, 49), gray: renderScene(f, { dx: 2, dy: -1 }) },
      {
        ...f,
        id: 2,
        keypoints: Float64Array.of(54, 48, 110, 80),
        gray: renderScene(f, { dx: 4, dy: -2 }),
      },
    ];
    const verifiedPairs = [
      { first: 0, second: 2, matches: Uint32Array.of(0, 0) },
      { first: 1, second: 2, matches: Uint32Array.of(0, 1) },
    ];
    const result = await api.expandObservationsGpu(device, frames, verifiedPairs, {
      verifiedPairs,
      klt: { levels: 3, halfWindow: 5 },
    });
    assert.ok(result.diagnostics.rejected.ownership >= 2);
    assert.ok(!result.candidatePairs.some((p) => p.first === 0 && p.second === 1));
    const budget = await api.expandObservationsGpu(
      device,
      input.frames.map((f) => ({
        ...f,
        gray: Float32Array.from(f.gray),
        keypoints: Float64Array.from(f.keypoints),
      })),
      [],
      {
        radius: 2,
        maxNewPerFrame: 1,
        klt: { levels: 3, halfWindow: 5 },
      },
    );
    assert.ok(
      budget.images.every(
        (f, i) => f.keypoints.length / 2 <= input.frames[i].keypoints.length / 2 + 1,
      ),
    );
    assert.ok(budget.diagnostics.rejected.capacity > 0);
    assert.equal(device.statistics.liveBytes, 0);
  } finally {
    await device.dispose();
  }
});

test('TypeShade CPU refines measured additions in the original map', async () => {
  const core = await import('../../dist/index.js');
  const { runRefinementCase } = await import('./observation-refinement-case.mjs');
  const { refinementInput, assertRefinement } =
    await import('./observation-refinement-fixture.mjs');
  const device = await api.ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    assertRefinement(await runRefinementCase(api, core, device, refinementInput()));
  } finally {
    await device.dispose();
  }
});

test('relative bearing rotations guide expansion before a finite map exists', async () => {
  // Supplied rotation hypothesis tests the initializer; it is never accepted as a 3D point.
  const { renderScene, gridPoints, affine } = await import('../klt-cases.ts');
  const { rotationExp } = await import('../../dist/index.js');
  const device = await api.ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    const width = 192,
      height = 144,
      camera = { fx: 180, fy: 180, cx: 95.5, cy: 71.5 },
      points = Float64Array.from(gridPoints(width, height, 40, 35, 4, 3)),
      degrees = 12,
      warp = { degrees },
      frames = [
        { id: 0, camera, width, height, keypoints: points, gray: renderScene({ width, height }) },
        {
          id: 1,
          camera,
          width,
          height,
          keypoints: new Float64Array(),
          gray: renderScene({ width, height }, warp),
        },
      ],
      rotations = [
        {
          first: 0,
          second: 1,
          firstCamera: camera,
          secondCamera: camera,
          rotation: {
            R: rotationExp([0, 0, (degrees * Math.PI) / 180]),
            inlierRows: new Uint32Array(),
            invertibleCount: 12,
            meanReprojectionErrorPx: 0,
            spreadDeg: [5, 5],
            translation: 'unobservable',
            depth: 'not-estimated',
          },
        },
      ];
    const result = await api.expandObservationsGpu(device, frames, [], {
      rotations,
      klt: { levels: 2, halfWindow: 5 },
    });
    assert.ok(result.newObservations.length >= 8, JSON.stringify(result.diagnostics));
    for (const o of result.newObservations) {
      assert.equal(o.prediction, 'rotation');
      assert.equal(o.geometry, 'candidate');
      const p = affine(width, height, warp).forward(
        points[2 * o.sourceFeatureId],
        points[2 * o.sourceFeatureId + 1],
      );
      // The current template translates without rotating: at 12 degrees its measured
      // centre is approximate. Require half-pixel accuracy, not equality to the projection.
      assert.ok(Math.hypot(o.measuredPixel[0] - p[0], o.measuredPixel[1] - p[1]) < 0.5);
    }
    assert.equal(device.statistics.liveBytes, 0);
  } finally {
    await device.dispose();
  }
});

test('geometry patch warps improve measured coordinates for calibrated roll and zoom', async () => {
  // Supplied calibrated rotation is a unit-test initializer, not RGB reconstruction truth input.
  const { renderScene, gridPoints, affine } = await import('../klt-cases.ts');
  const { rotationExp } = await import('../../dist/index.js');
  const device = await api.ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    const width = 192,
      height = 144,
      camera = { fx: 180, fy: 180, cx: 95.5, cy: 71.5 },
      camera2 = { ...camera, fx: 207, fy: 207 },
      warp = { degrees: 12, scale: 1.15 },
      pts = Float64Array.from(gridPoints(width, height, 40, 35, 4, 3)),
      frames = [
        { id: 0, width, height, camera, keypoints: pts, gray: renderScene({ width, height }) },
        {
          id: 1,
          width,
          height,
          camera: camera2,
          keypoints: new Float64Array(),
          gray: renderScene({ width, height }, warp),
        },
      ],
      rotations = [
        {
          first: 0,
          second: 1,
          firstCamera: camera,
          secondCamera: camera2,
          rotation: {
            R: rotationExp([0, 0, (12 * Math.PI) / 180]),
            inlierRows: new Uint32Array(),
            invertibleCount: 12,
            meanReprojectionErrorPx: 0,
            spreadDeg: [5, 5],
            translation: 'unobservable',
            depth: 'not-estimated',
          },
        },
      ],
      options = {
        rotations,
        klt: { levels: 2, halfWindow: 5, epsilonPx: 0.005, maxIterations: 30 },
      },
      plain = await api.expandObservationsGpu(device, frames, [], options),
      warped = await api.expandObservationsGpu(device, frames, [], {
        ...options,
        geometryPatchWarp: true,
        pyramidCache: true,
      });
    const error = (o) => {
      const p = affine(width, height, warp).forward(
        pts[2 * o.sourceFeatureId],
        pts[2 * o.sourceFeatureId + 1],
      );
      return Math.hypot(o.measuredPixel[0] - p[0], o.measuredPixel[1] - p[1]);
    };
    assert.equal(warped.newObservations.length, 12);
    assert.ok(Math.max(...warped.newObservations.map(error)) < 0.05);
    assert.ok(Math.max(...plain.newObservations.map(error)) > 0.3);
    assert.ok(
      warped.newObservations.every((o) => o.patchWarp.length === 4 && o.geometry === 'candidate'),
    );
    assert.equal(device.statistics.liveBytes, 0);
  } finally {
    await device.dispose();
  }
});
