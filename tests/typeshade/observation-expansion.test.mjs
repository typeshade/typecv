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

async function strictWitnessFixture(seeds) {
  const { renderScene } = await import('../klt-cases.ts');
  const width = 192,
    height = 144,
    camera = { fx: 180, fy: 180, cx: 95.5, cy: 71.5 };
  return seeds.map((keypoints, id) => ({
    id,
    camera,
    width,
    height,
    keypoints: Float64Array.from(keypoints),
    gray: renderScene({ width, height }, { dx: 2 * id, dy: -id }),
  }));
}

const strictWitnessOptions = {
  radius: 3,
  maxSeedsPerFrame: 16,
  klt: { levels: 2, halfWindow: 5 },
};

function assertMeasuredEdges(result, frames, originalPairs) {
  assert.equal(result.newEdges.length, result.diagnostics.addedEdges);
  const originalCounts = new Map(frames.map((f) => [f.id, f.keypoints.length / 2]));
  for (const frame of frames) {
    const current = result.images.find((v) => v.id === frame.id);
    assert.deepEqual(current.keypoints.slice(0, frame.keypoints.length), frame.keypoints);
  }
  for (const raw of originalPairs) {
    const pair = result.candidatePairs.find(
      (p) => p.first === raw.first && p.second === raw.second,
    );
    assert.deepEqual(pair.matches.slice(0, raw.matches.length), raw.matches);
  }
  const keys = new Set();
  for (const edge of result.newEdges) {
    assert.equal(edge.geometry, 'candidate');
    assert.equal(edge.corroboration.status, 'verified');
    assert.notEqual(edge.corroboration.viewId, edge.sourceViewId);
    assert.notEqual(edge.corroboration.viewId, edge.viewId);
    assert.ok(edge.corroboration.featureId < originalCounts.get(edge.corroboration.viewId));
    const frame = result.images.find((v) => v.id === edge.viewId);
    assert.deepEqual(
      edge.targetPixel,
      Array.from(frame.keypoints.slice(2 * edge.featureId, 2 * edge.featureId + 2)),
    );
    assert.equal(edge.createdFeature, edge.featureId >= originalCounts.get(edge.viewId));
    const first = Math.min(edge.sourceViewId, edge.viewId),
      second = Math.max(edge.sourceViewId, edge.viewId),
      a = edge.sourceViewId === first ? edge.sourceFeatureId : edge.featureId,
      b = edge.sourceViewId === first ? edge.featureId : edge.sourceFeatureId,
      key = `${first}:${a}:${second}:${b}`,
      pair = result.candidatePairs.find((p) => p.first === first && p.second === second);
    assert.ok(!keys.has(key), 'one provenance record per newly added pair edge');
    keys.add(key);
    assert.ok(
      Array.from({ length: pair.matches.length / 2 }, (_, i) => i).some(
        (i) => pair.matches[2 * i] === a && pair.matches[2 * i + 1] === b,
      ),
    );
  }
}

test('required witnesses defer untrusted seeds while true retains legacy unavailable admission', async () => {
  const frames = await strictWitnessFixture([[50, 50], []]),
    device = await api.ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    const plain = await api.expandObservationsGpu(device, frames, [], strictWitnessOptions),
      legacy = await api.expandObservationsGpu(device, frames, [], {
        ...strictWitnessOptions,
        multiViewConsistency: true,
      }),
      required = await api.expandObservationsGpu(device, frames, [], {
        ...strictWitnessOptions,
        multiViewConsistency: 'required',
      });
    assert.equal(plain.newObservations.length, 1);
    assert.deepEqual(legacy.images, plain.images);
    assert.deepEqual(legacy.candidatePairs, plain.candidatePairs);
    assert.equal(legacy.newEdges[0].corroboration.status, 'unavailable');
    assert.equal(required.newObservations.length, 0);
    assert.equal(required.newEdges.length, 0);
    assert.equal(required.candidatePairs.length, 0);
    assert.equal(required.diagnostics.attempts, 0, 'unavailable seeds do not spend the KLT budget');
    assert.ok(required.diagnostics.rejected.multiViewUnavailable > 0);
    assert.deepEqual(
      required.images.map((f) => f.keypoints),
      frames.map((f) => f.keypoints),
    );
    assert.equal(device.statistics.liveBytes, 0);
  } finally {
    await device.dispose();
  }
});

test('required mode trusts verified original groups, never raw descriptor candidates', async () => {
  const frames = await strictWitnessFixture([[50, 50], [], [54, 48]]),
    pairs = [
      { first: 0, second: 2, matches: Uint32Array.of(0, 0), distances: Float64Array.of(0.1) },
    ],
    snapshot = frames.map((f) => ({ pixels: f.gray.slice(), keypoints: f.keypoints.slice() })),
    device = await api.ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    const untrusted = await api.expandObservationsGpu(device, frames, pairs, {
      ...strictWitnessOptions,
      multiViewConsistency: 'required',
    });
    assert.equal(untrusted.newEdges.length, 0);
    assertMeasuredEdges(untrusted, frames, pairs);
    const trusted = await api.expandObservationsGpu(device, frames, pairs, {
      ...strictWitnessOptions,
      multiViewConsistency: 'required',
      verifiedPairs: pairs,
    });
    assert.equal(trusted.newObservations.length, 1);
    assert.equal(trusted.newEdges.length, 1);
    assert.equal(trusted.newEdges[0].createdFeature, true);
    assert.equal(trusted.newEdges[0].corroboration.viewId, 2);
    assertMeasuredEdges(trusted, frames, pairs);
    for (let i = 0; i < frames.length; i++) {
      assert.deepEqual(frames[i].gray, snapshot[i].pixels);
      assert.deepEqual(frames[i].keypoints, snapshot[i].keypoints);
    }
    assert.deepEqual(pairs[0].matches, Uint32Array.of(0, 0));
    assert.equal(device.statistics.liveBytes, 0);
  } finally {
    await device.dispose();
  }
});

test('required mode reserves the bounded seed budget for original-witness candidates', async () => {
  const unowned = Array.from({ length: 8 }, (_, i) => [50.1 + i * 0.1, 50.1]).flat(),
    frames = await strictWitnessFixture([[...unowned, 50, 50], [], [54, 48]]),
    pairs = [{ first: 0, second: 2, matches: Uint32Array.of(8, 0) }],
    device = await api.ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    const result = await api.expandObservationsGpu(device, frames, pairs, {
      ...strictWitnessOptions,
      maxSeedsPerFrame: 1,
      multiViewConsistency: 'required',
      verifiedPairs: pairs,
    });
    assert.equal(result.newEdges.length, 1);
    assert.equal(result.newEdges[0].sourceFeatureId, 8);
    assert.ok(result.diagnostics.rejected.multiViewUnavailable >= unowned.length / 2);
    assertMeasuredEdges(result, frames, pairs);
    assert.equal(device.statistics.liveBytes, 0);
  } finally {
    await device.dispose();
  }
});

test('strict provenance includes existing-feature joins without overwriting their measurements', async () => {
  const frames = await strictWitnessFixture([
      [50, 50],
      [52.2, 49],
      [54, 48],
    ]),
    pairs = [{ first: 0, second: 2, matches: Uint32Array.of(0, 0) }],
    device = await api.ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    const result = await api.expandObservationsGpu(device, frames, pairs, {
      ...strictWitnessOptions,
      multiViewConsistency: 'required',
      verifiedPairs: pairs,
    });
    assert.equal(result.newObservations.length, 0);
    assert.equal(result.newEdges.length, 1);
    const edge = result.newEdges[0];
    assert.equal(edge.createdFeature, false);
    assert.deepEqual(edge.targetPixel, [52.2, 49]);
    assert.ok(Math.hypot(edge.measuredPixel[0] - 52, edge.measuredPixel[1] - 49) < 0.1);
    assert.ok(Math.abs(edge.measuredPixel[0] - edge.targetPixel[0]) > 0.1);
    assertMeasuredEdges(result, frames, pairs);
    const distant = await strictWitnessFixture([
        [50, 50],
        [52.6, 49],
        [54, 48],
      ]),
      deferred = await api.expandObservationsGpu(device, distant, pairs, {
        ...strictWitnessOptions,
        multiViewConsistency: 'required',
        verifiedPairs: pairs,
      });
    assert.equal(deferred.newEdges.length, 0);
    assert.equal(deferred.newObservations.length, 0);
    assert.ok(deferred.diagnostics.rejected.multiViewJoin > 0);
    assert.equal(device.statistics.liveBytes, 0);
  } finally {
    await device.dispose();
  }
});

test('new descendants cannot witness another strict edge and reverse edges retain one provenance row', async () => {
  const frames = await strictWitnessFixture([[50, 50], [52, 49], [], []]),
    pairs = [{ first: 0, second: 1, matches: Uint32Array.of(0, 0) }],
    device = await api.ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    const result = await api.expandObservationsGpu(device, frames, pairs, {
      ...strictWitnessOptions,
      multiViewConsistency: 'required',
      verifiedPairs: pairs,
    });
    assert.equal(result.newEdges.length, 2);
    assert.equal(result.newEdges.find((e) => e.viewId === 3).corroboration.viewId, 1);
    assertMeasuredEdges(result, frames, pairs);
    const reverseFrames = await strictWitnessFixture([[], [52, 49], [54, 48]]),
      reversePairs = [{ first: 1, second: 2, matches: Uint32Array.of(0, 0) }],
      reverse = await api.expandObservationsGpu(device, reverseFrames, reversePairs, {
        ...strictWitnessOptions,
        multiViewConsistency: 'required',
        verifiedPairs: reversePairs,
      });
    assert.equal(reverse.newEdges.length, 1);
    assert.equal(reverse.newEdges[0].sourceViewId, 1);
    assert.equal(reverse.newEdges[0].viewId, 0);
    assertMeasuredEdges(reverse, reverseFrames, reversePairs);
    assert.equal(device.statistics.liveBytes, 0);
  } finally {
    await device.dispose();
  }
});

test('observation expansion validates the independent-witness mode before doing tracking', async () => {
  const frames = await strictWitnessFixture([[50, 50], []]),
    device = await api.ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    for (const multiViewConsistency of [null, 0, 1, 'yes', {}, []])
      await assert.rejects(
        api.expandObservationsGpu(device, frames, [], { multiViewConsistency }),
        /Invalid observation expansion bounds/,
      );
    assert.equal(device.statistics.liveBytes, 0);
    assert.equal(device.statistics.reads, 0);
  } finally {
    await device.dispose();
  }
});
