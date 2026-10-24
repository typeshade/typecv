import assert from 'node:assert/strict';
import { mulberry32, renderScene } from '../klt-cases.ts';

// This is a tracking/API experiment with a supplied, slightly inexact map. It is not
// an RGB-only SfM benchmark. Analytic plane geometry stays in the assertion-side truth.
export function surfaceFixture() {
  const width = 192,
    height = 144,
    camera = { fx: 180, fy: 180, cx: 95.5, cy: 71.5 },
    plane = [0.12, 0.04, 0.2],
    targetTranslation = [0.3, -0.05, -1],
    random = mulberry32(71),
    spots = Array.from({ length: 220 }, () => ({
      x: random() * (width + 100) - 50,
      y: random() * (height + 100) - 50,
      sigma: 2.5 + 4 * random(),
      weight: (random() - 0.5) * 0.6,
    })),
    texture = (x, y) => {
      let value = 0.5 + 0.04 * Math.sin(x * 0.49) * Math.sin(y * 0.53);
      for (const spot of spots) {
        const d = (x - spot.x) ** 2 + (y - spot.y) ** 2;
        if (d < 25 * spot.sigma ** 2) value += spot.weight * Math.exp(-d / (2 * spot.sigma ** 2));
      }
      return Math.max(0, Math.min(1, value));
    },
    xyz = (pixel) => {
      const x = (pixel[0] - camera.cx) / camera.fx,
        y = (pixel[1] - camera.cy) / camera.fy,
        z = 1 / (plane[0] * x + plane[1] * y + plane[2]);
      return [x * z, y * z, z];
    },
    project = (point, translation) => [
      (camera.fx * (point[0] + translation[0])) / (point[2] + translation[2]) + camera.cx,
      (camera.fy * (point[1] + translation[1])) / (point[2] + translation[2]) + camera.cy,
    ],
    render = (translation) =>
      Array.from({ length: width * height }, (_, k) => {
        const x = ((k % width) - camera.cx) / camera.fx,
          y = (Math.floor(k / width) - camera.cy) / camera.fy,
          z =
            (1 + plane.reduce((sum, v, i) => sum + v * translation[i], 0)) /
            (plane[0] * x + plane[1] * y + plane[2]),
          point = [x * z - translation[0], y * z - translation[1], z - translation[2]];
        return texture(
          (camera.fx * point[0]) / point[2] + camera.cx,
          (camera.fy * point[1]) / point[2] + camera.cy,
        );
      }),
    pixels = Array.from({ length: 7 * 9 }, (_, k) => [
      40.17 + 14 * (k % 9),
      30.23 + 14 * Math.floor(k / 9),
    ]),
    keypoints = pixels.flat(),
    identity = [1, 0, 0, 0, 1, 0, 0, 0, 1],
    estimatedTranslation = targetTranslation.map((v, i) => v + [0.003, -0.002, 0][i]);
  return {
    surface: {
      frames: [
        { id: 0, width, height, camera, gray: render([0, 0, 0]), keypoints },
        { id: 1, width, height, camera, gray: render(targetTranslation), keypoints: [] },
      ],
      model: {
        views: [
          { id: 0, camera, pose: { R: identity, t: [0, 0, 0] } },
          { id: 1, camera, pose: { R: identity, t: estimatedTranslation } },
        ],
        landmarks: pixels.map((pixel, k) => ({
          trackId: k,
          xyz: xyz(pixel),
          observations: [{ viewId: 0, featureId: k, pixel }],
        })),
      },
    },
    truth: pixels.map((pixel) => ({
      pixel: project(xyz(pixel), targetTranslation),
      prediction: project(xyz(pixel), estimatedTranslation),
      interior: pixel[0] > 50 && pixel[0] < 145 && pixel[1] > 40 && pixel[1] < 100,
    })),
  };
}

export function consensusFixture() {
  const width = 192,
    height = 144,
    camera = { fx: 180, fy: 180, cx: 95.5, cy: 71.5 },
    pixels = [
      [60.17, 50.23],
      [95.17, 50.23],
      [60.17, 90.23],
      [95.17, 90.23],
    ],
    frames = [0, 1, 2].map((id) => ({
      id,
      width,
      height,
      camera,
      gray: Array.from(renderScene({ width, height }, { dx: id * 2, dy: -id })),
      keypoints: id === 1 ? [] : pixels.flatMap(([x, y]) => [x + id * 2, y - id]),
    }));
  const identity = [1, 0, 0, 0, 1, 0, 0, 0, 1],
    geometryFrames = [
      frames[0],
      frames[1],
      {
        ...frames[2],
        gray: Array(width * height).fill(0.5),
        keypoints: pixels.flatMap(([x, y]) => [x + 36, y]),
      },
      {
        ...frames[0],
        id: 3,
        gray: Array.from(renderScene({ width, height }, { dx: 6, dy: -3 })),
        keypoints: pixels.flatMap(([x, y]) => [x + 6, y - 3]),
      },
    ],
    geometryModel = {
      views: [
        [0, 0, 0],
        [(2 * 5) / 180, -5 / 180, 0],
        [1, 0, 0],
        [(6 * 5) / 180, (-3 * 5) / 180, 0],
      ].map((t, id) => ({
        id,
        camera,
        pose: { R: identity, t },
      })),
      landmarks: pixels.map(([x, y], trackId) => ({
        trackId,
        xyz: [((x - camera.cx) * 5) / camera.fx, ((y - camera.cy) * 5) / camera.fy, 5],
        observations: [0, 2, 3].map((viewId) => ({
          viewId,
          featureId: trackId,
          pixel: geometryFrames[viewId].keypoints.slice(2 * trackId, 2 * trackId + 2),
        })),
      })),
    };
  return {
    frames,
    // The temporally closer alternate (2) has a much larger viewing-angle change
    // and no usable texture. Geometry should select the original near-view (3).
    geometryChoice: { frames: geometryFrames, model: geometryModel },
    // A caller-verified track can still contain sub-threshold keypoint localization
    // errors. Both patches remain distinctive, trackable RGB evidence; their
    // independently measured target centres disagree instead of failing texture.
    displacedTarget: frames.map((frame) =>
      frame.id !== 1
        ? frame
        : {
            ...frame,
            keypoints: pixels.flatMap(([x, y]) => [x + 2.7, y - 1]),
          },
    ),
    displacedAnchor: frames.map((frame) =>
      frame.id !== 2
        ? frame
        : {
            ...frame,
            keypoints: frame.keypoints.map((v, k) => v + (k % 2 === 0 ? 1.8 : 0)),
          },
    ),
    corrupted: frames.map((frame) =>
      frame.id !== 2
        ? frame
        : {
            ...frame,
            gray: frame.gray.map((v, k) =>
              pixels.some(
                ([x, y]) =>
                  Math.abs((k % width) - x - 4) < 15 &&
                  Math.abs(Math.floor(k / width) - y + 2) < 15,
              )
                ? 0.5
                : v,
            ),
          },
    ),
    verifiedPairs: [{ first: 0, second: 2, matches: pixels.flatMap((_, k) => [k, k]) }],
    truth: pixels.map(([x, y]) => [x + 2, y - 1]),
  };
}

export function assertSurface(result, input) {
  assert.equal(result.liveBytes, 0);
  const reference = new Map(result.rotation.observations.map((o) => [o.sourceFeatureId, o])),
    measured = result.surface.observations.filter(
      (o) => o.sourceViewId === 0 && input.truth[o.sourceFeatureId].interior,
    ),
    errors = [],
    referenceErrors = [];
  assert.ok(measured.length >= 12, JSON.stringify(result.surface.diagnostics));
  for (const o of measured) {
    const truth = input.truth[o.sourceFeatureId],
      error = Math.hypot(o.measuredPixel[0] - truth.pixel[0], o.measuredPixel[1] - truth.pixel[1]);
    assert.ok(error < 0.12, `surface feature${o.sourceFeatureId}: ${error}px`);
    assert.ok(o.surfaceSupport?.points >= 6, 'surface proposal retains its estimated-map support');
    assert.ok(o.surfaceSupport.maxRelativeDepthResidual <= 0.02);
    assert.equal(o.geometry, 'candidate');
    assert.ok(
      o.patchWarp && o.patchWarp[0] > 1.1,
      'translated surface scale must affect the patch',
    );
    assert.ok(
      Math.hypot(
        o.measuredPixel[0] - truth.prediction[0],
        o.measuredPixel[1] - truth.prediction[1],
      ) > 0.05,
      'slightly inexact projected centre is not copied as an observation',
    );
    errors.push(error);
    const old = reference.get(o.sourceFeatureId);
    if (old)
      referenceErrors.push({
        before: Math.hypot(
          old.measuredPixel[0] - truth.pixel[0],
          old.measuredPixel[1] - truth.pixel[1],
        ),
        after: error,
      });
  }
  assert.ok(referenceErrors.length >= 8, 'same source seeds must survive in both methods');
  const average = (values) => values.reduce((sum, v) => sum + v, 0) / values.length;
  assert.ok(
    average(referenceErrors.map((e) => e.after)) <
      average(referenceErrors.map((e) => e.before)) * 0.5,
    JSON.stringify(referenceErrors),
  );
  return {
    compared: referenceErrors.length,
    surfaceMaxErrorPx: Math.max(...errors),
    referenceMeanErrorPx: average(referenceErrors.map((e) => e.before)),
    surfaceMeanErrorPx: average(referenceErrors.map((e) => e.after)),
  };
}

export function assertConsensus(result, input) {
  assert.equal(result.liveBytes, 0);
  assert.equal(result.good.observations.length, input.truth.length);
  for (const o of result.good.observations) {
    const truth = input.truth[o.sourceFeatureId];
    assert.equal(o.corroboration?.status, 'verified');
    assert.equal(o.corroboration.viewId, 2);
    assert.equal(o.corroboration.featureId, o.sourceFeatureId);
    assert.ok(o.corroboration.disagreementPx <= 0.5);
    assert.ok(o.corroboration.ncc >= 0.8);
    assert.ok(o.corroboration.forwardBackwardErrorPx <= 0.5);
    assert.ok(Math.hypot(o.measuredPixel[0] - truth[0], o.measuredPixel[1] - truth[1]) < 0.1);
  }
  assert.ok(result.bad.diagnostics.corroboration.rejected >= input.truth.length);
  assert.equal(
    result.bad.observations.length,
    0,
    'another original frame does not confirm a corrupted patch',
  );
  assert.equal(
    result.displaced.observations.length,
    0,
    'two trackable original patches with inconsistent endpoint locations are rejected',
  );
  assert.ok(
    result.displaced.diagnostics.rejected.multiViewDisagreement >= input.truth.length,
    'the independent patch passed image quality but disagreed spatially',
  );
  assert.equal(result.joined.observations.length, 0);
  assert.ok(
    result.joined.diagnostics.rejected.multiViewJoin >= input.truth.length,
    'joining a nearby target keypoint must preserve independent endpoint agreement',
  );
  assert.deepEqual(
    result.joined.images.find((f) => f.id === 1).keypoints,
    input.displacedTarget[1].keypoints,
    'original target coordinates stay immutable',
  );
  assert.ok(
    result.joined.pairs.every((p) => p.first !== 1 && p.second !== 1),
    'the conflicting canonical coordinates are not silently admitted as track edges',
  );
  assert.equal(
    result.twoView.observations.length,
    input.truth.length,
    'two-view candidates remain available without a third witness',
  );
  assert.equal(result.geometryChoice.observations.length, input.truth.length);
  for (const o of result.geometryChoice.observations) {
    assert.equal(o.sourceViewId, 0);
    assert.equal(o.viewId, 1);
    assert.equal(o.corroboration?.status, 'verified');
    assert.equal(o.corroboration.viewId, 3, 'view geometry takes priority over timeline proximity');
    const truth = input.truth[o.sourceFeatureId];
    assert.ok(Math.hypot(o.measuredPixel[0] - truth[0], o.measuredPixel[1] - truth[1]) < 0.1);
  }
  assert.equal(result.twoView.diagnostics.corroboration.unavailable, input.truth.length);
  for (const o of result.twoView.observations)
    assert.deepEqual(o.corroboration, { status: 'unavailable' });
}
