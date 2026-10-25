import test from 'node:test';
import assert from 'node:assert/strict';
import {
  mvsNormalization,
  normalizeMvsScene,
  denormalizeMvsPoints,
  planMvsViews,
  type MvsScene,
  type MvsSceneView,
} from '../src/gpu/mvs-views.ts';

function scene(): MvsScene {
  const views = [
    [-1, 0, 5],
    [1, 0.3, 5],
    [0, -0.2, 6],
  ].map((center, id) => {
    const length = Math.hypot(...center),
      forward = center.map((x) => -x / length),
      rightLength = Math.hypot(forward[2]!, forward[0]!),
      right = [-forward[2]! / rightLength, 0, forward[0]! / rightLength],
      down = [
        forward[1]! * right[2]!,
        forward[2]! * right[0]! - forward[0]! * right[2]!,
        -forward[1]! * right[0]!,
      ],
      R = Float64Array.of(...right, ...down, ...forward),
      t = Float64Array.from(
        { length: 3 },
        (_, r) => -center.reduce((sum, x, c) => sum + R[3 * r + c]! * x, 0),
      );
    return {
      id,
      width: 640,
      height: 480,
      camera: { fx: 500, fy: 500, cx: 319.5, cy: 239.5 },
      pose: { R, t },
    };
  });
  const landmarks = Array.from({ length: 36 }, (_, i) => {
    const xyz = Float64Array.of((i % 6) * 0.4 - 1, Math.floor(i / 6) * 0.4 - 1, 0.1 * Math.sin(i));
    return {
      xyz,
      observations: views.map((view) => ({ viewId: view.id, pixel: project(view, xyz) })),
    };
  });
  return { views, landmarks };
}

function project(view: MvsSceneView, X: ArrayLike<number>): readonly [number, number] {
  const q = [0, 1, 2].map(
    (r) =>
      view.pose.R[3 * r]! * X[0]! +
      view.pose.R[3 * r + 1]! * X[1]! +
      view.pose.R[3 * r + 2]! * X[2]! +
      view.pose.t[r]!,
  );
  return [
    (view.camera.fx * q[0]!) / q[2]! + view.camera.cx,
    (view.camera.fy * q[1]!) / q[2]! + view.camera.cy,
  ];
}

test('one unobserved point behind every camera preserves packed visible geometry', () => {
  const original = scene(),
    extra = { xyz: Float64Array.of(0, 0, 1e10), observations: [] },
    augmented = { ...original, landmarks: [...original.landmarks, extra] },
    snapshot = JSON.stringify(augmented),
    before = planMvsViews(original),
    after = planMvsViews(augmented),
    normalized = normalizeMvsScene(augmented, after.normalization);
  assert.equal(normalized.landmarks.length, augmented.landmarks.length);
  assert.deepEqual(after.packed, before.packed);
  assert.deepEqual(after.sources, before.sources);
  assert.deepEqual(after.seedOffsets, before.seedOffsets);
  for (let v = 0; v < after.views.length; v++) {
    for (let bound = 0; bound < 2; bound++) {
      const a = after.views[v]!.depthRange![bound]! * after.normalization.scale,
        b = before.views[v]!.depthRange![bound]! * before.normalization.scale;
      assert.ok(Math.abs(a - b) < 1e-5, `physical depth bound ${v}/${bound}: ${a} vs ${b}`);
    }
  }
  let worst = 0;
  for (let v = 0; v < after.packed.length; v++) {
    const camera = after.cameras.subarray(v * 24, v * 24 + 24),
      view = original.views[after.packed[v]!]!;
    for (let i = 0; i < original.landmarks.length; i++) {
      const X = Float32Array.from(normalized.landmarks[i]!.xyz),
        // f64 evaluation deliberately isolates loss already present in the f32 inputs.
        q = [0, 1, 2].map(
          (r) =>
            camera[r * 3]! * X[0]! +
            camera[r * 3 + 1]! * X[1]! +
            camera[r * 3 + 2]! * X[2]! +
            camera[9 + r]!,
        ),
        expected = project(view, original.landmarks[i]!.xyz);
      assert.ok(q[2]! > 0, 'a visible point remains in front after packing');
      worst = Math.max(
        worst,
        Math.hypot(
          (camera[12]! * q[0]!) / q[2]! + camera[14]! - expected[0],
          (camera[13]! * q[1]!) / q[2]! + camera[15]! - expected[1],
        ),
      );
    }
  }
  assert.ok(worst < 0.01, `f32 packing moved a visible projection by ${worst} pixels`);
  const all = Float64Array.from(normalized.landmarks.flatMap((p) => Array.from(p.xyz))),
    restored = denormalizeMvsPoints(all, after.normalization);
  augmented.landmarks.forEach((p, i) => {
    for (let axis = 0; axis < 3; axis++)
      assert.ok(
        Math.abs(restored[i * 3 + axis]! - p.xyz[axis]!) <
          1e-12 * Math.max(1, Math.abs(p.xyz[axis]!)),
      );
  });
  assert.equal(
    JSON.stringify(augmented),
    snapshot,
    'planning never changes supplied coordinates/observations',
  );
});

test('normalization origin handles empty, singleton and even coordinate populations', () => {
  assert.deepEqual(mvsNormalization({ views: [], landmarks: [] }), {
    center: new Float64Array(3),
    scale: 1,
  });
  const single = mvsNormalization({ views: [], landmarks: [{ xyz: [7, 8, 9], observations: [] }] });
  assert.deepEqual(Array.from(single.center), [7, 8, 9]);
  assert.equal(single.scale, 1);
  const even = mvsNormalization({
    views: [],
    landmarks: [
      { xyz: [1, 2, 3], observations: [] },
      { xyz: [3, 4, 5], observations: [] },
      { xyz: [9, 10, 11], observations: [] },
      { xyz: [1e10, 1e10, 1e10], observations: [] },
    ],
  });
  assert.deepEqual(Array.from(even.center), [6, 7, 8]);
  const points = [
      [1, 2, 3],
      [3, 4, 5],
      [9, 10, 11],
      [1e10, 1e10, 1e10],
    ],
    centroid = [0, 1, 2].map(
      (axis) => points.reduce((sum, point) => sum + point[axis]!, 0) / points.length,
    ),
    expectedRadius = Math.sqrt(
      points.reduce(
        (sum, point) => sum + point.reduce((s, x, axis) => s + (x - centroid[axis]!) ** 2, 0),
        0,
      ) / points.length,
    );
  assert.equal(
    even.scale,
    expectedRadius,
    'translation origin does not redefine the existing RMS scale',
  );
});

test('a similarity gauge change preserves normalized projections and the physical inverse', () => {
  const original = scene(),
    angle = 0.31,
    cosine = Math.cos(angle),
    sine = Math.sin(angle),
    rotation = [cosine, -sine, 0, sine, cosine, 0, 0, 0, 1],
    translation = [1e4, -2e4, 3e4],
    scale = 100,
    changed: MvsScene = {
      views: original.views.map((view) => {
        const R = Float64Array.from({ length: 9 }, (_, index) => {
            const row = Math.floor(index / 3),
              column = index % 3;
            return [0, 1, 2].reduce(
              (sum, k) => sum + view.pose.R[row * 3 + k]! * rotation[column * 3 + k]!,
              0,
            );
          }),
          t = Float64Array.from(
            { length: 3 },
            (_, row) =>
              scale * view.pose.t[row]! -
              translation.reduce((sum, x, k) => sum + R[row * 3 + k]! * x, 0),
          );
        return { ...view, pose: { R, t } };
      }),
      landmarks: original.landmarks.map((point) => ({
        ...point,
        xyz: Float64Array.from(
          { length: 3 },
          (_, row) =>
            scale * [0, 1, 2].reduce((sum, k) => sum + rotation[row * 3 + k]! * point.xyz[k]!, 0) +
            translation[row]!,
        ),
      })),
    },
    normalization = mvsNormalization(changed),
    normalized = normalizeMvsScene(changed, normalization);
  assert.ok(Math.abs(normalization.scale / mvsNormalization(original).scale - scale) < 1e-9);
  for (let v = 0; v < original.views.length; v++)
    for (let p = 0; p < original.landmarks.length; p++) {
      const expected = project(original.views[v]!, original.landmarks[p]!.xyz),
        actual = project(normalized.views[v]!, normalized.landmarks[p]!.xyz);
      assert.ok(Math.hypot(expected[0] - actual[0], expected[1] - actual[1]) < 1e-8);
    }
  const restored = denormalizeMvsPoints(
    Float64Array.from(normalized.landmarks.flatMap((point) => Array.from(point.xyz))),
    normalization,
  );
  changed.landmarks.forEach((point, i) => {
    for (let axis = 0; axis < 3; axis++)
      assert.ok(Math.abs(restored[i * 3 + axis]! - point.xyz[axis]!) < 1e-9);
  });
});
