import test from 'node:test';
import assert from 'node:assert/strict';
import {
  verifyImagePair,
  unprojectPixel,
  estimateHomography,
  reconstructTwoView,
  projectPoint,
} from '../src/index.ts';
import { scene } from './scene.ts';

function fixture() {
  const f = scene({
    count: 83,
    views: 2,
    noise: 0.03,
    camera: { fx: 850, fy: 870, cx: 480, cy: 270, k1: -0.25 },
  });
  const first = f.inputs[0]!,
    second = f.inputs[1]!,
    pixels = new Float64Array(4 * f.points.length);
  f.points.forEach((_, i) =>
    pixels.set(
      [
        first.keypoints[2 * i]!,
        first.keypoints[2 * i + 1]!,
        second.keypoints[2 * i]!,
        second.keypoints[2 * i + 1]!,
      ],
      4 * i,
    ),
  );
  return { ...f, first, second, pixels };
}

test('pair verification preserves independent CPU stages and original row ownership', () => {
  const f = fixture(),
    { first, second, pixels } = f;
  for (let i = 0; i < f.points.length; i += 13) pixels[4 * i] = 100000;
  const snapshot = pixels.slice(),
    values: number[] = [],
    rows: number[] = [];
  for (let i = 0; i < f.points.length; i++) {
    const a = unprojectPixel(first.camera, [pixels[4 * i]!, pixels[4 * i + 1]!]),
      b = unprojectPixel(second.camera, [pixels[4 * i + 2]!, pixels[4 * i + 3]!]);
    if (!a || !b) continue;
    rows.push(i);
    values.push(
      a[0] * first.camera.fx + first.camera.cx,
      a[1] * first.camera.fy + first.camera.cy,
      b[0] * second.camera.fx + second.camera.cx,
      b[1] * second.camera.fy + second.camera.cy,
    );
  }
  const ideal = Float64Array.from(values),
    options = { iterations: 128, refinementIterations: 0, seed: 31 },
    h = estimateHomography(ideal, { iterations: 128, thresholdPx: 4, seed: 31 }),
    reference = reconstructTwoView(ideal, first.camera, second.camera, options),
    result = verifyImagePair(pixels, first.camera, second.camera, options);
  assert.equal(result.invertibleCount, rows.length);
  assert.equal(result.homographyInlierCount, h?.inlierCount ?? 0);
  assert.deepEqual(
    [...result.inlierRows],
    rows.filter((_, i) => reference.inliers[i]),
  );
  assert.deepEqual(result.geometry, {
    R: reference.R,
    t: reference.t,
    inlierCount: reference.inlierCount,
    positiveDepthCount: reference.positiveDepthCount,
    meanReprojectionErrorPx: reference.meanReprojectionErrorPx,
    diagnostics: reference.diagnostics,
  });
  assert.deepEqual(pixels, snapshot);
});

test('pair rejection retains H support while malformed data/options throw', () => {
  const f = fixture(),
    { first, second } = f;
  const planar = Float64Array.from(
    f.points.flatMap((X) => {
      const point = Float64Array.of(X[0]!, X[1]!, 6);
      return [
        ...projectPoint(first.camera, f.registered[0]!.pose, point)!,
        ...projectPoint(second.camera, f.registered[1]!.pose, point)!,
      ];
    }),
  );
  const result = verifyImagePair(planar, first.camera, second.camera, { iterations: 64 });
  assert.equal(result.homographyInlierCount, f.points.length);
  assert.equal(result.geometry, undefined);
  assert.equal(result.inlierRows.length, 0);
  assert.match(result.reason!, /valid two-view|parallax|Cheirality/);
  const short = verifyImagePair(f.pixels.slice(0, 12), first.camera, second.camera);
  assert.equal(short.invertibleCount, 3);
  assert.equal(short.homographyInlierCount, 0);
  assert.equal(short.geometry, undefined);
  assert.match(short.reason!, /Insufficient invertible/);
  for (const pixels of [Float64Array.of(1, 2, 3), Float64Array.of(1, 2, NaN, 4)])
    assert.throws(() => verifyImagePair(pixels, first.camera, second.camera), RangeError);
  for (const options of [
    { homographyThresholdPx: 0 },
    { homographyIterations: 0 },
    { iterations: 0 },
  ])
    assert.throws(
      () => verifyImagePair(new Float64Array(0), first.camera, second.camera, options),
      RangeError,
    );
});
