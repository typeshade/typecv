import test from 'node:test';
import assert from 'node:assert/strict';
import type { Camera, Pixel } from '../src/geometry/camera.ts';
import { identityPose, projectPoint } from '../src/geometry/pose.ts';
import type { Pose } from '../src/geometry/pose.ts';
import {
  triangulateTrack,
  triangulateTrackDetailed,
  type TriangulationObservation,
} from '../src/geometry/multiview.ts';

const camera: Camera = { fx: 800, fy: 800, cx: 320, cy: 240, k1: 0, k2: 0 };
const point = Float64Array.of(0.1, -0.15, 10);
const controls = { maxErrorPx: 3, minAngleDeg: 0.5, maxPairHypotheses: 32 };

function poseAt(x: number): Pose {
  return { ...identityPose(), t: Float64Array.of(-x, 0, 0) };
}

function observationsAt(centers: readonly number[]): TriangulationObservation[] {
  return centers.map((x) => {
    const pose = poseAt(x),
      pixel = projectPoint(camera, pose, point);
    assert.ok(pixel);
    return { camera: { ...camera }, pose, pixel };
  });
}

function expectRejected(
  observations: readonly TriangulationObservation[],
  status: 1 | 3 | 4,
): void {
  assert.deepEqual(triangulateTrackDetailed(observations, controls), { status });
  assert.equal(triangulateTrack(observations, controls), undefined);
}

test('detailed accepted track keeps the legacy values, actual consensus mask and inputs', () => {
  const observations = observationsAt([0, 0.4, 0.8, 1.2]);
  const last = observations[3]!;
  observations[3] = { ...last, pixel: [last.pixel[0] - 600, last.pixel[1] + 400] };
  const original = structuredClone(observations);
  const result = triangulateTrackDetailed(observations, controls);
  assert.ok(result.status === 0);
  assert.deepEqual(result.track.inliers, Uint8Array.of(1, 1, 1, 0));
  assert.ok(result.track.xyz instanceof Float64Array);
  assert.ok(result.track.inliers instanceof Uint8Array);
  assert.ok(Math.hypot(...result.track.xyz.map((value, i) => value - point[i]!)) < 1e-8);
  assert.ok(result.track.maxAngleDeg >= controls.minAngleDeg);
  assert.ok(result.track.meanErrorPx < 1e-8);
  assert.deepEqual(triangulateTrack(observations, controls), result.track);
  assert.deepEqual(observations, original);
});

test('valid zero/single observation inputs report status1 and retain legacy undefined', () => {
  expectRejected([], 1);
  expectRejected(observationsAt([0]), 1);
});

test('degenerate, inconsistent and behind-camera inputs are status3, not low parallax', () => {
  expectRejected(observationsAt([0, 0]), 3);
  const inconsistent = observationsAt([0, 1]);
  const second = inconsistent[1]!;
  inconsistent[1] = { ...second, pixel: [second.pixel[0], second.pixel[1] + 200] };
  expectRejected(inconsistent, 3);
  // These finite pixels meet only at (0.2,0.1,-4), behind both accepted cameras.
  const behind = [0, 1].map((x): TriangulationObservation => ({
    camera,
    pose: poseAt(x),
    pixel: [(camera.fx * (0.2 - x)) / -4 + camera.cx, (camera.fy * 0.1) / -4 + camera.cy],
  }));
  expectRejected(behind, 3);
});

test('a valid two-view consensus below the unchanged angle reports status4 without a track', () => {
  expectRejected(observationsAt([0, 0.06]), 4);
});

test('status4 with more observations exposes no unaccepted geometry or per-row authority', () => {
  const observations = observationsAt([0, 0.02, 0.04, 0.06]);
  const last = observations[3]!;
  observations[3] = { ...last, pixel: [last.pixel[0] - 600, last.pixel[1] + 400] };
  expectRejected(observations, 4);
});

test('individually low-angle neighboring links gain accepted depth from a third fixed view', () => {
  const observations = observationsAt([0, 0.06, 0.12]);
  expectRejected(observations.slice(0, 2), 4);
  expectRejected(observations.slice(1), 4);
  const result = triangulateTrackDetailed(observations, controls);
  assert.ok(result.status === 0);
  assert.deepEqual(result.track.inliers, Uint8Array.of(1, 1, 1));
  assert.ok(result.track.maxAngleDeg >= controls.minAngleDeg);
  assert.deepEqual(triangulateTrack(observations, controls), result.track);
});

test('the existing angle comparison remains inclusive at the computed boundary', () => {
  const observations = observationsAt([0, 0.06, 0.12]);
  const first = triangulateTrackDetailed(observations, controls);
  assert.ok(first.status === 0);
  const exact = { ...controls, minAngleDeg: first.track.maxAngleDeg };
  const atBoundary = triangulateTrackDetailed(observations, exact);
  assert.ok(atBoundary.status === 0);
  assert.deepEqual(atBoundary.track, first.track);
  assert.deepEqual(triangulateTrack(observations, exact), first.track);
  const above = { ...controls, minAngleDeg: first.track.maxAngleDeg + 1e-10 };
  assert.deepEqual(triangulateTrackDetailed(observations, above), { status: 4 });
  assert.equal(triangulateTrack(observations, above), undefined);
});

test('invalid options/observations still throw before short-input status classification', () => {
  for (const options of [
    { ...controls, maxErrorPx: 0 },
    { ...controls, minAngleDeg: 180 },
    { ...controls, maxPairHypotheses: 0 },
  ]) {
    for (const triangulate of [triangulateTrack, triangulateTrackDetailed])
      assert.throws(() => triangulate([], options), RangeError);
  }
  const good = observationsAt([0])[0]!;
  const invalid: TriangulationObservation[] = [
    { ...good, camera: { ...camera, fx: 0 } },
    { ...good, pose: { ...good.pose, R: new Float64Array(9) } },
    { ...good, pixel: [NaN, 240] as Pixel },
  ];
  for (const observation of invalid)
    for (const triangulate of [triangulateTrack, triangulateTrackDetailed])
      assert.throws(() => triangulate([observation], controls), RangeError);
});

test('the CPU API retains its existing capacity rather than adopting the GPU64-row cap', () => {
  const observations = observationsAt(Array.from({ length: 65 }, (_, i) => i * 0.02));
  const result = triangulateTrackDetailed(observations, controls);
  assert.ok(result.status === 0);
  assert.equal(result.track.inliers.length, 65);
  assert.ok(result.track.inliers.every((value) => value === 1));
  assert.deepEqual(triangulateTrack(observations, controls), result.track);
});
