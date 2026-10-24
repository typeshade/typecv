import test from 'node:test';
import assert from 'node:assert/strict';
import {
  cameraMaskFromPixelPaths,
  projectPoint,
  reconstructIncremental,
  triangulateTrack,
  type ObservationPathMask,
} from '../src/index.ts';
import { identityPose, transformPoint } from '../src/geometry/pose.ts';
import { seededRandom } from '../src/geometry/numerics.ts';
import { generalVideoFixture } from './fixtures/general-video.ts';
import { snapshotObservationPaths } from '../src/reconstruction/observation-paths.ts';

test('a static planar reflection passes ordinary triangulation; explicit optical paths separate the virtual map', () => {
  const camera = { fx: 480, fy: 480, cx: 320, cy: 240 },
    rng = seededRandom(461),
    real = Array.from({ length: 60 }, () => [3 * (rng() - 0.5), 1.2 + rng() * 0.8, 6 + rng() * 4]),
    // Reflection about y=0 has det=-1; it is applied to points, never passed as an SE(3) pose.
    virtual = real.map(([x, y, z]) => [x!, -y!, z!]),
    points = [...real, ...virtual],
    poses = [-0.8, -0.4, 0, 0.4, 0.8].map((x) => ({
      ...identityPose(),
      t: Float64Array.of(-x, -0.5, 0),
    }));
  const observations = poses.map((pose) => ({
      camera,
      pose,
      pixel: projectPoint(camera, pose, virtual[0]!)!,
    })),
    track = triangulateTrack(observations)!;
  assert.ok(track.meanErrorPx < 1e-8);
  assert.ok(track.inliers.every((v) => v === 1));
  assert.ok(Math.hypot(...track.xyz.map((x, i) => x - virtual[0]![i]!)) < 1e-7);
  assert.ok(poses.every((p) => transformPoint(p, track.xyz)[2]! > 0));
  assert.ok(track.xyz[1]! < 0, 'accepted virtual geometry is below the reflecting surface');

  const f = generalVideoFixture('translation'),
    images = poses.map((pose, id) => ({
      ...f.input.images[id]!,
      id,
      camera,
      keypoints: Float64Array.from(points.flatMap((p) => [...projectPoint(camera, pose, p)!])),
    })),
    pairs = f.input.pairs.map((p) => ({
      ...p,
      matches: Uint32Array.from(points.flatMap((_, i) => [i, i])),
    })),
    paths: ObservationPathMask = {
      width: 640,
      height: 480,
      source: 'ground-truth',
      values: Uint8Array.from({ length: 640 * 480 }, (_, i) => (i < 240 * 640 ? 1 : 0)),
    },
    masks = images.map((image) => cameraMaskFromPixelPaths(image, paths, 1)),
    controls = { ransacIterations: 128, bundleIterations: 1, finalBundleIterations: 2 };
  assert.ok(masks.every((m) => m.values.every((v, i) => v === (i < 60 ? 1 : 0))));
  const unmasked = reconstructIncremental(images, pairs, controls),
    masked = reconstructIncremental(images, pairs, { ...controls, cameraObservationMasks: masks });
  assert.ok(
    unmasked.reconstruction.landmarks.filter((l) => l.observations[0]!.featureId >= 60).length >=
      50,
  );
  assert.equal(masked.reconstruction.landmarks.length, 60);
  assert.ok(
    masked.reconstruction.landmarks.every((l) => l.observations.every((o) => o.featureId < 60)),
  );
  assert.equal(
    masked.observationSelection!.backgroundMatchCount * 2,
    masked.observationSelection!.inputMatchCount,
  );
  assert.ok(
    masked.observationSelection!.separatedPairs.every((p) => p.matches.every((id) => id >= 60)),
  );
});

test('pixel-path selection undistorts keypoints, pads descriptor support and snapshots provenance', () => {
  const paths: ObservationPathMask = {
      width: 11,
      height: 11,
      values: new Uint8Array(121),
      source: 'manual',
    },
    camera = { fx: 10, fy: 10, cx: 5, cy: 5, k1: 1 },
    image = { id: 0, camera, keypoints: Float64Array.of(5, 5, 8.27, 5, 7, 5, -100, 5) };
  paths.values[5 * 11 + 8] = 2; // x=8 undistorted projects to x=8.27 in the original RGB.
  assert.deepEqual(cameraMaskFromPixelPaths(image, paths).values, Uint8Array.of(1, 0, 1, 0));
  assert.deepEqual(cameraMaskFromPixelPaths(image, paths, 1).values, Uint8Array.of(1, 0, 0, 0));
  const copy = snapshotObservationPaths(paths);
  paths.values.fill(3);
  assert.equal(copy.values[5 * 11 + 5], 0);
  assert.equal(copy.source, 'manual');
  assert.throws(() => snapshotObservationPaths({ ...copy, values: Uint8Array.of(4) }), RangeError);
  assert.throws(() => cameraMaskFromPixelPaths(image, paths, -1), RangeError);
});
