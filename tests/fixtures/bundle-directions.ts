import { generalVideoFixture } from './general-video.ts';
import { directionObservationFixture } from './direction-observations.ts';
import { multiply } from '../../src/sfm/linear.ts';
import { rotationExp } from '../../src/geometry/pose.ts';
import type { SparseReconstruction } from '../../src/reconstruction/types.ts';
import type { BundleDirection } from '../../src/reconstruction/bundle-directions.ts';

/** Explicit optimizer-factor fixture, not automatic observation classification or an RGB run. */
export function bundleDirectionsFixture(focalScale = 1) {
  const f = directionObservationFixture(),
    observations = (featureId: number) =>
      f.input.images.map((v) => ({
        viewId: v.id,
        featureId,
        pixel: [v.keypoints[featureId * 2]!, v.keypoints[featureId * 2 + 1]!] as const,
      })),
    truth: SparseReconstruction = {
      views: f.truth.views,
      landmarks: f.truth.finitePoints.map((xyz, trackId) => ({
        trackId,
        xyz,
        observations: observations(trackId),
      })),
    },
    directions: BundleDirection[] = f.truth.bearings.map((direction, i) => ({
      trackId: 60 + i,
      direction: Float64Array.from(direction),
      observations: observations(60 + i),
    })),
    input: SparseReconstruction = {
      views: truth.views.map((v, i) => ({
        ...v,
        camera: { ...v.camera, fx: v.camera.fx * focalScale, fy: v.camera.fy * focalScale },
        pose: {
          R: i ? multiply(rotationExp([0.003, -0.004, 0.002]), v.pose.R, 3) : v.pose.R.slice(),
          t: v.pose.t.slice(),
        },
      })),
      landmarks: truth.landmarks.map((p) => ({
        ...p,
        xyz: p.xyz.map((x, i) => x + 0.02 * Math.sin(p.trackId + 3 * i)),
      })),
    };
  return { input, directions, truth };
}

export function disconnectedBundleDirectionsFixture() {
  const f = generalVideoFixture('disconnected');
  return {
    views: f.truth.views,
    landmarks: f.truth.points.map((xyz, trackId) => ({
      trackId,
      xyz,
      observations: f.input.images.slice(0, 3).map((v) => ({
        viewId: v.id,
        featureId: trackId,
        pixel: [v.keypoints[trackId * 2]!, v.keypoints[trackId * 2 + 1]!] as const,
      })),
    })),
  };
}
