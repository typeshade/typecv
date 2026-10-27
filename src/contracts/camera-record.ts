import type { Camera, Pose } from './camera.ts';

/**
 * The camera and pose records that the GPU programs and the host scene read, each written once
 * (docs/library-structure.md, section 3.2). A pose record is `R` row-major, then `t`: 12 doubles,
 * the `POSE_STRIDE` of `src/sfm/scene/scene.ts`. A camera record is `fx, fy, cx, cy, k1, k2`, with
 * 0 for an absent distortion term: 6 doubles, the `MODEL_STRIDE` of the same module. Each caller
 * puts the records at the offsets of its own layout.
 */

/** @internal The pose record: `R` row-major, then `t`. */
export function poseRecord(pose: Pose): number[] {
  return [...pose.R, ...pose.t];
}

/** @internal The camera record: `fx, fy, cx, cy, k1, k2`, with 0 for an absent `k1` or `k2`. */
export function cameraRecord(camera: Camera): number[] {
  return [camera.fx, camera.fy, camera.cx, camera.cy, camera.k1 ?? 0, camera.k2 ?? 0];
}

/** @internal The camera record as a struct value with the same six fields, in the same order. */
export function cameraStruct(camera: Camera): {
  fx: number;
  fy: number;
  cx: number;
  cy: number;
  k1: number;
  k2: number;
} {
  return {
    fx: camera.fx,
    fy: camera.fy,
    cx: camera.cx,
    cy: camera.cy,
    k1: camera.k1 ?? 0,
    k2: camera.k2 ?? 0,
  };
}
