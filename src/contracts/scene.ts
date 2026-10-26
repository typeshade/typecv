import type { Camera, Pose } from './camera.ts';
export interface ImageFeatures {
  readonly id: number;
  readonly camera: Camera;
  readonly keypoints: Float64Array;
}
export interface FeatureObservation {
  readonly viewId: number;
  readonly featureId: number;
}
/** Edges must be geometrically verified before building reconstruction tracks. Lower distance wins conflicts. */
export interface PairMatches {
  readonly first: number;
  readonly second: number;
  readonly matches: Uint32Array;
  readonly distances?: Float64Array;
}
export interface FeatureTrack {
  readonly id: number;
  readonly observations: readonly FeatureObservation[];
}
export interface RegisteredView {
  readonly id: number;
  readonly camera: Camera;
  readonly pose: Pose;
}
export interface LandmarkObservation {
  readonly viewId: number;
  readonly featureId: number;
  readonly pixel: readonly [number, number];
}
export interface Landmark {
  readonly trackId: number;
  readonly xyz: Float64Array;
  readonly observations: readonly LandmarkObservation[];
}
export interface SparseReconstruction {
  readonly views: readonly RegisteredView[];
  readonly landmarks: readonly Landmark[];
}
