/** Pixel-coordinate matches, packed [x1,y1,x2,y2, ...]. */
export interface CameraIntrinsics {
  readonly fx: number;
  readonly fy: number;
  readonly cx: number;
  readonly cy: number;
}

/** Undistorted pinhole if k1=k2=0; otherwise Brown radial distortion only. */
export interface Camera extends CameraIntrinsics {
  readonly k1?: number;
  readonly k2?: number;
}
export type Pixel = readonly [number, number];

/** World-to-camera convention: q = R X + t. Translation is not normalized. */
export interface Pose {
  readonly R: Float64Array;
  readonly t: Float64Array;
}
