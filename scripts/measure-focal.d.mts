/** Declarations for scripts/measure-focal.mjs, so that the tests that import it typecheck. */
import type { Camera } from '../src/geometry/camera.ts';
import type { PairMatches, SparseReconstruction } from '../src/reconstruction/types.ts';

/** A frame of a sample: COLMAP's SIFT keypoints and the sample's reference camera. */
export interface SampleFrame {
  readonly id: number;
  readonly width: number;
  readonly height: number;
  readonly truth: Camera;
  readonly keypoints: Float64Array;
}
export declare function loadSample(name: string): { frames: SampleFrame[]; pairs: PairMatches[] };
export declare function loadPyramid(): { frames: SampleFrame[]; pairs: PairMatches[] };
export declare function pyramidRelief(
  reconstruction: SparseReconstruction,
  viewId?: number,
): number;
/** The lab's numerical controls: the six mapper numbers of `src/sfm/config/lab-preset.ts`. */
export declare const LAB: {
  readonly ransacIterations: number;
  readonly maxErrorPx: number;
  readonly minInitialAngleDeg: number;
  readonly minTriangulationAngleDeg: number;
  readonly bundleIterations: number;
  readonly finalBundleIterations: number;
};
export declare function camerasAt(
  frames: readonly SampleFrame[],
  ratio: number,
  distortion: boolean,
): { id: number; keypoints: Float64Array; camera: Camera }[];
