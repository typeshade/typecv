import { unprojectPixel, validateCamera } from './camera.ts';
import type { Camera } from '../contracts/camera.ts';
import { GeometryEstimationError } from './errors.ts';
import { estimateHomography } from './homography.ts';
import {
  reconstructTwoView,
  resolveTwoViewOptions,
  type TrialHypotheses,
  type TwoViewOptions,
  type TwoViewResult,
} from './two-view.ts';
import { imagePairHomographyIterations } from './operator-defaults.ts';
import { CHEIRALITY_FAILURE_PREFIX, TwoViewFailure } from '../contracts/two-view-failures.ts';

export interface ImagePairOptions extends TwoViewOptions {
  /** Symmetric transfer threshold in ideal pixels. Default: maxReprojectionErrorPx (4). */
  readonly homographyThresholdPx?: number;
  /** H trial cap. Default: min(iterations, 500); H confidence is 0.999. */
  readonly homographyIterations?: number;
  /** @internal Trials of the first RANSAC run, from a cache (the focal search). */
  readonly hypotheses?: TrialHypotheses;
}
export type ImagePairGeometry = Pick<
  TwoViewResult,
  'R' | 't' | 'inlierCount' | 'positiveDepthCount' | 'meanReprojectionErrorPx' | 'diagnostics'
>;
export interface ImagePairVerification {
  readonly invertibleCount: number;
  readonly homographyInlierCount: number;
  /** Ascending indices in the original raw correspondence array, not feature ids. */
  readonly inlierRows: Uint32Array;
  readonly geometry?: ImagePairGeometry;
  /** Expected geometric rejection; invalid inputs and backend failures throw instead. */
  readonly reason?: string;
}

/** @internal */
export function resolveImagePairOptions(first: Camera, second: Camera, options: ImagePairOptions) {
  validateCamera(first);
  validateCamera(second);
  const controls = resolveTwoViewOptions({
      ...options,
      refinementIterations: options.refinementIterations ?? 0,
    }),
    homographyThreshold = options.homographyThresholdPx ?? controls.maxError,
    homographyIterations =
      options.homographyIterations ?? imagePairHomographyIterations(controls.iterations);
  if (
    !Number.isFinite(homographyThreshold) ||
    homographyThreshold <= 0 ||
    !Number.isSafeInteger(homographyIterations) ||
    homographyIterations < 1
  )
    throw new RangeError('Invalid image-pair homography options');
  return { ...controls, homographyThreshold, homographyIterations };
}
/** @internal */
export function validateImagePairPixels(pixels: Float64Array): void {
  if (!(pixels instanceof Float64Array) || pixels.length % 4 || !pixels.every(Number.isFinite))
    throw new RangeError('Image pairs require finite Float64Array rows [x1,y1,x2,y2]');
}

/** Calibrated pair verification; residual/quality thresholds use ideal pinhole pixels. */
export function verifyImagePair(
  matches: Float64Array,
  first: Camera,
  second: Camera,
  options: ImagePairOptions = {},
): ImagePairVerification {
  return verifyImagePairInternal(matches, first, second, options);
}

/** @internal Reuse the pixel-space homography support already measured by the focal pair base. */
export function verifyImagePairWithHomography(
  matches: Float64Array,
  first: Camera,
  second: Camera,
  homographyInlierCount: number,
  options: ImagePairOptions = {},
): ImagePairVerification {
  return verifyImagePairInternal(matches, first, second, options, homographyInlierCount);
}

function verifyImagePairInternal(
  matches: Float64Array,
  first: Camera,
  second: Camera,
  options: ImagePairOptions,
  cachedHomographyInlierCount?: number,
): ImagePairVerification {
  const controls = resolveImagePairOptions(first, second, options);
  validateImagePairPixels(matches);
  if (
    cachedHomographyInlierCount !== undefined &&
    (!Number.isSafeInteger(cachedHomographyInlierCount) ||
      cachedHomographyInlierCount < 0 ||
      cachedHomographyInlierCount > matches.length / 4)
  )
    throw new RangeError('Invalid cached image-pair homography count');
  const values: number[] = [],
    rows: number[] = [];
  for (let i = 0; i < matches.length / 4; i++) {
    const a = unprojectPixel(first, [matches[4 * i]!, matches[4 * i + 1]!]),
      b = unprojectPixel(second, [matches[4 * i + 2]!, matches[4 * i + 3]!]);
    if (!a || !b) continue;
    values.push(
      a[0] * first.fx + first.cx,
      a[1] * first.fy + first.cy,
      b[0] * second.fx + second.cx,
      b[1] * second.fy + second.cy,
    );
    rows.push(i);
  }
  const pixels = Float64Array.from(values),
    invertibleCount = rows.length;
  if (invertibleCount < 8)
    return {
      invertibleCount,
      homographyInlierCount: cachedHomographyInlierCount ?? 0,
      inlierRows: new Uint32Array(0),
      reason: TwoViewFailure.fewInvertible,
    };
  const homographyInlierCount =
    cachedHomographyInlierCount ??
    estimateHomography(pixels, {
      thresholdPx: controls.homographyThreshold,
      iterations: controls.homographyIterations,
      seed: controls.seed,
    })?.inlierCount ??
    0;
  const base = { invertibleCount, homographyInlierCount };
  // A near-planar pair can hand RANSAC a best-scoring essential matrix whose pose fails
  // cheirality; another seed usually finds the right one. Retry a bounded number of times.
  for (let attempt = 0; ; attempt++) {
    try {
      const result = reconstructTwoView(
        pixels,
        first,
        second,
        {
          ...options,
          seed: controls.seed + attempt * CHEIRALITY_RETRY_SEED_STEP,
          refinementIterations: controls.refinementIterations,
        },
        attempt === 0 ? options.hypotheses : undefined,
      );
      const inlierRows = Uint32Array.from(rows.filter((_, i) => result.inliers[i]));
      return {
        ...base,
        inlierRows,
        geometry: {
          R: result.R,
          t: result.t,
          inlierCount: result.inlierCount,
          positiveDepthCount: result.positiveDepthCount,
          meanReprojectionErrorPx: result.meanReprojectionErrorPx,
          diagnostics: result.diagnostics,
        },
      };
    } catch (error) {
      if (!(error instanceof GeometryEstimationError)) throw error;
      if (attempt < CHEIRALITY_RETRIES && isCheiralityFailure(error.message)) continue;
      return { ...base, inlierRows: new Uint32Array(0), reason: error.message };
    }
  }
}
/** Further RANSAC runs, with other seeds, after a pose failed cheirality. */
export const CHEIRALITY_RETRIES = 2;
/** @internal Seed offset between those runs; large and odd so the sample sequences do not overlap. */
export const CHEIRALITY_RETRY_SEED_STEP = 7919;
/** @internal */
export const isCheiralityFailure = (reason: string | undefined) =>
  typeof reason === 'string' && reason.startsWith(CHEIRALITY_FAILURE_PREFIX);
