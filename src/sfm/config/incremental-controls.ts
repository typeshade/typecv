import { depthSensitivityControls } from '../../geometry/multiview.ts';
import type { ImageFeatures, PairMatches } from '../../contracts/scene.ts';
import type { IncrementalControls, IncrementalOptions } from '../mapper/mapper.ts';
import { rotationTrackingControls } from '../tracks/tracking.ts';
import { validateFeatures } from '../tracks/tracks.ts';

/** @internal Validate the inputs and options; throws RangeError on an invalid one. */
export function incrementalControls(
  images: readonly ImageFeatures[],
  inputPairs: readonly PairMatches[],
  options: IncrementalOptions,
): { byId: Map<number, ImageFeatures>; controls: IncrementalControls } {
  const byId = validateFeatures(images, inputPairs),
    seed = options.seed ?? 17,
    iterations = options.ransacIterations ?? 750,
    maxError = options.maxErrorPx ?? 3;
  const pairRefinementIterations = options.pairRefinementIterations ?? 0;
  const separateDirectionalObservations =
    options.separateDirectionalObservations ?? !!options.depthTracking;
  if (typeof separateDirectionalObservations !== 'boolean')
    throw new RangeError('separateDirectionalObservations must be boolean');
  depthSensitivityControls(options);
  if (
    !Number.isSafeInteger(pairRefinementIterations) ||
    pairRefinementIterations < 0 ||
    pairRefinementIterations > 100
  )
    throw new RangeError('pairRefinementIterations must be 0..100');
  const initialAngle = options.minInitialAngleDeg ?? 1,
    pointAngle = options.minTriangulationAngleDeg ?? 0.5,
    minPnP = options.minPnPInliers ?? 12,
    minInitial = options.minInitialPoints ?? 12;
  const bundleIterations = options.bundleIterations ?? 8,
    finalBundle = options.finalBundleIterations ?? 25;
  const localSize = options.localBundleSize ?? 6,
    globalViewRatio = options.globalBundleViewRatio ?? DEFAULT_GLOBAL_BUNDLE_RATIO,
    globalPointRatio = options.globalBundlePointRatio ?? DEFAULT_GLOBAL_BUNDLE_RATIO,
    globalFrequency = options.globalBundleFrequency ?? 0,
    stallTolerance = options.bundleStallTolerance ?? DEFAULT_BUNDLE_STALL_TOLERANCE;
  if (
    images.length < 2 ||
    !Number.isSafeInteger(seed) ||
    !Number.isSafeInteger(iterations) ||
    iterations < 1 ||
    !Number.isFinite(maxError) ||
    maxError <= 0 ||
    !Number.isFinite(initialAngle) ||
    initialAngle <= 0 ||
    initialAngle >= 180 ||
    !Number.isFinite(pointAngle) ||
    pointAngle < 0 ||
    pointAngle >= 180 ||
    !Number.isSafeInteger(minPnP) ||
    minPnP < 6 ||
    !Number.isSafeInteger(minInitial) ||
    minInitial < 8 ||
    !Number.isSafeInteger(bundleIterations) ||
    bundleIterations < 0 ||
    !Number.isSafeInteger(finalBundle) ||
    finalBundle < 0 ||
    !Number.isSafeInteger(localSize) ||
    localSize < 1 ||
    !(globalViewRatio > 1) ||
    !(globalPointRatio > 1) ||
    !Number.isSafeInteger(globalFrequency) ||
    globalFrequency < (options.globalBundleFrequency === undefined ? 0 : 1) ||
    !Number.isFinite(stallTolerance) ||
    stallTolerance < 0 ||
    stallTolerance >= 1
  )
    throw new RangeError('Invalid incremental reconstruction inputs/options');
  if (options.localBundleSubset !== undefined && typeof options.localBundleSubset !== 'boolean')
    throw new RangeError('localBundleSubset must be a boolean');
  const rotationTracking = rotationTrackingControls(options.rotationTracking, !options.focalSearch);
  if (rotationTracking && options.focalSearch)
    throw new RangeError(
      'rotationTracking requires supplied calibration; focalSearch is not supported with it',
    );
  if (
    options.initialPair &&
    (!byId.has(options.initialPair[0]) ||
      !byId.has(options.initialPair[1]) ||
      options.initialPair[0] === options.initialPair[1])
  )
    throw new RangeError('Invalid initial pair');
  const pairKeys = new Set<string>();
  for (const p of inputPairs) {
    const key = [p.first, p.second].sort((a, b) => a - b).join(':');
    if (pairKeys.has(key)) throw new RangeError('Duplicate image pair');
    pairKeys.add(key);
  }
  return {
    byId,
    controls: {
      separateDirectionalObservations,
      pairRefinementIterations,
      depth: {
        maxRelativeDepthStdDev: options.maxRelativeDepthStdDev,
        pixelNoiseStdDevPx: options.pixelNoiseStdDevPx,
      },
      rotationTracking,
      seed,
      iterations,
      maxError,
      initialAngle,
      pointAngle,
      minPnP,
      minInitial,
      bundleIterations,
      finalBundle,
      localSize,
      localSubset: options.localBundleSubset ?? true,
      globalViewRatio,
      globalPointRatio,
      globalFrequency,
      stallTolerance,
    },
  };
}

/** COLMAP's `ba_global_images_ratio` and `ba_global_points_ratio`: 10 % growth. */
export const DEFAULT_GLOBAL_BUNDLE_RATIO = 1.1;
/** The default `bundleStallTolerance` of the mapper's BA passes. */
export const DEFAULT_BUNDLE_STALL_TOLERANCE = 1e-4;
