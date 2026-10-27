import { depthSensitivityControls } from '../../geometry/multiview.ts';
import type { ImageFeatures, PairMatches } from '../../contracts/scene.ts';
import type { IncrementalControls, IncrementalOptions } from '../mapper/mapper.ts';
import { rotationTrackingControls } from '../tracks/tracking.ts';
import { validateFeatures } from '../tracks/tracks.ts';
import {
  DEFAULT_BUNDLE_STALL_TOLERANCE,
  DEFAULT_GLOBAL_BUNDLE_RATIO,
  MAPPER_DEFAULTS,
} from './defaults.ts';
import { pairKey } from '../../contracts/keys.ts';

/** @internal Validate the inputs and options; throws RangeError on an invalid one. */
export function incrementalControls(
  images: readonly ImageFeatures[],
  inputPairs: readonly PairMatches[],
  options: IncrementalOptions,
): { byId: Map<number, ImageFeatures>; controls: IncrementalControls } {
  const byId = validateFeatures(images, inputPairs),
    seed = options.seed ?? MAPPER_DEFAULTS.seed,
    iterations = options.ransacIterations ?? MAPPER_DEFAULTS.ransacIterations,
    maxError = options.maxErrorPx ?? MAPPER_DEFAULTS.maxErrorPx;
  const pairRefinementIterations =
    options.pairRefinementIterations ?? MAPPER_DEFAULTS.pairRefinementIterations;
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
  const initialAngle = options.minInitialAngleDeg ?? MAPPER_DEFAULTS.minInitialAngleDeg,
    pointAngle = options.minTriangulationAngleDeg ?? MAPPER_DEFAULTS.minTriangulationAngleDeg,
    minPnP = options.minPnPInliers ?? MAPPER_DEFAULTS.minPnPInliers,
    minInitial = options.minInitialPoints ?? MAPPER_DEFAULTS.minInitialPoints;
  const bundleIterations = options.bundleIterations ?? MAPPER_DEFAULTS.bundleIterations,
    finalBundle = options.finalBundleIterations ?? MAPPER_DEFAULTS.finalBundleIterations;
  const localSize = options.localBundleSize ?? MAPPER_DEFAULTS.localBundleSize,
    globalViewRatio = options.globalBundleViewRatio ?? DEFAULT_GLOBAL_BUNDLE_RATIO,
    globalPointRatio = options.globalBundlePointRatio ?? DEFAULT_GLOBAL_BUNDLE_RATIO,
    globalFrequency = options.globalBundleFrequency ?? MAPPER_DEFAULTS.globalBundleFrequency,
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
    const key = pairKey(p.first, p.second);
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
      localSubset: options.localBundleSubset ?? MAPPER_DEFAULTS.localBundleSubset,
      globalViewRatio,
      globalPointRatio,
      globalFrequency,
      stallTolerance,
    },
  };
}

export { DEFAULT_BUNDLE_STALL_TOLERANCE, DEFAULT_GLOBAL_BUNDLE_RATIO } from './defaults.ts';
