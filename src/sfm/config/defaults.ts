import { DEFAULT_SEED } from '../../geometry/operator-defaults.ts';

/**
 * The defaults of the sparse pipeline, each written once (docs/library-structure.md, section 5,
 * phase 2). Where two entry points hold different values for one option, each keeps its own under
 * a name that says where it applies. The differences are facts of section 7, item 5; this module
 * does not resolve them.
 */

/** The defaults of `IncrementalOptions` (`incrementalControls`). */
export const MAPPER_DEFAULTS = {
  seed: DEFAULT_SEED,
  ransacIterations: 750,
  maxErrorPx: 3,
  minInitialAngleDeg: 1,
  minTriangulationAngleDeg: 0.5,
  minPnPInliers: 12,
  minInitialPoints: 12,
  bundleIterations: 8,
  /** The mapper's final BA. A session's is `SESSION_DEFAULTS.finalBundleIterations`. */
  finalBundleIterations: 25,
  localBundleSize: 6,
  localBundleSubset: true,
  globalBundleFrequency: 0,
  pairRefinementIterations: 0,
  refineSharedFocal: false,
} as const;

/** The defaults that `IncrementalSfmSession` holds for its own options. */
export const SESSION_DEFAULTS = {
  /** The local optimization window, 8..64 cameras. */
  windowSize: 32,
  /** The session's final BA: 8, against the mapper's 25 (`MAPPER_DEFAULTS`). */
  finalBundleIterations: 8,
} as const;

/** COLMAP's `ba_global_images_ratio` and `ba_global_points_ratio`: 10 % growth. */
export const DEFAULT_GLOBAL_BUNDLE_RATIO = 1.1;

/**
 * The default `bundleStallTolerance` of the mapper's BA passes: 1e-4. The solvers' own default,
 * for a direct call of the BA, is `BUNDLE_SOLVER_STALL_TOLERANCE`.
 */
export const DEFAULT_BUNDLE_STALL_TOLERANCE = 1e-4;

/**
 * The LM stop rule of the BA solvers: the CPU solvers' default `stallTolerance`, and the rule that
 * both BA shaders hold (`acceptBundleCandidate` of `bundle.shade.ts` and `joint-focal.shade.ts`).
 * The default ends a solve only at convergence.
 */
export const BUNDLE_SOLVER_STALL_TOLERANCE = 1e-9;

/** The defaults of the BA operators on both tiers (`bundleAdjust`, `bundleAdjustGpu`). */
export const BUNDLE_DEFAULTS = {
  iterations: 20,
  huberPx: 2,
  /** The PCG tolerance of the GPU solvers. */
  cgRelativeTolerance: 1e-6,
} as const;

/** The mapper's BA Huber threshold: half of `maxErrorPx`. */
export const bundleHuberPx = (maxErrorPx: number): number => maxErrorPx / 2;

/** The error bound of a mapper Huber threshold: the inverse of `bundleHuberPx`. */
export const maxErrorOfHuberPx = (huberPx: number): number => 2 * huberPx;
