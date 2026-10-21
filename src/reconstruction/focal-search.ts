import {
  incrementalControls,
  incrementalSteps,
  judgePair,
  ReconstructionInitializationError,
  ReconstructionOperationError,
  type IncrementalOptions,
  type PreVerifiedPair,
  type ReconstructionJob,
} from './mapper.ts';
import { pairBase, verificationFromBase, type PairBase } from './focal-pairs.ts';
import type { Camera } from '../geometry/camera.ts';
import type { ImageFeatures, PairMatches } from './types.ts';

/**
 * The automatic estimate of one shared focal length (docs/auto-focal.md). The mapper runs on a
 * few views at each focal of a fixed grid, with the focal held. Each run is scored by its
 * truncated reprojection error after bundle adjustment (`scoreCandidates`). The best grid point is refined by a parabola and
 * by a joint focal bundle adjustment on the same views.
 */
export interface FocalSearchOptions {
  /** Views in the first subset, in input order (default 5). The subset doubles while no grid point initializes. */
  readonly views?: number;
  /** The smallest focal of the grid, as focal / width (default 0.3). */
  readonly minRatio?: number;
  /** The largest focal of the grid, as focal / width (default 3). */
  readonly maxRatio?: number;
  /** The factor between neighbouring grid points (default 1.15). */
  readonly step?: number;
  /** LM iterations of the joint focal bundle adjustment on the subset (default 25). */
  readonly refineIterations?: number;
}

/** One grid point of the search. */
export interface FocalCandidate {
  /** Focal / width. */
  readonly ratio: number;
  /** Views the subset run registered; 0 when it did not initialize. */
  readonly views: number;
  /** The RMS reprojection error of the run's last bundle adjustment in pixels; Infinity when it failed. */
  readonly rmsePx: number;
  /** The observations the run kept after its last bundle adjustment. */
  readonly observations: number;
  /**
   * The truncated mean square error in px² (`scoreCandidates`): an observation that another grid
   * point kept and this one dropped costs the square of the BA's Huber threshold. Infinity when it failed.
   */
  readonly score: number;
}

export interface FocalSearchResult {
  /** The estimate: focal / width. Each view's `fx` becomes `ratio × width`; `fy` keeps its ratio to `fx`. */
  readonly ratio: number;
  /** The view ids of the subset that decided it. */
  readonly views: readonly number[];
  readonly candidates: readonly FocalCandidate[];
  /** The ratio before the joint refinement on the subset: the grid's best point after the parabola. */
  readonly gridRatio: number;
}

/** The defaults of `FocalSearchOptions`. */
export const FOCAL_SEARCH_DEFAULTS = {
  views: 5,
  minRatio: 0.3,
  maxRatio: 3,
  step: 1.15,
  refineIterations: 25,
} as const;

/**
 * The image width that the search measures a focal against: `2·cx + 1`, the width of an image
 * whose principal point is its centre (pixel centres at integers).
 */
export function cameraWidth(camera: Camera): number {
  return 2 * camera.cx + 1;
}

/** The camera with `fx = ratio × width`; `fy` keeps its ratio to `fx`, and the rest stays. */
export function cameraAtRatio(camera: Camera, ratio: number): Camera {
  const fx = ratio * cameraWidth(camera);
  // The aspect first: it is exactly 1 for square pixels, so the result does not depend on the start.
  return { ...camera, fx, fy: fx * (camera.fy / camera.fx) };
}

/** @internal Validate the options; throws RangeError on an invalid one. */
export function focalSearchControls(
  options: boolean | FocalSearchOptions | undefined,
): Required<FocalSearchOptions> | undefined {
  if (options === undefined || options === false) return undefined;
  const o = options === true ? {} : options;
  const c = {
    views: o.views ?? FOCAL_SEARCH_DEFAULTS.views,
    minRatio: o.minRatio ?? FOCAL_SEARCH_DEFAULTS.minRatio,
    maxRatio: o.maxRatio ?? FOCAL_SEARCH_DEFAULTS.maxRatio,
    step: o.step ?? FOCAL_SEARCH_DEFAULTS.step,
    refineIterations: o.refineIterations ?? FOCAL_SEARCH_DEFAULTS.refineIterations,
  };
  if (
    !Number.isSafeInteger(c.views) ||
    c.views < 2 ||
    !(c.minRatio > 0) ||
    !(c.maxRatio >= c.minRatio) ||
    !Number.isFinite(c.maxRatio) ||
    !(c.step > 1.01) ||
    !Number.isFinite(c.step) ||
    Math.log(c.maxRatio / c.minRatio) / Math.log(c.step) > 256 ||
    !Number.isSafeInteger(c.refineIterations) ||
    c.refineIterations < 0
  )
    throw new RangeError('Invalid focal search options');
  return c;
}

/** The grid: `minRatio · step^i` up to `maxRatio`. */
export function focalGrid(controls: Required<FocalSearchOptions>): number[] {
  const out: number[] = [];
  for (let i = 0; ; i++) {
    const ratio = controls.minRatio * controls.step ** i;
    if (ratio > controls.maxRatio * (1 + 1e-9)) break;
    out.push(ratio);
  }
  return out;
}

/**
 * The vertex of the parabola through three points of `(log ratio, rmse)` around the best one,
 * held between its neighbours. Without two scored neighbours, the best point.
 */
export function parabolaVertex(candidates: readonly FocalCandidate[], best: number): number {
  const a = candidates[best - 1],
    b = candidates[best]!,
    c = candidates[best + 1];
  if (!a || !c || !Number.isFinite(a.score) || !Number.isFinite(c.score)) return b.ratio;
  if (a.views !== b.views || c.views !== b.views) return b.ratio;
  const x0 = Math.log(a.ratio),
    x1 = Math.log(b.ratio),
    x2 = Math.log(c.ratio);
  const p = (x1 - x0) * (b.score - c.score),
    q = (x1 - x2) * (b.score - a.score),
    denominator = p - q;
  // The best point is lowest, so a parabola through the three opens upwards unless they are collinear.
  if (!(Math.abs(denominator) > 0)) return b.ratio;
  const x = x1 - (0.5 * ((x1 - x0) * p - (x1 - x2) * q)) / denominator;
  return Math.exp(Math.min(x2, Math.max(x0, x)));
}

/**
 * Score the runs of one subset. A run at a wrong focal both raises the error of what it keeps and
 * drops observations, and the RMS error alone rewards the dropping. Among the runs that
 * registered the same number of views, `n` is the most observations any of them kept; a run that
 * kept `m` observations at RMS error `e` scores `(m·e² + (n − m)·τ²) / n`, with `τ` the BA's Huber
 * threshold.
 */
export function scoreCandidates(
  runs: readonly Omit<FocalCandidate, 'score'>[],
  huberPx: number,
): FocalCandidate[] {
  const most = new Map<number, number>();
  for (const run of runs)
    if (run.views) most.set(run.views, Math.max(most.get(run.views) ?? 0, run.observations));
  return runs.map((run) => {
    const n = most.get(run.views) ?? 0;
    return {
      ...run,
      score:
        run.views && n && Number.isFinite(run.rmsePx)
          ? (run.observations * run.rmsePx ** 2 + (n - run.observations) * huberPx ** 2) / n
          : Infinity,
    };
  });
}

/** The index of the best candidate: the most views registered, then the smallest score, then the smaller ratio. */
export function bestCandidate(candidates: readonly FocalCandidate[]): number {
  let best = -1;
  candidates.forEach((c, i) => {
    if (!c.views) return;
    const b = candidates[best];
    if (!b || c.views > b.views || (c.views === b.views && c.score < b.score)) best = i;
  });
  return best;
}

function rescaled(images: readonly ImageFeatures[], ratio: number): ImageFeatures[] {
  return images.map((image) => ({ ...image, camera: cameraAtRatio(image.camera, ratio) }));
}

/**
 * @internal Search the shared focal on a subset of the views, through the mapper's own numerical
 * requests, so that the CPU reference and the device driver run the same search.
 */
export function* focalSearchSteps(
  images: readonly ImageFeatures[],
  pairs: readonly PairMatches[],
  options: IncrementalOptions,
  controls: Required<FocalSearchOptions>,
): Generator<ReconstructionJob, FocalSearchResult, unknown> {
  const grid = focalGrid(controls);
  const mapper: IncrementalOptions = {
    ...options,
    focalSearch: undefined,
    onReconstruction: undefined,
    refineSharedFocal: false,
    focalGroups: undefined,
    preVerified: undefined,
  };
  const { controls: pairControls } = incrementalControls(images, pairs, options);
  /**
   * The pair verifications of a subset at any focal, from one base for each pair (section 8 of
   * docs/auto-focal.md). Undefined when a camera has lens distortion: the ideal pixels then
   * depend on the focal, and each grid point verifies its pairs in full.
   */
  const sharedVerification = (subset: readonly ImageFeatures[], subsetPairs: PairMatches[]) => {
    if (subset.some((v) => v.camera.k1 || v.camera.k2)) return undefined;
    const byId = new Map(subset.map((v) => [v.id, v]));
    const bases = new Map<PairMatches, PairBase>();
    return (ratio: number): PreVerifiedPair[] => {
      const at = (id: number) => cameraAtRatio(byId.get(id)!.camera, ratio);
      return subsetPairs.map((pair) => {
        let base = bases.get(pair);
        if (!base) {
          const a = byId.get(pair.first)!,
            b = byId.get(pair.second)!;
          base = pairBase(
            a,
            b,
            pair,
            cameraAtRatio(a.camera, 1),
            cameraAtRatio(b.camera, 1),
            pairControls,
          );
          bases.set(pair, base);
        }
        const judged = judgePair(
          pair,
          verificationFromBase(base, at(pair.first), at(pair.second), pairControls),
          pairControls,
        );
        return {
          verified: judged.accepted ?? pair,
          verification: judged.diagnostic,
          geometry: judged.candidate?.result,
          rejected: !judged.accepted,
        };
      });
    };
  };
  /** One mapper run on `subset` at `ratio`; undefined when it does not initialize. */
  const run = function* (
    subset: readonly ImageFeatures[],
    subsetPairs: PairMatches[],
    ratio: number,
    refine: number,
    verifications?: (ratio: number) => PreVerifiedPair[],
  ) {
    const ids = new Set(subset.map((v) => v.id));
    const initialPair =
      options.initialPair && ids.has(options.initialPair[0]) && ids.has(options.initialPair[1])
        ? options.initialPair
        : undefined;
    try {
      return yield* incrementalSteps(rescaled(subset, ratio), subsetPairs, {
        ...mapper,
        preVerified: verifications?.(ratio),
        initialPair,
        refineSharedFocal: refine > 0,
        finalBundleIterations: refine > 0 ? refine : (options.bundleIterations ?? 8),
      });
    } catch (error) {
      // A device failure ends the run. A grid point that cannot build a map scores nothing.
      if (error instanceof ReconstructionOperationError) throw error;
      if (error instanceof ReconstructionInitializationError || error instanceof RangeError)
        return undefined;
      throw error;
    }
  };
  let size = Math.min(controls.views, images.length);
  for (;;) {
    const subset = images.slice(0, size),
      ids = new Set(subset.map((v) => v.id)),
      subsetPairs = pairs.filter((p) => ids.has(p.first) && ids.has(p.second));
    const verifications = sharedVerification(subset, subsetPairs);
    const runs: Omit<FocalCandidate, 'score'>[] = [];
    for (const ratio of grid) {
      const result = yield* run(subset, subsetPairs, ratio, 0, verifications);
      const last = result?.diagnostics.bundleAdjustments.at(-1);
      runs.push(
        result && last
          ? {
              ratio,
              views: result.reconstruction.views.length,
              rmsePx: last.finalRmsePx,
              observations: result.reconstruction.landmarks.reduce(
                (sum, point) => sum + point.observations.length,
                0,
              ),
            }
          : { ratio, views: 0, rmsePx: Infinity, observations: 0 },
      );
    }
    // The mapper's BA runs with a Huber threshold of half the error bound.
    const candidates = scoreCandidates(runs, (options.maxErrorPx ?? 3) / 2);
    const best = bestCandidate(candidates);
    if (best >= 0) {
      const gridRatio = parabolaVertex(candidates, best);
      let ratio = gridRatio;
      if (controls.refineIterations > 0) {
        const refined = yield* run(
          subset,
          subsetPairs,
          gridRatio,
          controls.refineIterations,
          verifications,
        );
        const view = refined?.reconstruction.views[0];
        const source = subset.find((v) => v.id === view?.id);
        const estimate = view && source ? view.camera.fx / cameraWidth(source.camera) : NaN;
        // The joint refinement starts in the grid's basin; keep it only inside the grid's bracket.
        if (
          Number.isFinite(estimate) &&
          estimate >= gridRatio / controls.step &&
          estimate <= gridRatio * controls.step
        )
          ratio = estimate;
      }
      return { ratio, views: subset.map((v) => v.id), candidates, gridRatio };
    }
    if (size >= images.length)
      throw new ReconstructionInitializationError(
        [],
        'No focal of the search built an initial map: check parallax and verified matches',
      );
    size = Math.min(images.length, size * 2);
  }
}

/** @internal The images with the found focal. */
export function applyFocal(images: readonly ImageFeatures[], ratio: number): ImageFeatures[] {
  return rescaled(images, ratio);
}
