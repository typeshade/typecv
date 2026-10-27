import type {
  IncrementalOptions,
  IncrementalResult,
  PreVerifiedPair,
  PairVerification,
  ReconstructionJob,
} from '../mapper/mapper.ts';
import { incrementalControls } from '../config/incremental-controls.ts';
import { judgePair } from '../pairs/pair-verdict.ts';
import {
  ReconstructionInitializationError,
  ReconstructionOperationError,
} from '../protocol/errors.ts';
import { pairBase, verificationFromBase, type PairBase } from './focal-pairs.ts';
import type { Camera } from '../../contracts/camera.ts';
import type {
  FeatureTrack,
  ImageFeatures,
  PairMatches,
  RegisteredView,
} from '../../contracts/scene.ts';
import { MAPPER_DEFAULTS, bundleHuberPx } from '../config/defaults.ts';
import { refineTrackPoint, triangulateLinear } from '../../geometry/multiview.ts';
import { projectPoint } from '../../geometry/pose.ts';
import { buildTracks } from '../tracks/tracks.ts';

/**
 * The automatic estimate of one shared focal length (docs/auto-focal.md). The mapper runs on a
 * few views at selected focals of a fixed grid, with the focal held. Each run is scored by the
 * truncated reprojection error of its map on one common observation set, the tracks of the
 * subset's input matches (`commonSetScore`, `scoreCandidates`). The best grid point is refined by
 * a parabola and by a joint focal bundle adjustment on the same views.
 */
export interface FocalSearchOptions {
  /** Views in the first subset of a match-connected component (default 5). Input order breaks component ties. */
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
  /** The observations the run kept; in directional mode, the last BA's full finite/bearing population. */
  readonly observations: number;
  /** Present in directional mode: 3 per finite point, 2 per bearing, plus free pose parameters. */
  readonly modelParameters?: number;
  /** Bearing-only observations included in this candidate's BA and RMS; present in directional mode. */
  readonly directionalObservations?: number;
  /**
   * The truncated mean square error in px² (`commonSetScore`): the run's poses triangulate every
   * track of the subset's input matches, and each observation costs its squared reprojection
   * error, at most the square of the BA's Huber threshold. Infinity when it failed.
   * With a supported bearing model at the largest view count, normalized BIC instead
   * (dimensionless; the search reports scoring='directional-bic'). See scoreCandidates.
   */
  readonly score: number;
}

export interface FocalSearchResult {
  /** The estimate: focal / width. Each view's `fx` becomes `ratio × width`; `fy` keeps its ratio to `fx`. */
  readonly ratio: number;
  /** The view ids of the subset that decided it. */
  readonly views: readonly number[];
  /** Evaluated grid points, sorted by ratio. Unpromising intervals may remain unevaluated. */
  readonly candidates: readonly FocalCandidate[];
  /** The ratio before the joint refinement on the subset: the grid's best point after the parabola. */
  readonly gridRatio: number;
  /** Only the experimental directional mode uses complexity-aware, normalized BIC scores. */
  readonly scoring?: 'directional-bic';
  /** Optional mixed-model basin refinements; ratios are the resulting focal estimates. */
  readonly refinements?: readonly (FocalCandidate & { readonly initialRatio: number })[];
}

/** One failed focal trial. Counts describe geometric checks, not calibrated probabilities. */
export interface FocalInitializationAttempt {
  readonly ratio: number;
  readonly views: number;
  readonly pairs: number;
  readonly eligiblePairs: number;
  readonly bestPair?: PairVerification;
  readonly reasons: readonly { readonly reason: string; readonly pairs: number }[];
  readonly failure: string;
}

/** Bounded evidence from a search that could not initialize any component. */
export interface FocalSearchFailure {
  readonly attempts: readonly FocalInitializationAttempt[];
  readonly omittedAttempts: number;
}

/**
 * @internal Prefer the largest match-connected component over a disconnected chronological
 * prefix. Raw edges only select evidence to test; the mapper still applies every geometric
 * and degeneracy check. Preserve input order within a component, and try smaller components
 * if larger ones cannot initialize. An explicit seed retains the caller's original ordering.
 */
export function focalSearchComponents(
  images: readonly ImageFeatures[],
  pairs: readonly PairMatches[],
  initialPair?: readonly [number, number],
): readonly (readonly ImageFeatures[])[] {
  if (initialPair) return [images];
  const adjacent = new Map(images.map((image) => [image.id, new Set<number>()]));
  for (const pair of pairs) {
    // The essential solver needs eight correspondences; weaker edges cannot connect a seed.
    if (pair.matches.length < 16) continue;
    adjacent.get(pair.first)!.add(pair.second);
    adjacent.get(pair.second)!.add(pair.first);
  }
  const remaining = new Set(images.map((image) => image.id));
  const components: ImageFeatures[][] = [];
  for (const image of images) {
    if (!remaining.delete(image.id)) continue;
    const ids = new Set([image.id]),
      pending = [image.id];
    for (let i = 0; i < pending.length; i++)
      for (const id of adjacent.get(pending[i]!)!)
        if (remaining.delete(id)) {
          ids.add(id);
          pending.push(id);
        }
    if (ids.size >= 2) components.push(images.filter((view) => ids.has(view.id)));
  }
  return components.length ? components.sort((a, b) => b.length - a.length) : [images];
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
  // A change of finite/bearing model is a discrete boundary, not a smooth focal basin.
  if (
    candidates.some((c) => c.views === b.views && c.directionalObservations) &&
    (a.modelParameters !== b.modelParameters || c.modelParameters !== b.modelParameters)
  )
    return b.ratio;
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

function directionalScoring(
  runs: readonly Pick<FocalCandidate, 'views' | 'directionalObservations'>[],
): boolean {
  const maxViews = Math.max(0, ...runs.map((r) => r.views));
  return runs.some((r) => r.views === maxViews && !!r.directionalObservations);
}

/**
 * @internal One grid point's run before scoring. `commonScore` is its `commonSetScore`; a run
 * without it is scored on the observations it kept.
 */
export type FocalRun = Omit<FocalCandidate, 'score'> & { readonly commonScore?: number };

/**
 * @internal The tracks that every grid point of a subset is scored on: the conflict-free tracks of
 * the subset's input matches (`buildTracks`), fixed before the sweep, so the set does not depend
 * on the focal (docs/auto-focal.md, section 10).
 */
export function commonTracks(
  subset: readonly ImageFeatures[],
  subsetPairs: readonly PairMatches[],
): readonly FeatureTrack[] {
  return buildTracks(subset, subsetPairs).tracks;
}

/**
 * @internal The truncated mean square error in px² of one map on the common tracks. The map's
 * registered views triangulate each track that has two or more observations in them (linear,
 * then refined; no angle gate), and each such observation costs `min(e², τ²)`, with `e` its
 * reprojection error and `τ` the BA's Huber threshold. A track that does not triangulate costs
 * `τ²` for each observation. A smaller focal makes every triangulation angle larger, so the
 * mapper's 0.5° gate keeps more tracks there; on this set a focal gains nothing from that.
 */
export function commonSetScore(
  tracks: readonly FeatureTrack[],
  subset: readonly ImageFeatures[],
  views: readonly RegisteredView[],
  huberPx: number,
): number {
  const registered = new Map(views.map((view) => [view.id, view])),
    keypoints = new Map(subset.map((image) => [image.id, image.keypoints])),
    cap = huberPx * huberPx;
  let cost = 0,
    count = 0;
  for (const track of tracks) {
    const observations = [];
    for (const { viewId, featureId } of track.observations) {
      const view = registered.get(viewId),
        k = keypoints.get(viewId);
      if (view && k)
        observations.push({
          camera: view.camera,
          pose: view.pose,
          pixel: [k[2 * featureId]!, k[2 * featureId + 1]!] as const,
        });
    }
    if (observations.length < 2) continue;
    count += observations.length;
    const initial = triangulateLinear(observations),
      point = initial && refineTrackPoint(initial, observations);
    for (const o of observations) {
      const pixel = point && projectPoint(o.camera, o.pose, point),
        e = pixel ? Math.hypot(pixel[0] - o.pixel[0], pixel[1] - o.pixel[1]) : Infinity;
      cost += Math.min(e * e, cap);
    }
  }
  return count ? cost / count : Infinity;
}

/**
 * Score the runs of one subset. A run with a `commonScore` scores it: the runs of a subset are
 * then compared on one observation set (`commonSetScore`). Otherwise, a run at a wrong focal both
 * raises the error of what it keeps and drops observations, and the RMS error alone rewards the
 * dropping. Among the runs that registered the same number of views, `n` is the most observations
 * any of them kept; a run that kept `m` observations at RMS error `e` scores
 * `(m·e² + (n − m)·τ²) / n`, with `τ` the BA's Huber threshold. A supported bearing model enables
 * the complexity correction below; its runs carry no `commonScore`.
 */
export function scoreCandidates(
  // FocalRun, spelled out: the declarations strip the internal alias (tsconfig.gpu.json).
  runs: readonly (Omit<FocalCandidate, 'score'> & { readonly commonScore?: number })[],
  huberPx: number,
): FocalCandidate[] {
  const most = new Map<number, number>();
  for (const run of runs)
    if (run.views) most.set(run.views, Math.max(most.get(run.views) ?? 0, run.observations));
  // Keep the established score when the best-covered calibration subset has no supported
  // bearing model. A weaker partial map must not switch finite-only calibration objectives.
  const directional = directionalScoring(runs);
  return runs.map(({ commonScore, ...run }) => {
    const n = most.get(run.views) ?? 0;
    let score =
      run.views && n && Number.isFinite(run.rmsePx)
        ? (commonScore ??
          (run.observations * run.rmsePx ** 2 + (n - run.observations) * huberPx ** 2) / n)
        : Infinity;
    // Under a common unknown-variance Gaussian residual model, normalized BIC is
    // log(SSE / N) + k log(N) / N. Each pixel observation contributes two scalar residuals.
    // This is a model-selection heuristic, not an observability or correctness certificate.
    if (directional && run.modelParameters !== undefined && Number.isFinite(score)) {
      const N = 2 * n,
        k = run.modelParameters;
      score =
        k < N ? Math.log(Math.max(score / 2, Number.EPSILON)) + (k * Math.log(N)) / N : Infinity;
    }
    return { ...run, score };
  });
}

/** The index of the best candidate: the most views registered, then the smallest score, then the smaller ratio. */
export function bestCandidate(candidates: readonly FocalCandidate[]): number {
  let best = -1;
  candidates.forEach((c, i) => {
    if (!c.views || !Number.isFinite(c.score)) return;
    const b = candidates[best];
    if (!b || c.views > b.views || (c.views === b.views && c.score < b.score)) best = i;
  });
  return best;
}

/**
 * @internal Bounded restart focals for a final joint fit of the archived map. A calibration
 * that selected a grid boundary has no bracket on that side; the initial short sequence must
 * not permanently exclude other basins once later views provide more evidence. Keep the
 * current calibration first, then try the log-grid middle and the opposite endpoint. The
 * caller must jointly optimize poses, points and focal, and compare the same observations;
 * these are starting values, not replacement camera calibrations or confidence estimates.
 */
export function historyFocalSeeds(search: FocalSearchResult, currentRatio: number): number[] {
  if (!(currentRatio > 0) || !Number.isFinite(currentRatio))
    throw new RangeError('Invalid current focal ratio');
  const candidates = search.candidates,
    best = bestCandidate(candidates);
  if (best < 0 || candidates.length < 2 || (best !== 0 && best !== candidates.length - 1))
    return [currentRatio];
  const ratios = [
    currentRatio,
    candidates[Math.floor((candidates.length - 1) / 2)]!.ratio,
    candidates[best === 0 ? candidates.length - 1 : 0]!.ratio,
  ];
  return ratios.filter(
    (ratio, i) => ratios.findIndex((r) => Math.abs(Math.log(r / ratio)) < 1e-9) === i,
  );
}

/**
 * @internal Evaluate alternate grid points first, then both neighbours of every coarse local
 * minimum with the most registered views. Re-score after each batch using its observation
 * maximum. Failed/partial initialization, an edge winner or a flat coarse
 * score falls back to the whole grid. Each ratio is requested at most once, in either mode.
 * `exhaustive` retains the previous search as a measurement and test oracle, without adding
 * a lab setting or changing the mapper's numerical controls.
 */
export function* focalCandidateSteps(
  grid: readonly number[],
  views: number,
  huberPx: number,
  exhaustive = false,
): Generator<number, FocalCandidate[], FocalRun> {
  const runs = new Map<number, FocalRun>();
  const all = grid.map((_, i) => i);
  const coarse = all.filter((i) => i % 2 === 0 || i === grid.length - 1);
  const evaluate = function* (indices: readonly number[]) {
    for (const i of indices) if (!runs.has(i)) runs.set(i, yield grid[i]!);
  };
  const scored = () =>
    scoreCandidates(
      [...runs].sort(([a], [b]) => a - b).map(([, run]) => run),
      huberPx,
    );
  yield* evaluate(exhaustive || grid.length < 6 ? all : coarse);
  for (;;) {
    const candidates = scored(),
      best = bestCandidate(candidates),
      winner = candidates[best];
    if (runs.size === grid.length) return candidates;
    const coarseScores = coarse.map((i) => candidates.find((c) => c.ratio === grid[i])!);
    // Insufficient evidence to prune any interval. In particular, do not give up when only
    // an unevaluated focal can initialize the subset.
    if (
      !winner ||
      winner.views < views ||
      winner.ratio === grid[0] ||
      winner.ratio === grid.at(-1) ||
      coarseScores.some(
        (c, i) => i > 0 && c.views === views && c.score === coarseScores[i - 1]!.score,
      )
    ) {
      yield* evaluate(all);
      return scored();
    }
    const pending = new Set<number>();
    for (let k = 1; k < coarse.length - 1; k++) {
      const c = coarseScores[k]!,
        left = coarseScores[k - 1]!,
        right = coarseScores[k + 1]!;
      if (
        c.views === views &&
        (left.views < c.views || c.score <= left.score) &&
        (right.views < c.views || c.score <= right.score)
      ) {
        pending.add(coarse[k]! - 1);
        pending.add(coarse[k]! + 1);
      }
    }
    // Always bracket the current winner on the original fine grid, even when a new candidate
    // moved it away from the coarse minimum that first led to it.
    const at = grid.indexOf(winner.ratio);
    pending.add(at - 1);
    pending.add(at + 1);
    const missing = [...pending].filter((i) => i >= 0 && i < grid.length && !runs.has(i));
    if (!missing.length) return candidates;
    yield* evaluate(missing.sort((a, b) => a - b));
  }
}

function rescaled(images: readonly ImageFeatures[], ratio: number): ImageFeatures[] {
  return images.map((image) => ({ ...image, camera: cameraAtRatio(image.camera, ratio) }));
}

/**
 * @internal The mapper that the search runs at each grid point: `incrementalSteps`, which passes
 * itself (docs/library-structure.md, section 5, phase 4). So this module imports the mapper's
 * types only, and no import cycle joins it to the mapper at run time.
 */
export type FocalSearchMapper = (
  images: readonly ImageFeatures[],
  pairs: readonly PairMatches[],
  options: IncrementalOptions,
) => Generator<ReconstructionJob, IncrementalResult, unknown>;

/**
 * @internal Search the shared focal on a subset of the views, through the mapper's own numerical
 * requests, so that the CPU reference and the device driver run the same search.
 */
export function* focalSearchSteps(
  images: readonly ImageFeatures[],
  pairs: readonly PairMatches[],
  options: IncrementalOptions,
  controls: Required<FocalSearchOptions>,
  /** @internal The exhaustive reference for tests and measurements. */
  exhaustive = false,
  /** @internal Streaming progress waits for this many registered calibration views; finish allows short input. */
  minimumViews = 2,
  /**
   * The mapper of each run. It comes last, after the two internal parameters, so that a caller
   * written for this signature runs the same search on a library that takes six parameters (the
   * merge base in the compare job of CI).
   */
  mapperSteps: FocalSearchMapper,
): Generator<ReconstructionJob, FocalSearchResult, unknown> {
  const grid = focalGrid(controls);
  const mapper: IncrementalOptions = {
    ...options,
    focalSearch: undefined,
    // Candidate cameras are assumptions under evaluation, not supplied calibration.
    rotationTracking: false,
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
  const attempts: FocalInitializationAttempt[] = [];
  let omittedAttempts = 0;
  let pairVerification: readonly PairVerification[] = [];
  const rememberFailure = (
    diagnostics: readonly PairVerification[],
    ratio: number,
    views: number,
    failure: string,
  ) => {
    if (diagnostics.length >= pairVerification.length) pairVerification = diagnostics;
    const reasons = new Map<string, number>();
    for (const pair of diagnostics) {
      let reason = (
        pair.reason ??
        (pair.seedEligible ? 'Seed eligible; calibration incomplete' : 'No stable geometry')
      ).slice(0, 160);
      if (!reasons.has(reason) && reasons.size >= 15) reason = 'Other geometric rejections';
      reasons.set(reason, (reasons.get(reason) ?? 0) + 1);
    }
    const bestPair = [...diagnostics].sort(
      (a, b) =>
        Number(b.seedEligible) - Number(a.seedEligible) ||
        b.inliers - a.inliers ||
        b.matches - a.matches ||
        b.medianAngleDeg - a.medianAngleDeg,
    )[0];
    attempts.push({
      ratio,
      views,
      pairs: diagnostics.length,
      eligiblePairs: diagnostics.filter((pair) => pair.seedEligible).length,
      ...(bestPair ? { bestPair } : {}),
      reasons: [...reasons].map(([reason, pairs]) => ({ reason, pairs })),
      failure: failure.slice(0, 256),
    });
    if (attempts.length > 64) {
      attempts.shift();
      omittedAttempts++;
    }
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
      return yield* mapperSteps(rescaled(subset, ratio), subsetPairs, {
        ...mapper,
        preVerified: verifications?.(ratio),
        initialPair,
        refineSharedFocal: refine > 0,
        finalBundleIterations:
          refine > 0 ? refine : (options.bundleIterations ?? MAPPER_DEFAULTS.bundleIterations),
      });
    } catch (error) {
      // A device failure ends the run. A grid point that cannot build a map scores nothing.
      if (error instanceof ReconstructionOperationError) throw error;
      if (error instanceof ReconstructionInitializationError || error instanceof RangeError) {
        const diagnostics =
          error instanceof ReconstructionInitializationError ? error.pairVerification : [];
        rememberFailure(diagnostics, ratio, subset.length, `${error.name}: ${error.message}`);
        return undefined;
      }
      throw error;
    }
  };
  const huberPx = bundleHuberPx(options.maxErrorPx ?? MAPPER_DEFAULTS.maxErrorPx);
  const candidateOf = (
    result: IncrementalResult | undefined,
    ratio: number,
    common?: {
      readonly tracks: readonly FeatureTrack[];
      readonly subset: readonly ImageFeatures[];
    },
  ): FocalRun => {
    const last = result?.diagnostics.bundleAdjustments.at(-1);
    if (!result || !last) return { ratio, views: 0, rmsePx: Infinity, observations: 0 };
    return {
      ...(common
        ? {
            commonScore: commonSetScore(
              common.tracks,
              common.subset,
              result.reconstruction.views,
              huberPx,
            ),
          }
        : {}),
      ratio,
      views: result.reconstruction.views.length,
      rmsePx: last.finalRmsePx,
      observations:
        options.separateDirectionalObservations && last.directionalObservations
          ? last.observations
          : result.reconstruction.landmarks.reduce((n, p) => n + p.observations.length, 0),
      ...(options.separateDirectionalObservations
        ? {
            modelParameters:
              (last.pointParameters ?? 3 * result.reconstruction.landmarks.length) +
              6 * result.reconstruction.views.length -
              7,
            directionalObservations: last.directionalObservations ?? 0,
          }
        : {}),
    };
  };
  for (const component of focalSearchComponents(images, pairs, options.initialPair)) {
    if (component.length < minimumViews) continue;
    let size = Math.min(controls.views, component.length);
    for (;;) {
      const subset = component.slice(0, size),
        ids = new Set(subset.map((v) => v.id)),
        subsetPairs = pairs.filter((p) => ids.has(p.first) && ids.has(p.second));
      const verifications = sharedVerification(subset, subsetPairs);
      // One observation set for every grid point of the subset (section 10 of docs/auto-focal.md).
      // The bearing model keeps its own complexity-aware score on the observations it kept.
      const common = options.separateDirectionalObservations
        ? undefined
        : { tracks: commonTracks(subset, subsetPairs), subset };
      // The mapper's BA runs with a Huber threshold of half the error bound.
      const schedule = focalCandidateSteps(grid, size, huberPx, exhaustive);
      let next = schedule.next();
      while (!next.done) {
        const ratio = next.value;
        const result = yield* run(subset, subsetPairs, ratio, 0, verifications);
        if (result && result.reconstruction.views.length < minimumViews)
          rememberFailure(
            result.diagnostics.pairVerification,
            ratio,
            subset.length,
            `Only ${result.reconstruction.views.length} views registered for focal calibration; need ${minimumViews}`,
          );
        next = schedule.next(candidateOf(result, ratio, common));
      }
      const candidates = next.value;
      const best = bestCandidate(
        candidates.map((candidate) =>
          candidate.views >= minimumViews ? candidate : { ...candidate, views: 0 },
        ),
      );
      if (best >= 0) {
        if (
          options.separateDirectionalObservations &&
          controls.refineIterations > 0 &&
          candidates.some((c) => c.views === candidates[best]!.views && c.directionalObservations)
        ) {
          // Finite and bearing models have different degrees of freedom. The discrete grid can
          // favour the flexible finite model before the bearing basin reaches its true focal.
          // Refine the best candidate of each model, at most two runs, then compare the same
          // complexity-aware objective. No semantic labels or evaluator truth enter selection.
          const seeds = [false, true]
            .map((directional) =>
              bestCandidate(
                candidates.map((c) =>
                  c.views === candidates[best]!.views && !!c.directionalObservations === directional
                    ? c
                    : { ...c, views: 0 },
                ),
              ),
            )
            .filter((i) => i >= 0);
          const refinedRuns: (Omit<FocalCandidate, 'score'> & { initialRatio: number })[] = [];
          for (const seed of seeds) {
            const initialRatio = parabolaVertex(candidates, seed),
              refined = yield* run(
                subset,
                subsetPairs,
                initialRatio,
                controls.refineIterations,
                verifications,
              ),
              view = refined?.reconstruction.views[0],
              source = subset.find((v) => v.id === view?.id),
              estimate = view && source ? view.camera.fx / cameraWidth(source.camera) : NaN;
            if (
              refined &&
              refined.reconstruction.views.length >= candidates[best]!.views &&
              Number.isFinite(estimate) &&
              estimate >= initialRatio / controls.step &&
              estimate <= initialRatio * controls.step
            ) {
              const c = candidateOf(refined, estimate);
              refinedRuns.push({ ...c, modelParameters: c.modelParameters! + 1, initialRatio });
            }
          }
          const scored = scoreCandidates(
              [...candidates, ...refinedRuns],
              bundleHuberPx(options.maxErrorPx ?? MAPPER_DEFAULTS.maxErrorPx),
            ),
            winner = bestCandidate(
              scored.map((c) => (c.views >= minimumViews ? c : { ...c, views: 0 })),
            ),
            selected = scored[winner]!,
            refinements = refinedRuns.map((r, i) => ({
              ...scored[candidates.length + i]!,
              initialRatio: r.initialRatio,
            }));
          return {
            ratio: selected.ratio,
            views: subset.map((v) => v.id),
            candidates: scored.slice(0, candidates.length),
            gridRatio:
              winner < candidates.length
                ? selected.ratio
                : refinedRuns[winner - candidates.length]!.initialRatio,
            ...(directionalScoring(scored) ? { scoring: 'directional-bic' as const } : {}),
            refinements,
          };
        }
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
          if (!refined || refined.reconstruction.views.length < candidates[best]!.views)
            ratio = candidates[best]!.ratio;
          // The joint refinement starts in the grid's basin; keep it only inside the grid's bracket.
          if (
            refined &&
            refined.reconstruction.views.length >= candidates[best]!.views &&
            Number.isFinite(estimate) &&
            estimate >= gridRatio / controls.step &&
            estimate <= gridRatio * controls.step
          )
            ratio = estimate;
        }
        return {
          ratio,
          views: subset.map((v) => v.id),
          candidates,
          gridRatio,
          ...(options.separateDirectionalObservations && directionalScoring(candidates)
            ? { scoring: 'directional-bic' as const }
            : {}),
        };
      }
      if (size >= component.length) break;
      size = Math.min(component.length, size * 2);
    }
  }
  throw new ReconstructionInitializationError(
    pairVerification,
    'No focal of the search built an initial map: check parallax and verified matches',
    { attempts, omittedAttempts },
  );
}

/** @internal The images with the found focal. */
export function applyFocal(images: readonly ImageFeatures[], ratio: number): ImageFeatures[] {
  return rescaled(images, ratio);
}
