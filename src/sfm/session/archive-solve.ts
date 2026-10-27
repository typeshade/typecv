import type { SparseReconstruction } from '../../contracts/scene.ts';
import type { BundleAdjustmentOptions } from '../bundle/bundle-adjustment.ts';
import { cameraWidth, historyFocalSeeds, type FocalSearchResult } from '../focal/focal-search.ts';
import { MAPPER_DEFAULTS, SESSION_DEFAULTS, bundleHuberPx } from '../config/defaults.ts';

// The archive-solve plan of `IncrementalSfmSession`'s final update (docs/library-structure.md,
// section 3.3, `sfm/session`; phase 6). A function of its inputs; the session runs the plan.

/** @internal The session options that the archive-solve plan reads. */
export interface ArchiveSolveOptions {
  readonly focalGroups?: ReadonlyMap<number, number>;
  readonly refineSharedFocal?: boolean;
  readonly separateDirectionalObservations?: boolean;
  readonly maxRelativeDepthStdDev?: number;
  readonly finalBundleIterations?: number;
  readonly historyBundleIterations?: number;
  readonly maxErrorPx?: number;
}

/**
 * @internal What the final update does with the archive: `solve` refines it with `refinement`,
 * `measure` records its cost and defers the solve, `none` leaves it as the window left it.
 */
export type ArchiveSolvePlan =
  | {
      readonly kind: 'solve';
      readonly refinement: BundleAdjustmentOptions & {
        readonly commonFocalRatios?: readonly number[];
      };
      /** The archive's focal refinement is deferred whatever the solve reports. */
      readonly focalDeferred: boolean;
    }
  | { readonly kind: 'measure' }
  | { readonly kind: 'none' };

/**
 * @internal The archive needs a full solve when it holds views outside the window, or when a
 * late view needs the shared focal of several seeds. The solve waits while the archive cannot own
 * bearing factors or conditional depth.
 */
export function archiveSolvePlan(
  archive: SparseReconstruction,
  window: SparseReconstruction,
  focal: FocalSearchResult | undefined,
  options: ArchiveSolveOptions,
): ArchiveSolvePlan {
  const fullWindow = archive.views.length > window.views.length,
    reference = window.views[0]!.camera,
    multipleFocalGroups =
      !!options.focalGroups &&
      new Set(archive.views.map((v) => options.focalGroups!.get(v.id))).size > 1,
    shape = (camera: typeof reference) => (camera.cy + 0.5) / (camera.cx + 0.5),
    sharedCamera =
      focal &&
      !options.focalGroups &&
      archive.views.every(
        ({ camera }) =>
          Math.abs(shape(camera) - shape(reference)) < 1e-9 &&
          Math.abs(camera.fy / camera.fx - reference.fy / reference.fx) < 1e-9 &&
          (camera.k1 ?? 0) === (reference.k1 ?? 0) &&
          (camera.k2 ?? 0) === (reference.k2 ?? 0),
      ),
    focalRatios =
      sharedCamera && options.refineSharedFocal
        ? historyFocalSeeds(focal!, reference.fx / cameraWidth(reference))
        : undefined,
    lateCalibration =
      (focalRatios?.length ?? 0) > 1 && archive.views.some((v) => !focal!.views.includes(v.id)),
    needsFullSolve = fullWindow || lateCalibration;
  if (
    needsFullSolve &&
    !options.separateDirectionalObservations &&
    !options.maxRelativeDepthStdDev
  ) {
    const finalIterations = options.finalBundleIterations ?? SESSION_DEFAULTS.finalBundleIterations;
    return {
      kind: 'solve',
      refinement: {
        iterations:
          finalIterations === 0
            ? 0
            : (options.historyBundleIterations ??
              ((focalRatios?.length ?? 0) > 1 ? Math.max(finalIterations, 32) : finalIterations)),
        huberPx: bundleHuberPx(options.maxErrorPx ?? MAPPER_DEFAULTS.maxErrorPx),
        refineSharedFocal: options.refineSharedFocal && !multipleFocalGroups,
        focalGroups: options.focalGroups,
        ...(focalRatios ? { commonFocalRatios: focalRatios } : {}),
      },
      focalDeferred: !!options.refineSharedFocal && multipleFocalGroups,
    };
  }
  if (needsFullSolve) return { kind: 'measure' };
  return { kind: 'none' };
}
