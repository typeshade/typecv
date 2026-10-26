/** Declarations for scripts/measure-mvs-sources.mjs, so that its unit test typechecks. */
import type {
  MvsScene,
  MvsSceneView,
  MvsViewOptions,
  MvsViewPlan,
  MvsViewStatus,
} from '../src/gpu/mvs-views.ts';

/** A view of the export: the MVS scene view and the time of its frame in seconds. */
export interface SourceSceneView extends MvsSceneView {
  readonly time?: number;
}
export interface SourceScene extends MvsScene {
  readonly views: readonly SourceSceneView[];
}
/** One source of a reference view. Distances are in the scene's units. */
export interface SourceLink {
  readonly id: number;
  /** The difference of the two views' positions in time order. */
  readonly keyframeGap: number;
  /** Seconds; NaN when a view has no time. */
  readonly timeGap: number;
  readonly sharedLandmarks: number;
  /** The median over the shared landmarks; NaN when there is none. */
  readonly medianAngleDeg: number;
  /** The distance of the camera centres over the reference's median landmark depth. */
  readonly baselineOverDepth: number;
}
export interface ReferenceRow {
  readonly id: number;
  readonly time?: number;
  readonly status: MvsViewStatus;
  /** Normalized units, as the plan holds it. */
  readonly depthRange?: readonly [number, number];
  readonly landmarks: number;
  /** The median depth of the landmarks that the view observes, in the scene's units. */
  readonly medianDepth: number;
  /** Best first. */
  readonly sources: readonly SourceLink[];
}
export interface SourceSummary {
  readonly views: number;
  /** The views that get a depth map. */
  readonly references: number;
  readonly links: number;
  readonly medianAngleDeg: number;
  readonly medianKeyframeGap: number;
  readonly medianTimeGap: number;
  /** NaN without a link. */
  readonly maxKeyframeGap: number;
  readonly shareBelow5Deg: number;
  readonly shareBelow10Deg: number;
}
export declare function parseOptions(argv: readonly string[]): {
  model?: string;
  out?: string;
  viewOptions: MvsViewOptions;
};
export declare function timeOrder<View extends { readonly time?: number }>(
  views: readonly View[],
): View[];
export declare function keyframeRanks(
  views: readonly { readonly id: number; readonly time?: number }[],
): Map<number, number>;
export declare function vectorAngleDeg(u: ArrayLike<number>, v: ArrayLike<number>): number;
export declare function summarizeSources(
  scene: SourceScene,
  plan: Pick<MvsViewPlan, 'views' | 'packed'>,
): { rows: ReferenceRow[]; summary: SourceSummary };
export declare function formatRow(row: ReferenceRow): string;
export declare function formatSummary(summary: SourceSummary): string[];
