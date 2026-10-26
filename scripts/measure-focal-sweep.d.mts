/** Declarations for scripts/measure-focal-sweep.mjs, so that its unit test typechecks. */
import type { Camera } from '../src/geometry/camera.ts';
import type { FocalCandidate } from '../src/reconstruction/focal-search.ts';
import type {
  ImageFeatures,
  PairMatches,
  SparseReconstruction,
} from '../src/reconstruction/types.ts';

export declare const MIN_PAIR_MATCHES: number;
export declare const HUBER_PX: number;

/** A view of the export (typecv-reconstruction-v1), as far as the sweep reads it. */
export interface ExportView {
  readonly id: number;
  readonly time?: number;
  readonly width?: number;
  readonly height?: number;
  readonly camera: Camera;
}
export interface ExportLandmark {
  readonly observations: readonly {
    readonly viewId: number;
    readonly pixel: ArrayLike<number>;
  }[];
}
export interface ExportModel {
  readonly views: readonly ExportView[];
  readonly landmarks: readonly ExportLandmark[];
}
export interface SweepOptions {
  readonly model: string;
  readonly ratios: number[];
  readonly views?: number;
  readonly minMatches: number;
  readonly finalBundleIterations: number;
  readonly out?: string;
  readonly models?: string;
}
export declare function parseOptions(argv: readonly string[]): SweepOptions;
export declare function calibrationViewIds(views: readonly ExportView[], n: number): number[];
export declare function tracksToInput(
  exported: ExportModel,
  viewIds: Iterable<number>,
  options?: { readonly minMatches?: number },
): { images: ImageFeatures[]; pairs: PairMatches[]; droppedPairs: number };
export declare function firstLastTurnDeg(
  views: readonly { readonly id: number; readonly pose: { readonly R: ArrayLike<number> } }[],
): number;
export interface SweepRow {
  readonly ratio: number;
  readonly views: number;
  readonly landmarks: number;
  readonly observations: number;
  readonly rmsePx: number;
  readonly meanErrorPx: number;
  readonly firstLastTurnDeg: number;
  readonly ms: number;
  readonly error?: undefined;
}
export interface SweepError {
  readonly ratio: number;
  readonly error: string;
}
/** The part of a mapper result (`IncrementalResult`) that a row reads. */
export interface SweepResult {
  readonly reconstruction: {
    readonly views: readonly {
      readonly id: number;
      readonly pose: { readonly R: ArrayLike<number> };
    }[];
    readonly landmarks: readonly { readonly observations: readonly unknown[] }[];
  };
  readonly diagnostics: {
    readonly bundleAdjustments: readonly { readonly finalRmsePx: number }[];
    readonly meanReprojectionErrorPx: number;
  };
}
export declare function sweepRow(ratio: number, result: SweepResult, ms: number): SweepRow;
export declare function scoreTable(
  rows: readonly (SweepRow | SweepError)[],
  huberPx?: number,
): FocalCandidate[];
export declare function scoreLine(candidate: FocalCandidate): string;
export declare function runAtRatio(
  images: readonly ImageFeatures[],
  pairs: readonly PairMatches[],
  ratio: number,
  options?: { readonly finalBundleIterations?: number },
):
  | { row: SweepRow; reconstruction: SparseReconstruction }
  | { row: SweepError; reconstruction?: undefined };
export interface RunModel {
  readonly ratio: number;
  readonly views: {
    readonly id: number;
    readonly time?: number;
    readonly width?: number;
    readonly height?: number;
    readonly camera: Camera;
    readonly pose: { readonly R: number[]; readonly t: number[] };
  }[];
  readonly landmarks: {
    readonly xyz: number[];
    readonly observations: { viewId: number; featureId: number; pixel: number[] }[];
  }[];
}
export declare function runModel(
  ratio: number,
  reconstruction: SparseReconstruction,
  exportViews: readonly ExportView[],
): RunModel;
