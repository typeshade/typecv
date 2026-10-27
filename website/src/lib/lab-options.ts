import { LAB_MAPPER, LAB_WINDOW_SIZE } from '../../../src/sfm/config/lab-preset.ts';
import type { KeyframeGateMode } from './keyframe-gate.ts';
import type { AnalysisEdge } from './sfm-video.ts';
import { VIDEO_SAMPLE_FPS } from './video-sampling.ts';

/**
 * The lab's options, each value once, for the page and both workers (docs/library-structure.md,
 * section 3.4 and section 5, phase 2). The mapper controls are the library's lab preset
 * (`src/sfm/config/lab-preset.ts`), which the scripts read too. Where the page and the worker
 * hold different defaults, each keeps its own here, under a name that says where it applies; the
 * differences are facts of section 7, item 5. `website/lab-options.test.mjs` holds the values.
 */

/** The dense MVS (`docs/dense-mvs.md` 5.3): the defaults of section 3, for both lab workers. */
export const DENSE_LAB_OPTIONS = {
  maxSources: 6,
  bestSources: 3,
  windowRadius: 5,
  windowStep: 2,
  photometricIterations: 8,
  geometricIterations: 4,
  minConsistentViews: 3,
  tsdf: { visibility: 'view-weighted' },
  planarDepth: true,
} as const;

/**
 * The fusion of the lab (`docs/dense-mvs.md` 10.4, the step 3 record): the TSDF fusion, which beat
 * the per-pixel fusion of 3.10 on ETH3D `pipes` in F-score at 2 and 5 cm and in landmark coverage.
 * `?denseFusion=pixel` runs the fusion of 3.10.
 */
export const DENSE_LAB_FUSION: DenseFusion = 'tsdf';

/** The fusions of `reconstructDenseGpu` (`fusion`). */
export type DenseFusion = 'pixel' | 'tsdf';

/**
 * Whether the lab cleans the fused cloud (3.12) by default: after the pixel fusion only. On
 * `pipes` the cleanup lowered the TSDF cloud's F-score at every threshold and raised its uncovered
 * landmarks (the step 3 record), as it did the pixel cloud's. `?denseClean=1` or `0` overrides it.
 */
export function denseCleanDefault(fusion: DenseFusion): boolean {
  return fusion === 'pixel';
}

/**
 * The cleanup of the fused cloud (3.12). The outlier and voxel stages take the defaults of
 * `cleanDenseCloudGpu` (k 16, 2 σ, a voxel of twice the median spacing). DBSCAN runs at 4
 * spacings and keeps the largest cluster and every cluster of at least 10 % of it.
 */
export const DENSE_CLEAN_OPTIONS = { cluster: { minFraction: 0.1 } } as const;

/** The lab's fixed numerical controls; reported with every result so a run record can carry them. */
export const LAB_OPTIONS = {
  windowSize: LAB_WINDOW_SIZE,
  ...LAB_MAPPER,
  /** `sift`: scale-space SIFT with resident f32 descriptors (the lab's front end); `corner`: the FAST/BRIEF pyramid front end, kept for comparison runs. */
  frontEnd: 'sift' as 'corner' | 'sift',
  /** Corner threshold of the `corner` front end; unused by SIFT. */
  threshold: 0.045,
  maxFeatures: 4096,
  smoothDescriptors: true,
  matchRatio: 0.8 as number,
  /** L2 bound of the SIFT matcher: no bound. */
  maxDistance: 1e19 as number,
  /** A frame becomes a keyframe when its median match displacement from the last keyframe reaches this many analysis pixels... */
  keyframeMinMotionPx: 8,
  /** ...or when this much time has passed since the last keyframe. The first frame is always a keyframe. */
  keyframeMaxGapSeconds: 1,
  /** Analysis frames per second the page takes from an uploaded video (`?fps=` overrides it); photos and samples are not sampled. */
  videoSampleFps: VIDEO_SAMPLE_FPS,
  /** Map snapshots to the page are posted at most this often; the final one is always posted. */
  snapshotIntervalMs: 500,
  /** Detailed stage entries kept for the page; per-operation totals cover every stage. */
  detailedStageLimit: 400,
  /** The cleanup of the fused cloud (3.12): `DENSE_CLEAN_OPTIONS`. */
  denseCleanOptions: DENSE_CLEAN_OPTIONS,
  /** The dense MVS (`docs/dense-mvs.md` 5.3): `DENSE_LAB_OPTIONS`. */
  dense: DENSE_LAB_OPTIONS,
} as const;

/** Still photographs with calibrated intrinsics: the shared focal is not refined. */
export const PHOTO_PROFILE = {
  refineSharedFocal: false,
} as const;

/** The ORB front end's lab defaults, chosen by measurement (docs/gpu-features.md, ORB): features per frame, ratio test and Hamming bound (none). */
export const ORB_LAB_FEATURES = 4000;
export const ORB_LAB_RATIO = 0.95;
export const ORB_MAX_DISTANCE = 256;

/**
 * The tracks the lab's tracker holds unless `?kltCapacity=` says otherwise: 512, against the
 * library's `KLT_DEFAULT_CAPACITY` of 1024 (section 7, item 5).
 */
export const LAB_KLT_CAPACITY = 512;

/**
 * The SfM worker's defaults for a `start` message that leaves a field out. The page sends each
 * of these fields with its own default: the edge `LAB_PAGE_ANALYSIS_EDGE`, the first octave and
 * guided matching of `siftQuality`, the gate `DEFAULT_KEYFRAME_GATE`, and the dense MVS and the
 * focal search on (section 7, item 5).
 */
export const LAB_WORKER_DEFAULTS: {
  readonly analysisEdge: number;
  readonly firstOctave: -1 | 0;
  readonly guided: boolean;
  readonly dense: boolean;
  readonly gate: KeyframeGateMode;
  readonly focalSearch: boolean;
} = {
  analysisEdge: 640,
  firstOctave: 0,
  guided: false,
  dense: false,
  gate: 'legacy',
  focalSearch: false,
};

/** The page's long side of a decoded video frame until the visitor picks another. */
export const LAB_PAGE_ANALYSIS_EDGE: AnalysisEdge = 1280;

/**
 * The options that the record of a multi-video run states as literals, whatever the run used: no
 * session window, no CLAHE and the phase gate (section 7, item 5).
 */
export const MULTI_VIDEO_RECORD_OPTIONS = {
  windowSize: 0,
  clahe: false,
  keyframeGate: 'phase',
} as const;
