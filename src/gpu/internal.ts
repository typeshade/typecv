/**
 * @internal The test entry of the GPU modules that `index.ts` does not export yet. It builds to
 * `dist/internal/gpu.js` (`scripts/build-gpu.mjs`), a bundle of its own with its own copy of the
 * runtime, so a test opens its `ShadeDevice` from this entry. It is no part of the package's
 * `exports`. `index.ts` exports `reconstructDenseGpu` (step 5 of `docs/dense-mvs.md`). This entry
 * keeps the stages of the dense MVS and `reconstructDense`, its form with the tests' options.
 */
export { ShadeDevice } from './device.ts';
export { planMvsViews } from './mvs-views.ts';
export type { MvsScene, MvsViewPlan, MvsViewOptions } from './mvs-views.ts';
export {
  decodeMvsNormal,
  mvsDepthMapsGpu,
  mvsPhotometricGpu,
  reconstructDense,
  reconstructDenseGpu,
} from './mvs.ts';
export type {
  GpuDenseOptions,
  GpuDenseFusionProgress,
  GpuDenseResult,
  MvsDenseOptions,
  MvsDepthMap,
  MvsDepthMapsOptions,
  MvsDepthMapsResult,
  MvsDepthSnapshot,
  MvsFilteredMaps,
  MvsImage,
  MvsPhotometricOptions,
  MvsPhotometricResult,
} from './mvs.ts';
export { cleanDenseCloudGpu, gpuCleanPrimitives } from './cloud-clean.ts';
export { fuseTsdfGpu } from './tsdf.ts';
export { completePlanarDepths } from './planar-depth.ts';
/** @internal Consuming view-slot entry used by MVS and differential ownership tests. */
export { completeOwnedPlanarDepths } from './planar-depth.ts';
export type { TsdfGpuOptions, TsdfReport, TsdfProgress, TsdfPhase, TsdfPart } from './tsdf.ts';
export { fuseTsdf, planTsdfVolume, tsdfSamples, tsdfViews } from '../reconstruction/tsdf.ts';
export { labelDenseCloud } from '../reconstruction/dense-label.ts';
export { labelLandmarks } from './mvs.ts';
export type { DenseLabelOptions, DenseLabelReport } from '../reconstruction/dense-label.ts';
export type { TsdfInput, TsdfOptions, TsdfPoints, TsdfView } from '../reconstruction/tsdf.ts';
export { cleanCloudWith, voxelFrame } from '../reconstruction/cloud-clean.ts';
export { meshDenseCloudGpu } from './dense-mesh.ts';
export { trackKltGpu } from './klt-device.ts';
export { createKltPairCache } from './klt-pair-cache.ts';
export { expandObservationsGpu } from './observation-expansion.ts';
export type { KltGpuOptions, KltGpuResult } from './klt-device.ts';
export {
  KLT_DEFAULTS_640,
  KLT_END,
  buildKltPyramid,
  checkKltReference,
  KLT_BLOCK,
  KLT_DEFAULT_CAPACITY,
  KLT_MAX_CAPACITY,
  buildKltGray,
  kltCellGrid,
  kltAssignNearest,
  kltCellOf,
  kltReject,
  kltEigenMapReference,
  kltEigenMaxReference,
  kltLayout,
  kltOptionsFor,
  kltScharrGradient,
  kltSelectReference,
  trackKlt,
  trackKltReference,
} from './klt.ts';
export { createKltTracker, kltResidentBytes } from './klt-tracker.ts';
export { extractSiftGpuResident } from './sift.ts';
export type { KltOptions, KltPyramid, KltTrackResult } from './klt.ts';
/**
 * @internal The GPU backend of `buildViewGraph` (`docs/multi-video.md` step 4) and the SIFT
 * extraction its test reads, from one runtime copy. Step 6 decides the public export.
 */
export { gpuViewGraphBackend } from './view-graph.ts';
export { verifyImagePairGpu, verifyImagePairsGpu } from './image-pair.ts';
/**
 * @internal The multi-video pipeline of the lab's worker (`docs/multi-video.md` step 6). The
 * record leaves the public export open; step 6 keeps it internal, and the site ships this entry
 * as `/runtime/internal/gpu.js`.
 */
export {
  describeKeyframeGpu,
  reconstructMultiVideoGpu,
  reconstructMultiVideoOwnedGpu,
} from './multi-video.ts';
export type {
  MultiVideoComponent,
  MultiVideoKeyframe,
  MultiVideoOptions,
  MultiVideoResult,
} from './multi-video.ts';
/** @internal Consuming worker-only lifetime; no immutable public API change. */
export type { OwnedMultiVideoOptions, OwnedMultiVideoResult } from './multi-video.ts';
export { globalDescriptorGpu } from './global-descriptor-gpu.ts';
export { parseVocabulary } from './global-descriptor.ts';
export { quantizeDescriptors } from '../reconstruction/view-graph.ts';
