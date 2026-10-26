export { ShadeDevice } from '../device/gpu/device.ts';
export { bundleAdjustInverseDepthGpu } from './inverse-depth.ts';
export { InverseDepthTracker } from '../reconstruction/inverse-depth-tracker.ts';
export type {
  InverseDepthTracking,
  InverseDepthTrackingOptions,
  InverseDepthSnapshot,
  DepthTrackState,
} from '../reconstruction/inverse-depth-tracker.ts';
export type {
  InverseDepthConstraint,
  InverseDepthBundleOptions,
  InverseDepthBundleResult,
} from '../reconstruction/inverse-depth-bundle.ts';
export type { DeviceDiagnostics } from '../device/gpu/device-diagnostics.ts';
export type {
  ShadeDeviceOptions,
  ShadePlanReport,
  ChunkRate,
  KernelStatistic,
} from '../device/gpu/device.ts';
export { GpuP3pBatch, GpuP3pStatus, solveP3pBatchGpu } from './p3p.ts';
export type { GpuP3pResult } from './p3p.ts';
export { estimatePosePnPGpu } from './pnp.ts';
export type { GpuPnPOptions, GpuPnPResult, GpuGeometryExecution } from './pnp.ts';
export { triangulateTracksGpu } from './triangulation.ts';
export type { GpuTriangulationOptions, GpuTriangulationResult } from './triangulation.ts';
export { evaluateObservationsGpu } from './observations.ts';
export type { GpuObservationResult } from './observations.ts';
export {
  reconstructIncrementalGpu,
  refineIncrementalGpu,
  reconstructIncrementalFloatGpu,
  IncrementalSfmSession,
} from './mapper.ts';
export type {
  GpuIncrementalOptions,
  GpuIncrementalResult,
  GpuFloatImageFeatures,
  GpuFloatImagePair,
  GpuFloatIncrementalOptions,
  GpuFloatIncrementalResult,
} from './mapper.ts';
export { matchFloatDescriptorsGpu } from '../features/gpu/float-matching.ts';
export { equalizeGrayCpu, equalizeGrayGpu } from '../image/gpu/clahe.ts';
export type { ClaheOptions } from '../image/gpu/clahe.ts';
export { extractOrbGpuResident } from '../features/gpu/orb.ts';
export type { GpuOrbOptions, GpuResidentOrb } from '../features/gpu/orb.ts';
export { extractOrbCpu } from '../features/gpu/orb-reference.ts';
export type { OrbFeaturesCpu, OrbParameters } from '../features/gpu/orb-reference.ts';
export { slic, slicGpu } from '../analysis/gpu/slic.ts';
export type { SlicOptions, SlicResult } from '../analysis/gpu/slic.ts';
export { canny, cannyGpu, houghLines, houghLinesGpu } from '../analysis/gpu/hough.ts';
export type {
  CannyOptions,
  CannyResult,
  HoughLine,
  HoughLinesOptions,
  HoughLinesResult,
  HoughSegment,
} from '../analysis/gpu/hough.ts';
export { extractSiftGpuResident } from '../features/gpu/sift.ts';
export type {
  GpuSiftOptions,
  GpuResidentSift,
  GpuSiftExecution,
  GpuSiftGray,
} from '../features/gpu/sift.ts';
export { createKltTracker, kltResidentBytes, KLT_LENGTH_EDGES } from './klt-tracker.ts';
export { expandObservationsGpu } from './observation-expansion.ts';
export { appendExpandedObservations } from './observation-append.ts';
export type {
  ObservationAppendOptions,
  ObservationAppendRejection,
  ObservationAppendResult,
} from './observation-append.ts';
export type {
  ObservationExpansionFrame,
  ObservationExpansionOptions,
  ObservationExpansionResult,
  ExpandedObservation,
  ExpandedObservationEdge,
} from './observation-expansion.ts';
export type {
  KltFrame,
  KltJoin,
  KltJoinInput,
  KltJoinPair,
  KltStats,
  KltPushOptions,
  KltStep,
  KltTracker,
  KltTrackerOptions,
  KltTracks,
} from './klt-tracker.ts';
export {
  KLT_DEFAULTS_640,
  KLT_DEFAULT_CAPACITY,
  KLT_END,
  KLT_MAX_CAPACITY,
  kltOptionsFor,
  kltReject,
} from './klt.ts';
export type { KltOptions } from './klt.ts';
export type {
  GpuResidentFloatDescriptors,
  GpuFloatMatchOptions,
  GpuFloatMatches,
} from '../features/gpu/float-matching.ts';
export { bundleAdjustGpu } from './bundle.ts';
export type { GpuBundleAdjustmentOptions, GpuBundleAdjustmentResult } from './bundle.ts';
export { EpipolarSession } from './epipolar.ts';
export type { EpipolarScore } from './epipolar.ts';
export { estimateEssentialRansacGpu, reconstructTwoViewGpu } from './two-view.ts';
export type {
  GpuTwoViewOptions,
  GpuConsensusExecution,
  GpuEssentialConsensus,
  GpuTwoViewResult,
} from './two-view.ts';

export { estimateHomographyGpu } from './homography.ts';
export type { GpuHomographyOptions, GpuHomographyResult } from './homography.ts';
export { undistortCorrespondencesGpu } from './pair-estimation.ts';
export type { GpuUndistortionResult } from './pair-estimation.ts';
export { verifyImagePairGpu } from './image-pair.ts';
/** Normalized camera coordinates of a pixel, radial distortion inverted (for guided matching). */
export { unprojectPixel } from '../geometry/camera.ts';
export type { GpuImagePairOptions, GpuImagePairVerification } from './image-pair.ts';
export {
  extractFeaturesGpu,
  extractFeaturesGpuPyramidResident,
  extractFeaturesGpuResident,
  matchFeaturesGpu,
  matchResidentFeaturesGpu,
} from '../features/gpu/features.ts';
export type {
  GpuFeatureExecution,
  GpuFeatureOptions,
  GpuFeatures,
  GpuFeatureMatches,
  GpuResidentFeatures,
} from '../features/gpu/features.ts';

export { reconstructDenseGpu } from './mvs.ts';
export type {
  GpuDenseOptions,
  GpuDenseDepthDiagnostic,
  GpuDensePoints,
  GpuDenseProgress,
  GpuDenseFusionProgress,
  GpuDenseResult,
  GpuDenseStage,
  GpuDenseView,
  GpuDenseViewEvent,
  GpuDenseViewStatus,
  DenseLabelOptions,
  DenseLabelReport,
  MvsImage,
  TsdfOptions,
  TsdfReport,
} from './mvs.ts';
export {
  cleanDenseCloudGpu,
  meanNeighbourDistancesGpu,
  removeStatisticalOutliersGpu,
  voxelDownsampleGpu,
} from './cloud-clean.ts';
export type {
  CleanCloud,
  DenseCleanOptions,
  DenseCleanReport,
  DenseCleanResult,
  OutlierOptions,
  OutlierResult,
  VoxelOptions,
  VoxelResult,
} from '../reconstruction/cloud-clean.ts';
export { meshDenseCloudGpu } from './dense-mesh.ts';
export type {
  DenseMesh,
  DenseMeshResult,
  MeshOptions,
  MeshReport,
} from '../reconstruction/dense-mesh.ts';
export type { MvsScene, MvsSceneLandmark, MvsSceneView, MvsViewOptions } from './mvs-views.ts';
export type {
  MvsDepthResidency,
  MvsFusionResidency,
  MvsResidency,
  MvsResidencyMode,
} from './mvs-residency.ts';

export { extractFeaturesTiledGpu, featureTileRegions } from '../features/gpu/tiled-features.ts';
export type {
  GpuImageSource,
  ImageRegion,
  GpuTiledFeatureOptions,
  GpuTiledFeatures,
} from '../features/gpu/tiled-features.ts';
export {
  BAG_OF_WORDS_VOCABULARY_ID,
  BAG_OF_WORDS_WORDS,
  GLOBAL_DESCRIPTOR_KIND,
  GLOBAL_DIMENSIONS,
  GLOBAL_VOCABULARY_ID,
  GLOBAL_WORDS,
  assignWordsReference,
  bagOfWordsDescriptor,
  bagOfWordsDescriptorReference,
  cosineSimilarity,
  globalDescriptorReference,
  inverseDocumentFrequency,
  parseVocabulary,
  vladDescriptorReference,
  vladFromWords,
  wordHistogram,
} from '../features/gpu/global-descriptor.ts';
export type {
  GlobalDescriptorKind,
  GlobalDescriptorOptions,
} from '../features/gpu/global-descriptor.ts';
export {
  assignWordsGpu,
  bagOfWordsDescriptorGpu,
  globalDescriptorGpu,
  vladDescriptorGpu,
} from '../features/gpu/global-descriptor-gpu.ts';
export type { GlobalAssignOptions } from '../features/gpu/global-descriptor-gpu.ts';
