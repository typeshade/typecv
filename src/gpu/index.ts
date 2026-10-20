export { ShadeDevice } from './device.ts';
export type { ShadeDeviceOptions, ShadePlanReport, ChunkRate } from './device.ts';
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
export { matchFloatDescriptorsGpu } from './float-matching.ts';
export { equalizeGrayCpu, equalizeGrayGpu } from './clahe.ts';
export type { ClaheOptions } from './clahe.ts';
export { slic, slicGpu } from './slic.ts';
export type { SlicOptions, SlicResult } from './slic.ts';
export { extractSiftGpuResident } from './sift.ts';
export type { GpuSiftOptions, GpuResidentSift, GpuSiftExecution } from './sift.ts';
export type {
  GpuResidentFloatDescriptors,
  GpuFloatMatchOptions,
  GpuFloatMatches,
} from './float-matching.ts';
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
} from './features.ts';
export type {
  GpuFeatureExecution,
  GpuFeatureOptions,
  GpuFeatures,
  GpuFeatureMatches,
  GpuResidentFeatures,
} from './features.ts';

export { reconstructDenseGpu } from './mvs.ts';
export type {
  GpuDenseOptions,
  GpuDensePoints,
  GpuDenseProgress,
  GpuDenseResult,
  GpuDenseStage,
  GpuDenseView,
  GpuDenseViewEvent,
  GpuDenseViewStatus,
  MvsImage,
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

export { extractFeaturesTiledGpu, featureTileRegions } from './tiled-features.ts';
export type {
  GpuImageSource,
  ImageRegion,
  GpuTiledFeatureOptions,
  GpuTiledFeatures,
} from './tiled-features.ts';
