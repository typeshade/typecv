export { reconstructTwoView, sampsonSquared } from './geometry/two-view.ts';
export { GeometryEstimationError } from './geometry/errors.ts';
export { triangulate } from './geometry/triangulation.ts';
export type { TwoViewOptions, TwoViewResult } from './geometry/two-view.ts';
export type { CameraIntrinsics, Camera, Pixel, Pose } from './contracts/camera.ts';
export { cameraProjection, unprojectPixel, validateCamera } from './geometry/camera.ts';
export {
  identityPose,
  validatePose,
  cameraCenter,
  projectPoint,
  rotationExp,
} from './geometry/pose.ts';
export {
  triangulateTrack,
  triangulateTrackDetailed,
  assessTrackDepth,
} from './geometry/multiview.ts';
export type {
  TrackTriangulation,
  TriangulationObservation,
  TrackTriangulationOptions,
  DepthSensitivityOptions,
  DetailedTrackTriangulation,
} from './geometry/multiview.ts';
export { estimateHomography } from './geometry/homography.ts';
export { estimateRelativeRotation } from './geometry/rotation.ts';
export type { RelativeRotationOptions, RelativeRotationResult } from './geometry/rotation.ts';
export type { PairRotationEvidence, ReconstructionTracking } from './sfm/tracks/tracking.ts';
export { verifyImagePair } from './geometry/image-pair.ts';
export type {
  ImagePairOptions,
  ImagePairGeometry,
  ImagePairVerification,
} from './geometry/image-pair.ts';
export { estimatePosePnP } from './geometry/pnp.ts';
export type { PnPOptions, PnPResult } from './geometry/pnp.ts';
export { buildTracks } from './sfm/tracks/tracks.ts';
export { InverseDepthTracker } from './sfm/depth/inverse-depth-tracker.ts';
export { inverseDepthRecord } from './sfm/depth/depth-record.ts';
export {
  fitInverseDepth,
  initializeInverseDepth,
  inverseDepthJacobian,
  inverseDepthPoint,
  reanchorInverseDepth,
  rescaleInverseDepth,
  inverseRangeUncertainty,
} from './geometry/inverse-depth.ts';
export type { AnchoredInverseDepth, InverseDepthLinearization } from './geometry/inverse-depth.ts';
export type {
  InverseDepthTracking,
  InverseDepthTrackingOptions,
  InverseDepthSnapshot,
  DepthTrackState,
  DepthTrackingInput,
  DepthTrackingResult,
} from './sfm/depth/inverse-depth-tracker.ts';
export { bundleAdjustInverseDepth } from './sfm/depth/inverse-depth-bundle.ts';
export type {
  InverseDepthConstraint,
  InverseDepthBundleOptions,
  InverseDepthBundleResult,
} from './sfm/depth/inverse-depth-bundle.ts';
export { cameraMaskFromRegions } from './sfm/tracks/observation-selection.ts';
export { cameraMaskFromPixelPaths } from './geometry/observation-paths.ts';
export type { ObservationPathMask } from './geometry/observation-paths.ts';
export type {
  CameraObservationSelection,
  ObservationExclusion,
} from './sfm/tracks/observation-selection.ts';
export type { CameraObservationMask } from './contracts/observation-mask.ts';
export { bundleAdjust } from './sfm/bundle/bundle-adjustment.ts';
export { validateReconstruction } from './geometry/validation.ts';
export type {
  BundleAdjustmentOptions,
  BundleAdjustmentResult,
} from './sfm/bundle/bundle-adjustment.ts';
export type { BundleDirection } from './sfm/bundle/bundle-directions.ts';
export type { BundleAdjustmentPartition } from './sfm/bundle/local-bundle.ts';
export { reconstructIncremental, refineIncremental } from './sfm/mapper/mapper.ts';
export { ReconstructionInitializationError } from './sfm/protocol/errors.ts';
export type {
  IncrementalOptions,
  IncrementalResult,
  PairVerification,
} from './sfm/mapper/mapper.ts';
export { FOCAL_SEARCH_DEFAULTS } from './sfm/focal/focal-search.ts';
export type {
  FocalCandidate,
  FocalSearchOptions,
  FocalSearchResult,
  FocalInitializationAttempt,
  FocalSearchFailure,
} from './sfm/focal/focal-search.ts';
export type {
  ImageFeatures,
  PairMatches,
  FeatureObservation,
  FeatureTrack,
  RegisteredView,
  LandmarkObservation,
  Landmark,
  SparseReconstruction,
} from './contracts/scene.ts';
export {
  cleanDenseCloud,
  dbscan,
  keepLargestClusters,
  removeStatisticalOutliers,
  voxelDownsample,
} from './mvs/cloud-clean.ts';
export { meshDenseCloud, MESH_DEFAULTS } from './mvs/dense-mesh.ts';
export type { DenseMesh, DenseMeshResult, MeshOptions, MeshReport } from './mvs/dense-mesh.ts';
export type {
  CleanCloud,
  ClusterResult,
  DbscanOptions,
  DbscanResult,
  DenseCleanOptions,
  DenseCleanReport,
  DenseCleanResult,
  OutlierOptions,
  OutlierResult,
  VoxelOptions,
  VoxelResult,
} from './mvs/cloud-clean.ts';
