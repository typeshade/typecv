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
  GpuDenseResult,
  MvsDenseOptions,
  MvsDepthMap,
  MvsDepthMapsOptions,
  MvsDepthMapsResult,
  MvsDepthSnapshot,
  MvsImage,
  MvsPhotometricOptions,
  MvsPhotometricResult,
} from './mvs.ts';
export { gpuCleanPrimitives } from './cloud-clean.ts';
export { cleanCloudWith, voxelFrame } from '../reconstruction/cloud-clean.ts';
export { meshDenseCloudGpu } from './dense-mesh.ts';
export { trackKltGpu } from './klt-device.ts';
export type { KltGpuOptions, KltGpuResult } from './klt-device.ts';
export {
  KLT_DEFAULTS_640,
  KLT_END,
  buildKltPyramid,
  checkKltReference,
  kltOptionsFor,
  trackKlt,
  trackKltReference,
} from './klt.ts';
export type { KltOptions, KltPyramid, KltTrackResult } from './klt.ts';
