/**
 * @internal The test entry of the GPU modules that `index.ts` does not export yet. It builds to
 * `dist/internal/gpu.js` (`scripts/build-gpu.mjs`), a bundle of its own with its own copy of the
 * runtime, so a test opens its `ShadeDevice` from this entry. It is no part of the package's
 * `exports`. Step 5 of `docs/dense-mvs.md` moves the dense MVS to `index.ts`.
 */
export { ShadeDevice } from './device.ts';
export { planMvsViews } from './mvs-views.ts';
export type { MvsScene, MvsViewPlan, MvsViewOptions } from './mvs-views.ts';
export { decodeMvsNormal, mvsDepthMapsGpu, mvsPhotometricGpu } from './mvs.ts';
export type {
  MvsDepthMap,
  MvsDepthMapsOptions,
  MvsDepthMapsResult,
  MvsDepthSnapshot,
  MvsImage,
  MvsPhotometricOptions,
  MvsPhotometricResult,
} from './mvs.ts';
