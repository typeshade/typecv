import type { Camera } from '../../contracts/camera.ts';
import { validateCamera } from '../camera.ts';

/** @internal Host validation only; numerical camera inversion/projection is in shader code. */
export function validateGpuCamera(camera: Camera): Camera {
  validateCamera(camera);
  if (
    camera.fx < 1e-6 ||
    camera.fy < 1e-6 ||
    camera.fx > 1e6 ||
    camera.fy > 1e6 ||
    Math.max(Math.abs(camera.cx), Math.abs(camera.cy)) > 1e9 ||
    Math.max(Math.abs(camera.k1 ?? 0), Math.abs(camera.k2 ?? 0)) > 1
  )
    throw new RangeError('Camera exceeds the supported GPU geometry bounds');
  return { ...camera, k1: camera.k1 ?? 0, k2: camera.k2 ?? 0 };
}
/** @internal */
export function finiteGpuArray(array: Float64Array): boolean {
  return (
    array instanceof Float64Array &&
    array.every(
      (x) => Number.isFinite(x) && Math.abs(x) <= 1e9 && (x === 0 || Math.abs(x) >= 1e-30),
    )
  );
}
