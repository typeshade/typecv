import { unprojectPixel, validateCamera } from '../geometry/camera.ts';
import type { ImageFeatures } from './types.ts';
import type { CameraObservationMask } from '../sfm/tracks/observation-selection.ts';

/** Optical path is independent of depth observability. These labels never estimate depth. */
export interface ObservationPathMask {
  /** Dimensions of the UNDISTORTED analysis-pixel lattice, also used by Dense depth maps. */
  readonly width: number;
  readonly height: number;
  /** 0 direct/static; 1 reflection; 2 refraction; 3 excluded/unknown. No automatic classification. */
  readonly values: Uint8Array;
  readonly source: CameraObservationMask['source'];
}

/** @internal Snapshot before queued work; RGB, labels and caller-owned arrays stay unchanged. */
export function snapshotObservationPaths(mask: ObservationPathMask): ObservationPathMask {
  if (
    !mask ||
    ![mask.width, mask.height].every((x) => Number.isSafeInteger(x) && x > 0) ||
    !(mask.values instanceof Uint8Array) ||
    mask.values.length !== mask.width * mask.height ||
    mask.values.some((v) => v > 3) ||
    !['manual', 'geometry', 'learned', 'ground-truth'].includes(mask.source)
  )
    throw new RangeError('Invalid undistorted observation-path mask');
  return { ...mask, values: mask.values.slice() };
}

/**
 * Select direct/static features using the SAME optical-path mask accepted by MvsImage.
 * Keypoints remain in original distorted pixel coordinates. Padding is an explicit square
 * descriptor-footprint assumption in undistorted pixels; out-of-image support is excluded.
 */
export function cameraMaskFromPixelPaths(
  image: ImageFeatures,
  paths: ObservationPathMask,
  paddingPx = 0,
): CameraObservationMask {
  const camera = image.camera;
  validateCamera(camera);
  const mask = snapshotObservationPaths(paths);
  if (
    !Number.isFinite(paddingPx) ||
    paddingPx < 0 ||
    image.keypoints.length % 2 ||
    !image.keypoints.every(Number.isFinite)
  )
    throw new RangeError('Invalid optical-path keypoints/padding');
  const values = Uint8Array.from({ length: image.keypoints.length / 2 }, (_, i) => {
    const ray = unprojectPixel(camera, [image.keypoints[2 * i]!, image.keypoints[2 * i + 1]!]);
    if (!ray) return 0;
    const u = camera.fx * ray[0] + camera.cx,
      v = camera.fy * ray[1] + camera.cy,
      x0 = Math.floor(u - paddingPx + 0.5),
      x1 = Math.floor(u + paddingPx + 0.5),
      y0 = Math.floor(v - paddingPx + 0.5),
      y1 = Math.floor(v + paddingPx + 0.5);
    if (x0 < 0 || y0 < 0 || x1 >= mask.width || y1 >= mask.height) return 0;
    for (let y = y0; y <= y1; y++)
      for (let x = x0; x <= x1; x++) if (mask.values[y * mask.width + x] !== 0) return 0;
    return 1;
  });
  return { viewId: image.id, values, source: mask.source };
}
