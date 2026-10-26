import { cameraProjection, unprojectPixel } from '../../../geometry/camera.ts';
import type { Pixel } from '../../../contracts/camera.ts';
import { solveSystem } from '../../../geometry/numerics.ts';
import { projectPoint, transformPoint } from '../../../geometry/pose.ts';
import { multiply, transpose } from '../../../geometry/linear.ts';
import type { Landmark, RegisteredView } from '../../../contracts/scene.ts';

/** A local surface is a search proposal. Neither its projected centre nor its depth is evidence. */
export interface ObservationSurfacePrediction {
  readonly pixel: Pixel;
  readonly affine: Float64Array;
  readonly support: number;
  readonly maxRelativeDepthResidual: number;
}

interface SurfaceSample {
  readonly pixel: Pixel;
  readonly ray: Pixel;
  readonly inverseDepth: number;
}

const RADIUS = 48;
const MAX_SAMPLES = 12;
const MIN_SAMPLES = 6;
const MAX_RESIDUAL = 0.02;

/**
 * Cache source-observed finite points in a pixel grid, then fit a small inverse-depth plane
 * around each query. A rejected fit leaves rotation/identity fallback to the caller. The fit
 * excludes a sample at the query itself and must predict an existing seed's depth when supplied.
 * Original RGB patches still have to measure and validate every proposed correspondence.
 */
export function createObservationSurfacePredictor(
  from: RegisteredView,
  to: RegisteredView,
  landmarks: readonly Landmark[],
  halfWindowPx = 5,
): (pixel: Pixel, finitePoint?: ArrayLike<number>) => ObservationSurfacePrediction | undefined {
  if (!(Number.isFinite(halfWindowPx) && halfWindowPx >= 1 && halfWindowPx <= 31))
    throw new RangeError('Invalid surface patch window');
  const cells = new Map<string, SurfaceSample[]>(),
    R = multiply(to.pose.R, transpose(from.pose.R, 3), 3),
    t = Float64Array.from(
      { length: 3 },
      (_, j) =>
        to.pose.t[j]! -
        (R[3 * j]! * from.pose.t[0]! +
          R[3 * j + 1]! * from.pose.t[1]! +
          R[3 * j + 2]! * from.pose.t[2]!),
    );
  for (const landmark of landmarks) {
    const observation = landmark.observations.find((o) => o.viewId === from.id);
    if (!observation || landmark.xyz.length !== 3 || !landmark.xyz.every(Number.isFinite)) continue;
    const q = transformPoint(from.pose, landmark.xyz),
      projected = cameraProjection(from.camera, q)?.pixel,
      ray = unprojectPixel(from.camera, observation.pixel);
    // A large sparse-map residual must not become a confident local surface proposal.
    if (
      !projected ||
      !ray ||
      Math.hypot(projected[0] - observation.pixel[0], projected[1] - observation.pixel[1]) > 2
    )
      continue;
    const key = `${Math.floor(observation.pixel[0] / RADIUS)}:${Math.floor(observation.pixel[1] / RADIUS)}`;
    if (!cells.has(key)) cells.set(key, []);
    cells.get(key)!.push({ pixel: observation.pixel, ray, inverseDepth: 1 / q[2]! });
  }

  return (pixel, finitePoint) => {
    const centre = unprojectPixel(from.camera, pixel);
    if (!centre) return undefined;
    const candidates: { sample: SurfaceSample; distance: number }[] = [],
      cellX = Math.floor(pixel[0] / RADIUS),
      cellY = Math.floor(pixel[1] / RADIUS);
    for (let cy = cellY - 1; cy <= cellY + 1; cy++)
      for (let cx = cellX - 1; cx <= cellX + 1; cx++)
        for (const sample of cells.get(`${cx}:${cy}`) ?? []) {
          const distance = (sample.pixel[0] - pixel[0]) ** 2 + (sample.pixel[1] - pixel[1]) ** 2;
          if (distance < 1 || distance > RADIUS * RADIUS) continue;
          // Retain only a bounded small system even in a densely observed grid cell.
          let index = candidates.findIndex((c) => c.distance > distance);
          if (index < 0) index = candidates.length;
          if (index < MAX_SAMPLES) candidates.splice(index, 0, { sample, distance });
          if (candidates.length > MAX_SAMPLES) candidates.pop();
        }
    if (candidates.length < MIN_SAMPLES) return undefined;
    const samples = candidates.map((c) => c.sample),
      dx = samples.map((s) => s.pixel[0] - pixel[0]),
      dy = samples.map((s) => s.pixel[1] - pixel[1]);
    // Do not extrapolate a plane from observations on one side of the queried patch.
    if (Math.min(...dx) > -8 || Math.max(...dx) < 8 || Math.min(...dy) > -8 || Math.max(...dy) < 8)
      return undefined;
    // A point lies strictly inside the support hull iff the samples surround its full circle.
    // Check all patch corners with a bilinear sample margin, not just the centre.
    const margin = halfWindowPx + 1;
    for (const ox of [-margin, margin])
      for (const oy of [-margin, margin]) {
        const angles = samples
          .map((s) => Math.atan2(s.pixel[1] - pixel[1] - oy, s.pixel[0] - pixel[0] - ox))
          .sort((a, b) => a - b);
        if (
          angles.some(
            (a, i) =>
              angles[(i + 1) % angles.length]! + (i + 1 === angles.length ? 2 * Math.PI : 0) - a >=
              Math.PI - 1e-6,
          )
        )
          return undefined;
      }

    const scale = Math.max(
        ...samples.map((s) => Math.hypot(s.ray[0] - centre[0], s.ray[1] - centre[1])),
      ),
      depths = samples.map((s) => s.inverseDepth).sort((a, b) => a - b),
      depthScale = depths[Math.floor(depths.length / 2)]!,
      rows = samples.map((s) => [
        (s.ray[0] - centre[0]) / scale,
        (s.ray[1] - centre[1]) / scale,
        1,
      ]),
      matrix = new Float64Array(9),
      rhs = new Float64Array(3);
    if (!(scale > 1e-12 && depthScale > 0)) return undefined;
    for (let i = 0; i < samples.length; i++)
      for (let r = 0; r < 3; r++) {
        rhs[r] += (rows[i]![r]! * samples[i]!.inverseDepth) / depthScale;
        for (let c = 0; c < 3; c++) matrix[3 * r + c] += rows[i]![r]! * rows[i]![c]!;
      }
    const n = samples.length,
      xx = matrix[0]! - matrix[2]! ** 2 / n,
      xy = matrix[1]! - (matrix[2]! * matrix[5]!) / n,
      yy = matrix[4]! - matrix[5]! ** 2 / n;
    if (!(xx * yy - xy * xy > 0.02 * (xx + yy) ** 2)) return undefined;
    const coefficients = solveSystem(matrix, rhs);
    if (!coefficients) return undefined;
    const inverseDepth = (ray: Pixel) =>
      depthScale *
      ((coefficients[0]! * (ray[0] - centre[0])) / scale +
        (coefficients[1]! * (ray[1] - centre[1])) / scale +
        coefficients[2]!);
    let residual = 0;
    for (const sample of samples)
      residual = Math.max(residual, Math.abs(inverseDepth(sample.ray) / sample.inverseDepth - 1));
    if (!(residual <= MAX_RESIDUAL && inverseDepth(centre) > 0)) return undefined;
    if (finitePoint) {
      const q = transformPoint(from.pose, finitePoint),
        p = cameraProjection(from.camera, q)?.pixel;
      if (
        !p ||
        Math.hypot(p[0] - pixel[0], p[1] - pixel[1]) > 2 ||
        Math.abs(inverseDepth(centre) * q[2]! - 1) > MAX_RESIDUAL
      )
        return undefined;
    }
    const project = (x: number, y: number): Pixel | undefined => {
      const ray = unprojectPixel(from.camera, [x, y]);
      if (!ray) return undefined;
      const rho = inverseDepth(ray);
      if (!(rho > 0)) return undefined;
      return cameraProjection(
        to.camera,
        [0, 1, 2].map(
          (j) => (R[3 * j]! * ray[0] + R[3 * j + 1]! * ray[1] + R[3 * j + 2]!) / rho + t[j]!,
        ),
      )?.pixel;
    };
    for (const ox of [-margin, margin])
      for (const oy of [-margin, margin])
        if (!project(pixel[0] + ox, pixel[1] + oy)) return undefined;
    const left = project(pixel[0] - 0.5, pixel[1]),
      right = project(pixel[0] + 0.5, pixel[1]),
      top = project(pixel[0], pixel[1] - 0.5),
      bottom = project(pixel[0], pixel[1] + 0.5),
      predicted = finitePoint
        ? projectPoint(to.camera, to.pose, finitePoint)
        : project(pixel[0], pixel[1]);
    if (!left || !right || !top || !bottom || !predicted) return undefined;
    const affine = Float64Array.of(
      right[0] - left[0],
      bottom[0] - top[0],
      right[1] - left[1],
      bottom[1] - top[1],
    );
    if (!affine.every(Number.isFinite)) return undefined;
    return { pixel: predicted, affine, support: n, maxRelativeDepthResidual: residual };
  };
}
