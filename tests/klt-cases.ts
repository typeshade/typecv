/**
 * The synthetic scenes of `docs/klt-tracking.md`, section 7.1. A scene is a sum of random
 * Gaussians and a smooth checker of a fixed period, so the structure tensor is large nearly
 * everywhere. The second frame is the first one under a known affine warp A: the Gaussians are
 * drawn at their warped centres with their warped radii, and the checker is evaluated at the
 * pre-image, so the second frame is exact and carries no resampling error.
 */

export interface SceneOptions {
  readonly width: number;
  readonly height: number;
  readonly seed?: number;
  readonly gaussians?: number;
  /** Period of the checker in pixels; 0 for none. */
  readonly checkerPeriod?: number;
  readonly checkerAmplitude?: number;
}

/** The warp A(p) = c + s R(theta) (p - c) + t, in pixels, about the centre c. */
export interface Warp {
  readonly dx?: number;
  readonly dy?: number;
  readonly degrees?: number;
  readonly scale?: number;
  readonly gain?: number;
  readonly offset?: number;
}

export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function affine(width: number, height: number, warp: Warp) {
  const c = [(width - 1) / 2, (height - 1) / 2],
    s = warp.scale ?? 1,
    th = ((warp.degrees ?? 0) * Math.PI) / 180,
    cos = Math.cos(th),
    sin = Math.sin(th),
    t = [warp.dx ?? 0, warp.dy ?? 0];
  return {
    forward: (x: number, y: number): [number, number] => {
      const px = x - c[0]!,
        py = y - c[1]!;
      return [c[0]! + s * (cos * px - sin * py) + t[0]!, c[1]! + s * (sin * px + cos * py) + t[1]!];
    },
    inverse: (x: number, y: number): [number, number] => {
      const px = x - c[0]! - t[0]!,
        py = y - c[1]! - t[1]!;
      return [c[0]! + (cos * px + sin * py) / s, c[1]! + (-sin * px + cos * py) / s];
    },
  };
}

/** A gray frame in [0, 1] of the scene under the warp (the identity for the first frame). */
export function renderScene(options: SceneOptions, warp: Warp = {}): Float32Array {
  const { width, height } = options,
    random = mulberry32(options.seed ?? 1),
    count = options.gaussians ?? 400,
    period = options.checkerPeriod ?? 11,
    amplitude = options.checkerAmplitude ?? 0.06,
    map = affine(width, height, warp),
    scale = warp.scale ?? 1,
    out = new Float64Array(width * height).fill(0.5);
  for (let g = 0; g < count; g++) {
    // The scene extends past the frame so that a shift brings new texture in.
    const cx = random() * (width + 200) - 100,
      cy = random() * (height + 200) - 100,
      sigma = 3 + random() * 6,
      weight = (random() - 0.5) * 0.7,
      [wx, wy] = map.forward(cx, cy),
      ws = sigma * scale,
      r = Math.ceil(3 * ws);
    for (let y = Math.max(0, Math.floor(wy) - r); y <= Math.min(height - 1, Math.ceil(wy) + r); y++)
      for (
        let x = Math.max(0, Math.floor(wx) - r);
        x <= Math.min(width - 1, Math.ceil(wx) + r);
        x++
      )
        out[y * width + x]! += weight * Math.exp(-((x - wx) ** 2 + (y - wy) ** 2) / (2 * ws * ws));
  }
  if (period > 0)
    for (let y = 0; y < height; y++)
      for (let x = 0; x < width; x++) {
        const [u, v] = map.inverse(x, y);
        out[y * width + x]! +=
          amplitude * Math.sin((2 * Math.PI * u) / period) * Math.sin((2 * Math.PI * v) / period);
      }
  const gain = warp.gain ?? 1,
    offset = warp.offset ?? 0,
    frame = new Float32Array(width * height);
  for (let i = 0; i < frame.length; i++)
    frame[i] = Math.min(Math.max(out[i]! * gain + offset, 0), 1);
  return frame;
}

/** A grid of points inside the margins, as x, y pairs. */
export function gridPoints(
  width: number,
  height: number,
  marginX: number,
  marginY: number,
  columns: number,
  rows: number,
  jitter = 0.37,
): Float32Array {
  const out = new Float32Array(2 * columns * rows);
  let k = 0;
  for (let j = 0; j < rows; j++)
    for (let i = 0; i < columns; i++) {
      out[2 * k] =
        marginX + ((width - 2 * marginX) * (i + 0.5)) / columns + jitter * ((i * 7 + j * 3) % 5);
      out[2 * k + 1] =
        marginY + ((height - 2 * marginY) * (j + 0.5)) / rows + jitter * ((i * 5 + j * 11) % 7);
      k++;
    }
  return out;
}

/** The error in pixels of each tracked position against the warp's flow, and the alive flags. */
export function flowErrors(
  records: Float32Array,
  points: ArrayLike<number>,
  warp: Warp,
  width: number,
  height: number,
): { error: Float64Array; alive: Uint8Array } {
  const map = affine(width, height, warp),
    n = points.length >> 1,
    error = new Float64Array(n),
    alive = new Uint8Array(n);
  for (let k = 0; k < n; k++) {
    const [tx, ty] = map.forward(points[2 * k]!, points[2 * k + 1]!);
    error[k] = Math.hypot(records[4 * k]! - tx, records[4 * k + 1]! - ty);
    alive[k] = records[4 * k + 3]! & 1;
  }
  return { error, alive };
}

export function percentile(values: ArrayLike<number>, p: number): number {
  const sorted = Array.from(values).sort((a, b) => a - b);
  return sorted.length ? sorted[Math.min(sorted.length - 1, Math.floor(p * sorted.length))]! : NaN;
}
