// The CPU reference of the pyramidal Lucas-Kanade tracker of `docs/klt-tracking.md`: the pyramid,
// the Scharr gradient, the inverse compositional solver, the Shi-Tomasi measure and the
// forward-backward check. It runs the same arithmetic as the kernels of `klt.shade.ts` in the same
// order, with `Math.fround` after each operation, so it is the oracle of the kernels on the CPU
// tier. It imports nothing of the device, so the unit tests and the host read it as plain
// TypeScript.

import { kltLayout, type KltLayout } from '../../image/gpu/pyramid-layout.ts';

export { kltLayout } from '../../image/gpu/pyramid-layout.ts';
export type { KltLayout } from '../../image/gpu/pyramid-layout.ts';

const f = Math.fround;

/** The end reasons of a track (section 3.5), stored in bits 1 to 4 of the status word. */
export const KLT_END = {
  alive: 0,
  outOfFrame: 1,
  lowStructure: 2,
  noConvergence: 3,
  forwardBackward: 4,
  residual: 5,
  epipolar: 6,
  age: 7,
} as const;

export interface KltOptions {
  /** Pyramid levels, level 0 included. */
  readonly levels: number;
  /** The window is (2 halfWindow + 1) pixels square. */
  readonly halfWindow: number;
  /** Maximum inverse-compositional updates at each level. */
  readonly maxIterations: number;
  /** A level has converged when the step is shorter than this, in pixels of that level. */
  readonly epsilonPx: number;
  /**
   * The least lambda-min of the template window per window pixel. The gray image is in [0, 1]
   * and the gradient is per pixel, so the unit is (gray per pixel) squared. OpenCV's
   * `minEigThreshold` works on 8-bit gray and unscaled Scharr, a factor of (255 x 32)^2 larger.
   */
  readonly minEigThreshold: number;
  /** A track whose forward-backward error is above this ends (section 2.3). */
  readonly fbThresholdPx: number;
  /** A track whose zero-mean NCC of the final window is below this ends (section 3.5). */
  readonly nccThreshold: number;
  /**
   * The residual of the solver. 0: plain SSD, I - J. 1: zero mean, the window means removed.
   * 2: zero mean and unit variance, J scaled to the template's variance (a per-window gain and
   * bias). Section 9 and the test of 7.1, item 3 decide the default.
   */
  readonly photometric: 0 | 1 | 2;
  readonly qualityLevel: number;
  readonly minDistancePx: number;
  readonly maxAgeFrames: number;
  /**
   * The join radius of section 3.3, in pixels of the frame: a track joins the nearest SIFT keypoint
   * within it. 2 at 640 px, scaled by the frame width.
   */
  readonly joinRadiusPx: number;
  /**
   * The most tracks alive at once, 1 to `KLT_MAX_CAPACITY`. When selection finds more points
   * than room, it keeps the ones with the greatest lambda-min.
   */
  readonly capacity: number;
}

/**
 * The tracks a tracker holds by default, for every frame width. On SwiftShader a push of 1024
 * tracks takes 4.3 s at 640 px with the minimum chunk of 64 tracks (Step 4 record). The lab asks
 * for more with `capacity`, up to `KLT_MAX_CAPACITY`.
 */
export const KLT_DEFAULT_CAPACITY = 1024;
/** The most tracks of a tracker (section 4.1). */
export const KLT_MAX_CAPACITY = 8192;

/** The defaults of section 2.4 for a frame 640 px wide. */
export const KLT_DEFAULTS_640: KltOptions = {
  levels: 5,
  halfWindow: 7,
  maxIterations: 20,
  epsilonPx: 0.03,
  minEigThreshold: 1e-6,
  fbThresholdPx: 0.5,
  nccThreshold: 0.8,
  photometric: 2,
  qualityLevel: 0.01,
  minDistancePx: 8,
  maxAgeFrames: 16,
  joinRadiusPx: 2,
  capacity: KLT_DEFAULT_CAPACITY,
};

/** The defaults for a frame of the given width: the 1280 px values for 1000 px and more. */
export function kltOptionsFor(width: number, overrides: Partial<KltOptions> = {}): KltOptions {
  const wide = width >= 1000;
  return {
    ...KLT_DEFAULTS_640,
    ...(wide ? { levels: 6, halfWindow: 10, joinRadiusPx: 4 } : {}),
    ...overrides,
  };
}

export interface KltPyramid extends KltLayout {
  readonly width: number;
  readonly height: number;
  /** All levels, level 0 first. */
  readonly gray: Float32Array;
  /** (Ix, Iy) for each pixel of all levels. */
  readonly grad: Float32Array;
}

/** The levels of the pyramid, 5-tap binomial blur and decimation, from the level 0 pixels. */
export function buildKltGray(
  gray: ArrayLike<number>,
  width: number,
  height: number,
  levels: number,
): { layout: KltLayout; gray: Float32Array } {
  const layout = kltLayout(width, height, levels);
  if (gray.length !== width * height) throw new RangeError('gray must hold width x height values');
  const out = new Float32Array(layout.total);
  for (let i = 0; i < width * height; i++) out[i] = gray[i]!;
  for (let l = 0; l + 1 < levels; l++) {
    const w = layout.widths[l]!,
      h = layout.heights[l]!,
      o = layout.offsets[l]!,
      ow = layout.widths[l + 1]!,
      oh = layout.heights[l + 1]!,
      oo = layout.offsets[l + 1]!,
      at = (x: number, y: number) =>
        out[o + Math.min(Math.max(y, 0), h - 1) * w + Math.min(Math.max(x, 0), w - 1)]!;
    for (let oy = 0; oy < oh; oy++)
      for (let ox = 0; ox < ow; ox++) {
        const cx = 2 * ox,
          cy = 2 * oy;
        let sum = 0;
        for (let r = -2; r <= 2; r++) {
          const wy = r === 0 ? 6 : r === -1 || r === 1 ? 4 : 1,
            row = cy + r;
          let line = at(cx - 2, row);
          line = f(line + f(4 * at(cx - 1, row)));
          line = f(line + f(6 * at(cx, row)));
          line = f(line + f(4 * at(cx + 1, row)));
          line = f(line + at(cx + 2, row));
          sum = f(sum + f(wy * line));
        }
        out[oo + oy * ow + ox] = f(sum / 256);
      }
  }
  return { layout, gray: out };
}

/** The Scharr gradient (3, 10, 3) / 32 of every level, the border clamped. */
export function kltScharrGradient(layout: KltLayout, gray: Float32Array): Float32Array {
  const grad = new Float32Array(2 * layout.total);
  for (let l = 0; l < layout.levels; l++) {
    const w = layout.widths[l]!,
      h = layout.heights[l]!,
      o = layout.offsets[l]!,
      at = (x: number, y: number) =>
        gray[o + Math.min(Math.max(y, 0), h - 1) * w + Math.min(Math.max(x, 0), w - 1)]!;
    for (let y = 0; y < h; y++)
      for (let x = 0; x < w; x++) {
        const tl = at(x - 1, y - 1),
          tc = at(x, y - 1),
          tr = at(x + 1, y - 1),
          ml = at(x - 1, y),
          mr = at(x + 1, y),
          bl = at(x - 1, y + 1),
          bc = at(x, y + 1),
          br = at(x + 1, y + 1),
          ix = f(f(f(f(3 * f(tr - tl)) + f(10 * f(mr - ml))) + f(3 * f(br - bl))) / 32),
          iy = f(f(f(f(3 * f(bl - tl)) + f(10 * f(bc - tc))) + f(3 * f(br - tr))) / 32);
        grad[2 * (o + y * w + x)] = ix;
        grad[2 * (o + y * w + x) + 1] = iy;
      }
  }
  return grad;
}

/** The pyramid and the Scharr gradient of a gray frame in [0, 1] (`kltPyrDown`, `kltScharr`). */
export function buildKltPyramid(
  gray: ArrayLike<number>,
  width: number,
  height: number,
  levels: number,
): KltPyramid {
  const built = buildKltGray(gray, width, height, levels);
  return {
    ...built.layout,
    width,
    height,
    gray: built.gray,
    grad: kltScharrGradient(built.layout, built.gray),
  };
}

/** A start record: x, y, 0, status 1 (alive). Length 4 for each point. */
export function kltSeedRecords(points: ArrayLike<number>): Float32Array {
  const n = points.length >> 1,
    out = new Float32Array(4 * n);
  for (let k = 0; k < n; k++) {
    out[4 * k] = points[2 * k]!;
    out[4 * k + 1] = points[2 * k + 1]!;
    out[4 * k + 3] = 1;
  }
  return out;
}

/** Validate and snapshot absolute target pixels. They initialize LK; they are not observations. */
export function kltInitialRecords(targets: ArrayLike<number>, count: number): Float32Array {
  if (targets.length !== 2 * count)
    throw new RangeError('initialTargets must hold one x, y pair per seed');
  const records = kltSeedRecords(targets);
  for (let k = 0; k < count; k++)
    if (!Number.isFinite(records[4 * k]) || !Number.isFinite(records[4 * k + 1]))
      throw new RangeError('initialTargets must be finite f32 pixels');
  return records;
}

/** Row-major local affine patches; inverse uses the same f32 operations as the shader. */
export function kltAffineRecords(
  warps: ArrayLike<number>,
  count: number,
): {
  forward: Float32Array;
  backward: Float32Array;
} {
  if (warps.length !== 4 * count)
    throw new RangeError('affineWarps must hold four values per seed');
  const forward = Float32Array.from(warps),
    backward = new Float32Array(4 * count);
  for (let k = 0; k < count; k++) {
    const a = forward[4 * k]!,
      b = forward[4 * k + 1]!,
      c = forward[4 * k + 2]!,
      d = forward[4 * k + 3]!,
      det = f(f(a * d) - f(b * c)),
      inv = f(1 / det);
    const values = [f(d * inv), f(-b * inv), f(-c * inv), f(a * inv)];
    if (
      ![a, b, c, d, ...values].every(Number.isFinite) ||
      !(det > 0) ||
      a * a + b * b + c * c + d * d > 16 ||
      values.reduce((n, x) => n + x * x, 0) > 16
    )
      throw new RangeError(
        'affineWarps require bounded, orientation-preserving invertible patches',
      );
    backward.set(values, 4 * k);
  }
  return { forward, backward };
}

/** The status word fields. */
export const kltAlive = (status: number) => (status & 1) === 1;
export const kltReason = (status: number) => (status >> 1) & 15;
export const kltIterations = (status: number) => (status >> 16) & 255;

const TINY_DET = f(1e-12);
const TINY_PRODUCT = f(1e-30);
const TINY_VARIANCE = f(1e-10);

function bilinear(
  data: Float32Array,
  stride: number,
  channel: number,
  offset: number,
  w: number,
  h: number,
  x: number,
  y: number,
): number {
  const cx = Math.min(Math.max(x, 0), f(w - 1)),
    cy = Math.min(Math.max(y, 0), f(h - 1)),
    x0 = Math.min(Math.floor(cx), w - 1),
    y0 = Math.min(Math.floor(cy), h - 1),
    x1 = Math.min(x0 + 1, w - 1),
    y1 = Math.min(y0 + 1, h - 1),
    ax = f(cx - x0),
    ay = f(cy - y0),
    at = (px: number, py: number) => data[stride * (offset + py * w + px) + channel]!,
    top = f(f(at(x0, y0) * f(1 - ax)) + f(at(x1, y0) * ax)),
    bottom = f(f(at(x0, y1) * f(1 - ax)) + f(at(x1, y1) * ax));
  return f(f(top * f(1 - ay)) + f(bottom * ay));
}

export interface KltRange {
  readonly first: number;
  readonly end: number;
}

/**
 * Track the points of `start` (records of four floats: x, y, unused, status) from the template
 * pyramid `from` to the target pyramid `to`. Returns records (x, y, NCC, status). A record that
 * is not alive is copied through. `range` selects tracks `first..end`; the others stay zero.
 */
export function trackKltReference(
  from: KltPyramid,
  to: KltPyramid,
  start: Float32Array,
  options: KltOptions,
  range: KltRange = { first: 0, end: start.length >> 2 },
  initialTargets?: ArrayLike<number>,
  affineWarps?: ArrayLike<number>,
): Float32Array {
  const initial = initialTargets && kltInitialRecords(initialTargets, start.length >> 2),
    affine = affineWarps && kltAffineRecords(affineWarps, start.length >> 2);
  if (from.width !== to.width || from.height !== to.height || from.levels !== to.levels)
    throw new RangeError('The two pyramids must have the same layout');
  if (from.levels !== options.levels) throw new RangeError('options.levels must match the pyramid');
  const out = new Float32Array(start.length),
    hw = options.halfWindow,
    levels = options.levels,
    windowCount = f((2 * hw + 1) * (2 * hw + 1)),
    epsilon = f(options.epsilonPx),
    epsilonSquared = f(epsilon * epsilon),
    minEig = f(options.minEigThreshold),
    w0 = from.width,
    h0 = from.height;
  for (let k = range.first; k < range.end; k++) {
    const sx = start[4 * k]!,
      sy = start[4 * k + 1]!,
      startStatus = start[4 * k + 3]!,
      a00 = affine?.forward[4 * k] ?? 1,
      a01 = affine?.forward[4 * k + 1] ?? 0,
      a10 = affine?.forward[4 * k + 2] ?? 0,
      a11 = affine?.forward[4 * k + 3] ?? 1;
    let dx = 0,
      dy = 0,
      gxx = 0,
      gxy = 0,
      gyy = 0,
      converged = false,
      used = 0;
    if (initial) {
      const scale = f(2 ** (levels - 1));
      dx = f(f(initial[4 * k]! - sx) / scale);
      dy = f(f(initial[4 * k + 1]! - sy) / scale);
      if (affine) {
        const ux = dx,
          uy = dy,
          inv = affine.backward;
        dx = f(f(inv[4 * k]! * ux) + f(inv[4 * k + 1]! * uy));
        dy = f(f(inv[4 * k + 2]! * ux) + f(inv[4 * k + 3]! * uy));
      }
    }
    for (let step = 0; step < levels; step++) {
      const level = levels - 1 - step,
        scale = f(2 ** level),
        off = from.offsets[level]!,
        w = from.widths[level]!,
        h = from.heights[level]!,
        px = f(sx / scale),
        py = f(sy / scale),
        grad = (x: number, y: number, c: number) => bilinear(from.grad, 2, c, off, w, h, x, y),
        template = (x: number, y: number) => bilinear(from.gray, 1, 0, off, w, h, x, y),
        target = (x: number, y: number) => bilinear(to.gray, 1, 0, off, w, h, x, y),
        warpedTarget = (i: number, j: number) =>
          affine
            ? target(
                f(px + f(f(a00 * f(i + dx)) + f(a01 * f(j + dy)))),
                f(py + f(f(a10 * f(i + dx)) + f(a11 * f(j + dy)))),
              )
            : target(f(f(px + i) + dx), f(f(py + j) + dy));
      let a = 0,
        b = 0,
        c = 0;
      for (let j = -hw; j <= hw; j++)
        for (let i = -hw; i <= hw; i++) {
          const tx = f(px + i),
            ty = f(py + j),
            gx = grad(tx, ty, 0),
            gy = grad(tx, ty, 1);
          a = f(a + f(gx * gx));
          b = f(b + f(gx * gy));
          c = f(c + f(gy * gy));
        }
      gxx = a;
      gxy = b;
      gyy = c;
      const det = f(f(a * c) - f(b * b)),
        inverse = det > TINY_DET ? f(1 / Math.max(det, TINY_DET)) : 0;
      // The template window statistics for the zero-mean forms.
      let sI = 0,
        sII = 0;
      for (let j = -hw; j <= hw; j++)
        for (let i = -hw; i <= hw; i++) {
          const v = template(f(px + i), f(py + j));
          sI = f(sI + v);
          sII = f(sII + f(v * v));
        }
      const meanT = f(sI / windowCount),
        varT = f(f(sII / windowCount) - f(meanT * meanT)),
        useMean = options.photometric > 0;
      let done = false;
      for (let it = 0; it < options.maxIterations; it++) {
        let sJ = 0,
          sJJ = 0;
        for (let j = -hw; j <= hw; j++)
          for (let i = -hw; i <= hw; i++) {
            const v = warpedTarget(i, j);
            sJ = f(sJ + v);
            sJJ = f(sJJ + f(v * v));
          }
        const meanJ = f(sJ / windowCount),
          varJ = f(f(sJJ / windowCount) - f(meanJ * meanJ)),
          gamma =
            options.photometric === 2 && varJ > TINY_VARIANCE && varT > TINY_VARIANCE
              ? f(Math.sqrt(f(varT / varJ)))
              : 1,
          subtractT = useMean ? meanT : 0,
          subtractJ = useMean ? meanJ : 0;
        let bx = 0,
          by = 0;
        for (let j = -hw; j <= hw; j++)
          for (let i = -hw; i <= hw; i++) {
            const tx = f(px + i),
              ty = f(py + j),
              r = f(f(template(tx, ty) - subtractT) - f(gamma * f(warpedTarget(i, j) - subtractJ)));
            bx = f(bx + f(r * grad(tx, ty, 0)));
            by = f(by + f(r * grad(tx, ty, 1)));
          }
        const stepX = f(f(f(c * bx) - f(b * by)) * inverse),
          stepY = f(f(f(a * by) - f(b * bx)) * inverse);
        if (!done) {
          dx = f(dx + stepX);
          dy = f(dy + stepY);
          if (level === 0) used++;
          const ex = affine ? f(f(a00 * stepX) + f(a01 * stepY)) : stepX,
            ey = affine ? f(f(a10 * stepX) + f(a11 * stepY)) : stepY;
          if (f(f(ex * ex) + f(ey * ey)) < epsilonSquared) done = true;
        }
      }
      if (level === 0) converged = done;
      else {
        dx = f(dx * 2);
        dy = f(dy * 2);
      }
    }
    const rx = affine ? f(sx + f(f(a00 * dx) + f(a01 * dy))) : f(sx + dx),
      ry = affine ? f(sy + f(f(a10 * dx) + f(a11 * dy))) : f(sy + dy),
      tpl = (x: number, y: number) => bilinear(from.gray, 1, 0, 0, w0, h0, x, y),
      tgt = (x: number, y: number) => bilinear(to.gray, 1, 0, 0, w0, h0, x, y),
      finalTarget = (i: number, j: number) =>
        affine
          ? tgt(f(rx + f(f(a00 * i) + f(a01 * j))), f(ry + f(f(a10 * i) + f(a11 * j))))
          : tgt(f(rx + i), f(ry + j));
    let meanI = 0,
      meanJ = 0;
    for (let j = -hw; j <= hw; j++)
      for (let i = -hw; i <= hw; i++) {
        meanI = f(meanI + tpl(f(sx + i), f(sy + j)));
        meanJ = f(meanJ + finalTarget(i, j));
      }
    meanI = f(meanI / windowCount);
    meanJ = f(meanJ / windowCount);
    let sii = 0,
      sjj = 0,
      sij = 0;
    for (let j = -hw; j <= hw; j++)
      for (let i = -hw; i <= hw; i++) {
        const ti = f(tpl(f(sx + i), f(sy + j)) - meanI),
          tj = f(finalTarget(i, j) - meanJ);
        sii = f(sii + f(ti * ti));
        sjj = f(sjj + f(tj * tj));
        sij = f(sij + f(ti * tj));
      }
    const product = f(sii * sjj),
      ncc = product > TINY_PRODUCT ? f(sij / f(Math.sqrt(Math.max(product, TINY_PRODUCT)))) : 0,
      lo = f(hw),
      hiX = f(f(f(w0) - 1) - lo),
      hiY = f(f(f(h0) - 1) - lo),
      extentX = f(f(hw) * f(Math.abs(a00) + Math.abs(a01))),
      extentY = f(f(hw) * f(Math.abs(a10) + Math.abs(a11))),
      outside =
        sx < lo ||
        sx > hiX ||
        sy < lo ||
        sy > hiY ||
        rx < (affine ? extentX : lo) ||
        rx > (affine ? f(f(f(w0) - 1) - extentX) : hiX) ||
        ry < (affine ? extentY : lo) ||
        ry > (affine ? f(f(f(h0) - 1) - extentY) : hiY),
      trace = f(gxx + gyy),
      root = f(Math.sqrt(f(f(f(gxx - gyy) * f(gxx - gyy)) + f(f(f(4 * gxy) * gxy))))),
      lowStructure = f(f(f(trace - root) / 2) / windowCount) < minEig;
    let reason: number = KLT_END.alive;
    if (outside) reason = KLT_END.outOfFrame;
    else if (lowStructure) reason = KLT_END.lowStructure;
    else if (!converged) reason = KLT_END.noConvergence;
    const alive = (startStatus & 1) === 1,
      status = alive ? (reason === 0 ? 1 | (used << 16) : reason << 1) : startStatus;
    out[4 * k] = alive ? rx : sx;
    out[4 * k + 1] = alive ? ry : sy;
    out[4 * k + 2] = alive ? ncc : 0;
    out[4 * k + 3] = status;
  }
  return out;
}

/**
 * The forward-backward check (section 2.3) and the NCC check (section 3.5). `start` holds the
 * seeds, `forward` the result of `trackKltReference(prev, cur, start)` and `back` the result of
 * `trackKltReference(cur, prev, forward)`. Returns (forward x, y, error, status).
 */
export function checkKltReference(
  start: Float32Array,
  forward: Float32Array,
  back: Float32Array,
  options: KltOptions,
  range: KltRange = { first: 0, end: start.length >> 2 },
): Float32Array {
  const out = new Float32Array(start.length),
    fb = f(options.fbThresholdPx),
    nccMin = f(options.nccThreshold);
  for (let k = range.first; k < range.end; k++) {
    const ncc = forward[4 * k + 2]!,
      forwardStatus = forward[4 * k + 3]!,
      backStatus = back[4 * k + 3]!,
      ex = f(back[4 * k]! - start[4 * k]!),
      ey = f(back[4 * k + 1]! - start[4 * k + 1]!),
      error = f(Math.sqrt(f(f(ex * ex) + f(ey * ey)))),
      alive = (forwardStatus & 1) === 1,
      backAlive = (backStatus & 1) === 1;
    let status = forwardStatus;
    if (alive) {
      if (!backAlive || !(error <= fb)) status = KLT_END.forwardBackward << 1;
      else if (!(ncc >= nccMin)) status = KLT_END.residual << 1;
    }
    out[4 * k] = forward[4 * k]!;
    out[4 * k + 1] = forward[4 * k + 1]!;
    out[4 * k + 2] = alive && backAlive ? error : 0;
    out[4 * k + 3] = status;
  }
  return out;
}

export interface KltTrackResult {
  /** (x, y, forward-backward error, status) for each point. */
  readonly records: Float32Array;
  /** The forward result (x, y, NCC, status) before the check. */
  readonly forward: Float32Array;
  readonly back: Float32Array;
  readonly alive: Uint8Array;
  readonly reason: Uint8Array;
}

/**
 * Track `points` (x, y pairs of the template frame) from `prev` to `cur`, then back, and check
 * both (sections 2.1 to 2.3, 3.5).
 */
export function trackKlt(
  prev: KltPyramid,
  cur: KltPyramid,
  points: ArrayLike<number>,
  options: KltOptions,
  initialTargets?: ArrayLike<number>,
  affineWarps?: ArrayLike<number>,
): KltTrackResult {
  const start = kltSeedRecords(points),
    inverse = affineWarps && kltAffineRecords(affineWarps, points.length >> 1).backward,
    forward = trackKltReference(prev, cur, start, options, undefined, initialTargets, affineWarps),
    back = trackKltReference(
      cur,
      prev,
      forward,
      options,
      undefined,
      initialTargets && points,
      inverse,
    ),
    records = checkKltReference(start, forward, back, options),
    n = points.length >> 1,
    alive = new Uint8Array(n),
    reason = new Uint8Array(n);
  for (let k = 0; k < n; k++) {
    alive[k] = records[4 * k + 3]! & 1;
    reason[k] = kltReason(records[4 * k + 3]!);
  }
  return { records, forward, back, alive, reason };
}

// Selection (sections 2.2 and 3.4). `kltEigenMap`, `kltEigenMax`, `kltEigenMaxFinal` and
// `kltSelect` of `klt.shade.ts` run the arithmetic below in the same order.

/** The side of the block of the Shi-Tomasi measure (section 2.2). */
export const KLT_BLOCK = 3;

/** The cells of `cell` pixels that cover a frame: columns, rows. */
export function kltCellGrid(
  width: number,
  height: number,
  cell: number,
): { cellsX: number; cellsY: number } {
  return { cellsX: Math.ceil(width / cell), cellsY: Math.ceil(height / cell) };
}

/**
 * The lambda-min map at level 0: lambda-min of the structure tensor summed over a block of
 * `KLT_BLOCK` pixels square, the border clamped, from the Scharr gradient of `kltScharrGradient`.
 * A pixel closer than `halfWindow` to the border is 0, because a track there ends out of frame.
 */
export function kltEigenMapReference(
  grad: Float32Array,
  width: number,
  height: number,
  halfWindow: number,
): Float32Array {
  const out = new Float32Array(width * height),
    reach = (KLT_BLOCK - 1) >> 1;
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      if (x < halfWindow || x + halfWindow >= width || y < halfWindow || y + halfWindow >= height)
        continue;
      let a = 0,
        b = 0,
        c = 0;
      for (let j = -reach; j <= reach; j++)
        for (let i = -reach; i <= reach; i++) {
          const px = Math.min(Math.max(x + i, 0), width - 1),
            py = Math.min(Math.max(y + j, 0), height - 1),
            gx = grad[2 * (py * width + px)]!,
            gy = grad[2 * (py * width + px) + 1]!;
          a = f(a + f(gx * gx));
          b = f(b + f(gx * gy));
          c = f(c + f(gy * gy));
        }
      const trace = f(a + c),
        root = f(Math.sqrt(f(f(f(a - c) * f(a - c)) + f(f(f(4 * b) * b)))));
      out[y * width + x] = f(f(trace - root) / 2);
    }
  return out;
}

/** The largest value of the map, as `kltEigenMax` and `kltEigenMaxFinal` find it. */
export function kltEigenMaxReference(eig: Float32Array): number {
  let best = 0;
  for (let i = 0; i < eig.length; i++) if (eig[i]! > best) best = eig[i]!;
  return best;
}

export interface KltPick {
  readonly x: number;
  readonly y: number;
  readonly score: number;
}

/**
 * The occupancy that `kltSelectReference` and `kltSelect` read: two floats (x, y) for each cell of
 * `cell` pixels, the position of one alive track in the cell, or -1 for a cell with none.
 */
export function kltOccupancy(
  x: ArrayLike<number>,
  y: ArrayLike<number>,
  count: number,
  width: number,
  height: number,
  cell: number,
): Float32Array {
  const { cellsX, cellsY } = kltCellGrid(width, height, cell),
    out = new Float32Array(2 * cellsX * cellsY).fill(-1);
  for (let k = 0; k < count; k++) {
    const c = kltCellOf(x[k]!, y[k]!, width, height, cell);
    if (out[2 * c]! < 0) {
      out[2 * c] = x[k]!;
      out[2 * c + 1] = y[k]!;
    }
  }
  return out;
}

/**
 * Section 3.4, steps 2 and 3: in each cell of `minDistancePx` pixels with no alive track
 * (`occupied`, `kltOccupancy`), the pixel of the greatest lambda-min, the first in row-major order
 * on a tie, if it is at least `qualityLevel` times the largest value of the frame and at least
 * `minEigThreshold` per pixel of the block. Then the suppression: a pick ends when another pick
 * within `minDistancePx` is stronger (the lower cell on a tie), or when an alive track of a
 * neighbouring cell is within `minDistancePx`. So the picks are at least `minDistancePx` apart from
 * each other and from the alive tracks that the occupancy holds. Returns the picks in cell order.
 */
export function kltSelectReference(
  eig: Float32Array,
  width: number,
  height: number,
  options: KltOptions,
  occupied: ArrayLike<number>,
): { max: number; best: KltPick[]; picks: KltPick[] } {
  const cell = options.minDistancePx,
    { cellsX, cellsY } = kltCellGrid(width, height, cell),
    cells = cellsX * cellsY,
    max = kltEigenMaxReference(eig),
    threshold = Math.max(f(f(options.qualityLevel) * max), f(f(options.minEigThreshold) * 9)),
    reach = f(cell * cell),
    found: (KltPick | undefined)[] = new Array(cells).fill(undefined),
    picks: KltPick[] = [];
  for (let k = 0; k < cells; k++) {
    const cx = k % cellsX,
      cy = Math.floor(k / cellsX);
    let best = 0,
      bx = 0,
      by = 0;
    for (let j = 0; j < cell; j++)
      for (let i = 0; i < cell; i++) {
        const x = cx * cell + i,
          y = cy * cell + j;
        if (x >= width || y >= height) continue;
        const v = eig[y * width + x]!;
        if (v > best) {
          best = v;
          bx = x;
          by = y;
        }
      }
    if (occupied[2 * k]! < 0 && best > 0 && best >= threshold)
      found[k] = { x: bx, y: by, score: best };
  }
  const nearer = (ax: number, ay: number, bx: number, by: number) => {
    const dx = f(bx - ax),
      dy = f(by - ay);
    return f(f(dx * dx) + f(dy * dy)) < reach;
  };
  for (let k = 0; k < cells; k++) {
    const mine = found[k];
    if (!mine) continue;
    const cx = k % cellsX,
      cy = Math.floor(k / cellsX);
    let keep = true;
    for (let j = -1; j <= 1; j++)
      for (let i = -1; i <= 1; i++) {
        const nx = cx + i,
          ny = cy + j;
        if (nx < 0 || nx >= cellsX || ny < 0 || ny >= cellsY) continue;
        const n = ny * cellsX + nx,
          other = found[n];
        if (other && n !== k && nearer(mine.x, mine.y, other.x, other.y))
          if (other.score > mine.score || (other.score === mine.score && n < k)) keep = false;
        const ex = occupied[2 * n]!;
        if (ex >= 0 && nearer(mine.x, mine.y, ex, occupied[2 * n + 1]!)) keep = false;
      }
    if (keep) picks.push(mine);
  }
  return { max, best: found.filter((p): p is KltPick => p !== undefined), picks };
}

/** The index of the cell that holds a point, for the occupancy of `kltSelectReference`. */
export function kltCellOf(
  x: number,
  y: number,
  width: number,
  height: number,
  cell: number,
): number {
  const { cellsX } = kltCellGrid(width, height, cell),
    cx = Math.min(Math.max(Math.floor(x / cell), 0), cellsX - 1),
    cy = Math.min(Math.max(Math.floor(y / cell), 0), Math.ceil(height / cell) - 1);
  return cy * cellsX + cx;
}

/**
 * The assignment of section 3.3, steps 1 and 2: each point of a list gets the nearest keypoint
 * within `radius` that no other point claims. Candidate pairs are taken by increasing distance
 * (the lower point index, then the lower keypoint index, on a tie), so a keypoint that two points
 * want goes to the nearer one, and the other point gets its next nearest. `taken` marks keypoints
 * that are not available. Returns the keypoint of each point, or -1.
 */
export function kltAssignNearest(
  x: ArrayLike<number>,
  y: ArrayLike<number>,
  count: number,
  keypoints: ArrayLike<number>,
  radius: number,
  taken?: ArrayLike<number>,
): Int32Array {
  const out = new Int32Array(count).fill(-1),
    total = keypoints.length >> 1;
  if (count === 0 || total === 0 || !(radius > 0)) return out;
  const grid = new Map<number, number[]>(),
    key = (cx: number, cy: number) => cy * 65536 + cx;
  for (let j = 0; j < total; j++) {
    if (taken && taken[j]) continue;
    const k = key(
      Math.floor(keypoints[2 * j]! / radius),
      Math.floor(keypoints[2 * j + 1]! / radius),
    );
    const list = grid.get(k);
    if (list) list.push(j);
    else grid.set(k, [j]);
  }
  const candidates: { d: number; i: number; j: number }[] = [];
  for (let i = 0; i < count; i++) {
    const cx = Math.floor(x[i]! / radius),
      cy = Math.floor(y[i]! / radius);
    for (let dy = -1; dy <= 1; dy++)
      for (let dx = -1; dx <= 1; dx++) {
        const list = grid.get(key(cx + dx, cy + dy));
        if (!list) continue;
        for (const j of list) {
          const d = Math.hypot(keypoints[2 * j]! - x[i]!, keypoints[2 * j + 1]! - y[i]!);
          if (d <= radius) candidates.push({ d, i, j });
        }
      }
  }
  candidates.sort((a, b) => a.d - b.d || a.i - b.i || a.j - b.j);
  const claimed = new Uint8Array(total);
  for (const { i, j } of candidates) {
    if (out[i]! >= 0 || claimed[j]) continue;
    out[i] = j;
    claimed[j] = 1;
  }
  return out;
}

/**
 * The tracks to remove after the verdict of section 3.5, item 6. `verdicts` holds, for each
 * track id, the number of its matches that a verified pair kept (`inlier`) and that it dropped
 * (`outlier`). A track is rejected when at least one verified pair dropped its match and no
 * verified pair kept it. A track that no verified pair judged stays.
 */
export function kltReject(
  verdicts: ReadonlyMap<number, { inlier: number; outlier: number }>,
): Uint32Array {
  const out: number[] = [];
  for (const [id, v] of verdicts) if (v.outlier > 0 && v.inlier === 0) out.push(id);
  return Uint32Array.from(out.sort((a, b) => a - b));
}

/** A hash grid of points for the distance tests of selection and of the join (sections 3.3, 3.4). */
export class KltPointGrid {
  private readonly cells = new Map<number, number[]>();
  private readonly cell: number;
  constructor(cell: number) {
    this.cell = cell;
  }
  private key(x: number, y: number): number {
    return Math.floor(y / this.cell) * 65536 + Math.floor(x / this.cell);
  }
  add(x: number, y: number): void {
    const key = this.key(x, y),
      list = this.cells.get(key);
    if (list) list.push(x, y);
    else this.cells.set(key, [x, y]);
  }
  /** Whether a stored point lies closer than `radius` to (x, y). `radius` must not exceed the cell. */
  near(x: number, y: number, radius: number): boolean {
    const cx = Math.floor(x / this.cell),
      cy = Math.floor(y / this.cell);
    for (let dy = -1; dy <= 1; dy++)
      for (let dx = -1; dx <= 1; dx++) {
        const list = this.cells.get((cy + dy) * 65536 + cx + dx);
        if (list)
          for (let i = 0; i < list.length; i += 2)
            if (Math.hypot(list[i]! - x, list[i + 1]! - y) < radius) return true;
      }
    return false;
  }
}
