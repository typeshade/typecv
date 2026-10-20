'use typeshade';

// Pyramidal Lucas-Kanade tracking (`docs/klt-tracking.md`, sections 2 to 4): the pyramid, the
// Scharr gradient, the inverse compositional tracker and the forward-backward check. Every
// kernel is one invocation per element with no workgroup memory and no barrier, so the CPU tier
// runs the same code as the GPU tier. `src/gpu/klt.ts` holds the CPU reference that runs the same
// arithmetic in the same order with `Math.fround` after each operation.
//
// Layouts (section 4.1).
// - A pyramid is one f32 array of all levels, level 0 first. Level l + 1 has
//   ceil(width_l / 2) x ceil(height_l / 2) pixels and follows level l. Offsets and sizes are
//   computed from the control, by `levelOffset`, `levelWidth` and `levelHeight`.
// - A gradient array has the same levels and two floats (Ix, Iy) for each pixel.
// - A track record is four f32 words: x, y, a measure, a status word as an exact integer in f32.
//   Input records of `kltTrack` are (x, y, unused, status). The output of `kltTrack` carries the
//   zero-mean NCC of the final window in word 2. The output of `kltCheck` carries the
//   forward-backward error in word 2.
// - A status word: bit 0 alive, bits 1 to 4 the end reason of section 3.5 (`KLT_END_*` in
//   `klt.ts`), bits 16 to 23 the iterations used at level 0.
//
// A lane past its chunk takes the chunk's last element, runs the same control flow and writes
// nothing (the clamp-and-mask rule of section 4.3). A lane that converged stops updating `d`,
// and the loop bound is the same for every lane.

class KltControl {
  width: u32;
  height: u32;
  levels: u32;
  halfWindow: u32;
  maxIterations: u32;
  // The step length in pixels below which a level has converged.
  epsilon: f32;
  // The least lambda-min per window pixel, in gray units of [0, 1] and gradients per pixel.
  minEig: f32;
  // The forward-backward error above which a track ends, and the least NCC.
  fbThreshold: f32;
  nccThreshold: f32;
}
// One level that `kltPyrDown` or `kltScharr` writes, and the level that `kltPyrDown` reads.
class KltLevel {
  width: u32;
  height: u32;
  offset: u32;
  srcWidth: u32;
  srcHeight: u32;
  srcOffset: u32;
}
// The tracks of one chunk: `first..end`.
class KltRange {
  first: u32;
  end: u32;
}

declare const kltControl: uniform<KltControl>;
declare const kltLevel: uniform<KltLevel>;
declare const kltRange: uniform<KltRange>;
declare const kltPyramid: storage<array<f32>, 'read_write'>;
declare const kltGrayIn: storage<array<f32>>;
declare const kltGradOut: storage<array<f32>, 'read_write'>;
// The template pyramid and its gradient, and the target pyramid.
declare const kltGray: storage<array<f32>>;
declare const kltGrad: storage<array<f32>>;
declare const kltTarget: storage<array<f32>>;
declare const kltStart: storage<array<f32>>;
declare const kltNext: storage<array<f32>, 'read_write'>;
declare const kltForward: storage<array<f32>>;
declare const kltBack: storage<array<f32>>;
declare const kltChecked: storage<array<f32>, 'read_write'>;

/** The offset of a level in a pyramid array: the sum of the pixels of the finer levels. */
function levelOffset(level: u32): u32 {
  let w: u32 = kltControl.width;
  let h: u32 = kltControl.height;
  let offset: u32 = 0;
  for (let l: u32 = 0; l < level; l++) {
    offset += w * h;
    w = (w + 1) / 2;
    h = (h + 1) / 2;
  }
  return offset;
}
function levelWidth(level: u32): u32 {
  let w: u32 = kltControl.width;
  for (let l: u32 = 0; l < level; l++) w = (w + 1) / 2;
  return w;
}
function levelHeight(level: u32): u32 {
  let h: u32 = kltControl.height;
  for (let l: u32 = 0; l < level; l++) h = (h + 1) / 2;
  return h;
}

/** A pixel of the pyramid being built, the coordinates clamped to the level. */
function pyramidAt(offset: u32, w: u32, h: u32, x: i32, y: i32): f32 {
  const cx: u32 = u32(clamp(x, 0, i32(w) - 1));
  const cy: u32 = u32(clamp(y, 0, i32(h) - 1));
  return kltPyramid[offset + cy * w + cx];
}
function grayAt(offset: u32, w: u32, h: u32, x: i32, y: i32): f32 {
  const cx: u32 = u32(clamp(x, 0, i32(w) - 1));
  const cy: u32 = u32(clamp(y, 0, i32(h) - 1));
  return kltGrayIn[offset + cy * w + cx];
}

/** One lane per output pixel of level l + 1: the 5-tap binomial blur of level l in x, then in y. */
@compute([64])
export function kltPyrDown(@builtin('global_invocation_id') gid: vec3u): void {
  const count: u32 = kltLevel.width * kltLevel.height;
  const active: bool = gid.x < count;
  const k: u32 = select(0, gid.x, active);
  const ox: u32 = k % kltLevel.width;
  const oy: u32 = k / kltLevel.width;
  const cx: i32 = 2 * i32(ox);
  const cy: i32 = 2 * i32(oy);
  const w: u32 = kltLevel.srcWidth;
  const h: u32 = kltLevel.srcHeight;
  const o: u32 = kltLevel.srcOffset;
  let sum: f32 = 0.;
  for (let r: i32 = -2; r <= 2; r++) {
    const wy: f32 = select(select(1., 4., r === -1 || r === 1), 6., r === 0);
    const row: i32 = cy + r;
    const line: f32 =
      pyramidAt(o, w, h, cx - 2, row) +
      4. * pyramidAt(o, w, h, cx - 1, row) +
      6. * pyramidAt(o, w, h, cx, row) +
      4. * pyramidAt(o, w, h, cx + 1, row) +
      pyramidAt(o, w, h, cx + 2, row);
    sum += wy * line;
  }
  if (active) kltPyramid[kltLevel.offset + k] = sum / 256.;
}

/** One lane per pixel of a level: the 3 x 3 Scharr derivatives (3, 10, 3) / 32, border clamped. */
@compute([64])
export function kltScharr(@builtin('global_invocation_id') gid: vec3u): void {
  const count: u32 = kltLevel.width * kltLevel.height;
  const active: bool = gid.x < count;
  const k: u32 = select(0, gid.x, active);
  const w: u32 = kltLevel.width;
  const h: u32 = kltLevel.height;
  const o: u32 = kltLevel.offset;
  const x: i32 = i32(k % w);
  const y: i32 = i32(k / w);
  const tl: f32 = grayAt(o, w, h, x - 1, y - 1);
  const tc: f32 = grayAt(o, w, h, x, y - 1);
  const tr: f32 = grayAt(o, w, h, x + 1, y - 1);
  const ml: f32 = grayAt(o, w, h, x - 1, y);
  const mr: f32 = grayAt(o, w, h, x + 1, y);
  const bl: f32 = grayAt(o, w, h, x - 1, y + 1);
  const bc: f32 = grayAt(o, w, h, x, y + 1);
  const br: f32 = grayAt(o, w, h, x + 1, y + 1);
  const ix: f32 = (3. * (tr - tl) + 10. * (mr - ml) + 3. * (br - bl)) / 32.;
  const iy: f32 = (3. * (bl - tl) + 10. * (bc - tc) + 3. * (br - tr)) / 32.;
  if (active) {
    kltGradOut[2 * (o + k)] = ix;
    kltGradOut[2 * (o + k) + 1] = iy;
  }
}

// The bilinear read of a level, the coordinates clamped to the level (section 2.1).
function bilinearGray(offset: u32, w: u32, h: u32, x: f32, y: f32): f32 {
  const cx: f32 = clamp(x, 0., f32(w - 1));
  const cy: f32 = clamp(y, 0., f32(h - 1));
  const x0: u32 = min(u32(floor(cx)), w - 1);
  const y0: u32 = min(u32(floor(cy)), h - 1);
  const x1: u32 = min(x0 + 1, w - 1);
  const y1: u32 = min(y0 + 1, h - 1);
  const ax: f32 = cx - f32(x0);
  const ay: f32 = cy - f32(y0);
  const top: f32 = kltGray[offset + y0 * w + x0] * (1. - ax) + kltGray[offset + y0 * w + x1] * ax;
  const bottom: f32 =
    kltGray[offset + y1 * w + x0] * (1. - ax) + kltGray[offset + y1 * w + x1] * ax;
  return top * (1. - ay) + bottom * ay;
}
function bilinearTarget(offset: u32, w: u32, h: u32, x: f32, y: f32): f32 {
  const cx: f32 = clamp(x, 0., f32(w - 1));
  const cy: f32 = clamp(y, 0., f32(h - 1));
  const x0: u32 = min(u32(floor(cx)), w - 1);
  const y0: u32 = min(u32(floor(cy)), h - 1);
  const x1: u32 = min(x0 + 1, w - 1);
  const y1: u32 = min(y0 + 1, h - 1);
  const ax: f32 = cx - f32(x0);
  const ay: f32 = cy - f32(y0);
  const top: f32 =
    kltTarget[offset + y0 * w + x0] * (1. - ax) + kltTarget[offset + y0 * w + x1] * ax;
  const bottom: f32 =
    kltTarget[offset + y1 * w + x0] * (1. - ax) + kltTarget[offset + y1 * w + x1] * ax;
  return top * (1. - ay) + bottom * ay;
}
// One channel (0: Ix, 1: Iy) of the template gradient.
function bilinearGrad(offset: u32, w: u32, h: u32, x: f32, y: f32, c: u32): f32 {
  const cx: f32 = clamp(x, 0., f32(w - 1));
  const cy: f32 = clamp(y, 0., f32(h - 1));
  const x0: u32 = min(u32(floor(cx)), w - 1);
  const y0: u32 = min(u32(floor(cy)), h - 1);
  const x1: u32 = min(x0 + 1, w - 1);
  const y1: u32 = min(y0 + 1, h - 1);
  const ax: f32 = cx - f32(x0);
  const ay: f32 = cy - f32(y0);
  const top: f32 =
    kltGrad[2 * (offset + y0 * w + x0) + c] * (1. - ax) +
    kltGrad[2 * (offset + y0 * w + x1) + c] * ax;
  const bottom: f32 =
    kltGrad[2 * (offset + y1 * w + x0) + c] * (1. - ax) +
    kltGrad[2 * (offset + y1 * w + x1) + c] * ax;
  return top * (1. - ay) + bottom * ay;
}

/**
 * One lane per track, all levels and iterations (section 4.3). The start record is
 * (x, y, unused, status) in the template frame. A start that is not alive is copied through.
 * The result is (x, y, NCC, status) in the target frame.
 */
@compute([64])
export function kltTrack(@builtin('global_invocation_id') gid: vec3u): void {
  const first: u32 = kltRange.first;
  let k: u32 = first + gid.x;
  const active: bool = k < kltRange.end;
  if (!active) k = max(kltRange.end, 1) - 1;
  const sx: f32 = kltStart[4 * k];
  const sy: f32 = kltStart[4 * k + 1];
  const startStatus: u32 = u32(kltStart[4 * k + 3]);
  const hw: i32 = i32(kltControl.halfWindow);
  const levels: u32 = kltControl.levels;
  const window: f32 = f32((2 * hw + 1) * (2 * hw + 1));
  let dx: f32 = 0.;
  let dy: f32 = 0.;
  let gxx: f32 = 0.;
  let gxy: f32 = 0.;
  let gyy: f32 = 0.;
  let converged: bool = false;
  let used: u32 = 0;
  for (let step: u32 = 0; step < levels; step++) {
    const level: u32 = levels - 1 - step;
    const scale: f32 = f32(u32(1) << level);
    const off: u32 = levelOffset(level);
    const w: u32 = levelWidth(level);
    const h: u32 = levelHeight(level);
    const px: f32 = sx / scale;
    const py: f32 = sy / scale;
    let a: f32 = 0.;
    let b: f32 = 0.;
    let c: f32 = 0.;
    for (let j: i32 = -hw; j <= hw; j++)
      for (let i: i32 = -hw; i <= hw; i++) {
        const tx: f32 = px + f32(i);
        const ty: f32 = py + f32(j);
        const gx: f32 = bilinearGrad(off, w, h, tx, ty, 0);
        const gy: f32 = bilinearGrad(off, w, h, tx, ty, 1);
        a += gx * gx;
        b += gx * gy;
        c += gy * gy;
      }
    gxx = a;
    gxy = b;
    gyy = c;
    const det: f32 = a * c - b * b;
    const inverse: f32 = select(0., 1. / max(det, 1e-12), det > 1e-12);
    let done: bool = false;
    for (let it: u32 = 0; it < kltControl.maxIterations; it++) {
      let bx: f32 = 0.;
      let by: f32 = 0.;
      for (let j: i32 = -hw; j <= hw; j++)
        for (let i: i32 = -hw; i <= hw; i++) {
          const tx: f32 = px + f32(i);
          const ty: f32 = py + f32(j);
          const r: f32 =
            bilinearGray(off, w, h, tx, ty) - bilinearTarget(off, w, h, tx + dx, ty + dy);
          bx += r * bilinearGrad(off, w, h, tx, ty, 0);
          by += r * bilinearGrad(off, w, h, tx, ty, 1);
        }
      const stepX: f32 = (c * bx - b * by) * inverse;
      const stepY: f32 = (a * by - b * bx) * inverse;
      if (!done) {
        dx += stepX;
        dy += stepY;
        if (level === 0) used += 1;
        if (stepX * stepX + stepY * stepY < kltControl.epsilon * kltControl.epsilon) done = true;
      }
    }
    if (level === 0) converged = done;
    else {
      dx *= 2.;
      dy *= 2.;
    }
  }
  const rx: f32 = sx + dx;
  const ry: f32 = sy + dy;
  // The zero-mean NCC of the final window at level 0.
  const w0: u32 = kltControl.width;
  const h0: u32 = kltControl.height;
  let meanI: f32 = 0.;
  let meanJ: f32 = 0.;
  for (let j: i32 = -hw; j <= hw; j++)
    for (let i: i32 = -hw; i <= hw; i++) {
      meanI += bilinearGray(0, w0, h0, sx + f32(i), sy + f32(j));
      meanJ += bilinearTarget(0, w0, h0, rx + f32(i), ry + f32(j));
    }
  meanI /= window;
  meanJ /= window;
  let sii: f32 = 0.;
  let sjj: f32 = 0.;
  let sij: f32 = 0.;
  for (let j: i32 = -hw; j <= hw; j++)
    for (let i: i32 = -hw; i <= hw; i++) {
      const ti: f32 = bilinearGray(0, w0, h0, sx + f32(i), sy + f32(j)) - meanI;
      const tj: f32 = bilinearTarget(0, w0, h0, rx + f32(i), ry + f32(j)) - meanJ;
      sii += ti * ti;
      sjj += tj * tj;
      sij += ti * tj;
    }
  const product: f32 = sii * sjj;
  const ncc: f32 = select(0., sij / sqrt(max(product, 1e-30)), product > 1e-30);
  // Section 3.5: out of frame (1), low structure (2), no convergence (3).
  const lo: f32 = f32(hw);
  const hiX: f32 = f32(w0) - 1. - lo;
  const hiY: f32 = f32(h0) - 1. - lo;
  const outside: bool =
    sx < lo || sx > hiX || sy < lo || sy > hiY || rx < lo || rx > hiX || ry < lo || ry > hiY;
  const trace: f32 = gxx + gyy;
  const root: f32 = sqrt((gxx - gyy) * (gxx - gyy) + 4. * gxy * gxy);
  const lowStructure: bool = (trace - root) / 2. / window < kltControl.minEig;
  let reason: u32 = 0;
  if (outside) reason = 1;
  else if (lowStructure) reason = 2;
  else if (!converged) reason = 3;
  const alive: bool = (startStatus & 1) === 1;
  const status: u32 = select(
    startStatus,
    select(reason << 1, 1 | (used << 16), reason === 0),
    alive,
  );
  if (active) {
    kltNext[4 * k] = select(sx, rx, alive);
    kltNext[4 * k + 1] = select(sy, ry, alive);
    kltNext[4 * k + 2] = select(0., ncc, alive);
    kltNext[4 * k + 3] = f32(status);
  }
}

/**
 * One lane per track: the forward-backward error (section 2.3) and the NCC check (section 3.5).
 * `kltStart` holds the seeds, `kltForward` the forward result and `kltBack` the backward result
 * that started at the forward position. The result is (forward x, y, error, status).
 */
@compute([64])
export function kltCheck(@builtin('global_invocation_id') gid: vec3u): void {
  let k: u32 = kltRange.first + gid.x;
  const active: bool = k < kltRange.end;
  if (!active) k = max(kltRange.end, 1) - 1;
  const fx: f32 = kltForward[4 * k];
  const fy: f32 = kltForward[4 * k + 1];
  const ncc: f32 = kltForward[4 * k + 2];
  const forwardStatus: u32 = u32(kltForward[4 * k + 3]);
  const backStatus: u32 = u32(kltBack[4 * k + 3]);
  const ex: f32 = kltBack[4 * k] - kltStart[4 * k];
  const ey: f32 = kltBack[4 * k + 1] - kltStart[4 * k + 1];
  const error: f32 = sqrt(ex * ex + ey * ey);
  const alive: bool = (forwardStatus & 1) === 1;
  const backAlive: bool = (backStatus & 1) === 1;
  let status: u32 = forwardStatus;
  if (alive) {
    if (!backAlive || !(error <= kltControl.fbThreshold)) status = 4 << 1;
    else if (!(ncc >= kltControl.nccThreshold)) status = 5 << 1;
  }
  if (active) {
    kltChecked[4 * k] = fx;
    kltChecked[4 * k + 1] = fy;
    kltChecked[4 * k + 2] = select(0., error, alive && backAlive);
    kltChecked[4 * k + 3] = f32(status);
  }
}
