'use typeshade';

// Canny edges and the Hough line transform on a resident RGBA image, after scikit-image's
// `feature.canny` and `transform.hough_line`, and OpenCV's `HoughLines`.
//
// Canny: gray, a separable Gaussian that renormalizes at the border (scikit-image's masked
// smoothing), a Sobel gradient with scikit-image's normalization, non-maximum suppression on four
// quantized directions, and hysteresis. Hysteresis runs as passes that grow the strong pixels over
// the weak ones until no pixel changes.
//
// Hough: every edge pixel votes once for each theta in `thetaBins` angles of [0, pi) with an atomic
// add. The line is `x cos(theta) + y sin(theta) = rho`. All trigonometry is fixed-point: the cos and
// sin tables hold integers scaled by 65536, so a vote is an integer expression and the GPU and the
// CPU reference give the same accumulator.
//
// Arrays: `houghGray`, `houghTmp` have one f32 per pixel; `houghGrad` has three f32 per pixel (gx,
// gy, magnitude); `houghState` has one atomic u32 per pixel (0 none, 1 weak, 2 edge);
// `houghAcc` has `thetaBins x rhoCount` atomic u32 cells, the cell of (theta t, rho r) at
// `t * rhoCount + r + maxRho`.

class HoughControl {
  width: u32;
  height: u32;
  // 0: blur along x, 1: blur along y.
  axis: u32;
  // Gaussian radius: the weights table holds 2 * radius + 1 values.
  radius: u32;
  low: f32;
  high: f32;
  thetaBins: u32;
  rhoCount: u32;
  maxRho: u32;
  // Peaks: the votes a peak needs and the half sizes of the suppression window.
  threshold: u32;
  windowRho: i32;
  windowTheta: i32;
  // Segments: the shortest segment, squared, and the longest gap.
  minLengthSquared: u32;
  maxGap: u32;
}
class HoughRange {
  first: u32;
  end: u32;
}
declare const hough: uniform<HoughControl>;
declare const houghRange: uniform<HoughRange>;
declare const houghRgba: storage<array<u32>>;
declare const houghWeights: storage<array<f32>>;
declare const houghGray: storage<array<f32>, 'read_write'>;
declare const houghSrc: storage<array<f32>>;
declare const houghDst: storage<array<f32>, 'read_write'>;
declare const houghGrad: storage<array<f32>, 'read_write'>;
declare const houghGradRead: storage<array<f32>>;
declare const houghState: storage<array<atomic<u32>>, 'read_write'>;
declare const houghChanged: storage<array<atomic<u32>>, 'read_write'>;
declare const houghList: storage<array<u32>, 'read_write'>;
declare const houghListRead: storage<array<u32>>;
declare const houghTrig: storage<array<f32>>;
declare const houghAcc: storage<array<atomic<u32>>, 'read_write'>;
declare const houghAccRead: storage<array<u32>>;
declare const houghPeakMap: storage<array<u32>, 'read_write'>;
declare const houghLines: storage<array<u32>>;
declare const houghSegments: storage<array<u32>, 'read_write'>;

// The most segments kept for one line, and the u32 per line in `houghSegments`: the count, then four
// coordinates per segment.
export const MAX_SEGMENTS: u32 = 16;
export const SEGMENT_STRIDE: u32 = 65;
// Two gradient magnitudes closer than this are a tie: the tie is broken the same way in f32 and f64.
export const TIE: f32 = 0.00001;

/** One invocation per pixel: gray with scikit-image's `rgb2gray` weights, in [0, 1]. */
@compute([64])
export function houghToGray(@builtin('global_invocation_id') gid: vec3u): void {
  let i: u32 = houghRange.first + gid.x;
  const active: bool = i < houghRange.end;
  if (!active) i = max(houghRange.end, 1) - 1;
  const pixel: u32 = houghRgba[i];
  const r: f32 = f32(pixel & 255);
  const g: f32 = f32((pixel >> 8) & 255);
  const b: f32 = f32((pixel >> 16) & 255);
  if (active) houghGray[i] = (0.2125 * r + 0.7154 * g + 0.0721 * b) / 255.0;
}

/**
 * One invocation per pixel: one pass of the separable Gaussian along `axis`. The weights of the
 * pixels outside the image drop out and the rest are renormalized.
 */
@compute([64])
export function houghBlur(@builtin('global_invocation_id') gid: vec3u): void {
  let i: u32 = houghRange.first + gid.x;
  const active: bool = i < houghRange.end;
  if (!active) i = max(houghRange.end, 1) - 1;
  const w: i32 = i32(hough.width);
  const h: i32 = i32(hough.height);
  const x: i32 = i32(i % hough.width);
  const y: i32 = i32(i / hough.width);
  const radius: i32 = i32(hough.radius);
  let sum: f32 = 0;
  let total: f32 = 0;
  for (let k: i32 = -radius; k <= radius; k++) {
    const nx: i32 = x + select(0, k, hough.axis === 0);
    const ny: i32 = y + select(k, 0, hough.axis === 0);
    if (nx >= 0 && ny >= 0 && nx < w && ny < h) {
      const weight: f32 = houghWeights[u32(k + radius)];
      sum += weight * houghSrc[u32(ny * w + nx)];
      total += weight;
    }
  }
  if (active) houghDst[i] = sum / total;
}

function sample(x: i32, y: i32): f32 {
  const w: i32 = i32(hough.width);
  const h: i32 = i32(hough.height);
  return houghSrc[u32(clamp(y, 0, h - 1) * w + clamp(x, 0, w - 1))];
}

/** One invocation per pixel: the Sobel gradient (scikit-image's normalization) and its magnitude. */
@compute([64])
export function houghGradient(@builtin('global_invocation_id') gid: vec3u): void {
  let i: u32 = houghRange.first + gid.x;
  const active: bool = i < houghRange.end;
  if (!active) i = max(houghRange.end, 1) - 1;
  const x: i32 = i32(i % hough.width);
  const y: i32 = i32(i / hough.width);
  const gx: f32 =
    (sample(x + 1, y - 1) + 2.0 * sample(x + 1, y) + sample(x + 1, y + 1) -
      sample(x - 1, y - 1) - 2.0 * sample(x - 1, y) - sample(x - 1, y + 1)) / 4.0;
  const gy: f32 =
    (sample(x - 1, y + 1) + 2.0 * sample(x, y + 1) + sample(x + 1, y + 1) -
      sample(x - 1, y - 1) - 2.0 * sample(x, y - 1) - sample(x + 1, y - 1)) / 4.0;
  if (active) {
    houghGrad[3 * i] = gx;
    houghGrad[3 * i + 1] = gy;
    houghGrad[3 * i + 2] = sqrt(gx * gx + gy * gy) * 0.7071067811865476;
  }
}

/**
 * One invocation per pixel: non-maximum suppression along the gradient direction, quantized to
 * four. A pixel survives when it is above the low threshold, above the neighbour behind it and not
 * below the neighbour ahead of it (so a two-pixel plateau keeps one pixel). The image border is
 * never an edge. The state is 2 when the magnitude is above the high threshold, else 1.
 */
@compute([64])
export function houghSuppress(@builtin('global_invocation_id') gid: vec3u): void {
  let i: u32 = houghRange.first + gid.x;
  const active: bool = i < houghRange.end;
  if (!active) i = max(houghRange.end, 1) - 1;
  const w: i32 = i32(hough.width);
  const h: i32 = i32(hough.height);
  const x: i32 = i32(i % hough.width);
  const y: i32 = i32(i / hough.width);
  const inside: bool = x > 0 && y > 0 && x < w - 1 && y < h - 1;
  // A border pixel reads its own value for both neighbours, so no read leaves the image.
  const gx: f32 = houghGradRead[3 * i];
  const gy: f32 = houghGradRead[3 * i + 1];
  const m: f32 = houghGradRead[3 * i + 2];
  const ax: f32 = abs(gx);
  const ay: f32 = abs(gy);
  let dx: i32 = 1;
  let dy: i32 = 0;
  if (ay <= 0.41421356237309503 * ax) {
    dx = 1;
    dy = 0;
  } else if (ax <= 0.41421356237309503 * ay) {
    dx = 0;
    dy = 1;
  } else if (gx * gy > 0.0) {
    dx = 1;
    dy = 1;
  } else {
    dx = 1;
    dy = -1;
  }
  const behind: u32 = select(i, u32((y - dy) * w + (x - dx)), inside);
  const ahead: u32 = select(i, u32((y + dy) * w + (x + dx)), inside);
  const keep: bool =
    inside && m >= hough.low && m > houghGradRead[3 * behind + 2] + TIE && m >= houghGradRead[3 * ahead + 2] - TIE;
  const weak: u32 = 1;
  const strong: u32 = 2;
  const none: u32 = 0;
  const level: u32 = select(weak, strong, m >= hough.high);
  if (active) atomicStore(houghState[i], select(none, level, keep));
}

/**
 * One invocation per pixel: a weak pixel (1) next to an edge pixel (2), in 8-connectivity, becomes
 * an edge. The update is in place and only ever raises a state, so passes can run without a
 * barrier between them and the result is the fixed point. `houghChanged[0]` counts the changes.
 */
@compute([64])
export function houghGrow(@builtin('global_invocation_id') gid: vec3u): void {
  let i: u32 = houghRange.first + gid.x;
  const active: bool = i < houghRange.end;
  if (!active) i = max(houghRange.end, 1) - 1;
  const w: i32 = i32(hough.width);
  const h: i32 = i32(hough.height);
  const x: i32 = i32(i % hough.width);
  const y: i32 = i32(i / hough.width);
  const self: u32 = atomicLoad(houghState[i]);
  let near: bool = false;
  for (let dy: i32 = -1; dy <= 1; dy++)
    for (let dx: i32 = -1; dx <= 1; dx++) {
      const nx: i32 = x + dx;
      const ny: i32 = y + dy;
      if (nx >= 0 && ny >= 0 && nx < w && ny < h && (dx !== 0 || dy !== 0)) {
        if (atomicLoad(houghState[u32(ny * w + nx)]) === 2) near = true;
      }
    }
  if (active && self === 1 && near) {
    atomicStore(houghState[i], 2);
    atomicAdd(houghChanged[0], 1);
  }
}

/** One invocation per pixel: appends the index of each edge pixel to `houghList`. */
@compute([64])
export function houghCollect(@builtin('global_invocation_id') gid: vec3u): void {
  let i: u32 = houghRange.first + gid.x;
  const active: bool = i < houghRange.end;
  if (!active) i = max(houghRange.end, 1) - 1;
  const isEdge: bool = atomicLoad(houghState[i]) === 2;
  if (active && isEdge) {
    const at: u32 = atomicAdd(houghChanged[0], 1);
    houghList[at] = i;
  }
}

/**
 * One invocation per (edge pixel, theta) pair: the vote. The rho is `round(x cos + y sin)` in
 * 16.16 fixed point, rounded half up.
 */
@compute([64])
export function houghVote(@builtin('global_invocation_id') gid: vec3u): void {
  let g: u32 = houghRange.first + gid.x;
  const active: bool = g < houghRange.end;
  if (!active) g = max(houghRange.end, 1) - 1;
  const e: u32 = g / hough.thetaBins;
  const t: u32 = g % hough.thetaBins;
  const p: u32 = houghListRead[e];
  const x: i32 = i32(p % hough.width);
  const y: i32 = i32(p / hough.width);
  const c: i32 = i32(houghTrig[2 * t]);
  const s: i32 = i32(houghTrig[2 * t + 1]);
  const rho: i32 = (x * c + y * s + 32768) >> 16;
  if (active) atomicAdd(houghAcc[t * hough.rhoCount + u32(rho + i32(hough.maxRho))], 1);
}

/**
 * One invocation per accumulator cell: the votes of the cell when they reach the threshold and no
 * cell in the window of +-windowRho and +-windowTheta has more (the lower cell index wins a tie),
 * else 0. Theta wraps at pi with the sign of rho flipped.
 */
@compute([64])
export function houghPeaks(@builtin('global_invocation_id') gid: vec3u): void {
  let cell: u32 = houghRange.first + gid.x;
  const active: bool = cell < houghRange.end;
  if (!active) cell = max(houghRange.end, 1) - 1;
  const t: i32 = i32(cell / hough.rhoCount);
  const r: i32 = i32(cell % hough.rhoCount);
  const bins: i32 = i32(hough.thetaBins);
  const count: i32 = i32(hough.rhoCount);
  const votes: u32 = houghAccRead[cell];
  let isPeak: bool = votes >= hough.threshold;
  if (isPeak) {
    for (let dt: i32 = -hough.windowTheta; dt <= hough.windowTheta; dt++)
      for (let dr: i32 = -hough.windowRho; dr <= hough.windowRho; dr++) {
        let t2: i32 = t + dt;
        let r2: i32 = r + dr;
        if (t2 < 0) {
          t2 += bins;
          r2 = count - 1 - r2;
        } else if (t2 >= bins) {
          t2 -= bins;
          r2 = count - 1 - r2;
        }
        if (r2 >= 0 && r2 < count) {
          const other: u32 = u32(t2 * count + r2);
          const v: u32 = houghAccRead[other];
          if (v > votes || (v === votes && other < cell)) isPeak = false;
        }
      }
  }
  if (active) houghPeakMap[cell] = select(0, votes, isPeak);
}

function floorDiv(n: i32, d: i32): i32 {
  let q: i32 = n / d;
  if (n % d !== 0 && n < 0) q -= 1;
  return q;
}

function emit(base: u32, count: u32, k0: i32, m0: i32, k1: i32, m1: i32, horizontal: bool): u32 {
  const dk: i32 = k1 - k0;
  const dm: i32 = m1 - m0;
  let next: u32 = count;
  if (u32(dk * dk + dm * dm) >= hough.minLengthSquared && count < MAX_SEGMENTS) {
    const at: u32 = base + 1 + 4 * count;
    houghSegments[at] = u32(select(m0, k0, horizontal));
    houghSegments[at + 1] = u32(select(k0, m0, horizontal));
    houghSegments[at + 2] = u32(select(m1, k1, horizontal));
    houghSegments[at + 3] = u32(select(k1, m1, horizontal));
    next = count + 1;
  }
  return next;
}

/**
 * One invocation per accepted line: walks the line along its major axis and cuts it into segments
 * of edge pixels. A step is supported when an edge pixel lies within one pixel of the line across
 * its minor axis. A run of supported steps ends when more than `maxGap` steps in a row are not
 * supported, and it is kept when its end points are `minLength` apart. The line `i` writes its
 * count and segments (x0, y0, x1, y1) at `i * SEGMENT_STRIDE`.
 */
@compute([64])
export function houghSegment(@builtin('global_invocation_id') gid: vec3u): void {
  let line: u32 = houghRange.first + gid.x;
  const active: bool = line < houghRange.end;
  if (!active) line = max(houghRange.end, 1) - 1;
  const w: i32 = i32(hough.width);
  const h: i32 = i32(hough.height);
  const t: u32 = houghLines[2 * line];
  const rho: i32 = i32(houghLines[2 * line + 1]) - i32(hough.maxRho);
  const c: i32 = i32(houghTrig[2 * t]);
  const s: i32 = i32(houghTrig[2 * t + 1]);
  const horizontal: bool = s >= abs(c);
  const steps: i32 = select(h, w, horizontal);
  const across: i32 = select(w, h, horizontal);
  // The minor coordinate is (rho * 65536 - k * other) / divisor, rounded.
  const other: i32 = select(s, c, horizontal);
  const divisor: i32 = select(c, s, horizontal);
  const flip: bool = divisor < 0;
  const den: i32 = select(divisor, -divisor, flip);
  const base: u32 = line * SEGMENT_STRIDE;
  let count: u32 = 0;
  let start: i32 = -1;
  let startM: i32 = 0;
  let last: i32 = -1;
  let lastM: i32 = 0;
  for (let k: i32 = 0; k < steps; k++) {
    const raw: i32 = rho * 65536 - k * other;
    const num: i32 = select(raw, -raw, flip);
    const m: i32 = floorDiv(2 * num + den, 2 * den);
    let supported: bool = false;
    for (let d: i32 = -1; d <= 1; d++) {
      const mm: i32 = m + d;
      if (mm >= 0 && mm < across && m >= 0 && m < across) {
        const px: i32 = select(mm, k, horizontal);
        const py: i32 = select(k, mm, horizontal);
        if (atomicLoad(houghState[u32(py * w + px)]) === 2) supported = true;
      }
    }
    if (supported) {
      if (start < 0) {
        start = k;
        startM = m;
      }
      last = k;
      lastM = m;
    } else if (start >= 0 && k - last > i32(hough.maxGap)) {
      count = emit(base, count, start, startM, last, lastM, horizontal);
      start = -1;
    }
  }
  if (start >= 0) count = emit(base, count, start, startM, last, lastM, horizontal);
  if (active) houghSegments[base] = count;
}
