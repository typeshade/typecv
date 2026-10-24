/**
 * Global translation between two frames by phase correlation, on a small grayscale thumbnail.
 * The method is the one of scikit-image `registration.phase_cross_correlation` and OpenCV
 * `phaseCorrelate`: the normalised cross-power spectrum of two Hann-windowed images has an
 * inverse transform with a sharp peak at the shift. This module has no DOM and no worker globals,
 * so a unit test runs it in Node and the lab's worker runs it on the CPU.
 *
 * Convention: the result is the displacement `d` of the content from the reference to the
 * candidate, so that `candidate(x) = reference(x - d)`. A candidate whose content moved right has
 * `dx > 0`.
 */

/** The side of the square thumbnail the lab correlates (a power of two). */
export const PHASE_SIZE = 128;

/** A radix-2 complex FFT plan of one size: bit reversal and twiddles are computed once. */
class FftPlan {
  readonly n: number;
  private readonly rev: Uint32Array;
  private readonly cos: Float64Array;
  private readonly sin: Float64Array;
  constructor(n: number) {
    if (n < 2 || (n & (n - 1)) !== 0) throw new Error('FFT size must be a power of two: ' + n);
    this.n = n;
    this.rev = new Uint32Array(n);
    let bits = 0;
    while (1 << bits < n) bits++;
    for (let i = 0; i < n; i++) {
      let r = 0;
      for (let b = 0; b < bits; b++) if (i & (1 << b)) r |= 1 << (bits - 1 - b);
      this.rev[i] = r;
    }
    this.cos = new Float64Array(n / 2);
    this.sin = new Float64Array(n / 2);
    for (let i = 0; i < n / 2; i++) {
      this.cos[i] = Math.cos((2 * Math.PI * i) / n);
      this.sin[i] = -Math.sin((2 * Math.PI * i) / n);
    }
  }
  /** In-place transform of `n` complex values at `offset`, strided by `stride` (in complex values). */
  transform(re: Float64Array, im: Float64Array, offset: number, stride: number, inverse: boolean) {
    const { n, rev } = this;
    for (let i = 0; i < n; i++) {
      const j = rev[i];
      if (j > i) {
        const a = offset + i * stride,
          b = offset + j * stride;
        let t = re[a];
        re[a] = re[b];
        re[b] = t;
        t = im[a];
        im[a] = im[b];
        im[b] = t;
      }
    }
    const sign = inverse ? -1 : 1;
    for (let size = 2; size <= n; size <<= 1) {
      const half = size >> 1,
        step = n / size;
      for (let start = 0; start < n; start += size) {
        for (let k = 0; k < half; k++) {
          const wr = this.cos[k * step],
            wi = sign * this.sin[k * step];
          const a = offset + (start + k) * stride,
            b = offset + (start + k + half) * stride;
          const xr = re[b] * wr - im[b] * wi,
            xi = re[b] * wi + im[b] * wr;
          re[b] = re[a] - xr;
          im[b] = im[a] - xi;
          re[a] += xr;
          im[a] += xi;
        }
      }
    }
  }
}

const plans = new Map<number, FftPlan>();
const planOf = (n: number) => {
  let plan = plans.get(n);
  if (!plan) plans.set(n, (plan = new FftPlan(n)));
  return plan;
};

/** In-place 2-D FFT of an `n × n` complex image (rows, then columns). The inverse is scaled by 1/n². */
export function fft2d(re: Float64Array, im: Float64Array, n: number, inverse = false) {
  const plan = planOf(n);
  for (let y = 0; y < n; y++) plan.transform(re, im, y * n, 1, inverse);
  for (let x = 0; x < n; x++) plan.transform(re, im, x, n, inverse);
  if (inverse) {
    const scale = 1 / (n * n);
    for (let i = 0; i < n * n; i++) {
      re[i] *= scale;
      im[i] *= scale;
    }
  }
}

type ThumbnailPlan = {
  width: number;
  height: number;
  n: number;
  xs: Uint16Array;
  counts: Float64Array;
};
// A stream normally keeps one shape. Retain only the latest plan, so switching videos or
// thumbnail sizes cannot accumulate a cache proportional to the number of encountered shapes.
let thumbnailPlan: ThumbnailPlan | undefined;
function planThumbnail(width: number, height: number, n: number): ThumbnailPlan {
  if (thumbnailPlan?.width === width && thumbnailPlan.height === height && thumbnailPlan.n === n)
    return thumbnailPlan;
  const xs = new Uint16Array(width),
    counts = new Float64Array(n * n),
    columns = new Float64Array(n),
    rows = new Float64Array(n);
  for (let x = 0; x < width; x++) {
    xs[x] = Math.min(n - 1, Math.floor((x * n) / width));
    columns[xs[x]]++;
  }
  for (let y = 0; y < height; y++) rows[Math.min(n - 1, Math.floor((y * n) / height))]++;
  // Cell populations are integer Cartesian products. Avoid another full-frame traversal
  // on the first frame as well as counts++ on every subsequent frame.
  for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) counts[y * n + x] = rows[y] * columns[x];
  return (thumbnailPlan = { width, height, n, xs, counts });
}

/**
 * The luma of an RGBA frame averaged over boxes into an `n × n` thumbnail (the frame is stretched,
 * so a shift of the thumbnail is a fraction of the frame width and height). Rec. 601 weights.
 */
export function grayThumbnail(
  rgba: Uint8ClampedArray | Uint8Array,
  width: number,
  height: number,
  n = PHASE_SIZE,
): Float64Array {
  const out = new Float64Array(n * n),
    { xs, counts } = planThumbnail(width, height, n);
  // Spread each source pixel over the thumbnail cell it falls in: a box filter that stays cheap
  // for the 640 × 360 and 1280 × 720 analysis frames. Each output cell keeps the original
  // source-pixel addition order; normalization reuses the same owned array instead of a copy.
  for (let y = 0; y < height; y++) {
    const row = Math.min(n - 1, Math.floor((y * n) / height)) * n;
    const base = y * width * 4;
    for (let x = 0; x < width; x++) {
      const p = base + x * 4,
        cell = row + xs[x];
      out[cell] += 0.299 * rgba[p] + 0.587 * rgba[p + 1] + 0.114 * rgba[p + 2];
    }
  }
  for (let i = 0; i < n * n; i++) out[i] = counts[i] > 0 ? out[i] / counts[i] : 0;
  return out;
}

/** The separable Hann window of an `n × n` image. */
const windows = new Map<number, Float64Array>();
function hann(n: number): Float64Array {
  let w = windows.get(n);
  if (!w) {
    const line = new Float64Array(n);
    for (let i = 0; i < n; i++) line[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * (i + 0.5)) / n);
    w = new Float64Array(n * n);
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) w[y * n + x] = line[y] * line[x];
    windows.set(n, w);
  }
  return w;
}

/** The spectrum of a windowed, mean-removed thumbnail. Reusable as the reference of many candidates. */
export type Spectrum = { n: number; re: Float64Array; im: Float64Array };

export function spectrumOf(gray: ArrayLike<number>, n = PHASE_SIZE): Spectrum {
  const w = hann(n);
  let mean = 0,
    weight = 0;
  for (let i = 0; i < n * n; i++) {
    mean += gray[i] * w[i];
    weight += w[i];
  }
  mean /= weight;
  const re = new Float64Array(n * n),
    im = new Float64Array(n * n);
  for (let i = 0; i < n * n; i++) re[i] = (gray[i] - mean) * w[i];
  fft2d(re, im, n);
  return { n, re, im };
}

export type PhaseShift = {
  /** Displacement of the content from the reference to the candidate, in thumbnail pixels. */
  dx: number;
  dy: number;
  /** The same as a fraction of the frame width and height. */
  fx: number;
  fy: number;
  /** The height of the correlation peak (1 for a pure shift of a periodic image). */
  peak: number;
  /** Peak over the mean absolute correlation: the confidence of the estimate. */
  ratio: number;
};

/** Half-width in pixels of the window refined around the integer peak, and its step. */
const REFINE_RADIUS = 1.5;
const REFINE_STEP = 0.05;

/**
 * The correlation surface evaluated on a fine grid around the integer peak by a matrix-product
 * DFT (the upsampled cross-correlation of Guizar-Sicairos et al., as scikit-image does with
 * `upsample_factor`). It has no model of the peak shape, so it does not bias the shift of a
 * band-limited image. Returns the peak offset from `(px, py)` after a parabolic fit.
 */
function refine(re: Float64Array, im: Float64Array, n: number, px: number, py: number) {
  const m = Math.round((2 * REFINE_RADIUS) / REFINE_STEP) + 1;
  const kernelRe = new Float64Array(m * n),
    kernelIm = new Float64Array(m * n);
  const signed = (k: number) => (k <= n / 2 ? k : k - n);
  const build = (centre: number, outRe: Float64Array, outIm: Float64Array) => {
    for (let j = 0; j < m; j++) {
      const position = centre + (j * REFINE_STEP - REFINE_RADIUS);
      for (let k = 0; k < n; k++) {
        const angle = (2 * Math.PI * signed(k) * position) / n;
        outRe[j * n + k] = Math.cos(angle);
        outIm[j * n + k] = Math.sin(angle);
      }
    }
  };
  const ky = [new Float64Array(m * n), new Float64Array(m * n)];
  build(py, ky[0], ky[1]);
  build(px, kernelRe, kernelIm);
  // T[j][kx] = sum over ky of Ky[j][ky] * R[ky][kx]
  const tRe = new Float64Array(m * n),
    tIm = new Float64Array(m * n);
  for (let j = 0; j < m; j++)
    for (let k = 0; k < n; k++) {
      const cr = ky[0][j * n + k],
        ci = ky[1][j * n + k];
      for (let x = 0; x < n; x++) {
        const rr = re[k * n + x],
          ri = im[k * n + x];
        tRe[j * n + x] += cr * rr - ci * ri;
        tIm[j * n + x] += cr * ri + ci * rr;
      }
    }
  // S[j][i] = real part of the sum over kx of T[j][kx] * Kx[i][kx]
  const surface = new Float64Array(m * m);
  for (let j = 0; j < m; j++)
    for (let i = 0; i < m; i++) {
      let sum = 0;
      for (let x = 0; x < n; x++)
        sum += tRe[j * n + x] * kernelRe[i * n + x] - tIm[j * n + x] * kernelIm[i * n + x];
      surface[j * m + i] = sum / (n * n);
    }
  let best = -Infinity,
    bi = 0,
    bj = 0;
  for (let j = 0; j < m; j++)
    for (let i = 0; i < m; i++)
      if (surface[j * m + i] > best) ((best = surface[j * m + i]), (bi = i), (bj = j));
  const parabola = (minus: number, centre: number, plus: number) => {
    const denominator = 2 * (2 * centre - plus - minus);
    return denominator > 0 ? (plus - minus) / denominator : 0;
  };
  const inside = bi > 0 && bi < m - 1 && bj > 0 && bj < m - 1;
  const fi = inside
    ? parabola(surface[bj * m + bi - 1], best, surface[bj * m + bi + 1]) * REFINE_STEP
    : 0;
  const fj = inside
    ? parabola(surface[(bj - 1) * m + bi], best, surface[(bj + 1) * m + bi]) * REFINE_STEP
    : 0;
  return {
    ox: bi * REFINE_STEP - REFINE_RADIUS + fi,
    oy: bj * REFINE_STEP - REFINE_RADIUS + fj,
    peak: best,
  };
}

/** Phase correlation of a candidate spectrum against a reference spectrum. */
export function phaseCorrelate(reference: Spectrum, candidate: Spectrum): PhaseShift {
  const n = reference.n;
  if (candidate.n !== n) throw new Error('Spectrum sizes differ');
  const re = new Float64Array(n * n),
    im = new Float64Array(n * n);
  for (let i = 0; i < n * n; i++) {
    // candidate * conj(reference) / |.|: its inverse peaks at the shift from reference to candidate.
    const cr = candidate.re[i] * reference.re[i] + candidate.im[i] * reference.im[i];
    const ci = candidate.im[i] * reference.re[i] - candidate.re[i] * reference.im[i];
    const magnitude = Math.hypot(cr, ci);
    if (magnitude > 1e-9) {
      re[i] = cr / magnitude;
      im[i] = ci / magnitude;
    }
  }
  const whiteRe = Float64Array.from(re),
    whiteIm = Float64Array.from(im);
  fft2d(re, im, n, true);
  let best = -Infinity,
    at = 0,
    sum = 0;
  for (let i = 0; i < n * n; i++) {
    if (re[i] > best) {
      best = re[i];
      at = i;
    }
    sum += Math.abs(re[i]);
  }
  const px = at % n,
    py = (at - px) / n;
  const sx = px > n / 2 ? px - n : px,
    sy = py > n / 2 ? py - n : py;
  const refined = refine(whiteRe, whiteIm, n, sx, sy);
  const dx = sx + refined.ox,
    dy = sy + refined.oy;
  const mean = sum / (n * n);
  return {
    dx,
    dy,
    fx: dx / n,
    fy: dy / n,
    peak: Math.max(best, refined.peak),
    ratio: mean > 0 ? best / mean : 0,
  };
}
