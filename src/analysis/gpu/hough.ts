/**
 * Canny edges and the Hough line transform on the GPU (`docs/gpu-features.md`, section "Straight
 * lines"). `cannyGpu` and `houghLinesGpu` run the TypeShade kernels of `hough.shade.ts`;
 * `canny` and `houghLines` are the CPU references, in double precision, with the same integer
 * voting, peak and segment code.
 */
import type { ShadeDevice } from '../../device/gpu/device.ts';

type Resident<T extends Float32Array | Uint32Array> = T extends Float32Array
  ? ReturnType<ShadeDevice['allocate']>
  : ReturnType<ShadeDevice['allocateU32']>;
import { chunkedPass } from '../../device/gpu/chunked-pass.ts';
import { packRgba } from '../../image/gpu/pack-rgba.ts';
import {
  houghBlur,
  houghCollect,
  houghGradient,
  houghGrow,
  houghPeaks,
  houghSegment,
  houghSuppress,
  houghToGray,
  houghVote,
} from './hough.shade.ts';

/** Options of `canny` and `cannyGpu`, with the defaults of scikit-image's `feature.canny`. */
export interface CannyOptions {
  /** Standard deviation of the Gaussian, in pixels. Default 1. */
  readonly sigma?: number;
  /**
   * What `lowThreshold` and `highThreshold` mean. `'quantile'` (default): fractions in [0, 1] of
   * the gradient magnitudes of the pixels whose magnitude is above `GRADIENT_FLOOR`, so the same
   * values suit a flat sky and a textured wall. `'absolute'`: thresholds on the Sobel magnitude of
   * the gray image in [0, 1], as scikit-image `feature.canny`.
   */
  readonly thresholdMode?: 'quantile' | 'absolute';
  /** Lower hysteresis threshold. Default 0.85 as a quantile, 0.1 as an absolute magnitude. */
  readonly lowThreshold?: number;
  /** Upper hysteresis threshold. Default 0.95 as a quantile, 0.2 as an absolute magnitude. */
  readonly highThreshold?: number;
  /** How `cannyGpu` and `houghLinesGpu` finish the hysteresis. `'cpu'` (default) reads the suppressed map back and flood-fills from the strong pixels. `'gpu'` grows the strong pixels with passes until none changes; it is slower on every tier measured (`docs/gpu-features.md`). */
  readonly hysteresis?: 'gpu' | 'cpu';
}

/** Options of `houghLines` and `houghLinesGpu`. */
export interface HoughLinesOptions extends CannyOptions {
  /** Number of angles in [0, pi). The angle step is `pi / thetaBins`. Default 180. */
  readonly thetaBins?: number;
  /** Fewest votes of a line. Default `ceil(min(width, height) / 6)`. */
  readonly threshold?: number;
  /** A line is kept when no cell within this many rho bins (pixels) has more votes. Default 9, as scikit-image. */
  readonly minDistance?: number;
  /** The same for the angle, in angle bins. Default 10, as scikit-image. */
  readonly minAngle?: number;
  /** At most this many lines, the most voted first. Default 256. */
  readonly maxLines?: number;
  /** Cut each line into segments (`HoughLinesP`-style). Default true. */
  readonly segments?: boolean;
  /** Shortest segment, in pixels between its end points. Default 30. */
  readonly minLineLength?: number;
  /** Longest run of steps without an edge pixel that a segment bridges. Default 3. */
  readonly maxLineGap?: number;
}

export interface CannyResult {
  readonly width: number;
  readonly height: number;
  /** 1 on an edge pixel, row-major. */
  readonly edges: Uint8Array;
  /** The number of edge pixels. */
  readonly count: number;
  /** The absolute thresholds applied (the quantiles of `thresholdMode: 'quantile'` resolved). */
  readonly low: number;
  readonly high: number;
  /** The passes the GPU hysteresis ran (0 for the CPU reference and `hysteresis: 'cpu'`). */
  readonly hysteresisPasses: number;
}

/** A line `x cos(theta) + y sin(theta) = rho`, theta in [0, pi), rho in pixels. */
export interface HoughLine {
  readonly rho: number;
  readonly theta: number;
  readonly votes: number;
}

export interface HoughSegment {
  readonly x1: number;
  readonly y1: number;
  readonly x2: number;
  readonly y2: number;
  /** Index into `lines`. */
  readonly line: number;
}

export interface HoughLinesResult extends CannyResult {
  readonly thetaBins: number;
  readonly rhoCount: number;
  /** The accumulator cell of (theta bin `t`, rho `r`) is `t * rhoCount + r + maxRho`. */
  readonly maxRho: number;
  readonly accumulator: Uint32Array;
  readonly lines: readonly HoughLine[];
  readonly segments: readonly HoughSegment[];
}

const LANES = 64;
// WebGPU runs at most 65535 workgroups in one dispatch.
const MAX_LANES_PER_DISPATCH = 65535 * LANES;
const MAX_SEGMENTS = 16;
const SEGMENT_STRIDE = 1 + 4 * MAX_SEGMENTS;
const TIE = 1e-5;
const TAN_22_5 = 0.41421356237309503;
const HYSTERESIS_ROUND = 8;
/** Pixels with a gradient magnitude at or below this (a quarter of a gray level) count as flat. */
export const GRADIENT_FLOOR = 1e-3;
const QUANTILE_LOW = 0.85;
const QUANTILE_HIGH = 0.95;
/** Thresholds that nothing reaches: the result when no pixel has a gradient. */
const NO_EDGE = 1e9;

export interface HoughPlan {
  readonly width: number;
  readonly height: number;
  readonly sigma: number;
  readonly radius: number;
  readonly thresholdMode: 'quantile' | 'absolute';
  /** The option value: a quantile or an absolute magnitude, by `thresholdMode`. */
  readonly low: number;
  readonly high: number;
  readonly thetaBins: number;
  readonly maxRho: number;
  readonly rhoCount: number;
  readonly threshold: number;
  readonly minDistance: number;
  readonly minAngle: number;
  readonly maxLines: number;
  readonly segments: boolean;
  readonly minLengthSquared: number;
  readonly maxGap: number;
  readonly hysteresis: 'gpu' | 'cpu';
}

export function planHough(options: HoughLinesOptions, width: number, height: number): HoughPlan {
  const mode = options.thresholdMode ?? 'quantile',
    sigma = options.sigma ?? 1,
    low = options.lowThreshold ?? (mode === 'quantile' ? QUANTILE_LOW : 0.1),
    high = options.highThreshold ?? (mode === 'quantile' ? QUANTILE_HIGH : 0.2),
    thetaBins = options.thetaBins ?? 180,
    threshold = options.threshold ?? Math.ceil(Math.min(width, height) / 6),
    minDistance = options.minDistance ?? 9,
    minAngle = options.minAngle ?? 10,
    maxLines = options.maxLines ?? 256,
    minLength = options.minLineLength ?? 30,
    maxGap = options.maxLineGap ?? 3;
  if (
    !Number.isSafeInteger(width) ||
    !Number.isSafeInteger(height) ||
    width < 1 ||
    height < 1 ||
    width + height > 32000 ||
    width * height > 20_000_000 ||
    !(sigma > 0 && sigma <= 16) ||
    !(low >= 0 && Number.isFinite(low)) ||
    !(high >= low && Number.isFinite(high)) ||
    (mode !== 'quantile' && mode !== 'absolute') ||
    (mode === 'quantile' && high > 1) ||
    !Number.isSafeInteger(thetaBins) ||
    thetaBins < 2 ||
    thetaBins > 3600 ||
    !Number.isSafeInteger(threshold) ||
    threshold < 1 ||
    !Number.isSafeInteger(minDistance) ||
    minDistance < 0 ||
    minDistance > 1000 ||
    !Number.isSafeInteger(minAngle) ||
    minAngle < 0 ||
    2 * minAngle >= thetaBins ||
    !Number.isSafeInteger(maxLines) ||
    maxLines < 1 ||
    !(minLength >= 0 && Number.isFinite(minLength)) ||
    !Number.isSafeInteger(maxGap) ||
    maxGap < 0 ||
    (options.segments !== undefined && typeof options.segments !== 'boolean') ||
    (options.hysteresis !== undefined &&
      options.hysteresis !== 'gpu' &&
      options.hysteresis !== 'cpu')
  )
    throw new RangeError('Invalid Hough options');
  const maxRho = Math.ceil(Math.hypot(width, height));
  return {
    width,
    height,
    sigma,
    radius: Math.ceil(4 * sigma),
    thresholdMode: mode,
    low,
    high,
    thetaBins,
    maxRho,
    rhoCount: 2 * maxRho + 1,
    threshold,
    minDistance,
    minAngle,
    maxLines,
    segments: options.segments ?? true,
    minLengthSquared: Math.ceil(minLength) ** 2,
    maxGap,
    hysteresis: options.hysteresis ?? 'cpu',
  };
}

const validRgba = (rgba: ArrayLike<number>, width: number, height: number) =>
  Number.isSafeInteger(width) &&
  Number.isSafeInteger(height) &&
  width >= 1 &&
  height >= 1 &&
  rgba.length === width * height * 4;

/** Gaussian weights `exp(-k^2 / 2 sigma^2)` for `k = -radius .. radius`, not normalized. */
function gaussianWeights(sigma: number, radius: number): Float32Array {
  const weights = new Float32Array(2 * radius + 1);
  for (let k = -radius; k <= radius; k++)
    weights[k + radius] = Math.exp(-(k * k) / (2 * sigma * sigma));
  return weights;
}

/** Cos and sin of each angle bin scaled by 65536 and rounded, as integer-valued f32: `[cos0, sin0, cos1, sin1, ...]`. */
export function houghTrigTable(thetaBins: number): Float32Array {
  const table = new Float32Array(2 * thetaBins);
  for (let t = 0; t < thetaBins; t++) {
    table[2 * t] = Math.round(Math.cos((Math.PI * t) / thetaBins) * 65536);
    table[2 * t + 1] = Math.round(Math.sin((Math.PI * t) / thetaBins) * 65536);
  }
  return table;
}

/** The pass of `chunkedPass` over `total` invocations, split so that no dispatch exceeds the workgroup limit. */
function* cappedPass(
  device: ShadeDevice,
  key: string,
  total: number,
  dispatch: (range: { first: number; end: number }, groups: number) => void,
) {
  for (let base = 0; base < total; base += MAX_LANES_PER_DISPATCH)
    yield* chunkedPass(
      device,
      key,
      key,
      Math.min(MAX_LANES_PER_DISPATCH, total - base),
      ({ first, end }, groups) => dispatch({ first: base + first, end: base + end }, groups),
    );
}

/**
 * The absolute thresholds of `plan` for the gradient magnitudes `magnitude`. In `'quantile'` mode
 * they are the nearest-rank quantiles of the magnitudes above `GRADIENT_FLOOR`, taken on f32
 * values so that the device and the CPU reference agree.
 */
export function resolveThresholds(
  plan: HoughPlan,
  magnitude: ArrayLike<number>,
): { low: number; high: number } {
  if (plan.thresholdMode === 'absolute') return { low: plan.low, high: plan.high };
  const kept: number[] = [];
  for (let i = 0; i < magnitude.length; i++) {
    const m = Math.fround(magnitude[i]!);
    if (m > GRADIENT_FLOOR) kept.push(m);
  }
  if (kept.length === 0) return { low: NO_EDGE, high: NO_EDGE };
  const sorted = Float32Array.from(kept).sort(),
    at = (q: number) => sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))]!;
  return { low: at(plan.low), high: at(plan.high) };
}

interface CannyRun {
  /** The state array: 0 none, 1 weak, 2 edge. The caller recycles it. */
  readonly state: Resident<Uint32Array>;
  readonly passes: number;
  readonly low: number;
  readonly high: number;
}

async function cannyStages(
  device: ShadeDevice,
  rgba: Uint8Array | Uint8ClampedArray,
  plan: HoughPlan,
): Promise<CannyRun> {
  const { width, height } = plan,
    pixels = width * height,
    base = {
      width,
      height,
      axis: 0,
      radius: plan.radius,
      low: plan.low,
      high: plan.high,
      thetaBins: plan.thetaBins,
      rhoCount: plan.rhoCount,
      maxRho: plan.maxRho,
      threshold: plan.threshold,
      windowRho: plan.minDistance,
      windowTheta: plan.minAngle,
      minLengthSquared: plan.minLengthSquared,
      maxGap: plan.maxGap,
    };
  const input = device.allocateU32(pixels, packRgba(rgba, pixels)),
    weights = device.allocate(2 * plan.radius + 1, gaussianWeights(plan.sigma, plan.radius)),
    gray = device.allocate(pixels),
    tmp = device.allocate(pixels),
    blurred = device.allocate(pixels),
    grad = device.allocate(3 * pixels);
  let state = device.allocateU32(pixels),
    passes = 0,
    ok = false;
  try {
    function* steps() {
      yield* cappedPass(device, 'houghGray', pixels, (houghRange, groups) =>
        houghToGray({ houghRange, houghRgba: input, houghGray: gray }, groups),
      );
      for (const [axis, src, dst] of [
        [0, gray, tmp],
        [1, tmp, blurred],
      ] as const)
        yield* cappedPass(device, 'houghBlur', pixels, (houghRange, groups) =>
          houghBlur(
            {
              hough: { ...base, axis },
              houghRange,
              houghWeights: weights,
              houghSrc: src,
              houghDst: dst,
            },
            groups,
          ),
        );
      yield* cappedPass(device, 'houghGradient', pixels, (houghRange, groups) =>
        houghGradient({ hough: base, houghRange, houghSrc: blurred, houghGrad: grad }, groups),
      );
    }
    await device.planSteps(steps());
    let { low, high } = plan;
    if (plan.thresholdMode === 'quantile') {
      const gradient = await device.read(grad, 3 * pixels),
        magnitude = new Float32Array(pixels);
      for (let i = 0; i < pixels; i++) magnitude[i] = gradient[3 * i + 2]!;
      ({ low, high } = resolveThresholds(plan, magnitude));
    }
    const suppressBase = { ...base, low, high };
    await device.planSteps(
      cappedPass(device, 'houghSuppress', pixels, (houghRange, groups) =>
        houghSuppress(
          { hough: suppressBase, houghRange, houghGradRead: grad, houghState: state },
          groups,
        ),
      ),
    );
    if (plan.hysteresis === 'cpu') {
      const resolved = hysteresisCpu(await device.read(state, pixels), width, height);
      device.recycle(state);
      state = device.allocateU32(pixels, resolved);
    } else {
      for (;;) {
        const changed = device.allocateU32(1, new Uint32Array(1));
        try {
          function* round() {
            for (let pass = 0; pass < HYSTERESIS_ROUND; pass++)
              yield* cappedPass(device, 'houghGrow', pixels, (houghRange, groups) =>
                houghGrow(
                  { hough: base, houghRange, houghState: state, houghChanged: changed },
                  groups,
                ),
              );
          }
          await device.planSteps(round());
          passes += HYSTERESIS_ROUND;
          if ((await device.read(changed, 1))[0] === 0) break;
        } finally {
          device.recycle(changed);
        }
      }
    }
    ok = true;
    return { state, passes, low, high };
  } finally {
    for (const buffer of [input, weights, gray, tmp, blurred, grad]) device.recycle(buffer);
    if (!ok) device.recycle(state);
  }
}

const edgeMap = (state: Uint32Array): { edges: Uint8Array; count: number } => {
  const edges = new Uint8Array(state.length);
  let count = 0;
  for (let i = 0; i < state.length; i++) if (state[i] === 2) ((edges[i] = 1), count++);
  return { edges, count };
};

/** Canny edges on the device. `rgba` is 8-bit. */
export function cannyGpu(
  device: ShadeDevice,
  rgba: Uint8Array | Uint8ClampedArray,
  width: number,
  height: number,
  options: CannyOptions = {},
): Promise<CannyResult> {
  if (!validRgba(rgba, width, height)) throw new RangeError('Invalid RGBA image for Canny');
  const plan = planHough(options, width, height);
  return device.enqueue(async () => {
    const run = await cannyStages(device, rgba, plan);
    try {
      const { edges, count } = edgeMap(await device.read(run.state, width * height));
      return {
        width,
        height,
        edges,
        count,
        low: run.low,
        high: run.high,
        hysteresisPasses: run.passes,
      };
    } finally {
      device.recycle(run.state);
    }
  });
}

/** Hough lines on the device: Canny, the vote, the peaks and, unless `segments` is false, the segments. */
export function houghLinesGpu(
  device: ShadeDevice,
  rgba: Uint8Array | Uint8ClampedArray,
  width: number,
  height: number,
  options: HoughLinesOptions = {},
): Promise<HoughLinesResult> {
  if (!validRgba(rgba, width, height)) throw new RangeError('Invalid RGBA image for Hough lines');
  const plan = planHough(options, width, height),
    pixels = width * height,
    cells = plan.thetaBins * plan.rhoCount;
  return device.enqueue(async () => {
    const run = await cannyStages(device, rgba, plan),
      owned: Resident<Float32Array | Uint32Array>[] = [run.state];
    try {
      const base = {
        width,
        height,
        axis: 0,
        radius: plan.radius,
        low: plan.low,
        high: plan.high,
        thetaBins: plan.thetaBins,
        rhoCount: plan.rhoCount,
        maxRho: plan.maxRho,
        threshold: plan.threshold,
        windowRho: plan.minDistance,
        windowTheta: plan.minAngle,
        minLengthSquared: plan.minLengthSquared,
        maxGap: plan.maxGap,
      };
      const state = run.state,
        counter = device.allocateU32(1, new Uint32Array(1)),
        list = device.allocateU32(pixels),
        trig = device.allocate(2 * plan.thetaBins, houghTrigTable(plan.thetaBins)),
        acc = device.allocateU32(cells, new Uint32Array(cells)),
        peakMap = device.allocateU32(cells);
      owned.push(counter, list, trig, acc, peakMap);
      await device.planSteps(
        cappedPass(device, 'houghCollect', pixels, (houghRange, groups) =>
          houghCollect(
            { houghRange, houghState: state, houghChanged: counter, houghList: list },
            groups,
          ),
        ),
      );
      const edgeCount = (await device.read(counter, 1))[0]!,
        { edges } = edgeMap(await device.read(state, pixels));
      if (edgeCount > 0)
        await device.planSteps(
          cappedPass(device, 'houghVote', edgeCount * plan.thetaBins, (houghRange, groups) =>
            houghVote(
              {
                hough: base,
                houghRange,
                houghListRead: list,
                houghTrig: trig,
                houghAcc: acc,
              },
              groups,
            ),
          ),
        );
      await device.planSteps(
        cappedPass(device, 'houghPeaks', cells, (houghRange, groups) =>
          houghPeaks({ hough: base, houghRange, houghAccRead: acc, houghPeakMap: peakMap }, groups),
        ),
      );
      const accumulator = await device.read(acc, cells),
        lines = pickPeaks(await device.read(peakMap, cells), plan);
      let segments: HoughSegment[] = [];
      if (plan.segments && lines.length > 0) {
        const lineCells = new Uint32Array(2 * lines.length);
        lines.forEach((line, i) => {
          lineCells[2 * i] = Math.round((line.theta * plan.thetaBins) / Math.PI);
          lineCells[2 * i + 1] = line.rho + plan.maxRho;
        });
        const lineBuffer = device.allocateU32(lineCells.length, lineCells),
          out = device.allocateU32(lines.length * SEGMENT_STRIDE);
        owned.push(lineBuffer, out);
        await device.planSteps(
          cappedPass(device, 'houghSegment', lines.length, (houghRange, groups) =>
            houghSegment(
              {
                hough: base,
                houghRange,
                houghState: state,
                houghTrig: trig,
                houghLines: lineBuffer,
                houghSegments: out,
              },
              groups,
            ),
          ),
        );
        segments = unpackSegments(
          await device.read(out, lines.length * SEGMENT_STRIDE),
          lines.length,
        );
      }
      return {
        width,
        height,
        edges,
        count: edgeCount,
        low: run.low,
        high: run.high,
        hysteresisPasses: run.passes,
        thetaBins: plan.thetaBins,
        rhoCount: plan.rhoCount,
        maxRho: plan.maxRho,
        accumulator,
        lines,
        segments,
      };
    } finally {
      for (const buffer of owned) device.recycle(buffer);
    }
  });
}

/** Cells that are peaks, the most voted first (the lower cell wins a tie), as lines. */
function pickPeaks(peakMap: Uint32Array, plan: HoughPlan): HoughLine[] {
  const found: number[] = [];
  for (let cell = 0; cell < peakMap.length; cell++) if (peakMap[cell]! > 0) found.push(cell);
  found.sort((a, b) => peakMap[b]! - peakMap[a]! || a - b);
  return found.slice(0, plan.maxLines).map((cell) => {
    const t = Math.floor(cell / plan.rhoCount);
    return {
      rho: (cell % plan.rhoCount) - plan.maxRho,
      theta: (Math.PI * t) / plan.thetaBins,
      votes: peakMap[cell]!,
    };
  });
}

function unpackSegments(raw: Uint32Array, lineCount: number): HoughSegment[] {
  const segments: HoughSegment[] = [];
  for (let line = 0; line < lineCount; line++) {
    const base = line * SEGMENT_STRIDE;
    for (let s = 0; s < raw[base]!; s++)
      segments.push({
        x1: raw[base + 1 + 4 * s]!,
        y1: raw[base + 2 + 4 * s]!,
        x2: raw[base + 3 + 4 * s]!,
        y2: raw[base + 4 + 4 * s]!,
        line,
      });
  }
  return segments;
}

/** Hysteresis by flood fill: the weak pixels (1) 8-connected to a strong pixel (2) become edges. */
function hysteresisCpu(state: ArrayLike<number>, width: number, height: number): Uint32Array {
  const out = new Uint32Array(state.length),
    stack = new Int32Array(state.length);
  let top = 0;
  for (let i = 0; i < state.length; i++)
    if (state[i] === 2) {
      out[i] = 2;
      stack[top++] = i;
    }
  while (top > 0) {
    const p = stack[--top]!,
      x = p % width,
      y = (p - x) / width;
    for (let dy = -1; dy <= 1; dy++)
      for (let dx = -1; dx <= 1; dx++) {
        const nx = x + dx,
          ny = y + dy;
        if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
        const q = ny * width + nx;
        if (state[q] === 1 && out[q] !== 2) {
          out[q] = 2;
          stack[top++] = q;
        }
      }
  }
  return out;
}

/** The CPU reference of the Canny stages: the state array (0 none, 1 weak, 2 strong) before hysteresis. */
function cannyStateCpu(
  rgba: ArrayLike<number>,
  plan: HoughPlan,
): { state: Uint32Array; low: number; high: number } {
  const { width, height, radius } = plan,
    pixels = width * height,
    weights = gaussianWeights(plan.sigma, radius),
    gray = new Float64Array(pixels);
  for (let i = 0; i < pixels; i++)
    gray[i] = (0.2125 * rgba[4 * i]! + 0.7154 * rgba[4 * i + 1]! + 0.0721 * rgba[4 * i + 2]!) / 255;
  const blur = (src: Float64Array, axis: number) => {
    const dst = new Float64Array(pixels);
    for (let y = 0; y < height; y++)
      for (let x = 0; x < width; x++) {
        let sum = 0,
          total = 0;
        for (let k = -radius; k <= radius; k++) {
          const nx = x + (axis === 0 ? k : 0),
            ny = y + (axis === 0 ? 0 : k);
          if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
          sum += weights[k + radius]! * src[ny * width + nx]!;
          total += weights[k + radius]!;
        }
        dst[y * width + x] = sum / total;
      }
    return dst;
  };
  const smooth = blur(blur(gray, 0), 1),
    at = (x: number, y: number) =>
      smooth[Math.min(height - 1, Math.max(0, y)) * width + Math.min(width - 1, Math.max(0, x))]!,
    grad = new Float64Array(3 * pixels);
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      const gx =
          (at(x + 1, y - 1) +
            2 * at(x + 1, y) +
            at(x + 1, y + 1) -
            at(x - 1, y - 1) -
            2 * at(x - 1, y) -
            at(x - 1, y + 1)) /
          4,
        gy =
          (at(x - 1, y + 1) +
            2 * at(x, y + 1) +
            at(x + 1, y + 1) -
            at(x - 1, y - 1) -
            2 * at(x, y - 1) -
            at(x + 1, y - 1)) /
          4,
        i = y * width + x;
      grad[3 * i] = gx;
      grad[3 * i + 1] = gy;
      grad[3 * i + 2] = Math.hypot(gx, gy) * Math.SQRT1_2;
    }
  const magnitude = new Float64Array(pixels);
  for (let i = 0; i < pixels; i++) magnitude[i] = grad[3 * i + 2]!;
  const { low, high } = resolveThresholds(plan, magnitude),
    state = new Uint32Array(pixels);
  for (let y = 1; y < height - 1; y++)
    for (let x = 1; x < width - 1; x++) {
      const i = y * width + x,
        gx = grad[3 * i]!,
        gy = grad[3 * i + 1]!,
        m = grad[3 * i + 2]!,
        ax = Math.abs(gx),
        ay = Math.abs(gy);
      let dx = 1,
        dy = 0;
      if (ay <= TAN_22_5 * ax) ((dx = 1), (dy = 0));
      else if (ax <= TAN_22_5 * ay) ((dx = 0), (dy = 1));
      else if (gx * gy > 0) ((dx = 1), (dy = 1));
      else ((dx = 1), (dy = -1));
      if (
        m >= low &&
        m > grad[3 * ((y - dy) * width + x - dx) + 2]! + TIE &&
        m >= grad[3 * ((y + dy) * width + x + dx) + 2]! - TIE
      )
        state[i] = m >= high ? 2 : 1;
    }
  return { state, low, high };
}

/** Canny edges, the CPU reference of `cannyGpu`. */
export function canny(
  rgba: ArrayLike<number>,
  width: number,
  height: number,
  options: CannyOptions = {},
): CannyResult {
  if (!validRgba(rgba, width, height)) throw new RangeError('Invalid RGBA image for Canny');
  const plan = planHough(options, width, height),
    { state, low, high } = cannyStateCpu(rgba, plan),
    { edges, count } = edgeMap(hysteresisCpu(state, width, height));
  return { width, height, edges, count, low, high, hysteresisPasses: 0 };
}

/** The votes of the edge pixels of `edges`: the same integer arithmetic as `houghVote`. */
function accumulateCpu(edges: Uint8Array, plan: HoughPlan): Uint32Array {
  const { width, thetaBins, rhoCount, maxRho } = plan,
    trig = houghTrigTable(thetaBins),
    acc = new Uint32Array(thetaBins * rhoCount);
  for (let p = 0; p < edges.length; p++) {
    if (!edges[p]) continue;
    const x = p % width,
      y = (p - x) / width;
    for (let t = 0; t < thetaBins; t++)
      acc[t * rhoCount + ((x * trig[2 * t]! + y * trig[2 * t + 1]! + 32768) >> 16) + maxRho]++;
  }
  return acc;
}

/** The CPU form of `houghPeaks`. */
function peakMapCpu(acc: Uint32Array, plan: HoughPlan): Uint32Array {
  const { thetaBins: bins, rhoCount: count } = plan,
    out = new Uint32Array(acc.length);
  for (let cell = 0; cell < acc.length; cell++) {
    const votes = acc[cell]!;
    if (votes < plan.threshold) continue;
    const t = Math.floor(cell / count),
      r = cell % count;
    let isPeak = true;
    for (let dt = -plan.minAngle; dt <= plan.minAngle && isPeak; dt++)
      for (let dr = -plan.minDistance; dr <= plan.minDistance; dr++) {
        let t2 = t + dt,
          r2 = r + dr;
        if (t2 < 0) ((t2 += bins), (r2 = count - 1 - r2));
        else if (t2 >= bins) ((t2 -= bins), (r2 = count - 1 - r2));
        if (r2 < 0 || r2 >= count) continue;
        const other = t2 * count + r2,
          v = acc[other]!;
        if (v > votes || (v === votes && other < cell)) {
          isPeak = false;
          break;
        }
      }
    if (isPeak) out[cell] = votes;
  }
  return out;
}

/** The CPU form of `houghSegment` for every line. */
function segmentsCpu(
  edges: Uint8Array,
  lines: readonly HoughLine[],
  plan: HoughPlan,
): HoughSegment[] {
  const { width, height, thetaBins } = plan,
    trig = houghTrigTable(thetaBins),
    segments: HoughSegment[] = [];
  lines.forEach((line, index) => {
    const t = Math.round((line.theta * thetaBins) / Math.PI),
      c = trig[2 * t]!,
      s = trig[2 * t + 1]!,
      horizontal = s >= Math.abs(c),
      steps = horizontal ? width : height,
      across = horizontal ? height : width,
      other = horizontal ? c : s,
      divisor = horizontal ? s : c,
      flip = divisor < 0,
      den = flip ? -divisor : divisor;
    let kept = 0,
      start = -1,
      startM = 0,
      last = -1,
      lastM = 0;
    const emit = (k0: number, m0: number, k1: number, m1: number) => {
      const dk = k1 - k0,
        dm = m1 - m0;
      if (dk * dk + dm * dm < plan.minLengthSquared || kept >= MAX_SEGMENTS) return;
      kept++;
      segments.push(
        horizontal
          ? { x1: k0, y1: m0, x2: k1, y2: m1, line: index }
          : { x1: m0, y1: k0, x2: m1, y2: k1, line: index },
      );
    };
    for (let k = 0; k < steps; k++) {
      const raw = line.rho * 65536 - k * other,
        num = flip ? -raw : raw,
        m = Math.floor((2 * num + den) / (2 * den));
      let supported = false;
      for (let d = -1; d <= 1; d++) {
        const mm = m + d;
        if (mm >= 0 && mm < across && m >= 0 && m < across) {
          const px = horizontal ? k : mm,
            py = horizontal ? mm : k;
          if (edges[py * width + px]) supported = true;
        }
      }
      if (supported) {
        if (start < 0) ((start = k), (startM = m));
        last = k;
        lastM = m;
      } else if (start >= 0 && k - last > plan.maxGap) {
        emit(start, startM, last, lastM);
        start = -1;
      }
    }
    if (start >= 0) emit(start, startM, last, lastM);
  });
  return segments;
}

/**
 * Hough lines, the CPU reference of `houghLinesGpu`: Canny in double precision, then the same
 * integer vote, peaks and segments.
 */
export function houghLines(
  rgba: ArrayLike<number>,
  width: number,
  height: number,
  options: HoughLinesOptions = {},
): HoughLinesResult {
  if (!validRgba(rgba, width, height)) throw new RangeError('Invalid RGBA image for Hough lines');
  const plan = planHough(options, width, height),
    { state, low, high } = cannyStateCpu(rgba, plan),
    { edges, count } = edgeMap(hysteresisCpu(state, width, height)),
    accumulator = accumulateCpu(edges, plan),
    lines = pickPeaks(peakMapCpu(accumulator, plan), plan);
  return {
    width,
    height,
    edges,
    count,
    low,
    high,
    hysteresisPasses: 0,
    thetaBins: plan.thetaBins,
    rhoCount: plan.rhoCount,
    maxRho: plan.maxRho,
    accumulator,
    lines,
    segments: plan.segments ? segmentsCpu(edges, lines, plan) : [],
  };
}
