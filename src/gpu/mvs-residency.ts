/**
 * The residency plan of the dense MVS (`docs/dense-mvs.md`, Amendment 3 in section 8, and 4.2):
 * the arrays each stage keeps on the device, sized from the device's budget, so that the dense
 * stage runs within any device budget. Size changes speed, not success.
 *
 * - The depth stage keeps a window: the reference's state (`mvsGeoDepth`, `mvsNormal`, `mvsCost`,
 *   and in the photometric pass the reference's `mvsDepth`), and `slots` view slots of `mvsGray`
 *   (and of `mvsDepth` in the geometric pass) for the reference and its sources. A view's
 *   finished outputs go back to the host and come up again when a later reference needs them.
 * - `window` mode holds the reference and all its sources at once and runs the kernels of steps
 *   2 to 4. `streamed` mode holds the reference and `slots − 1` sources at a time, streams the
 *   sources through in groups and keeps each hypothesis's least source costs in `mvsPartial`
 *   (`elements` elements of 9 hypotheses × `bestSources` costs) between the groups.
 * - The fusion keeps the reference and its fusion views (3.10, `fusionViews`) in `window` mode,
 *   or the reference and `slots − 1` of them at a time with per-pixel sums in `mvsFuseSums` for a
 *   chunk of `chunk` pixels in `streamed` mode.
 *
 * Every byte count here is the bytes of the arrays the stage allocates, exactly.
 */
import { MVS_CAMERA_FLOATS, MVS_SOURCE_SLOTS } from './mvs-views.ts';

/** Hypotheses per element of `mvsPartial`: the 8 propagation candidates and the random one. */
export const MVS_PARTIAL_HYPOTHESES = 9;
/** f32 per pixel of `mvsFuseSums`: the point sum (3), the normal sum (3), the count, the lower flag. */
export const MVS_FUSE_SUM_FLOATS = 8;
/** The fewest elements of a streamed chunk: one workgroup. */
const LANES = 64;
/** `scanBlockCount` of `scan.ts`, which this host module does not import with its kernels. */
const scanBlockCount = (count: number) => Math.max(1, Math.ceil(count / LANES));

export type MvsResidencyMode = 'window' | 'streamed';

export interface MvsDepthResidency {
  readonly mode: MvsResidencyMode;
  /** View slots of `mvsGray` (and of `mvsDepth` in the geometric pass): the reference and sources. */
  readonly slots: number;
  /** Elements of `mvsPartial` (9 hypotheses × `bestSources` costs each), 0 in window mode. */
  readonly elements: number;
  /** The bytes of the stage's arrays. */
  readonly bytes: number;
  /** The largest array, bytes. */
  readonly largestArray: number;
}

export interface MvsFusionResidency {
  readonly mode: MvsResidencyMode;
  /** View slots of `mvsGeoDepth` and `mvsNormal`: the reference and its fusion views. */
  readonly slots: number;
  /** Pixels of one chunk: P in window mode. */
  readonly chunk: number;
  readonly bytes: number;
  readonly largestArray: number;
}

export interface MvsResidency {
  /** The bytes the dense stage may keep on the device. */
  readonly budgetBytes: number;
  /** The largest array the device binds. */
  readonly bindingBytes: number;
  readonly photometric: MvsDepthResidency;
  readonly geometric: MvsDepthResidency;
  readonly fusion: MvsFusionResidency;
  /** The largest of the three stages, bytes. The stages run one after another. */
  readonly peakBytes: number;
}

export interface MvsResidencyInput {
  readonly width: number;
  readonly height: number;
  /** The most sources of one reference (K, at most 6). */
  readonly maxSources: number;
  /** The most views of one fusion window, the reference included. */
  readonly fusionSlots: number;
  /** The f32 of `mvsSeeds`: 3 × the most seeds of one reference, at least 3. */
  readonly seedFloats: number;
  /** The least source costs kept per hypothesis in streamed mode: `bestSources` (3.7). */
  readonly bestSources: number;
  readonly budgetBytes: number;
  readonly bindingBytes: number;
  /**
   * @internal Force streamed mode with at most this many sources (or fusion views) per group, a
   * test of the grouped paths. Default: window mode when it fits.
   */
  readonly streamGroup?: number;
}

/** The arrays of one depth pass, in elements of 4 bytes, by name. */
export function mvsDepthArrays(
  input: Pick<MvsResidencyInput, 'width' | 'height' | 'seedFloats' | 'bestSources'>,
  geometric: boolean,
  slots: number,
  elements: number,
): Record<string, number> {
  const P = input.width * input.height;
  return {
    mvsCameras: MVS_CAMERA_FLOATS * slots,
    mvsSources: MVS_SOURCE_SLOTS,
    // The geometric pass binds a seed array of one seed, which its kernels never read.
    mvsSeeds: geometric ? 3 : Math.max(3, input.seedFloats),
    mvsGray: slots * P,
    mvsDepth: geometric ? slots * P : P,
    mvsGeoDepth: P,
    mvsNormal: P,
    mvsCost: P,
    // The upload of a view: its packed RGBA and its gray image, or a depth map to copy.
    stageU32: P,
    stageF32: P,
    ...(elements > 0 ? { mvsPartial: elements * MVS_PARTIAL_HYPOTHESES * input.bestSources } : {}),
  };
}

/** The arrays of the fusion of one reference, in elements of 4 bytes, by name. */
export function mvsFusionArrays(
  input: Pick<MvsResidencyInput, 'width' | 'height'>,
  slots: number,
  chunk: number,
  streamed: boolean,
): Record<string, number> {
  const P = input.width * input.height;
  const flags = streamed ? scanBlockCount(chunk) * LANES : P;
  return {
    mvsCameras: MVS_CAMERA_FLOATS * slots,
    mvsSources: slots,
    mvsGeoDepth: slots * P,
    mvsNormal: slots * P,
    mvsRgba: P,
    stageU32: P,
    stageF32: P,
    ...(streamed ? { mvsFuseSums: MVS_FUSE_SUM_FLOATS * chunk } : {}),
    mvsFlags: flags,
    mvsFlagOffsets: 2 * scanBlockCount(flags),
    totals: 2,
    mvsPoints: 3 * chunk,
    mvsPointData: 2 * chunk,
  };
}

const sum = (arrays: Record<string, number>) =>
  Object.values(arrays).reduce((n, count) => n + 4 * count, 0);
const largest = (arrays: Record<string, number>) => 4 * Math.max(...Object.values(arrays));

/** The largest multiple of 64 at most `limit`, or `whole` when `limit` reaches it. */
function chunkOf(limit: number, whole: number): number {
  if (limit >= whole) return whole;
  return Math.max(0, limit - (limit % LANES));
}

function depthStage(
  input: MvsResidencyInput,
  geometric: boolean,
  oom: (stage: string, bytes: number) => Error,
): MvsDepthResidency {
  const K = Math.max(1, input.maxSources);
  const measure = (slots: number, elements: number) => {
    const arrays = mvsDepthArrays(input, geometric, slots, elements);
    return { bytes: sum(arrays), largestArray: largest(arrays) };
  };
  const fits = (m: { bytes: number; largestArray: number }) =>
    m.bytes <= input.budgetBytes && m.largestArray <= input.bindingBytes;
  const window = measure(1 + K, 0);
  if (input.streamGroup === undefined && fits(window))
    return { mode: 'window', slots: 1 + K, elements: 0, ...window };
  // Streamed: the fewest chunks of a propagation pass first, since each chunk loads every source
  // once, then the most sources per group, since each group is a kernel call per chunk.
  const whole = Math.ceil(input.width / 2) * input.height;
  const perElement = 4 * MVS_PARTIAL_HYPOTHESES * input.bestSources;
  let best: MvsDepthResidency | undefined;
  const top = Math.min(Math.max(1, K - 1), input.streamGroup ?? K);
  for (let G = top; G >= 1; G--) {
    const base = measure(1 + G, 0).bytes;
    const limit = Math.min(
      Math.floor((input.budgetBytes - base) / perElement),
      Math.floor(input.bindingBytes / perElement),
    );
    const elements = chunkOf(limit, whole);
    if (elements < Math.min(LANES, whole)) continue;
    const m = measure(1 + G, elements);
    if (!fits(m)) continue;
    const chunks = Math.ceil(whole / elements);
    if (!best || chunks < Math.ceil(whole / best.elements))
      best = { mode: 'streamed', slots: 1 + G, elements, ...m };
  }
  if (!best)
    throw oom(geometric ? 'geometric' : 'photometric', measure(2, Math.min(LANES, whole)).bytes);
  return best;
}

function fusionStage(
  input: MvsResidencyInput,
  oom: (stage: string, bytes: number) => Error,
): MvsFusionResidency {
  const P = input.width * input.height;
  const F = Math.max(1, input.fusionSlots);
  const measure = (slots: number, chunk: number, streamed: boolean) => {
    const arrays = mvsFusionArrays(input, slots, chunk, streamed);
    return { bytes: sum(arrays), largestArray: largest(arrays) };
  };
  const fits = (m: { bytes: number; largestArray: number }) =>
    m.bytes <= input.budgetBytes && m.largestArray <= input.bindingBytes;
  const window = measure(F, P, false);
  if (input.streamGroup === undefined && fits(window))
    return { mode: 'window', slots: F, chunk: P, ...window };
  // Streamed: the fewest chunks of pixels first, since each chunk loads every fusion view once,
  // then the most fusion views per group.
  let best: MvsFusionResidency | undefined;
  const top = Math.min(Math.max(1, F - 2), input.streamGroup ?? F);
  for (let g = top; g >= 1; g--) {
    // The largest chunk that fits, by bisection over blocks of 64 pixels.
    let low = 0,
      high = Math.ceil(P / LANES);
    while (low < high) {
      const mid = Math.ceil((low + high) / 2);
      if (fits(measure(1 + g, Math.min(P, mid * LANES), true))) low = mid;
      else high = mid - 1;
    }
    const chunk = Math.min(P, low * LANES);
    if (chunk === 0) continue;
    const chunks = Math.ceil(P / chunk);
    if (!best || chunks < Math.ceil(P / best.chunk))
      best = { mode: 'streamed', slots: 1 + g, chunk, ...measure(1 + g, chunk, true) };
  }
  if (!best) throw oom('fusion', measure(2, Math.min(LANES, P), true).bytes);
  return best;
}

/**
 * The residency of each stage of the dense MVS within `budgetBytes`, each array at most
 * `bindingBytes`. A stage takes window mode when it fits and streamed mode otherwise. The one
 * refusal is a budget below the floor of streamed mode: one source view and one chunk of 64
 * pixels, about 36 bytes per pixel of the image (Amendment 3).
 */
export function planMvsResidency(input: MvsResidencyInput): MvsResidency {
  for (const [name, value] of Object.entries(input))
    if (value !== undefined && (!Number.isSafeInteger(value) || value < 0))
      throw new RangeError(`planMvsResidency: ${name} must be a whole number`);
  const oom = (stage: string, bytes: number) =>
    new RangeError(
      `The dense MVS needs at least ${bytes} B for its ${stage} stage at ` +
        `${input.width} × ${input.height} (one source view at a time, 64 pixels a chunk), and ` +
        `the device's budget is ${input.budgetBytes} B with arrays of at most ${input.bindingBytes} B`,
    );
  const photometric = depthStage(input, false, oom),
    geometric = depthStage(input, true, oom),
    fusion = fusionStage(input, oom);
  return {
    budgetBytes: input.budgetBytes,
    bindingBytes: input.bindingBytes,
    photometric,
    geometric,
    fusion,
    peakBytes: Math.max(photometric.bytes, geometric.bytes, fusion.bytes),
  };
}
