import type { Resident } from 'typeshade/runtime';
import type { ShadeDevice, ShadePlanReport } from '../../device/gpu/device.ts';
import { kltCheck, kltPyrDown, kltScharr, kltTrack } from './klt.shade.ts';
import { readTogether } from '../../device/gpu/readback.ts';
import {
  kltInitialRecords,
  kltAffineRecords,
  kltLayout,
  kltSeedRecords,
  type KltLayout,
  type KltOptions,
} from './klt.ts';

export interface KltGpuOptions {
  /** Forward row-major 2x2 patch-offset warps, one per point. Optional; identity by default. */
  readonly affineWarps?: ArrayLike<number>;
  /** Skip the unused backward coordinate readback. `back` is empty when false. Default true. */
  readonly readBack?: boolean;
  /** Read checked records and NCC together; `forward` is empty and `ncc` is present. Default false. */
  readonly compactOutput?: boolean;
  /** Experimental: submit forward, backward and check together for each point chunk. Default false. */
  readonly batchPasses?: boolean;
  /** Absolute target x, y guesses. LK must still measure and validate the image patch. */
  readonly initialTargets?: ArrayLike<number>;
  /** The tracks of one chunk of `kltTrack` and `kltCheck`. Default: all in one chunk. */
  readonly chunkTracks?: number;
  /**
   * The fewest tracks in an adaptive chunk (default 128 in batched mode, 64 separately).
   * A larger minimum exposes more workgroups on software devices; its effect on wall time
   * depends on the workload and backend. Explicit chunkTracks still fixes the size.
   */
  readonly minChunkTracks?: number;
}

export interface KltGpuResult {
  /** (x, y, forward-backward error, status) for each point, as `checkKltReference`. */
  readonly records: Float32Array;
  /** The forward result (x, y, NCC, status), empty with `compactOutput:true`. */
  readonly forward: Float32Array;
  /** Per-track forward NCC in compact mode; shares the independent checked-result snapshot. */
  readonly ncc?: Float32Array;
  readonly back: Float32Array;
  /** The forward and backward pyramids as built on the device, for the tests. */
  readonly pyramids?: { gray: Float32Array[]; grad: Float32Array[] };
  readonly report: ShadePlanReport;
  /** Forward chunk sizes followed by backward sizes; batched mode executes each pair together. */
  readonly trackChunks: readonly number[];
}

export const KLT_LANES = 64;
const LANES = KLT_LANES;

export function kltControlOf(
  layout: KltLayout,
  width: number,
  height: number,
  options: KltOptions,
) {
  return {
    width,
    height,
    levels: layout.levels,
    halfWindow: options.halfWindow,
    maxIterations: options.maxIterations,
    epsilon: options.epsilonPx,
    minEig: options.minEigThreshold,
    fbThreshold: options.fbThresholdPx,
    nccThreshold: options.nccThreshold,
    photometric: options.photometric,
    initialGuess: 0,
    affineOffset: 0,
  };
}

/** Record the pyramid and the gradient of a frame into an open plan body. */
export function recordKltPyramid(
  layout: KltLayout,
  gray: Resident<Float32Array>,
  grad: Resident<Float32Array>,
): void {
  for (let l = 0; l + 1 < layout.levels; l++) {
    const level = {
      width: layout.widths[l + 1]!,
      height: layout.heights[l + 1]!,
      offset: layout.offsets[l + 1]!,
      srcWidth: layout.widths[l]!,
      srcHeight: layout.heights[l]!,
      srcOffset: layout.offsets[l]!,
    };
    kltPyrDown(
      { kltPyramid: gray, kltLevel: level },
      Math.ceil((level.width * level.height) / LANES),
    );
  }
  for (let l = 0; l < layout.levels; l++) {
    const level = {
      width: layout.widths[l]!,
      height: layout.heights[l]!,
      offset: layout.offsets[l]!,
      srcWidth: 0,
      srcHeight: 0,
      srcOffset: 0,
    };
    kltScharr(
      { kltGrayIn: gray, kltGradOut: grad, kltLevel: level },
      Math.ceil((level.width * level.height) / LANES),
    );
  }
}

export interface KltChunkPlan {
  /** The units to run. */
  readonly count: number;
  /** A forced chunk size; the device sizes each chunk from its measured rate when unset. */
  readonly fixed?: number;
  /** The chunk key of section 4.4; none runs the units as one chunk. */
  readonly key?: string;
  /** The fewest units of a chunk that the device sizes. */
  readonly min?: number;
}

type KltChunkRun = (range: { first: number; end: number }, groups: number) => void;

/**
 * The chunks of one pass over `count` units as plan steps. A forced size fixes the chunk. Otherwise
 * the device sizes each chunk from the rate it measured under `key` (section 4.4), and a chunk
 * before the last fills its workgroups. The warm step runs the dispatch over zero workgroups, so
 * the pipeline is made before the first timed chunk.
 */
export function* kltChunkSteps(
  device: ShadeDevice,
  plan: KltChunkPlan,
  run: KltChunkRun,
  onChunk?: (size: number) => void,
) {
  const { count, key } = plan,
    min = plan.min ?? LANES,
    fixed = plan.fixed ?? Math.max(count, 1),
    warm = key === undefined ? undefined : { id: key, run: () => run({ first: 0, end: 0 }, 0) };
  for (let first = 0; first < count;) {
    const remaining = count - first;
    let size = Math.min(remaining, fixed);
    if (plan.fixed === undefined && key !== undefined) {
      // A grouped roundtrip contains two iterative solves. Reserve another quarter of the
      // device's proposal for point-dependent work and rate changes between timed chunks.
      // Ask for capacity before clipping to remaining work, so a fitting tail stays in one chunk.
      const fill = device.backend === 'webgpu' && key === 'kltRoundtrip' ? 0.75 : 1;
      const capacity = device.chunkSize(key, Math.ceil(remaining / fill), Math.min(remaining, min));
      size = Math.min(remaining, Math.max(Math.min(remaining, min), Math.floor(capacity * fill)));
      if (size < remaining) size = Math.max(LANES, size - (size % LANES));
    }
    const range = { first, end: first + size };
    onChunk?.(size);
    const step = ['chunk', () => run(range, Math.ceil(size / LANES))] as const;
    yield key === undefined || warm === undefined
      ? step
      : ([...step, undefined, { key, units: size, warm }] as const);
    first += size;
  }
}

/** @internal Run the independent forward/backward/check chain in one plan step per point range. */
export function* kltRoundtripSteps(
  device: ShadeDevice,
  plan: Omit<KltChunkPlan, 'key'>,
  passes: readonly [KltChunkRun, KltChunkRun, KltChunkRun],
  trackChunks: number[],
) {
  const firstChunk = trackChunks.length;
  yield* kltChunkSteps(
    device,
    { ...plan, key: 'kltRoundtrip' },
    (range, groups) => {
      passes[0](range, groups);
      passes[1](range, groups);
      passes[2](range, groups);
    },
    (size) => trackChunks.push(size),
  );
  // Keep the existing logical forward-then-backward report despite grouped execution.
  const endChunk = trackChunks.length;
  for (let k = firstChunk; k < endChunk; k++) trackChunks.push(trackChunks[k]!);
}

/** @internal Resident image evidence; the owning caller keeps it alive through tracking. */
export interface KltResidentPyramid {
  readonly width: number;
  readonly height: number;
  readonly layout: KltLayout;
  readonly gray: Resident<Float32Array>;
  readonly grad: Resident<Float32Array>;
}

/** @internal Snapshot point inputs before entering the device's asynchronous job queue. */
export function prepareKltGpuPair(
  width: number,
  height: number,
  points: ArrayLike<number>,
  options: KltOptions,
  gpu: KltGpuOptions,
) {
  const layout = kltLayout(width, height, options.levels),
    n = points.length >> 1,
    chunk = gpu.chunkTracks ?? Math.max(n, 1);
  if (points.length % 2) throw new RangeError('points must hold x, y pairs');
  if (!(chunk >= 1)) throw new RangeError('chunkTracks must be at least 1');
  const initial = gpu.initialTargets && kltInitialRecords(gpu.initialTargets, n),
    affine = gpu.affineWarps && kltAffineRecords(gpu.affineWarps, n),
    seeds = kltSeedRecords(points),
    startData = affine ? new Float32Array(8 * n) : seeds,
    forwardData = affine ? new Float32Array(8 * n) : initial;
  if (affine) {
    startData.set(seeds);
    startData.set(affine.forward, 4 * n);
    if (initial) forwardData!.set(initial);
    forwardData!.set(affine.backward, 4 * n);
  }
  const compactOutput = gpu.compactOutput === true,
    checkedLength = (compactOutput ? 5 : 4) * n,
    scratchBytes = 4 * (startData.length + (forwardData?.length ?? 4 * n) + 4 * n + checkedLength),
    control = {
      ...kltControlOf(layout, width, height, options),
      initialGuess: initial ? 1 : 0,
      affineOffset: affine ? 4 * n : 0,
    };
  return {
    layout,
    n,
    startData,
    forwardData,
    backData: initial ? seeds : undefined,
    control,
    checkOutput: { nccOffset: compactOutput ? 4 * n : 0 },
    compactOutput,
    checkedLength,
    scratchBytes,
  };
}
export type KltGpuPreparedPair = ReturnType<typeof prepareKltGpuPair>;

/** @internal Zero-fill unused pyramid levels while uploading only retained source pixels. */
export function paddedKltGray(gray: ArrayLike<number>, layout: KltLayout): Float32Array {
  const values = new Float32Array(layout.total);
  for (let i = 0; i < layout.widths[0]! * layout.heights[0]!; i++) values[i] = gray[i]!;
  return values;
}

/**
 * @internal Runs within the caller's device.enqueue. Inputs and the two resident pyramids
 * are owned by the caller; all per-pair scratch is released even when execution/readback fails.
 */
export async function trackKltResidentPair(
  device: ShadeDevice,
  a: KltResidentPyramid,
  b: KltResidentPyramid,
  prepared: KltGpuPreparedPair,
  gpu: KltGpuOptions & { readonly readPyramids?: boolean } = {},
  buildPyramids = false,
): Promise<KltGpuResult> {
  const {
    layout,
    n,
    startData,
    forwardData,
    backData,
    control,
    checkOutput,
    compactOutput,
    checkedLength,
  } = prepared;
  if (
    a.width !== b.width ||
    a.height !== b.height ||
    a.width !== control.width ||
    a.height !== control.height ||
    a.layout.levels !== layout.levels ||
    b.layout.levels !== layout.levels
  )
    throw new RangeError('Resident KLT pyramids must share image coordinates and levels');
  if (!n)
    return {
      records: new Float32Array(),
      forward: new Float32Array(),
      ...(compactOutput ? { ncc: new Float32Array() } : {}),
      back: new Float32Array(),
      report: { tier: device.backend, dispatches: 0, submissions: 0, groups: [] },
      trackChunks: [],
    };
  const owned: Resident<Float32Array>[] = [],
    own = <T extends Resident<Float32Array>>(buffer: T) => (owned.push(buffer), buffer);
  try {
    const start = own(device.allocate(startData.length, startData)),
      forward = own(device.allocate(forwardData?.length ?? 4 * n, forwardData)),
      back = own(device.allocate(4 * n, backData)),
      checked = own(device.allocate(checkedLength));
    const trackChunks: number[] = [],
      minChunk = gpu.minChunkTracks ?? (gpu.batchPasses === true ? 2 * LANES : LANES),
      chunks = (
        key: string | undefined,
        run: (range: { first: number; end: number }, groups: number) => void,
      ) =>
        kltChunkSteps(
          device,
          {
            count: n,
            fixed: gpu.chunkTracks ?? (key === undefined ? n : undefined),
            key,
            min: minChunk,
          },
          run,
          key === 'kltTrack' ? (size) => trackChunks.push(size) : undefined,
        );
    const forwardPass: KltChunkRun = (kltRange, groups) =>
        kltTrack(
          {
            kltControl: control,
            kltRange,
            kltGray: a.gray,
            kltGrad: a.grad,
            kltTarget: b.gray,
            kltStart: start,
            kltNext: forward,
          },
          groups,
        ),
      backwardPass: KltChunkRun = (kltRange, groups) =>
        kltTrack(
          {
            kltControl: control,
            kltRange,
            kltGray: b.gray,
            kltGrad: b.grad,
            kltTarget: a.gray,
            kltStart: forward,
            kltNext: back,
          },
          groups,
        ),
      checkPass: KltChunkRun = (kltRange, groups) =>
        kltCheck(
          {
            kltControl: control,
            kltCheckOutput: checkOutput,
            kltRange,
            kltStart: start,
            kltForward: forward,
            kltBack: back,
            kltChecked: checked,
          },
          groups,
        );
    function* steps() {
      if (buildPyramids)
        yield [
          'kltPyramid',
          () => {
            recordKltPyramid(layout, a.gray, a.grad);
            recordKltPyramid(layout, b.gray, b.grad);
          },
        ] as const;
      if (gpu.batchPasses === true)
        yield* kltRoundtripSteps(
          device,
          { count: n, fixed: gpu.chunkTracks, min: minChunk },
          [forwardPass, backwardPass, checkPass],
          trackChunks,
        );
      else {
        yield* chunks('kltTrack', forwardPass);
        yield* chunks('kltTrack', backwardPass);
        yield* chunks(undefined, checkPass);
      }
    }
    const report = await device.planSteps(steps());
    const read = async (buffer: Resident<Float32Array>, length: number) =>
      (await device.read(buffer, length)).slice(0, length);
    // All outputs follow the final check. Start their independent snapshots together,
    // and let every read settle on failure before finally releases any resident array.
    const [checkedRecords, forwardRecords, backRecords, grayA, grayB, gradA, gradB] =
      await readTogether([
        read(checked, checkedLength),
        compactOutput ? Promise.resolve(new Float32Array()) : read(forward, 4 * n),
        gpu.readBack === false ? Promise.resolve(new Float32Array()) : read(back, 4 * n),
        gpu.readPyramids ? read(a.gray, layout.total) : Promise.resolve(undefined),
        gpu.readPyramids ? read(b.gray, layout.total) : Promise.resolve(undefined),
        gpu.readPyramids ? read(a.grad, 2 * layout.total) : Promise.resolve(undefined),
        gpu.readPyramids ? read(b.grad, 2 * layout.total) : Promise.resolve(undefined),
      ]);
    const pyramids = gpu.readPyramids
      ? { gray: [grayA!, grayB!], grad: [gradA!, gradB!] }
      : undefined;
    return {
      records: compactOutput ? checkedRecords.subarray(0, 4 * n) : checkedRecords,
      forward: forwardRecords,
      ...(compactOutput ? { ncc: checkedRecords.subarray(4 * n, checkedLength) } : {}),
      back: backRecords,
      ...(pyramids ? { pyramids } : {}),
      report,
      trackChunks,
    };
  } finally {
    for (const buffer of owned) device.recycle(buffer);
  }
}

/**
 * Track `points` from the gray frame `prev` to `cur`, then back, and check both. Stateless
 * calls build both pyramids. The bounded pair cache uses the same runner on retained pyramids.
 */
export function trackKltGpu(
  device: ShadeDevice,
  prev: ArrayLike<number>,
  cur: ArrayLike<number>,
  width: number,
  height: number,
  points: ArrayLike<number>,
  options: KltOptions,
  gpu: KltGpuOptions & { readonly readPyramids?: boolean } = {},
): Promise<KltGpuResult> {
  if (prev.length !== width * height || cur.length !== width * height)
    throw new RangeError('Each frame must hold width x height values');
  const prepared = prepareKltGpuPair(width, height, points, options, gpu),
    controls = { ...gpu },
    { layout } = prepared,
    prevPixels = paddedKltGray(prev, layout),
    curPixels = paddedKltGray(cur, layout);
  return device.enqueue(async () => {
    const owned: Resident<Float32Array>[] = [],
      own = <T extends Resident<Float32Array>>(buffer: T) => (owned.push(buffer), buffer);
    try {
      const a = {
          width,
          height,
          layout,
          gray: own(device.allocate(layout.total, prevPixels)),
          grad: own(device.allocate(2 * layout.total)),
        },
        b = {
          width,
          height,
          layout,
          gray: own(device.allocate(layout.total, curPixels)),
          grad: own(device.allocate(2 * layout.total)),
        };
      return await trackKltResidentPair(device, a, b, prepared, controls, true);
    } finally {
      for (const buffer of owned) device.recycle(buffer);
    }
  });
}
