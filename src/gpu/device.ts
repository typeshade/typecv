import {
  createDevice,
  recommendFp64Flavor,
  type ChunkRate,
  type Device,
  type Resident,
} from 'typeshade/runtime';
// The planner. `device.fuse()` runs through the slot this import fills (TypeShade change 0071,
// Amendment 1): the runtime ships none of the planner, so the application imports it once.
import 'typeshade/fuse';
import type { FuseReport, FuseScope } from 'typeshade/fuse';

export interface ShadeDeviceOptions {
  /** WebGPU is required by default. CPU is an explicit differential-testing tier. */
  readonly backend?: 'webgpu' | 'cpu';
  /**
   * A GPUDevice the application owns. TypeCV never destroys it. A call WebGPU cannot run then falls
   * to the CPU tier, in a group of its own (`createDevice({ device })`, typeshade/typeshade#684).
   */
  readonly gpuDevice?: object;
  /**
   * Request the adapter's `maxBufferSize` and `maxStorageBufferBindingSize` for the WebGPU device,
   * not WebGPU's defaults (`docs/dense-mvs.md`, Amendment 3): `bindingBytes` then reads the
   * adapter's limits. The device owns the GPUDevice it requests and destroys it in `dispose()`.
   * It hands the GPUDevice to TypeShade as `gpuDevice` does. Ignored with `gpuDevice` and on the
   * CPU tier. Default false.
   */
  readonly adapterLimits?: boolean;
  /**
   * Whether chunk keys honor their occupancy floors (`chunkSize(key, max, min, minUnits)`, GPU
   * throughput item 2). The default is true on a hardware adapter and false on a software one
   * (SwiftShader, llvmpipe, a fallback adapter): a CPU emulation has no idle lanes to fill, and a
   * larger batch only scores more models than a search that stops early needs.
   */
  readonly occupancyFloors?: boolean;
  /** The most bytes of released arrays the TypeShade device keeps for reuse (change 0071). */
  readonly poolBytes?: number;
  /** The most bytes of active arrays, the pooled ones aside: `reserveWorkingBytes` admits by it. */
  readonly workingBytes?: number;
  readonly maxModelsPerBatch?: number;
  /** Fence geometry work in bounded groups of this many dispatches. */
  readonly boundedSubmissions?: boolean;
  /** Read terminal solver flags to skip inactive iterations; defaults to boundedSubmissions. */
  readonly solverCheckpoints?: boolean;
  /** Maximum geometry dispatches between bounded queue fences; defaults to 1. */
  readonly maxDispatchesPerSubmission?: number;
  /**
   * The flavor of TypeShade's emulated double the kernels run (TypeShade change 0073). `'float'`
   * binds the `_fp64` guard; `'integer'` does the error-free transforms through integer
   * primitives, which a fast-math shader compiler cannot reassociate, and is what an Apple GPU
   * needs (Metal collapses the float flavor). `'auto'`, the default, asks
   * `recommendFp64Flavor` with the adapter's info and the user agent: `'integer'` on an Apple
   * GPU, `'float'` elsewhere and on the CPU tier.
   */
  readonly fp64Flavor?: 'float' | 'integer' | 'auto';
  /**
   * Host scheduling progress. In bounded mode `complete` follows a GPU fence.
   * With unbounded checkpoints it means submitted, unless the chunk ends in a readback.
   * Public solver results always await their final numerical readback.
   */
  readonly onProgress?: (event: { kernel: string; phase: 'start' | 'complete' }) => void;
  /**
   * The time one chunk of a long kernel may take, in milliseconds; 250 by default (TypeShade
   * change 0074: `createDevice({ timeBudgetMs })`). A long kernel (a RANSAC batch's scoring, the
   * sweeps of a triangulation) runs as chunks, one per submission, and the TypeShade device sizes
   * the next chunk from the time the last one took, so that no submission runs long enough for a
   * GPU watchdog to reset the device (`docs/gpu-resident-scene.md`, step 2).
   */
  readonly timeBudgetMs?: number;
  /**
   * The timed chunks of each key a device runs before it trusts the rate it learned, 8 by default
   * (`CHUNK_TIMING_SAMPLES`). Each timed chunk costs two queue fences, a full drain of the GPU
   * each. After this many, the chunks of the key run untimed, sized by the settled rate.
   * `Infinity` times every chunk, as before.
   */
  readonly chunkTimingSamples?: number;
}

/**
 * The timed chunks of a key after which a device stops timing it. The rate is the mean of the
 * previous rate and the last measured one (TypeShade change 0074, `learnRate`), so after eight
 * chunks the first one weighs 1/128 of it, and a chunk that starts at `min` has had seven chances
 * to grow fourfold to what fits the budget.
 */
const CHUNK_TIMING_SAMPLES = 8;
/** The share of the budget a chunk is sized to fill: the `FILL` of TypeShade's chunk arithmetic
 *  (`runtime/chunk.ts`, change 0074), which the package does not export. */
const CHUNK_FILL = 0.8;
/** The most a chunk grows over the last chunk of its key (`GROWTH` of TypeShade's arithmetic). */
const CHUNK_GROWTH = 4;
/**
 * The GPU time of one dispatch above which the key's occupancy floor is halved, in milliseconds.
 * The owner's GPU runs under a browser watchdog: no dispatch may become long enough to be killed
 * (`docs/gpu-throughput.md`, item 2). A chunk sized from a measured GPU time also fits this ceiling.
 */
export const GPU_CEILING_MS = 100;
/** A settled key that declares an occupancy floor is timed again every this many chunks, so that a
 *  larger input cannot push its chunks past the ceiling unseen. */
export const RETIME_EVERY = 16;
/**
 * A step of `planSteps`: the kernel's name, its dispatch, an optional hook that reads what the
 * dispatch wrote before the next step is yielded, and, for a chunk of a long kernel, the chunk's
 * key and the units of work it holds. A chunk step is its own plan, `device.fuse(body, { chunk })`
 * of the TypeShade device (change 0074): fenced and timed, so that `chunkSize(key)` sizes the
 * next chunk of that key from the time it took. A chunk with `timed: false` is still a plan of its
 * own, but untimed: its dispatch is indirect (TypeShade change 0075), so the GPU may have set its
 * workgroup count to zero, and the time of a plan that ran nothing would teach a wrong rate. A
 * chunk of a key whose rate has settled (`chunkTimingSamples` timed chunks) runs untimed too.
 */
export type GpuStep = readonly [
  name: string,
  run: () => void,
  after?: () => Promise<void>,
  chunk?: GpuChunk,
];
/** @internal What a chunk step tells the scheduler. */
export interface GpuChunk {
  readonly key: string;
  readonly units: number;
  /** False when the host cannot know the chunk runs its units: a plan of its own, not timed. */
  readonly timed?: boolean;
  /**
   * The occupancy floor the generator sized this chunk with (`chunkSize(key, max, min, minUnits)`):
   * the units that fill the device. A chunk that declares it is learned from its GPU time when
   * the device measured one, is timed again every `RETIME_EVERY` chunks after its key settled,
   * and halves the key's floor after one dispatch over `GPU_CEILING_MS` (GPU throughput, item 2).
   */
  readonly minUnits?: number;
  /**
   * The chunk's entries dispatched over zero workgroups. Before the first timed chunk of the key,
   * a WebGPU device runs it once as an untimed plan of its own, so that the device creates the
   * entries' pipelines (and compiles their shader module) there and not inside the timed plan: the
   * first rate is then the work's alone. No invocation runs, so it changes no result. `id` names
   * the program: two programs whose chunks share a key are warmed once each.
   */
  readonly warm?: { readonly id: string; readonly run: () => void };
}
/** What the TypeShade device measured of one chunk key (change 0074): the time per unit it
 *  learned, the timed chunks it ran (at most `chunkTimingSamples`: the later chunks of a settled
 *  key run untimed), the longest of them and the last timed chunk's units. */
export type { ChunkRate };

/**
 * The flavor of the emulated double a device needs, from the signals this context has (TypeShade
 * change 0073): the adapter info of a `GPUDevice` the application hands in, else the info of the
 * adapter WebGPU offers, and the user agent. The CPU tier computes doubles as doubles, so it takes
 * `'float'`, the default lowering.
 */
async function recommendedFlavor(
  backend: 'webgpu' | 'cpu',
  gpuDevice: object | undefined,
): Promise<'float' | 'integer'> {
  if (backend !== 'webgpu') return 'float';
  const nav = (globalThis as { navigator?: { gpu?: any; userAgent?: string } }).navigator;
  let adapterInfo = (gpuDevice as { adapterInfo?: unknown } | undefined)?.adapterInfo ?? null;
  if (adapterInfo === null && nav?.gpu) {
    try {
      adapterInfo = (await nav.gpu.requestAdapter())?.info ?? null;
    } catch {
      adapterInfo = null;
    }
  }
  return recommendFp64Flavor({
    adapterInfo: adapterInfo as { vendor?: string; architecture?: string } | null,
    userAgent: nav?.userAgent ?? null,
  });
}
/**
 * Whether the adapter behind the device is software: a fallback adapter, or one whose description
 * names SwiftShader, llvmpipe or a software rasterizer. Unknown counts as hardware.
 */
async function softwareAdapter(gpuDevice: object | undefined): Promise<boolean> {
  const nav = (globalThis as { navigator?: { gpu?: any } }).navigator;
  let info: any = (gpuDevice as { adapterInfo?: unknown } | undefined)?.adapterInfo ?? null,
    fallback = false;
  try {
    const adapter = await nav?.gpu?.requestAdapter();
    fallback = adapter?.isFallbackAdapter === true || adapter?.info?.isFallbackAdapter === true;
    info ??= adapter?.info ?? null;
  } catch {
    // An adapter that cannot be asked is treated as hardware.
  }
  const text = [info?.vendor, info?.architecture, info?.device, info?.description].join(' ');
  return (
    fallback || info?.isFallbackAdapter === true || /swiftshader|software|llvmpipe/i.test(text)
  );
}
/**
 * What the device summed of the groups of its plans for one fused call list (`calls.join('+')`):
 * how many groups ran, how many dispatches they made, and the GPU time of the groups that carry
 * one. `gpuNs` sums the `gpuTime` nanoseconds of `timedGroups` groups. A device without
 * `timestamp-query` has `timedGroups` 0 and `gpuNs` 0, and the row still counts.
 */
export interface KernelStatistic {
  readonly name: string;
  readonly groups: number;
  readonly dispatches: number;
  readonly timedGroups: number;
  readonly gpuNs: number;
}
/** The most call lists the per-kernel collector keeps. A list beyond it is counted in `kernelsDropped`. */
export const MAX_KERNEL_STATISTICS = 160;
type ScalarArray = Float32Array | Float64Array | Uint32Array;
type Buffer = Resident<ScalarArray>;
/** Consumer report without a dependency on TypeShade's build-time package. */
export interface ShadePlanReport {
  readonly tier: 'cpu' | 'webgl2' | 'webgpu';
  readonly dispatches: number;
  readonly submissions: number;
  readonly groups: readonly {
    readonly calls: readonly string[];
    readonly fused: boolean;
    readonly why?: string;
    readonly workgroupShapes?: readonly (readonly [number, number, number])[];
    readonly gpuTime?: number;
    /** An entry call whose workgroup count the GPU wrote (TypeShade change 0075). */
    readonly indirect?: true;
  }[];
  /** The chunk keys the plan ran, each with what the TypeShade device measured of it. */
  readonly chunks?: Readonly<Record<string, ChunkRate>>;
}

/**
 * A GPUDevice with the adapter's `maxBufferSize` and `maxStorageBufferBindingSize`, and its
 * `timestamp-query` feature when it has one, as TypeShade's own request does. Undefined when the
 * environment has no adapter: `createDevice` then reports the missing tier.
 */
async function requestAdapterLimits(): Promise<object | undefined> {
  const gpu = (
    globalThis as {
      navigator?: {
        gpu?: {
          requestAdapter(): Promise<{
            readonly limits: Readonly<Record<string, unknown>>;
            readonly features?: { has(name: string): boolean };
            requestDevice(descriptor: object): Promise<object>;
          } | null>;
        };
      };
    }
  ).navigator?.gpu;
  const adapter = await gpu?.requestAdapter();
  if (!adapter) return undefined;
  const requiredLimits: Record<string, number> = {};
  for (const name of ['maxBufferSize', 'maxStorageBufferBindingSize']) {
    const value = adapter.limits[name];
    if (typeof value === 'number') requiredLimits[name] = value;
  }
  const requiredFeatures = adapter.features?.has('timestamp-query') ? ['timestamp-query'] : [];
  return adapter.requestDevice({ requiredFeatures, requiredLimits });
}

/**
 * The TypeCV device: the serialized queue of geometry work, the bounded submissions, the working
 * budget and the application counters, over a TypeShade device owner (`createDevice()` of
 * `typeshade/runtime`, change 0071). TypeShade holds the tier, the arrays and their pool, the
 * limits, `lost` and the plans. Two devices in one JavaScript context are independent.
 */
export class ShadeDevice {
  readonly backend: 'webgpu' | 'cpu';
  /** The flavor of the emulated double this device's kernels run (TypeShade change 0073). */
  readonly fp64Flavor: 'float' | 'integer';
  readonly boundedSubmissions: boolean;
  readonly solverCheckpoints: boolean;
  /** Resolves when the browser invalidates its WebGPU device. */
  readonly lost?: Promise<{ message: string; reason: string }>;
  readonly maxModelsPerBatch: number;
  readonly maxDispatchesPerSubmission: number;
  /** The time one chunk of a long kernel may take, in milliseconds (TypeShade change 0074). */
  readonly timeBudgetMs: number;
  /** The timed chunks of each key before its chunks run untimed (`CHUNK_TIMING_SAMPLES`). */
  readonly chunkTimingSamples: number;
  readonly workingBytes: number;
  readonly bindingBytes: number;
  /** The device's `maxBufferSize`: 256 MiB, the WebGPU default, when the device reports none. */
  readonly bufferBytes: number;
  /** Whether the device requested the adapter's limits (`ShadeDeviceOptions.adapterLimits`). */
  readonly adapterLimits: boolean;
  /** Current compiled portable kernels use a 64 x 1 x 1 workgroup. */
  readonly maxKernelElements: number;
  /** The TypeShade device owner: the tier, the arrays and their pool, the limits, the plans. */
  private readonly shade: Device;
  /** The handles this device gave out and has not taken back, so that `recycle` and `drop` are
   *  idempotent: a handle that left the books is ignored. */
  private readonly active = new Set<Buffer>();
  private readonly history: ShadePlanReport[] = [];
  private readonly kernelRows = new Map<
    string,
    { -readonly [K in keyof KernelStatistic]: KernelStatistic[K] }
  >();
  private kernelsDropped = 0;
  /** The programs this device warmed, by chunk key and `warm.id`. */
  private readonly warmed = new Set<string>();
  /** Whether chunk keys honor their occupancy floors (`ShadeDeviceOptions.occupancyFloors`). */
  occupancyFloors = false;
  /** The GPU milliseconds a unit takes, by chunk key, learned from the `gpuTime` of its chunks. */
  private readonly gpuMsPerUnit = new Map<string, number>();
  /** The occupancy floor of a key after a dispatch overran `GPU_CEILING_MS`, halved each time. */
  private readonly lowered = new Map<string, number>();
  /** The settled chunks of a key since the last one timed, for the keys that declare a floor. */
  private readonly sinceTimed = new Map<string, number>();
  private uploads = 0;
  private downloads = 0;
  private reads = 0;
  private pendingReads = 0;
  private readWaves = 0;
  private snapshotBytes = 0;
  private tail: Promise<unknown> = Promise.resolve();
  private closing = false;
  private closed = false;
  private disposal?: Promise<void>;
  /** The GPUDevice this device requested with the adapter's limits, destroyed in `dispose()`. */
  private owned?: { destroy(): void };
  private constructor(
    backend: 'webgpu' | 'cpu',
    shade: Device,
    workingBytes: number,
    maxModels: number,
    maxDispatchesPerSubmission: number,
    private readonly onProgress: ShadeDeviceOptions['onProgress'],
    boundedSubmissions: boolean,
    solverCheckpoints: boolean,
    chunkTimingSamples: number,
  ) {
    this.backend = backend;
    this.chunkTimingSamples = chunkTimingSamples;
    this.shade = shade;
    this.boundedSubmissions = boundedSubmissions;
    this.solverCheckpoints = solverCheckpoints;
    this.timeBudgetMs = shade.timeBudgetMs;
    if (backend === 'webgpu')
      this.lost = shade.lost.then((info) => ({
        message: info.message ?? 'WebGPU device was lost',
        reason: info.reason ?? 'unknown',
      }));
    this.fp64Flavor = shade.fp64Flavor;
    this.workingBytes = workingBytes;
    this.maxModelsPerBatch = maxModels;
    this.maxDispatchesPerSubmission = maxDispatchesPerSubmission;
    // The limits of the WebGPU device, by their names in GPUSupportedLimits (TypeShade change
    // 0051); empty on the CPU tier, where the defaults are the WebGPU minimums.
    const limits = shade.limits;
    this.bindingBytes = Math.min(
      limits['maxStorageBufferBindingSize'] ?? 128 * 1024 ** 2,
      limits['maxBufferSize'] ?? 256 * 1024 ** 2,
    );
    this.maxKernelElements = (limits['maxComputeWorkgroupsPerDimension'] ?? 65535) * 64;
    this.bufferBytes = limits['maxBufferSize'] ?? 256 * 1024 ** 2;
    this.adapterLimits = false;
  }
  static async open(options: ShadeDeviceOptions = {}): Promise<ShadeDevice> {
    const backend = options.backend ?? 'webgpu',
      pool = options.poolBytes ?? 32 * 1024 ** 2;
    const working = options.workingBytes ?? 64 * 1024 ** 2,
      models = options.maxModelsPerBatch ?? (options.boundedSubmissions ? 64 : 1024),
      maxDispatchesPerSubmission = options.maxDispatchesPerSubmission ?? 1,
      timeBudgetMs = options.timeBudgetMs ?? 250,
      chunkTimingSamples = options.chunkTimingSamples ?? CHUNK_TIMING_SAMPLES;
    if (
      !(Number.isSafeInteger(chunkTimingSamples) || chunkTimingSamples === Infinity) ||
      chunkTimingSamples < 1 ||
      !Number.isFinite(timeBudgetMs) ||
      timeBudgetMs < 1 ||
      timeBudgetMs > 60000 ||
      !['webgpu', 'cpu'].includes(backend) ||
      !Number.isSafeInteger(pool) ||
      pool < 0 ||
      !Number.isSafeInteger(working) ||
      working < 4096 ||
      !Number.isSafeInteger(models) ||
      models < 1 ||
      models > 2 ** 24 ||
      !Number.isSafeInteger(maxDispatchesPerSubmission) ||
      maxDispatchesPerSubmission < 1 ||
      maxDispatchesPerSubmission > 64 ||
      (options.boundedSubmissions !== undefined &&
        typeof options.boundedSubmissions !== 'boolean') ||
      (options.solverCheckpoints !== undefined && typeof options.solverCheckpoints !== 'boolean') ||
      (options.onProgress !== undefined && typeof options.onProgress !== 'function') ||
      (options.fp64Flavor !== undefined &&
        !['float', 'integer', 'auto'].includes(options.fp64Flavor))
    )
      throw new RangeError('Invalid TypeShade device options');
    if (backend === 'cpu' && options.gpuDevice !== undefined)
      throw new TypeError('CPU tier cannot take a GPUDevice');
    // `webgpu` alone is required: a list of one tier makes it so. A GPUDevice the application
    // hands in keeps the later tiers as the fallback of a call WebGPU cannot run (#684).
    if (options.adapterLimits !== undefined && typeof options.adapterLimits !== 'boolean')
      throw new RangeError('Invalid TypeShade device options');
    // Amendment 3 of `docs/dense-mvs.md`: the adapter's buffer limits, on a GPUDevice of our own.
    const owned =
      backend === 'webgpu' && options.adapterLimits && options.gpuDevice === undefined
        ? await requestAdapterLimits()
        : undefined;
    const gpuDevice = options.gpuDevice ?? owned;
    const fp64Flavor =
      options.fp64Flavor === undefined || options.fp64Flavor === 'auto'
        ? await recommendedFlavor(backend, gpuDevice)
        : options.fp64Flavor;
    // The time budget is the TypeShade device's (change 0074): it fences and times each chunked
    // plan and learns the rate of its key.
    const shade = await createDevice(
      gpuDevice !== undefined
        ? { device: gpuDevice, poolBytes: pool, fp64Flavor, timeBudgetMs }
        : { prefer: [backend], poolBytes: pool, fp64Flavor, timeBudgetMs },
    );
    const device = new ShadeDevice(
      backend,
      shade,
      working,
      models,
      maxDispatchesPerSubmission,
      options.onProgress,
      options.boundedSubmissions ?? false,
      options.solverCheckpoints ?? options.boundedSubmissions ?? false,
      chunkTimingSamples,
    );
    device.occupancyFloors =
      options.occupancyFloors ?? (backend === 'webgpu' && !(await softwareAdapter(gpuDevice)));
    if (owned) {
      device.owned = owned as { destroy(): void };
      (device as { adapterLimits: boolean }).adapterLimits = true;
    }
    return device;
  }
  /**
   * @internal The size of the next chunk of `key`, in units, as the TypeShade device sizes it
   * (change 0074): as many as fit eight tenths of the time budget at the rate the chunks of that
   * key have measured, at most `max`, at most four times the last chunk (a chunk that measured
   * fast is not trusted to be a hundred times faster), and at least `min`. The first chunk of a
   * key is `min`: the rate is learned from it. The CPU tier has no watchdog: it runs the whole
   * work as one chunk, so that its results and its counts are those of the state machine run
   * once, bit for bit.
   *
   * Once the rate of `key` has settled (`chunkTimingSamples` timed chunks), the chunks are no
   * longer timed, and the last chunk's units no longer change. The fourfold cap over them would
   * then hold every later chunk to four times whatever the last timed chunk was (one trial, after
   * a refit's scoring), so a settled key takes what fits the budget at its rate: at most `max`,
   * at least `min`.
   */
  chunkSize(key: string, max: number, min = 1, minUnits = min): number {
    if (!this.occupancyFloors) minUnits = min;
    const size = this.shade.chunkSize(key, max, min);
    if (this.backend === 'cpu') return size;
    const settled = this.settled(key),
      measured = this.shade.chunkRates[key],
      gpu = this.gpuMsPerUnit.get(key);
    // The occupancy floor, lowered once for each dispatch that overran the ceiling.
    const floor = Math.min(
      max,
      Math.max(min, Math.min(minUnits, this.lowered.get(key) ?? minUnits)),
    );
    if (floor <= min && gpu === undefined) {
      if (settled === undefined) return size;
      const fit = Math.floor((this.timeBudgetMs * CHUNK_FILL) / settled.msPerUnit);
      return Math.min(max, Math.max(1, min, fit));
    }
    // A key that has not settled reaches its floor in steps of at most four times the last chunk,
    // from `min`: the first dispatch of a key is never the whole floor.
    const last = measured?.lastUnits ?? min,
      reach =
        settled !== undefined
          ? floor
          : measured === undefined
            ? min
            : Math.max(min, Math.min(floor, last * CHUNK_GROWTH));
    const msPerUnit = gpu ?? measured?.msPerUnit;
    if (msPerUnit === undefined) return Math.min(max, reach);
    // A time measured on the GPU also holds the chunk to the ceiling.
    const budget =
        gpu === undefined ? this.timeBudgetMs : Math.min(this.timeBudgetMs, GPU_CEILING_MS),
      fit = Math.floor((budget * CHUNK_FILL) / msPerUnit);
    return Math.min(
      max,
      Math.max(1, reach, settled === undefined ? Math.min(fit, last * CHUNK_GROWTH) : fit),
    );
  }
  /**
   * Whether the chunk is timed. The chunks of a key that has not settled are. A settled key runs
   * untimed, except that a key with an occupancy floor is timed every `RETIME_EVERY` chunks.
   */
  private times(chunk: GpuChunk): boolean {
    if (chunk.timed === false) return false;
    if (!this.settled(chunk.key)) return true;
    if (chunk.minUnits === undefined || !this.occupancyFloors) return false;
    const count = (this.sinceTimed.get(chunk.key) ?? 0) + 1;
    this.sinceTimed.set(chunk.key, count >= RETIME_EVERY ? 0 : count);
    return count >= RETIME_EVERY;
  }
  /**
   * Learn from the GPU time of a timed chunk that declared a floor: the rate per unit, and the
   * ceiling. A report without `gpuTime` in every group teaches nothing: the fence rate stays.
   */
  private observe(chunk: GpuChunk, report: ShadePlanReport): void {
    if (chunk.minUnits === undefined || !this.occupancyFloors || !report.groups.length) return;
    let total = 0,
      worst = 0;
    for (const { gpuTime } of report.groups) {
      if (typeof gpuTime !== 'number' || !Number.isFinite(gpuTime) || gpuTime < 0) return;
      total += gpuTime;
      worst = Math.max(worst, gpuTime);
    }
    const measured = Math.max(1e-3, total / 1e6) / Math.max(1, chunk.units),
      previous = this.gpuMsPerUnit.get(chunk.key);
    this.gpuMsPerUnit.set(
      chunk.key,
      previous === undefined ? measured : 0.5 * previous + 0.5 * measured,
    );
    if (worst / 1e6 > GPU_CEILING_MS) {
      const current = this.lowered.get(chunk.key) ?? chunk.minUnits;
      this.lowered.set(chunk.key, Math.max(1, Math.floor(current / 2)));
    }
  }
  /** The rate of `key` once the device has timed `chunkTimingSamples` chunks of it. */
  private settled(key: string): ChunkRate | undefined {
    const rate = this.shade.chunkRates[key];
    return rate !== undefined && rate.chunks >= this.chunkTimingSamples ? rate : undefined;
  }
  /** What the TypeShade device measured of each chunk key so far: its timed chunks alone. */
  get chunkRates(): Readonly<Record<string, ChunkRate>> {
    return this.shade.chunkRates;
  }
  /** The per-kernel table since the device opened or `resetKernelStatistics()`, in first-use order. */
  get kernelStatistics(): { rows: readonly KernelStatistic[]; dropped: number } {
    return {
      rows: [...this.kernelRows.values()].map((r) => ({ ...r })),
      dropped: this.kernelsDropped,
    };
  }
  get reports(): readonly ShadePlanReport[] {
    return this.history.slice();
  }
  /** What the TypeShade device counts of the arrays, and what TypeCV reads of them. */
  get statistics() {
    const s = this.shade.statistics;
    return {
      liveBuffers: s.live,
      liveBytes: s.liveBytes,
      pooledBytes: s.pooledBytes,
      allocations: s.allocations,
      reuses: s.reuses,
      /** The values `allocate*()` was given, and the zeros of a new f64 array. A later `write()`
       *  through a handle is the compiler's `uploadBytes`, not this count. */
      applicationUploadBytes: this.uploads,
      applicationReadbackBytes: this.downloads,
      /** The groups, dispatches and GPU time of each fused call list (at most 160 lists). */
      kernels: [...this.kernelRows.values()].map((r) => ({ ...r })) as readonly KernelStatistic[],
      /** Call lists beyond the 160 the collector keeps. */
      kernelsDropped: this.kernelsDropped,
      /** Calls of `read`: each is a round trip to the GPU. */
      reads: this.reads,
      /** Round trips of reads: reads that were in flight together count once. */
      readWaves: this.readWaves,
      /** Full arrays returned by Resident.read(); application counters may name a shorter prefix. */
      residentSnapshotBytes: this.snapshotBytes,
    };
  }
  /** @internal All jobs share the device and run in submission order. */
  enqueue<T>(operation: () => Promise<T>): Promise<T> {
    if (this.closing || this.closed)
      return Promise.reject(new Error('ShadeDevice is closing or disposed'));
    const job = this.tail.then(operation);
    this.tail = job.catch(() => undefined);
    return job;
  }
  /** @internal */
  async flush(): Promise<void> {
    await this.tail;
  }
  /** @internal */
  allocate(length: number, values?: Float32Array): Resident<Float32Array> {
    return this.allocateTyped('f32', length, values) as Resident<Float32Array>;
  }
  /** @internal f64 storage uses TypeShade's public Float64Array packing path. */
  allocateF64(length: number, values?: Float64Array): Resident<Float64Array> {
    return this.allocateTyped('f64', length, values) as Resident<Float64Array>;
  }
  /** @internal */
  allocateU32(length: number, values?: Uint32Array): Resident<Uint32Array> {
    return this.allocateTyped('u32', length, values) as Resident<Uint32Array>;
  }
  /**
   * @internal Conservative admission for additional active resident storage: the bytes of the
   * arrays given out and not taken back, plus `additionalBytes`, stay within `workingBytes`. The
   * released arrays the TypeShade device pools are bounded by `poolBytes` on their own, so the
   * whole is at most `workingBytes + poolBytes`. This does not bound TypeShade textures, staging
   * buffers or physical VRAM.
   */
  reserveWorkingBytes(additionalBytes: number): void {
    if (!Number.isSafeInteger(additionalBytes) || additionalBytes < 0)
      throw new RangeError('Invalid TypeShade working reservation');
    if (this.closed) throw new Error('ShadeDevice is disposed');
    const s = this.shade.statistics;
    if (s.liveBytes - s.pooledBytes + additionalBytes > this.workingBytes)
      throw new RangeError('Allocation exceeds TypeShade working budget');
  }
  private allocateTyped(kind: 'f32' | 'f64' | 'u32', length: number, values?: ScalarArray): Buffer {
    const bytes = length * (kind === 'f64' ? 8 : 4);
    if (
      !Number.isSafeInteger(length) ||
      length < 1 ||
      bytes > this.bindingBytes ||
      (values !== undefined && values.length !== length)
    )
      throw new RangeError('Array exceeds the device storage binding limit');
    // A reused array leaves the pool and becomes active, so the admission is the same either way.
    this.reserveWorkingBytes(bytes);
    // A zeroed array of f32 or u32 is the device's to zero (Rule 11.8). An f64 array holds a host
    // array of zeros, which its first use packs once.
    const reuses = this.shade.statistics.reuses;
    const result = (
      values !== undefined
        ? this.shade.array(values)
        : kind === 'f64'
          ? this.shade.array(new Float64Array(length))
          : this.shade.array(length, kind)
    ) as Buffer;
    this.active.add(result);
    if (values !== undefined) this.uploads += values.byteLength;
    else if (kind === 'f64' && this.shade.statistics.reuses === reuses) this.uploads += bytes;
    return result;
  }
  /** @internal Give an array back: the TypeShade device keeps it for a later array of the same
   *  element and length while the pool has room, and frees it otherwise. */
  recycle(buffer: Buffer): void {
    if (!this.active.delete(buffer)) return;
    this.shade.release(buffer);
  }
  /** @internal A failed plan's buffers must be destroyed, never reused. */
  drop(buffer: Buffer): void {
    if (!this.active.delete(buffer)) return;
    buffer.destroy();
  }
  /** @internal A plan, or with `chunk` one chunk of a long kernel: the TypeShade device fences it,
   *  times it and learns the rate of its key (change 0074). The report then carries `chunk`. */
  async plan(body: (scope: FuseScope) => void, chunk?: GpuChunk): Promise<FuseReport> {
    const report = await this.shade.fuse(
      body,
      chunk === undefined ? undefined : { chunk: { key: chunk.key, units: chunk.units } },
    );
    if (report.tier !== this.backend) throw new Error(`Unexpected execution tier: ${report.tier}`);
    if (chunk !== undefined) this.observe(chunk, report);
    // Only runtime submissions add work. planSteps/planStepsMany also record a composite
    // report for callers, but its groups already appeared in these individual reports.
    this.collect(report);
    this.record(report);
    return report;
  }
  /**
   * Add the groups of one plan to the per-kernel table. Each group is read from the report that
   * the TypeShade device already built: this makes no query, no readback and no submission.
   */
  private collect(report: ShadePlanReport): void {
    for (const group of report.groups) {
      const name = group.calls.join('+');
      let row = this.kernelRows.get(name);
      if (row === undefined) {
        if (this.kernelRows.size >= MAX_KERNEL_STATISTICS) {
          this.kernelsDropped++;
          continue;
        }
        row = { name, groups: 0, dispatches: 0, timedGroups: 0, gpuNs: 0 };
        this.kernelRows.set(name, row);
      }
      row.groups++;
      row.dispatches += group.workgroupShapes?.length ?? 0;
      if (
        typeof group.gpuTime === 'number' &&
        Number.isFinite(group.gpuTime) &&
        group.gpuTime >= 0
      ) {
        row.timedGroups++;
        row.gpuNs += group.gpuTime;
      }
    }
  }
  /** Empty the per-kernel table, at the start of a run on a reused device. */
  resetKernelStatistics(): void {
    this.kernelRows.clear();
    this.kernelsDropped = 0;
  }
  private record(report: ShadePlanReport): void {
    this.history.push(report);
    if (this.history.length > 64) this.history.shift();
  }
  /**
   * @internal Run the steps as plans: consecutive steps join one plan until a step reads what it
   * wrote (its after hook decides the next step) or a chunk step comes, which is a plan of its
   * own, `device.fuse(body, { chunk })` of the TypeShade device (change 0074): fenced on the queue
   * and timed, so the device learns the rate of its key. A chunk step with `timed: false`, or of a
   * key whose rate has settled (`chunkTimingSamples` timed chunks), is a plan of its own too, run
   * as an ordinary plan: no fences and no rate. Before the first timed chunk of a key, a WebGPU
   * device runs the chunk's `warm` as an untimed plan of its own, once for each program. Solver
   * state stays resident across the plans; no numerical readback. With bounded submissions every
   * plan holds at most `maxDispatchesPerSubmission` steps and is fenced.
   */
  async planSteps(steps: Iterable<GpuStep>): Promise<ShadePlanReport> {
    const reports: ShadePlanReport[] = [];
    const keys = new Set<string>();
    const iterator = steps[Symbol.iterator]();
    let done = false;
    let pending: GpuStep | undefined;
    while (!done) {
      const group: GpuStep[] = [];
      let alone: GpuChunk | undefined;
      if (pending) {
        group.push(pending);
        alone = pending[3];
        pending = undefined;
      }
      while (
        !alone &&
        (!this.boundedSubmissions || group.length < this.maxDispatchesPerSubmission)
      ) {
        const next = iterator.next();
        if (next.done) {
          done = true;
          break;
        }
        if (next.value[3]) {
          // A chunk step runs alone: a timed one is measured on its own.
          if (group.length) {
            pending = next.value;
            break;
          }
          group.push(next.value);
          alone = next.value[3];
          break;
        }
        group.push(next.value);
        // An after hook can read GPU state that determines the next yielded step.
        // Do not advance the generator until that hook has completed.
        if (next.value[2]) break;
      }
      if (!group.length) break;
      const timed = alone !== undefined && this.times(alone) ? alone : undefined;
      // The pipelines of a key's first timed chunk are made in a plan before it (`GpuChunk.warm`),
      // so that the timed plan holds the work alone. The timed chunk's first fence waits for it.
      const warming = timed?.warm && `${timed.key}\n${timed.warm.id}`;
      if (warming && this.backend === 'webgpu' && !this.warmed.has(warming)) {
        this.warmed.add(warming);
        reports.push(await this.plan(timed.warm!.run));
      }
      reports.push(
        await this.plan(() => {
          for (const [kernel, run] of group) {
            this.onProgress?.({ kernel, phase: 'start' });
            run();
          }
        }, timed),
      );
      // Awaiting fuse only submits: it is not a GPU completion fence. `wait()` is the queue fence
      // (`onSubmittedWorkDone`) that keeps each bounded group out of a long-running command
      // stream. A chunked plan is fenced by the TypeShade device, and its promise resolves after
      // the fence.
      if (this.boundedSubmissions && !timed && this.backend === 'webgpu') await this.shade.wait();
      if (alone) keys.add(alone.key);
      for (const [kernel, , after] of group) {
        await after?.();
        this.onProgress?.({ kernel, phase: 'complete' });
      }
    }
    // The steps are one plan to the caller and to `reports`, however many submissions they took.
    while (reports.includes(this.history[this.history.length - 1]!)) this.history.pop();
    const rates = this.shade.chunkRates;
    const chunks = [...keys].filter((k) => k in rates).map((k) => [k, rates[k]!] as const);
    const report: ShadePlanReport = {
      tier: this.backend,
      dispatches: reports.reduce((n, r) => n + r.dispatches, 0),
      submissions: reports.reduce((n, r) => n + r.submissions, 0),
      groups: reports.flatMap((r) => r.groups),
      ...(chunks.length ? { chunks: Object.fromEntries(chunks) } : {}),
    };
    this.record(report);
    return report;
  }

  /**
   * @internal Run several step streams as one job. The streams are independent: they share no
   * buffer. Each round pulls every live stream to its next break, a step with an after hook or a
   * chunk step, and then runs one plan of all the steps before the breaks, awaits all the hooks
   * together, and runs the chunk steps of the round as one plan for each chunk key. A round
   * costs the submissions and the fences of one stream, so K streams cost about 1/K of their sum.
   * The chunks of one key join into one timed chunk of the sum of their units, so the rate the
   * device learns is still a time for each unit. A stream that must stay in a bounded
   * submission (`boundedSubmissions`) belongs to `planSteps`, not here.
   */
  async planStepsMany(streams: readonly Iterable<GpuStep>[]): Promise<ShadePlanReport> {
    if (this.boundedSubmissions) throw new Error('planStepsMany needs unbounded submissions');
    const reports: ShadePlanReport[] = [];
    const keys = new Set<string>();
    const iterators = streams.map((stream) => stream[Symbol.iterator]());
    let live = iterators.map((_, index) => index);
    while (live.length) {
      const group: GpuStep[] = [],
        chunks: GpuStep[] = [],
        next: number[] = [];
      for (const index of live) {
        let ended = false;
        for (;;) {
          const step = iterators[index]!.next();
          if (step.done) {
            ended = true;
            break;
          }
          if (step.value[3]) {
            chunks.push(step.value);
            break;
          }
          group.push(step.value);
          // An after hook can read GPU state that determines the next step of this stream.
          if (step.value[2]) break;
        }
        if (!ended) next.push(index);
      }
      live = next;
      if (group.length) {
        reports.push(
          await this.plan(() => {
            for (const [kernel, run] of group) {
              this.onProgress?.({ kernel, phase: 'start' });
              run();
            }
          }),
        );
        await Promise.all(group.map(([, , after]) => after?.()));
        for (const [kernel] of group) this.onProgress?.({ kernel, phase: 'complete' });
      }
      const byKey = new Map<string, GpuStep[]>();
      for (const step of chunks)
        byKey.set(step[3]!.key, [...(byKey.get(step[3]!.key) ?? []), step]);
      for (const [key, steps] of byKey) {
        const units = steps.reduce((sum, step) => sum + step[3]!.units, 0);
        const timed =
          steps.some((step) => step[3]!.timed === false) || !this.times(steps[0]![3]!)
            ? undefined
            : { ...steps[0]![3]!, units };
        const warming = timed?.warm && `${key}\n${timed.warm.id}`;
        if (warming && this.backend === 'webgpu' && !this.warmed.has(warming)) {
          this.warmed.add(warming);
          reports.push(await this.plan(timed!.warm!.run));
        }
        reports.push(
          await this.plan(() => {
            for (const [kernel, run] of steps) {
              this.onProgress?.({ kernel, phase: 'start' });
              run();
            }
          }, timed),
        );
        keys.add(key);
        await Promise.all(steps.map(([, , after]) => after?.()));
        for (const [kernel] of steps) this.onProgress?.({ kernel, phase: 'complete' });
      }
    }
    while (reports.includes(this.history[this.history.length - 1]!)) this.history.pop();
    const rates = this.shade.chunkRates;
    const chunks = [...keys].filter((k) => k in rates).map((k) => [k, rates[k]!] as const);
    const report: ShadePlanReport = {
      tier: this.backend,
      dispatches: reports.reduce((n, r) => n + r.dispatches, 0),
      submissions: reports.reduce((n, r) => n + r.submissions, 0),
      groups: reports.flatMap((r) => r.groups),
      ...(chunks.length ? { chunks: Object.fromEntries(chunks) } : {}),
    };
    this.record(report);
    return report;
  }
  /** @internal The counts describe application arrays, not runtime uniforms or staging allocation. */
  async read<T extends ScalarArray>(buffer: Resident<T>, length: number): Promise<T> {
    this.reads++;
    if (this.pendingReads++ === 0) this.readWaves++;
    const data = await buffer.read().finally(() => {
      this.pendingReads--;
    });
    this.downloads += data.BYTES_PER_ELEMENT * length;
    this.snapshotBytes += data.byteLength;
    return data;
  }
  dispose(): Promise<void> {
    if (this.disposal) return this.disposal;
    this.closing = true;
    this.disposal = (async () => {
      await this.tail;
      // The owner destroys every array it holds, pooled or not, each once, and the WebGPU device
      // it requested; a GPUDevice the application handed in is left as it is.
      this.active.clear();
      await this.shade.dispose();
      this.owned?.destroy();
      this.closed = true;
    })();
    return this.disposal;
  }
}
