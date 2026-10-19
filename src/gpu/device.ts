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
}
/**
 * A step of `planSteps`: the kernel's name, its dispatch, an optional hook that reads what the
 * dispatch wrote before the next step is yielded, and, for a chunk of a long kernel, the chunk's
 * key and the units of work it holds. A chunk step is its own plan, `device.fuse(body, { chunk })`
 * of the TypeShade device (change 0074): fenced and timed, so that `chunkSize(key)` sizes the
 * next chunk of that key from the time it took.
 */
type GpuStep = readonly [
  name: string,
  run: () => void,
  after?: () => Promise<void>,
  chunk?: GpuChunk,
];
/** @internal What a chunk step tells the scheduler. */
export interface GpuChunk {
  readonly key: string;
  readonly units: number;
}
/** What the TypeShade device measured of one chunk key (change 0074): the time per unit it
 *  learned, the chunks it ran, the longest of them and the last chunk's units. */
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
  }[];
  /** The chunk keys the plan ran, each with what the TypeShade device measured of it. */
  readonly chunks?: Readonly<Record<string, ChunkRate>>;
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
  readonly workingBytes: number;
  readonly bindingBytes: number;
  /** Current compiled portable kernels use a 64 x 1 x 1 workgroup. */
  readonly maxKernelElements: number;
  /** The TypeShade device owner: the tier, the arrays and their pool, the limits, the plans. */
  private readonly shade: Device;
  /** The handles this device gave out and has not taken back, so that `recycle` and `drop` are
   *  idempotent: a handle that left the books is ignored. */
  private readonly active = new Set<Buffer>();
  private readonly history: ShadePlanReport[] = [];
  private uploads = 0;
  private downloads = 0;
  private snapshotBytes = 0;
  private tail: Promise<unknown> = Promise.resolve();
  private closing = false;
  private closed = false;
  private disposal?: Promise<void>;
  private constructor(
    backend: 'webgpu' | 'cpu',
    shade: Device,
    workingBytes: number,
    maxModels: number,
    maxDispatchesPerSubmission: number,
    private readonly onProgress: ShadeDeviceOptions['onProgress'],
    boundedSubmissions: boolean,
    solverCheckpoints: boolean,
  ) {
    this.backend = backend;
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
  }
  static async open(options: ShadeDeviceOptions = {}): Promise<ShadeDevice> {
    const backend = options.backend ?? 'webgpu',
      pool = options.poolBytes ?? 32 * 1024 ** 2;
    const working = options.workingBytes ?? 64 * 1024 ** 2,
      models = options.maxModelsPerBatch ?? (options.boundedSubmissions ? 64 : 1024),
      maxDispatchesPerSubmission = options.maxDispatchesPerSubmission ?? 1,
      timeBudgetMs = options.timeBudgetMs ?? 250;
    if (
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
    const fp64Flavor =
      options.fp64Flavor === undefined || options.fp64Flavor === 'auto'
        ? await recommendedFlavor(backend, options.gpuDevice)
        : options.fp64Flavor;
    // The time budget is the TypeShade device's (change 0074): it fences and times each chunked
    // plan and learns the rate of its key.
    const shade = await createDevice(
      options.gpuDevice !== undefined
        ? { device: options.gpuDevice, poolBytes: pool, fp64Flavor, timeBudgetMs }
        : { prefer: [backend], poolBytes: pool, fp64Flavor, timeBudgetMs },
    );
    return new ShadeDevice(
      backend,
      shade,
      working,
      models,
      maxDispatchesPerSubmission,
      options.onProgress,
      options.boundedSubmissions ?? false,
      options.solverCheckpoints ?? options.boundedSubmissions ?? false,
    );
  }
  /**
   * @internal The size of the next chunk of `key`, in units, as the TypeShade device sizes it
   * (change 0074): as many as fit eight tenths of the time budget at the rate the chunks of that
   * key have measured, at most `max`, at most four times the last chunk (a chunk that measured
   * fast is not trusted to be a hundred times faster), and at least `min`. The first chunk of a
   * key is `min`: the rate is learned from it. The CPU tier has no watchdog: it runs the whole
   * work as one chunk, so that its results and its counts are those of the state machine run
   * once, bit for bit.
   */
  chunkSize(key: string, max: number, min = 1): number {
    return this.shade.chunkSize(key, max, min);
  }
  /** What the TypeShade device measured of each chunk key so far. */
  get chunkRates(): Readonly<Record<string, ChunkRate>> {
    return this.shade.chunkRates;
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
    const report = await this.shade.fuse(body, chunk === undefined ? undefined : { chunk });
    if (report.tier !== this.backend) throw new Error(`Unexpected execution tier: ${report.tier}`);
    this.record(report);
    return report;
  }
  private record(report: ShadePlanReport): void {
    this.history.push(report);
    if (this.history.length > 64) this.history.shift();
  }
  /**
   * @internal Run the steps as plans: consecutive steps join one plan until a step reads what it
   * wrote (its after hook decides the next step) or a chunk step comes, which is a plan of its
   * own, `device.fuse(body, { chunk })` of the TypeShade device (change 0074): fenced on the queue
   * and timed, so the device learns the rate of its key. Solver state stays resident across the
   * plans; no numerical readback. With bounded submissions every plan holds at most
   * `maxDispatchesPerSubmission` steps and is fenced.
   */
  async planSteps(steps: Iterable<GpuStep>): Promise<ShadePlanReport> {
    const reports: ShadePlanReport[] = [];
    const keys = new Set<string>();
    const iterator = steps[Symbol.iterator]();
    let done = false;
    let pending: GpuStep | undefined;
    while (!done) {
      const group: GpuStep[] = [];
      let timed: GpuChunk | undefined;
      if (pending) {
        group.push(pending);
        timed = pending[3];
        pending = undefined;
      }
      while (
        !timed &&
        (!this.boundedSubmissions || group.length < this.maxDispatchesPerSubmission)
      ) {
        const next = iterator.next();
        if (next.done) {
          done = true;
          break;
        }
        if (next.value[3]) {
          // A chunk step runs alone: it is measured on its own.
          if (group.length) {
            pending = next.value;
            break;
          }
          group.push(next.value);
          timed = next.value[3];
          break;
        }
        group.push(next.value);
        // An after hook can read GPU state that determines the next yielded step.
        // Do not advance the generator until that hook has completed.
        if (next.value[2]) break;
      }
      if (!group.length) break;
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
      if (timed) keys.add(timed.key);
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
  /** @internal The counts describe application arrays, not runtime uniforms or staging allocation. */
  async read<T extends ScalarArray>(buffer: Resident<T>, length: number): Promise<T> {
    const data = await buffer.read();
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
      this.closed = true;
    })();
    return this.disposal;
  }
}
