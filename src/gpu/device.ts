import { configure, createRuntime, resident, type Resident, type Runtime } from 'typeshade/runtime';
import { fuse, type FuseReport } from 'typeshade/fuse';

export interface ShadeDeviceOptions {
  /** WebGPU is required by default. CPU is an explicit differential-testing tier. */
  readonly backend?: 'webgpu' | 'cpu';
  readonly gpuDevice?: object;
  readonly poolBytes?: number;
  readonly workingBytes?: number;
  readonly maxModelsPerBatch?: number;
  /** Fence geometry work in bounded groups of this many dispatches. */
  readonly boundedSubmissions?: boolean;
  /** Read terminal solver flags to skip inactive iterations; defaults to boundedSubmissions. */
  readonly solverCheckpoints?: boolean;
  /** Maximum geometry dispatches between bounded queue fences; defaults to 1. */
  readonly maxDispatchesPerSubmission?: number;
  /**
   * Host scheduling progress. In bounded mode `complete` follows a GPU fence.
   * With unbounded checkpoints it means submitted, unless the chunk ends in a readback.
   * Public solver results always await their final numerical readback.
   */
  readonly onProgress?: (event: { kernel: string; phase: 'start' | 'complete' }) => void;
}
type GpuStep = readonly [name: string, run: () => void, after?: () => Promise<void>];
type ScalarArray = Float32Array | Float64Array | Uint32Array;
type Buffer = Resident<ScalarArray>;
type Allocation = { length: number; bytes: number; kind: 'f32' | 'f64' | 'u32'; key: string };
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
}
let active: ShadeDevice | undefined;
let opening = false;

/** Owner of the TypeShade runtime, resident buffers, queue and bounded reuse pool. */
export class ShadeDevice {
  readonly backend: 'webgpu' | 'cpu';
  readonly boundedSubmissions: boolean;
  readonly solverCheckpoints: boolean;
  /** Resolves when the browser invalidates its WebGPU device. */
  readonly lost?: Promise<{ message: string; reason: string }>;
  readonly maxModelsPerBatch: number;
  readonly maxDispatchesPerSubmission: number;
  readonly workingBytes: number;
  readonly bindingBytes: number;
  /** Current compiled portable kernels use a 64 x 1 x 1 workgroup. */
  readonly maxKernelElements: number;
  private readonly live = new Map<Buffer, Allocation>();
  private readonly free = new Map<string, Buffer[]>();
  private readonly pooled = new Set<Buffer>();
  private readonly history: ShadePlanReport[] = [];
  private pooledBytes = 0;
  private allocations = 0;
  private reuses = 0;
  private uploads = 0;
  private downloads = 0;
  private snapshotBytes = 0;
  private tail: Promise<unknown> = Promise.resolve();
  private closing = false;
  private closed = false;
  private disposal?: Promise<void>;
  private constructor(
    backend: 'webgpu' | 'cpu',
    private readonly runtime: Runtime | undefined,
    private readonly poolBytes: number,
    workingBytes: number,
    maxModels: number,
    maxDispatchesPerSubmission: number,
    private readonly onProgress: ShadeDeviceOptions['onProgress'],
    boundedSubmissions: boolean,
    solverCheckpoints: boolean,
  ) {
    this.backend = backend;
    this.boundedSubmissions = boundedSubmissions;
    this.solverCheckpoints = solverCheckpoints;
    const rawDevice = runtime?.device as
      { lost?: Promise<{ message?: string; reason?: string }> } | undefined;
    this.lost = rawDevice?.lost?.then((info) => ({
      message: info.message ?? 'WebGPU device was lost',
      reason: info.reason ?? 'unknown',
    }));
    this.workingBytes = workingBytes;
    this.maxModelsPerBatch = maxModels;
    this.maxDispatchesPerSubmission = maxDispatchesPerSubmission;
    const limits = runtime?.device as
      | {
          limits?: {
            maxStorageBufferBindingSize?: number;
            maxBufferSize?: number;
            maxComputeWorkgroupsPerDimension?: number;
          };
        }
      | undefined;
    this.bindingBytes = Math.min(
      limits?.limits?.maxStorageBufferBindingSize ?? 128 * 1024 ** 2,
      limits?.limits?.maxBufferSize ?? 256 * 1024 ** 2,
    );
    this.maxKernelElements = (limits?.limits?.maxComputeWorkgroupsPerDimension ?? 65535) * 64;
  }
  static async open(options: ShadeDeviceOptions = {}): Promise<ShadeDevice> {
    const backend = options.backend ?? 'webgpu',
      pool = options.poolBytes ?? 32 * 1024 ** 2;
    const working = options.workingBytes ?? 64 * 1024 ** 2,
      models = options.maxModelsPerBatch ?? (options.boundedSubmissions ? 8 : 64),
      maxDispatchesPerSubmission = options.maxDispatchesPerSubmission ?? 1;
    if (
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
      (options.onProgress !== undefined && typeof options.onProgress !== 'function')
    )
      throw new RangeError('Invalid TypeShade device options');
    if (backend === 'cpu' && options.gpuDevice !== undefined)
      throw new TypeError('CPU tier cannot take a GPUDevice');
    if (active || opening)
      throw new Error('One active TypeCV ShadeDevice is supported per context');
    opening = true;
    let rt: Runtime | undefined;
    try {
      if (backend === 'webgpu')
        rt = await createRuntime(
          options.gpuDevice ? { device: options.gpuDevice } : { prefer: ['webgpu'] },
        );
      configure({ runtime: rt ?? null, prefer: [backend] });
      const device = new ShadeDevice(
        backend,
        rt,
        pool,
        working,
        models,
        maxDispatchesPerSubmission,
        options.onProgress,
        options.boundedSubmissions ?? false,
        options.solverCheckpoints ?? options.boundedSubmissions ?? false,
      );
      active = device;
      return device;
    } catch (error) {
      rt?.destroy();
      configure({ runtime: null });
      configure({});
      throw error;
    } finally {
      opening = false;
    }
  }
  get reports(): readonly ShadePlanReport[] {
    return this.history.slice();
  }
  get statistics() {
    return {
      liveBuffers: this.live.size,
      liveBytes: [...this.live.values()].reduce((s, allocation) => s + allocation.bytes, 0),
      pooledBytes: this.pooledBytes,
      allocations: this.allocations,
      reuses: this.reuses,
      applicationUploadBytes: this.uploads,
      applicationReadbackBytes: this.downloads,
      /** Full arrays returned by Resident.read(); application counters may name a shorter prefix. */
      residentSnapshotBytes: this.snapshotBytes,
    };
  }
  /** @internal All jobs share the runtime's configuration and run in submission order. */
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
  private allocateTyped(kind: Allocation['kind'], length: number, values?: ScalarArray): Buffer {
    const bytes = length * (kind === 'f64' ? 8 : 4),
      key = `${kind}:${length}`;
    if (
      !Number.isSafeInteger(length) ||
      length < 1 ||
      bytes > this.bindingBytes ||
      (values !== undefined && values.length !== length)
    )
      throw new RangeError('Array exceeds the device storage binding limit');
    const list = this.free.get(key),
      old = list?.pop();
    if (old) {
      this.pooled.delete(old);
      this.pooledBytes -= bytes;
      if (!list!.length) this.free.delete(key);
      this.reuses++;
      if (values) {
        try {
          old.write(values);
          this.uploads += values.byteLength;
        } catch (error) {
          this.drop(old);
          throw error;
        }
      }
      return old;
    }
    // Zeroed allocation currently supports only f32/i32/u32 in TypeShade.
    // f64 still stays resident: its first use packs the initial host array once.
    const source = values ?? (kind === 'f64' ? new Float64Array(length) : undefined);
    const result = source ? resident(source) : resident(length, kind as 'f32' | 'u32');
    this.live.set(result, { length, bytes, kind, key });
    this.allocations++;
    if (source) this.uploads += source.byteLength;
    return result;
  }
  /** @internal */
  recycle(buffer: Buffer): void {
    const allocation = this.live.get(buffer);
    if (allocation === undefined || this.pooled.has(buffer)) return;
    if (this.pooledBytes + allocation.bytes > this.poolBytes) {
      this.drop(buffer);
      return;
    }
    const list = this.free.get(allocation.key) ?? [];
    list.push(buffer);
    this.free.set(allocation.key, list);
    this.pooled.add(buffer);
    this.pooledBytes += allocation.bytes;
  }
  /** @internal A failed plan's buffers must be destroyed, never reused. */
  drop(buffer: Buffer): void {
    const allocation = this.live.get(buffer);
    if (allocation === undefined) return;
    if (this.pooled.delete(buffer)) {
      this.pooledBytes -= allocation.bytes;
      const list = this.free.get(allocation.key)!.filter((b) => b !== buffer);
      if (list.length) this.free.set(allocation.key, list);
      else this.free.delete(allocation.key);
    }
    this.live.delete(buffer);
    buffer.destroy();
  }
  /** @internal */
  async plan(body: Parameters<typeof fuse>[0]): Promise<FuseReport> {
    const report = await fuse(body);
    if (report.tier !== this.backend) throw new Error(`Unexpected execution tier: ${report.tier}`);
    this.history.push(report);
    if (this.history.length > 64) this.history.shift();
    return report;
  }
  /** @internal Solver state stays resident across submission boundaries; no numerical readback. */
  async planSteps(steps: Iterable<GpuStep>): Promise<ShadePlanReport> {
    if (!this.boundedSubmissions && !this.solverCheckpoints)
      return this.plan(() => {
        for (const [, run] of steps) run();
      });
    const reports: ShadePlanReport[] = [];
    const rawDevice = this.runtime?.device as
      { queue: { onSubmittedWorkDone(): Promise<void> } } | undefined;
    const iterator = steps[Symbol.iterator]();
    let done = false;
    while (!done) {
      const chunk: GpuStep[] = [];
      while (!this.boundedSubmissions || chunk.length < this.maxDispatchesPerSubmission) {
        const next = iterator.next();
        if (next.done) {
          done = true;
          break;
        }
        chunk.push(next.value);
        // An after hook can read GPU state that determines the next yielded step.
        // Do not advance the generator until that hook has completed.
        if (next.value[2]) break;
      }
      if (!chunk.length) break;
      reports.push(
        await this.plan(() => {
          for (const [kernel, run] of chunk) {
            this.onProgress?.({ kernel, phase: 'start' });
            run();
          }
        }),
      );
      // Awaiting fuse only submits: it is not a GPU completion fence. A queue
      // fence keeps each bounded group out of a long-running command stream.
      if (this.boundedSubmissions && rawDevice) await rawDevice.queue.onSubmittedWorkDone();
      for (const [kernel, , after] of chunk) {
        await after?.();
        this.onProgress?.({ kernel, phase: 'complete' });
      }
    }
    return {
      tier: this.backend,
      dispatches: reports.reduce((n, r) => n + r.dispatches, 0),
      submissions: reports.reduce((n, r) => n + r.submissions, 0),
      groups: reports.flatMap((r) => r.groups),
    };
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
      for (const b of this.live.keys()) b.destroy();
      this.live.clear();
      this.free.clear();
      this.pooled.clear();
      this.pooledBytes = 0;
      this.runtime?.destroy();
      configure({ runtime: null });
      configure({});
      this.closed = true;
      if (active === this) active = undefined;
    })();
    return this.disposal;
  }
}
