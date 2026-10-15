import { configure, createRuntime, resident, type Resident, type Runtime } from 'typeshade/runtime';
import { fuse, type FuseReport } from 'typeshade/fuse';

export interface ShadeDeviceOptions {
  /** WebGPU is required by default. CPU is an explicit differential-testing tier. */
  readonly backend?: 'webgpu' | 'cpu';
  readonly gpuDevice?: object;
  readonly poolBytes?: number;
  readonly workingBytes?: number;
  readonly maxModelsPerBatch?: number;
}
type Buffer = Resident<Float32Array>;
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
  readonly maxModelsPerBatch: number;
  readonly workingBytes: number;
  readonly bindingBytes: number;
  /** Current compiled portable kernels use a 64 x 1 x 1 workgroup. */
  readonly maxKernelElements: number;
  private readonly live = new Map<Buffer, number>();
  private readonly free = new Map<number, Buffer[]>();
  private readonly pooled = new Set<Buffer>();
  private readonly history: ShadePlanReport[] = [];
  private pooledBytes = 0;
  private allocations = 0;
  private reuses = 0;
  private uploads = 0;
  private downloads = 0;
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
  ) {
    this.backend = backend;
    this.workingBytes = workingBytes;
    this.maxModelsPerBatch = maxModels;
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
      models = options.maxModelsPerBatch ?? 64;
    if (
      !['webgpu', 'cpu'].includes(backend) ||
      !Number.isSafeInteger(pool) ||
      pool < 0 ||
      !Number.isSafeInteger(working) ||
      working < 4096 ||
      !Number.isSafeInteger(models) ||
      models < 1 ||
      models > 2 ** 24
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
      const device = new ShadeDevice(backend, rt, pool, working, models);
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
      liveBytes: [...this.live.values()].reduce((s, n) => s + 4 * n, 0),
      pooledBytes: this.pooledBytes,
      allocations: this.allocations,
      reuses: this.reuses,
      applicationUploadBytes: this.uploads,
      applicationReadbackBytes: this.downloads,
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
  allocate(length: number, values?: Float32Array): Buffer {
    if (!Number.isSafeInteger(length) || length < 1 || length * 4 > this.bindingBytes)
      throw new RangeError('Array exceeds the device storage binding limit');
    const list = this.free.get(length),
      old = list?.pop();
    if (old) {
      this.pooled.delete(old);
      this.pooledBytes -= length * 4;
      if (!list!.length) this.free.delete(length);
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
    const result = values ? resident(values) : resident(length, 'f32');
    this.live.set(result, length);
    this.allocations++;
    if (values) this.uploads += values.byteLength;
    return result;
  }
  /** @internal */
  recycle(buffer: Buffer): void {
    const n = this.live.get(buffer);
    if (n === undefined || this.pooled.has(buffer)) return;
    if (this.pooledBytes + 4 * n > this.poolBytes) {
      this.drop(buffer);
      return;
    }
    const list = this.free.get(n) ?? [];
    list.push(buffer);
    this.free.set(n, list);
    this.pooled.add(buffer);
    this.pooledBytes += 4 * n;
  }
  /** @internal A failed plan's buffers must be destroyed, never reused. */
  drop(buffer: Buffer): void {
    const n = this.live.get(buffer);
    if (n === undefined) return;
    if (this.pooled.delete(buffer)) {
      this.pooledBytes -= 4 * n;
      const list = this.free.get(n)!.filter((b) => b !== buffer);
      if (list.length) this.free.set(n, list);
      else this.free.delete(n);
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
  /** @internal The counts describe application arrays, not runtime uniforms or staging allocation. */
  async read(buffer: Buffer, length: number): Promise<Float32Array> {
    const data = await buffer.read();
    this.downloads += 4 * length;
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
