/** Logical resident storage only: excludes textures, staging buffers and physical VRAM. */
export interface DeviceDiagnostics {
  readonly disposalRequested: boolean;
  /** Null when TypeShade owns an opaque GPUDevice whose destroy call TypeCV cannot observe. */
  readonly destroyRequested: boolean | null;
  readonly errorEventsAvailable: boolean;
  readonly recentErrors: readonly { readonly name: string; readonly message: string }[];
  readonly memory: {
    readonly liveBytes: number;
    readonly pooledBytes: number;
    readonly peakLiveBytes: number;
    readonly peakActiveBytes: number;
    readonly workingBytes: number;
    readonly pendingReads: number;
  };
}

/** @internal Bounded host-side diagnostics; never submits GPU work or suppresses native errors. */
export class DeviceDiagnosticTracker {
  disposalRequested = false;
  destroyRequested: boolean | null;
  private peakLiveBytes = 0;
  private peakActiveBytes = 0;
  private readonly recentErrors: { name: string; message: string }[] = [];
  private readonly target?: EventTarget;
  private readonly statistics: () => { liveBytes: number; pooledBytes: number };
  private readonly workingBytes: number;
  private readonly pendingReads: () => number;
  private readonly onError = (event: Event) => {
    const error = (event as Event & { error?: { name?: string; message?: string } }).error;
    this.recentErrors.push({
      name: String(error?.name ?? error?.constructor?.name ?? 'GPUError').slice(0, 64),
      message: String(error?.message ?? '').slice(0, 400),
    });
    if (this.recentErrors.length > 4) this.recentErrors.shift();
  };

  constructor(
    statistics: () => { liveBytes: number; pooledBytes: number },
    workingBytes: number,
    pendingReads: () => number,
    gpuDevice?: object,
    opaqueOwner = false,
  ) {
    this.statistics = statistics;
    this.workingBytes = workingBytes;
    this.pendingReads = pendingReads;
    this.destroyRequested = opaqueOwner ? null : false;
    const target = gpuDevice as Partial<EventTarget> | undefined;
    if (
      typeof target?.addEventListener === 'function' &&
      typeof target.removeEventListener === 'function'
    ) {
      this.target = target as EventTarget;
      this.target.addEventListener('uncapturederror', this.onError);
    }
  }

  /** Observe successful resident allocations, including arrays taken back out of the pool. */
  observe(): void {
    const { liveBytes, pooledBytes } = this.statistics();
    this.peakLiveBytes = Math.max(this.peakLiveBytes, liveBytes);
    this.peakActiveBytes = Math.max(this.peakActiveBytes, liveBytes - pooledBytes);
  }

  snapshot(): DeviceDiagnostics {
    this.observe();
    const { liveBytes, pooledBytes } = this.statistics();
    return {
      disposalRequested: this.disposalRequested,
      destroyRequested: this.destroyRequested,
      errorEventsAvailable: this.target !== undefined,
      recentErrors: this.recentErrors.map((error) => ({ ...error })),
      memory: {
        liveBytes,
        pooledBytes,
        peakLiveBytes: this.peakLiveBytes,
        peakActiveBytes: this.peakActiveBytes,
        workingBytes: this.workingBytes,
        pendingReads: this.pendingReads(),
      },
    };
  }

  detach(): void {
    this.target?.removeEventListener('uncapturederror', this.onError);
  }
}
