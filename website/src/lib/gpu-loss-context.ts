import type { DeviceDiagnostics } from '../../../src/gpu/device-diagnostics.ts';

type RequestKind = 'start' | 'frame' | 'finish';
type RequestInfo = { type: RequestKind; frameId?: number; ageMs: number };

/** Optional, bounded context on a v6 failure; absent on older records. */
export interface GpuLossContext {
  readonly state: 'active' | 'idle';
  readonly request: RequestInfo | null;
  readonly lastCompletedRequest: RequestInfo | null;
  readonly lastKernel: {
    readonly name: string;
    /** Host progress only: submitted does not promise that the GPU finished. */
    readonly phase: 'started' | 'submitted';
    readonly ageMs: number;
    readonly inCurrentRequest: boolean;
  } | null;
  readonly device?: DeviceDiagnostics;
}

/** Keeps idle losses from being attributed to a request or kernel that already returned. */
export class GpuRequestContext {
  private serial = 0;
  private request?: { type: RequestKind; frameId?: number; at: number };
  private completed?: { type: RequestKind; frameId?: number; at: number };
  private kernel?: {
    name: string;
    phase: 'started' | 'submitted';
    at: number;
    serial: number;
  };

  private readonly now: () => number;

  constructor(now = () => performance.now()) {
    this.now = now;
  }

  begin(type: RequestKind, frameId?: number): void {
    this.serial++;
    this.request = { type, ...(frameId === undefined ? {} : { frameId }), at: this.now() };
  }

  end(): void {
    if (!this.request) return;
    this.completed = { ...this.request, at: this.now() };
    this.request = undefined;
  }

  progress(name: string, phase: string): void {
    this.kernel = {
      name: name.slice(0, 64),
      phase: phase === 'start' ? 'started' : 'submitted',
      at: this.now(),
      serial: this.serial,
    };
  }

  location(operation: string): { operation: string; kernel: string } {
    return {
      operation: this.request ? operation.slice(0, 200) : 'waiting for next request',
      kernel: this.request && this.kernel?.serial === this.serial ? this.kernel.name : '',
    };
  }

  snapshot(device?: DeviceDiagnostics): GpuLossContext {
    const now = this.now();
    const requestInfo = (request: typeof this.request): RequestInfo | null =>
      request
        ? {
            type: request.type,
            ...(request.frameId === undefined ? {} : { frameId: request.frameId }),
            ageMs: Math.max(0, now - request.at),
          }
        : null;
    return {
      state: this.request ? 'active' : 'idle',
      request: requestInfo(this.request),
      lastCompletedRequest: requestInfo(this.completed),
      lastKernel: this.kernel
        ? {
            name: this.kernel.name,
            phase: this.kernel.phase,
            ageMs: Math.max(0, now - this.kernel.at),
            inCurrentRequest: !!this.request && this.kernel.serial === this.serial,
          }
        : null,
      ...(device ? { device } : {}),
    };
  }
}

const object = (value: unknown): value is Record<string, unknown> =>
  !!value && typeof value === 'object' && !Array.isArray(value);
const count = (value: unknown): value is number =>
  Number.isSafeInteger(value) && (value as number) >= 0;
const age = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value) && value >= 0;
const text = (value: unknown, limit: number): value is string =>
  typeof value === 'string' && value.length <= limit;

/** Validate optional context before the collector accepts it. */
export function gpuLossContextValid(value: unknown): value is GpuLossContext {
  if (!object(value)) return false;
  const request = (r: unknown) =>
    r === null ||
    (object(r) &&
      (r.type === 'start' || r.type === 'frame' || r.type === 'finish') &&
      age(r.ageMs) &&
      (r.frameId === undefined || count(r.frameId)));
  if (
    (value.state !== 'active' && value.state !== 'idle') ||
    !request(value.request) ||
    !request(value.lastCompletedRequest) ||
    (value.state === 'idle') !== (value.request === null)
  )
    return false;
  const k = value.lastKernel;
  if (
    k !== null &&
    !(
      object(k) &&
      text(k.name, 64) &&
      (k.phase === 'started' || k.phase === 'submitted') &&
      age(k.ageMs) &&
      typeof k.inCurrentRequest === 'boolean' &&
      (value.state === 'active' || !k.inCurrentRequest)
    )
  )
    return false;
  const d = value.device;
  if (d === undefined) return true;
  if (
    !object(d) ||
    typeof d.disposalRequested !== 'boolean' ||
    !(d.destroyRequested === null || typeof d.destroyRequested === 'boolean') ||
    typeof d.errorEventsAvailable !== 'boolean' ||
    !Array.isArray(d.recentErrors) ||
    d.recentErrors.length > 4 ||
    !d.recentErrors.every((e) => object(e) && text(e.name, 64) && text(e.message, 400)) ||
    !object(d.memory)
  )
    return false;
  const memory = d.memory;
  return [
    'liveBytes',
    'pooledBytes',
    'peakLiveBytes',
    'peakActiveBytes',
    'workingBytes',
    'pendingReads',
  ].every((key) => count(memory[key]));
}
