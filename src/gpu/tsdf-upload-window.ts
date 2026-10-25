/** Logical view payload admitted since the last confirmed GPU completion, not driver/RSS bytes. */
export interface TsdfUploadWindowReport {
  readonly limitBytes: number;
  readonly peakSubmittedBytes: number;
  readonly drainCount: number;
}

/**
 * Bound view uploads before their synchronous Resident.write snapshots are made. An oversized
 * single view expands the effective limit rather than splitting its pixels or dropping data.
 * Calls must be sequential; the helper never queues write callbacks or retains view arrays.
 */
export class TsdfUploadWindow {
  private limit: number;
  private pending = 0;
  private peak = 0;
  private drains = 0;
  private running = false;
  private readonly drain: () => Promise<void>;
  private readonly signal?: AbortSignal;

  constructor(limitBytes: number, drain: () => Promise<void>, signal?: AbortSignal) {
    if (!Number.isSafeInteger(limitBytes) || limitBytes < 1)
      throw new RangeError('TSDF upload limit must be a positive safe integer');
    if (typeof drain !== 'function') throw new TypeError('TSDF upload drain must be a function');
    this.limit = limitBytes;
    this.drain = drain;
    this.signal = signal;
  }

  get pendingBytes(): number {
    return this.pending;
  }

  get report(): TsdfUploadWindowReport {
    return { limitBytes: this.limit, peakSubmittedBytes: this.peak, drainCount: this.drains };
  }

  /** A real readback/fence or a proven completed final GPU submission clears the watermark. */
  completed(): void {
    this.pending = 0;
  }

  /** Abort promptly while keeping the underlying fence rejection handled. */
  private async waitForDrain(): Promise<void> {
    const waiting = Promise.resolve(this.drain());
    const signal = this.signal;
    if (!signal) return waiting;
    return new Promise<void>((resolve, reject) => {
      let settled = false;
      const finish = (error?: unknown, failed = false) => {
        if (settled) return;
        settled = true;
        signal.removeEventListener('abort', abort);
        if (failed) reject(error);
        else resolve();
      };
      const abort = () => finish(signal.reason, true);
      signal.addEventListener('abort', abort, { once: true });
      waiting.then(
        () => finish(),
        (error: unknown) => finish(error, true),
      );
      // The drain callback may itself have synchronously aborted before the listener was added.
      if (signal.aborted) abort();
    });
  }

  /** Reserve the complete payload before invoking writes, including callbacks that fail midway. */
  async run(bytes: number, write: () => void): Promise<void> {
    if (!Number.isSafeInteger(bytes) || bytes < 0)
      throw new RangeError('TSDF upload bytes must be a nonnegative safe integer');
    if (typeof write !== 'function') throw new TypeError('TSDF upload writes must be a function');
    if (this.running) throw new TypeError('TSDF upload admission must be sequential');
    this.signal?.throwIfAborted();
    this.running = true;
    try {
      this.limit = Math.max(this.limit, bytes);
      // Subtraction avoids overflowing safe-integer addition when callers use a very large cap.
      if (this.pending > this.limit - bytes) {
        this.drains++;
        await this.waitForDrain();
        this.pending = 0;
      }
      this.signal?.throwIfAborted();
      this.pending += bytes;
      this.peak = Math.max(this.peak, this.pending);
      write();
    } finally {
      this.running = false;
    }
  }
}
