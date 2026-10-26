import type { CameraIntrinsics } from '../sfm/two-view.ts';
import type { Resident } from 'typeshade/runtime';
import { validateCamera } from '../geometry/camera.ts';
import { ShadeDevice, type ShadePlanReport } from '../device/gpu/device.ts';
import {
  normalizeMatches,
  sampsonBatch,
  inlierFlags,
  msacCosts,
  blockStats,
  modelStats,
  selectBest,
  winningMask,
} from './epipolar.shade.ts';

type Buffer = Resident<Float32Array>;
export interface EpipolarScore {
  readonly bestIndex: number;
  readonly inlierCount: number;
  readonly msacCost: number;
  readonly mask?: Uint8Array;
  /** Debug-only counts then costs; explicitly requests another small readback. */
  readonly modelStatistics?: Float32Array;
  readonly report: ShadePlanReport;
}

/** Undistorted observations uploaded/normalized once and retained across hypothesis batches. */
export class EpipolarSession {
  readonly count: number;
  readonly capacity: number;
  private closed = false;
  private disposal?: Promise<void>;
  private constructor(
    private readonly device: ShadeDevice,
    private readonly normalized: Buffer,
    count: number,
  ) {
    this.count = count;
    const blocks = Math.ceil(count / 256);
    this.capacity = Math.min(
      device.maxModelsPerBatch,
      Math.floor(device.bindingBytes / (count * 4)),
      Math.floor(device.bindingBytes / 36),
      Math.floor(device.maxKernelElements / count),
      Math.floor(device.maxKernelElements / (2 * blocks)),
      Math.floor((device.workingBytes - 4 * count - 12) / (8 * count + 8 * blocks + 44)),
    );
  }
  static create(
    device: ShadeDevice,
    pixels: Float64Array | Float32Array,
    cameraA: CameraIntrinsics,
    cameraB: CameraIntrinsics,
  ): Promise<EpipolarSession> {
    const a = { ...cameraA },
      b = { ...cameraB };
    validateCamera(a);
    validateCamera(b);
    if (
      [a, b].some((k) => {
        const d = k as CameraIntrinsics & { k1?: number; k2?: number };
        return (d.k1 ?? 0) !== 0 || (d.k2 ?? 0) !== 0;
      })
    )
      throw new RangeError('EpipolarSession requires undistorted pinhole pixels');
    if (
      pixels.length < 4 ||
      pixels.length % 4 ||
      pixels.length / 4 > 2 ** 24 ||
      pixels.length > device.maxKernelElements ||
      !pixels.every(Number.isFinite)
    )
      throw new RangeError('Invalid packed matches');
    const source = Float32Array.from(pixels);
    if (
      !source.every(Number.isFinite) ||
      ![a.fx, a.fy, a.cx, a.cy, b.fx, b.fy, b.cx, b.cy].map(Math.fround).every(Number.isFinite) ||
      Math.fround(a.fx) <= 0 ||
      Math.fround(a.fy) <= 0 ||
      Math.fround(b.fx) <= 0 ||
      Math.fround(b.fy) <= 0
    )
      throw new RangeError('Camera/matches cannot be represented in f32');
    for (let i = 0; i < pixels.length; i++) {
      const lane = i % 4,
        k = lane < 2 ? a : b,
        coord = (pixels[i]! - (lane % 2 ? k.cy : k.cx)) / (lane % 2 ? k.fy : k.fx);
      const focal = Math.fround(lane % 2 ? k.fy : k.fx);
      const deviceCoord = Math.fround(
        Math.fround(source[i]! - Math.fround(lane % 2 ? k.cy : k.cx)) / focal,
      );
      if (
        focal < 2 ** -126 ||
        !Number.isFinite(coord) ||
        Math.abs(coord) > 1e4 ||
        !Number.isFinite(deviceCoord) ||
        Math.abs(deviceCoord) > 1e4
      )
        throw new RangeError('Normalized coordinate exceeds the validated f32 range');
    }
    return device.enqueue(async () => {
      const input = device.allocate(source.length, source);
      let output: Buffer | undefined;
      try {
        output = device.allocate(source.length);
        await device.plan(() =>
          normalizeMatches(output!, input, a.fx, a.fy, a.cx, a.cy, b.fx, b.fy, b.cx, b.cy),
        );
        const session = new EpipolarSession(device, output, source.length / 4);
        if (session.capacity < 1) throw new RangeError('Working budget cannot hold one hypothesis');
        return session;
      } catch (error) {
        if (output) device.drop(output);
        throw error;
      } finally {
        device.drop(input);
      }
    });
  }
  score(
    models: Float64Array | Float32Array,
    threshold: number,
    options: { readMask?: boolean; readStatistics?: boolean } = {},
  ): Promise<EpipolarScore> {
    if (this.closed) return Promise.reject(new Error('EpipolarSession is disposed'));
    const readMask = options.readMask === true,
      readStatistics = options.readStatistics === true;
    const n = models.length / 9,
      squared = Math.fround(threshold * threshold);
    if (!Number.isInteger(n) || n < 1 || n > this.capacity || !models.every(Number.isFinite))
      throw new RangeError(`Pass 1..${this.capacity} finite E matrices per batch`);
    if (!Number.isFinite(threshold) || threshold <= 0 || threshold > 1e4 || squared < 2 ** -126)
      throw new RangeError('Invalid f32 Sampson threshold');
    // Unit E makes the denominator guard scale-independent. Zero E is an invalid all-outlier hypothesis.
    const packed = new Float32Array(models.length);
    for (let m = 0; m < n; m++) {
      const E = Array.from(models.subarray(9 * m, 9 * m + 9)),
        scale = Math.max(...E.map(Math.abs));
      if (scale === 0) continue;
      const unit = E.map((x) => x / scale),
        norm = Math.hypot(...unit);
      for (let j = 0; j < 9; j++) packed[m * 9 + j] = unit[j]! / norm;
    }
    return this.device.enqueue(async () => {
      const d = this.device,
        blocks = Math.ceil(this.count / 256),
        owned: Buffer[] = [];
      const acquire = (size: number, values?: Float32Array) => {
        const r = d.allocate(size, values);
        owned.push(r);
        return r;
      };
      let successful = false;
      try {
        const E = acquire(packed.length, packed),
          flags = acquire(n * this.count),
          costs = acquire(n * this.count),
          partial = acquire(2 * n * blocks),
          stats = acquire(2 * n),
          best = acquire(3),
          mask = readMask ? acquire(this.count) : undefined;
        const report = await d.plan((scope) => {
          const errors = scope.scratch(n * this.count);
          sampsonBatch(errors, this.normalized, E, this.count);
          inlierFlags(flags, errors, squared);
          msacCosts(costs, errors, squared);
          blockStats(partial, flags, costs, this.count, blocks, 256);
          modelStats(stats, partial, blocks);
          selectBest(best, stats, n);
          if (mask) winningMask(mask, flags, best, this.count);
        });
        const summary = await d.read(best, 3);
        const result: EpipolarScore = {
          bestIndex: summary[0]!,
          inlierCount: summary[1]!,
          msacCost: summary[2]!,
          report,
          ...(mask ? { mask: Uint8Array.from(await d.read(mask, this.count)) } : {}),
          ...(readStatistics ? { modelStatistics: await d.read(stats, 2 * n) } : {}),
        };
        successful = true;
        return result;
      } finally {
        for (const b of owned) successful ? d.recycle(b) : d.drop(b);
      }
    });
  }
  dispose(): Promise<void> {
    if (this.disposal) return this.disposal;
    this.closed = true;
    this.disposal = this.device.flush().then(() => this.device.drop(this.normalized));
    return this.disposal;
  }
}
