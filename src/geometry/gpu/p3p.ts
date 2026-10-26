import type { Resident } from 'typeshade/runtime';
import { ShadeDevice, type ShadePlanReport } from '../../device/gpu/device.ts';
import { solveP3p } from './p3p.shade.ts';

export const GpuP3pStatus = {
  success: 0,
  degenerateWorld: 1,
  degenerateBearings: 2,
  degeneratePolynomial: 3,
  noValidatedPose: 4,
  unresolvedBranches: 5,
  illConditionedWorld: 6,
} as const;

export interface GpuP3pResult {
  /** Four slots per problem, each row-major R[9] followed by t[3]. Unused slots are zero. */
  readonly poses: Float64Array;
  readonly valid: Uint32Array;
  readonly counts: Uint32Array;
  readonly status: Uint32Array;
}
type Outputs = {
  poses: Resident<Float64Array>;
  valid: Resident<Uint32Array>;
  counts: Resident<Uint32Array>;
  status: Resident<Uint32Array>;
};

/** GPU-generated P3P candidates retained on the device until read() or dispose(). */
export class GpuP3pBatch {
  private closed = false;
  private disposal?: Promise<void>;
  private constructor(
    private readonly device: ShadeDevice,
    private readonly outputs: Outputs,
    readonly count: number,
    readonly report: ShadePlanReport,
  ) {}

  /** @internal */
  static create(device: ShadeDevice, packed: Float64Array): Promise<GpuP3pBatch> {
    const count = packed.length / 18;
    if (
      !(packed instanceof Float64Array) ||
      !Number.isInteger(count) ||
      count < 1 ||
      count > device.maxKernelElements / 2 ||
      count * 552 > device.workingBytes ||
      count * 48 * 8 > device.bindingBytes ||
      !packed.every(
        (x) => Number.isFinite(x) && Math.abs(x) <= 1e9 && (x === 0 || Math.abs(x) >= 1e-30),
      )
    )
      throw new RangeError(
        'P3P requires packed 18-value problems within the device budget and validated f64 range',
      );
    const snapshot = packed.slice();
    return device.enqueue(async () => {
      const owned: (Resident<Float64Array> | Resident<Uint32Array>)[] = [];
      let successful = false;
      try {
        const input = device.allocateF64(snapshot.length, snapshot);
        owned.push(input);
        const poses = device.allocateF64(count * 48);
        owned.push(poses);
        const valid = device.allocateU32(count * 4);
        owned.push(valid);
        const counts = device.allocateU32(count);
        owned.push(counts);
        const status = device.allocateU32(count);
        owned.push(status);
        const report = await device.plan(() =>
          solveP3p({ problems: input, poses, valid, counts, status }, Math.ceil(count / 32)),
        );
        device.recycle(input);
        successful = true;
        return new GpuP3pBatch(device, { poses, valid, counts, status }, count, report);
      } finally {
        if (!successful) for (const buffer of owned) device.drop(buffer);
      }
    });
  }

  /** Explicit readback into caller-owned snapshots. Solving itself reads no candidates back. */
  read(): Promise<GpuP3pResult> {
    if (this.closed) return Promise.reject(new Error('GpuP3pBatch is disposed'));
    return this.device.enqueue(async () => {
      try {
        return {
          poses: (await this.device.read(this.outputs.poses, this.count * 48)).slice(),
          valid: (await this.device.read(this.outputs.valid, this.count * 4)).slice(),
          counts: (await this.device.read(this.outputs.counts, this.count)).slice(),
          status: (await this.device.read(this.outputs.status, this.count)).slice(),
        };
      } catch (error) {
        this.closed = true;
        for (const buffer of Object.values(this.outputs)) this.device.drop(buffer);
        throw error;
      }
    });
  }

  dispose(): Promise<void> {
    if (this.disposal) return this.disposal;
    this.closed = true;
    this.disposal = (async () => {
      await this.device.flush();
      for (const buffer of Object.values(this.outputs)) this.device.recycle(buffer);
    })();
    return this.disposal;
  }
}

/**
 * Batched Grunert P3P: each 18-value record is three world XYZs then three camera bearings.
 * All root isolation, distance refinement and pose validation run in the selected TypeShade tier.
 * GPU f64 uses ~48-bit significands and the f32 exponent range. No automatic CPU fallback.
 */
export function solveP3pBatchGpu(device: ShadeDevice, packed: Float64Array): Promise<GpuP3pBatch> {
  return GpuP3pBatch.create(device, packed);
}
