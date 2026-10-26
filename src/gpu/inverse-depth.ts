import type { Resident } from 'typeshade/runtime';
import { radialLimitSquared } from '../geometry/camera.ts';
import type { InverseDepthLinearization } from '../geometry/inverse-depth.ts';
import {
  inverseDepthBundleSteps,
  type InverseDepthBundleOptions,
  type InverseDepthConstraint,
  type InverseDepthMeasurementJob,
} from '../reconstruction/inverse-depth-bundle.ts';
import type { SparseReconstruction } from '../reconstruction/types.ts';
import { ShadeDevice } from '../device/gpu/device.ts';
import { linearizeInverseDepth } from './inverse-depth.shade.ts';

/** Measurement splitting changes residency only. Every raw row is evaluated exactly once. */
async function linearizeBatches(
  device: ShadeDevice,
  job: InverseDepthMeasurementJob,
  requested?: number,
) {
  if (requested !== undefined && (!Number.isSafeInteger(requested) || requested < 1))
    throw new RangeError('Invalid inverse-depth measurement batch size');
  const snapshot = structuredClone(job.measurements),
    count = snapshot.length;
  if (!count) return { rows: [], batches: 0, batchSize: 0 };
  return device.enqueue(async () => {
    const stats = device.statistics,
      available = device.workingBytes - (stats.liveBytes - stats.pooledBytes),
      batch = Math.min(
        count,
        requested ?? count,
        Math.floor(available / (8 * 57)),
        Math.floor(device.bindingBytes / (8 * 34)),
        device.maxKernelElements,
      );
    if (!(batch >= 1))
      throw new RangeError('Inverse-depth measurement budget cannot fit one observation');
    device.reserveWorkingBytes(8 * 57 * batch);
    const measurements = device.allocateF64(batch * 34);
    let output: Resident<Float64Array> | undefined,
      successful = false,
      batches = 0;
    const rows: (InverseDepthLinearization | undefined)[] = [],
      input = new Float64Array(batch * 34);
    try {
      output = device.allocateF64(batch * 23);
      const activeOutput = output;
      for (let first = 0; first < count; first += batch) {
        const n = Math.min(batch, count - first);
        input.fill(0);
        for (let i = 0; i < n; i++) {
          const { point, view } = snapshot[first + i]!,
            at = i * 34,
            k = view.camera;
          input.set(point.parameters, at);
          input.set(point.reference.R, at + 3);
          input.set(point.reference.t, at + 12);
          input.set(view.pose.R, at + 15);
          input.set(view.pose.t, at + 24);
          input.set(
            [
              k.fx,
              k.fy,
              k.cx,
              k.cy,
              k.k1 ?? 0,
              k.k2 ?? 0,
              Math.min(1e30, radialLimitSquared(k.k1 ?? 0, k.k2 ?? 0)),
            ],
            at + 27,
          );
        }
        measurements.write(input);
        await device.planSteps([
          [
            'linearizeInverseDepth',
            () =>
              linearizeInverseDepth(
                {
                  idMeasurements: measurements,
                  idLinearizations: activeOutput,
                  idControl: { count: n },
                },
                Math.ceil(n / 64),
              ),
          ],
        ]);
        const result = await device.read(output, batch * 23);
        batches++;
        for (let i = 0; i < n; i++) {
          const at = i * 23;
          rows.push(
            result[at] !== 1
              ? undefined
              : {
                  pixel: [result[at + 1]!, result[at + 2]!] as const,
                  point: Float64Array.from(result.subarray(at + 3, at + 9)),
                  pose: Float64Array.from(result.subarray(at + 9, at + 21)),
                  focal: Float64Array.from(result.subarray(at + 21, at + 23)),
                },
          );
        }
      }
      successful = true;
      return { rows, batches, batchSize: batch };
    } finally {
      if (successful) {
        device.recycle(measurements);
        if (output) device.recycle(output);
      } else {
        device.drop(measurements);
        if (output) device.drop(output);
      }
    }
  });
}
/** Measurements/Jacobians on the selected device; reduced solve and covariance explicitly host f64. */
export async function linearizeInverseDepthGpu(
  device: ShadeDevice,
  job: InverseDepthMeasurementJob,
  batchSize?: number,
): Promise<readonly (InverseDepthLinearization | undefined)[]> {
  return (await linearizeBatches(device, job, batchSize)).rows;
}
export async function bundleAdjustInverseDepthGpu(
  device: ShadeDevice,
  model: SparseReconstruction,
  constraints: readonly InverseDepthConstraint[] = [],
  options: InverseDepthBundleOptions = {},
) {
  const requestedBatch = options.measurementBatchSize,
    machine = inverseDepthBundleSteps(model, constraints, options);
  let step = machine.next(),
    batches = 0,
    batchSize = 0;
  while (!step.done) {
    const result = await linearizeBatches(device, step.value, requestedBatch);
    batches += result.batches;
    batchSize = Math.max(batchSize, result.batchSize);
    step = machine.next(result.rows);
  }
  return {
    ...step.value,
    diagnostics: {
      ...step.value.diagnostics,
      backend: `${device.backend}-jacobians/host-f64-Schur`,
      measurementBatches: batches,
      measurementBatchSize: batchSize,
    },
  };
}
