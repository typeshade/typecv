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
import { ShadeDevice } from './device.ts';
import { linearizeInverseDepth } from './inverse-depth.shade.ts';

/** Measurements/Jacobians on the selected device; reduced solve and covariance explicitly host f64. */
export async function linearizeInverseDepthGpu(
  device: ShadeDevice,
  job: InverseDepthMeasurementJob,
): Promise<readonly (InverseDepthLinearization | undefined)[]> {
  const count = job.measurements.length;
  if (!count) return [];
  const input = new Float64Array(count * 34);
  job.measurements.forEach(({ point, view }, i) => {
    const at = i * 34,
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
  });
  return device.enqueue(async () => {
    const bytes = 8 * count * (34 + 23);
    device.reserveWorkingBytes(bytes);
    if (count > device.maxKernelElements || count * 34 * 8 > device.bindingBytes)
      throw new RangeError('Inverse-depth measurement binding/dispatch budget exceeded');
    const measurements = device.allocateF64(count * 34);
    let output: Resident<Float64Array> | undefined,
      successful = false;
    try {
      output = device.allocateF64(count * 23);
      const activeOutput = output;
      measurements.write(input);
      await device.planSteps([
        [
          'linearizeInverseDepth',
          () =>
            linearizeInverseDepth(
              {
                idMeasurements: measurements,
                idLinearizations: activeOutput,
                idControl: { count },
              },
              Math.ceil(count / 64),
            ),
        ],
      ]);
      const result = await device.read(output, count * 23);
      successful = true;
      return job.measurements.map((_, i) => {
        const at = i * 23;
        if (result[at] !== 1) return undefined;
        return {
          pixel: [result[at + 1]!, result[at + 2]!] as const,
          point: Float64Array.from(result.subarray(at + 3, at + 9)),
          pose: Float64Array.from(result.subarray(at + 9, at + 21)),
          focal: Float64Array.from(result.subarray(at + 21, at + 23)),
        };
      });
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
export async function bundleAdjustInverseDepthGpu(
  device: ShadeDevice,
  model: SparseReconstruction,
  constraints: readonly InverseDepthConstraint[] = [],
  options: InverseDepthBundleOptions = {},
) {
  const machine = inverseDepthBundleSteps(model, constraints, options);
  let step = machine.next();
  while (!step.done) step = machine.next(await linearizeInverseDepthGpu(device, step.value));
  return {
    ...step.value,
    diagnostics: {
      ...step.value.diagnostics,
      backend: `${device.backend}-jacobians/host-f64-Schur`,
    },
  };
}
