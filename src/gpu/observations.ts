import type { Resident } from 'typeshade/runtime';
import type { SparseReconstruction } from '../reconstruction/types.ts';
import { validateReconstruction } from '../reconstruction/bundle-adjustment.ts';
import { ShadeDevice } from './device.ts';
import { finiteGpuArray, validateGpuCamera, type GpuGeometryExecution } from './pnp.ts';
import { projectObservations, observationSummary } from './triangulation.shade.ts';

export interface GpuObservationResult {
  readonly masks: readonly Uint8Array[];
  readonly inlierCount: number;
  readonly meanErrorPx: number;
  readonly execution: GpuGeometryExecution;
}

/** Project/filter fixed scene observations on the device; returns masks in landmark order. */
export function evaluateObservationsGpu(
  device: ShadeDevice,
  model: SparseReconstruction,
  maxErrorPx = 3,
): Promise<GpuObservationResult> {
  validateReconstruction(model);
  if (!Number.isFinite(maxErrorPx) || maxErrorPx < 1e-12 || maxErrorPx > 1e6)
    throw new RangeError('Invalid GPU observation threshold');
  const count = model.landmarks.reduce((n, p) => n + p.observations.length, 0),
    lengths = model.landmarks.map((p) => p.observations.length);
  if (
    count * 196 + 16 > device.workingBytes ||
    count * 160 > device.bindingBytes ||
    count > device.maxKernelElements
  )
    throw new RangeError('GPU observation evaluation exceeds budget');
  const raw = new Float64Array(Math.max(1, count * 20)),
    points = new Float64Array(Math.max(1, count * 3)),
    views = new Map(model.views.map((v) => [v.id, v]));
  let index = 0;
  for (const p of model.landmarks)
    for (const o of p.observations) {
      const v = views.get(o.viewId)!,
        k = validateGpuCamera(v.camera);
      const record = Float64Array.of(
        ...v.pose.R,
        ...v.pose.t,
        k.fx,
        k.fy,
        k.cx,
        k.cy,
        k.k1!,
        k.k2!,
        ...o.pixel,
      );
      if (!finiteGpuArray(record) || !finiteGpuArray(p.xyz))
        throw new RangeError('Invalid GPU scene coordinate bounds');
      raw.set(record, index * 20);
      points.set(p.xyz, index * 3);
      index++;
    }
  return device.enqueue(async () => {
    const before = device.statistics,
      owned: (Resident<Float64Array> | Resident<Uint32Array>)[] = [];
    let successful = false;
    const wide = (n: number, v?: Float64Array) => {
      const r = device.allocateF64(n, v);
      owned.push(r);
      return r;
    };
    try {
      const input = wide(raw.length, raw),
        xyz = wide(points.length, points),
        errors = wide(Math.max(1, count)),
        summary = wide(2),
        flags = device.allocateU32(Math.max(1, count));
      owned.push(flags);
      const control = { maxError: maxErrorPx, minAngle: 0, maxPairs: 1, iterations: 0 };
      const report = await device.plan(() => {
        if (count)
          projectObservations(
            { control, raw: input, points: xyz, errors, flags },
            Math.ceil(count / 64),
          );
        observationSummary({ points: xyz, flags, errors, summary }, 1);
      });
      const values = await device.read(summary, 2),
        flat = count ? await device.read(flags, count) : new Uint32Array(0);
      let at = 0;
      const masks = lengths.map((n) => {
        const r = Uint8Array.from(flat.subarray(at, at + n));
        at += n;
        return r;
      });
      const after = device.statistics;
      successful = true;
      return {
        masks,
        inlierCount: values[1]!,
        meanErrorPx: values[1]! > 0 ? values[0]! / values[1]! : Infinity,
        execution: {
          backend: device.backend,
          report,
          applicationUploadBytes: after.applicationUploadBytes - before.applicationUploadBytes,
          applicationReadbackBytes:
            after.applicationReadbackBytes - before.applicationReadbackBytes,
          cpuStages: [],
        },
      };
    } finally {
      for (const b of owned) successful ? device.recycle(b) : device.drop(b);
    }
  });
}
