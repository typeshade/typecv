import type { Resident } from 'typeshade/runtime';
import type { SparseReconstruction } from '../reconstruction/types.ts';
import { depthSensitivityControls, type DepthSensitivityOptions } from '../geometry/multiview.ts';
import { validateReconstruction } from '../reconstruction/bundle-adjustment.ts';
import { ShadeDevice } from './device.ts';
import { finiteGpuArray, validateGpuCamera, type GpuGeometryExecution } from './pnp.ts';
import {
  projectObservations,
  filterObservationDepth,
  observationPartials,
  observationPartialsOracle,
  observationSummary,
} from './triangulation.shade.ts';

export interface GpuObservationResult {
  readonly depthRejected?: Uint8Array;
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
  options: DepthSensitivityOptions = {},
): Promise<GpuObservationResult> {
  const depth = depthSensitivityControls(options),
    guarded = depth.maxRelativeDepthStdDev > 0;
  validateReconstruction(model);
  if (
    !finiteGpuArray(Float64Array.of(depth.maxRelativeDepthStdDev, depth.pixelNoiseStdDevPx)) ||
    !Number.isFinite(maxErrorPx) ||
    maxErrorPx < 1e-12 ||
    maxErrorPx > 1e6
  )
    throw new RangeError('Invalid GPU observation threshold');
  const count = model.landmarks.reduce((n, p) => n + p.observations.length, 0),
    lengths = model.landmarks.map((p) => p.observations.length);
  if (
    count * 196 +
      16 +
      Math.ceil(Math.max(1, count) / 64) * 16 +
      (guarded ? 8 * model.landmarks.length + 4 : 0) >
      device.workingBytes ||
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
      const starts = new Uint32Array(lengths.length + 1);
      lengths.forEach((n, i) => {
        starts[i + 1] = starts[i]! + n;
      });
      const offsets = guarded ? device.allocateU32(starts.length, starts) : undefined;
      if (offsets) owned.push(offsets);
      const input = wide(raw.length, raw),
        xyz = wide(points.length, points),
        errors = wide(Math.max(1, count)),
        summary = wide(2),
        blocks = Math.ceil(Math.max(1, count) / 64),
        partials = wide(2 * blocks),
        flags = device.allocateU32(Math.max(1, count + (guarded ? lengths.length : 0)));
      owned.push(flags);
      const control = { maxError: maxErrorPx, minAngle: 0, maxPairs: 1, iterations: 0, ...depth };
      // Named steps, so the device reports each kernel before it starts (a lost device names it).
      const report = await device.planSteps([
        [
          'projectObservations',
          () => {
            if (count)
              projectObservations(
                { control, raw: input, points: xyz, errors, flags },
                Math.ceil(count / 64),
              );
          },
        ],
        ...(offsets
          ? [
              [
                'filterObservationDepth',
                () =>
                  filterObservationDepth(
                    { control, raw: input, points: xyz, errors, flags, offsets },
                    Math.ceil(lengths.length / 64),
                  ),
              ] as const,
            ]
          : []),
        [
          'observationPartials',
          () => {
            if (device.backend === 'webgpu')
              observationPartials({ points: xyz, flags, errors, partials }, blocks);
            else
              observationPartialsOracle(
                { points: xyz, flags, errors, partials },
                Math.ceil(blocks / 8),
              );
          },
        ],
        ['observationSummary', () => observationSummary({ partials, summary }, 1)],
      ]);
      const [values, flat] = await Promise.all([
        device.read(summary, 2),
        count || (guarded && lengths.length)
          ? device.read(flags, count + (guarded ? lengths.length : 0))
          : new Uint32Array(0),
      ]);
      let at = 0;
      const masks = lengths.map((n) => {
        const r = Uint8Array.from(flat.subarray(at, at + n));
        at += n;
        return r;
      });
      const after = device.statistics;
      successful = true;
      return {
        ...(guarded
          ? { depthRejected: Uint8Array.from(flat.subarray(count, count + lengths.length)) }
          : {}),
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
