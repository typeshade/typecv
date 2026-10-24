import type { Resident } from 'typeshade/runtime';
import type { SparseReconstruction } from '../reconstruction/types.ts';
import { depthSensitivityControls, type DepthSensitivityOptions } from '../geometry/multiview.ts';
import { validateReconstruction } from '../reconstruction/bundle-adjustment.ts';
import { ShadeDevice } from './device.ts';
import { finiteGpuArray, validateGpuCamera, type GpuGeometryExecution } from './pnp.ts';
import { observationSummary } from './triangulation.shade.ts';
import {
  projectIndexedObservations,
  filterIndexedObservationDepth,
  indexedObservationPartials,
  indexedObservationPartialsOracle,
} from './observation-evaluation.shade.ts';
import { observationLayout } from './observation-layout.ts';

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
    lengths = model.landmarks.map((p) => p.observations.length),
    used = new Map<number, number>();
  for (const p of model.landmarks)
    for (const o of p.observations) if (!used.has(o.viewId)) used.set(o.viewId, used.size);
  const layout = observationLayout(used.size, model.landmarks.length, count, guarded);
  if (
    !Object.values(layout.lengths).every((n) => Number.isSafeInteger(n) && n <= 0xffffffff) ||
    layout.bytes > device.workingBytes ||
    layout.largestBindingBytes > device.bindingBytes ||
    count > device.maxKernelElements
  )
    throw new RangeError('GPU observation evaluation exceeds budget');
  const cameras = new Float64Array(layout.lengths.cameras),
    points = new Float64Array(layout.lengths.points),
    pixels = new Float64Array(layout.lengths.pixels),
    indices = new Uint32Array(layout.lengths.indices),
    views = new Map(model.views.map((v) => [v.id, v]));
  // Validate/pack each referenced camera once. Unreferenced views retain the public validator's
  // existing contract; they need no extra GPU coordinate bounds or device allocation.
  for (const [id, index] of used) {
    const v = views.get(id)!,
      k = validateGpuCamera(v.camera),
      row = cameras.subarray(index * 18, (index + 1) * 18);
    row.set(v.pose.R);
    row.set(v.pose.t, 9);
    row.set([k.fx, k.fy, k.cx, k.cy, k.k1!, k.k2!], 12);
    if (!finiteGpuArray(row)) throw new RangeError('Invalid GPU scene coordinate bounds');
  }
  let index = 0;
  for (const [pointIndex, p] of model.landmarks.entries()) {
    if (!finiteGpuArray(p.xyz)) throw new RangeError('Invalid GPU scene coordinate bounds');
    points.set(p.xyz, pointIndex * 3);
    for (const o of p.observations) {
      pixels.set(o.pixel, index * 2);
      indices[2 * index] = used.get(o.viewId)!;
      indices[2 * index + 1] = pointIndex;
      index++;
    }
  }
  if (!finiteGpuArray(pixels)) throw new RangeError('Invalid GPU scene coordinate bounds');
  return device.enqueue(async () => {
    const before = device.statistics,
      owned: (Resident<Float64Array> | Resident<Uint32Array>)[] = [];
    let successful = false;
    const wide = (n: number, v?: Float64Array) => {
      const r = device.allocateF64(n, v);
      owned.push(r);
      return r;
    };
    const words = (n: number, v?: Uint32Array) => {
      const r = device.allocateU32(n, v);
      owned.push(r);
      return r;
    };
    try {
      const starts = new Uint32Array(lengths.length + 1);
      lengths.forEach((n, i) => {
        starts[i + 1] = starts[i]! + n;
      });
      const offsets = guarded ? words(starts.length, starts) : undefined;
      const cameraRecords = wide(cameras.length, cameras),
        xyz = wide(points.length, points),
        measurements = wide(pixels.length, pixels),
        observationIndices = words(indices.length, indices),
        errors = wide(layout.lengths.errors),
        summary = wide(2),
        blocks = Math.ceil(Math.max(1, count) / 64),
        partials = wide(2 * blocks),
        flags = words(layout.lengths.flags);
      const control = { maxError: maxErrorPx, ...depth, observations: count };
      const scene = {
        control,
        cameras: cameraRecords,
        coordinates: xyz,
        indices: observationIndices,
        errors,
        flags,
      };
      // Named steps, so the device reports each kernel before it starts (a lost device names it).
      const report = await device.planSteps([
        [
          'projectIndexedObservations',
          () => {
            if (count)
              projectIndexedObservations({ ...scene, pixels: measurements }, Math.ceil(count / 64));
          },
        ],
        ...(offsets
          ? [
              [
                'filterIndexedObservationDepth',
                () =>
                  filterIndexedObservationDepth(
                    { ...scene, offsets },
                    Math.ceil(lengths.length / 64),
                  ),
              ] as const,
            ]
          : []),
        [
          'indexedObservationPartials',
          () => {
            if (device.backend === 'webgpu')
              indexedObservationPartials({ control, flags, errors, partials }, blocks);
            else
              indexedObservationPartialsOracle(
                { control, flags, errors, partials },
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
