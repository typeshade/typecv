import type { Resident } from 'typeshade/runtime';
import type { SparseReconstruction } from '../reconstruction/types.ts';
import { ShadeDevice, type ShadePlanReport } from './device.ts';
import { finiteGpuArray, validateGpuCamera } from './pnp.ts';
import { scoreFocalCandidates, sumFocalCandidates } from './focal.shade.ts';

/** Score all nine focal hypotheses together, retaining the scene for the fine pass. */
export function calibrateFocalGpu(device: ShadeDevice, model: SparseReconstruction, huber: number) {
  if (!Number.isFinite(huber) || huber < 1e-12 || huber > 1e6)
    throw new RangeError('Invalid bundle adjustment options');
  const count = model.landmarks.reduce((n, point) => n + point.observations.length, 0);
  if (
    !count ||
    count * 256 + 216 > device.workingBytes ||
    count * 184 > device.bindingBytes ||
    count * 9 > device.maxKernelElements
  )
    throw new RangeError('GPU focal calibration exceeds budget');
  const data = new Float64Array(count * 23),
    views = new Map(model.views.map((view) => [view.id, view]));
  let at = 0;
  for (const point of model.landmarks)
    for (const observation of point.observations) {
      const view = views.get(observation.viewId)!,
        camera = validateGpuCamera(view.camera);
      const record = Float64Array.of(
        ...view.pose.R,
        ...view.pose.t,
        ...point.xyz,
        camera.fx,
        camera.fy,
        camera.cx,
        camera.cy,
        camera.k1!,
        camera.k2!,
        ...observation.pixel,
      );
      if (!finiteGpuArray(record)) throw new RangeError('Invalid GPU scene coordinate bounds');
      data.set(record, at);
      at += 23;
    }
  return device.enqueue(async () => {
    const before = device.statistics,
      owned: Resident<Float64Array>[] = [],
      reports: ShadePlanReport[] = [];
    let successful = false;
    const allocate = (length: number, values?: Float64Array) => {
      const buffer = device.allocateF64(length, values);
      owned.push(buffer);
      return buffer;
    };
    try {
      const raw = allocate(data.length, data),
        costs = allocate(count * 9),
        totals = allocate(9);
      let best = 1,
        bestCost = Infinity;
      for (let pass = 0; pass < 2; pass++) {
        const center = Math.log(best),
          step = Math.log(2) / (pass === 0 ? 4 : 16);
        const candidates = Float64Array.from({ length: 9 }, (_, i) => {
          const value = Math.exp(center + (i - 4) * step);
          return value >= 0.5 && value <= 2 ? value : -1;
        });
        const scales = allocate(9, candidates),
          control = { observations: count, huber };
        reports.push(
          await device.plan(() => {
            scoreFocalCandidates({ control, raw, scales, costs }, Math.ceil((count * 9) / 64));
            sumFocalCandidates({ control, costs, totals }, 9);
          }),
        );
        const values = await device.read(totals, 9);
        for (let i = 0; i < 9; i++)
          if (values[i]! >= 0 && values[i]! < bestCost) {
            best = candidates[i]!;
            bestCost = values[i]!;
          }
      }
      if (!Number.isFinite(bestCost)) throw new RangeError('Invalid GPU focal projections');
      successful = true;
      const after = device.statistics;
      return {
        scale: best,
        execution: {
          report: {
            tier: device.backend,
            dispatches: reports.reduce((n, r) => n + r.dispatches, 0),
            submissions: reports.reduce((n, r) => n + r.submissions, 0),
            groups: reports.flatMap((r) => r.groups),
          },
          applicationUploadBytes: after.applicationUploadBytes - before.applicationUploadBytes,
          applicationReadbackBytes:
            after.applicationReadbackBytes - before.applicationReadbackBytes,
        },
      };
    } finally {
      for (const buffer of owned) successful ? device.recycle(buffer) : device.drop(buffer);
    }
  });
}
