import { observationModel, observationCases } from './indexed-observations-cases.mjs';
export { observationModel, observationCases } from './indexed-observations-cases.mjs';
import { ShadeDevice } from '../../src/gpu/device.ts';
import { evaluateObservationsGpu } from '../../src/gpu/observations.ts';
import {
  projectObservations,
  filterObservationDepth,
  observationPartials,
  observationPartialsOracle,
  observationSummary,
} from '../../src/gpu/triangulation.shade.ts';
export { ShadeDevice, evaluateObservationsGpu };

// Frozen expanded-layout oracle: these original kernels also serve triangulation. Keep the
// old observation/pose/point record order for exact comparisons with indexed scene storage.
export async function expandedObservationOracle(device, model, maxError = 3, options = {}) {
  const guarded = (options.maxRelativeDepthStdDev ?? 0) > 0;
  const lengths = model.landmarks.map((p) => p.observations.length);
  const count = lengths.reduce((sum, n) => sum + n, 0);
  const raw = new Float64Array(Math.max(1, count * 20));
  const points = new Float64Array(Math.max(1, count * 3));
  const views = new Map(model.views.map((v) => [v.id, v]));
  let at = 0;
  for (const p of model.landmarks)
    for (const o of p.observations) {
      const v = views.get(o.viewId),
        k = v.camera;
      raw.set(
        [...v.pose.R, ...v.pose.t, k.fx, k.fy, k.cx, k.cy, k.k1 ?? 0, k.k2 ?? 0, ...o.pixel],
        20 * at,
      );
      points.set(p.xyz, 3 * at++);
    }
  return device.enqueue(async () => {
    const owned = [];
    const wide = (n, value) => {
      const r = device.allocateF64(n, value);
      owned.push(r);
      return r;
    };
    const words = (n, value) => {
      const r = device.allocateU32(n, value);
      owned.push(r);
      return r;
    };
    try {
      const starts = new Uint32Array(lengths.length + 1);
      lengths.forEach((n, i) => {
        starts[i + 1] = starts[i] + n;
      });
      const offsets = guarded ? words(starts.length, starts) : undefined;
      const input = wide(raw.length, raw),
        xyz = wide(points.length, points);
      const errors = wide(Math.max(1, count)),
        summary = wide(2);
      const blocks = Math.ceil(Math.max(1, count) / 64),
        partials = wide(2 * blocks);
      const flags = words(Math.max(1, count + (guarded ? lengths.length : 0)));
      const control = {
        maxError,
        minAngle: 0,
        maxPairs: 1,
        iterations: 0,
        maxRelativeDepthStdDev: options.maxRelativeDepthStdDev ?? 0,
        pixelNoiseStdDevPx: options.pixelNoiseStdDevPx ?? 1,
      };
      await device.planSteps([
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
              ],
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
          : new Uint32Array(),
      ]);
      let offset = 0;
      return {
        masks: lengths.map((n) => {
          const mask = Uint8Array.from(flat.subarray(offset, offset + n));
          offset += n;
          return mask;
        }),
        ...(guarded
          ? { depthRejected: Uint8Array.from(flat.subarray(count, count + lengths.length)) }
          : {}),
        inlierCount: values[1],
        meanErrorPx: values[1] > 0 ? values[0] / values[1] : Infinity,
      };
    } finally {
      for (const r of owned) device.recycle(r);
    }
  });
}

function numericResult(result) {
  return {
    masks: result.masks.map((mask) => [...mask]),
    depthRejected: result.depthRejected ? [...result.depthRejected] : null,
    inlierCount: result.inlierCount,
    // Preserve every f64 bit of the actual reported mean, including Infinity and signed zero.
    meanBits: [...new Uint32Array(Float64Array.of(result.meanErrorPx).buffer)],
  };
}
export async function checkIndexedObservations(backend = 'cpu') {
  const device = await ShadeDevice.open({ backend, poolBytes: 0 });
  const cases = [];
  try {
    for (const row of observationCases()) {
      const before = JSON.stringify(row.model);
      const old = await expandedObservationOracle(device, row.model, 3, row.options);
      const result = await evaluateObservationsGpu(device, row.model, 3, row.options);
      cases.push({
        name: row.name,
        old: numericResult(old),
        indexed: numericResult(result),
        inputUnchanged: before === JSON.stringify(row.model),
        uploadBytes: result.execution.applicationUploadBytes,
      });
    }
    // Public calls synchronously snapshot all source bytes before their first suspension.
    const snapshot = observationModel({ points: 2 });
    const expected = numericResult(await expandedObservationOracle(device, snapshot));
    const pending = evaluateObservationsGpu(device, snapshot);
    for (const v of snapshot.views) {
      v.pose.R.fill(NaN);
      v.pose.t.fill(NaN);
      v.camera.fx = 1e20;
    }
    for (const p of snapshot.landmarks) {
      p.xyz.fill(NaN);
      for (const o of p.observations) o.pixel.fill(NaN);
    }
    const snapshotted = numericResult(await pending);
    // Fail immediately after allocating the new index buffer: every earlier allocation must retire.
    const allocate = device.allocateF64;
    let allocations = 0,
      failed = false;
    Object.defineProperty(device, 'allocateF64', {
      configurable: true,
      value(...args) {
        if (++allocations === 4) throw new Error('synthetic indexed allocation failure');
        return allocate.apply(device, args);
      },
    });
    try {
      await evaluateObservationsGpu(device, observationModel({ points: 2 }));
    } catch (error) {
      failed = String(error.message).includes('synthetic indexed allocation failure');
    } finally {
      delete device.allocateF64;
    }
    const afterFailure = device.statistics.liveBytes;
    // The indexed representation admits the same valid scene under a smaller per-binding limit.
    const descriptor = Object.getOwnPropertyDescriptor(device, 'bindingBytes');
    Object.defineProperty(device, 'bindingBytes', { configurable: true, value: 4096 });
    let tight;
    try {
      tight = numericResult(await evaluateObservationsGpu(device, observationModel()));
    } finally {
      if (descriptor) Object.defineProperty(device, 'bindingBytes', descriptor);
      else delete device.bindingBytes;
    }
    return {
      cases,
      expected,
      snapshotted,
      failed,
      afterFailure,
      tight,
      liveBytes: device.statistics.liveBytes,
    };
  } finally {
    await device.dispose();
  }
}
