import { mvsScene, depthMetrics } from './mvs-cases.mjs';
const bytes = (a) => new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
function same(a, b, label) {
  if (ArrayBuffer.isView(a)) {
    const x = bytes(a),
      y = bytes(b);
    if (x.length !== y.length || x.some((v, i) => v !== y[i]))
      throw new Error('Grouped MVS changed ' + label);
  } else if (a && typeof a === 'object')
    for (const k of Object.keys(a)) same(a[k], b[k], label + '.' + k);
  else if (a !== b) throw new Error('Grouped MVS changed ' + label);
}
export async function boundaryMapCases(api, deviceOptions = {}, names) {
  const scene = mvsScene({ width: 49, height: 37 }),
    plan = api.planMvsViews(scene),
    reference = plan.views[2].index;
  const device = await api.ShadeDevice.open({ ...deviceOptions, poolBytes: 0 }),
    rows = [];
  try {
    for (const [name, options] of Object.entries({
      legacy: {},
      boundary: { boundaryAware: true },
      observable: { minFiniteGain: 0.02 },
      combined: { boundaryAware: true, minFiniteGain: 0.02 },
    })) {
      if (names && !names.includes(name)) continue;
      const maps = [];
      let peak = 0;
      for (const grouped of [false, true]) {
        const r = await api.mvsDepthMapsGpu(device, scene, plan, scene.images, {
          photometricIterations: 2,
          geometricIterations: 1,
          readStages: true,
          confidence: true,
          ...options,
          ...(grouped ? { streamGroup: 1, chunkPixels: 65 } : {}),
        });
        if (device.statistics.liveBytes !== 0) throw new Error('MVS boundary arrays leaked');
        peak = Math.max(peak, r.residency.peakBytes);
        maps.push(r);
      }
      same(maps[0].views, maps[1].views, name);
      const r = maps[0],
        m = r.views[reference];
      const metrics = depthMetrics(
        m,
        scene.truth[2],
        plan.normalization.scale,
        plan.views[2].depthRange,
        api.decodeMvsNormal,
        (i) => m.depth[i] > 0,
      );
      rows.push({
        name,
        metrics,
        peakBytes: peak,
        depth: Array.from(m.depth),
        normal: Array.from(m.normal),
        cost: Array.from(m.cost),
      });
    }
    return { rows, liveBytes: device.statistics.liveBytes, backend: device.backend };
  } finally {
    await device.dispose();
  }
}
