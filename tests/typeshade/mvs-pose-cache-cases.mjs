import { mvsScene, mvsWidePlacements } from './mvs-cases.mjs';

function sameArrays(a, b, path = '') {
  if (ArrayBuffer.isView(a)) {
    const x = new Uint8Array(a.buffer, a.byteOffset, a.byteLength),
      y = new Uint8Array(b.buffer, b.byteOffset, b.byteLength);
    if (x.length !== y.length || x.some((v, i) => v !== y[i]))
      throw new Error(`Pose cache changed ${path}`);
  } else if (a && typeof a === 'object') {
    for (const key of Object.keys(a)) sameArrays(a[key], b[key], `${path}.${key}`);
  } else if (a !== b) throw new Error(`Pose cache changed ${path}: ${a} vs ${b}`);
}

export async function mvsPoseCacheCases(api, deviceOptions) {
  const { ShadeDevice, planMvsViews, mvsPhotometricGpu, mvsDepthMapsGpu, reconstructDense } = api,
    rows = [];
  for (const config of [
    { name: 'six sources, default 36 samples', views: 7, options: {} },
    {
      name: '121 bilateral samples, tail chunks',
      views: 5,
      options: { windowStep: 1, bilateral: true, chunkPixels: 65 },
    },
    { name: 'streamed sources', views: 5, options: { streamGroup: 1 } },
  ]) {
    const scene = mvsScene({
        width: 49,
        height: 37,
        placements: mvsWidePlacements(config.views),
        reference: 2,
      }),
      plan = planMvsViews(scene),
      reference = plan.views[2].index,
      device = await ShadeDevice.open({ ...deviceOptions, poolBytes: 0 }),
      results = [];
    try {
      for (const cache of [false, true])
        results.push(
          await mvsPhotometricGpu(device, scene, plan, scene.images, reference, {
            photometricIterations: 2,
            snapshots: true,
            ...config.options,
            relativePoseCache: cache,
          }),
        );
      const arrays = ({ depth, normal, cost, snapshots }) => ({ depth, normal, cost, snapshots });
      sameArrays(arrays(results[0]), arrays(results[1]), config.name);
      rows.push({
        name: config.name,
        sources: plan.sources[reference * 7],
        snapshots: results[0].snapshots.length,
        kept: results[0].cost.filter((c) => c <= 0.9).length,
      });
      if (device.statistics.liveBuffers) throw new Error('Photometric pose cache leaked arrays');
    } finally {
      await device.dispose();
    }
  }
  const scene = mvsScene({ width: 49, height: 37 }),
    plan = planMvsViews(scene);
  for (const config of [
    { name: 'full window', options: {} },
    { name: 'full grouped bilateral', options: { streamGroup: 1, bilateral: true, windowStep: 1 } },
    { name: 'full budget and row chunks', options: { residentBytes: 75_000 } },
  ]) {
    const device = await ShadeDevice.open({ ...deviceOptions, poolBytes: 0 }),
      maps = [],
      clouds = [];
    try {
      for (const cache of [false, true]) {
        const options = {
          photometricIterations: 1,
          geometricIterations: 1,
          ...config.options,
          relativePoseCache: cache,
        };
        maps.push(
          await mvsDepthMapsGpu(device, scene, plan, scene.images, {
            ...options,
            readStages: true,
          }),
        );
        clouds.push(await reconstructDense(device, scene, scene.images, options));
      }
      sameArrays(maps[0].views, maps[1].views, `${config.name}.depthStages`);
      sameArrays(clouds[0].points, clouds[1].points, `${config.name}.points`);
      sameArrays(clouds[0].views, clouds[1].views, `${config.name}.views`);
      sameArrays(maps[0].residency, maps[1].residency, `${config.name}.residency`);
      if (device.statistics.liveBuffers) throw new Error('Full pose cache leaked arrays');
      rows.push({
        name: config.name,
        kept: maps[0].views.reduce((n, v) => n + v.depth.filter((d) => d > 0).length, 0),
        points: clouds[0].points.count,
        residency: maps[0].residency,
      });
    } finally {
      await device.dispose();
    }
  }
  return rows;
}
