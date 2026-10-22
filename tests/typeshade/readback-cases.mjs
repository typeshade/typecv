import { feedbackInput, mvsFeedbackInput } from './tsdf-feedback-cases.mjs';

/** Compare identical kernels with reads forced serial versus issued together. */
export async function readbackCase(api, backend) {
  const d = await api.ShadeDevice.open({ backend, poolBytes: 0 });
  const original = d.read.bind(d);
  let serial = true,
    pending = Promise.resolve();
  d.read = (...args) =>
    serial ? (pending = pending.then(() => original(...args))) : original(...args);
  const { scene, plan, images, depthMaps } = mvsFeedbackInput(api.planMvsViews);
  const rows = [];
  const snapshot = () => {
    const s = d.statistics;
    return {
      reads: s.reads,
      waves: s.readWaves,
      bytes: s.applicationReadbackBytes,
      snapshots: s.residentSnapshotBytes,
    };
  };
  const run = async (operation, isSerial) => {
    serial = isSerial;
    const before = snapshot(),
      start = performance.now();
    const output = await operation();
    const after = snapshot();
    return {
      output,
      ms: performance.now() - start,
      counts: Object.fromEntries(Object.keys(before).map((k) => [k, after[k] - before[k]])),
    };
  };
  const cases = [
    [
      'photometric with cost',
      async () => {
        const r = await api.mvsPhotometricGpu(d, scene, plan, images, 1, {
          windowRadius: 1,
          windowStep: 1,
          photometricIterations: 1,
          snapshots: true,
        });
        return {
          depth: [...r.depth],
          normal: [...r.normal],
          cost: [...r.cost],
          snapshots: r.snapshots.map((s) => ({
            depth: [...s.depth],
            normal: [...s.normal],
            cost: [...s.cost],
          })),
        };
      },
    ],
    [
      'depth and normal',
      async () => {
        let maps;
        const r = await api.reconstructDense(d, scene, images, {
          windowRadius: 1,
          windowStep: 1,
          photometricIterations: 1,
          geometricIterations: 1,
          fusion: 'pixel',
          onDepthMaps: (value) => {
            maps = value;
          },
        });
        return {
          depth: maps.depth.map((a) => [...a]),
          normal: maps.normal.map((a) => [...a]),
          positions: [...r.points.positions],
          colors: [...r.points.colors],
          normals: [...r.points.normals],
        };
      },
    ],
    [
      'TSDF points and attributes',
      async () => {
        const r = await api.fuseTsdfGpu(d, feedbackInput(), {
          levels: 1,
          voxelScale: 1,
          truncation: 2,
        });
        return {
          count: r.count,
          positions: [...r.positions],
          colors: [...r.colors],
          normals: [...r.normals],
          weights: [...r.weights],
          free: [...r.free],
          tallies: [...r.tallies],
        };
      },
    ],
    [
      'pixel points and attributes',
      async () => {
        const r = await api.reconstructDense(d, scene, images, { fusion: 'pixel', depthMaps });
        return {
          count: r.points.count,
          positions: [...r.points.positions],
          colors: [...r.points.colors],
          normals: [...r.points.normals],
        };
      },
    ],
  ];
  try {
    for (const [name, operation] of cases) {
      await run(operation, false); // Compile/warm kernels before comparing counts.
      const baseline = await run(operation, true);
      const concurrent = await run(operation, false);
      rows.push({ name, baseline, concurrent });
    }
    return { rows, liveBytes: d.statistics.liveBytes };
  } finally {
    await d.dispose();
  }
}
