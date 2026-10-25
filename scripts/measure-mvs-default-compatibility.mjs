import { createHash } from 'node:crypto';
import { writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
const args = process.argv.slice(2),
  flag = (n) => (args.includes(n) ? args[args.indexOf(n) + 1] : undefined);
if (!flag('--baseline') || !flag('--out'))
  throw new Error('Usage: --baseline <baseline.mjs> --out <report.json>');
const baseline = await import(pathToFileURL(resolve(flag('--baseline'))).href);
import * as candidate from '../dist/internal/gpu.js';
import { mvsScene } from '../tests/typeshade/mvs-cases.mjs';
const scene = mvsScene({ width: 49, height: 37 });
const digest = (r) => {
  const h = createHash('sha256');
  const walk = (v) => {
    if (ArrayBuffer.isView(v)) h.update(new Uint8Array(v.buffer, v.byteOffset, v.byteLength));
    else if (v && typeof v === 'object') for (const k of Object.keys(v)) walk(v[k]);
  };
  walk(r);
  return h.digest('hex');
};
const rows = [];
for (const streamGroup of [undefined, 1]) {
  const results = [];
  for (const api of [baseline, candidate]) {
    const device = await api.ShadeDevice.open({ backend: 'cpu', poolBytes: 0 }),
      plan = api.planMvsViews(scene);
    try {
      const options = {
        photometricIterations: 2,
        geometricIterations: 1,
        readStages: true,
        confidence: true,
        streamGroup,
      };
      const maps = await api.mvsDepthMapsGpu(device, scene, plan, scene.images, options);
      const cloud = await api.reconstructDense(device, scene, scene.images, {
        ...options,
        tsdf: { visibility: 'view-weighted' },
        fusion: 'tsdf',
      });
      results.push({
        mapHash: digest(maps.views),
        cloudHash: digest(cloud.points),
        points: cloud.points.count,
        liveBytes: device.statistics.liveBytes,
      });
    } finally {
      await device.dispose();
    }
  }
  if (JSON.stringify(results[0]) !== JSON.stringify(results[1]))
    throw new Error('Default-disabled compatibility failed ' + JSON.stringify(results));
  rows.push({ streamGroup: streamGroup ?? 'window', baseline: results[0], candidate: results[1] });
}
console.log(JSON.stringify(rows));
await writeFile(resolve(flag('--out')), JSON.stringify(rows, null, 2));
