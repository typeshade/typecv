// Compare an existing internal bundle with the current build on the same synthetic truth.
// npm run build:gpu
// node scripts/measure-mvs-refinement.mjs --bundle /tmp/baseline.mjs --out baseline.json
// node scripts/measure-mvs-refinement.mjs --out refinement.json
// Add --geometric to run all five photometric maps and filter the central view, seed 0.
// Add --fusion to measure the fused cloud at 4 photometric / 4 geometric iterations, seed 0.
import { writeFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { depthMetrics, fusionMetrics, mvsScene } from '../tests/typeshade/mvs-cases.mjs';

const args = process.argv.slice(2);
const flag = (name, fallback) => (args.includes(name) ? args[args.indexOf(name) + 1] : fallback);
const geometric = args.includes('--geometric');
const fused = args.includes('--fusion');
if (geometric && fused) throw new Error('Choose --geometric or --fusion');
const {
  ShadeDevice,
  decodeMvsNormal,
  mvsPhotometricGpu,
  mvsDepthMapsGpu,
  planMvsViews,
  reconstructDense,
} = await import(pathToFileURL(resolve(flag('--bundle', 'dist/internal/gpu.js'))).href);
const scene = mvsScene();
const plan = planMvsViews(scene);
const reference = plan.views[2].index;
const rows = [];
for (const seed of geometric || fused ? [0] : [0, 1, 2]) {
  const device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    const start = performance.now();
    const result = fused
      ? await reconstructDense(device, scene, scene.images, { photometricIterations: 4 })
      : geometric
        ? await mvsDepthMapsGpu(device, scene, plan, scene.images, {
            seed,
            geometricOrder: [reference],
            readStages: true,
          })
        : await mvsPhotometricGpu(device, scene, plan, scene.images, reference, { seed });
    const map = geometric ? result.views[reference] : result;
    const metrics = (data, filtered = false) =>
      depthMetrics(
        data,
        scene.truth[2],
        plan.normalization.scale,
        plan.views[2].depthRange,
        decodeMvsNormal,
        filtered ? (i) => data.depth[i] > 0 : undefined,
      );
    const row = {
      seed,
      ms: performance.now() - start,
      ...(fused
        ? {
            count: result.points.count,
            ...fusionMetrics(scene, result, plan.normalization.scale, decodeMvsNormal),
          }
        : geometric
          ? {
              photometric: metrics(map.photometric),
              geometric: metrics(map.geometric),
              filtered: metrics(map, true),
            }
          : metrics(map)),
    };
    rows.push(row);
    console.log(JSON.stringify(row));
  } finally {
    await device.dispose();
  }
}
if (flag('--out')) await writeFile(flag('--out'), JSON.stringify(rows, null, 2) + '\n');
