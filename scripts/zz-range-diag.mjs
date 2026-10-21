import { readFile, writeFile, mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import sharp from 'sharp';
import { inBrowser } from '../tests/typeshade/browser.mjs';
const R = '/tmp/claude-0/-home-user/a2fd8d80-86db-52da-ba3c-ad2b03a32f13/scratchpad';
const k = 2;
const ids = new Set([0, 2, 4, 6, 8, 10]);
const model = JSON.parse(await readFile(`${R}/rec-ed93/typecv-reconstruction.json`, 'utf8'));
const shrink = (x) => (x + 0.5) / k - 0.5;
const views = model.views.filter((v) => ids.has(Number(v.id))).sort((a, b) => a.id - b.id).map((v) => ({
  id: Number(v.id), width: v.width / k, height: v.height / k, pose: v.pose,
  camera: { ...v.camera, fx: v.camera.fx / k, fy: v.camera.fy / k, cx: shrink(v.camera.cx), cy: shrink(v.camera.cy) },
}));
const landmarks = model.landmarks.map((l) => ({ xyz: l.xyz, observations: l.observations.filter((o) => ids.has(o.viewId)).map((o) => ({ viewId: o.viewId, pixel: [shrink(o.pixel[0]), shrink(o.pixel[1])] })) })).filter((l) => l.observations.length >= 2);
const scratch = await mkdtemp(join(tmpdir(), 'diag-'));
const serve = {};
for (const v of views) {
  const { data } = await sharp(`${R}/range/wt/frame-${String(v.id).padStart(2, '0')}.jpg`).resize(v.width, v.height, { kernel: 'lanczos3', fit: 'fill' }).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
  await writeFile(join(scratch, `${v.id}.rgba`), data);
  serve[`/data/${v.id}.rgba`] = join(scratch, `${v.id}.rgba`);
}
const { result, errors } = await inBrowser(async ({ scene }) => {
  const { ShadeDevice, mvsPhotometricGpu, planMvsViews } = await import('/dist/internal/gpu.js');
  const images = new Map();
  for (const v of scene.views) images.set(v.id, { rgba: new Uint8Array(await (await fetch(`/data/${v.id}.rgba`)).arrayBuffer()), width: v.width, height: v.height });
  const out = {};
  const variants = {
    old: { far: 1.25, seeds: true },
    oldNoSeeds: { far: 1.25, seeds: false },
    wide: { far: 8, seeds: true },
    wideNoSeeds: { far: 8, seeds: false },
    farBand: { far: 8, seeds: false, near: 6 },
  };
  for (const [name, o] of Object.entries(variants)) {
    const plan = planMvsViews(scene, { farStretch: o.far });
    const scale = plan.normalization.scale;
    const ref = 0;
    const base = ref * 24;
    if (o.near) plan.cameras[base + 18] = (19 / scale);
    if (!o.seeds) { plan.seedOffsets.fill(0); }
    const device = await ShadeDevice.open({ backend: 'webgpu', poolBytes: 0, workingBytes: 512 * 1024 ** 2 });
    try {
      const r = await mvsPhotometricGpu(device, scene, plan, images, ref, { photometricIterations: 8 });
      const W = scene.views[0].width, H = scene.views[0].height;
      const bands = [];
      for (let q = 0; q < 4; q++) {
        const c = [], d = [];
        for (let y = Math.floor(q * H / 4); y < Math.floor((q + 1) * H / 4); y++) for (let x = 0; x < W; x++) { c.push(r.cost[y * W + x]); d.push(r.depth[y * W + x] * scale); }
        c.sort((a, b) => a - b); d.sort((a, b) => a - b);
        bands.push({ medianCost: +c[c.length >> 1].toFixed(3), p25Cost: +c[c.length >> 2].toFixed(3), medianDepth: +d[d.length >> 1].toFixed(2), p10: +d[Math.floor(d.length * 0.1)].toFixed(2), p90: +d[Math.floor(d.length * 0.9)].toFixed(2) });
      }
      out[name] = { range: [plan.cameras[base + 18] * scale, plan.cameras[base + 19] * scale].map((x) => +x.toFixed(2)), bands };
    } finally { await device.dispose(); }
  }
  return out;
}, { scene: { views, landmarks } }, { timeoutMs: 3600000, serve });
if (errors.length) console.error(errors);
console.log(JSON.stringify(result, null, 1));
await rm(scratch, { recursive: true, force: true });
