// Dense MVS depth against the TUM ground-truth depth (docs/dense-mvs.md 7.3, step 8).
//
//   python scripts/compare-mvs-tum-landmarks.py website/public/sample/tum-fr3-structure-texture-far lm.json
//   npm run build && node scripts/compare-mvs-tum.mjs lm.json [--views 16] [--out result.json]
//
// Runs the depth stage of `reconstructDenseGpu` (mvsDepthMapsGpu: the same photometric pass,
// geometric pass and filter) on headless WebGPU with the sample's motion-capture poses, then
// compares every view's filtered depth with the sample's depth-NN.png.
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { inflateSync } from 'node:zlib';
import sharp from 'sharp';
import { inBrowser } from '../tests/typeshade/browser.mjs';

const args = process.argv.slice(2);
const flag = (name, fallback) => (args.includes(name) ? args[args.indexOf(name) + 1] : fallback);
const landmarksPath = args.find((a) => a.endsWith('.json') && a !== flag('--out'));
const folder = resolve(
  import.meta.dirname,
  '../website/public/sample/tum-fr3-structure-texture-far',
);
const sample = JSON.parse(await readFile(join(folder, 'sample.json'), 'utf8'));
const count = Number(flag('--views', sample.frames.length));
// --scale k divides the analysis size by k (the pixel centers are integers: x' = (x + 0.5) / k - 0.5).
const k = Number(flag('--scale', 1));
const shrink = (x) => (x + 0.5) / k - 0.5;
const frames = sample.frames.slice(0, count).map((f) => ({
  ...f,
  width: f.width / k,
  height: f.height / k,
  camera: {
    ...f.camera,
    fx: f.camera.fx / k,
    fy: f.camera.fy / k,
    cx: shrink(f.camera.cx),
    cy: shrink(f.camera.cy),
  },
}));
const landmarks = JSON.parse(await readFile(landmarksPath, 'utf8')).landmarks.filter((l) => {
  l.observations = l.observations
    .filter((o) => o.viewId < count)
    .map((o) => ({ ...o, pixel: [shrink(o.pixel[0]), shrink(o.pixel[1])] }));
  return l.observations.length >= 2;
});

const scratch = await mkdtemp(join(tmpdir(), 'compare-mvs-tum-'));
try {
  const serve = {};
  for (const f of frames) {
    const { data } = await sharp(join(folder, f.file))
      .resize(f.width, f.height, { kernel: 'lanczos3', fit: 'fill' })
      .ensureAlpha()
      .raw()
      .toBuffer({ resolveWithObject: true });
    await writeFile(join(scratch, `${f.id}.rgba`), data);
    serve[`/data/${f.id}.rgba`] = join(scratch, `${f.id}.rgba`);
  }
  const scene = {
    views: frames.map((f) => ({
      id: f.id,
      width: f.width,
      height: f.height,
      pose: f.reference,
      camera: f.camera,
    })),
    landmarks,
  };
  const started = Date.now();
  const { result, errors } = await inBrowser(
    async ({ scene, options }) => {
      const { ShadeDevice, mvsDepthMapsGpu, planMvsViews } = await import('/dist/internal/gpu.js');
      const images = new Map();
      for (const v of scene.views) {
        const rgba = new Uint8Array(await (await fetch(`/data/${v.id}.rgba`)).arrayBuffer());
        images.set(v.id, { rgba, width: v.width, height: v.height });
      }
      const plan = planMvsViews(scene);
      const device = await ShadeDevice.open({ backend: 'webgpu', poolBytes: 0 });
      try {
        const t0 = performance.now();
        const maps = await mvsDepthMapsGpu(device, scene, plan, images, options);
        const base64 = (a) => {
          const bytes = new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
          let s = '';
          for (let i = 0; i < bytes.length; i += 8192)
            s += String.fromCharCode(...bytes.subarray(i, i + 8192));
          return btoa(s);
        };
        return {
          ms: performance.now() - t0,
          scale: plan.normalization.scale,
          views: plan.views.map((v) => ({
            id: v.id,
            status: v.status,
            landmarks: v.landmarks,
            sources: v.sources,
            depth: v.index >= 0 ? base64(maps.views[v.index].depth) : null,
          })),
        };
      } finally {
        await device.dispose();
      }
    },
    { scene, options: {} },
    { timeoutMs: 7200000, serve },
  );
  if (errors.length) console.error(errors);
  const P = frames[0].width * frames[0].height;
  const rows = [];
  const pairs = [];
  for (const view of result.views) {
    const frame = frames[view.id];
    if (view.depth === null) {
      rows.push({ id: view.id, status: view.status });
      continue;
    }
    const raw = Buffer.from(view.depth, 'base64');
    const depth = new Float32Array(raw.buffer, raw.byteOffset, P);
    const full = await readDepthPng(join(folder, frame.depthFile));
    // Nearest neighbour: averaging depths across an edge invents surfaces.
    const gt = new Uint16Array(P);
    for (let y = 0; y < frame.height; y++)
      for (let x = 0; x < frame.width; x++)
        gt[y * frame.width + x] = full[(y * k + (k >> 1)) * frame.width * k + x * k + (k >> 1)];
    let withTruth = 0,
      valid = 0;
    const errors = [];
    for (let p = 0; p < P; p++) {
      if (gt[p] === 0) continue;
      withTruth++;
      if (!(depth[p] > 0)) continue;
      valid++;
      pairs.push([depth[p] * result.scale, gt[p] / sample.depthUnitsPerMeter]);
      errors.push(
        Math.abs(depth[p] * result.scale - gt[p] / sample.depthUnitsPerMeter) /
          (gt[p] / sample.depthUnitsPerMeter),
      );
    }
    rows.push({
      id: view.id,
      status: view.status,
      sources: view.sources,
      withTruth,
      valid,
      ...summary(errors, withTruth),
    });
  }
  function median(a) {
    const s = Float64Array.from(a).sort();
    return s.length ? s[s.length >> 1] : NaN;
  }
  function summary(errors, withTruth) {
    const within = (x) => errors.filter((e) => e <= x).length / Math.max(1, errors.length);
    return {
      validFraction: errors.length / withTruth,
      medianRelativeError: median(errors),
      within1: within(0.01),
      within2: within(0.02),
      within5: within(0.05),
    };
  }
  // The poses are metric, so the comparison needs no alignment. The median of ground truth over
  // estimate is the scale a free alignment would find: it shows whether the metric scale holds.
  const alignScale = median(pairs.map(([d, g]) => g / d));
  const aligned = pairs.map(([d, g]) => Math.abs(d * alignScale - g) / g);
  const raw = pairs.map(([d, g]) => Math.abs(d - g) / g);
  const totalWithTruth = rows.reduce((s, r) => s + (r.withTruth ?? 0), 0);
  const total = {
    views: rows.filter((r) => r.valid !== undefined).length,
    pixelsWithTruth: totalWithTruth,
    ...summary(raw, totalWithTruth),
    medianScaleAlignment: alignScale,
    alignedMedianRelativeError: median(aligned),
    alignedWithin1: aligned.filter((e) => e <= 0.01).length / aligned.length,
    alignedWithin2: aligned.filter((e) => e <= 0.02).length / aligned.length,
    alignedWithin5: aligned.filter((e) => e <= 0.05).length / aligned.length,
  };
  const record = {
    sample: sample.name,
    scale: k,
    size: [frames[0].width, frames[0].height],
    landmarks: landmarks.length,
    depthStageMs: Math.round(result.ms),
    wallMs: Date.now() - started,
    total,
    views: rows,
  };
  console.log(JSON.stringify(record, null, 1));
  if (flag('--out')) await writeFile(flag('--out'), JSON.stringify(record, null, 1) + '\n');
} finally {
  await rm(scratch, { recursive: true, force: true });
}

// A 16-bit grayscale PNG as native Uint16 values (sharp converts such a PNG to sRGB).
async function readDepthPng(path) {
  const png = await readFile(path);
  const width = png.readUInt32BE(16),
    height = png.readUInt32BE(20);
  if (png[24] !== 16 || png[25] !== 0 || png[28] !== 0)
    throw new Error(`${path}: not a plain 16-bit grayscale PNG`);
  const idat = [];
  for (let at = 8; at < png.length;) {
    const length = png.readUInt32BE(at);
    if (png.toString('latin1', at + 4, at + 8) === 'IDAT')
      idat.push(png.subarray(at + 8, at + 8 + length));
    at += 12 + length;
  }
  const raw = inflateSync(Buffer.concat(idat));
  const stride = width * 2;
  const out = Buffer.alloc(stride * height);
  for (let y = 0; y < height; y++) {
    const filter = raw[y * (stride + 1)];
    for (let x = 0; x < stride; x++) {
      const value = raw[y * (stride + 1) + 1 + x];
      const left = x >= 2 ? out[y * stride + x - 2] : 0;
      const up = y > 0 ? out[(y - 1) * stride + x] : 0;
      const upLeft = x >= 2 && y > 0 ? out[(y - 1) * stride + x - 2] : 0;
      const p = left + up - upLeft;
      const paeth =
        Math.abs(p - left) <= Math.abs(p - up) && Math.abs(p - left) <= Math.abs(p - upLeft)
          ? left
          : Math.abs(p - up) <= Math.abs(p - upLeft)
            ? up
            : upLeft;
      out[y * stride + x] = (value + [0, left, up, (left + up) >> 1, paeth][filter]) & 255;
    }
  }
  const values = new Uint16Array(width * height);
  for (let i = 0; i < values.length; i++) values[i] = out.readUInt16BE(2 * i);
  return values;
}
