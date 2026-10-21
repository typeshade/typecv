// Dense MVS depth range, measured on a sparse model with no ground-truth depth
// (docs/dense-mvs.md, Amendment 2 to 3.3).
//
//   node scripts/measure-mvs-range.mjs --scene water-tower --model typecv-reconstruction.json \
//     --frames dir-of-frame-NN.jpg --ids 0,2,4,6,8,10 --scale 2 --far 1.25 --out run.json
//   node scripts/measure-mvs-range.mjs --scene graham-hall --ids 0,1,2,3,4,5 --scale 1 --far 8
//
// --scene water-tower   the lab's export (`typecv-reconstruction.json`: views, landmarks) and the
//                       sample's frames `frame-NN.jpg`, NN = the view id.
// --scene graham-hall   COLMAP's poses and the dataset's own SIFT matches of
//                       `website/public/sample/graham-hall`, triangulated as in tests/mvs-views.test.ts.
// --ids i,j,k           the view ids that run (a subset keeps the run short); landmarks keep only
//                       their observations in these views.
// --scale k             divides the analysis size by k (pixel centers stay centers).
// --near a --far b      `nearStretch` and `farStretch` of the view options (3.3). 0.75 and 1.25 are
//                       the range before the amendment.
// --backend webgpu|cpu  headless WebGPU (SwiftShader without a physical GPU) or the CPU tier.
// --out file.json       the record. `file.bin` next to it holds the fused positions (f32 x 3),
//                       `file.previews.json` the depth previews.
//
// Reported per view: the range, the landmarks and seeds, the share of pixels with a filtered depth,
// and the coverage by image row quarter: the share of pixels that some fused point of the whole
// cloud lands on within one pixel (a 3 x 3 splat), top to bottom. Then the fused point count and
// the time of the dense run.
import { readFile, writeFile, mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import sharp from 'sharp';
import { triangulateTrack } from '../src/geometry/multiview.ts';
import { planMvsViews } from '../src/gpu/mvs-views.ts';
import { inBrowser } from '../tests/typeshade/browser.mjs';

const args = process.argv.slice(2);
const flag = (name, fallback) => (args.includes(name) ? args[args.indexOf(name) + 1] : fallback);
const sceneName = flag('--scene', 'water-tower');
const k = Number(flag('--scale', 2));
const near = flag('--near') === undefined ? undefined : Number(flag('--near'));
const far = flag('--far') === undefined ? undefined : Number(flag('--far'));
const backend = flag('--backend', 'webgpu');
const iterations = flag('--iterations') === undefined ? undefined : Number(flag('--iterations'));
const ids = flag('--ids')?.split(',').map(Number);
const shrink = (x) => (x + 0.5) / k - 0.5;

const unit = (v) => {
  const n = Math.hypot(...v);
  return v.map((x) => x / n);
};
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
/** COLMAP's R is rounded to about 1e-6: Gram-Schmidt on its rows makes it a rotation. */
function orthonormal(R) {
  const x = unit([R[0], R[1], R[2]]);
  const b = [R[3], R[4], R[5]];
  const d = dot(x, b);
  const y = unit([b[0] - d * x[0], b[1] - d * x[1], b[2] - d * x[2]]);
  return [...x, ...y, ...cross(x, y)];
}

/** Views { id, width, height, camera, pose, load() }, landmarks { xyz, observations }. */
async function loadScene() {
  if (sceneName === 'water-tower') {
    const model = JSON.parse(await readFile(flag('--model'), 'utf8'));
    const frames = flag('--frames');
    const views = model.views.map((v) => ({
      id: Number(v.id),
      width: v.width,
      height: v.height,
      camera: v.camera,
      pose: v.pose,
      file: join(frames, `frame-${String(v.id).padStart(2, '0')}.jpg`),
    }));
    return { views, landmarks: model.landmarks };
  }
  if (sceneName === 'graham-hall') {
    const dir = resolve(import.meta.dirname, '../website/public/sample/graham-hall');
    const sample = JSON.parse(await readFile(join(dir, 'sample.json'), 'utf8'));
    const bytes = await readFile(join(dir, 'colmap-sift.bin'));
    const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
    const keypoints = new Float32Array(buffer, 0, sample.features.keypointBytes / 4);
    const matches = new Uint16Array(buffer, sample.features.keypointBytes);
    const views = sample.frames.map((f) => ({
      id: f.id,
      width: f.width,
      height: f.height,
      camera: { ...f.camera },
      pose: { R: orthonormal(f.reference.R), t: f.reference.t },
      file: join(dir, f.file),
    }));
    const blocks = new Map(sample.features.keypoints.map((b) => [b.id, b]));
    const base = new Map(),
      owner = [];
    for (const f of sample.frames) {
      base.set(f.id, owner.length);
      for (let i = 0; i < blocks.get(f.id).count; i++) owner.push(f.id);
    }
    const parent = Int32Array.from({ length: owner.length }, (_, i) => i);
    const find = (i) => {
      while (parent[i] !== i) i = parent[i] = parent[parent[i]];
      return i;
    };
    for (const pair of sample.features.pairs)
      for (let i = 0; i < pair.count; i++) {
        const a = find(base.get(pair.first) + matches[pair.offset + 2 * i]),
          b = find(base.get(pair.second) + matches[pair.offset + 2 * i + 1]);
        if (a !== b) parent[Math.max(a, b)] = Math.min(a, b);
      }
    const tracks = new Map();
    for (let i = 0; i < owner.length; i++) {
      const root = find(i);
      if (!tracks.has(root)) tracks.set(root, []);
      tracks.get(root).push(i);
    }
    const byId = new Map(views.map((v) => [v.id, v]));
    const landmarks = [];
    for (const members of tracks.values()) {
      if (members.length < 2) continue;
      const owners = members.map((i) => owner[i]);
      if (new Set(owners).size !== owners.length) continue;
      const observations = members.map((i, j) => {
        const view = byId.get(owners[j]);
        const local = blocks.get(view.id).offset + 2 * (i - base.get(view.id));
        return {
          viewId: view.id,
          camera: view.camera,
          pose: view.pose,
          pixel: [keypoints[local], keypoints[local + 1]],
        };
      });
      const track = triangulateTrack(observations, { maxErrorPx: 2, minAngleDeg: 1 });
      if (!track) continue;
      const kept = observations.filter((_, j) => track.inliers[j]);
      if (kept.length >= 2)
        landmarks.push({
          xyz: Array.from(track.xyz),
          observations: kept.map(({ viewId, pixel }) => ({ viewId, pixel })),
        });
    }
    return { views, landmarks };
  }
  throw new Error('Unknown scene ' + sceneName);
}

const full = await loadScene();
const wanted = new Set(ids ?? full.views.map((v) => v.id));
const views = full.views
  .filter((v) => wanted.has(v.id))
  .sort((a, b) => a.id - b.id)
  .map((v) => ({
    ...v,
    width: v.width / k,
    height: v.height / k,
    camera: {
      ...v.camera,
      fx: v.camera.fx / k,
      fy: v.camera.fy / k,
      cx: shrink(v.camera.cx),
      cy: shrink(v.camera.cy),
    },
  }));
const landmarks = full.landmarks
  .map((l) => ({
    xyz: l.xyz,
    observations: l.observations
      .filter((o) => wanted.has(o.viewId))
      .map((o) => ({ viewId: o.viewId, pixel: [shrink(o.pixel[0]), shrink(o.pixel[1])] })),
  }))
  .filter((l) => l.observations.length >= 2);

/** The 99th percentile of the depths (model units) of the landmarks that the view observes. */
function observedP99(view) {
  const { R, t } = view.pose;
  const z = [];
  for (const l of landmarks) {
    if (!l.observations.some((o) => o.viewId === view.id)) continue;
    const d = R[6] * l.xyz[0] + R[7] * l.xyz[1] + R[8] * l.xyz[2] + t[2];
    if (d > 0) z.push(d);
  }
  z.sort((a, b) => a - b);
  return z[Math.ceil((z.length - 1) * 0.99)];
}

const scratch = await mkdtemp(join(tmpdir(), 'measure-mvs-range-'));
try {
  const serve = {};
  for (const v of views) {
    const { data } = await sharp(v.file)
      .resize(v.width, v.height, { kernel: 'lanczos3', fit: 'fill' })
      .ensureAlpha()
      .raw()
      .toBuffer({ resolveWithObject: true });
    await writeFile(join(scratch, `${v.id}.rgba`), data);
    serve[`/data/${v.id}.rgba`] = join(scratch, `${v.id}.rgba`);
  }
  const scene = {
    views: views.map((v) => ({
      id: v.id,
      width: v.width,
      height: v.height,
      pose: v.pose,
      camera: v.camera,
    })),
    landmarks,
  };
  const viewOptions = {
    ...(near === undefined ? {} : { nearStretch: near }),
    ...(far === undefined ? {} : { farStretch: far }),
  };
  // The host plan: the range and the seeds of every view (the same code runs in the page).
  const plan = planMvsViews(scene, viewOptions);
  const options = {
    ...viewOptions,
    ...(iterations === undefined
      ? {}
      : { photometricIterations: iterations, geometricIterations: Math.ceil(iterations / 2) }),
  };
  const started = Date.now();
  const { result, errors } = await inBrowser(
    async ({ scene, options, backend }) => {
      const { ShadeDevice, reconstructDenseGpu } = await import('/dist/internal/gpu.js');
      const images = new Map();
      for (const v of scene.views) {
        const rgba = new Uint8Array(await (await fetch(`/data/${v.id}.rgba`)).arrayBuffer());
        images.set(v.id, { rgba, width: v.width, height: v.height });
      }
      const device = await ShadeDevice.open({
        backend,
        poolBytes: 0,
        workingBytes: 512 * 1024 ** 2,
      });
      try {
        const t0 = performance.now();
        const stage = {};
        const previews = {};
        let last = t0;
        const dense = await reconstructDenseGpu(device, scene, images, {
          ...options,
          onView: (e) => {
            if (e.type !== 'depth') return;
            const b = new Uint8Array(e.depth.buffer, e.depth.byteOffset, e.depth.byteLength);
            let t = '';
            for (let i = 0; i < b.length; i += 8192)
              t += String.fromCharCode(...b.subarray(i, i + 8192));
            previews[e.viewId] = { width: e.width, height: e.height, depth: btoa(t) };
          },
          onProgress: (e) => {
            if (e.done === e.total) {
              const now = performance.now();
              stage[e.stage] = now - last;
              last = now;
            }
          },
        });
        const ms = performance.now() - t0;
        const bytes = new Uint8Array(
          dense.points.positions.buffer,
          dense.points.positions.byteOffset,
          dense.points.positions.byteLength,
        );
        let s = '';
        for (let i = 0; i < bytes.length; i += 8192)
          s += String.fromCharCode(...bytes.subarray(i, i + 8192));
        return {
          ms,
          stage,
          previews,
          count: dense.points.count,
          positions: btoa(s),
          views: dense.views.map((v) => ({
            id: v.id,
            status: v.status,
            sources: v.sources,
            depthPixels: v.depthPixels,
            firstPoint: v.firstPoint,
            points: v.points,
          })),
        };
      } finally {
        await device.dispose();
      }
    },
    { scene, options, backend },
    { timeoutMs: 7200000, serve },
  );
  if (errors.length) console.error(errors);

  const raw = Buffer.from(result.positions, 'base64');
  const xyz = new Float32Array(raw.buffer, raw.byteOffset, 3 * result.count);
  const perView = [];
  for (const v of scene.views) {
    const W = v.width,
      H = v.height,
      P = W * H;
    const { R, t } = v.pose,
      cam = v.camera;
    const hit = new Uint8Array(P);
    let inFront = 0;
    const depths = [];
    for (let i = 0; i < result.count; i++) {
      const X = xyz[3 * i],
        Y = xyz[3 * i + 1],
        Z = xyz[3 * i + 2];
      const z = R[6] * X + R[7] * Y + R[8] * Z + t[2];
      if (!(z > 0)) continue;
      const u = Math.round((cam.fx * (R[0] * X + R[1] * Y + R[2] * Z + t[0])) / z + cam.cx),
        w = Math.round((cam.fy * (R[3] * X + R[4] * Y + R[5] * Z + t[1])) / z + cam.cy);
      if (u < 0 || w < 0 || u >= W || w >= H) continue;
      inFront++;
      depths.push(z);
      for (let dy = -1; dy <= 1; dy++)
        for (let dx = -1; dx <= 1; dx++) {
          const x = u + dx,
            y = w + dy;
          if (x >= 0 && y >= 0 && x < W && y < H) hit[y * W + x] = 1;
        }
    }
    const quarters = [];
    for (let q = 0; q < 4; q++) {
      let c = 0;
      const y0 = Math.floor((q * H) / 4),
        y1 = Math.floor(((q + 1) * H) / 4);
      for (let y = y0; y < y1; y++) for (let x = 0; x < W; x++) c += hit[y * W + x];
      quarters.push(+(c / ((y1 - y0) * W)).toFixed(3));
    }
    depths.sort((a, b) => a - b);
    // The depth previews (a quarter of the size, the mean of each 4 x 4 block, 0 without depth):
    // the share of blocks with a depth in each row quarter, and of those, the share beyond the
    // far bound of the range before the amendment (1.25 x the 99th percentile of the observed
    // depths) and the median depth.
    const pv = result.previews[v.id];
    const pb = Buffer.from(pv.depth, 'base64');
    const pd = new Float32Array(pb.buffer, pb.byteOffset, pv.width * pv.height);
    const oldFar = 1.25 * observedP99(v);
    const previewQuarters = [];
    for (let q = 0; q < 4; q++) {
      const y0 = Math.floor((q * pv.height) / 4),
        y1 = Math.floor(((q + 1) * pv.height) / 4);
      const valid = [];
      for (let y = y0; y < y1; y++)
        for (let x = 0; x < pv.width; x++)
          if (pd[y * pv.width + x] > 0) valid.push(pd[y * pv.width + x]);
      valid.sort((a, b) => a - b);
      previewQuarters.push({
        valid: +(valid.length / ((y1 - y0) * pv.width)).toFixed(3),
        beyondOldFar: valid.length
          ? +(valid.filter((d) => d > oldFar).length / valid.length).toFixed(3)
          : 0,
        medianDepth: valid.length ? +valid[valid.length >> 1].toFixed(2) : null,
      });
    }
    const entry = plan.views.find((e) => e.id === v.id);
    const run = result.views.find((e) => e.id === v.id);
    const scale = plan.normalization.scale;
    perView.push({
      id: v.id,
      status: run.status,
      // The range in the model's units (the plan holds it in normalized units).
      range: entry.depthRange?.map((d) => +(d * scale).toFixed(2)),
      landmarks: entry.landmarks,
      seeds:
        entry.index >= 0 ? plan.seedOffsets[entry.index + 1] - plan.seedOffsets[entry.index] : 0,
      validShare: +(run.depthPixels / P).toFixed(4),
      coverageByRowQuarter: quarters,
      depthPreviewByRowQuarter: previewQuarters,
      coverage: +(hit.reduce((a, b) => a + b, 0) / P).toFixed(4),
      ownPoints: run.points,
      // Depths at which the whole cloud lands in this view: 50th and 99th percentile.
      cloudDepthP50P99: depths.length
        ? [
            +depths[Math.floor(0.5 * depths.length)].toFixed(2),
            +depths[Math.floor(0.99 * (depths.length - 1))].toFixed(2),
          ]
        : null,
    });
  }
  const mean = (f) => +(perView.reduce((a, v) => a + f(v), 0) / perView.length).toFixed(4);
  const record = {
    scene: sceneName,
    backend,
    scale: k,
    size: [views[0].width, views[0].height],
    viewIds: views.map((v) => v.id),
    options: { near: near ?? 0.75, far: far ?? 'default', iterations: iterations ?? 'default' },
    landmarks: landmarks.length,
    points: result.count,
    denseMs: Math.round(result.ms),
    stageMs: Object.fromEntries(Object.entries(result.stage).map(([a, b]) => [a, Math.round(b)])),
    wallMs: Date.now() - started,
    meanValidShare: mean((v) => v.validShare),
    meanCoverageByRowQuarter: [0, 1, 2, 3].map((q) => mean((v) => v.coverageByRowQuarter[q])),
    views: perView,
  };
  console.log(JSON.stringify(record, null, 1));
  if (flag('--out')) {
    await writeFile(flag('--out'), JSON.stringify(record, null, 1) + '\n');
    await writeFile(flag('--out').replace(/\.json$/, '.bin'), raw);
    // The depth previews: per view, width, height and f32 depths (model units).
    await writeFile(flag('--out').replace(/\.json$/, '.previews.json'), JSON.stringify(result.previews));
  }
} finally {
  await rm(scratch, { recursive: true, force: true });
}
