// Per-kernel GPU time of one pipeline stage on a headless WebGPU adapter (SwiftShader without a
// physical GPU), for docs/gpu-throughput.md. Every plan of the TypeShade device is wrapped: the
// report of `device.fuse()` names each group's calls and, with `timestamp-query`, its GPU time.
//
//   node scripts/profile-kernels.mjs --case mvs --scene south-building-24 --ids 0,1,2,3 --scale 2
//   node scripts/profile-kernels.mjs --case bundle --scene south-building-24 --ids 0,1,2,3,4,5
//
// --case mvs      `reconstructDenseGpu` (docs/dense-mvs.md) on a COLMAP sample's poses and landmarks.
// --case bundle   `bundleAdjustGpu` on the same poses, landmarks and a deterministic perturbation.
// --case sift     `extractSiftGpuResident` on the sample's frames.
// --case guided  Guided matching on consecutive frames, as `guidedMatches` of the lab's worker does it:
//                 SIFT, the plain match, the guide, the guided match. `--guide essential` (default)
//                 estimates E only (`estimateEssentialRansacGpu`). `--guide verify` runs the full
//                 `verifyImagePairGpu` and builds E from its pose, as the worker did before PR 3.
// --out file.json the table: for each call name, groups, GPU ms (null without timestamps) and wall
//                 ms of the plans that held it, plus the totals.
//
// SwiftShader is a CPU emulation. Its times rank the kernels by arithmetic and memory work. They are
// not the times of a physical GPU, and a kernel's share there can differ.
import { readFile, writeFile, mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import sharp from 'sharp';
import { triangulateTrack } from '../src/geometry/multiview.ts';
import { inBrowser } from '../tests/typeshade/browser.mjs';

const args = process.argv.slice(2);
const flag = (name, fallback) => (args.includes(name) ? args[args.indexOf(name) + 1] : fallback);
const caseName = flag('--case', 'mvs');
const sceneName = flag('--scene', 'south-building-24');
const k = Number(flag('--scale', 2));
const ids = flag('--ids')?.split(',').map(Number);
const out = flag('--out');
const iterations = flag('--iterations') === undefined ? undefined : Number(flag('--iterations'));
const guide = flag('--guide', 'essential');
if (!['essential', 'verify'].includes(guide)) throw new RangeError('--guide: essential|verify');
const shrink = (x) => (x + 0.5) / k - 0.5;
if (!['mvs', 'bundle', 'sift', 'pair', 'guided'].includes(caseName))
  throw new RangeError('--case: mvs|bundle|sift|pair|guided');

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
function orthonormal(R) {
  const x = unit([R[0], R[1], R[2]]);
  const b = [R[3], R[4], R[5]];
  const d = dot(x, b);
  const y = unit([b[0] - d * x[0], b[1] - d * x[1], b[2] - d * x[2]]);
  return [...x, ...y, ...cross(x, y)];
}

/** The sample's views (COLMAP poses) and the landmarks that its own SIFT matches triangulate to. */
async function loadScene() {
  const dir = resolve(import.meta.dirname, '../website/public/sample', sceneName);
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

const scratch = await mkdtemp(join(tmpdir(), 'profile-kernels-'));
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
  const { result, errors } = await inBrowser(
    async ({ scene, caseName, iterations, guide }) => {
      const api = await import('/dist/gpu.js');
      const device = await api.ShadeDevice.open({
        backend: 'webgpu',
        poolBytes: 0,
        workingBytes: 1024 * 1024 * 1024,
      });
      // Wrap the TypeShade device's fuse(): one record for each call name.
      const table = new Map();
      let plans = 0,
        planWallMs = 0,
        gpuMs = 0,
        dispatches = 0,
        submissions = 0;
      const fuse = device.shade.fuse.bind(device.shade);
      device.shade.fuse = async (body, options) => {
        const t0 = performance.now();
        const report = await fuse(body, options);
        const wall = performance.now() - t0;
        plans++;
        planWallMs += wall;
        dispatches += report.dispatches;
        submissions += report.submissions;
        const timed = report.groups.filter((g) => Number.isFinite(g.gpuTime));
        for (const g of report.groups) {
          const name = g.calls.join('+');
          const row = table.get(name) ?? { groups: 0, gpuNs: 0, timedGroups: 0, fused: g.fused };
          row.groups++;
          if (Number.isFinite(g.gpuTime)) {
            row.gpuNs += g.gpuTime;
            row.timedGroups++;
            gpuMs += g.gpuTime / 1e6;
          }
          table.set(name, row);
        }
        return report;
      };
      const images = new Map();
      for (const v of scene.views) {
        const rgba = new Uint8Array(await (await fetch(`/data/${v.id}.rgba`)).arrayBuffer());
        images.set(v.id, { rgba, width: v.width, height: v.height });
      }
      const t0 = performance.now();
      let detail = {};
      try {
        if (caseName === 'mvs') {
          const dense = await api.reconstructDenseGpu(device, scene, images, {
            ...(iterations === undefined
              ? {}
              : {
                  photometricIterations: iterations,
                  geometricIterations: Math.ceil(iterations / 2),
                }),
          });
          detail = { points: dense.points.count, views: dense.views.length };
        } else if (caseName === 'bundle') {
          const nextFeature = new Map();
          const perturbed = {
            views: scene.views.map((v, i) => ({
              id: v.id,
              camera: { ...v.camera },
              pose: {
                R: [...v.pose.R],
                t: v.pose.t.map((x, a) =>
                  i < 2 ? x : x + 0.005 * Math.sin((v.id + 1) * 0.73 + a * 1.17),
                ),
              },
            })),
            landmarks: scene.landmarks.map((l, n) => ({
              trackId: n,
              xyz: l.xyz.map((x, a) => x + 0.01 * Math.sin((n + 1) * 0.73 + a * 1.17)),
              observations: l.observations.map((o) => {
                const featureId = nextFeature.get(o.viewId) ?? 0;
                nextFeature.set(o.viewId, featureId + 1);
                return { viewId: o.viewId, featureId, pixel: [...o.pixel] };
              }),
            })),
          };
          const r = await api.bundleAdjustGpu(device, perturbed, {
            iterations: 4,
            refineSharedFocal: true,
          });
          detail = { iterations: r.iterations, finalCost: r.finalCost ?? null };
        } else if (caseName === 'pair') {
          // verifyImagePairGpu (the pair-verification RANSAC) on consecutive views, with the
          // correspondences that the sample's own matches triangulate.
          let inliers = 0;
          for (let i = 0; i + 1 < scene.views.length; i++) {
            const a = scene.views[i],
              b = scene.views[i + 1];
            const rows = [];
            for (const l of scene.landmarks) {
              const oa = l.observations.find((o) => o.viewId === a.id),
                ob = l.observations.find((o) => o.viewId === b.id);
              if (oa && ob) rows.push(oa.pixel[0], oa.pixel[1], ob.pixel[0], ob.pixel[1]);
            }
            const r = await api.verifyImagePairGpu(
              device,
              new Float64Array(rows),
              a.camera,
              b.camera,
              {
                iterations: 500,
                seed: 7 + i,
              },
            );
            inliers += r.geometry ? 1 : 0;
          }
          detail = { pairs: scene.views.length - 1, verified: inliers };
        } else if (caseName === 'guided') {
          const features = [];
          for (const v of scene.views) {
            const im = images.get(v.id);
            const sift = await api.extractSiftGpuResident(device, im.rgba, v.width, v.height, {
              maxFeatures: 2048,
              firstOctave: 0,
            });
            features.push({ v, sift, points: await sift.readKeypoints() });
          }
          const normalized = (camera, source) => {
            const out = new Float64Array(source.length);
            for (let i = 0; i < source.length; i += 2) {
              const xy = api.unprojectPixel(camera, [source[i], source[i + 1]]);
              out[i] = xy ? xy[0] : 1e6;
              out[i + 1] = xy ? xy[1] : 1e6;
            }
            return out;
          };
          // The guide only: the plain matches are outside the timed window.
          let plainTotal = 0,
            guidedTotal = 0,
            guideMs = 0,
            pairs = 0;
          for (let i = 0; i + 1 < features.length; i++) {
            const A = features[i],
              B = features[i + 1];
            const plain = await api.matchFloatDescriptorsGpu(
              device,
              A.sift.descriptors,
              B.sift.descriptors,
              { ratio: 0.8, mutual: true },
            );
            const n = plain.matches.length / 2;
            if (n < 16) continue;
            const pixels = new Float64Array(n * 4);
            for (let m = 0; m < n; m++) {
              const p = plain.matches[m * 2] * 2,
                q = plain.matches[m * 2 + 1] * 2;
              pixels.set([A.points[p], A.points[p + 1], B.points[q], B.points[q + 1]], m * 4);
            }
            const a = A.v.camera,
              b = B.v.camera;
            const threshold = 4 / Math.min(a.fx, a.fy, b.fx, b.fy);
            const started = performance.now();
            let E;
            try {
              if (guide === 'essential') {
                E = (
                  await api.estimateEssentialRansacGpu(device, pixels, a, b, {
                    iterations: 50,
                    minIterations: 50,
                    seed: 101 + B.v.id,
                    threshold,
                  })
                ).E;
              } else {
                const verified = await api.verifyImagePairGpu(device, pixels, a, b, {
                  iterations: 50,
                  seed: 101 + B.v.id,
                  threshold,
                  refinementIterations: 0,
                  maxReprojectionErrorPx: 4,
                  homographyThresholdPx: 4,
                  homographyIterations: 1,
                });
                if (verified.geometry) {
                  const { R, t } = verified.geometry;
                  const tx = [0, -t[2], t[1], t[2], 0, -t[0], -t[1], t[0], 0];
                  E = new Float64Array(9);
                  for (let r = 0; r < 3; r++)
                    for (let c = 0; c < 3; c++)
                      E[r * 3 + c] =
                        tx[r * 3] * R[c] + tx[r * 3 + 1] * R[3 + c] + tx[r * 3 + 2] * R[6 + c];
                }
              }
            } catch (error) {
              if (error?.name !== 'GeometryEstimationError') throw error;
            }
            guideMs += performance.now() - started;
            if (!E) continue;
            const guided = await api.matchFloatDescriptorsGpu(
              device,
              A.sift.descriptors,
              B.sift.descriptors,
              {
                ratio: 0.8,
                mutual: true,
                epipolar: {
                  fundamental: E,
                  firstPoints: normalized(a, A.points),
                  secondPoints: normalized(b, B.points),
                  maxLinePx: 4 / Math.max(a.fx, a.fy, b.fx, b.fy),
                  kernel: 'scalar',
                },
              },
            );
            pairs++;
            plainTotal += n;
            guidedTotal += guided.matches.length / 2;
          }
          for (const f of features) await f.sift.dispose();
          detail = { guide, pairs, plainMatches: plainTotal, guidedMatches: guidedTotal, guideMs };
        } else {
          let n = 0;
          for (const v of scene.views) {
            const im = images.get(v.id);
            const sift = await api.extractSiftGpuResident(device, im.rgba, v.width, v.height, {
              maxFeatures: 8192,
              firstOctave: -1,
            });
            n += sift.count ?? 0;
          }
          detail = { features: n };
        }
      } finally {
        const wallMs = performance.now() - t0;
        const rows = [...table].map(([name, r]) => ({
          name,
          groups: r.groups,
          fused: r.fused,
          gpuMs: r.timedGroups ? r.gpuNs / 1e6 : null,
        }));
        globalThis.__out = {
          wallMs,
          plans,
          planWallMs,
          gpuMs,
          dispatches,
          submissions,
          detail,
          rows,
        };
        await device.dispose();
      }
      return globalThis.__out;
    },
    { scene, caseName, iterations, guide },
    { timeoutMs: 7200000, serve },
  );
  if (errors.length) console.error(errors);
  rowsPrint(result);
  if (out) await writeFile(out, JSON.stringify(result, null, 2));
} finally {
  await rm(scratch, { recursive: true, force: true });
}

function rowsPrint(r) {
  const rows = r.rows.slice().sort((a, b) => (b.gpuMs ?? 0) - (a.gpuMs ?? 0));
  console.log(
    `case ${caseName}: wall ${r.wallMs.toFixed(0)} ms, plans ${r.plans} (wall in plans ${r.planWallMs.toFixed(0)} ms), ` +
      `dispatches ${r.dispatches}, submissions ${r.submissions}, GPU ${r.gpuMs.toFixed(0)} ms`,
    JSON.stringify(r.detail),
  );
  for (const row of rows)
    console.log(
      `${row.name.padEnd(48)} ${String(row.groups).padStart(6)} groups ${
        row.gpuMs === null ? '     n/a' : row.gpuMs.toFixed(1).padStart(9)
      } ms ${row.gpuMs === null ? '' : ((100 * row.gpuMs) / r.gpuMs).toFixed(1).padStart(5) + '%'}`,
    );
}
