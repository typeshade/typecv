// The bundle adjustment within any device budget, measured (docs/gpu-budget.md, section 6).
//
//   node scripts/measure-ba-budget.mjs --scene south-building-24 --max-points 1500 \
//     --budgets-kib default,1200,600 --out docs/experiments/gpu-budget/south-building-24-ba.json
//
// --scene name          a COLMAP sample of `website/public/sample` (poses and the dataset's SIFT
//                       matches, triangulated as in scripts/measure-mvs-range.mjs).
// --max-points n        keep the n landmarks with the most observations (default 1500), so that a
//                       run on SwiftShader takes minutes.
// --iterations n        LM iterations (default 4).
// --budgets-kib a,b,c   the device's `workingBytes` of each run in KiB, or `default` (512 MiB:
//                       the whole problem fits). The first run is the reference.
// --backend webgpu|cpu  headless WebGPU (SwiftShader without a physical GPU) or the CPU tier.
// --focal fixed|shared  `refineSharedFocal` off or on (default fixed).
//
// Per run: the time, the peak of the device's active bytes, whether the run streamed (the plan
// is in the window count of `checkBundlePoints`), the cost and the SHA-256 of the result.
// Equality of the hashes is the equality of the results, bit for bit.
import { readFile, writeFile, mkdir, mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { triangulateTrack } from '../src/geometry/multiview.ts';
import { inBrowser } from '../tests/typeshade/browser.mjs';

const args = process.argv.slice(2);
const flag = (name, fallback) => (args.includes(name) ? args[args.indexOf(name) + 1] : fallback);
const sceneName = flag('--scene', 'south-building-24');
const maxPoints = Number(flag('--max-points', 1500));
const iterations = Number(flag('--iterations', 4));
const backend = flag('--backend', 'webgpu');
const shared = flag('--focal', 'fixed') === 'shared';
const budgets = flag('--budgets-kib', 'default').split(',');
const out = flag('--out');

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

async function loadScene() {
  const dir = resolve(import.meta.dirname, '../website/public/sample', sceneName);
  const sample = JSON.parse(await readFile(join(dir, 'sample.json'), 'utf8'));
  const bytes = await readFile(join(dir, 'colmap-sift.bin'));
  const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  const keypoints = new Float32Array(buffer, 0, sample.features.keypointBytes / 4);
  const matches = new Uint16Array(buffer, sample.features.keypointBytes);
  const views = sample.frames.map((f) => ({
    id: f.id,
    camera: { ...f.camera },
    pose: { R: orthonormal(f.reference.R), t: f.reference.t },
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
        featureId: i,
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
        observations: kept.map(({ viewId, featureId, pixel }) => ({ viewId, featureId, pixel })),
      });
  }
  landmarks.sort((a, b) => b.observations.length - a.observations.length);
  return { views, landmarks: landmarks.slice(0, maxPoints) };
}

const scene = await loadScene();
// A deterministic perturbation, so that the optimizer has work to do.
const wave = (i, k) => Math.sin(12.9898 * (i + 1) + 78.233 * (k + 1));
const model = {
  views: scene.views.map((v, i) => ({
    id: v.id,
    camera: v.camera,
    pose: {
      R: v.pose.R,
      t: v.pose.t.map((x, j) => x + (i ? 0.004 * wave(i, j) : 0)),
    },
  })),
  landmarks: scene.landmarks.map((l, i) => ({
    trackId: i,
    xyz: l.xyz.map((x, j) => x + 0.004 * wave(i + 1000, j)),
    observations: l.observations,
  })),
};
const counts = {
  cameras: model.views.length,
  points: model.landmarks.length,
  observations: model.landmarks.reduce((n, l) => n + l.observations.length, 0),
};
console.log(JSON.stringify(counts));

const scratch = await mkdtemp(join(tmpdir(), 'measure-ba-budget-'));
try {
  await writeFile(join(scratch, 'model.json'), JSON.stringify(model));
  const { result, errors } = await inBrowser(
    async ({ budgets, backend, shared, iterations }) => {
      const { ShadeDevice, bundleAdjustGpu } = await import('/dist/gpu.js');
      const raw = await (await fetch('/data/model.json')).json();
      const reconstruction = {
        views: raw.views.map((v) => ({
          ...v,
          pose: { R: Float64Array.from(v.pose.R), t: Float64Array.from(v.pose.t) },
        })),
        landmarks: raw.landmarks.map((l) => ({ ...l, xyz: Float64Array.from(l.xyz) })),
      };
      const hex = async (text) =>
        [...new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text)))]
          .map((b) => b.toString(16).padStart(2, '0'))
          .join('');
      const runs = [];
      for (const budget of budgets) {
        const workingBytes = budget === 'default' ? 512 * 1024 ** 2 : Number(budget) * 1024;
        let peak = 0,
          device;
        const sample = () => {
          if (device)
            peak = Math.max(peak, device.statistics.liveBytes - device.statistics.pooledBytes);
        };
        device = await ShadeDevice.open({
          backend,
          poolBytes: 0,
          workingBytes,
          onProgress: sample,
        });
        try {
          const started = performance.now();
          const r = await bundleAdjustGpu(device, reconstruction, {
            iterations,
            huberPx: 1,
            refineSharedFocal: shared,
          });
          sample();
          const ms = performance.now() - started;
          const calls = r.execution.report.groups.flatMap((g) => g.calls);
          const text = JSON.stringify([
            r.reconstruction.views.map((v) => [...v.pose.R, ...v.pose.t, v.camera.fx]),
            r.reconstruction.landmarks.map((l) => [...l.xyz]),
            r.diagnostics,
            r.cgIterations,
          ]);
          runs.push({
            budgetKiB: budget,
            workingBytes,
            tier: r.execution.report.tier,
            ms,
            peakActiveBytes: peak,
            streamed: calls.includes('checkBundlePoints'),
            windows:
              calls.filter((c) => c === 'checkBundlePoints').length /
              Math.max(1, r.diagnostics.acceptedSteps + r.diagnostics.rejectedSteps),
            dispatches: r.execution.report.dispatches,
            cgIterations: r.cgIterations,
            acceptedSteps: r.diagnostics.acceptedSteps,
            initialRmsePx: r.diagnostics.initialRmsePx,
            finalRmsePx: r.diagnostics.finalRmsePx,
            sha256: await hex(text),
          });
        } finally {
          await device.dispose();
        }
      }
      return runs;
    },
    { budgets, backend, shared, iterations },
    { timeoutMs: 7200000, serve: { '/data/model.json': join(scratch, 'model.json') } },
  );
  if (errors.length) console.error(errors);
  const record = {
    scene: sceneName,
    focal: shared ? 'shared' : 'fixed',
    iterations,
    ...counts,
    runs: result,
  };
  console.log(JSON.stringify(record, null, 2));
  if (out) {
    await mkdir(dirname(resolve(out)), { recursive: true });
    await writeFile(resolve(out), JSON.stringify(record, null, 2) + '\n');
  }
} finally {
  await rm(scratch, { recursive: true, force: true });
}
