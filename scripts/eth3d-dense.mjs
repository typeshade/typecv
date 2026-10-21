// Scores the dense pipeline against a laser-scan ground truth (docs/dense-mvs.md, section 10).
//
//   node scripts/eth3d-dense.mjs --scene <dir> --scan <dir> --out <dir> [--scale 8] [options]
//
// <dir> of --scene is an ETH3D scene of the "undistorted" archive (`dslr_calibration_undistorted/`
// and `images/dslr_images_undistorted/`). <dir> of --scan is `dslr_scan_eval/` of the "scan_eval"
// archive (`scan1.ply` and `scan_alignment.mlp`). Download from https://www.eth3d.net/datasets
// (for example `pipes_dslr_undistorted.7z` and `pipes_dslr_scan_eval.7z`).
//
// The sparse model is the dataset's own: the known cameras (COLMAP, in the frame of the scan) and
// its triangulated points as the landmarks. The dense stage is `reconstructDenseGpu` on headless
// WebGPU (SwiftShader without a physical GPU). The cleanup is the lab's (`cleanDenseCloud`, the CPU
// tier, `cluster.minFraction` 0.1) or another variant named by `--cleanups`.
//
// Options:
//   --scale k          divide the image size by k (default 8)
//   --cloud file.ply   score this dense PLY (binary, xyz and rgb) instead of running the stage
//   --cleanups list    comma list of: raw, lab, none-voxel ... (default raw,lab)
//   --taus list        thresholds in meters (default 0.01,0.02,0.05,0.1)
//   --mask r           the radius around the scan inside which a point is judged (default 0.5)
//   --coverage-k k     the landmark radius is k pixel footprints (default 4)
//   --gt-voxel s       decimate the scan to one point per cube of side s meters (default 0.004)
//   --working-mib m    the device's working budget in MiB (default 256)
//   --views n          use the first n views (default all)
//   --extra json       a JSON object merged into the options of `reconstructDenseGpu`, for
//                      example '{"fusion":"tsdf"}' for the TSDF fusion of section 10.4
//   --save-maps dir    write the filtered depth maps (depth, normal and, with the TSDF fusion,
//                      confidence per packed view) to dir after the depth stage
//   --maps dir         fuse the maps of a --save-maps run: no depth stage
//   --dump-tsdf dir    write the TSDF fusion's points and their attributes (positions, weights,
//                      free counts, tallies, normals) to dir, with the landmarks and their radii
// Writes <out>/dense-raw.ply, <out>/dense-<cleanup>.ply and <out>/record.json.
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { existsSync } from 'node:fs';
import { join, resolve } from 'node:path';
import sharp from 'sharp';
import { cleanDenseCloud } from '../src/reconstruction/cloud-clean.ts';
import {
  decimateByVoxel,
  landmarkCoverage,
  readPly,
  scoreCloud,
  takePositions,
  visibleGroundTruth,
} from '../src/reconstruction/dense-score.ts';
import { inBrowser } from '../tests/typeshade/browser.mjs';

const args = process.argv.slice(2);
const flag = (name, fallback) => (args.includes(name) ? args[args.indexOf(name) + 1] : fallback);
const sceneDir = resolve(flag('--scene', ''));
const scanDir = resolve(flag('--scan', ''));
const outDir = resolve(flag('--out', '.'));
const k = Number(flag('--scale', 8));
const taus = flag('--taus', '0.01,0.02,0.05,0.1').split(',').map(Number);
const maskRadius = Number(flag('--mask', 0.5));
const coverageK = Number(flag('--coverage-k', 4));
const gtVoxel = Number(flag('--gt-voxel', 0.004));
const workingMiB = Number(flag('--working-mib', 256));
const cleanups = flag('--cleanups', 'raw,lab').split(',');
const extra = JSON.parse(flag('--extra', '{}'));
const saveMaps = flag('--save-maps') && resolve(flag('--save-maps'));
const mapsDir = flag('--maps') && resolve(flag('--maps'));
const dumpTsdf = flag('--dump-tsdf') && resolve(flag('--dump-tsdf'));
if (!flag('--scene') || !flag('--scan')) throw new Error('Usage: see the header of this file');

// ---- The COLMAP text model of the scene ---------------------------------------------------------

const calibration = join(sceneDir, 'dslr_calibration_undistorted');
const cameras = new Map();
for (const line of (await readFile(join(calibration, 'cameras.txt'), 'utf8')).split('\n')) {
  if (!line || line.startsWith('#')) continue;
  const w = line.split(/\s+/);
  if (w[1] !== 'PINHOLE') throw new Error(`Camera model ${w[1]} is not supported`);
  cameras.set(Number(w[0]), {
    width: +w[2],
    height: +w[3],
    fx: +w[4],
    fy: +w[5],
    cx: +w[6],
    cy: +w[7],
  });
}
const shrink = (x) => (x + 0.5) / k - 0.5;
/** World to camera from the quaternion of COLMAP (w x y z). */
function rotation(qw, qx, qy, qz) {
  return [
    1 - 2 * (qy * qy + qz * qz),
    2 * (qx * qy - qz * qw),
    2 * (qx * qz + qy * qw),
    2 * (qx * qy + qz * qw),
    1 - 2 * (qx * qx + qz * qz),
    2 * (qy * qz - qx * qw),
    2 * (qx * qz - qy * qw),
    2 * (qy * qz + qx * qw),
    1 - 2 * (qx * qx + qy * qy),
  ];
}
const imageLines = (await readFile(join(calibration, 'images.txt'), 'utf8'))
  .split('\n')
  .filter((l) => !l.startsWith('#'));
const frames = [];
const points2d = new Map(); // image id -> [x, y][]
for (let i = 0; i + 1 < imageLines.length; i += 2) {
  const w = imageLines[i].split(/\s+/);
  if (w.length < 10) continue;
  const cam = cameras.get(Number(w[8]));
  const coords = imageLines[i + 1].trim().split(/\s+/).map(Number);
  const list = [];
  for (let j = 0; j + 2 < coords.length; j += 3) list.push([coords[j], coords[j + 1]]);
  points2d.set(Number(w[0]), list);
  frames.push({
    imageId: Number(w[0]),
    name: w[9],
    width: Math.round(cam.width / k),
    height: Math.round(cam.height / k),
    R: rotation(+w[1], +w[2], +w[3], +w[4]),
    t: [+w[5], +w[6], +w[7]],
    camera: { fx: cam.fx / k, fy: cam.fy / k, cx: shrink(cam.cx), cy: shrink(cam.cy) },
  });
}
frames.sort((a, b) => (a.name < b.name ? -1 : 1));
const viewCount = Number(flag('--views', frames.length));
frames.length = Math.min(frames.length, viewCount);
frames.forEach((f, i) => (f.id = i));
const idOf = new Map(frames.map((f) => [f.imageId, f.id]));
const landmarks = [];
for (const line of (await readFile(join(calibration, 'points3D.txt'), 'utf8')).split('\n')) {
  if (!line || line.startsWith('#')) continue;
  const w = line.split(/\s+/).map(Number);
  const observations = [];
  for (let j = 8; j + 1 < w.length; j += 2) {
    const id = idOf.get(w[j]);
    if (id === undefined) continue;
    const [x, y] = points2d.get(w[j])[w[j + 1]];
    observations.push({ viewId: id, pixel: [shrink(x), shrink(y)] });
  }
  if (observations.length >= 2) landmarks.push({ xyz: [w[1], w[2], w[3]], observations });
}

// ---- The ground truth ---------------------------------------------------------------------------

async function loadGroundTruth() {
  const mlp = await readFile(join(scanDir, 'scan_alignment.mlp'), 'utf8');
  const m = /<MLMatrix44>\s*([\s\S]*?)<\/MLMatrix44>/.exec(mlp)[1].trim().split(/\s+/).map(Number);
  const scan = readPly(await readFile(join(scanDir, 'scan1.ply')));
  const p = scan.positions;
  for (let i = 0; i < p.length; i += 3) {
    const x = p[i],
      y = p[i + 1],
      z = p[i + 2];
    p[i] = m[0] * x + m[1] * y + m[2] * z + m[3];
    p[i + 1] = m[4] * x + m[5] * y + m[6] * z + m[7];
    p[i + 2] = m[8] * x + m[9] * y + m[10] * z + m[11];
  }
  const decimated = decimateByVoxel(p, gtVoxel);
  const seen = visibleGroundTruth(
    decimated,
    frames.map((f) => ({ R: f.R, t: f.t, ...f.camera, width: f.width, height: f.height })),
    { cell: 4 },
  );
  return {
    raw: scan.count,
    decimated: decimated.length / 3,
    visible: takePositions(decimated, seen),
  };
}

// ---- The dense cloud ----------------------------------------------------------------------------

function writePly(path, cloud) {
  const header = `ply\nformat binary_little_endian 1.0\nelement vertex ${cloud.count}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n`;
  const head = Buffer.from(header);
  const body = Buffer.alloc(cloud.count * 15);
  for (let i = 0; i < cloud.count; i++) {
    for (let a = 0; a < 3; a++) body.writeFloatLE(cloud.positions[3 * i + a], 15 * i + 4 * a);
    for (let a = 0; a < 3; a++)
      body[15 * i + 12 + a] = cloud.colors ? cloud.colors[3 * i + a] : 128;
  }
  return writeFile(path, Buffer.concat([head, body]));
}

async function denseCloud() {
  const scratch = await mkdtemp(join(tmpdir(), 'eth3d-dense-'));
  try {
    const serve = {};
    for (const f of frames) {
      const { data } = await sharp(join(sceneDir, 'images', f.name))
        .resize(f.width, f.height, { kernel: 'lanczos3', fit: 'fill' })
        .ensureAlpha()
        .raw()
        .toBuffer({ resolveWithObject: true });
      await writeFile(join(scratch, `${f.id}.rgba`), data);
      serve[`/data/${f.id}.rgba`] = join(scratch, `${f.id}.rgba`);
    }
    let maps = 0;
    if (mapsDir) {
      maps = JSON.parse(await readFile(join(mapsDir, 'maps.json'), 'utf8')).count;
      for (let v = 0; v < maps; v++)
        for (const kind of ['depth', 'normal', 'confidence'])
          if (existsSync(join(mapsDir, `${kind}-${v}.bin`)))
            serve[`/maps/${kind}-${v}.bin`] = join(mapsDir, `${kind}-${v}.bin`);
    }
    if (saveMaps) await mkdir(saveMaps, { recursive: true });
    if (dumpTsdf) await mkdir(dumpTsdf, { recursive: true });
    const scene = {
      views: frames.map((f) => ({
        id: f.id,
        width: f.width,
        height: f.height,
        pose: { R: f.R, t: f.t },
        camera: f.camera,
      })),
      landmarks,
    };
    const { result, errors } = await inBrowser(
      async ({ scene, workingMiB, extra, saveMaps, maps, dumpTsdf }) => {
        const { ShadeDevice, reconstructDenseGpu } = await import('/dist/internal/gpu.js');
        const internal = {};
        if (dumpTsdf)
          internal.onTsdfPoints = async (p) => {
            for (const kind of ['positions', 'normals', 'weights', 'free', 'tallies'])
              await fetch(`/upload/tsdf-${kind}.bin`, { method: 'PUT', body: p[kind] });
          };
        if (saveMaps)
          internal.onDepthMaps = async (m) => {
            for (let v = 0; v < m.depth.length; v++)
              for (const kind of ['depth', 'normal', 'confidence'])
                if (m[kind]?.[v])
                  await fetch(`/upload/${kind}-${v}.bin`, { method: 'PUT', body: m[kind][v] });
            await fetch('/upload/maps.json', {
              method: 'PUT',
              body: JSON.stringify({ count: m.depth.length }),
            });
          };
        if (maps) {
          const read = async (kind, v, Type) => {
            const response = await fetch(`/maps/${kind}-${v}.bin`);
            return response.ok ? new Type(await response.arrayBuffer()) : undefined;
          };
          const depthMaps = { depth: [], normal: [], confidence: [] };
          for (let v = 0; v < maps; v++) {
            depthMaps.depth.push(await read('depth', v, Float32Array));
            depthMaps.normal.push(await read('normal', v, Uint32Array));
            const confidence = await read('confidence', v, Float32Array);
            if (confidence) depthMaps.confidence.push(confidence);
          }
          if (!depthMaps.confidence.length) delete depthMaps.confidence;
          internal.depthMaps = depthMaps;
        }
        const images = new Map();
        for (const v of scene.views) {
          const rgba = new Uint8Array(await (await fetch(`/data/${v.id}.rgba`)).arrayBuffer());
          images.set(v.id, { rgba, width: v.width, height: v.height });
        }
        const device = await ShadeDevice.open({
          backend: 'webgpu',
          poolBytes: 0,
          workingBytes: workingMiB * 1024 ** 2,
        });
        try {
          const t0 = performance.now();
          const stages = {};
          const dense = await reconstructDenseGpu(device, scene, images, {
            ...extra,
            ...internal,
            onProgress: (e) => {
              stages[e.stage] = performance.now() - t0;
            },
          });
          const ms = performance.now() - t0;
          const b64 = (a) => {
            const bytes = new Uint8Array(a.buffer, a.byteOffset, a.byteLength);
            let s = '';
            for (let i = 0; i < bytes.length; i += 8192)
              s += String.fromCharCode(...bytes.subarray(i, i + 8192));
            return btoa(s);
          };
          return {
            ms,
            stages,
            count: dense.points.count,
            positions: b64(dense.points.positions.subarray(0, 3 * dense.points.count)),
            colors: b64(dense.points.colors.subarray(0, 3 * dense.points.count)),
            peakBytes: dense.residency?.peakBytes,
            tsdf: dense.tsdf && { ...dense.tsdf, report: undefined },
            dispatches: dense.report.dispatches,
            kernels: device.statistics.kernels.filter((k) => /tsdf|Fuse|Scatter|Flag/.test(k.name)),
            viewStatus: dense.views.map((v) => v.status),
          };
        } finally {
          await device.dispose();
        }
      },
      { scene, workingMiB, extra, saveMaps: Boolean(saveMaps), maps, dumpTsdf: Boolean(dumpTsdf) },
      { timeoutMs: 7200000, serve, uploads: saveMaps || dumpTsdf },
    );
    if (errors.length) console.error(errors);
    const bytes = (s) => {
      const b = Buffer.from(s, 'base64');
      return new Uint8Array(b.buffer, b.byteOffset, b.byteLength);
    };
    const positions = new Float32Array(bytes(result.positions).slice().buffer);
    const colors = bytes(result.colors).slice();
    return {
      cloud: { count: result.count, positions, colors, normals: new Int16Array(2 * result.count) },
      timing: {
        totalMs: Math.round(result.ms),
        stageEndMs: result.stages,
        // The fusion stage: from the end of the last geometric pass, or the whole run with --maps.
        fusionMs: Math.round(result.ms - (result.stages.geometric ?? 0)),
        peakBytes: result.peakBytes,
        dispatches: result.dispatches,
        kernels: result.kernels,
      },
      tsdf: result.tsdf,
      viewStatus: result.viewStatus,
    };
  } finally {
    await rm(scratch, { recursive: true, force: true });
  }
}

// ---- Run ----------------------------------------------------------------------------------------

await mkdir(outDir, { recursive: true });
const record = {
  scene: sceneDir.split('/').slice(-1)[0],
  scale: k,
  size: [frames[0].width, frames[0].height],
  views: frames.length,
  landmarks: landmarks.length,
  taus,
  maskRadius,
  coverageK,
  options: extra,
};
let dense;
if (flag('--cloud')) {
  const ply = readPly(await readFile(flag('--cloud')));
  dense = {
    cloud: {
      count: ply.count,
      positions: ply.positions,
      colors: ply.colors,
      normals: new Int16Array(2 * ply.count),
    },
  };
} else dense = await denseCloud();
Object.assign(record, { timing: dense.timing, tsdf: dense.tsdf, viewStatus: dense.viewStatus });
if (mapsDir) record.maps = mapsDir;
await writePly(join(outDir, 'dense-raw.ply'), dense.cloud);

const gt = await loadGroundTruth();
record.groundTruth = {
  rawPoints: gt.raw,
  afterVoxel: gt.decimated,
  visible: gt.visible.length / 3,
  voxel: gtVoxel,
};

// The radius of a landmark: coverageK pixel footprints at its nearest observing view.
const coverageLandmarks = landmarks.map((l) => {
  let z = Infinity;
  for (const o of l.observations) {
    const f = frames[o.viewId];
    const d = f.R[6] * l.xyz[0] + f.R[7] * l.xyz[1] + f.R[8] * l.xyz[2] + f.t[2];
    if (d > 0 && d < z) z = d;
  }
  const f = frames[l.observations[0].viewId];
  return { xyz: l.xyz, radius: (coverageK * z) / ((f.camera.fx + f.camera.fy) / 2) };
});

if (dumpTsdf) {
  await mkdir(dumpTsdf, { recursive: true });
  await writeFile(join(dumpTsdf, 'landmarks.json'), JSON.stringify(coverageLandmarks));
}
record.results = {};
for (const name of cleanups) {
  let cloud = dense.cloud,
    report;
  if (name === 'lab') {
    const result = await cleanDenseCloud(cloud, { cluster: { minFraction: 0.1 } });
    cloud = result.cloud;
    report = result.report;
  } else if (name !== 'raw') throw new Error(`Unknown cleanup ${name}`);
  if (name !== 'raw') await writePly(join(outDir, `dense-${name}.ply`), cloud);
  const started = performance.now();
  const score = scoreCloud(cloud.positions, gt.visible, { taus, maskRadius });
  const coverage = landmarkCoverage(cloud.positions, coverageLandmarks);
  record.results[name] = {
    points: cloud.count,
    cleanReport: report,
    score,
    coverage,
    scoreMs: Math.round(performance.now() - started),
  };
}
await writeFile(join(outDir, 'record.json'), JSON.stringify(record, null, 1) + '\n');
const pct = (x) => (100 * x).toFixed(1).padStart(5);
for (const [name, r] of Object.entries(record.results)) {
  console.log(
    `${name}: ${r.points} points, ${pct(r.coverage.missingFraction)} % of landmarks uncovered`,
  );
  for (const row of r.score.rows)
    console.log(
      `  tau ${row.tau}: accuracy ${pct(row.accuracy)}  completeness ${pct(row.completeness)}  F ${pct(row.fscore)}`,
    );
}
