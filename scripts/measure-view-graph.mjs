// Measures the view graph (docs/multi-video.md, step 4): the pair counts against N^2, the
// components, expansion on or off, and 8-bit against float32 descriptors.
//
//   npm run build
//   node scripts/measure-view-graph.mjs [--scale 1] [--features 1024] [--words 128]
//
// Inputs: a synthetic scene of three camera paths (CPU backend), and the photo samples of the
// lab (CPU tier of the device, ffmpeg decodes the JPEG files). The global descriptor is the
// stand-in of `tests/typeshade/view-graph-photos.mjs`, not step 3's.
import { ShadeDevice, gpuViewGraphBackend } from '../dist/internal/gpu.js';
import { buildViewGraph, cpuViewGraphBackend } from '../src/reconstruction/view-graph.ts';
import { pathScene } from '../tests/view-graph-scene.ts';
import {
  extractPhotoFrames,
  hasFfmpeg,
  photoKeyframes,
} from '../tests/typeshade/view-graph-photos.mjs';

const args = process.argv.slice(2);
const flag = (name, fallback) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? Number(args[i + 1]) : fallback;
};
const scale = flag('scale', 1),
  features = flag('features', 1024),
  words = flag('words', 128);

function summary(name, g) {
  const s = g.stats;
  console.log(
    `${name}: N=${s.keyframes} videos=${s.videos} N^2/2=${s.exhaustivePairs} candidates=${s.candidatePairs}` +
      ` (temporal ${s.temporalPairs}, retrieval ${s.retrievalPairs}, reserve ${s.reservePairs}, expansion ${s.expansionPairs})` +
      ` verified=${s.verifiedPairs} edges=${s.edges} cross-video=${s.crossVideoEdges}`,
  );
  console.log(
    `  components (size/videos): ${g.components.map((c) => `${c.viewIds.length}/${c.videoIds.length}`).join(' ')}`,
  );
  console.log(
    `  tags: ${g.videos.map((v) => `${v.videoId.slice(0, 4)}=${v.tag}(${v.share.toFixed(2)})`).join(' ')}`,
  );
}

function compareStorage(name, bits, floats) {
  const key = (e) => `${e.first}:${e.second}`;
  const f = new Map(floats.edges.map((e) => [key(e), e.inliers]));
  const common = bits.edges.filter((e) => f.has(key(e)));
  const rel = common.map((e) => (e.inliers - f.get(key(e))) / f.get(key(e)));
  const abs = rel.map(Math.abs);
  const mean = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : 0);
  const totalBits = common.reduce((x, e) => x + e.inliers, 0),
    totalFloat = common.reduce((x, e) => x + f.get(key(e)), 0);
  console.log(
    `${name}: 8-bit against float32: edges ${bits.edges.length} against ${floats.edges.length}, ` +
      `common ${common.length}, inliers ${totalBits} against ${totalFloat} ` +
      `(${(((totalBits - totalFloat) / totalFloat) * 100).toFixed(2)} %), ` +
      `mean |change| ${(mean(abs) * 100).toFixed(2)} %, max |change| ${(Math.max(0, ...abs) * 100).toFixed(1)} %, ` +
      `components equal: ${JSON.stringify(bits.components.map((c) => c.viewIds)) === JSON.stringify(floats.components.map((c) => c.viewIds))}`,
  );
}

async function measure(name, keyframes, backend, base) {
  const run = async (extra) => {
    const t = performance.now();
    const g = await buildViewGraph(keyframes, backend, { ...base, ...extra });
    return { g, ms: performance.now() - t };
  };
  const bits = await run({}),
    floats = await run({ descriptorStorage: 'float32' }),
    expanded = await run({ expansion: true });
  summary(`${name} [8-bit, expansion off]`, bits.g);
  compareStorage(name, bits.g, floats.g);
  summary(`${name} [8-bit, expansion on]`, expanded.g);
  const had = new Set(bits.g.edges.map((e) => `${e.first}:${e.second}`));
  const added = expanded.g.edges.filter((e) => !had.has(`${e.first}:${e.second}`));
  console.log(
    `${name}: expansion tried ${expanded.g.stats.expansionPairs} more pairs and added ${added.length} edges (${added.filter((e) => e.crossVideo).length} cross-video), ` +
      `components ${JSON.stringify(bits.g.components.map((c) => c.viewIds.length))} -> ${JSON.stringify(expanded.g.components.map((c) => c.viewIds.length))}` +
      `; build ${Math.round(bits.ms)} ms`,
  );
}

const synthetic = pathScene(
  [
    { label: 'orbit', cloud: 0, views: 12, from: -35, to: 35, radius: 9, height: -1 },
    { label: 'ascent', cloud: 0, views: 12, from: 5, to: 40, radius: 9, height: -3 },
    { label: 'pass', cloud: 0, views: 12, from: -60, to: -20, radius: 11, height: 1 },
    { label: 'elsewhere', cloud: 1, views: 12, from: -30, to: 30, radius: 9, height: -1 },
  ],
  { points: 100 },
);
await measure('synthetic 4 paths', synthetic.keyframes, cpuViewGraphBackend(), {
  verification: { ransacIterations: 150 },
});

if (hasFfmpeg()) {
  const device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    const t = performance.now();
    const frames = await extractPhotoFrames(
      device,
      {
        'gerrard-a': { sample: 'gerrard-hall', frames: [0, 1, 2, 3, 4, 5] },
        'gerrard-b': { sample: 'gerrard-hall', frames: [6, 7, 8, 9, 10, 11] },
        'person-hall': { sample: 'person-hall', frames: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] },
        'graham-hall': { sample: 'graham-hall', frames: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] },
      },
      { scale, maxFeatures: features },
    );
    console.log(
      `photo samples: ${frames.length} frames, SIFT ${Math.round(performance.now() - t)} ms`,
    );
    await measure('photos', photoKeyframes(frames, words), gpuViewGraphBackend(device), {
      verification: { ransacIterations: 300 },
    });
  } finally {
    await device.dispose();
  }
} else console.log('photo samples: not run (no ffmpeg)');
