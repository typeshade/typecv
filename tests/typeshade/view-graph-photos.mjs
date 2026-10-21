// Photo-sample inputs of the view graph (docs/multi-video.md step 4), shared by the CPU-tier test
// and `scripts/measure-view-graph.mjs`. Frames are decoded with ffmpeg, so a caller checks
// `hasFfmpeg()` first. SIFT runs on the device that the caller opens.
import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { extractSiftGpuResident } from '../../dist/internal/gpu.js';
import { assignViewIds } from '../../src/reconstruction/identity.ts';

const SAMPLES = new URL('../../website/public/sample/', import.meta.url);

export function hasFfmpeg() {
  try {
    execFileSync('ffmpeg', ['-version'], { stdio: 'ignore' });
    return true;
  } catch {
    return false;
  }
}

/** Seeded generator, so the vocabulary does not change from run to run. */
function random(seed) {
  let s = seed >>> 0;
  return () => {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    return (s >>> 0) / 2 ** 32;
  };
}

/**
 * SIFT of the listed frames. `videos` maps a video label to `{ sample, frames }`. The result is
 * one entry for each frame, without view ids, global descriptors or video ids.
 */
export async function extractPhotoFrames(device, videos, { scale = 1, maxFeatures = 1024 } = {}) {
  const frames = [];
  for (const [label, { sample, frames: indices }] of Object.entries(videos)) {
    const meta = JSON.parse(readFileSync(new URL(`${sample}/sample.json`, SAMPLES), 'utf8'));
    for (const index of indices) {
      const frame = meta.frames[index];
      const width = Math.round(frame.width * scale),
        height = Math.round(frame.height * scale);
      const rgba = new Uint8Array(
        execFileSync(
          'ffmpeg',
          [
            '-v',
            'error',
            '-i',
            new URL(`${sample}/${frame.file}`, SAMPLES).pathname,
            '-vf',
            `scale=${width}:${height}`,
            '-f',
            'rawvideo',
            '-pix_fmt',
            'rgba',
            '-',
          ],
          { maxBuffer: 1 << 28 },
        ),
      );
      const sift = await extractSiftGpuResident(device, rgba, width, height, { maxFeatures });
      const keypoints = await sift.readKeypoints();
      // `read` returns the whole capacity of the buffer. The first `count` rows are the keypoints'.
      const descriptors = (
        await device.enqueue(() => device.read(sift.descriptors.descriptors, sift.count * 128))
      ).slice(0, sift.count * 128);
      await sift.dispose();
      const s = width / frame.width;
      frames.push({
        label,
        sample,
        index,
        time: frame.time ?? index,
        frameKey: createHash('sha256').update(rgba).digest('hex').slice(0, 16),
        camera: {
          fx: frame.camera.fx * s,
          fy: frame.camera.fy * s,
          cx: (frame.camera.cx + 0.5) * s - 0.5,
          cy: (frame.camera.cy + 0.5) * s - 0.5,
          k1: frame.camera.k1,
          k2: frame.camera.k2,
        },
        keypoints: Float64Array.from(keypoints),
        descriptors,
      });
    }
  }
  return frames;
}

/**
 * A stand-in for step 3's global descriptor (`docs/multi-video.md` 3.2): a bag of visual words
 * from a k-means vocabulary of the run's own descriptors, square-rooted counts times idf,
 * L2-normalized. The real vocabulary is fixed and trained offline (step 3).
 */
export function bagOfWords(frames, words = 128, seed = 7) {
  const rng = random(seed);
  const pool = [];
  for (const f of frames) for (let i = 0; i < f.descriptors.length / 128; i++) pool.push([f, i]);
  const centres = Float32Array.from({ length: words * 128 }, () => 0);
  for (let w = 0; w < words; w++) {
    const [f, i] = pool[Math.floor(rng() * pool.length)];
    centres.set(f.descriptors.subarray(i * 128, i * 128 + 128), w * 128);
  }
  const nearest = (d, offset) => {
    let best = 0,
      bestD = Infinity;
    for (let w = 0; w < words; w++) {
      let s = 0;
      for (let k = 0; k < 128; k++) {
        const t = d[offset + k] - centres[w * 128 + k];
        s += t * t;
      }
      if (s < bestD) {
        bestD = s;
        best = w;
      }
    }
    return best;
  };
  for (let iteration = 0; iteration < 6; iteration++) {
    const sum = new Float64Array(words * 128),
      count = new Uint32Array(words);
    for (const [f, i] of pool.filter((_, n) => n % 3 === 0)) {
      const w = nearest(f.descriptors, i * 128);
      count[w]++;
      for (let k = 0; k < 128; k++) sum[w * 128 + k] += f.descriptors[i * 128 + k];
    }
    for (let w = 0; w < words; w++)
      if (count[w])
        for (let k = 0; k < 128; k++) centres[w * 128 + k] = sum[w * 128 + k] / count[w];
  }
  const histograms = frames.map((f) => {
    const h = new Float64Array(words);
    for (let i = 0; i < f.descriptors.length / 128; i++) h[nearest(f.descriptors, i * 128)]++;
    return h;
  });
  const df = new Float64Array(words);
  for (const h of histograms) for (let w = 0; w < words; w++) if (h[w]) df[w]++;
  return histograms.map((h) => {
    const g = Float32Array.from(
      h,
      (c, w) => Math.sqrt(c) * (Math.log((frames.length + 1) / (df[w] + 1)) + 1),
    );
    const n = Math.hypot(...g) || 1;
    return g.map((x) => x / n);
  });
}

/** Keyframes for `buildViewGraph`: canonical view ids, video ids from the labels. */
export function photoKeyframes(frames, words) {
  const global = bagOfWords(frames, words);
  const videoId = (label) => createHash('sha256').update(label).digest('hex').slice(0, 16);
  const identity = frames.map((f) => ({
    frameKey: f.frameKey,
    videoId: videoId(f.label),
    time: f.time,
  }));
  const ids = assignViewIds(identity);
  return frames.map((f, i) => ({
    viewId: ids[i],
    videoId: identity[i].videoId,
    frameKey: f.frameKey,
    time: f.time,
    camera: f.camera,
    keypoints: f.keypoints,
    descriptors: f.descriptors,
    global: global[i],
  }));
}
