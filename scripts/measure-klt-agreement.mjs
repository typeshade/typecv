/**
 * Dump what `docs/klt-tracking.md` step 6 needs to judge the tracker against the descriptors: for
 * the keyframes of a video, the SIFT keypoints, the tracker's matches of every pair and the
 * descriptor matches (ratio 0.8, mutual) of the same pairs. `scripts/analyze-klt-agreement.py`
 * reads the file.
 *
 *   node scripts/measure-klt-agreement.mjs video.webm out.json [fps=4] [edge=640]
 *
 * The keyframes are the ones of the phase gate (`website/src/lib/keyframe-gate.ts`), the lab's
 * default. Every other frame goes through the tracker as RGBA, as the worker does. Needs ffmpeg,
 * Chromium (Playwright) with SwiftShader, and `npm run build`. A run takes minutes on SwiftShader.
 */
import { execFileSync, spawnSync } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { PhaseGate } from '../website/src/lib/keyframe-gate.ts';
import { inBrowser } from '../tests/typeshade/browser.mjs';

const [file, out, fpsArg = '4', edgeArg = '640'] = process.argv.slice(2);
if (!file || !out)
  throw new Error('Usage: node scripts/measure-klt-agreement.mjs video out.json [fps] [edge]');
const fps = Number(fpsArg),
  edge = Number(edgeArg);
const [sw, sh] = execFileSync('ffprobe', [
  '-v',
  'error',
  '-select_streams',
  'v:0',
  '-show_entries',
  'stream=width,height',
  '-of',
  'csv=p=0',
  file,
])
  .toString()
  .trim()
  .split(',')
  .map(Number);
const scale = Math.min(1, edge / Math.max(sw, sh)),
  width = Math.round(sw * scale),
  height = Math.round(sh * scale);
const raw = spawnSync(
  'ffmpeg',
  [
    '-v',
    'error',
    '-i',
    file,
    '-vf',
    `fps=${fps},scale=${width}:${height}`,
    '-f',
    'rawvideo',
    '-pix_fmt',
    'rgba',
    '-',
  ],
  { maxBuffer: 1 << 30 },
).stdout;
const size = width * height * 4,
  count = Math.floor(raw.length / size);
const gate = new PhaseGate(),
  keyframes = [];
for (let i = 0; i < count; i++) {
  const rgba = new Uint8ClampedArray(raw.buffer, raw.byteOffset + i * size, size),
    decision = gate.decide(rgba, width, height, i / fps, i === count - 1);
  if (decision.action === 'skip') gate.discard();
  else {
    gate.accept();
    keyframes.push(i);
  }
}
console.log(`${count} frames, ${keyframes.length} keyframes: ${keyframes.join(' ')}`);
const dir = mkdtempSync(join(tmpdir(), 'klt-agreement-'));
writeFileSync(join(dir, 'frames.rgba'), raw.subarray(0, count * size));
try {
  const { result, errors } = await inBrowser(
    async (input) => {
      const {
        ShadeDevice,
        createKltTracker,
        extractSiftGpuResident,
        kltOptionsFor,
        matchFloatDescriptorsGpu,
      } = await import('/dist/gpu.js');
      const all = new Uint8ClampedArray(await (await fetch('/frames.rgba')).arrayBuffer()),
        size = input.width * input.height * 4,
        device = await ShadeDevice.open({ backend: 'webgpu', poolBytes: 0 }),
        tracker = await createKltTracker(device, input.width, input.height),
        done = [],
        trackerPairs = [],
        descriptorPairs = [];
      try {
        for (let i = 0; i < input.count; i++) {
          const rgba = all.subarray(i * size, (i + 1) * size);
          if (!input.keyframes.includes(i)) {
            await tracker.push({ rgba }, { frameId: i, select: false });
            continue;
          }
          const handle = await extractSiftGpuResident(device, rgba, input.width, input.height, {
            maxFeatures: 16384,
            firstOctave: -1,
            clahe: false,
            grayForKlt: { levels: kltOptionsFor(input.width).levels },
          });
          const points = await handle.readKeypoints();
          await tracker.push({ sift: handle }, { frameId: i, select: false });
          const joined = await tracker.join({
            frameId: i,
            keypoints: points,
            views: done.map((d) => d.id),
          });
          for (const pair of joined.pairs)
            trackerPairs.push({
              first: pair.first,
              second: pair.second,
              matches: Array.from(pair.matches),
            });
          for (const earlier of done) {
            const matched = await matchFloatDescriptorsGpu(
              device,
              earlier.handle.descriptors,
              handle.descriptors,
              { ratio: 0.8, mutual: true },
            );
            descriptorPairs.push({
              first: earlier.id,
              second: i,
              matches: Array.from(matched.matches),
            });
          }
          done.push({ id: i, handle, points: Array.from(points) });
        }
        return {
          keyframes: done.map((d) => ({ id: d.id, points: d.points })),
          trackerPairs,
          descriptorPairs,
          stats: tracker.stats(),
        };
      } finally {
        await tracker.dispose();
        for (const d of done) await d.handle.dispose();
      }
    },
    { width, height, count, keyframes },
    { timeoutMs: 3_000_000, serve: { '/frames.rgba': join(dir, 'frames.rgba') } },
  );
  if (errors.length) console.error(errors.join('\n'));
  writeFileSync(out, JSON.stringify({ width, height, fps, ...result }));
  console.log(`wrote ${out}`);
} finally {
  rmSync(dir, { recursive: true, force: true });
}
