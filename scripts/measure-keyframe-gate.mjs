/**
 * Measure the phase keyframe gate (`website/src/lib/keyframe-gate.ts`) on a video file: the frames
 * it keeps for a range of thresholds, the correlation ratios, and the milliseconds of the estimate.
 *
 *   node scripts/measure-keyframe-gate.mjs video.mp4 [fps=4] [edge=640]
 *
 * ffmpeg decodes the video at `fps` frames per second to `edge`-pixel RGBA frames, like the lab's
 * sampling of an uploaded video. Needs ffmpeg and ffprobe on the PATH. `VERBOSE=1` prints the decision of every frame.
 */
import { execFileSync, spawnSync } from 'node:child_process';
import { PhaseGate } from '../website/src/lib/keyframe-gate.ts';

const [file, fpsArg = '4', edgeArg = '640'] = process.argv.slice(2);
if (!file) throw new Error('Usage: node scripts/measure-keyframe-gate.mjs video.mp4 [fps] [edge]');
const fps = Number(fpsArg),
  edge = Number(edgeArg);
const probe = execFileSync('ffprobe', [
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
const scale = Math.min(1, edge / Math.max(probe[0], probe[1]));
const width = Math.round(probe[0] * scale),
  height = Math.round(probe[1] * scale);
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
const frames = Array.from(
  { length: count },
  (_, i) => new Uint8ClampedArray(raw.buffer, raw.byteOffset + i * size, size),
);
console.log(`${file}: ${count} frames at ${fps} fps, ${width}x${height}`);

/** Run the gate over all frames; `fallback` decisions are kept (a stand-in for the legacy rule). */
function run(options) {
  const gate = new PhaseGate(options);
  const kept = [];
  const rows = [];
  frames.forEach((rgba, i) => {
    const time = i / fps;
    const decision = gate.decide(rgba, width, height, time, i === count - 1);
    rows.push({ i, ...decision });
    if (decision.action === 'skip') gate.discard();
    else (gate.accept(), kept.push(i));
  });
  return { kept, rows, stats: gate.stats };
}

const base = run({});
if (process.env.VERBOSE)
  console.log('\nframe  action    reason               shift(px at 128)  fraction  ratio');
for (const r of process.env.VERBOSE ? base.rows : [])
  console.log(
    String(r.i).padStart(4),
    r.action.padEnd(9),
    r.reason.padEnd(20),
    r.shift ? `(${r.shift.dx.toFixed(2)}, ${r.shift.dy.toFixed(2)})`.padEnd(18) : ''.padEnd(18),
    r.shift ? Math.hypot(r.shift.fx, r.shift.fy).toFixed(3) : '',
    r.shift ? r.shift.ratio.toFixed(1) : '',
  );
console.log('\nminShift sweep (minRatio 12):');
for (const minShift of [0.01, 0.015, 0.02, 0.03, 0.05, 0.08]) {
  const r = run({ minShift });
  console.log(`  ${minShift}: kept ${r.kept.length} of ${count}, fallbacks ${r.stats.fallbacks}`);
}
console.log('minRatio sweep (minShift 0.05):');
for (const minRatio of [4, 8, 12, 20, 40]) {
  const r = run({ minRatio });
  console.log(`  ${minRatio}: kept ${r.kept.length} of ${count}, fallbacks ${r.stats.fallbacks}`);
}
const ms = base.rows.map((r) => r.ms).sort((a, b) => a - b);
console.log(
  `\nms per frame (gate.decide, thumbnail+FFT+correlation): median ${ms[ms.length >> 1].toFixed(2)}, p95 ${ms[Math.floor(ms.length * 0.95)].toFixed(2)}, max ${ms.at(-1).toFixed(2)}`,
);
console.log(
  `estimate-only: ${(base.stats.estimateMs / Math.max(1, base.stats.estimated)).toFixed(2)} ms per estimate (${base.stats.estimated} estimates)`,
);
