import { createHash } from 'node:crypto';
import { performance } from 'node:perf_hooks';
import { sortTsdfPoints, sortOwnedTsdfPoints } from '../src/reconstruction/tsdf.ts';

const mode = process.argv[2];
const count = Number(process.argv[3] ?? 2433283);
if (
  !['immutable', 'owned'].includes(mode) ||
  !Number.isSafeInteger(count) ||
  count < 0 ||
  count > 10000000
)
  throw new Error('Use immutable|owned and an integer count from 0 to 10000000');
const positions = new Float32Array(3 * count);
const normals = new Uint32Array(count);
const colors = new Uint32Array(count);
const weights = new Uint32Array(count);
const free = new Uint32Array(count);
const tallies = new Uint32Array(count);
const freeWeights = new Float32Array(count);
let seed = 0x5e27b93;
const next = () => (seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0);
for (let i = 0; i < count; i++) {
  positions[3 * i] = (next() % 100003) / 128 - 390;
  positions[3 * i + 1] = (next() % 100003) / 128 - 390;
  positions[3 * i + 2] = (next() % 100003) / 128 - 390;
  normals[i] = next();
  colors[i] = next();
  weights[i] = next() % 255;
  free[i] = next() % 64;
  tallies[i] = next();
  freeWeights[i] = (next() % 1048576) / 1024;
}
global.gc?.();
const before = process.memoryUsage();
const started = performance.now();
const result = (mode === 'owned' ? sortOwnedTsdfPoints : sortTsdfPoints)(
  positions,
  normals,
  colors,
  weights,
  free,
  tallies,
  freeWeights,
);
const elapsedMs = performance.now() - started;
const digest = createHash('sha256');
for (const key of ['positions', 'normals', 'colors', 'weights', 'free', 'tallies', 'freeWeights']) {
  const array = result[key];
  digest.update(new Uint8Array(array.buffer, array.byteOffset, array.byteLength));
}
console.log(
  JSON.stringify({
    mode,
    count,
    elapsedMs,
    before,
    after: process.memoryUsage(),
    peakRssKiB: process.resourceUsage().maxRSS,
    sha256: digest.digest('hex'),
  }),
);
