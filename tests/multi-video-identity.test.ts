import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { cameraCenter } from '../src/geometry/pose.ts';
import {
  assignViewIds,
  frameKeyOf,
  pairSeed,
  videoIdOf,
  type KeyframeIdentity,
} from '../src/reconstruction/identity.ts';
import {
  executeReconstructionJob,
  incrementalControls,
  incrementalSteps,
  verifyPairs,
  type IncrementalResult,
} from '../src/reconstruction/mapper.ts';
import type { ImageFeatures, PairMatches } from '../src/reconstruction/types.ts';
import { seededRandom } from '../src/geometry/numerics.ts';
import { scene } from './scene.ts';

// Verifies: docs/multi-video.md 2.4 (identity), 4.4 (tie-breaks), 7.1 (CPU tier of the shuffle gate).

const MIB = 1024 * 1024;

function reference(parts: Uint8Array[]): string {
  const hash = createHash('sha256');
  for (const part of parts) hash.update(part);
  return hash.digest('hex').slice(0, 16);
}

function sizeBytes(size: number): Uint8Array {
  const out = new Uint8Array(8);
  new DataView(out.buffer).setBigUint64(0, BigInt(size));
  return out;
}

test('videoId hashes the size, the first MiB and the last MiB of a file', async () => {
  const big = new Uint8Array(3 * MIB + 5).map((_, i) => (i * 31 + (i >> 8)) & 255);
  assert.equal(
    await videoIdOf(new Blob([big])),
    reference([sizeBytes(big.length), big.subarray(0, MIB), big.subarray(big.length - MIB)]),
  );
  // Bytes in the middle do not count: the id is cheap for large files.
  const changed = big.slice();
  changed[MIB + 10] ^= 255;
  assert.equal(await videoIdOf(new Blob([changed])), await videoIdOf(new Blob([big])));
  // A byte in the first MiB, a byte in the last MiB and the size all count.
  for (const index of [3, big.length - 3]) {
    const edited = big.slice();
    edited[index] ^= 1;
    assert.notEqual(await videoIdOf(new Blob([edited])), await videoIdOf(new Blob([big])));
  }
  assert.notEqual(await videoIdOf(new Blob([big.subarray(1)])), await videoIdOf(new Blob([big])));
});

test('videoId hashes a small file once, and an empty file', async () => {
  const small = Uint8Array.from({ length: 1000 }, (_, i) => i & 255);
  assert.equal(
    await videoIdOf(new Blob([small])),
    reference([sizeBytes(1000), small, new Uint8Array(0)]),
  );
  const two = new Uint8Array(MIB + 100).fill(7);
  assert.equal(
    await videoIdOf(new Blob([two])),
    reference([sizeBytes(two.length), two.subarray(0, MIB), two.subarray(MIB)]),
  );
  assert.equal(await videoIdOf(new Blob([])), reference([sizeBytes(0)]));
  assert.match(await videoIdOf(new Blob([small])), /^[0-9a-f]{16}$/);
});

test('frameKey is the hash of the RGBA bytes, whatever the view onto them', async () => {
  const rgba = Uint8ClampedArray.from({ length: 64 }, (_, i) => (i * 5) & 255);
  const key = await frameKeyOf(rgba);
  assert.equal(key, reference([new Uint8Array(rgba.buffer)]));
  assert.equal(await frameKeyOf(new Uint8Array(rgba)), key);
  const padded = new Uint8Array(100);
  padded.set(rgba, 20);
  assert.equal(await frameKeyOf(padded.subarray(20, 84)), key);
  const edited = rgba.slice();
  edited[63] ^= 1;
  assert.notEqual(await frameKeyOf(edited), key);
});

const keyframes: KeyframeIdentity[] = [
  { frameKey: 'bbbb', videoId: '2222', time: 0.5 },
  { frameKey: 'aaaa', videoId: '2222', time: 0 },
  { frameKey: 'aaaa', videoId: '1111', time: 9 },
  { frameKey: 'bbbb', videoId: '2222', time: 0.25 },
  { frameKey: 'aaaa', videoId: '1111', time: 1.5 },
];

test('viewId is the rank by frameKey, then videoId, then time', () => {
  // Sorted: (aaaa, 1111, 1.5), (aaaa, 1111, 9), (aaaa, 2222, 0), (bbbb, 2222, 0.25), (bbbb, 2222, 0.5).
  assert.deepEqual(assignViewIds(keyframes), [4, 2, 1, 3, 0]);
});

test('viewId does not depend on the order of the input', () => {
  const ids = assignViewIds(keyframes);
  const rng = seededRandom(5);
  for (let round = 0; round < 20; round++) {
    const order = keyframes.map((_, i) => i).sort(() => rng() - 0.5);
    const shuffled = assignViewIds(order.map((i) => keyframes[i]!));
    order.forEach((original, position) => assert.equal(shuffled[position], ids[original]));
  }
  assert.deepEqual(
    [...ids].sort((a, b) => a - b),
    [0, 1, 2, 3, 4],
  );
});

test('viewId refuses two keyframes with one identity', () => {
  assert.throws(() => assignViewIds([keyframes[0]!, { ...keyframes[0]! }]), RangeError);
  assert.throws(() => assignViewIds([{ ...keyframes[0]!, time: NaN }]), RangeError);
});

test('pairSeed depends on the two ids and the run seed only', () => {
  assert.equal(pairSeed(17, 3, 9), pairSeed(17, 9, 3));
  assert.notEqual(pairSeed(17, 3, 9), pairSeed(18, 3, 9));
  const seeds = new Set<number>();
  for (let a = 0; a < 30; a++) for (let b = a + 1; b < 30; b++) seeds.add(pairSeed(17, a, b));
  assert.equal(seeds.size, (30 * 29) / 2);
  for (const s of seeds) assert.ok(Number.isSafeInteger(s) && s >= 0 && s <= 0xffffffff);
});

// A synthetic scene of nine keyframes in three videos (an orbit, an ascent and a lateral pass in
// the record; here three runs of consecutive views). Each keyframe has a content key.
const VIDEOS = ['v-a', 'v-b', 'v-c'].map((label) =>
  createHash('sha256').update(label).digest('hex').slice(0, 16),
);

function labelled(noise: number) {
  const s = scene({ count: 120, views: 9, noise, seed: 91 });
  const identity: KeyframeIdentity[] = s.inputs.map((input, i) => ({
    frameKey: createHash('sha256').update(`frame ${input.id}`).digest('hex').slice(0, 16),
    videoId: VIDEOS[Math.floor(i / 3)]!,
    time: (i % 3) * 0.25,
  }));
  return { ...s, identity };
}

/** Relabel the scene's views with the given ids, in the given order of images and pairs. */
function relabel(
  s: ReturnType<typeof labelled>,
  ids: readonly number[],
  imageOrder: readonly number[],
  pairOrder: readonly number[],
): { inputs: ImageFeatures[]; pairs: PairMatches[] } {
  const idOf = new Map(s.inputs.map((input, i) => [input.id, ids[i]!]));
  const inputs = imageOrder.map((i) => ({ ...s.inputs[i]!, id: ids[i]! }));
  const pairs = pairOrder.map((p) => {
    const pair = s.pairs[p]!;
    const first = idOf.get(pair.first)!,
      second = idOf.get(pair.second)!;
    // The pair is stored with its ids in the order the caller met them.
    return p % 2
      ? { ...pair, first: second, second: first, matches: pair.matches.map((_, k, m) => m[k ^ 1]!) }
      : { ...pair, first, second };
  });
  return { inputs, pairs };
}

function run(inputs: ImageFeatures[], pairs: PairMatches[]): IncrementalResult {
  const machine = incrementalSteps(inputs, pairs, { seed: 7, ransacIterations: 200 });
  let step = machine.next();
  while (!step.done) step = machine.next(executeReconstructionJob(step.value));
  return step.value;
}

/** Cameras of a result keyed by frame key, so that runs with different ids compare. */
function byFrameKey(
  result: IncrementalResult,
  s: ReturnType<typeof labelled>,
  ids: readonly number[],
) {
  const keyOf = new Map(ids.map((id, i) => [id, s.identity[i]!.frameKey]));
  return new Map(
    result.reconstruction.views.map((v) => [
      keyOf.get(v.id)!,
      { R: Array.from(v.pose.R), c: Array.from(cameraCenter(v.pose)) },
    ]),
  );
}

function shuffleOf(n: number, seed: number): number[] {
  const rng = seededRandom(seed),
    order = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [order[i], order[j]] = [order[j]!, order[i]!];
  }
  return order;
}

test('shuffled input order with canonical ids gives the same map (shuffle gate, CPU tier)', () => {
  const s = labelled(0.3);
  const n = s.inputs.length,
    m = s.pairs.length;
  // Canonical ids come from content. The list order of the keyframes is irrelevant to them.
  const ids = assignViewIds(s.identity);
  const identityOrder = Array.from({ length: n }, (_, i) => i),
    identityPairs = Array.from({ length: m }, (_, i) => i);
  const base = run(
    ...(Object.values(relabel(s, ids, identityOrder, identityPairs)) as [
      ImageFeatures[],
      PairMatches[],
    ]),
  );
  assert.equal(base.reconstruction.views.length, n);
  const reference = byFrameKey(base, s, ids);
  const baseOrder = base.diagnostics.registrationOrder.map((id) => ids.indexOf(id));
  for (const round of [1, 2, 3, 4]) {
    // The keyframes arrive in another order: ids from the shuffled list, the same per keyframe.
    const arrival = shuffleOf(n, round);
    const shuffledIds = assignViewIds(arrival.map((i) => s.identity[i]!));
    const idsByKeyframe = new Array<number>(n);
    arrival.forEach((keyframe, position) => (idsByKeyframe[keyframe] = shuffledIds[position]!));
    assert.deepEqual(idsByKeyframe, ids);
    const { inputs, pairs } = relabel(s, idsByKeyframe, arrival, shuffleOf(m, 10 + round));
    const result = run(inputs, pairs);
    assert.deepEqual(
      result.diagnostics.registrationOrder.map((id) => ids.indexOf(id)),
      baseOrder,
    );
    assert.deepEqual(byFrameKey(result, s, ids), reference);
    assert.equal(result.reconstruction.landmarks.length, base.reconstruction.landmarks.length);
  }
});

test('the shuffle gate can fail: ids in arrival order change the seed pair', () => {
  const s = labelled(0.3);
  const n = s.inputs.length;
  const identity = Array.from({ length: n }, (_, i) => i);
  const outcome = (arrival: readonly number[]) => {
    // Non-canonical: the id is the position in the arrival list.
    const idsByKeyframe = new Array<number>(n);
    arrival.forEach((keyframe, position) => (idsByKeyframe[keyframe] = position));
    const { inputs, pairs } = relabel(
      s,
      idsByKeyframe,
      arrival,
      s.pairs.map((_, i) => i),
    );
    const result = run(inputs, pairs);
    return result.diagnostics.registrationOrder.map((id) => arrival[id]!);
  };
  const reference = outcome(identity);
  const differs = [1, 2, 3, 4, 5, 6].some(
    (round) => JSON.stringify(outcome(shuffleOf(n, round))) !== JSON.stringify(reference),
  );
  assert.ok(
    differs,
    'arrival-order ids must change the registration order for the gate to be able to fail',
  );
});

test('verifyPairs seeds a pair from its ids, not from its place in the list', () => {
  const s = labelled(0);
  const byId = new Map(s.inputs.map((v) => [v.id, v]));
  const seedsOf = (pairs: PairMatches[]) => {
    const { controls } = incrementalControls(s.inputs, pairs, { seed: 17, ransacIterations: 50 });
    const machine = verifyPairs(byId, pairs, controls);
    const seeds: number[] = [];
    let step = machine.next();
    while (!step.done) {
      const job = step.value;
      if (job.kind !== 'verifyPair') throw new Error('unexpected job');
      seeds.push(job.args[3]?.seed ?? NaN);
      step = machine.next(executeReconstructionJob(job));
    }
    return seeds;
  };
  const sorted = [...s.pairs].sort((a, b) => a.first - b.first || a.second - b.second);
  const full = seedsOf(s.pairs);
  assert.deepEqual(
    full,
    sorted.map((p) => pairSeed(17, p.first, p.second)),
  );
  // Dropping the first pair and the sixth pair moves every later index. No other seed changes.
  const kept = sorted.filter((_, i) => i !== 0 && i !== 5);
  assert.deepEqual(
    seedsOf(kept),
    full.filter((_, i) => i !== 0 && i !== 5),
  );
});
