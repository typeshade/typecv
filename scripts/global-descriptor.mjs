/**
 * Train the vocabulary of the global descriptor and measure its retrieval recall
 * (docs/multi-video.md, 3.2 and step 3).
 *
 *   node scripts/global-descriptor-data.mjs download <work-dir>
 *   node scripts/global-descriptor-data.mjs extract <work-dir> [--shard=i/n]
 *   node scripts/global-descriptor.mjs train <work-dir> [--words=512|32]
 *   node scripts/global-descriptor.mjs truth <work-dir> [--workers=3]
 *   node scripts/global-descriptor.mjs recall <work-dir> [--corpus=test|photos]
 *
 * `train` runs Lloyd's k-means (`kMeans`, 25 iterations, seed 1) on the RootSIFT descriptors of
 * the lab's video and photo samples, except the 15 videos of the test set, and writes
 * `src/gpu/vocabulary/<id>.bin` (words x 128 f32, little endian) and its record `<id>.json`.
 * `--words=512` (the default) writes `rootsift-512-v1`, the vocabulary of the bag of words.
 * `--words=32` writes `rootsift-32-v1`, the vocabulary of VLAD, the default descriptor.
 *
 * `truth` finds the true pairs. For each eligible pair of frames it matches the SIFT descriptors
 * (nearest and second nearest by L2, ratio 0.8, mutual: `matchFloatDescriptorsGpu`'s rule, on
 * the host), drops a pair with fewer than `MIN_MATCHES` (30) matches, and verifies the others with
 * `verifyImagePair` (the CPU verifier of the mapper). A pair with at least `MIN_EDGE_INLIERS`
 * (25) inliers is a true pair (3.4). It writes `<work-dir>/truth-<corpus>.json`.
 *
 * `recall` ranks the frames with each candidate and prints, for each corpus, the recall at k = 10 of
 * the true pairs, as JSON. See `docs/multi-video.md`, Amendment 2, for the protocol.
 */
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads';
import { listFrames, frameName, readSift } from './global-descriptor-data.mjs';
import {
  BAG_OF_WORDS_VOCABULARY_ID,
  GLOBAL_VOCABULARY_ID,
  assignWordsReference,
  bagOfWordsDescriptor,
  cosineSimilarity,
  inverseDocumentFrequency,
  parseVocabulary,
  vladDescriptorReference,
  wordHistogram,
} from '../src/gpu/global-descriptor.ts';
import { kMeans, thumbnailDescriptor } from '../src/gpu/global-descriptor-candidates.ts';
import { verifyImagePair } from '../src/geometry/image-pair.ts';

const root = resolve(import.meta.dirname, '..');
const VOCABULARY_DIR = join(root, 'src/gpu/vocabulary');
export const MIN_MATCHES = 30;
export const MIN_EDGE_INLIERS = 25;
export const RATIO = 0.8;
const TRAIN_PER_FRAME = 1000;
const TRAIN_ITERATIONS = 25;
const TRAIN_SEED = 1;
const VLAD_WORDS = 32;
const BAG_OF_WORDS = 512;
const THUMBNAIL = 16;
/** Frames of one video this close in time (in kept frames) are the temporal neighbours of 3.3. */
const SAME_VIDEO_WINDOW = 10;

/** The corpora: the test set (15 videos) and the photo samples. */
function corpora() {
  const frames = listFrames();
  const test = frames.filter((f) => f.set !== undefined);
  const photos = frames.filter((f) => f.kind === 'photos');
  const withPosition = (list, photo) => {
    const seen = new Map();
    return list.map((f) => {
      const key = photo ? `${f.id}/${f.index}` : f.id;
      const position = seen.get(f.id) ?? 0;
      seen.set(f.id, position + 1);
      return { ...f, video: photo ? key : f.id, position, scene: photo ? f.id : 'chimney-rock' };
    });
  };
  return { test: withPosition(test, false), photos: withPosition(photos, true) };
}

/** Whether the pair can be a candidate of retrieval (3.3): not the same video within the window. */
const eligible = (a, b) =>
  a.video !== b.video || Math.abs(a.position - b.position) > SAME_VIDEO_WINDOW;

function featuresOf(work, f) {
  return readSift(join(work, 'features', `${frameName(f)}.sift`));
}

// Mutual ratio-test matches of two RootSIFT sets, by the dot product (unit vectors: d^2 = 2 - 2 dot).
function mutualMatches(a, b) {
  const na = a.count,
    nb = b.count;
  if (na < 2 || nb < 2) return [];
  const best1 = new Float32Array(na).fill(-2),
    best2 = new Float32Array(na).fill(-2),
    arg1 = new Int32Array(na).fill(-1),
    bBest1 = new Float32Array(nb).fill(-2),
    bBest2 = new Float32Array(nb).fill(-2),
    bArg1 = new Int32Array(nb).fill(-1);
  const da = a.descriptors,
    db = b.descriptors;
  for (let i = 0; i < na; i++) {
    const oi = i * 128;
    for (let j = 0; j < nb; j++) {
      const oj = j * 128;
      let dot = 0;
      for (let d = 0; d < 128; d++) dot += da[oi + d] * db[oj + d];
      if (dot > best1[i]) {
        best2[i] = best1[i];
        best1[i] = dot;
        arg1[i] = j;
      } else if (dot > best2[i]) best2[i] = dot;
      if (dot > bBest1[j]) {
        bBest2[j] = bBest1[j];
        bBest1[j] = dot;
        bArg1[j] = i;
      } else if (dot > bBest2[j]) bBest2[j] = dot;
    }
  }
  const r2 = RATIO * RATIO,
    ok = (d1, d2) => 2 - 2 * d1 < r2 * (2 - 2 * d2);
  const out = [];
  for (let i = 0; i < na; i++) {
    const j = arg1[i];
    if (j >= 0 && bArg1[j] === i && ok(best1[i], best2[i]) && ok(bBest1[j], bBest2[j]))
      out.push([i, j]);
  }
  return out;
}

function trueWorker() {
  const { work, items } = workerData;
  const cache = new Map();
  const load = (f) => {
    const key = frameName(f);
    if (!cache.has(key)) cache.set(key, featuresOf(work, f));
    return cache.get(key);
  };
  const results = [];
  for (const [i, j, fa, fb] of items) {
    const a = load(fa),
      b = load(fb),
      m = mutualMatches(a, b);
    if (m.length < MIN_MATCHES) continue;
    const pixels = new Float64Array(m.length * 4);
    m.forEach(([p, q], k) =>
      pixels.set(
        [a.points[2 * p], a.points[2 * p + 1], b.points[2 * q], b.points[2 * q + 1]],
        4 * k,
      ),
    );
    let inliers = 0;
    try {
      inliers = verifyImagePair(pixels, fa.camera, fb.camera).inlierRows.length;
    } catch {
      inliers = 0;
    }
    results.push([i, j, m.length, inliers]);
  }
  parentPort.postMessage(results);
}

async function truth(work, workers) {
  const sets = corpora();
  for (const [name, frames] of Object.entries(sets)) {
    const items = [];
    for (let i = 0; i < frames.length; i++)
      for (let j = i + 1; j < frames.length; j++) {
        const a = frames[i],
          b = frames[j];
        const considered = name === 'photos' ? a.scene === b.scene : eligible(a, b);
        if (considered) items.push([i, j, a, b]);
      }
    const started = Date.now();
    const parts = Array.from({ length: workers }, (_, w) =>
      items.filter((_, k) => k % workers === w),
    );
    const results = (
      await Promise.all(
        parts.map(
          (part) =>
            new Promise((res, rej) => {
              const w = new Worker(new URL(import.meta.url), { workerData: { work, items: part } });
              w.on('message', res);
              w.on('error', rej);
            }),
        ),
      )
    ).flat();
    results.sort((x, y) => x[0] - y[0] || x[1] - y[1]);
    writeFileSync(
      join(work, `truth-${name}.json`),
      JSON.stringify({
        corpus: name,
        frames: frames.map(frameName),
        considered: items.length,
        minMatches: MIN_MATCHES,
        minEdgeInliers: MIN_EDGE_INLIERS,
        pairs: results,
      }),
    );
    const trueCount = results.filter((r) => r[3] >= MIN_EDGE_INLIERS).length;
    console.log(
      `${name}: ${frames.length} frames, ${items.length} pairs matched, ${results.length} with >= ${MIN_MATCHES} matches, ${trueCount} true pairs, ${((Date.now() - started) / 1000).toFixed(0)} s`,
    );
  }
}

/** Up to `cap` descriptors of a frame, evenly spaced in the SIFT order. */
function sample(features, cap) {
  if (features.count <= cap) return features.descriptors;
  const out = new Float32Array(cap * 128),
    step = features.count / cap;
  for (let i = 0; i < cap; i++)
    out.set(
      features.descriptors.subarray(Math.floor(i * step) * 128, Math.floor(i * step) * 128 + 128),
      i * 128,
    );
  return out;
}

function trainingSet(work, filter) {
  const parts = [],
    names = [];
  for (const f of listFrames()) {
    if (!filter(f)) continue;
    parts.push(sample(featuresOf(work, f), TRAIN_PER_FRAME));
    names.push(frameName(f));
  }
  const total = parts.reduce((s, p) => s + p.length, 0),
    all = new Float32Array(total);
  let at = 0;
  for (const p of parts) {
    all.set(p, at);
    at += p.length;
  }
  return { points: all, frames: names };
}

function train(work, words) {
  const id = { [BAG_OF_WORDS]: BAG_OF_WORDS_VOCABULARY_ID, [VLAD_WORDS]: GLOBAL_VOCABULARY_ID }[
    words
  ];
  if (!id) throw new Error('--words must be 512 or 32');
  const { points, frames } = trainingSet(work, (f) => f.set === undefined);
  console.log(`training ${id} on ${points.length / 128} descriptors of ${frames.length} frames`);
  let last = 0;
  const started = Date.now();
  const centers = kMeans(points, words, {
    iterations: TRAIN_ITERATIONS,
    seed: TRAIN_SEED,
    onIteration: (it, e) => {
      last = e;
      console.log(`iteration ${it + 1}: mean squared distance ${e.toFixed(6)}`);
    },
  });
  const bytes = new Uint8Array(centers.buffer);
  mkdirSync(VOCABULARY_DIR, { recursive: true });
  writeFileSync(join(VOCABULARY_DIR, `${id}.bin`), bytes);
  writeFileSync(
    join(VOCABULARY_DIR, `${id}.json`),
    JSON.stringify(
      {
        id,
        words,
        dimensions: 128,
        format: `f32 little endian, ${words} x 128, row = word`,
        sha256: createHash('sha256').update(bytes).digest('hex'),
        method: `Lloyd k-means, ${TRAIN_ITERATIONS} iterations, seed ${TRAIN_SEED}, initial centers = ${words} distinct training points`,
        script: `node scripts/global-descriptor.mjs train <work-dir> --words=${words}`,
        features: 'RootSIFT of the CPU tier, extractSiftGpuResident, at most 4096 per frame',
        perFrameCap: TRAIN_PER_FRAME,
        descriptors: points.length / 128,
        frames: frames.length,
        sources: [...new Set(frames.map((n) => n.split('/')[0]))],
        meanSquaredDistance: last,
      },
      null,
      2,
    ) + '\n',
  );
  console.log(`done in ${((Date.now() - started) / 1000).toFixed(0)} s`);
}

function thumbnailOf(work, f) {
  const r = spawnSync(
    'ffmpeg',
    [
      '-v',
      'error',
      '-i',
      join(work, 'frames', `${frameName(f)}.jpg`),
      '-vf',
      `scale=${THUMBNAIL}:${THUMBNAIL}:flags=area`,
      '-f',
      'rawvideo',
      '-pix_fmt',
      'gray',
      '-',
    ],
    { maxBuffer: 1 << 20 },
  );
  if (r.status) throw new Error('ffmpeg failed');
  return thumbnailDescriptor(Uint8Array.from(r.stdout));
}

function topK(similarity, frames, k) {
  const n = frames.length,
    out = [];
  for (let i = 0; i < n; i++) {
    const rows = [];
    for (let j = 0; j < n; j++)
      if (j !== i && eligible(frames[i], frames[j])) rows.push([similarity[i * n + j], j]);
    rows.sort((x, y) => y[0] - x[0] || x[1] - y[1]);
    out.push(rows.slice(0, k).map((r) => r[1]));
  }
  return out;
}

function recallOf(tops, pairs, k) {
  const sets = tops.map((t) => new Set(t.slice(0, k)));
  let hitsEither = 0,
    hitsDirected = 0,
    directed = 0;
  for (const [i, j] of pairs) {
    if (sets[i].has(j) || sets[j].has(i)) hitsEither++;
    directed += 2;
    hitsDirected += (sets[i].has(j) ? 1 : 0) + (sets[j].has(i) ? 1 : 0);
  }
  return {
    either: pairs.length ? hitsEither / pairs.length : NaN,
    directed: directed ? hitsDirected / directed : NaN,
  };
}

function recall(work, only) {
  const vocabulary = parseVocabulary(
    readFileSync(join(VOCABULARY_DIR, `${BAG_OF_WORDS_VOCABULARY_ID}.bin`)),
    BAG_OF_WORDS,
  );
  const shippedVlad = parseVocabulary(
    readFileSync(join(VOCABULARY_DIR, `${GLOBAL_VOCABULARY_ID}.bin`)),
    VLAD_WORDS,
  );
  const all = corpora();
  const report = {};
  const vladTraining = trainingSet(work, (f) => f.set === undefined);
  console.log('training the VLAD vocabulary (32 words)');
  const vlad32 = kMeans(vladTraining.points, VLAD_WORDS, {
    iterations: TRAIN_ITERATIONS,
    seed: TRAIN_SEED,
  });
  // The shipped vocabulary is the one that the measurement of Amendment 2 trained in memory.
  if (Buffer.compare(Buffer.from(vlad32.buffer), Buffer.from(shippedVlad.buffer)) !== 0)
    throw new Error(`${GLOBAL_VOCABULARY_ID} differs from the VLAD vocabulary of this training`);
  console.log(`${GLOBAL_VOCABULARY_ID} equals the retrained VLAD vocabulary, byte for byte`);
  for (const [name, frames] of Object.entries(all)) {
    if (only && name !== only) continue;
    const truthFile = JSON.parse(readFileSync(join(work, `truth-${name}.json`), 'utf8'));
    if (truthFile.frames.join() !== frames.map(frameName).join())
      throw new Error(`${name}: stale truth file`);
    const truePairs = truthFile.pairs.filter((p) => p[3] >= MIN_EDGE_INLIERS);
    const features = frames.map((f) => featuresOf(work, f));
    console.log(`${name}: ${frames.length} frames, ${truePairs.length} true pairs`);
    const n = frames.length;
    const wordsOf = (v) => features.map((f) => assignWordsReference(f.descriptors, v));
    const bow = (words, v) => {
      const hist = words.map((w) => wordHistogram(w, v.length / 128));
      const idf = inverseDocumentFrequency(hist, v.length / 128);
      return hist.map((h) => bagOfWordsDescriptor(h, idf));
    };
    const matrix = (vectors) => {
      const s = new Float32Array(n * n);
      for (let i = 0; i < n; i++)
        for (let j = 0; j < n; j++) s[i * n + j] = cosineSimilarity(vectors[i], vectors[j]);
      return s;
    };
    const candidates = {};
    candidates['bag of words 512, tf-idf (rootsift-512-v1)'] = bow(wordsOf(vocabulary), vocabulary);
    const ownTraining = trainingSet(work, (f) =>
      frames.some((g) => g.id === f.id && g.index === f.index),
    );
    console.log(
      `  training a vocabulary on the corpus's own descriptors (${ownTraining.points.length / 128})`,
    );
    const own = kMeans(ownTraining.points, 512, { iterations: TRAIN_ITERATIONS, seed: TRAIN_SEED });
    candidates["bag of words 512, tf-idf (trained on the corpus's own descriptors)"] = bow(
      wordsOf(own),
      own,
    );
    candidates['VLAD 32 words, intra-normalized'] = features.map((f) =>
      vladDescriptorReference(f.descriptors, shippedVlad),
    );
    candidates['downsampled image 16 x 16 gray'] = frames.map((f) => thumbnailOf(work, f));
    if (name === 'photos') {
      const noPhotos = trainingSet(work, (f) => f.set === undefined && f.kind !== 'photos');
      console.log(
        `  training a vocabulary without the photo samples (${noPhotos.points.length / 128})`,
      );
      const held = kMeans(noPhotos.points, 512, { iterations: TRAIN_ITERATIONS, seed: TRAIN_SEED });
      candidates['bag of words 512, tf-idf (trained without the photo samples)'] = bow(
        wordsOf(held),
        held,
      );
    }
    report[name] = {
      frames: n,
      considered: truthFile.considered,
      truePairs: truePairs.length,
      trueVideoPairs:
        name === 'test'
          ? new Set(truePairs.map(([i, j]) => [frames[i].video, frames[j].video].sort().join('|')))
              .size
          : undefined,
      candidates: Object.fromEntries(
        Object.entries(candidates).map(([label, vectors]) => {
          const tops = topK(matrix(vectors), frames, 20);
          return [
            label,
            {
              dimension: vectors[0].length,
              'recall@10': recallOf(tops, truePairs, 10),
              'recall@20': recallOf(tops, truePairs, 20),
            },
          ];
        }),
      ),
    };
  }
  console.log(JSON.stringify(report, null, 2));
}

if (!isMainThread) trueWorker();
else if (import.meta.url === `file://${process.argv[1]}`) {
  const [command, work, ...rest] = process.argv.slice(2);
  if (!work)
    throw new Error('Usage: node scripts/global-descriptor.mjs train|truth|recall <work-dir>');
  const dir = resolve(work);
  if (command === 'train')
    train(dir, Number(rest.find((a) => a.startsWith('--words='))?.slice(8) ?? BAG_OF_WORDS));
  else if (command === 'truth')
    await truth(dir, Number(rest.find((a) => a.startsWith('--workers='))?.slice(10) ?? 3));
  else if (command === 'recall') recall(dir, rest.find((a) => a.startsWith('--corpus='))?.slice(9));
  else throw new Error(`Unknown command ${command}`);
}
