import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import {
  GLOBAL_DIMENSIONS,
  GLOBAL_VOCABULARY_ID,
  GLOBAL_WORDS,
  assignWordsReference,
  bagOfWordsDescriptor,
  cosineSimilarity,
  globalDescriptorReference,
  inverseDocumentFrequency,
  parseVocabulary,
  wordHistogram,
} from '../src/gpu/global-descriptor.ts';
import {
  kMeans,
  thumbnailDescriptor,
  vladDescriptor,
} from '../src/gpu/global-descriptor-candidates.ts';
import { seededRandom } from '../src/geometry/numerics.ts';

// Verifies: docs/multi-video.md 3.2 (the descriptor) and 7.4 (the reference against a naive definition).

/** A vocabulary of `words` unit vectors on the first axes of a space of `dim` dimensions. */
function axes(words: number, dim: number): Float32Array {
  const v = new Float32Array(words * dim);
  for (let w = 0; w < words; w++) v[w * dim + w] = 1;
  return v;
}

test('the histogram of a hand-made descriptor set equals the hand count', () => {
  const dim = 4,
    vocabulary = axes(4, dim);
  // Six descriptors near the axes 0, 0, 0, 2, 2, 3. Axis 1 has no descriptor.
  const near = (axis: number, jitter: number) => {
    const d = new Float32Array(dim);
    d[axis] = 1;
    d[(axis + 1) % dim] = jitter;
    return d;
  };
  const descriptors = Float32Array.from([
    ...near(0, 0.1),
    ...near(0, 0.2),
    ...near(0, 0.05),
    ...near(2, 0.1),
    ...near(2, 0.3),
    ...near(3, 0.1),
  ]);
  const words = assignWordsReference(descriptors, vocabulary, dim);
  assert.deepEqual([...words], [0, 0, 0, 2, 2, 3]);
  assert.deepEqual([...wordHistogram(words, 4)], [3, 0, 2, 1]);
});

test('the nearest word is the least squared distance and a tie goes to the lower index', () => {
  const dim = 2,
    vocabulary = Float32Array.from([0, 0, 2, 0, 0, 2, 2, 0]); // words 1 and 3 coincide
  const descriptors = Float32Array.from([1.9, 0.1, 1, 0, 0.1, 1.9, -5, -5]);
  assert.deepEqual([...assignWordsReference(descriptors, vocabulary, dim)], [1, 0, 2, 0]);
  // (1, 0) is at distance 1 of words 0 and 1: the lower index wins.
  assert.throws(() => assignWordsReference(new Float32Array(3), vocabulary, dim), RangeError);
});

test('the descriptor is the damped, idf-weighted, L2-normalized histogram', () => {
  const histogram = Uint32Array.from([4, 0, 1, 9]);
  const idf = Float32Array.from([1, 2, 3, 0.5]);
  const expected = [2 * 1, 0, 1 * 3, 3 * 0.5];
  const norm = Math.hypot(...expected);
  const got = bagOfWordsDescriptor(histogram, idf);
  expected.forEach((x, i) => assert.ok(Math.abs(got[i]! - x / norm) < 1e-6));
  assert.ok(Math.abs(Math.hypot(...got) - 1) < 1e-6);
  // Without idf every weight is 1.
  const plain = bagOfWordsDescriptor(histogram);
  const plainNorm = Math.hypot(2, 0, 1, 3);
  assert.ok(Math.abs(plain[0]! - 2 / plainNorm) < 1e-6);
  // An empty histogram is the zero vector, and its cosine with anything is 0.
  const zero = bagOfWordsDescriptor(new Uint32Array(4));
  assert.deepEqual([...zero], [0, 0, 0, 0]);
  assert.equal(cosineSimilarity(zero, got), 0);
  // The cosine of a descriptor with itself is 1.
  assert.ok(Math.abs(cosineSimilarity(got, got) - 1) < 1e-6);
  assert.throws(() => bagOfWordsDescriptor(histogram, [1, 2]), RangeError);
});

test('idf is log((N + 1) / (df + 1)) + 1 and does not depend on the order of the keyframes', () => {
  const histograms = [
    Uint32Array.from([3, 0, 1]),
    Uint32Array.from([1, 1, 0]),
    Uint32Array.from([5, 0, 0]),
    Uint32Array.from([2, 0, 0]),
  ];
  const idf = inverseDocumentFrequency(histograms, 3);
  // df = [4, 1, 1]; N = 4.
  const want = [Math.log(5 / 5) + 1, Math.log(5 / 2) + 1, Math.log(5 / 2) + 1];
  want.forEach((x, w) => assert.ok(Math.abs(idf[w]! - x) < 1e-6, `word ${w}`));
  const reversed = inverseDocumentFrequency([...histograms].reverse(), 3);
  assert.deepEqual([...reversed], [...idf]);
  assert.deepEqual([...inverseDocumentFrequency([], 2)], [1, 1]);
  assert.throws(() => inverseDocumentFrequency([new Uint32Array(2)], 3), RangeError);
  assert.throws(() => wordHistogram([5], 3), RangeError);
});

test('globalDescriptorReference chains the steps and agrees with the parts', () => {
  const dim = GLOBAL_DIMENSIONS,
    random = seededRandom(7),
    vocabulary = new Float32Array(8 * dim).map(() => random()),
    descriptors = new Float32Array(30 * dim).map(() => random());
  const idf = inverseDocumentFrequency(
    [wordHistogram(assignWordsReference(descriptors, vocabulary), 8)],
    8,
  );
  const parts = bagOfWordsDescriptor(
    wordHistogram(assignWordsReference(descriptors, vocabulary), 8),
    idf,
  );
  assert.deepEqual([...globalDescriptorReference(descriptors, vocabulary, idf)], [...parts]);
  assert.equal(
    globalDescriptorReference(new Float32Array(0), vocabulary).every((x) => x === 0),
    true,
  );
});

test('similar keyframes are more alike than unlike ones', () => {
  const dim = GLOBAL_DIMENSIONS,
    random = seededRandom(3),
    words = 16,
    vocabulary = new Float32Array(words * dim).map(() => random());
  // A keyframe draws descriptors around the words of its scene.
  const scene = (ws: number[], n: number) => {
    const out = new Float32Array(n * dim);
    for (let i = 0; i < n; i++)
      for (let d = 0; d < dim; d++)
        out[i * dim + d] = vocabulary[ws[i % ws.length]! * dim + d]! + 0.02 * (random() - 0.5);
    return out;
  };
  const a1 = scene([0, 1, 2, 3], 40),
    a2 = scene([0, 1, 2, 3, 4], 50),
    b = scene([8, 9, 10, 11], 40);
  const hist = [a1, a2, b].map((d) => wordHistogram(assignWordsReference(d, vocabulary), words));
  const idf = inverseDocumentFrequency(hist, words);
  const [da1, da2, db] = hist.map((h) => bagOfWordsDescriptor(h, idf));
  assert.ok(cosineSimilarity(da1!, da2!) > 0.8);
  assert.ok(cosineSimilarity(da1!, db!) < 0.01);
});

test('the shipped vocabulary has the recorded size, checksum and unit-scale words', () => {
  const bytes = readFileSync(new URL('../src/gpu/vocabulary/rootsift-512-v1.bin', import.meta.url));
  const record = JSON.parse(
    readFileSync(new URL('../src/gpu/vocabulary/rootsift-512-v1.json', import.meta.url), 'utf8'),
  );
  assert.equal(record.id, GLOBAL_VOCABULARY_ID);
  assert.equal(bytes.byteLength, GLOBAL_WORDS * GLOBAL_DIMENSIONS * 4);
  assert.equal(createHash('sha256').update(bytes).digest('hex'), record.sha256);
  const vocabulary = parseVocabulary(bytes);
  assert.equal(vocabulary.length, GLOBAL_WORDS * GLOBAL_DIMENSIONS);
  for (let w = 0; w < GLOBAL_WORDS; w++) {
    let norm = 0;
    for (let d = 0; d < GLOBAL_DIMENSIONS; d++) {
      const x = vocabulary[w * GLOBAL_DIMENSIONS + d]!;
      assert.ok(x >= 0 && x <= 1, 'RootSIFT values lie in [0, 1]');
      norm += x * x;
    }
    // The mean of unit vectors is shorter than a unit vector and not close to zero.
    assert.ok(
      Math.sqrt(norm) > 0.3 && Math.sqrt(norm) <= 1 + 1e-6,
      `word ${w} has norm ${Math.sqrt(norm)}`,
    );
  }
});

test('parseVocabulary rejects a wrong size and a nonfinite value, and reads an offset view', () => {
  assert.throws(() => parseVocabulary(new Uint8Array(10)), RangeError);
  const values = Float32Array.from([1, 2, 3, 4]);
  assert.deepEqual([...parseVocabulary(new Uint8Array(values.buffer), 2, 2)], [1, 2, 3, 4]);
  const padded = new Uint8Array(20);
  padded.set(new Uint8Array(values.buffer), 3);
  assert.deepEqual([...parseVocabulary(padded.subarray(3, 19), 2, 2)], [1, 2, 3, 4]);
  assert.throws(
    () => parseVocabulary(new Uint8Array(Float32Array.of(NaN, 0, 0, 0).buffer), 2, 2),
    RangeError,
  );
});

test('k-means is deterministic and finds separated clusters', () => {
  const dim = 2,
    random = seededRandom(11),
    points = new Float32Array(600 * dim);
  const centers = [
    [0, 0],
    [10, 0],
    [0, 10],
  ];
  for (let i = 0; i < 600; i++) {
    const c = centers[i % 3]!;
    points[2 * i] = c[0]! + random() - 0.5;
    points[2 * i + 1] = c[1]! + random() - 0.5;
  }
  let previous = Infinity;
  // Seed 2 starts one center in each cluster; a seed that starts two in one cluster can stay there.
  const first = kMeans(points, 3, {
    iterations: 8,
    seed: 2,
    dimensions: dim,
    onIteration: (_, error) => {
      assert.ok(error <= previous + 1e-9, 'a Lloyd step does not raise the error');
      previous = error;
    },
  });
  const second = kMeans(points, 3, { iterations: 8, seed: 2, dimensions: dim });
  assert.deepEqual([...first], [...second]);
  const found = [0, 1, 2]
    .map((c) => [first[2 * c]!, first[2 * c + 1]!])
    .sort((a, b) => a[0]! - b[0]! || a[1]! - b[1]!);
  const want = [...centers].sort((a, b) => a[0]! - b[0]! || a[1]! - b[1]!);
  found.forEach((f, i) => {
    assert.ok(Math.abs(f[0]! - want[i]![0]!) < 0.3 && Math.abs(f[1]! - want[i]![1]!) < 0.3);
  });
  assert.throws(() => kMeans(points, 700, { iterations: 1, seed: 1, dimensions: dim }), RangeError);
});

test('the baseline candidates: VLAD is intra-normalized and the thumbnail removes brightness', () => {
  const dim = 4,
    vocabulary = axes(2, dim),
    descriptors = Float32Array.from([1, 0.5, 0, 0, 0.9, 0.5, 0, 0, 0, 1, 0.3, 0]);
  const v = vladDescriptor(descriptors, vocabulary, dim);
  assert.equal(v.length, 2 * dim);
  assert.ok(Math.abs(Math.hypot(...v) - 1) < 1e-6);
  // Word 0 gets the residuals (0, 0.5, 0, 0) and (-0.1, 0.5, 0, 0); word 1 gets (0, 0, 0.3, 0).
  // Each block is scaled to a unit vector, so the two blocks have the same norm.
  const block = (w: number) => Math.hypot(...v.subarray(w * dim, w * dim + dim));
  assert.ok(Math.abs(block(0) - block(1)) < 1e-6);
  const t1 = thumbnailDescriptor([10, 20, 30, 50]),
    t2 = thumbnailDescriptor([110, 120, 130, 150]);
  assert.ok(cosineSimilarity(t1, t2) > 1 - 1e-6);
  assert.deepEqual([...thumbnailDescriptor([5, 5, 5])], [0, 0, 0]);
});
