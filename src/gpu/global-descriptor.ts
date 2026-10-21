/**
 * The global descriptor of a keyframe (`docs/multi-video.md`, section 3.2): a bag of visual words
 * over RootSIFT, with the fixed vocabulary `rootsift-512-v1` and tf-idf weights. Retrieval ranks
 * the keyframes of a run by the cosine of two such descriptors (section 3.3).
 *
 * The recipe, for one keyframe:
 *
 * 1. Take its RootSIFT descriptors (`GpuResidentSift.descriptors`, or a host array of the same shape).
 * 2. Assign each to the nearest word of the vocabulary: `assignWordsGpu` (the kernel `globalAssign`
 *    in `global-descriptor.shade.ts`, called from `global-descriptor-gpu.ts`) or `assignWordsReference`.
 * 3. Count the words: `wordHistogram`.
 * 4. Over all keyframes of the run, take `inverseDocumentFrequency` of the histograms.
 * 5. `bagOfWordsDescriptor(histogram, idf)`: the square root of each count times its idf, L2
 *    normalized. The similarity of two keyframes is the dot product (`cosineSimilarity`).
 *
 * `globalDescriptorReference` runs steps 1 to 5 on the host, in plain TypeScript, for a single
 * keyframe. It is the oracle that the kernel is held to (section 7.4).
 *
 * The vocabulary is a file of 512 × 128 f32 values in little-endian order, trained offline by
 * `scripts/global-descriptor.mjs train`. `parseVocabulary` checks the bytes of that file. The idf
 * weights depend on the set of keyframes and not on their order.
 */
/** The name of the vocabulary that ships with the package. */
export const GLOBAL_VOCABULARY_ID = 'rootsift-512-v1';
/** The number of visual words, and so the dimension of the global descriptor. */
export const GLOBAL_WORDS = 512;
/** The dimension of a RootSIFT descriptor. */
export const GLOBAL_DIMENSIONS = 128;

/** The bytes of a vocabulary file as 512 × 128 f32 values. Throws on a wrong size or a nonfinite value. */
export function parseVocabulary(
  bytes: ArrayBuffer | Uint8Array,
  words = GLOBAL_WORDS,
  dimensions = GLOBAL_DIMENSIONS,
): Float32Array {
  const view = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  if (view.byteLength !== words * dimensions * 4)
    throw new RangeError(`A vocabulary of ${words} words takes ${words * dimensions * 4} bytes`);
  const values = new Float32Array(words * dimensions);
  new Uint8Array(values.buffer).set(view);
  const little = new Uint8Array(Uint32Array.of(1).buffer)[0] === 1;
  if (!little) {
    const dv = new DataView(view.buffer, view.byteOffset, view.byteLength);
    for (let i = 0; i < values.length; i++) values[i] = dv.getFloat32(4 * i, true);
  }
  for (let i = 0; i < values.length; i++)
    if (!Number.isFinite(values[i])) throw new RangeError('The vocabulary holds a nonfinite value');
  return values;
}

export function checkShape(
  descriptors: ArrayLike<number>,
  vocabulary: ArrayLike<number>,
  dimensions: number,
) {
  if (
    !Number.isSafeInteger(dimensions) ||
    dimensions < 1 ||
    descriptors.length % dimensions !== 0 ||
    vocabulary.length % dimensions !== 0 ||
    vocabulary.length === 0
  )
    throw new RangeError('Descriptors and vocabulary need a common dimension');
}

/**
 * The word of each descriptor on the host: the nearest word by squared L2 distance, summed in
 * f32 over the dimensions in order; the first word of the least distance wins. This is the
 * oracle of `globalAssign`.
 */
export function assignWordsReference(
  descriptors: Float32Array,
  vocabulary: Float32Array,
  dimensions = GLOBAL_DIMENSIONS,
): Uint32Array {
  checkShape(descriptors, vocabulary, dimensions);
  const count = descriptors.length / dimensions,
    words = vocabulary.length / dimensions,
    out = new Uint32Array(count);
  for (let i = 0; i < count; i++) {
    let best = 3.4028234663852886e38,
      bestWord = 0;
    for (let w = 0; w < words; w++) {
      let distance = 0;
      for (let d = 0; d < dimensions; d++) {
        const delta = Math.fround(
          descriptors[i * dimensions + d]! - vocabulary[w * dimensions + d]!,
        );
        distance = Math.fround(distance + Math.fround(delta * delta));
      }
      if (distance < best) {
        best = distance;
        bestWord = w;
      }
    }
    out[i] = bestWord;
  }
  return out;
}

/** How many times each word occurs. A word outside `[0, words)` throws. */
export function wordHistogram(assigned: ArrayLike<number>, words = GLOBAL_WORDS): Uint32Array {
  const histogram = new Uint32Array(words);
  for (let i = 0; i < assigned.length; i++) {
    const w = assigned[i]!;
    if (!(w >= 0 && w < words && Number.isInteger(w))) throw new RangeError('Word out of range');
    histogram[w]!++;
  }
  return histogram;
}

/**
 * `idf(w) = log((N + 1) / (df(w) + 1)) + 1`, where N is the number of keyframes and `df(w)` is the
 * number of keyframes whose histogram counts the word `w` (section 3.2). The sum does not depend on
 * the order of the histograms.
 */
export function inverseDocumentFrequency(
  histograms: readonly ArrayLike<number>[],
  words = GLOBAL_WORDS,
): Float32Array {
  const frequency = new Uint32Array(words);
  for (const h of histograms) {
    if (h.length !== words) throw new RangeError('A histogram has the wrong number of words');
    for (let w = 0; w < words; w++) if (h[w]! > 0) frequency[w]!++;
  }
  const n = histograms.length,
    idf = new Float32Array(words);
  for (let w = 0; w < words; w++) idf[w] = Math.log((n + 1) / (frequency[w]! + 1)) + 1;
  return idf;
}

/**
 * The global descriptor of one histogram: the square root of each count, times the idf weight
 * (1 when `idf` is not given), L2 normalized. An empty histogram gives the zero vector, whose
 * cosine with any descriptor is 0.
 */
export function bagOfWordsDescriptor(
  histogram: ArrayLike<number>,
  idf?: ArrayLike<number>,
): Float32Array {
  const words = histogram.length;
  if (idf && idf.length !== words) throw new RangeError('The idf has the wrong number of words');
  const values = new Float64Array(words);
  let sum = 0;
  for (let w = 0; w < words; w++) {
    values[w] = Math.sqrt(histogram[w]!) * (idf ? idf[w]! : 1);
    sum += values[w]! * values[w]!;
  }
  const out = new Float32Array(words),
    norm = Math.sqrt(sum);
  if (norm > 0) for (let w = 0; w < words; w++) out[w] = values[w]! / norm;
  return out;
}

/** The cosine of two global descriptors that are L2 normalized: their dot product. */
export function cosineSimilarity(a: ArrayLike<number>, b: ArrayLike<number>): number {
  if (a.length !== b.length) throw new RangeError('Descriptors differ in length');
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i]! * b[i]!;
  return sum;
}

/**
 * The global descriptor of one keyframe, from its RootSIFT descriptors, on the host. `idf` is
 * the weights of the run (`inverseDocumentFrequency`); without it every weight is 1.
 */
export function globalDescriptorReference(
  descriptors: Float32Array,
  vocabulary: Float32Array,
  idf?: ArrayLike<number>,
  dimensions = GLOBAL_DIMENSIONS,
): Float32Array {
  const words = vocabulary.length / dimensions;
  return bagOfWordsDescriptor(
    wordHistogram(assignWordsReference(descriptors, vocabulary, dimensions), words),
    idf,
  );
}
