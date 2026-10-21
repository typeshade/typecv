// Fixtures for the global descriptor tests (docs/multi-video.md, 7.4). Shared by the CPU-tier and
// the WebGPU test files. Descriptors have the shape of RootSIFT: non-negative, unit L2 norm.

/** A small seeded generator, so that every run and both tiers see the same fixture. */
export function random(seed) {
  let s = seed >>> 0 || 0x9e3779b9;
  return () => {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    return (s >>> 0) / 2 ** 32;
  };
}

/** `count` RootSIFT-like descriptors: random non-negative, L1 normalized, square-rooted. */
export function rootSiftFixture(count, seed = 1) {
  const next = random(seed),
    out = new Float32Array(count * 128);
  for (let i = 0; i < count; i++) {
    let sum = 0;
    for (let d = 0; d < 128; d++) {
      // Sparse, as a gradient histogram is: most bins are small.
      const x = next() ** 4;
      out[i * 128 + d] = x;
      sum += x;
    }
    for (let d = 0; d < 128; d++) out[i * 128 + d] = Math.fround(Math.sqrt(out[i * 128 + d] / sum));
  }
  return out;
}

/**
 * Descriptors that lie near the words of `vocabulary`: word `i % words`, plus a small jitter, so
 * that the nearest word is well defined for most of them, and a few exactly on a word.
 */
export function nearWords(vocabulary, count, seed = 2, jitter = 0.01) {
  const next = random(seed),
    words = vocabulary.length / 128,
    out = new Float32Array(count * 128);
  for (let i = 0; i < count; i++) {
    const w = Math.floor(next() * words);
    for (let d = 0; d < 128; d++)
      out[i * 128 + d] = Math.fround(
        Math.max(0, vocabulary[w * 128 + d] + (i % 5 === 0 ? 0 : jitter * (next() - 0.5))),
      );
  }
  return out;
}
