// A plain JavaScript oracle for the Hamming matcher and a generator of descriptor sets with ties,
// duplicates and a tail that is not a multiple of the tile size.

const popcount = (x) => {
  x -= (x >>> 1) & 0x55555555;
  x = (x & 0x33333333) + ((x >>> 2) & 0x33333333);
  return (((x + (x >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24;
};

/** Random 256-bit descriptors; every 7th row copies row i - 3 (ties), every 11th differs by one bit. */
export function descriptorSet(count, seed) {
  let state = seed >>> 0;
  const next = () => ((state = (Math.imul(state, 1664525) + 1013904223) >>> 0), state);
  const descriptors = new Uint32Array(count * 8);
  for (let i = 0; i < count * 8; i++) descriptors[i] = next();
  for (let i = 3; i < count; i++) {
    if (i % 7 === 0) descriptors.copyWithin(i * 8, (i - 3) * 8, (i - 3) * 8 + 8);
    else if (i % 11 === 0) {
      descriptors.copyWithin(i * 8, (i - 5) * 8, (i - 5) * 8 + 8);
      descriptors[i * 8 + (i % 8)] ^= 1 << (i % 32);
    }
  }
  return {
    descriptors,
    scores: Float32Array.from({ length: count }, () => 1),
    keypoints: new Float32Array(count * 2),
  };
}

/**
 * Two sets where most rows of `second` are noisy copies (up to 24 flipped bits) of rows of
 * `first`, so that many pairs pass the ratio test, with the ties of `descriptorSet`.
 */
export function descriptorPair(nf, ns, seed) {
  const first = descriptorSet(nf, seed),
    second = descriptorSet(ns, seed + 1);
  let state = (seed * 2654435761) >>> 0;
  const next = () => ((state = (Math.imul(state, 1664525) + 1013904223) >>> 0), state);
  for (let j = 0; j < ns; j++) {
    if (next() % 10 >= 7) continue;
    const i = next() % nf;
    second.descriptors.set(first.descriptors.subarray(i * 8, i * 8 + 8), j * 8);
    for (let k = next() % 25; k > 0; k--)
      second.descriptors[j * 8 + (next() % 8)] ^= 1 << (next() % 32);
  }
  return { first, second };
}

/** Best, second best distance, ratio test, Hamming bound, then the mutual check. */
export function hammingOracle(first, second, ratio, maxDistance) {
  const nearest = (queries, targets, nq, nt) => {
    const out = new Array(nq);
    for (let i = 0; i < nq; i++) {
      let best = -1,
        bestDistance = Infinity,
        secondDistance = Infinity;
      for (let j = 0; j < nt; j++) {
        let d = 0;
        for (let w = 0; w < 8; w++) d += popcount((queries[i * 8 + w] ^ targets[j * 8 + w]) >>> 0);
        if (d < bestDistance) {
          secondDistance = bestDistance;
          bestDistance = d;
          best = j;
        } else if (d < secondDistance) secondDistance = d;
      }
      out[i] = {
        best,
        bestDistance,
        accepted:
          best >= 0 &&
          secondDistance !== Infinity &&
          bestDistance <= maxDistance &&
          Math.fround(bestDistance) < Math.fround(Math.fround(ratio) * Math.fround(secondDistance)),
      };
    }
    return out;
  };
  const nf = first.scores.length,
    ns = second.scores.length,
    forward = nearest(first.descriptors, second.descriptors, nf, ns),
    reverse = nearest(second.descriptors, first.descriptors, ns, nf),
    matches = [],
    distances = [];
  for (let i = 0; i < nf; i++) {
    const f = forward[i];
    if (f.accepted && reverse[f.best].accepted && reverse[f.best].best === i) {
      matches.push(i, f.best);
      distances.push(f.bestDistance);
    }
  }
  return { matches: Uint32Array.from(matches), distances: Float64Array.from(distances) };
}
