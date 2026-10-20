'use typeshade';

class FloatMatchControl {
  queries: u32;
  candidates: u32;
  dimensions: u32;
  ratioSquared: f32;
  maxDistanceSquared: f32;
  mutual: u32;
}
declare const floatMatchControl: uniform<FloatMatchControl>;
/**
 * Guided matching: the 3 × 3 matrix M, row-major in m0..m8, maps a query point to the line in
 * the candidate image (F for queries in the first image, Fᵀ for queries in the second), and a
 * candidate counts only within `maxLinePx` of that line.
 */
class EpipolarGuide {
  m0: f32;
  m1: f32;
  m2: f32;
  m3: f32;
  m4: f32;
  m5: f32;
  m6: f32;
  m7: f32;
  m8: f32;
  maxLinePx: f32;
}
declare const epipolarGuide: uniform<EpipolarGuide>;
declare const queryPoints: storage<array<f32>>;
declare const candidatePoints: storage<array<f32>>;
declare const floatQueries: storage<array<f32>>;
declare const floatCandidates: storage<array<f32>>;
declare const nearestIndices: storage<array<u32>, 'read_write'>;
declare const nearestDistances: storage<array<f32>, 'read_write'>;
declare const forwardIndices: storage<array<u32>>;
declare const forwardDistances: storage<array<f32>>;
declare const backwardIndices: storage<array<u32>>;
declare const backwardDistances: storage<array<f32>>;
declare const floatMatches: storage<array<u32>, 'read_write'>;
declare const floatDistances: storage<array<f32>, 'read_write'>;
declare const floatMatchStatus: storage<array<u32>, 'read_write'>;
// Compaction flags (bit 0 accepted, bit 1 nonfinite) and exclusive block offsets from the shared scan.
declare const matchFlags: storage<array<u32>, 'read_write'>;
declare const matchOffsets: storage<array<u32>>;
let queryTile: workgroup<array<f32, 128>>;
let candidateTile: workgroup<array<f32, 128>>;
let bestTile: workgroup<array<f32, 128>>;
let indexTile: workgroup<array<u32, 128>>;
let errorTile: workgroup<array<u32, 64>>;
let querySlab: workgroup<array<f32, 1024>>;
let candidateSlab: workgroup<array<f32, 1024>>;

/** Explicit CPU differential tier; no barriers or implicit WebGPU fallback. */
@compute([64])
export function nearestFloatScalar(@builtin('global_invocation_id') gid: vec3u): void {
  const query: u32 = gid.x;
  if (query >= floatMatchControl.queries) return;
  let best: f32 = bitcast<f32>(u32(0x7f7fffff));
  let second: f32 = bitcast<f32>(u32(0x7f7fffff));
  let bestIndex: u32 = 0xffffffff;
  let secondIndex: u32 = 0xffffffff;
  let invalid: u32 = 0;
  for (let candidate: u32 = 0; candidate < floatMatchControl.candidates; candidate++) {
    let distance: f32 = 0;
    for (let d: u32 = 0; d < floatMatchControl.dimensions; d++) {
      const a: f32 = floatQueries[query * floatMatchControl.dimensions + d];
      const b: f32 = floatCandidates[candidate * floatMatchControl.dimensions + d];
      if ((bitcast<u32>(a) & 0x7f800000) === 0x7f800000 ||
          (bitcast<u32>(b) & 0x7f800000) === 0x7f800000) invalid = 1;
      const delta: f32 = a - b;
      distance += delta * delta;
    }
    if ((bitcast<u32>(distance) & 0x7f800000) === 0x7f800000) invalid = 1;
    if (distance < best || (distance === best && candidate < bestIndex)) {
      second = best; secondIndex = bestIndex;
      best = distance; bestIndex = candidate;
    } else if (distance < second || (distance === second && candidate < secondIndex)) {
      second = distance; secondIndex = candidate;
    }
  }
  nearestIndices[query * 3] = bestIndex;
  nearestIndices[query * 3 + 1] = secondIndex;
  nearestIndices[query * 3 + 2] = invalid;
  nearestDistances[query * 2] = best;
  nearestDistances[query * 2 + 1] = second;
}

/**
 * Guided nearest neighbours: as `nearestFloatScalar`, over the candidates within
 * `epipolarGuide.maxLinePx` of the query's epipolar line only, so the ratio test compares the
 * candidates the two-view geometry allows (COLMAP's guided matching). One invocation per query
 * without workgroup memory or barriers: the CPU tier runs it, and it is the oracle that
 * `nearestFloatGuidedTiles`, the WebGPU kernel, is held to.
 */
@compute([64])
export function nearestFloatGuided(@builtin('global_invocation_id') gid: vec3u): void {
  const query: u32 = gid.x;
  if (query >= floatMatchControl.queries) return;
  const x: f32 = queryPoints[query * 2];
  const y: f32 = queryPoints[query * 2 + 1];
  const la: f32 = epipolarGuide.m0 * x + epipolarGuide.m1 * y + epipolarGuide.m2;
  const lb: f32 = epipolarGuide.m3 * x + epipolarGuide.m4 * y + epipolarGuide.m5;
  const lc: f32 = epipolarGuide.m6 * x + epipolarGuide.m7 * y + epipolarGuide.m8;
  const norm: f32 = sqrt(la * la + lb * lb);
  let best: f32 = bitcast<f32>(u32(0x7f7fffff));
  let second: f32 = bitcast<f32>(u32(0x7f7fffff));
  let bestIndex: u32 = 0xffffffff;
  let secondIndex: u32 = 0xffffffff;
  let invalid: u32 = 0;
  // A degenerate line (the query at an epipole) admits no candidate: the query stays unmatched.
  let lines: u32 = 0;
  if (norm > 1e-12) lines = floatMatchControl.candidates;
  for (let candidate: u32 = 0; candidate < lines; candidate++) {
    const u: f32 = candidatePoints[candidate * 2];
    const v: f32 = candidatePoints[candidate * 2 + 1];
    if (abs(la * u + lb * v + lc) > epipolarGuide.maxLinePx * norm) continue;
    let distance: f32 = 0;
    for (let d: u32 = 0; d < floatMatchControl.dimensions; d++) {
      const a: f32 = floatQueries[query * floatMatchControl.dimensions + d];
      const b: f32 = floatCandidates[candidate * floatMatchControl.dimensions + d];
      if ((bitcast<u32>(a) & 0x7f800000) === 0x7f800000 ||
          (bitcast<u32>(b) & 0x7f800000) === 0x7f800000) invalid = 1;
      const delta: f32 = a - b;
      distance += delta * delta;
    }
    if ((bitcast<u32>(distance) & 0x7f800000) === 0x7f800000) invalid = 1;
    if (distance < best || (distance === best && candidate < bestIndex)) {
      second = best; secondIndex = bestIndex;
      best = distance; bestIndex = candidate;
    } else if (distance < second || (distance === second && candidate < secondIndex)) {
      second = distance; secondIndex = candidate;
    }
  }
  // A sole candidate on the line passes the ratio test, as in COLMAP's guided matching: the
  // missing second neighbour counts as infinitely far (index `candidates`, distance f32 max).
  if (bestIndex !== 0xffffffff && secondIndex === 0xffffffff)
    secondIndex = floatMatchControl.candidates;
  nearestIndices[query * 3] = bestIndex;
  nearestIndices[query * 3 + 1] = secondIndex;
  nearestIndices[query * 3 + 2] = invalid;
  nearestDistances[query * 2] = best;
  nearestDistances[query * 2 + 1] = second;
}

/** Exhaustive squared L2: 8 queries x 8 candidates, shared 16-channel slabs. */
@compute([64])
export function nearestFloatTiles(
  @builtin('workgroup_id') wid: vec3u,
  @builtin('local_invocation_id') lid: vec3u,
): void {
  const lane: u32 = lid.x;
  const query: u32 = wid.x * 8 + lane / 8;
  const column: u32 = lane % 8;
  let best: f32 = bitcast<f32>(u32(0x7f7fffff));
  let second: f32 = bitcast<f32>(u32(0x7f7fffff));
  let bestIndex: u32 = 0xffffffff;
  let secondIndex: u32 = 0xffffffff;
  let invalid: u32 = 0;
  for (let base: u32 = 0; base < floatMatchControl.candidates; base += 8) {
    let distance: f32 = 0;
    for (let channel: u32 = 0; channel < floatMatchControl.dimensions; channel += 16) {
      for (let half: u32 = 0; half < 2; half++) {
        const slot: u32 = lane + half * 64;
        const q: u32 = wid.x * 8 + slot / 16;
        const c: u32 = base + slot / 16;
        const d: u32 = channel + slot % 16;
        let a: f32 = 0;
        let b: f32 = 0;
        if (q < floatMatchControl.queries && d < floatMatchControl.dimensions)
          a = floatQueries[q * floatMatchControl.dimensions + d];
        if (c < floatMatchControl.candidates && d < floatMatchControl.dimensions)
          b = floatCandidates[c * floatMatchControl.dimensions + d];
        queryTile[slot] = a;
        candidateTile[slot] = b;
      }
      workgroupBarrier();
      for (let d: u32 = 0; d < 16; d++) {
        const a: f32 = queryTile[(lane / 8) * 16 + d];
        const b: f32 = candidateTile[column * 16 + d];
        if (query < floatMatchControl.queries && base + column < floatMatchControl.candidates &&
            ((bitcast<u32>(a) & 0x7f800000) === 0x7f800000 ||
             (bitcast<u32>(b) & 0x7f800000) === 0x7f800000)) invalid = 1;
        const delta: f32 = a - b;
        distance += delta * delta;
      }
      workgroupBarrier();
    }
    const candidate: u32 = base + column;
    if (candidate < floatMatchControl.candidates) {
      if (query < floatMatchControl.queries &&
          (bitcast<u32>(distance) & 0x7f800000) === 0x7f800000) invalid = 1;
      if (distance < best || (distance === best && candidate < bestIndex)) {
        second = best; secondIndex = bestIndex;
        best = distance; bestIndex = candidate;
      } else if (distance < second || (distance === second && candidate < secondIndex)) {
        second = distance; secondIndex = candidate;
      }
    }
  }
  bestTile[lane * 2] = best;
  bestTile[lane * 2 + 1] = second;
  indexTile[lane * 2] = bestIndex;
  indexTile[lane * 2 + 1] = secondIndex;
  errorTile[lane] = invalid;
  workgroupBarrier();
  if (column === 0 && query < floatMatchControl.queries) {
    best = bitcast<f32>(u32(0x7f7fffff)); second = bitcast<f32>(u32(0x7f7fffff));
    bestIndex = 0xffffffff; secondIndex = 0xffffffff; invalid = 0;
    for (let c: u32 = 0; c < 8; c++) {
      invalid |= errorTile[lane + c];
      for (let rank: u32 = 0; rank < 2; rank++) {
        const slot: u32 = (lane + c) * 2 + rank;
        const distance: f32 = bestTile[slot];
        const candidate: u32 = indexTile[slot];
        if (candidate === 0xffffffff) continue;
        if (distance < best || (distance === best && candidate < bestIndex)) {
          second = best; secondIndex = bestIndex;
          best = distance; bestIndex = candidate;
        } else if (distance < second || (distance === second && candidate < secondIndex)) {
          second = distance; secondIndex = candidate;
        }
      }
    }
    nearestIndices[query * 3] = bestIndex;
    nearestIndices[query * 3 + 1] = secondIndex;
    nearestIndices[query * 3 + 2] = invalid;
    nearestDistances[query * 2] = best;
    nearestDistances[query * 2 + 1] = second;
  }
}

/**
 * Guided nearest neighbours on the tiles of `nearestFloatTiles`: 8 queries x 8 candidates per
 * workgroup, with the epipolar gate of `nearestFloatGuided` applied to each (query, candidate)
 * cell. The shared slabs hold 128 channels, so a SIFT descriptor takes two barriers per tile, not
 * the 16 of 16-channel slabs; a software GPU (SwiftShader) spends most of a tiled kernel's time
 * in barriers. A cell off the line skips its sums, and its distance never enters the column's
 * best and second. Each sum runs in the scalar kernel's channel order, so the indices, the
 * distances, the sole-candidate rule and the degenerate-line rule are those of
 * `nearestFloatGuided`. No lane returns before the barriers: a lane past the queries reads the
 * last query and writes nothing.
 */
@compute([64])
export function nearestFloatGuidedTiles(
  @builtin('workgroup_id') wid: vec3u,
  @builtin('local_invocation_id') lid: vec3u,
): void {
  const lane: u32 = lid.x;
  const query: u32 = wid.x * 8 + lane / 8;
  const column: u32 = lane % 8;
  const active: bool = query < floatMatchControl.queries;
  const source: u32 = min(query, floatMatchControl.queries - 1);
  const x: f32 = queryPoints[source * 2];
  const y: f32 = queryPoints[source * 2 + 1];
  const la: f32 = epipolarGuide.m0 * x + epipolarGuide.m1 * y + epipolarGuide.m2;
  const lb: f32 = epipolarGuide.m3 * x + epipolarGuide.m4 * y + epipolarGuide.m5;
  const lc: f32 = epipolarGuide.m6 * x + epipolarGuide.m7 * y + epipolarGuide.m8;
  const norm: f32 = sqrt(la * la + lb * lb);
  // A degenerate line (the query at an epipole) admits no candidate: the query stays unmatched.
  const line: bool = active && norm > 1e-12;
  let best: f32 = bitcast<f32>(u32(0x7f7fffff));
  let second: f32 = bitcast<f32>(u32(0x7f7fffff));
  let bestIndex: u32 = 0xffffffff;
  let secondIndex: u32 = 0xffffffff;
  let invalid: u32 = 0;
  for (let base: u32 = 0; base < floatMatchControl.candidates; base += 8) {
    const candidate: u32 = base + column;
    let admitted: bool = false;
    if (line && candidate < floatMatchControl.candidates) {
      const u: f32 = candidatePoints[candidate * 2];
      const v: f32 = candidatePoints[candidate * 2 + 1];
      admitted = !(abs(la * u + lb * v + lc) > epipolarGuide.maxLinePx * norm);
    }
    let distance: f32 = 0;
    for (let channel: u32 = 0; channel < floatMatchControl.dimensions; channel += 128) {
      for (let k: u32 = 0; k < 16; k++) {
        const slot: u32 = lane + k * 64;
        const q: u32 = wid.x * 8 + slot / 128;
        const c: u32 = base + slot / 128;
        const d: u32 = channel + slot % 128;
        let a: f32 = 0;
        let b: f32 = 0;
        if (q < floatMatchControl.queries && d < floatMatchControl.dimensions)
          a = floatQueries[q * floatMatchControl.dimensions + d];
        if (c < floatMatchControl.candidates && d < floatMatchControl.dimensions)
          b = floatCandidates[c * floatMatchControl.dimensions + d];
        querySlab[slot] = a;
        candidateSlab[slot] = b;
      }
      workgroupBarrier();
      // A cell off the line runs no sum; every lane still reaches both barriers.
      let width: u32 = 0;
      if (admitted) width = min(floatMatchControl.dimensions - channel, 128);
      for (let d: u32 = 0; d < width; d++) {
        const a: f32 = querySlab[(lane / 8) * 128 + d];
        const b: f32 = candidateSlab[column * 128 + d];
        if ((bitcast<u32>(a) & 0x7f800000) === 0x7f800000 ||
            (bitcast<u32>(b) & 0x7f800000) === 0x7f800000) invalid = 1;
        const delta: f32 = a - b;
        distance += delta * delta;
      }
      workgroupBarrier();
    }
    if (admitted) {
      if ((bitcast<u32>(distance) & 0x7f800000) === 0x7f800000) invalid = 1;
      if (distance < best || (distance === best && candidate < bestIndex)) {
        second = best; secondIndex = bestIndex;
        best = distance; bestIndex = candidate;
      } else if (distance < second || (distance === second && candidate < secondIndex)) {
        second = distance; secondIndex = candidate;
      }
    }
  }
  bestTile[lane * 2] = best;
  bestTile[lane * 2 + 1] = second;
  indexTile[lane * 2] = bestIndex;
  indexTile[lane * 2 + 1] = secondIndex;
  errorTile[lane] = invalid;
  workgroupBarrier();
  if (column === 0 && active) {
    best = bitcast<f32>(u32(0x7f7fffff)); second = bitcast<f32>(u32(0x7f7fffff));
    bestIndex = 0xffffffff; secondIndex = 0xffffffff; invalid = 0;
    for (let c: u32 = 0; c < 8; c++) {
      invalid |= errorTile[lane + c];
      for (let rank: u32 = 0; rank < 2; rank++) {
        const slot: u32 = (lane + c) * 2 + rank;
        const distance: f32 = bestTile[slot];
        const candidate: u32 = indexTile[slot];
        if (candidate === 0xffffffff) continue;
        if (distance < best || (distance === best && candidate < bestIndex)) {
          second = best; secondIndex = bestIndex;
          best = distance; bestIndex = candidate;
        } else if (distance < second || (distance === second && candidate < secondIndex)) {
          second = distance; secondIndex = candidate;
        }
      }
    }
    // A sole candidate on the line passes the ratio test, as in `nearestFloatGuided`.
    if (bestIndex !== 0xffffffff && secondIndex === 0xffffffff)
      secondIndex = floatMatchControl.candidates;
    nearestIndices[query * 3] = bestIndex;
    nearestIndices[query * 3 + 1] = secondIndex;
    nearestIndices[query * 3 + 2] = invalid;
    nearestDistances[query * 2] = best;
    nearestDistances[query * 2 + 1] = second;
  }
}

/** Strict ratio in both directions when mutual is requested. */
function floatMatchAccepted(i: u32): bool {
  const j: u32 = forwardIndices[i * 3];
  const distance: f32 = forwardDistances[i * 2];
  if (j >= floatMatchControl.candidates || forwardIndices[i * 3 + 1] === 0xffffffff ||
      distance > floatMatchControl.maxDistanceSquared ||
      !(distance < floatMatchControl.ratioSquared * forwardDistances[i * 2 + 1])) return false;
  if (floatMatchControl.mutual !== 0) {
    if (backwardIndices[j * 3] !== i || backwardIndices[j * 3 + 1] === 0xffffffff ||
        !(backwardDistances[j * 2] < floatMatchControl.ratioSquared * backwardDistances[j * 2 + 1])) return false;
  }
  return true;
}
/** Flags over max(queries, candidates) elements: bit 0 accepts a query, bit 1 reports a nonfinite row. */
@compute([64])
export function flagFloatMatches(@builtin('global_invocation_id') gid: vec3u): void {
  const i: u32 = gid.x;
  if (i >= max(floatMatchControl.queries, floatMatchControl.candidates)) return;
  let flag: u32 = 0;
  if (i < floatMatchControl.queries) {
    if (forwardIndices[i * 3 + 2] !== 0) flag |= 2;
    if (floatMatchAccepted(i)) flag |= 1;
  }
  if (i < floatMatchControl.candidates && backwardIndices[i * 3 + 2] !== 0) flag |= 2;
  matchFlags[i] = flag;
}
/** Stable source order: accepted queries land at their exclusive scan offset. */
@compute([64])
export function scatterFloatMatches(@builtin('global_invocation_id') gid: vec3u): void {
  const i: u32 = gid.x;
  if (i >= floatMatchControl.queries || !floatMatchAccepted(i)) return;
  const row: u32 = matchOffsets[(i / 64) * 2] + matchFlags[i] - 1;
  floatMatches[row * 2] = i;
  floatMatches[row * 2 + 1] = forwardIndices[i * 3];
  floatDistances[row] = sqrt(forwardDistances[i * 2]);
}
