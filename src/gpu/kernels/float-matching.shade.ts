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
let queryTile: workgroup<array<f32, 128>>;
let candidateTile: workgroup<array<f32, 128>>;
let bestTile: workgroup<array<f32, 128>>;
let indexTile: workgroup<array<u32, 128>>;
let errorTile: workgroup<array<u32, 64>>;

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

/** Stable source order; strict ratio in both directions when mutual is requested. */
@compute([1])
export function compactFloatMatches(@builtin('global_invocation_id') gid: vec3u): void {
  if (gid.x !== 0) return;
  let count: u32 = 0;
  let invalid: u32 = 0;
  for (let j: u32 = 0; j < floatMatchControl.candidates; j++)
    invalid |= backwardIndices[j * 3 + 2];
  for (let i: u32 = 0; i < floatMatchControl.queries; i++) {
    invalid |= forwardIndices[i * 3 + 2];
    const j: u32 = forwardIndices[i * 3];
    const distance: f32 = forwardDistances[i * 2];
    if (j >= floatMatchControl.candidates || forwardIndices[i * 3 + 1] === 0xffffffff ||
        distance > floatMatchControl.maxDistanceSquared ||
        !(distance < floatMatchControl.ratioSquared * forwardDistances[i * 2 + 1])) continue;
    if (floatMatchControl.mutual !== 0) {
      if (backwardIndices[j * 3] !== i || backwardIndices[j * 3 + 1] === 0xffffffff ||
          !(backwardDistances[j * 2] < floatMatchControl.ratioSquared * backwardDistances[j * 2 + 1])) continue;
    }
    floatMatches[count * 2] = i;
    floatMatches[count * 2 + 1] = j;
    floatDistances[count] = sqrt(distance);
    count++;
  }
  floatMatchStatus[0] = count;
  floatMatchStatus[1] = invalid;
}
