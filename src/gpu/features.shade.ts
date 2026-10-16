'use typeshade';

class FeatureControl {
  width: u32;
  height: u32;
  tilesX: u32;
  tilesY: u32;
  slotsPerTile: u32;
  selectedTiles: u32;
  threshold: f32;
  capacity: u32;
}
class MatchControl {
  firstCount: u32;
  secondCount: u32;
  ratio: f32;
  maxDistance: u32;
}
declare const featureControl: uniform<FeatureControl>;
declare const matchControl: uniform<MatchControl>;
declare const rgba: storage<array<u32>>;
declare const gray: storage<array<f32>, 'read_write'>;
declare const scores: storage<array<f32>, 'read_write'>;
declare const features: storage<array<f32>, 'read_write'>;
declare const descriptors: storage<array<u32>, 'read_write'>;
declare const rightFeatures: storage<array<f32>>;
declare const leftDescriptors: storage<array<u32>>;
declare const rightDescriptors: storage<array<u32>>;
declare const matches: storage<array<u32>, 'read_write'>;
declare const packedFeatures: storage<array<f32>, 'read_write'>;
declare const packedDescriptors: storage<array<u32>, 'read_write'>;
declare const featureCount: storage<array<u32>, 'read_write'>;
declare const forwardMatches: storage<array<u32>>;
declare const reverseMatches: storage<array<u32>>;
declare const mutualMatches: storage<array<u32>, 'read_write'>;
declare const mutualDistances: storage<array<u32>, 'read_write'>;

function fastScore(x: u32, y: u32): f32 {
  const center: f32 = gray[y * featureControl.width + x];
  let best: f32 = 0;
  for (let start: u32 = 0; start < 16; start++) {
    let bright: bool = true;
    let dark: bool = true;
    let contrast: f32 = 1e6;
    for (let k: u32 = 0; k < 9; k++) {
      const j: u32 = (start + k) % 16;
      let dx: i32 = 0;
      let dy: i32 = 0;
      if (j === 0) dy = -3;
      else if (j === 1) { dx = 1; dy = -3; }
      else if (j === 2) { dx = 2; dy = -2; }
      else if (j === 3) { dx = 3; dy = -1; }
      else if (j === 4) dx = 3;
      else if (j === 5) { dx = 3; dy = 1; }
      else if (j === 6) { dx = 2; dy = 2; }
      else if (j === 7) { dx = 1; dy = 3; }
      else if (j === 8) dy = 3;
      else if (j === 9) { dx = -1; dy = 3; }
      else if (j === 10) { dx = -2; dy = 2; }
      else if (j === 11) { dx = -3; dy = 1; }
      else if (j === 12) dx = -3;
      else if (j === 13) { dx = -3; dy = -1; }
      else if (j === 14) { dx = -2; dy = -2; }
      else { dx = -1; dy = -3; }
      const sample: f32 = gray[u32(i32(y) + dy) * featureControl.width + u32(i32(x) + dx)];
      const delta: f32 = sample - center;
      if (delta <= featureControl.threshold) bright = false;
      if (delta >= -featureControl.threshold) dark = false;
      contrast = min(contrast, abs(delta));
    }
    if (bright || dark) best = max(best, contrast);
  }
  return best;
}
function hash(value: u32): u32 {
  let x: u32 = value + 0x9e3779b9;
  x = (x ^ (x >> 16)) * 0x85ebca6b;
  x = (x ^ (x >> 13)) * 0xc2b2ae35;
  return x ^ (x >> 16);
}
function grayAt(x: i32, y: i32): f32 {
  const xx: u32 = u32(clamp(x, 0, i32(featureControl.width) - 1));
  const yy: u32 = u32(clamp(y, 0, i32(featureControl.height) - 1));
  return gray[yy * featureControl.width + xx];
}

@compute([64])
export function rgbaToGray(@builtin('global_invocation_id') gid: vec3u): void {
  const i: u32 = gid.x;
  if (i >= rgba.length) return;
  const pixel: u32 = rgba[i];
  const r: f32 = f32(pixel & 255);
  const g: f32 = f32((pixel >> 8) & 255);
  const b: f32 = f32((pixel >> 16) & 255);
  gray[i] = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
}

@compute([64])
export function scoreFast9(@builtin('global_invocation_id') gid: vec3u): void {
  const x: u32 = gid.x % featureControl.width;
  const y: u32 = gid.x / featureControl.width;
  if (gid.x >= gray.length) return;
  scores[gid.x] = 0;
  if (x < 4 || y < 4 || x + 4 >= featureControl.width || y + 4 >= featureControl.height) return;
  scores[gid.x] = fastScore(x, y);
}

@compute([64])
export function selectTileCorners(@builtin('global_invocation_id') gid: vec3u): void {
  const selectedTile: u32 = gid.x;
  const tileCount: u32 = featureControl.tilesX * featureControl.tilesY;
  if (selectedTile >= featureControl.selectedTiles) return;
  const tile: u32 = ((2 * selectedTile + 1) * tileCount) / (2 * featureControl.selectedTiles);
  const tileX: u32 = tile % featureControl.tilesX;
  const tileY: u32 = tile / featureControl.tilesX;
  for (let slot: u32 = 0; slot < featureControl.slotsPerTile; slot++) {
    let bestIndex: u32 = 0xffffffff;
    let bestScore: f32 = featureControl.threshold;
    for (let dy: u32 = 0; dy < 24; dy++) {
      for (let dx: u32 = 0; dx < 24; dx++) {
        const x: u32 = tileX * 24 + dx;
        const y: u32 = tileY * 24 + dy;
        if (x >= featureControl.width || y >= featureControl.height) continue;
        const index: u32 = y * featureControl.width + x;
        const value: f32 = scores[index];
        if (value > bestScore) { bestScore = value; bestIndex = index; }
      }
    }
    const out: u32 = selectedTile * featureControl.slotsPerTile + slot;
    if (out >= featureControl.capacity) return;
    if (bestIndex === 0xffffffff) {
      features[out * 4] = 0; features[out * 4 + 1] = 0; features[out * 4 + 2] = 0; features[out * 4 + 3] = 0;
    } else {
      features[out * 4] = f32(bestIndex % featureControl.width);
      features[out * 4 + 1] = f32(bestIndex / featureControl.width);
      features[out * 4 + 2] = bestScore;
      features[out * 4 + 3] = 1;
      scores[bestIndex] = -1;
    }
  }
}

@compute([64])
export function orientBrief(@builtin('global_invocation_id') gid: vec3u): void {
  const i: u32 = gid.x;
  if (i >= featureControl.capacity) return;
  for (let word: u32 = 0; word < 8; word++) descriptors[i * 8 + word] = 0;
  if (features[i * 4 + 3] === 0) return;
  const x: i32 = i32(features[i * 4]);
  const y: i32 = i32(features[i * 4 + 1]);
  let m10: f32 = 0;
  let m01: f32 = 0;
  for (let dy: i32 = -8; dy <= 8; dy++) {
    for (let dx: i32 = -8; dx <= 8; dx++) {
      if (dx * dx + dy * dy > 64) continue;
      const intensity: f32 = grayAt(x + dx, y + dy);
      m10 += f32(dx) * intensity;
      m01 += f32(dy) * intensity;
    }
  }
  const angle: f32 = atan2(m01, m10);
  const c: f32 = cos(angle);
  const s: f32 = sin(angle);
  for (let bit: u32 = 0; bit < 256; bit++) {
    const seed: u32 = bit * 4 + 11;
    const a: u32 = hash(seed);
    const b: u32 = hash(seed + 1);
    const c0: u32 = hash(seed + 2);
    const d: u32 = hash(seed + 3);
    const ax: f32 = f32(i32(a % 17) - 8);
    const ay: f32 = f32(i32(b % 17) - 8);
    const bx: f32 = f32(i32(c0 % 17) - 8);
    const by: f32 = f32(i32(d % 17) - 8);
    const va: f32 = grayAt(x + i32(round(c * ax - s * ay)), y + i32(round(s * ax + c * ay)));
    const vb: f32 = grayAt(x + i32(round(c * bx - s * by)), y + i32(round(s * bx + c * by)));
    if (va < vb) descriptors[i * 8 + bit / 32] |= u32(1) << (bit % 32);
  }
}

/** Stable compact of tile slots; one invocation keeps feature indices deterministic. */
@compute([1])
export function compactFeatures(@builtin('global_invocation_id') gid: vec3u): void {
  if (gid.x !== 0) return;
  let count: u32 = 0;
  for (let i: u32 = 0; i < featureControl.capacity; i++) {
    if (features[i * 4 + 3] === 0) continue;
    for (let k: u32 = 0; k < 4; k++) packedFeatures[count * 4 + k] = features[i * 4 + k];
    for (let k: u32 = 0; k < 8; k++) packedDescriptors[count * 8 + k] = descriptors[i * 8 + k];
    count++;
  }
  featureCount[0] = count;
}

@compute([64])
export function matchBrief(@builtin('global_invocation_id') gid: vec3u): void {
  const i: u32 = gid.x;
  if (i >= matchControl.firstCount) return;
  let best: u32 = 0xffffffff;
  let bestDistance: u32 = 0xffffffff;
  let secondDistance: u32 = 0xffffffff;
  for (let j: u32 = 0; j < matchControl.secondCount; j++) {
    if (rightFeatures[j * 4 + 3] === 0) continue;
    let distance: u32 = 0;
    for (let word: u32 = 0; word < 8; word++) distance += countOneBits(leftDescriptors[i * 8 + word] ^ rightDescriptors[j * 8 + word]);
    if (distance < bestDistance) { secondDistance = bestDistance; bestDistance = distance; best = j; }
    else if (distance < secondDistance) secondDistance = distance;
  }
  const accepted: u32 = u32(best !== 0xffffffff && secondDistance !== 0xffffffff && bestDistance <= matchControl.maxDistance && f32(bestDistance) < matchControl.ratio * f32(secondDistance));
  matches[i * 3] = best;
  matches[i * 3 + 1] = bestDistance;
  matches[i * 3 + 2] = accepted;
}

/** Stable GPU mutual-best filter and compaction, preserving the source-feature order. */
@compute([1])
export function compactMutualBrief(@builtin('global_invocation_id') gid: vec3u): void {
  if (gid.x !== 0) return;
  let count: u32 = 0;
  for (let i: u32 = 0; i < matchControl.firstCount; i++) {
    const j: u32 = forwardMatches[i * 3];
    if (forwardMatches[i * 3 + 2] === 0 || j >= matchControl.secondCount) continue;
    if (reverseMatches[j * 3 + 2] === 0 || reverseMatches[j * 3] !== i) continue;
    mutualMatches[count * 2] = i;
    mutualMatches[count * 2 + 1] = j;
    mutualDistances[count] = forwardMatches[i * 3 + 1];
    count++;
  }
  featureCount[0] = count;
}
