'use typeshade';
import { CameraModel, unproject } from './numerics.shade.ts';

class PreparationControl {
  first: CameraModel;
  second: CameraModel;
}
declare const control: uniform<PreparationControl>;
declare const pixels: storage<array<f64>>;
// Per raw row: normalized xyxy, then ideal pixel xyxy.
declare const prepared: storage<array<f64>, 'read_write'>;
// Initially 0 invalid, 1 valid, 2 unsupported intermediate coordinate; then block-local inclusive count.
declare const prefix: storage<array<u32>, 'read_write'>;
// Per workgroup: exclusive block offset (initially count), domain issue count.
declare const blockOffsets: storage<array<u32>, 'read_write'>;
declare const coords: storage<array<f64>, 'read_write'>;
declare const ideal: storage<array<f64>, 'read_write'>;
declare const sourceRows: storage<array<u32>, 'read_write'>;
declare const invertible: storage<array<u32>, 'read_write'>;
declare const outputMask: storage<array<u32>, 'read_write'>;
declare const mask: storage<array<u32>>;
// 0..7 consensus state, 8 H count, 9 H attempts, 10 compact population, 11 domain issues.
declare const state: storage<array<u32>, 'read_write'>;
let scanTile: workgroup<array<u32, 128>>;

function supportedPixel(x: f64): bool {
  const a: f64 = abs(x);
  return a <= 1e9 && (a === 0 || a >= 1e-30);
}
@compute([64])
export function prepareImagePair(@builtin('global_invocation_id') gid: vec3u): void {
  const i: u32 = gid.x;
  if (i >= prefix.length) return;
  prefix[i] = 0;
  for (let j: u32 = 0; j < 8; j++) prepared[i * 8 + j] = 0;
  const a = unproject(control.first, vec2f64(pixels[i * 4], pixels[i * 4 + 1]));
  const b = unproject(control.second, vec2f64(pixels[i * 4 + 2], pixels[i * 4 + 3]));
  if (a.z === 0 || b.z === 0 || max(max(abs(a.x), abs(a.y)), max(abs(b.x), abs(b.y))) > 1e4) return;
  const left = vec2f64(
    a.x * control.first.fx + control.first.cx,
    a.y * control.first.fy + control.first.cy,
  );
  const right = vec2f64(
    b.x * control.second.fx + control.second.cx,
    b.y * control.second.fy + control.second.cy,
  );
  if (
    !supportedPixel(left.x) ||
    !supportedPixel(left.y) ||
    !supportedPixel(right.x) ||
    !supportedPixel(right.y)
  ) {
    prefix[i] = 2;
    return;
  }
  const first: CameraModel = {
    fx: control.first.fx,
    fy: control.first.fy,
    cx: control.first.cx,
    cy: control.first.cy,
    k1: 0,
    k2: 0,
  };
  const second: CameraModel = {
    fx: control.second.fx,
    fy: control.second.fy,
    cx: control.second.cx,
    cy: control.second.cy,
    k1: 0,
    k2: 0,
  };
  // Preserve the ideal-pixel normalization contract of the separate public APIs.
  const x = unproject(first, left),
    y = unproject(second, right);
  if (x.z === 0 || y.z === 0 || max(max(abs(x.x), abs(x.y)), max(abs(y.x), abs(y.y))) > 1e4) return;
  prepared[i * 8] = x.x;
  prepared[i * 8 + 1] = x.y;
  prepared[i * 8 + 2] = y.x;
  prepared[i * 8 + 3] = y.y;
  prepared[i * 8 + 4] = left.x;
  prepared[i * 8 + 5] = left.y;
  prepared[i * 8 + 6] = right.x;
  prepared[i * 8 + 7] = right.y;
  prefix[i] = 1;
}
@compute([64])
export function scanPairRows(
  @builtin('local_invocation_id') lid: vec3u,
  @builtin('workgroup_id') wid: vec3u,
): void {
  const i: u32 = wid.x * 64 + lid.x;
  let flag: u32 = 0;
  if (i < prefix.length) flag = prefix[i];
  scanTile[lid.x] = flag & 1;
  scanTile[64 + lid.x] = flag >> 1;
  workgroupBarrier();
  for (let stride: u32 = 1; stride < 64; stride *= 2) {
    let previous: u32 = 0,
      issue: u32 = 0;
    if (lid.x >= stride) {
      previous = scanTile[lid.x - stride];
      issue = scanTile[64 + lid.x - stride];
    }
    workgroupBarrier();
    scanTile[lid.x] += previous;
    scanTile[64 + lid.x] += issue;
    workgroupBarrier();
  }
  if (i < prefix.length) prefix[i] = scanTile[lid.x];
  if (lid.x === 63) {
    blockOffsets[wid.x * 2] = scanTile[63];
    blockOffsets[wid.x * 2 + 1] = scanTile[127];
  }
}
@compute([8])
export function scanPairRowsOracle(@builtin('global_invocation_id') gid: vec3u): void {
  const block: u32 = gid.x;
  if (block >= blockOffsets.length / 2) return;
  let sum: u32 = 0,
    issues: u32 = 0;
  for (let j: u32 = 0; j < 64; j++) {
    const i: u32 = block * 64 + j;
    if (i >= prefix.length) break;
    const flag: u32 = prefix[i];
    sum += flag & 1;
    issues += flag >> 1;
    prefix[i] = sum;
  }
  blockOffsets[block * 2] = sum;
  blockOffsets[block * 2 + 1] = issues;
}
@compute([1])
export function scanPairBlocks(): void {
  let sum: u32 = 0,
    issues: u32 = 0;
  for (let i: u32 = 0; i < blockOffsets.length / 2; i++) {
    const count: u32 = blockOffsets[i * 2];
    blockOffsets[i * 2] = sum;
    sum += count;
    issues += blockOffsets[i * 2 + 1];
  }
  state[8] = 0;
  state[9] = 0;
  state[10] = sum;
  state[11] = issues;
}
@compute([64])
export function compactImagePair(@builtin('global_invocation_id') gid: vec3u): void {
  const i: u32 = gid.x;
  if (i >= prefix.length) return;
  let previous: u32 = 0;
  if (i % 64 !== 0) previous = prefix[i - 1];
  if (prefix[i] === previous) return;
  const row: u32 = blockOffsets[(i / 64) * 2] + prefix[i] - 1;
  for (let j: u32 = 0; j < 4; j++) {
    coords[row * 4 + j] = prepared[i * 8 + j];
    ideal[row * 4 + j] = prepared[i * 8 + 4 + j];
  }
  sourceRows[row] = i;
}
@compute([64])
export function finishPairPreparation(@builtin('global_invocation_id') gid: vec3u): void {
  const i: u32 = gid.x;
  if (i >= invertible.length) return;
  invertible[i] = 0;
  outputMask[i] = 0;
  if (i < state[10]) invertible[i] = 1;
}
@compute([1])
export function preserveHomography(): void {
  state[8] = state[3];
  state[9] = state[1];
}
@compute([64])
export function scatterPairConsensus(@builtin('global_invocation_id') gid: vec3u): void {
  const i: u32 = gid.x;
  if (i >= mask.length || mask[i] === 0) return;
  outputMask[sourceRows[i]] = 1;
}
