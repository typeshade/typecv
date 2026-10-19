'use typeshade';

// Shared flag scan for device compaction. Bit 0 of a flag selects an element and bit 1
// marks an issue. After `scanFlags`, each flag holds the inclusive selected count within
// its block of 64; `scanFlagBlocks` turns the block counts into exclusive offsets.
declare const flags: storage<array<u32>, 'read_write'>;
// Per block: selected count then issue count; after the block scan, exclusive offset then issue count.
declare const blockSums: storage<array<u32>, 'read_write'>;
// [0] selected total, [1] issue total.
declare const totals: storage<array<u32>, 'read_write'>;
let scanTile: workgroup<array<u32, 128>>;

@compute([64])
export function scanFlags(
  @builtin('local_invocation_id') lid: vec3u,
  @builtin('workgroup_id') wid: vec3u,
): void {
  const i: u32 = wid.x * 64 + lid.x;
  let flag: u32 = 0;
  if (i < flags.length) flag = flags[i];
  scanTile[lid.x] = flag & 1;
  scanTile[64 + lid.x] = (flag >> 1) & 1;
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
  if (i < flags.length) flags[i] = scanTile[lid.x];
  if (lid.x === 63) {
    blockSums[wid.x * 2] = scanTile[63];
    blockSums[wid.x * 2 + 1] = scanTile[127];
  }
}
/** CPU differential tier: one invocation scans one block serially; no barriers. */
@compute([8])
export function scanFlagsOracle(@builtin('global_invocation_id') gid: vec3u): void {
  const block: u32 = gid.x;
  if (block >= blockSums.length / 2) return;
  let sum: u32 = 0,
    issues: u32 = 0;
  for (let j: u32 = 0; j < 64; j++) {
    const i: u32 = block * 64 + j;
    if (i >= flags.length) break;
    const flag: u32 = flags[i];
    sum += flag & 1;
    issues += (flag >> 1) & 1;
    flags[i] = sum;
  }
  blockSums[block * 2] = sum;
  blockSums[block * 2 + 1] = issues;
}
@compute([1])
export function scanFlagBlocks(): void {
  let sum: u32 = 0,
    issues: u32 = 0;
  for (let i: u32 = 0; i < blockSums.length / 2; i++) {
    const count: u32 = blockSums[i * 2];
    blockSums[i * 2] = sum;
    sum += count;
    issues += blockSums[i * 2 + 1];
  }
  totals[0] = sum;
  totals[1] = issues;
}
