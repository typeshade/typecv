'use typeshade';

// The voxel binning of a dense point cloud (`docs/dense-mvs.md`, section 3.12). One invocation
// takes one point, finds the voxel of the point, and claims a slot of an open-addressing table
// for it with a compare-exchange on the index of the first point to arrive. The result of a point
// is the index of the point that owns its voxel's slot, so all points of a voxel name one owner.
// Which member owns the slot depends on the order of the invocations. The host orders the voxels
// by first member and sums in double (`reduceVoxels`), so the output does not depend on it.
//
// - `voxelPoints`: f32 × 3 per point.
// - `voxelTable`: atomic u32, a power of two of at least 2 × count slots, each 0xffffffff at start.
// - `voxelRep`: u32 per point, the owner of the point's voxel.

class VoxelControl {
  count: u32;
  // The table size minus 1.
  mask: u32;
  originX: f32;
  originY: f32;
  originZ: f32;
  inverse: f32;
}
class VoxelRange {
  first: u32;
  end: u32;
}
declare const voxelControl: uniform<VoxelControl>;
declare const voxelRange: uniform<VoxelRange>;
declare const voxelPoints: storage<array<f32>>;
declare const voxelTable: storage<array<atomic<u32>>, 'read_write'>;
declare const voxelRep: storage<array<u32>, 'read_write'>;

const EMPTY: u32 = 0xffffffff;

/** The voxel coordinate of a point on one axis: `voxelFrame` of `src/reconstruction/cloud-clean.ts`. */
function voxelAxis(j: u32, axis: u32, origin: f32): u32 {
  return u32(floor((voxelPoints[j * 3 + axis] - origin) * voxelControl.inverse));
}

function voxelSlot(x: u32, y: u32, z: u32): u32 {
  let h: u32 = (x * 73856093) ^ (y * 19349663) ^ (z * 83492791);
  h = h ^ (h >> 16);
  h = h * 0x7feb352d;
  h = h ^ (h >> 15);
  h = h * 0x846ca68b;
  return h ^ (h >> 16);
}

@compute([64])
export function voxelClaim(@builtin('global_invocation_id') gid: vec3u): void {
  let i: u32 = voxelRange.first + gid.x;
  const active: bool = i < voxelRange.end;
  if (!active) i = max(voxelRange.end, 1) - 1;
  const x: u32 = voxelAxis(i, 0, voxelControl.originX);
  const y: u32 = voxelAxis(i, 1, voxelControl.originY);
  const z: u32 = voxelAxis(i, 2, voxelControl.originZ);
  let slot: u32 = voxelSlot(x, y, z) & voxelControl.mask;
  let owner: u32 = EMPTY;
  for (let probe: u32 = 0; probe <= voxelControl.mask; probe++) {
    if (owner !== EMPTY) break;
      const claim = atomicCompareExchangeWeak(voxelTable[slot], EMPTY, i);
      if (claim.exchanged) {
        owner = i;
      } else if (claim.old_value !== EMPTY) {
        const j: u32 = claim.old_value;
        if (
          voxelAxis(j, 0, voxelControl.originX) === x &&
          voxelAxis(j, 1, voxelControl.originY) === y &&
          voxelAxis(j, 2, voxelControl.originZ) === z
        ) {
          owner = j;
        } else {
          slot = (slot + 1) & voxelControl.mask;
        }
      }
  }
  if (active) voxelRep[i] = owner;
}
