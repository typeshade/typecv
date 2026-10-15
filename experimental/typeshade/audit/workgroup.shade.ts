"use typeshade";

const WG: override<u32> = 64;
declare const values: storage<array<f32>>;
declare const sums: storage<array<f32>, "read_write">;
declare const accepted: storage<array<atomic<u32>>, "read_write">;
let tile: workgroup<array<f32, WG>>;

@compute([WG])
export function reduce(
  @builtin("global_invocation_id") gid: vec3u,
  @builtin("local_invocation_id") lid: vec3u,
  @builtin("workgroup_id") wid: vec3u,
): void {
  let value: f32 = 0.;
  if (gid.x < values.length) {
    value = values[gid.x];
    if (value > 0.) atomicAdd(accepted[0], 1);
  }
  tile[lid.x] = value;
  workgroupBarrier();
  for (let stride: u32 = WG / 2; stride > 0; stride /= 2) {
    if (lid.x < stride) tile[lid.x] = tile[lid.x] + tile[lid.x + stride];
    workgroupBarrier();
  }
  if (lid.x === 0) sums[wid.x] = tile[0];
}
