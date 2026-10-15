"use typeshade";

declare const values: storage<array<f64>>;
declare const wide: storage<array<f64>, "read_write">;
declare const ordinary: storage<array<f32>, "read_write">;

@compute([64])
export function compareWords(@builtin("global_invocation_id") gid: vec3u): void {
  const i = gid.x;
  if (i >= values.length) return;
  const x: f64 = values[i];
  const step: f64 = 0.125;
  wide[i] = (x + step) - x;
  // Store the f32 sum before the host compares it with f32(x), so a driver
  // cannot reassociate the cancellation into a retained increment.
  ordinary[i] = f32(x) + 0.125;
}
