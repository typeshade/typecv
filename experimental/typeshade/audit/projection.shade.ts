"use typeshade";

export function projection(x: f32, z: f32, focal: f32, center: f32): f32 {
  return focal * x / z + center;
}
