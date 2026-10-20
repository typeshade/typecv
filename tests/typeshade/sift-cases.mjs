// Synthetic SIFT inputs, shared by the CPU-tier and the WebGPU tests.
/** A 256 × 192 texture of blobs and bars with smooth edges, so that scale space has structure. */
export function texture(width, height) {
  const gray = new Float32Array(width * height);
  let state = 12345;
  const blobs = [];
  for (let i = 0; i < 70; i++) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    const x = 12 + (state % (width - 24));
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    const y = 12 + (state % (height - 24));
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    const r = 2 + (state % 7);
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    const sign = state % 2 ? 1 : -1;
    blobs.push({ x, y, r, sign });
  }
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      let value = 0.5 + 0.08 * Math.sin(x / 23) * Math.cos(y / 17);
      for (const blob of blobs) {
        const d2 = (x - blob.x) ** 2 + (y - blob.y) ** 2;
        value += blob.sign * 0.4 * Math.exp(-d2 / (2 * blob.r * blob.r));
      }
      gray[y * width + x] = Math.max(0, Math.min(1, value));
    }
  return gray;
}
/** Sample the texture under a similarity (scale, rotation, translation) with bilinear interpolation. */
export function warped(gray, width, height, outWidth, outHeight, scale, angle, tx, ty) {
  const rgba = new Uint8Array(outWidth * outHeight * 4);
  const c = Math.cos(angle),
    s = Math.sin(angle);
  for (let y = 0; y < outHeight; y++)
    for (let x = 0; x < outWidth; x++) {
      // Inverse map: source = R⁻¹ ((target - t) / scale).
      const ux = (x - tx) / scale,
        uy = (y - ty) / scale,
        sx = c * ux + s * uy,
        sy = -s * ux + c * uy;
      let value = 0.5;
      if (sx >= 0 && sy >= 0 && sx < width - 1 && sy < height - 1) {
        const x0 = Math.floor(sx),
          y0 = Math.floor(sy),
          fx = sx - x0,
          fy = sy - y0;
        value =
          gray[y0 * width + x0] * (1 - fx) * (1 - fy) +
          gray[y0 * width + x0 + 1] * fx * (1 - fy) +
          gray[(y0 + 1) * width + x0] * (1 - fx) * fy +
          gray[(y0 + 1) * width + x0 + 1] * fx * fy;
      }
      const byte = Math.round(value * 255),
        i = (y * outWidth + x) * 4;
      rgba[i] = rgba[i + 1] = rgba[i + 2] = byte;
      rgba[i + 3] = 255;
    }
  return rgba;
}
