'use typeshade';

// Original ALIKED arithmetic, frozen before the row-normalization optimization.
// Only function names differ so both versions can execute on the same device.
export function legacyNormalizeL2(out: array<f32>, source: array<f32>, channels: u32): void {
  for (let i: u32 = 0; i < out.length; i++) {
    const pixel = i / channels;
    const channel = i % channels;
    let squared: f32 = 0.0;
    for (let c: u32 = 0; c < channels; c++) {
      const value = source[pixel * channels + c];
      squared += value * value;
    }
    out[i] = source[i] / max(sqrt(squared), 1e-12);
  }
}

export function legacyNormalizeDescriptors(
  out: array<f32>,
  source: array<f32>,
  keypointCount: u32,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    const kp = i / 128;
    let squared: f32 = 0.0;
    for (let d: u32 = 0; d < 128; d++) {
      const value = source[kp * 128 + d];
      squared += value * value;
    }
    out[i] = source[i] / max(sqrt(squared), 1e-12);
  }
}

export function unpad(
  out: array<f32>,
  source: array<f32>,
  paddedWidth: u32,
  width: u32,
  channels: u32,
  padTop: u32,
  padLeft: u32,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    const c = i % channels;
    const pixel = i / channels;
    const x = pixel % width;
    const y = pixel / width;
    out[i] = source[((y + padTop) * paddedWidth + x + padLeft) * channels + c];
  }
}

export function gatherPatch3x3(
  out: array<f32>,
  featureMap: array<f32>,
  keypoints: array<f32>,
  height: u32,
  width: u32,
  channels: u32,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    const channel = i % channels;
    const patchIndex = i / channels;
    const dx = patchIndex % 3;
    const dy = (patchIndex / 3) % 3;
    const kp = patchIndex / 9;
    const centerX = i32((keypoints[kp * 2] * 0.5 + 0.5) * f32(width - 1));
    const centerY = i32((keypoints[kp * 2 + 1] * 0.5 + 0.5) * f32(height - 1));
    const cornerX = clamp(centerX - 1, 0, i32(width) - 4);
    const cornerY = clamp(centerY - 1, 0, i32(height) - 4);
    const x = u32(cornerX + i32(dx));
    const y = u32(cornerY + i32(dy));
    out[i] = featureMap[(y * width + x) * channels + channel];
  }
}

// Frozen TypeNN grid sampler from 0aec8749486f342411d71adb4d05f390b8c38fe8.
export function gridSampleNHWC(
  out: array<f32>,
  source: array<f32>,
  grid: array<f32>,
  height: u32,
  width: u32,
  channels: u32,
  outHeight: u32,
  outWidth: u32,
  alignCorners: u32,
): void {
  for (let i: u32 = 0; i < out.length; i++) {
    const channel = i % channels;
    const outPixel = i / channels;
    const outX = outPixel % outWidth;
    const outY = (outPixel / outWidth) % outHeight;
    const b = outPixel / (outWidth * outHeight);
    const gridIndex = (b * outHeight * outWidth + outY * outWidth + outX) * 2;
    const gx = grid[gridIndex];
    const gy = grid[gridIndex + 1];

    // Classify exponent-all-ones values by their IEEE-754 bits. Arithmetic NaN checks can be
    // folded under GPU fast-math, and subtraction does not reliably manufacture a NaN there.
    const gxBits = bitcast<u32>(gx);
    const gyBits = bitcast<u32>(gy);
    const gxExponent = gxBits & 0x7f800000;
    const gyExponent = gyBits & 0x7f800000;
    if (gxExponent === 0x7f800000 || gyExponent === 0x7f800000) {
      const nonFiniteBits = select(gyBits, gxBits, gxExponent === 0x7f800000);
      out[i] = bitcast<f32>(nonFiniteBits | 0x00400000);
      continue;
    }

    let x: f32 = 0.0;
    let y: f32 = 0.0;
    if (alignCorners !== 0) {
      x = (gx + 1.0) * f32(width - 1) * 0.5;
      y = (gy + 1.0) * f32(height - 1) * 0.5;
    } else {
      x = ((gx + 1.0) * f32(width) - 1.0) * 0.5;
      y = ((gy + 1.0) * f32(height) - 1.0) * 0.5;
    }

    // Avoid undefined integer conversion for finite coordinates far outside the image.
    if (x < -1.0 || x > f32(width) || y < -1.0 || y > f32(height)) {
      out[i] = 0.0;
      continue;
    }

    const x0 = i32(floor(x));
    const y0 = i32(floor(y));
    const x1 = x0 + 1;
    const y1 = y0 + 1;
    const wx = x - f32(x0);
    const wy = y - f32(y0);
    const bOffset = b * height * width * channels;
    let value: f32 = 0.0;
    if (x0 >= 0 && x0 < i32(width) && y0 >= 0 && y0 < i32(height)) {
      value +=
        source[bOffset + (u32(y0) * width + u32(x0)) * channels + channel] *
        (1.0 - wx) *
        (1.0 - wy);
    }
    if (x1 >= 0 && x1 < i32(width) && y0 >= 0 && y0 < i32(height)) {
      value += source[bOffset + (u32(y0) * width + u32(x1)) * channels + channel] * wx * (1.0 - wy);
    }
    if (x0 >= 0 && x0 < i32(width) && y1 >= 0 && y1 < i32(height)) {
      value += source[bOffset + (u32(y1) * width + u32(x0)) * channels + channel] * (1.0 - wx) * wy;
    }
    if (x1 >= 0 && x1 < i32(width) && y1 >= 0 && y1 < i32(height)) {
      value += source[bOffset + (u32(y1) * width + u32(x1)) * channels + channel] * wx * wy;
    }
    out[i] = value;
  }
}
