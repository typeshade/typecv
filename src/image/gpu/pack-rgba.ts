/** @internal RGBA bytes as the packed u32 words that `rgbaToGray` reads. */
export function packRgba(rgba: Uint8Array | Uint8ClampedArray, pixelCount: number): Uint32Array {
  const littleEndian = new Uint8Array(Uint32Array.of(0x01020304).buffer)[0] === 4;
  if (littleEndian && rgba.byteOffset % 4 === 0 && rgba.byteLength % 4 === 0)
    return new Uint32Array(rgba.buffer, rgba.byteOffset, pixelCount).slice();
  const packed = new Uint32Array(pixelCount);
  for (let i = 0; i < pixelCount; i++) {
    const at = i * 4;
    packed[i] = rgba[at]! | (rgba[at + 1]! << 8) | (rgba[at + 2]! << 16) | (rgba[at + 3]! << 24);
  }
  return packed;
}
