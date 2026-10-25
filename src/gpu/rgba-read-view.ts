const littleEndian = new Uint8Array(Uint32Array.of(0x01020304).buffer)[0] === 4;

/**
 * @internal Borrow packed RGBA words for a read-only consumer while its source stays alive.
 * Unlike `packRgba`, this may alias the caller's bytes; never write or transfer the result.
 * MVS retains the source images until TSDF and planar completion have finished reading them.
 */
export function readOnlyRgbaWords(
  rgba: Uint8Array | Uint8ClampedArray,
  pixelCount: number,
): Uint32Array {
  if (littleEndian && rgba.byteOffset % 4 === 0 && rgba.byteLength === 4 * pixelCount)
    return new Uint32Array(rgba.buffer, rgba.byteOffset, pixelCount);
  const packed = new Uint32Array(pixelCount);
  for (let i = 0; i < pixelCount; i++) {
    const at = i * 4;
    packed[i] = rgba[at]! | (rgba[at + 1]! << 8) | (rgba[at + 2]! << 16) | (rgba[at + 3]! << 24);
  }
  return packed;
}
