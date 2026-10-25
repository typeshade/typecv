import type { TsdfView } from '../reconstruction/tsdf.ts';

/**
 * Borrow complete input channels instead of interleaving and retaining a copy of every view.
 * Resident.write snapshots its argument before returning. These views never escape to callers,
 * and no input bytes are changed. The uncommon short-channel padding preserves the old packer's
 * undefined-to-NaN (float) / undefined-to-zero (u32) behavior in bounded, reusable scratch arrays.
 */
export class TsdfViewChannels {
  private readonly scratch: (Uint32Array | undefined)[] = [];
  private ones?: Uint32Array;
  private readonly pixels: number;

  constructor(pixels: number) {
    this.pixels = pixels;
  }

  get hostBytes(): number {
    return (
      (this.ones?.byteLength ?? 0) +
      this.scratch.reduce((n, words) => n + (words?.byteLength ?? 0), 0)
    );
  }

  private words(values: Float32Array | Uint32Array, channel: number): Uint32Array {
    if (values.length >= this.pixels)
      return new Uint32Array(values.buffer, values.byteOffset, this.pixels);
    const words = (this.scratch[channel] ??= new Uint32Array(this.pixels));
    if (values instanceof Float32Array) {
      const floats = new Float32Array(words.buffer);
      floats.fill(NaN);
      floats.set(values);
    } else {
      words.fill(0);
      words.set(values);
    }
    return words;
  }

  of(view: TsdfView) {
    return {
      tsdfDepth: this.words(view.depth, 0),
      tsdfNormal: this.words(view.normal, 1),
      tsdfConfidence: view.confidence
        ? this.words(view.confidence, 2)
        : (this.ones ??= new Uint32Array(this.pixels).fill(0x3f800000)),
      tsdfColor: this.words(view.rgba, 3),
    };
  }
}
