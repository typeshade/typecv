/** The offsets and sizes of the levels of a pyramid of a frame. */
export interface KltLayout {
  readonly levels: number;
  readonly widths: Uint32Array;
  readonly heights: Uint32Array;
  /** Pixel offsets of the levels in `gray`; a gradient array has two floats for each pixel. */
  readonly offsets: Uint32Array;
  /** The pixels of all levels. */
  readonly total: number;
}

export function kltLayout(width: number, height: number, levels: number): KltLayout {
  if (!Number.isInteger(levels) || levels < 1 || levels > 8)
    throw new RangeError('levels must be an integer from 1 to 8');
  if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1)
    throw new RangeError('Invalid frame size');
  const widths = new Uint32Array(levels),
    heights = new Uint32Array(levels),
    offsets = new Uint32Array(levels);
  let w = width,
    h = height,
    offset = 0;
  for (let l = 0; l < levels; l++) {
    widths[l] = w;
    heights[l] = h;
    offsets[l] = offset;
    offset += w * h;
    w = (w + 1) >> 1;
    h = (h + 1) >> 1;
  }
  return { levels, widths, heights, offsets, total: offset };
}
