/**
 * The lab page's dense cloud (`docs/dense-mvs.md` section 6, step 7 of section 8): the points of
 * the worker's `densePoints` messages, the point budget of the viewer and the binary PLY of the
 * export. The viewer draws a subset of the points. The export writes every point.
 */

/** One `densePoints` message (5.3): the fused points of one reference view. */
export type DenseChunk = {
  readonly viewId: number;
  /** x, y, z per point, in the model's frame. */
  readonly positions: Float32Array;
  /** r, g, b per point, sRGB bytes. */
  readonly colors: Uint8Array;
  /** Two snorm16 values per point: the octahedral map of the unit world normal (4.1). */
  readonly normals?: Int16Array;
};

/**
 * The dense cloud of one run. `chunks` only grows during a run, so a reader that keeps the number
 * of chunks it has read can append the new ones. `run` changes when a new run starts.
 */
export type DenseCloud = {
  readonly run: number;
  readonly chunks: readonly DenseChunk[];
  /** The sum of the chunks' point counts. */
  readonly count: number;
};

/** The default point budget of section 6, and the budget of a touch device. */
export const DENSE_POINT_BUDGET = 1_000_000;
export const DENSE_TOUCH_POINT_BUDGET = 300_000;

/** The most points the viewer draws (section 6): 300,000 when the device has touch input. */
export function densePointBudget(maxTouchPoints: number): number {
  return maxTouchPoints > 0 ? DENSE_TOUCH_POINT_BUDGET : DENSE_POINT_BUDGET;
}

/**
 * The step of the drawn subset: the viewer draws every k-th point, with k the smallest integer
 * that keeps the drawn count in the budget (section 6). 1 when the cloud fits.
 */
export function denseStride(count: number, budget: number): number {
  if (!(budget >= 1)) throw new RangeError('The point budget must be at least 1');
  return Math.max(1, Math.ceil(count / budget));
}

/** The number of points that `denseStride` draws: the indices 0, k, 2k, … below `count`. */
export function denseDrawnCount(count: number, stride: number): number {
  return Math.ceil(count / stride);
}

/** The unit vector of an octahedral pair of snorm16 values, as `decodeMvsNormal` in `typecv/gpu`. */
export function decodeOctahedral(low: number, high: number): [number, number, number] {
  const px = Math.max(low / 32767, -1),
    py = Math.max(high / 32767, -1);
  const z = 1 - Math.abs(px) - Math.abs(py),
    t = Math.max(-z, 0);
  const x = px + (px >= 0 ? -t : t),
    y = py + (py >= 0 ? -t : t);
  const length = Math.hypot(x, y, z) || 1;
  return [x / length, y / length, z / length];
}

/** The bytes of one vertex: x, y, z (float), nx, ny, nz (float) and red, green, blue (uchar). */
export function densePlyStride(withNormals: boolean): number {
  return withNormals ? 27 : 15;
}

/**
 * The dense cloud as a binary PLY (section 6): `format binary_little_endian 1.0`, one `vertex`
 * element with x, y, z (float), nx, ny, nz (float, when every chunk has normals) and red, green,
 * blue (uchar). 27 B per point with normals, 15 B without.
 */
export function densePly(chunks: readonly DenseChunk[]): Uint8Array {
  let count = 0;
  for (const chunk of chunks) count += chunk.positions.length / 3;
  const withNormals = chunks.length > 0 && chunks.every((chunk) => chunk.normals);
  const header =
    [
      'ply',
      'format binary_little_endian 1.0',
      'comment typecv dense MVS',
      'element vertex ' + count,
      'property float x',
      'property float y',
      'property float z',
      ...(withNormals ? ['property float nx', 'property float ny', 'property float nz'] : []),
      'property uchar red',
      'property uchar green',
      'property uchar blue',
      'end_header',
    ].join('\n') + '\n';
  const head = new TextEncoder().encode(header);
  const stride = densePlyStride(withNormals);
  const bytes = new Uint8Array(head.length + count * stride);
  bytes.set(head);
  const view = new DataView(bytes.buffer);
  let at = head.length;
  for (const chunk of chunks) {
    const { positions, colors, normals } = chunk;
    const n = positions.length / 3;
    for (let i = 0; i < n; i++) {
      view.setFloat32(at, positions[3 * i], true);
      view.setFloat32(at + 4, positions[3 * i + 1], true);
      view.setFloat32(at + 8, positions[3 * i + 2], true);
      at += 12;
      if (withNormals) {
        const [nx, ny, nz] = decodeOctahedral(normals![2 * i], normals![2 * i + 1]);
        view.setFloat32(at, nx, true);
        view.setFloat32(at + 4, ny, true);
        view.setFloat32(at + 8, nz, true);
        at += 12;
      }
      bytes[at] = colors[3 * i];
      bytes[at + 1] = colors[3 * i + 1];
      bytes[at + 2] = colors[3 * i + 2];
      at += 3;
    }
  }
  return bytes;
}
