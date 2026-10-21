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

/** The mesh of a run (`docs/dense-mvs.md` section 3.13): the arrays of `meshDenseCloudGpu`. */
export type DenseMeshData = {
  readonly run: number;
  /** x, y, z per vertex. */
  readonly positions: Float32Array;
  /** r, g, b per vertex, sRGB bytes. */
  readonly colors: Uint8Array;
  /** Three vertex indices per face, counter-clockwise seen from the side the normals point to. */
  readonly indices: Uint32Array;
};

/**
 * The mesh as a binary PLY: `format binary_little_endian 1.0`, a `vertex` element with x, y, z
 * (float) and red, green, blue (uchar), 15 B per vertex, and a `face` element with a
 * `vertex_indices` list (uchar count, uint indices), 13 B per face.
 */
export function denseMeshPly(
  mesh: Pick<DenseMeshData, 'positions' | 'colors' | 'indices'>,
): Uint8Array {
  const vertices = mesh.positions.length / 3,
    faces = mesh.indices.length / 3;
  const header =
    [
      'ply',
      'format binary_little_endian 1.0',
      'comment typecv dense mesh',
      'element vertex ' + vertices,
      'property float x',
      'property float y',
      'property float z',
      'property uchar red',
      'property uchar green',
      'property uchar blue',
      'element face ' + faces,
      'property list uchar uint vertex_indices',
      'end_header',
    ].join('\n') + '\n';
  const head = new TextEncoder().encode(header);
  const bytes = new Uint8Array(head.length + vertices * 15 + faces * 13);
  bytes.set(head);
  const view = new DataView(bytes.buffer);
  let at = head.length;
  for (let v = 0; v < vertices; v++) {
    for (let a = 0; a < 3; a++) view.setFloat32(at + 4 * a, mesh.positions[3 * v + a], true);
    for (let a = 0; a < 3; a++) bytes[at + 12 + a] = mesh.colors[3 * v + a];
    at += 15;
  }
  for (let f = 0; f < faces; f++) {
    bytes[at] = 3;
    for (let a = 0; a < 3; a++) view.setUint32(at + 1 + 4 * a, mesh.indices[3 * f + a], true);
    at += 13;
  }
  return bytes;
}

/** The side of a depth preview's block in pixels (`PREVIEW_STEP` of `src/gpu/mvs.ts`). */
export const PREVIEW_BLOCK = 4;

/** One `denseDepth` message (`docs/dense-mvs.md`, Amendment 5): a reference's depth preview. */
export type DensePreviewView = {
  readonly viewId: number;
  readonly stage: 'photometric' | 'geometric';
  /** The photometric iteration of an iteration preview; absent after a pass. */
  readonly iteration?: number;
  /** The grid, ⌈W / 4⌉ × ⌈H / 4⌉ blocks. */
  readonly width: number;
  readonly height: number;
  /** The mean depth of each block along the optical axis, in the model's units; 0 for none. */
  readonly depth: Float32Array;
  /** r, g, b per block, sRGB bytes. */
  readonly rgb: Uint8Array;
};

/**
 * The preview cloud of a run (Amendment 5, rule 4): the latest preview of each view. `done` turns
 * true once the fusion has posted every reference, and the viewer then lets the preview go.
 */
export type DensePreview = {
  readonly run: number;
  readonly views: ReadonlyMap<number, DensePreviewView>;
  readonly done: boolean;
};

/** The preview cloud's share of the point budget (Amendment 5, rule 5). */
export const PREVIEW_BUDGET_SHARE = 0.5;

/**
 * The 3D points of a depth preview (Amendment 5, rule 4): each block with a depth, back-projected
 * at its center pixel through the view's camera and pose (x_camera = R · X + t). `stride` keeps
 * every k-th block, by block index, so a block keeps its identity across the previews of a view.
 */
export function densePreviewPoints(
  view: {
    pose: { R: ArrayLike<number>; t: ArrayLike<number> };
    camera: { fx: number; fy: number; cx: number; cy: number };
  },
  preview: Pick<DensePreviewView, 'width' | 'height' | 'depth' | 'rgb'>,
  stride = 1,
): { blocks: Uint32Array; positions: Float32Array; colors: Uint8Array } {
  const { R, t } = view.pose;
  const { fx, fy, cx, cy } = view.camera;
  const blocks: number[] = [],
    positions: number[] = [],
    colors: number[] = [];
  const total = preview.width * preview.height;
  for (let b = 0; b < total; b += Math.max(1, stride)) {
    const z = preview.depth[b]!;
    if (!(z > 0) || !Number.isFinite(z)) continue;
    const bx = b % preview.width,
      by = (b - bx) / preview.width;
    const u = bx * PREVIEW_BLOCK + (PREVIEW_BLOCK - 1) / 2,
      v = by * PREVIEW_BLOCK + (PREVIEW_BLOCK - 1) / 2;
    // The camera point, less t, turned back by Rᵀ.
    const yx = (z * (u - cx)) / fx - t[0]!,
      yy = (z * (v - cy)) / fy - t[1]!,
      yz = z - t[2]!;
    blocks.push(b);
    positions.push(
      R[0]! * yx + R[3]! * yy + R[6]! * yz,
      R[1]! * yx + R[4]! * yy + R[7]! * yz,
      R[2]! * yx + R[5]! * yy + R[8]! * yz,
    );
    colors.push(preview.rgb[3 * b]!, preview.rgb[3 * b + 1]!, preview.rgb[3 * b + 2]!);
  }
  return {
    blocks: Uint32Array.from(blocks),
    positions: Float32Array.from(positions),
    colors: Uint8Array.from(colors),
  };
}

/** The block stride that keeps `blocks` preview blocks within `budget` points (rule 5). */
export function previewStride(blocks: number, budget: number): number {
  return budget > 0 && blocks > budget ? Math.ceil(blocks / budget) : 1;
}
