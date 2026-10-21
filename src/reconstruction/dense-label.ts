/**
 * The visibility labeling of the TSDF cloud (`docs/dense-mvs.md`, section 10.5): each extracted
 * point is labeled `surface` or `noise` by the exact minimum of an energy of unary terms and a
 * pairwise smoothness term, found by a min-cut (Boykov and Kolmogorov 2004). The surface points
 * then fall into connected components, and a component is kept when it has `minComponent` points
 * or a sparse landmark. This file is the one implementation: both tiers run it on the host, on the
 * points and the attributes of `tsdfExtract` (`TsdfPoints`).
 *
 * The energy, in units of one point:
 * - Graph: a node per point, and an edge between each point and each of its `k` nearest
 *   neighbours (one edge per pair), of weight w_ij = exp(−d² / 2σ²) · max(0, n_i · n_j). σ is the
 *   larger local spacing of the two points, the local spacing being the median distance of a point
 *   to its k nearest neighbours.
 * - The cost of `surface`: `confidenceCost` · (1 − c_i) + `visibilityCost` · v_i. c_i is the
 *   confidence term, the mean weight of the voxel's samples (the confidence of 10.3 times the
 *   cosine) over `confidenceScale`, at most 1. v_i is the visibility term, the voxel's free-space
 *   count over its weight, over `visibilityScale`, at most 1.
 * - The cost of `noise`: `noiseCost`, plus `landmarkBias` when a sparse landmark lies within its
 *   radius of the point (the sparse support term).
 * - The cost of two neighbours with different labels: `smoothness` · w_ij.
 *
 * The result does not depend on the order of the input: the points are sorted by the TSDF's order
 * (x, y, z), the neighbours of a point are ordered by distance then index, and the solver visits
 * the nodes in index order.
 */
import { buildPointGrid, eachInCell, neighbourCellSize } from './cloud-clean.ts';
import { decodeTsdfNormal, tsdfSamples, TSDF_WEIGHT_SCALE, type TsdfPoints } from './tsdf.ts';

export interface DenseLabelOptions {
  /** The nearest neighbours of a point in the graph. Default 8. */
  readonly k?: number;
  /** The cost of `surface` for a point with no confidence. Default 0.5 (the step 4 record). */
  readonly confidenceCost?: number;
  /** The mean sample weight at which the confidence term is 1. Default 0.1. */
  readonly confidenceScale?: number;
  /** The cost of `surface` for a point whose voxel is carved to `visibilityScale`. Default 1. */
  readonly visibilityCost?: number;
  /** The free-space count over the weight at which the visibility term is 1. Default 0.5. */
  readonly visibilityScale?: number;
  /** The cost of `noise`. Default 0.5. */
  readonly noiseCost?: number;
  /** The cost of `noise` added for a point with a sparse landmark within its radius. Default 2. */
  readonly landmarkBias?: number;
  /** The weight of the pairwise term. Default 1. */
  readonly smoothness?: number;
  /** The least points of a kept component without a landmark. Default 10. */
  readonly minComponent?: number;
  /** The least w_ij of an edge that joins two surface points into one component. Default 0.1. */
  readonly componentEdge?: number;
}

export const DENSE_LABEL_DEFAULTS = {
  k: 8,
  confidenceCost: 0.5,
  confidenceScale: 0.1,
  visibilityCost: 1,
  visibilityScale: 0.5,
  noiseCost: 0.5,
  landmarkBias: 2,
  smoothness: 1,
  minComponent: 10,
  componentEdge: 0.1,
} as const;

/** A sparse landmark and the radius within which it supports a dense point (10.2). */
export interface LabelLandmark {
  readonly xyz: ArrayLike<number>;
  readonly radius: number;
}

/** The points of the labeling: the TSDF's points and attributes. */
export type LabelInput = Pick<
  TsdfPoints,
  'count' | 'positions' | 'normals' | 'weights' | 'free' | 'tallies'
>;

export interface DenseLabelReport {
  readonly points: number;
  /** The points the cut labeled noise, by the larger of the two terms of their surface cost. */
  readonly cutByConfidence: number;
  readonly cutByVisibility: number;
  /** The surface points of the components that the component rule removed. */
  readonly removedByComponents: number;
  readonly kept: number;
  /** The points with a sparse landmark within its radius, and the landmarks with such a point. */
  readonly supported: number;
  readonly supportingLandmarks: number;
  /** The components of the surface points, and those kept, with the sizes of the largest kept. */
  readonly components: number;
  readonly keptComponents: number;
  readonly largestKept: readonly number[];
  /** The edges of the graph, and the flow of the cut (the energy less the labels' unary costs). */
  readonly edges: number;
  readonly flow: number;
}

export interface DenseLabelResult {
  /** 1 for a kept point, 0 for a removed one, in input order. */
  readonly keep: Uint8Array;
  readonly report: DenseLabelReport;
}

export function checkedLabelOptions(options: DenseLabelOptions = {}): Required<DenseLabelOptions> {
  const o = { ...DENSE_LABEL_DEFAULTS, ...options };
  if (!Number.isInteger(o.k) || o.k < 1 || o.k > 32)
    throw new RangeError('k must be an integer in [1, 32]');
  for (const name of [
    'confidenceCost',
    'confidenceScale',
    'visibilityCost',
    'visibilityScale',
    'noiseCost',
    'landmarkBias',
    'smoothness',
    'componentEdge',
  ] as const)
    if (!(o[name] >= 0 && Number.isFinite(o[name])))
      throw new RangeError(`${name} must be a finite number of at least 0`);
  if (!(o.confidenceScale > 0)) throw new RangeError('confidenceScale must be positive');
  if (!(o.visibilityScale > 0)) throw new RangeError('visibilityScale must be positive');
  if (!Number.isInteger(o.minComponent) || o.minComponent < 1)
    throw new RangeError('minComponent must be a positive integer');
  return o;
}

/** The farthest ring of grid cells a neighbour search visits. A point farther away is no neighbour. */
const MAX_RING = 6;

/**
 * The `k` nearest neighbours of every point (input indices, −1 where fewer exist) and their
 * squared distances, ascending, ties to the lower index.
 */
export function nearestNeighbours(
  positions: Float32Array,
  k: number,
): { readonly index: Int32Array; readonly d2: Float64Array } {
  const n = positions.length / 3;
  const index = new Int32Array(n * k).fill(-1),
    d2 = new Float64Array(n * k).fill(Infinity);
  if (n < 2) return { index, d2 };
  const grid = buildPointGrid(positions, neighbourCellSize(positions, k));
  const p = grid.points,
    order = grid.order;
  for (let s = 0; s < n; s++) {
    const i = order[s]!;
    const base = i * k;
    const px = p[3 * s]!,
      py = p[3 * s + 1]!,
      pz = p[3 * s + 2]!;
    const consider = (t: number) => {
      if (t === s) return;
      const j = order[t]!;
      const dx = p[3 * t]! - px,
        dy = p[3 * t + 1]! - py,
        dz = p[3 * t + 2]! - pz;
      const q = dx * dx + dy * dy + dz * dz;
      const last = base + k - 1;
      if (q > d2[last]! || (q === d2[last]! && j > index[last]!)) return;
      let m = k - 1;
      while (
        m > 0 &&
        (d2[base + m - 1]! > q || (d2[base + m - 1]! === q && index[base + m - 1]! > j))
      ) {
        d2[base + m] = d2[base + m - 1]!;
        index[base + m] = index[base + m - 1]!;
        m--;
      }
      d2[base + m] = q;
      index[base + m] = j;
    };
    const cx = grid.cells[3 * s]!,
      cy = grid.cells[3 * s + 1]!,
      cz = grid.cells[3 * s + 2]!;
    for (let r = 0; r <= MAX_RING; r++) {
      for (let dz = -r; dz <= r; dz++)
        for (let dy = -r; dy <= r; dy++)
          for (let dx = -r; dx <= r; dx++) {
            if (Math.max(Math.abs(dx), Math.abs(dy), Math.abs(dz)) !== r) continue;
            eachInCell(grid, cx + dx, cy + dy, cz + dz, consider);
          }
      const reach = r * grid.cellSize;
      if (d2[base + k - 1]! <= reach * reach) break;
    }
  }
  return { index, d2 };
}

/** The points with a landmark within the landmark's radius, and the landmarks with such a point. */
export function landmarkSupport(
  positions: Float32Array,
  landmarks: readonly LabelLandmark[],
): { readonly supported: Uint8Array; readonly landmarks: number } {
  const n = positions.length / 3;
  const supported = new Uint8Array(n);
  let count = 0;
  if (n === 0 || landmarks.length === 0) return { supported, landmarks: 0 };
  const widest = landmarks.reduce((a, l) => Math.max(a, l.radius), 0);
  if (!(widest > 0)) return { supported, landmarks: 0 };
  const grid = buildPointGrid(positions, widest);
  const h = grid.cellSize,
    origin = grid.origin,
    p = grid.points;
  for (const l of landmarks) {
    const x = l.xyz[0]!,
      y = l.xyz[1]!,
      z = l.xyz[2]!,
      r2 = l.radius * l.radius;
    const c = [x - origin[0], y - origin[1], z - origin[2]].map((v) => Math.floor(v / h));
    let found = false;
    for (let dz = -1; dz <= 1; dz++)
      for (let dy = -1; dy <= 1; dy++)
        for (let dx = -1; dx <= 1; dx++)
          eachInCell(grid, c[0]! + dx, c[1]! + dy, c[2]! + dz, (t) => {
            const q = (p[3 * t]! - x) ** 2 + (p[3 * t + 1]! - y) ** 2 + (p[3 * t + 2]! - z) ** 2;
            if (q <= r2) {
              supported[grid.order[t]!] = 1;
              found = true;
            }
          });
    if (found) count++;
  }
  return { supported, landmarks: count };
}

// ---- The min-cut --------------------------------------------------------------------------------

const TERMINAL = -2;
const NONE = -1;
const FREE = 0,
  SOURCE = 1,
  SINK = 2;

/**
 * A graph for the min-cut of Boykov and Kolmogorov (2004), with the terminal capacities of each
 * node and undirected edges of one capacity in each direction. `maxflow` labels each node
 * `SOURCE` (1) or `SINK` (2): a node is on the source side when the residual graph reaches it from
 * the source, so a node that neither terminal reaches is on the sink side.
 */
export class MinCutGraph {
  readonly n: number;
  /** The source and the sink capacity of each node. */
  private readonly source: Float64Array;
  private readonly sink: Float64Array;
  private readonly from: number[] = [];
  private readonly to: number[] = [];
  private readonly cap: number[] = [];

  constructor(n: number) {
    this.n = n;
    this.source = new Float64Array(n);
    this.sink = new Float64Array(n);
  }

  /** Adds capacity from the source to `i` and from `i` to the sink. */
  addTerminal(i: number, source: number, sink: number): void {
    this.source[i]! += source;
    this.sink[i]! += sink;
  }

  /** Adds an edge of capacity `capacity` in each direction between `i` and `j`. */
  addEdge(i: number, j: number, capacity: number): void {
    if (!(capacity > 0) || i === j) return;
    this.from.push(i);
    this.to.push(j);
    this.cap.push(capacity);
  }

  get edges(): number {
    return this.cap.length;
  }

  /** The maximum flow and the side of each node. */
  maxflow(): { readonly flow: number; readonly side: Uint8Array } {
    const n = this.n,
      m = this.cap.length;
    // Arcs in CSR order by tail: arc 2e is from[e] → to[e], 2e + 1 the reverse; sister = a ^ 1.
    const start = new Int32Array(n + 1);
    for (let e = 0; e < m; e++) {
      start[this.from[e]! + 1]!++;
      start[this.to[e]! + 1]!++;
    }
    for (let i = 0; i < n; i++) start[i + 1]! += start[i]!;
    const fill = start.slice(0, n);
    const arcAt = new Int32Array(2 * m);
    for (let e = 0; e < m; e++) {
      arcAt[fill[this.from[e]!]!++] = 2 * e;
      arcAt[fill[this.to[e]!]!++] = 2 * e + 1;
    }
    const head = new Int32Array(2 * m),
      rcap = new Float64Array(2 * m);
    for (let e = 0; e < m; e++) {
      head[2 * e] = this.to[e]!;
      head[2 * e + 1] = this.from[e]!;
      rcap[2 * e] = this.cap[e]!;
      rcap[2 * e + 1] = this.cap[e]!;
    }
    // The common part of a node's two terminal arcs flows at once, and the rest is its residual
    // terminal arc: to the source when positive, to the sink when negative.
    const tr = new Float64Array(n);
    let flow = 0;
    for (let i = 0; i < n; i++) {
      flow += Math.min(this.source[i]!, this.sink[i]!);
      tr[i] = this.source[i]! - this.sink[i]!;
    }
    const tree = new Uint8Array(n);
    const parent = new Int32Array(n).fill(NONE);
    // The time of the last check that a node reaches its terminal, and its distance then.
    const stamp = new Int32Array(n),
      dist = new Int32Array(n);
    let time = 0;
    const active: number[] = [];
    let activeAt = 0;
    const isActive = new Uint8Array(n);
    const activate = (i: number) => {
      if (!isActive[i]) {
        isActive[i] = 1;
        active.push(i);
      }
    };
    for (let i = 0; i < n; i++) {
      if (tr[i]! > 0) tree[i] = SOURCE;
      else if (tr[i]! < 0) tree[i] = SINK;
      else continue;
      parent[i] = TERMINAL;
      dist[i] = 1;
      activate(i);
    }
    let orphans: number[] = [];
    for (;;) {
      // Growth: the first active node with an arc to the other tree gives a path.
      let bridge = -1;
      while (bridge < 0 && activeAt < active.length) {
        const i = active[activeAt]!;
        if (tree[i] === FREE) {
          isActive[i] = 0;
          activeAt++;
          continue;
        }
        for (let s = start[i]!; s < start[i + 1]!; s++) {
          const a = arcAt[s]!,
            j = head[a]!;
          const residual = tree[i] === SOURCE ? rcap[a]! : rcap[a ^ 1]!;
          if (!(residual > 0)) continue;
          if (tree[j] === FREE) {
            tree[j] = tree[i]!;
            parent[j] = a ^ 1;
            stamp[j] = stamp[i]!;
            dist[j] = dist[i]! + 1;
            activate(j);
          } else if (tree[j] !== tree[i]) {
            bridge = tree[i] === SOURCE ? a : a ^ 1;
            break;
          } else if (stamp[j]! <= stamp[i]! && dist[j]! > dist[i]!) {
            // A shorter path to the terminal (the heuristic of the paper).
            parent[j] = a ^ 1;
            stamp[j] = stamp[i]!;
            dist[j] = dist[i]! + 1;
          }
        }
        if (bridge < 0) {
          isActive[i] = 0;
          activeAt++;
        }
      }
      if (bridge < 0) break;
      if (activeAt > 4096 && activeAt * 2 > active.length) {
        active.splice(0, activeAt);
        activeAt = 0;
      }
      time++;
      // Augmentation along source … tail(bridge) → head(bridge) … sink.
      const tail = head[bridge ^ 1]!,
        tip = head[bridge]!;
      let bottleneck = rcap[bridge]!;
      for (let u = tail; ;) {
        const pa = parent[u]!;
        if (pa === TERMINAL) {
          bottleneck = Math.min(bottleneck, tr[u]!);
          break;
        }
        bottleneck = Math.min(bottleneck, rcap[pa ^ 1]!);
        u = head[pa]!;
      }
      for (let u = tip; ;) {
        const pa = parent[u]!;
        if (pa === TERMINAL) {
          bottleneck = Math.min(bottleneck, -tr[u]!);
          break;
        }
        bottleneck = Math.min(bottleneck, rcap[pa]!);
        u = head[pa]!;
      }
      rcap[bridge]! -= bottleneck;
      rcap[bridge ^ 1]! += bottleneck;
      for (let u = tail; ;) {
        const pa = parent[u]!;
        if (pa === TERMINAL) {
          tr[u]! -= bottleneck;
          if (!(tr[u]! > 0)) {
            tr[u] = Math.max(tr[u]!, 0);
            parent[u] = NONE;
            orphans.push(u);
          }
          break;
        }
        rcap[pa ^ 1]! -= bottleneck;
        rcap[pa]! += bottleneck;
        if (!(rcap[pa ^ 1]! > 0)) {
          parent[u] = NONE;
          orphans.push(u);
        }
        u = head[pa]!;
      }
      for (let u = tip; ;) {
        const pa = parent[u]!;
        if (pa === TERMINAL) {
          tr[u]! += bottleneck;
          if (!(tr[u]! < 0)) {
            tr[u] = Math.min(tr[u]!, 0);
            parent[u] = NONE;
            orphans.push(u);
          }
          break;
        }
        rcap[pa]! -= bottleneck;
        rcap[pa ^ 1]! += bottleneck;
        if (!(rcap[pa]! > 0)) {
          parent[u] = NONE;
          orphans.push(u);
        }
        u = head[pa]!;
      }
      flow += bottleneck;
      // Adoption: each orphan finds a parent in its tree that still reaches the terminal, or frees.
      for (let at = 0; at < orphans.length; at++) {
        const u = orphans[at]!;
        const side = tree[u]!;
        let best = NONE,
          bestDist = Infinity;
        for (let s = start[u]!; s < start[u + 1]!; s++) {
          const a = arcAt[s]!,
            j = head[a]!;
          if (tree[j] !== side) continue;
          const residual = side === SOURCE ? rcap[a ^ 1]! : rcap[a]!;
          if (!(residual > 0)) continue;
          // Walk to the root: valid when it ends at the terminal.
          let d = 0,
            w = j,
            valid = false;
          for (;;) {
            if (stamp[w] === time) {
              d += dist[w]!;
              valid = true;
              break;
            }
            const pw = parent[w]!;
            d++;
            if (pw === TERMINAL) {
              stamp[w] = time;
              dist[w] = 1;
              valid = true;
              break;
            }
            if (pw === NONE) break;
            w = head[pw]!;
          }
          if (!valid) continue;
          if (d < bestDist) {
            best = a;
            bestDist = d;
          }
          // Mark the path with its distances for the next walks of this adoption.
          for (let x = j, dx = d; stamp[x] !== time; x = head[parent[x]!]!) {
            stamp[x] = time;
            dist[x] = dx--;
          }
        }
        if (best !== NONE) {
          parent[u] = best;
          stamp[u] = time;
          dist[u] = bestDist + 1;
          continue;
        }
        // No parent: the neighbours become active, and its children orphans.
        for (let s = start[u]!; s < start[u + 1]!; s++) {
          const a = arcAt[s]!,
            j = head[a]!;
          if (tree[j] !== side) continue;
          const residual = side === SOURCE ? rcap[a ^ 1]! : rcap[a]!;
          if (residual > 0) activate(j);
          const pj = parent[j]!;
          if (pj >= 0 && head[pj] === u) {
            parent[j] = NONE;
            orphans.push(j);
          }
        }
        tree[u] = FREE;
        parent[u] = NONE;
      }
      orphans = [];
    }
    const side = new Uint8Array(n);
    for (let i = 0; i < n; i++) side[i] = tree[i] === SOURCE ? SOURCE : SINK;
    return { flow, side };
  }
}

// ---- The labeling -------------------------------------------------------------------------------

/**
 * The labeling of section 10.5 of the points of a TSDF fusion, with the landmarks that support
 * them. `keep` marks the points of the kept surface components.
 */
export function labelDenseCloud(
  points: LabelInput,
  landmarks: readonly LabelLandmark[] = [],
  options: DenseLabelOptions = {},
): DenseLabelResult {
  const o = checkedLabelOptions(options);
  const n = points.count;
  const P = points.positions.subarray(0, 3 * n);
  const keep = new Uint8Array(n);
  const empty: DenseLabelReport = {
    points: n,
    cutByConfidence: 0,
    cutByVisibility: 0,
    removedByComponents: 0,
    kept: 0,
    supported: 0,
    supportingLandmarks: 0,
    components: 0,
    keptComponents: 0,
    largestKept: [],
    edges: 0,
    flow: 0,
  };
  if (n === 0) return { keep, report: empty };
  const k = Math.min(o.k, Math.max(1, n - 1));
  const { index, d2 } = nearestNeighbours(P, k);
  // The local spacing: the median distance to the k nearest neighbours.
  const spacing = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let m = 0;
    while (m < k && index[i * k + m]! >= 0) m++;
    spacing[i] = m ? Math.sqrt(d2[i * k + ((m - 1) >> 1)]!) : 0;
  }
  const normals = Array.from(points.normals.subarray(0, n), (w) => decodeTsdfNormal(w));
  const support = landmarkSupport(P, landmarks);
  const graph = new MinCutGraph(n);
  const confidenceCost = new Float64Array(n),
    visibilityCost = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const weight = points.weights[i]! / TSDF_WEIGHT_SCALE;
    const samples = Math.max(1, tsdfSamples(points.tallies[i]!));
    const c = Math.min(1, weight / samples / o.confidenceScale);
    const v = weight > 0 ? Math.min(1, points.free[i]! / weight / o.visibilityScale) : 1;
    confidenceCost[i] = o.confidenceCost * (1 - c);
    visibilityCost[i] = o.visibilityCost * v;
    const noise = o.noiseCost + (support.supported[i] ? o.landmarkBias : 0);
    graph.addTerminal(i, noise, confidenceCost[i]! + visibilityCost[i]!);
  }
  // The edges: one per pair of neighbours, from the lower index.
  for (let i = 0; i < n; i++)
    for (let m = 0; m < k; m++) {
      const j = index[i * k + m]!;
      if (j < 0) break;
      // A pair that are each other's neighbours is added once, from the lower index.
      if (j < i) {
        let mutual = false;
        for (let q = 0; q < k; q++) if (index[j * k + q] === i) mutual = true;
        if (mutual) continue;
      }
      const w = edgeWeight(
        d2[i * k + m]!,
        Math.max(spacing[i]!, spacing[j]!),
        normals[i]!,
        normals[j]!,
      );
      graph.addEdge(i, j, o.smoothness * w);
    }
  const { flow, side } = graph.maxflow();
  let cutByConfidence = 0,
    cutByVisibility = 0;
  for (let i = 0; i < n; i++)
    if (side[i] !== 1) {
      if (confidenceCost[i]! >= visibilityCost[i]!) cutByConfidence++;
      else cutByVisibility++;
    }
  // The components of the surface points under the edges of at least `componentEdge`.
  const root = new Int32Array(n);
  for (let i = 0; i < n; i++) root[i] = i;
  const find = (i: number) => {
    while (root[i] !== i) {
      root[i] = root[root[i]!]!;
      i = root[i]!;
    }
    return i;
  };
  for (let i = 0; i < n; i++) {
    if (side[i] !== 1) continue;
    for (let m = 0; m < k; m++) {
      const j = index[i * k + m]!;
      if (j < 0) break;
      if (side[j] !== 1) continue;
      const w = edgeWeight(
        d2[i * k + m]!,
        Math.max(spacing[i]!, spacing[j]!),
        normals[i]!,
        normals[j]!,
      );
      if (!(w >= o.componentEdge)) continue;
      const a = find(i),
        b = find(j);
      if (a !== b) root[Math.max(a, b)] = Math.min(a, b);
    }
  }
  const size = new Int32Array(n),
    anchored = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    if (side[i] !== 1) continue;
    const r = find(i);
    size[r]!++;
    if (support.supported[i]) anchored[r] = 1;
  }
  const keptSizes: number[] = [];
  let components = 0,
    removedByComponents = 0,
    kept = 0;
  for (let i = 0; i < n; i++) {
    if (side[i] !== 1 || root[i] !== i) continue;
    components++;
    if (size[i]! >= o.minComponent || anchored[i]) keptSizes.push(size[i]!);
  }
  for (let i = 0; i < n; i++) {
    if (side[i] !== 1) continue;
    const r = find(i);
    if (size[r]! >= o.minComponent || anchored[r]) {
      keep[i] = 1;
      kept++;
    } else removedByComponents++;
  }
  let supported = 0;
  for (let i = 0; i < n; i++) supported += support.supported[i]!;
  keptSizes.sort((a, b) => b - a);
  return {
    keep,
    report: {
      points: n,
      cutByConfidence,
      cutByVisibility,
      removedByComponents,
      kept,
      supported,
      supportingLandmarks: support.landmarks,
      components,
      keptComponents: keptSizes.length,
      largestKept: keptSizes.slice(0, 8),
      edges: graph.edges,
      flow,
    },
  };
}

/** w_ij = exp(−d² / 2σ²) · max(0, n_i · n_j), 0 when σ is 0. */
function edgeWeight(d2: number, sigma: number, a: readonly number[], b: readonly number[]): number {
  if (!(sigma > 0)) return 0;
  const cos = a[0]! * b[0]! + a[1]! * b[1]! + a[2]! * b[2]!;
  return Math.exp(-d2 / (2 * sigma * sigma)) * Math.max(0, cos);
}
