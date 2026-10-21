/**
 * The view graph of a multi-video run (docs/multi-video.md, sections 3.3 to 3.5 and 4.1).
 *
 * Nodes are keyframes. An edge joins two keyframes that a geometric check accepted. The graph is
 * built from content only: the global descriptor decides the candidates, the canonical `viewId`
 * (2.4) breaks every tie, and the input order decides nothing.
 */
import type { Camera } from '../geometry/camera.ts';
import type { ImagePairGeometry } from '../geometry/image-pair.ts';
import {
  executeReconstructionJob,
  incrementalControls,
  verifyPairs,
  type IncrementalOptions,
  type PairVerification,
  type PairVerificationJob,
} from './mapper.ts';
import type { ImageFeatures, PairMatches } from './types.ts';

/** The RootSIFT dimension. */
export const DESCRIPTOR_DIMENSIONS = 128;
/** The 8-bit scale of 3.5: `round(512 * v)`, clamped to 255. */
export const DESCRIPTOR_SCALE = 512;

export interface ViewGraphOptions {
  /** Temporal neighbours on each side of a keyframe, in its own video. Default 3. */
  readonly temporalNeighbours?: number;
  /** Retrieved neighbours of each keyframe. Default 10. */
  readonly retrievalK?: number;
  /** Retrieval skips the same video within this many keyframes in time. Default 10. */
  readonly exclusionWindow?: number;
  /** Best candidate pairs kept for each pair of videos. Default 3. */
  readonly reservePerVideoPair?: number;
  /** Query expansion after the first round (3.3, guard 2). Off by default. */
  readonly expansion?: boolean;
  /** Neighbours of one end tried against the other end in the expansion. Default 3. */
  readonly expansionK?: number;
  /** Ratio test of the matching. Default 0.8. */
  readonly ratio?: number;
  /** A pair with fewer matches is dropped before verification. Default 30. */
  readonly minMatches?: number;
  /** A pair with fewer inliers is not an edge. Default 25. */
  readonly minEdgeInliers?: number;
  /** A component with fewer keyframes is not reconstructed. Default 5. */
  readonly minComponentViews?: number;
  /** Share of a video in one component that makes it connected. Default 0.5. */
  readonly minVideoShare?: number;
  /** `uint8` stores 3.5's 8-bit descriptors. `float32` is the measurement baseline. */
  readonly descriptorStorage?: 'uint8' | 'float32';
  /** Options of the pair verification (`verifyPairs` controls), such as `seed`. */
  readonly verification?: IncrementalOptions;
  /**
   * Pairs verified in one backend call (`executeMany`). Default 48. A larger value holds more
   * pixel rows in memory; it does not change the result.
   */
  readonly verifyBatch?: number;
  /** Called as the build advances (Amendment 8). It never changes a result. */
  readonly onProgress?: (event: ViewGraphProgress) => void;
}

/**
 * One progress event of `buildViewGraph`. `candidates` comes once with the pairs of the first
 * round. `matching` and `verification` carry the pairs done and the pairs of the build so far:
 * the total grows once if the expansion round adds pairs. `components` comes at the end.
 */
export type ViewGraphProgress =
  | { readonly stage: 'candidates'; readonly total: number }
  | {
      readonly stage: 'matching' | 'verification';
      readonly done: number;
      readonly total: number;
    }
  | { readonly stage: 'components'; readonly edges: number };

/** One keyframe as the view graph reads it. */
export interface ViewGraphKeyframe {
  /** The canonical id of 2.4. */
  readonly viewId: number;
  readonly videoId: string;
  readonly frameKey: string;
  /** Time in the video, in seconds. */
  readonly time: number;
  readonly camera: Camera;
  /** Packed x, y pixels, one pair for each feature. */
  readonly keypoints: Float64Array;
  /** RootSIFT, row-major, `keypoints.length / 2` rows of 128 values in [0, 1]. */
  readonly descriptors: Float32Array;
  /**
   * The global descriptor (3.2): L2-normalized, so the cosine similarity of two keyframes is the
   * dot product. Step 3 produces it (`globalDescriptorReference`: VLAD of 4,096 values by default, or the bag of words of 512). This step takes it as input.
   */
  readonly global: Float32Array;
}

/** A keyframe as stored on the host (3.5). */
export interface StoredView {
  readonly viewId: number;
  readonly count: number;
  /** The descriptors as float32, decoded from the store when it is 8-bit. */
  floatDescriptors(): Float32Array;
}

export interface MatchedPair {
  readonly matches: Uint32Array;
  readonly distances: Float64Array;
}

/** The two numerical steps of the build. CPU and GPU backends implement it. */
export interface ViewGraphBackend {
  /** Match `first` against each of `others`. One upload of `first` serves the whole batch. */
  match(
    first: StoredView,
    others: readonly StoredView[],
    options: { readonly ratio: number },
  ): Promise<MatchedPair[]>;
  /** Run one `verifyPair` job of the mapper. */
  execute(job: PairVerificationJob): unknown;
  /**
   * Run many `verifyPair` jobs, which share no input, and return the results in the order of the
   * jobs. Optional: without it the graph runs `execute` on each job in turn. A backend gives the
   * same result for a job in either form.
   */
  executeMany?(jobs: readonly PairVerificationJob[]): Promise<unknown[]>;
}

export type CandidateSource = 'temporal' | 'retrieval' | 'reserve' | 'expansion';

export interface ViewGraphEdge {
  /** `first < second`, both canonical view ids. */
  readonly first: number;
  readonly second: number;
  readonly crossVideo: boolean;
  /** The inlier matches (consecutive feature index pairs), as `verifyPairs` keeps them. */
  readonly matches: Uint32Array;
  readonly distances: Float64Array;
  /** The weight of the edge. */
  readonly inliers: number;
  readonly homographyInliers: number;
  readonly medianAngleDeg: number;
  readonly seedEligible: boolean;
  /** The row `verifyPairs` wrote for this pair. The mapper reuses it (`preVerified`). */
  readonly diagnostic: PairVerification;
  /** The relative pose and triangulation. Present when the pair can seed a map only. */
  readonly geometry?: ImagePairGeometry;
  readonly source: CandidateSource;
}

export type VideoTag = 'connected' | 'not connected' | 'alone';

export interface ViewGraphComponent {
  /** The rank of 4.1 rule 4: by size, then by the lowest view id. */
  readonly index: number;
  readonly viewIds: readonly number[];
  readonly videoIds: readonly string[];
  /** False when the component has fewer than `minComponentViews` keyframes. */
  readonly reconstructed: boolean;
}

export interface ViewGraphVideo {
  readonly videoId: string;
  readonly tag: VideoTag;
  /** The component that holds most of its keyframes, or -1 without keyframes. */
  readonly component: number;
  /** The share of its keyframes in that component. */
  readonly share: number;
}

export interface ViewGraphStats {
  readonly keyframes: number;
  readonly videos: number;
  /** N (N - 1) / 2. */
  readonly exhaustivePairs: number;
  readonly candidatePairs: number;
  readonly temporalPairs: number;
  readonly retrievalPairs: number;
  readonly reservePairs: number;
  readonly expansionPairs: number;
  /** Pairs with at least `minMatches` matches, which reach verification. */
  readonly verifiedPairs: number;
  readonly edges: number;
  readonly crossVideoEdges: number;
}

/** Wall time of the stages of one build, in milliseconds (6.4, `timing.stages`). */
export interface ViewGraphTiming {
  /** Candidate selection: the retrieval of 3.3. */
  readonly retrievalMs: number;
  /** The time inside `backend.match`. */
  readonly pairMatchingMs: number;
  /** The time inside `verifyPairs`, with the backend's `execute`. */
  readonly pairVerificationMs: number;
  /** The union-find and the tags of 4.1. */
  readonly componentsMs: number;
}

export interface ViewGraph {
  readonly edges: readonly ViewGraphEdge[];
  readonly components: readonly ViewGraphComponent[];
  readonly videos: readonly ViewGraphVideo[];
  readonly stats: ViewGraphStats;
  readonly timing: ViewGraphTiming;
  /** Every candidate pair of the build, as [lower, higher] view ids in sorted order. */
  readonly candidates: readonly (readonly [number, number])[];
}

interface Resolved {
  temporalNeighbours: number;
  retrievalK: number;
  exclusionWindow: number;
  reservePerVideoPair: number;
  expansion: boolean;
  expansionK: number;
  ratio: number;
  minMatches: number;
  minEdgeInliers: number;
  minComponentViews: number;
  minVideoShare: number;
  descriptorStorage: 'uint8' | 'float32';
}

function resolve(options: ViewGraphOptions): Resolved {
  const r: Resolved = {
    temporalNeighbours: options.temporalNeighbours ?? 3,
    retrievalK: options.retrievalK ?? 10,
    exclusionWindow: options.exclusionWindow ?? 10,
    reservePerVideoPair: options.reservePerVideoPair ?? 3,
    expansion: options.expansion ?? false,
    expansionK: options.expansionK ?? 3,
    ratio: options.ratio ?? 0.8,
    minMatches: options.minMatches ?? 30,
    minEdgeInliers: options.minEdgeInliers ?? 25,
    minComponentViews: options.minComponentViews ?? 5,
    minVideoShare: options.minVideoShare ?? 0.5,
    descriptorStorage: options.descriptorStorage ?? 'uint8',
  };
  const counts = [
    r.temporalNeighbours,
    r.retrievalK,
    r.exclusionWindow,
    r.reservePerVideoPair,
    r.expansionK,
    r.minMatches,
    r.minEdgeInliers,
    r.minComponentViews,
  ];
  if (
    counts.some((v) => !Number.isSafeInteger(v) || v < 0) ||
    !(r.ratio > 0 && r.ratio < 1) ||
    !(r.minVideoShare > 0 && r.minVideoShare <= 1) ||
    (r.descriptorStorage !== 'uint8' && r.descriptorStorage !== 'float32')
  )
    throw new RangeError('Invalid view graph options');
  return r;
}

/** 3.5: `round(512 v)`, clamped to 255. */
export function quantizeDescriptors(values: Float32Array): Uint8Array {
  const out = new Uint8Array(values.length);
  for (let i = 0; i < values.length; i++)
    out[i] = Math.min(255, Math.max(0, Math.round(DESCRIPTOR_SCALE * values[i]!)));
  return out;
}

export function dequantizeDescriptors(values: Uint8Array): Float32Array {
  const out = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) out[i] = values[i]! / DESCRIPTOR_SCALE;
  return out;
}

/** The host store of 3.5. It keeps the descriptors of every keyframe, 8-bit by default. */
export class DescriptorStore {
  readonly storage: 'uint8' | 'float32';
  private readonly rows = new Map<number, Uint8Array | Float32Array>();
  constructor(storage: 'uint8' | 'float32' = 'uint8') {
    this.storage = storage;
  }
  add(viewId: number, descriptors: Float32Array): void {
    if (this.rows.has(viewId)) throw new RangeError('Duplicate view id in the descriptor store');
    if (descriptors.length % DESCRIPTOR_DIMENSIONS)
      throw new RangeError('Descriptors must be rows of 128 values');
    this.rows.set(
      viewId,
      this.storage === 'uint8' ? quantizeDescriptors(descriptors) : descriptors.slice(),
    );
  }
  view(viewId: number): StoredView {
    const rows = this.rows.get(viewId);
    if (!rows) throw new RangeError('Unknown view id in the descriptor store');
    return {
      viewId,
      count: rows.length / DESCRIPTOR_DIMENSIONS,
      floatDescriptors: () => (rows instanceof Uint8Array ? dequantizeDescriptors(rows) : rows),
    };
  }
  /** Bytes held on the host. */
  get bytes(): number {
    let total = 0;
    for (const rows of this.rows.values()) total += rows.byteLength;
    return total;
  }
}

/**
 * Brute-force mutual ratio matching in float64 arithmetic: the CPU reference of
 * `matchFloatDescriptorsGpu` (ratio test in both directions, mutual nearest, ties to the lower
 * index, matches in source order).
 */
export function matchDescriptorsCpu(
  first: Float32Array,
  second: Float32Array,
  ratio = 0.8,
): MatchedPair {
  const d = DESCRIPTOR_DIMENSIONS,
    n = first.length / d,
    m = second.length / d;
  const dist = new Float64Array(n * m);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < m; j++) {
      let s = 0;
      for (let k = 0; k < d; k++) {
        const t = first[i * d + k]! - second[j * d + k]!;
        s += t * t;
      }
      dist[i * m + j] = s;
    }
  // Nearest and second nearest, in both directions, from the one matrix. Ties go to the lower index.
  const near = (count: number, other: number, at: (x: number, y: number) => number) => {
    const best = new Int32Array(count).fill(-1),
      d1 = new Float64Array(count).fill(Infinity),
      d2 = new Float64Array(count).fill(Infinity);
    for (let x = 0; x < count; x++)
      for (let y = 0; y < other; y++) {
        const v = at(x, y);
        if (v < d1[x]!) {
          d2[x] = d1[x]!;
          d1[x] = v;
          best[x] = y;
        } else if (v < d2[x]!) d2[x] = v;
      }
    return { best, d1, d2 };
  };
  const forward = near(n, m, (i, j) => dist[i * m + j]!),
    backward = near(m, n, (j, i) => dist[i * m + j]!);
  const matches: number[] = [],
    distances: number[] = [];
  const r2 = ratio * ratio;
  for (let i = 0; i < n; i++) {
    const j = forward.best[i]!;
    if (j < 0 || backward.best[j] !== i) continue;
    if (forward.d1[i]! >= r2 * forward.d2[i]!) continue;
    if (backward.d1[j]! >= r2 * backward.d2[j]!) continue;
    matches.push(i, j);
    distances.push(Math.sqrt(forward.d1[i]!));
  }
  return { matches: Uint32Array.from(matches), distances: Float64Array.from(distances) };
}

/** The CPU backend: float64 matching and the CPU `verifyPair`. For tests and small runs. */
export function cpuViewGraphBackend(): ViewGraphBackend {
  return {
    async match(first, others, options) {
      const a = first.floatDescriptors();
      return others.map((o) =>
        a.length && o.count ? matchDescriptorsCpu(a, o.floatDescriptors(), options.ratio) : empty(),
      );
    },
    execute: (job) => executeReconstructionJob(job),
  };
}

function empty(): MatchedPair {
  return { matches: new Uint32Array(0), distances: new Float64Array(0) };
}

function dot(a: Float32Array, b: Float32Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i]! * b[i]!;
  return s;
}

const pairKey = (a: number, b: number) => (a < b ? `${a}:${b}` : `${b}:${a}`);

export interface CandidateSet {
  /** Sorted by (first, second). `first < second`. */
  readonly pairs: readonly (readonly [number, number])[];
  readonly source: ReadonlyMap<string, CandidateSource>;
  /** For each keyframe, its retrieved neighbours, best first (the input of the expansion). */
  readonly retrieved: ReadonlyMap<number, readonly number[]>;
}

/**
 * 3.3: temporal neighbours, the top-k retrieved neighbours outside the exclusion window, and the
 * best pairs of each pair of videos. `similarity(i, j)` takes two keyframe indices.
 */
export function selectCandidates(
  keyframes: readonly ViewGraphKeyframe[],
  options: ViewGraphOptions = {},
  similarity: (a: ViewGraphKeyframe, b: ViewGraphKeyframe) => number = (a, b) =>
    dot(a.global, b.global),
): CandidateSet {
  const o = resolve(options);
  const frames = [...keyframes].sort((a, b) => a.viewId - b.viewId);
  const n = frames.length;
  // Rank in time inside the video, ties by view id.
  const rank = new Map<number, number>(),
    byVideo = new Map<string, ViewGraphKeyframe[]>();
  for (const f of frames) {
    const list = byVideo.get(f.videoId) ?? [];
    list.push(f);
    byVideo.set(f.videoId, list);
  }
  const sequences = new Map<string, ViewGraphKeyframe[]>();
  for (const [video, list] of byVideo) {
    const sorted = [...list].sort((a, b) => a.time - b.time || a.viewId - b.viewId);
    sorted.forEach((f, i) => rank.set(f.viewId, i));
    sequences.set(video, sorted);
  }
  const source = new Map<string, CandidateSource>();
  const add = (a: ViewGraphKeyframe, b: ViewGraphKeyframe, why: CandidateSource) => {
    if (a.viewId === b.viewId || a.frameKey === b.frameKey) return;
    const key = pairKey(a.viewId, b.viewId);
    if (!source.has(key)) source.set(key, why);
  };
  for (const sequence of sequences.values())
    for (let i = 0; i < sequence.length; i++)
      for (let d = 1; d <= o.temporalNeighbours && i + d < sequence.length; d++)
        add(sequence[i]!, sequence[i + d]!, 'temporal');
  const retrieved = new Map<number, number[]>();
  const sim = (i: number, j: number) => similarity(frames[i]!, frames[j]!);
  // The cross-video pairs for the reserve, per ordered pair of video ids.
  const reserve = new Map<string, { s: number; a: number; b: number }[]>();
  const keepBest = (list: { s: number; a: number; b: number }[], item: (typeof list)[number]) => {
    list.push(item);
    list.sort((x, y) => y.s - x.s || x.a - y.a || x.b - y.b);
    if (list.length > o.reservePerVideoPair) list.pop();
  };
  for (let i = 0; i < n; i++) {
    const a = frames[i]!;
    const scored: { s: number; j: number }[] = [];
    for (let j = 0; j < n; j++) {
      if (j === i) continue;
      const b = frames[j]!;
      const s = sim(i, j);
      if (a.videoId !== b.videoId) {
        if (a.viewId < b.viewId && a.frameKey !== b.frameKey && o.reservePerVideoPair > 0) {
          const key =
            a.videoId < b.videoId ? `${a.videoId}|${b.videoId}` : `${b.videoId}|${a.videoId}`;
          const list = reserve.get(key) ?? [];
          reserve.set(key, list);
          keepBest(list, { s, a: a.viewId, b: b.viewId });
        }
      } else if (Math.abs(rank.get(a.viewId)! - rank.get(b.viewId)!) <= o.exclusionWindow) continue;
      scored.push({ s, j });
    }
    scored.sort((x, y) => y.s - x.s || frames[x.j]!.viewId - frames[y.j]!.viewId);
    const top = scored.slice(0, o.retrievalK);
    retrieved.set(
      a.viewId,
      top.map((t) => frames[t.j]!.viewId),
    );
    for (const t of top) add(a, frames[t.j]!, 'retrieval');
  }
  const byViewId = new Map(frames.map((f) => [f.viewId, f]));
  for (const list of reserve.values())
    for (const item of list) add(byViewId.get(item.a)!, byViewId.get(item.b)!, 'reserve');
  const pairs = [...source.keys()]
    .map((k) => k.split(':').map(Number) as [number, number])
    .sort((x, y) => x[0] - y[0] || x[1] - y[1]);
  return { pairs, source, retrieved };
}

class UnionFind {
  private readonly parent = new Map<number, number>();
  add(x: number): void {
    if (!this.parent.has(x)) this.parent.set(x, x);
  }
  find(x: number): number {
    let r = x;
    while (this.parent.get(r) !== r) r = this.parent.get(r)!;
    while (this.parent.get(x) !== r) {
      const next = this.parent.get(x)!;
      this.parent.set(x, r);
      x = next;
    }
    return r;
  }
  union(a: number, b: number): void {
    const ra = this.find(a),
      rb = this.find(b);
    // The lower view id stays the root, so the structure does not depend on the edge order.
    if (ra !== rb) this.parent.set(Math.max(ra, rb), Math.min(ra, rb));
  }
}

/** 4.1: union-find over the edges, the size rule, and the tags of the videos. */
export function connectedComponents(
  keyframes: readonly Pick<ViewGraphKeyframe, 'viewId' | 'videoId'>[],
  edges: readonly Pick<ViewGraphEdge, 'first' | 'second'>[],
  options: Pick<ViewGraphOptions, 'minComponentViews' | 'minVideoShare'> = {},
): { components: ViewGraphComponent[]; videos: ViewGraphVideo[] } {
  const minViews = options.minComponentViews ?? 5,
    minShare = options.minVideoShare ?? 0.5;
  const uf = new UnionFind();
  for (const k of keyframes) uf.add(k.viewId);
  for (const e of edges) uf.union(e.first, e.second);
  const groups = new Map<number, number[]>();
  for (const k of keyframes) {
    const root = uf.find(k.viewId);
    groups.set(root, [...(groups.get(root) ?? []), k.viewId]);
  }
  const videoOf = new Map(keyframes.map((k) => [k.viewId, k.videoId]));
  const ordered = [...groups.values()]
    .map((ids) => ids.sort((a, b) => a - b))
    .sort((a, b) => b.length - a.length || a[0]! - b[0]!);
  const components: ViewGraphComponent[] = ordered.map((viewIds, index) => ({
    index,
    viewIds,
    videoIds: [...new Set(viewIds.map((v) => videoOf.get(v)!))].sort(),
    reconstructed: viewIds.length >= minViews,
  }));
  const componentOf = new Map<number, number>();
  for (const c of components) for (const v of c.viewIds) componentOf.set(v, c.index);
  const perVideo = new Map<string, Map<number, number>>();
  for (const k of keyframes) {
    const counts = perVideo.get(k.videoId) ?? new Map<number, number>();
    perVideo.set(k.videoId, counts);
    const c = componentOf.get(k.viewId)!;
    counts.set(c, (counts.get(c) ?? 0) + 1);
  }
  const videos: ViewGraphVideo[] = [...perVideo.entries()]
    .sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0))
    .map(([videoId, counts]) => {
      let best = -1,
        bestCount = 0,
        total = 0;
      for (const [c, count] of [...counts.entries()].sort((a, b) => a[0] - b[0])) {
        total += count;
        if (count > bestCount) {
          best = c;
          bestCount = count;
        }
      }
      const share = total ? bestCount / total : 0,
        component = components[best];
      let tag: VideoTag = 'not connected';
      if (component?.reconstructed && share >= minShare)
        tag = component.videoIds.length >= 2 ? 'connected' : 'alone';
      return { videoId, tag, component: best, share };
    });
  return { components, videos };
}

/**
 * Build the view graph: candidates, matching, verification, components. `backend` runs the
 * numerical steps (`cpuViewGraphBackend`, or `gpuViewGraphBackend` of `typecv/gpu`).
 */
export async function buildViewGraph(
  keyframes: readonly ViewGraphKeyframe[],
  backend: ViewGraphBackend,
  options: ViewGraphOptions = {},
  similarity?: (a: ViewGraphKeyframe, b: ViewGraphKeyframe) => number,
): Promise<ViewGraph> {
  const o = resolve(options);
  const frames = [...keyframes].sort((a, b) => a.viewId - b.viewId);
  const byViewId = new Map(frames.map((f) => [f.viewId, f]));
  if (byViewId.size !== frames.length) throw new RangeError('View ids must be unique');
  for (const f of frames) {
    if (f.keypoints.length * 64 !== f.descriptors.length)
      throw new RangeError('Each keypoint needs one descriptor row of 128 values');
    if (f.global.length !== frames[0]!.global.length)
      throw new RangeError('Global descriptors must have one length');
  }
  const store = new DescriptorStore(o.descriptorStorage);
  for (const f of frames) store.add(f.viewId, f.descriptors);
  const retrievalStarted = performance.now();
  const candidates = selectCandidates(frames, o, similarity);
  const retrievalMs = performance.now() - retrievalStarted;
  let pairMatchingMs = 0,
    pairVerificationMs = 0;
  const images: ImageFeatures[] = frames.map((f) => ({
    id: f.viewId,
    camera: f.camera,
    keypoints: f.keypoints,
  }));
  // `incrementalControls` needs two images. A graph of fewer keyframes has no pair to verify.
  const controls =
    frames.length >= 2
      ? incrementalControls(images, [], options.verification ?? {}).controls
      : null;
  const byId = new Map(images.map((i) => [i.id, i]));
  const edges = new Map<string, ViewGraphEdge>();
  const tried = new Set<string>();
  let verifiedPairs = 0;

  const report = options.onProgress;
  const verifyBatch = options.verifyBatch ?? 48;
  if (!Number.isSafeInteger(verifyBatch) || verifyBatch < 1)
    throw new RangeError('Invalid view graph options');
  let matchTotal = 0,
    matchDone = 0,
    verifyTotal = 0,
    verifyDone = 0;

  const round = async (
    pairs: readonly (readonly [number, number])[],
    why: (k: string) => CandidateSource,
  ) => {
    // Matching first, for every pair of the round. The pairs are sorted by the first keyframe:
    // one upload serves its whole batch.
    const groups = new Map<number, number[]>();
    for (const [a, b] of pairs) groups.set(a, [...(groups.get(a) ?? []), b]);
    matchTotal += pairs.length;
    report?.({ stage: 'matching', done: matchDone, total: matchTotal });
    const matched: { a: number; b: number; m: MatchedPair }[] = [];
    for (const [a, list] of groups) {
      const matchStarted = performance.now();
      const results = await backend.match(
        store.view(a),
        list.map((b) => store.view(b)),
        { ratio: o.ratio },
      );
      pairMatchingMs += performance.now() - matchStarted;
      for (let t = 0; t < list.length; t++)
        if (results[t]!.matches.length / 2 >= o.minMatches)
          matched.push({ a, b: list[t]!, m: results[t]! });
      matchDone += list.length;
      report?.({ stage: 'matching', done: matchDone, total: matchTotal });
    }
    // Then verification, a slice of the matched pairs in one backend call.
    verifyTotal += matched.length;
    report?.({ stage: 'verification', done: verifyDone, total: verifyTotal });
    for (let at = 0; at < matched.length; at += verifyBatch) {
      const slice = matched.slice(at, at + verifyBatch);
      const batch = slice.map(({ a, b, m }) => {
        verifiedPairs++;
        const pair: PairMatches = {
          first: a,
          second: b,
          matches: m.matches,
          distances: m.distances,
        };
        const machine = verifyPairs(byId, [pair], controls!);
        // One pair yields one job and then returns.
        const job = machine.next();
        if (job.done) throw new Error('A pair verification ended without a job');
        return { a, b, machine, job: job.value };
      });
      const verifyStarted = performance.now();
      const answers = backend.executeMany
        ? await backend.executeMany(batch.map((x) => x.job))
        : await batch.reduce<Promise<unknown[]>>(
            (chain, x) => chain.then(async (all) => [...all, await backend.execute(x.job)]),
            Promise.resolve([]),
          );
      pairVerificationMs += performance.now() - verifyStarted;
      for (let t = 0; t < batch.length; t++) {
        const { a, b, machine } = batch[t]!;
        const step = machine.next(answers[t]);
        if (!step.done) throw new Error('A pair verification asked for a second job');
        const { verified, diagnostics, candidates: seeds } = step.value;
        const d = diagnostics[0]!,
          v = verified[0];
        if (!v || d.inliers < o.minEdgeInliers) continue;
        const key = pairKey(a, b);
        edges.set(key, {
          first: a,
          second: b,
          crossVideo: byViewId.get(a)!.videoId !== byViewId.get(b)!.videoId,
          matches: v.matches,
          distances: v.distances ?? new Float64Array(0),
          inliers: d.inliers,
          homographyInliers: d.homographyInliers,
          medianAngleDeg: d.medianAngleDeg,
          seedEligible: d.seedEligible,
          diagnostic: d,
          geometry: seeds[0]?.result,
          source: why(key),
        });
      }
      verifyDone += slice.length;
      report?.({ stage: 'verification', done: verifyDone, total: verifyTotal });
    }
  };

  const first = candidates.pairs;
  report?.({ stage: 'candidates', total: first.length });
  for (const [a, b] of first) tried.add(pairKey(a, b));
  if (controls) await round(first, (k) => candidates.source.get(k)!);

  let expansionPairs = 0;
  let expansionList: (readonly [number, number])[] = [];
  if (controls && o.expansion) {
    // Once, on the edges of the first round: the best neighbours of one end against the other.
    const extra = new Map<string, [number, number]>();
    for (const e of [...edges.values()].filter((e) => e.crossVideo)) {
      for (const [from, to] of [
        [e.second, e.first],
        [e.first, e.second],
      ] as const) {
        for (const nb of (candidates.retrieved.get(from) ?? []).slice(0, o.expansionK)) {
          const key = pairKey(nb, to);
          if (nb === to || tried.has(key) || extra.has(key)) continue;
          if (byViewId.get(nb)!.frameKey === byViewId.get(to)!.frameKey) continue;
          extra.set(key, nb < to ? [nb, to] : [to, nb]);
        }
      }
    }
    const pairs = [...extra.values()].sort((x, y) => x[0] - y[0] || x[1] - y[1]);
    expansionPairs = pairs.length;
    expansionList = pairs;
    for (const key of extra.keys()) tried.add(key);
    await round(pairs, () => 'expansion');
  }

  const sortedEdges = [...edges.values()].sort((x, y) => x.first - y.first || x.second - y.second);
  const componentsStarted = performance.now();
  const { components, videos } = connectedComponents(frames, sortedEdges, o);
  const componentsMs = performance.now() - componentsStarted;
  report?.({ stage: 'components', edges: sortedEdges.length });
  const count = (s: CandidateSource) =>
    [...candidates.source.values()].filter((v) => v === s).length;
  return {
    edges: sortedEdges,
    components,
    videos,
    candidates: [...first, ...expansionList],
    timing: { retrievalMs, pairMatchingMs, pairVerificationMs, componentsMs },
    stats: {
      keyframes: frames.length,
      videos: videos.length,
      exhaustivePairs: (frames.length * (frames.length - 1)) / 2,
      candidatePairs: first.length,
      temporalPairs: count('temporal'),
      retrievalPairs: count('retrieval'),
      reservePairs: count('reserve'),
      expansionPairs,
      verifiedPairs,
      edges: sortedEdges.length,
      crossVideoEdges: sortedEdges.filter((e) => e.crossVideo).length,
    },
  };
}
