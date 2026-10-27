import test from 'node:test';
import assert from 'node:assert/strict';
import { carryAnchorEdges, chooseAnchors, evictWindow } from '../src/sfm/session/window.ts';
import { archiveSolvePlan, type ArchiveSolveOptions } from '../src/sfm/session/archive-solve.ts';
import { pairKey } from '../src/contracts/keys.ts';
import { cameraCenter } from '../src/geometry/pose.ts';
import {
  cameraWidth,
  historyFocalSeeds,
  type FocalSearchResult,
} from '../src/sfm/focal/focal-search.ts';
import { MAPPER_DEFAULTS, SESSION_DEFAULTS, bundleHuberPx } from '../src/sfm/config/defaults.ts';
import type { Camera } from '../src/contracts/camera.ts';
import type { ImageFeatures, PairMatches, SparseReconstruction } from '../src/contracts/scene.ts';

// The pure policies of `IncrementalSfmSession` (docs/library-structure.md, section 5, phase 6).
// Each row holds a policy to the inline code of `add` and `update` that it replaces, copied from
// `src/sfm/gpu/mapper.ts` on `60b4dd9` with `this.` read from the row, and to the value that the
// branch gives.

const camera = (fx = 100, extra: Partial<Camera> = {}): Camera => ({
  fx,
  fy: fx,
  cx: 49.5,
  cy: 39.5,
  ...extra,
});
const image = (id: number): ImageFeatures => ({
  id,
  camera: camera(),
  keypoints: new Float64Array(64),
});
const pose = (x: number) => ({ R: [1, 0, 0, 0, 1, 0, 0, 0, 1], t: [-x, 0, 0] });
const edge = (first: number, second: number): PairMatches => ({
  first,
  second,
  matches: Uint32Array.of(0, 0),
});

/**
 * A model of the registered views at the given camera centres (on the x axis). `shared` lists the
 * landmarks each view set observes: [viewIds, count]. Feature ids count up per view.
 */
function model(
  centres: readonly (readonly [number, number])[],
  shared: readonly (readonly [readonly number[], number])[],
  cameras: ReadonlyMap<number, Camera> = new Map(),
): SparseReconstruction {
  const next = new Map<number, number>();
  const landmarks = shared.flatMap(([ids, count]) =>
    Array.from({ length: count }, () => ({
      xyz: [0, 0, 5],
      observations: ids.map((viewId) => {
        const featureId = next.get(viewId) ?? 0;
        next.set(viewId, featureId + 1);
        return { viewId, featureId, pixel: [1, 2] as const };
      }),
    })),
  );
  return {
    views: centres.map(([id, x]) => ({ id, camera: cameras.get(id) ?? camera(), pose: pose(x) })),
    landmarks,
  } as unknown as SparseReconstruction;
}

/** The inline window block of `add` on `60b4dd9`. */
function inlineEviction(
  imagesIn: ImageFeatures[],
  edgesIn: PairMatches[],
  windowSize: number,
  stateModel: SparseReconstruction | undefined,
  initialPair: readonly number[] | undefined,
) {
  let images = imagesIn,
    edges = edgesIn;
  if (!(images.length > windowSize)) return undefined;
  const recent = images.slice(-(windowSize - 2));
  const recentIds = new Set(recent.map((view) => view.id));
  let anchors: readonly number[] | undefined = initialPair;
  if (!stateModel) anchors = [images[0]!.id];
  if (stateModel) {
    const shared = new Map<string, number>();
    for (const point of stateModel.landmarks) {
      const ids = point.observations.map((o) => o.viewId).sort((a, b) => a - b);
      for (let i = 0; i < ids.length; i++)
        for (let j = i + 1; j < ids.length; j++) {
          const key = pairKey(ids[i]!, ids[j]!);
          shared.set(key, (shared.get(key) ?? 0) + 1);
        }
    }
    const common = (a: number, b: number) => shared.get(pairKey(a, b)) ?? 0;
    const candidates = stateModel.views.filter((view) => recentIds.has(view.id));
    let best = 0;
    for (let i = 0; i < candidates.length; i++)
      for (let j = i + 1; j < candidates.length; j++) {
        const a = candidates[i]!,
          b = candidates[j]!;
        const ac = cameraCenter(a.pose),
          bc = cameraCenter(b.pose);
        const distance = Math.hypot(...ac.map((v, k) => v - bc[k]!));
        if (distance > best && distance > 1e-8 && common(a.id, b.id) >= 12) {
          anchors = [a.id, b.id];
          best = distance;
        }
      }
  }
  const keep = new Set([...recentIds, ...(anchors ?? [])]);
  images = images.filter((view) => keep.has(view.id));
  edges = edges.filter((pair) => keep.has(pair.first) && keep.has(pair.second));
  let windowModel: SparseReconstruction | undefined;
  if (stateModel && anchors) {
    const existing = new Set(edges.map((pair) => pairKey(pair.first, pair.second)));
    for (const anchor of anchors)
      for (const view of images) {
        if (view.id === anchor) continue;
        const key = pairKey(anchor, view.id);
        if (existing.has(key)) continue;
        const matches: number[] = [];
        for (const point of stateModel.landmarks) {
          let a: number | undefined, b: number | undefined;
          for (const o of point.observations) {
            if (o.viewId === anchor) a = o.featureId;
            else if (o.viewId === view.id) b = o.featureId;
          }
          if (a !== undefined && b !== undefined) matches.push(a, b);
        }
        if (matches.length >= 16) {
          edges.push({ first: anchor, second: view.id, matches: Uint32Array.from(matches) });
          existing.add(key);
        }
      }
    windowModel = {
      views: stateModel.views.filter((view) => keep.has(view.id)),
      landmarks: stateModel.landmarks
        .map((point) => ({
          ...point,
          observations: point.observations.filter((o) => keep.has(o.viewId)),
        }))
        .filter((point) => point.observations.length >= 2),
    };
  }
  return { images, edges, anchors, ...(windowModel ? { model: windowModel } : {}) };
}

const ids = (n: number, from = 1) => Array.from({ length: n }, (_, i) => from + i);
const views = (n: number) => ids(n).map(image);
const chain = (n: number) => ids(n - 1).map((id) => edge(id, id + 1));

const evictionRows: {
  name: string;
  images: ImageFeatures[];
  edges: PairMatches[];
  windowSize: number;
  model?: SparseReconstruction;
  initialPair?: readonly number[];
  anchors: readonly number[] | undefined;
  evicts: boolean;
}[] = [
  {
    name: 'a window at its size evicts nothing',
    images: views(8),
    edges: chain(8),
    windowSize: 8,
    anchors: undefined,
    evicts: false,
  },
  {
    name: 'no model: the first view is the anchor',
    images: views(9),
    edges: chain(9),
    windowSize: 8,
    initialPair: [4, 5],
    anchors: [1],
    evicts: true,
  },
  {
    name: 'the farthest recent pair that shares 12 landmarks',
    images: views(9),
    edges: chain(9),
    windowSize: 8,
    model: model(
      [
        [1, 0],
        [2, 1],
        [4, 3],
        [6, 5],
        [9, 8],
      ],
      [
        [[1, 2], 20],
        [[4, 6], 12],
        [[4, 9], 11],
        [[6, 9, 1], 12],
      ],
    ),
    initialPair: [1, 2],
    anchors: [6, 9],
    evicts: true,
  },
  {
    name: 'no recent pair shares 12 landmarks: the initial pair stays',
    images: views(9),
    edges: chain(9),
    windowSize: 8,
    model: model(
      [
        [1, 0],
        [2, 1],
        [4, 3],
        [9, 8],
      ],
      [
        [[1, 2], 20],
        [[1, 4], 8],
        [[2, 9], 7],
        [[4, 9], 11],
      ],
    ),
    initialPair: [1, 2],
    anchors: [1, 2],
    evicts: true,
  },
  {
    name: 'coincident centres are no baseline',
    images: views(9),
    edges: chain(9),
    windowSize: 8,
    model: model(
      [
        [4, 3],
        [5, 3],
      ],
      [[[4, 5], 30]],
    ),
    initialPair: [1, 2],
    anchors: [1, 2],
    evicts: true,
  },
  {
    name: 'a baseline of 1e-9 is no baseline',
    images: views(9),
    edges: chain(9),
    windowSize: 8,
    model: model(
      [
        [4, 3],
        [5, 3 + 1e-9],
      ],
      [[[4, 5], 30]],
    ),
    initialPair: [1, 2],
    anchors: [1, 2],
    evicts: true,
  },
  {
    name: 'a model and no anchor: the newest views only, and no model change',
    images: views(10),
    edges: [...chain(10), edge(1, 9)],
    windowSize: 8,
    model: model([[5, 1]], [[[5, 6], 3]]),
    anchors: undefined,
    evicts: true,
  },
  {
    name: 'an existing anchor edge is not carried again',
    images: views(9),
    edges: [...chain(9), edge(1, 9)],
    windowSize: 8,
    model: model(
      [
        [1, 0],
        [2, 1],
      ],
      [
        [[1, 9], 8],
        [[1, 8], 8],
        [[2, 7], 7],
        [[1, 2, 6], 9],
      ],
    ),
    initialPair: [1, 2],
    anchors: [1, 2],
    evicts: true,
  },
];

for (const row of evictionRows)
  test(`evictWindow: ${row.name}`, () => {
    const got = evictWindow(row.images, row.edges, row.windowSize, row.model, row.initialPair);
    const want = inlineEviction(
      row.images.slice(),
      row.edges.slice(),
      row.windowSize,
      row.model,
      row.initialPair,
    );
    assert.deepEqual(got, want);
    assert.equal(got !== undefined, row.evicts);
    if (got) assert.deepEqual(got.anchors, row.anchors);
  });

test('chooseAnchors and carryAnchorEdges: the carried edges hold the shared features in landmark order', () => {
  const m = model(
    [
      [1, 0],
      [2, 1],
    ],
    [
      [[1, 9], 8],
      [[1, 8], 7],
      [[2, 9], 8],
    ],
  );
  const images = [1, 2, 8, 9].map(image);
  assert.deepEqual(chooseAnchors(images, new Set([8, 9]), m, [1, 2]), [1, 2]);
  assert.deepEqual(chooseAnchors(images, new Set([8, 9]), undefined, [1, 2]), [1]);
  const carried = carryAnchorEdges([1, 2], images, [edge(8, 9)], m);
  assert.deepEqual(
    carried.map((e) => [e.first, e.second, [...e.matches]]),
    [
      [8, 9, [0, 0]],
      [1, 9, [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7]],
      [2, 9, [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15]],
    ],
  );
});

/** The inline plan of `update`'s final branch on `60b4dd9`, down to the choice of the branch. */
function inlinePlan(
  reconstruction: SparseReconstruction,
  windowModel: SparseReconstruction,
  focal: FocalSearchResult | undefined,
  options: ArchiveSolveOptions,
) {
  const fullWindow = reconstruction.views.length > windowModel.views.length,
    reference = windowModel.views[0]!.camera,
    multipleFocalGroups =
      !!options.focalGroups &&
      new Set(reconstruction.views.map((v) => options.focalGroups!.get(v.id))).size > 1,
    shape = (c: typeof reference) => (c.cy + 0.5) / (c.cx + 0.5),
    sharedCamera =
      focal &&
      !options.focalGroups &&
      reconstruction.views.every(
        ({ camera: c }) =>
          Math.abs(shape(c) - shape(reference)) < 1e-9 &&
          Math.abs(c.fy / c.fx - reference.fy / reference.fx) < 1e-9 &&
          (c.k1 ?? 0) === (reference.k1 ?? 0) &&
          (c.k2 ?? 0) === (reference.k2 ?? 0),
      ),
    focalRatios =
      sharedCamera && options.refineSharedFocal
        ? historyFocalSeeds(focal!, reference.fx / cameraWidth(reference))
        : undefined,
    lateCalibration =
      (focalRatios?.length ?? 0) > 1 &&
      reconstruction.views.some((v) => !focal!.views.includes(v.id)),
    needsFullSolve = fullWindow || lateCalibration;
  if (
    needsFullSolve &&
    !options.separateDirectionalObservations &&
    !options.maxRelativeDepthStdDev
  ) {
    const finalIterations = options.finalBundleIterations ?? SESSION_DEFAULTS.finalBundleIterations;
    return {
      kind: 'solve',
      refinement: {
        iterations:
          finalIterations === 0
            ? 0
            : (options.historyBundleIterations ??
              ((focalRatios?.length ?? 0) > 1 ? Math.max(finalIterations, 32) : finalIterations)),
        huberPx: bundleHuberPx(options.maxErrorPx ?? MAPPER_DEFAULTS.maxErrorPx),
        refineSharedFocal: options.refineSharedFocal && !multipleFocalGroups,
        focalGroups: options.focalGroups,
        ...(focalRatios ? { commonFocalRatios: focalRatios } : {}),
      },
      focalDeferred: !!options.refineSharedFocal && multipleFocalGroups,
    };
  }
  if (needsFullSolve) return { kind: 'measure' };
  return { kind: 'none' };
}

const archive3 = model(
  [
    [1, 0],
    [2, 1],
    [3, 2],
  ],
  [],
);
const window2 = model(
  [
    [2, 1],
    [3, 2],
  ],
  [],
);
const window3 = archive3;
/** A search whose best grid point is an endpoint, so the seeds are three ratios. */
const edgeSearch: FocalSearchResult = {
  ratio: 1,
  views: [1, 2],
  gridRatio: 1,
  candidates: [
    { ratio: 0.5, views: 3, rmsePx: 1, observations: 10, score: 1 },
    { ratio: 1, views: 3, rmsePx: 1, observations: 10, score: 2 },
    { ratio: 2, views: 3, rmsePx: 1, observations: 10, score: 3 },
  ],
} as unknown as FocalSearchResult;
/** A search whose best grid point is inside the grid, so the one seed is the current ratio. */
const middleSearch: FocalSearchResult = {
  ...edgeSearch,
  candidates: [
    { ratio: 0.5, views: 3, rmsePx: 1, observations: 10, score: 2 },
    { ratio: 1, views: 3, rmsePx: 1, observations: 10, score: 1 },
    { ratio: 2, views: 3, rmsePx: 1, observations: 10, score: 3 },
  ],
} as unknown as FocalSearchResult;
const groups = new Map([
  [1, 0],
  [2, 1],
  [3, 1],
]);

const planRows: {
  name: string;
  archive: SparseReconstruction;
  window: SparseReconstruction;
  focal?: FocalSearchResult;
  options: ArchiveSolveOptions;
  kind: 'solve' | 'measure' | 'none';
  iterations?: number;
}[] = [
  {
    name: 'the window is the archive',
    archive: archive3,
    window: window3,
    options: {},
    kind: 'none',
  },
  {
    name: 'an archive past the window: the session default iterations',
    archive: archive3,
    window: window2,
    options: {},
    kind: 'solve',
    iterations: 8,
  },
  {
    name: 'finalBundleIterations 0 solves no iteration',
    archive: archive3,
    window: window2,
    options: { finalBundleIterations: 0, historyBundleIterations: 5 },
    kind: 'solve',
    iterations: 0,
  },
  {
    name: 'historyBundleIterations wins over the final iterations',
    archive: archive3,
    window: window2,
    options: { finalBundleIterations: 12, historyBundleIterations: 5, maxErrorPx: 4 },
    kind: 'solve',
    iterations: 5,
  },
  {
    name: 'separate bearing observations defer the solve',
    archive: archive3,
    window: window2,
    options: { separateDirectionalObservations: true },
    kind: 'measure',
  },
  {
    name: 'a depth-uncertainty guard defers the solve',
    archive: archive3,
    window: window2,
    options: { maxRelativeDepthStdDev: 0.1 },
    kind: 'measure',
  },
  {
    name: 'a late view with several focal seeds solves at least 32 iterations',
    archive: archive3,
    window: window3,
    focal: edgeSearch,
    options: { refineSharedFocal: true },
    kind: 'solve',
    iterations: 32,
  },
  {
    name: 'a late view without refineSharedFocal needs no solve',
    archive: archive3,
    window: window3,
    focal: edgeSearch,
    options: {},
    kind: 'none',
  },
  {
    name: 'several focal groups: no shared focal, and its refinement deferred',
    archive: archive3,
    window: window2,
    focal: edgeSearch,
    options: { refineSharedFocal: true, focalGroups: groups },
    kind: 'solve',
    iterations: 8,
  },
  {
    name: 'a late view with one focal seed needs no solve',
    archive: archive3,
    window: window3,
    focal: middleSearch,
    options: { refineSharedFocal: true },
    kind: 'none',
  },
  {
    name: 'several focal groups without refineSharedFocal defer no focal',
    archive: archive3,
    window: window2,
    options: { focalGroups: groups },
    kind: 'solve',
    iterations: 8,
  },
  {
    name: 'a camera of another principal point shares no focal',
    archive: model(
      [
        [1, 0],
        [2, 1],
        [3, 2],
      ],
      [],
      new Map([[3, camera(100, { cx: 59.5 })]]),
    ),
    window: window3,
    focal: edgeSearch,
    options: { refineSharedFocal: true },
    kind: 'none',
  },
  {
    name: 'a camera of another shape shares no focal',
    archive: model(
      [
        [1, 0],
        [2, 1],
        [3, 2],
      ],
      [],
      new Map([[3, camera(100, { k1: 0.1 })]]),
    ),
    window: window3,
    focal: edgeSearch,
    options: { refineSharedFocal: true },
    kind: 'none',
  },
];

for (const row of planRows)
  test(`archiveSolvePlan: ${row.name}`, () => {
    const got = archiveSolvePlan(row.archive, row.window, row.focal, row.options);
    assert.deepEqual(got, inlinePlan(row.archive, row.window, row.focal, row.options));
    assert.equal(got.kind, row.kind);
    if (got.kind === 'solve') {
      assert.equal(got.refinement.iterations, row.iterations);
      assert.equal(got.focalDeferred, !!row.options.refineSharedFocal && !!row.options.focalGroups);
    }
  });

test('archiveSolvePlan: several focal groups set refineSharedFocal false and defer the focal', () => {
  const got = archiveSolvePlan(archive3, window2, edgeSearch, {
    refineSharedFocal: true,
    focalGroups: groups,
  });
  assert.equal(got.kind, 'solve');
  if (got.kind !== 'solve') return;
  assert.equal(got.refinement.refineSharedFocal, false);
  assert.equal(got.focalDeferred, true);
  assert.equal(got.refinement.huberPx, 1.5);
  assert.equal('commonFocalRatios' in got.refinement, false);
});

test('archiveSolvePlan: the focal seeds of a shared camera are the common focal ratios', () => {
  const got = archiveSolvePlan(archive3, window3, edgeSearch, { refineSharedFocal: true });
  assert.equal(got.kind, 'solve');
  if (got.kind !== 'solve') return;
  // The current ratio 100 / (2 * 49.5 + 1) = 1, the grid middle 1 (a duplicate) and the far end 2.
  assert.deepEqual(got.refinement.commonFocalRatios, [1, 2]);
  assert.equal(got.focalDeferred, false);
});
