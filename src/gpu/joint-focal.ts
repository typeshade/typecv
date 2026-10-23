import type { Resident } from 'typeshade/runtime';
import { validateReconstruction } from '../reconstruction/bundle-adjustment.ts';
import type { SparseReconstruction } from '../reconstruction/types.ts';
import { ShadeDevice, type ChunkRate, type ShadePlanReport } from './device.ts';
import { planBundleResidency } from './budget-plan.ts';
import { finiteGpuArray, validateGpuCamera } from './pnp.ts';
import {
  initializeBundle,
  linearizeBundle,
  bundleMetricPartials,
  bundleMetricPartialsOracle,
  finishBundleMetrics,
  bundlePointBlocks,
  bundleCameraBlocks,
  bundleCameraRows,
  invertBundleCameraColumns,
  finalizeBundleCameraInverse,
  initializeBundleCg,
  bundleSchurPoint,
  bundleSchurCamera,
  bundleSchurCameraCooperative,
  bundleSchurCameraWorkgroup,
  finishBundleSchurFocal,
  advanceBundleCg,
  snapshotBundleActivity,
  updateBundleCandidate,
  bundleCandidateGauge,
  normalizeBundleCandidate,
  bundleCandidatePartials,
  bundleCandidatePartialsOracle,
  acceptBundleCandidate,
  commitBundleCandidate,
  packBundleResult,
  bundleFocalCameraRows,
  bundleFocalObservations,
  dampBundleFocal,
  bundleFocalPoints,
  checkBundlePoints,
  cacheBundleSchurCross,
  cacheBundleSchurRows,
  bundleCachedSchurProduct,
  advanceCachedBundleCg,
  prepareCachedBundleCertificate,
  certifyCachedBundleCg,
} from './joint-focal.shade.ts';

import {
  bundleStallMonitor,
  cooperativeBundleSchur,
  type GpuBundleAdjustmentOptions,
  type GpuBundleAdjustmentResult,
} from './bundle.ts';

/** Matrix-free Schur PCG with a single global log-focal coordinate coupled to every pose and point. */
export function bundleAdjustJointFocalGpu(
  device: ShadeDevice,
  input: SparseReconstruction,
  options: GpuBundleAdjustmentOptions = {},
): Promise<GpuBundleAdjustmentResult> {
  validateReconstruction(input);
  const C = input.views.length,
    P = input.landmarks.length,
    M = input.landmarks.reduce((n, p) => n + p.observations.length, 0);
  if (![C, P, M].every((n) => Number.isSafeInteger(n) && n >= 0 && n <= 0xffffffff))
    throw new RangeError('GPU BA scene dimensions exceed integer index bounds');
  const iterations = options.iterations ?? 20,
    huber = options.huberPx ?? 2,
    maxCg = options.maxCgIterations ?? Math.min(256, Math.max(32, C * 12 + 1)),
    tolerance = options.cgRelativeTolerance ?? 1e-6;
  if (
    C < 2 ||
    P < 1 ||
    !Number.isSafeInteger(iterations) ||
    iterations < 0 ||
    iterations > 100 ||
    !Number.isFinite(huber) ||
    huber < 1e-12 ||
    huber > 1e6 ||
    !Number.isSafeInteger(maxCg) ||
    maxCg < 1 ||
    maxCg > 256 ||
    !Number.isFinite(tolerance) ||
    tolerance < 1e-12 ||
    tolerance > 1e-2
  )
    throw new RangeError('Invalid GPU BA scene/options');
  const byId = new Map(input.views.map((v, i) => [v.id, i])),
    anchorId = options.anchorViewId ?? input.views[0]!.id,
    scaleId = options.scaleViewId ?? input.views[1]!.id;
  if (!byId.has(anchorId) || !byId.has(scaleId) || anchorId === scaleId)
    throw new RangeError('Invalid GPU BA gauge cameras');
  const variable =
    options.variableViewIds === undefined ? undefined : new Set(options.variableViewIds);
  if (
    variable &&
    (variable.size !== options.variableViewIds!.length ||
      [...variable].some((id) => id === anchorId || !byId.has(id)))
  )
    throw new RangeError('Invalid GPU BA variable camera window');
  const active = input.views.map((v) => v.id !== anchorId && (!variable || variable.has(v.id))),
    fixed = active.filter((x) => !x).length;
  if (fixed > 1 && variable?.has(scaleId))
    throw new RangeError('Local GPU BA scale camera must be fixed');
  const connected = new Set([anchorId]);
  let grew = true;
  while (grew) {
    grew = false;
    for (const p of input.landmarks)
      if (p.observations.some((o) => connected.has(o.viewId)))
        for (const o of p.observations)
          if (!connected.has(o.viewId)) {
            connected.add(o.viewId);
            grew = true;
          }
  }
  if (input.views.some((v) => !connected.has(v.id)))
    throw new RangeError('GPU BA requires one connected observation graph');
  // Conservative structural bound on the builder's per-row observation traversals.
  // It is not a GPU-duration guarantee.
  const cacheWorkLimit = 262144;
  const wantsSchurCache =
    options.schurCache === true &&
    device.backend === 'webgpu' &&
    !device.boundedSubmissions &&
    C <= 16 &&
    M * (6 * C + 1) <= cacheWorkLimit;
  const cameraWorkgroups =
    device.backend === 'webgpu' &&
    device.fp64Flavor === 'float' &&
    options.cameraProduct === 'workgroups';
  const sceneSize = 18 * C + 3 * P + 2 * M,
    trialSize = 12 * C + 3 * P,
    indexSize = 3 * M + P + 2 * C + 2,
    cameraScratchSize = 36 * C,
    vectorSize = 5 * (6 * C + 1) + 3 * P + (cameraWorkgroups ? C : 0),
    stateSize = 26,
    partialsSize = 3 * Math.max(1, Math.ceil(M / 64)),
    outputSize = trialSize + 27;
  // These logical lengths bound every positive u32 shader offset sum. The sizes of the arrays are
  // not a limit of the problem: the plan below cuts the observation and point sections of `blocks`
  // and `focal` to the budget (docs/gpu-budget.md, section 3).
  if (
    ![
      sceneSize,
      trialSize,
      45 * M + 13 * P + 78 * C,
      vectorSize,
      indexSize,
      6 * C + 6 + 2 * M,
    ].every((n) => Number.isSafeInteger(n) && n >= 0 && n <= 0xffffffff)
  )
    throw new RangeError('GPU BA array lengths exceed integer index bounds');
  const pointStarts = new Uint32Array(P + 1);
  let longestTrack = 1;
  for (let p = 0; p < P; p++) {
    const n = input.landmarks[p]!.observations.length;
    pointStarts[p + 1] = pointStarts[p]! + n;
    longestTrack = Math.max(longestTrack, n);
  }
  if (C * 6 + 1 > device.maxKernelElements)
    throw new RangeError(
      'GPU BA reduced camera system exceeds the dispatch limit (docs/gpu-budget.md, 3.6)',
    );
  // Snapshot metadata separately from the mutable numerical arrays.
  const model: SparseReconstruction = {
    views: input.views.map((v) => ({
      ...v,
      camera: { ...v.camera },
      pose: { R: v.pose.R.slice(), t: v.pose.t.slice() },
    })),
    landmarks: input.landmarks.map((p) => ({
      ...p,
      xyz: p.xyz.slice(),
      observations: p.observations.map((o) => ({ ...o, pixel: [...o.pixel] as const })),
    })),
  };
  const source = new Float64Array(sceneSize),
    indices = new Uint32Array(indexSize),
    cameraLists: number[][] = Array.from({ length: C }, () => []);
  for (let c = 0; c < C; c++) {
    const v = model.views[c]!,
      k = validateGpuCamera(v.camera);
    source.set([...v.pose.R, ...v.pose.t], c * 12);
    source.set([k.fx, k.fy, k.cx, k.cy, k.k1!, k.k2!], 12 * C + 3 * P + c * 6);
    indices[3 * M + P + C + 2 + c] = Number(active[c]);
  }
  let obs = 0;
  for (let p = 0; p < P; p++) {
    const point = model.landmarks[p]!;
    source.set(point.xyz, 12 * C + p * 3);
    indices[2 * M + p] = obs;
    for (const o of point.observations) {
      const c = byId.get(o.viewId)!;
      indices[obs] = c;
      indices[M + obs] = p;
      source.set(o.pixel, 18 * C + 3 * P + obs * 2);
      cameraLists[c]!.push(obs);
      obs++;
    }
  }
  indices[2 * M + P] = obs;
  let cursor = 0;
  for (let c = 0; c < C; c++) {
    indices[2 * M + P + 1 + c] = cursor;
    indices.set(cameraLists[c]!, 2 * M + P + C + 2 + cursor);
    cursor += cameraLists[c]!.length;
  }
  indices[2 * M + P + 1 + C] = cursor;
  if (!finiteGpuArray(source))
    throw new RangeError('GPU BA input exceeds supported coordinate bounds');
  const control = {
    cameras: C,
    points: P,
    observations: M,
    anchor: byId.get(anchorId)!,
    scaleView: byId.get(scaleId)!,
    normalizeScale: Number(fixed === 1),
    huber,
    relativeTolerance: tolerance,
    cgIterations: maxCg,
  };
  // The units of the key `bundleCg` are the rows of one matrix-free PCG iteration, as on the
  // fixed-focal route (`src/gpu/bundle.ts`): one per observation and one per row of the reduced
  // system, `6C + 1` with the shared focal row. A chunk holds at least one iteration.
  const cgWork = M + 6 * C + 1;
  // The cached route's iteration is the dense product over the `(6C + 1)^2` entries of the
  // cached Schur matrix and one PCG step. Its own key `bundleCachedCg` keeps its rate apart from
  // the matrix-free rows, whose cost per unit differs.
  const cachedCgWork = (6 * C + 1) ** 2;
  return device.enqueue(async () => {
    const before = device.statistics,
      owned: (Resident<Float64Array> | Resident<Uint32Array>)[] = [];
    let successful = false;
    // The residency plan (docs/gpu-budget.md, section 3.3): the whole problem when it fits the
    // budget, else windows of whole points. The budget is what the device holds free now. The
    // opt-in Schur cache needs the whole problem resident: a streamed run does without it.
    const budget = Math.max(0, device.workingBytes - (before.liveBytes - before.pooledBytes));
    const sum = (xs: readonly number[]) => xs.reduce((n, x) => n + x, 0);
    const planWith = (cache: boolean) => {
      const common = [
        sceneSize * 8,
        trialSize * 8,
        vectorSize * 8,
        stateSize * 8,
        partialsSize * 8,
        outputSize * 8,
        indexSize * 4,
      ];
      const cacheArrays = cache ? [(6 * C + 1) ** 2 * 8, (18 * M + 3 * P) * 8] : [],
        activityBytes = (cache ? 3 : 2) * 4,
        cameraBlocksBytes = 78 * C * 8,
        scratchBytes = cameraScratchSize * 8,
        focalFixed = (6 * C + 6) * 8;
      const residentArrays = [
          ...common,
          (45 * M + 13 * P + 78 * C) * 8,
          (6 * C + 6 + 2 * M) * 8,
          ...(device.backend === 'webgpu' ? [scratchBytes] : []),
          ...cacheArrays,
        ],
        floorArrays = [
          ...common,
          cameraBlocksBytes,
          scratchBytes,
          focalFixed,
          ...(cameraWorkgroups ? [7 * 64 * C * 8] : []),
        ];
      return planBundleResidency({
        points: P,
        observations: M,
        pointStarts,
        longestTrack,
        budgetBytes: budget,
        bindingBytes: device.bindingBytes,
        maxKernelElements: device.maxKernelElements,
        residentBytes: sum(residentArrays) + activityBytes,
        residentArrays,
        floorBytes: sum(floorArrays) + activityBytes,
        floorArrays,
        observationBytes: 45 * 8 + 16,
        pointBytes: 13 * 8,
        windowArrays: (m, p) => [(45 * m + 13 * p + 78 * C) * 8, (6 * C + 6 + 2 * m) * 8],
      });
    };
    let plan = planWith(wantsSchurCache);
    const useSchurCache = wantsSchurCache && plan.mode === 'resident';
    if (wantsSchurCache && !useSchurCache) plan = planWith(false);
    if (!plan.fits)
      throw new RangeError(
        'GPU BA needs ' +
          plan.bytes +
          ' bytes for the arrays that stay resident and a window of one track, more than the working budget of ' +
          budget +
          ' bytes (docs/gpu-budget.md, 3.6)',
      );
    const streamed = plan.mode === 'streamed',
      blockSize = 45 * plan.obsCapacity + 13 * plan.pointCapacity + 78 * C,
      focalSize = 6 * C + 6 + 2 * plan.obsCapacity,
      schurSize = useSchurCache ? (6 * C + 1) ** 2 : 0,
      schurCrossSize = useSchurCache ? 18 * M + 3 * P : 0,
      activitySize = useSchurCache ? 3 : 2;
    // The CPU tier keeps the per-camera kernel of a resident run; a streamed run continues the
    // camera rows from window to window, as the WebGPU route does.
    const rows = device.backend === 'webgpu' || streamed;
    const full = {
      ...control,
      obsBase: 0,
      obsEnd: M,
      pointBase: 0,
      pointEnd: P,
      obsCapacity: plan.obsCapacity,
      pointCapacity: plan.pointCapacity,
      first: 1,
      last: 1,
      sweep: 0,
      streamed: Number(streamed),
      base: 0,
    };
    // The control of window `k` of a sweep (1: PCG, 2: update, 0: assembly), and of cost window `k`.
    const windowControl = (k: number, sweep: number) => ({
      ...full,
      ...plan.windows[k]!,
      first: Number(k === 0),
      last: Number(k === plan.windows.length - 1),
      sweep,
    });
    const costControl = (k: number) => ({ ...full, ...plan.costWindows[k]! });
    const groups = (n: number) => Math.ceil(n / 64);
    const wide = (n: number, v?: Float64Array) => {
      const b = device.allocateF64(n, v);
      owned.push(b);
      return b;
    };
    // A dispatch over `n` elements that exceeds the dispatch limit is split: piece `base`.
    const piece = Math.max(64, device.maxKernelElements - (device.maxKernelElements % 64));
    function* pieces(name: string, n: number, run: (control: typeof full, count: number) => void) {
      for (let at = 0; at < n; at += piece)
        yield [name, () => run({ ...full, base: at }, groups(Math.min(piece, n - at)))] as const;
    }
    const reports: ShadePlanReport[] = [];
    try {
      const scene = wide(sceneSize, source),
        trial = wide(trialSize),
        index = device.allocateU32(indexSize, indices);
      owned.push(index);
      const blocks = wide(blockSize),
        cameraScratch = rows ? wide(cameraScratchSize) : undefined,
        schur = useSchurCache ? wide(schurSize) : undefined,
        schurCross = useSchurCache ? wide(schurCrossSize) : undefined,
        focal = wide(focalSize),
        vectors = wide(vectorSize),
        state = wide(stateSize),
        partials = wide(partialsSize),
        cameraLanes = cameraWorkgroups && streamed ? wide(7 * 64 * C) : partials,
        activity = device.allocateU32(activitySize),
        output = wide(outputSize);
      owned.push(activity);
      const monitor = bundleStallMonitor(device, state, options);
      // The linearization and the cost partials of one cost window (or of the whole problem).
      function* metrics() {
        for (let k = 0; k < plan.costWindows.length; k++) {
          const w = plan.costWindows[k]!,
            ctl = costControl(k),
            n = w.obsEnd - w.obsBase;
          yield [
            device.backend === 'webgpu' ? 'bundleMetricPartials' : 'bundleMetricPartialsOracle',
            () => {
              if (device.backend === 'webgpu')
                bundleMetricPartials(
                  { control: ctl, scene, trial, index, partials, focal },
                  groups(Math.max(1, n)),
                );
              else
                bundleMetricPartialsOracle(
                  { control: ctl, scene, trial, index, partials, focal },
                  Math.ceil(groups(n) / 8),
                );
            },
          ] as const;
        }
      }
      reports.push(
        await device.planSteps(
          (function* () {
            yield [
              'initializeBundle',
              () => {
                initializeBundle({ control: full, scene, trial, state, focal }, 1);
              },
            ] as const;
            yield* metrics();
            yield [
              'finishBundleMetrics',
              () => {
                finishBundleMetrics({ control: full, state, partials }, 1);
              },
            ] as const;
          })(),
        ),
      );
      let lmActive = true;
      await monitor.start();
      for (let iteration = 0; iteration < iterations && lmActive; iteration++) {
        reports.push(
          await device.planSteps(
            (function* () {
              // Assembly: the point blocks, the camera rows and the focal row. A resident run
              // linearizes once here; a streamed run linearizes each window, in the assembly sweep
              // and once more for the focal point terms.
              if (!streamed) {
                yield [
                  'linearizeBundle',
                  () => {
                    linearizeBundle(
                      { control: full, scene, trial, index, blocks, state, focal },
                      groups(M),
                    );
                  },
                ] as const;
                yield [
                  'bundlePointBlocks',
                  () => {
                    bundlePointBlocks({ control: full, index, blocks, state, focal }, groups(P));
                  },
                ] as const;
              }
              if (!rows) {
                yield [
                  'bundleCameraBlocks',
                  () => {
                    bundleCameraBlocks({ control: full, index, blocks, state }, groups(C));
                  },
                ] as const;
              }
              if (rows && !cameraScratch)
                throw new Error('GPU BA camera scratch was not allocated');
              for (let k = 0; k < (streamed ? plan.windows.length : 1); k++) {
                const ctl = streamed ? windowControl(k, 0) : full,
                  w = plan.windows[k]!;
                if (streamed) {
                  yield [
                    'linearizeBundle',
                    () => {
                      linearizeBundle(
                        { control: ctl, scene, trial, index, blocks, state, focal },
                        groups(w.obsEnd - w.obsBase),
                      );
                    },
                  ] as const;
                  yield [
                    'bundlePointBlocks',
                    () => {
                      bundlePointBlocks(
                        { control: ctl, index, blocks, state, focal },
                        groups(w.pointEnd - w.pointBase),
                      );
                    },
                  ] as const;
                  yield [
                    'checkBundlePoints',
                    () => {
                      checkBundlePoints({ control: ctl, blocks, state }, 1);
                    },
                  ] as const;
                }
                if (rows)
                  yield [
                    'bundleCameraRows',
                    () => {
                      bundleCameraRows(
                        { control: ctl, index, blocks, state, cameraScratch: cameraScratch! },
                        groups(C * 6),
                      );
                    },
                  ] as const;
                yield [
                  'bundleFocalCameraRows',
                  () => {
                    bundleFocalCameraRows(
                      { control: ctl, index, blocks, state, focal },
                      groups(C * 6),
                    );
                  },
                ] as const;
                yield [
                  'bundleFocalObservations',
                  () => {
                    bundleFocalObservations({ control: ctl, blocks, state, focal }, 1);
                  },
                ] as const;
              }
              if (rows) {
                yield [
                  'invertBundleCameraColumns',
                  () => {
                    invertBundleCameraColumns(
                      { control: full, blocks, cameraScratch: cameraScratch!, state },
                      groups(C * 6),
                    );
                  },
                ] as const;
                yield [
                  'finalizeBundleCameraInverse',
                  () => {
                    finalizeBundleCameraInverse({ control: full, blocks, state }, groups(C));
                  },
                ] as const;
              }
              yield [
                'dampBundleFocal',
                () => {
                  dampBundleFocal({ control: full, state, focal }, 1);
                },
              ] as const;
              // The point terms of the focal diagonal and RHS need the point blocks again.
              for (let k = 0; k < (streamed ? plan.windows.length : 1); k++) {
                const ctl = streamed ? windowControl(k, 0) : full,
                  w = plan.windows[k]!;
                if (streamed) {
                  yield [
                    'linearizeBundle',
                    () => {
                      linearizeBundle(
                        { control: ctl, scene, trial, index, blocks, state, focal },
                        groups(w.obsEnd - w.obsBase),
                      );
                    },
                  ] as const;
                  yield [
                    'bundlePointBlocks',
                    () => {
                      bundlePointBlocks(
                        { control: ctl, index, blocks, state, focal },
                        groups(w.pointEnd - w.pointBase),
                      );
                    },
                  ] as const;
                }
                yield [
                  'bundleFocalPoints',
                  () => {
                    bundleFocalPoints({ control: ctl, index, blocks, state, focal }, 1);
                  },
                ] as const;
              }
              yield [
                'initializeBundleCg',
                () => {
                  initializeBundleCg({ control: full, blocks, vectors, state, focal }, 1);
                },
              ] as const;
              let cgActive = true;
              const snapshotActivity = async () => {
                const flags = await device.read(activity, 2);
                cgActive = flags[0] !== 0;
                lmActive = flags[1] !== 0;
              };
              const snapshot = [
                'snapshotBundleActivity',
                () => {
                  snapshotBundleActivity({ state, activity }, 1);
                },
                snapshotActivity,
              ] as const;
              // The dispatches of one PCG iteration, in order. A streamed run goes over the
              // windows: each linearizes its observations and rebuilds its point blocks, the
              // values of the assembly, and continues the camera rows of S p.
              const schurPoint = (ctl: typeof full, scale: number, count: number) =>
                bundleSchurPoint({ control: ctl, index, blocks, vectors, state }, scale * count);
              const cooperative = cooperativeBundleSchur(
                device,
                C,
                M,
                true,
                options.schurReduction,
              );
              const schurCameraSteps = function* (ctl: typeof full, scale: number) {
                if (!cameraWorkgroups) {
                  yield [
                    'bundleSchurCamera',
                    () =>
                      (cooperative ? bundleSchurCameraCooperative : bundleSchurCamera)(
                        { control: ctl, index, blocks, vectors, state, focal },
                        scale * (cooperative ? C * 6 + 1 : groups(C * 6 + 1)),
                      ),
                  ] as const;
                  return;
                }
                for (let at = 0; at < C; at += device.maxWorkgroups)
                  yield [
                    'bundleSchurCameraPartial',
                    () => {
                      bundleSchurCameraWorkgroup(
                        {
                          control: { ...ctl, base: at },
                          index,
                          blocks,
                          vectors,
                          state,
                          focal,
                          cameraLanes,
                        },
                        scale * Math.min(device.maxWorkgroups, C - at),
                      );
                    },
                  ] as const;
                if (ctl.last !== 0)
                  yield [
                    'bundleSchurCamera',
                    () => {
                      finishBundleSchurFocal({ control: ctl, vectors, state, focal }, scale);
                    },
                  ] as const;
              };
              const schurCamera = (ctl: typeof full, scale: number) => {
                for (const [, run] of schurCameraSteps(ctl, scale)) run();
              };
              const windowSweep = (k: number, scale: number) => {
                const ctl = windowControl(k, 1),
                  w = plan.windows[k]!;
                linearizeBundle(
                  { control: ctl, scene, trial, index, blocks, state, focal },
                  scale * groups(w.obsEnd - w.obsBase),
                );
                bundlePointBlocks(
                  { control: ctl, index, blocks, state, focal },
                  scale * groups(w.pointEnd - w.pointBase),
                );
                schurPoint(ctl, scale, groups(w.pointEnd - w.pointBase));
                schurCamera(ctl, scale);
              };
              // One PCG iteration; `scale` 0 dispatches its entries over no workgroups (the warm-up
              // of `GpuChunk.warm`).
              const cgIteration = (scale = 1) => {
                if (streamed) for (let k = 0; k < plan.windows.length; k++) windowSweep(k, scale);
                else {
                  schurPoint(full, scale, groups(P));
                  schurCamera(full, scale);
                }
                advanceBundleCg({ control: full, blocks, vectors, state, focal }, scale);
              };
              // The PCG loop runs under the time budget (`docs/gpu-sfm.md`, "The BA's PCG loop"):
              // a chunk is `n` iterations of this LM iteration, one plan that the TypeShade device
              // fences and times under `key`, and it ends with the flags, which the host reads
              // before it sizes the next chunk or stops. The iterations are the dispatches of the
              // per-iteration path in the same order; an iteration past convergence is a guarded
              // no-op, so skipping it changes no result. The CPU tier runs the whole loop as one
              // chunk.
              const cgChunks = function* (
                key: string,
                limit: number,
                work: number,
                iteration: (scale: number) => void,
              ) {
                for (let k = 0; k < limit && cgActive;) {
                  const remaining = limit - k,
                    n = Math.floor(device.chunkSize(key, remaining * work, work) / work);
                  yield [
                    key,
                    () => {
                      for (let i = 0; i < n; i++) iteration(1);
                      snapshotBundleActivity({ state, activity }, 1);
                    },
                    snapshotActivity,
                    {
                      key,
                      units: n * work,
                      // Cached solves read their initial flags before the chunks; a matrix-free
                      // solve first reads them here, after its guarded first chunk.
                      warm: { id: 'joint-focal', run: () => iteration(0) },
                    },
                  ] as const;
                  k += n;
                }
              };
              if (useSchurCache) {
                if (!schur || !schurCross || !cameraScratch)
                  throw new Error('GPU Schur cache was not allocated');
                yield [
                  'cacheBundleSchurCross',
                  () => {
                    cacheBundleSchurCross(
                      { control: full, index, blocks, state, schurCross },
                      Math.ceil(P / 64),
                    );
                  },
                ] as const;
                yield [
                  'cacheBundleSchurRows',
                  () => {
                    cacheBundleSchurRows(
                      {
                        control: full,
                        index,
                        blocks,
                        cameraScratch,
                        focal,
                        state,
                        schurCross,
                        schur,
                      },
                      Math.ceil((C * 6 + 1) / 64),
                    );
                  },
                ] as const;
                // The cached PCG loop: two dispatches an iteration over the cached Schur matrix,
                // the arithmetic of the former one-workgroup kernel in the same order.
                yield snapshot;
                yield* cgChunks('bundleCachedCg', maxCg, cachedCgWork, (scale) => {
                  bundleCachedSchurProduct(
                    { control: full, vectors, state, schur },
                    scale * Math.ceil((C * 6 + 1) / 64),
                  );
                  advanceCachedBundleCg({ control: full, blocks, vectors, state, focal }, scale);
                });
                let remainingCg = 0;
                const certificate = function* () {
                  yield [
                    'prepareCachedBundleCertificate',
                    () => {
                      prepareCachedBundleCertificate(
                        { control: full, blocks, vectors, state, focal },
                        1,
                      );
                    },
                  ] as const;
                  yield [
                    'bundleSchurPoint',
                    () => {
                      bundleSchurPoint(
                        { control: full, index, blocks, vectors, state },
                        Math.ceil(P / 64),
                      );
                    },
                  ] as const;
                  yield* schurCameraSteps(full, 1);
                  yield [
                    'certifyCachedBundleCg',
                    () => {
                      certifyCachedBundleCg(
                        { control: full, blocks, vectors, state, focal, activity },
                        1,
                      );
                    },
                    async () => {
                      const flags = await device.read(activity, 3);
                      cgActive = flags[0] !== 0;
                      lmActive = flags[1] !== 0;
                      remainingCg = Math.max(0, maxCg - flags[2]!);
                    },
                  ] as const;
                };
                // The certificate is mandatory, with or without solver checkpoints.
                yield* certificate();
                // Continue the original operator from the certified residual, never
                // spending more than the caller's total per-LM CG iteration cap.
                const retry = remainingCg > 0 && cgActive;
                yield* cgChunks(
                  streamed ? 'bundleCgStreamed' : 'bundleCg',
                  remainingCg,
                  cgWork,
                  cgIteration,
                );
                // A restarted original solve must also meet the true-residual test.
                if (retry) yield* certificate();
              } else if (!device.boundedSubmissions) {
                // The first chunk also snapshots the flags. Inactive PCG/LM work is guarded,
                // so a separate pre-chunk snapshot would only add a GPU round trip.
                yield* cgChunks(
                  streamed ? 'bundleCgStreamed' : 'bundleCg',
                  maxCg,
                  cgWork,
                  cgIteration,
                );
              } else {
                // Bounded submissions keep one step per dispatch, so that no plan holds more than
                // `maxDispatchesPerSubmission` dispatches, and read the flags every four
                // iterations with solver checkpoints.
                if (device.solverCheckpoints) yield snapshot;
                for (let k = 0; k < maxCg && cgActive; k++) {
                  for (let w = 0; w < (streamed ? plan.windows.length : 1); w++) {
                    const ctl = streamed ? windowControl(w, 1) : full,
                      win = plan.windows[w]!;
                    if (streamed) {
                      yield [
                        'linearizeBundle',
                        () => {
                          linearizeBundle(
                            { control: ctl, scene, trial, index, blocks, state, focal },
                            groups(win.obsEnd - win.obsBase),
                          );
                        },
                      ] as const;
                      yield [
                        'bundlePointBlocks',
                        () => {
                          bundlePointBlocks(
                            { control: ctl, index, blocks, state, focal },
                            groups(win.pointEnd - win.pointBase),
                          );
                        },
                      ] as const;
                    }
                    yield [
                      'bundleSchurPoint',
                      () => {
                        schurPoint(ctl, 1, groups(win.pointEnd - win.pointBase));
                      },
                    ] as const;
                    yield* schurCameraSteps(ctl, 1);
                  }
                  yield [
                    'advanceBundleCg',
                    () => {
                      advanceBundleCg({ control: full, blocks, vectors, state, focal }, 1);
                    },
                  ] as const;
                  if (device.solverCheckpoints && (k + 1) % 4 === 0 && k + 1 < maxCg)
                    yield snapshot;
                }
              }
              // The update: the focal scale, the cameras and the points of the candidate.
              for (let k = 0; k < (streamed ? plan.windows.length : 1); k++) {
                const ctl = streamed ? windowControl(k, 2) : full,
                  w = plan.windows[k]!;
                if (streamed) {
                  yield [
                    'linearizeBundle',
                    () => {
                      linearizeBundle(
                        { control: ctl, scene, trial, index, blocks, state, focal },
                        groups(w.obsEnd - w.obsBase),
                      );
                    },
                  ] as const;
                  yield [
                    'bundlePointBlocks',
                    () => {
                      bundlePointBlocks(
                        { control: ctl, index, blocks, state, focal },
                        groups(w.pointEnd - w.pointBase),
                      );
                    },
                  ] as const;
                }
                yield [
                  'updateBundleCandidate',
                  () => {
                    updateBundleCandidate(
                      { control: ctl, scene, trial, index, blocks, vectors, state, focal },
                      groups(Math.max(k === 0 ? C : 1, w.pointEnd - w.pointBase)),
                    );
                  },
                ] as const;
              }
              yield [
                'bundleCandidateGauge',
                () => {
                  bundleCandidateGauge({ control: full, scene, trial, state, focal }, 1);
                },
              ] as const;
              yield* pieces('normalizeBundleCandidate', Math.max(C, P), (ctl, count) => {
                normalizeBundleCandidate({ control: ctl, scene, trial, state }, count);
              });
              for (let k = 0; k < plan.costWindows.length; k++) {
                const w = plan.costWindows[k]!,
                  ctl = costControl(k),
                  n = w.obsEnd - w.obsBase;
                yield [
                  device.backend === 'webgpu'
                    ? 'bundleCandidatePartials'
                    : 'bundleCandidatePartialsOracle',
                  () => {
                    if (device.backend === 'webgpu')
                      bundleCandidatePartials(
                        { control: ctl, scene, trial, index, partials, focal },
                        groups(Math.max(1, n)),
                      );
                    else
                      bundleCandidatePartialsOracle(
                        { control: ctl, scene, trial, index, partials, focal },
                        Math.ceil(groups(n) / 8),
                      );
                  },
                ] as const;
              }
              yield [
                'acceptBundleCandidate',
                () => {
                  acceptBundleCandidate({ control: full, state, partials }, 1);
                },
              ] as const;
              yield* pieces('commitBundleCandidate', trialSize, (ctl, count) => {
                commitBundleCandidate({ control: ctl, scene, trial, state, focal }, count);
              });
              if (device.solverCheckpoints) yield snapshot;
            })(),
          ),
        );
        // A stall ends the LM loop (`stallTolerance`); the shaders' own rule may end it earlier.
        if (lmActive && (await monitor.stalled())) lmActive = false;
      }
      reports.push(
        await device.planSteps(
          (function* () {
            yield* pieces('packBundleResult', outputSize, (ctl, count) => {
              packBundleResult({ control: ctl, scene, state, output, focal }, count);
            });
          })(),
        ),
      );
      const values = await device.read(output, outputSize),
        base = trialSize;
      if (values[base + 14] === 1) throw new RangeError('GPU BA gauge requires a nonzero baseline');
      if (values[base + 14] === 2)
        throw new RangeError(
          'GPU BA observations must initially project in front of their cameras',
        );
      // Preserve fixed host constants exactly, including bits beyond the emulated format.
      const changed = values[base + 10]! > 0;
      const focalScale = changed ? Math.exp(values[base + 26]!) : 1;
      const reconstruction: SparseReconstruction = {
        views: model.views.map((v, c) => ({
          ...v,
          camera: changed
            ? { ...v.camera, fx: v.camera.fx * focalScale, fy: v.camera.fy * focalScale }
            : v.camera,
          pose:
            changed && active[c]
              ? { R: values.slice(c * 12, c * 12 + 9), t: values.slice(c * 12 + 9, c * 12 + 12) }
              : v.pose,
        })),
        landmarks: model.landmarks.map((p, i) => ({
          ...p,
          xyz: changed ? values.slice(C * 12 + i * 3, C * 12 + i * 3 + 3) : p.xyz,
        })),
      };
      // The rates of the chunk keys the plans ran, as the last plan of each key reported them.
      const after = device.statistics,
        chunks: Record<string, ChunkRate> = Object.assign(
          {},
          ...reports.map((r) => r.chunks ?? {}),
        ),
        report: ShadePlanReport = {
          tier: device.backend,
          dispatches: reports.reduce((n, r) => n + r.dispatches, 0),
          submissions: reports.reduce((n, r) => n + r.submissions, 0),
          groups: reports.flatMap((r) => r.groups),
          ...(Object.keys(chunks).length ? { chunks } : {}),
        };
      successful = true;
      return {
        reconstruction,
        diagnostics: {
          initialCost: values[base + 8]!,
          finalCost: values[base + 6]!,
          initialRmsePx: values[base + 24]!,
          finalRmsePx: values[base + 25]!,
          acceptedSteps: values[base + 10]!,
          rejectedSteps: values[base + 11]!,
          observations: M,
          baseline: values[base + 2]!,
        },
        cgIterations: values[base + 16]!,
        linearSolverFailures: values[base + 17]!,
        execution: {
          backend: device.backend,
          report,
          applicationUploadBytes: after.applicationUploadBytes - before.applicationUploadBytes,
          applicationReadbackBytes:
            after.applicationReadbackBytes - before.applicationReadbackBytes,
          cpuStages: [],
        },
      };
    } finally {
      for (const b of owned) successful ? device.recycle(b) : device.drop(b);
    }
  });
}
