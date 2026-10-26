import type { Resident } from 'typeshade/runtime';
import { planCameraProduct } from './bundle-camera-product.ts';
import { BundleCgSchedule } from './bundle-cg-schedule.ts';
import { validateReconstruction } from '../../../geometry/validation.ts';
import type { SparseReconstruction } from '../../../contracts/scene.ts';
import { appendBundleDirections, separateBundleDirections } from '../bundle-directions.ts';
import { ShadeDevice, type ChunkRate, type ShadePlanReport } from '../../../device/gpu/device.ts';
import { planBundleResidency } from '../../../device/gpu/budget-plan.ts';
import { finiteGpuArray, validateGpuCamera } from '../../../geometry/gpu/bounds.ts';
import { bundleAdjustJointFocalGpu } from './joint-focal.ts';
import {
  initializeBundle,
  linearizeBundle,
  bundleMetricPartials,
  bundleMetricPartialsOracle,
  finishBundleMetrics,
  bundlePointBlocks,
  bundleCameraBlocks,
  bundleCameraRows,
  bundleCameraRowsCooperative,
  checkBundlePoints,
  invertBundleCameraColumns,
  finalizeBundleCameraInverse,
  initializeBundleCg,
  bundleSchurPoint,
  bundleSchurCamera,
  bundleSchurCameraCooperative,
  bundleSchurCameraWorkgroup,
  bundleSchurCameraSegments,
  finishBundleSchurCameraSegments,
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
} from './bundle.shade.ts';
import {
  bundleStallMonitor,
  cooperativeBundleSchur,
  type GpuBundleAdjustmentOptions,
  type GpuBundleAdjustmentResult,
} from './bundle-shared.ts';

/** @internal */
export { bundleStallMonitor, cooperativeBundleSchur } from './bundle-shared.ts';
export type { GpuBundleAdjustmentOptions, GpuBundleAdjustmentResult } from './bundle-shared.ts';

/** Keep several scalar groups alive for small camera systems. Cooperative rows use a
 *  smaller tree for shorter observation windows. These widths fit WebGPU's minimum limits;
 *  native timing can refine the policy without adding another lab setting. */
function cameraAssemblyWorkgroup(
  device: ShadeDevice,
  cameras: number,
  observations: number,
  cooperative: boolean,
): number {
  if (device.backend !== 'webgpu') return 64;
  if (cooperative) return observations <= cameras * 256 ? 32 : 64;
  if (!device.occupancyFloors) return 64;
  const rows = cameras * 6;
  const width = rows <= 32 ? 8 : rows <= 64 ? 16 : rows <= 128 ? 32 : 64;
  return Math.ceil(rows / width) <= device.maxKernelElements / 64 ? width : 64;
}

/** Matrix-free point-eliminated Schur PCG, block preconditioning and joint GPU pose/point LM. */
function bundleAdjustGpuFixed(
  device: ShadeDevice,
  input: SparseReconstruction,
  options: GpuBundleAdjustmentOptions = {},
): Promise<GpuBundleAdjustmentResult> {
  validateReconstruction(input);
  const finiteLandmarks = input.landmarks,
    finitePoints = finiteLandmarks.length;
  input = appendBundleDirections(input, options);
  const C = input.views.length,
    P = input.landmarks.length,
    M = input.landmarks.reduce((n, p) => n + p.observations.length, 0);
  if (![C, P, M].every((n) => Number.isSafeInteger(n) && n >= 0 && n <= 0xffffffff))
    throw new RangeError('GPU BA scene dimensions exceed integer index bounds');
  const iterations = options.iterations ?? 20,
    huber = options.huberPx ?? 2,
    maxCg = options.maxCgIterations ?? Math.min(96, Math.max(24, C * 12)),
    tolerance = options.cgRelativeTolerance ?? 1e-6;
  if (
    C < 2 ||
    finitePoints < 1 ||
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
    for (const p of finiteLandmarks)
      if (p.observations.some((o) => connected.has(o.viewId)))
        for (const o of p.observations)
          if (!connected.has(o.viewId)) {
            connected.add(o.viewId);
            grew = true;
          }
  }
  if (input.views.some((v) => !connected.has(v.id)))
    throw new RangeError('GPU BA requires one connected observation graph');
  const sceneSize = 18 * C + 3 * P + 2 * M,
    trialSize = 12 * C + 3 * P,
    indexSize = 3 * M + P + 2 * C + 2,
    cameraScratchSize = 36 * C,
    vectorSize = 30 * C + 3 * P,
    stateSize = 26,
    partialsSize = 3 * Math.max(1, Math.ceil(M / 64)),
    outputSize = trialSize + 26;
  // These logical lengths bound every positive u32 shader offset sum. The sizes of the arrays are
  // not a limit of the problem: the plan below cuts the observation and point sections of `blocks`
  // to the budget (docs/gpu-budget.md, section 3).
  if (
    ![sceneSize, trialSize, 42 * M + 13 * P + 78 * C, vectorSize, indexSize].every(
      (n) => Number.isSafeInteger(n) && n >= 0 && n <= 0xffffffff,
    )
  )
    throw new RangeError('GPU BA array lengths exceed integer index bounds');
  const pointStarts = new Uint32Array(P + 1);
  let longestTrack = 1;
  for (let p = 0; p < P; p++) {
    const n = input.landmarks[p]!.observations.length;
    pointStarts[p + 1] = pointStarts[p]! + n;
    longestTrack = Math.max(longestTrack, n);
  }
  if (C * 6 > device.maxKernelElements)
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
  const cameraWorkgroups =
    device.backend === 'webgpu' &&
    device.fp64Flavor === 'float' &&
    options.cameraProduct === 'workgroups';
  const cameraPlan = cameraWorkgroups
    ? planCameraProduct(
        device,
        cameraLists.map((list) => list.length),
        false,
        options.cameraProductWorkgroupSize,
        options.cameraProductTileSize,
      )
    : undefined;
  const segmented = cameraPlan !== undefined && options.cameraProductTileSize !== undefined;
  const cameraConstants = cameraPlan
    ? { constants: { BA_CAMERA_PRODUCT_WG: cameraPlan.workgroupSize } }
    : undefined;

  if (!finiteGpuArray(source))
    throw new RangeError('GPU BA input exceeds supported coordinate bounds');
  const control = {
    cameras: C,
    points: P,
    directionalPoints: P - finitePoints,
    observations: M,
    anchor: byId.get(anchorId)!,
    scaleView: byId.get(scaleId)!,
    normalizeScale: Number(fixed === 1),
    huber,
    relativeTolerance: tolerance,
  };
  // The units of the key `bundleCg` are rows of one PCG iteration, one per observation and one per
  // camera parameter, so that the rate a device learns of a small bundle sizes the chunks of a
  // larger one: an iteration's dispatches grow with both. The least chunk is one iteration: the
  // minimum wins over what fits the budget, so a larger one could run past it on a slow device.
  const cgWork = M + 6 * C;
  return device.enqueue(async () => {
    const before = device.statistics,
      owned: (Resident<Float64Array> | Resident<Uint32Array>)[] = [];
    const activityWords = device.backend === 'webgpu' && !device.boundedSubmissions ? 3 : 2;
    const cgSchedule = new BundleCgSchedule(activityWords === 3);
    let successful = false;
    // The residency plan (docs/gpu-budget.md, section 3.3): the whole problem when it fits the
    // budget, else windows of whole points. The budget is what the device holds free now.
    const budget = Math.max(0, device.workingBytes - (before.liveBytes - before.pooledBytes));
    const common = [
      sceneSize * 8,
      trialSize * 8,
      vectorSize * 8,
      stateSize * 8,
      partialsSize * 8,
      outputSize * 8,
      indexSize * 4,
      ...(segmented ? [cameraPlan!.tasks.byteLength, cameraPlan!.partialElements * 8] : []),
    ];
    const cameraBlocksBytes = 78 * C * 8,
      scratchBytes = cameraScratchSize * 8,
      sum = (xs: readonly number[]) => xs.reduce((n, x) => n + x, 0);
    const residentArrays = [
        ...common,
        (42 * M + 13 * P + 78 * C) * 8,
        ...(device.backend === 'webgpu' ? [scratchBytes] : []),
      ],
      floorArrays = [
        ...common,
        cameraBlocksBytes,
        scratchBytes,
        ...(cameraPlan ? [cameraPlan.laneElements * 8] : []),
      ];
    const plan = planBundleResidency({
      points: P,
      observations: M,
      pointStarts,
      longestTrack,
      budgetBytes: budget,
      bindingBytes: device.bindingBytes,
      maxKernelElements: device.maxKernelElements,
      residentBytes: sum(residentArrays) + 4 * activityWords,
      residentArrays,
      floorBytes: sum(floorArrays) + 4 * activityWords,
      floorArrays,
      observationBytes: 42 * 8,
      pointBytes: 13 * 8,
      windowArrays: (m, p) => [(42 * m + 13 * p + 78 * C) * 8],
    });
    if (!plan.fits)
      throw new RangeError(
        'GPU BA needs ' +
          plan.bytes +
          ' bytes for the arrays that stay resident and a window of one track, more than the working budget of ' +
          budget +
          ' bytes (docs/gpu-budget.md, 3.6)',
      );
    const streamed = plan.mode === 'streamed',
      blockSize = 42 * plan.obsCapacity + 13 * plan.pointCapacity + 78 * C;
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
        vectors = wide(vectorSize),
        state = wide(stateSize),
        partials = wide(partialsSize),
        cameraLanes = cameraPlan && streamed ? wide(cameraPlan.laneElements) : partials,
        cameraParts = segmented ? wide(cameraPlan!.partialElements) : partials,
        activity = device.allocateU32(activityWords),
        output = wide(outputSize);
      owned.push(activity);
      const cameraTasks = segmented
        ? device.allocateU32(cameraPlan!.tasks.length, cameraPlan!.tasks)
        : index;
      if (segmented) owned.push(cameraTasks);

      const cameraSpecialization = cameraPlan
        ? `Camera${cameraPlan.workgroupSize}Tile${cameraPlan.tileSize ?? 0}`
        : '';
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
                  { control: ctl, scene, trial, index, partials },
                  groups(Math.max(1, n)),
                );
              else
                bundleMetricPartialsOracle(
                  { control: ctl, scene, trial, index, partials },
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
                initializeBundle({ control: full, scene, trial, state }, 1);
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
      // The monitor reads the cost where the first read of the first iteration is, and the cost
      // after an iteration where the iteration's last read is: one round trip each, not two.
      let baselineTaken = false;
      // Bounded submissions read before every chunk, so the cost is read here, as before.
      if (device.boundedSubmissions) {
        await monitor.start();
        baselineTaken = true;
      }
      for (let iteration = 0; iteration < iterations && lmActive; iteration++) {
        cgSchedule.begin();
        let afterIteration: Float64Array | undefined;
        let fetched = false;
        reports.push(
          await device.planSteps(
            (function* () {
              // Assembly: the point blocks and the camera rows. A resident run linearizes once here; a
              // streamed run linearizes each window, here and in every sweep that needs its blocks.
              if (!streamed) {
                yield [
                  'linearizeBundle',
                  () => {
                    linearizeBundle(
                      { control: full, scene, trial, index, blocks, state },
                      groups(M),
                    );
                  },
                ] as const;
                yield [
                  'bundlePointBlocks',
                  () => {
                    bundlePointBlocks({ control: full, index, blocks, state }, groups(P));
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
              } else {
                if (!cameraScratch) throw new Error('GPU BA camera scratch was not allocated');
                for (let k = 0; k < (streamed ? plan.windows.length : 1); k++) {
                  const ctl = streamed ? windowControl(k, 0) : full,
                    w = plan.windows[k]!;
                  if (streamed) {
                    yield [
                      'linearizeBundle',
                      () => {
                        linearizeBundle(
                          { control: ctl, scene, trial, index, blocks, state },
                          groups(w.obsEnd - w.obsBase),
                        );
                      },
                    ] as const;
                    yield [
                      'bundlePointBlocks',
                      () => {
                        bundlePointBlocks(
                          { control: ctl, index, blocks, state },
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
                  const cooperative = cooperativeBundleSchur(
                    device,
                    C,
                    M,
                    false,
                    options.schurReduction,
                  );
                  const rowWorkgroup = cameraAssemblyWorkgroup(
                    device,
                    C,
                    w.obsEnd - w.obsBase,
                    cooperative,
                  );
                  yield [
                    cooperative ? 'bundleCameraRowsCooperative' : 'bundleCameraRows',
                    () => {
                      (cooperative ? bundleCameraRowsCooperative : bundleCameraRows)(
                        { control: ctl, index, blocks, state, cameraScratch },
                        cooperative ? C * 6 : Math.ceil((C * 6) / rowWorkgroup),
                        rowWorkgroup === 64
                          ? undefined
                          : {
                              constants: cooperative
                                ? { BA_CAMERA_REDUCE_WG: rowWorkgroup }
                                : { BA_CAMERA_ROWS_WG: rowWorkgroup },
                            },
                      );
                    },
                  ] as const;
                }
                yield [
                  'invertBundleCameraColumns',
                  () => {
                    invertBundleCameraColumns(
                      { control: full, blocks, cameraScratch, state },
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
                'initializeBundleCg',
                () => {
                  initializeBundleCg({ control: full, blocks, vectors, state }, 1);
                },
              ] as const;
              let cgActive = true;
              const snapshotActivity = async () => {
                const flags = await device.read(activity, activityWords);
                cgActive = flags[0] !== 0;
                lmActive = flags[1] !== 0;
                cgSchedule.observe(flags[2], cgActive);
              };
              // The first read of the first iteration also reads the cost before any step.
              const firstActivity = async () => {
                if (baselineTaken) return snapshotActivity();
                baselineTaken = true;
                const [, values] = await Promise.all([snapshotActivity(), monitor.fetch()]);
                monitor.begin(values);
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
                false,
                options.schurReduction,
              );
              const schurCameraSteps = function* (ctl: typeof full, scale: number) {
                if (!cameraWorkgroups) {
                  yield [
                    'bundleSchurCamera',
                    () =>
                      (cooperative ? bundleSchurCameraCooperative : bundleSchurCamera)(
                        { control: ctl, index, blocks, vectors, state },
                        scale * (cooperative ? C * 6 : groups(C * 6)),
                      ),
                  ] as const;
                  return;
                }
                const taskCount = cameraPlan!.taskCount;
                for (let at = 0; at < taskCount; at += device.maxWorkgroups) {
                  const last = at + device.maxWorkgroups >= taskCount;
                  yield [
                    !segmented && last ? 'bundleSchurCamera' : 'bundleSchurCameraPartial',
                    () => {
                      const bindings = {
                        control: { ...ctl, base: at },
                        index,
                        blocks,
                        vectors,
                        state,
                        cameraLanes,
                      };
                      const count = scale * Math.min(device.maxWorkgroups, taskCount - at);
                      if (segmented)
                        bundleSchurCameraSegments(
                          { ...bindings, cameraTasks, cameraParts },
                          count,
                          cameraConstants,
                        );
                      else bundleSchurCameraWorkgroup({ ...bindings }, count, cameraConstants);
                    },
                  ] as const;
                }
                if (segmented && ctl.last !== 0)
                  for (let at = 0; at < C; at += piece) {
                    const last = at + piece >= C;
                    yield [
                      last ? 'bundleSchurCamera' : 'bundleSchurCameraPartial',
                      () =>
                        finishBundleSchurCameraSegments(
                          {
                            control: { ...ctl, base: at },
                            blocks,
                            vectors,
                            state,
                            cameraTasks,
                            cameraParts,
                          },
                          scale * groups(Math.min(piece, C - at)),
                        ),
                    ] as const;
                  }
              };
              const schurCamera = (ctl: typeof full, scale: number) => {
                for (const [, run] of schurCameraSteps(ctl, scale)) run();
              };
              const windowSweep = (k: number, scale: number, last: boolean) => {
                const ctl = windowControl(k, 1),
                  w = plan.windows[k]!;
                linearizeBundle(
                  { control: ctl, scene, trial, index, blocks, state },
                  scale * groups(w.obsEnd - w.obsBase),
                );
                bundlePointBlocks(
                  { control: ctl, index, blocks, state },
                  scale * groups(w.pointEnd - w.pointBase),
                );
                schurPoint(ctl, scale, groups(w.pointEnd - w.pointBase));
                schurCamera(ctl, scale);
                if (last) advanceBundleCg({ control: full, blocks, vectors, state }, scale);
              };
              // One PCG iteration; `scale` 0 dispatches its entries over no workgroups (the warm-up
              // of `GpuChunk.warm`).
              const cgIteration = (scale = 1) => {
                if (streamed) {
                  for (let k = 0; k < plan.windows.length; k++)
                    windowSweep(k, scale, k === plan.windows.length - 1);
                  return;
                }
                schurPoint(full, scale, groups(P));
                schurCamera(full, scale);
                advanceBundleCg({ control: full, blocks, vectors, state }, scale);
              };
              if (!device.boundedSubmissions) {
                // The PCG loop runs under the time budget (`docs/gpu-sfm.md`, "The BA's PCG
                // loop"): a chunk is `n` iterations of this LM iteration, one plan that the
                // TypeShade device fences and times (key `bundleCg`), and it ends with the flags,
                // which the host reads before it sizes the next chunk or stops. The iterations
                // are the dispatches of the per-iteration path in the same order; an iteration
                // past convergence is a guarded no-op, so skipping it changes no result. The CPU
                // tier runs the whole loop as one chunk. A streamed iteration costs more per
                // unit, so it learns the rate of a key of its own.
                // No read before the first chunk. The LM flag cannot change between the last read of
                // the previous iteration and here, every kernel of an inactive iteration is a
                // guarded no-op, and so is a CG iteration past convergence. A read costs a round
                // trip to the GPU (Amendment 8 of docs/multi-video.md).
                const key = (streamed ? 'bundleCgStreamed' : 'bundleCg') + cameraSpecialization;
                for (let k = 0; k < maxCg && cgActive;) {
                  const remaining = maxCg - k,
                    proposed = Math.floor(
                      device.chunkSize(key, remaining * cgWork, cgWork) / cgWork,
                    ),
                    n = cgSchedule.next(proposed, k);
                  yield [
                    'bundleCg',
                    () => {
                      for (let i = 0; i < n; i++) cgIteration();
                      snapshotBundleActivity({ state, activity }, 1);
                    },
                    firstActivity,
                    {
                      key,
                      units: n * cgWork,
                      // The chunk ends with the flags' entry.
                      warm: {
                        id: (streamed ? 'bundleStreamed' : 'bundle') + cameraSpecialization,
                        run: () => cgIteration(0),
                      },
                    },
                  ] as const;
                  k += n;
                }
              } else {
                // Bounded submissions keep one step per dispatch, so that no plan holds more than
                // `maxDispatchesPerSubmission` dispatches, and read the flags every four
                // iterations with solver checkpoints.
                if (device.solverCheckpoints) yield snapshot;
                for (let k = 0; k < maxCg && cgActive; k++) {
                  if (streamed) {
                    for (let w = 0; w < plan.windows.length; w++) {
                      const ctl = windowControl(w, 1),
                        win = plan.windows[w]!;
                      yield [
                        'linearizeBundle',
                        () => {
                          linearizeBundle(
                            { control: ctl, scene, trial, index, blocks, state },
                            groups(win.obsEnd - win.obsBase),
                          );
                        },
                      ] as const;
                      yield [
                        'bundlePointBlocks',
                        () => {
                          bundlePointBlocks(
                            { control: ctl, index, blocks, state },
                            groups(win.pointEnd - win.pointBase),
                          );
                        },
                      ] as const;
                      yield [
                        'bundleSchurPoint',
                        () => {
                          schurPoint(ctl, 1, groups(win.pointEnd - win.pointBase));
                        },
                      ] as const;
                      yield* schurCameraSteps(ctl, 1);
                    }
                  } else {
                    yield [
                      'bundleSchurPoint',
                      () => {
                        schurPoint(full, 1, groups(P));
                      },
                    ] as const;
                    yield* schurCameraSteps(full, 1);
                  }
                  yield [
                    'advanceBundleCg',
                    () => {
                      advanceBundleCg({ control: full, blocks, vectors, state }, 1);
                    },
                  ] as const;
                  if (device.solverCheckpoints && (k + 1) % 4 === 0 && k + 1 < maxCg)
                    yield snapshot;
                }
              }
              // The update: the cameras and the points of the candidate.
              for (let k = 0; k < (streamed ? plan.windows.length : 1); k++) {
                const ctl = streamed ? windowControl(k, 2) : full,
                  w = plan.windows[k]!;
                if (streamed) {
                  yield [
                    'linearizeBundle',
                    () => {
                      linearizeBundle(
                        { control: ctl, scene, trial, index, blocks, state },
                        groups(w.obsEnd - w.obsBase),
                      );
                    },
                  ] as const;
                  yield [
                    'bundlePointBlocks',
                    () => {
                      bundlePointBlocks(
                        { control: ctl, index, blocks, state },
                        groups(w.pointEnd - w.pointBase),
                      );
                    },
                  ] as const;
                }
                yield [
                  'updateBundleCandidate',
                  () => {
                    updateBundleCandidate(
                      { control: ctl, scene, trial, index, blocks, vectors, state },
                      groups(Math.max(k === 0 ? C : 0, w.pointEnd - w.pointBase)),
                    );
                  },
                ] as const;
              }
              yield [
                'bundleCandidateGauge',
                () => {
                  bundleCandidateGauge({ control: full, scene, trial, state }, 1);
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
                        { control: ctl, scene, trial, index, partials },
                        groups(Math.max(1, n)),
                      );
                    else
                      bundleCandidatePartialsOracle(
                        { control: ctl, scene, trial, index, partials },
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
                commitBundleCandidate({ control: ctl, scene, trial, state }, count);
              });
              if (device.solverCheckpoints)
                yield [
                  snapshot[0],
                  snapshot[1],
                  async () => {
                    // The flags and the cost of the monitor come in one round trip.
                    const [, values] = await Promise.all([snapshotActivity(), monitor.fetch()]);
                    afterIteration = values;
                    fetched = true;
                  },
                ] as const;
            })(),
          ),
        );
        // A stall ends the LM loop (`stallTolerance`); the shaders' own rule may end it earlier.
        if (!fetched) afterIteration = await monitor.fetch();
        if (lmActive && monitor.decide(afterIteration)) lmActive = false;
      }
      reports.push(
        await device.planSteps(
          (function* () {
            yield* pieces('packBundleResult', outputSize, (ctl, count) => {
              packBundleResult({ control: ctl, scene, state, output }, count);
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
      const reconstruction: SparseReconstruction = {
        views: model.views.map((v, c) => ({
          ...v,
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
      return separateBundleDirections(
        {
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
        },
        finitePoints,
      );
    } finally {
      for (const b of owned) successful ? device.recycle(b) : device.drop(b);
    }
  });
}

/** Jointly optimize one log-focal scale, camera poses and points on the GPU. */
export function bundleAdjustGpu(
  device: ShadeDevice,
  input: SparseReconstruction,
  options: GpuBundleAdjustmentOptions = {},
): Promise<GpuBundleAdjustmentResult> {
  if (
    options.cameraProductWorkgroupSize !== undefined &&
    ![32, 64, 128, 256, 'auto'].includes(options.cameraProductWorkgroupSize)
  )
    throw new RangeError('cameraProductWorkgroupSize must be 32, 64, 128, 256 or auto');
  if (
    options.cameraProductTileSize !== undefined &&
    (!Number.isSafeInteger(options.cameraProductTileSize) ||
      options.cameraProductTileSize < 1 ||
      options.cameraProductTileSize > 0xffffffff)
  )
    throw new RangeError('cameraProductTileSize must be a positive u32 integer');
  if (
    options.cameraProduct !== undefined &&
    !['workgroups', 'rows'].includes(options.cameraProduct)
  )
    throw new RangeError("cameraProduct must be 'workgroups' or 'rows'");
  if (options.schurCache !== undefined && typeof options.schurCache !== 'boolean')
    throw new RangeError('schurCache must be a boolean');
  if (
    options.schurReduction !== undefined &&
    !['auto', 'serial', 'cooperative'].includes(options.schurReduction)
  )
    throw new RangeError('Invalid Schur reduction mode');
  if (!options.refineSharedFocal) return bundleAdjustGpuFixed(device, input, options);
  if (
    options.focalGroups &&
    new Set(input.views.map((v) => options.focalGroups!.get(v.id))).size > 1
  )
    throw new RangeError('Per-group focal refinement is not implemented on the GPU');
  return bundleAdjustJointFocalGpu(device, input, options);
}
