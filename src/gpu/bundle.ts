import type { Resident } from 'typeshade/runtime';
import {
  validateReconstruction,
  type BundleAdjustmentOptions,
  type BundleAdjustmentResult,
} from '../reconstruction/bundle-adjustment.ts';
import type { SparseReconstruction } from '../reconstruction/types.ts';
import { ShadeDevice, type ShadePlanReport } from './device.ts';
import { finiteGpuArray, validateGpuCamera, type GpuGeometryExecution } from './pnp.ts';
import { bundleAdjustJointFocalGpu } from './joint-focal.ts';
import {
  initializeBundle,
  linearizeBundle,
  initializeBundleMetrics,
  bundlePointBlocks,
  bundleCameraBlocks,
  bundleCameraRows,
  invertBundleCameraColumns,
  finalizeBundleCameraInverse,
  initializeBundleCg,
  bundleSchurPoint,
  bundleSchurCamera,
  bundlePersistentCg,
  advanceBundleCg,
  snapshotBundleActivity,
  updateBundleCandidate,
  bundleCandidateGauge,
  normalizeBundleCandidate,
  bundleCandidateMetrics,
  acceptBundleCandidate,
  commitBundleCandidate,
  packBundleResult,
} from './bundle.shade.ts';

export interface GpuBundleAdjustmentOptions extends BundleAdjustmentOptions {
  readonly maxCgIterations?: number;
  readonly cgRelativeTolerance?: number;
}
export interface GpuBundleAdjustmentResult extends BundleAdjustmentResult {
  readonly cgIterations: number;
  readonly linearSolverFailures: number;
  readonly execution: GpuGeometryExecution;
}

/** Matrix-free point-eliminated Schur PCG, block preconditioning and joint GPU pose/point LM. */
function bundleAdjustGpuFixed(
  device: ShadeDevice,
  input: SparseReconstruction,
  options: GpuBundleAdjustmentOptions = {},
): Promise<GpuBundleAdjustmentResult> {
  validateReconstruction(input);
  const C = input.views.length,
    P = input.landmarks.length,
    M = input.landmarks.reduce((n, p) => n + p.observations.length, 0);
  const iterations = options.iterations ?? 20,
    huber = options.huberPx ?? 2,
    maxCg = options.maxCgIterations ?? Math.min(96, Math.max(24, C * 12)),
    tolerance = options.cgRelativeTolerance ?? 1e-6;
  if (
    C < 2 ||
    C > 64 ||
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
    throw new RangeError('Invalid GPU BA scene/options (2..64 cameras)');
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
  const usePersistentCg =
    device.backend === 'webgpu' && !device.boundedSubmissions && M * maxCg <= 262144;
  const sceneSize = 18 * C + 3 * P + 2 * M,
    trialSize = 12 * C + 3 * P,
    indexSize = 3 * M + P + 2 * C + 2,
    blockSize = 42 * M + 13 * P + 78 * C,
    cameraScratchSize = device.backend === 'webgpu' ? 36 * C : 0,
    vectorSize = 30 * C + 3 * P,
    stateSize = 26 + 2 * M,
    outputSize = trialSize + 26;
  const sizes = [
    sceneSize,
    trialSize,
    blockSize,
    cameraScratchSize,
    vectorSize,
    stateSize,
    outputSize,
  ];
  if (
    sizes.some((n) => n * 8 > device.bindingBytes) ||
    indexSize * 4 > device.bindingBytes ||
    sizes.reduce((n, s) => n + s * 8, indexSize * 4 + 8) > device.workingBytes ||
    Math.max(M, P, C * 6, trialSize, outputSize) > device.maxKernelElements
  )
    throw new RangeError('GPU BA exceeds working/storage/dispatch budget');
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
  return device.enqueue(async () => {
    const before = device.statistics,
      owned: (Resident<Float64Array> | Resident<Uint32Array>)[] = [];
    let successful = false;
    const wide = (n: number, v?: Float64Array) => {
      const b = device.allocateF64(n, v);
      owned.push(b);
      return b;
    };
    const reports: ShadePlanReport[] = [];
    try {
      const scene = wide(sceneSize, source),
        trial = wide(trialSize),
        index = device.allocateU32(indexSize, indices);
      owned.push(index);
      const blocks = wide(blockSize),
        cameraScratch = cameraScratchSize > 0 ? wide(cameraScratchSize) : undefined,
        vectors = wide(vectorSize),
        state = wide(stateSize),
        activity = device.allocateU32(2),
        output = wide(outputSize);
      owned.push(activity);
      reports.push(
        await device.planSteps(
          (function* () {
            yield [
              'initializeBundle',
              () => {
                initializeBundle({ control, scene, trial, state }, 1);
              },
            ] as const;
            yield [
              'linearizeBundle',
              () => {
                linearizeBundle({ control, scene, trial, index, blocks, state }, Math.ceil(M / 64));
              },
            ] as const;
            yield [
              'initializeBundleMetrics',
              () => {
                initializeBundleMetrics({ control, blocks, state }, 1);
              },
            ] as const;
          })(),
        ),
      );
      let lmActive = true;
      for (let iteration = 0; iteration < iterations && lmActive; iteration++)
        reports.push(
          await device.planSteps(
            (function* () {
              if (iteration)
                yield [
                  'linearizeBundle',
                  () => {
                    linearizeBundle(
                      { control, scene, trial, index, blocks, state },
                      Math.ceil(M / 64),
                    );
                  },
                ] as const;
              yield [
                'bundlePointBlocks',
                () => {
                  bundlePointBlocks({ control, index, blocks, state }, Math.ceil(P / 64));
                },
              ] as const;
              if (device.backend === 'cpu') {
                yield [
                  'bundleCameraBlocks',
                  () => {
                    bundleCameraBlocks({ control, index, blocks, state }, Math.ceil(C / 64));
                  },
                ] as const;
              } else {
                if (!cameraScratch) throw new Error('GPU BA camera scratch was not allocated');
                yield [
                  'bundleCameraRows',
                  () => {
                    bundleCameraRows(
                      { control, index, blocks, state, cameraScratch },
                      Math.ceil((C * 6) / 64),
                    );
                  },
                ] as const;
                yield [
                  'invertBundleCameraColumns',
                  () => {
                    invertBundleCameraColumns(
                      { control, blocks, cameraScratch, state },
                      Math.ceil((C * 6) / 64),
                    );
                  },
                ] as const;
                yield [
                  'finalizeBundleCameraInverse',
                  () => {
                    finalizeBundleCameraInverse({ control, blocks, state }, Math.ceil(C / 64));
                  },
                ] as const;
              }
              yield [
                'initializeBundleCg',
                () => {
                  initializeBundleCg({ control, blocks, vectors, state }, 1);
                },
              ] as const;
              let cgActive = true;
              const snapshotActivity = async () => {
                const flags = await device.read(activity, 2);
                cgActive = flags[0] !== 0;
                lmActive = flags[1] !== 0;
              };
              if (usePersistentCg) {
                yield [
                  'bundlePersistentCg',
                  () => {
                    bundlePersistentCg({ control, index, blocks, vectors, state }, 1);
                  },
                ] as const;
              } else {
                if (device.solverCheckpoints)
                  yield [
                    'snapshotBundleActivity',
                    () => {
                      snapshotBundleActivity({ state, activity }, 1);
                    },
                    snapshotActivity,
                  ] as const;
                for (let k = 0; k < maxCg && cgActive; k++) {
                  yield [
                    'bundleSchurPoint',
                    () => {
                      bundleSchurPoint(
                        { control, index, blocks, vectors, state },
                        Math.ceil(P / 64),
                      );
                    },
                  ] as const;
                  yield [
                    'bundleSchurCamera',
                    () => {
                      bundleSchurCamera(
                        { control, index, blocks, vectors, state },
                        Math.ceil((C * 6) / 64),
                      );
                    },
                  ] as const;
                  yield [
                    'advanceBundleCg',
                    () => {
                      advanceBundleCg({ control, blocks, vectors, state }, 1);
                    },
                  ] as const;
                  if (device.solverCheckpoints && (k + 1) % 4 === 0 && k + 1 < maxCg)
                    yield [
                      'snapshotBundleActivity',
                      () => {
                        snapshotBundleActivity({ state, activity }, 1);
                      },
                      snapshotActivity,
                    ] as const;
                }
              }
              yield [
                'updateBundleCandidate',
                () => {
                  updateBundleCandidate(
                    { control, scene, trial, index, blocks, vectors, state },
                    Math.ceil(Math.max(C, P) / 64),
                  );
                },
              ] as const;
              yield [
                'bundleCandidateGauge',
                () => {
                  bundleCandidateGauge({ control, scene, trial, state }, 1);
                },
              ] as const;
              yield [
                'normalizeBundleCandidate',
                () => {
                  normalizeBundleCandidate(
                    { control, scene, trial, state },
                    Math.ceil(Math.max(C, P) / 64),
                  );
                },
              ] as const;
              yield [
                'bundleCandidateMetrics',
                () => {
                  bundleCandidateMetrics(
                    { control, scene, trial, index, state },
                    Math.ceil(M / 64),
                  );
                },
              ] as const;
              yield [
                'acceptBundleCandidate',
                () => {
                  acceptBundleCandidate({ control, state }, 1);
                },
              ] as const;
              yield [
                'commitBundleCandidate',
                () => {
                  commitBundleCandidate({ scene, trial, state }, Math.ceil(trialSize / 64));
                },
              ] as const;
              if (device.solverCheckpoints)
                yield [
                  'snapshotBundleActivity',
                  () => {
                    snapshotBundleActivity({ state, activity }, 1);
                  },
                  snapshotActivity,
                ] as const;
            })(),
          ),
        );
      reports.push(
        await device.planSteps(
          (function* () {
            yield [
              'packBundleResult',
              () => {
                packBundleResult({ control, scene, state, output }, Math.ceil(outputSize / 64));
              },
            ] as const;
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
      const after = device.statistics,
        report: ShadePlanReport = {
          tier: device.backend,
          dispatches: reports.reduce((n, r) => n + r.dispatches, 0),
          submissions: reports.reduce((n, r) => n + r.submissions, 0),
          groups: reports.flatMap((r) => r.groups),
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

/** Jointly optimize one log-focal scale, camera poses and points on the GPU. */
export function bundleAdjustGpu(
  device: ShadeDevice,
  input: SparseReconstruction,
  options: GpuBundleAdjustmentOptions = {},
): Promise<GpuBundleAdjustmentResult> {
  if (!options.refineSharedFocal) return bundleAdjustGpuFixed(device, input, options);
  return bundleAdjustJointFocalGpu(device, input, options);
}
