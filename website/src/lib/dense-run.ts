import { DepthArchive } from './depth-archive.ts';
import { densePerformanceRecord } from './run-record.ts';
/**
 * The lab worker's dense MVS (`docs/dense-mvs.md` 5.3, step 6 of section 8): the RGBA it keeps
 * for each keyframe, the run of `reconstructDenseGpu` after the sparse result, and the messages it
 * posts. The worker (`website/src/workers/sfm.worker.ts`) owns the device and the job. This module
 * takes them as arguments, so that `website/dense-run.test.mjs` runs it with a fake MVS.
 */

/** The dense MVS (`docs/dense-mvs.md` 5.3): the defaults of section 3, for both lab workers. */
export const DENSE_LAB_OPTIONS = {
  maxSources: 6,
  bestSources: 3,
  windowRadius: 5,
  windowStep: 2,
  photometricIterations: 8,
  geometricIterations: 4,
  minConsistentViews: 3,
  tsdf: { visibility: 'view-weighted' },
  planarDepth: true,
} as const;

/**
 * The fusion of the lab (`docs/dense-mvs.md` 10.4, the step 3 record): the TSDF fusion, which beat
 * the per-pixel fusion of 3.10 on ETH3D `pipes` in F-score at 2 and 5 cm and in landmark coverage.
 * `?denseFusion=pixel` runs the fusion of 3.10.
 */
export const DENSE_LAB_FUSION: DenseFusion = 'tsdf';

/** The fusions of `reconstructDenseGpu` (`fusion`). */
export type DenseFusion = 'pixel' | 'tsdf';

/** Only enabled features of the actual native Dense device establish timestamp mode. Adapter
 * support, requested options and the presence of timed groups are different measurements. */
export function denseGpuTimestampFields(device: unknown): { gpuTimestamps?: boolean } {
  try {
    const features = (device as { info?: { webgpu?: { features?: unknown } } } | undefined)?.info
      ?.webgpu?.features;
    return Array.isArray(features) && features.every((feature) => typeof feature === 'string')
      ? { gpuTimestamps: features.includes('timestamp-query') }
      : {};
  } catch {
    return {};
  }
}

/**
 * Whether the lab cleans the fused cloud (3.12) by default: after the pixel fusion only. On
 * `pipes` the cleanup lowered the TSDF cloud's F-score at every threshold and raised its uncovered
 * landmarks (the step 3 record), as it did the pixel cloud's. `?denseClean=1` or `0` overrides it.
 */
export function denseCleanDefault(fusion: DenseFusion): boolean {
  return fusion === 'pixel';
}

/**
 * The cleanup of the fused cloud (3.12). The outlier and voxel stages take the defaults of
 * `cleanDenseCloudGpu` (k 16, 2 σ, a voxel of twice the median spacing). DBSCAN runs at 4
 * spacings and keeps the largest cluster and every cluster of at least 10 % of it.
 */
export const DENSE_CLEAN_OPTIONS = { cluster: { minFraction: 0.1 } } as const;

/** The stages of the MVS, in their order (5.3). */
export type DenseStage = 'photometric' | 'geometric' | 'fusion';

/** One kept image: the RGBA bytes of a keyframe at the analysis size. */
export type DenseImage = { rgba: Uint8ClampedArray; width: number; height: number };

/** What a run with the dense MVS keeps from its start to `denseResult`. */
export type DenseJob = {
  /** The RGBA of each keyframe at the analysis size, 4 B per pixel (4.2), with no view limit. */
  rgba: Map<number, DenseImage>;
  /** Why the run did not start the MVS, or empty. */
  skipped: string;
  abort: AbortController;
  /** True from the start of the MVS to `denseResult`: a failure then ends the MVS only. */
  running: boolean;
  /** The stage that runs now, for a failure. */
  stage: DenseStage | 'clean' | 'mesh' | '';
  /** True once `denseResult` is posted. */
  ended: boolean;
  /** Clean the fused cloud after the fusion (section 3.12). Off with `?denseClean=0`. */
  clean: boolean;
  /** The fusion (3.10 or 10.4): `DENSE_LAB_FUSION` unless `?denseFusion=pixel`. */
  fusion: DenseFusion;
  /** Mesh the final cloud (section 3.13). On with `?denseMesh=1`. */
  mesh: boolean;
};

export function createDenseJob(
  clean = true,
  mesh = false,
  fusion: DenseFusion = DENSE_LAB_FUSION,
): DenseJob {
  return {
    clean,
    mesh,
    fusion,
    rgba: new Map(),
    skipped: '',
    abort: new AbortController(),
    running: false,
    stage: '',
    ended: false,
  };
}

/**
 * The device's working budget for a run with the dense MVS (`docs/dense-mvs.md`, Amendment 3,
 * item 5): the larger of the SfM's budget (`sfmBytes`, which scales with the analysis edge) and
 * half of the device's `maxBufferSize`, the conservative fraction. WebGPU reports no memory size,
 * and `maxBufferSize` is the largest buffer the device says it can make. The MVS sizes every array
 * from this budget (`planMvsResidency`), so no view count enters it.
 */
export function denseWorkingBytes(sfmBytes: number, maxBufferSize: number | undefined): number {
  return Math.max(sfmBytes, Math.floor((maxBufferSize ?? 0) / 2));
}

/**
 * Keep a frame's RGBA until the MVS (5.3, item 3). The caller deletes the entry of a frame that
 * is no keyframe. Amendment 3: the MVS keeps a window of views on the device, so the map has no
 * view limit.
 */
export function keepFrame(
  dense: DenseJob,
  frame: { id: number; width: number; height: number; pixels: Uint8ClampedArray },
): void {
  dense.rgba.set(frame.id, { rgba: frame.pixels, width: frame.width, height: frame.height });
}

/**
 * The mean color of each block of a depth preview's grid (`docs/dense-mvs.md`, Amendment 5, rule
 * 3): `width` × `height` blocks over the kept image, r g b per block. A block covers the pixels
 * that its share of the image holds, at least one.
 */
export function blockColors(image: DenseImage, width: number, height: number): Uint8Array {
  const rgb = new Uint8Array(width * height * 3);
  for (let by = 0; by < height; by++) {
    const y0 = Math.floor((by * image.height) / height),
      y1 = Math.max(y0 + 1, Math.floor(((by + 1) * image.height) / height));
    for (let bx = 0; bx < width; bx++) {
      const x0 = Math.floor((bx * image.width) / width),
        x1 = Math.max(x0 + 1, Math.floor(((bx + 1) * image.width) / width));
      let r = 0,
        g = 0,
        b = 0,
        n = 0;
      for (let y = y0; y < Math.min(y1, image.height); y++)
        for (let x = x0; x < Math.min(x1, image.width); x++) {
          const at = 4 * (y * image.width + x);
          r += image.rgba[at]!;
          g += image.rgba[at + 1]!;
          b += image.rgba[at + 2]!;
          n++;
        }
      const at = 3 * (by * width + bx);
      if (n) rgb.set([Math.round(r / n), Math.round(g / n), Math.round(b / n)], at);
    }
  }
  return rgb;
}

/** The failure of the MVS (5.3, item 7): the message, the stage, the kernel and the reason. */
export function denseFailure(
  dense: DenseJob,
  message: string,
  kernel: string,
  reason: string,
  context?: import('./gpu-loss-context').GpuLossContext,
) {
  return {
    message: message.slice(0, 2000),
    stage: dense.stage,
    kernel: kernel.slice(0, 64),
    reason: reason.slice(0, 64),
    ...(context ? { context } : {}),
  };
}

/** The arrays of the fused cloud (`GpuDensePoints`). */
export type DenseCleanInput = {
  count: number;
  positions: Float32Array;
  colors: Uint8Array;
  normals: Int16Array;
};

/** What the worker gives a dense run. */
export interface DenseRunContext {
  readonly component?: number;
  readonly dense: DenseJob;
  /** The MVS options, `LAB_OPTIONS.dense` of the worker. */
  readonly options: Readonly<Record<string, unknown>>;
  /** The device's byte counts now (`ShadeDevice.statistics`). */
  readonly statistics: () => { liveBytes: number; pooledBytes: number };
  /** `reconstructDenseGpu` bound to the device. */
  readonly reconstruct: (scene: any, images: Map<number, DenseImage>, options: any) => Promise<any>;
  /**
   * The cleanup of the fused cloud (section 3.12): `cleanDenseCloudGpu` bound to the device and
   * the lab's options. Absent in a test that runs without it.
   */
  readonly clean?: (points: DenseCleanInput) => Promise<{
    cloud: DenseCleanInput;
    report: Record<string, any>;
  }>;
  /**
   * The mesh of the final cloud (section 3.13): `meshDenseCloudGpu` bound to the device. Absent in
   * a test that runs without it.
   */
  readonly mesh?: (points: DenseCleanInput) => Promise<{
    mesh: {
      positions: Float32Array;
      colors: Uint8Array;
      indices: Uint32Array;
    };
    report: Record<string, any>;
  }>;
  /** Post a message to the page, with the transfer list. The worker adds the job id. */
  readonly post: (message: Record<string, unknown>, transfer?: Transferable[]) => void;
  /** The extra fields of `denseResult`: the kernels and the chunk rates when it is posted. */
  readonly finalFields: () => Record<string, unknown>;
  /** The kernel that started last, for a failure. */
  readonly lastKernel: () => string;
}

/** Post `denseResult` once (5.3): the end of the MVS, completed, cancelled, failed or skipped. */
export function endDense(context: DenseRunContext, outcome: Record<string, unknown>): void {
  const { dense } = context;
  if (dense.ended) return;
  dense.ended = true;
  dense.running = false;
  dense.rgba.clear();
  // A failure from outside the run (a device loss, an uncaught error) stops the run too.
  dense.abort.abort();
  context.post({
    type: 'denseResult',
    cancelled: false,
    failure: null,
    skipped: '',
    points: 0,
    views: [],
    ...outcome,
    ...context.finalFields(),
    options: {
      ...context.options,
      fusion: dense.fusion,
      denseClean: dense.clean,
      ...(context.options.planarDepth !== undefined
        ? { planarDepth: dense.fusion === 'tsdf' && context.options.planarDepth === true }
        : {}),
      ...(context.mesh ? { denseMesh: dense.mesh } : {}),
    },
  });
}

/**
 * The dense MVS after the sparse result (5.3, items 4 to 7): `reconstruct` over the registered
 * views whose RGBA the run kept. It posts `denseProgress` after each reference of each stage,
 * `denseDepth` per reference and stage with its block colors (Amendment 5), `densePoints` per reference and `denseResult` at the end, with the arrays
 * transferred, not copied. The device's bytes before the first MVS allocation, at the depth
 * stage's peak and at the fusion stage go to `denseResult.memory` (4.2, 4.3).
 */
export async function runDense(
  context: DenseRunContext,
  model: { views: readonly any[]; landmarks: readonly any[] },
): Promise<void> {
  const { dense } = context;
  dense.running = true;
  const before = context.statistics();
  const activeBytes = () => {
    const s = context.statistics();
    return s.liveBytes - s.pooledBytes;
  };
  const memory = {
    liveBytesBefore: before.liveBytes,
    pooledBytesBefore: before.pooledBytes,
    depthStageBytes: null as number | null,
    depthStageFormulaBytes: null as number | null,
    fusionStageBytes: null as number | null,
  };
  const started = performance.now();
  let activityKey = '',
    phaseKey = '',
    activityAt = -Infinity,
    phaseAt = started;
  const activity = (event: {
    phase: string;
    done: number;
    total: number;
    phaseDone?: number;
    phaseTotal?: number;
    viewId?: number;
    view?: number;
    pass?: number;
    passes?: number;
    views?: number;
    points?: number;
  }) => {
    const now = performance.now();
    const key = `${event.phase}:${event.pass ?? 0}:${event.viewId ?? -1}`;
    const phase = `${event.phase}:${event.pass ?? 0}`;
    if (phase !== phaseKey) phaseAt = now;
    phaseKey = phase;
    if (key === activityKey && event.done !== event.total && now - activityAt < 200) return;
    activityKey = key;
    activityAt = now;
    context.post({
      type: 'denseActivity',
      ...event,
      elapsedMs: now - started,
      phaseMs: now - phaseAt,
    });
  };
  const stageMs: Record<DenseStage, number> = { photometric: 0, geometric: 0, fusion: 0 };
  let mark = started,
    points = 0;
  const timing = () => ({ totalMs: performance.now() - started, stageMs: { ...stageMs } });
  const images = new Map<number, DenseImage>();
  for (const view of model.views) {
    const image = dense.rgba.get(view.id);
    if (image) images.set(view.id, image);
  }
  if (dense.skipped || !images.size) {
    endDense(context, {
      skipped: dense.skipped || 'No registered view kept its image',
      memory,
      timing: timing(),
    });
    return;
  }
  const scene = {
    views: model.views
      .filter((view) => images.has(view.id))
      .map((view) => ({ id: view.id, pose: view.pose, camera: view.camera })),
    landmarks: model.landmarks,
  };
  const archive = new DepthArchive(context.component ?? 0, ({ name, data }) => {
    context.post(
      { type: 'denseArtifact', name, data },
      typeof data === 'string' ? [] : [data.buffer],
    );
  });
  const finishArchive = (
    outcome: 'completed' | 'cancelled' | 'failed',
    expectedViews = scene.views.map((v) => v.id),
  ) =>
    archive.finish(
      outcome,
      expectedViews,
      dense.fusion === 'tsdf' && context.options.planarDepth === true,
    );
  dense.stage = 'photometric';
  try {
    activity({ phase: 'photometric', done: 0, total: scene.views.length });
    const result = await context.reconstruct(scene, images, {
      ...context.options,
      fusion: dense.fusion,
      ...(context.options.planarDepth !== undefined
        ? { planarDepth: dense.fusion === 'tsdf' && context.options.planarDepth === true }
        : {}),
      signal: dense.abort.signal,
      onDepthDiagnostic: (event: import('../../../src/gpu/mvs.ts').GpuDenseDepthDiagnostic) => {
        const image = images.get(event.viewId);
        return archive.capture(
          event,
          image
            ? { ...image, camera: scene.views.find((v) => v.id === event.viewId)?.camera }
            : undefined,
        );
      },
      onFusionProgress: (event: Parameters<typeof activity>[0]) => {
        dense.stage = 'fusion';
        activity(event);
      },
      onFusionPreview: (event: any) => {
        const { positions, colors, normals } = event;
        context.post({ type: 'denseFusionPoints', viewId: -1, positions, colors, normals }, [
          positions.buffer,
          colors.buffer,
          normals.buffer,
        ]);
      },
      onProgress: ({
        stage,
        viewId,
        done,
        total,
      }: {
        stage: DenseStage;
        viewId: number;
        done: number;
        total: number;
      }) => {
        if (done === total) {
          const now = performance.now();
          stageMs[stage] = now - mark;
          mark = now;
          dense.stage = stage === 'photometric' ? 'geometric' : 'fusion';
        }
        context.post({ type: 'denseProgress', stage, viewId, done, total });
        if (stage !== 'fusion' || dense.fusion === 'pixel')
          activity({ phase: stage, viewId, done, total });
        if (done === total && stage !== 'fusion')
          activity({ phase: stage === 'photometric' ? 'geometric' : 'fusion', done: 0, total });
      },
      onView: (event: any) => {
        if (event.type === 'depth') {
          // Amendment 5: a preview per reference and stage. The depth stage's peak is read at the
          // geometric previews, which come while the geometric pass's arrays are resident.
          if (event.stage === 'geometric') memory.depthStageBytes ??= activeBytes();
          const { viewId, stage, iteration, width, height, depth } = event;
          const image = dense.rgba.get(viewId);
          const rgb = image
            ? blockColors(image, width, height)
            : new Uint8Array(width * height * 3);
          context.post(
            {
              type: 'denseDepth',
              viewId,
              stage,
              ...(iteration === undefined ? {} : { iteration }),
              width,
              height,
              depth,
              rgb,
            },
            [depth.buffer, rgb.buffer],
          );
          return;
        }
        memory.fusionStageBytes ??= activeBytes();
        points += event.count;
        const { viewId, positions, colors, normals } = event;
        context.post({ type: 'densePoints', viewId, positions, colors, normals }, [
          positions.buffer,
          colors.buffer,
          normals.buffer,
        ]);
      },
    });
    const views = result.views.map((view: any) => ({
      id: view.id,
      status: view.status,
      sources: view.sources,
      depthPixels: view.depthPixels,
      points: view.points,
    }));
    // Amendment 3: the plan's bytes of the depth stage, the larger of its two passes.
    const residency = result.residency;
    if (memory.depthStageBytes !== null && residency)
      memory.depthStageFormulaBytes = Math.max(
        residency.photometric.bytes,
        residency.geometric.bytes,
      );
    // 3.12: the cleanup follows the fusion. The raw points are on the page already, and the
    // cleaned cloud replaces them. A failed cleanup leaves the raw cloud and says why.
    let clean: Record<string, any> | null = null,
      cleanError = '',
      finalPoints = result.points.count,
      finalCloud: DenseCleanInput = result.points;
    if (dense.clean && context.clean && result.points.count > 0) {
      dense.stage = 'clean';
      activity({ phase: 'clean', done: 0, total: 0 });
      try {
        const cleaned = await context.clean(result.points);
        clean = cleaned.report;
        finalPoints = cleaned.cloud.count;
        finalCloud = cleaned.cloud;
        // The page gets copies: a transfer detaches the arrays, and the mesh below still reads
        // the cloud.
        const positions = cleaned.cloud.positions.slice(),
          colors = cleaned.cloud.colors.slice(),
          normals = cleaned.cloud.normals.slice();
        context.post({ type: 'denseClean', positions, colors, normals, report: clean }, [
          positions.buffer,
          colors.buffer,
          normals.buffer,
        ]);
      } catch (error) {
        if (dense.abort.signal.aborted) throw error;
        cleanError = String(error instanceof Error ? error.message : error).slice(0, 200);
      }
    }
    // 3.13: the mesh follows the cleanup, on the cloud the page shows. A failed mesh leaves the
    // points and says why.
    let mesh: Record<string, any> | null = null,
      meshError = '';
    if (dense.mesh && context.mesh && finalPoints > 0) {
      dense.stage = 'mesh';
      activity({ phase: 'mesh', done: 0, total: 0 });
      try {
        const meshed = await context.mesh(finalCloud);
        mesh = meshed.report;
        const { positions, colors, indices } = meshed.mesh;
        context.post({ type: 'denseMesh', positions, colors, indices, report: mesh }, [
          positions.buffer,
          colors.buffer,
          indices.buffer,
        ]);
      } catch (error) {
        if (dense.abort.signal.aborted) throw error;
        meshError = String(error instanceof Error ? error.message : error).slice(0, 200);
      }
    }
    await finishArchive(
      'completed',
      views.filter((v: any) => v.status === 'done').map((v: any) => v.id),
    );
    const performanceMeasurements = densePerformanceRecord({
      planarHostMaps: result.planar?.hostMaps,
      tsdf: result.tsdf,
    });
    endDense(context, {
      points: result.points.count,
      ...(result.planar ? { planarFilled: result.planar.filled } : {}),
      ...(performanceMeasurements ? { performance: performanceMeasurements } : {}),
      views,
      memory,
      ...(residency ? { residency } : {}),
      timing: timing(),
      clean,
      cleanError,
      finalPoints,
      mesh,
      meshError,
    });
  } catch (error) {
    await finishArchive(dense.abort.signal.aborted ? 'cancelled' : 'failed');
    if (dense.ended) return;
    if (dense.abort.signal.aborted)
      endDense(context, { cancelled: true, points, memory, timing: timing() });
    else
      endDense(context, {
        failure: denseFailure(
          dense,
          error instanceof Error ? error.stack || String(error) : String(error),
          context.lastKernel(),
          '',
        ),
        points,
        memory,
        timing: timing(),
      });
  }
}
