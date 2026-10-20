/**
 * The lab worker's dense MVS (`docs/dense-mvs.md` 5.3, step 6 of section 8): the RGBA it keeps
 * for each keyframe, the run of `reconstructDenseGpu` after the sparse result, and the messages it
 * posts. The worker (`website/src/workers/sfm.worker.ts`) owns the device and the job. This module
 * takes them as arguments, so that `website/dense-run.test.mjs` runs it with a fake MVS.
 */

/** The stages of the MVS, in their order (5.3). */
export type DenseStage = 'photometric' | 'geometric' | 'fusion';

/** One kept image: the RGBA bytes of a keyframe at the analysis size. */
export type DenseImage = { rgba: Uint8ClampedArray; width: number; height: number };

/** What a run with the dense MVS keeps from its start to `denseResult`. */
export type DenseJob = {
  /** The RGBA of each keyframe at the analysis size, 4 B per pixel (4.2). */
  rgba: Map<number, DenseImage>;
  /** The most views the all-resident layout holds at the analysis size (4.2), 0 before a frame. */
  limit: number;
  /** Why the map was cleared before the MVS, or empty. */
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
  /** Mesh the final cloud (section 3.13). On with `?denseMesh=1`. */
  mesh: boolean;
};

export function createDenseJob(clean = true, mesh = false): DenseJob {
  return {
    clean,
    mesh,
    rgba: new Map(),
    limit: 0,
    skipped: '',
    abort: new AbortController(),
    running: false,
    stage: '',
    ended: false,
  };
}

/**
 * The most views the all-resident layout of 4.2 holds: the depth stage takes (16 N + 4) bytes per
 * pixel, so N ≤ ⌊(workingBytes / P − 4) / 16⌋. 15 views at 640 × 427 with 64 MiB.
 */
export function denseViewLimit(workingBytes: number, pixels: number): number {
  return Math.max(0, Math.floor((workingBytes / pixels - 4) / 16));
}

/**
 * Keep a frame's RGBA until the MVS (5.3, item 3). More keyframes than `denseViewLimit` cannot
 * all run, so the map is then cleared and the run reports why. The caller deletes the entry of a
 * frame that is no keyframe.
 */
export function keepFrame(
  dense: DenseJob,
  frame: { id: number; width: number; height: number; pixels: Uint8ClampedArray },
  workingBytes: number,
): void {
  if (dense.skipped) return;
  dense.limit ||= denseViewLimit(workingBytes, frame.width * frame.height);
  dense.rgba.set(frame.id, { rgba: frame.pixels, width: frame.width, height: frame.height });
  if (dense.rgba.size > dense.limit) {
    dense.rgba.clear();
    dense.skipped =
      'More keyframes than the ' + dense.limit + ' views the dense layout holds (section 4.2)';
  }
}

/** The failure of the MVS (5.3, item 7): the message, the stage, the kernel and the reason. */
export function denseFailure(dense: DenseJob, message: string, kernel: string, reason: string) {
  return {
    message: message.slice(0, 2000),
    stage: dense.stage,
    kernel: kernel.slice(0, 64),
    reason: reason.slice(0, 64),
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
  readonly dense: DenseJob;
  /** The MVS options, `LAB_OPTIONS.dense` of the worker. */
  readonly options: Readonly<Record<string, number>>;
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
      denseClean: dense.clean,
      ...(context.mesh ? { denseMesh: dense.mesh } : {}),
    },
  });
}

/**
 * The dense MVS after the sparse result (5.3, items 4 to 7): `reconstruct` over the registered
 * views whose RGBA the run kept. It posts `denseProgress` after each reference of each stage,
 * `denseDepth` per view, `densePoints` per reference and `denseResult` at the end, with the arrays
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
  const first = images.values().next().value!,
    P = first.width * first.height;
  dense.stage = 'photometric';
  try {
    const result = await context.reconstruct(scene, images, {
      ...context.options,
      signal: dense.abort.signal,
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
      },
      onView: (event: any) => {
        if (event.type === 'depth') {
          // The depth events follow the depth stage of every view, before its arrays recycle.
          memory.depthStageBytes ??= activeBytes();
          const { viewId, width, height, depth } = event;
          context.post({ type: 'denseDepth', viewId, width, height, depth }, [depth.buffer]);
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
    // 4.2: the formula counts the views with a depth map.
    const N = views.filter((view: any) => view.status === 'done').length;
    if (memory.depthStageBytes !== null) memory.depthStageFormulaBytes = (16 * N + 4) * P;
    // 3.12: the cleanup follows the fusion. The raw points are on the page already, and the
    // cleaned cloud replaces them. A failed cleanup leaves the raw cloud and says why.
    let clean: Record<string, any> | null = null,
      cleanError = '',
      finalPoints = result.points.count,
      finalCloud: DenseCleanInput = result.points;
    if (dense.clean && context.clean && result.points.count > 0) {
      dense.stage = 'clean';
      try {
        const cleaned = await context.clean(result.points);
        clean = cleaned.report;
        finalPoints = cleaned.cloud.count;
        finalCloud = cleaned.cloud;
        const { positions, colors, normals } = cleaned.cloud;
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
    endDense(context, {
      points: result.points.count,
      views,
      memory,
      timing: timing(),
      clean,
      cleanError,
      finalPoints,
      mesh,
      meshError,
    });
  } catch (error) {
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
