/** The builders of a run record's fields from what the workers send: the stage totals, the GPU
 *  rows, the chunk rates and the dense block. */
import { gpuLossContextValid } from '../gpu-loss-context.ts';
import {
  RUN_RECORD_DENSE_STAGES,
  RUN_RECORD_MAX_CHUNK_KEYS,
  RUN_RECORD_MAX_GPU_NAME,
  RUN_RECORD_MAX_GPU_ROWS,
  type RunRecord,
  type RunRecordChunk,
  type RunRecordDense,
  type RunRecordDensePerformance,
  type RunRecordKernelTime,
  type RunRecordStage,
} from './schema.ts';
import {
  chunksValid,
  count,
  countFields,
  descriptorCacheRecord,
  finite,
  HOST_RETENTION_COUNTS,
  PLANAR_HOST_MAP_COUNTS,
  queuedViewUploadsRecord,
  TSDF_COUNTS,
  TSDF_HOST_TIMINGS,
  TSDF_OPTIONAL_COUNTS,
} from './validate.ts';

/** Omit unavailable/invalid groups, preserve measured zeros and copy only known counters. */
export function sparsePerformanceRecord(
  message: unknown,
): Pick<RunRecord['result'], 'hostRetention' | 'descriptorCache'> {
  if (!message || typeof message !== 'object' || Array.isArray(message)) return {};
  const input = message as Record<string, unknown>;
  const hostRetention = countFields(input.hostRetention, HOST_RETENTION_COUNTS);
  const descriptorCache = descriptorCacheRecord(input.descriptorCache);
  return {
    ...(hostRetention ? { hostRetention } : {}),
    ...(descriptorCache ? { descriptorCache } : {}),
  };
}

/** Omit unavailable/invalid measurements rather than recording an invented zero. */
export function densePerformanceRecord(value: unknown): RunRecordDensePerformance | undefined {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return undefined;
  const input = value as Record<string, any>;
  const planar = input.planarHostMaps;
  const source = input.tsdf;
  const planarHostMaps =
    planar &&
    typeof planar === 'object' &&
    !Array.isArray(planar) &&
    PLANAR_HOST_MAP_COUNTS.every((key) => count(planar[key])) &&
    planar.peakRetainedBytes >= planar.inputBytes
      ? (Object.fromEntries(PLANAR_HOST_MAP_COUNTS.map((key) => [key, planar[key]])) as NonNullable<
          RunRecordDensePerformance['planarHostMaps']
        >)
      : undefined;
  let tsdf: RunRecordDensePerformance['tsdf'];
  if (
    source &&
    typeof source === 'object' &&
    !Array.isArray(source) &&
    TSDF_COUNTS.every((key) => count(source[key]))
  ) {
    const timings = source.hostTimings;
    const queuedViewUploads = queuedViewUploadsRecord(source.queuedViewUploads);
    const hostTimings =
      timings &&
      typeof timings === 'object' &&
      !Array.isArray(timings) &&
      TSDF_HOST_TIMINGS.every((key) => finite(timings[key]) && timings[key] >= 0)
        ? (Object.fromEntries(
            TSDF_HOST_TIMINGS.map((key) => {
              const rounded = Math.round(timings[key] * 10) / 10;
              return [key, Number.isFinite(rounded) ? rounded : timings[key]];
            }),
          ) as NonNullable<NonNullable<RunRecordDensePerformance['tsdf']>['hostTimings']>)
        : undefined;
    tsdf = {
      blocks: source.blocks,
      passes: source.passes,
      peakBytes: source.peakBytes,
      uploadBytes: source.uploadBytes,
      ...Object.fromEntries(
        TSDF_OPTIONAL_COUNTS.filter((key) => count(source[key])).map((key) => [key, source[key]]),
      ),
      ...(hostTimings ? { hostTimings } : {}),
      ...(queuedViewUploads ? { queuedViewUploads } : {}),
    };
  }
  return planarHostMaps || tsdf
    ? { ...(planarHostMaps ? { planarHostMaps } : {}), ...(tsdf ? { tsdf } : {}) }
    : undefined;
}

/**
 * The GPU time of one report in ms, summed from the groups that carry a `gpuTime` (nanoseconds),
 * or undefined when no group does: a device without `timestamp-query` reports no time, not 0.
 */
export function reportGpuMs(report: unknown): number | undefined {
  const groups = (report as { groups?: readonly { gpuTime?: unknown }[] } | undefined)?.groups;
  if (!Array.isArray(groups)) return undefined;
  let ns = 0,
    timed = false;
  for (const group of groups)
    if (
      typeof group?.gpuTime === 'number' &&
      Number.isFinite(group.gpuTime) &&
      group.gpuTime >= 0
    ) {
      ns += group.gpuTime;
      timed = true;
    }
  return timed ? Math.round(ns / 1e3) / 1e3 : undefined;
}

/**
 * The record's `timing.gpu` from the rows the worker sends (`ShadeDevice.statistics.kernels`):
 * names cut to `RUN_RECORD_MAX_GPU_NAME` (rows that then share a name are summed), at most
 * `RUN_RECORD_MAX_GPU_ROWS` rows, `gpuMs` in ms with microsecond precision, null without a timed
 * group. `timestamps` is whether any group was timed.
 */
export function gpuRecord(
  rows: unknown,
  dropped: unknown = 0,
): { gpu: RunRecordKernelTime[]; gpuTimestamps: boolean; gpuDropped: number } {
  const merged = new Map<
    string,
    { groups: number; dispatches: number; timed: number; ns: number }
  >();
  let lost = count(dropped) ? dropped : 0;
  if (Array.isArray(rows))
    for (const row of rows) {
      if (!row || typeof row.name !== 'string' || row.name.length === 0) continue;
      const name = row.name.slice(0, RUN_RECORD_MAX_GPU_NAME);
      let entry = merged.get(name);
      if (!entry) {
        if (merged.size >= RUN_RECORD_MAX_GPU_ROWS) {
          lost += 1;
          continue;
        }
        entry = { groups: 0, dispatches: 0, timed: 0, ns: 0 };
        merged.set(name, entry);
      }
      const whole = (v: unknown) => (count(v) ? v : 0);
      entry.groups += whole(row.groups);
      entry.dispatches += whole(row.dispatches);
      const timed = Math.min(whole(row.timedGroups), whole(row.groups));
      entry.timed += timed;
      if (timed > 0 && finite(row.gpuNs) && row.gpuNs >= 0) entry.ns += row.gpuNs;
    }
  const gpu = [...merged].map(([name, e]) => ({
    name,
    groups: e.groups,
    dispatches: e.dispatches,
    timedGroups: Math.min(e.timed, e.groups),
    gpuMs: e.timed > 0 ? Math.round(e.ns / 1e3) / 1e3 : null,
  }));
  return { gpu, gpuTimestamps: gpu.some((row) => row.gpuMs !== null), gpuDropped: lost };
}

/** Collapse the lab's stage list into per-operation totals. */
export function summarizeStages(
  stages: readonly {
    operation: string;
    elapsedMs?: number;
    dispatches?: number;
    submissions?: number;
    gpuMs?: number;
  }[],
): Record<string, RunRecordStage> {
  const out: Record<string, RunRecordStage> = {};
  for (const stage of stages) {
    if (stage.operation === 'runTotal') continue;
    const entry = out[stage.operation] ?? { calls: 0, elapsedMs: 0, dispatches: 0, submissions: 0 };
    out[stage.operation] = {
      calls: entry.calls + 1,
      elapsedMs: entry.elapsedMs + (stage.elapsedMs ?? 0),
      dispatches: entry.dispatches + (stage.dispatches ?? 0),
      submissions: entry.submissions + (stage.submissions ?? 0),
      ...(entry.gpuMs !== undefined || stage.gpuMs !== undefined
        ? { gpuMs: Math.round(((entry.gpuMs ?? 0) + (stage.gpuMs ?? 0)) * 1e3) / 1e3 }
        : {}),
    };
  }
  return out;
}

/**
 * The record's `timing.chunks` from what a worker sends: at most `RUN_RECORD_MAX_CHUNK_KEYS` keys,
 * each with the four numbers of a chunk rate. Anything else is dropped.
 */
export function chunkRecord(rates: unknown): Record<string, RunRecordChunk> {
  const out: Record<string, RunRecordChunk> = {};
  if (!rates || typeof rates !== 'object') return out;
  for (const [key, rate] of Object.entries(rates as Record<string, any>)) {
    if (Object.keys(out).length >= RUN_RECORD_MAX_CHUNK_KEYS) break;
    if (!key || key.length > 64 || !rate || typeof rate !== 'object') continue;
    const entry = {
      msPerUnit: rate.msPerUnit,
      chunks: rate.chunks,
      longestMs: rate.longestMs,
      lastUnits: rate.lastUnits,
    };
    if (chunksValid({ [key]: entry })) out[key] = entry;
  }
  return out;
}

/**
 * The record's `dense` block from the worker's `denseResult` (5.3), or undefined without one.
 * The worker's numbers are bounded and rounded here, so that the validator accepts the block.
 */
export function denseRecord(message: any): RunRecordDense | undefined {
  if (!message || typeof message !== 'object') return undefined;
  const whole = (v: unknown) => (Number.isSafeInteger(v) && (v as number) >= 0 ? (v as number) : 0);
  const ms = (v: unknown) => (finite(v) && v >= 0 ? Math.round(v * 10) / 10 : 0);
  const bytes = (v: unknown) =>
    Number.isSafeInteger(v) && (v as number) >= 0 ? (v as number) : null;
  const o = message.options ?? {},
    m = message.memory ?? {},
    views: any[] = Array.isArray(message.views) ? message.views : [];
  const status: RunRecordDense['status'] = message.failure
    ? 'failed'
    : message.cancelled
      ? 'cancelled'
      : message.skipped
        ? 'skipped'
        : 'completed';
  const stageMs = Object.fromEntries(
    RUN_RECORD_DENSE_STAGES.map((stage) => [stage, ms(message.timing?.stageMs?.[stage])]),
  ) as RunRecordDense['stageMs'];
  const failure = message.failure ?? {};
  const performance = densePerformanceRecord(message.performance);
  return {
    status,
    skipped: status === 'skipped' ? String(message.skipped).slice(0, 200) : '',
    options: {
      maxSources: whole(o.maxSources),
      bestSources: whole(o.bestSources),
      windowRadius: whole(o.windowRadius),
      windowStep: whole(o.windowStep),
      photometricIterations: whole(o.photometricIterations),
      geometricIterations: whole(o.geometricIterations),
      minConsistentViews: whole(o.minConsistentViews),
      denseClean: o.denseClean === true,
      ...(o.fusion === 'pixel' || o.fusion === 'tsdf' ? { fusion: o.fusion } : {}),
      ...(typeof o.denseMesh === 'boolean' ? { denseMesh: o.denseMesh } : {}),
      ...(typeof o.planarDepth === 'boolean' ? { planarDepth: o.planarDepth } : {}),
      ...(typeof o.boundaryAware === 'boolean' ? { boundaryAware: o.boundaryAware } : {}),
      ...(finite(o.minFiniteGain) && o.minFiniteGain >= 0 && o.minFiniteGain <= 2
        ? { minFiniteGain: o.minFiniteGain }
        : {}),
      ...(typeof message.gpuTimestamps === 'boolean'
        ? { gpuTimestamps: message.gpuTimestamps }
        : {}),
      ...(o.fusion === 'tsdf' &&
      (o.tsdf?.visibility === 'ray-count' || o.tsdf?.visibility === 'view-weighted')
        ? { tsdfVisibility: o.tsdf.visibility }
        : {}),
    },
    ...(Number.isSafeInteger(message.planarFilled) && message.planarFilled >= 0
      ? { planarFilled: whole(message.planarFilled) }
      : {}),
    ...(performance ? { performance } : {}),
    ...cleanRecord(message),
    ...meshRecord(message),
    views: views.filter((view) => view?.status === 'done').length,
    viewsWithoutDepth: views.filter((view) => view && view.status !== 'done').length,
    points: whole(message.points),
    totalMs: ms(message.timing?.totalMs),
    stageMs,
    memory: {
      liveBytesBefore: whole(m.liveBytesBefore),
      pooledBytesBefore: whole(m.pooledBytesBefore),
      depthStageBytes: bytes(m.depthStageBytes),
      depthStageFormulaBytes: bytes(m.depthStageFormulaBytes),
      fusionStageBytes: bytes(m.fusionStageBytes),
    },
    failure:
      status === 'failed'
        ? {
            message: String(failure.message ?? '').slice(0, 2000),
            stage: String(failure.stage ?? '').slice(0, 64),
            kernel: String(failure.kernel ?? '').slice(0, 64),
            reason: String(failure.reason ?? '').slice(0, 64),
            ...(gpuLossContextValid(failure.context) ? { context: failure.context } : {}),
          }
        : null,
  };
}

/**
 * The record's one `dense` block of a multi-video run: the sum of the blocks of its components
 * (`docs/multi-video.md`, Amendment 9, item 6). Counts and times add up. A byte count takes the
 * largest value, and the "before" counts those of the first component. The status is `failed` if
 * any component failed, else `cancelled` if any was cancelled, else `completed` if any completed,
 * else `skipped`. The thresholds of `clean` and `mesh` come from the component with the most points.
 * Per-component performance diagnostics are retained only for one block; multi-component
 * records keep them in `result.components[].dense` rather than adding memory peaks.
 */
export function sumDenseRecords(blocks: readonly RunRecordDense[]): RunRecordDense | undefined {
  const first = blocks[0];
  if (!first) return undefined;
  const sum = (pick: (d: RunRecordDense) => number) => blocks.reduce((n, d) => n + pick(d), 0);
  const round = (v: number) => Math.round(v * 10) / 10;
  const largest = (pick: (d: RunRecordDense) => number | null) =>
    blocks.reduce<number | null>((m, d) => {
      const v = pick(d);
      return v === null ? m : m === null ? v : Math.max(m, v);
    }, null);
  const failed = blocks.find((d) => d.status === 'failed');
  const status: RunRecordDense['status'] = failed
    ? 'failed'
    : blocks.some((d) => d.status === 'cancelled')
      ? 'cancelled'
      : blocks.some((d) => d.status === 'completed')
        ? 'completed'
        : 'skipped';
  const cleans = blocks.filter((d) => d.clean).map((d) => ({ d, c: d.clean! }));
  const meshes = blocks.filter((d) => d.mesh).map((d) => ({ d, m: d.mesh! }));
  const biggest = <T extends { d: RunRecordDense }>(list: T[]) =>
    list.reduce((a, b) => (b.d.points > a.d.points ? b : a));
  const clean = cleans.length
    ? (() => {
        const ref = biggest(cleans).c;
        const add = (pick: (c: NonNullable<RunRecordDense['clean']>) => number) =>
          cleans.reduce((n, { c }) => n + pick(c), 0);
        return {
          tier: cleans.every(({ c }) => c.tier === 'webgpu')
            ? ('webgpu' as const)
            : ('cpu' as const),
          before: add((c) => c.before),
          afterOutliers: add((c) => c.afterOutliers),
          afterVoxel: add((c) => c.afterVoxel),
          after: add((c) => c.after),
          outlierThreshold: ref.outlierThreshold,
          voxelSize: ref.voxelSize,
          eps: ref.eps,
          clusters: add((c) => c.clusters),
          keptClusters: add((c) => c.keptClusters),
          outliersMs: round(add((c) => c.outliersMs)),
          voxelMs: round(add((c) => c.voxelMs)),
          clusterMs: round(add((c) => c.clusterMs)),
          error: cleans.find(({ c }) => c.error)?.c.error ?? '',
        };
      })()
    : undefined;
  const mesh = meshes.length
    ? (() => {
        const ref = biggest(meshes).m;
        const add = (pick: (m: NonNullable<RunRecordDense['mesh']>) => number) =>
          meshes.reduce((n, { m }) => n + pick(m), 0);
        return {
          tier: meshes.every(({ m }) => m.tier === 'webgpu')
            ? ('webgpu' as const)
            : ('cpu' as const),
          points: add((m) => m.points),
          voxelSize: ref.voxelSize,
          radius: ref.radius,
          nodes: add((m) => m.nodes),
          cubes: add((m) => m.cubes),
          vertices: add((m) => m.vertices),
          faces: add((m) => m.faces),
          latticeMs: round(add((m) => m.latticeMs)),
          fieldMs: round(add((m) => m.fieldMs)),
          cubesMs: round(add((m) => m.cubesMs)),
          verticesMs: round(add((m) => m.verticesMs)),
          error: meshes.find(({ m }) => m.error)?.m.error ?? '',
        };
      })()
    : undefined;
  return {
    status,
    skipped:
      status === 'skipped' ? (blocks.find((d) => d.skipped)?.skipped ?? '').slice(0, 200) : '',
    options: first.options,
    ...(blocks.some((d) => d.planarFilled !== undefined)
      ? { planarFilled: sum((d) => d.planarFilled ?? 0) }
      : {}),
    ...(blocks.length === 1 && first.performance ? { performance: first.performance } : {}),
    ...(clean ? { clean } : {}),
    ...(mesh ? { mesh } : {}),
    views: sum((d) => d.views),
    viewsWithoutDepth: sum((d) => d.viewsWithoutDepth),
    points: sum((d) => d.points),
    totalMs: round(sum((d) => d.totalMs)),
    stageMs: Object.fromEntries(
      RUN_RECORD_DENSE_STAGES.map((stage) => [stage, round(sum((d) => d.stageMs[stage]))]),
    ) as RunRecordDense['stageMs'],
    memory: {
      liveBytesBefore: first.memory.liveBytesBefore,
      pooledBytesBefore: first.memory.pooledBytesBefore,
      depthStageBytes: largest((d) => d.memory.depthStageBytes),
      depthStageFormulaBytes: largest((d) => d.memory.depthStageFormulaBytes),
      fusionStageBytes: largest((d) => d.memory.fusionStageBytes),
    },
    failure: failed?.failure ?? null,
  };
}

/** The `clean` field of a dense record, or nothing when the cleanup neither ran nor failed. */
function cleanRecord(message: any): Pick<RunRecordDense, 'clean'> | Record<string, never> {
  const r = message.clean,
    error = typeof message.cleanError === 'string' ? message.cleanError.slice(0, 200) : '';
  if (!(r && typeof r === 'object') && !error) return {};
  const whole = (v: unknown) => (Number.isSafeInteger(v) && (v as number) >= 0 ? (v as number) : 0);
  const num = (v: unknown) => (finite(v) && v >= 0 ? v : 0);
  const ms = (v: unknown) => Math.round(num(v) * 10) / 10;
  const c = r ?? {};
  return {
    clean: {
      tier: c.tier === 'webgpu' ? 'webgpu' : 'cpu',
      before: whole(c.before),
      afterOutliers: whole(c.afterOutliers),
      afterVoxel: whole(c.afterVoxel),
      after: whole(c.after),
      outlierThreshold: num(c.outlierThreshold),
      voxelSize: num(c.voxelSize),
      eps: num(c.eps),
      clusters: whole(c.clusters),
      keptClusters: whole(c.keptClusters),
      outliersMs: ms(c.ms?.outliers),
      voxelMs: ms(c.ms?.voxel),
      clusterMs: ms(c.ms?.cluster),
      error,
    },
  };
}

/** The `mesh` field of a dense record, or nothing when the mesh neither ran nor failed. */
function meshRecord(message: any): Pick<RunRecordDense, 'mesh'> | Record<string, never> {
  const r = message.mesh,
    error = typeof message.meshError === 'string' ? message.meshError.slice(0, 200) : '';
  if (!(r && typeof r === 'object') && !error) return {};
  const whole = (v: unknown) => (Number.isSafeInteger(v) && (v as number) >= 0 ? (v as number) : 0);
  const num = (v: unknown) => (finite(v) && v >= 0 ? v : 0);
  const ms = (v: unknown) => Math.round(num(v) * 10) / 10;
  const m = r ?? {};
  return {
    mesh: {
      tier: m.tier === 'webgpu' ? 'webgpu' : 'cpu',
      points: whole(m.points),
      voxelSize: num(m.voxelSize),
      radius: num(m.radius),
      nodes: whole(m.nodes),
      cubes: whole(m.cubes),
      vertices: whole(m.vertices),
      faces: whole(m.faces),
      latticeMs: ms(m.ms?.lattice),
      fieldMs: ms(m.ms?.field),
      cubesMs: ms(m.ms?.cubes),
      verticesMs: ms(m.ms?.vertices),
      error,
    },
  };
}
