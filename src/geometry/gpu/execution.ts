import type { ChunkRate, ShadeDevice, ShadePlanReport } from '../../device/gpu/device.ts';

export interface GpuGeometryExecution {
  readonly backend: 'webgpu' | 'cpu';
  readonly report: ShadePlanReport;
  readonly applicationUploadBytes: number;
  readonly applicationReadbackBytes: number;
  readonly cpuStages: readonly string[];
}

/**
 * @internal The execution report of one GPU operation: the device's backend, the operation's plan
 * report, and the application bytes that the device uploaded and read back between `before` and
 * `after`, two reads of its statistics.
 */
export function geometryExecution(
  device: ShadeDevice,
  report: ShadePlanReport,
  before: ShadeDevice['statistics'],
  after: ShadeDevice['statistics'],
): GpuGeometryExecution {
  return {
    backend: device.backend,
    report,
    applicationUploadBytes: after.applicationUploadBytes - before.applicationUploadBytes,
    applicationReadbackBytes: after.applicationReadbackBytes - before.applicationReadbackBytes,
    cpuStages: [],
  };
}

/**
 * @internal The reports of several plans as one report: their dispatches and submissions summed,
 * their groups in order, and the rates of the chunk keys that the plans ran, as the last plan of
 * each key reported them.
 */
export function mergedReport(
  tier: ShadePlanReport['tier'],
  reports: readonly ShadePlanReport[],
): ShadePlanReport {
  const chunks: Record<string, ChunkRate> = Object.assign(
    {},
    ...reports.map((r) => r.chunks ?? {}),
  );
  return {
    tier,
    dispatches: reports.reduce((n, r) => n + r.dispatches, 0),
    submissions: reports.reduce((n, r) => n + r.submissions, 0),
    groups: reports.flatMap((r) => r.groups),
    ...(Object.keys(chunks).length ? { chunks } : {}),
  };
}
