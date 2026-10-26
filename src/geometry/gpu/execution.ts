import type { ShadePlanReport } from '../../device/gpu/device.ts';

export interface GpuGeometryExecution {
  readonly backend: 'webgpu' | 'cpu';
  readonly report: ShadePlanReport;
  readonly applicationUploadBytes: number;
  readonly applicationReadbackBytes: number;
  readonly cpuStages: readonly string[];
}
