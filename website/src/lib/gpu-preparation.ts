/** Preparation events report actual work boundaries; they are not a time estimate. */
export const GPU_PREPARATION_PHASES = [
  'viewer',
  'worker',
  'runtime',
  'vocabulary',
  'adapter',
  'device',
  'session',
] as const;

export type GpuPreparationPhase = (typeof GPU_PREPARATION_PHASES)[number];

export interface GpuPreparation {
  phase: GpuPreparationPhase;
  startedAt: number;
  multi: boolean;
}

export function isGpuPreparationPhase(value: unknown): value is GpuPreparationPhase {
  return GPU_PREPARATION_PHASES.some((phase) => phase === value);
}
