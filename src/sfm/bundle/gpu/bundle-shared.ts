import type { Resident } from 'typeshade/runtime';
import type { ShadeDevice } from '../../../device/gpu/device.ts';
import type { GpuGeometryExecution } from '../../../geometry/gpu/execution.ts';
import type { BundleAdjustmentOptions, BundleAdjustmentResult } from '../bundle-adjustment.ts';
import { bundleStallTolerance, bundleStalled } from '../bundle-shared.ts';
import type { CameraProductWorkgroupSize } from './bundle-camera-product.ts';
import { BUNDLE_SOLVER_STALL_TOLERANCE } from '../../config/defaults.ts';

export interface GpuBundleAdjustmentOptions extends BundleAdjustmentOptions {
  readonly maxCgIterations?: number;
  readonly cgRelativeTolerance?: number;
  /** Opt-in dense Schur cache for shared-focal WebGPU problems with at most 16 cameras. */
  readonly schurCache?: boolean;
  /** Camera assembly/product reduction. Auto uses cooperative WebGPU rows for long observation lists. */
  readonly schurReduction?: 'auto' | 'serial' | 'cooperative';
  /** @internal Experimental PCG product: per-camera workgroups or the existing row policy (default).
   *  CPU and integer-fp64 owners retain the existing row policy. Native timings must choose the default. */
  readonly cameraProduct?: 'workgroups' | 'rows';
  /** @internal Per-camera product width, selected once per BA run. Default: 64. */
  readonly cameraProductWorkgroupSize?: CameraProductWorkgroupSize;
  /** @internal Split a camera into tasks of at most this many observations; default: one task per camera. */
  readonly cameraProductTileSize?: number;
}
export interface GpuBundleAdjustmentResult extends BundleAdjustmentResult {
  readonly cgIterations: number;
  readonly linearSolverFailures: number;
  readonly execution: GpuGeometryExecution;
}

/**
 * The words of the BA `state` that the hosts read, the same in both BA shaders (`bundle.shade.ts`
 * and `joint-focal.shade.ts`): the LM part `[lambda, active, baseline, origin.xyz, current
 * cost/SSE, initial cost/SSE, accepted, rejected, commit, gauge scale, input error, trial valid,
 * CG steps, failed solves]`, then the PCG part. `packBundleResult` copies the state after the
 * scene, with the initial and final RMSE in words 24 and 25; the joint focal shader adds the log
 * scale of the shared focal after the state.
 */
export const BUNDLE_STATE = {
  baseline: 2,
  cost: 6,
  initialCost: 8,
  accepted: 10,
  rejected: 11,
  commit: 12,
  inputError: 14,
  cgSteps: 16,
  failedSolves: 17,
  initialRmsePx: 24,
  finalRmsePx: 25,
  /** The words of the state. */
  size: 26,
  /** The joint focal result's log scale of the shared focal: the word after the state. */
  focalScale: 26,
} as const;

/** The convergence rule of `acceptBundleCandidate` in both BA shaders: the solvers' default. */
const SHADER_STALL_TOLERANCE = BUNDLE_SOLVER_STALL_TOLERANCE;

/** @internal Keep the serial CPU oracle and stay inside the one-dimensional dispatch limit. */
export function cooperativeBundleSchur(
  device: ShadeDevice,
  cameras: number,
  observations: number,
  sharedFocal: boolean,
  mode: GpuBundleAdjustmentOptions['schurReduction'],
): boolean {
  return (
    device.backend === 'webgpu' &&
    cameras * 6 + (sharedFocal ? 1 : 0) <= device.maxKernelElements / 64 &&
    (mode === 'cooperative' ||
      ((mode === undefined || mode === 'auto') &&
        device.occupancyFloors &&
        observations >= cameras * 128))
  );
}

/**
 * @internal The host side of a `stallTolerance` above the shaders' own rule: after an LM
 * iteration, read the cost and the commit flag (`BUNDLE_STATE`), and report a stall
 * when the accepted step lowered the cost by less than the tolerance. `cost` holds the cost after
 * the previous iteration. Below the shaders' rule this reads nothing and reports no stall.
 */
export function bundleStallMonitor(
  device: ShadeDevice,
  state: Resident<Float64Array>,
  options: BundleAdjustmentOptions,
): {
  start(): Promise<void>;
  stalled(): Promise<boolean>;
  /** The state words the monitor needs, or `undefined` when the tolerance is the shaders' own. */
  fetch(): Promise<Float64Array | undefined>;
  /** `start` from words that `fetch` returned. */
  begin(values: Float64Array | undefined): void;
  /** `stalled` from words that `fetch` returned. */
  decide(values: Float64Array | undefined): boolean;
} {
  const tolerance = bundleStallTolerance(options);
  let cost = Infinity;
  const monitor = {
    async fetch() {
      if (tolerance <= SHADER_STALL_TOLERANCE) return undefined;
      return device.read(state, BUNDLE_STATE.commit + 1);
    },
    begin(values: Float64Array | undefined) {
      if (values) cost = values[BUNDLE_STATE.cost]!;
    },
    decide(values: Float64Array | undefined) {
      if (!values) return false;
      const next = values[BUNDLE_STATE.cost]!,
        accepted = values[BUNDLE_STATE.commit] === 1,
        change = cost - next;
      cost = next;
      return accepted && bundleStalled(change, next, tolerance);
    },
    async start() {
      monitor.begin(await monitor.fetch());
    },
    async stalled() {
      return monitor.decide(await monitor.fetch());
    },
  };
  return monitor;
}
