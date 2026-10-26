import type { Resident } from 'typeshade/runtime';
import type { ShadeDevice } from '../../../device/gpu/device.ts';
import type { GpuGeometryExecution } from '../../../geometry/gpu/execution.ts';
import type { BundleAdjustmentOptions, BundleAdjustmentResult } from '../bundle-adjustment.ts';
import { bundleStallTolerance } from '../bundle-shared.ts';
import type { CameraProductWorkgroupSize } from './bundle-camera-product.ts';

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

/** The convergence rule of `acceptBundleCandidate` in both BA shaders. */
const SHADER_STALL_TOLERANCE = 1e-9;

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
 * iteration, read the cost (`state[6]`) and the accepted flag (`state[12]`), and report a stall
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
      return device.read(state, 13);
    },
    begin(values: Float64Array | undefined) {
      if (values) cost = values[6]!;
    },
    decide(values: Float64Array | undefined) {
      if (!values) return false;
      const next = values[6]!,
        accepted = values[12] === 1,
        change = cost - next;
      cost = next;
      return accepted && change < tolerance * Math.max(1, next);
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
