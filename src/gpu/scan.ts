import type { Resident } from 'typeshade/runtime';
import type { ShadeDevice } from './device.ts';
import { scanFlags, scanFlagsOracle, scanFlagBlocks } from './scan.shade.ts';

export interface FlagScanBindings {
  /** One flag per element, `scanBlockCount(count) * 64` or exactly `count` long; rewritten in place. */
  readonly flags: Resident<Uint32Array>;
  /** `2 * scanBlockCount(count)` words. */
  readonly blockSums: Resident<Uint32Array>;
  /** Two words: selected total, issue total. */
  readonly totals: Resident<Uint32Array>;
}

/** Blocks of 64 elements covering `count` elements; at least one. */
export function scanBlockCount(count: number): number {
  return Math.max(1, Math.ceil(count / 64));
}

/** Record the two scan kernels of a flag compaction inside a plan body. */
export function planFlagScan(device: ShadeDevice, bindings: FlagScanBindings, count: number): void {
  const blocks = scanBlockCount(count),
    scan = { flags: bindings.flags, blockSums: bindings.blockSums };
  if (device.backend === 'webgpu') scanFlags(scan, blocks);
  else scanFlagsOracle(scan, Math.ceil(blocks / 8));
  scanFlagBlocks({ blockSums: bindings.blockSums, totals: bindings.totals }, 1);
}
