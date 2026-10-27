/**
 * The rule that chose the flavor of the emulated double in TypeCV before TypeShade change 0085, as
 * one pure function of the signals that a page can read. It is `recommendedFlavor` of
 * `src/device/gpu/device.ts` at `87724f9` (lines 152 to 176) with the selector that it called,
 * `isAppleGpu` and `recommendFp64Flavor` of TypeShade at the pin `725b2969`
 * (`src/core/fp64/flavor-select.ts`). Since change 0085 the TypeShade device selects the flavor
 * itself. The hardware check (`docs/lab-run-records.md`, "Hardware check") computes this rule
 * beside the flavor that the device selected, and `scripts/compare-hardware-check.mjs` computes it
 * again from the recorded signals. `tests/hardware-check-flavor.test.ts` holds it to a table, to
 * the pin's selector and, where the history is present, to the code of `87724f9`.
 */
export type PreviousFlavor = 'float' | 'integer';
/** The fields of a `GPUAdapterInfo` that the rule reads. */
export interface FlavorAdapterSignal {
  readonly vendor?: string | null;
  readonly architecture?: string | null;
}
export interface PreviousFlavorSignals {
  /** The tier. The CPU tier computes doubles as doubles, so it takes `'float'`. */
  readonly backend?: 'webgpu' | 'cpu';
  /**
   * Whether the device was opened on a `GPUDevice` that TypeCV or the application holds, as the
   * lab opens its device (`adapterLimits: true`). The rule then read that device's `adapterInfo`
   * and requested no adapter.
   */
  readonly handedDevice: boolean;
  /** `GPUDevice.adapterInfo` of that device, or null. */
  readonly deviceAdapterInfo?: FlavorAdapterSignal | null;
  /** `(await navigator.gpu.requestAdapter()).info`, or null. */
  readonly requestedAdapterInfo?: FlavorAdapterSignal | null;
  readonly userAgent?: string | null;
}

/** The flavor that TypeCV's rule before change 0085 gives on these signals. */
export function previousFp64Flavor(signals: PreviousFlavorSignals): PreviousFlavor {
  if ((signals.backend ?? 'webgpu') !== 'webgpu') return 'float';
  const info = signals.handedDevice
    ? (signals.deviceAdapterInfo ?? null)
    : (signals.requestedAdapterInfo ?? null);
  const vendor = info?.vendor ?? '';
  const architecture = info?.architecture ?? '';
  // An Apple adapter: Metal runs underneath, and its compiler collapses the float flavor.
  if (/apple/i.test(vendor) || /^apple|metal/i.test(architecture)) return 'integer';
  // Any WebKit on an iPhone, an iPad or a Mac: Metal is the only GPU API there.
  const userAgent = signals.userAgent ?? '';
  if (/\b(iPhone|iPad|Macintosh)\b/.test(userAgent) && /AppleWebKit/.test(userAgent))
    return 'integer';
  return 'float';
}
