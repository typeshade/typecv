/**
 * The working budget of a worker's device (`docs/gpu-budget.md`, section 2): the larger of the
 * budget that scales with the analysis frame (`scaleBytes`, 64 MiB at 640 px) and half of the
 * adapter's `maxBufferSize`, the conservative fraction. WebGPU reports no memory size, and
 * `maxBufferSize` is the largest buffer the adapter says it can make. The GPU stages split their
 * work to fit this budget (size changes speed, not success), so no frame count enters it. Without a
 * reported limit the budget is `scaleBytes`.
 */
export function workingBytesFromAdapter(
  scaleBytes: number,
  maxBufferSize: number | undefined,
): number {
  return Math.max(scaleBytes, Math.floor((maxBufferSize ?? 0) / 2));
}

/** The long side, in pixels, of the analysis frame at which the budgets have their base values. */
export const BUDGET_REFERENCE_EDGE = 640;

/** The working budget that scales with the analysis frame, at 640 px: 64 MiB. */
export const REFERENCE_WORKING_BYTES = 64 * 1024 ** 2;

/** The pool budget of a worker's device at 640 px: 32 MiB. */
export const REFERENCE_POOL_BYTES = 32 * 1024 ** 2;

/**
 * The scale of a worker's budgets and of its feature cap (`featureCap` of `lab-features.ts`): the
 * pixel count of the analysis frame over that of a 640 px frame. The SIFT pyramid of a 1280 px
 * frame needs about 72 MB against the 64 MB default (`sift.ts`, `estimatedPeakBytes`), and the
 * resident descriptors of the matching window grow with the feature cap. Octave -1 builds the
 * pyramid at twice the frame's width and height: four times the pixels.
 */
export function analysisScale(analysisEdge: number, firstOctave: -1 | 0): number {
  return (
    ((analysisEdge * analysisEdge) / (BUDGET_REFERENCE_EDGE * BUDGET_REFERENCE_EDGE)) *
    (firstOctave === -1 ? 4 : 1)
  );
}

/**
 * The working and pool budgets of a worker's device at `scale` (`analysisScale`): 64 and 32 MiB at
 * 640 px and 256 and 128 MiB at 1280 px, and the working budget raised by `workingRule` from the
 * adapter's `maxBufferSize`. Each device keeps its own rule: `workingBytesFromAdapter` for the SfM
 * device of both workers, `denseWorkingBytes` of `dense-run.ts` for the dense device of the
 * multi-video worker.
 */
export function deviceBudgets(
  scale: number,
  maxBufferSize: number | undefined,
  workingRule: (
    scaleBytes: number,
    maxBufferSize: number | undefined,
  ) => number = workingBytesFromAdapter,
): { workingBytes: number; poolBytes: number } {
  return {
    workingBytes: workingRule(Math.round(REFERENCE_WORKING_BYTES * scale), maxBufferSize),
    poolBytes: Math.round(REFERENCE_POOL_BYTES * scale),
  };
}
