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
