/**
 * Host-side plans that cut a GPU stage into pieces that fit the device's budget
 * (`docs/gpu-budget.md`). Size changes speed, not success: a plan never refuses a problem for
 * its size. These functions are pure, so a test reads them without a device.
 */

/** The bytes per observation of the triangulation: `raw` 160, `geometry` 48, `flags` 4. */
export const TRIANGULATION_OBSERVATION_BYTES = 212;
/** The bytes per track of the triangulation: `offsets` 4, `result` 40, `statuses` 4. */
export const TRIANGULATION_TRACK_BYTES = 48;

export interface TriangulationChunkLimits {
  /** The bytes of arrays the chunk may allocate. */
  readonly budgetBytes: number;
  /** The device's storage binding limit, bytes. */
  readonly bindingBytes: number;
  /** The device's dispatch limit, elements. */
  readonly maxKernelElements: number;
}

/**
 * Cut `counts` (the observations of each track, in order) into contiguous chunks of tracks.
 * A chunk holds as many tracks as fit `limits`; a chunk has at least one track, so a budget below
 * one track's arrays gives one track per chunk. Returns the first track of each chunk and the
 * end: `[first, end)`.
 */
export function planTriangulationChunks(
  counts: ArrayLike<number>,
  limits: TriangulationChunkLimits,
): { first: number; end: number }[] {
  const chunks: { first: number; end: number }[] = [];
  const maxObservations = Math.max(
    1,
    Math.min(limits.maxKernelElements, limits.bindingBytes / 160),
  );
  const maxTracks = Math.max(1, Math.min(limits.maxKernelElements / 4, limits.bindingBytes / 40));
  let first = 0;
  while (first < counts.length) {
    let end = first,
      observations = 0;
    while (end < counts.length) {
      const next = observations + counts[end]!;
      const bytes =
        TRIANGULATION_OBSERVATION_BYTES * next + TRIANGULATION_TRACK_BYTES * (end + 1 - first) + 4;
      if (
        end > first &&
        (bytes > limits.budgetBytes || next > maxObservations || end + 1 - first > maxTracks)
      )
        break;
      observations = next;
      end++;
    }
    chunks.push({ first, end });
    first = end;
  }
  return chunks;
}
