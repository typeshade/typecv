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

/** A range of points and of the observations they own, in the global numbering. End exclusive. */
export interface BundleWindow {
  readonly pointBase: number;
  readonly pointEnd: number;
  readonly obsBase: number;
  readonly obsEnd: number;
}

export interface BundleResidencyInput {
  readonly points: number;
  readonly observations: number;
  /** `P + 1` offsets of the points into the observations: point `p` owns `[starts[p], starts[p+1])`. */
  readonly pointStarts: ArrayLike<number>;
  /** The observations of the longest track. A window holds at least one whole point. */
  readonly longestTrack: number;
  /** The bytes the route may allocate now (the working budget less the live arrays). */
  readonly budgetBytes: number;
  readonly bindingBytes: number;
  readonly maxKernelElements: number;
  /** The bytes of every array when the whole problem is resident, and of each array. */
  readonly residentBytes: number;
  readonly residentArrays: readonly number[];
  /** The bytes of the arrays that stay resident when the problem streams, and of each. */
  readonly floorBytes: number;
  readonly floorArrays: readonly number[];
  /** The window bytes per observation and per point (`blocks` section, `focal` for shared focal). */
  readonly observationBytes: number;
  readonly pointBytes: number;
  /** The arrays of a window of `obsCapacity` observations and `pointCapacity` points, bytes. */
  readonly windowArrays: (obsCapacity: number, pointCapacity: number) => readonly number[];
}

export interface BundleResidency {
  readonly mode: 'resident' | 'streamed';
  readonly obsCapacity: number;
  readonly pointCapacity: number;
  /** Windows of whole points, for the assembly, PCG and update sweeps. */
  readonly windows: readonly BundleWindow[];
  /** Windows of observations that start at multiples of 64, for the cost sweeps. */
  readonly costWindows: readonly { readonly obsBase: number; readonly obsEnd: number }[];
  /** The bytes of the route's arrays under this plan. */
  readonly bytes: number;
  /** False when the floor and the least window exceed the budget (`docs/gpu-budget.md`, 3.6). */
  readonly fits: boolean;
}

/** The observations of a cost window and the least observation window: 64 lanes of a block. */
export const BUNDLE_BLOCK_OBSERVATIONS = 64;

/**
 * The residency plan of the bundle adjustment (`docs/gpu-budget.md`, section 3.3). The whole
 * problem is resident when every array fits the budget, the binding limit and the dispatch limit.
 * Otherwise the observation and point sections of `blocks` become a window of whole points, and
 * the sweeps of the route run over the windows in order.
 */
export function planBundleResidency(input: BundleResidencyInput): BundleResidency {
  const P = input.points,
    M = input.observations;
  const all = { pointBase: 0, pointEnd: P, obsBase: 0, obsEnd: M };
  if (
    input.residentBytes <= input.budgetBytes &&
    input.residentArrays.every((b) => b <= input.bindingBytes) &&
    Math.max(M, P) <= input.maxKernelElements
  )
    return {
      mode: 'resident',
      obsCapacity: M,
      pointCapacity: P,
      windows: [all],
      costWindows: [{ obsBase: 0, obsEnd: M }],
      bytes: input.residentBytes,
      fits: true,
    };
  const least = Math.max(BUNDLE_BLOCK_OBSERVATIONS, input.longestTrack);
  const points = (capacity: number) => Math.max(1, Math.ceil((capacity * P) / M));
  const windowBytes = (capacity: number) =>
    capacity * input.observationBytes + points(capacity) * input.pointBytes;
  const fitsWindow = (capacity: number) =>
    input.floorBytes + windowBytes(capacity) <= input.budgetBytes &&
    input.windowArrays(capacity, points(capacity)).every((b) => b <= input.bindingBytes) &&
    input.floorArrays.every((b) => b <= input.bindingBytes);
  const cap = Math.max(1, Math.min(M, input.maxKernelElements));
  // The largest capacity that fits, by bisection; the least window when none does.
  let lo = Math.min(least, cap),
    hi = cap;
  const fits = fitsWindow(lo);
  if (fits)
    while (lo < hi) {
      const mid = Math.ceil((lo + hi) / 2);
      if (fitsWindow(mid)) lo = mid;
      else hi = mid - 1;
    }
  const obsCapacity = lo,
    pointCapacity = points(obsCapacity);
  const windows: BundleWindow[] = [];
  for (let base = 0; base < P;) {
    let end = base + 1;
    const first = input.pointStarts[base]!;
    while (
      end < P &&
      end - base < pointCapacity &&
      input.pointStarts[end + 1]! - first <= obsCapacity
    )
      end++;
    windows.push({
      pointBase: base,
      pointEnd: end,
      obsBase: first,
      obsEnd: input.pointStarts[end]!,
    });
    base = end;
  }
  const step = Math.max(
    BUNDLE_BLOCK_OBSERVATIONS,
    Math.floor(obsCapacity / BUNDLE_BLOCK_OBSERVATIONS) * BUNDLE_BLOCK_OBSERVATIONS,
  );
  const costWindows: { obsBase: number; obsEnd: number }[] = [];
  for (let base = 0; base < M; base += step)
    costWindows.push({ obsBase: base, obsEnd: Math.min(M, base + step) });
  return {
    mode: 'streamed',
    obsCapacity,
    pointCapacity,
    windows,
    costWindows,
    bytes: input.floorBytes + windowBytes(obsCapacity),
    fits,
  };
}
