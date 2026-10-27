import type { ShadeDevice } from './device.ts';

const LANES = 64;

/** The most elements of one dispatch: the adapter's workgroups per dimension, LANES lanes each. */
export function maxLanes(device: Pick<ShadeDevice, 'maxKernelElements'>): number {
  return Math.max(LANES, Math.floor(device.maxKernelElements / LANES) * LANES);
}

/** The parameters of `chunkedPass` in which the chunked families differ. */
export interface ChunkedPassOptions {
  /**
   * The occupancy floor of each chunk, the `minUnits` of `ShadeDevice.chunkSize` (the TSDF's
   * `TSDF_OCCUPANCY_FLOOR`). On a device that does not honor floors
   * (`ShadeDevice.occupancyFloors`), it is also the chunk's `min`. Unset: the floor is one
   * workgroup, and the step carries no `minUnits`.
   */
  readonly floor?: number;
  /**
   * How a dispatch stays within `maxLanes(device)`. `'cap'` caps each chunk at it (the TSDF).
   * `'split'` cuts the pass into parts of `maxLanes(device)` elements first and chunks each part
   * on its own, with its own warm-up (planar depth). Unset: no limit.
   */
  readonly limit?: 'cap' | 'split';
}

/**
 * The chunks of one pass over `total` points: sized by the device from the rate of `key`. Each
 * step is named `name`.
 */
export function* chunkedPass(
  device: ShadeDevice,
  key: string,
  name: string,
  total: number,
  dispatch: (range: { first: number; end: number }, groups: number) => void,
  options: ChunkedPassOptions = {},
) {
  if (options.limit !== 'split') {
    yield* chunks(device, key, name, total, dispatch, options.floor, options.limit === 'cap');
    return;
  }
  const limit = maxLanes(device);
  for (let first = 0; first < total; first += limit) {
    const offset = first;
    yield* chunks(
      device,
      key,
      name,
      Math.min(limit, total - first),
      (range, groups) => dispatch({ first: offset + range.first, end: offset + range.end }, groups),
      options.floor,
      false,
    );
  }
}

function* chunks(
  device: ShadeDevice,
  key: string,
  name: string,
  total: number,
  dispatch: (range: { first: number; end: number }, groups: number) => void,
  floor: number | undefined,
  cap: boolean,
) {
  const warm = { id: key, run: () => dispatch({ first: 0, end: 0 }, 0) };
  for (let first = 0; first < total;) {
    const remaining = cap ? Math.min(total - first, maxLanes(device)) : total - first;
    const least = Math.min(
      remaining,
      floor === undefined || device.occupancyFloors ? LANES : floor,
    );
    let n = Math.min(remaining, device.chunkSize(key, remaining, least, floor ?? least));
    if (n < remaining) n = Math.max(least, n - (n % LANES));
    const range = { first, end: first + n };
    yield [
      name,
      () => dispatch(range, Math.ceil(n / LANES)),
      undefined,
      floor === undefined ? { key, units: n, warm } : { key, units: n, minUnits: floor, warm },
    ] as const;
    first += n;
  }
}
