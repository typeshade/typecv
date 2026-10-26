import type { ShadeDevice } from './device.ts';

const LANES = 64;

/** The chunks of one pass over `total` points: sized by the device from the rate of `key`. */
export function* chunkedPass(
  device: ShadeDevice,
  key: string,
  name: string,
  total: number,
  dispatch: (range: { first: number; end: number }, groups: number) => void,
) {
  const warm = { id: key, run: () => dispatch({ first: 0, end: 0 }, 0) };
  for (let first = 0; first < total;) {
    const remaining = total - first;
    let n = Math.min(remaining, device.chunkSize(key, remaining, Math.min(remaining, LANES)));
    if (n < remaining) n = Math.max(LANES, n - (n % LANES));
    n = Math.min(n, remaining);
    const range = { first, end: first + n };
    yield [
      name,
      () => dispatch(range, Math.ceil(n / LANES)),
      undefined,
      { key, units: n, warm },
    ] as const;
    first += n;
  }
}
