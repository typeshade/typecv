/** @internal Compact fixed-scene observation storage, with one camera and point record each. */
export function observationLayout(
  cameras: number,
  points: number,
  observations: number,
  guarded: boolean,
) {
  const lengths = {
    cameras: Math.max(1, 18 * cameras),
    points: Math.max(1, 3 * points),
    pixels: Math.max(1, 2 * observations),
    indices: Math.max(2, 2 * observations),
    errors: Math.max(1, observations),
    flags: Math.max(1, observations + (guarded ? points : 0)),
    offsets: guarded ? points + 1 : 0,
    partials: 2 * Math.ceil(Math.max(1, observations) / 64),
    summary: 2,
  };
  const wide = ['cameras', 'points', 'pixels', 'errors', 'partials', 'summary'] as const;
  const words = ['indices', 'flags', 'offsets'] as const;
  const bytes = [...wide.map((key) => 8 * lengths[key]), ...words.map((key) => 4 * lengths[key])];
  return {
    lengths,
    bytes: bytes.reduce((sum, value) => sum + value, 0),
    largestBindingBytes: Math.max(...bytes),
    inputBytes: 8 * (lengths.cameras + lengths.points + lengths.pixels) + 4 * lengths.indices,
  };
}
