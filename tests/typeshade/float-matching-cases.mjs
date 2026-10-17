import assert from 'node:assert/strict';

/** Independent double-precision exhaustive oracle; keep fixtures away from rounding boundaries. */
export function floatMatchOracle(first, second, dimensions, options = {}) {
  const ratio = options.ratio ?? 0.8,
    maxDistance = options.maxDistance ?? 1e19,
    mutual = options.mutual ?? true;
  const nearest = (left, right) =>
    Array.from({ length: left.length / dimensions }, (_, i) => {
      const rows = Array.from({ length: right.length / dimensions }, (_, j) => {
        let squared = 0;
        for (let d = 0; d < dimensions; d++) {
          const delta = left[i * dimensions + d] - right[j * dimensions + d];
          squared += delta * delta;
        }
        return { index: j, squared };
      }).sort((a, b) => a.squared - b.squared || a.index - b.index);
      return {
        ...rows[0],
        accepted:
          rows.length >= 2 &&
          rows[0].squared <= maxDistance * maxDistance &&
          rows[0].squared < ratio * ratio * rows[1].squared,
      };
    });
  const forward = nearest(first, second),
    backward = nearest(second, first),
    matches = [],
    distances = [];
  forward.forEach((row, i) => {
    if (
      !row.accepted ||
      (mutual && (!backward[row.index].accepted || backward[row.index].index !== i))
    )
      return;
    matches.push(i, row.index);
    distances.push(Math.sqrt(row.squared));
  });
  return { matches, distances };
}

export function floatDescriptorFixture(count, dimensions, seed = 17) {
  let state = seed >>> 0;
  const values = new Float32Array(count * dimensions);
  for (let i = 0; i < values.length; i++) {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    values[i] = (state / 2 ** 32) * 2 - 1;
  }
  return values;
}

export async function checkFloatMatches(device, match, first, second, dimensions, options = {}) {
  const a = device.allocate(Math.max(1, first.length), first.length ? first : new Float32Array(1)),
    b = device.allocate(Math.max(1, second.length), second.length ? second : new Float32Array(1));
  try {
    const result = await match(
      device,
      { descriptors: a, count: first.length / dimensions, dimensions, elements: first.length },
      { descriptors: b, count: second.length / dimensions, dimensions, elements: second.length },
      options,
    );
    const expected = floatMatchOracle(first, second, dimensions, options);
    assert.deepEqual([...result.matches], expected.matches);
    result.distances.forEach((distance, i) =>
      assert.ok(
        Math.abs(distance - expected.distances[i]) <= 2e-5 * Math.max(1, expected.distances[i]),
        `distance ${i}: ${distance} versus ${expected.distances[i]}`,
      ),
    );
    assert.equal(result.execution.applicationUploadBytes, 0);
    const n = result.matches.length / 2;
    assert.equal(
      result.execution.applicationReadbackBytes,
      first.length && second.length ? 8 + 12 * n : 0,
    );
    assert.equal(
      result.execution.residentSnapshotBytes,
      first.length && second.length ? 8 + (n ? (12 * first.length) / dimensions : 0) : 0,
    );
    return result;
  } finally {
    device.drop(a);
    device.drop(b);
  }
}
