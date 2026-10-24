// Same-device raw-kernel differential checks. Descriptors and nearest-neighbour outputs stay
// separate from the high-level ratio/compaction path, so ties and invalid flags are observable.
function sameBits(first, second, label) {
  if (first.byteLength !== second.byteLength) throw new Error(`${label}: output length changed`);
  const a = new Uint32Array(first.buffer, first.byteOffset, first.byteLength / 4),
    b = new Uint32Array(second.buffer, second.byteOffset, second.byteLength / 4);
  for (let i = 0; i < a.length; i++)
    if (a[i] !== b[i])
      throw new Error(
        `${label}: output bits differ at ${i}: 0x${a[i].toString(16)} vs 0x${b[i].toString(16)}`,
      );
  return a.length;
}

function descriptors(rows, dimensions, seed) {
  let state = seed;
  return Float32Array.from({ length: rows * dimensions }, (_, index) => {
    state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
    return ((state >>> 8) / 0xffffff - 0.5) * (1 + (index % 5) * 0.125);
  });
}

export async function floatMatchingWideCases(api, deviceOptions = { backend: 'webgpu' }) {
  const device = await api.ShadeDevice.open({ ...deviceOptions, poolBytes: 0 });
  if (device.backend !== 'webgpu') {
    await device.dispose();
    throw new Error('Wide float-matching differential fixture requires WebGPU');
  }
  const cases = [
      { name: 'one row, short descriptors', queries: 1, candidates: 1, dimensions: 17 },
      { name: 'seven by nine, standard SIFT', queries: 7, candidates: 9, dimensions: 128 },
      { name: 'one exact tile', queries: 8, candidates: 8, dimensions: 128 },
      { name: 'channel and row tails', queries: 9, candidates: 7, dimensions: 129 },
      { name: 'multiple slabs and tiles', queries: 63, candidates: 65, dimensions: 257 },
      { name: 'reversed multiple tile tails', queries: 65, candidates: 63, dimensions: 128 },
      {
        name: 'duplicate equal-distance ties',
        queries: 9,
        candidates: 17,
        dimensions: 129,
        kind: 'ties',
      },
      {
        name: 'all identical zero rows',
        queries: 7,
        candidates: 9,
        dimensions: 257,
        kind: 'zeros',
      },
      {
        name: 'finite squared-distance overflow',
        queries: 9,
        candidates: 7,
        dimensions: 129,
        kind: 'overflow',
      },
      { name: 'NaN on both sides', queries: 9, candidates: 8, dimensions: 128, kind: 'nan' },
      {
        name: 'positive infinity across slab tail',
        queries: 8,
        candidates: 9,
        dimensions: 129,
        kind: 'positive-infinity',
      },
      {
        name: 'negative infinity across multiple slabs',
        queries: 9,
        candidates: 7,
        dimensions: 257,
        kind: 'negative-infinity',
      },
    ],
    rows = [];
  let comparedValues = 0;
  try {
    for (let index = 0; index < cases.length; index++) {
      const c = cases[index],
        first = descriptors(c.queries, c.dimensions, 0x193b5 + index),
        second = descriptors(c.candidates, c.dimensions, 0x719d3 + index);
      if (c.kind === 'zeros') {
        first.fill(0);
        second.fill(0);
      } else if (c.kind === 'ties') {
        second.set(first.subarray(0, c.dimensions), 0);
        second.set(first.subarray(0, c.dimensions), c.dimensions);
        first.set(first.subarray(0, c.dimensions), c.dimensions);
      } else if (c.kind === 'overflow') {
        first.fill(1e30, 0, c.dimensions);
        second.fill(-1e30, (c.candidates - 1) * c.dimensions);
      } else if (c.kind) {
        const value =
          c.kind === 'nan' ? NaN : c.kind === 'positive-infinity' ? Infinity : -Infinity;
        first[(c.queries - 1) * c.dimensions + c.dimensions - 1] = value;
        second[Math.floor(c.candidates / 2) * c.dimensions] = value;
      }
      const result = await device.enqueue(async () => {
        const owned = [],
          f32 = (values) => {
            const handle = device.allocate(values.length, values);
            owned.push(handle);
            return handle;
          },
          u32 = (values) => {
            const handle = device.allocateU32(values.length, values);
            owned.push(handle);
            return handle;
          };
        try {
          const firstBuffer = f32(first),
            secondBuffer = f32(second),
            directions = [];
          for (const direction of ['forward', 'backward']) {
            const queries = direction === 'forward' ? c.queries : c.candidates,
              candidates = direction === 'forward' ? c.candidates : c.queries,
              query = direction === 'forward' ? firstBuffer : secondBuffer,
              candidate = direction === 'forward' ? secondBuffer : firstBuffer,
              // Whole outputs include untouched sentinel words after the live region.
              indices = new Uint32Array(queries * 3 + 5).fill(0x88aaff17),
              distances = new Float32Array(queries * 2 + 7).fill(-17.25),
              originalIndices = u32(indices),
              wideIndices = u32(indices),
              originalDistances = f32(distances),
              wideDistances = f32(distances),
              control = {
                queries,
                candidates,
                dimensions: c.dimensions,
                ratioSquared: 0.64,
                maxDistanceSquared: 1e38,
                mutual: 1,
              };
            await device.planSteps([
              [
                'nearestFloatTiles',
                () =>
                  api.nearestFloatTiles(
                    {
                      floatQueries: query,
                      floatCandidates: candidate,
                      nearestIndices: originalIndices,
                      nearestDistances: originalDistances,
                      floatMatchControl: control,
                    },
                    Math.ceil(queries / 8),
                  ),
              ],
              [
                'nearestFloatTilesWide',
                () =>
                  api.nearestFloatTilesWide(
                    {
                      floatQueries: query,
                      floatCandidates: candidate,
                      nearestIndices: wideIndices,
                      nearestDistances: wideDistances,
                      floatMatchControl: control,
                    },
                    Math.ceil(queries / 8),
                  ),
              ],
            ]);
            const outputs = await Promise.all([
              device.read(originalIndices, indices.length),
              device.read(wideIndices, indices.length),
              device.read(originalDistances, distances.length),
              device.read(wideDistances, distances.length),
            ]);
            const compared =
              sameBits(outputs[0], outputs[1], `${c.name}/${direction}/indices`) +
              sameBits(outputs[2], outputs[3], `${c.name}/${direction}/distances`);
            let invalidQueries = 0;
            for (let q = 0; q < queries; q++) if (outputs[0][q * 3 + 2]) invalidQueries++;
            if (
              ['overflow', 'nan', 'positive-infinity', 'negative-infinity'].includes(c.kind) &&
              !invalidQueries
            )
              throw new Error(`${c.name}/${direction}: fixture did not exercise invalid flags`);
            if (c.kind === 'zeros') {
              for (let q = 0; q < queries; q++) {
                if (outputs[0][q * 3] !== 0 || outputs[0][q * 3 + 1] !== 1)
                  throw new Error(`${c.name}/${direction}: duplicate tie order changed`);
              }
            }
            directions.push({
              direction,
              queries,
              candidates,
              comparedValues: compared,
              invalidQueries,
            });
          }
          return directions;
        } finally {
          for (const handle of owned) device.drop(handle);
        }
      });
      comparedValues += result.reduce((sum, row) => sum + row.comparedValues, 0);
      rows.push({ ...c, directions: result });
    }
    if (device.statistics.liveBytes) throw new Error('Wide float-matching fixture leaked arrays');
    return {
      backend: device.backend,
      cases: rows,
      comparedValues,
      bitwiseIdentical: true,
      liveBytes: device.statistics.liveBytes,
    };
  } finally {
    await device.dispose();
  }
}
