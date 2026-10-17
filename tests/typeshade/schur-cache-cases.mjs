// Independent scalar Schur algebra over captured normal blocks; no TypeCV solvers.
export function scalarSchur(system) {
  const { C, P, M, blocks, index, focal } = system,
    n = 6 * C,
    m = n + 1,
    S = new Float64Array(m * m);
  for (let c = 0; c < C; c++)
    for (let r = 0; r < 6; r++) {
      S[(c * 6 + r) * m + n] = focal[c * 6 + r];
      S[n * m + c * 6 + r] = focal[c * 6 + r];
      for (let k = 0; k < 6; k++)
        S[(c * 6 + r) * m + c * 6 + k] = blocks[45 * M + 13 * P + 36 * c + 6 * r + k];
    }
  S[n * m + n] = focal[n];
  for (let p = 0; p < P; p++) {
    const cross = new Float64Array(m * 3),
      at = 45 * M + p * 13;
    for (let i = index[2 * M + p]; i < index[2 * M + p + 1]; i++) {
      const c = index[i],
        base = 24 * M + i * 21;
      for (let r = 0; r < 6; r++)
        for (let k = 0; k < 3; k++) cross[(c * 6 + r) * 3 + k] += blocks[base + r * 3 + k];
      for (let k = 0; k < 3; k++) cross[n * 3 + k] += blocks[base + 18 + k];
    }
    for (let r = 0; r < m; r++)
      for (let k = 0; k < m; k++)
        for (let a = 0; a < 3; a++)
          for (let b = 0; b < 3; b++)
            S[r * m + k] -= cross[r * 3 + a] * blocks[at + a * 3 + b] * cross[k * 3 + b];
  }
  return S;
}

export function validSpaceVector(system, seed) {
  const { C, M, P, index } = system,
    vector = Float64Array.from(
      { length: 6 * C + 1 },
      (_, i) => 0.013 * Math.sin(i * 0.7 + seed) + 0.007 * Math.cos(i * 1.3 - seed),
    );
  for (let c = 0; c < C; c++) if (!index[3 * M + P + C + 2 + c]) vector.fill(0, c * 6, c * 6 + 6);
  return vector;
}

export function scalarMatrixFree(system, vector) {
  const { C, P, M, blocks, index, focal } = system,
    n = C * 6,
    result = new Float64Array(n + 1),
    eliminated = new Float64Array(P * 3);
  for (let p = 0; p < P; p++) {
    const gathered = new Float64Array(3),
      at = 45 * M + p * 13;
    for (let i = index[2 * M + p]; i < index[2 * M + p + 1]; i++) {
      const c = index[i],
        base = 24 * M + i * 21;
      for (let k = 0; k < 3; k++) {
        gathered[k] += blocks[base + 18 + k] * vector[n];
        for (let r = 0; r < 6; r++) gathered[k] += blocks[base + r * 3 + k] * vector[c * 6 + r];
      }
    }
    for (let r = 0; r < 3; r++)
      for (let k = 0; k < 3; k++) eliminated[p * 3 + r] += blocks[at + r * 3 + k] * gathered[k];
  }
  for (let c = 0; c < C; c++)
    for (let r = 0; r < 6; r++) {
      const row = c * 6 + r;
      result[row] = focal[row] * vector[n];
      for (let k = 0; k < 6; k++)
        result[row] += blocks[45 * M + 13 * P + c * 36 + r * 6 + k] * vector[c * 6 + k];
    }
  result[n] = focal[n] * vector[n];
  for (let row = 0; row < n; row++) result[n] += focal[row] * vector[row];
  for (let i = 0; i < M; i++) {
    const c = index[i],
      p = index[M + i],
      base = 24 * M + i * 21;
    for (let k = 0; k < 3; k++) {
      result[n] -= blocks[base + 18 + k] * eliminated[p * 3 + k];
      for (let r = 0; r < 6; r++)
        result[c * 6 + r] -= blocks[base + r * 3 + k] * eliminated[p * 3 + k];
    }
  }
  return result;
}

export function multiplySchur(matrix, vector, columnMajor = false) {
  const m = vector.length;
  return Float64Array.from({ length: m }, (_, row) => {
    let value = 0;
    for (let k = 0; k < m; k++)
      value += matrix[columnMajor ? k * m + row : row * m + k] * vector[k];
    return value;
  });
}

export function captureSchurSystems(device, input, { corruptCache = false } = {}) {
  const C = input.views.length,
    P = input.landmarks.length,
    M = input.landmarks.reduce((sum, p) => sum + p.observations.length, 0),
    lengths = {
      blocks: 45 * M + 13 * P + 78 * C,
      focal: 6 * C + 6 + 2 * M,
      schur: (6 * C + 1) ** 2,
      state: 26 + 2 * M,
      vectors: 5 * (6 * C + 1) + 3 * P,
    },
    buffers = {},
    systems = [],
    certificates = [],
    allocate = device.allocateF64.bind(device),
    allocateU32 = device.allocateU32.bind(device),
    plan = device.planSteps.bind(device);
  device.allocateF64 = (length, values) => {
    const buffer = allocate(length, values);
    for (const [name, expected] of Object.entries(lengths))
      if (length === expected) buffers[name] = buffer;
    return buffer;
  };
  device.allocateU32 = (length, values) => {
    const buffer = allocateU32(length, values);
    if (length === 3 * M + P + 2 * C + 2) buffers.index = buffer;
    return buffer;
  };
  device.planSteps = (steps) =>
    plan(
      (function* () {
        for (const step of steps) {
          const snapshot =
            step[0] === 'cacheBundleSchurRows' ||
            (device.backend === 'cpu' && step[0] === 'initializeBundleCg');
          if (!snapshot && step[0] !== 'certifyCachedBundleCg') {
            yield step;
            continue;
          }
          yield [
            step[0],
            step[1],
            async () => {
              await step[2]?.();
              if (snapshot) {
                const [blocks, index, focal] = await Promise.all([
                  device.read(buffers.blocks, lengths.blocks),
                  device.read(buffers.index, 3 * M + P + 2 * C + 2),
                  device.read(buffers.focal, lengths.focal),
                ]);
                const system = {
                  C,
                  P,
                  M,
                  blocks: [...blocks],
                  index: [...index],
                  focal: [...focal],
                };
                if (step[0] === 'cacheBundleSchurRows') {
                  system.schur = [...(await device.read(buffers.schur, lengths.schur))];
                  if (corruptCache) await buffers.schur.write(new Float64Array(lengths.schur));
                }
                systems.push(system);
              } else {
                const [state, vectors] = await Promise.all([
                  device.read(buffers.state, lengths.state),
                  device.read(buffers.vectors, lengths.vectors),
                ]);
                certificates.push({ state: [...state], vectors: [...vectors] });
              }
            },
          ];
        }
      })(),
    );
  return { systems, certificates };
}
