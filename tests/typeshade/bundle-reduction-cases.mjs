import { bundleFixture, bundleMetrics } from './geometry-cases.mjs';

// Evaluate the camera product with scalar JS arithmetic from the *same* GPU point product
// and search direction. This isolates reduction error from later PCG/LM trajectory changes.
function cameraProduct({ C, P, M, shared, blocks, vectors, state }) {
  const m = C * 6 + Number(shared),
    expected = vectors.slice(m * 3, m * 4);
  if (!state[19]) return expected;
  const cameraAt = (shared ? 45 : 42) * M + P * 13;
  expected.fill(0);
  for (let row = 0; row < m; row++) {
    if (row === C * 6) {
      // The focal buffer is supplied separately below.
      continue;
    }
    const c = Math.floor(row / 6),
      j = row % 6;
    for (let k = 0; k < 6; k++)
      expected[row] += blocks[cameraAt + c * 36 + j * 6 + k] * vectors[m * 2 + c * 6 + k];
  }
  return expected;
}

function captureProducts(device, input, shared) {
  const C = input.views.length,
    P = input.landmarks.length,
    M = input.landmarks.reduce((n, p) => n + p.observations.length, 0),
    m = C * 6 + Number(shared),
    lengths = {
      blocks: (shared ? 45 : 42) * M + P * 13 + C * 78,
      vectors: m * 5 + P * 3,
      state: 26,
      focal: shared ? C * 6 + 6 + M * 2 : -1,
    },
    buffers = {},
    costs = [],
    allocate = device.allocateF64.bind(device),
    plan = device.planSteps.bind(device);
  let before,
    checked = 0,
    idle = 0,
    maxRelativeError = 0;
  device.allocateF64 = (length, values) => {
    const buffer = allocate(length, values);
    for (const [name, n] of Object.entries(lengths)) if (n === length) buffers[name] = buffer;
    return buffer;
  };
  device.planSteps = (steps) =>
    plan(
      (function* () {
        for (const step of steps) {
          yield [
            step[0],
            step[1],
            async () => {
              await step[2]?.();
              if (step[0] === 'bundleSchurPoint') {
                const [blocks, vectors, state, focal] = await Promise.all([
                  device.read(buffers.blocks),
                  device.read(buffers.vectors),
                  device.read(buffers.state),
                  shared ? device.read(buffers.focal) : undefined,
                ]);
                before = { blocks, vectors, state, focal };
              } else if (step[0].startsWith('bundleSchurCamera')) {
                const { blocks, vectors, state, focal } = before,
                  expected = cameraProduct({ C, P, M, shared, blocks, vectors, state }),
                  actual = (await device.read(buffers.vectors)).slice(m * 3, m * 4);
                if (state[19]) {
                  checked++;
                  if (shared) {
                    for (let row = 0; row < C * 6; row++)
                      expected[row] += focal[row] * vectors[m * 2 + m - 1];
                    expected[m - 1] = focal[m - 1] * vectors[m * 2 + m - 1];
                    for (let row = 0; row < C * 6; row++)
                      expected[m - 1] += focal[row] * vectors[m * 2 + row];
                  }
                  let i = 0;
                  for (const point of input.landmarks) {
                    for (const observation of point.observations) {
                      const c = input.views.findIndex((v) => v.id === observation.viewId),
                        p = Math.floor(i / C),
                        cross = M * 24 + i * (shared ? 21 : 18);
                      for (let k = 0; k < 3; k++) {
                        for (let j = 0; j < 6; j++)
                          expected[c * 6 + j] -=
                            blocks[cross + j * 3 + k] * vectors[m * 5 + p * 3 + k];
                        if (shared)
                          expected[m - 1] -= blocks[cross + 18 + k] * vectors[m * 5 + p * 3 + k];
                      }
                      i++;
                    }
                  }
                } else idle++;
                const scale = Math.max(1, ...expected.map(Math.abs));
                for (let row = 0; row < m; row++)
                  maxRelativeError = Math.max(
                    maxRelativeError,
                    Math.abs(actual[row] - expected[row]) / scale,
                  );
              } else if (step[0] === 'acceptBundleCandidate') {
                costs.push((await device.read(buffers.state))[6]);
              }
            },
            step[3],
          ];
        }
      })(),
    );
  return () => ({ checked, idle, maxRelativeError, costs });
}

function streamingBudget(input, shared) {
  const C = input.views.length,
    P = input.landmarks.length,
    M = P * C;
  const doubles =
    18 * C +
    3 * P +
    2 * M +
    (12 * C + 3 * P) +
    (30 * C + 3 * P + (shared ? 5 : 0)) +
    26 +
    3 * Math.ceil(M / 64) +
    (12 * C + 3 * P + (shared ? 27 : 26)) +
    78 * C +
    36 * C +
    (shared ? 6 * C + 6 : 0);
  return 8 * doubles + 4 * (3 * M + P + 2 * C + 2) + 8 + 64 * (shared ? 376 : 336) + 64 * 104;
}

export async function bundleReductionCases({ ShadeDevice, bundleAdjustGpu }, gpu) {
  const rows = [];
  for (const config of [
    { name: 'fixed', shared: false, count: 131, views: 4, bounded: true, iterations: 6 },
    { name: 'focal', shared: true, count: 131, views: 4, bounded: true, iterations: 6 },
    { name: '17 cameras', shared: false, count: 131, views: 17, iterations: 3 },
    { name: '17 cameras with focal', shared: true, count: 131, views: 17, iterations: 3 },
    {
      name: 'streamed local fixed',
      shared: false,
      count: 67,
      views: 6,
      streamed: true,
      iterations: 3,
    },
    {
      name: 'streamed local focal',
      shared: true,
      count: 67,
      views: 6,
      streamed: true,
      iterations: 3,
    },
    {
      name: 'cached focal certificate',
      shared: true,
      count: 131,
      views: 4,
      cached: true,
      iterations: 3,
    },
  ]) {
    const input = bundleFixture({ count: config.count, views: config.views, outlier: true }),
      original = JSON.stringify(input),
      results = [];
    for (const mode of ['serial', 'cooperative']) {
      const device = await ShadeDevice.open({
        gpuDevice: gpu,
        poolBytes: 0,
        boundedSubmissions: config.bounded ?? false,
        solverCheckpoints: true,
        workingBytes: config.streamed ? streamingBudget(input, config.shared) : 64 * 1024 ** 2,
      });
      const capture = config.bounded ? captureProducts(device, input, config.shared) : undefined;
      try {
        const result = await bundleAdjustGpu(device, input, {
          iterations: config.iterations,
          maxCgIterations: 64,
          huberPx: 1,
          refineSharedFocal: config.shared,
          schurReduction: mode,
          schurCache: config.cached ?? false,
          ...(config.streamed ? { variableViewIds: [30, 40] } : {}),
        });
        results.push({
          model: result.reconstruction,
          metrics: bundleMetrics(result.reconstruction, 1),
          diagnostics: result.diagnostics,
          failures: result.linearSolverFailures,
          tier: result.execution.report.tier,
          streamed: result.execution.report.groups.some((g) =>
            g.calls.includes('checkBundlePoints'),
          ),
          capture: capture?.(),
        });
      } finally {
        if (device.statistics.liveBuffers) throw new Error('BA reduction leaked resident buffers');
        await device.dispose();
      }
    }
    if (JSON.stringify(input) !== original) throw new Error('BA reduction mutated input');
    const [serial, cooperative] = results,
      coordinates = (model) => [
        ...model.views.flatMap((v) => [...v.pose.R, ...v.pose.t, v.camera.fx, v.camera.fy]),
        ...model.landmarks.flatMap((p) => [...p.xyz]),
      ],
      a = coordinates(serial.model),
      b = coordinates(cooperative.model);
    rows.push({
      name: config.name,
      maxCoordinateError: Math.max(
        ...a.map((x, i) => Math.abs(x - b[i]) / Math.max(1, Math.abs(x))),
      ),
      runs: results.map(({ model, ...rest }) => rest),
    });
  }
  return rows;
}
