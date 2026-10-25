import { fixedJacobianCases, fixedJacobianDigest } from './bundle-fixed-jacobian-cases.mjs';

export function cgScheduleFixtures() {
  const fixtures = fixedJacobianCases({ count: 16, views: 4, iterations: 4 }).map((fixture) => ({
    ...fixture,
    options: { ...fixture.options, maxCgIterations: 96 },
  }));
  const cached = fixtures.find((fixture) => fixture.name === 'local-shared');
  const streamed = fixtures.find((fixture) => fixture.name === 'local');
  return [
    ...fixtures,
    { ...streamed, name: 'local-streamed', device: { workingBytes: 16384 } },
    { ...cached, name: 'local-shared-cache', options: { ...cached.options, schurCache: true } },
  ];
}

export async function runCgScheduleFixture(api, fixture, bounded = false) {
  const device = await api.ShadeDevice.open({
    backend: 'webgpu',
    poolBytes: 0,
    fp64Flavor: 'float',
    boundedSubmissions: bounded,
    solverCheckpoints: false,
    maxDispatchesPerSubmission: 64,
    ...fixture.device,
  });
  // A fast adapter can submit the entire cap in a chunk. This deterministically exposes
  // guarded work after convergence without changing the solve's numerical instructions.
  const originalChunkSize = device.chunkSize.bind(device);
  device.chunkSize = (key, max, min, ...rest) =>
    key.startsWith('bundleCg') || key.startsWith('bundleCachedCg')
      ? max
      : originalChunkSize(key, max, min, ...rest);
  const before = device.statistics;
  const source = await fixedJacobianDigest(fixture.model);
  try {
    const start = performance.now();
    const result = await api.bundleAdjustGpu(device, fixture.model, fixture.options);
    const elapsedMs = performance.now() - start;
    const calls = result.execution.report.groups.flatMap((group) => group.calls);
    return {
      name: fixture.name,
      bounded,
      sourceBefore: source,
      sourceAfter: await fixedJacobianDigest(fixture.model),
      output: await fixedJacobianDigest(result.reconstruction, result),
      cgIterations: result.cgIterations,
      linearSolverFailures: result.linearSolverFailures,
      diagnostics: result.diagnostics,
      elapsedMs,
      dispatches: result.execution.report.dispatches,
      advanceCalls: calls.filter((call) => call === 'advanceBundleCg').length,
      cachedAdvanceCalls: calls.filter((call) => call === 'advanceCachedBundleCg').length,
      certificates: calls.filter((call) => call === 'certifyCachedBundleCg').length,
      streamed: calls.includes('checkBundlePoints'),
      readWaves: device.statistics.readWaves - before.readWaves,
      readbackBytes: result.execution.applicationReadbackBytes,
      liveBytes: device.statistics.liveBytes,
    };
  } finally {
    await device.dispose();
  }
}
