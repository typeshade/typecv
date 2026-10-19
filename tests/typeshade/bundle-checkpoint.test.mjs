import test from 'node:test';
import assert from 'node:assert/strict';
import { bundleAdjustGpu, ShadeDevice } from '../../dist/gpu.js';
import { bundleFixture } from './geometry-cases.mjs';

/** Opens a CPU device that records each step name and, per plan, its kernel calls and chunk. */
async function recordingDevice(settings) {
  const device = await ShadeDevice.open({ backend: 'cpu', ...settings });
  const schedule = [],
    plans = [],
    planSteps = device.planSteps.bind(device),
    plan = device.plan.bind(device);
  device.planSteps = (steps) =>
    planSteps(
      (function* () {
        for (const step of steps) {
          schedule.push(step[0]);
          yield step;
        }
      })(),
    );
  device.plan = async (body, chunk) => {
    const report = await plan(body, chunk);
    plans.push({ calls: report.groups.flatMap((group) => group.calls), chunk });
    return report;
  };
  return { device, schedule, plans };
}

const calls = (result, name) =>
  result.execution.report.groups.flatMap((group) => group.calls).filter((call) => call === name)
    .length;

test('bounded BA checkpoints skip guarded no-op work without changing solver results', async () => {
  const input = bundleFixture({ outlier: true }),
    options = { iterations: 15, huberPx: 1 },
    results = [],
    runs = [];
  for (const settings of [
    { boundedSubmissions: false },
    { boundedSubmissions: true, maxDispatchesPerSubmission: 1 },
    { boundedSubmissions: true, maxDispatchesPerSubmission: 16 },
  ]) {
    const run = await recordingDevice(settings);
    try {
      results.push(await bundleAdjustGpu(run.device, input, options));
    } finally {
      await run.device.dispose();
    }
    assert.equal(run.device.statistics.liveBuffers, 0);
    runs.push({ ...run, settings });
  }
  const [unbounded, boundedSingle, boundedGrouped] = results;
  // The unbounded path runs the PCG loop as chunks (key `bundleCg`) and reads the activity flags
  // after each; the CPU tier runs the whole loop of an LM iteration as one chunk.
  assert.ok(runs[0].schedule.includes('bundleCg'));
  assert.ok(runs[0].plans.some((p) => p.chunk?.key === 'bundleCg'));
  for (let i = 1; i < results.length; i++) {
    const bounded = results[i];
    assert.deepEqual(bounded.reconstruction, unbounded.reconstruction);
    assert.deepEqual(bounded.diagnostics, unbounded.diagnostics);
    assert.equal(bounded.cgIterations, unbounded.cgIterations);
    assert.equal(bounded.linearSolverFailures, unbounded.linearSolverFailures);
    assert.deepEqual(bounded.execution.cpuStages, unbounded.execution.cpuStages);
    // Bounded submissions keep one step per dispatch and no chunk step, so that no plan holds
    // more than `maxDispatchesPerSubmission` dispatches.
    assert.ok(runs[i].schedule.includes('snapshotBundleActivity'));
    assert.ok(!runs[i].schedule.includes('bundleCg'));
    assert.ok(runs[i].plans.every((p) => !p.chunk));
    assert.ok(
      runs[i].plans.every((p) => p.calls.length <= runs[i].settings.maxDispatchesPerSubmission),
    );
    // Its checkpoints every four PCG iterations skip more no-op iterations than one chunk of
    // the whole loop does.
    assert.ok(
      calls(bounded, 'advanceBundleCg') < calls(unbounded, 'advanceBundleCg'),
      JSON.stringify({
        unbounded: calls(unbounded, 'advanceBundleCg'),
        bounded: calls(bounded, 'advanceBundleCg'),
      }),
    );
    assert.ok(
      bounded.execution.applicationReadbackBytes > unbounded.execution.applicationReadbackBytes,
    );
  }
  assert.deepEqual(boundedGrouped.reconstruction, boundedSingle.reconstruction);
  assert.deepEqual(boundedGrouped.diagnostics, boundedSingle.diagnostics);
  assert.equal(boundedGrouped.cgIterations, boundedSingle.cgIterations);
});

test('the BA PCG loop runs as timed chunks of the time budget with unchanged results', async () => {
  const options = { iterations: 12, huberPx: 1 };
  for (const fixture of [{}, { outlier: true }, { count: 96, views: 6 }]) {
    const input = bundleFixture(fixture),
      C = input.views.length,
      M = input.landmarks.reduce((n, p) => n + p.observations.length, 0),
      maxCg = Math.min(96, Math.max(24, C * 12)),
      // The units of one PCG iteration: a row per observation and per camera parameter.
      work = M + 6 * C;
    // The reference: bounded submissions without solver checkpoints run every dispatch of every
    // PCG and LM iteration, one step each, which is the per-iteration sequence of the former
    // unbounded path.
    const reference = await recordingDevice({
      boundedSubmissions: true,
      maxDispatchesPerSubmission: 64,
      solverCheckpoints: false,
    });
    // The CPU tier: `chunkSize` returns `max`, so one chunk holds the whole loop.
    const whole = await recordingDevice({});
    // A forced least chunk: what the GPU tier does on a slow device, one iteration a chunk.
    const small = await recordingDevice({});
    small.device.chunkSize = (key, max, min) => {
      assert.equal(key, 'bundleCg');
      assert.equal(min, work);
      assert.equal(max % work, 0);
      return min;
    };
    const out = [];
    for (const run of [reference, whole, small]) {
      try {
        out.push(await bundleAdjustGpu(run.device, input, options));
      } finally {
        await run.device.dispose();
      }
      assert.equal(run.device.statistics.liveBuffers, 0);
    }
    const [expected, ...chunked] = out;
    assert.equal(calls(expected, 'advanceBundleCg'), options.iterations * maxCg);
    for (const [i, result] of chunked.entries()) {
      const run = [whole, small][i];
      // Bit for bit: the chunks issue the same dispatches in the same order, and the iterations
      // they skip after convergence are guarded no-ops.
      assert.deepEqual(result.reconstruction, expected.reconstruction);
      assert.deepEqual(result.diagnostics, expected.diagnostics);
      assert.equal(result.cgIterations, expected.cgIterations);
      assert.equal(result.linearSolverFailures, expected.linearSolverFailures);
      assert.ok(calls(result, 'advanceBundleCg') < calls(expected, 'advanceBundleCg'));
      // Each chunk is a plan of its own: `n` iterations of bundleSchurPoint, bundleSchurCamera
      // and advanceBundleCg, then the activity flags the host reads.
      const chunks = run.plans.filter((p) => p.chunk);
      assert.ok(chunks.length > 0);
      for (const { calls: kernels, chunk } of chunks) {
        assert.equal(chunk.key, 'bundleCg');
        assert.equal(chunk.units % work, 0);
        const iterations = chunk.units / work;
        assert.equal(kernels.length, 3 * iterations + 1);
        for (let k = 0; k < iterations; k++)
          assert.deepEqual(kernels.slice(3 * k, 3 * k + 3), [
            'bundleSchurPoint',
            'bundleSchurCamera',
            'advanceBundleCg',
          ]);
        assert.equal(kernels.at(-1), 'snapshotBundleActivity');
        assert.equal(iterations, run === whole ? maxCg : 1);
      }
      // The report carries the rate the TypeShade device learned of the key.
      const rate = result.execution.report.chunks?.bundleCg;
      assert.ok(rate, JSON.stringify(result.execution.report.chunks));
      assert.equal(rate.chunks, chunks.length);
      assert.equal(rate.lastUnits, chunks.at(-1).chunk.units);
      assert.ok(rate.longestMs >= 0);
    }
    // A smaller chunk reads the flags more often, so it stops sooner after convergence.
    assert.ok(calls(chunked[1], 'advanceBundleCg') < calls(chunked[0], 'advanceBundleCg'));
    assert.equal(reference.plans.filter((p) => p.chunk).length, 0);
  }
});
