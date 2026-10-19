import test from 'node:test';
import assert from 'node:assert/strict';
import {
  executeReconstructionJob,
  incrementalSteps,
  reconstructIncremental,
} from '../src/reconstruction/mapper.ts';
import type { IncrementalResult, ReconstructionJob } from '../src/reconstruction/mapper.ts';
import {
  HostSceneDriver,
  reconstructIncrementalScene,
  sceneSteps,
} from '../src/reconstruction/scene-mapper.ts';
import type { SceneRequest } from '../src/reconstruction/scene-mapper.ts';
import type { SparseReconstruction } from '../src/reconstruction/types.ts';
import { scene } from './scene.ts';

/** The summation order of the mean error differs (arrival order against point order); the rest is equal. */
function sameDiagnostics(
  a: IncrementalResult['diagnostics'],
  b: IncrementalResult['diagnostics'],
  message: string,
) {
  const { meanReprojectionErrorPx: ea, ...restA } = a,
    { meanReprojectionErrorPx: eb, ...restB } = b;
  assert.deepEqual(restA, restB, message);
  assert.ok(Math.abs(ea - eb) <= 1e-12 * Math.max(1, ea), `${message}: mean error ${ea} vs ${eb}`);
}
function sameModel(a: SparseReconstruction, b: SparseReconstruction, message: string) {
  assert.deepEqual(
    a.views.map((v) => v.id),
    b.views.map((v) => v.id),
    `${message}: views`,
  );
  assert.deepEqual(
    a.landmarks.map((p) => p.trackId),
    b.landmarks.map((p) => p.trackId),
    `${message}: landmarks`,
  );
  for (const [i, v] of a.views.entries()) {
    assert.deepEqual(v.camera, b.views[i]!.camera, `${message}: camera ${v.id}`);
    assert.deepEqual([...v.pose.R], [...b.views[i]!.pose.R], `${message}: rotation ${v.id}`);
    assert.deepEqual([...v.pose.t], [...b.views[i]!.pose.t], `${message}: translation ${v.id}`);
  }
  for (const [i, p] of a.landmarks.entries()) {
    assert.deepEqual([...p.xyz], [...b.landmarks[i]!.xyz], `${message}: point ${p.trackId}`);
    assert.deepEqual(
      p.observations,
      b.landmarks[i]!.observations,
      `${message}: observations ${p.trackId}`,
    );
  }
}

test('the scene mapper on the CPU driver builds the same model as the host-model mapper', () => {
  for (const [name, options] of [
    ['global', { seed: 42, ransacIterations: 300, initialPair: [10, 60] as const }],
    [
      'local subset',
      {
        seed: 42,
        ransacIterations: 300,
        localBundleSubset: true,
        globalBundleFrequency: 4,
        localBundleSize: 2,
      },
    ],
    ['shared focal', { seed: 7, ransacIterations: 300, refineSharedFocal: true }],
  ] as const) {
    const { inputs, pairs } = scene({
      count: 90,
      views: 6,
      noise: 0.2,
      camera: { fx: 850, fy: 870, cx: 480, cy: 270, k1: -0.03, k2: 0.004 },
    });
    const snapshots: SparseReconstruction[] = [],
      sceneSnapshots: SparseReconstruction[] = [];
    const reference = reconstructIncremental(inputs, pairs, {
        ...options,
        onReconstruction: (m) => snapshots.push(m),
      }),
      result = reconstructIncrementalScene(inputs, pairs, {
        ...options,
        onReconstruction: (m) => sceneSnapshots.push(m),
      });
    assert.equal(result.reconstruction.views.length, 6, name);
    sameModel(result.reconstruction, reference.reconstruction, name);
    assert.deepEqual(result.unregistered, reference.unregistered, name);
    sameDiagnostics(result.diagnostics, reference.diagnostics, name);
    assert.equal(sceneSnapshots.length, snapshots.length, name);
    snapshots.forEach((m, i) => sameModel(sceneSnapshots[i]!, m, `${name} snapshot ${i}`));
  }
});

test('the scene mapper retries a refused view and recovers, as the host-model mapper does', () => {
  const { inputs, pairs } = scene({ count: 60, views: 5, noise: 0.1 });
  const options = { seed: 3, ransacIterations: 200, initialPair: [10, 20] as const };
  // Refuse each view's first PnP, by its pixels, in both drivers.
  const refused = new Set<string>();
  const refuse = (pixels: Float64Array) => {
    const key = pixels.subarray(0, 4).join(',');
    if (refused.has(key)) return false;
    refused.add(key);
    return true;
  };
  const machine = sceneSteps(inputs, pairs, options),
    driver = new HostSceneDriver();
  let step = machine.next();
  while (!step.done) {
    let value: unknown;
    try {
      const job = step.value as SceneRequest;
      if (job.kind === 'pnp' && refuse(job.pixels)) throw new Error('refused');
      value = driver.execute(job);
    } catch (error) {
      step = machine.throw(error);
      continue;
    }
    step = machine.next(value);
  }
  const result = step.value;
  refused.clear();
  const reference = incrementalSteps(inputs, pairs, options);
  let ref = reference.next();
  while (!ref.done) {
    let value: unknown;
    try {
      const job = ref.value as ReconstructionJob;
      if (job.kind === 'pnp' && refuse(job.args[1])) throw new Error('refused');
      value = executeReconstructionJob(job);
    } catch (error) {
      ref = reference.throw(error);
      continue;
    }
    ref = reference.next(value);
  }
  assert.equal(result.reconstruction.views.length, 5);
  sameModel(result.reconstruction, ref.value.reconstruction, 'recovery');
  assert.deepEqual(result.diagnostics.registrationOrder, ref.value.diagnostics.registrationOrder);
});

test('the scene mapper resumes from a model with the same result as the host-model mapper', () => {
  const { inputs, pairs } = scene({ count: 70, views: 6, noise: 0.1 });
  const options = { seed: 11, ransacIterations: 200, initialPair: [10, 20] as const };
  const partial = reconstructIncremental(
    inputs.slice(0, 4),
    pairs.filter((p) => p.second <= 40),
    options,
  );
  const run = <J, R>(machine: Generator<J, R, unknown>, execute: (job: J) => unknown): R => {
    let step = machine.next();
    while (!step.done) {
      let value: unknown;
      try {
        value = execute(step.value);
      } catch (error) {
        step = machine.throw(error);
        continue;
      }
      step = machine.next(value);
    }
    return step.value;
  };
  const reference = run(
      incrementalSteps(inputs, pairs, options, partial.reconstruction),
      executeReconstructionJob,
    ),
    driver = new HostSceneDriver(),
    result = run(sceneSteps(inputs, pairs, options, partial.reconstruction), (job) =>
      driver.execute(job),
    );
  assert.equal(result.reconstruction.views.length, 6);
  sameModel(result.reconstruction, reference.reconstruction, 'resume');
  sameDiagnostics(result.diagnostics, reference.diagnostics, 'resume');
});
