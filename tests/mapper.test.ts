import test from 'node:test';
import assert from 'node:assert/strict';
import { reconstructIncremental } from '../src/index.ts';
import { executeReconstructionJob, incrementalSteps } from '../src/reconstruction/mapper.ts';
import { cameraCenter, projectPoint, rotationExp } from '../src/geometry/pose.ts';
import { multiply, transpose } from '../src/sfm/linear.ts';
import type { ImageFeatures, PairMatches } from '../src/index.ts';
import { scene } from './scene.ts';

test('incremental mapper registers six views in one frame and scale under noisy observations', () => {
  const { inputs, pairs, registered } = scene({
    count: 90,
    views: 6,
    noise: 0.2,
    camera: { fx: 850, fy: 870, cx: 480, cy: 270, k1: -0.03, k2: 0.004 },
  });
  const inputSnapshot = structuredClone(inputs),
    pairSnapshot = structuredClone(pairs);
  const result = reconstructIncremental(inputs, pairs, {
    seed: 42,
    ransacIterations: 300,
    initialPair: [10, 60],
  });
  assert.equal(result.reconstruction.views.length, 6);
  assert.equal(result.unregistered.length, 0);
  assert.ok(result.reconstruction.landmarks.length >= 85);
  assert.ok(result.diagnostics.meanReprojectionErrorPx < 0.3);
  const first = registered[0]!,
    second = registered[5]!,
    scale =
      1 / Math.hypot(...cameraCenter(second.pose).map((v, i) => v - cameraCenter(first.pose)[i]!));
  for (const v of result.reconstruction.views) {
    const truth = registered.find((t) => t.id === v.id)!,
      R = multiply(v.pose.R, transpose(truth.pose.R, 3), 3);
    const angle = Math.acos(Math.max(-1, Math.min(1, (R[0]! + R[4]! + R[8]! - 1) / 2)));
    const c = cameraCenter(v.pose),
      gt = cameraCenter(truth.pose);
    assert.ok(angle < 0.01, `${v.id} rotation ${angle}`);
    assert.ok(Math.hypot(...c.map((x, i) => x - scale * gt[i]!)) < 0.03, `${v.id} center`);
  }
  assert.ok(result.reconstruction.landmarks.every((p) => p.observations.length >= 5));
  assert.deepEqual(inputs, inputSnapshot);
  assert.deepEqual(pairs, pairSnapshot);
  console.log(
    'mapper',
    JSON.stringify({
      views: result.reconstruction.views.length,
      points: result.reconstruction.landmarks.length,
      error: result.diagnostics.meanReprojectionErrorPx,
    }),
  );
});

test('tracks and PnP register a camera without direct matches to the initial pair', () => {
  const { inputs, pairs } = scene({ count: 60, views: 4, noise: 0 });
  const selected = pairs.filter(
    (p) =>
      (p.first === 10 && p.second === 30) ||
      (p.first === 20 && p.second === 30) ||
      (p.first === 20 && p.second === 40),
  );
  const result = reconstructIncremental(inputs, selected, {
    initialPair: [10, 30],
    ransacIterations: 150,
  });
  assert.equal(result.reconstruction.views.length, 4);
  assert.equal(result.unregistered.length, 0);
  assert.deepEqual(result.diagnostics.registrationOrder, [10, 30, 20, 40]);
});

test('a recovery pass retries registrations that failed at one map size', () => {
  const { inputs, pairs } = scene({ count: 60, views: 4 });
  const machine = incrementalSteps(inputs, pairs, { ransacIterations: 150, seed: 7 });
  // Every first PnP attempt at a map size fails, as a marginal consensus does on another device.
  const refused = new Set<string>();
  const kinds: string[] = [];
  let lastRefusal = -1,
    firstSuccess = -1;
  let step = machine.next();
  while (!step.done) {
    const job = step.value;
    kinds.push(job.kind);
    if (job.kind === 'pnp') {
      const key = `${job.args[0].length}:${job.args[3]?.seed}`;
      if (!refused.has(key)) {
        refused.add(key);
        lastRefusal = kinds.length - 1;
        step = machine.throw(
          new Error('PnP could not register camera: insufficient nondegenerate consensus'),
        );
        continue;
      }
      if (firstSuccess < 0) firstSuccess = kinds.length - 1;
    }
    step = machine.next(executeReconstructionJob(job));
  }
  const result = step.value;
  assert.equal(result.reconstruction.views.length, 4);
  assert.equal(result.unregistered.length, 0);
  // Both candidates were refused at the initial map size; the recovery pass refined the map (every
  // track of this scene was already triangulated) before the first retry succeeded.
  const firstRetry = kinds.indexOf('pnp', kinds.indexOf('pnp') + 2);
  assert.ok(firstSuccess === firstRetry && firstRetry > 0, kinds.join(','));
  const between = kinds.slice(lastRefusal + 1, firstSuccess);
  assert.ok(between.includes('bundle'), kinds.join(','));
  assert.ok(!between.includes('pnp'), kinds.join(','));
});

test('disconnected frames remain explicitly unregistered', () => {
  const { inputs, pairs } = scene({ count: 60, views: 3 });
  const result = reconstructIncremental(
    inputs,
    pairs.filter((p) => p.first === 10 && p.second === 30),
    { ransacIterations: 150 },
  );
  assert.equal(result.reconstruction.views.length, 2);
  assert.equal(result.unregistered[0]!.viewId, 20);
  assert.match(result.unregistered[0]!.reason, /correspondences/);
});

test('homography-dominated planar and pure rotation sequences cannot initialize a map', () => {
  const { inputs, points } = scene({ count: 50, views: 3 });
  const planar: ImageFeatures[] = inputs.map((v, i) => ({
    id: v.id,
    camera: v.camera,
    keypoints: Float64Array.from(
      points.flatMap((p) => {
        const pose = { R: rotationExp([0, 0.02 * i, 0]), t: Float64Array.of(i * 0.3, 0, 0) };
        return [...projectPoint(v.camera, pose, Float64Array.of(p[0]!, p[1]!, 6))!];
      }),
    ),
  }));
  const pair: PairMatches = {
    first: 10,
    second: 30,
    matches: Uint32Array.from(points.flatMap((_, i) => [i, i])),
  };
  assert.throws(
    () => reconstructIncremental(planar, [pair], { ransacIterations: 100 }),
    /stable initial pair/,
  );
  const rotation: ImageFeatures[] = inputs.map((v, i) => ({
    ...v,
    keypoints: Float64Array.from(
      points.flatMap((p) => [
        ...projectPoint(v.camera, { R: rotationExp([0, 0.05 * i, 0]), t: new Float64Array(3) }, p)!,
      ]),
    ),
  }));
  assert.throws(
    () => reconstructIncremental(rotation, [pair], { ransacIterations: 100 }),
    /stable initial pair/,
  );
});

test('mapper validates controls and duplicate pair identity', () => {
  const { inputs, pairs } = scene({ count: 20, views: 2 });
  assert.throws(() => reconstructIncremental(inputs, [...pairs, ...pairs]), /Duplicate/);
  assert.throws(() => reconstructIncremental(inputs, pairs, { minPnPInliers: 3 }), RangeError);
  assert.throws(
    () => reconstructIncremental(inputs, pairs, { initialPair: [10, 999] }),
    RangeError,
  );
});

test('new tracks triangulate after registration and extend beyond initial visibility', () => {
  const { inputs, pairs } = scene({ count: 80, views: 5, noise: 0.1 });
  const visible = (viewId: number, i: number) => (i < 40 ? viewId <= 30 : viewId >= 20);
  const masked = pairs
    .map((p) => ({
      ...p,
      matches: Uint32Array.from(
        Array.from({ length: 80 }, (_, i) => i)
          .filter((i) => visible(p.first, i) && visible(p.second, i))
          .flatMap((i) => [i, i]),
      ),
    }))
    .filter((p) => p.matches.length);
  const result = reconstructIncremental(inputs, masked, {
    initialPair: [10, 30],
    ransacIterations: 250,
  });
  assert.equal(result.reconstruction.views.length, 5);
  assert.equal(result.reconstruction.landmarks.length, 80);
  assert.ok(
    result.reconstruction.landmarks.some(
      (p) =>
        !p.observations.some((o) => o.viewId === 10) && p.observations.some((o) => o.viewId === 50),
    ),
  );
  assert.ok(result.diagnostics.meanReprojectionErrorPx < 0.2);
});

test('geometric verification prevents random pairwise mismatches from corrupting the map', () => {
  const { inputs, pairs } = scene({ count: 80, views: 4, noise: 0.2 });
  const corrupted = pairs.map((p) => ({
    ...p,
    matches: Uint32Array.from(p.matches, (v, i) => (i % 2 && i < 40 ? (v + 13) % 80 : v)),
  }));
  const result = reconstructIncremental(inputs, corrupted, {
    ransacIterations: 500,
    initialPair: [10, 40],
  });
  assert.equal(result.reconstruction.views.length, 4);
  assert.ok(result.reconstruction.landmarks.length >= 55);
  assert.ok(result.diagnostics.meanReprojectionErrorPx < 0.5);
});

test('incremental reconstruction is deterministic under input ordering changes', () => {
  const { inputs, pairs } = scene({ count: 30, views: 3, noise: 0.1 });
  const options = { initialPair: [10, 30] as const, ransacIterations: 100 };
  const a = reconstructIncremental(inputs, pairs, options),
    b = reconstructIncremental([...inputs].reverse(), [...pairs].reverse(), options);
  assert.deepEqual(a.reconstruction, b.reconstruction);
  assert.deepEqual(a.diagnostics, b.diagnostics);
});

test('noisy planar and rotation-only inputs expose initialization diagnostics', () => {
  for (const rotationOnly of [false, true]) {
    const { inputs, points, rng } = scene({ count: 60, views: 3 });
    const views = inputs.map((v, i) => ({
      ...v,
      keypoints: Float64Array.from(
        points.flatMap((p) => {
          const X = rotationOnly ? p : Float64Array.of(p[0]!, p[1]!, 6);
          const pixel = projectPoint(
            v.camera,
            {
              R: rotationExp([0, 0.03 * i, 0]),
              t: Float64Array.of(rotationOnly ? 0 : 0.3 * i, 0, 0),
            },
            X,
          )!;
          return [pixel[0] + (rng() - 0.5) * 0.3, pixel[1] + (rng() - 0.5) * 0.3];
        }),
      ),
    }));
    const pair = {
      first: 10,
      second: 30,
      matches: Uint32Array.from(points.flatMap((_, i) => [i, i])),
    };
    assert.throws(
      () => reconstructIncremental(views, [pair], { ransacIterations: 150 }),
      (error) => {
        assert.ok(error instanceof Error);
        assert.equal(error.name, 'ReconstructionInitializationError');
        return true;
      },
    );
  }
});

test('landmarks persist across registrations: each track is triangulated once and continued', () => {
  const { inputs, pairs, reconstruction } = scene({ count: 90, views: 6, noise: 0.3 }),
    machine = incrementalSteps(inputs, pairs, { seed: 3 });
  let step = machine.next(),
    triangulated = 0,
    continued = 0;
  while (!step.done) {
    const job = step.value;
    if (job.kind === 'triangulation') triangulated += job.observations.length;
    if (job.kind === 'observations' && job.stage === 'continuation')
      continued += job.model.landmarks.length;
    step = machine.next(executeReconstructionJob(job));
  }
  const result = step.value;
  assert.equal(result.reconstruction.views.length, 6);
  // Every track triangulates once; the old mapper re-triangulated all 90 tracks at each of
  // the four later registrations (450 groups).
  assert.ok(triangulated <= 100, `triangulated ${triangulated} groups`);
  // Each later view continues the 90 tracks it observes.
  assert.equal(continued, 4 * 90);
  assert.ok(result.reconstruction.landmarks.every((p) => p.observations.length === 6));
  assert.equal(result.reconstruction.landmarks.length, reconstruction.landmarks.length);
});
