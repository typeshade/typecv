import test from 'node:test';
import assert from 'node:assert/strict';
import {
  corroborateObservations,
  type ConsensusFrame,
  type ObservationConsensusCandidate,
} from '../src/gpu/observation-consensus.ts';

function frame(id: number, extra: Partial<ConsensusFrame> = {}): ConsensusFrame {
  return {
    id,
    width: 64,
    height: 48,
    keypoints: Float64Array.of(15, 16, 30, 25, 40, 30),
    shotId: 'A',
    ...extra,
  };
}
function candidate(
  id: number,
  group: number | undefined = 0,
  extra: Partial<ObservationConsensusCandidate> = {},
): ObservationConsensusCandidate {
  return {
    id,
    group,
    sourceViewId: 10,
    sourceFeatureId: 0,
    targetViewId: 20,
    measuredPixel: [18.25, 16.5],
    ...extra,
  };
}
const measurement = (x: number, y: number) => ({
  pixel: [x, y] as const,
  ncc: 0.98,
  forwardBackwardErrorPx: 0.08,
});

test('consensus batches original anchor pairs and preserves candidate order and measured initial values', async () => {
  const frames = [frame(10), frame(20), frame(99)],
    groups = [
      new Map([
        [10, 0],
        [99, 1],
      ]),
      new Map([
        [10, 1],
        [99, 2],
      ]),
    ],
    calls: unknown[] = [],
    result = await corroborateObservations(
      frames,
      groups,
      [candidate(7), candidate(3, 1, { sourceFeatureId: 1, measuredPixel: [30.5, 25.25] })],
      async (anchor, target, seeds, initials, featureIds) => {
        calls.push([anchor.id, target.id, [...seeds], [...initials], featureIds]);
        return [measurement(18.35, 16.5), measurement(30.5, 25.15)];
      },
    );
  assert.deepEqual(calls, [[99, 20, [30, 25, 40, 30], [18.25, 16.5, 30.5, 25.25], [1, 2]]]);
  assert.deepEqual(
    result.map((row) => [row.id, row.status, row.anchorFeatureId]),
    [
      [7, 'accepted', 1],
      [3, 'accepted', 2],
    ],
  );
  assert.ok(result.every((row) => row.disagreementPx! < 0.101));
  assert.deepEqual(frames[0]!.keypoints, Float64Array.of(15, 16, 30, 25, 40, 30));
});

test('consensus prefers a nearby target patch and breaks playback-distance ties by source separation', async () => {
  const frames = [frame(700), frame(10), frame(20), frame(1), frame(5)],
    groups = [
      new Map([
        [1, 1],
        [700, 1],
        [5, 2],
        [10, 0],
        [20, 0],
      ]),
    ],
    calls: number[] = [];
  const result = await corroborateObservations(frames, groups, [candidate(1)], async (anchor) => {
    calls.push(anchor.id);
    return [measurement(18.25, 16.5)];
  });
  assert.deepEqual(calls, [1]);
  assert.equal(result[0]!.anchorFeatureId, 1);
  assert.equal(result[0]!.status, 'accepted');
  // Equidistant target neighbours remain independent of Map insertion order or numeric ids.
  groups[0]!.delete(1);
  const tied = await corroborateObservations(frames, groups, [candidate(2)], async (anchor) => {
    calls.push(anchor.id);
    return [measurement(18.25, 16.5)];
  });
  assert.equal(tied[0]!.anchorViewId, 5);
  assert.deepEqual(calls, [1, 5]);
});

test('finite patch-comparability scores select a witness per candidate without accepting its coordinates', async () => {
  const frames = [frame(700), frame(10), frame(20), frame(1), frame(5)],
    groups = [
      new Map([
        [10, 0],
        [700, 1],
        [1, 1],
        [5, 2],
      ]),
    ],
    scores: unknown[] = [],
    calls: number[] = [],
    result = await corroborateObservations(
      frames,
      groups,
      [candidate(1), candidate(2)],
      async (anchor) => {
        calls.push(anchor.id);
        // The best geometric proposal still fails if its actual image measurement disagrees.
        return [anchor.id === 5 ? measurement(19.25, 16.5) : measurement(18.25, 16.5)];
      },
      {},
      (source, target, anchor, featureId, primary) => {
        scores.push([source.id, target.id, anchor.id, featureId, primary.id]);
        if (primary.id === 1) return anchor.id === 5 ? 0.1 : 0.5;
        return anchor.id === 700 ? -0.2 : undefined;
      },
    );
  assert.deepEqual(calls, [5, 700]);
  assert.deepEqual(
    result.map((row) => [row.anchorViewId, row.status]),
    [
      [5, 'disagreement'],
      [700, 'accepted'],
    ],
  );
  assert.equal(scores.length, 6);
  assert.deepEqual(scores[0], [10, 20, 700, 1, 1]);
});

test('unavailable and nonfinite anchor priorities use deterministic nearby-target fallback', async () => {
  const frames = [frame(700), frame(10), frame(20), frame(1), frame(5)],
    groups = [
      new Map([
        [5, 2],
        [700, 1],
        [1, 1],
        [10, 0],
      ]),
    ];
  for (const score of [undefined, NaN, Infinity, -Infinity]) {
    const calls: number[] = [],
      result = await corroborateObservations(
        frames,
        groups,
        [candidate(1)],
        async (anchor) => {
          calls.push(anchor.id);
          return [measurement(18.25, 16.5)];
        },
        {},
        () => score,
      );
    assert.deepEqual(calls, [1]);
    assert.equal(result[0]!.status, 'accepted');
  }
});

test('consensus distinguishes disagreement and failed independent measurement', async () => {
  const frames = [frame(10), frame(20), frame(30)],
    groups = [0, 1, 2].map(
      (i) =>
        new Map([
          [10, i],
          [30, i],
        ]),
    ),
    result = await corroborateObservations(
      frames,
      groups,
      [0, 1, 2].map((i) => candidate(i, i, { sourceFeatureId: i })),
      async () => [measurement(19, 16.5), null, measurement(NaN, 16.5)],
    );
  assert.deepEqual(
    result.map((row) => row.status),
    ['disagreement', 'measurementFailure', 'measurementFailure'],
  );
  assert.equal(result[0]!.disagreementPx, 0.75);
  assert.equal(result[0]!.measurement!.ncc, 0.98);
  assert.ok(result.slice(1).every((row) => row.measurement === undefined));
});

test('two-view and untrusted groups are unavailable without invoking measurement', async () => {
  let calls = 0;
  const result = await corroborateObservations(
    [frame(10), frame(20), frame(30)],
    [
      new Map([
        [10, 0],
        [20, 0],
      ]),
    ],
    [candidate(0), candidate(1, 0, { group: undefined }), candidate(2, 9)],
    async () => {
      calls++;
      return [];
    },
  );
  assert.equal(calls, 0);
  assert.deepEqual(
    result.map((row) => row.status),
    ['unavailable', 'unavailable', 'unavailable'],
  );
});

test('a witness cannot cross an intervening cut or change of image coordinates', async () => {
  const groups = [
    new Map([
      [10, 0],
      [40, 0],
    ]),
  ];
  for (const middle of [frame(30, { shotId: 'B' }), frame(30, { width: 65 })]) {
    let calls = 0;
    const result = await corroborateObservations(
      [frame(10), frame(20), middle, frame(40)],
      groups,
      [candidate(0)],
      async () => {
        calls++;
        return [];
      },
    );
    assert.equal(calls, 0);
    assert.equal(result[0]!.status, 'unavailable');
  }
});

test('witness selection checks original indices, image borders and full source masks', async () => {
  const mask = new Uint8Array(64 * 48).fill(1);
  mask[16 * 64 + 14] = 0;
  const frames = [frame(10), frame(20), frame(30, { validMask: mask })],
    groups = [
      new Map([
        [10, 0],
        [30, 0],
      ]),
      new Map([
        [10, 0],
        [30, 3],
      ]),
    ];
  let calls = 0;
  const result = await corroborateObservations(
    frames,
    groups,
    [candidate(0), candidate(1, 1), candidate(2, 0, { measuredPixel: [1, 1] })],
    async () => {
      calls++;
      return [];
    },
    { halfWindow: 2 },
  );
  assert.equal(calls, 0);
  assert.ok(result.every((row) => row.status === 'unavailable'));
});

test('new descendants and asynchronous ownership mutations cannot become original witnesses', async () => {
  const frames = [frame(10), frame(20), frame(30), frame(40)],
    groups = [
      new Map([
        [10, 0],
        [30, 0],
      ]),
      new Map([
        [10, 1],
        [20, 1],
      ]),
    ],
    calls: number[] = [],
    result = await corroborateObservations(
      frames,
      groups,
      [candidate(0), candidate(1, 1, { sourceFeatureId: 1 })],
      async (anchor) => {
        calls.push(anchor.id);
        // Simulate a caller's primary pass adding an observation while a batch is in flight.
        groups[1]!.set(40, 1);
        return [measurement(18.25, 16.5)];
      },
    );
  assert.deepEqual(calls, [30]);
  assert.deepEqual(
    result.map((row) => row.status),
    ['accepted', 'unavailable'],
  );
});

test('consensus bounds callback batches and rejects invalid callback cardinality', async () => {
  const frames = [frame(10), frame(20), frame(30)],
    groups = [
      new Map([
        [10, 0],
        [30, 0],
      ]),
    ],
    candidates = Array.from({ length: 5 }, (_, i) => candidate(i)),
    counts: number[] = [],
    result = await corroborateObservations(
      frames,
      groups,
      candidates,
      async (_anchor, _target, seeds, initials) => {
        counts.push(seeds.length / 2);
        return Array.from({ length: initials.length / 2 }, (_, i) =>
          measurement(initials[2 * i]!, initials[2 * i + 1]!),
        );
      },
      { batchSize: 2 },
    );
  assert.deepEqual(counts, [2, 2, 1]);
  assert.ok(result.every((row) => row.status === 'accepted'));
  await assert.rejects(
    corroborateObservations(frames, groups, [candidate(0)], async () => []),
    /different measurement count/,
  );
  await assert.rejects(
    corroborateObservations(frames, groups, [candidate(0)], async () => [], {
      maxDisagreementPx: 0,
    }),
    /Invalid observation consensus bounds/,
  );
});
