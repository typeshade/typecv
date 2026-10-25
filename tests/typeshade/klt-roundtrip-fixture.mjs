import assert from 'node:assert/strict';
import { affine, renderScene } from '../klt-cases.ts';
import { workReferences } from './klt-work-fixture.mjs';

/** Image measurements for grouped roundtrips, with an unchanged f32 oracle. */
export function roundtripFixture() {
  const width = 192,
    height = 144,
    first = Array.from(renderScene({ width, height })),
    points = [65.37, 50.29, 111.63, 84.71],
    warp = { degrees: 8, scale: 1.1, gain: 1.12, offset: -0.03 },
    angle = (warp.degrees * Math.PI) / 180,
    matrix = [
      warp.scale * Math.cos(angle),
      -warp.scale * Math.sin(angle),
      warp.scale * Math.sin(angle),
      warp.scale * Math.cos(angle),
    ],
    truth = Array.from({ length: points.length / 2 }, (_, i) =>
      affine(width, height, warp).forward(points[2 * i], points[2 * i + 1]),
    ).flat(),
    oddWidth = 97,
    oddHeight = 73,
    oddFirst = Array.from(renderScene({ width: oddWidth, height: oddHeight }));
  return {
    groups: [
      {
        width: oddWidth,
        height: oddHeight,
        first: oddFirst,
        cases: [
          {
            name: 'odd-image-fractional-seeds-and-coarse-border-clamps',
            points: [11.37, 12.29, 84.63, 60.71, 45.17, 35.41],
            second: Array.from(
              renderScene({ width: oddWidth, height: oddHeight }, { dx: 0.55, dy: -0.4 }),
            ),
            options: { levels: 3, halfWindow: 5, maxIterations: 25, photometric: 2 },
            gpu: { chunkTracks: 2 },
          },
        ],
      },
      {
        width,
        height,
        first,
        cases: [
          {
            name: 'minimum-window-plain-ssd',
            points,
            second: first,
            options: {
              levels: 1,
              halfWindow: 1,
              maxIterations: 2,
              minEigThreshold: 1e-10,
              photometric: 0,
            },
            gpu: { chunkTracks: 1 },
          },
          {
            name: 'window-10-multiple-levels-and-mean-removal',
            points,
            second: first,
            options: { levels: 2, halfWindow: 10, maxIterations: 2, photometric: 1 },
            gpu: { chunkTracks: 1 },
          },
          {
            name: 'window-11-gain-normalization',
            points,
            second: Array.from(renderScene({ width, height }, { gain: 1.08, offset: -0.03 })),
            options: { levels: 1, halfWindow: 11, maxIterations: 2, photometric: 2 },
            gpu: { chunkTracks: 1 },
          },
          {
            name: 'window-31-finite-shift',
            points: [95.37, 70.29],
            second: Array.from(renderScene({ width, height }, { dx: 0.35, dy: -0.25 })),
            options: { levels: 1, halfWindow: 31, maxIterations: 2, photometric: 2 },
            gpu: { chunkTracks: 1 },
          },
          {
            name: 'affine-offset-initializer-and-photometric-gain',
            points,
            second: Array.from(renderScene({ width, height }, warp)),
            options: { levels: 2, halfWindow: 5, maxIterations: 30, photometric: 2 },
            gpu: {
              chunkTracks: 1,
              initialTargets: truth.map((v, i) => v + (i % 2 ? -0.3 : 0.4)),
              affineWarps: Array.from({ length: points.length * 2 }, (_, i) => matrix[i % 4]),
            },
          },
        ],
      },
    ],
  };
}

export function roundtripReferences(api, input) {
  return input.groups.flatMap((group) => workReferences(api, group));
}

export function assertRoundtrips(result, input, references, tolerance = 0) {
  assert.equal(result.liveBytes, 0);
  const cases = input.groups.flatMap((group) => group.cases);
  assert.equal(result.rows.length, cases.length);
  assert.deepEqual(
    result.defaultRow,
    result.separateRows[0],
    'omitting batchPasses preserves explicit separate-pass outputs, entry counts and plan calls',
  );
  for (let j = 0; j < cases.length; j++) {
    const actual = result.rows[j],
      expected = references[j],
      name = cases[j].name;
    assert.equal(actual.name, name);
    assert.equal(result.separateRows[j].name, name);
    assert.ok(
      actual.planCalls < result.separateRows[j].planCalls,
      `${name}: actual device plans ${actual.planCalls} versus ${result.separateRows[j].planCalls}`,
    );
    assert.equal(actual.pairEntries, 0, name + ': no replacement tracking kernel');
    assert.equal(result.separateRows[j].pairEntries, 0, name + ': original tracking kernels');
    assert.ok(actual.directionalEntries > 0, name + ': grouped original directions ran');
    assert.ok(actual.checkEntries > 0, name + ': grouped original check ran');
    assert.ok(result.separateRows[j].directionalEntries > 0, name + ': separate directions ran');
    assert.ok(result.separateRows[j].checkEntries > 0, name + ': separate check ran');
    assert.equal(actual.directionalEntries, result.separateRows[j].directionalEntries);
    assert.equal(actual.checkEntries, result.separateRows[j].checkEntries);
    assert.equal(actual.entryDispatches, result.separateRows[j].entryDispatches);
    for (const key of ['forward', 'back', 'records']) {
      assert.deepEqual(
        actual[key],
        result.separateRows[j][key],
        `${name}: grouped roundtrips must preserve every output against separate passes`,
      );
      assert.equal(actual[key].length, expected[key].length, name + ' ' + key);
      for (let i = 0; i < expected[key].length; i++)
        if (tolerance === 0 || i % 4 === 3)
          assert.equal(actual[key][i], expected[key][i], `${name}: ${key}[${i}]`);
        else
          assert.ok(
            Math.abs(actual[key][i] - expected[key][i]) <= tolerance,
            `${name}: ${key}[${i}] ${actual[key][i]} versus f32 ${expected[key][i]}`,
          );
    }
  }
  assert.equal(result.tight.liveBytes, 0, 'both schedules release their arrays');
  assert.ok(result.tight.peakActiveBytes <= result.tight.workingBytes);
  assert.ok(result.tight.grouped.planCalls < result.tight.separate.planCalls);
  assert.equal(result.tight.grouped.pairEntries, 0);
  assert.ok(result.tight.grouped.directionalEntries > 0);
  assert.ok(result.tight.grouped.checkEntries > 0);
  assert.equal(result.tight.separate.pairEntries, 0);
  assert.ok(result.tight.separate.directionalEntries > 0);
  assert.ok(result.tight.separate.checkEntries > 0);
  assert.equal(result.tight.grouped.directionalEntries, result.tight.separate.directionalEntries);
  assert.equal(result.tight.grouped.checkEntries, result.tight.separate.checkEntries);
  assert.equal(result.tight.grouped.entryDispatches, result.tight.separate.entryDispatches);
  for (const row of [result.tight.grouped, result.tight.separate]) {
    assert.equal(row.name, cases[0].name);
    for (const key of ['forward', 'back', 'records'])
      assert.deepEqual(
        row[key],
        result.separateRows[0][key],
        'both schedules work at the exact original memory budget: ' + key,
      );
  }
  for (const name of [
    'odd-image-fractional-seeds-and-coarse-border-clamps',
    'window-10-multiple-levels-and-mean-removal',
    'window-11-gain-normalization',
    'window-31-finite-shift',
  ]) {
    const row = result.rows.find((candidate) => candidate.name === name);
    assert.ok(
      row.records.some((v, i) => i % 4 === 3 && (v & 1) === 1),
      name + ': compare real accepted tracking, not only failed/empty records',
    );
  }
}
