import assert from 'node:assert/strict';
import { affine, gridPoints, renderScene } from '../klt-cases.ts';

/** Small image experiments, not reconstruction inputs or a hardware performance benchmark. */
export function workFixture() {
  const width = 192,
    height = 144,
    first = Array.from(renderScene({ width, height })),
    grid = Array.from(gridPoints(width, height, 30, 25, 12, 6)),
    interior = Array.from(gridPoints(width, height, 40, 35, 4, 3)),
    warp = { degrees: 7, scale: 1.12, gain: 1.1, offset: 0.02 },
    theta = (warp.degrees * Math.PI) / 180,
    matrix = [
      warp.scale * Math.cos(theta),
      -warp.scale * Math.sin(theta),
      warp.scale * Math.sin(theta),
      warp.scale * Math.cos(theta),
    ],
    truth = Array.from({ length: interior.length / 2 }, (_, k) =>
      affine(width, height, warp).forward(interior[2 * k], interior[2 * k + 1]),
    ).flat(),
    translated = Array.from(renderScene({ width, height }, { dx: 2.7, dy: -1.3 }));
  return {
    width,
    height,
    first,
    cases: [
      {
        name: 'one-track-converges-at-first-iteration',
        points: interior.slice(0, 2),
        second: first,
        options: { levels: 3, halfWindow: 5, maxIterations: 80, photometric: 2 },
        gpu: { chunkTracks: 1 },
      },
      {
        name: '67-tracks-63-and-4-tail-chunks',
        points: grid.slice(0, 134),
        second: Array.from(renderScene({ width, height }, { dx: 0.4, dy: -0.2 })),
        options: { levels: 2, halfWindow: 4, maxIterations: 20, photometric: 0 },
        gpu: { chunkTracks: 63 },
      },
      {
        name: 'affine-initial-targets-and-5-track-chunks',
        points: interior,
        second: Array.from(renderScene({ width, height }, warp)),
        options: { levels: 2, halfWindow: 5, maxIterations: 40, photometric: 2 },
        gpu: {
          chunkTracks: 5,
          initialTargets: truth.map((v, i) => v + (i % 2 ? -0.25 : 0.45)),
          affineWarps: Array.from({ length: interior.length * 2 }, (_, i) => matrix[i % 4]),
        },
      },
      {
        name: 'one-iteration-limit-is-not-convergence',
        points: interior,
        second: translated,
        options: {
          levels: 1,
          halfWindow: 5,
          maxIterations: 1,
          epsilonPx: 1e-6,
          photometric: 1,
        },
        gpu: { chunkTracks: 5 },
      },
      {
        name: 'flat-target-and-outside-forward-seeds-end-before-backward',
        points: [...interior.slice(0, 8), 1.25, 1.75, width - 1.25, height - 1.75],
        second: Array(width * height).fill(0.5),
        options: { levels: 2, halfWindow: 5, maxIterations: 12, photometric: 2 },
        gpu: { chunkTracks: 5 },
      },
      {
        name: 'zero-structure-template-preserves-end-reason',
        points: interior.slice(0, 6),
        first: Array(width * height).fill(0.5),
        second: translated,
        options: { levels: 2, halfWindow: 5, maxIterations: 30, photometric: 1 },
        gpu: { chunkTracks: 2 },
      },
    ],
  };
}

export function workReferences(api, input) {
  return input.cases.map((c) => {
    const options = api.kltOptionsFor(input.width, c.options),
      reference = api.trackKlt(
        api.buildKltPyramid(c.first ?? input.first, input.width, input.height, options.levels),
        api.buildKltPyramid(c.second, input.width, input.height, options.levels),
        c.points,
        options,
        c.gpu.initialTargets,
        c.gpu.affineWarps,
      );
    return Object.fromEntries(
      ['records', 'forward', 'back'].map((key) => [key, Array.from(reference[key])]),
    );
  });
}

export function assertWork(result, input, references, tolerance = 0) {
  assert.equal(result.liveBytes, 0);
  assert.equal(result.rows.length, input.cases.length);
  for (let j = 0; j < input.cases.length; j++) {
    const c = input.cases[j],
      actual = result.rows[j],
      expected = references[j];
    for (const key of ['records', 'forward', 'back']) {
      assert.equal(actual[key].length, expected[key].length, `${c.name}: ${key} length`);
      for (let i = 0; i < expected[key].length; i++) {
        const a = actual[key][i],
          b = expected[key][i];
        if (tolerance === 0 || i % 4 === 3) assert.equal(a, b, `${c.name}: ${key}[${i}]`);
        else
          assert.ok(
            Math.abs(a - b) <= tolerance,
            `${c.name}: ${key}[${i}] ${a} versus exhaustive f32 ${b}`,
          );
      }
    }
  }
  const first = result.rows[0];
  assert.equal(first.forward[3] & 1, 1);
  assert.equal(first.forward[3] >>> 16, 1, 'a converging first update still counts once');
  assert.ok(
    result.rows[3].forward.some((v, i) => i % 4 === 3 && (v & 1) === 0),
    'the one-iteration fixture must reach a real failed convergence case',
  );
  for (const j of [3, 4, 5]) {
    const row = result.rows[j];
    let dead = 0;
    for (let i = 0; i < row.forward.length; i += 4)
      if ((row.forward[i + 3] & 1) === 0) {
        dead++;
        assert.deepEqual(
          row.back.slice(i, i + 4),
          [row.forward[i], row.forward[i + 1], 0, row.forward[i + 3]],
          'a dead backward seed preserves its coordinates and reason, with zero NCC',
        );
      }
    assert.ok(dead > 0, `${input.cases[j].name}: exercise ended backward seed copy-through`);
  }
}
