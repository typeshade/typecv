import { renderScene, gridPoints, affine } from '../klt-cases.ts';
import assert from 'node:assert/strict';
export function patchFixture() {
  const width = 192,
    height = 144,
    points = gridPoints(width, height, 40, 35, 4, 3);
  return {
    width,
    height,
    points: Array.from(points),
    first: Array.from(renderScene({ width, height })),
    cases: [{ degrees: 12 }, { degrees: 25 }, { scale: 1.2 }, { degrees: 12, scale: 1.15 }].map(
      (warp) => {
        const theta = ((warp.degrees ?? 0) * Math.PI) / 180,
          s = warp.scale ?? 1,
          A = [s * Math.cos(theta), -s * Math.sin(theta), s * Math.sin(theta), s * Math.cos(theta)],
          truth = Array.from({ length: points.length / 2 }, (_, i) =>
            affine(width, height, warp).forward(points[2 * i], points[2 * i + 1]),
          ).flat();
        return {
          warp,
          truth,
          guesses: truth.map((v, i) => v + (i % 2 ? -0.5 : 0.7)),
          affine: Array.from({ length: points.length * 2 }, (_, i) => A[i % 4]),
          second: Array.from(renderScene({ width, height }, warp)),
        };
      },
    ),
  };
}
export function assertPatches(result, input) {
  assert.equal(result.liveBytes, 0);
  for (let j = 0; j < input.cases.length; j++) {
    const row = result.cases[j],
      c = input.cases[j],
      errors = [];
    for (let k = 0; k < input.points.length / 2; k++) {
      assert.ok(row.records[4 * k + 3] & 1, `case${j} seed${k}`);
      const e = Math.hypot(
        row.records[4 * k] - c.truth[2 * k],
        row.records[4 * k + 1] - c.truth[2 * k + 1],
      );
      errors.push(e);
      assert.ok(e < 0.05, `case${j} error${e}`);
    }
    assert.ok(
      Math.max(...errors) < row.plainMax * 0.15,
      `warp must improve measured error: ${JSON.stringify(row)}`,
    );
  }
}
