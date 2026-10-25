import assert from 'node:assert/strict';
import { gridPoints, renderScene } from '../klt-cases.ts';

export function expansionFixture() {
  const width = 192,
    height = 144,
    camera = { fx: 180, fy: 180, cx: 95.5, cy: 71.5 },
    points = Array.from(gridPoints(width, height, 40, 35, 4, 3)),
    frame = (id, seed, image = {}) => ({
      id,
      width,
      height,
      camera,
      gray: Array.from(renderScene({ width, height, ...image }, { dx: id * 2, dy: -id })),
      keypoints: seed.flatMap((k) => [points[2 * k] + id * 2, points[2 * k + 1] - id]),
    });
  return {
    frames: [frame(0, [0, 1, 2, 3, 4, 5]), frame(1, []), frame(2, [6, 7, 8, 9, 10, 11])],
    repeated: [
      frame(0, [0, 1, 2, 3], { gaussians: 0, checkerPeriod: 8, checkerAmplitude: 0.2 }),
      frame(1, [], { gaussians: 0, checkerPeriod: 8, checkerAmplitude: 0.2 }),
    ],
    points,
  };
}
export function assertExpansion(result, input) {
  assert.equal(result.liveBytes, 0);
  assert.deepEqual(
    result.original,
    input.frames.map((f) => f.keypoints),
    'inputs were not mutated',
  );
  assert.ok(
    result.expanded.diagnostics.newCoordinates >= 15,
    JSON.stringify(result.expanded.diagnostics),
  );
  assert.ok(
    result.expanded.counts.every((n) => n >= 8),
    JSON.stringify(result.expanded.counts),
  );
  const old = new Map(input.frames.map((f) => [f.id, f.keypoints.length / 2]));
  assert.ok(
    result.expanded.observations.some((o) => o.sourceViewId > o.viewId),
    'backwards from late seeds',
  );
  assert.ok(
    result.expanded.observations.some((o) => o.sourceViewId < o.viewId),
    'forwards from early seeds',
  );
  for (const frame of result.expanded.images) {
    assert.deepEqual(
      frame.keypoints.slice(0, 2 * old.get(frame.id)),
      input.frames[frame.id].keypoints,
    );
    for (let i = 0; i < frame.keypoints.length; i += 2) {
      const x = frame.keypoints[i] - frame.id * 2,
        y = frame.keypoints[i + 1] + frame.id;
      const nearest = Math.min(
        ...Array.from({ length: input.points.length / 2 }, (_, k) =>
          Math.hypot(x - input.points[2 * k], y - input.points[2 * k + 1]),
        ),
      );
      assert.ok(nearest < 0.1, `image-measured coordinate error ${nearest}`);
      for (let j = 0; j < i; j += 2)
        assert.ok(
          Math.hypot(
            frame.keypoints[i] - frame.keypoints[j],
            frame.keypoints[i + 1] - frame.keypoints[j + 1],
          ) > 0.75,
        );
    }
  }
  for (const o of result.expanded.observations) {
    assert.equal(o.geometry, 'candidate');
    assert.ok(o.featureId >= old.get(o.viewId));
    assert.ok(o.ncc >= 0.8 && o.forwardBackwardErrorPx <= 0.5 && o.uniquenessMargin >= 0.03);
  }
  for (const name of ['flat', 'masked', 'cut', 'ambiguous'])
    assert.equal(result[name].newCoordinates, 0, name);
  assert.ok(result.cut.rejected.shotBoundary > 0);
  assert.ok(result.ambiguous.rejected.ambiguousPatch > 0);
}
