import test from 'node:test';
import assert from 'node:assert/strict';
import { planMvsViews } from '../src/gpu/mvs-views.ts';
import {
  formatRow,
  formatSummary,
  keyframeRanks,
  parseOptions,
  summarizeSources,
  type SourceScene,
  type SourceSceneView,
} from '../scripts/measure-mvs-sources.mjs';

const FX = 100,
  CX = 63.5,
  CY = 47.5;
type Vec3 = [number, number, number];

/** A camera at (x, 0, 0) that looks down +z: R = I, t = −C. */
function view(id: number, x: number, time: number): SourceSceneView {
  return {
    id,
    time,
    width: 128,
    height: 96,
    camera: { fx: FX, fy: FX, cx: CX, cy: CY },
    pose: { R: [1, 0, 0, 0, 1, 0, 0, 0, 1], t: [-x, 0, 0] },
  };
}

/**
 * Views 10, 20 and 30 at x = 0, 1 and 3, in the input order 30, 10, 20 but at the times 1, 2 and
 * 3 s, observe a 7 × 3 grid of landmarks at z = 10. View 40 at x = 6 observes 5 other landmarks
 * only: too few for a depth map (3.3), and too few shared to be a source (3.2).
 */
function lineScene(): SourceScene {
  const views = [view(30, 3, 3), view(10, 0, 1), view(20, 1, 2), view(40, 6, 4)];
  const observe = (xyz: Vec3, ids: number[]) => ({
    xyz,
    observations: ids.map((id) => {
      const x = views.find((v) => v.id === id)!.pose.t[0]!;
      return {
        viewId: id,
        pixel: [(FX * (xyz[0] + x)) / xyz[2] + CX, (FX * xyz[1]) / xyz[2] + CY] as const,
      };
    }),
  });
  const landmarks = [];
  for (let i = 0; i <= 6; i++)
    for (const y of [-1, 0, 1]) landmarks.push(observe([0.5 * i, y, 10], [10, 20, 30]));
  for (const y of [-1, -0.5, 0, 0.5, 1]) landmarks.push(observe([6, y, 10], [40]));
  return { views, landmarks };
}

/** The upper median of the triangulation angles of the grid between the cameras at xa and xb. */
function gridMedianAngle(xa: number, xb: number): number {
  const angles: number[] = [];
  for (let i = 0; i <= 6; i++)
    for (const y of [-1, 0, 1]) {
      const u = [xa - 0.5 * i, -y, -10],
        v = [xb - 0.5 * i, -y, -10];
      const cos =
        (u[0]! * v[0]! + u[1]! * v[1]! + u[2]! * v[2]!) / (Math.hypot(...u) * Math.hypot(...v));
      angles.push((Math.acos(cos) * 180) / Math.PI);
    }
  return angles.sort((a, b) => a - b)[angles.length >> 1]!;
}

test('each source link: plan order, gap in time order, median angle and baseline', () => {
  const scene = lineScene();
  // α0 = 20°: every grid angle is below it, so a longer baseline scores strictly higher.
  const plan = planMvsViews(scene, { fullAngleDeg: 20 });
  const { rows, summary } = summarizeSources(scene, plan);
  assert.deepEqual(
    rows.map((row) => [row.id, row.status, row.sources.map((s) => s.id)]),
    [
      [30, 'ready', [10, 20]],
      [10, 'ready', [30, 20]],
      [20, 'ready', [30, 10]],
      [40, 'fewLandmarks', []],
    ],
  );
  const link = (r: number, s: number) =>
    rows.find((row) => row.id === r)!.sources.find((source) => source.id === s)!;
  // Time order is 10, 20, 30, 40. In the input order, 10 and 30 would be neighbours.
  assert.equal(link(10, 30).keyframeGap, 2);
  assert.equal(link(10, 20).keyframeGap, 1);
  assert.equal(link(20, 30).keyframeGap, 1);
  assert.equal(link(30, 10).timeGap, 2);
  assert.equal(link(20, 10).timeGap, 1);
  const x = new Map([
    [10, 0],
    [20, 1],
    [30, 3],
  ]);
  for (const row of rows.slice(0, 3)) {
    assert.equal(row.medianDepth, 10);
    for (const source of row.sources) {
      const xa = x.get(row.id)!,
        xb = x.get(source.id)!;
      assert.equal(source.sharedLandmarks, 21);
      assert.ok(Math.abs(source.medianAngleDeg - gridMedianAngle(xa, xb)) < 1e-9);
      assert.ok(Math.abs(source.baselineOverDepth - Math.abs(xa - xb) / 10) < 1e-12);
    }
  }
  assert.deepEqual(rows[1]!.depthRange, plan.views[1]!.depthRange);
  assert.equal(rows[3]!.landmarks, 5);
  // The six link medians: two near 5.5° (baseline 1), two near 11° (2), two near 16° (3).
  const medians = rows.flatMap((row) => row.sources.map((s) => s.medianAngleDeg));
  assert.deepEqual(
    { ...summary, medianAngleDeg: 0 },
    {
      views: 4,
      references: 3,
      links: 6,
      medianAngleDeg: 0,
      medianKeyframeGap: 1,
      medianTimeGap: 1,
      maxKeyframeGap: 2,
      shareBelow5Deg: 0,
      shareBelow10Deg: 2 / 6,
    },
  );
  assert.equal(summary.medianAngleDeg, medians.sort((a, b) => a - b)[3]);
  assert.ok(summary.medianAngleDeg > 10 && summary.medianAngleDeg < 12);
});

test('the flags pass through to planMvsViews', () => {
  assert.deepEqual(
    parseOptions([
      '--model',
      'm.json',
      '--max-sources',
      '1',
      '--full-angle',
      '20',
      '--min-angle',
      '2',
      '--min-shared',
      '21',
      '--out',
      'o.json',
    ]),
    {
      model: 'm.json',
      out: 'o.json',
      viewOptions: { maxSources: 1, fullAngleDeg: 20, minAngleDeg: 2, minSharedLandmarks: 21 },
    },
  );
  assert.deepEqual(parseOptions(['--model', 'm.json']), {
    model: 'm.json',
    out: undefined,
    viewOptions: {},
  });
  assert.throws(() => parseOptions(['--model', 'm.json', '--sources', '3']));
  const scene = lineScene();
  const { viewOptions } = parseOptions(['--max-sources', '1', '--full-angle', '20']);
  const { rows } = summarizeSources(scene, planMvsViews(scene, viewOptions));
  assert.deepEqual(
    rows.map((row) => row.sources.map((s) => s.id)),
    [[10], [30], [30], []],
  );
  // 22 shared landmarks are more than the grid has: no pair scores, and no view gets a depth map.
  const none = summarizeSources(scene, planMvsViews(scene, { minSharedLandmarks: 22 }));
  assert.deepEqual(
    none.rows.map((row) => row.status),
    ['noSource', 'noSource', 'noSource', 'fewLandmarks'],
  );
  assert.equal(none.summary.links, 0);
  assert.ok(Number.isNaN(none.summary.medianAngleDeg) && Number.isNaN(none.summary.maxKeyframeGap));
});

test('time order: equal times keep the input order, and views without a time come last', () => {
  assert.deepEqual(
    [...keyframeRanks([{ id: 5 }, { id: 3, time: 1 }, { id: 4 }, { id: 9, time: 1 }])],
    [
      [3, 0],
      [9, 1],
      [5, 2],
      [4, 3],
    ],
  );
});

test('the printed lines', () => {
  assert.equal(
    formatRow({
      id: 7,
      time: 2.2,
      status: 'ready',
      depthRange: [0.1234, 5.6789],
      landmarks: 9,
      medianDepth: 3,
      sources: [
        {
          id: 9,
          keyframeGap: 2,
          timeGap: 1.04,
          sharedLandmarks: 20,
          medianAngleDeg: 4.56,
          baselineOverDepth: 0.0812,
        },
        {
          id: 11,
          keyframeGap: 4,
          timeGap: NaN,
          sharedLandmarks: 16,
          medianAngleDeg: 12,
          baselineOverDepth: 0.2,
        },
      ],
    }),
    '   7    2.2 s | ready | 0.123-5.679 | 9(2, 1.0 s, 4.6°, 0.081) 11(4, - s, 12.0°, 0.200)',
  );
  assert.equal(
    formatRow({ id: 40, status: 'fewLandmarks', landmarks: 5, medianDepth: 10, sources: [] }),
    '  40      - s | fewLandmarks | - | -',
  );
  assert.deepEqual(
    formatSummary({
      views: 47,
      references: 47,
      links: 282,
      medianAngleDeg: 5.734,
      medianKeyframeGap: 4,
      medianTimeGap: 3,
      maxKeyframeGap: 9,
      shareBelow5Deg: 0.35461,
      shareBelow10Deg: 0.92908,
    }),
    [
      'median over 282 source links: triangulation angle 5.73°, keyframe gap 4, time gap 3.00 s; largest keyframe gap 9',
      'share of source links with a median triangulation angle under 5°: 0.355, under 10°: 0.929',
    ],
  );
});
