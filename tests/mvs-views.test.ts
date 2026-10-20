// Verifies: docs/dense-mvs.md step 1 (sections 3.1 to 3.3, the packing of 4.1).
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { cameraProjection } from '../src/geometry/camera.ts';
import type { Camera } from '../src/geometry/camera.ts';
import { triangulateTrack } from '../src/geometry/multiview.ts';
import { seededRandom } from '../src/geometry/numerics.ts';
import {
  MVS_CAMERA_FLOATS,
  MVS_SOURCE_SLOTS,
  denormalizeMvsPoints,
  mvsAngleWeight,
  mvsCameraCenter,
  mvsDepth,
  mvsDepthRange,
  mvsNormalization,
  mvsObservedDepths,
  mvsViewScores,
  normalizeMvsScene,
  planMvsViews,
} from '../src/gpu/mvs-views.ts';
import type { MvsScene, MvsSceneLandmark, MvsSceneView } from '../src/gpu/mvs-views.ts';

type Vec3 = [number, number, number];
const cross = (a: Vec3, b: Vec3): Vec3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const dot = (a: ArrayLike<number>, b: ArrayLike<number>) =>
  a[0]! * b[0]! + a[1]! * b[1]! + a[2]! * b[2]!;
const unit = (a: Vec3): Vec3 => {
  const n = Math.hypot(...a);
  return [a[0] / n, a[1] / n, a[2] / n];
};
const deg = Math.PI / 180;

const camera: Camera = { fx: 500, fy: 500, cx: 319.5, cy: 239.5, k1: -0.05, k2: 0.01 };
const WIDTH = 640,
  HEIGHT = 480;

/** A camera at C that looks at the origin. Rows of R: right, down, forward, so right × down = forward. */
function lookAt(id: number, C: Vec3): MvsSceneView {
  const f = unit([-C[0], -C[1], -C[2]]);
  const r = unit(cross(f, [0, 1, 0]));
  const d = cross(f, r);
  const R = Float64Array.of(...r, ...d, ...f);
  const t = Float64Array.of(-dot(r, C), -dot(d, C), -dot(f, C));
  return { id, width: WIDTH, height: HEIGHT, camera: { ...camera }, pose: { R, t } };
}

function project(view: MvsSceneView, X: ArrayLike<number>) {
  const { R, t } = view.pose;
  const q = [0, 1, 2].map(
    (i) => R[3 * i]! * X[0]! + R[3 * i + 1]! * X[1]! + R[3 * i + 2]! * X[2]! + t[i]!,
  );
  const p = cameraProjection(view.camera, q)?.pixel;
  return p && p[0] >= 0 && p[0] < WIDTH && p[1] >= 0 && p[1] < HEIGHT ? p : undefined;
}

/**
 * Nine cameras on an arc of radius 8 around a sphere of radius 1.5, at −60° to 60° in steps of
 * 15° (ids 0 to 8, id 4 at 0°). A point is seen from a camera when its outward normal is within
 * 70° of the direction to the camera, so that nearer cameras share more points. Id 9 is a twin
 * of id 4 at 0.5°: every shared point has α < 1°. Id 10 looks at a cluster of 12 points at
 * (0, 0, 40) that no other view sees. Id 11 sits at 0° and observes only 5 landmarks.
 */
function arcScene(): MvsScene {
  const views: MvsSceneView[] = [];
  for (let i = 0; i < 9; i++) {
    const a = (-60 + 15 * i) * deg;
    views.push(lookAt(i, [8 * Math.sin(a), 0.5, 8 * Math.cos(a)]));
  }
  views.push(lookAt(9, [8 * Math.sin(0.5 * deg), 0.5, 8 * Math.cos(0.5 * deg)]));
  // Id 10 at z = 48 looks down −z at the cluster at z = 40: right −x, down +y, forward −z.
  const R = Float64Array.of(-1, 0, 0, 0, 1, 0, 0, 0, -1);
  views.push({ ...lookAt(10, [0, 0, 1]), pose: { R, t: Float64Array.of(0, 0, 48) } });
  views.push(lookAt(11, [0, 0.5, 8]));
  const rng = seededRandom(17);
  const landmarks: MvsSceneLandmark[] = [];
  for (let k = 0; k < 1500; k++) {
    const z = 2 * rng() - 1,
      phi = 2 * Math.PI * rng(),
      s = Math.sqrt(1 - z * z);
    const n: Vec3 = [s * Math.cos(phi), z * 0.6, s * Math.sin(phi)];
    const normal = unit(n);
    const xyz = Float64Array.of(1.5 * normal[0], 1.5 * normal[1], 1.5 * normal[2]);
    const observations: MvsSceneLandmark['observations'][number][] = [];
    for (const view of views.slice(0, 10)) {
      const C = mvsCameraCenter(view);
      const toCamera = unit([C[0]! - xyz[0]!, C[1]! - xyz[1]!, C[2]! - xyz[2]!]);
      if (dot(normal, toCamera) < Math.cos(70 * deg)) continue;
      const pixel = project(view, xyz);
      if (pixel) observations.push({ viewId: view.id, pixel });
    }
    if (observations.length >= 2) landmarks.push({ xyz, observations });
  }
  for (let k = 0; k < 12; k++) {
    const xyz = Float64Array.of(0.3 * Math.cos(k), 0.3 * Math.sin(k), 40 + 0.1 * (k % 3));
    const pixel = project(views[10]!, xyz);
    assert.ok(pixel);
    landmarks.push({ xyz, observations: [{ viewId: 10, pixel }] });
  }
  let seen = 0;
  for (const landmark of landmarks) {
    const pixel = seen < 5 ? project(views[11]!, landmark.xyz) : undefined;
    if (!pixel) continue;
    (landmark.observations as { viewId: number; pixel: readonly [number, number] }[]).push({
      viewId: 11,
      pixel,
    });
    seen++;
  }
  return { views, landmarks };
}

test('g(α) is 0 below 1°, (α / 5°)² up to 5° and 1 above', () => {
  assert.equal(mvsAngleWeight(0.5), 0);
  assert.equal(mvsAngleWeight(0.999), 0);
  assert.ok(Math.abs(mvsAngleWeight(1) - 0.04) < 1e-15);
  assert.ok(Math.abs(mvsAngleWeight(2.5) - 0.25) < 1e-15);
  assert.equal(mvsAngleWeight(5), 1);
  assert.equal(mvsAngleWeight(30), 1);
});

test('a pair scores 0 below 16 shared landmarks and Σ g(α) from 16', () => {
  // Two cameras 1 apart on x, points at depth 10 on the axis between them: α = 2 atan(0.05).
  const views = [lookAt(0, [-0.5, 0, 10]), lookAt(1, [0.5, 0, 10])].map((view) => ({
    ...view,
    pose: {
      R: Float64Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1),
      t: mvsCameraCenter(view).map((x) => -x),
    },
  }));
  const point = (k: number): MvsSceneLandmark => ({
    xyz: Float64Array.of(0, 0.001 * k, 20),
    observations: [
      { viewId: 0, pixel: [320, 240] },
      { viewId: 1, pixel: [320, 240] },
    ],
  });
  const alpha = (2 * Math.atan2(0.5, 10)) / deg;
  for (const [count, shared] of [
    [15, 0],
    [16, 16],
  ] as const) {
    const scene = { views, landmarks: Array.from({ length: count }, (_, k) => point(k)) };
    const scores = mvsViewScores(scene);
    assert.equal(scores[0], 0);
    assert.equal(scores[1], scores[2]);
    assert.ok(Math.abs(scores[1]! - shared * mvsAngleWeight(alpha)) < 1e-9, String(scores[1]));
  }
  assert.ok(alpha > 5.7 && alpha < 5.8);
});

test('normalization: centroid 0, RMS radius 1, same pixels, and the inverse map', () => {
  const scene = arcScene();
  const normalization = mvsNormalization(scene);
  const normalized = normalizeMvsScene(scene, normalization);
  const centroid = [0, 0, 0];
  let sum = 0;
  for (const { xyz } of normalized.landmarks) {
    for (let i = 0; i < 3; i++) centroid[i]! += xyz[i]! / normalized.landmarks.length;
    sum += dot(xyz, xyz);
  }
  assert.ok(Math.hypot(...centroid) < 1e-12);
  assert.ok(Math.abs(Math.sqrt(sum / normalized.landmarks.length) - 1) < 1e-12);
  let worst = 0;
  scene.landmarks.forEach((landmark, k) => {
    for (const view of scene.views) {
      const a = project(view, landmark.xyz),
        b = project(normalized.views[scene.views.indexOf(view)]!, normalized.landmarks[k]!.xyz);
      assert.equal(a === undefined, b === undefined);
      if (a && b) worst = Math.max(worst, Math.hypot(a[0] - b[0], a[1] - b[1]));
      const z = mvsDepth(view, landmark.xyz);
      const zn = mvsDepth(
        normalized.views[scene.views.indexOf(view)]!,
        normalized.landmarks[k]!.xyz,
      );
      assert.ok(Math.abs(zn * normalization.scale - z) < 1e-9 * Math.max(1, Math.abs(z)));
    }
  });
  assert.ok(worst < 1e-9, String(worst));
  const flat = Float64Array.from(normalized.landmarks.flatMap(({ xyz }) => [...Array.from(xyz)]));
  const back = denormalizeMvsPoints(flat, normalization);
  scene.landmarks.forEach(({ xyz }, k) => {
    for (let i = 0; i < 3; i++) assert.ok(Math.abs(back[3 * k + i]! - xyz[i]!) < 1e-12);
  });
});

test('arc scene: the source lists are the nearest arc neighbours, scores match a brute force', () => {
  const scene = arcScene();
  const plan = planMvsViews(scene);
  const status = Object.fromEntries(plan.views.map((v) => [v.id, v.status]));
  assert.equal(status[10], 'noSource');
  assert.equal(status[11], 'fewLandmarks');
  for (let id = 0; id < 10; id++) assert.equal(status[id], 'ready');
  assert.deepEqual(plan.packed, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  const ids = (view: number) => plan.views[view]!.sources.map((v) => plan.packed[v]!);
  // id 4 (0°): the twin at 0.5° scores 0 and is no source. The rest are the ±15°, ±30°, ±45° cameras.
  assert.deepEqual(new Set(ids(4)), new Set([1, 2, 3, 5, 6, 7]));
  assert.deepEqual(new Set(ids(4).slice(0, 2)), new Set([3, 5]));
  assert.ok(!ids(4).includes(9));
  // id 0 (−60°) has neighbours on one side only: −45°, −30°, −15°, then 0° and its twin, then 15°.
  assert.deepEqual(ids(0).slice(0, 3), [1, 2, 3]);
  assert.deepEqual(new Set(ids(0)), new Set([1, 2, 3, 4, 9, 5]));

  // A brute force of 3.2 on the normalized scene.
  const normalized = normalizeMvsScene(scene, plan.normalization);
  const scores = mvsViewScores(normalized);
  const V = scene.views.length;
  for (let r = 0; r < V; r++)
    for (let s = 0; s < V; s++) {
      if (r === s) {
        assert.equal(scores[r * V + s], 0);
        continue;
      }
      let sum = 0,
        shared = 0;
      const a = mvsCameraCenter(normalized.views[r]!),
        b = mvsCameraCenter(normalized.views[s]!);
      for (const landmark of normalized.landmarks) {
        const seen = new Set(landmark.observations.map((o) => o.viewId));
        if (!seen.has(scene.views[r]!.id) || !seen.has(scene.views[s]!.id)) continue;
        const u = [0, 1, 2].map((i) => a[i]! - landmark.xyz[i]!),
          w = [0, 1, 2].map((i) => b[i]! - landmark.xyz[i]!);
        const cos = dot(u, w) / Math.hypot(...u) / Math.hypot(...w);
        sum += mvsAngleWeight(Math.acos(Math.min(1, cos)) / deg);
        shared++;
      }
      const expected = shared >= 16 ? sum : 0;
      assert.ok(Math.abs(scores[r * V + s]! - expected) < 1e-6, `${r} ${s}`);
    }
  // Every list is sorted by score, best first, and holds at most 6 sources.
  plan.views.forEach((view, r) => {
    assert.ok(view.sources.length <= 6);
    const list = view.sources.map((v) => scores[r * V + plan.packed[v]!]!);
    for (let k = 1; k < list.length; k++) assert.ok(list[k - 1]! >= list[k]!);
    assert.ok(list.every((score) => score > 0));
  });
});

test('arc scene: each depth range holds every observed landmark depth and follows 3.3', () => {
  const scene = arcScene();
  const plan = planMvsViews(scene);
  const normalized = normalizeMvsScene(scene, plan.normalization);
  let checked = 0;
  for (const entry of plan.views.filter((v) => v.status === 'ready')) {
    const view = normalized.views.find((v) => v.id === entry.id)!;
    const depths = normalized.landmarks
      .filter((l) => l.observations.some((o) => o.viewId === entry.id))
      .map((l) => mvsDepth(view, l.xyz))
      .sort((a, b) => a - b);
    const [lo, hi] = entry.depthRange!;
    assert.equal(entry.landmarks, depths.length);
    assert.equal(lo, 0.75 * depths[Math.floor((depths.length - 1) * 0.01)]!);
    assert.equal(hi, 1.25 * depths[Math.ceil((depths.length - 1) * 0.99)]!);
    for (const z of depths) assert.ok(z >= lo && z <= hi);
    checked += depths.length;
    // mvsCameras carries the range, the size and the center.
    const base = entry.index * MVS_CAMERA_FLOATS;
    assert.equal(plan.cameras[base + 18], Math.fround(lo));
    assert.equal(plan.cameras[base + 19], Math.fround(hi));
    assert.deepEqual(Array.from(plan.cameras.subarray(base + 16, base + 18)), [WIDTH, HEIGHT]);
    const C = mvsCameraCenter(view);
    for (let i = 0; i < 3; i++) assert.equal(plan.cameras[base + 20 + i], Math.fround(C[i]!));
    assert.equal(plan.cameras[base + 23], 0);
    for (let i = 0; i < 9; i++) assert.equal(plan.cameras[base + i], Math.fround(view.pose.R[i]!));
    for (let i = 0; i < 3; i++)
      assert.equal(plan.cameras[base + 9 + i], Math.fround(view.pose.t[i]!));
    const slots = plan.sources.subarray(
      entry.index * MVS_SOURCE_SLOTS,
      (entry.index + 1) * MVS_SOURCE_SLOTS,
    );
    assert.equal(slots[0], entry.sources.length);
    assert.deepEqual(Array.from(slots.subarray(1, 1 + entry.sources.length)), entry.sources);
  }
  assert.ok(checked > 3000, String(checked));
  assert.equal(plan.cameras.length, 10 * MVS_CAMERA_FLOATS);
  assert.equal(plan.sources.length, 10 * MVS_SOURCE_SLOTS);
  assert.equal(mvsDepthRange([]), undefined);
});

test('arc scene: every seed back-projects onto a landmark within the rounding of its pixel', () => {
  const scene = arcScene();
  const plan = planMvsViews(scene);
  const normalized = normalizeMvsScene(scene, plan.normalization);
  let seeds = 0,
    worst = 0;
  for (const entry of plan.views.filter((v) => v.status === 'ready')) {
    const view = normalized.views.find((v) => v.id === entry.id)!;
    const { R, t } = view.pose;
    const first = plan.seedOffsets[entry.index]!,
      end = plan.seedOffsets[entry.index + 1]!;
    // The landmarks are exact and far apart, so nearly every observation is its own seed.
    assert.ok(end - first >= 0.98 * entry.landmarks, `${end - first} of ${entry.landmarks}`);
    let previous = -1;
    for (let k = first; k < end; k++) {
      const [x, y, z] = plan.seeds.subarray(3 * k, 3 * k + 3) as unknown as [
        number,
        number,
        number,
      ];
      assert.ok(Number.isInteger(x) && Number.isInteger(y) && x < WIDTH && y < HEIGHT);
      assert.ok(y * WIDTH + x > previous, 'seeds are ordered by pixel index');
      previous = y * WIDTH + x;
      // Pinhole back-projection on the undistorted image, then camera to world'.
      const q = [((x - camera.cx) / camera.fx) * z, ((y - camera.cy) / camera.fy) * z, z];
      const d = [q[0]! - t[0]!, q[1]! - t[1]!, q[2]! - t[2]!];
      const X = [0, 1, 2].map((j) => R[j]! * d[0]! + R[3 + j]! * d[1]! + R[6 + j]! * d[2]!);
      let nearest = Infinity;
      for (const landmark of normalized.landmarks)
        if (landmark.observations.some((o) => o.viewId === entry.id))
          nearest = Math.min(
            nearest,
            Math.hypot(...[0, 1, 2].map((i) => X[i]! - landmark.xyz[i]!)),
          );
      // Rounding moves the pixel by at most √2 / 2 px, so the point by 0.71 · z / f laterally.
      const bound = (Math.SQRT1_2 * z) / camera.fx + 1e-6 * z;
      worst = Math.max(worst, nearest / bound);
      assert.ok(nearest <= bound, `${nearest} > ${bound}`);
      seeds++;
    }
  }
  assert.equal(seeds, plan.seeds.length / 3);
  assert.equal(plan.seedOffsets.length, plan.packed.length + 1);
  assert.ok(seeds > 3000 && worst <= 1, `${seeds} ${worst}`);
});

test('a source must get a depth map itself: the selection repeats until no view drops out', () => {
  // Views 0 and 1 both look down +z, from the origin and from (1, 0, 12). They share 20
  // landmarks, so their score is positive. 14 of them lie at z = 10, behind view 1, so view 1
  // has 6 depths and gets no depth map. View 0 then has no source left.
  const identity = Float64Array.of(1, 0, 0, 0, 1, 0, 0, 0, 1);
  const view = (id: number, t: number[]): MvsSceneView => ({
    id,
    width: WIDTH,
    height: HEIGHT,
    camera: { ...camera },
    pose: { R: identity, t: Float64Array.from(t) },
  });
  const views = [view(0, [0, 0, 0]), view(1, [-1, 0, -12])];
  const landmarks: MvsSceneLandmark[] = Array.from({ length: 20 }, (_, k) => ({
    xyz: Float64Array.of(0.1 * (k % 5), 0.1 * Math.floor(k / 5), k < 14 ? 10 : 20),
    observations: [
      { viewId: 0, pixel: [320, 240] },
      { viewId: 1, pixel: [320, 240] },
    ],
  }));
  const scene = { views, landmarks };
  assert.ok(mvsViewScores(scene)[1]! > 0);
  assert.equal(mvsObservedDepths(scene, 0).length, 20);
  assert.equal(mvsObservedDepths(scene, 1).length, 6);
  const plan = planMvsViews(scene);
  assert.deepEqual(
    plan.views.map((v) => [v.status, v.index]),
    [
      ['noSource', -1],
      ['fewLandmarks', -1],
    ],
  );
  assert.equal(plan.packed.length, 0);
  assert.equal(plan.cameras.length, 0);
  assert.equal(plan.seeds.length, 0);
});

test('invalid input is refused', () => {
  const scene = arcScene();
  assert.throws(() => planMvsViews(scene, { maxSources: 7 }), RangeError);
  assert.throws(() => planMvsViews(scene, { maxSources: 0 }), RangeError);
  const { width: _width, ...sizeless } = scene.views[0]!;
  assert.throws(
    () => planMvsViews({ ...scene, views: [sizeless, ...scene.views.slice(1)] }),
    /width and height/,
  );
  assert.throws(
    () => planMvsViews({ ...scene, views: [scene.views[0]!, scene.views[0]!] }),
    /Duplicate view id/,
  );
});

// gerrard-hall: COLMAP's poses and intrinsics from sample.json, and the dataset's own SIFT
// keypoints and raw matches from colmap-sift.bin. The tracks are the connected components of
// the raw matches. A component that holds two keypoints of one view is dropped. The rest are
// triangulated with COLMAP's poses (`triangulateTrack`, 2 px, 1°), and only the inlier
// observations are kept. No mapper runs, so the fixture depends only on the sample's files.
type SampleFrame = {
  id: number;
  width: number;
  height: number;
  camera: { fx: number; fy: number; cx: number; cy: number; k1: number; k2: number };
  reference: { R: number[]; t: number[] };
};
type Sample = {
  frames: SampleFrame[];
  features: {
    keypointBytes: number;
    keypoints: { id: number; offset: number; count: number }[];
    pairs: { first: number; second: number; offset: number; count: number }[];
  };
};

/** The sample's R is rounded to about 1e-6. Gram–Schmidt on its rows makes it a rotation. */
function orthonormal(R: readonly number[]): Float64Array {
  const x = unit([R[0]!, R[1]!, R[2]!]);
  const b: Vec3 = [R[3]!, R[4]!, R[5]!];
  const d = dot(x, b);
  const y = unit([b[0] - d * x[0], b[1] - d * x[1], b[2] - d * x[2]]);
  return Float64Array.of(...x, ...y, ...cross(x, y));
}

function gerrardHall(): MvsScene {
  const dir = new URL('../website/public/sample/gerrard-hall/', import.meta.url);
  const sample = JSON.parse(readFileSync(new URL('sample.json', dir), 'utf8')) as Sample;
  const bytes = readFileSync(new URL('colmap-sift.bin', dir));
  const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  const keypoints = new Float32Array(buffer, 0, sample.features.keypointBytes / 4);
  const matches = new Uint16Array(buffer, sample.features.keypointBytes);
  const views: MvsSceneView[] = sample.frames.map((frame) => ({
    id: frame.id,
    width: frame.width,
    height: frame.height,
    camera: { ...frame.camera },
    pose: { R: orthonormal(frame.reference.R), t: Float64Array.from(frame.reference.t) },
  }));
  const blocks = new Map(sample.features.keypoints.map((block) => [block.id, block]));
  const base = new Map<number, number>(),
    owner: number[] = [];
  for (const frame of sample.frames) {
    base.set(frame.id, owner.length);
    for (let k = 0; k < blocks.get(frame.id)!.count; k++) owner.push(frame.id);
  }
  const parent = Int32Array.from({ length: owner.length }, (_, i) => i);
  const find = (i: number): number => {
    while (parent[i] !== i) i = parent[i] = parent[parent[i]!]!;
    return i;
  };
  for (const pair of sample.features.pairs)
    for (let k = 0; k < pair.count; k++) {
      const a = find(base.get(pair.first)! + matches[pair.offset + 2 * k]!),
        b = find(base.get(pair.second)! + matches[pair.offset + 2 * k + 1]!);
      if (a !== b) parent[Math.max(a, b)] = Math.min(a, b);
    }
  const tracks = new Map<number, number[]>();
  for (let i = 0; i < owner.length; i++) {
    const root = find(i);
    if (!tracks.has(root)) tracks.set(root, []);
    tracks.get(root)!.push(i);
  }
  const byId = new Map(views.map((view) => [view.id, view]));
  const landmarks: MvsSceneLandmark[] = [];
  for (const members of tracks.values()) {
    if (members.length < 2) continue;
    const ids = members.map((i) => owner[i]!);
    if (new Set(ids).size !== ids.length) continue;
    const observations = members.map((i, k) => {
      const view = byId.get(ids[k]!)!,
        local = blocks.get(view.id)!.offset + 2 * (i - base.get(view.id)!);
      const pixel = [keypoints[local]!, keypoints[local + 1]!] as const;
      return { viewId: view.id, camera: view.camera, pose: view.pose as never, pixel };
    });
    const track = triangulateTrack(observations, { maxErrorPx: 2, minAngleDeg: 1 });
    if (!track) continue;
    const kept = observations.filter((_, k) => track.inliers[k]);
    if (kept.length >= 2)
      landmarks.push({
        xyz: track.xyz,
        observations: kept.map(({ viewId, pixel }) => ({ viewId, pixel })),
      });
  }
  return { views, landmarks };
}

test('gerrard-hall from the dataset features: the source lists, and their stability', (t) => {
  const scene = gerrardHall();
  const plan = planMvsViews(scene);
  t.diagnostic(`${scene.landmarks.length} landmarks`);
  assert.ok(scene.landmarks.length > 7000, String(scene.landmarks.length));
  assert.ok(plan.views.every((view) => view.status === 'ready'));
  assert.deepEqual(plan.packed, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
  const lists = plan.views.map((view) => view.sources);
  // Recorded on 2026-10-20 from this fixture. The photographs walk around the building in two
  // runs, 0 to 4 and 5 to 11. Each list holds the nearest frames of its run.
  assert.deepEqual(lists, [
    [1, 2, 3, 8, 7, 9],
    [0, 2, 3, 4, 8, 7],
    [1, 0, 3, 4, 8, 9],
    [1, 2, 0, 4, 5, 8],
    [3, 2, 1, 5, 0, 6],
    [6, 7, 8, 9, 10, 4],
    [7, 8, 9, 10, 5, 0],
    [9, 10, 8, 6, 5, 11],
    [10, 7, 9, 6, 11, 0],
    [7, 10, 8, 6, 11, 5],
    [8, 7, 9, 6, 11, 5],
    [9, 8, 7, 10, 6, 0],
  ]);
  // Stability: drop 20 % of the landmarks at random, five times. Each reference keeps its first
  // three sources (the `bestSources` of 3.7) in its list, and at least 5 of its 6 sources.
  for (const seed of [1, 2, 3, 4, 5]) {
    const rng = seededRandom(seed);
    const subset = { ...scene, landmarks: scene.landmarks.filter(() => rng() >= 0.2) };
    const other = planMvsViews(subset).views.map((view) => view.sources);
    lists.forEach((list, r) => {
      const kept = list.filter((s) => other[r]!.includes(s)).length;
      assert.ok(kept >= 5, `seed ${seed}, view ${r}: ${list} against ${other[r]}`);
      assert.ok(
        list.slice(0, 3).every((s) => other[r]!.includes(s)),
        `seed ${seed}, view ${r}`,
      );
    });
  }
  // Each reference has seeds, and its range holds the depths of its own landmarks.
  for (const entry of plan.views) {
    const count = plan.seedOffsets[entry.index + 1]! - plan.seedOffsets[entry.index]!;
    assert.ok(count > 0.8 * entry.landmarks, `${entry.id}: ${count} of ${entry.landmarks}`);
    assert.ok(entry.depthRange![0] > 0 && entry.depthRange![1] > entry.depthRange![0]);
  }
});
