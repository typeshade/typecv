import { seededRandom } from '../../src/geometry/numerics.ts';
import { cameraCenter, rotationExp, transformPoint } from '../../src/geometry/pose.ts';
import type { Camera } from '../../src/geometry/camera.ts';
import type { ImageFeatures, PairMatches, RegisteredView } from '../../src/reconstruction/types.ts';

export const GENERAL_VIDEO_CASES = [
  'translation',
  'rotation',
  'rotation-start',
  'mapped-rotation',
  'reobserved',
  'orbit',
  'planar',
  'far',
  'zoom',
  'rotation-zoom',
  'same-place-shots',
  'disconnected',
] as const;
export type GeneralVideoCase = (typeof GENERAL_VIDEO_CASES)[number];

export interface GeneralVideoFixture {
  scenario: GeneralVideoCase;
  scenarioId: string;
  input: { images: ImageFeatures[]; pairs: PairMatches[] };
  truth: {
    views: RegisteredView[];
    points: Float64Array[];
    visiblePointIds: number[][];
    componentOfView: number[];
    shotOfView: number[];
    editTimes: number[];
  };
  provenance: {
    calibration: 'known-synthetic';
    correspondences: 'supplied-synthetic-identities';
    mode: 'geometry-only';
  };
  expected: {
    initialization: 'map' | 'deferred';
    componentCount: number;
    unobservedAfterMotion: number[];
  };
}

/** Geometry fixture only: identities are supplied, so this is NOT an RGB-only accuracy benchmark. */
export function generalVideoFixture(scenario: GeneralVideoCase): GeneralVideoFixture {
  if (!GENERAL_VIDEO_CASES.includes(scenario))
    throw new RangeError('Unknown general video fixture');
  const rng = seededRandom(260123);
  const points = Array.from({ length: 60 }, () =>
    Float64Array.of(
      (rng() - 0.5) * 3.6,
      (rng() - 0.5) * 2.4,
      scenario === 'planar' ? 7 : 5 + rng() * 4,
    ),
  );
  if (scenario === 'far') for (const point of points) point[2] += 5000;
  const count =
    scenario === 'reobserved' || scenario === 'disconnected'
      ? 6
      : scenario === 'same-place-shots'
        ? 4
        : 5;
  const componentOfView = Array.from({ length: count }, (_, i) =>
    scenario === 'disconnected' && i >= 3 ? 1 : 0,
  );
  const shotOfView = Array.from({ length: count }, (_, i) =>
    (scenario === 'same-place-shots' && i >= 2) || (scenario === 'disconnected' && i >= 3) ? 1 : 0,
  );
  const views: RegisteredView[] = Array.from({ length: count }, (_, i) => {
    const j = scenario === 'disconnected' ? i % 3 : i;
    let center = Float64Array.of(j * 0.3, Math.sin(j * 1.7) * 0.07, j * j * 0.008);
    let angle = [0.006 * j, 0.018 * j, -0.004 * j];
    if (['rotation', 'zoom', 'rotation-zoom'].includes(scenario)) {
      center = new Float64Array(3);
      angle = scenario === 'zoom' ? [0, 0, 0] : [0.01 * j, 0.07 * j, 0];
    }
    if (scenario === 'rotation-start' && i < 3) center = new Float64Array(3);
    if (scenario === 'mapped-rotation' || scenario === 'reobserved') {
      if (i >= 1 && i <= 3) center = Float64Array.of(0.3, 0.07 * Math.sin(1.7), 0.008);
      angle = [0, 0.035 * i, 0];
    }
    if (scenario === 'orbit') {
      const theta = (i - 2) * 0.22;
      center = Float64Array.of(0.9 * Math.sin(theta), 0, 0.9 * (1 - Math.cos(theta)));
      angle = [0, Math.atan2(center[0]!, 6.5 - center[2]!), 0];
    }
    if (scenario === 'same-place-shots') {
      center = i < 2 ? new Float64Array(3) : Float64Array.of(0.8, 0.1, 0.04);
      angle = [0, (i % 2) * 0.06, 0];
    }
    const scale = scenario === 'zoom' || scenario === 'rotation-zoom' ? 1 + 0.16 * i : 1;
    const camera: Camera = { fx: 480 * scale, fy: 480 * scale, cx: 319.5, cy: 239.5 };
    const R = rotationExp(angle);
    return {
      id: i,
      camera,
      pose: { R, t: transformPoint({ R, t: new Float64Array(3) }, center).map((x) => -x) },
    };
  });
  const visiblePointIds = views.map((_, i) =>
    points.flatMap((_, p) => {
      const newPoint = p >= 40;
      if (scenario === 'mapped-rotation' && newPoint && i !== 2 && i !== 3) return [];
      if (scenario === 'reobserved' && newPoint && i < 2) return [];
      return [p];
    }),
  );
  // Use independent scalar projection, not the reconstruction's cameraProjection implementation.
  const images: ImageFeatures[] = views.map((view) => ({
    id: view.id,
    camera: { ...view.camera },
    keypoints: Float64Array.from(
      points.flatMap((point) => {
        const { R, t } = view.pose;
        const q = [0, 1, 2].map(
          (axis) =>
            R[axis * 3]! * point[0]! +
            R[axis * 3 + 1]! * point[1]! +
            R[axis * 3 + 2]! * point[2]! +
            t[axis]!,
        );
        return [
          (view.camera.fx * q[0]!) / q[2]! + view.camera.cx,
          (view.camera.fy * q[1]!) / q[2]! + view.camera.cy,
        ];
      }),
    ),
  }));
  const pairs: PairMatches[] = [];
  for (let a = 0; a < count; a++)
    for (let b = a + 1; b < count; b++) {
      if (componentOfView[a] !== componentOfView[b]) continue;
      const ids = visiblePointIds[a]!.filter((p) => visiblePointIds[b]!.includes(p));
      pairs.push({ first: a, second: b, matches: Uint32Array.from(ids.flatMap((p) => [p, p])) });
    }
  const deferred = ['rotation', 'planar', 'far', 'zoom', 'rotation-zoom'].includes(scenario);
  const id: Record<GeneralVideoCase, string> = {
    translation: 'S01',
    rotation: 'S02',
    'rotation-start': 'S03-start',
    'mapped-rotation': 'S03-unseen',
    reobserved: 'S03-reseen',
    orbit: 'S04',
    planar: 'S05-planar',
    far: 'S05-far',
    zoom: 'S06-zoom',
    'rotation-zoom': 'S06-rotation-zoom',
    'same-place-shots': 'S15',
    disconnected: 'S16',
  };
  return {
    scenario,
    scenarioId: id[scenario],
    input: { images, pairs },
    truth: {
      views,
      points,
      visiblePointIds,
      componentOfView,
      shotOfView,
      editTimes: views.map((_, i) => i / 4),
    },
    provenance: {
      calibration: 'known-synthetic',
      correspondences: 'supplied-synthetic-identities',
      mode: 'geometry-only',
    },
    expected: {
      initialization: deferred ? 'deferred' : 'map',
      componentCount: scenario === 'disconnected' ? 2 : 1,
      unobservedAfterMotion:
        scenario === 'mapped-rotation' ? Array.from({ length: 20 }, (_, i) => i + 40) : [],
    },
  };
}

/** Separate RGB source: analytical static room. No feature identities are returned to its pipeline. */
export function renderGeneralVideoFrame(view: RegisteredView, width = 320, height = 240) {
  if (
    !Number.isSafeInteger(width) ||
    !Number.isSafeInteger(height) ||
    width < 48 ||
    height < 48 ||
    width > 1280 ||
    height > 960
  )
    throw new RangeError('Fixture image dimensions must be bounded');
  const sx = width / 640,
    sy = height / 480;
  const camera = {
    ...view.camera,
    fx: view.camera.fx * sx,
    fy: view.camera.fy * sy,
    cx: (view.camera.cx + 0.5) * sx - 0.5,
    cy: (view.camera.cy + 0.5) * sy - 0.5,
  };
  const center = cameraCenter(view.pose),
    { R } = view.pose;
  const rgba = new Uint8ClampedArray(width * height * 4);
  const depthZ = new Float32Array(width * height);
  const boxes = [
    [-1.8, -0.6, 4.8, -0.8, 0.8, 5.8],
    [0.3, -0.8, 6.1, 1.4, 0.7, 7.2],
    [-0.4, 0.5, 3.5, 0.4, 1.3, 4.2],
  ];
  for (let y = 0; y < height; y++)
    for (let x = 0; x < width; x++) {
      const ray = [(x - camera.cx) / camera.fx, (y - camera.cy) / camera.fy, 1];
      const d = [0, 1, 2].map((axis) => R[axis]! * ray[0]! + R[3 + axis]! * ray[1]! + R[6 + axis]!);
      let distance = Infinity,
        normalAxis = 2,
        surface = 0;
      for (const [axis, position, id] of [
        [2, 10, 1],
        [1, 1.3, 2],
        [0, -3.2, 3],
        [0, 3.2, 4],
      ]) {
        const t = (position! - center[axis!]!) / d[axis!]!;
        if (t > 0 && t < distance) {
          distance = t;
          normalAxis = axis!;
          surface = id!;
        }
      }
      for (const [index, box] of boxes.entries()) {
        let near = 0,
          far = distance,
          face = 0;
        for (let axis = 0; axis < 3; axis++) {
          if (Math.abs(d[axis]!) < 1e-12) {
            if (center[axis]! < box[axis]! || center[axis]! > box[axis + 3]!) far = -1;
            continue;
          }
          const t1 = (box[axis]! - center[axis]!) / d[axis]!,
            t2 = (box[axis + 3]! - center[axis]!) / d[axis]!;
          const entry = Math.min(t1, t2);
          if (entry > near) {
            near = entry;
            face = axis;
          }
          far = Math.min(far, Math.max(t1, t2));
        }
        if (near > 0 && near < far && near < distance) {
          distance = near;
          normalAxis = face;
          surface = 5 + index;
        }
      }
      const at = y * width + x;
      if (Number.isFinite(distance)) {
        const position = [0, 1, 2].map((axis) => center[axis]! + distance * d[axis]!);
        const axes = [0, 1, 2].filter((axis) => axis !== normalAxis);
        const u = Math.floor(position[axes[0]!]! * 14),
          v = Math.floor(position[axes[1]!]! * 14);
        const hash =
          (Math.imul(u, 73856093) ^ Math.imul(v, 19349663) ^ Math.imul(surface, 83492791)) >>> 0;
        const value = 35 + (hash % 190);
        rgba.set([value, 25 + ((hash >>> 8) % 200), 25 + ((hash >>> 16) % 200), 255], at * 4);
        depthZ[at] = distance; // Camera +Z, since camera-space ray.z=1; not Euclidean ray length.
      } else {
        rgba.set([20, 30, 40, 255], at * 4);
        depthZ[at] = Infinity;
      }
    }
  return { id: view.id, width, height, camera, rgba, depthZ };
}
