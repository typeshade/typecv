import { createHash } from 'node:crypto';
import { seededRandom } from '../src/geometry/numerics.ts';
import { cameraProjection, type Camera } from '../src/geometry/camera.ts';
import { transformPoint } from '../src/geometry/pose.ts';
import { assignViewIds } from '../src/reconstruction/identity.ts';
import type { ViewGraphKeyframe } from '../src/reconstruction/view-graph.ts';

/** A synthetic scene of several camera paths for the view graph (docs/multi-video.md 3.3, 4.1). */

export const CAMERA: Camera = { fx: 850, fy: 850, cx: 480, cy: 270 };
export const WIDTH = 960,
  HEIGHT = 540;
export const WORDS = 32;

type Vec = [number, number, number];
const sub = (a: Vec, b: Vec): Vec => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a: Vec, b: Vec): Vec => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];
const norm = (a: Vec): Vec => {
  const n = Math.hypot(...a);
  return [a[0] / n, a[1] / n, a[2] / n];
};

/** World-to-camera pose of a camera at `center` that looks at `target`. */
function lookAt(center: Vec, target: Vec) {
  const z = norm(sub(target, center)),
    x = norm(cross(z, [0, -1, 0])),
    y = cross(z, x);
  const R = Float64Array.of(...x, ...y, ...z);
  const t = Float64Array.of(
    -(x[0] * center[0] + x[1] * center[1] + x[2] * center[2]),
    -(y[0] * center[0] + y[1] * center[1] + y[2] * center[2]),
    -(z[0] * center[0] + z[1] * center[1] + z[2] * center[2]),
  );
  return { R, t };
}

export interface PathSpec {
  readonly label: string;
  /** Which point cloud the path looks at. */
  readonly cloud: number;
  readonly views: number;
  /** Arc of the path around the cloud, in degrees. */
  readonly from: number;
  readonly to: number;
  readonly radius: number;
  readonly height: number;
}

/** A unit RootSIFT-like vector: the square root of an L1-normalized positive vector. */
function rootSift(base: Float64Array, rng: () => number, noise: number): Float32Array {
  const v = Float64Array.from(base, (x) => Math.max(0, x + (rng() - 0.5) * noise));
  let l1 = 0;
  for (const x of v) l1 += x;
  const out = new Float32Array(128);
  for (let i = 0; i < 128; i++) out[i] = Math.sqrt(v[i]! / l1);
  return out;
}

/**
 * Paths of `views` keyframes each. A path is an arc around a cloud of `points` landmarks. Each
 * landmark has a descriptor and a visual word. The global descriptor of a keyframe is a stand-in
 * for step 3's: the square-rooted histogram of the words of the landmarks it sees, L2-normalized.
 */
export function pathScene(paths: readonly PathSpec[], { points = 60, seed = 41 } = {}) {
  const rng = seededRandom(seed);
  const clouds = new Map<number, { xyz: Vec; base: Float64Array; word: number }[]>();
  const keyframes: ViewGraphKeyframe[] = [];
  const truth: { path: string; cloud: number }[] = [];
  for (const path of paths) {
    if (!clouds.has(path.cloud))
      clouds.set(
        path.cloud,
        Array.from({ length: points }, () => ({
          xyz: [path.cloud * 100 + (rng() - 0.5) * 4, (rng() - 0.5) * 3, (rng() - 0.5) * 4] as Vec,
          base: Float64Array.from({ length: 128 }, () => 0.05 + rng()),
          word: Math.floor(rng() * WORDS),
        })),
      );
  }
  const videoIds = new Map(
    paths.map((p) => [p.label, createHash('sha256').update(p.label).digest('hex').slice(0, 16)]),
  );
  for (const path of paths) {
    const cloud = clouds.get(path.cloud)!;
    const centre: Vec = [path.cloud * 100, 0, 0];
    for (let i = 0; i < path.views; i++) {
      const angle = ((path.from + ((path.to - path.from) * i) / (path.views - 1)) * Math.PI) / 180;
      const center: Vec = [
        centre[0] + path.radius * Math.sin(angle),
        path.height,
        centre[2] - path.radius * Math.cos(angle),
      ];
      const pose = lookAt(center, centre);
      const keypoints: number[] = [],
        rows: Float32Array[] = [],
        histogram = new Float64Array(WORDS);
      for (const p of cloud) {
        const q = cameraProjection(CAMERA, transformPoint(pose, Float64Array.from(p.xyz)));
        if (!q || q.pixel[0] < 0 || q.pixel[0] >= WIDTH || q.pixel[1] < 0 || q.pixel[1] >= HEIGHT)
          continue;
        keypoints.push(q.pixel[0] + (rng() - 0.5) * 0.6, q.pixel[1] + (rng() - 0.5) * 0.6);
        rows.push(rootSift(p.base, rng, 0.06));
        histogram[p.word]! += 1;
      }
      const global = Float32Array.from(histogram, Math.sqrt);
      const length = Math.hypot(...global);
      for (let k = 0; k < WORDS; k++) global[k] = global[k]! / length;
      const descriptors = new Float32Array(rows.length * 128);
      rows.forEach((row, k) => descriptors.set(row, k * 128));
      keyframes.push({
        viewId: -1,
        videoId: videoIds.get(path.label)!,
        frameKey: createHash('sha256').update(`${path.label} ${i}`).digest('hex').slice(0, 16),
        time: i * 0.25,
        camera: { ...CAMERA },
        keypoints: Float64Array.from(keypoints),
        descriptors,
        global,
      });
      truth.push({ path: path.label, cloud: path.cloud });
    }
  }
  const ids = assignViewIds(keyframes);
  return {
    keyframes: keyframes.map((k, i) => ({ ...k, viewId: ids[i]! })),
    truth,
    videoIds,
  };
}
