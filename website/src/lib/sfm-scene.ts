export type SceneView = {
  id: number;
  time?: number;
  sequence?: number;
  /** The video of a view in a multi-video scene (`docs/multi-video.md`); absent for one video. */
  videoId?: string;
  width?: number;
  height?: number;
  image?: string;
  pose: { R: Float64Array; t: Float64Array };
  camera: { fx: number; fy: number; cx: number; cy: number; k1?: number; k2?: number };
};
export type ScenePoint = {
  trackId: number;
  xyz: Float64Array;
  color?: [number, number, number];
  observations: { viewId: number; featureId: number; pixel: readonly [number, number] }[];
};
export type Scene = { views: SceneView[]; landmarks: ScenePoint[] };
export type DepthPoint = { id: number; x: number; y: number; depth: number };

/** Keep frame timing/pixels metadata while the solved pose and calibration remain authoritative. */
export function sceneViewWithFrame<T extends SceneView>(view: T, frame?: object): T {
  return { ...frame, ...view };
}

/** Adjacent cameras belong to the same video; never bridge a missing pose. */
export function previousCamera(scene: Scene | undefined, view: SceneView | undefined) {
  if (!view) return undefined;
  return scene?.views.find(
    (candidate) =>
      candidate.videoId === view.videoId &&
      (view.sequence !== undefined
        ? candidate.sequence === view.sequence - 1
        : candidate.id === view.id - 1),
  );
}

export function cameraPosition(view: SceneView): [number, number, number] {
  const { R, t } = view.pose;
  return [0, 1, 2].map((i) => -(R[i] * t[0] + R[i + 3] * t[1] + R[i + 6] * t[2])) as [
    number,
    number,
    number,
  ];
}

/**
 * The camera's axes in the world frame: right, down (image rows grow downwards) and forward.
 * The pose is world-to-camera (x_c = R x_w + t), so axis k is Rᵀ e_k, row k of R.
 */
export function cameraAxes(view: SceneView): [number, number, number][] {
  const { R } = view.pose;
  return [0, 1, 2].map((k) => [R[3 * k], R[3 * k + 1], R[3 * k + 2]]);
}

/** Optical-axis depth, not distance to world origin or camera-to-point Euclidean distance. */
export function depthPoints(scene: Scene | undefined, viewId: number): DepthPoint[] {
  const view = scene?.views.find((candidate) => candidate.id === viewId);
  if (!view || !scene) return [];
  const { R, t } = view.pose;
  const result: DepthPoint[] = [];
  for (const point of scene.landmarks) {
    const observation = point.observations.find((o) => o.viewId === viewId);
    if (!observation) continue;
    const X = point.xyz;
    const z = R[6] * X[0] + R[7] * X[1] + R[8] * X[2] + t[2];
    if (!(z > 0) || !Number.isFinite(z)) continue;
    result.push({ id: point.trackId, x: observation.pixel[0], y: observation.pixel[1], depth: z });
  }
  return result;
}

export function depthRange(points: DepthPoint[]): [number, number] {
  const sorted = points.map((point) => point.depth).sort((a, b) => a - b);
  if (!sorted.length) return [0, 1];
  return [
    sorted[Math.floor((sorted.length - 1) * 0.05)],
    sorted[Math.ceil((sorted.length - 1) * 0.95)],
  ];
}

export function depthColor(
  depth: number,
  [near, far]: readonly [number, number],
): [number, number, number] {
  const t = far > near ? Math.max(0, Math.min(1, (depth - near) / (far - near))) : 0.5;
  const stops = [
    [1, 0.2, 0.12],
    [1, 0.9, 0.15],
    [0.2, 0.9, 0.5],
    [0.1, 0.75, 1],
    [0.2, 0.25, 1],
  ];
  const index = Math.min(3, Math.floor(t * 4));
  const f = t * 4 - index;
  return stops[index].map((value, channel) => value * (1 - f) + stops[index + 1][channel] * f) as [
    number,
    number,
    number,
  ];
}

export function cameraMotion(previous: SceneView | undefined, current: SceneView | undefined) {
  if (!previous || !current) return undefined;
  const before = cameraPosition(previous),
    after = cameraPosition(current);
  const delta = after.map((v, i) => v - before[i]);
  const R = previous.pose.R;
  const local = [0, 1, 2].map(
    (i) => R[i * 3] * delta[0] + R[i * 3 + 1] * delta[1] + R[i * 3 + 2] * delta[2],
  );
  const trace = previous.pose.R.reduce((sum, value, i) => sum + value * current.pose.R[i], 0);
  return {
    distance: Math.hypot(...delta),
    local,
    degrees: (Math.acos(Math.max(-1, Math.min(1, (trace - 1) / 2))) * 180) / Math.PI,
  };
}

/**
 * A preview seek for the video element while a run is in flight. The analysis reads decoded
 * frames, not the element, so the element only shows where the run is. A 4K seek costs the main
 * thread a long task (65 ms per seek, measured 2026-10-20), so the seek is not awaited, waits
 * for the previous seek and starts at most once per `intervalMs`. Returns whether it seeked.
 */
export function previewSeeker(
  video: Pick<HTMLVideoElement, 'seeking' | 'currentTime'>,
  intervalMs = 500,
  now: () => number = () => performance.now(),
): (time: number) => boolean {
  let last = -Infinity;
  return (time) => {
    if (video.seeking || now() - last < intervalMs) return false;
    last = now();
    video.currentTime = time;
    return true;
  };
}

export async function seekFrame(
  video: HTMLVideoElement,
  time: number,
  signal: AbortSignal,
): Promise<void> {
  signal.throwIfAborted();
  if (Math.abs(video.currentTime - time) < 1e-6 && video.readyState >= 2 && !video.seeking) return;
  await new Promise<void>((resolve, reject) => {
    const cleanup = () => {
      clearTimeout(timer);
      video.removeEventListener('seeked', done);
      video.removeEventListener('error', error);
      signal.removeEventListener('abort', abort);
    };
    const done = () => {
      cleanup();
      resolve();
    };
    const error = () => {
      cleanup();
      reject(new Error('Video frame could not be decoded.'));
    };
    const abort = () => {
      cleanup();
      reject(new DOMException('Cancelled', 'AbortError'));
    };
    const timer = window.setTimeout(error, 15000);
    video.addEventListener('seeked', done, { once: true });
    video.addEventListener('error', error, { once: true });
    signal.addEventListener('abort', abort, { once: true });
    video.currentTime = time;
  });
}

/** A square ground grid in the xz plane: its center, its side and its cell size, in scene units. */
export type GridFit = { x: number; z: number; size: number; cell: number; divisions: number };

/** The smallest 1, 2 or 5 × 10^k that is at least `value`. */
export function niceStep(value: number): number {
  if (!(value > 0) || !Number.isFinite(value)) return 1;
  const power = Math.pow(10, Math.floor(Math.log10(value)));
  for (const factor of [1, 2, 5, 10])
    if (factor * power >= value * (1 - 1e-9)) return factor * power;
  return 10 * power;
}

function percentile(sorted: number[], fraction: number): number {
  const at = fraction * (sorted.length - 1);
  const low = Math.floor(at);
  const high = Math.min(sorted.length - 1, low + 1);
  return sorted[low] + (sorted[high] - sorted[low]) * (at - low);
}

/**
 * The ground grid that fits a point set: `xz` holds x, z pairs (landmarks and camera centers). The
 * extent is the 2nd to 98th percentile on each axis, so a few outliers do not stretch it, plus 10 %
 * on each side. The cell is 1, 2 or 5 × 10^k, giving 8 to 20 cells across. The center sits on a
 * multiple of the cell, so the lines fall on round coordinates. At most `limit` points are sorted.
 */
export function fitGrid(xz: ArrayLike<number>, limit = 4096): GridFit | undefined {
  const total = Math.floor(xz.length / 2);
  const stride = Math.max(1, Math.ceil(total / limit));
  const xs: number[] = [],
    zs: number[] = [];
  for (let i = 0; i < total; i += stride) {
    const x = xz[2 * i],
      z = xz[2 * i + 1];
    if (Number.isFinite(x) && Number.isFinite(z)) {
      xs.push(x);
      zs.push(z);
    }
  }
  if (!xs.length) return undefined;
  xs.sort((a, b) => a - b);
  zs.sort((a, b) => a - b);
  const x0 = percentile(xs, 0.02),
    x1 = percentile(xs, 0.98),
    z0 = percentile(zs, 0.02),
    z1 = percentile(zs, 0.98);
  const span = Math.max(x1 - x0, z1 - z0, 1e-6) * 1.2;
  const cell = niceStep(span / 20);
  const divisions = Math.max(1, Math.ceil(span / cell - 1e-9));
  return {
    x: Math.round((x0 + x1) / 2 / cell) * cell,
    z: Math.round((z0 + z1) / 2 / cell) * cell,
    size: divisions * cell,
    cell,
    divisions,
  };
}

/** Whether a new fit differs enough from the shown one to rebuild: a side or center 20 % off. */
export function gridChanged(shown: GridFit | undefined, next: GridFit): boolean {
  if (!shown) return true;
  const ratio = next.size / shown.size;
  if (ratio > 1.2 || ratio < 1 / 1.2) return true;
  return Math.hypot(next.x - shown.x, next.z - shown.z) > 0.2 * shown.size;
}

/**
 * A growing map as the worker posts it (`type: 'model', partial: true`). The page's main thread
 * deserializes every message. A map of 50,000 landmarks as objects, each with its own typed array
 * and observation list, took 100 to 300 ms to deserialize, twice a second (an owner run on
 * 2026-10-21: 164 long tasks, 20 s). The landmarks go as three flat typed arrays instead, and the
 * page unpacks only what the viewer reads.
 */
export type PackedLandmarks = {
  /** The number of landmarks. */
  count: number;
  trackIds: Uint32Array;
  /** x, y, z of each landmark. */
  xyz: Float64Array;
  /** r, g, b of each landmark as bytes. */
  rgb: Uint8Array;
  /** The observations of all landmarks. The packed form drops them. */
  observations: number;
};
export type PackedScene = { views: SceneView[]; packed: PackedLandmarks };
/** A scene unpacked from a growing map: landmarks have no observations. */
export type PartialScene = Scene & { observationCount: number };

export function packLandmarks(
  landmarks: readonly {
    trackId: number;
    xyz: ArrayLike<number>;
    color?: readonly number[];
    observations: readonly unknown[];
  }[],
): PackedLandmarks {
  const count = landmarks.length;
  const packed: PackedLandmarks = {
    count,
    trackIds: new Uint32Array(count),
    xyz: new Float64Array(count * 3),
    rgb: new Uint8Array(count * 3),
    observations: 0,
  };
  for (let i = 0; i < count; i++) {
    const point = landmarks[i]!;
    packed.trackIds[i] = point.trackId;
    for (let k = 0; k < 3; k++) {
      packed.xyz[3 * i + k] = point.xyz[k]!;
      packed.rgb[3 * i + k] = Math.round(Math.min(1, Math.max(0, point.color?.[k] ?? 0.7)) * 255);
    }
    packed.observations += point.observations.length;
  }
  return packed;
}

/** The landmarks of a packed map as `ScenePoint`s (plain arrays for the position and color). */
export function unpackScene(scene: PackedScene): PartialScene {
  const { packed } = scene;
  const landmarks: ScenePoint[] = new Array(packed.count);
  for (let i = 0; i < packed.count; i++)
    landmarks[i] = {
      trackId: packed.trackIds[i]!,
      xyz: [packed.xyz[3 * i]!, packed.xyz[3 * i + 1]!, packed.xyz[3 * i + 2]!] as never,
      color: [packed.rgb[3 * i]! / 255, packed.rgb[3 * i + 1]! / 255, packed.rgb[3 * i + 2]! / 255],
      observations: [],
    };
  return { views: scene.views, landmarks, observationCount: packed.observations };
}
