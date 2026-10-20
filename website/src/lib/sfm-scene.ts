export type SceneView = {
  id: number;
  time?: number;
  sequence?: number;
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
