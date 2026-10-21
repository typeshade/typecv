/**
 * The live view of a run (`docs/lab-live-view.md`): what changed between two maps, which cameras
 * the work of a stage touches, and how fresh the worker's last message is. The functions hold no
 * state of the page or of three.js, so a unit test reads them without a browser.
 */

import type { Scene, ScenePoint } from './sfm-scene';

/** A landmark's place, keyed by its track id. */
export type PositionIndex = Map<number, readonly [number, number, number]>;

/** What happened to one landmark of the new map. */
export const Change = {
  /** Not drawn: a coordinate is not finite. */
  Skipped: -1,
  /** The same place as in the previous map. */
  Kept: 0,
  /** A track the previous map did not have. */
  Added: 1,
  /** A track of the previous map at a new place (bundle adjustment moved it). */
  Moved: 2,
} as const;

export interface LandmarkChanges {
  /** One entry per landmark of the new map, in its order. */
  readonly changes: Int8Array;
  /** The places of the tracks that the new map dropped. */
  readonly removed: Float32Array;
  readonly added: number;
  readonly moved: number;
  /** The index of the new map, for the next comparison. */
  readonly index: PositionIndex;
}

const finite = (point: ScenePoint) =>
  Number.isFinite(point.xyz[0]) && Number.isFinite(point.xyz[1]) && Number.isFinite(point.xyz[2]);

/**
 * Compare the landmarks of a new map with the previous one. A move shorter than `epsilon` (in
 * map units) counts as kept, so float noise does not animate a point.
 */
export function landmarkChanges(
  previous: PositionIndex,
  landmarks: readonly ScenePoint[],
  epsilon: number,
): LandmarkChanges {
  const changes = new Int8Array(landmarks.length);
  const index: PositionIndex = new Map();
  let added = 0,
    moved = 0;
  const limit = epsilon * epsilon;
  for (let i = 0; i < landmarks.length; i++) {
    const point = landmarks[i]!;
    if (!finite(point)) {
      changes[i] = Change.Skipped;
      continue;
    }
    const at = [point.xyz[0], point.xyz[1], point.xyz[2]] as const;
    index.set(point.trackId, at);
    const before = previous.get(point.trackId);
    if (!before) {
      changes[i] = Change.Added;
      added++;
      continue;
    }
    const dx = at[0] - before[0],
      dy = at[1] - before[1],
      dz = at[2] - before[2];
    if (dx * dx + dy * dy + dz * dz > limit) {
      changes[i] = Change.Moved;
      moved++;
    } else changes[i] = Change.Kept;
  }
  const gone: number[] = [];
  for (const [trackId, at] of previous) if (!index.has(trackId)) gone.push(at[0], at[1], at[2]);
  return { changes, removed: Float32Array.from(gone), added, moved, index };
}

/** The ids of the views of `next` that `previous` did not hold. */
export function addedViews(previous: ReadonlySet<number>, next: Scene): Set<number> {
  const result = new Set<number>();
  for (const view of next.views) if (!previous.has(view.id)) result.add(view.id);
  return result;
}

/**
 * The work of a stage, as the viewer shows it. `all`: the stage refines every camera of the
 * active map (bundle adjustment). `latest`: the stage works next to the newest camera (matching,
 * verification, pose, triangulation). `none`: the stage has nothing in the map yet.
 */
export type PulseTarget = 'all' | 'latest' | 'none';

/** The target of a progress detail of the worker, such as `bundle · kernel (start)`. */
export function pulseTarget(detail: string): PulseTarget {
  const stage = detail.split(/[ #·]/)[0] ?? '';
  if (stage === 'bundle' || stage === 'final' || stage === 'reconstruction') return 'all';
  if (/^(matching|verifyPair|pnp|triangulation|observations)$/.test(stage)) return 'latest';
  return 'none';
}

/** A pulse runs while the worker's last message is younger than this. */
export const PULSE_HOLD_MS = 1500;

/** The worker counts as quiet after this long without a message; the badge then says so. */
export const QUIET_MS = 5000;

/**
 * The freshness badge: how long ago the worker last reported, in whole seconds. While it is
 * recent the badge says when the last update came; past `QUIET_MS` it says how long the GPU has
 * worked without one, which is still work, not a stall.
 */
export function freshness(
  lastEventAt: number,
  now: number,
  ko: boolean,
): { text: string; quiet: boolean } {
  const seconds = Math.max(0, Math.floor((now - lastEventAt) / 1000));
  const quiet = now - lastEventAt >= QUIET_MS;
  if (quiet)
    return {
      quiet,
      text: ko
        ? `GPU 작업 중 · ${seconds}초째 새 소식 없음`
        : `GPU working · no update for ${seconds} s`,
    };
  return {
    quiet,
    text: ko ? `작업 중 · ${seconds}초 전 갱신` : `Working · updated ${seconds} s ago`,
  };
}

/** The 2D–3D links of one view: what the mapper accepted for it (`docs/lab-live-view.md` 3.5). */
export interface ViewLinks {
  /** The linked features' pixels, x and y, in analysis pixels. */
  readonly pixels: Float32Array;
  /** The 3D points, x y z in map units, in the order of `pixels`. */
  readonly points: Float32Array;
  /** Each point's depth along the view's optical axis. */
  readonly depths: Float32Array;
  /** Each point's track id. */
  readonly trackIds: Uint32Array;
}

/**
 * Collect the observations of `viewId` across the landmarks of a scene. A landmark that is not
 * finite, or that lies behind the camera, is left out, as the frame card leaves it out.
 */
export function viewLinks(scene: Scene, viewId: number, only?: ReadonlySet<number>): ViewLinks {
  const view = scene.views.find((candidate) => candidate.id === viewId);
  const pixels: number[] = [],
    points: number[] = [],
    depths: number[] = [],
    trackIds: number[] = [];
  if (view) {
    const { R, t } = view.pose;
    for (const landmark of scene.landmarks) {
      if (!finite(landmark)) continue;
      const observation = landmark.observations.find((o) => o.viewId === viewId);
      if (!observation) continue;
      const [x, y, z] = landmark.xyz;
      const depth = R[6]! * x! + R[7]! * y! + R[8]! * z! + t[2]!;
      if (!(depth > 0)) continue;
      pixels.push(observation.pixel[0], observation.pixel[1]);
      points.push(x!, y!, z!);
      depths.push(depth);
      trackIds.push(landmark.trackId);
    }
  }
  return {
    pixels: Float32Array.from(pixels),
    points: Float32Array.from(points),
    depths: Float32Array.from(depths),
    trackIds: Uint32Array.from(trackIds),
  };
}

/**
 * Where a new point's flight starts (3.6): its observation in the first of `order` that observes
 * it, as a view id and a pixel. `order` lists the views to try, the newest first.
 */
export function flightOrigin(
  point: ScenePoint,
  order: readonly number[],
): { viewId: number; pixel: readonly [number, number] } | undefined {
  for (const viewId of order) {
    const observation = point.observations.find((o) => o.viewId === viewId);
    if (observation) return { viewId, pixel: observation.pixel };
  }
  return undefined;
}
