import type { ImageFeatures } from '../../contracts/scene.ts';
import type { TrackEntry } from './track-index.ts';

/** A registered view is retried after each PnP failure only while the map has grown, at most this often. */
export const MAX_REGISTRATION_ATTEMPTS = 3;

/** @internal Why a view did not register, at which map size, after how many PnP attempts. */
export interface RegistrationFailure {
  readonly reason: string;
  readonly at: number;
  readonly attempts: number;
}

/**
 * @internal The unregistered views with the points that their tracks reach, the best supported
 * first: the most 2D–3D correspondences, then the lower view id. `pointOf` gives the point of a
 * track: a landmark (`incrementalSteps`) or a point index (`sceneSteps`).
 */
export function rankViews<Point>(
  images: readonly ImageFeatures[],
  registered: (viewId: number) => boolean,
  viewTracks: ReadonlyMap<number, readonly TrackEntry[]>,
  pointOf: (trackId: number) => Point | undefined,
): { view: ImageFeatures; correspondences: { featureId: number; point: Point }[] }[] {
  return images
    .filter((v) => !registered(v.id))
    .map((view) => ({
      view,
      correspondences: viewTracks.get(view.id)!.flatMap((entry) => {
        const point = pointOf(entry.trackId);
        return point === undefined ? [] : [{ featureId: entry.featureId, point }];
      }),
    }))
    .sort((a, b) => b.correspondences.length - a.correspondences.length || a.view.id - b.view.id);
}

/**
 * @internal The failure of a view with fewer than `minPnP` 2D–3D correspondences at the map size
 * `size`. It counts no PnP attempt.
 */
export function tooFewCorrespondences(
  count: number,
  minPnP: number,
  size: number,
  previous: RegistrationFailure | undefined,
): RegistrationFailure {
  return {
    reason: `Only ${count} 2D–3D correspondences; need ${minPnP}`,
    at: size,
    attempts: previous?.attempts ?? 0,
  };
}

/**
 * @internal Whether a view that failed waits for a later attempt: it failed at this map size, or it
 * used its `MAX_REGISTRATION_ATTEMPTS`. A failed registration is retried once for each map growth.
 */
export function retryWaits(previous: RegistrationFailure | undefined, size: number): boolean {
  return !!previous && (previous.at === size || previous.attempts >= MAX_REGISTRATION_ATTEMPTS);
}

/** @internal The failure of a PnP attempt at the map size `size`: its message, and one more attempt. */
export function pnpFailure(
  error: unknown,
  size: number,
  previous: RegistrationFailure | undefined,
): RegistrationFailure {
  return {
    reason: (error as Error).message,
    at: size,
    attempts: (previous?.attempts ?? 0) + 1,
  };
}

/** @internal After a recovery pass, every view that failed may try once more. */
export function retryAfterRecovery(failure: Map<number, RegistrationFailure>): void {
  for (const [id, entry] of failure) failure.set(id, { ...entry, at: -1 });
}
