import type {
  FeatureTrack,
  ImageFeatures,
  Landmark,
  RegisteredView,
  SparseReconstruction,
} from '../../contracts/scene.ts';
import { observationKey } from '../../contracts/keys.ts';

/** @internal One track that a view observes, through one of its features. */
export interface TrackEntry {
  readonly trackId: number;
  readonly featureId: number;
}

/**
 * @internal The per-view index of a run's track graph: the tracks that each view observes, through
 * which feature (`viewTracks`); the track of each observation, by `observationKey`
 * (`trackOfObservation`); and the pixel of a feature (`pixelOf`).
 */
export function indexTracks(
  images: readonly ImageFeatures[],
  tracks: readonly FeatureTrack[],
  byId: ReadonlyMap<number, ImageFeatures>,
): {
  viewTracks: Map<number, TrackEntry[]>;
  trackOfObservation: Map<string, number>;
  pixelOf: (viewId: number, featureId: number) => [number, number];
} {
  const viewTracks = new Map<number, TrackEntry[]>(),
    trackOfObservation = new Map<string, number>();
  for (const view of images) viewTracks.set(view.id, []);
  for (const track of tracks)
    for (const o of track.observations) {
      viewTracks.get(o.viewId)!.push({ trackId: track.id, featureId: o.featureId });
      trackOfObservation.set(observationKey(o.viewId, o.featureId), track.id);
    }
  const pixelOf = (viewId: number, featureId: number): [number, number] => {
    const k = byId.get(viewId)!.keypoints;
    return [k[2 * featureId]!, k[2 * featureId + 1]!];
  };
  return { viewTracks, trackOfObservation, pixelOf };
}

/**
 * @internal The map that a run resumes: a copy of its views, and the landmarks that it keeps, by
 * track. The track ids are rebuilt from the current edges, so a landmark is adopted through its
 * observations. It takes the track of its first observation that has one, and keeps its
 * observations of registered views on that track. A track keeps the landmark with the most such
 * observations, at least two.
 */
export function resumeMap(
  resume: SparseReconstruction,
  trackOfObservation: ReadonlyMap<string, number>,
): { views: RegisteredView[]; landmarks: Map<number, Landmark> } {
  const views = resume.views.map((v) => ({
      ...v,
      camera: { ...v.camera },
      pose: { R: v.pose.R.slice(), t: v.pose.t.slice() },
    })),
    registered = new Set(views.map((v) => v.id)),
    landmarks = new Map<number, Landmark>();
  for (const point of resume.landmarks) {
    const trackId = point.observations
      .map((o) => trackOfObservation.get(observationKey(o.viewId, o.featureId)))
      .find((id) => id !== undefined);
    if (trackId === undefined) continue;
    const observations = point.observations.filter(
      (o) =>
        registered.has(o.viewId) &&
        trackOfObservation.get(observationKey(o.viewId, o.featureId)) === trackId,
    );
    const existing = landmarks.get(trackId);
    if (
      observations.length >= 2 &&
      (!existing || existing.observations.length < observations.length)
    )
      landmarks.set(trackId, { trackId, xyz: point.xyz.slice(), observations });
  }
  return { views, landmarks };
}
