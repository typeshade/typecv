import type { FeatureTrack, ImageFeatures, LandmarkObservation } from './types.ts';

export interface PersistentDepthTrack {
  readonly id: number;
  readonly observations: readonly LandmarkObservation[];
}
const key = (o: { viewId: number; featureId: number }) => `${o.viewId}:${o.featureId}`;
const copy = (t: PersistentDepthTrack): PersistentDepthTrack => ({
  id: t.id,
  observations: t.observations.map((o) => ({ ...o, pixel: [o.pixel[0], o.pixel[1]] })),
});

/**
 * Observation-owned identities survive window-local track renumbering. Stores only keypoints,
 * not RGB/descriptors. Merges rebuild from raw observations; a split invalidates the old lineage
 * rather than assigning one depth posterior to two tracks. No old posterior is counted as data.
 */
export class DepthTrackHistory {
  private next = 0;
  private tracks = new Map<number, PersistentDepthTrack>();
  private owner = new Map<string, number>();

  snapshot(): readonly PersistentDepthTrack[] {
    return [...this.tracks.values()].sort((a, b) => a.id - b.id).map(copy);
  }

  update(
    input: readonly FeatureTrack[],
    images: readonly ImageFeatures[],
    excluded: ReadonlySet<string> = new Set(),
  ): readonly PersistentDepthTrack[] {
    const pixels = new Map(images.map((v) => [v.id, v.keypoints])),
      seen = new Set<string>(),
      rows = input.map((t) => {
        const views = new Set<number>();
        return t.observations
          .filter((o) => !excluded.has(key(o)))
          .map((o) => {
            const p = pixels.get(o.viewId),
              at = 2 * o.featureId;
            if (
              !Number.isSafeInteger(o.featureId) ||
              o.featureId < 0 ||
              !p ||
              at + 1 >= p.length ||
              !Number.isFinite(p[at]) ||
              !Number.isFinite(p[at + 1]) ||
              views.has(o.viewId) ||
              seen.has(key(o))
            )
              throw new RangeError('Invalid or multiply owned depth-track observation');
            views.add(o.viewId);
            seen.add(key(o));
            return { ...o, pixel: [p[at]!, p[at + 1]!] as const };
          });
      });
    // Validate before mutating the store. A late optical-path exclusion also revokes old evidence.
    for (const [id, t] of this.tracks) {
      const observations = t.observations.filter((o) => !excluded.has(key(o)));
      if (!observations.length) this.tracks.delete(id);
      else this.tracks.set(id, { id, observations });
    }
    this.reindex();
    const appearances = new Map<number, Set<number>>();
    rows.forEach((obs, row) =>
      obs.forEach((o) => {
        const id = this.owner.get(key(o));
        if (id !== undefined) {
          if (!appearances.has(id)) appearances.set(id, new Set());
          appearances.get(id)!.add(row);
        }
      }),
    );
    for (const [id, groups] of appearances) if (groups.size > 1) this.tracks.delete(id);
    this.reindex();
    for (const observations of rows) {
      if (!observations.length) continue;
      const parents = [
          ...new Set(
            observations.map((o) => this.owner.get(key(o))).filter((i) => i !== undefined),
          ),
        ],
        all = new Map<string, LandmarkObservation>();
      for (const id of parents)
        for (const o of this.tracks.get(id)!.observations) all.set(key(o), o);
      for (const o of observations) all.set(key(o), o);
      const views = new Set<number>();
      let conflict = false;
      for (const o of all.values()) {
        if (views.has(o.viewId)) conflict = true;
        views.add(o.viewId);
      }
      // Conflicting evicted histories are not fused into a fictitious static track.
      const id = parents.length && !conflict ? Math.min(...parents) : this.next++;
      for (const parent of parents) {
        for (const o of this.tracks.get(parent)!.observations) this.owner.delete(key(o));
        this.tracks.delete(parent);
      }
      this.tracks.set(id, {
        id,
        observations: (conflict ? [...observations] : [...all.values()]).sort(
          (a, b) => a.viewId - b.viewId || a.featureId - b.featureId,
        ),
      });
      for (const o of this.tracks.get(id)!.observations) this.owner.set(key(o), id);
    }
    return this.snapshot();
  }

  private reindex() {
    this.owner.clear();
    for (const t of this.tracks.values())
      for (const o of t.observations) this.owner.set(key(o), t.id);
  }
}
