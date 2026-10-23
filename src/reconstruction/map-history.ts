import type { Landmark, RegisteredView, SparseReconstruction } from './types.ts';
import { copyReconstruction } from './mapper.ts';

/** Historical observations identify a point even when local track indices are rebuilt. */
export class MapHistory {
  private views = new Map<number, RegisteredView>();
  private points = new Map<number, Landmark>();
  private identities = new Map<string, number>();
  private nextId = 0;

  private removePoint(id: number): void {
    const point = this.points.get(id)!;
    for (const observation of point.observations) {
      const key = observation.viewId + ':' + observation.featureId;
      if (this.identities.get(key) === id) this.identities.delete(key);
    }
    this.points.delete(id);
  }

  update(model: SparseReconstruction): SparseReconstruction {
    const active = new Set(model.views.map((view) => view.id));
    const retained = new Set<number>();
    for (const view of model.views) this.views.set(view.id, view);
    for (const point of model.landmarks) {
      const keys = point.observations.map((o) => o.viewId + ':' + o.featureId);
      const existing = [
        ...new Set(
          keys.flatMap((key) => {
            const id = this.identities.get(key);
            return id === undefined ? [] : [id];
          }),
        ),
      ].filter((id) => this.points.has(id));
      const id = existing[0] ?? this.nextId++;
      const observations = new Map<string, Landmark['observations'][number]>();
      for (const previousId of existing) {
        const previous = this.points.get(previousId)!;
        // The current window owns its observations, including rejection. Only observations
        // outside it belong to the archive; a union would resurrect filtered local features.
        for (const observation of previous.observations)
          if (!active.has(observation.viewId))
            observations.set(observation.viewId + ':' + observation.featureId, observation);
        // Retire removed feature identities too: they must not join a later, separate track.
        this.removePoint(previousId);
      }
      for (const observation of point.observations)
        observations.set(observation.viewId + ':' + observation.featureId, observation);
      const merged = { ...point, trackId: id, observations: [...observations.values()] };
      this.points.set(id, merged);
      retained.add(id);
      for (const key of observations.keys()) this.identities.set(key, id);
    }
    // A point rejected within the active window must not survive as a stale historical point.
    // Points with fewer than two remaining local observations belong to the archive.
    for (const [id, point] of this.points) {
      if (!retained.has(id) && point.observations.filter((o) => active.has(o.viewId)).length >= 2)
        this.removePoint(id);
    }
    return this.snapshot();
  }

  snapshot(): SparseReconstruction {
    return copyReconstruction({
      views: [...this.views.values()],
      landmarks: [...this.points.values()],
    });
  }
}
