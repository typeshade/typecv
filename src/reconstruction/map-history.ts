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
      const observations = new Map<number, Landmark['observations'][number]>();
      for (const previousId of existing) {
        const previous = this.points.get(previousId)!;
        // The current window owns its observations, including rejection. Only observations
        // outside it belong to the archive; a union would resurrect filtered local features.
        const archived = previous.observations.filter((o) => !active.has(o.viewId));
        // A new local track can touch two older tracks that disagree in an archived view.
        // They cannot be one finite point: preserve the conflicting archive as a separate
        // point instead of publishing two different features of one camera in a BA track.
        const conflict = archived.some((o) => {
          const known = observations.get(o.viewId);
          return known && known.featureId !== o.featureId;
        });
        // Retire removed feature identities too: they must not join a later, separate track.
        this.removePoint(previousId);
        if (conflict) {
          if (archived.length >= 2) {
            const separate = { ...previous, observations: archived };
            this.points.set(previousId, separate);
            retained.add(previousId);
            for (const o of archived) this.identities.set(o.viewId + ':' + o.featureId, previousId);
          }
        } else
          for (const observation of archived) observations.set(observation.viewId, observation);
      }
      for (const observation of point.observations)
        observations.set(observation.viewId, observation);
      const merged = { ...point, trackId: id, observations: [...observations.values()] };
      this.points.set(id, merged);
      retained.add(id);
      for (const o of observations.values()) this.identities.set(o.viewId + ':' + o.featureId, id);
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

  /** Replace geometry after a full-observation solve, preserving its stable track identities. */
  replace(model: SparseReconstruction): void {
    const snapshot = copyReconstruction(model);
    this.views = new Map(snapshot.views.map((v) => [v.id, v]));
    this.points = new Map(snapshot.landmarks.map((p) => [p.trackId, p]));
    this.identities.clear();
    for (const point of snapshot.landmarks) {
      this.nextId = Math.max(this.nextId, point.trackId + 1);
      for (const o of point.observations)
        this.identities.set(o.viewId + ':' + o.featureId, point.trackId);
    }
  }
}
