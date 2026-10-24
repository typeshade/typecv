import type { TsdfView } from '../reconstruction/tsdf.ts';

/** @internal Potential readers include self and every configured source, even grouped cameras. */
export function planarLifetimePlan(sources: readonly (readonly number[])[]) {
  const n = sources.length;
  const dependencies = sources.map((indices, reference) => {
    if (indices.some((index) => !Number.isSafeInteger(index) || index < 0 || index >= n))
      throw new RangeError('Invalid planar lifetime source');
    return [...new Set([reference, ...indices])];
  });
  const readers = new Int32Array(n);
  for (const indices of dependencies) for (const index of indices) readers[index]!++;
  const remaining = readers.slice();
  const processed = new Uint8Array(n);
  const order: number[] = [];
  let extra = 0,
    peakExtraViews = 0;
  for (let step = 0; step < n; step++) {
    let best = -1,
      bestExtra = Infinity;
    for (let reference = 0; reference < n; reference++) {
      if (processed[reference]) continue;
      let next = extra + (remaining[reference]! > 1 ? 1 : 0);
      for (const index of dependencies[reference]!)
        if (processed[index] && remaining[index] === 1) next--;
      // Input index resolves ties, making the ownership schedule independent of timing.
      if (next < bestExtra) {
        best = reference;
        bestExtra = next;
      }
    }
    peakExtraViews = Math.max(peakExtraViews, extra + 1);
    order.push(best);
    processed[best] = 1;
    for (const index of dependencies[best]!) remaining[index]!--;
    extra = bestExtra;
  }
  return { dependencies, readers, order, peakExtraViews };
}

function mapBuffers(view: TsdfView): Set<ArrayBufferLike> {
  return new Set([
    view.depth.buffer,
    view.normal.buffer,
    ...(view.confidence ? [view.confidence.buffer] : []),
  ]);
}

/** Unique backing storage reachable from these map roots, including subview backing bytes. */
class MapBuffers {
  private readonly references = new Map<ArrayBufferLike, number>();
  bytes = 0;

  add(view: TsdfView): void {
    for (const buffer of mapBuffers(view)) {
      const previous = this.references.get(buffer) ?? 0;
      if (!previous) this.bytes += buffer.byteLength;
      this.references.set(buffer, previous + 1);
    }
  }

  remove(view: TsdfView): void {
    for (const buffer of mapBuffers(view)) {
      const previous = this.references.get(buffer);
      if (!previous) throw new Error('Planar map ownership underflow');
      if (previous === 1) {
        this.references.delete(buffer);
        this.bytes -= buffer.byteLength;
      } else this.references.set(buffer, previous - 1);
    }
  }

  has(buffer: ArrayBufferLike): boolean {
    return this.references.has(buffer);
  }
}

export interface PlanarHostMaps {
  /** Unique original map backing bytes; excludes RGBA, GPU and external caller ownership. */
  readonly inputBytes: number;
  /** Peak unique map backing bytes under these roots; not browser RSS or GPU readback scratch. */
  readonly peakRetainedBytes: number;
  /** Processed originals retaining backing storage not already held by completed output maps. */
  readonly peakExtraOriginalViews: number;
  readonly retiredOriginalViews: number;
  readonly reorderedViews: number;
}

/**
 * @internal Consumes only an exclusively owned array of view slots, never the source bytes.
 * Originals remain frozen until their self and potential source readers have all finished.
 * Failure may leave mixed private slots; no caller may publish them as a completed result.
 */
export class OwnedPlanarMaps {
  readonly plan: ReturnType<typeof planarLifetimePlan>;
  private readonly views: TsdfView[];
  private readonly originals: (TsdfView | undefined)[];
  private readonly remaining: Int32Array;
  private readonly processed: Uint8Array;
  private readonly roots = new MapBuffers();
  private readonly completedRoots = new MapBuffers();
  private readonly inputBytes: number;
  private peakRetainedBytes: number;
  private peakExtraOriginalViews = 0;
  private retiredOriginalViews = 0;

  constructor(views: TsdfView[], sources: readonly (readonly number[])[]) {
    if (sources.length !== views.length) throw new RangeError('Invalid owned planar view count');
    this.views = views;
    this.originals = views.slice();
    this.plan = planarLifetimePlan(sources);
    this.remaining = this.plan.readers.slice();
    this.processed = new Uint8Array(views.length);
    for (const view of views) this.roots.add(view);
    this.inputBytes = this.roots.bytes;
    this.peakRetainedBytes = this.inputBytes;
  }

  original(index: number): TsdfView {
    const view = this.originals[index];
    if (!view) throw new Error('Planar evidence was retired before its last reader');
    return view;
  }

  complete(reference: number, completed: TsdfView): void {
    if (this.processed[reference]) throw new Error('Planar view completed twice');
    this.original(reference);
    // Observe the allocation overlap before removing any old roots, including the current view.
    this.roots.add(completed);
    this.completedRoots.add(completed);
    this.peakRetainedBytes = Math.max(this.peakRetainedBytes, this.roots.bytes);
    this.views[reference] = completed;
    this.processed[reference] = 1;
    let extra = 0;
    for (let index = 0; index < this.originals.length; index++) {
      const view = this.originals[index];
      if (
        this.processed[index] &&
        view &&
        [...mapBuffers(view)].some((buffer) => !this.completedRoots.has(buffer))
      )
        extra++;
    }
    this.peakExtraOriginalViews = Math.max(this.peakExtraOriginalViews, extra);
    for (const index of this.plan.dependencies[reference]!) {
      if (--this.remaining[index]! !== 0) continue;
      this.roots.remove(this.original(index));
      this.originals[index] = undefined;
      this.retiredOriginalViews++;
    }
  }

  finish(): PlanarHostMaps {
    if (this.retiredOriginalViews !== this.views.length)
      throw new Error('Planar completion did not consume every original reader');
    return {
      inputBytes: this.inputBytes,
      peakRetainedBytes: this.peakRetainedBytes,
      peakExtraOriginalViews: this.peakExtraOriginalViews,
      retiredOriginalViews: this.retiredOriginalViews,
      reorderedViews: this.plan.order.reduce(
        (count, view, index) => count + (view !== index ? 1 : 0),
        0,
      ),
    };
  }
}
