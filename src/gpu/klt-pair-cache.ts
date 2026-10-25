import type { ShadeDevice } from './device.ts';
import {
  paddedKltGray,
  prepareKltGpuPair,
  recordKltPyramid,
  trackKltResidentPair,
  type KltGpuOptions,
  type KltGpuResult,
  type KltResidentPyramid,
} from './klt-device.ts';
import type { KltOptions } from './klt.ts';

/** Original image evidence must stay immutable while the cache owns its resident copy. */
export interface KltCacheFrame {
  readonly id: number;
  readonly width: number;
  readonly height: number;
  readonly gray: Float32Array;
}
export interface KltPairCacheOptions {
  /** LRU capacity, default three frames. Explicit bounds are 2..8. */
  readonly capacity?: number;
}
export interface KltPairCacheStatistics {
  readonly hits: number;
  readonly misses: number;
  readonly evictions: number;
  readonly pyramidBuilds: number;
  /** Complete gray/gradient pairs allocated, including pairs later evicted or dropped. */
  readonly pyramidAllocations: number;
  /** Gray and gradient resident handles retained for compatible replacement (two per pyramid). */
  readonly bufferReuses: number;
  readonly frames: number;
  /** Gray and gradient storage only; point scratch, pooling and driver memory are excluded. */
  readonly residentBytes: number;
  readonly peakFrames: number;
  readonly peakResidentBytes: number;
}
export interface KltPairCache {
  track(
    from: KltCacheFrame,
    to: KltCacheFrame,
    points: ArrayLike<number>,
    options: KltOptions,
    gpu?: KltGpuOptions,
  ): Promise<KltGpuResult>;
  statistics(): KltPairCacheStatistics;
  /** Serializes after queued tracking calls and releases every retained frame. Idempotent. */
  dispose(): Promise<void>;
}
interface Slot extends KltResidentPyramid {
  id: number;
  pixels: Float32Array;
  ready: boolean;
  used: number;
}

/**
 * Bounded resident image evidence for ordered frame pairs. It runs the same KLT kernels as
 * trackKltGpu; the cache changes buffer lifetime and image uploads, never the measurements.
 */
export function createKltPairCache(
  device: ShadeDevice,
  options: KltPairCacheOptions = {},
): KltPairCache {
  const capacity = options.capacity ?? 3;
  if (!Number.isSafeInteger(capacity) || capacity < 2 || capacity > 8)
    throw new RangeError('KLT pair cache capacity must be an integer from 2 to 8');
  const slots = new Map<number, Slot>();
  let hits = 0,
    misses = 0,
    evictions = 0,
    pyramidBuilds = 0,
    pyramidAllocations = 0,
    bufferReuses = 0,
    peakFrames = 0,
    peakResidentBytes = 0,
    clock = 0,
    closed = false,
    disposing: Promise<void> | undefined;
  let protectedIds = new Set<number>();
  const bytes = () => [...slots.values()].reduce((n, slot) => n + slot.layout.total * 12, 0);
  const release = (slot: Slot, invalid = false) => {
    slots.delete(slot.id);
    if (invalid) {
      device.drop(slot.gray);
      device.drop(slot.grad);
    } else {
      device.recycle(slot.gray);
      device.recycle(slot.grad);
    }
  };
  const oldest = () =>
    [...slots.values()]
      .filter((slot) => !protectedIds.has(slot.id))
      .sort((a, b) => a.used - b.used)[0];
  const evict = (): boolean => {
    const victim = oldest();
    if (!victim) return false;
    release(victim);
    evictions++;
    return true;
  };
  // Outside a tracking job all frames are merely a cache, not an allocation lease
  // allowed to starve another job. The active pair remains protected across awaits.
  const unregister = device.registerWorkingReclaimer(evict);
  return {
    track(from, to, points, tracking, gpu = {}) {
      if (closed) return Promise.reject(new Error('KLT pair cache is disposed'));
      for (const frame of [from, to])
        if (
          !Number.isSafeInteger(frame.id) ||
          frame.id < 0 ||
          frame.gray.length !== frame.width * frame.height
        )
          return Promise.reject(new RangeError('Invalid KLT cache frame'));
      if (from.width !== to.width || from.height !== to.height)
        return Promise.reject(new RangeError('KLT cache pair must share image coordinates'));
      if (from.id === to.id && from.gray !== to.gray)
        return Promise.reject(
          new RangeError('Different image evidence cannot share a KLT cache frame id'),
        );
      const prepared = prepareKltGpuPair(from.width, from.height, points, tracking, gpu),
        controls = { ...gpu };
      return device.enqueue(async () => {
        protectedIds = new Set([from.id, to.id]);
        const { scratchBytes } = prepared;
        const admit = (additionalBytes: number) => {
          const budget = device.diagnostics.memory.workingBytes;
          while (
            device.statistics.liveBytes - device.statistics.pooledBytes + additionalBytes >
            budget
          )
            if (!evict()) break; // The device reports the ordinary allocation failure below.
          device.reserveWorkingBytes(additionalBytes);
        };
        const get = (frame: KltCacheFrame): Slot => {
          let slot = slots.get(frame.id);
          const compatible = (candidate: Slot) =>
              candidate.width === frame.width &&
              candidate.height === frame.height &&
              candidate.layout.levels === prepared.layout.levels,
            reuse = (candidate: Slot): Slot => {
              slots.delete(candidate.id);
              candidate.id = frame.id;
              candidate.pixels = frame.gray;
              candidate.used = ++clock;
              candidate.ready = false;
              slots.set(candidate.id, candidate);
              device.writeF32(candidate.gray, paddedKltGray(frame.gray, prepared.layout));
              bufferReuses += 2;
              return candidate;
            };
          if (
            slot &&
            (slot.pixels !== frame.gray ||
              slot.width !== frame.width ||
              slot.height !== frame.height ||
              slot.layout.levels !== prepared.layout.levels)
          ) {
            if (compatible(slot)) {
              misses++;
              return reuse(slot);
            }
            release(slot);
            slot = undefined;
          }
          if (slot) {
            hits++;
            slot.used = ++clock;
            return slot;
          }
          misses++;
          const layout = prepared.layout,
            pyramidBytes = layout.total * 12,
            needRoom =
              slots.size >= capacity ||
              device.statistics.liveBytes -
                device.statistics.pooledBytes +
                pyramidBytes +
                scratchBytes >
                device.workingBytes;
          if (needRoom) {
            const victim = oldest();
            if (victim && compatible(victim)) {
              evictions++;
              return reuse(victim);
            }
            if (slots.size >= capacity && !evict())
              throw new Error('KLT cache cannot evict an active image');
          }
          admit(pyramidBytes + scratchBytes);
          const gray = device.allocate(layout.total, paddedKltGray(frame.gray, layout));
          try {
            // The pinned TypeShade ZeroedState.view() materializes a fresh zero array
            // for host validation on every entry call while its host value is absent.
            // A persistent gradient would repeat that full-size allocation on every
            // cache hit. Retain one host mirror instead; Scharr still overwrites the
            // complete gradient. Its initial zero upload is counted explicitly.
            const grad = device.allocate(2 * layout.total, new Float32Array(2 * layout.total));
            slot = {
              id: frame.id,
              width: frame.width,
              height: frame.height,
              pixels: frame.gray,
              layout,
              gray,
              grad,
              used: ++clock,
              ready: false,
            };
          } catch (error) {
            device.recycle(gray);
            throw error;
          }
          slots.set(frame.id, slot);
          pyramidAllocations++;
          peakFrames = Math.max(peakFrames, slots.size);
          peakResidentBytes = Math.max(peakResidentBytes, bytes());
          return slot;
        };
        try {
          const a = get(from),
            b = get(to),
            build = [...new Set([a, b])].filter((slot) => !slot.ready);
          if (build.length) {
            await device.planSteps([
              [
                'kltPyramid',
                () => {
                  for (const slot of build) recordKltPyramid(slot.layout, slot.gray, slot.grad);
                },
              ],
            ]);
            for (const slot of build) {
              slot.ready = true;
              pyramidBuilds++;
            }
          }
          admit(scratchBytes);
          return await trackKltResidentPair(device, a, b, prepared, controls);
        } catch (error) {
          // Partly initialized images never survive a failed allocation or pyramid dispatch.
          for (const slot of [...slots.values()]) if (!slot.ready) release(slot, true);
          throw error;
        } finally {
          protectedIds.clear();
        }
      });
    },
    statistics() {
      return {
        hits,
        misses,
        evictions,
        pyramidBuilds,
        pyramidAllocations,
        bufferReuses,
        frames: slots.size,
        residentBytes: bytes(),
        peakFrames,
        peakResidentBytes,
      };
    },
    dispose() {
      if (disposing) return disposing;
      closed = true;
      disposing = device
        .enqueue(async () => {
          for (const slot of [...slots.values()]) release(slot);
        })
        .finally(unregister);
      return disposing;
    },
  };
}
