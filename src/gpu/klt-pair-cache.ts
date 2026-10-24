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
  readonly id: number;
  readonly pixels: Float32Array;
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
    peakFrames = 0,
    peakResidentBytes = 0,
    clock = 0,
    closed = false,
    disposing: Promise<void> | undefined;
  const bytes = () => [...slots.values()].reduce((n, slot) => n + slot.layout.total * 12, 0);
  const release = (slot: Slot) => {
    slots.delete(slot.id);
    device.recycle(slot.gray);
    device.recycle(slot.grad);
  };
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
        const protectedIds = new Set([from.id, to.id]);
        const scratchBytes =
          4 *
          (prepared.startData.length +
            (prepared.forwardData?.length ?? 4 * prepared.n) +
            8 * prepared.n);
        const evict = (): boolean => {
          const oldest = [...slots.values()]
            .filter((slot) => !protectedIds.has(slot.id))
            .sort((a, b) => a.used - b.used)[0];
          if (!oldest) return false;
          release(oldest);
          evictions++;
          return true;
        };
        const admit = (additionalBytes: number) => {
          const budget = device.diagnostics.memory.workingBytes;
          while (
            device.statistics.liveBytes - device.statistics.pooledBytes + additionalBytes >
            budget
          )
            if (!evict()) break; // The device reports the ordinary allocation failure below.
        };
        const get = (frame: KltCacheFrame): Slot => {
          let slot = slots.get(frame.id);
          if (
            slot &&
            (slot.pixels !== frame.gray ||
              slot.width !== frame.width ||
              slot.height !== frame.height ||
              slot.layout.levels !== prepared.layout.levels)
          ) {
            release(slot);
            slot = undefined;
          }
          if (slot) {
            hits++;
            slot.used = ++clock;
            return slot;
          }
          misses++;
          if (slots.size >= capacity) {
            if (!evict()) throw new Error('KLT cache cannot evict an active image');
          }
          const layout = prepared.layout,
            pyramidBytes = layout.total * 12;
          admit(pyramidBytes + scratchBytes);
          const gray = device.allocate(layout.total, paddedKltGray(frame.gray, layout));
          try {
            const grad = device.allocate(2 * layout.total);
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
          for (const slot of [...slots.values()]) if (!slot.ready) release(slot);
          throw error;
        }
      });
    },
    statistics() {
      return {
        hits,
        misses,
        evictions,
        pyramidBuilds,
        frames: slots.size,
        residentBytes: bytes(),
        peakFrames,
        peakResidentBytes,
      };
    },
    dispose() {
      if (disposing) return disposing;
      closed = true;
      disposing = device.enqueue(async () => {
        for (const slot of [...slots.values()]) release(slot);
      });
      return disposing;
    },
  };
}
