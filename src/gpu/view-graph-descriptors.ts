import type { Resident } from 'typeshade/runtime';
import { descriptorStoreEvidence, type StoredView } from '../reconstruction/view-graph.ts';
import type { ShadeDevice } from '../device/gpu/device.ts';

/** Payload bytes reserved by this cache; temporary inputs, pools and driver storage are separate. */
export interface ViewGraphDescriptorStatistics {
  readonly hits: number;
  readonly misses: number;
  readonly uncached: number;
  readonly decodes: number;
  readonly decodedBytes: number;
  readonly applicationUploadBytes: number;
  readonly evictions: number;
  readonly entries: number;
  readonly residentBytes: number;
  readonly peakEntries: number;
  readonly peakResidentBytes: number;
  readonly limitBytes: number;
}

interface Slot {
  readonly key: object;
  readonly buffer: Resident<Float32Array>;
  readonly elements: number;
  readonly bytes: number;
  leases: number;
  used: number;
  retired: boolean;
  invalid: boolean;
}

export interface DescriptorLease {
  readonly buffer: Resident<Float32Array>;
  readonly elements: number;
  release(success: boolean): void;
}

/** Internal immutable-evidence LRU. A lease protects an input while its GPU job is queued. */
export function createViewGraphDescriptorCache(device: ShadeDevice, requestedBytes?: number) {
  if (requestedBytes !== undefined && (!Number.isSafeInteger(requestedBytes) || requestedBytes < 0))
    throw new RangeError('Descriptor cache bytes must be a nonnegative integer');
  const limitBytes = Math.min(
    requestedBytes ?? 16 * 1024 ** 2,
    Math.floor(device.workingBytes / 4),
  );
  if (!Number.isSafeInteger(limitBytes) || limitBytes < 0)
    throw new RangeError('Descriptor cache bytes must be a nonnegative integer');
  const slots = new Map<object, Slot>();
  const allocated = new Set<Slot>();
  let unregister: (() => void) | undefined;
  let clock = 0,
    hits = 0,
    misses = 0,
    uncached = 0,
    decodes = 0,
    decodedBytes = 0;
  let applicationUploadBytes = 0,
    evictions = 0,
    residentBytes = 0,
    peakResidentBytes = 0;
  let peakEntries = 0;

  const free = (slot: Slot) => {
    if (!allocated.delete(slot)) return;
    residentBytes -= slot.bytes;
    slot.invalid ? device.drop(slot.buffer) : device.recycle(slot.buffer);
    if (!allocated.size) {
      unregister?.();
      unregister = undefined;
    }
  };
  const retire = (slot: Slot, invalid = false) => {
    if (slots.get(slot.key) === slot) slots.delete(slot.key);
    slot.retired = true;
    slot.invalid ||= invalid;
    if (!slot.leases) free(slot);
  };
  const evict = (): boolean => {
    let oldest: Slot | undefined;
    for (const slot of slots.values())
      if (!slot.leases && (!oldest || slot.used < oldest.used)) oldest = slot;
    if (!oldest) return false;
    retire(oldest);
    evictions++;
    return true;
  };
  const lease = (slot: Slot): DescriptorLease => {
    slot.leases++;
    slot.used = ++clock;
    let released = false;
    return {
      buffer: slot.buffer,
      elements: slot.elements,
      release(success) {
        if (released) return;
        released = true;
        slot.leases--;
        if (!success) retire(slot, true);
        if (slot.retired && !slot.leases) free(slot);
      },
    };
  };

  return {
    acquire(view: StoredView): DescriptorLease {
      const key = limitBytes ? descriptorStoreEvidence(view) : undefined;
      const existing = key && slots.get(key);
      if (existing) {
        hits++;
        return lease(existing);
      }
      key ? misses++ : uncached++;
      const values = view.floatDescriptors();
      decodes++;
      decodedBytes += values.byteLength;
      const bytes = values.byteLength;
      let keep = key !== undefined && bytes <= limitBytes && bytes > 0;
      while (keep && (residentBytes + bytes > limitBytes || allocated.size >= 16)) {
        if (!evict()) keep = false;
      }
      const buffer = device.allocate(values.length, values);
      applicationUploadBytes += bytes;
      if (!keep) {
        if (key) uncached++;
        let released = false;
        return {
          buffer,
          elements: values.length,
          release(success) {
            if (released) return;
            released = true;
            success ? device.recycle(buffer) : device.drop(buffer);
          },
        };
      }
      const slot: Slot = {
        key: key!,
        buffer,
        elements: values.length,
        bytes,
        leases: 0,
        used: 0,
        retired: false,
        invalid: false,
      };
      slots.set(key!, slot);
      allocated.add(slot);
      residentBytes += bytes;
      peakResidentBytes = Math.max(peakResidentBytes, residentBytes);
      peakEntries = Math.max(peakEntries, allocated.size);
      try {
        unregister ??= device.registerWorkingReclaimer(evict);
      } catch (error) {
        retire(slot, true);
        throw error;
      }
      return lease(slot);
    },
    clear(invalid = false) {
      // Active leases can belong to another concurrently queued match. Retire them and release
      // only when their final consumer settles; clearing never invalidates borrowed GPU inputs.
      for (const slot of [...slots.values()]) retire(slot, invalid);
    },
    statistics(): ViewGraphDescriptorStatistics {
      return {
        hits,
        misses,
        uncached,
        decodes,
        decodedBytes,
        applicationUploadBytes,
        evictions,
        entries: allocated.size,
        residentBytes,
        peakEntries,
        peakResidentBytes,
        limitBytes,
      };
    },
  };
}
