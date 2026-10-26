import type { ShadeDevice } from '../device/gpu/device.ts';

export type CameraProductWorkgroupSize = 32 | 64 | 128 | 256 | 'auto';

/** Select once, before residency admission. A streamed sweep must retain its lane ownership.
 * Software/disabled occupancy floors use 32 lanes, favored by the software measurements.
 * Native auto targets four observations per lane at the 95th percentile; it remains an
 * opt-in structural heuristic, not a measured native-GPU performance threshold. */
export function planCameraProduct(
  device: Pick<
    ShadeDevice,
    'maxWorkgroupSize' | 'workgroupStorageBytes' | 'bindingBytes' | 'occupancyFloors'
  >,
  counts: readonly number[],
  shared: boolean,
  requested: CameraProductWorkgroupSize = 64,
  tileSize?: number,
) {
  const rows = shared ? 7 : 6;
  const supported = [32, 64, 128, 256].filter(
    (n) => n <= device.maxWorkgroupSize && rows * n * 8 <= device.workgroupStorageBytes,
  );
  const sorted = [...counts].sort((a, b) => a - b);
  const p95 = sorted[Math.max(0, Math.ceil(sorted.length * 0.95) - 1)] ?? 0;
  const workgroupSize =
    requested === 'auto' && !device.occupancyFloors
      ? supported[0]
      : requested === 'auto'
        ? (supported.find((n) => n * 4 >= Math.min(p95, tileSize ?? Infinity)) ?? supported.at(-1))
        : requested;
  if (workgroupSize === undefined || !supported.includes(workgroupSize))
    throw new RangeError('cameraProductWorkgroupSize exceeds device workgroup limits');
  const taskCount =
    tileSize === undefined
      ? counts.length
      : counts.reduce((n, count) => n + Math.max(1, Math.ceil(count / tileSize)), 0);
  const laneElements = rows * workgroupSize * taskCount;
  const taskWords = tileSize === undefined ? 0 : counts.length + 1 + 3 * taskCount;
  if (![laneElements, taskWords].every((n) => Number.isSafeInteger(n) && n <= 0xffffffff))
    throw new RangeError('Camera product scratch exceeds integer index bounds');
  if (taskWords * 4 > device.bindingBytes || rows * taskCount * 8 > device.bindingBytes)
    throw new RangeError('Camera product segments exceed the device binding limit');
  const tasks = new Uint32Array(taskWords);
  if (tileSize !== undefined) {
    let task = 0,
      cursor = 0;
    for (let c = 0; c < counts.length; c++) {
      tasks[c] = task;
      const count = counts[c]!;
      for (let at = 0; at < Math.max(1, count); at += tileSize) {
        tasks.set(
          [c, cursor + at, cursor + Math.min(count, at + tileSize)],
          counts.length + 1 + 3 * task,
        );
        task++;
      }
      cursor += count;
    }
    tasks[counts.length] = task;
  }
  return {
    workgroupSize,
    tileSize,
    laneElements,
    taskCount,
    tasks,
    partialElements: rows * taskCount,
  };
}
