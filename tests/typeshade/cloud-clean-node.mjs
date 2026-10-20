// Node-side helpers of the cloud cleanup tests (the browser imports only the cases module).
import assert from 'node:assert/strict';
import { ShadeDevice } from '../../dist/internal/gpu.js';

/** The voxel partition as a list of member lists, independent of which member owns a voxel. */
export function partition(rep) {
  const groups = new Map();
  rep.forEach((r, i) => groups.set(r, [...(groups.get(r) ?? []), i]));
  return [...groups.values()].map((g) => g.join()).sort();
}

export async function withDevice(fn) {
  const device = await ShadeDevice.open({ backend: 'cpu', poolBytes: 0 });
  try {
    await fn(device);
    assert.equal(device.statistics.liveBytes - device.statistics.pooledBytes, 0);
  } finally {
    await device.dispose();
  }
}
