import test from 'node:test';
import assert from 'node:assert/strict';
import { planCameraProduct } from '../../src/gpu/bundle-camera-product.ts';

const device = {
  maxWorkgroupSize: 256,
  workgroupStorageBytes: 16384,
  bindingBytes: 2 ** 27,
  occupancyFloors: true,
};

test('camera product planning selects a supported width before sizing streamed storage', () => {
  for (const shared of [false, true]) {
    for (const width of [32, 64, 128, 256]) {
      const plan = planCameraProduct(device, [3, 133, 2051], shared, width);
      assert.equal(plan.workgroupSize, width);
      assert.equal(plan.laneElements, (shared ? 7 : 6) * width * 3);
      assert.equal(plan.tasks.length, 0);
    }
    assert.equal(planCameraProduct(device, [20, 101, 120], shared, 'auto').workgroupSize, 32);
    assert.equal(planCameraProduct(device, [20, 101, 1500], shared, 'auto').workgroupSize, 256);
    assert.equal(planCameraProduct(device, [1500], shared, 'auto', 100).workgroupSize, 32);
  }
  assert.equal(
    planCameraProduct({ ...device, occupancyFloors: false }, [2000], true, 'auto').workgroupSize,
    32,
  );
  const constrained = { ...device, maxWorkgroupSize: 128, workgroupStorageBytes: 4000 };
  assert.equal(planCameraProduct(constrained, [2000], true, 'auto').workgroupSize, 64);
  assert.throws(() => planCameraProduct(constrained, [2000], true, 128), /workgroup limits/);
  assert.throws(
    () => planCameraProduct({ ...device, maxWorkgroupSize: 16 }, [20], false, 'auto'),
    /workgroup limits/,
  );
});

test('camera task descriptors partition every observation once, including empty cameras and tails', () => {
  const counts = [0, 1, 137, 138, 1031];
  const plan = planCameraProduct(device, counts, true, 32, 137);
  assert.equal(plan.taskCount, 13);
  assert.equal(plan.partialElements, 7 * 13);
  assert.equal(plan.laneElements, 7 * 32 * 13);
  let cursor = 0;
  for (let c = 0; c < counts.length; c++) {
    let at = cursor;
    for (let task = plan.tasks[c]; task < plan.tasks[c + 1]; task++) {
      const offset = counts.length + 1 + task * 3;
      const [camera, first, end] = plan.tasks.slice(offset, offset + 3);
      assert.equal(camera, c);
      assert.equal(first, at);
      assert.ok(end >= first && end - first <= 137);
      at = end;
    }
    cursor += counts[c];
    assert.equal(at, cursor);
  }
  assert.throws(
    () => planCameraProduct({ ...device, bindingBytes: 64 }, counts, false, 32, 137),
    /binding limit/,
  );
  assert.throws(() => planCameraProduct(device, [2 ** 32], false, 256, 1), /integer index/);
});
