import test from 'node:test';
import assert from 'node:assert/strict';
import { pathToFileURL } from 'node:url';
import { buildComputeProbe } from './compute-probe-build.mjs';
import { buildViewGraph } from '../../src/reconstruction/view-graph.ts';
import { writeFile } from 'node:fs/promises';

test('immutable descriptor residency preserves pairs and graph geometry with bounded payloads', async () => {
  const file =
    process.env.TYPECV_VIEW_GRAPH_CACHE_PROBE ??
    (await buildComputeProbe('view-graph-cache-probe'));
  const probe = await import(pathToFileURL(file).href);
  const plain = await probe.cacheWorkload({ bytes: 0 });
  const cached = await probe.cacheWorkload();
  assert.deepEqual(cached.output, plain.output);
  assert.equal(plain.beforeRelease.decodes, 15);
  assert.ok(cached.beforeRelease.hits > 0);
  assert.ok(cached.beforeRelease.decodes < plain.beforeRelease.decodes);
  assert.ok(
    cached.beforeRelease.applicationUploadBytes < plain.beforeRelease.applicationUploadBytes,
  );
  assert.ok(cached.beforeRelease.peakResidentBytes <= cached.beforeRelease.limitBytes);
  assert.ok(cached.beforeRelease.peakEntries <= 16);
  assert.equal(cached.afterRelease.residentBytes, 0);
  assert.equal(cached.device.liveBuffers, 0);
  const first = await probe.cacheGraph({ bytes: 0 });
  const second = await probe.cacheGraph();
  assert.deepEqual(second.graph, first.graph);
  assert.equal(second.graph.edges.length, 5);
  assert.equal(second.graph.edges.filter((e) => e.geometry).length, 5);
  assert.equal(second.cache.decodes, 4);
  assert.equal(first.cache.decodes, 8);
  assert.equal(second.cache.residentBytes, 0);
  assert.equal(second.device.liveBuffers, 0);
  const failure = await probe.cacheGraph({ fail: true });
  assert.equal(failure.failure, true);
  assert.equal(failure.cache.residentBytes, 0);
  assert.equal(failure.device.liveBuffers, 0);
  const ownership = await probe.cacheOwnershipAndFailure();
  for (const field of [
    'sameEvidence',
    'mutableExcluded',
    'changedMethodExcluded',
    'allocationFailure',
    'numericalFailure',
    'emptyFailure',
  ])
    assert.equal(ownership[field], true, field);
  for (const field of [
    'afterConcurrent',
    'afterAllocationFailure',
    'afterNumericalFailure',
    'afterEmptyFailure',
  ])
    assert.equal(ownership[field], 0, field);
  assert.notDeepEqual(ownership.beforeMutable, ownership.afterMutable);
  assert.equal(ownership.final.liveBuffers, 0);
  const pressure = await probe.cacheWorkingPressure();
  assert.ok(pressure.after.evictions > pressure.before.evictions);
  assert.equal(pressure.protectedFailure, true);
  assert.ok(pressure.protectedBytes > 0);
  assert.equal(pressure.final.residentBytes, 0);
  assert.equal(pressure.device.liveBuffers, 0);
  if (process.env.TYPECV_VIEW_GRAPH_CACHE_CPU_RECEIPT)
    await writeFile(
      process.env.TYPECV_VIEW_GRAPH_CACHE_CPU_RECEIPT,
      JSON.stringify({ plain, cached, first, second, failure, ownership, pressure }),
    );
});

test('graph matching cleanup also runs for empty graphs and validation failures', async () => {
  let released = 0;
  const backend = {
    match: async () => [],
    execute: () => undefined,
    releaseMatching() {
      released++;
    },
  };
  await assert.rejects(buildViewGraph([], backend, { verifyBatch: 0 }), RangeError);
  assert.equal(released, 1);
  await buildViewGraph([], backend);
  assert.equal(released, 2);
  const sentinel = new Error('candidate progress failure');
  await assert.rejects(
    buildViewGraph([], backend, {
      onProgress() {
        throw sentinel;
      },
    }),
    (e) => e === sentinel,
  );
  assert.equal(released, 3);
});
