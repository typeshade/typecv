import test from 'node:test';
import assert from 'node:assert/strict';
import { buildComputeProbe } from './compute-probe-build.mjs';
import { inBrowser } from './browser.mjs';
import { writeFile } from 'node:fs/promises';

test('WebGPU descriptor reuse keeps exact matches, graph geometry and queued input ownership', async () => {
  const file =
    process.env.TYPECV_VIEW_GRAPH_CACHE_PROBE ??
    (await buildComputeProbe('view-graph-cache-probe'));
  const { result, errors } = await inBrowser(
    async () => {
      const p = await import('/probe/view-graph-cache-probe.js');
      return {
        plain: await p.cacheWorkload({ backend: 'webgpu', bytes: 0 }),
        cached: await p.cacheWorkload({ backend: 'webgpu' }),
        first: await p.cacheGraph({ backend: 'webgpu', bytes: 0 }),
        second: await p.cacheGraph({ backend: 'webgpu' }),
        failure: await p.cacheGraph({ backend: 'webgpu', fail: true }),
        ownership: await p.cacheOwnershipAndFailure({ backend: 'webgpu' }),
      };
    },
    null,
    { timeoutMs: 90000, serve: { '/probe/view-graph-cache-probe.js': file } },
  );
  assert.deepEqual(errors, []);
  assert.deepEqual(result.cached.output, result.plain.output);
  assert.deepEqual(result.second.graph, result.first.graph);
  assert.equal(result.second.graph.edges.length, 5);
  assert.equal(result.second.graph.edges.filter((e) => e.geometry).length, 5);
  assert.equal(result.second.cache.decodes, 4);
  assert.equal(result.first.cache.decodes, 8);
  assert.ok(result.cached.beforeRelease.hits > 0);
  assert.ok(
    result.cached.beforeRelease.applicationUploadBytes <
      result.plain.beforeRelease.applicationUploadBytes,
  );
  assert.ok(
    result.cached.beforeRelease.peakResidentBytes <= result.cached.beforeRelease.limitBytes,
  );
  assert.equal(result.cached.afterRelease.residentBytes, 0);
  assert.equal(result.cached.device.liveBuffers, 0);
  assert.equal(result.second.device.liveBuffers, 0);
  assert.equal(result.failure.failure, true);
  assert.equal(result.failure.device.liveBuffers, 0);
  const own = result.ownership;
  for (const field of [
    'sameEvidence',
    'mutableExcluded',
    'changedMethodExcluded',
    'allocationFailure',
    'numericalFailure',
    'emptyFailure',
  ])
    assert.equal(own[field], true, field);
  for (const field of [
    'afterConcurrent',
    'afterAllocationFailure',
    'afterNumericalFailure',
    'afterEmptyFailure',
  ])
    assert.equal(own[field], 0, field);
  assert.notDeepEqual(own.beforeMutable, own.afterMutable);
  assert.equal(own.final.liveBuffers, 0);
  if (process.env.TYPECV_VIEW_GRAPH_CACHE_WEBGPU_RECEIPT)
    await writeFile(process.env.TYPECV_VIEW_GRAPH_CACHE_WEBGPU_RECEIPT, JSON.stringify(result));
});
