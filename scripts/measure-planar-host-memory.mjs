#!/usr/bin/env node
/**
 * Lifetime-only simulation: recorded source dependencies, generated map values, no geometry.
 * Run each variant in a fresh process, for example:
 * node --expose-gc scripts/measure-planar-host-memory.mjs owned depths-c000.json 256 256
 * node --expose-gc scripts/measure-planar-host-memory.mjs immutable depths-c000.json 256 256
 * Output is aggregate metadata only. Source poses, view IDs and graph edges are never printed.
 */
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { setImmediate } from 'node:timers/promises';
import { performance } from 'node:perf_hooks';
import { OwnedPlanarMaps, planarLifetimePlan } from '../src/gpu/planar-depth-lifetime.ts';

const [mode, filename, w = '256', h = '256'] = process.argv.slice(2);
const width = Number(w),
  height = Number(h);
if (
  !['owned', 'immutable'].includes(mode) ||
  !filename ||
  !Number.isSafeInteger(width) ||
  !Number.isSafeInteger(height) ||
  width < 1 ||
  height < 1
)
  throw new Error(
    'Usage: node --expose-gc scripts/measure-planar-host-memory.mjs owned|immutable depth-manifest.json [width height]',
  );
const sourceBytes = await readFile(filename);
const manifest = JSON.parse(sourceBytes.toString('utf8'));
const maps = manifest.maps.filter((map) => map.stage === 'photometric');
const indexOf = new Map(maps.map((map, index) => [map.viewId, index]));
if (!maps.length || indexOf.size !== maps.length)
  throw new Error('Expected complete unique photometric metadata');
const sources = maps.map((map) =>
  map.sourceViewIds.map((id) => {
    const index = indexOf.get(id);
    if (index === undefined)
      throw new Error('Manifest is missing a source view; refusing an incomplete dependency graph');
    return index;
  }),
);
const count = maps.length,
  pixels = width * height;
const sourcePixels = maps[0].width * maps[0].height;
if (maps.some((map) => map.width * map.height !== sourcePixels))
  throw new Error('Mixed map shapes');
const plan = planarLifetimePlan(sources);
const rgba = new Uint32Array(1);
const input = Array.from({ length: count }, (_, index) => ({
  rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1],
  center: [index, 0, 0],
  fx: 32,
  fy: 32,
  cx: 0,
  cy: 0,
  depth: new Float32Array(pixels).fill(index + 1),
  normal: new Uint32Array(pixels).fill(index + 1),
  confidence: new Float32Array(pixels).fill(1),
  rgba,
}));
global.gc?.();
const before = process.memoryUsage();
const started = performance.now();
let output, hostMaps;
if (mode === 'owned') {
  const owner = new OwnedPlanarMaps(input, sources);
  for (const reference of owner.plan.order) {
    // Check that every configured dependency is still the original, frozen generated value.
    for (const source of owner.plan.dependencies[reference])
      if (owner.original(source).depth[0] !== source + 1)
        throw new Error('Frozen evidence changed');
    const view = owner.original(reference);
    owner.complete(reference, {
      ...view,
      depth: view.depth.slice(),
      normal: view.normal.slice(),
      confidence: view.confidence.slice(),
    });
    await setImmediate();
  }
  hostMaps = owner.finish();
  output = input;
} else {
  output = [];
  for (const view of input) {
    output.push({
      ...view,
      depth: view.depth.slice(),
      normal: view.normal.slice(),
      confidence: view.confidence.slice(),
    });
    await setImmediate();
  }
  hostMaps = {
    inputBytes: 12 * count * pixels,
    peakRetainedBytes: 24 * count * pixels,
    peakExtraOriginalViews: count,
    retiredOriginalViews: 0,
    reorderedViews: 0,
  };
}
const elapsedMs = performance.now() - started;
const hash = createHash('sha256');
for (const view of output)
  for (const array of [view.depth, view.normal, view.confidence])
    hash.update(new Uint8Array(array.buffer, array.byteOffset, array.byteLength));
console.log(
  JSON.stringify({
    scope:
      'Generated-map lifetime simulation only. Source dependency graph is recorded; map values and dimensions are synthetic. No plane fitting, depth reconstruction, TSDF fusion or Chrome RSS is measured.',
    mode,
    sourceManifestSha256: createHash('sha256').update(sourceBytes).digest('hex'),
    views: count,
    width,
    height,
    sourceWidth: maps[0].width,
    sourceHeight: maps[0].height,
    conservativeScheduleExtraViews: plan.peakExtraViews,
    scheduleSha256: createHash('sha256').update(JSON.stringify(plan.order)).digest('hex'),
    hostMaps,
    predictedMapBytesAtRecordedResolution: (hostMaps.peakRetainedBytes * sourcePixels) / pixels,
    elapsedMs,
    before,
    after: process.memoryUsage(),
    peakRssKiB: process.resourceUsage().maxRSS,
    outputSha256: hash.digest('hex'),
  }),
);
