/** Actual-sized synthetic host lifetime receipt. This allocates backing buffers and counts their
 * reachable roots, without decoding video or estimating Chrome RSS. No source imagery/features
 * are loaded. With --record only aggregate sizes/counts are read from an existing run record. */
import { readFile } from 'node:fs/promises';
import { backingBytes, OwnedMultiVideoFeatures } from '../src/gpu/multi-video-lifetime.ts';

const args = process.argv.slice(2);
if (args.length && (args.length !== 2 || args[0] !== '--record')) {
  throw new Error(
    'Usage: node scripts/measure-multi-video-host-memory.mjs [--record run-record.json]',
  );
}
const record = args.length ? JSON.parse(await readFile(args[1], 'utf8')) : undefined;
const featureCounts =
  record?.result?.featureCounts ?? Array.from({ length: 333 }, (_, i) => 2141 + (i < 239 ? 1 : 0));
const views = featureCounts.length;
const features = featureCounts.reduce((sum, count) => sum + count, 0);
const components = record?.result?.components?.map((component) => component.registeredViews) ?? [
  240, 93,
];
const width = record?.input?.width ?? 1280;
const height = record?.input?.height ?? 720;
if (
  !views ||
  views > 1000 ||
  !Number.isSafeInteger(features) ||
  features < 0 ||
  features > 16_384_000 ||
  featureCounts.some((count) => !Number.isSafeInteger(count) || count < 0 || count > 16384) ||
  !Number.isSafeInteger(width) ||
  !Number.isSafeInteger(height) ||
  width < 1 ||
  height < 1 ||
  width * height > 2_000_000 ||
  components.reduce((sum, count) => sum + count, 0) !== views
) {
  throw new Error(
    'This bounded fixture requires complete registered components and finite feature/image sizes',
  );
}
let componentMembers;
if (record) {
  const videos = record.input?.videos;
  if (
    !videos?.length ||
    videos.reduce((sum, video) => sum + video.keyframes, 0) !== views ||
    videos.some(
      (video) =>
        video.width !== width ||
        video.height !== height ||
        video.registeredViews !== video.keyframes ||
        !Number.isSafeInteger(video.component) ||
        video.component < 0 ||
        video.component >= components.length,
    )
  ) {
    throw new Error(
      'A record must identify all retained feature rows, equal analysis dimensions and registered component memberships',
    );
  }
  componentMembers = components.map(() => []);
  let cursor = 0;
  for (const video of videos)
    for (let i = 0; i < video.keyframes; i++) componentMembers[video.component].push(cursor++);
  if (componentMembers.some((ids, i) => ids.length !== components[i]))
    throw new Error('Recorded component membership does not match registered counts');
} else {
  let cursor = 0;
  componentMembers = components.map((count) => Array.from({ length: count }, () => cursor++));
}
const camera = { fx: width, fy: width, cx: width / 2, cy: height / 2 };
const worker = featureCounts.map((count, viewId) => ({
  videoId: 'synthetic',
  frameKey: String(viewId),
  time: viewId,
  camera,
  keypoints: new Float64Array(2 * count),
  descriptors: new Uint8Array(128 * count),
  global: new Float32Array(4096),
  colors: new Uint8Array(3 * count),
  rgba: new Uint8ClampedArray(width * height * 4),
}));
// Separate owner containers share identical synthetic allocations. This lets the census
// compare before/after lifetimes without doubling the fixture or claiming a GC/RSS result.
const legacyWorker = worker.map((frame) => ({ ...frame }));
const lease = new OwnedMultiVideoFeatures(
  worker.slice(),
  worker.map((_, i) => i),
);
const images = new Map();
const rows = [];
function snapshot(stage) {
  const workerFeatures = worker.flatMap((frame) => [
    frame.keypoints,
    frame.descriptors,
    frame.global,
    frame.colors,
  ]);
  const pixels = [
    ...worker.flatMap((frame) => (frame.rgba ? [frame.rgba] : [])),
    ...images.values(),
  ];
  // The lease and worker share the same original buffers; the combined census must deduplicate
  // them instead of summing independent counters and claiming twice the feature memory.
  const libraryFeatures = lease.keyframes.flatMap((frame) => [frame.keypoints, frame.global]);
  const legacyFeatures = legacyWorker.flatMap((frame) => [
    frame.keypoints,
    frame.descriptors,
    frame.global,
    frame.colors,
  ]);
  const legacyPixels = legacyWorker.flatMap((frame) => (frame.rgba ? [frame.rgba] : []));
  rows.push({
    stage,
    workerFeatureBytes: backingBytes(workerFeatures),
    libraryFeatureBytes: lease.bytes,
    rgbaBytes: backingBytes(pixels),
    combinedRetainedBytes: backingBytes([...workerFeatures, ...libraryFeatures, ...pixels]),
    legacyWorkerFeatureBytes: backingBytes(legacyFeatures),
    legacyCombinedRetainedBytes: backingBytes([...legacyFeatures, ...legacyPixels]),
  });
}
snapshot('beforeGraph');
lease.releaseDescriptors();
for (const frame of worker) {
  frame.descriptors = new Uint8Array(0);
  frame.global = new Float32Array(0);
}
snapshot('afterGraph');
for (let i = 0; i < components.length; i++) {
  const ids = componentMembers[i];
  lease.releaseKeypoints(ids);
  for (const id of ids) {
    worker[id].keypoints = new Float64Array(0);
    worker[id].colors = new Uint8Array(0);
  }
  snapshot(`afterSparseComponent${i}`);
}
for (const [viewId, frame] of worker.entries()) {
  images.set(viewId, frame.rgba);
  frame.rgba = undefined;
}
worker.length = 0;
snapshot('beforeDense');
for (let i = 0; i < components.length; i++) {
  for (const id of componentMembers[i]) {
    images.delete(id);
    legacyWorker[id].rgba = undefined;
  }
  snapshot(`afterDenseComponent${i}`);
}
legacyWorker.length = 0;
snapshot('afterWorkerClose');
console.log(
  JSON.stringify(
    {
      kind: 'logical-allocated-backing-roots',
      scope:
        'Synthetic feature/RGBA arrays with recorded aggregate sizes. Counts deduplicate complete backing allocations; source bytes are synthetic zeros. Excludes JS object/string/observation overhead, intermediate DescriptorStore/mapper/device arrays, archives, driver allocation and Chrome RSS. This is not a reconstruction runtime or browser RSS benchmark.',
      views,
      features,
      width,
      height,
      componentViews: components,
      featureBudget: {
        descriptors: features * 128,
        keypoints: features * 16,
        colors: features * 3,
        globals: views * 4096 * 4,
      },
      stages: rows,
    },
    null,
    2,
  ),
);
