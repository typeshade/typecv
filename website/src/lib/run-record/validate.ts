/** The validator of a run record (`runRecordProblem`) and the field checks that the builders and
 *  the probe record share. */
import { gpuLossContextValid } from '../gpu-loss-context.ts';
import { videoRangeValid } from '../video-range.ts';
import { initializationFailureValid } from '../initialization-failure.ts';
import {
  RESULT_FILE_MAX_BYTES,
  RESULT_FILE_MAX_COUNT,
  RESULT_FILE_PART_BYTES,
  RESULT_FILES_MAX_BYTES,
  resultFileBaseNameValid,
  resultFileNameValid,
  RUN_RECORD_DENSE_STAGES,
  RUN_RECORD_FORMAT,
  RUN_RECORD_FORMATS,
  RUN_RECORD_MAX_CHUNK_KEYS,
  RUN_RECORD_MAX_FEATURES,
  RUN_RECORD_MAX_FRAMES,
  RUN_RECORD_MAX_GPU_NAME,
  RUN_RECORD_MAX_GPU_ROWS,
  RUN_RECORD_MAX_KERNELS,
  RUN_RECORD_MAX_NOTE,
  RUN_RECORD_MAX_VIDEOS,
  RUN_RECORD_VIDEO_TAGS,
  type RunRecord,
  type RunRecordChunk,
  type RunRecordDense,
  type RunRecordDensePerformance,
  type RunRecordDescriptorCache,
  type RunRecordKlt,
  type RunRecordMultiVideoOptions,
  type RunRecordVideo,
} from './schema.ts';

export const finite = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value);
export const count = (value: unknown): value is number =>
  Number.isSafeInteger(value) && (value as number) >= 0;
export const text = (value: unknown, max: number): value is string =>
  typeof value === 'string' && value.length <= max;

export const HOST_RETENTION_COUNTS = [
  'featuresBeforeSfMBytes',
  'featuresAfterSfMBytes',
  'featuresBeforeGraphBytes',
  'featuresAfterGraphBytes',
  'featuresAfterMappingBytes',
  'graphMatchesBeforeMappingBytes',
  'graphMatchesAfterMappingBytes',
] as const;
const DESCRIPTOR_CACHE_COUNTS = [
  'hits',
  'misses',
  'uncached',
  'decodes',
  'decodedBytes',
  'applicationUploadBytes',
  'evictions',
  'entries',
  'residentBytes',
  'peakEntries',
  'peakResidentBytes',
  'limitBytes',
] as const;
const TSDF_QUEUED_VIEW_COUNTS = ['limitBytes', 'peakSubmittedBytes', 'drainCount'] as const;

export function countFields<const Keys extends readonly string[]>(
  value: unknown,
  keys: Keys,
): { readonly [Key in Keys[number]]: number } | undefined {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return undefined;
  const input = value as Record<string, unknown>;
  return keys.every((key) => count(input[key]))
    ? (Object.fromEntries(keys.map((key) => [key, input[key]])) as {
        readonly [Key in Keys[number]]: number;
      })
    : undefined;
}

export function descriptorCacheRecord(value: unknown): RunRecordDescriptorCache | undefined {
  const counts = countFields(value, DESCRIPTOR_CACHE_COUNTS);
  return counts &&
    counts.entries <= counts.peakEntries &&
    counts.peakEntries <= 16 &&
    counts.residentBytes <= counts.peakResidentBytes &&
    counts.peakResidentBytes <= counts.limitBytes
    ? counts
    : undefined;
}

export function queuedViewUploadsRecord(
  value: unknown,
): NonNullable<NonNullable<RunRecordDensePerformance['tsdf']>['queuedViewUploads']> | undefined {
  const counts = countFields(value, TSDF_QUEUED_VIEW_COUNTS);
  return counts && counts.limitBytes >= 1 && counts.peakSubmittedBytes <= counts.limitBytes
    ? counts
    : undefined;
}

/** New optional diagnostic objects are bounded to their selected, known integer fields. */
function selectedCountFieldsValid(value: unknown, keys: readonly string[]): boolean {
  return (
    !!countFields(value, keys) && Object.keys(value as object).every((key) => keys.includes(key))
  );
}

const positive = (value: unknown): value is number => finite(value) && value > 0;
export function chunksValid(value: unknown): boolean {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false;
  const entries = Object.entries(value as Record<string, RunRecordChunk>);
  return (
    entries.length <= RUN_RECORD_MAX_CHUNK_KEYS &&
    entries.every(
      ([key, rate]) =>
        text(key, 64) &&
        key.length > 0 &&
        rate &&
        typeof rate === 'object' &&
        finite(rate.msPerUnit) &&
        rate.msPerUnit >= 0 &&
        count(rate.chunks) &&
        finite(rate.longestMs) &&
        rate.longestMs >= 0 &&
        count(rate.lastUnits),
    )
  );
}

function cleanValid(value: unknown): boolean {
  if (value === undefined) return true;
  const c = value as NonNullable<RunRecordDense['clean']>;
  const ms = (v: unknown) => finite(v) && (v as number) >= 0;
  return (
    !!c &&
    typeof c === 'object' &&
    (c.tier === 'cpu' || c.tier === 'webgpu') &&
    count(c.before) &&
    count(c.afterOutliers) &&
    count(c.afterVoxel) &&
    count(c.after) &&
    ms(c.outlierThreshold) &&
    ms(c.voxelSize) &&
    ms(c.eps) &&
    count(c.clusters) &&
    count(c.keptClusters) &&
    ms(c.outliersMs) &&
    ms(c.voxelMs) &&
    ms(c.clusterMs) &&
    text(c.error, 200)
  );
}

function meshValid(value: unknown): boolean {
  if (value === undefined) return true;
  const m = value as NonNullable<RunRecordDense['mesh']>;
  const ms = (v: unknown) => finite(v) && (v as number) >= 0;
  return (
    !!m &&
    typeof m === 'object' &&
    (m.tier === 'cpu' || m.tier === 'webgpu') &&
    count(m.points) &&
    ms(m.voxelSize) &&
    ms(m.radius) &&
    count(m.nodes) &&
    count(m.cubes) &&
    count(m.vertices) &&
    count(m.faces) &&
    ms(m.latticeMs) &&
    ms(m.fieldMs) &&
    ms(m.cubesMs) &&
    ms(m.verticesMs) &&
    text(m.error, 200)
  );
}

function kltSetupValid(value: unknown): boolean {
  const k = value as NonNullable<RunRecord['options']['kltSetup']> | null;
  return (
    typeof k === 'object' &&
    k !== null &&
    (k.capacity === undefined ||
      (Number.isInteger(k.capacity) && k.capacity >= 1 && k.capacity <= 8192)) &&
    (k.fillIn === undefined || k.fillIn === true) &&
    (k.matching === undefined || k.matching === 'tracks') &&
    (k.gate === undefined || k.gate === true)
  );
}

function kltValid(value: unknown): boolean {
  const k = value as RunRecordKlt;
  if (!k || typeof k !== 'object') return false;
  const ms = (v: unknown) => finite(v) && (v as number) >= 0;
  return (
    count(k.pushes) &&
    count(k.joins) &&
    count(k.alive) &&
    Array.isArray(k.ended) &&
    k.ended.length === 8 &&
    k.ended.every(count) &&
    count(k.joined) &&
    count(k.seeded) &&
    count(k.filledIn) &&
    count(k.filtered) &&
    count(k.matches) &&
    count(k.pairMatches) &&
    count(k.duplicateMatches) &&
    count(k.conflictingMatches) &&
    count(k.pairs) &&
    count(k.trackerOnlyPairs) &&
    count(k.rejected) &&
    Array.isArray(k.lengths) &&
    k.lengths.length === 6 &&
    k.lengths.every(count) &&
    ms(k.meanLived) &&
    ms(k.msPush) &&
    ms(k.msSelect) &&
    ms(k.msJoin)
  );
}

export const PLANAR_HOST_MAP_COUNTS = [
  'inputBytes',
  'peakRetainedBytes',
  'peakExtraOriginalViews',
  'retiredOriginalViews',
  'reorderedViews',
] as const;
export const TSDF_COUNTS = ['blocks', 'passes', 'peakBytes', 'uploadBytes'] as const;
export const TSDF_OPTIONAL_COUNTS = [
  'hostViewStagingBytes',
  'allocationPixels',
  'reusedBoxTables',
  'inlineVoteResetGroups',
] as const;
export const TSDF_HOST_TIMINGS = [
  'volumePlanningMs',
  'boxPlanningMs',
  'boxTableMs',
  'viewUploadMs',
  'sortingMs',
] as const;

function densePerformanceValid(value: unknown): boolean {
  if (value === undefined) return true;
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false;
  const p = value as RunRecordDensePerformance;
  const ms = (v: unknown) => finite(v) && v >= 0;
  return (
    (p.planarHostMaps !== undefined || p.tsdf !== undefined) &&
    (p.planarHostMaps === undefined ||
      (!!p.planarHostMaps &&
        typeof p.planarHostMaps === 'object' &&
        !Array.isArray(p.planarHostMaps) &&
        PLANAR_HOST_MAP_COUNTS.every((key) => count(p.planarHostMaps![key])) &&
        p.planarHostMaps.peakRetainedBytes >= p.planarHostMaps.inputBytes)) &&
    (p.tsdf === undefined ||
      (!!p.tsdf &&
        typeof p.tsdf === 'object' &&
        !Array.isArray(p.tsdf) &&
        TSDF_COUNTS.every((key) => count(p.tsdf![key])) &&
        TSDF_OPTIONAL_COUNTS.every((key) => p.tsdf![key] === undefined || count(p.tsdf![key])) &&
        (p.tsdf.queuedViewUploads === undefined ||
          (selectedCountFieldsValid(p.tsdf.queuedViewUploads, TSDF_QUEUED_VIEW_COUNTS) &&
            !!queuedViewUploadsRecord(p.tsdf.queuedViewUploads))) &&
        (p.tsdf.hostTimings === undefined ||
          (!!p.tsdf.hostTimings &&
            typeof p.tsdf.hostTimings === 'object' &&
            !Array.isArray(p.tsdf.hostTimings) &&
            TSDF_HOST_TIMINGS.every((key) => ms(p.tsdf!.hostTimings![key]))))))
  );
}

function denseValid(value: unknown): boolean {
  const d = value as RunRecordDense;
  if (!d || typeof d !== 'object') return false;
  const o = d.options,
    m = d.memory,
    f = d.failure;
  const bytes = (v: unknown) => v === null || count(v);
  const ms = (v: unknown) => finite(v) && (v as number) >= 0;
  return (
    ['completed', 'cancelled', 'failed', 'skipped'].includes(d.status) &&
    text(d.skipped, 200) &&
    !!o &&
    typeof o === 'object' &&
    count(o.maxSources) &&
    count(o.bestSources) &&
    count(o.windowRadius) &&
    count(o.windowStep) &&
    count(o.photometricIterations) &&
    count(o.geometricIterations) &&
    count(o.minConsistentViews) &&
    (o.denseClean === undefined || typeof o.denseClean === 'boolean') &&
    (o.fusion === undefined || o.fusion === 'pixel' || o.fusion === 'tsdf') &&
    (o.denseMesh === undefined || typeof o.denseMesh === 'boolean') &&
    (o.planarDepth === undefined || typeof o.planarDepth === 'boolean') &&
    (o.boundaryAware === undefined || typeof o.boundaryAware === 'boolean') &&
    (o.minFiniteGain === undefined ||
      (finite(o.minFiniteGain) && o.minFiniteGain >= 0 && o.minFiniteGain <= 2)) &&
    (o.gpuTimestamps === undefined || typeof o.gpuTimestamps === 'boolean') &&
    (o.tsdfVisibility === undefined ||
      o.tsdfVisibility === 'ray-count' ||
      o.tsdfVisibility === 'view-weighted') &&
    (d.planarFilled === undefined || count(d.planarFilled)) &&
    densePerformanceValid(d.performance) &&
    cleanValid(d.clean) &&
    meshValid(d.mesh) &&
    count(d.views) &&
    count(d.viewsWithoutDepth) &&
    count(d.points) &&
    ms(d.totalMs) &&
    !!d.stageMs &&
    typeof d.stageMs === 'object' &&
    Object.keys(d.stageMs).length === RUN_RECORD_DENSE_STAGES.length &&
    RUN_RECORD_DENSE_STAGES.every((stage) => ms(d.stageMs[stage])) &&
    !!m &&
    typeof m === 'object' &&
    count(m.liveBytesBefore) &&
    count(m.pooledBytesBefore) &&
    bytes(m.depthStageBytes) &&
    bytes(m.depthStageFormulaBytes) &&
    bytes(m.fusionStageBytes) &&
    (d.status === 'failed'
      ? !!f &&
        typeof f === 'object' &&
        text(f.message, 2000) &&
        text(f.stage, 64) &&
        text(f.kernel, 64) &&
        text(f.reason, 64) &&
        (f.context === undefined || gpuLossContextValid(f.context))
      : f === null)
  );
}

/** Return the reason a value is not a run record, or undefined when it is one. */
function mainThreadValid(value: unknown): boolean {
  if (value === undefined) return true;
  const m = value as Record<string, unknown>;
  const measure = (v: unknown) => typeof v === 'number' && Number.isFinite(v) && v >= 0;
  const stage = (v: unknown) => typeof v === 'string' && v.length <= 128;
  return (
    !!m &&
    typeof m === 'object' &&
    Number.isSafeInteger(m.longTasks) &&
    (m.longTasks as number) >= 0 &&
    measure(m.longTaskMs) &&
    measure(m.longestTaskMs) &&
    stage(m.longestTaskStage) &&
    Number.isSafeInteger(m.frameGaps) &&
    (m.frameGaps as number) >= 0 &&
    measure(m.frameGapMs) &&
    measure(m.longestGapMs) &&
    stage(m.longestGapStage) &&
    (m.hiddenMs === undefined || measure(m.hiddenMs))
  );
}

function filesValid(value: unknown): boolean {
  if (value === undefined) return true;
  if (!Array.isArray(value) || value.length > RESULT_FILE_MAX_COUNT) return false;
  const names = new Set<unknown>();
  const originals = new Map<string, { total: number; end: number; count: number }>();
  let uploadedBytes = 0;
  for (const file of value) {
    if (!file || typeof file !== 'object') return false;
    const f = file as Record<string, unknown>;
    if (
      !resultFileNameValid(f.name) ||
      names.has(f.name) ||
      (f.bytes !== undefined && (!Number.isSafeInteger(f.bytes) || (f.bytes as number) < 0)) ||
      (f.sha256 !== undefined &&
        (typeof f.sha256 !== 'string' || !/^[0-9a-f]{64}$/.test(f.sha256))) ||
      (f.skipped !== undefined && (!text(f.skipped, 200) || f.skipped === '')) ||
      (f.skipped === undefined &&
        (f.bytes === undefined ||
          f.sha256 === undefined ||
          (f.bytes as number) > RESULT_FILE_MAX_BYTES))
    )
      return false;
    names.add(f.name);
    const partName = /^(.*)\.part-([0-9]{4})$/.exec(f.name);
    if (!!partName !== (f.part !== undefined)) return false;
    if (partName) {
      if (!f.part || typeof f.part !== 'object') return false;
      const p = f.part as Record<string, unknown>;
      if (
        p.original !== partName[1] ||
        !resultFileBaseNameValid(p.original) ||
        !Number.isSafeInteger(p.totalBytes) ||
        (p.totalBytes as number) <= RESULT_FILE_MAX_BYTES ||
        (p.totalBytes as number) > RESULT_FILES_MAX_BYTES ||
        !Number.isSafeInteger(p.offset) ||
        (p.offset as number) < 0 ||
        !Number.isSafeInteger(f.bytes) ||
        (f.bytes as number) <= 0 ||
        (f.bytes as number) > RESULT_FILE_PART_BYTES
      )
        return false;
      const previous = originals.get(p.original) ?? {
        total: p.totalBytes as number,
        end: 0,
        count: 0,
      };
      if (
        previous.total !== p.totalBytes ||
        previous.end !== p.offset ||
        previous.count !== Number(partName[2]) ||
        previous.end + (f.bytes as number) > previous.total
      )
        return false;
      originals.set(p.original, {
        total: previous.total,
        end: previous.end + (f.bytes as number),
        count: previous.count + 1,
      });
    }
    if (f.skipped === undefined) uploadedBytes += f.bytes as number;
    if (uploadedBytes > RESULT_FILES_MAX_BYTES) return false;
  }
  for (const [original, part] of originals)
    if (names.has(original) || part.end !== part.total) return false;
  return true;
}

export function runRecordProblem(value: unknown): string | undefined {
  const r = value as RunRecord;
  if (!r || typeof r !== 'object') return 'not an object';
  if (!(RUN_RECORD_FORMATS as readonly unknown[]).includes(r.format)) return 'format';
  if (!text(r.createdAt, 40) || Number.isNaN(Date.parse(r.createdAt))) return 'createdAt';
  if (!r.site || !text(r.site.commit, 64) || !text(r.site.runtime, 200)) return 'site';
  const c = r.client;
  if (
    !c ||
    !text(c.userAgent, 512) ||
    !text(c.platform, 64) ||
    !text(c.language, 32) ||
    !count(c.hardwareConcurrency) ||
    !(c.deviceMemoryGb === null || finite(c.deviceMemoryGb)) ||
    !c.screen ||
    !count(c.screen.width) ||
    !count(c.screen.height) ||
    !finite(c.screen.dpr)
  )
    return 'client';
  const g = r.gpu;
  if (
    !g ||
    !text(g.vendor, 128) ||
    !text(g.architecture, 128) ||
    !text(g.device, 128) ||
    !text(g.description, 512) ||
    typeof g.fallbackAdapter !== 'boolean' ||
    !g.limits ||
    !count(g.limits.maxStorageBufferBindingSize) ||
    !count(g.limits.maxBufferSize)
  )
    return 'gpu';
  if (!gpuV6Valid(r.format === RUN_RECORD_FORMAT, g)) return 'gpu';
  const i = r.input;
  if (
    !i ||
    !['sample', 'prepared', 'video'].includes(i.kind) ||
    !(i.name === null || text(i.name, 128)) ||
    !count(i.frames) ||
    i.frames > RUN_RECORD_MAX_FRAMES ||
    !count(i.width) ||
    !count(i.height) ||
    !count(i.sourceWidth) ||
    !count(i.sourceHeight) ||
    !finite(i.durationSeconds) ||
    i.durationSeconds < 0 ||
    (i.range !== undefined && !videoRangeValid(i.range, i.durationSeconds)) ||
    !videosValid(i.videos)
  )
    return 'input';
  const o = r.options;
  if (
    !o ||
    !finite(o.focalRatio) ||
    !count(o.windowSize) ||
    !count(o.ransacIterations) ||
    !finite(o.maxErrorPx) ||
    !count(o.bundleIterations) ||
    !count(o.finalBundleIterations) ||
    typeof o.refineSharedFocal !== 'boolean' ||
    (o.focalSearch !== undefined && typeof o.focalSearch !== 'boolean') ||
    !count(o.maxFeatures) ||
    !finite(o.matchRatio) ||
    !finite(o.maxDistance) ||
    o.maxDistance < 0 ||
    !['corner', 'sift'].includes(o.frontEnd) ||
    typeof o.boundedSubmissions !== 'boolean' ||
    !['float', 'integer'].includes(o.fp64Flavor) ||
    (o.guidedMatching !== undefined && typeof o.guidedMatching !== 'boolean') ||
    (o.guidedKernel !== undefined && o.guidedKernel !== 'tiled' && o.guidedKernel !== 'scalar') ||
    (o.siftFirstOctave !== undefined && o.siftFirstOctave !== -1 && o.siftFirstOctave !== 0) ||
    (o.clahe !== undefined && typeof o.clahe !== 'boolean') ||
    (o.videoSampleFps !== undefined && (!finite(o.videoSampleFps) || o.videoSampleFps <= 0)) ||
    (o.videoDecode !== undefined &&
      o.videoDecode !== 'worker' &&
      o.videoDecode !== 'main-thread') ||
    (o.videoSource !== undefined &&
      o.videoSource !== 'source' &&
      o.videoSource !== 'frames' &&
      o.videoSource !== 'mixed') ||
    (o.keyframeGate !== undefined && o.keyframeGate !== 'phase' && o.keyframeGate !== 'legacy') ||
    (o.features !== undefined && o.features !== 'orb') ||
    (o.keyframes !== undefined && !count(o.keyframes)) ||
    (o.klt !== undefined && typeof o.klt !== 'boolean') ||
    (o.kltSetup !== undefined && !kltSetupValid(o.kltSetup)) ||
    !multiVideoOptionsValid(o.multiVideo)
  )
    return 'options';
  const s = r.result;
  if (
    !s ||
    !count(s.registeredViews) ||
    !count(s.landmarks) ||
    !count(s.observations) ||
    !(s.meanReprojectionErrorPx === null || finite(s.meanReprojectionErrorPx)) ||
    !count(s.unregistered) ||
    !count(s.matchedPairs) ||
    !Array.isArray(s.featureCounts) ||
    s.featureCounts.length > RUN_RECORD_MAX_FRAMES ||
    !s.featureCounts.every(count) ||
    (s.hostRetention !== undefined &&
      !selectedCountFieldsValid(s.hostRetention, HOST_RETENTION_COUNTS)) ||
    (s.descriptorCache !== undefined &&
      (!selectedCountFieldsValid(s.descriptorCache, DESCRIPTOR_CACHE_COUNTS) ||
        !descriptorCacheRecord(s.descriptorCache))) ||
    (s.klt !== undefined && !kltValid(s.klt)) ||
    (s.focalSearch !== undefined &&
      (!s.focalSearch ||
        !(finite(s.focalSearch.ratio) && s.focalSearch.ratio > 0) ||
        !(finite(s.focalSearch.gridRatio) && s.focalSearch.gridRatio > 0) ||
        !count(s.focalSearch.views))) ||
    !multiVideoResultValid(s)
  )
    return 'result';
  const t = r.timing;
  if (
    !t ||
    !finite(t.totalMs) ||
    typeof t.reusedDevice !== 'boolean' ||
    !t.stages ||
    typeof t.stages !== 'object' ||
    Object.keys(t.stages).length > 64 ||
    !Object.entries(t.stages).every(
      ([name, stage]) =>
        text(name, 64) &&
        stage &&
        count(stage.calls) &&
        finite(stage.elapsedMs) &&
        count(stage.dispatches) &&
        count(stage.submissions) &&
        (stage.gpuMs === undefined
          ? true
          : r.format === RUN_RECORD_FORMAT && finite(stage.gpuMs) && stage.gpuMs >= 0),
    ) ||
    !Array.isArray(t.frameMs) ||
    t.frameMs.length > RUN_RECORD_MAX_FRAMES ||
    !t.frameMs.every(finite) ||
    !Array.isArray(t.kernels) ||
    t.kernels.length > RUN_RECORD_MAX_KERNELS ||
    !t.kernels.every((name) => text(name, 64)) ||
    !gpuTimeValid(r.format === RUN_RECORD_FORMAT, t) ||
    (r.format !== 'typecv-run-record-v3'
      ? !chunksValid(t.chunks) ||
        !(t.budgetMs === null || positive(t.budgetMs)) ||
        !mainThreadValid(t.mainThread)
      : t.chunks !== undefined || t.budgetMs !== undefined || t.mainThread !== undefined)
  )
    return 'timing';
  const x = r.transfers;
  if (!x || !count(x.applicationUploadBytes) || !count(x.applicationReadbackBytes))
    return 'transfers';
  if (!(
    r.benchmark === null ||
    (r.benchmark &&
      text(r.benchmark.name, 64) &&
      (r.benchmark.pass === 'cold' || r.benchmark.pass === 'warm'))
  ))
    return 'benchmark';
  const f = r.reference;
  const measure = (value: unknown) => value === null || (finite(value) && (value as number) >= 0);
  if (!(
    f === null ||
    (f &&
      text(f.name, 128) &&
      text(f.status, 128) &&
      count(f.commonViews) &&
      measure(f.medianRotationDeg) &&
      measure(f.maxRotationDeg) &&
      measure(f.medianCenterOfRadius) &&
      measure(f.maxCenterOfRadius))
  ))
    return 'reference';
  const e = r.failure;
  if (!(
    e === null ||
    (e &&
      text(e.message, 2000) &&
      text(e.operation, 200) &&
      text(e.kernel, 64) &&
      text(e.reason, 64) &&
      count(e.framesProcessed) &&
      (e.context === undefined || gpuLossContextValid(e.context)) &&
      (e.initialization === undefined || initializationFailureValid(e.initialization)))
  ))
    return 'failure';
  if (!text(r.note, RUN_RECORD_MAX_NOTE)) return 'note';
  if (
    r.format === RUN_RECORD_FORMAT || r.format === 'typecv-run-record-v5'
      ? r.dense !== undefined && !denseValid(r.dense)
      : r.dense !== undefined
  )
    return 'dense';
  if (!filesValid(r.files)) return 'files';
  if (r.filePreparationFailure !== undefined) {
    const f = r.filePreparationFailure;
    if (
      !f ||
      typeof f !== 'object' ||
      !text(f.reason, 200) ||
      !f.reason ||
      !Number.isSafeInteger(f.requested) ||
      f.requested < 1 ||
      f.requested > RESULT_FILE_MAX_COUNT
    )
      return 'filePreparationFailure';
  }
  return undefined;
}

/** `gpu.features` and the compute limits: required in a v6 record, absent before it. */
function gpuV6Valid(current: boolean, g: RunRecord['gpu']): boolean {
  const limits = g.limits;
  const added = [
    limits.maxComputeWorkgroupStorageSize,
    limits.maxComputeInvocationsPerWorkgroup,
    limits.maxStorageBuffersPerShaderStage,
  ];
  if (!current) return g.features === undefined && added.every((v) => v === undefined);
  return (
    Array.isArray(g.features) &&
    g.features.length <= RUN_RECORD_MAX_FEATURES &&
    g.features.every((name) => text(name, 64)) &&
    added.every(count)
  );
}

/** `timing.gpu`, `gpuTimestamps` and `gpuDropped`: required in a v6 record, absent before it. */
function gpuTimeValid(current: boolean, t: RunRecord['timing']): boolean {
  if (!current)
    return t.gpu === undefined && t.gpuTimestamps === undefined && t.gpuDropped === undefined;
  return (
    Array.isArray(t.gpu) &&
    t.gpu.length <= RUN_RECORD_MAX_GPU_ROWS &&
    t.gpu.every(
      (row) =>
        !!row &&
        typeof row === 'object' &&
        text(row.name, RUN_RECORD_MAX_GPU_NAME) &&
        row.name.length > 0 &&
        count(row.groups) &&
        count(row.dispatches) &&
        count(row.timedGroups) &&
        row.timedGroups <= row.groups &&
        (row.gpuMs === null ? row.timedGroups === 0 : finite(row.gpuMs) && row.gpuMs >= 0),
    ) &&
    typeof t.gpuTimestamps === 'boolean' &&
    count(t.gpuDropped)
  );
}

const VIDEO_ID = /^[0-9a-f]{16}$/;

function videosValid(value: unknown): boolean {
  if (value === undefined) return true;
  if (!Array.isArray(value) || value.length > RUN_RECORD_MAX_VIDEOS) return false;
  return (value as RunRecordVideo[]).every(
    (v) =>
      !!v &&
      typeof v === 'object' &&
      typeof v.videoId === 'string' &&
      VIDEO_ID.test(v.videoId) &&
      (v.label === null || (text(v.label, 64) && v.label.length > 0)) &&
      finite(v.durationSeconds) &&
      v.durationSeconds >= 0 &&
      (v.range === undefined || videoRangeValid(v.range, v.durationSeconds)) &&
      count(v.keyframes) &&
      count(v.registeredViews) &&
      v.registeredViews <= v.keyframes &&
      (v.frames === undefined || (count(v.frames) && v.frames >= v.keyframes)) &&
      (v.width === undefined || count(v.width)) &&
      (v.height === undefined || count(v.height)) &&
      positive(v.focalRatio) &&
      (v.component === null || count(v.component)) &&
      Array.isArray(v.tags) &&
      v.tags.length <= 5 &&
      v.tags.every((t) => (RUN_RECORD_VIDEO_TAGS as readonly unknown[]).includes(t)),
  );
}

function multiVideoOptionsValid(value: unknown): boolean {
  if (value === undefined) return true;
  const m = value as RunRecordMultiVideoOptions;
  return (
    !!m &&
    typeof m === 'object' &&
    text(m.descriptor, 32) &&
    m.descriptor.length > 0 &&
    count(m.k) &&
    count(m.minMatches) &&
    count(m.minEdgeInliers) &&
    count(m.maxKeyframesPerVideo) &&
    (m.growth === 'incremental' || m.growth === 'submaps') &&
    (m.canonical === undefined || typeof m.canonical === 'boolean') &&
    (m.focalPerVideo === undefined || typeof m.focalPerVideo === 'boolean')
  );
}

function multiVideoResultValid(s: RunRecord['result']): boolean {
  const components = s.components;
  if (
    components !== undefined &&
    !(
      Array.isArray(components) &&
      components.length <= RUN_RECORD_MAX_VIDEOS &&
      components.every(
        (c) =>
          !!c &&
          typeof c === 'object' &&
          count(c.keyframes) &&
          count(c.registeredViews) &&
          c.registeredViews <= c.keyframes &&
          count(c.landmarks) &&
          count(c.videos) &&
          (c.meanReprojectionErrorPx === null || finite(c.meanReprojectionErrorPx)) &&
          (c.dense === undefined || denseValid(c.dense)),
      )
    )
  )
    return false;
  const x = s.crossVideoEdges;
  if (x !== undefined && !(!!x && typeof x === 'object' && count(x.edges) && count(x.inliers)))
    return false;
  const g = s.viewGraph;
  if (
    g !== undefined &&
    !(
      !!g &&
      typeof g === 'object' &&
      count(g.nodes) &&
      count(g.candidatePairs) &&
      count(g.matchedPairs) &&
      count(g.verifiedEdges) &&
      count(g.exhaustivePairs)
    )
  )
    return false;
  const n = s.notConnected;
  return (
    n === undefined ||
    (Array.isArray(n) &&
      n.length <= RUN_RECORD_MAX_VIDEOS &&
      n.every((id) => typeof id === 'string' && VIDEO_ID.test(id)))
  );
}
