import {
  Button,
  Card,
  Checkbox,
  ConfigProvider,
  Input,
  InputNumber,
  Popover,
  Progress,
  Radio,
  Space,
  Switch,
  Tag,
  Tooltip,
  Typography,
  Upload,
  theme,
} from 'antd';
import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react';
import SfmViewer, { type ViewerState } from './SfmViewer';
import {
  deviceLabel,
  isResult,
  isVideoSet,
  resultFileNames,
  type CatalogEntry,
  type ResultEntry,
  type SampleCatalog,
  type SampleEntry,
  type VideoSetEntry,
} from '../lib/sample-catalog';
import {
  ComponentSwitch,
  InlierMatrix,
  VideoRows,
  VideoSetCard,
  type VideoRow,
} from './MultiVideoPanel';
import {
  MULTI_VIDEO_DEFAULTS,
  componentColor,
  markDuplicates,
  multiVideoOptions,
  multiVideoQuery,
} from '../lib/multi-video';
import {
  fetchSampleFrames,
  fileVideoId,
  probeDuration,
  sampleVideoId,
} from '../lib/multi-video-input';
import { catalogVideoId, runVideoSource, sourceUrl, videoSourceOf } from '../lib/sample-source';
import type { VideoSource } from '../lib/sample-source';
import { loadResultFiles } from '../lib/result-load';
import {
  ResultMissingError,
  fetchResultFiles,
  resultView,
  resultViewFromFiles,
  type ResultInfo,
  type ResultView,
} from '../lib/result-view';
import { reconstructionJson, reconstructionPly } from '../lib/result-export';
import { cameraInSourcePixels, parsePreparedVideo } from '../lib/sfm-input';
import { alignPoses, type CameraPose, type PoseAlignment } from '../lib/pose-alignment';
import {
  RUN_RECORD_FORMAT,
  RUN_RECORD_MAX_FRAMES,
  RUN_RECORD_MAX_NOTE,
  chunkLines,
  chunkRecord,
  denseLine,
  kltLine,
  denseRecord,
  runRecordProblem,
  summarizeStages,
  type RunRecord,
} from '../lib/run-record';
import { mainThreadLine, watchMainThread } from '../lib/main-thread';
import { ANALYSIS_EDGES, type AnalysisEdge } from '../lib/sfm-video';
import {
  featureFrontEnd,
  guidedKernel,
  kltSetup,
  kltTracking,
  orbTuning,
  siftQuality,
} from '../lib/sift-quality';
import { videoSampleFps } from '../lib/video-sampling';
import { keyframeGateMode, type KeyframeGateMode } from '../lib/keyframe-gate';
import {
  planFiles,
  postRecord,
  resultFilesAvailable,
  type ResultFilePayload,
  type SendState,
} from '../lib/send-record';
import {
  densePly,
  denseMeshPly,
  densePointBudget,
  denseStride,
  type DenseCloud,
  type DenseMeshData,
} from '../lib/dense-cloud';
import {
  cameraMotion,
  depthColor,
  depthPoints,
  depthRange,
  previewSeeker,
  seekFrame,
  type Scene,
} from '../lib/sfm-scene';
const { Title } = Typography;
type Frame = {
  id: number;
  width: number;
  height: number;
  pixels: Uint8ClampedArray;
  time: number;
  sourceWidth?: number;
  sourceHeight?: number;
  image?: string;
  /** The final frame of an uploaded video: the phase keyframe gate always keeps it. */
  last?: boolean;
  metrics?: { features: number; matches: number; inliers: number; registered: boolean };
  /** Known intrinsics in analysis pixels (a calibrated sample). */
  camera?: { fx: number; fy: number; cx: number; cy: number; k1: number; k2: number };
  /** Keypoints and raw matches from another front end (the COLMAP sample's SIFT). */
  precomputed?: { keypoints: Float64Array; pairs: { first: number; matches: Uint32Array }[] };
};
/** The worker's feature and matching controls: `video` for footage, `photo` for still photographs. */
type Profile = 'video' | 'photo';
/** `website/public/sample/<name>/sample.json`, written by `scripts/prepare-lab-colmap-sample.py`. */
type LabSample = {
  format: 'typecv-lab-sample-v1';
  name: string;
  credit: string;
  reference: string;
  source: { width: number; height: number; images: number };
  frames: {
    id: number;
    file: string;
    time: number;
    width: number;
    height: number;
    sourceWidth: number;
    sourceHeight: number;
    camera: { fx: number; fy: number; cx: number; cy: number; k1: number; k2: number };
    reference: { R: number[]; t: number[] };
  }[];
  features?: {
    file: string;
    keypointBytes: number;
    keypoints: { id: number; offset: number; count: number }[];
    pairs: { first: number; second: number; offset: number; count: number }[];
  };
};
/** The catalog types live in `lib/sample-catalog.ts`: samples to run and results to open. */
/** The standard benchmark's input, kept in code so the benchmark runs without the catalog. */
const PYRAMID_SAMPLE: SampleEntry = {
  id: 'pyramid',
  kind: 'video',
  dir: '/sample/pyramid/',
  name: 'Pyramid benchmark (8 frames)',
  frames: 8,
  size: [640, 360],
  thumbnail: 'frame-00.jpg',
  description: { en: '', ko: '' },
  video: { name: 'Pexels pyramid sample', times: [0, 1, 2, 4, 8, 12, 16, 20], duration: 20 },
};
/** A loaded sample with reference cameras: the input the page compares the result against. */
type LoadedSample = {
  name: string;
  credit: string;
  referenceName: string;
  reference: CameraPose[];
  /** Whether the run uses the sample's own features (`features`) or the GPU front end (`images`). */
  mode: 'images' | 'features';
};
type Reconstruction = Scene;
type Overlay = { frame: Frame; points?: Float64Array; lines?: number[]; matches?: number };
type InputKind = RunRecord['input']['kind'];
/** The worker's `graph` message of a multi-video run (`lib/multi-video-job.ts`). */
type MvGraph = {
  canonical: boolean;
  components: { index: number; keyframes: number; videos: string[]; reconstructed: boolean }[];
  matrix: { order: string[]; cells: number[][] };
};
type Benchmark = { name: string };
const STANDARD_BENCHMARK: Benchmark = { name: 'pyramid-8' };
/**
 * The focal length / image width assumed for an uncalibrated video. 1.2 (a DSLR's normal lens)
 * inverted the pyramid sample's relief; its drone lens is about 0.58, and every start from 0.5
 * to 1.0 gave the right shape (measured 2026-10-20). Phone main cameras are about 0.72.
 */
const DEFAULT_VIDEO_FOCAL = 0.8;

/**
 * Build the record of one run from the worker's result and the page's environment. A failed run
 * passes the worker's partial outcome (no reconstruction; the last snapshot's counts) and the
 * failure.
 */
/** The lab's `?fp64=float|integer`, which forces a flavor of the emulated double for a measurement;
 *  undefined lets the device pick from the adapter (TypeShade change 0073). */
/** The lab's `?bounded=1`, the diagnostic mode with fenced submissions. */
function boundedOverride(): boolean {
  if (typeof location === 'undefined') return false;
  return new URLSearchParams(location.search).get('bounded') === '1';
}

/** The lab's `?budget=<ms>`: the device's time budget per submission (TypeShade change 0074),
 *  for a measurement of how the slice length affects the page's responsiveness; the device's
 *  default (250 ms) otherwise. */
/** The lab's `?gate=phase|legacy`: the keyframe gate of a video run (`lib/keyframe-gate.ts`). */
function pageKeyframeGate(): KeyframeGateMode {
  return keyframeGateMode(typeof location === 'undefined' ? '' : location.search);
}

/** The analysis frames per second for an uploaded video: `?fps=` or the lab default. */
function pageSampleFps(): number {
  return videoSampleFps(typeof location === 'undefined' ? '' : location.search);
}

function budgetOverride(): number | undefined {
  if (typeof location === 'undefined') return undefined;
  const value = Number(new URLSearchParams(location.search).get('budget'));
  return Number.isFinite(value) && value >= 1 && value <= 60000 ? value : undefined;
}

/** The lab's `?dense=1`: the dense MVS after the sparse result (`docs/dense-mvs.md` 5.3). It
 *  turns the run option "Dense MVS" on (section 6, step 7), and a run reads it again at its start. */
function denseOverride(): boolean {
  if (typeof location === 'undefined') return false;
  return new URLSearchParams(location.search).get('dense') === '1';
}

/** The lab's `?denseMesh=1`: mesh the final cloud (`docs/dense-mvs.md` 3.13, off by default). */
function denseMeshOverride(): boolean {
  if (typeof location === 'undefined') return false;
  return new URLSearchParams(location.search).get('denseMesh') === '1';
}

/** The lab's `?denseClean=0`: no cleanup of the fused cloud (`docs/dense-mvs.md` 3.12, on by default). */
function denseCleanOverride(): boolean {
  if (typeof location === 'undefined') return true;
  return new URLSearchParams(location.search).get('denseClean') !== '0';
}

/** The page's words for the stages of the dense MVS (5.3). */
const DENSE_STAGES: Record<string, [string, string]> = {
  photometric: ['광도 단계', 'photometric pass'],
  geometric: ['기하 단계', 'geometric pass'],
  fusion: ['융합', 'fusion'],
};

/** The words of a dense status (`denseRecord`). */
const DENSE_STATUS: Record<string, [string, string]> = {
  completed: ['완료', 'completed'],
  cancelled: ['취소됨', 'cancelled'],
  failed: ['실패', 'failed'],
  skipped: ['생략됨', 'skipped'],
};

/** The dense MVS's summary in the run column (section 6): status, points, stage times, drawing. */
function DenseSummary(props: {
  ko: boolean;
  summary: ReturnType<typeof denseRecord>;
  cloud: DenseCloud | undefined;
  budget: number;
}) {
  const { ko, summary, cloud, budget } = props;
  const count = cloud?.count ?? 0;
  const stride = denseStride(count, budget);
  const cleaned = !!summary?.clean && !summary.clean.error;
  const status = summary
    ? (DENSE_STATUS[summary.status] ?? [summary.status, summary.status])[ko ? 0 : 1]
    : ko
      ? '실행 중'
      : 'running';
  return (
    <div className="lab-dense-summary" data-status={summary?.status ?? 'running'}>
      <p className="field-help">
        <strong>{ko ? '조밀 MVS' : 'Dense MVS'}</strong> · {status} · {ko ? '점 ' : ''}
        {(cleaned ? summary.clean!.after : (summary?.points ?? count)).toLocaleString()}
        {ko ? '개' : ' points'}
        {cleaned &&
          (ko
            ? ` (원본 ${summary.clean!.before.toLocaleString()})`
            : ` (raw ${summary.clean!.before.toLocaleString()})`)}
        {summary && (ko ? ` · 깊이 맵 ${summary.views}개` : ` · ${summary.views} depth maps`)}
      </p>
      {summary && summary.status !== 'skipped' && (
        <p className="field-help">
          {(ko ? '광도 ' : 'photometric ') + (summary.stageMs.photometric / 1000).toFixed(1)} s ·{' '}
          {(ko ? '기하 ' : 'geometric ') + (summary.stageMs.geometric / 1000).toFixed(1)} s ·{' '}
          {(ko ? '융합 ' : 'fusion ') + (summary.stageMs.fusion / 1000).toFixed(1)} s ·{' '}
          {(ko ? '전체 ' : 'total ') + (summary.totalMs / 1000).toFixed(1)} s
        </p>
      )}
      {stride > 1 && (
        <p className="field-help">
          {ko
            ? `표시 예산 ${budget.toLocaleString()}점: ${stride}개마다 1개를 그립니다. 내보내기는 모든 점을 씁니다.`
            : `Display budget ${budget.toLocaleString()} points: 1 point in ${stride} is drawn. The export writes every point.`}
        </p>
      )}
    </div>
  );
}

function fp64Override(): 'float' | 'integer' | undefined {
  if (typeof location === 'undefined') return undefined;
  const value = new URLSearchParams(location.search).get('fp64');
  return value === 'float' || value === 'integer' ? value : undefined;
}

function buildRunRecord(input: {
  result: any;
  failure?: RunRecord['failure'];
  adapter: any;
  commit: string;
  kind: InputKind;
  sampleName: string | null;
  frames: readonly { width: number; height: number; sourceWidth?: number; sourceHeight?: number }[];
  durationSeconds: number;
  /** Set for an uploaded video only. */
  videoSampleFps?: number;
  /** Set for an uploaded video only: the path that decoded it. */
  videoDecode?: 'worker' | 'main-thread';
  /** Set for a video sample that stores a source video: the input its frames came from. */
  videoSource?: VideoSource | 'mixed';
  benchmark?: Benchmark;
  reference: RunRecord['reference'];
  note: string;
  mainThread?: RunRecord['timing']['mainThread'];
  /** The worker's `denseResult`, for a run with the dense MVS. */
  dense?: any;
}): RunRecord {
  const { result, adapter } = input,
    dense = denseRecord(input.dense),
    total = result.stages.find((stage: any) => stage.operation === 'runTotal'),
    first = input.frames[0],
    options = result.options ?? {};
  const nav = navigator as Navigator & { deviceMemory?: number };
  return {
    format: RUN_RECORD_FORMAT,
    createdAt: new Date().toISOString(),
    site: { commit: input.commit, runtime: '/runtime/gpu.js' },
    client: {
      userAgent: navigator.userAgent.slice(0, 512),
      platform: String((navigator as any).userAgentData?.platform ?? '').slice(0, 64),
      language: navigator.language.slice(0, 32),
      hardwareConcurrency: navigator.hardwareConcurrency ?? 0,
      deviceMemoryGb: typeof nav.deviceMemory === 'number' ? nav.deviceMemory : null,
      screen: { width: screen.width, height: screen.height, dpr: devicePixelRatio },
    },
    gpu: {
      vendor: String(adapter?.vendor ?? '').slice(0, 128),
      architecture: String(adapter?.architecture ?? '').slice(0, 128),
      device: String(adapter?.device ?? '').slice(0, 128),
      description: String(adapter?.description ?? '').slice(0, 512),
      fallbackAdapter: adapter?.fallbackAdapter === true,
      limits: {
        maxStorageBufferBindingSize: adapter?.limits?.maxStorageBufferBindingSize ?? 0,
        maxBufferSize: adapter?.limits?.maxBufferSize ?? 0,
      },
    },
    input: {
      kind: input.kind,
      name: input.kind === 'sample' ? input.sampleName : null,
      frames: input.frames.length,
      width: first?.width ?? 0,
      height: first?.height ?? 0,
      sourceWidth: first?.sourceWidth ?? first?.width ?? 0,
      sourceHeight: first?.sourceHeight ?? first?.height ?? 0,
      durationSeconds: Number.isFinite(input.durationSeconds) ? input.durationSeconds : 0,
    },
    options: {
      frontEnd: options.frontEnd === 'sift' ? 'sift' : 'corner',
      focalRatio: options.focalRatio ?? 0,
      windowSize: options.windowSize ?? 0,
      ransacIterations: options.ransacIterations ?? 0,
      maxErrorPx: options.maxErrorPx ?? 0,
      bundleIterations: options.bundleIterations ?? 0,
      finalBundleIterations: options.finalBundleIterations ?? 0,
      refineSharedFocal: options.refineSharedFocal === true,
      maxFeatures: options.maxFeatures ?? 0,
      matchRatio: options.matchRatio ?? 0,
      maxDistance: options.maxDistance ?? 0,
      boundedSubmissions: options.boundedSubmissions === true,
      fp64Flavor: options.fp64Flavor === 'integer' ? 'integer' : 'float',
      guidedMatching: options.guidedMatching === true,
      ...(options.guidedMatching === true && options.guidedKernel === 'tiled'
        ? { guidedKernel: 'tiled' as const }
        : {}),
      siftFirstOctave: options.siftFirstOctave === -1 ? -1 : 0,
      clahe: options.clahe === true,
      ...(input.videoSampleFps ? { videoSampleFps: input.videoSampleFps } : {}),
      ...(input.videoDecode ? { videoDecode: input.videoDecode } : {}),
      ...(input.videoSource ? { videoSource: input.videoSource } : {}),
      ...(options.keyframeGate === 'phase' || options.keyframeGate === 'legacy'
        ? { keyframeGate: options.keyframeGate as 'phase' | 'legacy' }
        : {}),
      ...(options.features === 'orb' ? { features: 'orb' as const } : {}),
      ...(typeof result.keyframes === 'number' ? { keyframes: result.keyframes } : {}),
      ...(typeof options.klt === 'boolean' ? { klt: options.klt } : {}),
      ...(options.klt && options.kltSetup ? { kltSetup: options.kltSetup } : {}),
    },
    result: {
      registeredViews: result.reconstruction?.views.length ?? result.latest?.views ?? 0,
      landmarks: result.reconstruction?.landmarks.length ?? result.latest?.landmarks ?? 0,
      observations:
        result.reconstruction?.landmarks.reduce(
          (sum: number, point: any) => sum + point.observations.length,
          0,
        ) ??
        result.latest?.observations ??
        0,
      meanReprojectionErrorPx: Number.isFinite(result.diagnostics?.meanReprojectionErrorPx)
        ? result.diagnostics.meanReprojectionErrorPx
        : null,
      unregistered: result.unregistered?.length ?? 0,
      matchedPairs: result.matchedPairs ?? 0,
      ...(result.guided
        ? { guided: { pairs: result.guided.pairs, addedMatches: result.guided.addedMatches } }
        : {}),
      featureCounts: (result.featureCounts ?? []).slice(0, 2048),
      ...(result.klt ? { klt: result.klt } : {}),
    },
    timing: {
      totalMs: total?.elapsedMs ?? 0,
      reusedDevice: total?.reusedDevice === true,
      stages: result.summary ?? summarizeStages(result.stages ?? []),
      frameMs: (
        result.frameMs ??
        (result.stages ?? [])
          .filter((stage: any) => stage.operation === 'keyframe')
          .map((stage: any) => stage.elapsedMs)
      ).slice(0, 2048),
      // The dense MVS's kernels and chunk rates follow the sparse run's (5.3).
      kernels: (input.dense?.kernels ?? result.kernels ?? [])
        .slice(0, 256)
        .map((name: unknown) => String(name).slice(0, 64)),
      chunks: chunkRecord(input.dense?.chunks ?? result.chunks),
      budgetMs: Number.isFinite(result.budgetMs) && result.budgetMs > 0 ? result.budgetMs : null,
      ...(input.mainThread ? { mainThread: input.mainThread } : {}),
    },
    transfers: {
      applicationUploadBytes: total?.applicationUploadBytes ?? 0,
      applicationReadbackBytes: total?.applicationReadbackBytes ?? 0,
    },
    benchmark: input.benchmark
      ? { name: input.benchmark.name, pass: total?.reusedDevice ? 'warm' : 'cold' }
      : null,
    reference: input.reference,
    failure: input.failure ?? null,
    note: input.note.slice(0, RUN_RECORD_MAX_NOTE),
    ...(dense ? { dense } : {}),
  };
}

/**
 * The record of a multi-video run (`docs/multi-video.md` 6.4): the single-video record built from
 * the sums of the run, plus the optional multi-video fields. `result` is the worker's `result`
 * message; it is undefined for a run that failed before it.
 */
function buildMultiRunRecord(input: {
  result: any;
  graph: any;
  models: ReadonlyMap<number, Scene>;
  meanErrors: ReadonlyMap<number, number | null>;
  adapter: any;
  commit: string;
  rows: readonly VideoRow[];
  query: ReturnType<typeof multiVideoQuery>;
  maxFeatures: number;
  firstOctave: -1 | 0;
  frames: number;
  first?: { width: number; height: number; sourceWidth?: number; sourceHeight?: number };
  durationSeconds: number;
  videoDecode?: 'worker' | 'main-thread';
  videoSource?: VideoSource | 'mixed';
  videoSampleFps?: number;
  mainThread?: RunRecord['timing']['mainThread'];
  setName?: string | null;
  note: string;
  failure?: RunRecord['failure'];
}): RunRecord {
  const { result, models } = input;
  const scenes = [...models.values()];
  const registered = scenes.reduce((n, m) => n + m.views.length, 0);
  const landmarks = scenes.reduce((n, m) => n + m.landmarks.length, 0);
  const observations = scenes.reduce(
    (n, m) => n + m.landmarks.reduce((k, p) => k + p.observations.length, 0),
    0,
  );
  const keyframes: number = result?.keyframeCount ?? 0;
  const pseudo = {
    stages: [
      {
        operation: 'runTotal',
        elapsedMs: result?.totalMs ?? 0,
        reusedDevice: false,
        applicationUploadBytes: 0,
        applicationReadbackBytes: 0,
      },
    ],
    summary: result?.stages ?? {},
    // The numbers of `LAB_OPTIONS` in the worker that a multi-video map uses.
    options: {
      frontEnd: 'sift',
      focalRatio: input.rows[0]?.focal ?? 0,
      windowSize: 0,
      ransacIterations: 500,
      maxErrorPx: 3,
      bundleIterations: 4,
      finalBundleIterations: 8,
      refineSharedFocal: true,
      maxFeatures: input.maxFeatures,
      matchRatio: 0.8,
      maxDistance: 1e19,
      siftFirstOctave: input.firstOctave,
      clahe: false,
      keyframeGate: 'phase',
    },
    keyframes,
    latest: { views: registered, landmarks, observations },
    diagnostics: { meanReprojectionErrorPx: input.meanErrors.get(0) ?? Number.NaN },
    unregistered: Array.from({ length: Math.max(0, keyframes - registered) }),
    matchedPairs: input.graph?.stats.verifiedPairs ?? 0,
    featureCounts: result?.featureCounts ?? [],
    frameMs: [],
    kernels: [],
    chunks: {},
  };
  const allSamples = input.rows.length > 0 && input.rows.every((r) => r.sampleId);
  const base = buildRunRecord({
    result: pseudo,
    failure: input.failure,
    adapter: input.adapter,
    commit: input.commit,
    kind: allSamples ? 'sample' : 'video',
    sampleName: allSamples ? (input.setName ?? null) : null,
    frames: Array.from({ length: Math.min(input.frames, RUN_RECORD_MAX_FRAMES) }, () => ({
      width: input.first?.width ?? 0,
      height: input.first?.height ?? 0,
      sourceWidth: input.first?.sourceWidth,
      sourceHeight: input.first?.sourceHeight,
    })),
    durationSeconds: input.durationSeconds,
    videoSampleFps: input.videoSampleFps,
    videoDecode: input.videoDecode,
    videoSource: input.videoSource,
    reference: null,
    note: input.note,
    mainThread: input.mainThread,
  });
  const summary = result?.summary;
  const videos = (summary?.videos ?? []).map((v: any) => ({
    ...v,
    // The typed name of an upload is never sent; a lab sample is public (the record, 6.4).
    label: v.label ?? null,
  }));
  return {
    ...base,
    input: { ...base.input, ...(videos.length ? { videos } : {}) },
    options: { ...base.options, multiVideo: multiVideoOptions({ query: input.query }) },
    result: {
      ...base.result,
      ...(summary
        ? {
            components: summary.components,
            crossVideoEdges: summary.crossVideoEdges,
            viewGraph: summary.viewGraph,
            notConnected: summary.notConnected,
          }
        : {}),
    },
  };
}

/** The time budget and one line per chunk key of a record, or nothing when it has none. */
function ChunkSummary({
  record,
  ko,
  className,
}: {
  record: RunRecord;
  ko: boolean;
  className: string;
}) {
  const lines = chunkLines(record.timing.chunks);
  const mainThread = mainThreadLine(record.timing.mainThread);
  if (mainThread) lines.push(mainThread);
  const dense = denseLine(record.dense);
  if (dense) lines.push(dense);
  const klt = kltLine(record.result.klt);
  if (klt) lines.push(klt);
  if (!lines.length) return null;
  return (
    <p className={className}>
      {(ko ? '시간 예산 ' : 'Time budget ') + (record.timing.budgetMs ?? '?') + ' ms'}
      {lines.map((line) => (
        <span key={line}>
          <br />
          {line}
        </span>
      ))}
    </p>
  );
}

/** The record's `reference` field from an alignment, or null without a sample. */
function referenceRecord(
  sample: LoadedSample | undefined,
  alignment: PoseAlignment | undefined,
): RunRecord['reference'] {
  if (!sample || !alignment) return null;
  const measure = (value: number) => (Number.isFinite(value) ? value : null);
  return {
    name: sample.name,
    status: alignment.status,
    commonViews: alignment.commonViews,
    medianRotationDeg: measure(alignment.rotationDeg.median),
    maxRotationDeg: measure(alignment.rotationDeg.max),
    medianCenterOfRadius: measure(alignment.centerOfRadius.median),
    maxCenterOfRadius: measure(alignment.centerOfRadius.max),
  };
}

/** Align the registered cameras of a scene to the sample's reference cameras. */
function compareToReference(sample: LoadedSample, views: Scene['views']): PoseAlignment {
  return alignPoses(
    views.map((view) => ({ id: view.id, R: view.pose.R, t: view.pose.t })),
    sample.reference,
  );
}

function AnalysisFrame({
  overlay,
  scene,
  annotate,
  transparent,
  visible,
  children,
}: {
  overlay?: Overlay;
  scene?: Scene;
  /** Draw the detections, matches and depth points over the frame. */
  annotate: boolean;
  /** Draw the marks only, on a transparent canvas, so the video shows through. */
  transparent: boolean;
  /** The marks match the video's current frame. */
  visible: boolean;
  /** The source video: it fills the same box under the canvas. */
  children: ReactNode;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  const depths = depthPoints(scene, overlay?.frame.id ?? -1);
  const range = depthRange(depths);
  useEffect(() => {
    if (!overlay || !ref.current) return;
    const { frame, points } = overlay;
    const canvas = ref.current;
    canvas.width = frame.width;
    canvas.height = frame.height;
    const ctx = canvas.getContext('2d')!;
    if (transparent) ctx.clearRect(0, 0, canvas.width, canvas.height);
    else {
      ctx.putImageData(
        new ImageData(new Uint8ClampedArray(frame.pixels), frame.width, frame.height),
        0,
        0,
      );
    }
    if (!annotate) return;
    if (!depths.length && overlay.lines) {
      ctx.strokeStyle = '#ffbf69';
      ctx.lineWidth = 1;
      for (let i = 0; i < overlay.lines.length; i += 4) {
        ctx.beginPath();
        ctx.moveTo(overlay.lines[i], overlay.lines[i + 1]);
        ctx.lineTo(overlay.lines[i + 2], overlay.lines[i + 3]);
        ctx.stroke();
      }
    }
    ctx.fillStyle = '#a3a3a3';
    for (let i = 0; i < (points?.length ?? 0); i += 2) {
      ctx.beginPath();
      ctx.arc(points![i], points![i + 1], 2, 0, 2 * Math.PI);
      ctx.fill();
    }
    for (const point of depths) {
      ctx.fillStyle =
        'rgb(' +
        depthColor(point.depth, range)
          .map((v) => Math.round(v * 255))
          .join(',') +
        ')';
      ctx.beginPath();
      ctx.arc(point.x, point.y, 3, 0, 2 * Math.PI);
      ctx.fill();
    }
  }, [overlay, scene, annotate, transparent]);
  // While a run is busy the canvas draws the decoded analysis frame itself, so the marks sit on
  // the pixels they were measured on, whatever frame the throttled preview video shows. Once the
  // run is idle the video is seeked exactly to the frame and the canvas draws the marks alone.
  // Both fill one box with object-fit: contain, so the letterboxing is the same.
  return (
    <div className="sfm-stage">
      {children}
      <canvas
        ref={ref}
        className="sfm-analysis-canvas"
        data-shown={!!overlay && visible}
        data-mode={transparent ? 'marks' : 'frame'}
        aria-label="Actual video frame with reconstructed points colored by camera depth"
      />
    </div>
  );
}

/** One statistic: a small label over a large value, "—" while the value is missing. */
function Tile({
  id,
  label,
  value,
  hint,
}: {
  id: string;
  label: string;
  value?: string | number;
  hint?: string;
}) {
  const body = (
    <div className="sfm-tile" data-tile={id} tabIndex={hint ? 0 : undefined}>
      <span className="sfm-tile-label">{label}</span>
      <strong className="sfm-tile-value">{value ?? '—'}</strong>
    </div>
  );
  return hint ? <Tooltip title={hint}>{body}</Tooltip> : body;
}

export default function SfmLab({ locale = 'en' }: { locale?: 'en' | 'ko' }) {
  const [dark, setDark] = useState(false);
  useEffect(() => {
    const update = () => setDark(document.documentElement.dataset.theme === 'dark');
    update();
    const observer = new MutationObserver(update);
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ['data-theme'],
    });
    return () => observer.disconnect();
  }, []);
  return (
    <ConfigProvider theme={{ algorithm: dark ? theme.darkAlgorithm : theme.defaultAlgorithm }}>
      <SfmLabContent locale={locale} dark={dark} />
    </ConfigProvider>
  );
}

function SfmLabContent({ locale = 'en', dark = false }: { locale?: 'en' | 'ko'; dark?: boolean }) {
  const ko = locale === 'ko';
  const [file, setFile] = useState<File>();
  const [videoURL, setVideoURL] = useState('');
  /** The paused video shows the selected frame, so the marks drawn for it line up. */
  const [videoAligned, setVideoAligned] = useState(false);
  const [source, setSource] = useState<{
    name: string;
    width: number;
    height: number;
    duration?: number;
  }>();
  const [frames, setFrames] = useState<string[]>([]);
  const [prepared, setPrepared] = useState<Frame[]>([]);
  const [runFrames, setRunFrames] = useState<Frame[]>([]);
  const [reconstruction, setReconstruction] = useState<Reconstruction>();
  const [diagnostics, setDiagnostics] = useState<any>();
  const [unregistered, setUnregistered] = useState<{ viewId: number; reason: string }[]>([]);
  const [stages, setStages] = useState<any[]>([]);
  const [overlay, setOverlay] = useState<Overlay>();
  const [status, setStatus] = useState('');
  const [busy, setBusy] = useState(false);
  const [focal, setFocal] = useState(DEFAULT_VIDEO_FOCAL);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [replaying, setReplaying] = useState(false);
  const [frameStep, setFrameStep] = useState(0);
  const [frameFinished, setFrameFinished] = useState(false);
  const [frameCounts, setFrameCounts] = useState({ features: 0, matches: 0, inliers: 0 });
  const [playTime, setPlayTime] = useState(0);
  const [analysisTime, setAnalysisTime] = useState(0);
  const [processed, setProcessed] = useState(0);
  const [skipped, setSkipped] = useState(0);
  /** Analysis frames per second of the last video run; 0 when the input is not sampled. */
  const [sampleFps, setSampleFps] = useState(0);
  const [frameMs, setFrameMs] = useState(0);
  const [totalMs, setTotalMs] = useState(0);
  const [complete, setComplete] = useState(false);
  const [viewerState, setViewerState] = useState<ViewerState>({ status: 'initializing' });
  const [record, setRecord] = useState<RunRecord>();
  const [consent, setConsent] = useState(false);
  const [note, setNote] = useState('');
  const [sendState, setSendState] = useState<SendState>({ status: 'idle' });
  // "Also send the result files": off by default, since the data may come from the visitor's own
  // photos or video. The collector says whether it has a store for them.
  const [sendFiles, setSendFiles] = useState(false);
  const [filesAvailable, setFilesAvailable] = useState(false);
  const sendFilesRef = useRef(false);
  sendFilesRef.current = sendFiles;
  useEffect(() => {
    let live = true;
    void resultFilesAvailable().then((available) => live && setFilesAvailable(available));
    return () => {
      live = false;
    };
  }, []);
  const [benchmarkRunning, setBenchmarkRunning] = useState(false);
  /**
   * A multi-video run (`docs/multi-video.md` 6): the rows of the upload list (two or more videos),
   * the view graph the worker posted, the model of each component and the component on show.
   */
  const [mvRows, setMvRows] = useState<VideoRow[]>([]);
  const [mvGraph, setMvGraph] = useState<MvGraph>();
  const [mvSelected, setMvSelected] = useState(0);
  const [mvProgress, setMvProgress] = useState(0);
  /** The camera of the multi-video scene the visitor picked in the viewer. */
  const [mvView, setMvView] = useState<number>();
  const mvModels = useRef(new Map<number, Scene>());
  const mvRowsRef = useRef<VideoRow[]>([]);
  mvRowsRef.current = mvRows;
  const mvVideoColors = new Map(
    mvRows.filter((r) => r.videoId && !r.duplicate).map((r, i) => [r.videoId!, componentColor(i)]),
  );
  const [sample, setSample] = useState<LoadedSample>();
  const [catalog, setCatalog] = useState<(SampleEntry | ResultEntry | VideoSetEntry)[]>([
    PYRAMID_SAMPLE,
  ]);
  /** How a calibrated sample with features runs: the GPU front end, or the dataset's features. */
  /** The long side of an analysis frame: uploaded video is decoded at it, and a sample loads the
   *  preview set of that edge when it ships one. The standard benchmark stays at 640. */
  const [analysisEdge, setAnalysisEdge] = useState<AnalysisEdge>(640);
  const analysisEdgeRef = useRef<AnalysisEdge>(640);
  analysisEdgeRef.current = analysisEdge;
  const [alignment, setAlignment] = useState<PoseAlignment>();
  /** The viewer's camera: look through the selected view (`cameraView`) or refit (`fitNonce`). */
  const [cameraView, setCameraView] = useState<{ id: number; nonce: number }>();
  const [fitNonce, setFitNonce] = useState(0);
  /** The viewer's display options. The background follows the page theme until the visitor
   *  picks one. */
  const [showCameras, setShowCameras] = useState(true);
  const [showPhoto, setShowPhoto] = useState(true);
  const [showOverlay, setShowOverlay] = useState(true);
  const [background, setBackground] = useState<'auto' | 'light' | 'dark'>('auto');
  const lightBackground = background === 'auto' ? !dark : background === 'light';
  const [navigation, setNavigation] = useState<'orbit' | 'fly'>('orbit');
  /** With the option on, selecting a frame also moves the 3D view to its camera. */
  const [followCamera, setFollowCamera] = useState(false);
  /** The run option "Dense MVS" (`docs/dense-mvs.md` section 6): off by default, on with
   *  `?dense=1`. */
  const [denseEnabled, setDenseEnabled] = useState(false);
  const denseEnabledRef = useRef(false);
  denseEnabledRef.current = denseEnabled;
  useEffect(() => {
    if (denseOverride()) setDenseEnabled(true);
  }, []);
  /** The dense cloud of the last run, the viewer's switch and the summary of `denseResult`. */
  const [denseCloud, setDenseCloud] = useState<DenseCloud>();
  /** The mesh of the final cloud (`docs/dense-mvs.md` 3.13). Hidden until the Display switch. */
  const [denseMesh, setDenseMesh] = useState<DenseMeshData>();
  const [showMesh, setShowMesh] = useState(false);
  const [showDense, setShowDense] = useState(true);
  const [denseSummary, setDenseSummary] = useState<ReturnType<typeof denseRecord>>();
  const pointBudget =
    typeof navigator === 'undefined'
      ? densePointBudget(0)
      : densePointBudget(navigator.maxTouchPoints);
  /** The catalog entry whose frames are loaded, for the card's highlight. */
  const [loadedSampleId, setLoadedSampleId] = useState<string>();
  /** The remote samples whose thumbnail failed to load: not published to the bucket yet. */
  const [missingSamples, setMissingSamples] = useState<ReadonlySet<string>>(new Set());
  /** The samples whose card asks for the dataset's SIFT features (the mapper alone). */
  const [featureSamples, setFeatureSamples] = useState<ReadonlySet<string>>(new Set());
  /** The sample being loaded and how far: files read of the files it needs. */
  const [loading, setLoading] = useState<{ id: string; done: number; total: number }>();
  /** The result in the viewer, opened from a card or from files: what it holds, for the tags. */
  const [resultInfo, setResultInfo] = useState<ResultInfo>();
  const resultInput = useRef<HTMLInputElement>(null);
  const commitRef = useRef('unknown');
  const inputKindRef = useRef<InputKind>('video');
  const sampleRef = useRef<LoadedSample | undefined>(undefined);
  /** The loaded video sample that stores a source video: whether this browser decodes it. A run
   *  reads the source video when it does, and the JPEG frames in `prepared` when it does not. */
  const videoSourceRef = useRef<{ url: string; decodable: boolean } | undefined>(undefined);
  const profileRef = useRef<Profile>('video');
  const loadSample = (loaded: LoadedSample | undefined) => {
    sampleRef.current = loaded;
    profileRef.current = loaded ? 'photo' : 'video';
    setSample(loaded);
    setAlignment(undefined);
  };
  const consentRef = useRef(false);
  const noteRef = useRef('');
  const adapterRef = useRef<any>(null);
  consentRef.current = consent;
  noteRef.current = note;
  useEffect(() => {
    fetch('/sample/index.json')
      .then((response) => (response.ok ? response.json() : undefined))
      .then((index: SampleCatalog | undefined) => {
        if (index?.format === 'typecv-lab-samples-v1' && Array.isArray(index.samples))
          setCatalog(index.samples);
      })
      .catch(() => {});
    fetch('/version.json')
      .then((response) => (response.ok ? response.json() : undefined))
      .then((version) => {
        if (typeof version?.commit === 'string') commitRef.current = version.commit.slice(0, 64);
      })
      .catch(() => {});
  }, []);
  const viewerStateRef = useRef(viewerState);
  const viewerWaiter = useRef<((state: ViewerState) => void) | undefined>(undefined);
  const onViewerState = useCallback((state: ViewerState) => {
    viewerStateRef.current = state;
    setViewerState(state);
    viewerWaiter.current?.(state);
  }, []);
  const waitForViewer = (signal: AbortSignal) => {
    signal.throwIfAborted();
    const current = viewerStateRef.current;
    // A viewer that failed to initialize (Safari after a GPU process loss has no WebGL context)
    // does not stop the run: the map stays empty and the alert says so.
    if (current.status === 'ready' || current.status === 'error') return Promise.resolve();
    return new Promise<void>((resolve, reject) => {
      const cleanup = () => {
        viewerWaiter.current = undefined;
        signal.removeEventListener('abort', abort);
      };
      const abort = () => {
        cleanup();
        reject(new DOMException('Cancelled', 'AbortError'));
      };
      viewerWaiter.current = (state) => {
        if (state.status === 'initializing') return;
        cleanup();
        resolve();
      };
      signal.addEventListener('abort', abort, { once: true });
    });
  };
  const videoRef = useRef<HTMLVideoElement>(null);
  const workerRef = useRef<Worker | undefined>(undefined);
  const jobRef = useRef(0);
  const runFocal = useRef(focal);
  const abortRef = useRef<AbortController | undefined>(undefined);
  /** Set while the dense MVS runs: the cancel that stops it and keeps the sparse result (5.3). */
  const denseCancelRef = useRef<(() => void) | undefined>(undefined);
  const [denseRunning, setDenseRunning] = useState(false);
  /** The dense MVS's messages of the last run (5.3). The points are the full arrays of the
   *  export. The viewer draws a subset of them (section 6). */
  const denseRef = useRef<{ depth: any[]; points: any[] }>({ depth: [], points: [] });
  const denseRunRef = useRef(0);
  const clearDense = () => {
    denseRef.current = { depth: [], points: [] };
    denseRunRef.current++;
    setDenseCloud(undefined);
    setDenseMesh(undefined);
    setDenseSummary(undefined);
  };
  const terminateWorker = () => {
    workerRef.current?.terminate();
    workerRef.current = undefined;
  };
  const selectedFrame = runFrames[selectedIndex];
  const displayedIndex = runFrames.findIndex((frame) => frame.id === overlay?.frame.id);
  const selectedView = reconstruction?.views.find((view) => view.id === overlay?.frame.id);
  const previousView = reconstruction?.views.find(
    (view) => view.id === runFrames[displayedIndex - 1]?.id,
  );
  const movement = cameraMotion(previousView, selectedView);
  // The frame the tiles describe: the analysed one while marks exist, else the selected one.
  const frameSize = overlay?.frame ?? selectedFrame;
  const shownFeatures = overlay?.points ? overlay.points.length / 2 : frameCounts.features;
  const shownMatches = overlay?.matches ?? frameCounts.matches;
  const currentDepths = depthPoints(
    mvRows.length ? undefined : reconstruction,
    overlay?.frame.id ?? -1,
  );
  const currentRange = depthRange(currentDepths);
  useEffect(() => {
    if (!replaying || !complete || overlay?.frame.id !== selectedFrame?.id) return;
    const timer = window.setTimeout(() => {
      if (selectedIndex >= runFrames.length - 1) setReplaying(false);
      else setSelectedIndex(selectedIndex + 1);
    }, 250);
    return () => window.clearTimeout(timer);
  }, [replaying, complete, selectedIndex, runFrames.length, overlay?.frame.id]);
  const syncAligned = () => {
    const video = videoRef.current;
    setVideoAligned(
      !!video &&
        !!selectedFrame &&
        video.paused &&
        !video.seeking &&
        Math.abs(video.currentTime - selectedFrame.time) < 0.02,
    );
  };
  useEffect(() => {
    if (busy || !complete || !selectedFrame) return;
    const abort = new AbortController();
    const load = async () => {
      const original = prepared.find((frame) => frame.id === selectedFrame.id);
      if (selectedFrame.metrics) {
        setFrameCounts(selectedFrame.metrics);
        setFrameStep(reconstruction?.views.some((view) => view.id === selectedFrame.id) ? 4 : 2);
        setFrameFinished(true);
      }
      if (original) {
        setOverlay({ frame: original });
        return;
      }
      if (!file || !videoURL || !videoRef.current) return;
      const video = videoRef.current;
      video.pause();
      setVideoAligned(false);
      await seekFrame(video, selectedFrame.time, abort.signal);
      syncAligned();
      const { decodeVideoFrames } = await import('../lib/sfm-video');
      for await (const decoded of decodeVideoFrames(
        file,
        abort.signal,
        selectedFrame.time,
        analysisEdgeRef.current,
      )) {
        if (!abort.signal.aborted) setOverlay({ frame: { ...selectedFrame, ...decoded } });
      }
    };
    load().catch((error) => {
      if (!abort.signal.aborted) setStatus(String(error));
    });
    return () => abort.abort();
  }, [selectedIndex, complete, busy, prepared, videoURL]);
  useEffect(() => {
    if (!followCamera || busy || !selectedFrame) return;
    if (!reconstruction?.views.some((view) => view.id === selectedFrame.id)) return;
    setCameraView((previous) => ({ id: selectedFrame.id, nonce: (previous?.nonce ?? 0) + 1 }));
    // The reconstruction is read, not watched: a snapshot must not re-aim the view.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [followCamera, selectedFrame?.id, busy]);
  useEffect(
    () => () => {
      abortRef.current?.abort();
      terminateWorker();
    },
    [],
  );
  useEffect(() => {
    if (!file || file.name.toLowerCase().endsWith('.json')) {
      setVideoURL('');
      return;
    }
    const url = URL.createObjectURL(file);
    setVideoURL(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  function reset() {
    abortRef.current?.abort();
    terminateWorker();
    setPrepared([]);
    setRunFrames([]);
    setFrames([]);
    setOverlay(undefined);
    setReconstruction(undefined);
    clearDense();
    setDiagnostics(undefined);
    setStages([]);
    setUnregistered([]);
    setSource(undefined);
    setLoadedSampleId(undefined);
    setResultInfo(undefined);
    loadSample(undefined);
    videoSourceRef.current = undefined;
    setStatus('');
    setComplete(false);
    setProcessed(0);
    setSkipped(0);
    setPlayTime(0);
    setAnalysisTime(0);
    setFrameMs(0);
    setTotalMs(0);
    setReplaying(false);
    setSelectedIndex(0);
    setFrameStep(0);
    setRecord(undefined);
    setSendState({ status: 'idle' });
    clearMulti();
  }
  /** Forget the rows and the result of a multi-video run. */
  function clearMulti() {
    setMvRows([]);
    setMvGraph(undefined);
    setMvSelected(0);
    setMvProgress(0);
    mvModels.current.clear();
  }
  /** Forget the result of a multi-video run and keep the rows. */
  function clearMultiResult() {
    setMvGraph(undefined);
    setMvSelected(0);
    setMvProgress(0);
    mvModels.current.clear();
    setReconstruction(undefined);
    setRecord(undefined);
    setComplete(false);
    setMvRows((rows) =>
      rows.map((r) => ({
        ...r,
        keyframes: undefined,
        registeredViews: undefined,
        component: undefined,
        tags: undefined,
        crossInliers: undefined,
      })),
    );
  }
  async function sendRecord(current: RunRecord | undefined = record) {
    if (!current) return;
    const problem = runRecordProblem(current);
    if (problem) {
      setSendState({ status: 'failed', message: 'Invalid record field: ' + problem });
      return;
    }
    setSendState({ status: 'sending' });
    const files = sendFiles && filesAvailable ? await planFiles(resultFiles()) : undefined;
    setSendState(
      await postRecord(current, ko, files, (done, total) =>
        setSendState({ status: 'sending', progress: { done, total } }),
      ),
    );
  }
  /** The files that exist for this run, as the lab's downloads make them. */
  function resultFiles(): ResultFilePayload[] {
    const files: ResultFilePayload[] = [];
    if (reconstruction) {
      files.push({ name: 'typecv-reconstruction.json', data: reconstructionText('json') });
      files.push({ name: 'typecv-reconstruction.ply', data: reconstructionText('ply') });
    }
    if (denseRef.current.points.length)
      files.push({ name: 'typecv-dense.ply', data: densePly(denseRef.current.points) });
    if (denseMesh) files.push({ name: 'typecv-mesh.ply', data: denseMeshPly(denseMesh) });
    return files;
  }
  function saveRecord() {
    if (!record) return;
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(record, null, 2)], { type: 'application/json' }),
    );
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = 'typecv-run-record.json';
    anchor.click();
    URL.revokeObjectURL(url);
  }
  async function loadPreparedFrames() {
    if (!file) return;
    setBusy(true);
    setReplaying(false);
    try {
      if (file.size > 64 * 1024 * 1024) throw new RangeError('Frame bundle exceeds 64 MiB.');
      const input = parsePreparedVideo(await file.text());
      inputKindRef.current = 'prepared';
      loadSample(undefined);
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
      const sampled: Frame[] = [];
      for (const frame of input.frames) {
        const bitmap = await createImageBitmap(await (await fetch(frame.image)).blob());
        try {
          if (bitmap.width !== frame.width || bitmap.height !== frame.height)
            throw new RangeError('Encoded frame dimensions differ.');
          canvas.width = frame.width;
          canvas.height = frame.height;
          ctx.drawImage(bitmap, 0, 0);
          sampled.push({
            id: frame.id,
            width: frame.width,
            height: frame.height,
            time: frame.time,
            pixels: ctx.getImageData(0, 0, frame.width, frame.height).data,
            sourceWidth: input.source.width,
            sourceHeight: input.source.height,
            image: frame.image,
          });
        } finally {
          bitmap.close();
        }
      }
      setSource(input.source);
      setPrepared(sampled);
      setFrames(input.frames.map((frame) => frame.image));
      setStatus(
        ko ? '분석 프레임이 준비됐습니다.' : sampled.length + ' analysis frames are ready.',
      );
    } catch (error) {
      setStatus(String(error));
    } finally {
      setBusy(false);
    }
  }
  /** A `video` entry of the catalog: frames cut from footage, no calibration, no reference. */
  async function loadVideoSample(entry: SampleEntry): Promise<Frame[]> {
    reset();
    setFile(undefined);
    setBusy(true);
    inputKindRef.current = 'sample';
    try {
      const times = entry.video?.times ?? [];
      if (times.length !== entry.frames) throw new Error('The sample lists no frame times.');
      const sampled: Frame[] = [],
        thumbs: string[] = [];
      const canvas = document.createElement('canvas'),
        ctx = canvas.getContext('2d', { willReadFrequently: true })!;
      for (let i = 0; i < times.length; i++) {
        const url = entry.dir + 'frame-' + String(i).padStart(2, '0') + '.jpg';
        const response = await fetch(url);
        if (!response.ok) throw new Error('Could not load sample frame ' + i);
        const bitmap = await createImageBitmap(await response.blob());
        canvas.width = bitmap.width;
        canvas.height = bitmap.height;
        ctx.drawImage(bitmap, 0, 0);
        bitmap.close();
        sampled.push({
          id: i,
          width: canvas.width,
          height: canvas.height,
          time: times[i],
          pixels: ctx.getImageData(0, 0, canvas.width, canvas.height).data,
          image: url,
        });
        thumbs.push(url);
        setLoading({ id: entry.id, done: i + 1, total: times.length });
      }
      // A sample that stores its source video: the run decodes it at the analysis edge when this
      // browser can, and uses the frames loaded above when it cannot.
      const url = sourceUrl(entry);
      const decodable = url
        ? await (
            await import('../lib/sfm-video')
          ).sourceDecodable(url, new AbortController().signal)
        : false;
      videoSourceRef.current = url ? { url, decodable } : undefined;
      setPrepared(sampled);
      setFrames(thumbs);
      setSource({
        name: entry.video?.name ?? entry.name,
        width: decodable ? entry.source!.width : sampled[0].width,
        height: decodable ? entry.source!.height : sampled[0].height,
        duration: entry.video?.duration ?? times[times.length - 1],
      });
      const tag = url
        ? ko
          ? decodable
            ? ' (원본 영상)'
            : ' (프레임 파일)'
          : decodable
            ? ' (source video)'
            : ' (frame files)'
        : '';
      setStatus(
        ko
          ? '실제 영상에서 추출한 ' + sampled.length + '개 샘플 프레임입니다.' + tag
          : sampled.length + ' sample frames from real footage are ready.' + tag,
      );
      return sampled;
    } catch (error) {
      setStatus(String(error));
      return [];
    } finally {
      setBusy(false);
    }
  }
  /**
   * A `photos` entry of the catalog (the COLMAP south-building sample and the others): calibrated
   * photographs with the dataset's poses as the reference. `images` runs the GPU front end on the
   * photographs; `features` feeds the dataset's own SIFT keypoints and raw matches to the GPU
   * mapper, when the sample ships them.
   */
  async function loadPhotoSample(entry: SampleEntry, mode: LoadedSample['mode']): Promise<Frame[]> {
    reset();
    setFile(undefined);
    setBusy(true);
    inputKindRef.current = 'sample';
    try {
      // The preview set of the chosen edge, when the sample ships one; 640 otherwise.
      const edge = entry.edges?.includes(analysisEdge) ? analysisEdge : 640;
      const base = edge === 640 ? entry.dir : entry.dir + edge + '/';
      const response = await fetch(base + 'sample.json');
      if (!response.ok) throw new Error('Could not load the sample ' + entry.id + '.');
      const manifest = (await response.json()) as LabSample;
      if (manifest.format !== 'typecv-lab-sample-v1') throw new Error('Unknown sample format.');
      let keypoints: Float32Array | undefined, matches: Uint16Array | undefined;
      if (mode === 'features' && !manifest.features) mode = 'images';
      if (mode === 'features') {
        if (!manifest.features) throw new Error('The sample ships no features.');
        const binary = await fetch(base + manifest.features.file);
        if (!binary.ok) throw new Error('Could not load the sample features.');
        const bytes = await binary.arrayBuffer();
        keypoints = new Float32Array(bytes, 0, manifest.features.keypointBytes / 4);
        matches = new Uint16Array(bytes, manifest.features.keypointBytes);
      }
      const sampled: Frame[] = [],
        thumbs: string[] = [];
      const canvas = document.createElement('canvas'),
        ctx = canvas.getContext('2d', { willReadFrequently: true })!;
      for (const frame of manifest.frames) {
        const url = base + frame.file;
        const image = await fetch(url);
        if (!image.ok) throw new Error('Could not load sample frame ' + frame.id);
        const bitmap = await createImageBitmap(await image.blob());
        if (bitmap.width !== frame.width || bitmap.height !== frame.height)
          throw new RangeError('Sample frame dimensions differ.');
        canvas.width = frame.width;
        canvas.height = frame.height;
        ctx.drawImage(bitmap, 0, 0);
        bitmap.close();
        let precomputed: Frame['precomputed'];
        if (keypoints && matches && manifest.features) {
          const block = manifest.features.keypoints.find((entry) => entry.id === frame.id);
          if (!block) throw new Error('Sample frame ' + frame.id + ' has no keypoints.');
          precomputed = {
            keypoints: Float64Array.from(
              keypoints.subarray(block.offset, block.offset + block.count * 2),
            ),
            pairs: manifest.features.pairs
              .filter((pair) => pair.second === frame.id)
              .map((pair) => ({
                first: pair.first,
                matches: Uint32Array.from(
                  matches!.subarray(pair.offset, pair.offset + pair.count * 2),
                ),
              })),
          };
        }
        sampled.push({
          id: frame.id,
          width: frame.width,
          height: frame.height,
          time: frame.time,
          pixels: ctx.getImageData(0, 0, frame.width, frame.height).data,
          sourceWidth: frame.sourceWidth,
          sourceHeight: frame.sourceHeight,
          camera: frame.camera,
          precomputed,
          image: url,
        });
        thumbs.push(url);
        setLoading({ id: entry.id, done: sampled.length, total: manifest.frames.length });
      }
      setPrepared(sampled);
      setFrames(thumbs);
      setSource({
        name: manifest.name + (mode === 'features' ? ' (COLMAP SIFT)' : ''),
        width: manifest.source.width,
        height: manifest.source.height,
        duration: manifest.frames.at(-1)?.time ?? 0,
      });
      loadSample({
        name: manifest.name + (mode === 'features' ? ' (COLMAP SIFT)' : ''),
        credit: manifest.credit,
        referenceName: manifest.reference,
        reference: manifest.frames.map((frame) => ({ id: frame.id, ...frame.reference })),
        mode,
      });
      const size = sampled[0] ? sampled[0].width + ' × ' + sampled[0].height : '';
      setStatus(
        mode === 'features'
          ? ko
            ? sampled.length +
              '장의 사진(' +
              size +
              ')과 데이터셋의 SIFT 특징점·매칭이 준비됐습니다. GPU는 매퍼만 실행하고, 결과를 참조 카메라와 비교합니다.'
            : sampled.length +
              ' photographs at ' +
              size +
              " with the dataset's own SIFT keypoints and matches are ready. The GPU runs the mapper alone; the result is compared with the reference cameras."
          : ko
            ? sampled.length +
              '장의 사진(' +
              size +
              ')이 보정된 초점거리·왜곡과 함께 준비됐습니다. 결과를 참조 카메라와 비교합니다.'
            : sampled.length +
              ' photographs at ' +
              size +
              ' are ready with calibrated intrinsics. The result is compared with the reference cameras.',
      );
      return sampled;
    } catch (error) {
      setStatus(String(error));
      return [];
    } finally {
      setBusy(false);
    }
  }
  /** Load one entry of the catalog: footage frames, or calibrated photographs in `mode`. The
   *  card of a sample with the dataset's SIFT features has a checkbox for the `features` mode. */
  async function loadCatalogSample(entry: SampleEntry, mode: LoadedSample['mode'] = 'images') {
    // The loaders reset the page first, which clears the highlight; it is set once they return.
    setLoading({ id: entry.id, done: 0, total: entry.frames });
    try {
      await (entry.kind === 'video'
        ? loadVideoSample(entry)
        : loadPhotoSample(entry, entry.features ? mode : 'images'));
    } finally {
      setLoading(undefined);
    }
    setLoadedSampleId(entry.id);
  }
  /**
   * Show a finished result in the viewer: the cameras and sparse points, the dense cloud and the
   * mesh. It replaces the page's run state the way loading a sample does. The logic that builds
   * the viewer's props is `lib/result-view.ts`, which a page of its own can reuse.
   */
  function showResult(build: (run: number) => ResultView) {
    reset();
    const view = build(denseRunRef.current);
    setReconstruction(view.scene);
    setDiagnostics(view.diagnostics);
    setUnregistered(view.unregistered);
    setSource(view.source);
    if (view.dense) {
      denseRef.current = { depth: [], points: [...view.dense.chunks] };
      setDenseCloud(view.dense);
    }
    if (view.mesh) setDenseMesh(view.mesh);
    setShowDense(true);
    setShowMesh(view.showMesh);
    setFitNonce((nonce) => nonce + 1);
    setLoadedSampleId(view.info.entry?.id);
    setResultInfo(view.info);
    setStatus(
      ko
        ? '저장된 결과를 열었습니다. 실행 없이 뷰어에 표시합니다.'
        : 'A saved result is open. The viewer shows it without a run.',
    );
  }
  /** Open result files from the visitor's disk: `.json` and `.ply`, several at once. */
  async function openResultFiles(list: FileList | null) {
    const picked = Array.from(list ?? []);
    if (!picked.length || busy) return;
    setBusy(true);
    try {
      const view = await resultViewFromFiles(picked, denseRunRef.current + 1);
      showResult(() => view);
    } catch (error) {
      setStatus((ko ? '결과를 열 수 없습니다: ' : 'Cannot open the result: ') + String(error));
    } finally {
      setBusy(false);
    }
  }
  /** The View button of a result card: download its files with progress, then show them. */
  async function viewResult(entry: ResultEntry) {
    if (busy) return;
    setLoading({ id: entry.id, done: 0, total: entry.size ?? 1 });
    setBusy(true);
    try {
      const files = await fetchResultFiles(entry, (done, total) =>
        setLoading({ id: entry.id, done, total }),
      );
      showResult((run) => resultView(loadResultFiles(files), entry.name, run, entry));
    } catch (error) {
      if (error instanceof ResultMissingError)
        setMissingSamples((current) => new Set(current).add(entry.id));
      setStatus((ko ? '결과를 열 수 없습니다: ' : 'Cannot open the result: ') + String(error));
    } finally {
      setLoading(undefined);
      setBusy(false);
    }
  }
  /** The standard benchmark: the eight-frame sample twice, a cold pass and a warm pass. */
  async function runBenchmark() {
    if (busy || benchmarkRunning) return;
    setBenchmarkRunning(true);
    try {
      const sample = await loadVideoSample(PYRAMID_SAMPLE);
      if (sample.length !== 8) return;
      for (let pass = 0; pass < 2 && !abortRef.current?.signal.aborted; pass++) {
        const done = await runSfM(sample, STANDARD_BENCHMARK);
        if (!done) break;
      }
    } finally {
      setBenchmarkRunning(false);
    }
  }

  /** The files the visitor chose: one video (or a frame JSON) runs as before, several make rows. */
  async function chooseFiles(list: File[]) {
    reset();
    const json = list.find((f) => f.name.toLowerCase().endsWith('.json'));
    const videos = list.filter((f) => f !== json);
    if (json || videos.length <= 1) {
      setFile(json ?? videos[0]);
      return;
    }
    setFile(undefined);
    setBusy(true);
    try {
      const made = await Promise.all(
        videos.map(async (f, i) => ({
          key: i + ':' + f.name + ':' + f.size,
          label: f.name,
          file: f,
          focal,
          videoId: await fileVideoId(f),
          durationSeconds: await probeDuration(f),
        })),
      );
      const marked = markDuplicates(made);
      setMvRows(marked.map(({ duplicateOf, ...row }) => ({ ...row, duplicate: !!duplicateOf })));
      setStatus(
        ko
          ? `영상 ${made.length}개가 준비됐습니다. 실행을 누르면 한 번에 복원합니다.`
          : `${made.length} videos are ready. Run reconstructs them together.`,
      );
    } catch (error) {
      setStatus(String(error));
    } finally {
      setBusy(false);
    }
  }
  /** A `video-set` card: every member becomes a row of the upload list (6.3). */
  function loadVideoSet(entry: VideoSetEntry) {
    reset();
    setFile(undefined);
    inputKindRef.current = 'sample';
    const byId = new Map(
      catalog.filter((e): e is SampleEntry => e.kind === 'video').map((e) => [e.id, e]),
    );
    const rows: VideoRow[] = entry.members
      .map((id) => byId.get(id))
      .filter((m): m is SampleEntry => !!m)
      .map((m) => ({
        key: m.id,
        label: m.name,
        sampleId: m.id,
        entry: m,
        videoId: catalogVideoId(m),
        focal: DEFAULT_VIDEO_FOCAL,
        thumbnail: m.dir + m.thumbnail,
        durationSeconds: m.video?.duration,
      }));
    setMvRows(rows);
    setLoadedSampleId(entry.id);
    setStatus(
      ko
        ? `영상 ${rows.length}개가 준비됐습니다. 실행을 누르면 한 번에 복원합니다.`
        : `${rows.length} videos are ready. Run reconstructs them together.`,
    );
  }
  function selectComponent(index: number) {
    const model = mvModels.current.get(index);
    if (!model) return;
    setMvSelected(index);
    setMvView(undefined);
    setReconstruction(model);
    setCameraView(undefined);
    setOverlay(undefined);
    setFitNonce((nonce) => nonce + 1);
  }

  /**
   * A run of two or more videos (`docs/multi-video.md` 6.2). The page decodes the videos one after
   * another and sends the frames of each in time order. The worker's `multiVideo` job gates,
   * extracts and describes the keyframes, then builds the view graph, finds the components and
   * reconstructs each. The list order decides nothing about the result.
   */
  async function runMultiVideo(): Promise<boolean> {
    const rows = mvRowsRef.current.filter((r) => !r.duplicate);
    if (busy || !rows.length) return false;
    clearMultiResult();
    setRecord(undefined);
    setSendState({ status: 'idle' });
    const abort = new AbortController();
    abortRef.current = abort;
    const signal = abort.signal;
    const id = ++jobRef.current;
    runFocal.current = rows[0]!.focal;
    setBusy(true);
    setReplaying(false);
    setProcessed(0);
    setSkipped(0);
    setRunFrames([]);
    setOverlay(undefined);
    setReconstruction(undefined);
    setDiagnostics(undefined);
    setUnregistered([]);
    setStages([]);
    setTotalMs(0);
    setPlayTime(0);
    setAnalysisTime(0);
    setStatus(ko ? '3D 뷰어를 준비하고 있습니다…' : 'Preparing the 3D viewer…');
    const query = multiVideoQuery(typeof location === 'undefined' ? '' : location.search);
    const search = typeof location === 'undefined' ? '' : location.search;
    const sampleFps = pageSampleFps();
    const runEdge = analysisEdge;
    const quality = siftQuality(runEdge, search);
    let worker: Worker;
    let pending:
      { type: string; resolve: (message: any) => void; reject: (error: Error) => void } | undefined;
    let timeout = 0;
    let failure: Error | undefined;
    let failed: any;
    let current: Frame | undefined;
    let stage = 'start';
    const decodeInfo: { path: 'worker' | 'main-thread' } = { path: 'main-thread' };
    const watch = watchMainThread(() => stage);
    const models = new Map<number, Scene>();
    const meanErrors = new Map<number, number | null>();
    let graph: any;
    let fed = 0;
    let firstFrame: Frame | undefined;
    const totalSeconds = rows.reduce((n, r) => n + (r.durationSeconds ?? 0), 0);
    const fail = (error: Error) => {
      failure = error;
      pending?.reject(error);
      abort.abort();
    };
    const armTimeout = () => {
      window.clearTimeout(timeout);
      timeout = window.setTimeout(() => {
        if (pending) fail(new Error('No response from GPU for 4 minutes.'));
      }, 240000);
    };
    const onAbort = () => pending?.reject(new DOMException('Cancelled', 'AbortError'));
    signal.addEventListener('abort', onAbort);
    const request = (type: string, message: object, transfer: Transferable[] = []) =>
      new Promise<any>((resolve, reject) => {
        signal.throwIfAborted();
        pending = { type, resolve, reject };
        armTimeout();
        worker.postMessage({ id, ...message }, transfer);
      });
    const onWorkerMessage = ({ data: message }: MessageEvent) => {
      if (message.id !== id || signal.aborted) return;
      armTimeout();
      if (message.type === 'progress') {
        stage = String(message.detail);
        const words: Record<string, [string, string]> = {
          viewGraph: ['뷰 그래프를 만들고 있습니다…', 'Building the view graph…'],
        };
        const component = /^component (\d+)/.exec(stage);
        setStatus(
          words[stage]?.[ko ? 0 : 1] ??
            (component
              ? ko
                ? `구성요소 ${Number(component[1]) + 1}을 복원하고 있습니다…`
                : `Reconstructing component ${Number(component[1]) + 1}…`
              : ko
                ? '영상을 분석하고 있습니다…'
                : 'Analyzing video…'),
        );
      }
      if (message.type === 'features' && current)
        setOverlay({ frame: current, points: message.points });
      if (message.type === 'graph') {
        graph = message;
        setMvGraph({
          canonical: message.canonical,
          components: message.components,
          matrix: message.matrix,
        });
      }
      if (message.type === 'model' && message.reconstruction) {
        models.set(message.component, message.reconstruction);
        meanErrors.set(message.component, message.meanReprojectionErrorPx ?? null);
        mvModels.current.set(message.component, message.reconstruction);
        if (message.component === 0) {
          setReconstruction(message.reconstruction);
          setFitNonce((nonce) => nonce + 1);
        }
      }
      if (message.type === 'ready' && message.adapter) adapterRef.current = message.adapter;
      if (message.type === 'multiReady' && message.adapter) adapterRef.current = message.adapter;
      if (message.type === 'error') {
        failed = message;
        fail(new Error(message.message || 'The worker reported an error with no message'));
      }
      if (pending && message.type === pending.type) {
        const waiter = pending;
        pending = undefined;
        window.clearTimeout(timeout);
        waiter.resolve(message);
      }
    };
    const update = (key: string, patch: Partial<VideoRow>) =>
      setMvRows((list) => list.map((r) => (r.key === key ? { ...r, ...patch } : r)));
    /** Send one frame and wait for the gate's verdict and, for a keyframe, its features. */
    const feed = async (row: VideoRow, videoId: string, frame: Frame) => {
      current = frame;
      stage = row.label + ' #' + frame.id;
      firstFrame ??= frame;
      setOverlay({ frame });
      setPlayTime(frame.time);
      if (!frame.image) {
        const photo = document.createElement('canvas');
        photo.width = frame.width;
        photo.height = frame.height;
        photo
          .getContext('2d')!
          .putImageData(
            new ImageData(new Uint8ClampedArray(frame.pixels), frame.width, frame.height),
            0,
            0,
          );
        frame.image = photo.toDataURL('image/jpeg', 0.8);
      }
      const pixels = frame.pixels.slice();
      const done = await request(
        'frameComplete',
        { type: 'multiFrame', videoId, frame: { ...frame, pixels } },
        [pixels.buffer],
      );
      fed++;
      if (done.skipped) setSkipped((n) => n + 1);
      else setProcessed((n) => n + 1);
      if (done.keyframe)
        setMvRows((list) =>
          list.map((r) => (r.key === row.key ? { ...r, keyframes: (r.keyframes ?? 0) + 1 } : r)),
        );
      if (done.keyframe && !row.thumbnail && frame.image)
        update(row.key, { thumbnail: frame.image });
      if (!done.skipped) setFrameMs(done.elapsedMs);
      setStatus(
        done.skipped
          ? done.skipped
          : ko
            ? `${row.label}: 키프레임 처리 완료`
            : `${row.label}: keyframe processed`,
      );
      await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
    };
    try {
      await waitForViewer(signal);
      signal.throwIfAborted();
      worker =
        workerRef.current ??
        new Worker(new URL('../workers/sfm.worker.ts', import.meta.url), { type: 'module' });
      workerRef.current = worker;
      worker.onmessage = onWorkerMessage;
      worker.onerror = (event) =>
        fail(
          new Error(
            'Worker error: ' +
              (event.message || '(no message)') +
              (event.filename
                ? ' at ' + event.filename + ':' + event.lineno + ':' + event.colno
                : ''),
          ),
        );
      setStatus(ko ? 'GPU를 준비하고 있습니다…' : 'Preparing the GPU…');
      setSampleFps(0);
      const ready = await request('multiReady', {
        type: 'multiStart',
        query,
        analysisEdge: runEdge,
        firstOctave: quality.firstOctave,
        clahe: quality.clahe,
        budget: budgetOverride(),
        fp64: fp64Override(),
      });
      const used: VideoRow[] = [];
      const seen = new Set<string>();
      // The input each sample row took, by row key (`options.videoSource` of the record).
      const sourceOf = new Map<string, VideoSource | undefined>();
      for (const [index, row] of rows.entries()) {
        signal.throwIfAborted();
        // The frames of a video, in time order. An upload, and a sample whose source video this
        // browser decodes, go through the decoder. Any other sample is fetched whole first: its
        // `videoId` is then the catalog's, or the hash of its frame files (Amendment 7).
        let videoId = row.videoId;
        let source: AsyncGenerator<Frame>;
        const decoded = (input: string | File) =>
          (async function* () {
            const { decodeUploadedFrames } = await import('../lib/sfm-video');
            let n = 0;
            let held: Frame | undefined;
            for await (const frame of decodeUploadedFrames(
              input,
              signal,
              analysisEdge,
              sampleFps,
              decodeInfo,
              search,
            )) {
              if (held) yield held;
              held = { id: n++, ...frame };
            }
            if (held) yield { ...held, last: true };
          })();
        const url = row.entry ? sourceUrl(row.entry) : undefined;
        const decodable = url
          ? await (await import('../lib/sfm-video')).sourceDecodable(url, signal)
          : false;
        if (row.entry) sourceOf.set(row.key, videoSourceOf(row.entry, decodable));
        if (row.file) {
          videoId ??= await fileVideoId(row.file);
          source = decoded(row.file);
        } else if (decodable && row.entry) {
          setStatus((ko ? '원본 영상 ' : 'Source video ') + row.label);
          videoId = catalogVideoId(row.entry)!;
          source = decoded(new URL(url!, location.href).href);
        } else {
          const entry = row.entry!;
          const fetched = await fetchSampleFrames(entry, signal, (done, total) =>
            setStatus(
              (ko ? '샘플 프레임 ' : 'Sample frames ') + row.label + ' ' + done + ' / ' + total,
            ),
          );
          videoId = catalogVideoId(entry) ?? (await sampleVideoId(fetched.map((f) => f.blob)));
          source = (async function* () {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
            for (const [i, f] of fetched.entries()) {
              const bitmap = await createImageBitmap(f.blob);
              canvas.width = bitmap.width;
              canvas.height = bitmap.height;
              ctx.drawImage(bitmap, 0, 0);
              bitmap.close();
              yield {
                id: i,
                width: canvas.width,
                height: canvas.height,
                time: f.time,
                pixels: ctx.getImageData(0, 0, canvas.width, canvas.height).data,
                image: f.url,
                last: i === fetched.length - 1,
              };
            }
          })();
        }
        if (seen.has(videoId)) {
          update(row.key, { videoId, duplicate: true });
          continue;
        }
        seen.add(videoId);
        const item = { ...row, videoId };
        used.push(item);
        update(row.key, { videoId });
        await request('videoStarted', {
          type: 'multiVideo',
          video: {
            videoId,
            label: row.sampleId ?? null,
            durationSeconds: row.durationSeconds ?? 0,
            focalRatio: row.focal,
          },
        });
        for await (const frame of source) {
          signal.throwIfAborted();
          await feed(item, videoId, frame);
          setMvProgress(
            Math.min(
              95,
              Math.round(
                ((index + frame.time / Math.max(row.durationSeconds ?? 0, frame.time || 1)) * 95) /
                  rows.length,
              ),
            ),
          );
        }
        await request('videoComplete', { type: 'multiEndVideo', videoId });
      }
      stage = 'finish';
      setStatus(ko ? '뷰 그래프를 만들고 있습니다…' : 'Building the view graph…');
      const result = await request('result', { type: 'multiFinish' });
      const mainThread = watch.stop();
      const summary = result.summary;
      // The rows learn their tags, component, counts and cross-video inliers.
      const order: string[] = graph?.matrix.order ?? [];
      const crossOf = (videoId: string) => {
        const i = order.indexOf(videoId);
        return i < 0 ? 0 : graph.matrix.cells[i].reduce((a: number, b: number) => a + b, 0);
      };
      const byId = new Map<string, any>(summary.videos.map((v: any) => [v.videoId, v]));
      setMvRows((list) =>
        list.map((r) => {
          const v = r.videoId ? byId.get(r.videoId) : undefined;
          return v
            ? {
                ...r,
                keyframes: v.keyframes,
                registeredViews: v.registeredViews,
                component: v.component,
                tags: v.tags,
                crossInliers: crossOf(r.videoId!),
              }
            : r;
        }),
      );
      setMvProgress(100);
      setStages(
        Object.entries(result.stages).map(([operation, st]: [string, any]) => ({
          operation,
          elapsedMs: st.elapsedMs,
        })),
      );
      setTotalMs(result.totalMs);
      setComplete(true);
      const best = models.get(0);
      if (best) setReconstruction(best);
      const registered = [...models.values()].reduce((n, m) => n + m.views.length, 0);
      setStatus(
        ko
          ? `복원 완료: 구성요소 ${models.size}개, 등록된 뷰 ${registered}개, 연결되지 않은 영상 ${summary.notConnected.length}개.`
          : `Reconstruction completed: ${models.size} components, ${registered} registered views, ${summary.notConnected.length} videos not connected.`,
      );
      if (/[?&]canonical=/.test(search))
        (window as any).__typecvMultiVideo = {
          canonical: result.canonical,
          keyframes: result.keyframes,
          components: result.components,
          summary,
          timing: result.stages,
          totalMs: result.totalMs,
        };
      // A decoder ran for an upload, or for a sample whose source video this browser decodes.
      const decoded = used.some((r) => r.file || sourceOf.get(r.key) === 'source');
      const built = buildMultiRunRecord({
        result,
        graph,
        models,
        meanErrors,
        adapter: ready.adapter ?? adapterRef.current,
        commit: commitRef.current,
        rows: used,
        query,
        maxFeatures: ready.maxFeatures,
        firstOctave: quality.firstOctave,
        frames: fed,
        first: firstFrame,
        durationSeconds: totalSeconds,
        setName: loadedSampleId ?? null,
        videoDecode: decoded ? decodeInfo.path : undefined,
        videoSource: runVideoSource(used.map((r) => sourceOf.get(r.key))),
        videoSampleFps: decoded ? sampleFps : undefined,
        mainThread,
        note: noteRef.current,
      });
      setRecord(built);
      if (consentRef.current) await sendRecord(built);
      return true;
    } catch (error) {
      const mainThread = watch.stop();
      const cancelled = signal.aborted && !failure;
      setStatus(
        cancelled
          ? ko
            ? '분석을 취소했습니다.'
            : 'Analysis cancelled.'
          : (ko ? '복원 실패: ' : 'Reconstruction failed: ') + String(failure ?? error),
      );
      terminateWorker();
      if (!cancelled) {
        const message = String(failure ?? error);
        const built = buildMultiRunRecord({
          result: undefined,
          graph,
          models,
          meanErrors,
          adapter: adapterRef.current,
          commit: commitRef.current,
          rows: mvRowsRef.current.filter((r) => !r.duplicate),
          query,
          maxFeatures: 0,
          firstOctave: quality.firstOctave,
          frames: fed,
          first: firstFrame,
          durationSeconds: totalSeconds,
          setName: loadedSampleId ?? null,
          mainThread,
          note: noteRef.current,
          failure: failed?.failure ?? {
            message: message.slice(0, 2000),
            operation: stage.slice(0, 200),
            kernel: '',
            reason: '',
            framesProcessed: fed,
          },
        });
        setRecord(built);
        if (consentRef.current) await sendRecord(built);
      }
      return false;
    } finally {
      window.clearTimeout(timeout);
      signal.removeEventListener('abort', onAbort);
      setBusy(false);
      abortRef.current = undefined;
    }
  }

  async function runSfM(framesOverride?: Frame[], benchmark?: Benchmark): Promise<boolean> {
    const inputFrames = framesOverride ?? prepared;
    if (busy || (!inputFrames.length && !videoURL)) return false;
    if (!inputFrames.length) inputKindRef.current = 'video';
    // A video sample with a decodable source video: the run decodes it, as it decodes an upload.
    const stored = framesOverride ? undefined : videoSourceRef.current;
    const sourceRun = stored?.decodable ? stored : undefined;
    const videoSource: VideoSource | undefined = stored
      ? stored.decodable
        ? 'source'
        : 'frames'
      : undefined;
    setRecord(undefined);
    setAlignment(undefined);
    setSendState({ status: 'idle' });
    const abort = new AbortController();
    abortRef.current = abort;
    const signal = abort.signal;
    const id = ++jobRef.current;
    runFocal.current = focal;
    let worker: Worker;
    setBusy(true);
    setReplaying(false);
    setSelectedIndex(0);
    setComplete(false);
    setProcessed(0);
    setSkipped(0);
    setFrameMs(0);
    setRunFrames([]);
    setOverlay(undefined);
    setReconstruction(undefined);
    setDiagnostics(undefined);
    setUnregistered([]);
    setStages([]);
    setTotalMs(0);
    setPlayTime(0);
    setAnalysisTime(0);
    if (!inputFrames.length || sourceRun) setFrames([]);
    setStatus(ko ? '3D 뷰어를 준비하고 있습니다…' : 'Preparing the 3D viewer…');
    let pending:
      { type: string; resolve: (message: any) => void; reject: (error: Error) => void } | undefined;
    let timeout = 0;
    let currentFrame: Frame | undefined;
    let counts = { features: 0, matches: 0, inliers: 0 };
    let failure: Error | undefined;
    // The worker's error message with its partial outcome, for the failure record.
    let failed: any;
    // The dense MVS (5.3): its final message, its waiter and the counts the status shows.
    const denseRun = denseEnabledRef.current || denseOverride();
    let denseFinal: any;
    let denseCounts = { depth: 0, points: 0 };
    clearDense();
    const denseRunId = denseRunRef.current;
    let denseShown = false;
    // A reference of the MVS can take minutes on a software GPU, so the watchdog waits longer.
    let watchdogMs = 120000;
    // The lab's stage for the main-thread watch: the worker's last progress detail.
    let stage = 'start';
    // The decode path of an uploaded video, set by `decodeUploadedFrames` for the run record.
    const decodeInfo: { path: 'worker' | 'main-thread' } = { path: 'main-thread' };
    const watch = watchMainThread(() => stage);
    const fail = (error: Error) => {
      failure = error;
      pending?.reject(error);
      abort.abort();
    };
    const collected: Frame[] = [];
    const armTimeout = () => {
      window.clearTimeout(timeout);
      timeout = window.setTimeout(() => {
        if (pending)
          fail(
            new Error('No response from GPU for ' + Math.round(watchdogMs / 60000) + ' minutes.'),
          );
      }, watchdogMs);
    };
    const onAbort = () => pending?.reject(new DOMException('Cancelled', 'AbortError'));
    signal.addEventListener('abort', onAbort);
    const request = (type: string, message: object, transfer: Transferable[] = []) =>
      new Promise<any>((resolve, reject) => {
        signal.throwIfAborted();
        pending = { type, resolve, reject };
        armTimeout();
        worker.postMessage({ id, ...message }, transfer);
      });
    const onWorkerMessage = ({ data: message }: MessageEvent) => {
      if (message.id !== id || signal.aborted) return;
      armTimeout();
      if (message.type === 'progress') {
        const labels: Record<string, [string, string]> = {
          features: ['특징점을 찾고 있습니다…', 'Detecting features…'],
          matching: ['이전 프레임과 대응점을 찾고 있습니다…', 'Matching previous frames…'],
          verifyPair: ['이미지 쌍의 기하를 검증하고 있습니다…', 'Verifying image-pair geometry…'],
          pnp: ['카메라 위치를 추정하고 있습니다…', 'Estimating camera position…'],
          triangulation: ['3D 점을 계산하고 있습니다…', 'Triangulating 3D points…'],
          observations: ['재투영 오차를 검사하고 있습니다…', 'Checking reprojection errors…'],
          bundle: ['지도를 정밀화하고 있습니다…', 'Refining the map…'],
          final: ['마지막 정밀화를 진행하고 있습니다…', 'Final map refinement…'],
        };
        stage = String(message.detail);
        const label = labels[String(message.detail).split(/[ #·]/)[0]];
        if (/^(pnp|verifyPair)/.test(message.detail)) setFrameStep(2);
        if (/^(triangulation|observations|bundle)/.test(message.detail)) setFrameStep(3);
        const base = label
          ? label[ko ? 0 : 1]
          : ko
            ? '영상을 분석하고 있습니다…'
            : 'Analyzing video…';
        // The first job compiles each GPU kernel on first use; show that count while it is cold.
        setStatus(
          message.cold && message.kernels
            ? base +
                (ko
                  ? ' GPU 커널 ' + message.kernels + '개 컴파일됨 (첫 실행)'
                  : ' ' + message.kernels + ' GPU kernels compiled (first run)')
            : base,
        );
      }
      if (message.type === 'features' && currentFrame && currentFrame.id === message.frameId) {
        counts.features = message.points.length / 2;
        setOverlay({ frame: currentFrame, points: message.points });
        setFrameCounts((counts) => ({ ...counts, features: message.points.length / 2 }));
        setFrameStep(1);
      }
      if (message.type === 'matches') {
        counts.matches = message.count;
        setFrameCounts((counts) => ({ ...counts, matches: message.count }));
        setFrameStep(2);
        setOverlay((previous) =>
          previous && previous.frame.id === message.frameId
            ? { ...previous, lines: message.lines, matches: message.count }
            : previous,
        );
      }
      if (message.type === 'snapshot') {
        setReconstruction(message.reconstruction);
        if (sampleRef.current)
          setAlignment(compareToReference(sampleRef.current, message.reconstruction.views));
        if (
          message.reconstruction.views.some((view: { id: number }) => view.id === currentFrame?.id)
        )
          setFrameStep(4);
      }
      if (message.type === 'ready' && message.adapter) {
        adapterRef.current = message.adapter;
      }
      // 5.3: the dense MVS posts at most one message per reference and stage.
      if (message.type === 'denseDepth') {
        denseRef.current.depth.push(message);
        denseCounts = { ...denseCounts, depth: denseCounts.depth + 1 };
      }
      if (message.type === 'densePoints') {
        denseRef.current.points.push(message);
        denseCounts = { ...denseCounts, points: denseCounts.points + message.positions.length / 3 };
        // Section 6: the cloud grows per reference, and its first points replace the sparse cloud.
        if (!denseShown && denseCounts.points) {
          denseShown = true;
          setShowDense(true);
        }
        setDenseCloud({
          run: denseRunId,
          chunks: denseRef.current.points,
          count: denseCounts.points,
        });
      }
      if (message.type === 'denseClean') {
        // 3.12: the cleaned cloud replaces the raw chunks. A run id of its own makes the viewer
        // rebuild its subset, whatever the number of chunks was.
        denseRef.current.points = [message];
        denseCounts = { ...denseCounts, points: message.positions.length / 3 };
        setDenseCloud({ run: denseRunId + 0.5, chunks: [message], count: denseCounts.points });
      }
      if (message.type === 'denseMesh') {
        // 3.13: the mesh of the final cloud. It stays hidden until the Display switch is on.
        setDenseMesh({
          run: denseRunId,
          positions: message.positions,
          colors: message.colors,
          indices: message.indices,
        });
      }
      if (message.type === 'denseProgress') {
        stage = 'dense ' + message.stage + ' ' + message.done + '/' + message.total;
        const words = DENSE_STAGES[message.stage] ?? [message.stage, message.stage];
        setStatus(
          (ko ? '조밀 MVS: ' : 'Dense MVS: ') +
            words[ko ? 0 : 1] +
            ' ' +
            message.done +
            '/' +
            message.total +
            (ko
              ? ' · 깊이 맵 ' + denseCounts.depth + '개 · 점 ' + denseCounts.points + '개'
              : ' · ' + denseCounts.depth + ' depth maps · ' + denseCounts.points + ' points'),
        );
      }
      if (message.type === 'denseResult') {
        denseFinal = message;
        setDenseSummary(denseRecord(message));
      }
      if (message.type === 'error') {
        failed = message;
        fail(new Error(message.message || 'The worker reported an error with no message'));
      }
      if (pending && message.type === pending.type) {
        const waiter = pending;
        pending = undefined;
        window.clearTimeout(timeout);
        waiter.resolve(message);
      }
    };
    const consume = async (frame: Frame) => {
      currentFrame = frame;
      stage = 'frame #' + frame.id;
      setFrameFinished(false);
      counts = { features: 0, matches: 0, inliers: 0 };
      setFrameStep(0);
      setFrameCounts({ features: 0, matches: 0, inliers: 0 });
      collected.push({
        ...frame,
        pixels: new Uint8ClampedArray(),
        image: undefined,
        precomputed: undefined,
      });
      setRunFrames([...collected]);
      setSelectedIndex(collected.length - 1);
      setOverlay({ frame });
      const pixels = frame.pixels.slice();
      // A sample or prepared frame carries its image URL; decoded video carries a JPEG data URL.
      // The encode below is the fallback for a frame that has neither (36 ms of main-thread work
      // per 640 × 360 frame, measured 2026-10-20).
      if (!frame.image) {
        const photo = document.createElement('canvas');
        photo.width = frame.width;
        photo.height = frame.height;
        photo
          .getContext('2d')!
          .putImageData(
            new ImageData(new Uint8ClampedArray(frame.pixels), frame.width, frame.height),
            0,
            0,
          );
        frame.image = photo.toDataURL('image/jpeg', 0.8);
      }
      const message = await request(
        'frameComplete',
        { type: 'frame', frame: { ...frame, pixels } },
        [pixels.buffer],
      );
      if (message.skipped) setSkipped((count) => count + 1);
      else setProcessed((count) => count + 1);
      setFrameFinished(true);
      setAnalysisTime(frame.time);
      if (!message.skipped) setFrameMs(message.elapsedMs);
      frame.metrics = {
        ...counts,
        inliers: message.verifiedInliers ?? 0,
        registered: !!message.cameraRegistered,
      };
      collected[collected.length - 1].metrics = frame.metrics;
      setRunFrames([...collected]);
      setFrameCounts((counts) => ({ ...counts, inliers: message.verifiedInliers ?? 0 }));
      setFrameStep(message.cameraRegistered ? 4 : 2);
      setUnregistered(message.unregistered ?? []);
      setStatus(
        message.skipped
          ? message.skipped
          : message.waiting
            ? ko
              ? '초기 복원에 필요한 시차와 연결을 기다립니다.'
              : 'Waiting for a stable initial pair.'
            : ko
              ? '카메라와 지도를 갱신했습니다.'
              : 'Camera and map updated.',
      );
      await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
    };
    try {
      await waitForViewer(signal);
      signal.throwIfAborted();
      worker =
        workerRef.current ??
        new Worker(new URL('../workers/sfm.worker.ts', import.meta.url), { type: 'module' });
      workerRef.current = worker;
      worker.onmessage = onWorkerMessage;
      // Safari reports a worker's uncaught error with no message: name what it does give.
      worker.onerror = (event) =>
        fail(
          new Error(
            'Worker error: ' +
              (event.message || '(no message)') +
              (event.filename
                ? ' at ' + event.filename + ':' + event.lineno + ':' + event.colno
                : ''),
          ),
        );
      setStatus(ko ? 'GPU를 준비하고 있습니다…' : 'Preparing the GPU…');
      const sampleFps = pageSampleFps();
      const sampledVideo = inputFrames.length === 0 || !!sourceRun;
      setSampleFps(sampledVideo ? sampleFps : 0);
      const runEdge =
        !sourceRun && inputFrames[0]?.width
          ? Math.max(inputFrames[0].width, inputFrames[0].height)
          : analysisEdge;
      await request('ready', {
        type: 'start',
        focalRatio: focal,
        profile: profileRef.current,
        // `?bounded=1`: fenced submissions, a diagnostic mode for a device loss (see the worker).
        bounded: boundedOverride(),
        // `?fp64=float|integer`: force a flavor of the emulated double, for a measurement; the
        // device picks from the adapter otherwise (TypeShade change 0073).
        fp64: fp64Override(),
        // `?budget=<ms>`: the device's time budget per submission (change 0074), for a
        // measurement of the page's responsiveness against the slice length.
        budget: budgetOverride(),
        // `?dense=1`: the dense MVS after the sparse result (`docs/dense-mvs.md` 5.3).
        dense: denseRun,
        // `?denseClean=0`: keep the raw fused cloud (`docs/dense-mvs.md` 3.12).
        denseClean: denseCleanOverride(),
        // `?denseMesh=1`: mesh the final cloud (`docs/dense-mvs.md` 3.13).
        denseMesh: denseMeshOverride(),
        // `?gate=phase|legacy`: the keyframe gate (phase correlation before SIFT, or the match rule).
        gate: pageKeyframeGate(),
        // The first octave and guided matching follow the analysis edge (`siftQuality`);
        // `?octave=-1|0` and `?guided=1|0` override them, for a measurement. `?clahe=1` turns
        // CLAHE on before the pyramid.
        ...siftQuality(runEdge, typeof location === 'undefined' ? '' : location.search),
        // `?features=orb`: the ORB front end instead of SIFT.
        features: featureFrontEnd(typeof location === 'undefined' ? '' : location.search),
        // `?guided=tiled`: the tiled kernel of guided matching instead of the scalar one.
        guidedKernel: guidedKernel(typeof location === 'undefined' ? '' : location.search),
        // `?orbFeatures=`, `?orbRatio=` and `?orbDistance=` tune the ORB front end, for a measurement.
        orbTuning: orbTuning(typeof location === 'undefined' ? '' : location.search),
        // `?klt=1`: track points between the frames of an uploaded video (`docs/klt-tracking.md`).
        klt: sampledVideo && kltTracking(typeof location === 'undefined' ? '' : location.search),
        kltSetup: kltSetup(typeof location === 'undefined' ? '' : location.search),
        analysisEdge: runEdge,
      });
      if (inputFrames.length && !sourceRun) {
        for (const frame of inputFrames) {
          signal.throwIfAborted();
          setPlayTime(frame.time);
          await consume(frame);
        }
      } else {
        const video = videoRef.current;
        video?.pause();
        const { decodeUploadedFrames } = await import('../lib/sfm-video');
        const previewSeek = video ? previewSeeker(video) : () => false;
        let index = 0;
        // One frame of lookahead, so that the final frame of the video is marked `last`.
        let held: Frame | undefined;
        const feed = async (frame: Frame) => {
          setFrames((thumbs) => [...thumbs, frame.image!].slice(-24));
          await consume(frame);
        };
        for await (const decoded of decodeUploadedFrames(
          sourceRun ? new URL(sourceRun.url, location.href).href : file!,
          signal,
          analysisEdge,
          sampleFps,
          decodeInfo,
          typeof location === 'undefined' ? '' : location.search,
        )) {
          signal.throwIfAborted();
          previewSeek(decoded.time);
          setPlayTime(decoded.time);
          const frame: Frame = { id: index++, ...decoded };
          if (held) await feed(held);
          held = frame;
        }
        if (held) await feed({ ...held, last: true });
      }
      const result = await request('result', { type: 'finish' });
      let dense: any;
      if (denseRun) {
        // The sparse result is on the page while the MVS runs in the worker (5.3).
        setReconstruction(result.reconstruction);
        setStatus(
          ko
            ? 'GPU SfM 복원이 완료되었습니다. 조밀 MVS를 시작합니다…'
            : 'GPU SfM reconstruction completed. Starting the dense MVS…',
        );
        stage = 'dense';
        watchdogMs = 600000;
        denseCancelRef.current = () => worker.postMessage({ type: 'cancelDense', id });
        setDenseRunning(true);
        try {
          dense =
            denseFinal ??
            (await new Promise<any>((resolve, reject) => {
              signal.throwIfAborted();
              pending = { type: 'denseResult', resolve, reject };
              armTimeout();
            }));
        } finally {
          denseCancelRef.current = undefined;
          setDenseRunning(false);
        }
      }
      const mainThread = watch.stop();
      setReconstruction(result.reconstruction);
      setDiagnostics(result.diagnostics);
      setUnregistered(result.unregistered);
      setStages(result.stages);
      setTotalMs(result.stages[0].elapsedMs);
      const aligned = sampleRef.current
        ? compareToReference(sampleRef.current, result.reconstruction.views)
        : undefined;
      setAlignment(aligned);
      setComplete(true);
      setStatus(
        !dense
          ? ko
            ? 'GPU SfM 복원이 완료되었습니다.'
            : 'GPU SfM reconstruction completed.'
          : dense.failure
            ? (ko ? '조밀 MVS 실패: ' : 'Dense MVS failed: ') + dense.failure.message
            : dense.skipped
              ? (ko ? '조밀 MVS 생략: ' : 'Dense MVS skipped: ') + dense.skipped
              : (dense.cancelled
                  ? ko
                    ? '조밀 MVS를 취소했습니다: 점 '
                    : 'Dense MVS cancelled: '
                  : ko
                    ? '조밀 MVS가 완료되었습니다: 점 '
                    : 'Dense MVS completed: ') +
                dense.points +
                (ko
                  ? '개, 깊이 맵 ' + denseCounts.depth + '개.'
                  : ' points from ' + denseCounts.depth + ' depth maps.'),
      );
      const built = buildRunRecord({
        result,
        dense,
        mainThread,
        adapter: result.adapter ?? adapterRef.current,
        commit: commitRef.current,
        kind: inputKindRef.current,
        sampleName:
          inputKindRef.current === 'sample'
            ? (sampleRef.current?.name ?? source?.name ?? null)
            : null,
        frames: collected,
        durationSeconds:
          source?.duration ?? collected.at(-1)?.time ?? inputFrames.at(-1)?.time ?? 0,
        videoSampleFps: sampledVideo ? sampleFps : undefined,
        videoDecode: sampledVideo ? decodeInfo.path : undefined,
        videoSource,
        benchmark,
        reference: referenceRecord(sampleRef.current, aligned),
        note: noteRef.current,
      });
      setRecord(built);
      // Consent given before the run sends the record as soon as it exists.
      if (consentRef.current) await sendRecord(built);
      return true;
    } catch (error) {
      const mainThread = watch.stop();
      const cancelled = signal.aborted && !failure;
      setStatus(
        cancelled
          ? ko
            ? '분석을 취소했습니다.'
            : 'Analysis cancelled.'
          : (ko ? '복원 실패: ' : 'Reconstruction failed: ') + String(failure ?? error),
      );
      terminateWorker();
      if (!cancelled) {
        // The failure record: everything measured before the error, and the error itself.
        const message = String(failure ?? error);
        const built = buildRunRecord({
          result: failed?.partial ?? { stages: [], adapter: adapterRef.current },
          mainThread,
          failure: failed?.failure ?? {
            message: message.slice(0, 2000),
            operation: '',
            kernel: '',
            reason: '',
            framesProcessed: collected.length,
          },
          adapter: failed?.partial?.adapter ?? adapterRef.current,
          commit: commitRef.current,
          kind: inputKindRef.current,
          sampleName:
            inputKindRef.current === 'sample'
              ? (sampleRef.current?.name ?? source?.name ?? null)
              : null,
          frames: collected,
          durationSeconds: source?.duration ?? collected.at(-1)?.time ?? 0,
          videoSampleFps: inputFrames.length === 0 || sourceRun ? pageSampleFps() : undefined,
          videoDecode: inputFrames.length === 0 || sourceRun ? decodeInfo.path : undefined,
          videoSource,
          benchmark,
          reference: null,
          note: noteRef.current,
        });
        setRecord(built);
        // The result files are built from the page's state, which this run has not set yet: with
        // "Also send the result files" on, the visitor sends the record and the files by hand.
        if (consentRef.current && !sendFilesRef.current) await sendRecord(built);
      }
      return false;
    } finally {
      window.clearTimeout(timeout);
      signal.removeEventListener('abort', onAbort);
      videoRef.current?.pause();
      setBusy(false);
      abortRef.current = undefined;
    }
  }

  /** The dense cloud as a binary PLY (section 6): every point, not the drawn subset. */
  function downloadDense() {
    if (!denseRef.current.points.length) return;
    const bytes = densePly(denseRef.current.points);
    const url = URL.createObjectURL(
      new Blob([bytes as Uint8Array<ArrayBuffer>], { type: 'application/octet-stream' }),
    );
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = 'typecv-dense.ply';
    anchor.click();
    URL.revokeObjectURL(url);
  }

  /** The mesh as a binary PLY with faces (section 3.13). */
  function downloadMesh() {
    if (!denseMesh) return;
    const url = URL.createObjectURL(
      new Blob([denseMeshPly(denseMesh) as Uint8Array<ArrayBuffer>], {
        type: 'application/octet-stream',
      }),
    );
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = 'typecv-mesh.ply';
    anchor.click();
    URL.revokeObjectURL(url);
  }

  /** The reconstruction export as text; `reconstruction` is set. */
  function reconstructionText(format: 'json' | 'ply'): string {
    if (!reconstruction) return '';
    return format === 'json'
      ? reconstructionJson({
          source,
          assumedFocalRatio: runFocal.current,
          diagnostics,
          unregistered,
          analysisFrames: runFrames.map(({ pixels: _pixels, image: _image, ...frame }) => frame),
          scene: reconstruction,
          sourceCamera: (view) => {
            const frame = runFrames.find((candidate) => candidate.id === view.id);
            return frame
              ? cameraInSourcePixels(view.camera, frame, {
                  width: frame.sourceWidth ?? frame.width,
                  height: frame.sourceHeight ?? frame.height,
                })
              : view.camera;
          },
          runRecord: record,
        })
      : reconstructionPly(reconstruction);
  }

  function download(format: 'json' | 'ply') {
    if (!reconstruction) return;
    const text = reconstructionText(format);
    const url = URL.createObjectURL(new Blob([text], { type: 'application/octet-stream' }));
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = 'typecv-reconstruction.' + format;
    anchor.click();
    URL.revokeObjectURL(url);
  }
  /** The card of a precomputed result: tags, a View button with a progress bar. */
  function resultCard(entry: ResultEntry) {
    const missing = missingSamples.has(entry.id);
    const active = loading?.id === entry.id;
    const description = entry.description?.[ko ? 'ko' : 'en'];
    const percent =
      active && loading.total ? Math.min(100, (100 * loading.done) / loading.total) : 0;
    return (
      <figure
        key={entry.id}
        className="sample-card"
        role="listitem"
        data-sample={entry.id}
        data-kind="result"
        data-loaded={loadedSampleId === entry.id || undefined}
      >
        <Tooltip
          placement="bottom"
          title={
            <>
              {description && <div>{description}</div>}
              <div>
                {entry.run.date} · {entry.run.commit.slice(0, 7)}
                {entry.size !== undefined && ' · ' + (entry.size / 1048576).toFixed(1) + ' MB'}
              </div>
            </>
          }
        >
          {entry.thumbnail && !missing ? (
            <img
              src={entry.dir + entry.thumbnail}
              alt={description ?? entry.name}
              loading="lazy"
              onError={() => setMissingSamples((current) => new Set(current).add(entry.id))}
            />
          ) : (
            <div className="sample-card-blank" role="img" aria-label={entry.name} />
          )}
        </Tooltip>
        <figcaption>
          <strong>{entry.name}</strong>
          <div className="sample-card-tags">
            <Tag color="gold">{deviceLabel(entry.run.device)}</Tag>
            <Tag color="blue">{ko ? '결과' : 'result'}</Tag>
            {entry.files.dense && <Tag>{ko ? '조밀' : 'dense'}</Tag>}
            {entry.files.mesh && <Tag>{ko ? '메시' : 'mesh'}</Tag>}
            {entry.source && <Tag color="purple">{entry.source}</Tag>}
          </div>
          <div className="sample-card-actions">
            <Button
              size="small"
              disabled={(busy && !active) || missing}
              loading={active}
              aria-label={'View ' + entry.name}
              onClick={() => void viewResult(entry)}
            >
              {active
                ? (ko ? '받는 중 ' : 'Downloading ') + Math.round(percent) + ' %'
                : missing
                  ? ko
                    ? '준비 중'
                    : 'Not published yet'
                  : ko
                    ? '보기'
                    : 'View'}
            </Button>
            {active && (
              <Progress
                className="sample-card-progress"
                size="small"
                showInfo={false}
                percent={percent}
                aria-label={ko ? '내려받기 진행률' : 'Download progress'}
              />
            )}
          </div>
        </figcaption>
      </figure>
    );
  }
  const duration = source?.duration ?? prepared.at(-1)?.time ?? 0;
  const progress = mvRows.length
    ? mvProgress
    : complete
      ? 100
      : duration
        ? Math.min(99, Math.round((100 * analysisTime) / duration))
        : 0;
  return (
    <div className="lab-app">
      <Card className="lab-panel lab-input">
        <div className="lab-input-grid">
          <div className="lab-input-main">
            <div className="sample-catalog-head">
              <Title level={4}>{ko ? '입력' : 'Input'}</Title>
            </div>
            <div className="sample-catalog" role="list">
              {catalog.map((entry) =>
                isResult(entry) ? (
                  resultCard(entry)
                ) : isVideoSet(entry) ? (
                  <VideoSetCard
                    key={entry.id}
                    ko={ko}
                    entry={entry}
                    members={entry.members
                      .map((id) => catalog.find((m) => m.id === id))
                      .filter((m): m is SampleEntry => !!m && m.kind === 'video')}
                    loaded={loadedSampleId === entry.id}
                    busy={busy}
                    missing={entry.members.some((id) => missingSamples.has(id))}
                    onLoad={() => loadVideoSet(entry)}
                  />
                ) : (
                  <figure
                    key={entry.id}
                    className="sample-card"
                    role="listitem"
                    data-sample={entry.id}
                    data-kind={entry.kind}
                    data-loaded={loadedSampleId === entry.id || undefined}
                  >
                    <Tooltip title={entry.description[ko ? 'ko' : 'en']} placement="bottom">
                      <img
                        src={entry.dir + entry.thumbnail}
                        alt={entry.description[ko ? 'ko' : 'en']}
                        loading="lazy"
                        onError={() =>
                          setMissingSamples((current) => new Set(current).add(entry.id))
                        }
                      />
                    </Tooltip>
                    <figcaption>
                      <strong>{entry.name}</strong>
                      <div className="sample-card-tags">
                        <Tag>
                          {entry.kind === 'video'
                            ? ko
                              ? '영상'
                              : 'video'
                            : ko
                              ? '사진'
                              : 'photos'}
                        </Tag>
                        {entry.kind !== 'video' && (
                          <Tag color="blue">{ko ? '보정' : 'calibrated'}</Tag>
                        )}
                        {entry.features && <Tag color="purple">COLMAP SIFT</Tag>}
                        {entry.size && (
                          <Tag>
                            {(() => {
                              const edge = entry.edges?.includes(analysisEdge) ? analysisEdge : 640;
                              return (
                                Math.round((entry.size[0] * edge) / 640) +
                                '×' +
                                Math.round((entry.size[1] * edge) / 640) +
                                ' px'
                              );
                            })()}
                          </Tag>
                        )}
                      </div>
                      <div className="sample-card-actions">
                        {entry.features && (
                          <Tooltip
                            title={
                              ko
                                ? 'COLMAP이 이 데이터셋에서 찾은 SIFT 특징점과 매칭을 그대로 씁니다. 특징 검출과 매칭은 건너뛰고, TypeCV의 매퍼(검증, 등록, 삼각측량, BA)만 실행해 COLMAP 카메라와 비교합니다.'
                                : "Uses the SIFT keypoints and matches COLMAP found in this dataset. Feature detection and matching are skipped; only TypeCV's mapper (verification, registration, triangulation, BA) runs, held to COLMAP's cameras."
                            }
                          >
                            <Checkbox
                              className="sample-card-mode"
                              aria-describedby={'sample-mode-help-' + entry.id}
                              disabled={busy}
                              checked={featureSamples.has(entry.id)}
                              onChange={(event) =>
                                setFeatureSamples((current) => {
                                  const next = new Set(current);
                                  if (event.target.checked) next.add(entry.id);
                                  else next.delete(entry.id);
                                  return next;
                                })
                              }
                            >
                              {ko ? '특징만 사용' : 'Features only'}
                            </Checkbox>
                          </Tooltip>
                        )}
                        {entry.features && (
                          <span className="sr-only" id={'sample-mode-help-' + entry.id}>
                            {ko
                              ? '켜면 COLMAP이 찾은 SIFT 특징점과 매칭을 그대로 쓰고, 특징 검출과 매칭을 건너뛰어 TypeCV의 매퍼만 실행합니다.'
                              : "When on, the run uses COLMAP's SIFT keypoints and matches, skips feature detection and matching, and runs only TypeCV's mapper."}
                          </span>
                        )}
                        <Button
                          size="small"
                          disabled={
                            (busy && loading?.id !== entry.id) || missingSamples.has(entry.id)
                          }
                          loading={loading?.id === entry.id}
                          aria-label={'Load ' + entry.name}
                          onClick={() =>
                            void loadCatalogSample(
                              entry,
                              featureSamples.has(entry.id) ? 'features' : 'images',
                            )
                          }
                        >
                          {loading?.id === entry.id
                            ? (ko ? '불러오는 중 ' : 'Loading ') +
                              loading.done +
                              ' / ' +
                              loading.total
                            : missingSamples.has(entry.id)
                              ? ko
                                ? '준비 중'
                                : 'Not published yet'
                              : ko
                                ? '불러오기'
                                : 'Load'}
                        </Button>
                        {loading?.id === entry.id && (
                          <Progress
                            className="sample-card-progress"
                            size="small"
                            showInfo={false}
                            percent={loading.total ? (100 * loading.done) / loading.total : 0}
                            aria-label={ko ? '불러오기 진행률' : 'Load progress'}
                          />
                        )}
                      </div>
                    </figcaption>
                  </figure>
                ),
              )}
            </div>
            <div className="lab-upload">
              <Upload
                disabled={busy}
                multiple
                accept="video/*,application/json,.json"
                fileList={[]}
                showUploadList={false}
                beforeUpload={(input, list) => {
                  // One call for each file: the first of the batch takes the whole selection.
                  if (input === list[0]) void chooseFiles(list as File[]);
                  return false;
                }}
              >
                <Button>
                  {ko
                    ? '내 영상(여러 개 가능) / 프레임 JSON 선택'
                    : 'Choose my videos / frame JSON'}
                </Button>
              </Upload>
              <Button
                disabled={busy || !file?.name.toLowerCase().endsWith('.json')}
                onClick={loadPreparedFrames}
              >
                {ko ? '분석 프레임 준비' : 'Prepare analysis frames'}
              </Button>
              <Tooltip
                title={
                  ko
                    ? '내보낸 reconstruction JSON과 조밀·메시 PLY를 실행 없이 뷰어에 엽니다. 여러 파일을 함께 고를 수 있습니다.'
                    : 'Open exported reconstruction JSON and dense or mesh PLY files in the viewer without a run. Pick several at once.'
                }
              >
                <Button disabled={busy} onClick={() => resultInput.current?.click()}>
                  {ko ? '결과 파일 열기' : 'Open result files'}
                </Button>
              </Tooltip>
              <input
                ref={resultInput}
                type="file"
                multiple
                hidden
                accept=".json,.ply,application/json"
                aria-label={ko ? '결과 파일' : 'Result files'}
                onChange={(event) => {
                  void openResultFiles(event.target.files);
                  event.target.value = '';
                }}
              />
              {(mvRows.length > 0 || file) && (
                <VideoRows
                  ko={ko}
                  busy={busy}
                  rows={
                    mvRows.length
                      ? mvRows
                      : [{ key: 'single', label: file!.name, file, focal } as VideoRow]
                  }
                  onFocal={(key, value) =>
                    key === 'single'
                      ? setFocal(value)
                      : setMvRows((list) =>
                          list.map((r) => (r.key === key ? { ...r, focal: value } : r)),
                        )
                  }
                  onRemove={(key) => {
                    if (key === 'single') {
                      reset();
                      setFile(undefined);
                    } else {
                      if (mvGraph) clearMultiResult();
                      setMvRows((list) => list.filter((r) => r.key !== key));
                    }
                  }}
                />
              )}
              {mvGraph && mvGraph.matrix.order.length > 1 && (
                <InlierMatrix
                  ko={ko}
                  names={mvGraph.matrix.order.map(
                    (id) => mvRows.find((r) => r.videoId === id)?.label ?? id,
                  )}
                  cells={mvGraph.matrix.cells}
                />
              )}
              <p className="field-help">
                {ko
                  ? '영상에서 초당 4프레임만 읽고(`?fps=`로 바꿀 수 있음) 그 사이 프레임은 디코딩하지 않습니다. 읽은 프레임 중 마지막 키프레임에서 화면이 4% 이상 움직였거나(작은 흑백 썸네일의 위상 상관, 특징점 추출 전에 판단) 1초가 지난 프레임만 키프레임으로 처리하고, 나머지는 건너뜁니다. `?gate=legacy`는 이전 규칙(특징점 8px 이상 이동)을 씁니다. 브라우저가 지원하는 코덱이 필요합니다. 초점거리/너비 0.8, 렌즈 왜곡 없음은 기본 가정입니다. 영상은 브라우저 안에서만 처리됩니다.'
                  : 'Decodes the video at 4 frames per second (`?fps=` overrides it) and skips the frames between. A sampled frame becomes a keyframe when the image moved at least 4 % of the frame since the last keyframe (phase correlation of a small grayscale thumbnail, checked before feature extraction) or one second has passed; other frames are skipped. `?gate=legacy` uses the older rule (features moved at least 8 px). Requires a browser-supported codec. Assumes focal length / width 0.8 and zero distortion. Video stays in this browser.'}
              </p>
            </div>
          </div>
          <aside className="lab-run-column" aria-label={ko ? '실행' : 'Run'}>
            <Title level={5}>{ko ? '실행' : 'Run'}</Title>
            <label className="lab-field">
              <span>{ko ? '분석 해상도' : 'Analysis resolution'}</span>
              <Radio.Group
                size="small"
                disabled={busy}
                value={analysisEdge}
                onChange={(event) => setAnalysisEdge(event.target.value)}
                aria-label={ko ? '분석 프레임의 긴 변' : 'Long side of an analysis frame'}
              >
                {ANALYSIS_EDGES.map((edge) => (
                  <Radio.Button key={edge} value={edge}>
                    {edge} px
                  </Radio.Button>
                ))}
              </Radio.Group>
            </label>
            <label className="lab-field">
              <span>{ko ? '초점거리 / 영상 너비' : 'Focal length / image width'}</span>
              <InputNumber
                disabled={busy}
                min={0.2}
                max={4}
                step={0.1}
                value={focal}
                onChange={(v) => setFocal(Number(v ?? DEFAULT_VIDEO_FOCAL))}
              />
            </label>
            <label className="lab-switch">
              <Switch
                size="small"
                disabled={busy}
                checked={denseEnabled}
                onChange={setDenseEnabled}
                aria-describedby="lab-dense-help"
              />
              {ko ? '조밀 MVS' : 'Dense MVS'}
            </label>
            <p className="field-help" id="lab-dense-help">
              {ko
                ? '희소 복원 뒤에 같은 GPU에서 조밀 점군을 만듭니다. 소프트웨어 GPU에서는 수십 분이 걸릴 수 있습니다.'
                : 'Builds a dense point cloud on the same GPU after the sparse map. It can take tens of minutes on a software GPU.'}
            </p>
            {source && (
              <p className="field-help">
                {ko ? '원본' : 'Source'} {source.width} × {source.height} · {source.name}
              </p>
            )}
            {sample && (
              <p className="field-help sfm-sample-credit">
                {sample.credit}{' '}
                {ko
                  ? '초점거리와 왜곡은 데이터셋의 보정값을 사용하고, 참조 카메라는 ' +
                    sample.referenceName +
                    '입니다.'
                  : 'Intrinsics are the dataset calibration; the reference cameras are ' +
                    sample.referenceName +
                    '.'}
              </p>
            )}
            <div className="lab-run-actions">
              <Button
                type="primary"
                block
                disabled={busy || (!mvRows.length && !prepared.length && !source)}
                onClick={() => void (mvRows.length ? runMultiVideo() : runSfM())}
              >
                {ko ? 'GPU SfM 실행' : 'Run GPU SfM'}
              </Button>
              {busy && (
                <Button
                  danger
                  block
                  onClick={() => {
                    // 5.3: during the dense MVS the cancel stops the MVS and keeps the sparse map.
                    if (denseCancelRef.current) return denseCancelRef.current();
                    abortRef.current?.abort();
                    videoRef.current?.pause();
                    terminateWorker();
                  }}
                >
                  {denseRunning
                    ? ko
                      ? '조밀 MVS 취소'
                      : 'Cancel dense MVS'
                    : ko
                      ? '취소'
                      : 'Cancel'}
                </Button>
              )}
              <Button block disabled={busy || benchmarkRunning} onClick={() => void runBenchmark()}>
                {ko ? '표준 벤치마크' : 'Standard benchmark'}
              </Button>
              <p className="field-help">
                {ko
                  ? '표준 벤치마크는 피라미드 샘플 8프레임을 콜드와 웜으로 두 번 실행합니다.'
                  : 'The standard benchmark runs the eight-frame pyramid sample twice, cold and warm.'}
              </p>
            </div>
            <div className="lab-status" role="status" aria-live="polite">
              {status || (ko ? '영상 또는 샘플을 선택하세요.' : 'Choose a video or sample.')}
            </div>
            {(busy || complete) && (
              <Progress percent={progress} status={busy ? 'active' : 'success'} />
            )}
            {complete && (
              <p className="field-help">
                {ko ? '전체 소요 시간' : 'Total elapsed'} {(totalMs / 1000).toFixed(2)}s ·{' '}
                {ko
                  ? '전체 프레임 분석과 최근 구간의 마지막 정밀화 포함'
                  : 'Includes all frame analysis and final local refinement'}
              </p>
            )}
            {(denseSummary || denseCloud) && (
              <DenseSummary
                ko={ko}
                summary={denseSummary}
                cloud={denseCloud}
                budget={pointBudget}
              />
            )}
          </aside>
        </div>
      </Card>
      <div className="sfm-live-grid">
        <section className="sfm-stream lab-panel">
          <div className="sfm-map-head">
            <Title level={4}>{ko ? '분석 프레임' : 'Analyzed frame'}</Title>
            <label className="lab-switch">
              <Switch size="small" checked={showOverlay} onChange={setShowOverlay} />
              {ko ? '프레임 분석 표시' : 'Frame overlay'}
            </label>
          </div>
          <div className="sfm-media">
            <AnalysisFrame
              overlay={overlay}
              scene={mvRows.length ? undefined : reconstruction}
              annotate={showOverlay}
              transparent={!busy && !!videoURL}
              visible={
                busy ||
                !videoURL ||
                (videoAligned && !!selectedFrame && overlay?.frame.id === selectedFrame.id)
              }
            >
              <video
                ref={videoRef}
                src={videoURL || undefined}
                controls={!busy}
                muted
                playsInline
                preload="metadata"
                onPlay={syncAligned}
                onPause={syncAligned}
                onSeeking={syncAligned}
                onSeeked={syncAligned}
                onLoadedMetadata={() => {
                  const video = videoRef.current!;
                  if (!Number.isFinite(video.duration) || video.duration < 1) {
                    setSource(undefined);
                    setStatus(
                      ko
                        ? '1초 이상인 영상을 선택하세요.'
                        : 'Choose a video at least one second long.',
                    );
                    return;
                  }
                  if (!file) return;
                  setSource({
                    name: file.name,
                    width: video.videoWidth,
                    height: video.videoHeight,
                    duration: video.duration,
                  });
                  setStatus(
                    ko
                      ? 'GPU SfM 실행을 누르면 영상 전체를 순서대로 분석합니다.'
                      : 'Run GPU SfM to analyze the whole video in order.',
                  );
                }}
                onError={() =>
                  setStatus(
                    ko
                      ? '브라우저에서 영상을 디코딩할 수 없습니다. prepare-sfm-video.py로 만든 JSON을 선택하세요.'
                      : 'This browser cannot decode the video. Select JSON from prepare-sfm-video.py.',
                  )
                }
              />
            </AnalysisFrame>
            <div
              className="sfm-depth-legend"
              title={
                ko
                  ? '깊이는 카메라 앞쪽 축의 상대값이며 미터가 아닙니다.'
                  : 'Depth is along the camera forward axis in relative units, not meters.'
              }
            >
              <span>
                {ko ? '가까움' : 'Near'} {currentDepths.length ? currentRange[0].toFixed(2) : '—'}
              </span>
              <span className="sfm-depth-ramp" />
              <span>
                {ko ? '멀음' : 'Far'} {currentDepths.length ? currentRange[1].toFixed(2) : '—'}
              </span>
            </div>
          </div>
          <div className="sfm-timeline">
            <span className="sfm-timeline-text">
              {ko ? '프레임' : 'Frame'} {runFrames.length ? selectedIndex + 1 : '—'} /{' '}
              {runFrames.length || '—'} ·{' '}
              {selectedFrame ? selectedFrame.time.toFixed(2) + 's' : '—'}
            </span>
            <input
              type="range"
              aria-label="Frame timeline"
              min={0}
              max={Math.max(0, runFrames.length - 1)}
              value={selectedIndex}
              disabled={busy || runFrames.length === 0}
              onChange={(event) => {
                setReplaying(false);
                setSelectedIndex(Number(event.target.value));
              }}
            />
            <div className="sfm-timeline-actions">
              <Button
                size="small"
                aria-label={ko ? '이전 프레임' : 'Previous frame'}
                disabled={busy || selectedIndex === 0}
                onClick={() => setSelectedIndex(selectedIndex - 1)}
              >
                {ko ? '이전' : 'Previous'}
              </Button>
              <Button
                size="small"
                aria-label={ko ? '다음 프레임' : 'Next frame'}
                disabled={busy || selectedIndex >= runFrames.length - 1}
                onClick={() => setSelectedIndex(selectedIndex + 1)}
              >
                {ko ? '다음' : 'Next'}
              </Button>
              <Button
                size="small"
                aria-label={
                  replaying
                    ? ko
                      ? '재생 중지'
                      : 'Pause trajectory'
                    : ko
                      ? '카메라 이동 재생'
                      : 'Replay camera motion'
                }
                disabled={busy || runFrames.length === 0}
                onClick={() => {
                  if (selectedIndex >= runFrames.length - 1) setSelectedIndex(0);
                  setReplaying(!replaying);
                }}
              >
                {replaying ? (ko ? '중지' : 'Pause') : ko ? '재생' : 'Replay'}
              </Button>
            </div>
          </div>
          <ol
            className="sfm-pipeline"
            aria-label={ko ? '프레임 처리 과정' : 'Frame processing stages'}
          >
            {[
              [
                ko ? '특징점 검출' : 'Detect features',
                frameCounts.features + (ko ? '개' : ' points'),
              ],
              [
                ko ? '프레임 연결' : 'Match frames',
                frameCounts.matches + (ko ? ' 후보' : ' candidates'),
              ],
              [
                ko ? '카메라 추정' : 'Estimate camera',
                frameCounts.inliers + (ko ? ' 기하 인라이어' : ' geometric inliers'),
              ],
              [
                ko ? '3D 깊이 갱신' : 'Update 3D depth',
                currentDepths.length + (ko ? ' 깊이점' : ' depth points'),
              ],
            ].map(([label, count], i) => {
              const unavailable = frameFinished && i >= 2 && !selectedView;
              return (
                <li
                  key={label}
                  title={
                    unavailable
                      ? ko
                        ? '이 프레임은 카메라 위치·깊이를 확정하지 못했습니다. 검출점과 매칭 후보는 위에서 확인할 수 있습니다.'
                        : 'This frame has no registered camera or verified depth. Its detections and match candidates remain visible above.'
                      : undefined
                  }
                  data-state={
                    unavailable
                      ? 'unavailable'
                      : frameStep > i
                        ? 'complete'
                        : frameStep === i && busy
                          ? 'active'
                          : 'pending'
                  }
                >
                  <strong>
                    {i + 1}. {label}
                  </strong>
                  <span title={count}>{count}</span>
                </li>
              );
            })}
          </ol>
          <div className="sfm-tiles" aria-label={ko ? '프레임 통계' : 'Frame statistics'}>
            <Tile
              id="resolution"
              label={ko ? '해상도' : 'Resolution'}
              value={frameSize ? frameSize.width + ' × ' + frameSize.height : undefined}
            />
            <Tile
              id="features"
              label={ko ? '특징' : 'Features'}
              value={frameSize ? shownFeatures : undefined}
            />
            <Tile
              id="matches"
              label={ko ? '후보 매칭' : 'Candidate matches'}
              value={frameSize ? shownMatches : undefined}
            />
            <Tile
              id="processed"
              label={ko ? '처리 프레임' : 'Frames processed'}
              value={processed + skipped > 0 ? processed : undefined}
            />
            <Tile
              id="skipped"
              label={ko ? '건너뜀' : 'Skipped'}
              hint={
                (ko ? '건너뛴 프레임 (키프레임 아님)' : 'Frames skipped (not keyframes)') +
                (sampleFps > 0
                  ? ko
                    ? ` · 영상은 초당 ${sampleFps}프레임 추출`
                    : ` · video sampled at ${sampleFps} frames/s`
                  : '')
              }
              value={processed + skipped > 0 ? skipped : undefined}
            />
            <Tile
              id="rate"
              label={ko ? '분석 프레임/초' : 'Analysis frames/s'}
              value={frameMs > 0 ? (1000 / frameMs).toFixed(2) : undefined}
            />
            <Tile
              id="lag"
              label={ko ? '지연 (초)' : 'Lag (s)'}
              value={
                playTime > 0 || analysisTime > 0
                  ? Math.max(0, playTime - analysisTime).toFixed(2)
                  : undefined
              }
            />
            <Tile
              id="time"
              label={ko ? '영상 / 분석 (초)' : 'Video / analyzed (s)'}
              value={
                playTime > 0 || analysisTime > 0
                  ? playTime.toFixed(2) + ' / ' + analysisTime.toFixed(2)
                  : undefined
              }
            />
          </div>
          <details className="sfm-help">
            <summary>{ko ? '표시와 조작 안내' : 'Legend and controls'}</summary>
            <p>
              {ko
                ? '회색은 검출 특징, 주황 선은 후보 매칭, 색 점은 카메라 깊이(빨강 가까움, 파랑 멀음)입니다. 깊이는 카메라 앞쪽 축의 상대값이며 미터가 아닙니다.'
                : 'Gray marks detected features, orange lines candidate matches, and colored points the camera depth (red near, blue far). Depth is along the camera forward axis in relative units, not meters.'}
            </p>
            <p>
              {ko
                ? '분석이 시작되면 실제 특징점과 매칭이 표시됩니다. 영상은 키프레임이 아닌 프레임을 건너뛰며, 건너뛴 수는 통계에 표시됩니다.'
                : 'Actual features and matches appear when analysis starts. Frames that are not keyframes are skipped, and the statistics count them.'}
            </p>
            {sampleFps > 0 ? (
              <p className="sfm-sampling">
                {ko
                  ? `영상은 초당 ${sampleFps}프레임만 추출합니다. 나머지 프레임은 디코딩하지 않습니다.`
                  : `The video is sampled at ${sampleFps} frames/s. The other frames are not decoded.`}
              </p>
            ) : null}
          </details>
        </section>
        <section className="sfm-map lab-panel">
          <div className="sfm-map-head">
            <Title level={4}>{ko ? '3D 복원' : '3D reconstruction'}</Title>
            <div className="sfm-toolbar">
              <Radio.Group
                size="small"
                value={navigation}
                onChange={(event) => setNavigation(event.target.value)}
                aria-label={ko ? '이동 방식' : 'Navigation'}
              >
                <Radio.Button value="orbit">{ko ? '궤도' : 'Orbit'}</Radio.Button>
                <Radio.Button value="fly">{ko ? '자유 이동' : 'Fly'}</Radio.Button>
              </Radio.Group>
              <Tooltip title={ko ? '선택한 카메라 시점' : 'Look through selected camera'}>
                <Button
                  size="small"
                  aria-label={ko ? '선택한 카메라 시점' : 'Look through selected camera'}
                  disabled={
                    !selectedFrame ||
                    !reconstruction?.views.some((view) => view.id === selectedFrame.id)
                  }
                  onClick={() =>
                    selectedFrame &&
                    setCameraView({ id: selectedFrame.id, nonce: (cameraView?.nonce ?? 0) + 1 })
                  }
                >
                  {ko ? '카메라 시점' : 'Camera view'}
                </Button>
              </Tooltip>
              <Tooltip title={ko ? '전체 보기' : 'Fit the map'}>
                <Button
                  size="small"
                  aria-label={ko ? '전체 보기' : 'Fit the map'}
                  disabled={!reconstruction?.views.length}
                  onClick={() => setFitNonce((nonce) => nonce + 1)}
                >
                  {ko ? '전체' : 'Fit'}
                </Button>
              </Tooltip>
              <Popover
                trigger="click"
                placement="bottomRight"
                content={
                  <div className="sfm-map-options">
                    <label className="lab-switch">
                      <Switch size="small" checked={showCameras} onChange={setShowCameras} />
                      {ko ? '카메라' : 'Cameras'}
                    </label>
                    <label className="lab-switch">
                      <Switch size="small" checked={showPhoto} onChange={setShowPhoto} />
                      {ko ? '사진' : 'Photo'}
                    </label>
                    <label className="lab-switch">
                      <Switch
                        size="small"
                        checked={lightBackground}
                        onChange={(checked) => setBackground(checked ? 'light' : 'dark')}
                      />
                      {ko ? '밝은 배경' : 'Light background'}
                    </label>
                    <label className="lab-switch">
                      <Switch size="small" checked={followCamera} onChange={setFollowCamera} />
                      {ko ? '선택 카메라 따라가기' : 'Follow selected camera'}
                    </label>
                    <label className="lab-switch">
                      <Switch
                        size="small"
                        disabled={!denseCloud?.count}
                        checked={!!denseCloud?.count && showDense}
                        onChange={setShowDense}
                      />
                      {ko ? '조밀 점군' : 'Dense'}
                    </label>
                    <label className="lab-switch">
                      <Switch
                        size="small"
                        disabled={!denseMesh?.indices.length}
                        checked={!!denseMesh?.indices.length && showMesh}
                        onChange={setShowMesh}
                      />
                      {ko ? '메시' : 'Mesh'}
                    </label>
                  </div>
                }
              >
                <Button size="small" aria-haspopup="true">
                  {ko ? '표시' : 'Display'} ▾
                </Button>
              </Popover>
            </div>
          </div>
          {resultInfo && (
            <div className="result-info" aria-label={ko ? '열린 결과' : 'Open result'}>
              <Tooltip
                title={
                  [resultInfo.entry?.description?.[ko ? 'ko' : 'en'], ...resultInfo.notes]
                    .filter(Boolean)
                    .join(' · ') || resultInfo.name
                }
              >
                <strong>{resultInfo.name}</strong>
              </Tooltip>
              {resultInfo.entry && (
                <Tag color="gold">{deviceLabel(resultInfo.entry.run.device)}</Tag>
              )}
              {resultInfo.entry && (
                <Tag>
                  {resultInfo.entry.run.date} · {resultInfo.entry.run.commit.slice(0, 7)}
                </Tag>
              )}
              {resultInfo.cameras > 0 && (
                <Tag>
                  {resultInfo.cameras} {ko ? '카메라' : 'cameras'}
                </Tag>
              )}
              {resultInfo.sparse > 0 && (
                <Tag>
                  {resultInfo.sparse.toLocaleString('en-US')} {ko ? '점' : 'points'}
                </Tag>
              )}
              {resultInfo.dense > 0 && (
                <Tag>
                  {ko ? '조밀' : 'dense'} {resultInfo.dense.toLocaleString('en-US')}
                </Tag>
              )}
              {resultInfo.faces > 0 && (
                <Tag>
                  {ko ? '메시' : 'mesh'} {resultInfo.faces.toLocaleString('en-US')}
                </Tag>
              )}
              {resultInfo.record && !resultInfo.record.failure && (
                <Tag>
                  {resultInfo.record.result.registeredViews}/{resultInfo.record.input.frames}{' '}
                  {ko ? '등록' : 'registered'} ·{' '}
                  {(resultInfo.record.timing.totalMs / 1000).toFixed(0)} s
                </Tag>
              )}
              {resultInfo.record && (
                <Tooltip title={resultInfo.record.gpu.description || resultInfo.record.gpu.vendor}>
                  <Tag>{resultInfo.record.gpu.architecture || 'GPU'}</Tag>
                </Tooltip>
              )}
            </div>
          )}
          {mvGraph && mvGraph.components.some((c) => c.reconstructed) && (
            <ComponentSwitch
              ko={ko}
              components={mvGraph.components
                .filter((c) => c.reconstructed)
                .map((c) => ({ index: c.index, keyframes: c.keyframes, videos: c.videos.length }))}
              selected={mvSelected}
              onSelect={selectComponent}
            />
          )}
          <div className="sfm-media">
            <SfmViewer
              reconstruction={reconstruction}
              videoColors={mvRows.length ? mvVideoColors : undefined}
              onState={onViewerState}
              selectedViewId={mvRows.length ? mvView : overlay?.frame.id}
              photo={mvRows.length ? undefined : overlay?.frame}
              cameraView={cameraView}
              fitNonce={fitNonce}
              showCameras={showCameras}
              showPhoto={showPhoto}
              lightBackground={lightBackground}
              navigation={navigation}
              dense={denseCloud}
              showDense={showDense}
              mesh={denseMesh}
              showMesh={showMesh}
              densePointBudget={pointBudget}
              onSelectView={(id) => {
                if (mvRows.length) setMvView(id);
                else if (!busy) {
                  const index = runFrames.findIndex((frame) => frame.id === id);
                  if (index >= 0) setSelectedIndex(index);
                }
              }}
            />
            <p className="sfm-media-note">
              {navigation === 'orbit'
                ? ko
                  ? '드래그: 회전 · 휠: 확대 · 카메라 더블클릭: 회전 중심'
                  : 'Drag: orbit · Wheel: zoom · Double-click a camera: center'
                : ko
                  ? '클릭 후 드래그: 시선 · W A S D Q E: 이동'
                  : 'Click, then drag: look · W A S D Q E: move'}
            </p>
          </div>

          <div className="sfm-map-message">
            {viewerState.status === 'error' ? (
              <p role="alert">{viewerState.message}</p>
            ) : !reconstruction ? (
              <p role="status" className="field-help">
                {viewerState.status === 'initializing'
                  ? ko
                    ? '3D 뷰어를 준비하고 있습니다…'
                    : 'Preparing the 3D viewer…'
                  : ko
                    ? '안정적인 초기 시점 쌍을 찾으면 지도가 나타납니다.'
                    : 'The map appears after a stable initial pair is found.'}
              </p>
            ) : (
              <p role="status" className="field-help">
                {ko ? '지도를 돌려 보세요.' : 'The map is live. Turn it to look around.'}
              </p>
            )}
          </div>
          <div className="sfm-camera-motion">
            <strong>{ko ? '이전 → 선택 프레임' : 'Previous → selected frame'}</strong>
            <span className="field-help">
              {movement
                ? ko
                  ? '이동량은 복원 장면의 상대 단위입니다.'
                  : 'Relative scene units.'
                : ko
                  ? '두 프레임의 카메라가 등록되면 표시됩니다.'
                  : 'Shown when both frame poses are registered.'}
            </span>
          </div>
          <div className="sfm-tiles" aria-label={ko ? '복원 통계' : 'Reconstruction statistics'}>
            <Tile
              id="views"
              label={ko ? '등록 카메라' : 'Registered views'}
              value={reconstruction?.views.length}
            />
            <Tile
              id="landmarks"
              label={ko ? '희소 3D 점' : 'Sparse landmarks'}
              value={reconstruction?.landmarks.length}
            />
            <Tile
              id="error"
              label={ko ? '평균 오차' : 'Mean error'}
              hint={ko ? '평균 재투영 오차 (px)' : 'Mean reprojection error (px)'}
              value={
                typeof diagnostics?.meanReprojectionErrorPx === 'number'
                  ? diagnostics.meanReprojectionErrorPx.toFixed(3) + ' px'
                  : undefined
              }
            />
            <Tile
              id="translation"
              label={ko ? '이동량' : 'Translation'}
              hint={
                ko
                  ? '이전 → 선택 프레임. 복원 장면의 상대 단위이며, 카메라 위치가 없는 프레임은 보간하지 않습니다.'
                  : 'Previous → selected frame, in relative scene units. Missing poses are not interpolated.'
              }
              value={movement ? movement.distance.toFixed(4) : undefined}
            />
            <Tile
              id="rotation"
              label={ko ? '회전' : 'Rotation'}
              hint={ko ? '이전 → 선택 프레임' : 'Previous → selected frame'}
              value={movement ? movement.degrees.toFixed(2) + '°' : undefined}
            />
            {[
              ['right', ko ? '오른쪽' : 'Right'],
              ['down', ko ? '아래' : 'Down'],
              ['forward', ko ? '앞' : 'Forward'],
            ].map(([id, label], i) => (
              <Tile
                key={id}
                id={id}
                label={label}
                hint={
                  ko
                    ? '이전 → 선택 프레임, 카메라 기준 오른쪽 / 아래 / 앞 이동'
                    : 'Previous → selected frame, camera-relative right / down / forward'
                }
                value={movement ? movement.local[i].toFixed(4) : undefined}
              />
            ))}
          </div>
          <details className="sfm-help">
            <summary>
              {ko ? '표시와 조작 안내' : 'Legend and controls'}
              {unregistered.length > 0 &&
                ' · ' + unregistered.length + (ko ? '개 미등록' : ' unregistered')}
            </summary>
            <p>
              {ko
                ? '주황색은 카메라 경로, 밝은 초록색은 선택 카메라의 앞쪽, 파란 화살표는 직전 프레임에서의 이동입니다. 사진 평면은 선택 프레임의 실제 이미지입니다. 3D 점은 실제 영상의 색상입니다.'
                : 'Orange traces the camera path; green shows the selected camera forward direction; the blue arrow shows movement from the previous frame. The photo plane is the actual selected image. The 3D points carry the image colors.'}
            </p>
            <p>
              {ko
                ? '궤도: 휠은 커서 쪽으로 확대하고, 지도가 작아질수록 한 번의 움직임도 작아집니다. 카메라를 더블클릭하면 그곳이 회전 중심이 됩니다.'
                : 'Orbit: the wheel zooms toward the cursor, and each move shrinks with the zoom. Double-click a camera to make it the center of rotation.'}
            </p>
            <p>
              {ko
                ? '자유 이동: 캔버스를 클릭한 뒤 드래그로 시선을 돌리고, W A S D로 앞뒤·좌우, Q E로 아래위로 움직입니다. Shift는 4배 빠르게, Alt는 4배 느리게, 휠은 속도를 바꿉니다.'
                : 'Fly: click the canvas, then drag to look around. W A S D move forward, back, left and right; Q E move down and up. Shift moves 4× faster, Alt 4× slower, and the wheel changes the speed.'}
            </p>
            <p>
              {ko
                ? '선택 카메라 따라가기를 켜면 이전·다음 프레임과 썸네일 선택이 3D 시점도 그 카메라로 옮깁니다.'
                : 'With Follow selected camera on, Previous, Next and a thumbnail click also move the 3D view to that camera.'}
            </p>
            <p>
              {ko
                ? '이동량은 복원 장면의 상대 단위이며, 카메라 위치가 없는 프레임은 보간하지 않습니다.'
                : 'Translation uses relative scene units. Missing poses are not interpolated.'}
            </p>
            {alignment?.status === 'aligned' && (
              <p>
                {ko ? 'COLMAP 대비 회전 중앙값 ' : 'Median rotation vs COLMAP '}
                {alignment.rotationDeg.median.toFixed(3)}° ·{' '}
                {ko ? '중심 중앙값 ' : 'median center '}
                {(alignment.centerOfRadius.median * 100).toFixed(2)}
                {ko ? '% (궤적 반경)' : '% of radius'}
              </p>
            )}
            {sample && alignment && (
              <p
                className="sfm-reference field-help"
                data-status={alignment.status}
                data-common-views={alignment.commonViews}
              >
                {alignment.status === 'aligned'
                  ? (ko
                      ? 'COLMAP 참조와 공통 카메라 ' + alignment.commonViews + '개 · 회전 최대 '
                      : alignment.commonViews + ' cameras in common with COLMAP · rotation max ') +
                    alignment.rotationDeg.max.toFixed(3) +
                    (ko ? '° · 중심 최대 ' : '° · center max ') +
                    (alignment.centerOfRadius.max * 100).toFixed(2) +
                    (ko ? '% (참조 궤적 반경 기준)' : '% of the reference trajectory radius')
                  : (ko ? 'COLMAP 참조와 비교 불가: ' : 'No comparison with COLMAP: ') +
                    alignment.status +
                    (ko
                      ? ' (공통 카메라 ' + alignment.commonViews + '개)'
                      : ' (' + alignment.commonViews + ' cameras in common)')}
              </p>
            )}
            {unregistered.length > 0 && (
              <ul>
                {unregistered.map((view) => (
                  <li key={view.viewId}>
                    {view.viewId}: {view.reason}
                  </li>
                ))}
              </ul>
            )}
          </details>
        </section>
      </div>
      {frames.length > 0 && (
        <section
          className="lab-panel sfm-frames"
          aria-label={ko ? '분석 프레임 목록' : 'Analyzed frames'}
        >
          <Title level={5}>{ko ? '프레임' : 'Frames'}</Title>
          <div className="frame-grid">
            {frames.map((src, i) => {
              const index = Math.max(0, runFrames.length - frames.length) + i;
              const frame = prepared.length ? prepared[i] : runFrames[index];
              const selectable = !busy && runFrames.length > 0;
              const selected = runFrames.length > 0 && index === selectedIndex;
              return (
                <figure key={i} data-selected={selected || undefined}>
                  <button
                    type="button"
                    disabled={!selectable}
                    aria-current={selected ? 'true' : undefined}
                    aria-label={(ko ? '프레임 ' : 'Frame ') + (index + 1)}
                    onClick={() => {
                      setReplaying(false);
                      setSelectedIndex(index);
                    }}
                  >
                    <img src={src} alt="" />
                  </button>
                  <figcaption>
                    {index + 1} · {frame?.time.toFixed(2)}s
                  </figcaption>
                </figure>
              );
            })}
          </div>
          <p className="field-help">
            {ko
              ? '썸네일을 누르면 그 프레임이 선택됩니다. 동영상 썸네일은 최근 24개를 표시하고, 전체 처리 시점은 타임라인에서 선택할 수 있습니다.'
              : 'Click a thumbnail to select that frame. Video thumbnails show the latest 24 frames; the timeline includes every processed time.'}
          </p>
        </section>
      )}
      <Card className="lab-panel">
        {complete && (
          <div className="run-details">
            <Space>
              <Button size="small" onClick={() => download('ply')}>
                Export PLY
              </Button>
              <Button size="small" onClick={() => download('json')}>
                Export JSON
              </Button>
              {!!denseCloud?.count && (
                <Button size="small" onClick={downloadDense}>
                  {ko ? '조밀 PLY 내보내기' : 'Export dense PLY'}
                </Button>
              )}
              {!!denseMesh?.indices.length && (
                <Button size="small" onClick={downloadMesh}>
                  {ko ? '메시 PLY 내보내기' : 'Export mesh PLY'}
                </Button>
              )}
              <details>
                <summary>{ko ? 'GPU 실행 단계' : 'GPU execution stages'}</summary>
                {record && (
                  <table className="stage-summary">
                    <thead>
                      <tr>
                        <th>{ko ? '단계' : 'Operation'}</th>
                        <th>{ko ? '호출' : 'Calls'}</th>
                        <th>ms</th>
                        <th>{ko ? '디스패치' : 'Dispatches'}</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(record.timing.stages).map(([operation, stage]) => (
                        <tr key={operation}>
                          <td>{operation}</td>
                          <td>{stage.calls}</td>
                          <td>{Math.round(stage.elapsedMs)}</td>
                          <td>{stage.dispatches}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
                {record && <ChunkSummary record={record} ko={ko} className="chunk-summary" />}
                <pre>{JSON.stringify(stages, null, 2)}</pre>
              </details>
            </Space>
          </div>
        )}
        <section className="run-record" aria-label={ko ? '실행 기록' : 'Run record'}>
          <Title level={4}>{ko ? '실행 기록 보내기' : 'Send run records'}</Title>
          <p className="field-help">
            {ko
              ? '기록에는 GPU 어댑터 문자열, 브라우저 User-Agent, 화면 크기, 입력 프레임 수와 해상도, 옵션, 결과 개수, 단계별 시간, 긴 커널의 청크 측정값과 시간 예산, 실행한 GPU 커널 이름이 들어갑니다. 실패한 실행은 오류 메시지와 마지막 커널도 담습니다. 영상, 이미지, 파일 이름, IP 주소는 들어가지 않습니다. 아래 JSON이 전송되는 전부입니다.'
              : 'A record holds the GPU adapter strings, the browser user agent, screen size, input frame count and resolution, options, result counts, per-stage timings, the chunk measurements of the long kernels with the time budget, and the names of the GPU kernels that ran. A failed run also carries the error and the last kernel. It holds no video, image, file name or IP address. The JSON below is everything that is sent.'}
          </p>
          <Checkbox checked={consent} onChange={(event) => setConsent(event.target.checked)}>
            {ko
              ? '실행 기록을 TypeCV 개발자에게 보내는 데 동의합니다. 실행 전에 선택하면 완료와 실패 기록이 바로 전송됩니다.'
              : 'I agree to send run records to the TypeCV developers. Checked before a run, each record is sent as soon as the run completes or fails.'}
          </Checkbox>
          {record && (
            <>
              <p className="run-record-summary">
                {record.gpu.description || record.gpu.architecture || record.gpu.vendor || 'GPU'} ·{' '}
                {(record.timing.totalMs / 1000).toFixed(2)}s{' '}
                {record.timing.reusedDevice ? '(warm)' : '(cold)'} ·{' '}
                {record.failure
                  ? (ko ? '실패: ' : 'failed: ') +
                    (record.failure.operation || '-') +
                    (record.failure.kernel ? ' · ' + record.failure.kernel : '')
                  : record.result.registeredViews +
                    '/' +
                    record.input.frames +
                    (ko ? ' 등록 · ' : ' registered · ') +
                    record.result.landmarks +
                    (ko ? ' 점' : ' points')}
                {record.benchmark
                  ? ' · ' + record.benchmark.name + ' ' + record.benchmark.pass
                  : ''}
              </p>
              <ChunkSummary record={record} ko={ko} className="run-record-chunks" />
              <Space wrap align="start">
                <Input
                  className="run-record-note"
                  maxLength={RUN_RECORD_MAX_NOTE}
                  placeholder={ko ? '메모 (선택)' : 'Note (optional)'}
                  value={note}
                  onChange={(event) => {
                    setNote(event.target.value);
                    setRecord({
                      ...record,
                      note: event.target.value.slice(0, RUN_RECORD_MAX_NOTE),
                    });
                  }}
                />
                <Button
                  type="primary"
                  disabled={
                    !consent || sendState.status === 'sending' || sendState.status === 'sent'
                  }
                  onClick={() => void sendRecord()}
                >
                  {ko ? '기록 보내기' : 'Send record'}
                </Button>
                <Tooltip
                  title={
                    filesAvailable
                      ? ko
                        ? '이 실행의 결과 파일(재구성 JSON, 희소 PLY, 있으면 조밀 PLY와 메시 PLY)을 기록과 함께 보냅니다. 입력이 본인의 사진이나 영상이면 결과에 그 장면이 담길 수 있습니다.'
                        : "Sends this run's result files with the record: the reconstruction JSON and sparse PLY, and the dense and mesh PLY when the run made them. If the input is your own photos or video, the result can show that scene."
                      : ko
                        ? '이 서버는 결과 파일을 받지 않습니다.'
                        : 'This server does not accept result files.'
                  }
                >
                  <Checkbox
                    checked={sendFiles && filesAvailable}
                    disabled={!filesAvailable}
                    onChange={(event) => setSendFiles(event.target.checked)}
                  >
                    {ko ? '결과 데이터도 보내기' : 'Also send the result files'}
                  </Checkbox>
                </Tooltip>
                <Button onClick={saveRecord}>{ko ? 'JSON 다운로드' : 'Download JSON'}</Button>
                <Button
                  onClick={() => navigator.clipboard?.writeText(JSON.stringify(record, null, 2))}
                >
                  {ko ? '복사' : 'Copy'}
                </Button>
              </Space>
              <p className="run-record-status" role="status">
                {sendState.status === 'sent'
                  ? (ko ? '전송됨: ' : 'Sent: ') +
                    sendState.id +
                    (sendState.files
                      ? (ko ? ' · 파일 ' : ' · files ') +
                        sendState.files.stored.length +
                        (sendState.files.failed.length
                          ? (ko ? ', 보내지 못함: ' : ', not stored: ') +
                            sendState.files.failed
                              .map((file) => file.name + ' (' + file.reason + ')')
                              .join(', ')
                          : '')
                      : '')
                  : sendState.status === 'failed'
                    ? (ko ? '전송 실패: ' : 'Not sent: ') + sendState.message
                    : sendState.status === 'sending'
                      ? (ko ? '전송 중…' : 'Sending…') +
                        (sendState.progress
                          ? (ko ? ' 파일 ' : ' files ') +
                            sendState.progress.done +
                            '/' +
                            sendState.progress.total
                          : '')
                      : ko
                        ? '아직 보내지 않았습니다.'
                        : 'Not sent yet.'}
              </p>
              <details>
                <summary>{ko ? '전송될 JSON' : 'JSON to be sent'}</summary>
                <pre>{JSON.stringify(record, null, 2)}</pre>
              </details>
            </>
          )}
        </section>
        <p className="sample-credit">
          Sample: Blanka OG,{' '}
          <a href="https://www.pexels.com/video/drone-footage-of-a-pyramid-8134952/">
            Pexels pyramid video
          </a>{' '}
          · <a href="https://www.pexels.com/license/">Pexels License</a>
        </p>
      </Card>
    </div>
  );
}
