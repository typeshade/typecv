import { ViewerBoundary, ViewerDiagnostics } from './ViewerBoundary';
import {
  Button,
  Card,
  Checkbox,
  ConfigProvider,
  Drawer,
  Input,
  InputNumber,
  Popover,
  Radio,
  Space,
  Switch,
  Tabs,
  Tag,
  Tooltip,
  Typography,
  Upload,
  theme,
} from 'antd';
import {
  BorderOutlined,
  CaretRightOutlined,
  CopyOutlined,
  EyeOutlined,
  FullscreenExitOutlined,
  FullscreenOutlined,
  PauseOutlined,
  SettingOutlined,
  SlidersOutlined,
  StepBackwardOutlined,
  StepForwardOutlined,
  SwapOutlined,
} from '@ant-design/icons';
import {
  useCallback,
  useMemo,
  useEffect,
  useRef,
  useState,
  type CSSProperties,
  type ReactNode,
} from 'react';
import DenseActivityPanel from './DenseActivity';
import WorkProgress from './WorkProgress';
import VideoRangeEditor from './VideoRange';
import { videoRangeProgress, type VideoRange } from '../lib/video-range';
import GpuPreparationProgress from './GpuPreparation';
import MultiVideoActivityPanel from './MultiVideoActivity';
import {
  appendKeyframeThumbnail,
  type KeyframeThumbnail,
  type MultiVideoActivity,
} from '../lib/multi-video-activity';
import { isGpuPreparationPhase, type GpuPreparation } from '../lib/gpu-preparation';
import type { VideoConnectionGraph } from '../lib/video-connection-scene';
import {
  appendFusionPreview,
  denseActivityProgress,
  denseActivityText,
  type DenseActivity,
} from '../lib/dense-activity';
import SfmViewer, { type PlacementState, type ViewerState } from './SfmViewer';
import PlacementMenu, {
  UNPLACED,
  type PlacementCommand,
  type TransformMode,
} from './PlacementMenu';
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
  partialVideos,
  type MultiVideoSummary,
  type VideoFacts,
} from '../lib/multi-video';
import {
  STAGE_WORDS,
  advance as advanceStages,
  finish as finishStages,
  formatElapsed,
  newBoard as newStageBoard,
  stageCount,
  stageTimings,
  statusLine as stageStatusLine,
  tick as tickStages,
  touch as touchStages,
  type StageBoard,
  type StageEvent,
} from '../lib/multi-video-stages';
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
  RUN_RECORD_MAX_FEATURES,
  gpuRecord,
  RUN_RECORD_MAX_FRAMES,
  RUN_RECORD_MAX_NOTE,
  chunkLines,
  chunkRecord,
  denseLine,
  kltLine,
  denseRecord,
  runRecordProblem,
  sumDenseRecords,
  summarizeStages,
  type RunRecord,
  type RunRecordDense,
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
  type DensePreview,
  type DensePreviewView,
} from '../lib/dense-cloud';
import { DENSE_NO_COMPONENT, denseReason, denseSwitchTitle } from '../lib/dense-reason';
import { ModelAssembler } from '../lib/model-chunks';
import { freshness, pulseTarget, type PulseTarget } from '../lib/live-view';
import {
  cameraMotion,
  depthColor,
  depthPoints,
  depthRange,
  previewSeeker,
  previousCamera,
  unpackScene,
  type DepthPoint,
  type PartialScene,
  type Scene,
} from '../lib/sfm-scene';
const { Title } = Typography;
type Frame = {
  id: number;
  width: number;
  height: number;
  pixels: Uint8ClampedArray;
  time: number;
  videoId?: string;
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
/** The input drawer's tabs: every entry, or one kind. */
const INPUT_TABS = ['all', 'video', 'photos', 'set', 'result'] as const;
type InputTab = (typeof INPUT_TABS)[number];
const INPUT_TAB_LABELS: Record<InputTab, [string, string]> = {
  all: ['All', '전체'],
  video: ['Videos', '영상'],
  photos: ['Photos', '사진'],
  set: ['Video sets', '영상 세트'],
  result: ['Saved results', '저장된 결과'],
};
function inputTabOf(entry: CatalogEntry, tab: InputTab): boolean {
  if (tab === 'all') return true;
  if (tab === 'set') return isVideoSet(entry);
  if (tab === 'result') return isResult(entry);
  return !isVideoSet(entry) && !isResult(entry) && entry.kind === tab;
}

/**
 * The focal length / image width of an uncalibrated video before the run's own estimate
 * (`focalSearch`, docs/auto-focal.md), and the start of a run with a typed focal and of the
 * standard benchmark. 1.2 (a DSLR's normal lens) inverted the pyramid sample's relief. The search
 * does not depend on this value. Phone main cameras are about 0.72.
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

/**
 * The lab's `?denseClean=1` or `0`: the cleanup of the fused cloud (`docs/dense-mvs.md` 3.12) on or
 * off. Undefined without the parameter: the worker cleans after the pixel fusion only
 * (`denseCleanDefault`).
 */
function denseCleanOverride(): boolean | undefined {
  if (typeof location === 'undefined') return undefined;
  const value = new URLSearchParams(location.search).get('denseClean');
  return value === '1' ? true : value === '0' ? false : undefined;
}

/** The lab's `?denseFusion=pixel|tsdf`: the fusion (3.10 or 10.4). Undefined: the lab's default. */
function denseFusionOverride(): 'pixel' | 'tsdf' | undefined {
  if (typeof location === 'undefined') return undefined;
  const value = new URLSearchParams(location.search).get('denseFusion');
  return value === 'pixel' || value === 'tsdf' ? value : undefined;
}

const DENSE_PRESETS = [
  {
    id: 'tsdf-raw',
    fusion: 'tsdf',
    clean: false,
    label: ['TSDF · 정리 끄기 (권장)', 'TSDF · Cleanup off (recommended)'],
  },
  {
    id: 'tsdf-clean',
    fusion: 'tsdf',
    clean: true,
    label: ['TSDF · 정리 켜기', 'TSDF · Cleanup on'],
  },
  {
    id: 'pixel-raw',
    fusion: 'pixel',
    clean: false,
    label: ['픽셀 융합 · 정리 끄기', 'Pixel fusion · Cleanup off'],
  },
  {
    id: 'pixel-clean',
    fusion: 'pixel',
    clean: true,
    label: ['픽셀 융합 · 정리 켜기', 'Pixel fusion · Cleanup on'],
  },
] as const;

/** The page's words for the stages of the dense MVS (5.3). */
const DENSE_STAGES: Record<string, [string, string]> = {
  photometric: ['깊이 추정', 'photometric pass'],
  geometric: ['깊이 검증', 'geometric pass'],
  fusion: ['포인트 합치기', 'fusion'],
};

/** The words of a dense status (`denseRecord`). */
const DENSE_STATUS: Record<string, [string, string]> = {
  completed: ['완료', 'completed'],
  cancelled: ['취소됨', 'cancelled'],
  failed: ['실패', 'failed'],
  skipped: ['건너뜀', 'skipped'],
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
        <strong>{ko ? 'Dense MVS' : 'Dense MVS'}</strong> · {status} · {ko ? '점 ' : ''}
        {(cleaned ? summary.clean!.after : (summary?.points ?? count)).toLocaleString()}
        {ko ? '개' : ' points'}
        {cleaned &&
          (ko
            ? ` (원본 ${summary.clean!.before.toLocaleString()})`
            : ` (raw ${summary.clean!.before.toLocaleString()})`)}
        {summary && (ko ? ` · 깊이 맵 ${summary.views}개` : ` · ${summary.views} depth maps`)}
        {summary?.planarFilled !== undefined &&
          (ko
            ? ` · 평면 보완 ${summary.planarFilled.toLocaleString()}픽셀`
            : ` · ${summary.planarFilled.toLocaleString()} depth pixels completed`)}
      </p>
      {summary && denseReason(summary, true, ko) && (
        <p className="field-help" data-testid="dense-reason">
          {denseReason(summary, true, ko)}
        </p>
      )}
      {summary && summary.status !== 'skipped' && (
        <p className="field-help">
          {(ko ? '추정 ' : 'photometric ') + (summary.stageMs.photometric / 1000).toFixed(1)} s ·{' '}
          {(ko ? '검증 ' : 'geometric ') + (summary.stageMs.geometric / 1000).toFixed(1)} s ·{' '}
          {(ko ? '융합 ' : 'fusion ') + (summary.stageMs.fusion / 1000).toFixed(1)} s ·{' '}
          {(ko ? '전체 ' : 'total ') + (summary.totalMs / 1000).toFixed(1)} s
        </p>
      )}
      {stride > 1 && (
        <p className="field-help">
          {ko
            ? `화면에는 ${budget.toLocaleString()}점까지만 그려서 ${stride}개 중 1개만 보입니다. 내보내기에는 모든 점이 들어갑니다.`
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
  range?: VideoRange;
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
        maxComputeWorkgroupStorageSize: adapter?.limits?.maxComputeWorkgroupStorageSize ?? 0,
        maxComputeInvocationsPerWorkgroup: adapter?.limits?.maxComputeInvocationsPerWorkgroup ?? 0,
        maxStorageBuffersPerShaderStage: adapter?.limits?.maxStorageBuffersPerShaderStage ?? 0,
      },
      features: (Array.isArray(adapter?.features) ? adapter.features : [])
        .slice(0, RUN_RECORD_MAX_FEATURES)
        .map((name: unknown) => String(name).slice(0, 64)),
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
      ...(input.range ? { range: input.range } : {}),
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
      ...(options.focalSearch === true ? { focalSearch: true } : {}),
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
      ...(result.focalSearch
        ? {
            focalSearch: {
              ratio: result.focalSearch.ratio,
              gridRatio: result.focalSearch.gridRatio,
              views: result.focalSearch.views,
            },
          }
        : {}),
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
      // The dense MVS's table is the whole job's: it covers the sparse run too.
      ...gpuRecord(
        input.dense?.gpuKernels ?? result.gpuKernels,
        input.dense?.gpuDropped ?? result.gpuDropped,
      ),
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
  /**
   * The dense MVS of each component, by component index (`docs/multi-video.md`, Amendment 9), and
   * the chunk rates the dense device reported last. Absent for a run without the dense MVS.
   */
  dense?: { blocks: ReadonlyMap<number, RunRecordDense>; chunks?: unknown };
  /**
   * A failed run has no worker `result`. It passes what the page saw: the time of the stages so
   * far, the videos it fed, and the maps that existed at the failure (the final map of a
   * component, or else its last partial map).
   */
  partial?: {
    stages: ReturnType<typeof stageTimings>;
    totalMs: number;
    videos: MultiVideoSummary['videos'][number][];
    scenes: ReadonlyMap<number, Scene>;
    /** The device's per-kernel table up to the failure, from the worker's `error` message. */
    gpuKernels?: unknown[];
    gpuDropped?: number;
  };
}): RunRecord {
  const { models } = input;
  const result =
    input.result ??
    (input.partial && {
      stages: input.partial.stages,
      totalMs: input.partial.totalMs,
      gpuKernels: input.partial.gpuKernels,
      gpuDropped: input.partial.gpuDropped,
    });
  const scenes = [...(input.partial?.scenes ?? models).values()];
  const registered = scenes.reduce((n, m) => n + m.views.length, 0);
  const landmarks = scenes.reduce((n, m) => n + m.landmarks.length, 0);
  const observations = scenes.reduce(
    (n, m) =>
      n +
      ((m as Partial<PartialScene>).observationCount ??
        m.landmarks.reduce((k, p) => k + p.observations.length, 0)),
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
    chunks: input.dense?.chunks ?? {},
    gpuKernels: result?.gpuKernels,
    gpuDropped: result?.gpuDropped,
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
  const summary = input.result?.summary;
  const videos = (summary?.videos ?? input.partial?.videos ?? []).map((v: any) => {
    const range = input.rows.find((row) => row.videoId === v.videoId)?.range;
    return {
      ...v,
      // The typed name of an upload is never sent; a lab sample is public (the record, 6.4).
      label: v.label ?? null,
      ...(range ? { range } : {}),
    };
  });
  // The summary lists the reconstructed components in the order of the graph's components.
  const reconstructed: number[] = (input.graph?.components ?? [])
    .filter((c: any) => c.reconstructed)
    .map((c: any) => c.index);
  const denseBlocks = input.dense?.blocks;
  const components = summary
    ? summary.components.map((c: any, i: number) => {
        const block = denseBlocks?.get(reconstructed[i]!);
        return block ? { ...c, dense: block } : c;
      })
    : undefined;
  const denseSum = denseBlocks?.size
    ? sumDenseRecords([...denseBlocks.entries()].sort((a, b) => a[0] - b[0]).map(([, d]) => d))
    : undefined;
  return {
    ...base,
    input: { ...base.input, ...(videos.length ? { videos } : {}) },
    options: { ...base.options, multiVideo: multiVideoOptions({ query: input.query }) },
    ...(denseSum ? { dense: denseSum } : {}),
    result: {
      ...base.result,
      ...(summary
        ? {
            components,
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

/**
 * Draw an analysis frame and its marks: the decoded pixels (unless `transparent`), the candidate
 * matches while no depth point exists, the detections and the depth points.
 */
function paintOverlay(
  canvas: HTMLCanvasElement,
  overlay: Overlay,
  depths: DepthPoint[],
  range: [number, number],
  annotate: boolean,
  transparent: boolean,
): void {
  const { frame, points } = overlay;
  canvas.width = frame.width;
  canvas.height = frame.height;
  const ctx = canvas.getContext('2d')!;
  if (transparent) ctx.clearRect(0, 0, canvas.width, canvas.height);
  else if (frame.pixels.length !== frame.width * frame.height * 4) {
    // The pixels went to a worker: the marks are drawn on a plain background.
    ctx.fillStyle = '#0b1020';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  } else {
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
    paintOverlay(ref.current, overlay, depths, range, annotate, transparent);
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
        data-frame={overlay?.frame.id}
        data-shown={!!overlay && visible}
        data-mode={transparent ? 'marks' : 'frame'}
        aria-label="Actual video frame with reconstructed points colored by camera depth"
      />
    </div>
  );
}

/**
 * The frame the worker analyses now, small, in a corner of the 3D view (`docs/lab-live-view.md`):
 * its detections and candidate matches as they arrive, so the map panel moves with every frame
 * even while no camera is added.
 */
function FrameInset({ overlay, ko }: { overlay?: Overlay; ko: boolean }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    if (overlay && ref.current) paintOverlay(ref.current, overlay, [], [0, 1], true, false);
  }, [overlay]);
  if (!overlay) return null;
  const features = (overlay.points?.length ?? 0) / 2;
  return (
    <figure className="sfm-inset" aria-live="off">
      <canvas
        ref={ref}
        aria-label={ko ? '지금 분석 중인 프레임' : 'The frame being analysed now'}
      />
      <figcaption>
        {(ko ? '분석 중 #' : 'Analysing #') +
          overlay.frame.id +
          ' · ' +
          features +
          (ko ? ' 특징' : ' features') +
          (overlay.matches !== undefined
            ? ' · ' + overlay.matches + (ko ? ' 매칭' : ' matches')
            : '')}
      </figcaption>
    </figure>
  );
}

/**
 * How long ago the worker last reported, redrawn every second while a run is busy. Past
 * `QUIET_MS` it says how long the GPU has worked without a message (`docs/lab-live-view.md`).
 */
function LiveBadge({ lastEventAt, ko }: { lastEventAt: { current: number }; ko: boolean }) {
  const [now, setNow] = useState(() => performance.now());
  useEffect(() => {
    const timer = window.setInterval(() => setNow(performance.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);
  const { text, quiet } = freshness(lastEventAt.current, now, ko);
  return (
    <span className="sfm-live" data-quiet={quiet} role="status">
      <span className="sfm-live-dot" aria-hidden="true" />
      {text}
    </span>
  );
}

/** The frames whose features a single-video run keeps for the 2D–3D links (about 32 KB each). */
const FEATURE_FRAMES = 1024;

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
  const [analysisRange, setAnalysisRange] = useState<VideoRange>();
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
  const [running, setRunning] = useState(false);
  const [finalizing, setFinalizing] = useState(false);
  const [preparation, setPreparation] = useState<GpuPreparation>();
  // The live view (`docs/lab-live-view.md`): the stage the worker reported last and when, and the
  // time of its last message of any kind.
  const [activity, setActivity] = useState<{ target: PulseTarget; at: number; viewId?: number }>();
  const lastEventAt = useRef(0);
  const noteActivity = useCallback((target: PulseTarget, viewId?: number) => {
    const now = performance.now();
    lastEventAt.current = now;
    // A quarter second between updates of the same target keeps the page's renders few.
    setActivity((previous) =>
      previous &&
      previous.target === target &&
      previous.viewId === viewId &&
      now - previous.at < 250
        ? previous
        : { target, at: now, viewId },
    );
  }, []);
  const [focal, setFocal] = useState(DEFAULT_VIDEO_FOCAL);
  /** The run finds the focal itself (`docs/auto-focal.md`); off when the visitor types one. */
  const [focalAuto, setFocalAuto] = useState(true);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [selectionNonce, setSelectionNonce] = useState(0);
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
  // A record to send by consent, once the page has rendered the run's results: `run` holds the
  // state of the render it started in, so the result files must be built in a later render.
  const [autoSend, setAutoSend] = useState<RunRecord>();
  useEffect(() => {
    let live = true;
    void resultFilesAvailable().then((available) => live && setFilesAvailable(available));
    return () => {
      live = false;
    };
  }, []);
  const [benchmarkRunning, setBenchmarkRunning] = useState(false);
  const benchmarkStopRef = useRef(false);
  /**
   * A multi-video run (`docs/multi-video.md` 6): the rows of the upload list (two or more videos),
   * the view graph the worker posted, the model of each component and the component on show.
   */
  const [mvRows, setMvRows] = useState<VideoRow[]>([]);
  const [mvGraph, setMvGraph] = useState<MvGraph>();
  const [mvSelected, setMvSelected] = useState(0);
  const [mvProgress, setMvProgress] = useState(0);
  /** The stage list of the multi-video run that is under way or just ended. */
  const [mvBoard, setMvBoard] = useState<StageBoard | undefined>();
  const [mvActivity, setMvActivity] = useState<MultiVideoActivity>();
  const [mvConnectionGraph, setMvConnectionGraph] = useState<VideoConnectionGraph>();
  const [mvThumbnails, setMvThumbnails] = useState<KeyframeThumbnail[]>([]);
  const [mvActiveVideo, setMvActiveVideo] = useState<string>();
  /** The timeline stays within one source video of the displayed component. */
  const [selectedVideo, setSelectedVideo] = useState<string>();
  const mvReplaySources = useRef(new Map<string, File | string>());
  const mvModels = useRef(new Map<number, Scene>());
  const mvSelectedRef = useRef(0);
  mvSelectedRef.current = mvSelected;
  /**
   * The dense MVS of each component of a multi-video run (`docs/multi-video.md`, Amendment 9): the
   * points that arrived, the mesh and the summary. The viewer and the export read the selected one.
   */
  const mvDense = useRef(
    new Map<
      number,
      {
        points: any[];
        count: number;
        cloud?: DenseCloud;
        fusionCloud?: DenseCloud;
        mesh?: DenseMeshData;
        summary?: RunRecordDense;
      }
    >(),
  );
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
  /** A preset view of the map (top, front, side), applied again on each click. */
  // The focal length / image width the run ended with: the median over the registered views.
  const estimatedFocal = (() => {
    const ratios = (reconstruction?.views ?? [])
      .map((view) => {
        const width = (view as { width?: number }).width;
        return width ? view.camera.fx / width : NaN;
      })
      .filter(Number.isFinite)
      .sort((a, b) => a - b);
    return ratios.length ? ratios[Math.floor(ratios.length / 2)] : undefined;
  })();
  const [viewPreset, setViewPreset] = useState<{ kind: 'top' | 'front' | 'side'; nonce: number }>();
  // The placement of the map on the grid (the 3D card's Place menu).
  const [transformMode, setTransformMode] = useState<TransformMode>('off');
  const [placementCommand, setPlacementCommand] = useState<PlacementCommand>();
  const [placement, setPlacement] = useState<PlacementState>(UNPLACED);
  const runPlacement = useCallback(
    (kind: PlacementCommand['kind'], value?: PlacementCommand['value']) =>
      setPlacementCommand((p) => ({
        kind,
        nonce: (p?.nonce ?? 0) + 1,
        ...(value ? { value } : {}),
      })),
    [],
  );
  /** The viewer's display options. The background follows the page theme until the visitor
   *  picks one. */
  const [showCameras, setShowCameras] = useState(true);
  const [showPhoto, setShowPhoto] = useState(true);
  const [showLinks, setShowLinks] = useState(true);
  // The features of each analysed frame of a single-video run, for the 3D view's 2D–3D links
  // (`docs/lab-live-view.md` 3.5). Float32 halves the memory; the oldest go past the bound.
  const frameFeatures = useRef(new Map<number, Float32Array>());
  const [showOverlay, setShowOverlay] = useState(true);
  const [background, setBackground] = useState<'auto' | 'light' | 'dark'>('auto');
  const lightBackground = background === 'auto' ? !dark : background === 'light';
  const [navigation, setNavigation] = useState<'orbit' | 'fly'>('orbit');
  /** With the option on, selecting a frame also moves the 3D view to its camera. */
  const [followCamera, setFollowCamera] = useState(false);
  /** Dense MVS follows sparse reconstruction by default; it can be disabled in run settings. */
  const [denseEnabled, setDenseEnabled] = useState(true);
  const [densePreset, setDensePreset] = useState<(typeof DENSE_PRESETS)[number]>(DENSE_PRESETS[0]);
  useEffect(() => {
    // Old links initialize the visible controls once; edits in the page govern the next run.
    const fusion = denseFusionOverride() ?? 'tsdf';
    const clean = denseCleanOverride() ?? fusion === 'pixel';
    setDensePreset(
      DENSE_PRESETS.find((preset) => preset.fusion === fusion && preset.clean === clean)!,
    );
  }, []);

  /** The mesh of the dense cloud (`docs/dense-mvs.md` 3.13): the run settings' switch, or `?denseMesh=1`. */
  const [meshEnabled, setMeshEnabled] = useState(false);
  const meshEnabledRef = useRef(false);
  meshEnabledRef.current = meshEnabled;
  useEffect(() => {
    if (denseMeshOverride()) setMeshEnabled(true);
  }, []);
  /** The 3D panel: in the page, wide (full width of the grid) or fullscreen. */
  const mapRef = useRef<HTMLElement>(null);
  const [mapWide, setMapWide] = useState(false);
  const [mapFullscreen, setMapFullscreen] = useState(false);
  useEffect(() => {
    const onChange = () => setMapFullscreen(document.fullscreenElement === mapRef.current);
    document.addEventListener('fullscreenchange', onChange);
    return () => document.removeEventListener('fullscreenchange', onChange);
  }, []);
  const denseEnabledRef = useRef(true);
  denseEnabledRef.current = denseEnabled;
  useEffect(() => {
    if (denseOverride()) setDenseEnabled(true);
  }, []);
  /** The dense cloud of the last run, the viewer's switch and the summary of `denseResult`. */
  const [denseCloud, setDenseCloud] = useState<DenseCloud>();
  const [denseFusionCloud, setDenseFusionCloud] = useState<DenseCloud>();
  const [denseActivity, setDenseActivity] = useState<DenseActivity>();
  const displayedDense = denseFusionCloud ?? denseCloud;
  // The dense MVS's depth previews (`docs/dense-mvs.md`, Amendment 5): the latest of each view.
  const [densePreview, setDensePreview] = useState<DensePreview>();
  /** The mesh of the final cloud (`docs/dense-mvs.md` 3.13). Hidden until the Display switch. */
  const [denseMesh, setDenseMesh] = useState<DenseMeshData>();
  const [showMesh, setShowMesh] = useState(false);
  const [showDense, setShowDense] = useState(true);
  const [denseSummary, setDenseSummary] = useState<ReturnType<typeof denseRecord>>();
  /** Whether the last run asked for the dense MVS; undefined for a result loaded from files. */
  const [denseAsked, setDenseAsked] = useState<boolean>();
  const pointBudget =
    typeof navigator === 'undefined'
      ? densePointBudget(0)
      : densePointBudget(navigator.maxTouchPoints);
  /** The catalog entry whose frames are loaded, for the card's highlight. */
  const [loadedSampleId, setLoadedSampleId] = useState<string>();
  // The input drawer: opened by the input bar's button, closed once an input is in the lab.
  const [inputOpen, setInputOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [inputTab, setInputTab] = useState<InputTab>('all');
  // The drawer is a portal: it renders after hydration so the server's HTML matches the client's.
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  const inTab = (entry: CatalogEntry) => inputTabOf(entry, inputTab);
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
    mvDense.current.clear();
    setDenseCloud(undefined);
    setDenseFusionCloud(undefined);
    setDenseActivity(undefined);
    setDensePreview(undefined);
    setDenseMesh(undefined);
    setDenseSummary(undefined);
  };
  const terminateWorker = () => {
    workerRef.current?.terminate();
    workerRef.current = undefined;
  };
  const componentFrames = useMemo<Frame[]>(
    () =>
      complete && mvRows.length
        ? (reconstruction?.views ?? [])
            .map((view) => ({
              id: view.id,
              videoId: view.videoId,
              time: view.time ?? 0,
              width: view.width ?? Math.round(view.camera.cx * 2 + 1),
              height: view.height ?? Math.round(view.camera.cy * 2 + 1),
              image: view.image,
              pixels: new Uint8ClampedArray(),
            }))
            .sort((a, b) => a.time - b.time || a.id - b.id)
        : [],
    [complete, !!mvRows.length, reconstruction],
  );
  const videoIds = [...new Set(componentFrames.map((frame) => frame.videoId))];
  const activeVideo = videoIds.includes(selectedVideo) ? selectedVideo : videoIds[0];
  const navigationFrames = useMemo(
    () =>
      complete && mvRows.length
        ? componentFrames.filter((frame) => frame.videoId === activeVideo)
        : runFrames,
    [complete, !!mvRows.length, componentFrames, activeVideo, runFrames],
  );
  const gallery =
    complete && mvRows.length
      ? navigationFrames
          .slice(-24)
          .map((frame, i) => ({
            src: frame.image,
            frame,
            index: Math.max(0, navigationFrames.length - 24) + i,
          }))
          .filter((item): item is typeof item & { src: string } => !!item.src)
      : frames.map((src, i) => {
          const index = Math.max(0, runFrames.length - frames.length) + i;
          return { src, index, frame: prepared.length ? prepared[i] : runFrames[index] };
        });
  const navigationIndex = Math.min(selectedIndex, Math.max(0, navigationFrames.length - 1));
  const selectedFrame = navigationFrames[navigationIndex];
  const displayedOverlay = busy || overlay?.frame.id === selectedFrame?.id ? overlay : undefined;
  const selectedViewId = busy ? (mvRows.length ? undefined : overlay?.frame.id) : selectedFrame?.id;
  const selectedView = reconstruction?.views.find((view) => view.id === selectedViewId);
  const previousView = previousCamera(reconstruction, selectedView);
  const movement = cameraMotion(previousView, selectedView);
  const frameSize = displayedOverlay?.frame ?? selectedFrame;
  const shownFeatures = displayedOverlay?.points
    ? displayedOverlay.points.length / 2
    : frameCounts.features;
  const selectedFeatures = useMemo(() => {
    const id = displayedOverlay?.frame.id;
    const points = id === undefined || mvRows.length ? undefined : frameFeatures.current.get(id);
    return points && id !== undefined ? { viewId: id, points } : undefined;
  }, [displayedOverlay, !!mvRows.length]);
  const shownMatches = displayedOverlay?.matches ?? frameCounts.matches;
  const overlayScene = busy && mvRows.length ? undefined : reconstruction;
  const currentDepths = depthPoints(overlayScene, displayedOverlay?.frame.id ?? -1);
  const currentRange = depthRange(currentDepths);
  const playbackInput = mvRows.length
    ? selectedFrame?.videoId && mvReplaySources.current.get(selectedFrame.videoId)
    : (file ?? (videoSourceRef.current?.decodable ? videoSourceRef.current.url : undefined));
  // Native playback can advance the selection without seeking itself back to sampled times.
  const videoSelection = useRef<Frame | undefined>(undefined);
  const programmaticSeek = useRef<number | undefined>(undefined);
  const videoSourceLoading = useRef(false);
  const [videoReady, setVideoReady] = useState(0);
  const [frameLoadError, setFrameLoadError] = useState(false);
  const selectFrame = (index: number) => {
    setReplaying(false);
    videoSelection.current = undefined;
    videoRef.current?.pause();
    setSelectedIndex(index);
    setSelectionNonce((nonce) => nonce + 1);
  };
  useEffect(() => {
    if (!replaying || !complete || displayedOverlay?.frame.id !== selectedFrame?.id) return;
    const timer = window.setTimeout(() => {
      if (navigationIndex >= navigationFrames.length - 1) setReplaying(false);
      else setSelectedIndex(navigationIndex + 1);
    }, 250);
    return () => window.clearTimeout(timer);
  }, [replaying, complete, navigationIndex, navigationFrames.length, displayedOverlay?.frame.id]);
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
  const syncVideoSelection = () => {
    syncAligned();
    const video = videoRef.current;
    if (busy || !complete || !video || videoSourceLoading.current || !navigationFrames.length)
      return;
    if (programmaticSeek.current !== undefined) {
      if (!video.seeking && Math.abs(video.currentTime - programmaticSeek.current) < 0.02)
        programmaticSeek.current = undefined;
      return;
    }
    if (video.seeking) return;
    const nearest = navigationFrames.reduce(
      (best, frame, index) =>
        Math.abs(frame.time - video.currentTime) <
        Math.abs(navigationFrames[best].time - video.currentTime)
          ? index
          : best,
      0,
    );
    if (nearest !== navigationIndex) {
      videoSelection.current = navigationFrames[nearest];
      setReplaying(false);
      setSelectedIndex(nearest);
    }
  };
  useEffect(() => {
    setVideoAligned(false);
    videoSourceLoading.current = !!playbackInput;
    programmaticSeek.current = undefined;
    if (
      !playbackInput ||
      (playbackInput instanceof File && playbackInput.name.toLowerCase().endsWith('.json'))
    ) {
      setVideoURL('');
      return;
    }
    if (typeof playbackInput === 'string') {
      setVideoURL(new URL(playbackInput, location.href).href);
      return;
    }
    const url = URL.createObjectURL(playbackInput);
    setVideoURL(url);
    return () => URL.revokeObjectURL(url);
  }, [playbackInput]);
  useEffect(() => {
    if (busy || !complete || !selectedFrame || !videoURL) return;
    if (videoSelection.current === selectedFrame) {
      videoSelection.current = undefined;
      return;
    }
    const video = videoRef.current;
    if (!video || !video.readyState) return;
    video.pause();
    if (Math.abs(video.currentTime - selectedFrame.time) >= 0.02) {
      setVideoAligned(false);
      programmaticSeek.current = selectedFrame.time;
      video.currentTime = selectedFrame.time;
    } else {
      programmaticSeek.current = undefined;
      syncAligned();
    }
  }, [selectedFrame, selectionNonce, complete, busy, videoURL, videoReady]);
  useEffect(() => {
    if (busy || !complete || !selectedFrame) return;
    const abort = new AbortController();
    setFrameLoadError(false);
    const load = async () => {
      const original = mvRows.length
        ? undefined
        : prepared.find((frame) => frame.id === selectedFrame.id);
      setFrameCounts(selectedFrame.metrics ?? { features: 0, matches: 0, inliers: 0 });
      setFrameStep(reconstruction?.views.some((view) => view.id === selectedFrame.id) ? 4 : 2);
      setFrameFinished(true);
      if (original) {
        setOverlay({ frame: original });
        return;
      }
      // Multi-video scenes retain keyframe images with canonical view ids. Decode those exact
      // pixels; a source frame's numeric id belongs to its video, not to the combined scene.
      const image = selectedFrame.image;
      if (image) {
        const response = await fetch(image, { signal: abort.signal });
        if (!response.ok) throw new Error('Could not load frame');
        const bitmap = await createImageBitmap(await response.blob());
        try {
          abort.signal.throwIfAborted();
          const canvas = document.createElement('canvas');
          canvas.width = selectedFrame.width;
          canvas.height = selectedFrame.height;
          const context = canvas.getContext('2d')!;
          context.drawImage(bitmap, 0, 0, canvas.width, canvas.height);
          setOverlay({
            frame: {
              ...selectedFrame,
              pixels: context.getImageData(0, 0, canvas.width, canvas.height).data,
            },
          });
        } finally {
          bitmap.close();
        }
        return;
      }
      if (!playbackInput) throw new Error('No frame image available');
      const { decodeVideoFrames } = await import('../lib/sfm-video');
      let loaded = false;
      for await (const decoded of decodeVideoFrames(
        typeof playbackInput === 'string'
          ? new URL(playbackInput, location.href).href
          : playbackInput,
        abort.signal,
        selectedFrame.time,
        Math.max(selectedFrame.width, selectedFrame.height) > 640 ? 1280 : 640,
      )) {
        if (!abort.signal.aborted) {
          loaded = true;
          setOverlay({ frame: { ...selectedFrame, ...decoded } });
        }
      }
      if (!loaded) throw new Error('No decoded frame at this time');
    };
    load().catch(() => {
      if (!abort.signal.aborted) {
        setFrameLoadError(true);
        setReplaying(false);
      }
    });
    return () => abort.abort();
  }, [selectedFrame, complete, busy, prepared, playbackInput]);
  useEffect(() => {
    if (!followCamera || busy || !selectedView) return;
    setCameraView((previous) => ({ id: selectedView.id, nonce: (previous?.nonce ?? 0) + 1 }));
    // A model snapshot must not re-aim the view during a run.
  }, [followCamera, selectedView?.id, busy, mvSelected, selectionNonce]);
  useEffect(
    () => () => {
      abortRef.current?.abort();
      terminateWorker();
    },
    [],
  );

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
    setAnalysisRange(undefined);
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
    // A note belongs to one run: a new run or input starts with an empty note, or the record that
    // is sent at once (consent on) would carry the last run's note.
    setNote('');
    noteRef.current = '';
    clearMulti();
  }
  /** Forget the rows and the result of a multi-video run. */
  function clearMulti() {
    clearDense();
    setMvRows([]);
    setMvGraph(undefined);
    setMvBoard(undefined);
    setMvActivity(undefined);
    setMvThumbnails([]);
    setMvActiveVideo(undefined);
    setMvSelected(0);
    setMvProgress(0);
    mvModels.current.clear();
    mvReplaySources.current.clear();
    setSelectedVideo(undefined);
    setSelectedIndex(0);
    setCameraView(undefined);
  }
  /** Forget the result of a multi-video run and keep the rows. */
  function clearMultiResult() {
    clearDense();
    setMvGraph(undefined);
    setMvBoard(undefined);
    setMvActivity(undefined);
    setMvThumbnails([]);
    setMvActiveVideo(undefined);
    setMvSelected(0);
    setMvProgress(0);
    mvModels.current.clear();
    mvReplaySources.current.clear();
    setSelectedVideo(undefined);
    setSelectedIndex(0);
    setCameraView(undefined);
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
  // The collector takes a record's files only with the record that declared them, so files for a
  // record already sent go with a new copy of it.
  const resendWithFiles =
    sendState.status === 'sent' && !sendState.files && sendFiles && filesAvailable;
  useEffect(() => {
    if (!autoSend || autoSend !== record) return;
    setAutoSend(undefined);
    void sendRecord(autoSend);
  });
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
        ko ? '분석할 프레임을 준비했습니다.' : sampled.length + ' analysis frames are ready.',
      );
      setInputOpen(false);
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
      // A sample that stores its source video: the run decodes it at the analysis edge when this
      // browser can, and nothing else is downloaded now. Only when it cannot are the stored
      // frames fetched, one file per frame.
      const url = sourceUrl(entry);
      const decodable = url
        ? await (
            await import('../lib/sfm-video')
          ).sourceDecodable(url, new AbortController().signal)
        : false;
      videoSourceRef.current = url ? { url, decodable } : undefined;
      const sampled: Frame[] = [],
        thumbs: string[] = [];
      const canvas = document.createElement('canvas'),
        ctx = canvas.getContext('2d', { willReadFrequently: true })!;
      for (let i = 0; !decodable && i < times.length; i++) {
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
      setPrepared(sampled);
      setFrames(thumbs);
      setSource({
        name: entry.video?.name ?? entry.name,
        width: decodable ? entry.source!.width : sampled[0]!.width,
        height: decodable ? entry.source!.height : sampled[0]!.height,
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
          ? decodable
            ? '원본 영상을 실행 때 바로 읽습니다. 미리 받은 프레임은 없습니다.' + tag
            : '영상에서 뽑은 ' + sampled.length + '개 프레임입니다.' + tag
          : decodable
            ? 'The run reads the source video directly; no frames were downloaded.' + tag
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
              ')과 데이터셋에 들어 있는 SIFT 특징점·매칭을 불러왔습니다. GPU에서는 복원 단계만 실행하고, 결과를 정답 카메라와 비교합니다.'
            : sampled.length +
              ' photographs at ' +
              size +
              " with the dataset's own SIFT keypoints and matches are ready. The GPU runs the mapper alone; the result is compared with the reference cameras."
          : ko
            ? sampled.length +
              '장의 사진(' +
              size +
              ')을 보정된 초점거리·왜곡과 함께 불러왔습니다. 결과를 정답 카메라와 비교합니다.'
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
    // Picking a card closes the drawer at once; the status line shows the load. The drawer does
    // not close again when the load finishes, so a drawer opened meanwhile stays open.
    setInputOpen(false);
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
        ? '저장된 결과를 열었습니다. 다시 계산하지 않고 그대로 보여 줍니다.'
        : 'A saved result is open. The viewer shows it without a run.',
    );
  }
  /** Open result files from the visitor's disk: `.json` and `.ply`, several at once. */
  async function openResultFiles(list: FileList | null) {
    const picked = Array.from(list ?? []);
    if (!picked.length || busy) return;
    setInputOpen(false);
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
      // The result is open: the drawer closes. A result that is not published leaves it open.
      setInputOpen(false);
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
    benchmarkStopRef.current = false;
    setBenchmarkRunning(true);
    try {
      const sample = await loadVideoSample(PYRAMID_SAMPLE);
      if (sample.length !== 8) return;
      for (let pass = 0; pass < 2 && !benchmarkStopRef.current; pass++) {
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
          ? `영상 ${made.length}개를 불러왔습니다. 설정에서 영상별 분석 구간을 고를 수 있습니다.`
          : `${made.length} videos are ready. Choose each video's analysis range in Settings.`,
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
        ? `영상 ${rows.length}개를 불러왔습니다. 실행하면 한 장면으로 복원합니다.`
        : `${rows.length} videos are ready. Run reconstructs them together.`,
    );
  }
  /** Show the dense cloud, mesh and summary of a component (Amendment 9, item 5). */
  function showComponentDense(index: number) {
    const entry = mvDense.current.get(index);
    denseRef.current = { depth: [], points: entry?.points ?? [] };
    setDenseCloud(entry?.cloud);
    setDenseFusionCloud(entry?.fusionCloud);
    setDenseMesh(entry?.mesh);
    setDenseSummary(entry?.summary);
  }
  function selectComponent(index: number) {
    const model = mvModels.current.get(index);
    if (!model) return;
    showComponentDense(index);
    setMvSelected(index);
    setSelectedVideo(undefined);
    setSelectedIndex(0);
    setReplaying(false);
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
    // A note belongs to one run: a new run or input starts with an empty note, or the record that
    // is sent at once (consent on) would carry the last run's note.
    setNote('');
    noteRef.current = '';
    const abort = new AbortController();
    abortRef.current = abort;
    const signal = abort.signal;
    const id = ++jobRef.current;
    runFocal.current = rows[0]!.focal;
    setBusy(true);
    setRunning(true);
    setPreparation({ phase: 'viewer', startedAt: performance.now(), multi: true });
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
    setStatus(ko ? '3D 화면 준비 중…' : 'Preparing the 3D viewer…');
    setMvProgress(0);
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
    let multiStage: StageEvent['stage'] | undefined;
    lastEventAt.current = performance.now();
    setActivity(undefined);
    const decodeInfo: { path: 'worker' | 'main-thread' } = { path: 'main-thread' };
    const watch = watchMainThread(() => stage);
    // The dense MVS of each component (`docs/multi-video.md`, Amendment 9): the run runs it after
    // the whole reconstruction, and the page shows the points of the selected component.
    const denseRun = denseEnabledRef.current || denseOverride();
    setDenseAsked(denseRun);
    const denseMeshRun = denseRun && (meshEnabledRef.current || denseMeshOverride());
    const denseRunId = denseRunRef.current;
    const denseBlocks = new Map<number, RunRecordDense>();
    let denseChunks: unknown;
    let denseShown = false;
    let denseSerial = 0;
    let watchdogMs = 240000;
    const denseEntry = (component: number) => {
      let entry = mvDense.current.get(component);
      if (!entry) mvDense.current.set(component, (entry = { points: [], count: 0 }));
      return entry;
    };
    // The stage list (Amendment 8). The ticker rebuilds the status line twice a second while the
    // run lasts, so the line moves even when no message arrives.
    let board = newStageBoard(performance.now());
    let framesNote = '';
    let expectedFrames = 0;
    let keyframesSoFar = 0;
    const showBoard = () => {
      const now = performance.now();
      setMvBoard(board);
      if (board.rows.some((row) => row.state !== 'pending'))
        setStatus(stageStatusLine(board, now, ko));
      const done = board.rows.filter((r) => r.state === 'done').length;
      const running = board.rows.find((r) => r.state === 'running');
      const fraction = running && running.total ? Math.min(1, running.done / running.total) : 0;
      setMvProgress(Math.round((100 * (done + fraction)) / board.rows.length));
    };
    const stageEvent = (event: StageEvent) => {
      board = advanceStages(board, event, performance.now());
      showBoard();
    };
    const ticker = window.setInterval(() => {
      board = tickStages(board, performance.now());
      showBoard();
    }, 500);
    const models = new Map<number, Scene>();
    // For a failure record: the last partial map of each component, and what the page saw of
    // each video. The final map of a component replaces its partial map.
    const partialModels = new Map<number, Scene>();
    const videoFacts = new Map<string, VideoFacts>();
    let fitted = false;
    const meanErrors = new Map<number, number | null>();
    let graph: any;
    let fed = 0;
    let firstFrame: Frame | undefined;
    const totalSeconds = rows.reduce((n, r) => n + (r.durationSeconds ?? 0), 0);
    expectedFrames = Math.round(
      rows.reduce(
        (n, row) => n + (row.range ? row.range.end - row.range.start : (row.durationSeconds ?? 0)),
        0,
      ) * sampleFps,
    );
    const fail = (error: Error) => {
      failure = error;
      pending?.reject(error);
      abort.abort();
    };
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
    const assembler = new ModelAssembler();
    const onWorkerMessage = ({ data: received }: MessageEvent) => {
      if (received.id !== id || signal.aborted) return;
      armTimeout();
      lastEventAt.current = performance.now();
      // A final map arrives in parts (`lib/model-chunks.ts`); a part waits for its `model`.
      const message = assembler.accept(received);
      if (!message) return;
      if (message.type === 'preparing' && isGpuPreparationPhase(message.phase)) {
        stage = 'preparing ' + message.phase;
        setPreparation((previous) => previous && { ...previous, phase: message.phase });
      }
      if (message.type === 'progress') {
        // A stage name or a kernel: both are signs of life for the stall warning.
        stage = String(message.detail);
        board = touchStages(board, performance.now());
        // Only the reconstruction has cameras to pulse; it grows next to the newest one.
        noteActivity(multiStage === 'reconstruction' ? 'latest' : 'none');
      }
      if (message.type === 'multiStage') {
        const { type: _type, id: _id, ...event } = message;
        multiStage = (event as StageEvent).stage;
        stageEvent(event as StageEvent);
      }
      if (message.type === 'graphActivity') {
        setMvActivity(message as MultiVideoActivity);
        board = touchStages(board, performance.now());
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
      if (message.type === 'model' && message.partial && message.reconstruction) {
        // A map that is still growing: the viewer shows it, and the camera fits it once.
        const growing = unpackScene(message.reconstruction);
        partialModels.set(message.component, growing);
        setReconstruction(growing);
        if (!fitted) {
          fitted = true;
          setFitNonce((nonce) => nonce + 1);
        }
      }
      if (message.type === 'videoComplete') {
        const fact = videoFacts.get(message.videoId);
        if (fact) videoFacts.set(message.videoId, { ...fact, keyframes: message.keyframes });
        setMvRows((rows) =>
          rows.map((row) =>
            row.videoId === message.videoId ? { ...row, keyframes: message.keyframes } : row,
          ),
        );
        if (Array.isArray(message.keptFrameIds)) {
          const kept = new Set(message.keptFrameIds);
          setMvThumbnails((frames) =>
            frames.filter((frame) => frame.videoId !== message.videoId || kept.has(frame.frameId)),
          );
        }
      }
      if (message.type === 'model' && !message.partial && message.reconstruction) {
        models.set(message.component, message.reconstruction);
        meanErrors.set(message.component, message.meanReprojectionErrorPx ?? null);
        mvModels.current.set(message.component, message.reconstruction);
        if (message.component === 0) {
          setReconstruction(message.reconstruction);
          if (!fitted) setFitNonce((nonce) => nonce + 1);
          fitted = true;
        }
      }
      if (message.type === 'ready' && message.adapter) adapterRef.current = message.adapter;
      if (message.type === 'multiReady' && message.adapter) adapterRef.current = message.adapter;
      if (typeof message.type === 'string' && message.type.startsWith('dense')) {
        const component: number = message.component ?? 0;
        const entry = denseEntry(component);
        const shown = component === mvSelectedRef.current;
        // The run id of a cloud changes with the component, so that the viewer rebuilds its subset.
        const runOf = (extra: number) => denseRunId * 10000 + component * 4 + extra;
        if (message.type === 'denseActivity') {
          const activity = { ...message, receivedAt: performance.now() } as DenseActivity;
          setDenseActivity(activity);
          stage = 'dense ' + activity.phase;
          setStatus('Dense MVS: ' + denseActivityText(activity, ko));
          noteActivity(
            shown ? (activity.viewId === undefined ? 'all' : 'latest') : 'none',
            shown ? activity.viewId : undefined,
          );
        } else if (message.type === 'denseFusionPoints') {
          entry.fusionCloud = appendFusionPreview(entry.fusionCloud, runOf(3), message);
          if (shown) {
            setDenseFusionCloud(entry.fusionCloud);
            if (!denseShown) {
              denseShown = true;
              setShowDense(true);
            }
          }
        } else if (message.type === 'densePoints') {
          entry.fusionCloud = undefined;
          if (shown) setDenseFusionCloud(undefined);
          entry.points.push(message);
          entry.count += message.positions.length / 3;
          entry.cloud = { run: runOf(0), chunks: entry.points, count: entry.count };
          if (shown) {
            denseRef.current = { depth: [], points: entry.points };
            setDenseCloud(entry.cloud);
            if (!denseShown) {
              denseShown = true;
              setShowDense(true);
            }
          }
        } else if (message.type === 'denseClean') {
          // 3.12: the cleaned cloud replaces the raw chunks of the component.
          entry.points = [message];
          entry.count = message.positions.length / 3;
          entry.cloud = { run: runOf(1), chunks: entry.points, count: entry.count };
          if (shown) {
            denseRef.current = { depth: [], points: entry.points };
            setDenseCloud(entry.cloud);
          }
        } else if (message.type === 'denseMesh') {
          entry.mesh = {
            run: runOf(2),
            positions: message.positions,
            colors: message.colors,
            indices: message.indices,
          };
          if (shown) setDenseMesh(entry.mesh);
        } else if (message.type === 'denseProgress') {
          stage =
            'dense ' +
            (component + 1) +
            ' ' +
            message.stage +
            ' ' +
            message.done +
            '/' +
            message.total;
          const words = DENSE_STAGES[message.stage] ?? [message.stage, message.stage];
          const total = graph?.components.filter((c: any) => c.reconstructed).length ?? 0;
          const points = [...mvDense.current.values()].reduce((n, e) => n + e.count, 0);
          setStatus(
            (ko
              ? `Dense MVS 묶음 ${component + 1}${total ? '/' + total : ''}: `
              : `Dense MVS component ${component + 1}${total ? ' of ' + total : ''}: `) +
              words[ko ? 0 : 1] +
              ' ' +
              message.done +
              '/' +
              message.total +
              (ko ? ' · 점 ' + points + '개' : ' · ' + points + ' points'),
          );
        } else if (message.type === 'denseResult') {
          entry.fusionCloud = undefined;
          if (shown) setDenseFusionCloud(undefined);
          setDenseActivity(undefined);
          const block = denseRecord(message);
          if (block) {
            denseBlocks.set(component, block);
            entry.summary = block;
            if (shown) setDenseSummary(block);
          }
          denseChunks = message.chunks ?? denseChunks;
        }
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
      const fact = videoFacts.get(videoId);
      videoFacts.set(videoId, {
        frames: (fact?.frames ?? 0) + 1,
        width: fact?.width ?? frame.width,
        height: fact?.height ?? frame.height,
      });
      if (done.skipped) setSkipped((n) => n + 1);
      else setProcessed((n) => n + 1);
      if (done.keyframe)
        setMvRows((list) =>
          list.map((r) => (r.key === row.key ? { ...r, keyframes: (r.keyframes ?? 0) + 1 } : r)),
        );
      if (done.keyframe && !row.thumbnail && frame.image)
        update(row.key, { thumbnail: frame.image });
      if (done.keyframe && frame.image)
        setMvThumbnails((previous) =>
          appendKeyframeThumbnail(previous, {
            videoId,
            frameId: frame.id,
            time: frame.time,
            image: frame.image!,
            features: done.features ?? 0,
          }),
        );
      if (!done.skipped) setFrameMs(done.elapsedMs);
      if (done.keyframe) keyframesSoFar++;
      stageEvent({
        stage: 'frames',
        done: fed,
        total: Math.max(expectedFrames, fed),
        detail:
          (framesNote ? framesNote + ' · ' : '') +
          (ko ? `키프레임 ${keyframesSoFar}개` : `${keyframesSoFar} keyframes`),
      });
      await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
    };
    try {
      await waitForViewer(signal);
      signal.throwIfAborted();
      setPreparation((previous) => previous && { ...previous, phase: 'worker' });
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
      setStatus(ko ? 'GPU 준비 중…' : 'Preparing the GPU…');
      setSampleFps(0);
      const ready = await request('multiReady', {
        type: 'multiStart',
        query,
        analysisEdge: runEdge,
        firstOctave: quality.firstOctave,
        clahe: quality.clahe,
        budget: budgetOverride(),
        fp64: fp64Override(),
        dense: denseRun,
        denseClean: densePreset.clean,
        denseFusion: densePreset.fusion,
        denseMesh: denseMeshRun,
      });
      setPreparation(undefined);
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
              row.range,
            )) {
              if (held) yield held;
              held = { id: n++, ...frame };
            }
            if (held) yield { ...held, last: true };
          })();
        framesNote = `${row.label} (${index + 1} / ${rows.length})`;
        stageEvent({
          stage: 'frames',
          done: fed,
          total: Math.max(expectedFrames, fed),
          detail: framesNote,
        });
        const url = row.entry ? sourceUrl(row.entry) : undefined;
        const decodable = url
          ? await (await import('../lib/sfm-video')).sourceDecodable(url, signal)
          : false;
        if (row.entry) sourceOf.set(row.key, videoSourceOf(row.entry, decodable));
        if (row.file) {
          videoId ??= await fileVideoId(row.file);
          source = decoded(row.file);
        } else if (decodable && row.entry) {
          videoId = catalogVideoId(row.entry)!;
          source = decoded(new URL(url!, location.href).href);
        } else {
          const entry = row.entry!;
          const fetched = await fetchSampleFrames(entry, signal, (done, total) =>
            stageEvent({
              stage: 'frames',
              done: fed,
              total: Math.max(expectedFrames, fed),
              detail: `${framesNote} · ${ko ? '샘플 프레임 내려받기' : 'fetching frames'} ${done} / ${total}`,
            }),
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
        if (row.file) mvReplaySources.current.set(videoId, row.file);
        else if (decodable && url)
          mvReplaySources.current.set(videoId, new URL(url, location.href).href);
        const item = { ...row, videoId };
        used.push(item);
        update(row.key, { videoId });
        setMvActiveVideo(videoId);
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
                ((index +
                  videoRangeProgress(frame.time, row.durationSeconds ?? frame.time, row.range)) *
                  95) /
                  rows.length,
              ),
            ),
          );
        }
        await request('videoComplete', { type: 'multiEndVideo', videoId });
      }
      stage = 'finish';
      setMvActiveVideo(undefined);
      // The frames are done: the worker now names each stage as it reaches it.
      stageEvent({ stage: 'candidates', done: 0, total: 0 });
      const result = await request('result', { type: 'multiFinish' });
      board = finishStages(board, performance.now());
      window.clearInterval(ticker);
      setMvBoard(board);
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
      setStages([
        ...Object.entries(result.stages).map(([operation, st]: [string, any]) => ({
          operation,
          elapsedMs: st.elapsedMs,
        })),
        // Where the reconstruction went: the calls of the mapper by kind.
        ...Object.entries(result.operations ?? {}).map(([operation, st]: [string, any]) => ({
          operation: 'reconstruction / ' + operation,
          calls: st.calls,
          submissions: st.submissions,
          elapsedMs: st.elapsedMs,
        })),
      ]);
      setTotalMs(result.totalMs);
      const best = models.get(0);
      if (best) setReconstruction(best);
      const registered = [...models.values()].reduce((n, m) => n + m.views.length, 0);
      const sparseStatus = ko
        ? `복원 끝: 묶음 ${models.size}개, 카메라 ${registered}개, 이어지지 않은 영상 ${summary.notConnected.length}개.`
        : `Reconstruction completed: ${models.size} components, ${registered} registered views, ${summary.notConnected.length} videos not connected.`;
      if (denseRun) {
        // The sparse maps are on the page while the worker runs the dense MVS on each component
        // (`docs/multi-video.md`, Amendment 9). The Cancel control stops it and keeps the maps.
        setStatus(
          sparseStatus +
            (ko ? ' 이어서 Dense MVS를 실행합니다…' : ' Starting the dense MVS on each component…'),
        );
        stage = 'dense';
        watchdogMs = 600000;
        denseCancelRef.current = () => worker.postMessage({ type: 'cancelDense', id });
        setDenseRunning(true);
        try {
          const done = await new Promise<any>((resolve, reject) => {
            signal.throwIfAborted();
            pending = { type: 'denseDone', resolve, reject };
            armTimeout();
          });
          if (Array.isArray(done.gpuKernels)) {
            // These are two device totals, not successive snapshots of one device.
            result.gpuKernels = [...(result.gpuKernels ?? []), ...done.gpuKernels];
            result.gpuDropped = (result.gpuDropped ?? 0) + (done.gpuDropped ?? 0);
          }
        } finally {
          denseCancelRef.current = undefined;
          setDenseRunning(false);
        }
      }
      if (denseRun && !denseBlocks.size) {
        // No component reached the dense stage: the record says why (key -1 is no component).
        const none = denseRecord({ skipped: DENSE_NO_COMPONENT });
        if (none) {
          denseBlocks.set(-1, none);
          setDenseSummary(none);
        }
      }
      const mainThread = watch.stop();
      setOverlay(undefined);
      setComplete(true);
      const denseTotal = [...denseBlocks.values()].reduce((n, d) => n + d.points, 0);
      const denseFailed = [...denseBlocks.values()].find((d) => d.status === 'failed');
      const denseSkipped = [...denseBlocks.values()].find((d) => d.status === 'skipped');
      setStatus(
        !denseRun
          ? sparseStatus
          : denseFailed
            ? (denseReason(denseFailed, true, ko) ?? '')
            : denseSkipped && !denseTotal
              ? sparseStatus + ' ' + denseReason(denseSkipped, true, ko)
              : sparseStatus +
                (ko
                  ? ` Dense MVS 점 ${denseTotal}개.`
                  : ` Dense MVS: ${denseTotal} points in ${denseBlocks.size} components.`),
      );
      if (/[?&]canonical=/.test(search))
        (window as any).__typecvMultiVideo = {
          canonical: result.canonical,
          keyframes: result.keyframes,
          components: result.components,
          summary,
          timing: result.stages,
          totalMs: result.totalMs,
          dense: Object.fromEntries(denseBlocks),
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
        dense: denseRun ? { blocks: denseBlocks, chunks: denseChunks } : undefined,
      });
      setRecord(built);
      if (consentRef.current) setAutoSend(built);
      return true;
    } catch (error) {
      window.clearInterval(ticker);
      const mainThread = watch.stop();
      const cancelled = signal.aborted && !failure;
      setStatus(
        cancelled
          ? ko
            ? '분석을 멈췄습니다.'
            : 'Analysis cancelled.'
          : (ko ? '복원 실패: ' : 'Reconstruction failed: ') + String(failure ?? error),
      );
      terminateWorker();
      if (!cancelled) {
        const message = String(failure ?? error);
        const now = performance.now();
        const timed = tickStages(board, now);
        const scenes = new Map<number, Scene>([...partialModels, ...models]);
        const used = mvRowsRef.current.filter((r) => !r.duplicate);
        const built = buildMultiRunRecord({
          result: undefined,
          partial: {
            stages: stageTimings(timed),
            totalMs: now - timed.startedAt,
            videos: partialVideos({ rows: used, facts: videoFacts, scenes }),
            scenes,
            gpuKernels: failed?.gpuKernels,
            gpuDropped: failed?.gpuDropped,
          },
          graph,
          models,
          meanErrors,
          adapter: adapterRef.current,
          commit: commitRef.current,
          rows: used,
          query,
          maxFeatures: 0,
          firstOctave: quality.firstOctave,
          frames: fed,
          first: firstFrame,
          durationSeconds: totalSeconds,
          setName: loadedSampleId ?? null,
          mainThread,
          note: noteRef.current,
          dense: denseBlocks.size ? { blocks: denseBlocks, chunks: denseChunks } : undefined,
          failure: failed?.failure ?? {
            message: message.slice(0, 2000),
            operation: stage.slice(0, 200),
            kernel: '',
            reason: '',
            framesProcessed: fed,
          },
        });
        setRecord(built);
        if (consentRef.current) setAutoSend(built);
      }
      return false;
    } finally {
      window.clearInterval(ticker);
      window.clearTimeout(timeout);
      signal.removeEventListener('abort', onAbort);
      setBusy(false);
      setRunning(false);
      setFinalizing(false);
      setPreparation(undefined);
      setMvActivity(undefined);
      setMvThumbnails([]);
      setMvActiveVideo(undefined);
      abortRef.current = undefined;
    }
  }

  async function runSfM(framesOverride?: Frame[], benchmark?: Benchmark): Promise<boolean> {
    const inputFrames = framesOverride ?? prepared;
    const runRange = framesOverride ? undefined : analysisRange;
    const storedSource = framesOverride ? undefined : videoSourceRef.current;
    if (busy || (!inputFrames.length && !videoURL && !storedSource?.decodable)) return false;
    if (!inputFrames.length && !storedSource) inputKindRef.current = 'video';
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
    // A note belongs to one run: a new run or input starts with an empty note, or the record that
    // is sent at once (consent on) would carry the last run's note.
    setNote('');
    noteRef.current = '';
    const abort = new AbortController();
    abortRef.current = abort;
    const signal = abort.signal;
    const id = ++jobRef.current;
    runFocal.current = focal;
    let worker: Worker;
    setBusy(true);
    setRunning(true);
    setFinalizing(false);
    setPreparation({ phase: 'viewer', startedAt: performance.now(), multi: false });
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
    setStatus(ko ? '3D 화면 준비 중…' : 'Preparing the 3D viewer…');
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
    setDenseAsked(denseRun);
    let denseFinal: any;
    let denseCounts = { depth: 0, points: 0 };
    const previewViews = new Map<number, DensePreviewView>();
    clearDense();
    const denseRunId = denseRunRef.current;
    let denseShown = false;
    // A reference of the MVS can take minutes on a software GPU, so the watchdog waits longer.
    let watchdogMs = 120000;
    // The lab's stage for the main-thread watch: the worker's last progress detail.
    let stage = 'start';
    lastEventAt.current = performance.now();
    setActivity(undefined);
    frameFeatures.current.clear();
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
      lastEventAt.current = performance.now();
      if (message.type === 'preparing' && isGpuPreparationPhase(message.phase)) {
        stage = 'preparing ' + message.phase;
        setPreparation((previous) => previous && { ...previous, phase: message.phase });
      }
      if (message.type === 'progress') {
        noteActivity(pulseTarget(String(message.detail)));
        const labels: Record<string, [string, string]> = {
          features: ['특징점 찾는 중…', 'Detecting features…'],
          matching: ['앞 프레임과 같은 점 찾는 중…', 'Matching previous frames…'],
          verifyPair: ['두 프레임이 맞는지 확인하는 중…', 'Verifying image-pair geometry…'],
          pnp: ['카메라 위치 계산 중…', 'Estimating camera position…'],
          triangulation: ['3D 점 계산 중…', 'Triangulating 3D points…'],
          observations: ['오차 확인 중…', 'Checking reprojection errors…'],
          bundle: ['결과 다듬는 중…', 'Refining the map…'],
          final: ['마지막으로 다듬는 중…', 'Final map refinement…'],
        };
        stage = String(message.detail);
        const label = labels[String(message.detail).split(/[ #·]/)[0]];
        if (/^(pnp|verifyPair)/.test(message.detail)) setFrameStep(2);
        if (/^(triangulation|observations|bundle)/.test(message.detail)) setFrameStep(3);
        const base = label ? label[ko ? 0 : 1] : ko ? '영상 분석 중…' : 'Analyzing video…';
        // The first job compiles each GPU kernel on first use; show that count while it is cold.
        setStatus(
          message.cold && message.kernels
            ? base +
                (ko
                  ? ' GPU 커널 ' + message.kernels + '개 준비함 (처음 한 번만)'
                  : ' ' + message.kernels + ' GPU kernels compiled (first run)')
            : base,
        );
      }
      if (message.type === 'features') {
        const store = frameFeatures.current;
        store.set(message.frameId, Float32Array.from(message.points));
        if (store.size > FEATURE_FRAMES) store.delete(store.keys().next().value!);
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
      // Amendment 5: a depth preview per reference and stage, and after iterations 1 and 3 of its
      // photometric pass. The viewer draws the latest preview of each view; the status counts
      // the filtered depth maps, the geometric previews.
      if (message.type === 'denseDepth') {
        previewViews.set(message.viewId, message as DensePreviewView);
        setDensePreview({ run: denseRunId, views: new Map(previewViews), done: false });
        if (message.stage === 'geometric')
          denseCounts = { ...denseCounts, depth: denseCounts.depth + 1 };
      }
      if (message.type === 'denseActivity') {
        const activity = { ...message, receivedAt: performance.now() } as DenseActivity;
        setDenseActivity(activity);
        stage = 'dense ' + activity.phase;
        setStatus('Dense MVS: ' + denseActivityText(activity, ko));
        noteActivity(activity.viewId === undefined ? 'all' : 'latest', activity.viewId);
      }
      if (message.type === 'denseFusionPoints') {
        setDenseFusionCloud((previous) =>
          appendFusionPreview(previous, denseRunId + 0.25, message),
        );
        if (!denseShown) {
          denseShown = true;
          setShowDense(true);
        }
      }
      if (message.type === 'densePoints') {
        setDenseFusionCloud(undefined);
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
        // The fusion has posted every reference: the preview goes (Amendment 5, rule 4).
        if (message.stage === 'fusion' && message.done === message.total)
          setDensePreview((preview) => preview && { ...preview, done: true });
        const words = DENSE_STAGES[message.stage] ?? [message.stage, message.stage];
        setStatus(
          (ko ? 'Dense MVS: ' : 'Dense MVS: ') +
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
        setDenseFusionCloud(undefined);
        setDenseActivity(undefined);
        denseFinal = message;
        setDensePreview((preview) => preview && { ...preview, done: true });
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
              ? '복원을 시작할 만큼 카메라가 움직이기를 기다리는 중입니다.'
              : 'Waiting for a stable initial pair.'
            : ko
              ? '카메라와 3D 점을 업데이트했습니다.'
              : 'Camera and map updated.',
      );
      await new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
    };
    try {
      await waitForViewer(signal);
      signal.throwIfAborted();
      setPreparation((previous) => previous && { ...previous, phase: 'worker' });
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
      setStatus(ko ? 'GPU 준비 중…' : 'Preparing the GPU…');
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
        // The standard benchmark keeps the fixed start, so that its pass times stay comparable.
        focalSearch: focalAuto && !benchmark,
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
        // The combination selected in the run settings, shared with multi-video runs.
        denseClean: densePreset.clean,
        denseFusion: densePreset.fusion,
        // `?denseMesh=1`: mesh the final cloud (`docs/dense-mvs.md` 3.13).
        denseMesh: denseRun && (meshEnabledRef.current || denseMeshOverride()),
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
      setPreparation(undefined);
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
          runRange,
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
      setFinalizing(true);
      const result = await request('result', { type: 'finish' });
      let dense: any;
      if (denseRun) {
        // The sparse result is on the page while the MVS runs in the worker (5.3).
        setReconstruction(result.reconstruction);
        setStatus(
          ko
            ? '복원이 끝났습니다. 이어서 Dense MVS를 실행합니다…'
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
            ? '복원이 끝났습니다.'
            : 'GPU SfM reconstruction completed.'
          : dense.failure
            ? (ko ? 'Dense MVS 실패: ' : 'Dense MVS failed: ') + dense.failure.message
            : dense.skipped
              ? (ko ? 'Dense MVS 건너뜀: ' : 'Dense MVS skipped: ') + dense.skipped
              : (dense.cancelled
                  ? ko
                    ? 'Dense MVS를 멈췄습니다: 점 '
                    : 'Dense MVS cancelled: '
                  : ko
                    ? 'Dense MVS 끝: 점 '
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
        range: runRange,
        videoSampleFps: sampledVideo ? sampleFps : undefined,
        videoDecode: sampledVideo ? decodeInfo.path : undefined,
        videoSource,
        benchmark,
        reference: referenceRecord(sampleRef.current, aligned),
        note: noteRef.current,
      });
      setRecord(built);
      // Consent given before the run sends the record as soon as it exists.
      if (consentRef.current) setAutoSend(built);
      return true;
    } catch (error) {
      const mainThread = watch.stop();
      const cancelled = signal.aborted && !failure;
      setStatus(
        cancelled
          ? ko
            ? '분석을 멈췄습니다.'
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
          range: runRange,
          videoSampleFps: inputFrames.length === 0 || sourceRun ? pageSampleFps() : undefined,
          videoDecode: inputFrames.length === 0 || sourceRun ? decodeInfo.path : undefined,
          videoSource,
          benchmark,
          reference: null,
          note: noteRef.current,
        });
        setRecord(built);
        if (consentRef.current) setAutoSend(built);
      }
      return false;
    } finally {
      window.clearTimeout(timeout);
      signal.removeEventListener('abort', onAbort);
      videoRef.current?.pause();
      setBusy(false);
      setRunning(false);
      setFinalizing(false);
      setPreparation(undefined);
      abortRef.current = undefined;
    }
  }

  /**
   * The name of a dense or mesh export. A multi-video run with two or more components has no
   * shared frame, so the export is the selected component's, and its name says which
   * (`docs/multi-video.md`, Amendment 9, item 7).
   */
  function plyName(kind: 'dense' | 'mesh'): string {
    const components = mvGraph?.components.filter((c) => c.reconstructed).length ?? 0;
    return components > 1 ? `typecv-${kind}-component-${mvSelected + 1}.ply` : `typecv-${kind}.ply`;
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
    anchor.download = plyName('dense');
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
    anchor.download = plyName('mesh');
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
            {entry.files.dense && <Tag>{ko ? 'Dense' : 'dense'}</Tag>}
            {entry.files.mesh && <Tag>{ko ? '메시' : 'mesh'}</Tag>}
            {entry.source && <Tag color="purple">{entry.source}</Tag>}
          </div>
          <div className="sample-card-actions">
            <Button
              size="small"
              className={
                active
                  ? 'sample-load-button is-loading'
                  : loadedSampleId === entry.id
                    ? 'sample-load-button is-loaded'
                    : 'sample-load-button'
              }
              style={active ? ({ '--load-progress': percent + '%' } as CSSProperties) : undefined}
              disabled={(busy && !active) || missing}
              aria-busy={active || undefined}
              aria-label={'View ' + entry.name}
              onClick={() => void viewResult(entry)}
            >
              {active
                ? (ko ? '받는 중 ' : 'Downloading ') + Math.round(percent) + ' %'
                : missing
                  ? ko
                    ? '준비 중'
                    : 'Not published yet'
                  : loadedSampleId === entry.id
                    ? ko
                      ? '보는 중 ✓'
                      : 'Open ✓'
                    : ko
                      ? '보기'
                      : 'View'}
            </Button>
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
        ? Math.min(99, Math.round(100 * videoRangeProgress(analysisTime, duration, analysisRange)))
        : 0;
  const denseActive = busy && (denseRunning || !!denseActivity);
  const runProgress =
    complete && !busy
      ? 100
      : preparation
        ? undefined
        : denseActive
          ? denseActivity && denseActivityProgress(denseActivity)?.percent
          : finalizing
            ? undefined
            : progress;
  const progressLabel =
    complete && !busy
      ? ko
        ? '완료'
        : 'Completed'
      : preparation
        ? ko
          ? 'GPU 준비'
          : 'GPU preparation'
        : denseActive
          ? ko
            ? 'Dense 현재 작업'
            : 'Dense current operation'
          : finalizing
            ? ko
              ? '복원 마무리'
              : 'Final refinement'
            : ko
              ? '영상·복원 단계'
              : 'Video and reconstruction';
  const hasInput = !!(loadedSampleId || file || mvRows.length || reconstruction);
  // A chosen file or video set closes the drawer. A catalog card closes it when it is picked
  // (loadCatalogSample); a frame JSON closes it once its frames load (loadPreparedFrames).
  useEffect(() => {
    if ((file || mvRows.length) && !file?.name.toLowerCase().endsWith('.json')) setInputOpen(false);
  }, [file, mvRows.length]);
  // The input in the lab, shown as one chip above the workspace; the drawer holds the catalog.
  const chosenEntry = loadedSampleId
    ? catalog.find((entry) => entry.id === loadedSampleId)
    : undefined;
  const chipImage = (() => {
    if (!chosenEntry) return undefined;
    if (isVideoSet(chosenEntry)) {
      const first = catalog.find((entry) => entry.id === chosenEntry.members[0]);
      return first && !isResult(first) && !isVideoSet(first)
        ? first.dir + first.thumbnail
        : undefined;
    }
    if (isResult(chosenEntry))
      return chosenEntry.thumbnail ? chosenEntry.dir + chosenEntry.thumbnail : undefined;
    return chosenEntry.dir + chosenEntry.thumbnail;
  })();
  const chipName = chosenEntry
    ? chosenEntry.name
    : mvRows.length
      ? (ko ? '내 영상 ' : 'My videos: ') + mvRows.length + (ko ? '개' : '')
      : file
        ? file.name
        : reconstruction
          ? ko
            ? '연 결과 파일'
            : 'Opened result files'
          : undefined;
  const chipMeta = chosenEntry
    ? isVideoSet(chosenEntry)
      ? (ko ? '영상 세트 · ' : 'Video set · ') + chosenEntry.members.length
      : isResult(chosenEntry)
        ? ko
          ? '저장된 결과'
          : 'Saved result'
        : (chosenEntry.kind === 'video' ? (ko ? '영상' : 'Video') : ko ? '사진' : 'Photos') +
          (prepared.length ? ' · ' + prepared.length + (ko ? '장' : ' frames') : '')
    : '';
  const showPyramidCredit =
    loadedSampleId === PYRAMID_SAMPLE.id ||
    benchmarkRunning ||
    record?.benchmark?.name === STANDARD_BENCHMARK.name;
  const inputChip = (
    <div className="lab-input-chip" data-empty={!chipName || undefined}>
      {chipImage && <img src={chipImage} alt="" />}
      <div>
        <strong>{chipName ?? (ko ? '아직 고른 입력이 없습니다' : 'No input yet')}</strong>
        {chipMeta && <small>{chipMeta}</small>}
      </div>
    </div>
  );
  const runSettings = (
    <div className="lab-run-column" aria-label={ko ? '실행 설정' : 'Run settings'}>
      {(mvRows.length > 0 || file) && (
        <VideoRows
          ko={ko}
          busy={busy}
          rows={
            mvRows.length
              ? mvRows
              : [{ key: 'single', label: file!.name, file, focal, focalAuto } as VideoRow]
          }
          onFocal={(key, value) =>
            key === 'single'
              ? setFocal(value)
              : setMvRows((list) => list.map((r) => (r.key === key ? { ...r, focal: value } : r)))
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
          onRange={
            mvRows.length
              ? (key, range) =>
                  setMvRows((list) =>
                    list.map((row) => (row.key === key ? { ...row, range } : row)),
                  )
              : undefined
          }
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
      <label className="lab-field">
        <span>{ko ? '분석 해상도' : 'Analysis resolution'}</span>
        <Radio.Group
          size="small"
          disabled={busy}
          value={analysisEdge}
          onChange={(event) => setAnalysisEdge(event.target.value)}
          aria-label={ko ? '분석할 프레임의 긴 변' : 'Long side of an analysis frame'}
        >
          {ANALYSIS_EDGES.map((edge) => (
            <Radio.Button key={edge} value={edge}>
              {edge} px
            </Radio.Button>
          ))}
        </Radio.Group>
      </label>
      <div className="lab-field">
        <Tooltip
          title={
            focalAuto
              ? ko
                ? '영상에서 초점거리를 자동으로 찾습니다. 앞쪽 프레임 몇 장을 여러 값으로 계산해 보고, 가장 잘 맞는 값을 씁니다. 오른쪽 숫자가 그 결과입니다. 보정값이 있는 사진 샘플은 자기 값을 씁니다.'
                : 'Found from the video: the first few frames are tried at many focal lengths, and the best fit is kept. The number on the right is the result. Photo samples with calibration keep theirs.'
              : ko
                ? '넣은 값에서 계산을 시작하고, 계산하면서 조금 고칩니다. 오른쪽 숫자가 그 결과입니다.'
                : 'The run starts from the typed value and refines it a little. The number on the right is the result.'
          }
        >
          <span>{ko ? '초점거리 / 영상 너비' : 'Focal length / image width'} ⓘ</span>
        </Tooltip>
        <span className="lab-focal-pair">
          <span className="lab-focal-mode" data-auto={focalAuto || undefined}>
            {focalAuto ? (ko ? '자동' : 'Auto') : focal.toFixed(2)}
          </span>
          {estimatedFocal !== undefined && (
            <span className="lab-focal-estimate" aria-label={ko ? '추정값' : 'Estimate'}>
              {ko ? '추정 ' : 'estimate '}
              {estimatedFocal.toFixed(2)}
            </span>
          )}
        </span>
      </div>
      <details className="lab-advanced">
        <summary>{ko ? '고급 설정' : 'Advanced'}</summary>
        <div className="lab-advanced-row">
          <label className="lab-switch">
            <Switch
              size="small"
              disabled={busy}
              checked={!focalAuto}
              onChange={(manual) => setFocalAuto(!manual)}
            />
            {ko ? '초점거리 직접 넣기' : 'Set the focal length'}
          </label>
          <InputNumber
            size="small"
            disabled={busy || focalAuto}
            min={0.2}
            max={4}
            step={0.1}
            value={focal}
            aria-label={ko ? '초점거리 / 영상 너비' : 'Focal length / image width'}
            onChange={(v) => setFocal(Number(v ?? DEFAULT_VIDEO_FOCAL))}
          />
        </div>
        <p className="field-help">
          {ko
            ? '렌즈 값을 알 때만 켜세요. 초점거리를 영상 너비로 나눈 값입니다. 휴대폰 기본 카메라는 0.7 안팎입니다.'
            : 'Turn on only when you know the lens. The value is the focal length divided by the image width: about 0.7 for the main camera of a phone.'}
        </p>
      </details>
      <div className="lab-switch-pair">
        <label className="lab-switch">
          <Switch
            size="small"
            disabled={busy}
            checked={denseEnabled}
            onChange={setDenseEnabled}
            aria-describedby="lab-dense-help"
          />
          {ko ? 'Dense MVS' : 'Dense MVS'}
        </label>
        <Tooltip
          title={
            denseEnabled
              ? undefined
              : ko
                ? '메시는 Dense 포인트 클라우드로 만듭니다. Dense MVS를 먼저 켜 주세요.'
                : 'The mesh is built from the dense cloud. Turn on Dense MVS first.'
          }
        >
          <label className="lab-switch">
            <Switch
              size="small"
              disabled={busy || !denseEnabled}
              checked={denseEnabled && meshEnabled}
              onChange={setMeshEnabled}
            />
            {ko ? '메시' : 'Mesh'}
          </label>
        </Tooltip>
      </div>
      <p className="field-help" id="lab-dense-help">
        {ko
          ? 'Sparse 복원이 끝나면 이어서 Dense 포인트 클라우드를 만듭니다. GPU가 없는 환경에서는 수십 분 걸릴 수 있습니다.'
          : 'Builds a dense point cloud on the same GPU after the sparse map. It can take tens of minutes on a software GPU.'}
      </p>
      <label className="lab-field lab-dense-preset">
        <span>{ko ? 'Dense MVS 조합' : 'Dense MVS combination'}</span>
        <select
          value={densePreset.id}
          disabled={busy || !denseEnabled}
          aria-describedby="lab-dense-preset-help"
          onChange={(event) =>
            setDensePreset(DENSE_PRESETS.find((preset) => preset.id === event.target.value)!)
          }
        >
          {DENSE_PRESETS.map((preset) => (
            <option key={preset.id} value={preset.id}>
              {preset.label[ko ? 0 : 1]}
            </option>
          ))}
        </select>
      </label>
      <p className="field-help" id="lab-dense-preset-help">
        {densePreset.fusion === 'tsdf'
          ? ko
            ? 'TSDF는 시점별 신뢰도로 표면을 검증하고, 다른 두 시점에서 확인된 평면의 빈 곳을 보완합니다. 부정확한 점은 남을 수 있습니다.'
            : 'TSDF weighs visibility by camera confidence and completes planar gaps verified by two other views. Inaccurate points may remain.'
          : ko
            ? '픽셀 융합은 여러 시점에서 일치하는 점을 남겨, 빈 영역이 더 많아질 수 있습니다.'
            : 'Pixel fusion keeps points that agree across multiple views, so more gaps may remain.'}{' '}
        {densePreset.clean
          ? ko
            ? '정리를 켜면 잡음과 중복 점을 줄이지만, 바닥 같은 실제 표면도 제거될 수 있습니다.'
            : 'Cleanup reduces noise and duplicate points, but may also remove real surfaces such as the floor.'
          : ko
            ? '정리를 끄면 융합된 점을 추가로 제거하지 않습니다.'
            : 'Cleanup is off, so the fused cloud keeps its points without further removal.'}{' '}
        {ko
          ? '정리를 꺼도 깊이 검증과 융합은 수행합니다. 임시 깊이 미리보기는 완료 시 최종 점군으로 교체됩니다.'
          : 'Depth validation and fusion still run with cleanup off. Temporary depth previews are replaced by the final cloud when processing ends.'}
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
            ? '초점거리와 왜곡은 데이터셋의 보정값을 쓰고, 비교할 정답 카메라는 ' +
              sample.referenceName +
              '입니다.'
            : 'Intrinsics are the dataset calibration; the reference cameras are ' +
              sample.referenceName +
              '.'}
        </p>
      )}
      <Button
        block
        disabled={busy || benchmarkRunning}
        onClick={() => {
          setSettingsOpen(false);
          void runBenchmark();
        }}
      >
        {ko ? '표준 벤치마크' : 'Standard benchmark'}
      </Button>
      <p className="field-help">
        {ko
          ? '표준 벤치마크는 피라미드 샘플 8프레임을 두 번(처음 실행, 다시 실행) 돌려 속도를 잽니다.'
          : 'The standard benchmark runs the eight-frame pyramid sample twice, cold and warm.'}
      </p>
    </div>
  );
  const runActions = (
    <div className="lab-run-actions">
      <Tooltip
        title={
          running
            ? denseRunning
              ? ko
                ? 'Dense MVS 정지 (Sparse 복원 유지)'
                : 'Stop Dense MVS (keep sparse reconstruction)'
              : ko
                ? '정지'
                : 'Stop'
            : ko
              ? '실행'
              : 'Run'
        }
      >
        <Button
          className="lab-run-toggle"
          type="primary"
          danger={running}
          icon={running ? <BorderOutlined aria-hidden /> : <CaretRightOutlined aria-hidden />}
          aria-label={running ? (ko ? '정지' : 'Stop') : ko ? '실행' : 'Run'}
          disabled={
            !running &&
            (busy || benchmarkRunning || (!mvRows.length && !prepared.length && !source))
          }
          onClick={() => {
            if (running) {
              benchmarkStopRef.current = true;
              // 5.3: stop Dense MVS without discarding the sparse reconstruction.
              if (denseCancelRef.current) return denseCancelRef.current();
              abortRef.current?.abort();
              videoRef.current?.pause();
              terminateWorker();
            } else {
              setSettingsOpen(false);
              void (mvRows.length ? runMultiVideo() : runSfM());
            }
          }}
        />
      </Tooltip>
    </div>
  );
  const recordControls = (
    <details className="lab-record-panel lab-panel">
      <summary>
        {ko ? '실행 기록과 공유' : 'Run record and sharing'}{' '}
        <span>
          {consent
            ? ko
              ? '자동 전송 켜짐'
              : 'Automatic sharing on'
            : ko
              ? '선택 사항'
              : 'Optional'}
        </span>
      </summary>
      <div className="run-record-consent">
        <Checkbox checked={consent} onChange={(event) => setConsent(event.target.checked)}>
          <Tooltip
            title={
              ko
                ? '실행 기록을 TypeCV 개발자에게 보내는 데 동의합니다. 실행 전에 선택하면 완료와 실패 기록이 바로 전송됩니다. 기록에 담기는 내용은 아래 "기록 보기"에 있습니다.'
                : 'I agree to send run records to the TypeCV developers. Checked before a run, each record is sent as soon as the run completes or fails. What a record holds is under "View the record" below.'
            }
          >
            {ko ? '실행 기록 보내기 (끝나면 바로)' : 'Send the run record (when it ends)'}
          </Tooltip>
        </Checkbox>
        <Tooltip
          title={
            filesAvailable
              ? ko
                ? '이번 실행의 결과 파일(복원 JSON, Sparse PLY, 있으면 Dense PLY와 메시 PLY)을 기록과 함께 보냅니다. 내 사진이나 영상을 썼다면 결과에 그 장면이 담깁니다.'
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
            {ko ? '결과 파일도 보내기' : 'Also send the result files'}
          </Checkbox>
        </Tooltip>
        <span className="run-record-consent-status">
          {sendState.status === 'sent'
            ? (ko ? '보냄 ✓ ' : 'Sent ✓ ') + sendState.id.slice(-8)
            : sendState.status === 'sending'
              ? ko
                ? '보내는 중…'
                : 'Sending…'
              : sendState.status === 'failed'
                ? ko
                  ? '전송 실패'
                  : 'Not sent'
                : ''}
        </span>
      </div>
      {record && (
        <details id="run-record" className="run-record">
          <summary>{ko ? '기록 보기' : 'View the record'}</summary>
          <p className="field-help">
            {ko
              ? '기록에는 GPU 이름, 브라우저 정보, 화면 크기, 프레임 수와 해상도, 설정, 결과 수치, 단계별 시간, 실행한 GPU 계산 이름과 계산별 GPU 시간, GPU의 기능 목록과 계산 한도가 들어갑니다. 실패하면 오류 메시지도 들어갑니다. 영상, 이미지, 파일 이름, IP 주소는 들어가지 않습니다. 아래 JSON이 보내는 내용 전부입니다.'
              : 'A record holds the GPU adapter strings, the browser user agent, screen size, input frame count and resolution, options, result counts, per-stage timings, the chunk measurements of the long kernels with the time budget, the names and GPU time of the kernels that ran, and the feature list and compute limits of the GPU adapter. A failed run also carries the error and the last kernel. It holds no video, image, file name or IP address. The JSON below is everything that is sent.'}
          </p>
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
            {record.benchmark ? ' · ' + record.benchmark.name + ' ' + record.benchmark.pass : ''}
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
                !consent ||
                sendState.status === 'sending' ||
                (sendState.status === 'sent' && !resendWithFiles)
              }
              onClick={() => void sendRecord()}
            >
              {resendWithFiles
                ? ko
                  ? '결과 파일과 함께 다시 보내기'
                  : 'Send again with the result files'
                : ko
                  ? '기록 보내기'
                  : 'Send record'}
            </Button>
            <Button onClick={saveRecord}>{ko ? 'JSON 다운로드' : 'Download JSON'}</Button>
            <Tooltip title={ko ? '복사' : 'Copy'}>
              <Button
                aria-label={ko ? '복사' : 'Copy'}
                icon={<CopyOutlined aria-hidden />}
                onClick={() => navigator.clipboard?.writeText(JSON.stringify(record, null, 2))}
              />
            </Tooltip>
          </Space>
          <p className="run-record-status" role="status">
            {sendState.status === 'sent'
              ? (ko ? '보냄: ' : 'Sent: ') +
                sendState.id +
                (sendState.files
                  ? (ko ? ' · 파일 ' : ' · files ') +
                    sendState.files.stored.length +
                    (sendState.files.failed.length
                      ? (ko ? ', 못 보냄: ' : ', not stored: ') +
                        sendState.files.failed
                          .map((file) => file.name + ' (' + file.reason + ')')
                          .join(', ')
                      : '')
                  : '')
              : sendState.status === 'failed'
                ? (ko ? '보내지 못했습니다: ' : 'Not sent: ') + sendState.message
                : sendState.status === 'sending'
                  ? (ko ? '보내는 중…' : 'Sending…') +
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
            <summary>{ko ? '보낼 JSON' : 'JSON to be sent'}</summary>
            <pre>{JSON.stringify(record, null, 2)}</pre>
          </details>
        </details>
      )}
    </details>
  );
  return (
    <div className="lab-app">
      <div className="lab-command-bar">
        <div className="lab-topbar">
          {inputChip}
          <Button
            type={hasInput ? 'default' : 'primary'}
            onClick={() => setInputOpen(true)}
            aria-expanded={inputOpen}
            aria-label={
              hasInput
                ? ko
                  ? '입력 바꾸기…'
                  : 'Change input…'
                : ko
                  ? '샘플이나 내 영상 고르기…'
                  : 'Choose a sample or my video…'
            }
            title={hasInput ? (ko ? '입력 바꾸기' : 'Change input') : undefined}
            icon={hasInput ? <SwapOutlined aria-hidden /> : undefined}
            disabled={busy}
          >
            {!hasInput && (ko ? '샘플이나 내 영상 고르기…' : 'Choose a sample or my video…')}
          </Button>
        </div>
        <div className="lab-command-actions">
          <Tooltip title={ko ? '설정' : 'Settings'}>
            <Button
              icon={<SettingOutlined aria-hidden />}
              aria-label={ko ? '설정' : 'Settings'}
              onClick={() => setSettingsOpen(true)}
              aria-expanded={settingsOpen}
              aria-haspopup="dialog"
              aria-controls="lab-run-settings"
            />
          </Tooltip>
          {runActions}
        </div>
      </div>
      {(busy || complete || !!status) && (
        <div className="lab-run-status" data-running={busy || undefined}>
          {(busy || complete) && (
            <div className="lab-run-progress" data-testid="run-progress">
              <span>{progressLabel}</span>
              <WorkProgress label={progressLabel} percent={runProgress} />
              <span className="lab-run-progress-value" aria-hidden>
                {runProgress === undefined ? '…' : `${runProgress}%`}
              </span>
            </div>
          )}
          <div className="lab-status" role="status" aria-live="polite">
            {preparation ? (
              <GpuPreparationProgress preparation={preparation} ko={ko} />
            ) : (
              status ||
              (ko ? '위에서 샘플을 고르거나 내 영상을 올려 주세요.' : 'Choose a video or sample.')
            )}
          </div>
          {!preparation && mvBoard && mvRows.length > 0 && (
            <details className="lab-stage-details">
              <summary>{ko ? '실행 단계 보기' : 'Run stages'}</summary>
              <ol
                className="mv-stages"
                aria-label={ko ? '실행 단계' : 'Run stages'}
                data-testid="mv-stages"
              >
                {mvBoard.rows.map((row, i) => (
                  <li key={row.id} data-state={row.state} data-stage={row.id}>
                    <strong>
                      {i + 1}. {STAGE_WORDS[row.id][ko ? 1 : 0]}
                    </strong>
                    <span>
                      {row.state === 'pending'
                        ? '—'
                        : stageCount(row, ko) + ' · ' + formatElapsed(row.elapsedMs, ko)}
                    </span>
                  </li>
                ))}
              </ol>
            </details>
          )}
          {complete && (
            <p className="field-help">
              {ko ? '걸린 시간' : 'Total elapsed'} {(totalMs / 1000).toFixed(2)}s ·{' '}
              {ko
                ? '모든 프레임 분석과 마지막 다듬기를 포함한 시간'
                : 'Includes all frame analysis and final local refinement'}
            </p>
          )}
          {(denseSummary || denseCloud) && (
            <DenseSummary ko={ko} summary={denseSummary} cloud={denseCloud} budget={pointBudget} />
          )}
        </div>
      )}
      {mounted && (
        <Drawer
          className="lab-settings-drawer"
          open={settingsOpen}
          onClose={() => setSettingsOpen(false)}
          width="min(560px, 100vw)"
          title={ko ? '실행 설정' : 'Run settings'}
        >
          <div id="lab-run-settings">{runSettings}</div>
        </Drawer>
      )}
      {mounted && (
        <Drawer
          className="lab-input-drawer"
          open={inputOpen}
          onClose={() => setInputOpen(false)}
          placement="left"
          width="min(820px, 100vw)"
          forceRender
          title={ko ? '입력 고르기' : 'Choose the input'}
        >
          <Tabs
            size="small"
            activeKey={inputTab}
            onChange={(key) => setInputTab(key as InputTab)}
            items={INPUT_TABS.map((tab) => ({
              key: tab,
              label:
                (ko ? INPUT_TAB_LABELS[tab][1] : INPUT_TAB_LABELS[tab][0]) +
                ' ' +
                catalog.filter((entry) => inputTabOf(entry, tab)).length,
            }))}
          />
          <div className="sample-catalog" role="list">
            {catalog.filter(inTab).map((entry) =>
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
                      onError={() => setMissingSamples((current) => new Set(current).add(entry.id))}
                    />
                  </Tooltip>
                  <figcaption>
                    <strong>{entry.name}</strong>
                    <div className="sample-card-tags">
                      <Tag>
                        {entry.kind === 'video' ? (ko ? '영상' : 'video') : ko ? '사진' : 'photos'}
                      </Tag>
                      {entry.kind !== 'video' && (
                        <Tag color="blue">{ko ? '보정값 있음' : 'calibrated'}</Tag>
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
                              ? 'COLMAP이 이 데이터셋에서 찾은 SIFT 특징점과 매칭을 그대로 씁니다. 특징점 찾기와 매칭은 건너뛰고, TypeCV의 복원 단계(검증, 카메라 등록, 삼각측량, 번들 조정)만 실행해 COLMAP 카메라와 비교합니다.'
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
                            {ko ? 'COLMAP 특징점 사용' : 'Features only'}
                          </Checkbox>
                        </Tooltip>
                      )}
                      {entry.features && (
                        <span className="sr-only" id={'sample-mode-help-' + entry.id}>
                          {ko
                            ? '켜면 COLMAP이 찾은 SIFT 특징점과 매칭을 그대로 쓰고, TypeCV의 복원 단계만 실행합니다.'
                            : "When on, the run uses COLMAP's SIFT keypoints and matches, skips feature detection and matching, and runs only TypeCV's mapper."}
                        </span>
                      )}
                      <Button
                        size="small"
                        className={
                          loading?.id === entry.id
                            ? 'sample-load-button is-loading'
                            : loadedSampleId === entry.id
                              ? 'sample-load-button is-loaded'
                              : 'sample-load-button'
                        }
                        style={
                          loading?.id === entry.id
                            ? ({
                                '--load-progress':
                                  (loading.total ? (100 * loading.done) / loading.total : 0) + '%',
                              } as CSSProperties)
                            : undefined
                        }
                        disabled={
                          (busy && loading?.id !== entry.id) || missingSamples.has(entry.id)
                        }
                        aria-label={'Load ' + entry.name}
                        aria-busy={loading?.id === entry.id || undefined}
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
                            : loadedSampleId === entry.id
                              ? ko
                                ? '불러옴 ✓'
                                : 'Loaded ✓'
                              : ko
                                ? '불러오기'
                                : 'Load'}
                      </Button>
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
                {ko ? '내 영상 고르기 (여러 개 가능)' : 'Choose my videos / frame JSON'}
              </Button>
            </Upload>
            <Button
              disabled={busy || !file?.name.toLowerCase().endsWith('.json')}
              onClick={loadPreparedFrames}
            >
              {ko ? '프레임 JSON 불러오기' : 'Prepare analysis frames'}
            </Button>
            <Tooltip
              title={
                ko
                  ? '예전에 내보낸 복원 JSON과 Dense·메시 PLY를 다시 계산하지 않고 엽니다. 여러 파일을 함께 고를 수 있습니다.'
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
            <p className="field-help">
              {ko
                ? '영상은 1초에 4장만 읽습니다(`?fps=`로 바꿀 수 있음). 그중 앞 키프레임보다 화면이 4% 넘게 움직였거나 1초가 지난 프레임만 키프레임으로 쓰고, 나머지는 건너뜁니다. `?gate=legacy`를 붙이면 예전 규칙(특징점 8px 이상 이동)을 씁니다. 브라우저가 재생할 수 있는 형식이어야 합니다. 초점거리는 자동으로 찾고(고급 설정에서 직접 넣을 수 있음), 렌즈 왜곡은 없다고 봅니다. 영상은 이 브라우저 밖으로 나가지 않습니다.'
                : 'Decodes the video at 4 frames per second (`?fps=` overrides it) and skips the frames between. A sampled frame becomes a keyframe when the image moved at least 4 % of the frame since the last keyframe (phase correlation of a small grayscale thumbnail, checked before feature extraction) or one second has passed; other frames are skipped. `?gate=legacy` uses the older rule (features moved at least 8 px). Requires a browser-supported codec. Finds the focal length from the video (Advanced sets it by hand) and assumes zero distortion. Video stays in this browser.'}
            </p>
          </div>
        </Drawer>
      )}
      {(mvGraph?.components.some((component) => component.reconstructed) ||
        (complete && !!mvRows.length && videoIds.length > 0)) && (
        <div
          className="lab-source-navigation"
          aria-label={ko ? '복원 묶음과 영상 선택' : 'Component and video selection'}
        >
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
          {complete && !!mvRows.length && videoIds.length > 0 && (
            <label className="sfm-video-selection">
              <span>{ko ? '영상' : 'Video'}</span>
              <select
                aria-label={ko ? '탐색할 영상' : 'Browse video'}
                value={activeVideo}
                onChange={(event) => {
                  setSelectedVideo(event.target.value);
                  selectFrame(0);
                }}
              >
                {videoIds.map((id) => (
                  <option key={id} value={id}>
                    {mvRows.find((row) => row.videoId === id)?.label ?? id}
                  </option>
                ))}
              </select>
              <span>{ko ? '복원된 카메라 프레임' : 'Registered camera frames'}</span>
            </label>
          )}
        </div>
      )}
      <div className={mapWide ? 'sfm-live-grid map-wide' : 'sfm-live-grid'}>
        <section
          className="sfm-stream lab-panel"
          tabIndex={complete ? 0 : undefined}
          aria-label={ko ? '영상과 분석 프레임' : 'Video and analyzed frame'}
          aria-keyshortcuts={complete ? 'ArrowLeft ArrowRight' : undefined}
          onKeyDown={(event) => {
            if (
              busy ||
              !complete ||
              !navigationFrames.length ||
              event.altKey ||
              event.ctrlKey ||
              event.metaKey
            )
              return;
            if (event.target !== event.currentTarget && event.target !== videoRef.current) return;
            if (event.key !== 'ArrowLeft' && event.key !== 'ArrowRight') return;
            event.preventDefault();
            selectFrame(
              Math.max(
                0,
                Math.min(
                  navigationFrames.length - 1,
                  navigationIndex + (event.key === 'ArrowRight' ? 1 : -1),
                ),
              ),
            );
          }}
        >
          <div className="sfm-map-head">
            <Title level={4}>{ko ? '영상과 프레임' : 'Video & frame'}</Title>
            <label className="lab-switch">
              <Switch size="small" checked={showOverlay} onChange={setShowOverlay} />
              {ko ? '분석 표시' : 'Frame overlay'}
            </label>
          </div>
          <div className="sfm-media">
            <AnalysisFrame
              overlay={displayedOverlay}
              scene={overlayScene}
              annotate={showOverlay}
              transparent={!busy && !!videoURL}
              visible={
                busy ||
                !videoURL ||
                (videoAligned && !!selectedFrame && overlay?.frame.id === selectedFrame.id)
              }
            >
              {!busy && !videoURL && !displayedOverlay && !selectedFrame && (
                <div className="sfm-empty-state">
                  <span className="sfm-empty-icon" aria-hidden>
                    ▻
                  </span>
                  <strong>{ko ? '영상에서 시작하세요' : 'Start with a video'}</strong>
                  <span>
                    {ko
                      ? '샘플이나 내 영상을 고르면 이곳에 표시됩니다.'
                      : 'Choose a sample or your own footage to see it here.'}
                  </span>
                </div>
              )}
              <video
                ref={videoRef}
                src={videoURL || undefined}
                aria-label={ko ? '원본 영상 재생과 탐색' : 'Source video playback and seeking'}
                controls={!busy && !!videoURL}
                muted
                playsInline
                preload="metadata"
                onPlay={() => {
                  setReplaying(false);
                  syncAligned();
                }}
                onPause={syncVideoSelection}
                onSeeking={() => {
                  setVideoAligned(false);
                  const video = videoRef.current;
                  if (
                    !videoSourceLoading.current &&
                    video &&
                    programmaticSeek.current !== undefined &&
                    Math.abs(video.currentTime - programmaticSeek.current) >= 0.02
                  )
                    programmaticSeek.current = undefined;
                }}
                onSeeked={syncVideoSelection}
                onTimeUpdate={syncVideoSelection}
                onLoadedMetadata={() => {
                  const video = videoRef.current!;
                  if (!Number.isFinite(video.duration) || video.duration < 1) {
                    setSource(undefined);
                    setStatus(
                      ko
                        ? '1초보다 긴 영상을 골라 주세요.'
                        : 'Choose a video at least one second long.',
                    );
                    return;
                  }
                  videoSourceLoading.current = false;
                  programmaticSeek.current = complete ? selectedFrame?.time : undefined;
                  setVideoReady((ready) => ready + 1);
                  if (!file || complete || mvRows.length) return;
                  setSource({
                    name: file.name,
                    width: video.videoWidth,
                    height: video.videoHeight,
                    duration: video.duration,
                  });
                  setStatus(
                    ko
                      ? '분석 구간을 고르고 실행하세요. 기본값은 영상 전체입니다.'
                      : 'Choose an analysis range and press Run. The whole video is selected by default.',
                  );
                }}
                onError={() =>
                  setStatus(
                    ko
                      ? '이 브라우저는 이 영상을 열 수 없습니다. prepare-sfm-video.py로 만든 JSON을 골라 주세요.'
                      : 'This browser cannot decode the video. Select JSON from prepare-sfm-video.py.',
                  )
                }
              />
            </AnalysisFrame>{' '}
            {complete && selectedFrame && !displayedOverlay && (
              <p className="sfm-frame-loading" role="status" data-testid="frame-loading">
                {frameLoadError
                  ? ko
                    ? '이 프레임의 이미지를 불러올 수 없습니다.'
                    : 'The image for this frame is unavailable.'
                  : ko
                    ? '선택한 프레임 불러오는 중…'
                    : 'Loading selected frame…'}
              </p>
            )}
            <div
              className="sfm-depth-legend"
              title={
                ko
                  ? '깊이는 카메라에서 앞으로 잰 상대값이고, 미터 단위가 아닙니다.'
                  : 'Depth is along the camera forward axis in relative units, not meters.'
              }
            >
              <span>
                {ko ? '가까움' : 'Near'} {currentDepths.length ? currentRange[0].toFixed(2) : '—'}
              </span>
              <span className="sfm-depth-ramp" />
              <span>
                {ko ? '멂' : 'Far'} {currentDepths.length ? currentRange[1].toFixed(2) : '—'}
              </span>
            </div>
          </div>
          {!mvRows.length &&
            file &&
            !file.name.toLowerCase().endsWith('.json') &&
            !!source?.duration && (
              <VideoRangeEditor
                duration={source.duration}
                value={analysisRange}
                disabled={busy}
                ko={ko}
                onChange={setAnalysisRange}
                onPreview={(time) => {
                  const video = videoRef.current;
                  if (video) {
                    video.pause();
                    video.currentTime = time;
                  }
                }}
              />
            )}
          {(busy || navigationFrames.length > 0) && (
            <div className="sfm-timeline" data-video={!!videoURL || undefined}>
              <span className="sfm-timeline-text">
                {ko ? '프레임' : 'Frame'} {navigationFrames.length ? navigationIndex + 1 : '—'} /{' '}
                {navigationFrames.length || '—'} ·{' '}
                {selectedFrame ? selectedFrame.time.toFixed(2) + 's' : '—'}
              </span>
              {!videoURL && (
                <input
                  type="range"
                  aria-label="Frame timeline"
                  min={0}
                  max={Math.max(0, navigationFrames.length - 1)}
                  value={navigationIndex}
                  disabled={busy || navigationFrames.length === 0}
                  onChange={(event) => {
                    selectFrame(Number(event.target.value));
                  }}
                />
              )}
              {!videoURL && (
                <div className="sfm-timeline-actions">
                  <Button
                    size="small"
                    icon={<StepBackwardOutlined aria-hidden />}
                    title={ko ? '이전 프레임' : 'Previous frame'}
                    aria-label={ko ? '이전 프레임' : 'Previous frame'}
                    disabled={busy || navigationIndex === 0}
                    onClick={() => selectFrame(navigationIndex - 1)}
                  />
                  <Button
                    size="small"
                    icon={<StepForwardOutlined aria-hidden />}
                    title={ko ? '다음 프레임' : 'Next frame'}
                    aria-label={ko ? '다음 프레임' : 'Next frame'}
                    disabled={busy || navigationIndex >= navigationFrames.length - 1}
                    onClick={() => selectFrame(navigationIndex + 1)}
                  />
                  <Button
                    size="small"
                    icon={
                      replaying ? <PauseOutlined aria-hidden /> : <CaretRightOutlined aria-hidden />
                    }
                    title={
                      replaying
                        ? ko
                          ? '재생 중지'
                          : 'Pause trajectory'
                        : ko
                          ? '카메라 움직임 재생'
                          : 'Replay camera motion'
                    }
                    aria-label={
                      replaying
                        ? ko
                          ? '재생 중지'
                          : 'Pause trajectory'
                        : ko
                          ? '카메라 움직임 재생'
                          : 'Replay camera motion'
                    }
                    disabled={busy || navigationFrames.length === 0}
                    onClick={() => {
                      if (!replaying)
                        selectFrame(
                          navigationIndex >= navigationFrames.length - 1 ? 0 : navigationIndex,
                        );
                      setReplaying(!replaying);
                    }}
                  />
                </div>
              )}
              {videoURL && (
                <span
                  className="sfm-keyboard-hint"
                  title={
                    ko
                      ? '영상이나 이 패널을 선택한 뒤 방향키로 분석 프레임을 한 장씩 이동합니다.'
                      : 'Focus the video or this panel, then use the arrow keys to step through analyzed frames.'
                  }
                >
                  ← → {ko ? '프레임 이동' : 'Step frames'}
                </span>
              )}
            </div>
          )}
          {complete && (
            <div className="sfm-frame-selection">
              <label className="lab-switch">
                <Switch
                  size="small"
                  checked={followCamera}
                  onChange={(follow) => {
                    setFollowCamera(follow);
                    if (!follow) setCameraView(undefined);
                  }}
                  disabled={busy}
                />
                {ko ? '선택한 카메라 따라가기' : 'Follow selected camera'}
              </label>
              <span>
                {selectedView
                  ? (ko ? '카메라 #' : 'Camera #') + selectedView.id
                  : ko
                    ? '카메라 위치 없는 프레임'
                    : 'Frame has no registered camera'}
              </span>
            </div>
          )}
          <ol
            className="sfm-pipeline"
            aria-label={ko ? '프레임 처리 단계' : 'Frame processing stages'}
          >
            {[
              [
                ko ? '특징점 찾기' : 'Detect features',
                frameCounts.features + (ko ? '개' : ' points'),
              ],
              [
                ko ? '앞 프레임과 매칭' : 'Match frames',
                frameCounts.matches + (ko ? ' 후보' : ' candidates'),
              ],
              [
                ko ? '카메라 추정' : 'Estimate camera',
                frameCounts.inliers + (ko ? ' 맞는 점' : ' geometric inliers'),
              ],
              [
                ko ? '3D 점 계산' : 'Update 3D depth',
                currentDepths.length + (ko ? ' 3D 점' : ' depth points'),
              ],
            ].map(([label, count], i) => {
              const unavailable = frameFinished && i >= 2 && !selectedView;
              return (
                <li
                  key={label}
                  title={
                    unavailable
                      ? ko
                        ? '이 프레임은 카메라 위치를 정하지 못했습니다. 찾은 특징점과 매칭 후보는 위에서 볼 수 있습니다.'
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
          <details className="sfm-analysis-details">
            <summary>{ko ? '프레임 통계와 조작법' : 'Frame statistics and controls'}</summary>
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
                  (ko ? '건너뛴 프레임 (키프레임이 아님)' : 'Frames skipped (not keyframes)') +
                  (sampleFps > 0
                    ? ko
                      ? ` · 영상은 1초에 ${sampleFps}장 읽음`
                      : ` · video sampled at ${sampleFps} frames/s`
                    : '')
                }
                value={processed + skipped > 0 ? skipped : undefined}
              />
              <Tile
                id="rate"
                label={ko ? '초당 분석 프레임' : 'Analysis frames/s'}
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
                label={ko ? '영상 길이 / 분석 시간 (초)' : 'Video / analyzed (s)'}
                value={
                  playTime > 0 || analysisTime > 0
                    ? playTime.toFixed(2) + ' / ' + analysisTime.toFixed(2)
                    : undefined
                }
              />
            </div>
            <div className="sfm-help">
              <p>
                {ko
                  ? '회색 점은 찾은 특징점, 주황 선은 매칭 후보, 색 점은 깊이(빨강은 가까움, 파랑은 멂)입니다. 깊이는 카메라에서 앞으로 잰 상대값이고, 미터 단위가 아닙니다.'
                  : 'Gray marks detected features, orange lines candidate matches, and colored points the camera depth (red near, blue far). Depth is along the camera forward axis in relative units, not meters.'}
              </p>
              <p>
                {ko
                  ? '분석을 시작하면 특징점과 매칭이 여기에 나타납니다. 키프레임이 아닌 프레임은 건너뛰고, 건너뛴 수는 통계에 나옵니다.'
                  : 'Actual features and matches appear when analysis starts. Frames that are not keyframes are skipped, and the statistics count them.'}
              </p>
              {sampleFps > 0 ? (
                <p className="sfm-sampling">
                  {ko
                    ? `영상은 1초에 ${sampleFps}장만 읽고, 나머지 프레임은 읽지 않습니다.`
                    : `The video is sampled at ${sampleFps} frames/s. The other frames are not decoded.`}
                </p>
              ) : null}
            </div>
          </details>
        </section>
        <section className="sfm-map lab-panel" ref={mapRef}>
          <div className="sfm-map-head">
            <Title level={4}>{ko ? '3D 복원' : '3D reconstruction'}</Title>
            <div className="sfm-toolbar">
              <Radio.Group
                value={navigation}
                onChange={(event) => setNavigation(event.target.value)}
                aria-label={ko ? '조작 방식' : 'Navigation'}
              >
                <Radio.Button value="orbit">{ko ? '둘러보기' : 'Orbit'}</Radio.Button>
                <Radio.Button value="fly">{ko ? '자유 이동' : 'Fly'}</Radio.Button>
              </Radio.Group>
              <div className="sfm-map-actions">
                <Popover
                  trigger="click"
                  placement="bottomRight"
                  getPopupContainer={() => (mapFullscreen && mapRef.current) || document.body}
                  content={
                    <div className="sfm-view-menu" role="menu">
                      {(
                        [
                          [
                            'camera',
                            '선택한 카메라에서 보기',
                            'Through the selected camera',
                            !selectedFrame ||
                              !reconstruction?.views.some((view) => view.id === selectedFrame.id),
                            () =>
                              selectedFrame &&
                              setCameraView({
                                id: selectedFrame.id,
                                nonce: (cameraView?.nonce ?? 0) + 1,
                              }),
                          ],
                          [
                            'fit',
                            '전체 보기',
                            'Whole map',
                            !reconstruction?.views.length,
                            () => setFitNonce((nonce) => nonce + 1),
                          ],
                          ...(
                            [
                              ['top', '위에서', 'From above'],
                              ['front', '앞에서', 'From the front'],
                              ['side', '옆에서', 'From the side'],
                            ] as const
                          ).map(
                            ([kind, koLabel, enLabel]) =>
                              [
                                kind,
                                koLabel,
                                enLabel,
                                !reconstruction?.views.length,
                                () => setViewPreset((p) => ({ kind, nonce: (p?.nonce ?? 0) + 1 })),
                              ] as const,
                          ),
                        ] as const
                      ).map(([key, koLabel, enLabel, off, act]) => (
                        <Button
                          key={key}
                          type="text"
                          role="menuitem"
                          block
                          disabled={off}
                          onClick={act}
                        >
                          {ko ? koLabel : enLabel}
                        </Button>
                      ))}
                    </div>
                  }
                >
                  <Button
                    aria-haspopup="true"
                    aria-label={ko ? '시점' : 'View'}
                    title={ko ? '시점' : 'View'}
                    icon={<EyeOutlined aria-hidden />}
                  />
                </Popover>
                <PlacementMenu
                  ko={ko}
                  disabled={!reconstruction?.views.length}
                  mode={transformMode}
                  onMode={setTransformMode}
                  state={placement}
                  onCommand={runPlacement}
                  getPopupContainer={() => (mapFullscreen && mapRef.current) || document.body}
                />
                <Tooltip
                  title={
                    mapFullscreen
                      ? ko
                        ? '전체 화면 끝내기 (Esc)'
                        : 'Exit full screen (Esc)'
                      : ko
                        ? '전체 화면'
                        : 'Full screen'
                  }
                  getPopupContainer={() => mapRef.current ?? document.body}
                >
                  <Button
                    icon={
                      mapFullscreen ? (
                        <FullscreenExitOutlined aria-hidden />
                      ) : (
                        <FullscreenOutlined aria-hidden />
                      )
                    }
                    aria-label={ko ? '전체 화면' : 'Full screen'}
                    aria-pressed={mapFullscreen}
                    disabled={typeof document !== 'undefined' && !document.fullscreenEnabled}
                    onClick={() => {
                      if (document.fullscreenElement) void document.exitFullscreen();
                      else void mapRef.current?.requestFullscreen();
                    }}
                  />
                </Tooltip>
                <Popover
                  trigger="click"
                  placement="bottomRight"
                  getPopupContainer={() => (mapFullscreen && mapRef.current) || document.body}
                  content={
                    <div className="sfm-map-options">
                      <label className="lab-switch sfm-wide-switch">
                        <Switch size="small" checked={mapWide} onChange={setMapWide} />
                        {ko ? '3D 화면 넓게' : 'Wide 3D'}
                      </label>
                      <label className="lab-switch">
                        <Switch size="small" checked={showCameras} onChange={setShowCameras} />
                        {ko ? '카메라' : 'Cameras'}
                      </label>
                      <label className="lab-switch">
                        <Switch size="small" checked={showPhoto} onChange={setShowPhoto} />
                        {ko ? '사진' : 'Photo'}
                      </label>
                      <label className="lab-switch">
                        <Switch size="small" checked={showLinks} onChange={setShowLinks} />
                        {ko ? '2D–3D 연결' : '2D–3D links'}
                      </label>
                      <label className="lab-switch">
                        <Switch
                          size="small"
                          checked={lightBackground}
                          onChange={(checked) => setBackground(checked ? 'light' : 'dark')}
                        />
                        {ko ? '밝은 배경' : 'Light background'}
                      </label>
                      <Tooltip
                        placement="left"
                        getPopupContainer={() => (mapFullscreen && mapRef.current) || document.body}
                        title={
                          displayedDense?.count
                            ? undefined
                            : denseSwitchTitle(denseSummary, denseAsked, ko)
                        }
                      >
                        <label className="lab-switch">
                          <Switch
                            size="small"
                            disabled={!displayedDense?.count}
                            checked={!!displayedDense?.count && showDense}
                            onChange={setShowDense}
                          />
                          {ko ? 'Dense 포인트 클라우드' : 'Dense'}
                        </label>
                      </Tooltip>
                      <Tooltip
                        placement="left"
                        getPopupContainer={() => (mapFullscreen && mapRef.current) || document.body}
                        title={
                          denseMesh?.indices.length
                            ? undefined
                            : !denseCloud?.count
                              ? denseSwitchTitle(denseSummary, denseAsked, ko)
                              : ko
                                ? '메시가 아직 없습니다. 실행에서 Dense MVS와 메시를 켜고 다시 실행해 주세요.'
                                : 'No mesh. Turn on Dense MVS and Mesh in the run settings and run.'
                        }
                      >
                        <label className="lab-switch">
                          <Switch
                            size="small"
                            disabled={!denseMesh?.indices.length}
                            checked={!!denseMesh?.indices.length && showMesh}
                            onChange={setShowMesh}
                          />
                          {ko ? '메시' : 'Mesh'}
                        </label>
                      </Tooltip>
                    </div>
                  }
                >
                  <Button
                    aria-haspopup="true"
                    aria-label={ko ? '표시' : 'Display'}
                    title={ko ? '표시' : 'Display'}
                    icon={<SlidersOutlined aria-hidden />}
                  />
                </Popover>
              </div>
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
                  {ko ? 'Dense' : 'dense'} {resultInfo.dense.toLocaleString('en-US')}
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
                  {ko ? '카메라 등록' : 'registered'} ·{' '}
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
          <div
            className="sfm-media"
            data-multi-stage={
              (running &&
                mvRows.length > 0 &&
                !denseActive &&
                (!reconstruction?.views.length || !!mvConnectionGraph)) ||
              undefined
            }
          >
            <ViewerBoundary onError={(message) => onViewerState({ status: 'error', message })}>
              <SfmViewer
                reconstruction={reconstruction}
                videoGraph={
                  running && mvRows.length > 0 && !denseActive ? mvConnectionGraph : undefined
                }
                videoColors={mvRows.length ? mvVideoColors : undefined}
                onState={onViewerState}
                selectedViewId={selectedViewId}
                photo={displayedOverlay?.frame}
                cameraView={cameraView}
                fitNonce={fitNonce}
                viewPreset={viewPreset}
                transformMode={transformMode}
                placementCommand={placementCommand}
                onPlacement={setPlacement}
                showCameras={showCameras}
                showPhoto={showPhoto}
                showLinks={showLinks}
                features={selectedFeatures}
                lightBackground={lightBackground}
                navigation={navigation}
                dense={displayedDense}
                showDense={showDense}
                mesh={denseMesh}
                showMesh={showMesh}
                densePointBudget={pointBudget}
                densePreview={densePreview}
                live={busy}
                activity={
                  denseActivity?.component !== undefined && denseActivity.component !== mvSelected
                    ? undefined
                    : activity
                }
                onSelectView={(id) => {
                  if (busy) return;
                  const frame = (mvRows.length ? componentFrames : navigationFrames).find(
                    (frame) => frame.id === id,
                  );
                  if (!frame) return;
                  if (mvRows.length) setSelectedVideo(frame.videoId);
                  const list = mvRows.length
                    ? componentFrames.filter((candidate) => candidate.videoId === frame.videoId)
                    : navigationFrames;
                  selectFrame(list.findIndex((candidate) => candidate.id === id));
                }}
              />
            </ViewerBoundary>
            {running && mvRows.length > 0 && !denseActive && (
              <MultiVideoActivityPanel
                rows={mvRows}
                board={mvBoard}
                graph={mvActivity}
                thumbnails={mvThumbnails}
                activeVideo={mvActiveVideo}
                preparation={preparation}
                mapReady={!!reconstruction?.views.length}
                ko={ko}
                preview={<FrameInset overlay={overlay} ko={ko} />}
                onGraphChange={setMvConnectionGraph}
              />
            )}
            {!busy && !reconstruction && (
              <div className="sfm-empty-state sfm-map-empty">
                <span className="sfm-empty-icon" aria-hidden>
                  ◇
                </span>
                <strong>
                  {ko ? '복원된 장면을 둘러보세요' : 'Explore the reconstructed scene'}
                </strong>
                <span>
                  {ko
                    ? '실행하면 카메라 경로와 3D 점이 나타납니다.'
                    : 'Run the reconstruction to reveal cameras and 3D points.'}
                </span>
              </div>
            )}
            {busy && !denseActive && (!mvRows.length || !!reconstruction?.views.length) && (
              <FrameInset overlay={overlay} ko={ko} />
            )}
            {denseActive && <DenseActivityPanel activity={denseActivity} ko={ko} />}
            {busy && (!mvRows.length || denseActive) && (
              <LiveBadge lastEventAt={lastEventAt} ko={ko} />
            )}
            {placement.picking >= 0 && (
              <div className="sfm-pick-banner" role="status">
                <span>
                  {ko
                    ? `바닥의 점을 더블클릭하세요 (${placement.picking} / 3)`
                    : `Double-click a point on the ground (${placement.picking} / 3)`}
                  {placement.message === 'miss' &&
                    (ko ? ' · 그 자리에 점이 없습니다' : ' · No point there')}
                </span>
                <Button
                  size="small"
                  onClick={() =>
                    runPlacement('set', {
                      position: placement.position,
                      rotationDeg: placement.rotationDeg,
                      scale: placement.scale,
                    })
                  }
                >
                  {ko ? '취소' : 'Cancel'}
                </Button>
              </div>
            )}
            <Tooltip
              placement="topLeft"
              getPopupContainer={() => (mapFullscreen && mapRef.current) || document.body}
              title={
                navigation === 'orbit'
                  ? ko
                    ? '드래그: 돌리기 · 오른쪽 드래그 또는 Shift+드래그: 옮기기 · 휠: 확대 · 점이나 카메라 더블클릭: 그곳을 중심으로 돌리기 · 두 손가락: 옮기기와 확대 · 시점 ▾: 위·앞·옆에서 보기'
                    : 'Drag: orbit · Right-drag or Shift+drag: pan · Wheel: zoom · Double-click a point or camera: orbit about it · Two fingers: pan and zoom · View ▾: from above, front or side'
                  : ko
                    ? '클릭 후 드래그: 둘러보기 · W A S D Q E: 움직이기 · Shift: 빠르게 · Alt: 느리게 · 휠: 속도'
                    : 'Click, then drag: look · W A S D Q E: move · Shift: faster · Alt: slower · Wheel: speed'
              }
            >
              <p className="sfm-media-note">
                {navigation === 'orbit'
                  ? ko
                    ? '드래그: 돌리기 · 오른쪽 드래그: 옮기기 · 휠: 확대 · 더블클릭: 중심 바꾸기 ⓘ'
                    : 'Drag: orbit · Right-drag: pan · Wheel: zoom · Double-click: center ⓘ'
                  : ko
                    ? '클릭 후 드래그: 둘러보기 · W A S D Q E: 움직이기 ⓘ'
                    : 'Click, then drag: look · W A S D Q E: move ⓘ'}
              </p>
            </Tooltip>
          </div>

          <ViewerDiagnostics
            error={viewerState.status === 'error' ? viewerState.message : ''}
            ko={ko}
          />
          <div className="sfm-map-notes">
            <div className="sfm-map-message">
              {viewerState.status === 'error' ? (
                <p role="alert">{viewerState.message}</p>
              ) : !reconstruction ? (
                <p role="status" className="field-help">
                  {viewerState.status === 'initializing'
                    ? ko
                      ? '3D 화면 준비 중…'
                      : 'Preparing the 3D viewer…'
                    : ko
                      ? '복원을 시작할 두 프레임을 찾으면 3D 점이 나타납니다.'
                      : 'The map appears after a stable initial pair is found.'}
                </p>
              ) : (
                <p role="status" className="field-help">
                  {ko
                    ? '카메라를 누르면 해당 프레임을 봅니다.'
                    : 'Select a camera to see its frame.'}
                </p>
              )}
            </div>
          </div>
          <div className="sfm-tiles" aria-label={ko ? '복원 통계' : 'Reconstruction statistics'}>
            <Tile
              id="views"
              label={ko ? '카메라' : 'Registered views'}
              value={reconstruction?.views.length}
            />
            <Tile
              id="landmarks"
              label={ko ? 'Sparse 3D 포인트' : 'Sparse landmarks'}
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
              id="dense-points"
              label={ko ? 'Dense 포인트' : 'Dense points'}
              value={displayedDense?.count}
            />
          </div>
          <details className="sfm-help">
            <summary>
              {ko ? '보는 법과 조작법' : 'Legend and controls'}
              {unregistered.length > 0 &&
                ' · ' + unregistered.length + (ko ? '개 위치 못 찾음' : ' unregistered')}
            </summary>
            <div className="sfm-motion-stats">
              <Tile
                id="translation"
                label={ko ? '카메라 이동' : 'Translation'}
                hint={
                  ko
                    ? '이전 → 선택 프레임. 복원 장면 안의 상대 단위이고, 카메라 위치가 없는 프레임은 건너뜁니다.'
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
            <div className="sfm-camera-motion">
              <strong>{ko ? '이전 → 선택 프레임' : 'Previous → selected frame'}</strong>
              <span className="field-help">
                {movement
                  ? ko
                    ? '이동 거리는 복원 장면 안의 상대 단위입니다.'
                    : 'Relative scene units.'
                  : ko
                    ? '두 프레임의 카메라 위치가 나오면 표시됩니다.'
                    : 'Shown when both frame poses are registered.'}
              </span>
            </div>{' '}
            <p>
              {ko
                ? '주황색은 카메라가 지나간 길, 밝은 초록색은 선택한 카메라가 보는 방향, 파란 화살표는 바로 앞 프레임에서 움직인 방향입니다. 사진은 선택한 프레임의 실제 이미지이고, 3D 점의 색은 영상에서 가져왔습니다.'
                : 'Orange traces the camera path; green shows the selected camera forward direction; the blue arrow shows movement from the previous frame. The photo plane is the actual selected image. The 3D points carry the image colors.'}
            </p>
            <p>
              {ko
                ? '둘러보기: 휠은 커서 쪽으로 확대합니다. 카메라나 점을 더블클릭하면 그곳을 중심으로 돕니다.'
                : 'Orbit: the wheel zooms toward the cursor, and each move shrinks with the zoom. Double-click a camera to make it the center of rotation.'}
            </p>
            <p>
              {ko
                ? '자유 이동: 화면을 클릭하고 드래그로 둘러보고, W A S D로 앞뒤·좌우, Q E로 위아래로 움직입니다. Shift는 4배 빠르게, Alt는 4배 느리게, 휠은 속도를 바꿉니다.'
                : 'Fly: click the canvas, then drag to look around. W A S D move forward, back, left and right; Q E move down and up. Shift moves 4× faster, Alt 4× slower, and the wheel changes the speed.'}
            </p>
            <p>
              {ko
                ? '‘선택한 카메라 따라가기’를 켜면 프레임을 고를 때 3D 화면도 그 카메라 시점으로 바뀝니다.'
                : 'Follow selected camera moves the 3D viewpoint with video playback, seeking, arrow keys and thumbnail selection.'}
            </p>
            <p>
              {ko
                ? '이동 거리는 복원 장면 안의 상대 단위이고, 카메라 위치가 없는 프레임은 건너뜁니다.'
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
                      ? 'COLMAP 정답과 겹치는 카메라 ' + alignment.commonViews + '개 · 회전 최대 '
                      : alignment.commonViews + ' cameras in common with COLMAP · rotation max ') +
                    alignment.rotationDeg.max.toFixed(3) +
                    (ko ? '° · 중심 최대 ' : '° · center max ') +
                    (alignment.centerOfRadius.max * 100).toFixed(2) +
                    (ko ? '% (참조 궤적 반경 기준)' : '% of the reference trajectory radius')
                  : (ko ? 'COLMAP 정답과 비교할 수 없음: ' : 'No comparison with COLMAP: ') +
                    alignment.status +
                    (ko
                      ? ' (겹치는 카메라 ' + alignment.commonViews + '개)'
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
      {(complete || showPyramidCredit) && (
        <Card className="lab-panel lab-results">
          {complete && <Title level={5}>{ko ? '결과 내보내기' : 'Export results'}</Title>}
          {complete && (
            <div className="run-details">
              <Space wrap align="start">
                <Button size="small" onClick={() => download('ply')}>
                  Export PLY
                </Button>
                <Button size="small" onClick={() => download('json')}>
                  Export JSON
                </Button>
                {!!denseCloud?.count && (
                  <Button size="small" onClick={downloadDense}>
                    {ko ? 'Dense PLY 내보내기' : 'Export dense PLY'}
                  </Button>
                )}
                {!!denseMesh?.indices.length && (
                  <Button size="small" onClick={downloadMesh}>
                    {ko ? '메시 PLY 내보내기' : 'Export mesh PLY'}
                  </Button>
                )}
                <details>
                  <summary>{ko ? 'GPU 단계별 시간' : 'GPU execution stages'}</summary>
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
          {showPyramidCredit && (
            <p className="sample-credit">
              Sample: Blanka OG,{' '}
              <a href="https://www.pexels.com/video/drone-footage-of-a-pyramid-8134952/">
                Pexels pyramid video
              </a>{' '}
              · <a href="https://www.pexels.com/license/">Pexels License</a>
            </p>
          )}
        </Card>
      )}
      {gallery.length > 0 && (
        <section
          className="lab-panel sfm-frames"
          aria-label={ko ? '분석 프레임 목록' : 'Analyzed frames'}
        >
          <Title level={5}>{ko ? '프레임' : 'Frames'}</Title>
          <div className="frame-grid">
            {gallery.map(({ src, index, frame }) => {
              const selectable = !busy && navigationFrames.length > 0;
              const selected = navigationFrames.length > 0 && index === navigationIndex;
              return (
                <figure key={index} data-selected={selected || undefined}>
                  <button
                    type="button"
                    disabled={!selectable}
                    aria-current={selected ? 'true' : undefined}
                    aria-label={(ko ? '프레임 ' : 'Frame ') + (index + 1)}
                    onClick={() => {
                      selectFrame(index);
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
              ? '썸네일을 누르면 그 프레임을 봅니다. 최근 24장을 표시하며, 영상에서 시간을 이동하거나 방향키로 프레임을 고를 수 있습니다.'
              : 'Select a thumbnail to see that frame. Video thumbnails show the latest 24; seek in the video or use the arrow keys to explore the rest.'}
          </p>
        </section>
      )}
      {recordControls}
      <p className="lab-session-note">
        {ko
          ? '영상 처리는 이 브라우저에서 실행됩니다. 복원의 크기와 거리는 상대 단위입니다.'
          : 'Video processing runs in this browser. Reconstructed size and distance use relative units.'}
      </p>
    </div>
  );
}
