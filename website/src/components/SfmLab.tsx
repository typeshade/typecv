import {
  Alert,
  Button,
  Card,
  ConfigProvider,
  InputNumber,
  Progress,
  Space,
  Statistic,
  Typography,
  Upload,
  theme,
} from 'antd';
import { lazy, Suspense, useEffect, useRef, useState } from 'react';
import { cameraInSourcePixels, parsePreparedVideo } from '../lib/sfm-input';
import {
  cameraMotion,
  depthColor,
  depthPoints,
  depthRange,
  seekFrame,
  type Scene,
} from '../lib/sfm-scene';
const SfmViewer = lazy(() => import('./SfmViewer'));
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
  metrics?: { features: number; matches: number; inliers: number; registered: boolean };
};
type Reconstruction = Scene;
type Overlay = { frame: Frame; points?: Float64Array; lines?: number[]; matches?: number };

function AnalysisFrame({
  overlay,
  scene,
  selectedPoint,
  onSelect,
}: {
  overlay?: Overlay;
  scene?: Scene;
  selectedPoint?: number;
  onSelect: (id: number) => void;
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
    ctx.putImageData(
      new ImageData(new Uint8ClampedArray(frame.pixels), frame.width, frame.height),
      0,
      0,
    );
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
      if (point.id === selectedPoint) {
        ctx.strokeStyle = '#ffffff';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(point.x, point.y, 7, 0, 2 * Math.PI);
        ctx.stroke();
      }
    }
  }, [overlay, scene, selectedPoint]);
  return (
    <canvas
      ref={ref}
      className="sfm-analysis-canvas"
      aria-label="Actual video frame with reconstructed points colored by camera depth"
      onClick={(event) => {
        const canvas = ref.current!;
        const rect = canvas.getBoundingClientRect();
        const x = ((event.clientX - rect.left) * canvas.width) / rect.width;
        const y = ((event.clientY - rect.top) * canvas.height) / rect.height;
        const point = depths.reduce<ReturnType<typeof depthPoints>[number] | undefined>(
          (best, point) =>
            Math.hypot(point.x - x, point.y - y) <
            Math.hypot((best?.x ?? Infinity) - x, (best?.y ?? Infinity) - y)
              ? point
              : best,
          undefined,
        );
        if (point && Math.hypot(point.x - x, point.y - y) < (12 * canvas.width) / rect.width)
          onSelect(point.id);
      }}
    />
  );
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
      <SfmLabContent locale={locale} />
    </ConfigProvider>
  );
}

function SfmLabContent({ locale = 'en' }: { locale?: 'en' | 'ko' }) {
  const ko = locale === 'ko';
  const [file, setFile] = useState<File>();
  const [videoURL, setVideoURL] = useState('');
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
  const [focal, setFocal] = useState(1.2);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [selectedPoint, setSelectedPoint] = useState<number>();
  const [replaying, setReplaying] = useState(false);
  const [frameStep, setFrameStep] = useState(0);
  const [frameFinished, setFrameFinished] = useState(false);
  const [frameCounts, setFrameCounts] = useState({ features: 0, matches: 0, inliers: 0 });
  const [playTime, setPlayTime] = useState(0);
  const [analysisTime, setAnalysisTime] = useState(0);
  const [processed, setProcessed] = useState(0);
  const [skipped, setSkipped] = useState(0);
  const [frameMs, setFrameMs] = useState(0);
  const [totalMs, setTotalMs] = useState(0);
  const [complete, setComplete] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const workerRef = useRef<Worker | undefined>(undefined);
  const jobRef = useRef(0);
  const runFocal = useRef(focal);
  const abortRef = useRef<AbortController | undefined>(undefined);
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
  const currentDepths = depthPoints(reconstruction, overlay?.frame.id ?? -1);
  const currentRange = depthRange(currentDepths);
  useEffect(() => {
    if (!replaying || !complete || overlay?.frame.id !== selectedFrame?.id) return;
    const timer = window.setTimeout(() => {
      if (selectedIndex >= runFrames.length - 1) setReplaying(false);
      else setSelectedIndex(selectedIndex + 1);
    }, 250);
    return () => window.clearTimeout(timer);
  }, [replaying, complete, selectedIndex, runFrames.length, overlay?.frame.id]);
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
      await seekFrame(video, selectedFrame.time, abort.signal);
      const { decodeVideoFrames } = await import('../lib/sfm-video');
      for await (const decoded of decodeVideoFrames(file, abort.signal, selectedFrame.time)) {
        if (!abort.signal.aborted) setOverlay({ frame: { ...selectedFrame, ...decoded } });
      }
    };
    load().catch((error) => {
      if (!abort.signal.aborted) setStatus(String(error));
    });
    return () => abort.abort();
  }, [selectedIndex, complete, busy, prepared, videoURL]);
  const selectPoint = (id: number) => {
    setSelectedPoint(id);
    if (!busy && reconstruction && selectedFrame) {
      const point = reconstruction.landmarks.find((point) => point.trackId === id);
      if (point && !point.observations.some((o) => o.viewId === selectedFrame.id)) {
        const index = runFrames.findIndex((frame) =>
          point.observations.some((o) => o.viewId === frame.id),
        );
        if (index >= 0) setSelectedIndex(index);
      }
    }
  };
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
    setDiagnostics(undefined);
    setStages([]);
    setUnregistered([]);
    setSource(undefined);
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
    setSelectedPoint(undefined);
    setFrameStep(0);
  }
  async function loadPreparedFrames() {
    if (!file) return;
    setBusy(true);
    setReplaying(false);
    setSelectedPoint(undefined);
    try {
      if (file.size > 64 * 1024 * 1024) throw new RangeError('Frame bundle exceeds 64 MiB.');
      const input = parsePreparedVideo(await file.text());
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
  async function loadPyramidSample() {
    reset();
    setFile(undefined);
    setBusy(true);
    try {
      const times = [0, 1, 2, 4, 8, 12, 16, 20];
      const sampled: Frame[] = [],
        thumbs: string[] = [];
      const canvas = document.createElement('canvas'),
        ctx = canvas.getContext('2d')!;
      for (let i = 0; i < times.length; i++) {
        const url = '/sample/pyramid/frame-' + String(i).padStart(2, '0') + '.jpg';
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
        });
        thumbs.push(url);
      }
      setPrepared(sampled);
      setFrames(thumbs);
      setSource({
        name: 'Pexels pyramid sample',
        width: sampled[0].width,
        height: sampled[0].height,
        duration: 20,
      });
      setStatus(
        ko
          ? '실제 영상에서 추출한 8개 샘플 프레임입니다.'
          : 'Eight sample frames from real footage are ready.',
      );
    } catch (error) {
      setStatus(String(error));
    } finally {
      setBusy(false);
    }
  }

  async function runSfM() {
    if (busy || (!prepared.length && !videoURL)) return;
    const abort = new AbortController();
    abortRef.current = abort;
    const signal = abort.signal;
    const id = ++jobRef.current;
    runFocal.current = focal;
    const worker =
      workerRef.current ??
      new Worker(new URL('../workers/sfm.worker.ts', import.meta.url), { type: 'module' });
    workerRef.current = worker;
    setBusy(true);
    setReplaying(false);
    setSelectedPoint(undefined);
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
    if (!prepared.length) setFrames([]);
    setStatus(ko ? 'GPU를 준비하고 있습니다…' : 'Preparing the GPU…');
    let pending:
      { type: string; resolve: (message: any) => void; reject: (error: Error) => void } | undefined;
    let timeout = 0;
    let currentFrame: Frame | undefined;
    let counts = { features: 0, matches: 0, inliers: 0 };
    let failure: Error | undefined;
    const fail = (error: Error) => {
      failure = error;
      pending?.reject(error);
      abort.abort();
    };
    const collected: Frame[] = [];
    const armTimeout = () => {
      window.clearTimeout(timeout);
      timeout = window.setTimeout(() => {
        if (pending) fail(new Error('No response from GPU for two minutes.'));
      }, 120000);
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
    worker.onmessage = ({ data: message }) => {
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
        const label = labels[String(message.detail).split(/[ #·]/)[0]];
        if (/^(pnp|verifyPair)/.test(message.detail)) setFrameStep(2);
        if (/^(triangulation|observations|bundle)/.test(message.detail)) setFrameStep(3);
        setStatus(
          label ? label[ko ? 0 : 1] : ko ? '영상을 분석하고 있습니다…' : 'Analyzing video…',
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
        if (
          message.reconstruction.views.some((view: { id: number }) => view.id === currentFrame?.id)
        )
          setFrameStep(4);
      }
      if (message.type === 'error') fail(new Error(message.message));
      if (pending && message.type === pending.type) {
        const waiter = pending;
        pending = undefined;
        window.clearTimeout(timeout);
        waiter.resolve(message);
      }
    };
    worker.onerror = (event) => fail(new Error(event.message));
    const consume = async (frame: Frame) => {
      currentFrame = frame;
      setFrameFinished(false);
      counts = { features: 0, matches: 0, inliers: 0 };
      setFrameStep(0);
      setFrameCounts({ features: 0, matches: 0, inliers: 0 });
      collected.push({ ...frame, pixels: new Uint8ClampedArray(), image: undefined });
      setRunFrames([...collected]);
      setSelectedIndex(collected.length - 1);
      setOverlay({ frame });
      const pixels = frame.pixels.slice();
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
      setProcessed((count) => count + 1);
      setFrameFinished(true);
      setAnalysisTime(frame.time);
      setFrameMs(message.elapsedMs);
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
      await request('ready', { type: 'start', focalRatio: focal });
      if (prepared.length) {
        for (const frame of prepared) {
          signal.throwIfAborted();
          setPlayTime(frame.time);
          await consume(frame);
        }
      } else {
        const video = videoRef.current!;
        video.pause();
        const { decodeVideoFrames } = await import('../lib/sfm-video');
        let index = 0;
        for await (const decoded of decodeVideoFrames(file!, signal)) {
          signal.throwIfAborted();
          await seekFrame(video, decoded.time, signal);
          setPlayTime(decoded.time);
          const frame: Frame = { id: index++, ...decoded };
          setFrames((thumbs) => [...thumbs, frame.image!].slice(-24));
          await consume(frame);
        }
      }
      const result = await request('result', { type: 'finish' });
      setReconstruction(result.reconstruction);
      setDiagnostics(result.diagnostics);
      setUnregistered(result.unregistered);
      setStages(result.stages);
      setTotalMs(result.stages[0].elapsedMs);
      setComplete(true);
      setStatus(ko ? 'GPU SfM 복원이 완료되었습니다.' : 'GPU SfM reconstruction completed.');
    } catch (error) {
      setStatus(
        signal.aborted && !failure
          ? ko
            ? '분석을 취소했습니다.'
            : 'Analysis cancelled.'
          : (ko ? '복원 실패: ' : 'Reconstruction failed: ') + String(failure ?? error),
      );
      terminateWorker();
    } finally {
      window.clearTimeout(timeout);
      signal.removeEventListener('abort', onAbort);
      videoRef.current?.pause();
      setBusy(false);
      abortRef.current = undefined;
    }
  }

  function download(format: 'json' | 'ply') {
    if (!reconstruction) return;
    const vertices = [
      ...reconstruction.landmarks.map(
        (point) =>
          Array.from(point.xyz).join(' ') +
          ' ' +
          (point.color ?? [0.7, 0.7, 0.7]).map((v) => Math.round(v * 255)).join(' '),
      ),
      ...reconstruction.views.map(
        ({ pose: { R, t } }) =>
          [
            -(R[0] * t[0] + R[3] * t[1] + R[6] * t[2]),
            -(R[1] * t[0] + R[4] * t[1] + R[7] * t[2]),
            -(R[2] * t[0] + R[5] * t[1] + R[8] * t[2]),
          ].join(' ') + ' 255 180 84',
      ),
    ];
    const text =
      format === 'json'
        ? JSON.stringify(
            {
              source: source?.name,
              sourceVideo: source,
              assumedFocalRatio: runFocal.current,
              diagnostics,
              unregistered,
              analysisFrames: runFrames.map(({ pixels: _pixels, ...frame }) => frame),
              views: reconstruction.views.map((view) => {
                const frame = runFrames.find((candidate) => candidate.id === view.id);
                return {
                  ...view,
                  sourceCamera: frame
                    ? cameraInSourcePixels(view.camera, frame, {
                        width: frame.sourceWidth ?? frame.width,
                        height: frame.sourceHeight ?? frame.height,
                      })
                    : view.camera,
                  pose: { R: Array.from(view.pose.R), t: Array.from(view.pose.t) },
                };
              }),
              landmarks: reconstruction.landmarks.map((point) => ({
                ...point,
                xyz: Array.from(point.xyz),
              })),
            },
            null,
            2,
          )
        : [
            'ply',
            'format ascii 1.0',
            'element vertex ' + vertices.length,
            'property float x',
            'property float y',
            'property float z',
            'property uchar red',
            'property uchar green',
            'property uchar blue',
            'end_header',
            ...vertices,
          ].join('\n') + '\n';
    const url = URL.createObjectURL(new Blob([text], { type: 'application/octet-stream' }));
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = 'typecv-reconstruction.' + format;
    anchor.click();
    URL.revokeObjectURL(url);
  }
  const duration = source?.duration ?? prepared.at(-1)?.time ?? 0;
  const progress = complete
    ? 100
    : duration
      ? Math.min(99, Math.round((100 * analysisTime) / duration))
      : 0;
  return (
    <div className="lab-app">
      <Card className="lab-panel">
        <Title level={4}>{ko ? '영상 입력' : 'Video input'}</Title>
        <Space wrap align="end">
          <Upload
            disabled={busy}
            accept="video/*,application/json,.json"
            maxCount={1}
            beforeUpload={(input) => {
              reset();
              setFile(input);
              return false;
            }}
            showUploadList={{ showRemoveIcon: !busy }}
            onRemove={() => {
              reset();
              setFile(undefined);
              return true;
            }}
          >
            <Button>{ko ? '영상 / 프레임 JSON 선택' : 'Choose video / frame JSON'}</Button>
          </Upload>
          <label>
            {ko ? '초점거리 / 영상 너비' : 'Focal length / image width'}{' '}
            <InputNumber
              disabled={busy}
              min={0.2}
              max={4}
              step={0.1}
              value={focal}
              onChange={(v) => setFocal(Number(v ?? 1.2))}
            />
          </label>
          <Button
            disabled={busy || !file?.name.toLowerCase().endsWith('.json')}
            onClick={loadPreparedFrames}
          >
            {ko ? '분석 프레임 준비' : 'Prepare analysis frames'}
          </Button>
          <Button disabled={busy} onClick={loadPyramidSample}>
            {ko ? '피라미드 영상 샘플 (8프레임)' : 'Load pyramid video sample (8 frames)'}
          </Button>
          <Button type="primary" disabled={busy || (!prepared.length && !source)} onClick={runSfM}>
            {ko ? 'GPU SfM 실행' : 'Run GPU SfM'}
          </Button>
          {busy && (
            <Button
              danger
              onClick={() => {
                abortRef.current?.abort();
                videoRef.current?.pause();
                terminateWorker();
              }}
            >
              {ko ? '취소' : 'Cancel'}
            </Button>
          )}
        </Space>
        <p className="field-help">
          {ko
            ? '영상 파일의 실제 프레임을 처음부터 끝까지 순서대로 읽습니다. 분석이 느리면 다음 프레임이 기다리며, FPS 샘플링이나 전체 프레임 수 제한은 없습니다. 브라우저가 지원하는 코덱이 필요합니다. 초점거리/너비 1.2, 렌즈 왜곡 없음은 기본 가정입니다.'
            : 'Decodes every actual video frame in order from start to end, waiting for analysis without FPS sampling or a total-frame cap. Requires a browser-supported codec. Assumes focal length / width 1.2 and zero distortion.'}
        </p>
        {source && (
          <p className="field-help">
            {ko ? '원본' : 'Source'} {source.width} × {source.height} · {source.name}
          </p>
        )}
        <Alert
          type="info"
          showIcon
          message={ko ? '영상은 브라우저 안에서만 처리됩니다.' : 'Video stays in this browser.'}
          description={
            ko
              ? '원본 영상 아래에서 처리 과정을 확인하세요. 회색은 검출 특징, 주황 선은 후보 매칭입니다. 3D 복원이 끝난 점만 카메라 깊이에 따라 빨강(가까움)부터 파랑(멀음)까지 표시합니다. 오른쪽 3D 점은 실제 영상의 색상입니다. 점을 누르면 사진과 3D에서 같은 점이 선택됩니다.'
              : 'Follow each frame below the source video: gray detections, orange candidate matches, then reconstructed points colored red (near) to blue (far) by camera depth. The 3D cloud uses actual image colors. Click a point to select it in both views.'
          }
        />
        <div className="sfm-live-grid">
          <section className="sfm-stream">
            <Title level={4}>
              {ko ? '실제 영상 · 분석 프레임' : 'Source video · analyzed frame'}
            </Title>
            <video
              ref={videoRef}
              src={videoURL || undefined}
              controls={!busy}
              muted
              playsInline
              hidden={!videoURL}
              preload="metadata"
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
              ].map(([label, count], i) => (
                <li
                  key={label}
                  data-state={
                    frameFinished && i >= 2 && !selectedView
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
                  <span>{count}</span>
                </li>
              ))}
            </ol>
            {frameFinished && !selectedView && (
              <p>
                {ko
                  ? '이 프레임은 카메라 위치·깊이를 확정하지 못했습니다. 아래에서 검출점과 매칭 후보를 확인할 수 있습니다.'
                  : 'This frame has no registered camera or verified depth. Its detections and match candidates remain visible.'}
              </p>
            )}
            <AnalysisFrame
              overlay={overlay}
              scene={reconstruction}
              selectedPoint={selectedPoint}
              onSelect={selectPoint}
            />
            <div className="sfm-depth-legend">
              <span>
                {ko ? '가까움' : 'Near'} {currentDepths.length ? currentRange[0].toFixed(2) : '—'}
              </span>
              <span className="sfm-depth-ramp" />
              <span>
                {ko ? '멀음' : 'Far'} {currentDepths.length ? currentRange[1].toFixed(2) : '—'}
              </span>
            </div>
            <p>
              {ko
                ? '깊이는 카메라 앞쪽 축의 상대값입니다. 회색 점은 깊이 미확정이며, 실제 미터 단위가 아닙니다.'
                : 'Depth is along the camera forward axis in relative scene units. Gray points have no verified depth; values are not meters.'}
            </p>
            {!overlay && (
              <p>
                {ko
                  ? '분석이 시작되면 실제 특징점과 매칭이 표시됩니다.'
                  : 'Actual features and matches appear when analysis starts.'}
              </p>
            )}
            <p className="sfm-timing">
              {ko ? '영상' : 'Video'} {playTime.toFixed(2)}s · {ko ? '분석 완료' : 'Analyzed'}{' '}
              {analysisTime.toFixed(2)}s · {ko ? '지연' : 'Lag'}{' '}
              {Math.max(0, playTime - analysisTime).toFixed(2)}s
            </p>
            <p className="sfm-throughput">
              {processed} {ko ? '프레임 처리' : 'frames processed'} · {skipped}{' '}
              {ko ? '영상 프레임 건너뜀' : 'video frames skipped'} ·{' '}
              {frameMs > 0 ? (1000 / frameMs).toFixed(2) : '—'}{' '}
              {ko ? '분석 프레임/초' : 'analysis frames/s'}
            </p>
            {overlay && (
              <p>
                {overlay.frame.width} × {overlay.frame.height} ·{' '}
                {overlay.points ? overlay.points.length / 2 : 0} {ko ? '특징' : 'features'} ·{' '}
                {overlay.matches ?? 0} {ko ? '후보 매칭' : 'candidate matches'}
              </p>
            )}
          </section>
          <section className="sfm-map">
            <Title level={4}>{ko ? '누적 3D 복원' : 'Growing 3D reconstruction'}</Title>
            {reconstruction ? (
              <Suspense fallback={<div className="viewer">Loading 3D viewer…</div>}>
                <SfmViewer
                  reconstruction={reconstruction}
                  selectedViewId={overlay?.frame.id}
                  photo={overlay?.frame}
                  selectedPoint={selectedPoint}
                  onSelectPoint={selectPoint}
                  onSelectView={(id) => {
                    if (!busy) {
                      const index = runFrames.findIndex((frame) => frame.id === id);
                      if (index >= 0) setSelectedIndex(index);
                    }
                  }}
                />
              </Suspense>
            ) : (
              <div className="viewer sfm-map-empty">
                {ko
                  ? '안정적인 초기 시점 쌍을 찾으면 지도가 나타납니다.'
                  : 'The map appears after a stable initial pair is found.'}
              </div>
            )}
            <div className="sfm-summary">
              <Statistic
                title={ko ? '등록 카메라' : 'Registered views'}
                value={reconstruction?.views.length ?? 0}
              />
              <Statistic
                title={ko ? '희소 3D 점' : 'Sparse landmarks'}
                value={reconstruction?.landmarks.length ?? 0}
              />
              {diagnostics && (
                <Statistic
                  title={ko ? '평균 재투영 오차' : 'Mean reprojection error'}
                  value={diagnostics.meanReprojectionErrorPx}
                  precision={3}
                  suffix="px"
                />
              )}
            </div>
            <p>
              {ko
                ? '주황색은 카메라 경로, 밝은 초록색은 선택 카메라의 앞쪽, 파란 화살표는 직전 프레임에서의 이동입니다. 사진 평면은 선택 프레임의 실제 이미지입니다.'
                : 'Orange traces the camera path; green shows the selected camera forward direction; the blue arrow shows movement from the previous frame. The photo plane is the actual selected image.'}
            </p>
            <div className="sfm-camera-motion">
              <strong>{ko ? '이전 프레임 → 선택 프레임' : 'Previous → selected frame'}</strong>
              {movement ? (
                <p>
                  {ko ? '이동량' : 'Translation'} {movement.distance.toFixed(4)} ·{' '}
                  {ko ? '회전' : 'Rotation'} {movement.degrees.toFixed(2)}°<br />
                  {ko ? '카메라 기준 오른쪽 / 아래 / 앞' : 'Camera-relative right / down / forward'}
                  : {movement.local.map((value) => value.toFixed(4)).join(' / ')}
                </p>
              ) : (
                <p>
                  {ko
                    ? '두 프레임의 카메라가 등록되면 이동·회전이 표시됩니다.'
                    : 'Movement and rotation appear when both frame poses are registered.'}
                </p>
              )}
              <small>
                {ko
                  ? '이동량은 복원 장면의 상대 단위이며, 카메라 위치가 없는 프레임은 보간하지 않습니다.'
                  : 'Translation uses relative scene units. Missing poses are not interpolated.'}
              </small>
            </div>
            {unregistered.length > 0 && (
              <ul>
                {unregistered.map((view) => (
                  <li key={view.viewId}>
                    {view.viewId}: {view.reason}
                  </li>
                ))}
              </ul>
            )}
          </section>
        </div>
        {runFrames.length > 0 && (
          <div className="sfm-timeline">
            <label>
              {ko ? '전체 영상 프레임 탐색' : 'Explore all analyzed frames'} · {selectedIndex + 1} /{' '}
              {runFrames.length} · {selectedFrame?.time.toFixed(3)}s
              <input
                type="range"
                aria-label="Frame timeline"
                min={0}
                max={Math.max(0, runFrames.length - 1)}
                value={selectedIndex}
                disabled={busy}
                onChange={(event) => {
                  setReplaying(false);
                  setSelectedIndex(Number(event.target.value));
                }}
              />
            </label>
            <Space>
              <Button
                disabled={busy || selectedIndex === 0}
                onClick={() => setSelectedIndex(selectedIndex - 1)}
              >
                {ko ? '이전 프레임' : 'Previous frame'}
              </Button>
              <Button
                disabled={busy || selectedIndex >= runFrames.length - 1}
                onClick={() => setSelectedIndex(selectedIndex + 1)}
              >
                {ko ? '다음 프레임' : 'Next frame'}
              </Button>
              <Button
                disabled={busy}
                onClick={() => {
                  if (selectedIndex >= runFrames.length - 1) setSelectedIndex(0);
                  setReplaying(!replaying);
                }}
              >
                {replaying
                  ? ko
                    ? '재생 중지'
                    : 'Pause trajectory'
                  : ko
                    ? '카메라 이동 재생'
                    : 'Replay camera motion'}
              </Button>
            </Space>
            {selectedPoint !== undefined && (
              <p>
                {ko ? '선택한 3D 점' : 'Selected 3D point'} #{selectedPoint}
              </p>
            )}
          </div>
        )}
        <div className="lab-status" role="status" aria-live="polite">
          {status || (ko ? '영상 또는 샘플을 선택하세요.' : 'Choose a video or sample.')}
        </div>
        {(busy || complete) && <Progress percent={progress} status={busy ? 'active' : 'success'} />}
        {complete && (
          <p>
            {ko ? '전체 소요 시간' : 'Total elapsed'} {(totalMs / 1000).toFixed(2)}s ·{' '}
            {ko
              ? '전체 프레임 분석과 최근 구간의 마지막 정밀화 포함'
              : 'Includes all frame analysis and final local refinement'}
          </p>
        )}
        {frames.length > 0 && (
          <div className="frame-grid">
            {frames.map((src, i) => (
              <figure key={i}>
                <img src={src} alt={(ko ? '프레임 ' : 'Frame ') + (i + 1)} />
                <figcaption>
                  {Math.max(0, runFrames.length - frames.length) + i + 1} ·{' '}
                  {(prepared.length
                    ? prepared[i]
                    : runFrames[Math.max(0, runFrames.length - frames.length) + i]
                  )?.time.toFixed(2)}
                  s
                </figcaption>
              </figure>
            ))}
          </div>
        )}
        {frames.length > 0 && (
          <p>
            {ko
              ? '동영상 썸네일은 최근 24개를 표시합니다. 전체 처리 시점은 위 타임라인에서 선택할 수 있습니다.'
              : 'Video thumbnails show the latest 24 frames; the timeline includes every processed time.'}
          </p>
        )}
        {complete && (
          <div className="run-details">
            <Space>
              <Button size="small" onClick={() => download('ply')}>
                Export PLY
              </Button>
              <Button size="small" onClick={() => download('json')}>
                Export JSON
              </Button>
              <details>
                <summary>{ko ? 'GPU 실행 단계' : 'GPU execution stages'}</summary>
                <pre>{JSON.stringify(stages, null, 2)}</pre>
              </details>
            </Space>
          </div>
        )}
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
