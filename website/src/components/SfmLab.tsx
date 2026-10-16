import {
  Alert,
  Button,
  Card,
  InputNumber,
  Progress,
  Space,
  Statistic,
  Typography,
  Upload,
} from 'antd';
import { lazy, Suspense, useEffect, useRef, useState } from 'react';
const SfmViewer = lazy(() => import('./SfmViewer'));

const { Text, Title } = Typography;
type SampledFrame = {
  id: number;
  width: number;
  height: number;
  pixels: Uint8ClampedArray;
  time: number;
};
type Reconstruction = {
  views: {
    id: number;
    pose: { R: Float64Array; t: Float64Array };
    camera: { fx: number; fy: number; cx: number; cy: number };
  }[];
  landmarks: { xyz: Float64Array }[];
};

export default function SfmLab({ locale = 'en' }: { locale?: 'en' | 'ko' }) {
  const ko = locale === 'ko';
  const [file, setFile] = useState<File>();
  const [frames, setFrames] = useState<string[]>([]);
  const [frameData, setFrameData] = useState<SampledFrame[]>([]);
  const [reconstruction, setReconstruction] = useState<Reconstruction>();
  const [diagnostics, setDiagnostics] = useState<any>();
  const [unregistered, setUnregistered] = useState<number[]>([]);
  const [stages, setStages] = useState<any[]>([]);
  const [featureCounts, setFeatureCounts] = useState<number[]>([]);
  const [matchedPairs, setMatchedPairs] = useState(0);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState('');
  const [focal, setFocal] = useState(1.2);
  const [busy, setBusy] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const workerRef = useRef<Worker | undefined>(undefined);
  const jobRef = useRef(0);
  const timeoutRef = useRef<number | undefined>(undefined);
  useEffect(
    () => () => {
      workerRef.current?.terminate();
      window.clearTimeout(timeoutRef.current);
    },
    [],
  );

  async function sampleFrames() {
    if (!file) return;
    workerRef.current?.terminate();
    setReconstruction(undefined);
    setDiagnostics(undefined);
    setBusy(true);
    setFrames([]);
    setFrameData([]);
    setStatus(
      ko ? '브라우저에서 영상 메타데이터를 읽는 중…' : 'Reading video metadata in this browser…',
    );
    const video = videoRef.current!;
    const url = URL.createObjectURL(file);
    video.src = url;
    try {
      await new Promise<void>((resolve, reject) => {
        video.onloadedmetadata = () => resolve();
        video.onerror = () =>
          reject(new Error(ko ? '영상 디코딩에 실패했습니다.' : 'Video decoding failed.'));
      });
      const duration = video.duration;
      if (!Number.isFinite(duration) || duration < 1)
        throw new Error(ko ? '영상 길이가 너무 짧습니다.' : 'The video is too short.');
      const canvas = canvasRef.current!;
      const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
      // Bound the feature workload of uploaded video frames.
      const scale = Math.min(1, 512 / Math.max(video.videoWidth, video.videoHeight));
      canvas.width = Math.round(video.videoWidth * scale);
      canvas.height = Math.round(video.videoHeight * scale);
      const times = Array.from({ length: 8 }, (_, i) => duration * (i / 7));
      const thumbs: string[] = [],
        sampled: SampledFrame[] = [];
      for (const time of times) {
        setStatus(
          `${ko ? '프레임 디코딩' : 'Decoding frame'} ${thumbs.length + 1}/8 · ${time.toFixed(2)} s`,
        );
        await new Promise<void>((resolve, reject) => {
          video.onseeked = () => resolve();
          video.onerror = () => reject(new Error('Video seek failed.'));
          video.currentTime = Math.min(time, duration - 0.05);
        });
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        const image = ctx.getImageData(0, 0, canvas.width, canvas.height);
        sampled.push({
          id: sampled.length,
          width: canvas.width,
          height: canvas.height,
          pixels: image.data.slice(),
          time,
        });
        thumbs.push(canvas.toDataURL('image/jpeg', 0.78));
      }
      setFrames(thumbs);
      setFrameData(sampled);
      setStatus(
        ko
          ? '8개 프레임을 준비했습니다. GPU SfM 실행을 선택하세요.'
          : 'Eight frames are ready. Start the GPU SfM run.',
      );
    } catch (err) {
      setStatus(err instanceof Error ? err.message : String(err));
    } finally {
      video.pause();
      video.removeAttribute('src');
      video.load();
      URL.revokeObjectURL(url);
      setBusy(false);
    }
  }

  async function loadPyramidSample() {
    workerRef.current?.terminate();
    setBusy(true);
    setReconstruction(undefined);
    setStatus(
      ko
        ? 'Pexels 피라미드 영상의 샘플 프레임을 불러오는 중…'
        : 'Loading sample frames from the Pexels pyramid video…',
    );
    try {
      const canvas = canvasRef.current!;
      const ctx = canvas.getContext('2d', { willReadFrequently: true })!;
      const times = [0, 1, 2, 4, 8, 12, 16, 20],
        thumbs: string[] = [],
        sampled: SampledFrame[] = [];
      for (let i = 0; i < times.length; i++) {
        const response = await fetch(`/sample/pyramid/frame-${String(i).padStart(2, '0')}.jpg`);
        if (!response.ok) throw new Error(`Sample frame ${i + 1} could not be loaded.`);
        const bitmap = await createImageBitmap(await response.blob());
        canvas.width = bitmap.width;
        canvas.height = bitmap.height;
        ctx.drawImage(bitmap, 0, 0);
        bitmap.close();
        const pixels = ctx.getImageData(0, 0, canvas.width, canvas.height).data.slice();
        sampled.push({ id: i, width: canvas.width, height: canvas.height, pixels, time: times[i] });
        thumbs.push(canvas.toDataURL('image/jpeg', 0.78));
      }
      setFrames(thumbs);
      setFrameData(sampled);
      setStatus(
        ko
          ? '실제 드론 영상에서 추출한 8개 프레임을 준비했습니다.'
          : 'Eight frames extracted from the real drone video are ready.',
      );
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }

  function runSfM() {
    if (frameData.length < 2 || busy) return;
    workerRef.current?.terminate();
    setBusy(true);
    setProgress(0);
    setReconstruction(undefined);
    setStatus(ko ? 'TypeShade WebGPU 초기화 중…' : 'Opening TypeShade WebGPU…');
    const worker = new Worker(new URL('../workers/sfm.worker.ts', import.meta.url), {
      type: 'module',
    });
    workerRef.current = worker;
    let settled = false;
    let lastDetail = 'WebGPU initialization';
    const finish = () => {
      if (settled) return;
      settled = true;
      if (timeoutRef.current !== undefined) window.clearTimeout(timeoutRef.current);
      timeoutRef.current = undefined;
      setBusy(false);
      worker.terminate();
      if (workerRef.current === worker) workerRef.current = undefined;
    };
    const armTimeout = () => {
      window.clearTimeout(timeoutRef.current);
      timeoutRef.current = window.setTimeout(() => {
        setStatus(
          ko
            ? `2분 동안 진행 응답이 없어 대기를 종료했습니다. 마지막 단계: ${lastDetail}`
            : `Stopped waiting after 2 minutes without progress. Last step: ${lastDetail}`,
        );
        finish();
      }, 120_000);
    };
    armTimeout();
    const id = ++jobRef.current;
    worker.onmessage = (event: MessageEvent<any>) => {
      const message = event.data;
      if (message.id !== id) return;
      if (message.type === 'progress') {
        lastDetail = message.detail ?? message.stage;
        armTimeout();
        const value =
          message.stage === 'features'
            ? 5 + (35 * message.completed) / message.total
            : message.stage === 'matches'
              ? 40 + (35 * message.completed) / message.total
              : 80;
        setProgress(Math.round(value));
        const stage =
          message.stage === 'features'
            ? ko
              ? 'GPU 특징 추출'
              : 'GPU feature extraction'
            : message.stage === 'matches'
              ? ko
                ? 'GPU 디스크립터 매칭'
                : 'GPU descriptor matching'
              : ko
                ? '증분 SfM 복원'
                : 'Incremental SfM';
        setStatus(
          `${stage} · ${message.detail ?? `${message.completed}/${message.total}`}${message.features ? ` · ${message.features} ${ko ? '특징' : 'features'}` : ''}`,
        );
      } else if (message.type === 'result') {
        setReconstruction(message.reconstruction);
        setDiagnostics(message.diagnostics);
        setUnregistered(message.unregistered);
        setStages(message.stages);
        setFeatureCounts(message.featureCounts);
        setMatchedPairs(message.matchedPairs);
        setProgress(100);
        setStatus(ko ? 'GPU SfM 복원이 완료되었습니다.' : 'GPU SfM reconstruction completed.');
        finish();
      } else if (message.type === 'error') {
        setStatus(`${ko ? '복원 실패' : 'Reconstruction failed'}: ${message.message}`);
        finish();
      }
    };
    worker.onerror = (event) => {
      setStatus(`${ko ? '워커 오류' : 'Worker error'}: ${event.message}`);
      finish();
    };
    const transferableFrames = frameData.map((frame) => ({
      ...frame,
      pixels: frame.pixels.slice(),
    }));
    worker.postMessage(
      { id, frames: transferableFrames, focalRatio: focal },
      transferableFrames.map((frame) => frame.pixels.buffer),
    );
  }

  function cancel() {
    if (timeoutRef.current !== undefined) window.clearTimeout(timeoutRef.current);
    timeoutRef.current = undefined;
    workerRef.current?.terminate();
    workerRef.current = undefined;
    setBusy(false);
    setStatus(ko ? 'GPU 작업을 취소했습니다.' : 'GPU run cancelled.');
  }

  function download(format: 'json' | 'ply') {
    if (!reconstruction) return;
    let text: string;
    let name: string;
    if (format === 'json') {
      text = JSON.stringify(
        {
          source: file?.name ?? 'Pexels pyramid sample',
          assumedFocalRatio: focal,
          diagnostics,
          unregistered,
          views: reconstruction.views.map((view) => ({
            id: view.id,
            camera: view.camera,
            pose: { R: Array.from(view.pose.R), t: Array.from(view.pose.t) },
          })),
          landmarks: reconstruction.landmarks.map((point) => Array.from(point.xyz)),
        },
        null,
        2,
      );
      name = 'typecv-reconstruction.json';
    } else {
      const xyz = reconstruction.landmarks.map((point) => [...point.xyz]);
      const cameras = reconstruction.views.map((view) => {
        const { R, t } = view.pose;
        return [
          -(R[0] * t[0] + R[3] * t[1] + R[6] * t[2]),
          -(R[1] * t[0] + R[4] * t[1] + R[7] * t[2]),
          -(R[2] * t[0] + R[5] * t[1] + R[8] * t[2]),
        ];
      });
      const vertices = [
        ...xyz.map((v) => `${v.join(' ')} 105 177 255`),
        ...cameras.map((v) => `${v.join(' ')} 255 180 84`),
      ];
      text = `ply\nformat ascii 1.0\nelement vertex ${vertices.length}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n${vertices.join('\n')}\n`;
      name = 'typecv-reconstruction.ply';
    }
    const url = URL.createObjectURL(
      new Blob([text], {
        type: format === 'json' ? 'application/json' : 'application/octet-stream',
      }),
    );
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = name;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="lab-app">
      <Card className="lab-panel">
        <Title level={4}>{ko ? '영상 입력' : 'Video input'}</Title>
        <Space wrap align="end">
          <Upload
            accept="video/mp4,video/webm,video/quicktime"
            maxCount={1}
            beforeUpload={(f) => {
              workerRef.current?.terminate();
              setFile(f);
              setFrames([]);
              setFrameData([]);
              setReconstruction(undefined);
              return false;
            }}
            showUploadList={{ showRemoveIcon: true }}
            onRemove={() => {
              setFile(undefined);
              setFrames([]);
              setFrameData([]);
              setReconstruction(undefined);
              return true;
            }}
          >
            <Button>{ko ? '로컬 영상 선택' : 'Choose local video'}</Button>
          </Upload>
          <label>
            {ko ? '초점거리 / 영상 너비' : 'Focal length / image width'}{' '}
            <InputNumber
              min={0.2}
              max={4}
              step={0.1}
              value={focal}
              onChange={(v) => setFocal(Number(v ?? 1.2))}
            />
          </label>
          <Button disabled={!file || busy} onClick={sampleFrames}>
            {ko ? '8개 프레임 추출' : 'Extract 8 frames'}
          </Button>
          <Button disabled={busy} onClick={loadPyramidSample}>
            {ko ? '피라미드 영상 샘플' : 'Load pyramid video sample'}
          </Button>
          <Button type="primary" disabled={frameData.length < 2 || busy} onClick={runSfM}>
            {ko ? 'GPU SfM 실행' : 'Run GPU SfM'}
          </Button>
          {busy && (
            <Button danger onClick={cancel}>
              {ko ? '취소' : 'Cancel'}
            </Button>
          )}
        </Space>
        <p className="field-help">
          {ko
            ? '초점거리/너비 1.2를 기본 가정값으로 사용합니다. 실제 카메라 보정 데이터가 아닙니다.'
            : 'Uses an assumed focal-length/width ratio of 1.2, not calibrated camera metadata.'}
        </p>
        <Alert
          style={{ marginTop: 18 }}
          type="info"
          showIcon
          message={ko ? '영상은 브라우저 안에서만 처리됩니다.' : 'Video stays in this browser.'}
          description={
            ko
              ? '프레임 특징 추출, BRIEF 매칭과 기하 복원은 TypeShade WebGPU로 실행합니다. 화면 표시는 Three.js를 사용합니다.'
              : 'Frame features, BRIEF matching and geometric reconstruction run through TypeShade WebGPU. Three.js renders the result.'
          }
        />
        <div className="lab-status" role="status" aria-live="polite">
          {status ||
            (ko
              ? '영상을 선택한 뒤 프레임을 추출하세요.'
              : 'Choose a video and extract frames to begin.')}
        </div>
        {busy && <Progress percent={progress} status="active" />}
        {frames.length > 0 && (
          <div className="frame-grid">
            {frames.map((src, i) => (
              <figure key={i}>
                <img src={src} alt={`${ko ? '프레임' : 'Frame'} ${i + 1}`} />
                <figcaption>
                  {i + 1} / 8 · {frameData[i]?.time.toFixed(2)}s
                </figcaption>
              </figure>
            ))}
          </div>
        )}
        <p className="sample-credit">
          {ko ? (
            <>
              샘플 출처: Blanka OG,{' '}
              <a href="https://www.pexels.com/video/drone-footage-of-a-pyramid-8134952/">
                Pexels 드론 피라미드 영상
              </a>{' '}
              · <a href="https://www.pexels.com/license/">Pexels 라이선스</a>
            </>
          ) : (
            <>
              Sample: Blanka OG,{' '}
              <a href="https://www.pexels.com/video/drone-footage-of-a-pyramid-8134952/">
                Drone Footage of a Pyramid on Pexels
              </a>{' '}
              · <a href="https://www.pexels.com/license/">Pexels License</a>
            </>
          )}
        </p>
        {reconstruction && (
          <>
            <div className="sfm-summary">
              <Statistic
                title={ko ? '복원 시점' : 'Registered views'}
                value={reconstruction.views.length}
                suffix={`/ ${frameData.length}`}
              />
              <Statistic
                title={ko ? '희소 3D 점' : 'Sparse landmarks'}
                value={reconstruction.landmarks.length}
              />
              <Statistic
                title={ko ? '평균 재투영 오차' : 'Mean reprojection error'}
                value={diagnostics?.meanReprojectionErrorPx ?? 0}
                precision={3}
                suffix="px"
              />
              <Statistic
                title={ko ? 'GPU 검출 특징 / 매칭 쌍' : 'Detected features / matched pairs'}
                value={`${featureCounts.reduce((a, b) => a + b, 0)} / ${matchedPairs}`}
              />
            </div>
            <div className="viewer-head">
              <Title level={4}>{ko ? '복원된 희소 장면' : 'Sparse reconstruction'}</Title>
              <Text type="secondary">
                {ko
                  ? '마우스로 회전·확대할 수 있습니다. 점은 랜드마크, 주황색 선은 카메라입니다.'
                  : 'Drag to orbit and zoom. Points are landmarks; orange lines mark cameras.'}
              </Text>
            </div>
            <Suspense fallback={<div className="viewer viewer-loading">Loading 3D viewer…</div>}>
              <SfmViewer reconstruction={reconstruction} />
            </Suspense>
            <div className="run-details">
              <Text>
                {ko
                  ? `미등록 시점: ${unregistered.length ? unregistered.join(', ') : '없음'}`
                  : `Unregistered views: ${unregistered.length ? unregistered.join(', ') : 'none'}`}
              </Text>
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
          </>
        )}
      </Card>
      <video ref={videoRef} muted playsInline hidden />
      <canvas ref={canvasRef} hidden />
      <Text type="secondary">
        {ko
          ? '영상은 네트워크로 전송되지 않습니다. 복원 결과는 재현 가능성을 위해 다운로드할 수 있습니다.'
          : 'Video is not sent over the network. The reconstruction stays available locally for inspection.'}
      </Text>
    </div>
  );
}
