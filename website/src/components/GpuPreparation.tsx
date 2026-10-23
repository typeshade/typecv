import { CheckOutlined, LoadingOutlined } from '@ant-design/icons';
import { useEffect, useState } from 'react';
import { GPU_PREPARATION_PHASES, type GpuPreparation } from '../lib/gpu-preparation';
import { formatElapsed } from '../lib/multi-video-stages';

const WORDS = {
  viewer: ['3D 화면', '3D view'],
  worker: ['분석 작업 시작', 'Starting analysis'],
  runtime: ['분석 엔진 로딩', 'Loading analysis engine'],
  vocabulary: ['영상 연결 데이터 로딩', 'Loading video matching data'],
  adapter: ['GPU 확인', 'Checking GPU'],
  device: ['GPU 연결', 'Connecting GPU'],
  session: ['분석 준비', 'Setting up analysis'],
} as const;

/** Stay visibly active while an awaited preparation phase has no measurable percentage. */
export default function GpuPreparationProgress({
  preparation,
  ko,
}: {
  preparation: GpuPreparation;
  ko: boolean;
}) {
  const [now, setNow] = useState(() => performance.now());
  useEffect(() => {
    const timer = window.setInterval(() => setNow(performance.now()), 500);
    return () => window.clearInterval(timer);
  }, []);
  const phases = GPU_PREPARATION_PHASES.filter(
    (phase) => preparation.multi || phase !== 'vocabulary',
  );
  const current = phases.indexOf(preparation.phase);
  const elapsed = Math.max(0, now - preparation.startedAt);
  return (
    <section
      className="lab-preparation"
      aria-label={ko ? '실행 준비' : 'Run preparation'}
      data-testid="gpu-preparation"
      data-phase={preparation.phase}
    >
      <div className="lab-preparation-heading">
        <LoadingOutlined
          spin
          role="progressbar"
          aria-label={ko ? 'GPU 준비' : 'GPU preparation'}
          aria-valuetext={WORDS[preparation.phase][ko ? 0 : 1]}
        />
        <strong>{ko ? 'GPU 준비 중' : 'Preparing the GPU'}</strong>
        <span
          className="lab-preparation-elapsed"
          data-elapsed-seconds={Math.floor(elapsed / 1000)}
          aria-live="off"
        >
          {formatElapsed(elapsed, ko)}
        </span>
      </div>
      <ol className="lab-preparation-steps" aria-label={ko ? '준비 단계' : 'Preparation stages'}>
        {phases.map((phase, index) => (
          <li
            key={phase}
            data-phase={phase}
            data-state={index < current ? 'done' : index === current ? 'running' : 'pending'}
            aria-current={index === current ? 'step' : undefined}
          >
            {index < current ? (
              <CheckOutlined aria-hidden />
            ) : (
              <span className="lab-preparation-dot" aria-hidden />
            )}
            {WORDS[phase][ko ? 0 : 1]}
          </li>
        ))}
      </ol>
      <p>
        {ko
          ? '첫 실행은 준비에 시간이 더 걸릴 수 있습니다.'
          : 'The first run can take longer to prepare.'}
      </p>
    </section>
  );
}
