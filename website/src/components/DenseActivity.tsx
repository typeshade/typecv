import { useEffect, useState } from 'react';
import {
  denseActivityDescription,
  denseActivityProgress,
  denseActivityText,
  type DenseActivity as Activity,
} from '../lib/dense-activity';
import WorkProgress from './WorkProgress';

const STEPS = [
  { phases: ['photometric', 'geometric'], words: ['깊이', 'Depth'] },
  { phases: ['allocate', 'assign', 'integrate', 'fusion'], words: ['융합', 'Fusion'] },
  { phases: ['carve'], words: ['빈 공간', 'Free space'] },
  { phases: ['count', 'extract'], words: ['표면', 'Surface'] },
  { phases: ['readback'], words: ['결과 회수', 'Read results'] },
  { phases: ['sort', 'clean', 'mesh'], words: ['정리', 'Finish'] },
];

/** Visible during the quiet tail of the MVS, including a GPU readback or synchronous cleanup. */
export default function DenseActivity({ activity, ko }: { activity?: Activity; ko: boolean }) {
  const [startedAt] = useState(() => performance.now());
  const [waitingAt, setWaitingAt] = useState(startedAt);
  const [now, setNow] = useState(startedAt);
  useEffect(() => {
    if (!activity) setWaitingAt(performance.now());
  }, [activity]);
  useEffect(() => {
    const timer = window.setInterval(() => setNow(performance.now()), 1000);
    return () => window.clearInterval(timer);
  }, []);
  const current = activity ?? {
    phase: 'waiting',
    done: 0,
    total: 0,
    elapsedMs: waitingAt - startedAt,
    phaseMs: 0,
    receivedAt: waitingAt,
  };
  const since = Math.max(0, now - current.receivedAt);
  const elapsed = Math.floor((current.elapsedMs + since) / 1000);
  const phaseElapsed = Math.floor((current.phaseMs + since) / 1000);
  const label = denseActivityText(current, ko);
  const progress = denseActivityProgress(current);
  return (
    <div className="sfm-dense-activity" data-phase={current.phase} role="status">
      <ol className="sfm-dense-steps" aria-label={ko ? 'Dense 작업 단계' : 'Dense work stages'}>
        {STEPS.map((step) => (
          <li
            key={step.words[1]}
            aria-current={step.phases.includes(current.phase) ? 'step' : undefined}
          >
            {step.words[ko ? 0 : 1]}
          </li>
        ))}
      </ol>
      <strong>{label}</strong>
      <p>{denseActivityDescription(current.phase, ko)}</p>
      <WorkProgress label={label} percent={progress?.percent} />
      <span className="sfm-dense-count" aria-live="off">
        {progress
          ? (progress.submitted
              ? ko
                ? '현재 단계 작업 전달'
                : 'Current step queued'
              : ko
                ? '처리 완료'
                : 'Completed') + ` ${progress.percent}%`
          : ko
            ? '처리 중'
            : 'Working'}
        {(current.points ?? 0) > 0 &&
          (ko
            ? ` · 추출 ${current.points!.toLocaleString()}점`
            : ` · ${current.points!.toLocaleString()} points extracted`)}
      </span>
      <span className="sfm-dense-time" aria-live="off" data-phase-seconds={phaseElapsed}>
        {ko
          ? `이 단계 ${phaseElapsed}초 · Dense 작업 ${elapsed}초`
          : `This step ${phaseElapsed} s · Dense work ${elapsed} s`}
      </span>
    </div>
  );
}
