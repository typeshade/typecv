/**
 * Why a run has no dense cloud (`docs/multi-video.md`, Amendment 9 and its follow-up on the reason).
 * The reason comes from the run's `dense` block (`RunRecordDense`: `status`, `skipped`, `failure`)
 * or, when the run has no block, from the run option. The record has no field of its own for it.
 */
import type { RunRecordDense } from './run-record.ts';

/** The reason a multi-video run with the dense MVS had no component to run it on. */
export const DENSE_NO_COMPONENT = 'No component was eligible for the dense MVS';
/** The reason of a component with fewer than two registered views (`multi-video-job.ts`). */
export const DENSE_FEW_VIEWS = 'Fewer than two registered views';
/** The reason when no registered view kept its image (`dense-run.ts`). */
export const DENSE_NO_IMAGE = 'No registered view kept its image';

const KNOWN: Record<string, string> = {
  [DENSE_NO_COMPONENT]: '조건에 맞는 묶음이 없습니다',
  [DENSE_FEW_VIEWS]: '묶음의 등록된 시점이 2개 미만입니다',
  [DENSE_NO_IMAGE]: '등록된 시점의 이미지가 남아 있지 않습니다',
};

/** The longest reason the lab shows. */
const LONGEST = 160;

function clip(text: string): string {
  return text.length > LONGEST ? text.slice(0, LONGEST - 1) + '…' : text;
}

/**
 * The short reason a dense cloud is missing, or undefined when the run gave no reason.
 * `asked` is the run option at the start of the run: false for off, true for on, undefined when
 * the page does not know (a result loaded from files).
 */
export function denseReason(
  summary: RunRecordDense | undefined,
  asked: boolean | undefined,
  ko: boolean,
): string | undefined {
  if (!summary) {
    if (asked === false)
      return ko
        ? 'Dense MVS가 실행 설정에서 꺼져 있었습니다.'
        : 'Dense MVS was off in the run settings.';
    return undefined;
  }
  if (summary.status === 'failed') {
    const message = summary.failure?.message || '';
    return (ko ? 'Dense MVS 실패: ' : 'Dense MVS failed: ') + clip(message || 'no message');
  }
  if (summary.status === 'cancelled')
    return ko ? 'Dense MVS를 멈췄습니다.' : 'Dense MVS was cancelled.';
  if (summary.status === 'skipped') {
    const why = summary.skipped || 'no reason';
    return (
      (ko ? 'Dense MVS 건너뜀: ' : 'Dense MVS skipped: ') +
      (ko ? (KNOWN[why] ?? clip(why)) : clip(why))
    );
  }
  return undefined;
}

/** The tooltip of the Dense switch: the reason, or the generic hint without one. */
export function denseSwitchTitle(
  summary: RunRecordDense | undefined,
  asked: boolean | undefined,
  ko: boolean,
): string {
  return (
    denseReason(summary, asked, ko) ??
    (ko
      ? 'Dense 포인트 클라우드가 아직 없습니다. 실행에서 Dense MVS를 켜고 다시 실행해 주세요.'
      : 'No dense cloud. Turn on Dense MVS in the run settings and run.')
  );
}
