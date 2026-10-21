/**
 * The stage list of a multi-video run (`docs/multi-video.md`, Amendment 8). The run takes minutes,
 * so the page shows where it is: the stages in order, the one that runs, its counts and the time
 * of each stage. The functions here hold no state of the page: a board goes in and a board comes
 * out, so a unit test reads them without a browser.
 */

export type StageId =
  'frames' | 'candidates' | 'matching' | 'verification' | 'components' | 'reconstruction';

export const STAGE_IDS: readonly StageId[] = [
  'frames',
  'candidates',
  'matching',
  'verification',
  'components',
  'reconstruction',
];

/** The words of the page, English first and Korean second. */
export const STAGE_WORDS: Record<StageId, [en: string, ko: string]> = {
  frames: ['Frames and descriptors', '특징점 찾는 중…'],
  candidates: ['Candidate pairs', '겹칠 영상 짝 고르는 중…'],
  matching: ['Matching', '점 맞추는 중…'],
  verification: ['Pair verification', '맞는 점 확인하는 중…'],
  components: ['Components', '묶음 나누는 중…'],
  reconstruction: ['Reconstruction', '3D 복원 중…'],
};

/** The unit of the count of a stage, English and Korean. */
const UNIT_WORDS: Record<StageId, [en: string, ko: string]> = {
  frames: ['frames', '프레임'],
  candidates: ['pairs', '쌍'],
  matching: ['pairs', '쌍'],
  verification: ['pairs', '쌍'],
  components: ['components', '개'],
  reconstruction: ['views registered', '장 등록'],
};

export interface StageRow {
  readonly id: StageId;
  readonly state: 'pending' | 'running' | 'done';
  readonly done: number;
  /** 0 when the total is not known yet. */
  readonly total: number;
  /** The time the stage started, in the clock of the caller. */
  readonly startedAt?: number;
  /** The time of the stage so far, or the whole time of a finished stage. */
  readonly elapsedMs: number;
  /** Text after the count, such as the keyframes found. */
  readonly detail?: string;
}

export interface StageBoard {
  readonly rows: readonly StageRow[];
  readonly startedAt: number;
  /** The time of the last event from the worker, kernel progress included. */
  readonly lastEventAt: number;
}

export interface StageEvent {
  readonly stage: StageId;
  readonly done?: number;
  readonly total?: number;
  readonly detail?: string;
  /** The stage is complete with this event. */
  readonly final?: boolean;
}

/** A board with every stage pending. `now` is the start of the run. */
export function newBoard(now: number): StageBoard {
  return {
    rows: STAGE_IDS.map((id) => ({ id, state: 'pending', done: 0, total: 0, elapsedMs: 0 })),
    startedAt: now,
    lastEventAt: now,
  };
}

/**
 * Apply one event. The stages before the event's stage are done. A stage after it stays pending.
 * An event for a stage that is done changes nothing but `lastEventAt`.
 */
export function advance(board: StageBoard, event: StageEvent, now: number): StageBoard {
  const at = STAGE_IDS.indexOf(event.stage);
  const rows = board.rows.map((row, i) => {
    if (i < at && row.state !== 'done')
      return {
        ...row,
        state: 'done' as const,
        done: row.total || row.done,
        startedAt: row.startedAt ?? now,
        elapsedMs: now - (row.startedAt ?? now),
      };
    if (i !== at || row.state === 'done') return row;
    const started = row.startedAt ?? now;
    const total = event.total ?? row.total;
    const done = event.done ?? row.done;
    const final = event.final === true;
    return {
      ...row,
      state: final ? ('done' as const) : ('running' as const),
      done: final && total ? total : done,
      total,
      startedAt: started,
      elapsedMs: now - started,
      detail: event.detail ?? row.detail,
    };
  });
  return { ...board, rows, lastEventAt: now };
}

/** Note a sign of life from the worker (kernel progress) without a change of stage. */
export function touch(board: StageBoard, now: number): StageBoard {
  return { ...board, lastEventAt: now };
}

/** Update the elapsed time of the running stage to `now`. */
export function tick(board: StageBoard, now: number): StageBoard {
  return {
    ...board,
    rows: board.rows.map((row) =>
      row.state === 'running' ? { ...row, elapsedMs: now - (row.startedAt ?? now) } : row,
    ),
  };
}

/** The end of the run: every stage is done. */
export function finish(board: StageBoard, now: number): StageBoard {
  return {
    ...board,
    lastEventAt: now,
    rows: board.rows.map((row) =>
      row.state === 'done'
        ? row
        : {
            ...row,
            state: 'done' as const,
            done: row.total || row.done,
            elapsedMs: row.startedAt === undefined ? 0 : now - row.startedAt,
          },
    ),
  };
}

/** The stage that runs, or the last one that is done, or the first. */
export function currentRow(board: StageBoard): StageRow {
  return (
    board.rows.find((r) => r.state === 'running') ??
    [...board.rows].reverse().find((r) => r.state === 'done') ??
    board.rows[0]!
  );
}

/** No sign of life from the worker for this long is a stall (ms). */
export const STALL_MS = 30000;

/** True when no kernel progress or event has arrived for `limitMs`. */
export function stalled(board: StageBoard, now: number, limitMs = STALL_MS): boolean {
  return now - board.lastEventAt >= limitMs;
}

/** `2 min 5 s`, `41 s`, `0.8 s`. */
export function formatElapsed(ms: number, ko = false): string {
  const seconds = Math.max(0, ms) / 1000;
  if (seconds < 10) return seconds.toFixed(1) + (ko ? '초' : ' s');
  const whole = Math.floor(seconds);
  if (whole < 60) return whole + (ko ? '초' : ' s');
  const minutes = Math.floor(whole / 60),
    rest = whole % 60;
  return ko ? `${minutes}분 ${rest}초` : `${minutes} min ${rest} s`;
}

/** The count of a stage: `312 / 908 pairs`, or `312 pairs` when the total is not known. */
export function stageCount(row: StageRow, ko = false): string {
  const unit = UNIT_WORDS[row.id][ko ? 1 : 0];
  const n = row.total ? `${row.done} / ${row.total}` : String(row.done);
  const text = `${n} ${unit}`;
  return row.detail ? `${text} · ${row.detail}` : text;
}

/**
 * The status line, rebuilt at least once a second while work runs: the running stage, its count,
 * its time and the time of the run. `warning` follows when the worker was silent for too long.
 */
export function statusLine(board: StageBoard, now: number, ko = false): string {
  const row = currentRow(board);
  const name = STAGE_WORDS[row.id][ko ? 1 : 0];
  const parts = [
    name + (row.state === 'running' && row.id !== 'frames' && !ko ? '…' : ''),
    row.state === 'pending' ? '' : stageCount(row, ko),
    formatElapsed(row.elapsedMs, ko),
    (ko ? '전체 ' : 'total ') + formatElapsed(now - board.startedAt, ko),
  ].filter(Boolean);
  let line = parts.join(' · ');
  if (stalled(board, now)) line += ' · ' + stallWarning(now - board.lastEventAt, ko);
  return line;
}

export function stallWarning(silentMs: number, ko = false): string {
  const t = formatElapsed(silentMs, ko);
  return ko
    ? `${t}째 GPU 응답이 없습니다. GPU가 바쁘거나 멈췄을 수 있습니다.`
    : `No progress from the GPU for ${t}. The GPU may be busy or stopped.`;
}
