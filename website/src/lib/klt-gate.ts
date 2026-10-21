/**
 * The lab gate of `docs/klt-tracking.md` 7.4 and 7.5: two run records of one input on one build,
 * `?klt=0` and `?klt=1`, held against the table of 7.4. The functions take parsed run records
 * (the JSON of `RunRecord`) so that a test runs them without a browser, and
 * `scripts/klt-lab-gate.mjs` runs the lab and prints the rows.
 */

/** The numbers of one run that the gate reads. */
export interface KltGateRun {
  registeredViews: number;
  landmarks: number;
  observations: number;
  meanReprojectionErrorPx: number;
  /** features + matches + guidedMatches + kltPush + kltJoin, in ms (the stages that KLT can move). */
  frontMs: number;
  /** The longest chunk of any key in ms, and the time budget of the device. */
  longestChunkMs: number;
  /** The longest chunk of the keys that begin with `klt`. */
  kltChunkMs: number;
  budgetMs: number;
  totalMs: number;
  /** The tracks alive at the end, and the tracks ended by reason (`KLT_END` order). */
  klt?: { alive: number; ended: number[]; meanLived: number };
}

/** The margins of the gate. The first run set them (Step 6 record). */
export interface KltGateMargins {
  /** Observations per landmark must exceed the `?klt=0` value by this many. */
  readonly trackLengthGain: number;
  /** The mean reprojection error may exceed the `?klt=0` value by this many pixels. */
  readonly errorPx: number;
  /** The landmarks may not fall below this share of the `?klt=0` value (amendment 35 of the Step 6 record). */
  readonly landmarkShare: number;
}

export const KLT_GATE_MARGINS: KltGateMargins = {
  trackLengthGain: 0.05,
  errorPx: 0.05,
  landmarkShare: 0.9,
};

export interface KltGateRow {
  readonly name: string;
  readonly off: string;
  readonly on: string;
  /** null for a row that is recorded and has no gate. */
  readonly pass: boolean | null;
  readonly rule: string;
}

const STAGE_KEYS = ['features', 'matches', 'guidedMatches', 'kltPush', 'kltJoin'] as const;

/** Read the numbers of a run record (`typecv-run-record-v5`). */
export function kltGateRun(record: any): KltGateRun {
  const stages = record?.timing?.stages ?? {},
    keyed = Object.entries(record?.timing?.chunks ?? {}) as [string, { longestMs?: number }][],
    chunks = keyed.map(([, c]) => c),
    result = record?.result ?? {};
  return {
    registeredViews: result.registeredViews,
    landmarks: result.landmarks,
    observations: result.observations,
    meanReprojectionErrorPx: result.meanReprojectionErrorPx,
    frontMs: STAGE_KEYS.reduce((sum, key) => sum + (stages[key]?.elapsedMs ?? 0), 0),
    longestChunkMs: Math.max(0, ...chunks.map((c) => c.longestMs ?? 0)),
    kltChunkMs: Math.max(
      0,
      ...keyed.filter(([key]) => key.startsWith('klt')).map(([, c]) => c.longestMs ?? 0),
    ),
    budgetMs: record?.timing?.budgetMs ?? Infinity,
    totalMs: record?.timing?.totalMs ?? 0,
    ...(result.klt
      ? {
          klt: {
            alive: result.klt.alive,
            ended: result.klt.ended,
            meanLived: result.klt.meanLived,
          },
        }
      : {}),
  };
}

const perLandmark = (run: KltGateRun) => run.observations / Math.max(1, run.landmarks);

/** The rows of table 7.4 for one input. `off` is the `?klt=0` run and `on` the `?klt=1` run. */
export function kltGateRows(
  off: KltGateRun,
  on: KltGateRun,
  margins: KltGateMargins = KLT_GATE_MARGINS,
): KltGateRow[] {
  const f = (v: number, digits = 3) => v.toFixed(digits),
    s = (ms: number) => f(ms / 1000, 1) + ' s';
  return [
    {
      name: 'Registered views',
      off: String(off.registeredViews),
      on: String(on.registeredViews),
      pass: on.registeredViews >= off.registeredViews,
      rule: 'not less than klt=0',
    },
    {
      name: 'Mean track length (observations per landmark)',
      off: f(perLandmark(off)),
      on: f(perLandmark(on)),
      pass: perLandmark(on) > perLandmark(off) + margins.trackLengthGain,
      rule: `greater than klt=0 by ${margins.trackLengthGain}`,
    },
    {
      name: 'Landmarks and observations',
      off: `${off.landmarks} / ${off.observations}`,
      on: `${on.landmarks} / ${on.observations}`,
      pass:
        perLandmark(on) > perLandmark(off) && on.landmarks >= margins.landmarkShare * off.landmarks,
      rule: `observations per landmark up, landmarks not below ${margins.landmarkShare} of klt=0`,
    },
    {
      name: 'Mean reprojection error (px)',
      off: f(off.meanReprojectionErrorPx, 4),
      on: f(on.meanReprojectionErrorPx, 4),
      pass: on.meanReprojectionErrorPx <= off.meanReprojectionErrorPx + margins.errorPx,
      rule: `not more than klt=0 plus ${margins.errorPx}`,
    },
    {
      name: 'features + matching + KLT stages',
      off: s(off.frontMs),
      on: s(on.frontMs),
      pass: on.frontMs < off.frontMs,
      rule: 'less than klt=0',
    },
    {
      name: 'Longest chunk (ms)',
      off: f(off.longestChunkMs, 0),
      on: `${f(on.longestChunkMs, 0)}, KLT keys ${f(on.kltChunkMs, 0)}`,
      pass: on.kltChunkMs <= on.budgetMs,
      rule: 'the KLT keys not above the time budget',
    },
    {
      name: 'Tracks alive, ended by reason',
      off: '-',
      on: on.klt ? `${on.klt.alive} alive, ended ${on.klt.ended.join('/')}` : '-',
      pass: null,
      rule: 'recorded, no gate',
    },
  ];
}

/** True when every gated row passes. */
export const kltGatePasses = (rows: readonly KltGateRow[]) =>
  rows.every((row) => row.pass !== false);
