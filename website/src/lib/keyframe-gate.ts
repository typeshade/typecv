/**
 * The keyframe gate of the lab's video path: which decoded frames the SfM keeps.
 *
 * Two gates exist, chosen by the page's `?gate=phase|legacy` and named in the run record
 * (`options.keyframeGate`):
 *
 * - `legacy` decides after SIFT and matching: a frame is skipped when the median displacement of
 *   its matches from the last keyframe is under 8 px (at 640 px) and under one second has passed
 *   (`LAB_OPTIONS.keyframeMinMotionPx` and `keyframeMaxGapSeconds` of `lib/lab-options.ts`).
 * - `phase` decides before SIFT, from the global translation between the last keyframe and the
 *   candidate that phase correlation of a 128 × 128 grayscale thumbnail gives
 *   (`lib/phase-correlation.ts`). A skipped frame costs no feature extraction and no matching.
 *
 * The phase gate keeps the guarantees of the legacy gate and adds a minimum spacing:
 *
 * 1. The first frame is a keyframe, and so is the last frame of the video (`last`).
 * 2. A frame is a keyframe when `maxGapSeconds` have passed since the last keyframe.
 * 3. A frame is skipped when less than `minGapSeconds` have passed.
 * 4. Otherwise a frame is a keyframe when its translation is at least `minShift` of the frame
 *    (the length of `(dx / width, dy / height)`), and skipped when it is smaller.
 * 5. When the correlation peak is weak (`minRatio`, a ratio of 20), the translation says nothing: the frame goes
 *    on to SIFT and the legacy rule decides.
 *
 * The SfM worker also applies two rules of its own, which this module names: the legacy rule
 * (`legacyGateSkips`) and the track gate of `?kltGate=1` (`trackGateDecision`).
 *
 * This module has no DOM and no worker globals, so a unit test runs it in Node.
 */
import {
  PHASE_SIZE,
  grayThumbnail,
  phaseCorrelate,
  spectrumOf,
  type PhaseShift,
  type Spectrum,
} from './phase-correlation.ts';

export type KeyframeGateMode = 'phase' | 'legacy';

/** The default gate, and the one a query without `?gate=` gets. */
export const DEFAULT_KEYFRAME_GATE: KeyframeGateMode = 'phase';

/** The gate from the page's query string (`?gate=phase|legacy`), or the default. */
export function keyframeGateMode(
  search = '',
  fallback: KeyframeGateMode = DEFAULT_KEYFRAME_GATE,
): KeyframeGateMode {
  const value = new URLSearchParams(search).get('gate');
  return value === 'phase' || value === 'legacy' ? value : fallback;
}

export type PhaseGateOptions = {
  /** The thumbnail side in pixels (a power of two). */
  size: number;
  /** The translation, as a fraction of the frame, that makes a frame a keyframe. */
  minShift: number;
  /** The correlation peak over the mean correlation below which the estimate is not trusted. */
  minRatio: number;
  /** No keyframe sooner than this after the last one, in seconds. */
  minGapSeconds: number;
  /** A keyframe after this long at the latest, in seconds. */
  maxGapSeconds: number;
};

export const PHASE_GATE_OPTIONS: PhaseGateOptions = {
  size: PHASE_SIZE,
  minShift: 0.04,
  minRatio: 20,
  minGapSeconds: 0.2,
  maxGapSeconds: 1,
};

export type GateDecision =
  | { action: 'keep'; reason: string; shift?: PhaseShift; ms: number }
  | { action: 'skip'; reason: string; shift?: PhaseShift; ms: number }
  /** No decision: the frame goes on to SIFT and the legacy rule decides. */
  | { action: 'fallback'; reason: string; shift?: PhaseShift; ms: number };

export class PhaseGate {
  readonly options: PhaseGateOptions;
  private reference?: { spectrum: Spectrum; time: number };
  private candidate?: { spectrum: Spectrum; time: number };
  /** Counters of the run, for the run record and the measurement. */
  stats = { estimated: 0, estimateMs: 0, fallbacks: 0, totalMs: 0 };

  constructor(options: Partial<PhaseGateOptions> = {}) {
    this.options = { ...PHASE_GATE_OPTIONS, ...options };
  }

  /**
   * Decide on a frame of `width × height` RGBA pixels at `time`. `last` marks the final frame of
   * the video. The frame stays the candidate until `accept()` makes it the reference.
   */
  decide(
    rgba: Uint8ClampedArray | Uint8Array,
    width: number,
    height: number,
    time: number,
    last = false,
  ): GateDecision {
    const started = performance.now();
    const o = this.options;
    const spectrum = spectrumOf(grayThumbnail(rgba, width, height, o.size), o.size);
    this.candidate = { spectrum, time };
    const done = (action: GateDecision['action'], reason: string, shift?: PhaseShift) => {
      const ms = performance.now() - started;
      this.stats.totalMs += ms;
      return { action, reason, shift, ms } as GateDecision;
    };
    const reference = this.reference;
    if (!reference) return done('keep', 'first frame');
    const gap = time - reference.time;
    if (last) return done('keep', 'last frame');
    if (gap >= o.maxGapSeconds) return done('keep', 'maximum gap');
    if (gap < o.minGapSeconds) return done('skip', 'minimum gap');
    const shift = phaseCorrelate(reference.spectrum, spectrum);
    this.stats.estimated++;
    this.stats.estimateMs += performance.now() - started;
    if (!(shift.ratio >= o.minRatio)) {
      this.stats.fallbacks++;
      return done('fallback', 'weak correlation peak', shift);
    }
    return Math.hypot(shift.fx, shift.fy) >= o.minShift
      ? done('keep', 'moved', shift)
      : done('skip', 'barely moved', shift);
  }

  /** The last decided frame became a keyframe: it is the reference of the next frames. */
  accept(): void {
    if (this.candidate) this.reference = this.candidate;
    this.candidate = undefined;
  }

  /** The last decided frame was skipped. */
  discard(): void {
    this.candidate = undefined;
  }
}

/**
 * The analysis width at which `keyframeMinMotionPx` holds. A frame `width` pixels wide scales it
 * by `width / MOTION_REFERENCE_WIDTH`: 8 px at 640 px is 16 px at 1280 px.
 */
export const MOTION_REFERENCE_WIDTH = 640;

/** The motion threshold in pixels for a frame `width` pixels wide (`minMotionPx` at 640 px). */
export function scaledMotionThresholdPx(minMotionPx: number, width: number): number {
  return (minMotionPx * width) / MOTION_REFERENCE_WIDTH;
}

/** The legacy rule judges a pair with at least this many match entries (8 matches). */
export const LEGACY_GATE_MIN_MATCH_ENTRIES = 16;

/** The worker's keyframe numbers that the legacy and track rules read (`LAB_OPTIONS`). */
export type KeyframeMotionOptions = {
  /** Pixels at 640 px. */
  readonly keyframeMinMotionPx: number;
  /** Seconds. */
  readonly keyframeMaxGapSeconds: number;
};

/**
 * The legacy rule (`?gate=legacy`, or a weak peak of the phase gate): true when the frame is not a
 * keyframe. Three conditions hold together. The pair with the last keyframe has at least
 * `LEGACY_GATE_MIN_MATCH_ENTRIES` match entries. The median motion of its matches is under the
 * scaled threshold. Less than `keyframeMaxGapSeconds` have passed since the last keyframe.
 */
export function legacyGateSkips(
  matchEntries: number,
  motionPx: number,
  gapSeconds: number,
  width: number,
  options: KeyframeMotionOptions,
): boolean {
  return (
    matchEntries >= LEGACY_GATE_MIN_MATCH_ENTRIES &&
    motionPx < scaledMotionThresholdPx(options.keyframeMinMotionPx, width) &&
    gapSeconds < options.keyframeMaxGapSeconds
  );
}

/** The decision of the track gate: skip the frame, keep it, or leave it to the legacy rule. */
export type TrackGateDecision = 'skip' | 'keep' | 'undecided';

/**
 * The track gate (`?kltGate=1`, `docs/klt-tracking.md` 3.6), from the median displacement of the
 * tracks since the last keyframe (`medianTrackDisplacement` of `klt-join.ts`). A median that is
 * not finite (too few common tracks) decides nothing. Under the scaled threshold, and with less
 * than `keyframeMaxGapSeconds` since the last keyframe, the frame is skipped. Otherwise it is kept.
 */
export function trackGateDecision(
  medianPx: number,
  gapSeconds: number,
  width: number,
  options: KeyframeMotionOptions,
): TrackGateDecision {
  if (!Number.isFinite(medianPx)) return 'undecided';
  return medianPx < scaledMotionThresholdPx(options.keyframeMinMotionPx, width) &&
    gapSeconds < options.keyframeMaxGapSeconds
    ? 'skip'
    : 'keep';
}
