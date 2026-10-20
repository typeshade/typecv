/**
 * The state of the triangulation probe of `/lab/probe/` between two page loads. On D3D12 a lost
 * device takes the adapter with it for the page, so the probe saves this state in
 * `sessionStorage` after every step, reloads the page when no new device opens, and goes on from
 * the saved plan (`docs/lab-run-records.md`, "Probe").
 */
import type { ProbeOverride, ProbeRange, ProbeRecord, ProbeStep } from './probe-record';

/** One task of the plan. A `narrow` and a `halves-done` task decide; the others run a step. */
export type ProbeTask =
  | { readonly kind: 'variant'; readonly index: number }
  /** A range that lost the device; `inFlight` is the chunk in flight at that loss. */
  | { readonly kind: 'narrow'; readonly range: ProbeRange; readonly inFlight: ProbeRange | null }
  /** Run the chunk in flight of `parent` alone. */
  | { readonly kind: 'chunk'; readonly parent: ProbeRange; readonly test: ProbeRange }
  /** Run one half of `parent`. */
  | { readonly kind: 'half'; readonly parent: ProbeRange; readonly half: ProbeRange }
  /** After both halves of `parent`: narrow each half that lost, in order. */
  | {
      readonly kind: 'halves-done';
      readonly parent: ProbeRange;
      readonly lost: { readonly range: ProbeRange; readonly inFlight: ProbeRange | null }[];
    };

export interface ProbeState {
  readonly version: 1;
  readonly startedAt: string;
  range: ProbeRange;
  adapter: ProbeRecord['adapter'];
  steps: ProbeStep[];
  plan: ProbeTask[];
  bisection: {
    override: ProbeOverride;
    hanging: number[];
    minimal: ProbeRange[];
    unresolved: ProbeRange[];
  } | null;
  aborted: string | null;
  /** The page reloads so far, and those in a row that opened no device. */
  reloads: number;
  failedReloads: number;
  /** Test hook (`?probe-fail=`): the next open fails once, after a simulated loss. */
  simulatedOpenFailure: boolean;
}

/** What the page keeps in `sessionStorage`: the worker's state, the page's consent and log. */
export interface SavedProbe {
  readonly href: string;
  readonly consent: boolean;
  readonly lines: readonly string[];
  readonly state: ProbeState;
}

export const PROBE_STORAGE_KEY = 'typecv-triangulation-probe';
/** The most reloads of one probe; past it the probe aborts with `resume`. */
export const PROBE_MAX_RELOADS = 40;
/** The most reloads in a row that open no device before the probe aborts. */
export const PROBE_MAX_FAILED_RELOADS = 3;
/** The most log lines kept across reloads. */
export const PROBE_MAX_LINES = 400;

export function loadProbe(): SavedProbe | null {
  try {
    const text = sessionStorage.getItem(PROBE_STORAGE_KEY);
    const saved = text ? (JSON.parse(text) as SavedProbe) : null;
    return saved?.state?.version === 1 ? saved : null;
  } catch {
    return null;
  }
}
export function saveProbe(saved: SavedProbe): boolean {
  try {
    sessionStorage.setItem(
      PROBE_STORAGE_KEY,
      JSON.stringify({ ...saved, lines: saved.lines.slice(-PROBE_MAX_LINES) }),
    );
    return true;
  } catch {
    return false;
  }
}
export function clearProbe() {
  try {
    sessionStorage.removeItem(PROBE_STORAGE_KEY);
  } catch {
    /* no storage: nothing to clear */
  }
}
