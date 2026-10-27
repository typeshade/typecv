/** Declarations for the comparison of a hardware-check record, so that its unit test typechecks. */
import type { HardwareCheckRecord } from '../website/src/lib/hardware-check/record.ts';
import type { RunRecord } from '../website/src/lib/run-record.ts';

export declare const DEFAULT_BASELINE: string;
export type Verdict = 'PASS' | 'FAIL' | 'WARN' | 'INFO' | 'SKIP' | 'n.a.';
export interface ComparisonRow {
  readonly check: string;
  readonly expected: string;
  readonly observed: string;
  readonly verdict: Verdict;
}
/** The CPU-tier digests of Node: the `cases` of `tests/digests/hardware-check.json`. */
export type Baseline = Readonly<Record<string, { readonly input?: string; readonly cpu?: string }>>;
export declare function compareHardwareCheck(
  record: HardwareCheckRecord & { readonly server?: { readonly id: string } },
  options?: {
    readonly baseline?: Baseline;
    readonly against?: HardwareCheckRecord & { readonly server?: { readonly id: string } };
    readonly benchmarks?: readonly (RunRecord & { readonly server?: { readonly id: string } })[];
  },
): ComparisonRow[];
export declare function verdict(rows: readonly ComparisonRow[]): 0 | 1;
export declare function markdown(rows: readonly ComparisonRow[]): string;
