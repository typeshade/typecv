/**
 * @internal A hint for submitted PCG work, never a convergence rule or iteration budget.
 * The solver reports completed active iterations in an existing checkpoint. Later LM solves
 * start near that count and keep going normally if needed. A miss disables the next hint.
 */
export class BundleCgSchedule {
  private previous?: number;
  private firstLimit?: number;
  private reducedFirst = false;
  private missed = false;
  private readonly enabled: boolean;

  constructor(enabled: boolean) {
    this.enabled = enabled;
  }

  begin(): void {
    this.firstLimit =
      this.previous === undefined
        ? undefined
        : this.previous === 0
          ? 1
          : this.previous + Math.max(4, Math.ceil(this.previous / 4));
    this.previous = undefined;
    this.reducedFirst = false;
    this.missed = false;
  }

  next(proposed: number, submitted: number): number {
    if (!this.enabled || submitted !== 0 || this.firstLimit === undefined) return proposed;
    const result = Math.min(proposed, this.firstLimit);
    this.reducedFirst = result < proposed;
    return result;
  }

  /** Zero means no completed solve; success encodes active iterations plus one. */
  observe(encoded: number | undefined, cgActive: boolean): void {
    if (!this.enabled) return;
    if (cgActive && this.reducedFirst) {
      this.missed = true;
      this.reducedFirst = false;
    }
    if (
      !cgActive &&
      !this.missed &&
      encoded !== undefined &&
      Number.isSafeInteger(encoded) &&
      encoded >= 1 &&
      encoded <= 257
    )
      this.previous = encoded - 1;
  }
}
