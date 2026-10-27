import { GeometryEstimationError } from '../../geometry/errors.ts';
import { ReconstructionOperationError } from './errors.ts';

/**
 * @internal The error rule of a driver: what it does when a request of the mapper throws
 * (docs/library-structure.md, section 3.3, `sfm/protocol`). The drivers keep the rules they had
 * before the rules were written here (section 4, rule 1):
 *
 * - `'pass'`: the error goes back into the mapper as it is. The CPU drivers use it:
 *   `reconstructIncremental`, `refineIncremental` and `reconstructIncrementalScene`.
 * - `'wrap'`: a `GeometryEstimationError`, a numerical rejection such as a failed PnP, goes back
 *   as it is. Any other error goes back as a `ReconstructionOperationError`. The mapper throws
 *   that error again, so the run ends. The device driver uses it: `reconstructIncrementalGpu`,
 *   `refineIncrementalGpu` and `IncrementalSfmSession`.
 * - `'end'`: no error goes back. The first error ends the run. The component driver
 *   (`reconstructComponent`) uses it. Section 7, item 5 lists this rule as a behaviour difference.
 */
export type DriverErrorRule = 'pass' | 'wrap' | 'end';

/** The error that goes back into the mapper under `rule`. Under `'end'` the error is thrown. */
function returnedError(error: unknown, rule: DriverErrorRule): unknown {
  if (rule === 'end') throw error;
  return rule === 'wrap' && !(error instanceof GeometryEstimationError)
    ? new ReconstructionOperationError(error)
    : error;
}

/**
 * @internal Drive a mapper to its result. Each request goes to `execute`, and the result of
 * `execute` goes back into the mapper. An error of `execute` follows `rule`.
 */
export function drive<Job, Result>(
  machine: Generator<Job, Result, unknown>,
  execute: (job: Job) => unknown,
  rule: DriverErrorRule,
): Result {
  let step = machine.next();
  while (!step.done) {
    let result: unknown;
    try {
      result = execute(step.value);
    } catch (error) {
      step = machine.throw(returnedError(error, rule));
      continue;
    }
    step = machine.next(result);
  }
  return step.value;
}

/** @internal `drive` with an asynchronous `execute`, for a device driver. */
export async function driveAsync<Job, Result>(
  machine: Generator<Job, Result, unknown>,
  execute: (job: Job) => Promise<unknown>,
  rule: DriverErrorRule,
): Promise<Result> {
  let step = machine.next();
  while (!step.done) {
    let result: unknown;
    try {
      result = await execute(step.value);
    } catch (error) {
      step = machine.throw(returnedError(error, rule));
      continue;
    }
    step = machine.next(result);
  }
  return step.value;
}
