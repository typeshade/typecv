/**
 * The run record of the SfM lab (`docs/lab-run-records.md`): the barrel of `run-record/`.
 * `schema.ts` holds the format, its limits and its types, `validate.ts` the validator,
 * `assemble.ts` the builders of its fields and `summary.ts` the lines the page shows. The page,
 * the workers, the collector (`website/worker.mjs`), the scripts and the tests import this path.
 * The page builds a whole record with `run-record/page.ts`.
 */
export * from './run-record/schema.ts';
export { runRecordProblem } from './run-record/validate.ts';
export * from './run-record/assemble.ts';
export * from './run-record/summary.ts';
