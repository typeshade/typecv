# Compiler changes this repository has handled

A change to the compiler that changes what TypeCV or its docs describe is agreed first as a
proposal in the compiler's `changes/` directory (the compiler's `changes/README.md` explains the
process). Each proposal lists, under `downstream`, the work it will owe the repositories it names.

When a pull request moves the compiler pin (`typeshade` in `package.json`) past a proposal that
names `typecv`, `scripts/downstream-impact.ts` of the compiler
(`bun <typeshade>/scripts/downstream-impact.ts --repo typecv`) lists the proposal until its work is
done on that branch and its id is recorded below. Record one list item per proposal: the id first,
then the pull request that did the work.

- 0071, pull request #45: `ShadeDevice` is built on `createDevice()` of `typeshade/runtime`. The
  pool, the single-device guard, the limits cast, `lost` and the `configure()` reset are gone. The
  option `poolSmallBufferBytes` is removed, and `workingBytes` bounds the active arrays (the owner
  decided so on 2026-10-19). The proposal names `typecv` in its expected work, not under
  `downstream`; the registration of `typecv` as a downstream repository is typeshade/typeshade#681.
- 0073, pull request #47: `ShadeDevice.open({ fp64Flavor })` picks the flavor of the emulated
  double, `'auto'` by default (`recommendFp64Flavor` on the adapter's info and the user agent:
  `'integer'` on an Apple GPU, `'float'` elsewhere), and `device.fp64Flavor` reports it. The lab
  takes `?fp64=float|integer` and records `options.fp64Flavor` (run record v3). The pin moves from
  `be90a21` to `107be2b`. The proposal names `typecv` in its expected work, not under `downstream`.
