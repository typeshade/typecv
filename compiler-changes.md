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
- 0074, pull request #54: the time budget of pull request #49 moves into the compiler's device.
  `ShadeDevice.open({ timeBudgetMs })` passes the budget to `createDevice()`; a chunk step of
  `planSteps` runs as `device.fuse(body, { chunk: { key, units } })`, which the TypeShade device
  fences and times; `ShadeDevice.chunkSize()` and `chunkRates` are the TypeShade device's. The
  arithmetic `fitChunk` and `learnRate` and the local `ChunkRate` are removed (`ChunkRate` is
  re-exported from `typeshade/runtime`). The pin moves from `107be2b` to `41a140a`. The proposal
  names `typecv` under `downstream`.
- 0075, pull request #57: the batch loop of pair model generation dispatches indirectly. A
  `Uint32Array` resident `counts` holds the workgroup counts of a batch's prepare, row appends,
  null solve, finish and scoring; `planPairModelBatch` and `summarizePairModelWork` write them,
  zero past the adaptive budget or with no active model, and those entries dispatch with
  `{ indirect: counts, offset }`. The host reads nothing between the kernels of a batch, and reads
  the scheduling word (`snapshotPairModelBudget`) after every fourth batch only
  (`PROGRESS_INTERVAL`). A scoring the host cannot know to run is a chunk step with
  `timed: false`, so no empty plan teaches the time budget a rate. `ShadeDevice` wraps no
  workgroup argument; `ShadePlanReport` groups gain `indirect`. The pin moves from `41a140a` to
  `d70b539`. The proposal names `typecv` under `downstream`; at the pin its status is `accepted`,
  so `downstream-impact.ts` does not list it yet.
- 0079, pull request #164: nothing is owed at the pin. A hand-written kernel loop that reduces,
  searches or fills can move to the array method later, held to its current output. The pin
  moves from `e0bf150` to `3c6e964`, which also carries typeshade/typeshade#720 (the fix of
  typeshade/typeshade#717) and so removes the plugin `typecv-skip-resident-dry-run`.
- 0078, this pull request: no source migration is required. Existing stable compactions retain
  their current output. The compiler pin moves from `3c6e964` to `94203d3c` with 0081.
- 0080, this pull request: no source migration is required. Existing unique-label passes retain
  their current output. The new compiler supports kernel `Map` and `Set` parameters.
- 0081, this pull request: the compiler pin moves from `3c6e964` to `94203d3c`, the merge of
  typeshade/typeshade#733. No source migration is required. Named tables can adopt `Record`
  separately against their current output. Proposal 0081 deviation D1 remains open upstream.
- 0057, pull request #208: BA workgroup probes use `program.compute(..., { constants })` and
  check resolved shapes, cached pipelines and scalar CPU results. The compiler pin moves from
  `94203d3c` to `d36ac502` (merged typeshade/typeshade#743 and #744, including the
  typeshade/typeshade#742 correctness fix). It
  preserves symbolic fp64 workgroup-array extents. Width 128 remains in the numerical probe.
- 0084, pull request #208: BA camera assembly passes `{ constants }` to imported entries inside
  the existing device plan. It selects scalar widths 8/16/32/64 for native camera-row counts,
  and cooperative widths 32/64 for observation windows, with dispatch limits and the CPU/software
  baseline retained. No lab parameter is added. All eight measured variants also run through
  the imported-entry path against the CPU oracle, with selected report shapes and a single
  submission. The native integer-fp64 gate covers those variants and remains opt-in on software
  adapters. The compiler implementation is typeshade/typeshade#743; the accepted proposal is
  typeshade/typeshade#741. See `docs/ba-camera-assembly.md` for the policy and validation limits.
- 0083, pull request #208: `ShadeDevice.info` exposes the owner's frozen build and actual-device
  snapshot, including enabled features/limits and identity provenance. Occupancy selection reads
  that snapshot instead of requesting another adapter. An external device without identity
  stays unknown; automatic fp64 selection uses the user agent without probing an unrelated
  adapter. Tests cover supplied devices, unavailable identity, CPU metadata and disposal.
  The lab's separate preflight adapter query remains for buffer-budget sizing; it is not the
  owner's device information. The existing explicit sending consent and workload records are
  unchanged. The pin is `d36ac502`, after typeshade/typeshade#736.
- 0085, pull request #260: no source migration is required. TypeCV takes the defaults that give
  it the same device, flavor and plans. The registration-only `import 'typeshade/fuse'` of
  `src/device/gpu/device.ts` is removed: `device.fuse()` holds its recorder and loads the planner
  from its own chunk. `ShadeDevice.open()` passes `fp64Flavor` through, `'auto'` by default. The
  TypeShade device then selects the flavor from the device it opens or its creation adapter, and
  TypeCV requests no adapter of its own for it. The build and `scripts/kernel-faces.mjs` drop
  `fuse: true`, now the compiler's default; `dist/` is byte-identical without it. The site ships
  the planner and shared chunks of `dist/gpu` and `dist/internal` (`website/scripts/prepare.mjs`).
  Four things stay explicit. The one-tier `prefer` keeps WebGPU required and the CPU tier on
  request; the default order falls back to WebGL2 and the CPU tier. TypeCV's own `GPUDevice`
  serves `adapterLimits` and `gpuTimestamps: false`: 0085's `limits` and `features` options cannot
  express adapter-dependent limits or `gpuTimestamps: false`. `limits` are fixed numbers that the
  adapter is checked against, and a device TypeShade requests always adds `timestamp-query`
  (TypeShade `src/core/host-entry.ts:933-982` at `725b2969`). Every kernel call stays in a device
  plan: the reports, the kernel statistics and the chunk timing read the plan's report. The lab
  and the tests keep their explicit flavors. `tshc init` is not used: the build writes and
  removes the host views itself. No kernel calls `console`. The pin stays `725b2969`; the console
  variants and the Apple/Metal validation of 0085 remain open upstream.
