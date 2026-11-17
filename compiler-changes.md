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
- 0096, pull request #278: the pin moves from `725b2969` to `78a9060f`, which carries 0096. A
  kernel function's map entry no longer returns for a lane past the trip count: only a wholly idle
  workgroup returns, and a lane of the last workgroup runs iteration `n - 1` with every write
  masked. Probe S2 is renewed to hold that form; the old assertion read the workgroup's return as
  the lane's and still passed. `docs/gpu-sfm.md` restates the rule for kernel authors: it holds for
  an entry an author writes, and the compiler applies it to a kernel function. The owner decided on
  2026-11-02 that no map converts and `triangulateTracks` stays an entry in this pull request: each
  conversion is phase 10's step "C6 and C7" of `docs/library-structure.md`, with phase 9's gate
  and the hardware run of section 7, item 11, which stays open. The pin also carries 0103: the WebGL2 pass program passes the k-NN
  list as a module place, so probes K1 and K2 hold the new GLSL form.
- 0138, pull request #283 (stacked on #282, which moves the pin from `78a9060f` to `f149b403`):
  milestone 4 slice 1 only. The host code moves from `Resident` to `Buffer` (Amendments A10 and
  A11): `Buffer<f32>`, `Buffer<f64>` and `Buffer<u32>`, `set` in place of `write`, `dispose` in
  place of `destroy`, and typed-array reads. `ShadeDevice.read` returns the typed array of its
  `Buffer` by overload. The public declarations change type only: `GpuResidentFloatDescriptors`
  `descriptors`, and `GpuSiftGray` `buffer` and `take()`, are `Buffer<f32>` in place of
  `Resident<Float32Array>`. No name changes. A `Buffer` refuses `set` while a call's values are
  on the device (A11, Q1), so the probes that restore a device-written `Buffer` read it first.
  The library overwrote device-written arrays with `write`, which queued after the calls before
  it, and `set` refuses that. The owner kept the rule on 2026-11-17 ("staging 재설계"): no
  `Buffer` is both written by the device and set by the host. The KLT pair cache's reused slot
  and the KLT tracker's gray frame give the old gray array back and take a new one made with the
  frame. The Dense depth stage gives `stageF32` back, after `rgbaToGray` wrote it, and takes a
  new one made with the observation mask or the depth map. Each old array goes back first, so the
  pool can give it again and the active peak does not grow. No readback and no GPU wait is added.
  The owner accepted this design on 2026-11-17 ("반납 후 새 할당 수용"). If the new array cannot be
  made, the cache drops the lost slot's gradient and the tracker fails every later call.
  `ShadeDevice.writeF32` lost its one caller and is removed.
  The audit host passes its plain-object quartic problems as plain data (A11, Q4). The
  `fuse(fn)` graphs, the legacy `device.fuse` scope and the other milestones of 0138 are not
  migrated: TypeCV still uses `device.fuse()` and `scope`, which the pin keeps. Rule 3.2 of
  `docs/design-rules.md` names a `Buffer` for an array since #282 (the owner's decision of
  2026-11-17, "`Buffer`로 용어만 교체").
- 0108, pull request #283: nothing must change. The host views that the build generates (not
  committed) carry the JSDoc of the shader modules.
- 0109, pull request #283: nothing must change. The host code imports no class of a shader
  module.
- 0110, pull request #283: nothing must change.
- 0116, pull request #283: the host views spell the shader's own types. The host code
  type-checks at `f149b403` after the `Buffer` migration of 0138, with no annotation added for
  0116.
- 0118, pull request #283: nothing must change in what runs. TypeCV builds no `FuseReport`
  literal. Its report literals are its own `ShadePlanReport`, whose groups do not carry `tier`.
- 0119, pull request #283: no source change. Which integer reductions of TypeCV get the subgroup
  variant is not audited in this pull request; gap G2 of `docs/gpu-throughput.md` stays open
  until it is.
- 0121, pull request #283: no report literal must change: `ShadePlanReport` groups do not carry
  `dispatches`, and the hand-built groups of the tests are of that shape. The kernel statistics still
  count `workgroupShapes` and do not read a group's `dispatches` yet. That is a follow-up.
- 0122, open: the plugin `typecv-fuse-origin` of `vite.gpu.config.ts` stays. Its retirement,
  after the origins of the new pin are checked equal, is a follow-up pull request.
- 0126, pull request #283: seven kernel modules are pure aliases (`export * from` one target):
  `src/gpu/{bundle,joint-focal,pair-models,triangulation}.shade.ts`,
  `src/geometry/gpu/{numerics,projective}.shade.ts` and `src/sfm/bundle/gpu/numerics.shade.ts`.
  `npm run test:surface` holds each alias to its target's WGSL and GLSL at `f149b403`. No alias
  is extended.
- 0128, pull request #283: nothing must change. `ShadePlanReport` and the kernel statistics do
  not read the scratch counts, and no fixture holds a full report.
- 0129, open: no test of a failed producer and its dependent image plans is added yet. The test
  is a follow-up.
- 0130, open: the 224 `@builtin` parameter decorators of `src/` move to types with `tshc migrate`
  of a compiler at or after `f149b403`, in a follow-up pull request. The pin still accepts them.
  The 18 `@inout` parameters stay (0137).
- 0137, pull request #283: nothing must change. The 18 `@inout` parameters of `src/` (10, and the
  8 that #282 adds to `epipolar.shade.ts`) are the language's form.
