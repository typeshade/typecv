# TypeCV library structure and the refactoring programme: proposal

Status: proposed, 2026-10-26. This document decides nothing until the owner accepts it. An
accepted version becomes the record that every refactoring pull request cites. It changes no
code and no behaviour. Labels: **Fact** (read in the code, a document or a measurement),
**Inference** (reasoned from facts), **Proposal** (a choice this document asks the owner to make).

Evidence: the three audit records of 2026-10-26, [`audits/2026-10-26-sparse-structure.md`](audits/2026-10-26-sparse-structure.md)
(the sparse pipeline on both tiers), [`audits/2026-10-26-lab-structure.md`](audits/2026-10-26-lab-structure.md)
(the lab's orchestration) and [`audits/2026-10-26-kernel-style.md`](audits/2026-10-26-kernel-style.md)
(how the kernels use the compiler). All three were read on `main` at `5e1c95c` with the compiler
pin `d36ac502`.

## 1. Why

The owner's verdict on 2026-10-26, in three parts:

1. The SfM code is written haphazardly. It is neither structured nor divided into parts.
2. The repository must match its name. TypeCV is a computer vision library, and the code is
   organized as if SfM were the whole of it.
3. The kernels use explicit `@compute` entries everywhere. TypeShade lifts a plain loop in a
   kernel function into a compute kernel by itself. Was the explicit style a choice, or a missed
   advantage?

The measurements agree with the verdict.

| Measurement (**Fact**, `main` at `5e1c95c`) | Value |
| --- | --- |
| `src/gpu` | 89 files, 36,737 lines; one flat directory for SIFT, ORB, KLT, CLAHE, SLIC, Hough, the mapper, MVS, TSDF and the device |
| `src/reconstruction` | 33 files, 12,472 lines |
| `website/src/lib` | 48 files, 9,978 lines |
| Largest files | `SfmLab.tsx` 5,441; `mvs.ts` 2,579; `SfmViewer.tsx` 2,404; `run-record.ts` 1,885; `mvs.shade.ts` 1,821; `sfm.worker.ts` 1,635; `mapper.ts` 1,448 |
| Functions over 590 lines | 5: `bundleAdjustJointFocalGpu` 989, `bundleAdjustGpuFixed` 846, `checkPairEstimation` 768, `expandObservationsGpu` 732, `incrementalSteps` 590 |
| Runtime import cycles | 3: `mapper.ts ↔ focal-search.ts`, `bundle-adjustment.ts ↔ joint-focal.ts`, `gpu/bundle.ts ↔ gpu/joint-focal.ts` |
| Two mapper state machines | `incrementalSteps` (590 lines) and `sceneSteps` (409 lines): 151 lines verbatim, the rest a near copy; `sceneSteps` lacks six features and no test sees the drift |
| Two GPU bundle-adjustment routes | 538 of 1,043 host lines and 896 of 1,125 kernel lines of the fixed route recur verbatim in the shared-focal route |
| The lab's six mapper values | written in 7 places; seed 17 in 8 places; LM stall 1e-9 in 4; `finalBundleIterations` has two library defaults (25 and 8) |
| Lab worker | `start` 243 lines with 23 positional parameters; `processFrame` 266 lines; 13 `post()` calls inside computation; policy rules (descriptor retention, pair minimums, keyframe rules) inline and untested |
| Public surfaces | `src/index.ts` exports geometry, two-view, tracks, BA and focal search; `src/gpu/index.ts` exports 68 names from one plane; the lab's multi-video path imports `src/gpu/internal.ts`; the website imports 7 library-internal modules by deep path |
| Kernel style | 30 `.shade.ts` files, 13,186 lines, 251 `@compute` entries: 56 single-invocation steps, 136 per-element maps, 29 cooperative (workgroup memory or barriers), 12 with atomics, 18 `...Oracle` CPU twins; 8 kernel functions, all in `epipolar.shade.ts`, which the SfM pipeline does not call; 0 array methods in kernels; every dispatch 1D; an entry call never fuses with another call in a plan |

**Inference.** No boundary separates an area of computer vision from another, a policy from its
numerics, or the oracle from the product path. A change in one place is a change in several.

## 2. What already stands

This proposal adds no new direction. It arranges the code under decisions the owner has made.

- `engine-v2.md` (accepted 2026-10-19): the engine is a GPU-resident program with f32 arithmetic
  and a scene state machine over indices; the CPU f64 path is the oracle; one npm package with one
  entry point per area over one shared device layer; the present layout binds nothing, area entry
  points come with the engine, and old paths stay as aliases until the next major version.
- `gpu-resident-scene.md` (accepted): the resident scene; `reconstruction-status.md` R03 is the
  work in flight that integrates `sceneSteps` with a device scene.
- `architecture.md`: shared data contracts (images, features, matches, cameras, observations)
  independent of any one pipeline; pipeline state stays in the pipeline.
- `design-rules.md`: Rule 1.1 (one answer on every tier), 1.2 (proved by tests, bit for bit where
  the operation has one answer), 1.4 (a fused plan gives the bits the calls give run apart), 2.1
  (the order of priorities), 3.1 to 3.3 (composition with the TypeShade family; a compiler gap
  becomes a TypeShade proposal, not a workaround). §4 (the `cv` namespace front end) is a
  proposal and §5 lists its open decisions; this document does not decide them.
- `roadmap.md`: the capability areas. They are the areas of section 3.1.

## 3. Target structure

### 3.1 Areas and entry points

**Proposal.** The source tree and the package entry points follow the roadmap's areas. SfM is one
consumer under reconstruction. Every operator that exists today moves to its area.

| Area (roadmap) | Directory | Entry point | What moves there today |
| --- | --- | --- | --- |
| Shared data and device execution | `src/device/`, `src/contracts/` | `typecv/device` | `gpu/device.ts`, `budget-plan.ts`, `readback.ts`, `scan.ts` + `scan.shade.ts`, `device-diagnostics.ts`, `rgba-read-view.ts`; the shared kernel helpers (`numerics.shade.ts`, `projective.shade.ts`); the image, feature, match, camera and observation contracts |
| Image processing | `src/image/` | `typecv/image` | `clahe`, gray conversion and pyramids now inside `features.shade.ts`, `sift.shade.ts` and `klt.shade.ts` |
| Image analysis | `src/analysis/` | `typecv/analysis` | `slic`, `hough` |
| Features and correspondence | `src/features/` | `typecv/features` | `sift`, `orb`, `orb-pattern`, `orb-reference`, `features` (corner and BRIEF), `tiled-features`, `kernels/float-matching`, Hamming matching, `global-descriptor*` |
| Geometric vision | `src/geometry/` | `typecv/geometry` | today's `src/geometry` and `src/sfm` (the oracle primitives: camera, pose, linear algebra, two-view, homography, P3P and PnP, triangulation, rotation, inverse depth, image pair); the GPU operators `pair-estimation`, `image-pair`, `two-view`, `homography`, `p3p`, `pnp`, `triangulation`, `observations`, `epipolar` and their kernels |
| Motion and video | `src/motion/` | `typecv/motion` | `klt`, `klt-tracker`, `klt-device`, `klt-pair-cache`; the keyframe gate and the decode schedule when they leave the website (section 3.4) |
| 3D vision: sparse reconstruction | `src/sfm/` | `typecv/sfm` | today's `src/reconstruction` and the GPU mapper, bundle, joint focal, inverse depth, view graph, multi-video and observation modules, arranged as section 3.3 says |
| 3D vision: dense reconstruction | `src/mvs/` | `typecv/mvs` | `mvs`, `mvs-views`, `mvs-residency`, `planar-depth*`, `tsdf*`, `cloud-clean`, `cloud-knn`, `cloud-voxel`, `dense-mesh*`, `reconstruction/tsdf.ts`, `dense-mesh.ts`, `cloud-clean.ts`, `dense-label.ts`, `marching-cubes-tables.ts`, `dense-score.ts` |
| Learned vision | — | — | nothing today (`roadmap.md`: planned) |
| Evaluation and interoperability | `scripts/`, `tests/` | — | as today; `scripts/record_analysis/` and the measurement scripts |

Rules of the layout (**Proposal**):

- A directory holds one area. A file holds one responsibility. A kernel file sits beside the host
  module that drives it.
- An area imports a lower area only: `contracts` and `device` at the bottom; `image`, `analysis`,
  `features`, `geometry`, `motion` above them; `sfm` and `mvs` at the top. No import cycle.
- The website and the scripts import entry points only, never a deep path. The internal entry
  `src/gpu/internal.ts` stays until the lab's orchestration has moved (section 3.4), then goes.
- `typecv` (the root) and `typecv/gpu` stay as barrels of the area entries until the next major
  version. Every moved file leaves a re-export at its old path until then.
- The area entry points go live when the owner says so (engine-v2: "introduced as the engine
  lands"). Until then the directories exist and the barrels export them.

### 3.2 Shared contracts and the device layer

**Fact.** `architecture.md` asks for data contracts (dimensions, channels, numeric type, layout,
camera conventions, index and ownership rules) that no one pipeline owns. **Fact.** Today a
camera record is packed by seven packers, a triangulation status is spelled in three places, and
a pair key in three spellings (audit sparse, 2.3 and 2.4).

**Proposal.** `src/contracts/` holds the types and the one encoder and decoder of each record that
crosses a boundary: image, feature set and descriptors, matches, camera and pose, observation,
scene (views, landmarks, observations), status enums, failure-reason table. Host and kernel code
read the layout constants from one place (the host view can import a kernel's exported
constants, audit sparse L3; a build check confirms it). `src/device/` stays what `engine-v2.md`
keeps: the device owner, plans, budgets, readback, scan.

### 3.3 The sparse pipeline (`src/sfm/`)

**Proposal** (audit sparse, section 7): one mapper policy as a state machine over indices, one
typed request protocol, two drivers of it (the host f64 oracle and the resident GPU scene), one
device layer.

| Module | Responsibility | Contract |
| --- | --- | --- |
| `sfm/model` | object model, copy, validation, `observationKey`, `pairKey`, content ids | pure, no tier |
| `sfm/config` | every default once; `resolveMapperControls`; session and focal controls; the lab preset | pure; frozen controls; throws `RangeError` |
| `sfm/tracks` | track union, rotation evidence, observation masks and paths | pure |
| `sfm/pairs` | pair verdict, seed score, pair seeds, verification order, pre-verified index | pure policy and `verifyPair` requests |
| `sfm/mapper` | initialization, ranking, registration gates and retries, weak registration, triangulation and continuation rules, filter rules, bundle schedule and window | a generator over `SceneRequest`; no numerics, no callbacks, no packing beyond small typed arrays |
| `sfm/protocol` | request and result types (a typed map), errors, `drive` (sync and async) with one error rule, the depth post-step | the only place of the error rule |
| `sfm/scene` | `HostScene`, layout constants shared with kernels, `HostSceneDriver` (oracle), `GpuSceneDriver` (R03) | executes one request; no policy |
| `sfm/bundle` | CPU solvers, bearing constraints, the problem plan and shared validation; the GPU host half once | pure numerics; tier-specific messages kept |
| `sfm/focal` | focal search, pair base, history seeds | a generator over the protocol |
| `sfm/depth` | inverse-depth tracker, depth BA, publication, record, sparse Schur | its own generator |
| `sfm/session` | window eviction, anchors, carried edges, archive, archive-solve plan | pure functions and one orchestrator |
| `sfm/multiview` | view graph, components, global descriptors (host part), multi-video | pure and a backend interface |

The mapping from today's files to these modules is the table in audit sparse, section 7.

### 3.4 The lab's orchestration

**Proposal** (audit lab, section 6): a worker of about 150 lines that only moves messages; the
pipeline stages as modules with one responsibility each, calling the GPU through an injected
port (the precedents are `JobApi` and `DenseRunContext`); typed events; one options schema for
the page, both workers, the record and the scripts; the run record split into schema, validator,
builder and summary behind a barrel.

| Module | Responsibility |
| --- | --- |
| `options` | the schema: key, flag, parser, default, where it applies, where the record keeps it |
| `schedule` | timestamps, fps, ranges, the one-frame look-ahead |
| `keyframe-gate` | the phase, track and legacy rules as one unit |
| `features` | the front end through the port, the feature minimum, colours |
| `matching` | candidates, descriptor retention, pair admission, guided matching |
| `klt` | tracker life, join, merge, verdict |
| `registration` | the driver over the session (the scene engine later) |
| `single-video-job`, `multi-video-job`, `dense-driver` | the three jobs over the stages |
| `ledger` | stage totals, GPU accounting, the record's measured fields |
| `lab-device` | open, reuse, loss, swap, for every worker |
| `lab-protocol` | typed page-to-worker messages |
| `lab-run` | the page's run controller |
| `run-record/` | schema, validate, assemble, summary, and the barrel the collector imports |

Where the stage modules live is the owner's decision (section 7, item 4): the modules that a
script needs (the gate, the matching policy, the options) belong in a library area; a module that
only the page needs stays in `website/src/lib/`.

### 3.5 The dense pipeline (`src/mvs/`)

Not audited yet. **Proposal.** The same shape as section 3.3: host view selection, the depth stage
driver, fusion, cleanup and mesh as modules with one responsibility, the kernels beside their
drivers, the lab's dense driver as a job over them. The audit of `mvs.ts` (2,579 lines) and
`tsdf.ts` follows the sparse and lab phases, and its record joins this document by amendment.

### 3.6 Kernels

The kernel audit answers the owner's third question. **Fact.** Most entries are intentional.
Four needs force an entry today: device-side control without a host round trip (the 56
single-invocation steps, counters read by later dispatches, 11 indirect dispatches); workgroup
memory and barriers (29 entries; the compiler refuses both in a kernel loop and itself names the
entry as the remedy); atomics with a returned value or compare-exchange (the TSDF and cloud hash
tables); and the idle-lane hang A1 (`hardware-observations.md`), since the compiler's kernel
lowering emits the early return that A1 forbids. **Fact.** The advantages not taken: about 60
barrier-free per-element maps in the image families (SLIC, planar depth, CLAHE, mesh, inverse
depth, KLT, Hough, the maps of SIFT and ORB) that a kernel function expresses directly, with the
compiler owning the workgroup counts (about 70 host expressions), the 65,535-workgroup split and
fusion; `scan.shade.ts`, its copy in `pair-preparation.shade.ts` and about 25 flag, scatter and
count entries that re-implement `push` (change 0078); exact reductions and histograms
(the KLT eigenvalue maximum, the CLAHE and Hough counts) that re-implement kernel reductions
and the integer scatter; 10 of the 18 oracle twins that add floating-point lanes in an order
other than the GPU's 64-lane tree, so the CPU tier does not give the GPU's bits; shader
constants copied into host files instead of exported.

**Proposal.** The rule of the target: a kernel is an entry where only an entry provides what it
needs (a device-side count or state word read by a later dispatch, indirect dispatch, workgroup
memory and barriers, an atomic with a returned value, a specific invocation-to-work mapping, a
long per-lane loop while A1 is open). A per-element map, an exact reduction and a compaction are
kernel functions, array methods and `push`. A shader constant is exported and imported by the
host. One `chunkedPass` serves every chunked family. The families that stay entries: the tiled
matchers, the BA Schur products, the PCG and LM steps, the pair-model and PnP batches, the TSDF
and cloud hashing, the MVS propagation with its pose cache, the Hough hysteresis, `orbPickTop`,
and `triangulateTracks` until the compiler's idle-lane scaffold changes. The families that
convert, in two phases, are the rows of section 5 (phases 9 and 10); the compiler changes they
wait for are in section 6.

## 4. The rules of every phase

1. **Structure only.** A phase moves, splits, renames, extracts or deduplicates. It changes no
   number, no threshold, no default, no message text, no record field and no public declaration.
   A behaviour difference found on the way (audit sparse, section 8, the last list; audit lab,
   section 9) gets its own record and its own pull request.
2. **One answer, proved.** The gate of a phase is identity: the characterization digests of
   phase 0 (section 5), the existing test suites with the same test count, the public
   declarations of `typecv` and `typecv/gpu` unchanged, the emitted WGSL and GLSL of every kernel
   unchanged unless the phase names the kernel, the run record's `options` identical. On the GPU
   tiers the comparison is baseline against candidate on the same runner (Rule 1.2; identity
   across machines is not established, `engine-v2.md`).
3. **One pull request per phase**, citing this proposal by section. The first commit goes up
   early as a draft (the owner's rule); the pull request is ready when its gate is green.
4. **Aliases.** A moved file leaves a re-export at its old path until the next major version. The
   51 files that tests, scripts and the website import by deep path (audit sparse, 1.3) keep
   working through them.
5. **Order of risk.** Pure moves first, then one definition per constant, then extraction of pure
   functions, then the splits of long functions, then boundary changes. A boundary change waits
   for the owner's go-ahead named in section 5.
6. **What waits for the scene engine** (`engine-v2.md`, `gpu-resident-scene.md`, R03): no edit of
   a sparse-scene kernel (pair models, PnP, triangulation, observations, BA) beyond the
   conversions of section 3.6 and the compiler changes of section 6; no restructuring of
   `IncrementalSfmSession` or `runIncrementalGpu` beyond the extraction of pure policies; no
   extension of `ReconstructionJob` (new requests go to `SceneRequest`); no removal of a deep
   path; no change of a lab preset value while it is being centralized.
7. **The documents follow the code.** A phase that moves a symbol updates every document that
   names it (`sfm.worker.ts` has 23 mentions in 11 documents; `run-record.ts` 10 in 5;
   `SfmLab.tsx` 11 in 5). The website's generated-pages test and the docs index are part of the
   gate.
8. **Execution.** Phases run as sub-agent tasks under the owner's rules: at most three at a time,
   each with its pull request first, each stopped when its report is in. The coordinator runs the
   gate again before a phase is marked ready.

## 5. Phases

The common gate of every phase: `npm run typecheck`, `npm run format:check`, `npm test`,
`npm run build`, `npm run test:shade`, `npm run test:webgpu`, `npm run build:website`,
`npm run test:website`, the same test counts, `dist/**/*.d.ts` unchanged, the phase 0 digests
unchanged, and the document audit of rule 7. Day counts are estimates (**Inference**).

| Phase | Days | Change (structure only) | Own gate |
| --- | --- | --- | --- |
| 0 Characterize | 2–3 | Tests only. Sparse: canonical digests of the full output of `reconstructIncremental`, `reconstructIncrementalScene`, `refineIncremental`, `reconstructComponent`, the focal search and `IncrementalSfmSession` (CPU device and SwiftShader) on `tests/scene.ts` scenes and the four lab samples; digests of the emitted WGSL and GLSL per kernel module; a per-request trace of each driver. Lab: a transcript test that bundles the worker with esbuild, drives it through `onmessage` with a deterministic fake runtime and records every message and every `session.add` input for the gate, legacy, feature-minimum, guided, KLT, ORB, precomputed, dense, failure and loss cases; the same for `MultiVideoJob.finish`. | Passes on unchanged code. Digests and transcripts recorded in `tests/`. |
| 1 Moves | 2–3 | Area directories of section 3.1 created; pure moves with re-exports; the three runtime cycles broken (`judgePair`, `incrementalControls`, errors and `copyReconstruction` out of `mapper.ts`; BA validation helpers out of `bundle-adjustment.ts`; `bundleStallMonitor` and `cooperativeBundleSchur` out of `gpu/bundle.ts`); `pnp.ts` helpers to their own module; `CameraIntrinsics` to the camera module; the worker's imports gain `.ts` extensions. | Cycle search empty; WGSL and GLSL digests unchanged; every deep import still resolves. |
| 2 One definition | 3–4 | `sfm/config` and the lab's `options` module hold each value once: the mapper defaults with the named per-entry differences kept (25 and 8), seed 17, the stall rules, the operator defaults shared by CPU and GPU operators, `observationKey` and `pairKey` with today's strings, the triangulation status enum, the two-view failure table, the host layout constants once per kernel, the lab preset read by both workers, the page and the scripts. `scripts/measure-focal.mjs` reads the preset instead of its copy. | Run record `options` identical; `scripts/measure-focal.mjs` output identical; a test holds the preset to today's values. |
| 3 Record split and named policies | 3–4 | `run-record.ts` split into schema, validate, assemble and summary behind a barrel; `buildRunRecord` and `buildMultiRunRecord` move in with the browser globals as an `environment` argument; the probe record shares the validator helpers. Every inline lab rule becomes a named unit with its current value and unit: pair minimums (16 entries = 8 matches; 32 entries; 16 tracker matches), the feature minimum 8, the retention rule (0.5 s, 3, 3), the legacy and track decisions, the guided trigger, KLT cover. | Golden records byte-identical with `createdAt` masked, v3 to v6 fixtures valid; transcripts identical; table tests copied from today's branches. |
| 4 Shared mapper parts | 3–4 | The pure parts the two mappers copy (masks, seed pick, initialization error, track index, resume adoption, ranking, failure and retry state, result assembly) move to `sfm/mapper/*`; both generators call them. One `drive` (sync and async) with an explicit error rule, the component driver keeping its current rule as a named option; one depth post-step; one bundle-stage telemetry builder; the execution-delta and report-merge helpers; the record packers with today's layouts. | Phase 0 digests of both mappers; request traces identical; plan reports identical (kernel names, dispatches, submissions). |
| 5 Worker by stage | 4–5, three pull requests | 5a `SingleVideoJob` over the stage modules (as `MultiVideoJob` is), the worker keeping intake and the device; 5b `lab-device` for the five open sites and four loss handlers; 5c `lab-protocol`, `start(message)` instead of 23 positional parameters, typed `request<T>()` on the page. | Transcripts identical; the white-box hooks of `dense-gpu-mode-worker.test.mjs` and the source-regex test of `working-budget.test.mjs` rewritten to check the same thing (owner's decision, section 7, item 2); the browser tests in CI; new Node tests of `SingleVideoJob` with a fake port. |
| 6 Long functions | 3–4 | The GPU host functions split without a change to the device steps: request checks, step generators and unpack of `checkPairEstimation`, `estimatePosePnPGpu`, `triangulateTracksGpu`, `evaluateSceneObservationsGpu`; the host half of the two BA routes once where it is verbatim; `IncrementalSfmSession.add` and `update` reduced to pure policies (`evictWindow`, `chooseAnchors`, `carryAnchorEdges`, `archiveSolvePlan`) with unit tests. | Step-name sequences and plan reports identical; WGSL unchanged; session digests on the CPU device and SwiftShader. |
| 7 Page controller and options schema | 4–5 | `lab-run` takes the run control that `runSfM`, `runMultiVideo` and `runBenchmark` copy (React keeps the state setters); `run-options` replaces the ten `*Override()` helpers and the scattered parser calls with one table; the values the page sends stay the same. | Table test over about 40 query strings: old and new start payloads and record options identical; transcripts and golden records identical. |
| 8 Boundaries (owner's go-ahead) | 3–5 | BA window, subset, partition and depth-filter policy move from `HostSceneDriver` into the mapper; `sceneSteps` reaches parity with `incrementalSteps` (the six missing features and `verifiedPairs`) from the phase 4 parts; a typed request-to-result map. The lab's stage modules move to their library area (section 7, item 4). | New parity tests: `sceneSteps` with `HostSceneDriver` equals `incrementalSteps` with the CPU driver bit for bit for every phase 0 option set; `incrementalSteps` digest unchanged; `test:shade` and `test:webgpu`. |
| 9 Kernels, no compiler change | 4–6, one pull request per step | The conversions that change no bits (audit kernel style, section 7, phase 1): the barrier-free maps of SLIC, planar depth, CLAHE apply, mesh field and vertices, and inverse depth (with struct arrays, which drops the host packing) become kernel functions; `kltEigenMax` and its final step become one kernel `max` reduction; `houghVote` becomes an integer scatter and `houghCollect` a `push`; the mesh classify and emit become a `push`; the shader constants are exported and the host imports them; the three `chunkedPass` copies become one; the four test-only triangulation entries go, or a line says why they stay. | CPU-tier outputs equal before and after on the family's inputs (`test:shade`); the family's WebGPU test unchanged (`test:webgpu`); no new readback and no new submission in the plan report; a run on the RTX 5060 Ti while A1 is open (section 7, item 11). |
| 10 Kernels, after the compiler changes | — | One BA module (C1); `push` for every compaction (C3): `scan.shade.ts`, the pair scan copy and the compactions of SIFT, ORB, KLT, matching and MVS; the 18 oracle twins deleted (C2; the CPU digests change once, section 7, item 10); the loop-heavy maps (SIFT orient and describe, the global descriptor, the scalar matchers, k-NN) after C6 and C7; retirement of the object-model protocol and `runIncrementalGpu` with the scene engine; the area entry points go live. | Every BA digest and WebGPU BA test unchanged for C1; row order and counts bit-identical for C3; WebGPU unchanged for C2; the phase 9 gate plus a hardware run for the loop-heavy maps. |
| 11 Dense | — | The audit of `src/mvs/`, then its phases by amendment. | — |

Phases 1 to 3 can run in parallel in different directories; phases 4 to 7 follow them; 8 waits
for the owner; 9 touches only the image, feature, motion and dense families, which the scene
engine does not rewrite, so it can run beside phases 4 to 7; 10 and 11 wait for the compiler
changes, R03 and the dense audit.

## 6. Compiler changes this programme asks for

Each is a proposal in TypeShade's `changes/`, by Rule 3.3, or an issue where the row says so.
The line counts are the kernel audit's estimates (**Inference**) of what each change removes
from TypeCV. The compiler probes that establish each limit are in that audit, section 0.1.

| # | Limit (**Fact**) | Effect in TypeCV | Change asked | Lines |
| --- | --- | --- | --- | --- |
| C1 | `TS8020`: a storage array cannot be a helper's parameter, so a helper that reads a binding belongs to the file that declares it | the fixed and shared-focal BA kernels repeat about 900 of their lines with different record sizes | a helper that takes a storage array by reference, or a module parameterized by constants that a file instantiates twice; without a compiler change, a shared bindings module with one control struct (a layout change) | ~900 |
| C2 | the CPU tier runs one invocation per call and cannot run a barrier (Rule 8.24), although the compiler's own oracle `dispatch` can | 18 `...Oracle` twins, the MVS splits and 29 backend branches; the oracle proves the twin, not the tiled entry, and 10 twins add in another order than the GPU's tree | the host CPU tier of an entry runs barrier entries in passes, as `dispatch` does | ~250 |
| C3 | a kernel function's trip count is computed on the host, and a reduction result or a `push` count reaches only the host | `scan.shade.ts`, its copy, and about 25 flag, scatter, advance and mark entries keep counts on the device by hand | a `push` that appends after a count held in a `Resident` and leaves the new count there; a reduction into a `Resident`; a loop bound read from a `Resident`, lowered to an indirect dispatch | 300–400 |
| C4 | the compiler does not cut a loop into timed chunks (change 0074) | `chunkedPass` three times, `cappedPass`, and per-module chunk generators | a kernel call that splits a proved loop into chunks under `timeBudgetMs` | ~100 |
| C5 | `write(value)` replaces a whole `Resident` | `mvsCopyF32`, `mvsCopyU32` and their staging buffers copy a view into a slot | `write(value, { offset })` on the queue | ~40 |
| C6 | the proof refuses `out[base + i]` with a loop-invariant `base`, several fields at a runtime stride, a nested affine record and two record forms in one loop (rule R3 of 8.22) | several maps cannot convert as written | accept an injective offset, offsets `c < s` checked at the call as `a·i + c` is, and nested affine records | enabler |
| C7 | the map lowering starts every invocation with an early return past the trip count; A1 (`hardware-observations.md`, TypeShade #703) forbids that return on the RTX 5060 Ti | loop-heavy maps must stay entries written clamp-and-mask | an iteration scaffold that clamps the index and masks the writes | enabler |
| C8 | a kernel call takes no workgroup argument and no indirect count | 11 entries dispatch indirectly and cannot become kernel functions | a kernel call whose trip count is read from a `Resident` (shares C3's lowering) | enabler |
| C9 | surface §65 lists `min` and `max` among the scatter forms; the compiler refuses `a[k] = min(a[k], e)` | a conversion that needs it fails | a TypeShade issue: the form the surface states, or a surface correction | bug |
| C10 | timestamp profiling is fixed when a device is created | the worker swaps the device for the dense stage and merges statistics across devices | a per-plan or per-scope switch for timestamp writes | lab |
| C11 | a device loss names no in-flight work | the lab tracks the last kernel itself | a loss record that names the plan or kernel submitted and not completed | lab |
| C12 | chunk rates live as long as the device | a record on a reused device includes earlier runs | a per-run epoch or snapshot of `chunkRates` | lab |
| C13 | GPU time cannot be attributed to stages inside one library call | the job diffs kernel statistics; view-graph stages carry no `gpuMs` | a named profiling scope that labels plan reports | lab |

Not gaps (**Fact**, from the kernel audit): 2D and 3D kernel loops (TypeCV uses only `.x`); f64
in kernel functions; integer scatter atomics; exported constants for the host (surface §64, a
facility not taken; the build check of phase 2 confirms it in the esbuild bundle); workgroup
memory in kernel functions (the entry is the remedy); kernel-to-kernel calls (`fuse` composes
calls on the host). Device-resident hashing and an on-device sort are gaps that remove no lines
today and are not asked for.

## 7. Decisions for the owner

1. **The areas and their names** (section 3.1): the directories, the entry point names, and whether
   `sfm` and `mvs` sit as two areas or under one `reconstruction`.
2. **White-box tests.** May the structure pull requests rewrite the tests that depend on internal
   names and source text (`dense-gpu-mode-worker.test.mjs`, `working-budget.test.mjs`), keeping
   what they check?
3. **Which mapper survives:** converge on `sceneSteps` (the R03 path) and retire
   `incrementalSteps`, or keep both until R03 lands? Phase 8 needs this.
4. **Where the lab's stage modules and the lab preset live:** in a library area (so scripts and
   Node tests share them) or in `website/src/lib/`. The run-record format depends on it.
5. **Behaviour differences found by the audits, each its own pull request if fixed:** the
   component driver's error rule; `finalBundleIterations` 25 against 8 and stall 1e-4 against 1e-9;
   the pair minimums in mixed units (8, 16 and 16 matches); the multi-video record's literal
   `clahe: false` and `keyframeGate: 'phase'`; the defaults that disagree between the page and the
   worker (gate, edge, guided, octave, KLT capacity); two devices in one worker after a single-video
   run; the session's pair-verification seed rewrite.
6. **Code without a product caller:** `posed-evidence.ts` (329 lines) and the observation-expansion
   family (1,692 lines, exported from `typecv/gpu`). Keep, move to `experimental/`, or delete?
7. **Compiler changes** (section 6): which to open as TypeShade proposals now.
8. **Freeze or sequence.** Functional work on the files of a phase stops while the phase is open,
   or follows it.
9. **The front end (`design-rules.md` §4) and the package path (§5.1):** decided separately, or
   together with the area entry points of phase 10.
10. **Floating-point reductions as kernel reductions** (the observation summary, the BA metrics,
    the PnP and consensus scores): the CPU tier and WebGPU would then agree bit for bit, but
    today's numbers and the recorded digests (`PREVIOUS_RESULTS` of the pair-batch and PnP-batch
    tests) change, and a tie in a score can pick another model. A behaviour change under Rule 2.2,
    not proposed here. The same decision covers deleting the 18 oracle twins after C2, which
    renews the CPU digests once.
11. **The hardware gate of the kernel phases.** Phases 9 and 10 need a run on the RTX 5060 Ti while
    A1 is open; CI has no such device. Who runs it, and on which samples.
12. **`EpipolarSession`**, the only kernel-function path today, called by tests and one benchmark:
    public or not. **The four test-only triangulation entries:** removed or kept with a reason.

## 8. Verification

- Functional verification is the gate of section 5 on every phase; it runs in CI (`check`,
  `webgpu`) and locally before a pull request is ready. GPU identity is proved on the same runner;
  native GPUs are not in CI (`physical-gpu-runs.md`).
- A kernel conversion (phases 9 and 10) is proved as the kernel audit's section 5 says: a map, an
  exact fold and a compaction change no bits, so old and new CPU-tier outputs are equal and the
  family's WebGPU test stays; a floating-point sum or a deleted oracle twin changes bits and is a
  decision (section 7, item 10); a chunked reduction keeps one fold order per call.
- The document audit of rule 7 is separate from the tests: the moved symbols are named in the
  documents, the docs index lists what exists, the website's generated pages link.
- Status accounting: each phase's pull request records what moved, which gate ran on which
  commit, and every deviation from this document with its disposition. This document gains a
  line in section 9 per accepted amendment.

## 9. Record of this document

- 2026-10-26: written from the three audits of the same day; proposed (#241).
