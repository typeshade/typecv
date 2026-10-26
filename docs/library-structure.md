# TypeCV library structure and the refactoring programme: proposal

Status: accepted, 2026-10-26 (the owner's go-ahead in the session, #241). This document is the
record that every refactoring pull request cites. Section 7 records the disposition of each
decision. An open item is decided before the phase that needs it. It changes no code and no
behaviour. Labels: **Fact** (read in the code, a document or a measurement),
**Inference** (reasoned from facts), **Proposal** (a choice this document asks the owner to make).

Evidence: three audit records of 2026-10-26.

- [`audits/2026-10-26-sparse-structure.md`](audits/2026-10-26-sparse-structure.md): the sparse
  pipeline on both tiers.
- [`audits/2026-10-26-lab-structure.md`](audits/2026-10-26-lab-structure.md): the lab's
  orchestration.
- [`audits/2026-10-26-kernel-style.md`](audits/2026-10-26-kernel-style.md): how the kernels use
  the compiler.

**Fact.** The audits read the code on the branch of #239 at `bac4261`, with the compiler pin
`d36ac502`. That branch's `src/` and `website/src/` are identical to `main` at `5e1c95c`
(`git diff 5e1c95c bac4261 -- src website/src` is empty). Its `tests/`, `docs/` and `scripts/`
are not, so every count over them below is restated for `5e1c95c`.

## 1. Why

The owner's verdict on 2026-10-26, in three parts:

1. The SfM code is written haphazardly. It is neither structured nor divided into parts.
2. The repository must match its name. TypeCV is a computer vision library, and the code is
   organized as if SfM were the whole of it.
3. The kernels use explicit `@compute` entries everywhere. TypeShade lifts a plain loop in a
   kernel function into a compute kernel by itself. Was the explicit style a choice, or a missed
   advantage?

**Inference.** The measurements below support the verdict.

| Measurement (**Fact**, `main` at `5e1c95c`) | Value |
| --- | --- |
| `src/gpu` | 90 `.ts` files, 37,217 lines: 89 files in one flat directory for SIFT, ORB, KLT, CLAHE, SLIC, Hough, the mapper, MVS, TSDF and the device, and `kernels/float-matching.shade.ts` |
| `src/reconstruction` | 33 files, 12,472 lines |
| `website/src/lib` | 48 files, 9,978 lines |
| Largest files | `SfmLab.tsx` 5,441; `mvs.ts` 2,579; `SfmViewer.tsx` 2,404; `run-record.ts` 1,885; `mvs.shade.ts` 1,821; `sfm.worker.ts` 1,635; `mapper.ts` 1,448 |
| Functions of 590 lines or more | 10 (TypeScript AST over `src/` and `website/src/`): `SfmLabContent` 4,411, `SfmViewer` 2,094, `bundleAdjustJointFocalGpu` 989, `bundleAdjustGpuFixed` 846, `checkPairEstimation` 768, `expandObservationsGpu` 732, `runMultiVideo` 713 (inside `SfmLabContent`), `createKltTracker` 685, `reconstructDense` 613, `incrementalSteps` 590 |
| Runtime import cycles | 3: `mapper.ts ↔ focal-search.ts`, `bundle-adjustment.ts ↔ joint-focal.ts`, `gpu/bundle.ts ↔ gpu/joint-focal.ts` |
| Two mapper state machines | `incrementalSteps` (590 lines) and `sceneSteps` (409 lines). Their files share 151 verbatim lines, about 120 of them inside the two functions; most of the rest is a near copy. `sceneSteps` lacks six features. **Inference:** no test can see the drift |
| Two GPU bundle-adjustment routes | at least 538 of 1,043 host lines and 896 of 1,125 kernel lines of the fixed route recur verbatim in the shared-focal route |
| The lab's six mapper values | in full in 9 places outside the tests (`LAB_OPTIONS`, `MAPPER` and 7 copies in 6 scripts) and in part in `SfmLab.tsx`; seed 17 in 8 places; LM stall 1e-9 in 4; `finalBundleIterations` has two library defaults (25 and 8) |
| Lab worker | `start` 243 lines with 23 positional parameters; `processFrame` 266 lines; 13 `post()` calls inside computation; policy rules (descriptor retention, pair minimums, keyframe rules) inline and without a unit test (only `site.test.mjs` reaches them, end to end in CI) |
| Public surfaces | `src/index.ts` exports 115 names: CPU geometry, two-view, tracks, the incremental mapper, BA, focal search, the inverse-depth tracker and BA, observation masks, and dense cloud cleanup and meshing. `src/gpu/index.ts` has 68 `export … from` statements that export 200 names (77 values, 123 types) from 39 modules. The lab's multi-video path imports `src/gpu/internal.ts`. Tests, scripts and the website import 74 `src` modules by deep path; `website/src/` alone imports 7 |
| Kernel style | 30 `.shade.ts` files, 13,186 lines, 251 `@compute` entries: 56 single-invocation steps, 136 per-element maps, 29 cooperative (workgroup memory or barriers), 12 with atomics, 18 `...Oracle` CPU twins; 8 kernel functions, all in `epipolar.shade.ts`, which the SfM pipeline does not call; 0 array methods in kernels; every dispatch 1D; an entry call never fuses with another call in a plan |

**Inference.** No boundary separates an area of computer vision from another, a policy from its
numerics, or the oracle from the product path. A change in one place is a change in several.

## 2. What already stands

**Proposal.** The programme arranges the code under the decisions below and adds no direction of
its own, with two exceptions. Section 7 asks the owner to amend `engine-v2.md` for the shared scan
(item 14) and `gpu-resident-scene.md` for the `bundle` request (item 16).

- `engine-v2.md` (accepted 2026-10-19): the engine is a GPU-resident program with f32 arithmetic
  and a scene state machine over indices. The CPU f64 path is the oracle. One npm package has one
  entry point per area over one shared device layer. The present layout binds nothing: area entry
  points come with the engine, and old paths stay as aliases until the next major version. The
  shared scan and the partial reductions are kept as the device layer.
- `gpu-resident-scene.md` (accepted): the resident scene and its index protocol (`SceneRequest`,
  `sceneSteps`, `HostSceneDriver`). It derives the CSR orders and the compaction from the shared
  scan.
- `architecture.md`: shared data contracts (images, features, matches, cameras, observations)
  that no one pipeline owns; pipeline state stays in the pipeline. The root `typecv` exports CPU
  geometry and reconstruction, and `typecv/gpu` bundles the TypeShade runtime for its kernels.
  Proposed layers add no export.
- `design-rules.md` (§1 to §3 accepted 2026-10-21):
  - Rule 1.1: one answer on every tier.
  - Rule 1.2: proved by tests, bit for bit where the operation has one answer.
  - Rule 1.4: a fused plan gives the bits that the calls give run apart.
  - Rule 2.1: the order of priorities.
  - Rules 3.1 to 3.3: composition with the TypeShade family; a compiler gap becomes a TypeShade
    proposal, not a workaround.
  - §4 (the `cv` namespace front end) is a proposal, and §5 lists its open decisions. This
    document does not decide them.
- `roadmap.md`: the capability areas. Section 3.1 follows them with two differences. It splits
  "3D vision and reconstruction" into sparse (`sfm`) and dense (`mvs`) reconstruction, and it
  shortens "Learned vision integration" to "Learned vision".
- `multi-video.md`, Amendment 6, decision 1: the public export stays closed, and the multi-video
  symbols are exported from `src/gpu/internal.ts` only.
- `gpu-sfm.md`, the rule for kernel authors after A1 of `hardware-observations.md`: an idle lane
  does not return before a loop or a barrier. A workgroup whose idle lanes returned early lost the
  device on an RTX 5060 Ti, and TypeShade #703 is open.

**The build split** (**Fact**, `5e1c95c`). The build selects GPU code by directory:

- `tsconfig.json` (`npm run typecheck`) and `tsconfig.build.json` (`npm run build:core`, the root
  entry) exclude `src/gpu`.
- `tsconfig.gpu.json` includes `src/gpu/**/*.ts` only. It resolves a kernel import to the
  kernel's generated host view through `moduleSuffixes: [".typeshade", ""]`.
- `scripts/build-gpu.mjs` builds `dist/gpu` from `src/gpu/index.ts` (with `vite.gpu.config.ts`)
  and `dist/internal` from `src/gpu/internal.ts`. It deletes the generated host views under
  `src/gpu` only (lines 14 and 43).
- `.prettierignore` lists 27 kernel files under `src/gpu/` by path.
- `package.json` exports `./gpu` with the types `./dist/types/gpu/index.d.ts`.
- `tsc` with the core settings cannot compile a kernel. On `slic.shade.ts` it reports 166 errors,
  8 of them TS1206 (decorators).
- No module outside `src/gpu` imports one inside it. Thirty modules inside import modules outside,
  so `dist/types/` also holds declarations of `src/geometry`, `src/sfm` and `src/reconstruction`.

**Where the programme stands.** **Fact.** `reconstruction-status.md`, a dated snapshot of
2026-10-22 at `3d33c257`, orders the remaining work. R01 (complete the verification of #199) and
R02 (a fresh native-GPU baseline on the RTX 2080 and 5060 Ti) are P0. R03 (the persistent common
`GpuScene`, which integrates `sceneSteps` with a device scene) is P1. **Fact.** No document on
`main` records any of the three as done. No open pull request on 2026-10-26 (#148, #239, #241)
implements a `GpuScene`. **Inference.** R03 has not started. So the steps of phase 10 that edit a
sparse-scene kernel, and the retirement of the object-model protocol, have no expected start.
**Proposal.** Phases 0 to 9 wait on none of R01 to R03. Decision (e), item 17, ranks the
programme against them.

## 3. Target structure

### 3.1 Areas and entry points

**Proposal.** The source tree follows the roadmap's areas, with the two differences of section 2.
SfM is one consumer under reconstruction. Every operator that exists today moves to its area. The
package entry points stay as they are until phase 10.

| Area (roadmap) | Directory | Entry point | What moves there today |
| --- | --- | --- | --- |
| Shared data and device execution | `src/device/`, `src/contracts/` | `typecv/device` | device: `gpu/device.ts`, `budget-plan.ts`, `readback.ts`, `scan.ts` + `scan.shade.ts`, `device-diagnostics.ts`, `rgba-read-view.ts`, the shared kernel helpers (`numerics.shade.ts`, `projective.shade.ts`) and `chunkedPass` (now in `gpu/cloud-clean.ts`); contracts: the object-model types of `reconstruction/types.ts`, the camera and pose types (`Camera`, `CameraIntrinsics`, `Pixel`, `Pose`) and `CameraObservationMask`, and later the image, feature and match contracts |
| Image processing | `src/image/` | `typecv/image` | `clahe`; `packRgba` (now in `sift.ts`); the pyramid layout `kltLayout` that `sift.ts` and the KLT modules share (now in `klt.ts`) |
| Image analysis | `src/analysis/` | `typecv/analysis` | `slic`, `hough` |
| Features and correspondence | `src/features/` | `typecv/features` | `sift`, `orb`, `orb-pattern`, `orb-reference`, `features` (corner and BRIEF), `tiled-features`, `float-matching` with `kernels/float-matching`, Hamming matching, and the global descriptors (`global-descriptor`, `global-descriptor-gpu`, `global-descriptor-candidates` and their kernel) |
| Geometric vision | `src/geometry/` | `typecv/geometry` | today's `src/geometry` and `src/sfm` (the oracle primitives: camera, pose, linear algebra, two-view, homography, P3P and PnP, triangulation, rotation, inverse depth, image pair); `validateReconstruction` and `observation-paths.ts` from `src/reconstruction`; the GPU operators `pair-estimation`, `image-pair`, `two-view`, `homography`, `p3p`, `pnp`, `triangulation`, `observations`, `observation-layout`, `epipolar` and their kernels |
| Motion and video | `src/motion/` | `typecv/motion` | `klt`, `klt-tracker`, `klt-device`, `klt-pair-cache`; the keyframe gate and the decode schedule if item 4 places them in the library (section 3.4) |
| 3D vision: sparse reconstruction | `src/sfm/` | `typecv/sfm` | the rest of today's `src/reconstruction` and the GPU mapper, bundle, joint focal, inverse depth, view graph, multi-video and observation-expansion modules, arranged as section 3.3 says |
| 3D vision: dense reconstruction | `src/mvs/` | `typecv/mvs` | `mvs`, `mvs-views`, `mvs-residency`, `planar-depth*`, `tsdf*`, `cloud-clean` (without `chunkedPass`), `cloud-knn`, `cloud-voxel`, `dense-mesh*`, and from `src/reconstruction`: `tsdf.ts`, `dense-mesh.ts`, `cloud-clean.ts`, `dense-label.ts`, `marching-cubes-tables.ts`, `dense-score.ts` |
| Learned vision | — | — | nothing today (`roadmap.md`: planned) |
| Evaluation and interoperability | `scripts/`, `tests/` | — | as today: the measurement and comparison scripts and the test fixtures |

The gray conversion and pyramid entries stay inside `features.shade.ts`, `sift.shade.ts` and
`klt.shade.ts`. Moving them is a kernel edit. The WGSL gate (section 4, rule 2) allows a kernel
edit only where a phase names the kernel, and no phase names these.

Rules of the layout (**Proposal**):

- **One area, one responsibility.** A directory holds one area. A file holds one responsibility.
  A kernel file sits beside the host module that drives it.
- **GPU code in `gpu/` subdirectories.** Each area keeps its kernels and GPU host modules in
  `src/<area>/gpu/`, and a nested module in its own `gpu/`, such as `src/sfm/bundle/gpu/`. The
  build configuration selects `src/*/gpu/**` and `src/*/*/gpu/**` besides `src/gpu/**`. The
  directory `src/gpu/` keeps the barrels `index.ts` and `internal.ts`, the aliases (section 4,
  rule 4) and the vocabulary files. Tests, scripts and the site read the vocabulary files by
  path. A file of `src/gpu/` moves only into a `gpu/` subdirectory, and a file outside it only
  outside one. So no file changes build. A module outside a `gpu/` directory never imports one
  inside it, as today. Phase 1a makes this change (decision (a), item 13).
- **Layers.** An area imports only the areas below it, type-only imports included. `contracts`
  and `device` are at the bottom. `image` and `geometry` sit above them, `features` and
  `analysis` above those, and `motion` above `features`. `sfm` and `mvs` are at the top. No
  import cycle joins two areas.
- **The layers today.** **Fact.** An import search over `src/` at `5e1c95c` puts each file in
  the area that the table gives it. It finds 8 imports that break the layering rule, 7 of them at
  run time. Four more imports follow the rule only because of this level order (`sift.ts` →
  `clahe.ts`, `klt-tracker.ts` → `sift.ts` and `features.shade.ts`,
  `global-descriptor-candidates.ts` → `geometry/numerics.ts`). **Proposal.** Phase 1c removes
  the 8 by moving the shared pieces down:
  - `chunkedPass` to `device`, since `slic.ts` and `hough.ts` import it from the dense
    `cloud-clean.ts`;
  - `packRgba` to `image`, since `slic.ts` and `hough.ts` import it from `sift.ts`;
  - the pyramid layout `kltLayout` to `image`, since `sift.ts` imports it from `klt.ts`;
  - the object-model types to `contracts`, with the camera and pose types and
    `CameraObservationMask` that they and `observation-paths.ts` use;
  - `validateReconstruction` to `geometry`, since `gpu/observations.ts` imports it;
  - `observation-paths.ts` to `geometry`, since `mvs.ts` imports it and it calls
    `geometry/camera.ts`.

  **Fact.** With these moves the same search finds no import that breaks the rule.
- **Imports from outside `src/`.** The website and new scripts import entry points: the root,
  `typecv/gpu` and the internal entry. Today's deep imports keep working through the aliases
  (section 4, rule 4). No phase rewrites them, and section 4, rule 6 keeps every deep path until
  the scene engine.
- **`src/gpu/internal.ts` stays.** Amendment 6 of `multi-video.md` keeps the public export
  closed. The lab's stage modules that phase 8b moves into the library are exported from
  `internal.ts` only. Closing it is decision (c), item 15.
- **Entry points.** The root `typecv` (CPU only) and `typecv/gpu` (the GPU barrel, one runtime
  copy) keep their export lists until the next major version. Phase 0's surface bake holds them.
  Every moved file leaves a re-export at its old path until then, except a kernel file (section
  4, rule 4).
- **Area entry points.** They go live in phase 10, when the owner says so (`engine-v2.md`:
  "introduced as the engine lands"). Until then the directories exist and the two barrels export
  them. **Proposal.** `typecv/<area>` then exports the area's CPU half, and the area's GPU half
  stays in `typecv/gpu`. **Inference** (audit lab, L5): every GPU module thus shares one runtime
  copy, since a device of one copy cannot pass to the other. The names presuppose the package path
  `typecv`, which `design-rules.md` §5.1 leaves open (item 9).

### 3.2 Shared contracts and the device layer

**Fact.** `architecture.md` asks for data contracts (dimensions, channels, numeric type, layout,
camera conventions, index and ownership rules) that no one pipeline owns. **Fact** (audit
sparse, 2.3 and 2.4). Today seven packers pack a camera record. A triangulation status is spelled
in four places, and a pair key in three spellings.

**Proposal.** `src/contracts/` holds the types and the one encoder and decoder of each record
that crosses a boundary. The records are: image, feature set and descriptors, matches, camera and
pose, observation, scene (views, landmarks, observations), status enums and the failure-reason
table. Host and kernel code read each layout constant from one place: the kernel exports it, and
the host imports it. `src/device/` stays what `engine-v2.md` keeps: the device owner, plans,
budgets, readback and the shared scan. `chunkedPass` joins it (section 3.1).

**Fact** (audit sparse, L3). The pinned host view exposes the exported constants of a kernel
(surface §64), and no TypeCV host file imports one. **Inference.** The copied constants are a
facility not taken, not a compiler limit. Nobody has checked such an import in the esbuild
bundle yet. Phase 9 checks it on the non-sparse kernels, and phase 10 extends it to the sparse
kernels after R03.

### 3.3 The sparse pipeline (`src/sfm/`)

**Proposal** (audit sparse, section 7):

- one mapper policy as a state machine over indices;
- one typed request protocol;
- two drivers of it: the host f64 oracle and the resident GPU scene;
- one device layer.

| Module | Responsibility | Contract |
| --- | --- | --- |
| `sfm/model` | copy of the object model, `observationKey`, `pairKey`, content ids (the model's types are in `contracts`, its validation in `geometry`) | pure, no tier |
| `sfm/config` | every default once; `resolveMapperControls`; session and focal controls; the lab preset, if item 4 places it in the library | pure; frozen controls; throws `RangeError` |
| `sfm/tracks` | track union, rotation evidence, observation masks | pure |
| `sfm/pairs` | pair verdict, seed score, pair seeds, verification order, pre-verified index | pure policy and `verifyPair` requests |
| `sfm/mapper` | initialization, ranking, registration gates and retries, weak registration, triangulation and continuation rules, filter rules, bundle schedule and window | a generator over `SceneRequest`; no numerics, no callbacks, no packing beyond small typed arrays |
| `sfm/protocol` | request and result types (a typed map), errors, `drive` (sync and async) with one error rule, the depth post-step | the only place of the error rule |
| `sfm/scene` | `HostScene`, layout constants shared with kernels, `HostSceneDriver` (oracle), `GpuSceneDriver` (R03) | executes one request; no policy |
| `sfm/bundle` | CPU solvers, bearing constraints, the problem plan and shared validation; the GPU host half once, in `sfm/bundle/gpu/` | pure numerics; tier-specific messages kept |
| `sfm/focal` | focal search, pair base, history seeds | a generator over the protocol |
| `sfm/depth` | inverse-depth tracker, depth BA, publication, record, sparse Schur | its own generator |
| `sfm/session` | window eviction, anchors, carried edges, archive, archive-solve plan | pure functions and one orchestrator |
| `sfm/multiview` | view graph, components, multi-video; it uses the global descriptors of `features` | pure and a backend interface |

The mapping from today's files to these modules is the table in audit sparse, section 7. Section
3.1 changes four of its entries. The object-model types go to `contracts`, and
`validateReconstruction` and `observation-paths.ts` to `geometry`. The global descriptors go to
`features`, and the view graph that uses them stays in `sfm/multiview`.

### 3.4 The lab's orchestration

**Proposal** (audit lab, section 6):

- a worker of about 150 lines that only moves messages;
- the pipeline stages as modules with one responsibility each, calling the GPU through an injected
  port (the precedents are `JobApi` and `DenseRunContext`);
- typed events;
- one options schema for the page, both workers, the record and the scripts;
- the run record split into schema, validator, builder and summary behind a barrel.

| Module | Responsibility |
| --- | --- |
| `options` | the schema: key, flag, parser, default, where it applies, where the record keeps it |
| `schedule` | timestamps, fps, ranges, the one-frame look-ahead |
| `keyframe-gate` | the phase, track and legacy rules as one unit |
| `features` | the front end through the port, the feature minimum, colours |
| `matching` | candidates, descriptor retention, pair admission, guided matching |
| `klt` | tracker life, join, merge, verdict |
| `registration` | the driver over the session (the scene engine later; item 18 asks how much to invest in it) |
| `single-video-job`, `multi-video-job`, `dense-driver` | the three jobs over the stages |
| `ledger` | stage totals, GPU accounting, the record's measured fields |
| `lab-device` | open, reuse, loss, swap: one module for every worker. A device stays with the runtime copy that opened it (audit lab, L5) |
| `lab-protocol` | typed page-to-worker messages |
| `lab-run` | the page's run controller |
| `run-record/` | schema, validate, assemble, summary, and the barrel the collector imports |

Where the stage modules live is the owner's decision (section 7, item 4). **Proposal.** The
modules that a script needs (the gate, the matching policy, the options) go to a library area. A
module that only the page needs stays in `website/src/lib/`.

### 3.5 The dense pipeline (`src/mvs/`)

Not audited yet. **Proposal.** The dense pipeline takes the shape of section 3.3. Host view
selection, the depth stage driver, fusion, cleanup and mesh become modules with one
responsibility, with the kernels beside their drivers. The lab's dense driver becomes a job over
them. The audit of `mvs.ts` (2,579 lines) and `tsdf.ts` follows the sparse and lab phases, and its
record joins this document by amendment (phase 11).

### 3.6 Kernels

The kernel audit answers the owner's third question. **Inference** (audit kernel style, section
1): most entries are intentional. Four needs force an entry today:

1. device-side control without a host round trip: the 56 single-invocation steps, counters read
   by later dispatches, and 12 indirect dispatches;
2. workgroup memory and barriers (29 entries): the compiler refuses both in a kernel loop and
   itself names the entry as the remedy;
3. atomics with a returned value or compare-exchange: the TSDF and cloud hash tables;
4. the idle-lane hang A1 (`hardware-observations.md`): the compiler's map lowering emits the
   early return that A1 forbids.

**Inference** (the same audit, with its estimates). These are the advantages not taken:

- About 60 barrier-free per-element maps could be kernel functions. They are in SLIC, planar
  depth, CLAHE, mesh, inverse depth, KLT, Hough and the maps of SIFT and ORB. The compiler would
  then own the workgroup counts (about 70 host expressions), the 65,535-workgroup split and fusion.
- `scan.shade.ts`, its copy in `pair-preparation.shade.ts` and about 25 flag, scatter and count
  entries re-implement `push` (change 0078). `engine-v2.md` keeps the shared scan as the device
  layer (item 14).
- Exact reductions and histograms (the KLT eigenvalue maximum, the CLAHE and Hough counts)
  re-implement kernel reductions and the integer scatter.
- Ten of the 18 oracle twins add floating-point lanes in an order other than the GPU's 64-lane
  tree. So the CPU tier does not give the GPU's bits.
- Shader constants are copied into host files instead of exported.

**Proposal.** The rule of the target: a kernel is an entry where only an entry provides what it
needs. Only an entry provides:

- a device-side count or state word read by a later dispatch;
- indirect dispatch;
- workgroup memory and barriers;
- an atomic with a returned value;
- a specific invocation-to-work mapping;
- a long per-lane loop written clamp-and-mask, while A1 is open.

A per-element map, an exact reduction and a compaction are kernel functions, array methods and
`push`. A shader constant is exported and imported by the host. One `chunkedPass` serves every
chunked family.

These families stay entries:

- the tiled matchers;
- the BA Schur products and the PCG and LM steps;
- the pair-model and PnP batches;
- the TSDF and cloud hashing;
- the MVS propagation with its pose cache;
- the Hough hysteresis and `orbPickTop`;
- `triangulateTracks`, until the compiler's idle-lane scaffold changes (C7).

The families that convert are the rows of phases 9 and 10 (section 5), and the compiler changes
they wait for are in section 6.

## 4. The rules of every phase

**Proposal.**

1. **Structure only.** A phase moves, splits, renames, extracts or deduplicates. It changes no
   number, no threshold, no default, no message text, no record field and no public declaration.
   A behaviour difference found on the way gets its own record and its own pull request (audit
   sparse, section 8; audit lab, section 9). Phases 8 and 10 change behaviour and stand outside
   this rule. Their gates are parity gates, and each behaviour change has its own record.
2. **One answer, proved.** The gate of a phase is identity:
   - the characterization digests and the baselines of phase 0;
   - the existing test suites, whose test count changes only by the tests that the phase's pull
     request names;
   - the public declaration surface of `typecv` and `typecv/gpu`, baked by phase 0's script,
     unchanged except for additions that the phase names;
   - the emitted WGSL and GLSL of every kernel, unchanged unless the phase names the kernel;
   - the run record's `options`, identical.

   On the GPU tiers the comparison is baseline against candidate on the same runner, in the CI
   job of phase 0. Identity across machines is not established (Rule 1.2, `engine-v2.md`).
3. **One pull request per step named in section 5**, citing this proposal by section. A phase
   with several steps is several pull requests. The first commit goes up early as a draft (the
   owner's rule), and the pull request is ready when its gate is green.
4. **Aliases.** A moved file leaves a re-export at its old path until the next major version.
   The 74 `src` modules that tests, scripts and the website import by deep path on `5e1c95c` keep
   working through them. **Inference.** A kernel file leaves no alias, since `typeshade/vite`
   compiles every `*.shade.ts` that a bundle reads as a shader module. Its importers change in
   the same pull request, among them the 23 files under `tests/` and `scripts/` that name a
   kernel path.
5. **Order of risk.** Pure moves come first, then one definition per constant, then extraction of
   pure functions, then the splits of long functions, then boundary changes. A boundary change
   waits for the owner's go-ahead named in section 5.
6. **What waits for the scene engine** (`engine-v2.md`, `gpu-resident-scene.md`, R03):
   - no edit of a sparse-scene kernel (pair models, PnP, triangulation, observations, BA, inverse
     depth) before R03;
   - no restructuring of `IncrementalSfmSession` or `runIncrementalGpu` beyond the extraction of
     pure policies;
   - no extension of `ReconstructionJob` (new requests go to `SceneRequest`);
   - no removal of a deep path;
   - no change of a lab preset value while it is being centralized.

   The sparse audit forbids every kernel edit before the scene work (its section 8). This rule
   narrows that to the kernels that `engine-v2.md` rewrites to bind the resident scene. So phase
   9 may convert image and dense kernels.
7. **The documents follow the code.** A phase that moves a symbol updates every living document
   that names it. On `5e1c95c`, `sfm.worker.ts` has 22 mentions in 10 files (20 in 8 Markdown
   documents), `run-record.ts` 10 in 5 and `SfmLab.tsx` 11 in 5. Dated records stay as written:
   the audits, the JSON experiment records and dated status entries such as
   `reconstruction-status.md`. Each records a configuration, and an edit would falsify it. The
   website's generated-pages test and the docs index are part of the gate.
8. **Execution.** Phases run as sub-agent tasks under the owner's rules. At most three run at a
   time; each opens its pull request first and stops when its report is in. The local gate covers
   the CPU tier. The sub-agent container has no Chromium. So `test:webgpu`, the browser half of
   `test:website` and the GPU-tier digests run in CI only. The coordinator runs the local gate
   again and reads the CI result before a phase is marked ready.

## 5. Phases

**Proposal.** The common gate of every phase is what `.github/workflows/ci.yml` runs:

- job `check`: `npm run check`, `npm --prefix website run check`, `npm run build:website`,
  `npm run example` and `npm run example:incremental`. `npm run check` runs `format:check`,
  `typecheck`, `npm test`, `build` and `test:shade`;
- job `webgpu`: `npm run test:webgpu`, `npm run audit:typeshade` and `npm run test:website`;
- the job of phase 0: the GPU-tier digests of base and head, compared on one runner.

It adds the conditions of rule 2 and the document audit of rule 7 (section 4). Day counts are
estimates (**Inference**).

| Phase | Days | Change | Own gate |
| --- | --- | --- | --- |
| 0 Characterize | 3–5 | Tests and tools only, in five parts. **Digests:** canonical digests (Float64 bit patterns) of the full output of `reconstructIncremental`, `reconstructIncrementalScene`, `refineIncremental`, `reconstructComponent`, the focal search and `IncrementalSfmSession`, on the CPU device and on SwiftShader, on the `tests/scene.ts` scenes and the four samples that `scripts/measure-focal.mjs` reads (`south-building`, `gerrard-hall`, `person-hall`, `graham-hall`); digests of the emitted WGSL and GLSL per kernel module; a per-request trace of each driver. **Baselines** of the later gates: plan reports (kernel names, dispatches, submissions) and step-name sequences of the GPU drivers; golden run records (single, multi, failed, dense) with `createdAt` masked; the output of `scripts/measure-focal.mjs`; the start payloads and record options of about 40 query strings. **Lab transcripts:** a test that bundles the worker with esbuild, drives it through `onmessage` with a deterministic fake runtime and records every message and every `session.add` input for the gate, legacy, feature-minimum, guided, KLT, ORB, precomputed, dense, failure and loss cases; the same for `MultiVideoJob.finish`. **Tools:** `scripts/api-surface.mjs` writes the exported names and declaration text of `typecv` and `typecv/gpu` to `tests/api-surface.txt`; `scripts/import-graph.mjs` prints the runtime import cycles of `src/` and `website/src/` and the imports that break the layering rule of section 3.1. **CI:** a job that builds the base and the head of a pull request on one runner, runs the GPU-tier digest tests on both and compares them (audit sparse, P0). | Passes on unchanged code. The CPU-tier digests and the baselines are recorded in `tests/`. The CI job reports equal digests when base and head are one commit. The cycle search prints today's three cycles. |
| 1 Moves | 3–5, three pull requests | **1a Build selection** (decision (a)): `tsconfig.json`, `tsconfig.build.json`, `tsconfig.gpu.json`, `scripts/build-gpu.mjs` and `.prettierignore` select the `gpu/` subdirectories of section 3.1 besides `src/gpu/**`; no file moves. **1b Moves:** the area directories; pure moves with re-exports (a kernel file without one, its importers changed); the two BA cycles broken (`validateReconstruction`, `cloneBundleReconstruction`, `bundleMetrics` and `bundleStallTolerance` out of `bundle-adjustment.ts`; `bundleStallMonitor` and `cooperativeBundleSchur` out of `gpu/bundle.ts`); `judgePair`, `incrementalControls`, the errors and `copyReconstruction` out of `mapper.ts`; the `pnp.ts` helpers to their own module; `.ts` extensions in the worker's imports. The mapper–focal cycle stays: `incrementalSteps` and `focalSearchSteps` call each other, which no move can break (phase 4). **1c Cross-area imports**, resolved by moving the shared pieces of section 3.1: `chunkedPass`, `packRgba`, `kltLayout`, the object-model, camera, pose and mask types, `validateReconstruction` and `observation-paths.ts`. | 1a: the `dist/` file list and the surface bake identical before and after; `typecheck`, `build` and `format:check` pass. 1b: the surface bake unchanged; the `dist/` file list differs only by the new paths, with every old path kept as an alias; WGSL and GLSL digests unchanged; every deep import resolves; the two BA cycles gone, the mapper–focal cycle the only one left. 1c: `scripts/import-graph.mjs` finds no import that breaks the layering rule; the surface bake unchanged. |
| 2 One definition | 3–4 | `sfm/config` and the lab's `options` module hold each value once: the mapper defaults with the named per-entry differences kept (25 and 8), seed 17, the stall rules, the operator defaults shared by CPU and GPU operators, `observationKey` and `pairKey` with today's strings, the triangulation status enum, the two-view failure table, the host layout constants once per kernel. The lab preset sits in one module, placed by item 4 (`website/src/lib/` until then). Both workers, the page and the six scripts that copy it read it: `measure-focal.mjs`, `measure-multi-video.mjs`, `benchmark-sfm-pyramid-e2e.mjs`, `benchmark-rotation-tracking.mjs`, `benchmark-sift-clahe.mjs`, `measure-video-sfm-performance.mjs`. | Run record `options` identical; the output of `scripts/measure-focal.mjs` identical to phase 0's baseline; a test holds the preset to today's values. |
| 3 Record split and named policies | 3–4 | `run-record.ts` split into schema, validate, assemble and summary behind a barrel; `buildRunRecord` and `buildMultiRunRecord` move in with the browser globals as an `environment` argument; the probe record shares the validator helpers. Every inline lab rule becomes a named unit with its current value and unit: pair minimums (16 entries = 8 matches; 32 entries; 16 tracker matches), the feature minimum 8, the retention rule (0.5 s, 3, 3), the legacy and track decisions, the guided trigger, KLT cover. | Golden records byte-identical to phase 0's baseline with `createdAt` masked; v3 to v6 fixtures valid; transcripts identical; table tests copied from today's branches. |
| 4 Shared mapper parts | 3–4 | The pure parts the two mappers copy (masks, seed pick, initialization error, track index, resume adoption, ranking, failure and retry state, result assembly) move to `sfm/mapper/*`; both generators call them. One `drive` (sync and async) with an explicit error rule, the component driver keeping its current rule as a named option; one depth post-step; one bundle-stage telemetry builder; the execution-delta and report-merge helpers; the record packers with today's layouts. The mapper–focal cycle is broken by injection: `incrementalSteps` passes itself to `focalSearchSteps`, so `focal-search.ts` no longer imports `mapper.ts` at run time. | Phase 0 digests of both mappers; request traces identical; plan reports identical to phase 0's baselines (kernel names, dispatches, submissions); the cycle search finds no cycle. |
| 5 Worker by stage | 5–7, four pull requests | 5a the lab's pure helpers (audit lab, P1): budgets, feature cap, `medianMotionPx`, `trimStages`, the frame camera, colours, match lines and landmark colours, each with its variants kept (the worker's persistent colour cache, the job's per-call cache). 5b `SingleVideoJob` over the stage modules (as `MultiVideoJob` is), the worker keeping intake and the device. 5c `lab-device` for the five open sites and four loss handlers. 5d `lab-protocol`, `start(message)` instead of 23 positional parameters, typed `request<T>()` on the page. | Transcripts identical; table tests for each helper of 5a; the white-box hooks of `dense-gpu-mode-worker.test.mjs` and the source-regex test of `working-budget.test.mjs` rewritten to check the same thing (item 2); the browser tests in CI; new Node tests of `SingleVideoJob` with a fake port. |
| 6 Long functions | 3–4 | The GPU host functions split without a change to the device steps: request checks, step generators and unpack of `checkPairEstimation`, `estimatePosePnPGpu`, `triangulateTracksGpu`, `evaluateSceneObservationsGpu`; the host half of the two BA routes once where it is verbatim; `IncrementalSfmSession.add` and `update` reduced to pure policies (`evictWindow`, `chooseAnchors`, `carryAnchorEdges`, `archiveSolvePlan`) with unit tests. | Step-name sequences and plan reports identical to phase 0's baselines; WGSL unchanged; session digests on the CPU device and, in the CI job, on SwiftShader. |
| 7 Page controller and options schema | 4–5 | `lab-run` takes the run control that `runSfM`, `runMultiVideo` and `runBenchmark` copy (React keeps the state setters). `run-options` replaces the nine `*Override()` helpers (eight in `SfmLab.tsx` and `decodeOverride` in `decode-protocol.ts`) and the scattered parser calls with one table; the values the page sends stay the same. | Table test over about 40 query strings: start payloads and record options identical to phase 0's baseline; a Node test of the controller against a scripted fake worker (audit lab, P5); transcripts and golden records identical. |
| 8 Boundaries (owner's go-ahead) | 3–5, two pull requests | Behaviour; outside rule 1. **8a Mapper boundary** (items 3 and 16): BA window, subset, partition and depth-filter policy move from `HostSceneDriver` into the mapper. This changes the accepted `bundle` request of `gpu-resident-scene.md`: the mapper decides the subset, and the result stops reporting it. So 8a carries that document's amendment. `sceneSteps` reaches parity with `incrementalSteps` (the six missing features and `verifiedPairs`) from the phase 4 parts; a typed request-to-result map. **8b Lab placement** (item 4): the lab's stage modules move to the area that item 4 names, exported from `src/gpu/internal.ts` only (Amendment 6); the scripts that copy the pipeline are ported to the stages, so their outputs change (audit lab, P7). | Parity gates, each behaviour change with its own record. 8a: `sceneSteps` with `HostSceneDriver` equals `incrementalSteps` with the CPU driver bit for bit for every phase 0 option set; the `incrementalSteps` digest unchanged; the `reconstructIncrementalScene` digests of the option sets that use the six features renewed once, in 8a's record; `test:shade` and `test:webgpu`. 8b: transcripts identical; a CPU-tier run of `SingleVideoJob` on a tiny synthetic sequence compared with WebGPU under Rule 1.2; each ported script's new output recorded. |
| 9 Kernels, no compiler change | 2–3, one pull request per step | The conversions of audit kernel style, section 7, phase 1, narrowed by A1. An entry converts only if it has no per-lane loop and does not use the clamp-and-mask pattern today; the conversion then keeps its idle-lane shape. Checked in the kernels at `5e1c95c`: `planarPropose`, `planarVerify`, `planarFinish` and `claheApply` qualify, since they return early and run no loop. Every SLIC and mesh entry, `planarFit`, `kltEigenMax`, `houghVote` and `houghCollect` loop or clamp and mask, so they wait for C7 (phase 10). The compiler's map lowering emits the early return that A1 forbids (`kernel-lower.ts:756-767`), so a loop-carrying map cannot convert before C7. Further steps: the non-sparse kernels export the constants their hosts copy (`hough.shade.ts`: `MAX_SEGMENTS`, `SEGMENT_STRIDE`, `TIE`; `tsdf.shade.ts`: `EMPTY`), and the hosts import them; the three `chunkedPass` copies become one helper in `device`, with the TSDF floor and the planar split as parameters. Phase 9 touches no sparse-scene kernel: the inverse-depth conversion and the four test-only triangulation entries move to phase 10, after R03. | CPU-tier outputs equal before and after on the family's inputs (`test:shade`); the family's WebGPU test unchanged (`test:webgpu`); no new readback and no new submission in the plan report; an esbuild bundle check that a host's import of a shader constant resolves; a run on the RTX 5060 Ti while A1 is open (item 11). |
| 10 Kernels, after the compiler changes | — | Behaviour; outside rule 1; one pull request per step, each after its compiler change. A step that edits a sparse-scene kernel also waits for R03 (rule 6). **C1:** one BA module, after R03. **C3:** the hand-written compactions of SIFT, ORB, KLT, matching and MVS become `push` where the compiler holds the count on the device, and so does the pair-preparation copy of the scan after R03; `scan.shade.ts` stays the device layer's scan until an amendment of `engine-v2.md` retires it (item 14). **C2:** the 18 oracle twins go only if item 10 is decided yes, the 12 sparse ones after R03. **C6 and C7:** the loop-heavy maps (SIFT orient and describe, the global descriptor, the scalar matchers, k-NN) and the entries that phase 9 left (SLIC, mesh, `planarFit`, the KLT maximum, the Hough vote and collect) convert. **After R03:** the inverse-depth conversion (struct arrays drop the host packing), the sparse kernels' exported constants, and the four test-only triangulation entries removed or kept with a reason (item 12). **Entry points:** they go live when the owner says so (items 9 and 13). R03 itself retires the object-model protocol and `runIncrementalGpu`, under its own records. | C1: every BA digest and WebGPU BA test unchanged. C3: row order and counts bit-identical; no new readback. C2: WebGPU unchanged; the CPU digests renewed once, in item 10's record. C6 and C7: phase 9's gate plus a hardware run. After R03: phase 9's gate; the tests of removed entries named in the pull request. Entry points: the surface bake of `typecv` and `typecv/gpu` unchanged; each new entry's surface baked; every deep path still resolves. |
| 11 Dense | — | The audit of `src/mvs/`, a read-only record like the three of 2026-10-26, then its phases by amendment of this document. | The audit record states its configuration, method and checks; the amendment names the change and the gate of each phase; the document audit of rule 7. |

**Proposal** (the order). Phases 1, 2 and 3 run one after the other, in the order of rule 5. They
share files:

- all three edit `sfm.worker.ts`: the `.ts` extensions, the preset and the inline rules;
- phases 2 and 3 edit `SfmLab.tsx`: the pseudo options and `buildRunRecord`;
- phase 1 moves `incrementalControls`, which phase 2 turns into `sfm/config`.

Phases 4 and 6 then run in that order, since both edit the GPU drivers and `gpu/mapper.ts`.
Phases 5 and 7 run in that order too, since both edit `sfm.worker.ts` and `SfmLab.tsx`. The two
pairs share no file and may run side by side. Phase 8 waits for the owner. Phase 9 touches no
sparse-scene kernel and may run beside phases 4 to 7. Phase 10 waits for the compiler changes,
and its sparse-kernel steps for R03 too. Phase 11 waits for the dense audit.

## 6. Compiler changes this programme asks for

Each is a proposal in TypeShade's `changes/`, by Rule 3.3, or an issue where the row says so. C13
may instead be a change of TypeCV's device (audit lab, L8). The line counts are the kernel
audit's estimates (**Inference**) of what each change removes from TypeCV.

The compiler probes of the kernel audit (section 0.1) cover C1 (P4), C3 (P6), C6 (P9b, P20, P30),
C8 (P1) and C9 (P11, P20). C2, C4, C5 and C7 rest on the compiler's rules and source
without a probe: Rule 8.24, change 0074, surface §65 and `kernel-lower.ts:756-767`. C10 to C13
rest on the lab audit's L1, L3, L4 and L8, also without a probe.

| # | Limit (**Fact**) | Effect in TypeCV | Change asked | Lines |
| --- | --- | --- | --- | --- |
| C1 | `TS8020`: a storage array cannot be a helper's parameter, so a helper that reads a binding belongs to the file that declares it | the fixed and shared-focal BA kernels repeat about 900 of their lines with different record sizes | a helper that takes a storage array by reference, or a module parameterized by constants that a file instantiates twice; without a compiler change, a shared bindings module with one control struct (a layout change), within the 8 storage buffers per stage that WebGPU guarantees (audit sparse, L4) | ~900 |
| C2 | the CPU tier runs one invocation per call and cannot run a barrier (Rule 8.24), although the compiler's own oracle `dispatch` can | 18 `...Oracle` twins, the MVS splits and 29 backend branches; the oracle proves the twin, not the tiled entry, and 10 twins add in another order than the GPU's tree | the host CPU tier of an entry runs barrier entries in passes, as `dispatch` does | ~250 |
| C3 | a kernel function's trip count is computed on the host, and a reduction result or a `push` count reaches only the host | `scan.shade.ts`, its copy, and about 25 flag, scatter, advance and mark entries keep counts on the device by hand | a `push` that appends after a count held in a `Resident` and leaves the new count there; a reduction into a `Resident`; a loop bound read from a `Resident`, lowered to an indirect dispatch | 300–400 |
| C4 | the compiler does not cut a loop into timed chunks (change 0074) | `chunkedPass` three times, `cappedPass`, and per-module chunk generators | a kernel call that splits a proved loop into chunks under `timeBudgetMs` | ~100 |
| C5 | `write(value)` replaces a whole `Resident` | `mvsCopyF32`, `mvsCopyU32` and their staging buffers copy a view into a slot | `write(value, { offset })` on the queue | ~40 |
| C6 | the proof refuses `out[base + i]` with a loop-invariant `base`, several fields at a runtime stride, a nested affine record and two record forms in one loop (rule R3 of 8.22) | several maps cannot convert as written | accept an injective offset, offsets `c < s` checked at the call as `a·i + c` is, and nested affine records | enabler |
| C7 | the map lowering starts every invocation with an early return past the trip count; A1 (`hardware-observations.md`, TypeShade #703) forbids that return on the RTX 5060 Ti | loop-heavy maps must stay entries written clamp-and-mask | an iteration scaffold that clamps the index and masks the writes | enabler |
| C8 | a kernel call takes no workgroup argument and no indirect count | 12 entries dispatch indirectly and cannot become kernel functions | a kernel call whose trip count is read from a `Resident` (shares C3's lowering) | enabler |
| C9 | surface §65 lists `min` and `max` among the scatter forms; the compiler refuses `a[k] = min(a[k], e)` | a conversion that needs it fails | a TypeShade issue: the form the surface states, or a surface correction | bug |
| C10 | timestamp profiling is fixed when a device is created | the worker swaps the device for the dense stage and merges statistics across devices | a per-plan or per-scope switch for timestamp writes | lab |
| C11 | a device loss names no in-flight work | the lab tracks the last kernel itself | a loss record that names the plan or kernel submitted and not completed | lab |
| C12 | chunk rates live as long as the device | a record on a reused device includes earlier runs | a per-run epoch or snapshot of `chunkRates` | lab |
| C13 | GPU time cannot be attributed to stages inside one library call | the job diffs kernel statistics; view-graph stages carry no `gpuMs` | a named profiling scope that labels plan reports, in the TypeShade runtime or in TypeCV's device | lab |

Not gaps (**Fact**, from the kernel audit's probes and rules):

- 2D and 3D kernel loops (TypeCV uses only `.x`);
- f64 in kernel functions;
- integer scatter atomics;
- workgroup memory in kernel functions (the entry is the remedy);
- kernel-to-kernel calls (`fuse` composes calls on the host);
- exported constants for the host (surface §64).

**Inference.** TypeCV does not use the exported constants, a facility not taken. Phase 9's
esbuild bundle check confirms the import. Device-resident hashing and an on-device sort are gaps
that remove no lines today and are not asked for.

## 7. Decisions

The owner's go-ahead of 2026-10-26 fixes each item at the choice this document proposes. An
item without a proposed choice stays open until the phase that needs it, and its decision is
recorded here by amendment.

1. **The areas and their names** (section 3.1). Which directories and entry point names, and do
   `sfm` and `mvs` sit as two areas or under one `reconstruction`?
   **Decision (2026-10-26).** Decided: the directories and entry point names of the table in section
   3.1; `sfm` and `mvs` are two areas.
2. **White-box tests.** May the structure pull requests rewrite the tests that depend on internal
   names and source text (`dense-gpu-mode-worker.test.mjs`, `working-budget.test.mjs`), keeping
   what they check?
   **Decision (2026-10-26).** Decided: yes. The structure pull requests rewrite the two tests and
   keep what they check.
3. **Which mapper survives:** converge on `sceneSteps` (the R03 path) and retire
   `incrementalSteps`, or keep both until R03 lands? Phase 8a needs this.
   **Decision (2026-10-26).** Decided: converge on `sceneSteps`. `incrementalSteps` stays until
   parity (phase 8a) and retires with R03.
4. **Where the lab's stage modules and the lab preset live:** in a library area (so scripts and
   Node tests share them) or in `website/src/lib/`? Phases 2 and 8b need this. **Inference.** The
   placement changes no record, since phase 2's gate holds the record's `options` identical.
   **Decision (2026-10-26).** Decided: the modules a script needs go to a library area; a module
   that only the page needs stays in `website/src/lib/`.
5. **Behaviour differences found by the audits**, each its own pull request if fixed:
   - the component driver's error rule;
   - `finalBundleIterations` 25 against 8, and stall 1e-4 against 1e-9;
   - the pair minimums in mixed units (8, 16 and 16 matches);
   - the multi-video record's literal `clahe: false` and `keyframeGate: 'phase'`;
   - the defaults that disagree between the page and the worker: gate, edge, guided, octave;
   - the dense fusion and clean defaults, written in the page and in `dense-run.ts` beside unused
     worker defaults;
   - the KLT capacity: 512 in the lab against 1024 in the library;
   - two devices in one worker after a single-video run;
   - the session's pair-verification seed rewrite;
   - the GPU refusal of focal groups (`gpu/bundle.ts:1037-1041`).

   **Decision (2026-10-26).** Open. Each difference gets its own record and pull request when its
   phase reaches it. The structure phases keep today's behaviour (rule 1).
6. **Code without a product caller:** `posed-evidence.ts` (329 lines) and the observation-expansion
   family (1,692 lines, exported from `typecv/gpu`). Keep, move to `experimental/`, or delete? A
   move or a deletion of the family removes exports of `typecv/gpu` (`src/gpu/index.ts:60-73`).
   Rules 1 and 4 of section 4 and the surface gate exclude that before the next major version.
   **Decision (2026-10-26).** Open. Until decided, both move with their area and keep their exports
   (rules 1 and 4).
7. **Compiler changes** (section 6): which to open as TypeShade proposals now.
   **Decision (2026-10-26).** Decided: C1, C2, C3 with C8, C6 and C7 open first as TypeShade
   proposals, and C9 as an issue. The others open when a phase needs them.
8. **Freeze or sequence.** Functional work on the files of a phase stops while the phase is open,
   or follows it.
   **Decision (2026-10-26).** Open. Decided before phase 1 opens.
9. **The front end (`design-rules.md` §4) and the package path (§5.1):** decided separately, or
   together with the area entry points of phase 10? The `typecv/<area>` names of section 3.1
   presuppose the path `typecv`.
   **Decision (2026-10-26).** Open. Decided with the go-ahead of phase 10's entry points.
10. **Floating-point reductions as kernel reductions** (the observation summary, the BA metrics,
    the PnP and consensus scores). **Inference.** The CPU tier and WebGPU would then agree bit for
    bit. Today's numbers and the recorded digests (`PREVIOUS_RESULTS` of the pair-batch and
    PnP-batch tests) would change, and a tie in a score could pick another model. It is a
    behaviour change under Rule 2.2, not proposed here. The same decision covers deleting the 18
    oracle twins after C2, which renews the CPU digests once. Phase 10 deletes them only if this
    item is decided yes.
    **Decision (2026-10-26).** Decided: not now. The reductions and the oracle twins stay; a later
    record may propose the change.
11. **The hardware gate of the kernel phases.** Phases 9 and 10 need a run on the RTX 5060 Ti
    while A1 is open, and CI has no such device. Who runs it, and on which samples? R02 asks for
    runs on the same device.
    **Decision (2026-10-26).** Open. Decided before phase 9's first step: who runs it, and on which
    samples.
12. **`EpipolarSession`**, the only kernel-function path today, called by tests and one
    benchmark: public or not? Making it non-public removes an export of `typecv/gpu`
    (`src/gpu/index.ts:102`), which waits for the next major version. **The four test-only
    triangulation entries:** removed in phase 10, or kept with a reason?
    **Decision (2026-10-26).** Open. Decided before phase 10's steps after R03.
13. **(a) The build and packaging change** (section 2, the build split; section 3.1; phase 1a).
    The build selects `gpu/` subdirectories instead of `src/gpu` alone, and five configuration
    files change. The `dist/` file list changes by the moved paths, and the surface bake holds
    both entries. The same item decides the form of the area entries in phase 10:
    - `typecv/<area>` exports the area's CPU half, and `typecv/gpu` keeps every GPU half
      (proposed; the boundary that `architecture.md` describes today); or
    - `typecv/<area>` is an entry of the GPU bundle that exports the whole area, beside the
      CPU-only root. This form is closer to the "one entry point per area" of `engine-v2.md`.

    **Decision (2026-10-26).** Decided: the first form. `typecv/<area>` exports the CPU half and
    `typecv/gpu` keeps the GPU half; phase 1a changes the five configuration files.
14. **(b) Amend `engine-v2.md` for the shared scan?** `engine-v2.md` keeps the shared scan and the
    partial reductions as the device layer. The resident scene derives its CSR orders and
    compaction from that scan. Phase 10 converts only the hand-written compactions. Retiring
    `scan.shade.ts` for `push` needs this amendment.
    **Decision (2026-10-26).** Open. Needed only to retire `scan.shade.ts`; phase 10's C3 step
    converts the hand-written compactions without it.
15. **(c) `src/gpu/internal.ts`:** keep Amendment 6 of `multi-video.md`, or close `internal.ts`?
    The default keeps it: `internal.ts` stays, and the moved stage modules are exported from it
    only. Closing it makes its symbols public or changes the tests, scripts and pages that import
    it.
    **Decision (2026-10-26).** Decided: keep Amendment 6. `internal.ts` stays.
16. **(d) The resident-scene protocol in phase 8a.** Moving the BA subset policy into the mapper
    changes the `bundle` request of `gpu-resident-scene.md`. The mapper would decide the subset,
    and the result would no longer report whether it was used. Amend that document in 8a, or keep
    the policy in the driver?
    **Decision (2026-10-26).** Open. Decided before phase 8a.
17. **(e) The programme against R01 to R03** (`reconstruction-status.md`) and the order of
    `engine-v2.md`: before them, after them or beside them? R01 and R02 are verification and
    measurement work. R03 changes the mapper and the GPU drivers that phases 4, 6 and 8
    restructure.
    **Decision (2026-10-26).** Open. Phases 0 to 9 wait on none of R01 to R03 (section 2); the order
    among them is the owner's.
18. **(f) The registration driver** (audit lab, question 6): how much should the programme invest
    in a `registration` stage around `IncrementalSfmSession`, which `engine-v2.md` replaces?
    **Decision (2026-10-26).** Open. Decided before phase 5b.

## 8. Verification

**Proposal.**

- Functional verification is the gate of section 5 on every phase. The local gate covers the CPU
  tier, since the sub-agent container has no Chromium. The CI job of phase 0 covers the GPU tiers:
  it compares the digests of the base and the head on one runner. `test:webgpu` and the browser
  half of `test:website` also run in CI only. Native GPUs are not in CI (`physical-gpu-runs.md`).
- A kernel conversion (phases 9 and 10) is proved as the kernel audit's section 5 says.
  **Inference** (that section): a map, an exact fold and a compaction change no bits. So old and
  new CPU-tier outputs are equal, and the family's WebGPU test stays. A floating-point sum or a
  deleted oracle twin changes bits and is a decision (section 7, item 10). A chunked reduction
  keeps one fold order per call.
- The document audit of rule 7 is separate from the tests. It checks that the living documents
  name the moved symbols, that the docs index lists what exists and that the website's generated
  pages link.
- Status accounting: each phase's pull request records what moved, which gate ran on which
  commit, and every deviation from this document with its disposition. This document gains a
  line in section 9 per accepted amendment.

## 9. Record of this document

- 2026-10-26: written from the three audits of the same day; proposed (#241).
- 2026-10-26: revised after the review of #241 (the configuration and counts, the build split,
  the layers, the phase scopes and gates, decisions 13 to 18).
- 2026-10-26: accepted (the owner's go-ahead in the session, #241), with the dispositions of
  section 7.
