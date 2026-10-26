# TypeCV library structure and the refactoring programme: proposal

Status: accepted, 2026-10-26 (the owner's go-ahead in the session, #241). This document is the
record that every refactoring pull request cites. Section 7 records the disposition of each
decision. An open item is decided before the phase that needs it. It changes no code and no
behaviour. Amendments 1 to 3 (section 9) record what the work of 2026-10-26 established. Labels:
**Fact** (read in the code, a document or a measurement), **Inference** (reasoned from facts),
**Proposal** (a choice this document asks the owner to make).

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

**The glob** (amendment 1). **Fact** (TypeShade proposal 0090, "What it owes downstream").
`.prettierignore` can select the kernel files by the glob `*.shade.ts` instead of 27 paths.
Prettier still cannot parse a decorated top-level function. **Fact** (`main` at `0f28d8d`). The
glob also matches three kernel files that the 27 paths leave out: `src/gpu/epipolar.shade.ts`,
`src/gpu/numerics.shade.ts` and `src/gpu/projective.shade.ts`. `format:check` checks them today,
and `npx prettier --check` passes on them. **Inference.** The glob takes those three out of
`format:check`. Section 6, row C14 records 0090 itself.

**The delivered glob** (amendment 3). **Fact** (#247, merged as `3e2777b` on 2026-10-26).
`.prettierignore` holds the glob `src/**/*.shade.ts` and three negated lines that name no
directory: `!src/**/epipolar.shade.ts`, `!src/**/numerics.shade.ts` and
`!src/**/projective.shade.ts`. `npm run format:check` checks the same 706 files before and after
#247, the three kernel files among them (#247, "Gate"). **Deviation.** The inference above, "The
glob takes those three out of `format:check`", does not hold for the delivered form. The negated
lines keep the three files checked. **Disposition.** Accepted by this amendment. The delivered form
keeps the checked files as they were, which rule 1 (structure only) prefers to a change of
coverage. **Inference** (#247). A line without a directory keeps each file checked when phase 1b
moves it into a `gpu/` subdirectory.

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
  Every moved file leaves a re-export at its old path until then. A moved kernel file may leave
  one too (section 4, rule 4, as amendment 1 corrects it).
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

**What 0094 asks of this list** (amendment 3). **Fact** (TypeShade change 0094, `status:
accepted`, "What changes"). 0094 gives a kernel function three forms. They are a trip count held
on the device, a `push` after a length that a `Resident` holds, and stores after the loops.
**Inference** (0094, "What it owes downstream"). For those forms it removes two needs of the list
above: a device-side count read by a later dispatch, and indirect dispatch. An entry that also needs
workgroup memory or a specific invocation-to-work mapping stays an entry. **Fact** (the same
section). 0094 asks that the list be amended before a conversion. **Fact** (#246). 0094 is not
implemented at the pin `725b2969`. So the list stands as written until a pin carries 0094's
implementation.

**Fact** (the kernel audit, sections 2 and 3, and the kernels at `3e2777b`). These entries exist
only to keep a count or an offset on the device, or to move one:

- the compactions that phase 10's step C3 names. Each entry flags the elements to keep for the
  shared scan, or scatters them after it:
  - SIFT: `siftFlagExtrema`, `siftScatterExtrema` and `siftScatterOriented`;
  - ORB: `orbFlagCorners`, `orbScatterCorners`, `orbScatterFast` and `orbEmit`;
  - `features.shade.ts`: `flagFeatures`, `scatterFeatures`, `flagMutualBrief` and
    `scatterMutualBrief`;
  - float matching: `flagFloatMatches` and `scatterFloatMatches`;
  - KLT: `kltCompact`;
  - MVS fusion: `mvsFuseFlags`, `mvsScatterPoints`, `mvsFuseEmit` and `mvsFuseScatterSums`;
- the steps of one invocation that move a count: `siftAdvance`, `siftMark`, `orbAdvance` and
  `orbMark`;
- the shared scan: `scanFlags`, `scanFlagsOracle` and `scanFlagBlocks` (`scan.shade.ts`);
- the scan copy of the pair preparation: `scanPairRows`, `scanPairRowsOracle`, `scanPairBlocks`
  and `compactImagePair` (`pair-preparation.shade.ts`).

Not in the list: `meshClassify`, `meshEmit` and `houghCollect`, whose counts the host reads anyway
(the audit, section 3). The same holds for `kltEigenMax` and `kltEigenMaxFinal`, whose maximum the
host reads. `push` and a kernel reduction serve them without 0094, and the phase 9 row holds them
for C7.

**Proposal.** This applies after a pin that carries 0094's implementation. A count that a later
dispatch reads, and an indirect dispatch, then force no entry where one of 0094's three forms
expresses them. The entries of the first two groups then convert in phase 10's step C3, under its
gate. `kltCompact` and the four MVS fusion entries are written clamp-and-mask today. So by the
phase 9 row's reading of A1 they also wait for C7. The shared scan stays in the device layer until
section 7, item 14 is decided; 0094 does not need that decision. The scan copy, the audit's 12
indirect entries (its section 2) and the planners of their counts (`planPairModelBatch`,
`summarizePairModelWork` and `planPnPBatch`) are sparse-scene kernels. They wait for R03 (section
4, rule 6), and each is read against the list then. **Inference** (0094, "What it owes
downstream"). A step of one invocation that only moves a count becomes a store after the loops of
the function that computes the count. A step without a loop depends on 0094's decision D7.

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

   **Fact** (amendment 1; #242, CI job `check (24)` on `09894a7`). CPU-tier identity across Node
   majors is not established for the focal-search path. Two digests of `scene-90x6-focal-1.2`
   differ between Node 22 and Node 24: `reconstructIncremental/scene-90x6-focal-1.2/focal-search`
   and `focalSearchSteps/scene-90x6-focal-1.2`. Every trace digest and request count matches.
   **Fact** (amendment 3; the investigation of 2026-10-26, below). This replaces the inference of
   commit `5f02df4` of #242. V8 13.6, the V8 of Node 24, has the flag `--use-std-math-pow`, on by
   default: `**` and `Math.pow` call the C library's `pow`. V8 12.4, the V8 of Node 22, has no such
   flag. The site is `focalGrid` (`src/reconstruction/focal-search.ts:184`, `controls.step ** i`).
   On Node 24.21.0 the two digests are `418f1d2e…` and `35c95321…`. With `--no-use-std-math-pow`
   they are `ee204b95…` and `8d4aebdf…`, the Node 22 fixture's. The traces are equal either way,
   and the two `scene-90x6-distorted/global` digests do not change with the flag. **Fact** (#242 at
   `252f5e5`). So each digest fixture records its configuration: the Node and V8 versions, the
   platform, the commit and the git tree of `src/` (`tests/typeshade/digest-fixture.mjs`). The
   fixtures of #242 were recorded on Node 22.22.0. The `check` job compares them only on the Node
   major that recorded them. The compare job of phase 0 runs the digests on base and head on every
   Node version of the matrix: `digests (22)` and `digests (24)`.

   **Deviation** (amendment 1). The approved text of phase 0 says: "The CPU-tier digests and the
   baselines are recorded in `tests/`." They are recorded, and each fixture binds the Node major
   that recorded it. Disposition: accepted by this amendment.

   **Deviation** (amendment 2; #242, "Deviations from the approved text", item 2). The approved
   gate runs the tests through `npm test` (section 5, job `check`). #242 reads it as putting the
   fast digests there. They run in `test:digests` instead: in the `check` job on Node 22, and in
   the `digests` job. The cause is the Node-major binding of amendment 1. Disposition: accepted by
   the owner's go-ahead in the session on 2026-10-26, and recorded by this amendment.

   **Fact** (amendment 3; #246, CI job `digests (24)` 121668286140 on `ba5cb5f`, Node 24.21.0).
   The step "Digests of the head" failed the self-check of `characterize`
   (`tests/characterization.ts:584`) in two cases of `tests/digests/cpu-scenes.test.ts`:
   `reconstructIncremental/scene-90x6-distorted/global` and
   `reconstructIncrementalScene/scene-90x6-distorted/global`. In one process, the entry point's
   output differed from the traced run's. `digests (22)` was green on the same head.

   **Fact** (amendment 3; the investigation of 2026-10-26, a read-only sub-agent's report to the
   coordinator). It compared the head `ba5cb5f` (pin `725b2969`) with the base `091a3e9` (pin
   `d36ac502`), whose `src/` is the same tree, `4c577995`. It ran three Node builds in a shared
   container with 4 vCPU: 24.21.0 (V8 13.6.233.17-node.53, the build of CI), 24.22.1 (V8
   13.6.233.17-node.58) and 22.22.0 (V8 12.4.254.21-node.33). The command of CI was
   `TYPECV_DIGEST_OUT=<directory> node --test --test-name-pattern='scene-90x6-distorted/global'
   tests/digests/cpu-scenes.test.ts`, in a worktree of the commit.

   - The failure came once: 1 of 223 fresh processes of that command, on the head and Node
     24.21.0. The second and third computations of the process were off, and the first and fourth
     normal. The off digests differ from CI's. No failure came in 610 more runs with an in-memory
     patch that saves the outputs, or in 40 runs with flags. That is 1 in 873 in all. No off output
     was captured.
   - Every other run gave the fixture's digests. These were the head on Node 24.22.1 (3 of 3) and
     on Node 22.22.0 (1 of 1), and the base on Node 24.22.1 (3 of 3). About 220 runs of standalone
     scripts or of a copy of the test gave them too.
   - In CI on Node 24.21.0, `digests (24)` was red on `ba5cb5f` only. It was green on `252f5e5`
     and `b2917db` (#242), on `0d34ef5` and `fab38b9` (#246) and on `d037770` (#247). That is
     about 1 failure in 11 Node 24 processes of this file. `digests (22)` was always green.
   - The pin is not in the process. `git diff --stat 091a3e9 ba5cb5f` lists `package.json` and
     `package-lock.json` only. A resolve hook listed the modules of the test process: 42 under
     `src/`, 8 under `tests/` and Node built-ins, none from `node_modules`, `typeshade` or `dist/`.
     In CI, the two test files beside it import only `tests/characterization.ts` and
     `tests/typeshade/digest-fixture.mjs`, and `npm run build` had ended. So no code that differs
     between the base and the head ran on the runner then.
   - The 42 modules read no clock, call no `Math.random` and hold no mutable state at module level.
     The digest encoder gave equal bytes for 200 encodings of one output. The off computation was
     the direct run in one case and the traced run in the other. So the disagreement follows the
     position in the process, not the driver.
   - The tiers: 25 intrinsic forms on 100,000 inputs each, forced to the interpreter, Maglev and
     Turbofan, gave identical bits. So did a copy of `roots()` of `src/geometry/p3p.ts` in 7 tier
     configurations.
   - Flags, 10 runs each of the command of CI: `--no-concurrent-recompilation`, `--no-opt`,
     `--no-maglev` and `--single-threaded` each passed 10 of 10. About 40 further configurations
     were normal. At 1 in 873, these counts cannot show that a flag removes the failure.

   **Inference** (the same report). The failure is a nondeterminism of the V8 13.6 runtime inside
   one process. A code version that exists only for a while computes some value differently, and
   its existence depends on the timing of concurrent compilation. Node 22 has Maglev and concurrent
   Sparkplug off by default and shows no failure. This fits a Maglev or concurrent-tiering cause
   but does not prove it. The rate grows with CPU contention: CI runs three V8 processes on 4 vCPU.
   The pin did not cause the failure, and the green runs of #242 were the common outcome, not luck.
   So the two `global` cases are not deterministic on Node 24.21.0.

   **Proposal** (amendment 3). The `digests` job runs its two `node --test` steps on Node 24 with
   `--single-threaded`. **Inference.** V8 then compiles and collects on the program's thread, at
   points that the program determines. So a tier-up comes at the same point in every run. Node 22
   runs as before. `runConfiguration()` of `tests/typeshade/digest-fixture.mjs` records
   `process.execArgv`, and `describe` of `scripts/compare-digests.mjs` prints it. The self-check of
   `characterize` stays on every Node major. **Fact** (checked on 2026-10-26 with a test file that
   prints `process.execArgv`, on Node 24.21.0 and 24.22.1). `node --single-threaded --test` passes
   the flag to the process of each test file.

   Considered and not taken:

   - the self-check on Node 22 only: the compare of base and head also runs on Node 24, and it
     stays exposed;
   - a second run of a failed job: it is not a fix, and the gate stays nondeterministic;
   - a change of the library: no evidence supports one, and a numeric change is a behaviour change
     under rule 1;
   - `--no-use-std-math-pow` on Node 24, or a `focalGrid` without `**` (its own record under rule
     1): either hides a real platform difference. The fixtures already bind that difference to the
     Node major. They are a gate of identity on one runner, not of equality across majors.

   **Disposition** (amendment 3). The owner accepts the proposal by merging this amendment. The
   gate change is its own pull request, #249. The fix is established when 32 consecutive
   `digests (24)` jobs are green. **Inference.** At an unchanged rate of 1 in 11 processes, 32 green
   processes in a row have a chance of (10/11)^32, about 0.047. A `digests (24)` job runs the file
   at least once, so 32 green jobs give at least 95% confidence that the rate has fallen. Until
   then, a red `digests (24)` on an unrelated change is run again once, and the run is recorded on
   its pull request. A V8 bug report follows when an off output is captured.

   **Deviation** (amendment 3). The gate of this rule is identity, and on Node 24 it was not
   deterministic from #242 to the gate change. **Disposition.** Open until 32 consecutive
   `digests (24)` jobs are green after the gate change.
3. **One pull request per step named in section 5**, citing this proposal by section. A phase
   with several steps is several pull requests. The first commit goes up early as a draft (the
   owner's rule), and the pull request is ready when its gate is green.
4. **Aliases.** A moved file leaves a re-export at its old path until the next major version.
   The 74 `src` modules that tests, scripts and the website import by deep path on `5e1c95c` keep
   working through them. A moved kernel file may leave an alias too.

   **Fact** (amendment 1; TypeShade proposal 0090, "Measurements of 2026-10-26"). A shader module
   whose only statement after the directive is `export * from "./new/k.shade.ts"` compiles. Its
   generated module is byte-identical to its target's. Its host view carries the target's
   `@compute` entry. The measurement compiled the alias with `hostFace`; Vite did not bundle it.
   **Inference** (the same proposal). A bundle that reads both paths holds two copies of the
   compiled module. So phase 1b may keep an alias for a moved kernel file. The 23 files under
   `tests/` and `scripts/` that name a kernel path need not change in the same pull request.
   This corrects the inference of the accepted text that a kernel file leaves no alias.
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
   the CPU tier. The sub-agent container may lack Chromium. **Fact** (amendment 1, observed on
   2026-10-26). The container had Playwright's Chromium in `/opt/pw-browsers`
   (`chromium_headless_shell-1243`), which the browser tests launch through `TYPESHADE_CHROMIUM`.
   The lab-side phase 0 tests ran their browser half locally (the coordinating session's report).
   So `test:webgpu`, the browser half of `test:website` and the GPU-tier digests run in CI, and
   locally where the container has Chromium. The coordinator runs the local gate again and reads
   the CI result before a phase is marked ready.

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
| 0 Characterize | 3–5 | Tests and tools only, in five parts. **Digests:** canonical digests (Float64 bit patterns) of the full output of `reconstructIncremental`, `reconstructIncrementalScene`, `refineIncremental`, `reconstructComponent`, the focal search and `IncrementalSfmSession`, on the CPU device and on SwiftShader, on the `tests/scene.ts` scenes and the four samples that `scripts/measure-focal.mjs` reads (`south-building`, `gerrard-hall`, `person-hall`, `graham-hall`); digests of the emitted WGSL and GLSL per kernel module; a per-request trace of each driver. **Baselines** of the later gates: plan reports (kernel names, dispatches, submissions) and step-name sequences of the GPU drivers; golden run records (single, multi, failed, dense) with `createdAt` masked; the output of `scripts/measure-focal.mjs`; the start payloads and record options of about 40 query strings. **Lab transcripts:** a test that bundles the worker with esbuild, drives it through `onmessage` with a deterministic fake runtime and records every message and every `session.add` input for the gate, legacy, feature-minimum, guided, KLT, ORB, precomputed, dense, failure and loss cases; the same for `MultiVideoJob.finish`. **Tools:** `scripts/api-surface.mjs` writes the exported names and declaration text of `typecv` and `typecv/gpu` to `tests/api-surface.txt`; `scripts/import-graph.mjs` prints the runtime import cycles of `src/` and `website/src/` and the imports that break the layering rule of section 3.1. **CI:** a job that builds the base and the head of a pull request on one runner, runs the GPU-tier digest tests on both and compares them (audit sparse, P0). Amendment 1: the job runs on every Node major of the matrix, `digests (22)` and `digests (24)`, and compares the CPU-tier digests too. Amendment 3: on Node 24 the job runs its two `node --test` steps with `--single-threaded`, and each digest file records the flags of its process (section 4, rule 2). | Passes on unchanged code. The CPU-tier digests and the baselines are recorded in `tests/`. Amendment 1: each digest fixture records its configuration and is compared only on the Node major that recorded it (section 4, rule 2, the deviation and its disposition). Amendment 2: the fast CPU-tier digests run in `test:digests` and the `digests` job, not in `npm test` (section 4, rule 2, deviation 2 of #242). The CI job reports equal digests when base and head are one commit. The cycle search prints today's three cycles. |
| 1 Moves | 3–5, three pull requests | **1a Build selection** (decision (a)): `tsconfig.json`, `tsconfig.build.json`, `tsconfig.gpu.json`, `scripts/build-gpu.mjs` and `.prettierignore` select the `gpu/` subdirectories of section 3.1 besides `src/gpu/**`; no file moves. **1b Moves:** the area directories; pure moves with re-exports (a kernel file may keep one too, section 4, rule 4, amendment 1); the two BA cycles broken (`validateReconstruction`, `cloneBundleReconstruction`, `bundleMetrics` and `bundleStallTolerance` out of `bundle-adjustment.ts`; `bundleStallMonitor` and `cooperativeBundleSchur` out of `gpu/bundle.ts`); `judgePair`, `incrementalControls`, the errors and `copyReconstruction` out of `mapper.ts`; the `pnp.ts` helpers to their own module; `.ts` extensions in the worker's imports; the cleanup of `tests/typeshade/compute-probe-build.mjs:71`, which deletes host views under `src/gpu` only, extended to the `gpu/` subdirectories as `scripts/build-gpu.mjs` does (amendment 3; #247, open item 2). The mapper–focal cycle stays: `incrementalSteps` and `focalSearchSteps` call each other, which no move can break (phase 4). **1c Cross-area imports**, resolved by moving the shared pieces of section 3.1: `chunkedPass`, `packRgba`, `kltLayout`, the object-model, camera, pose and mask types, `validateReconstruction` and `observation-paths.ts`. | 1a: the `dist/` file list and the surface bake identical before and after; `typecheck`, `build` and `format:check` pass. 1b: the surface bake unchanged; the `dist/` file list differs only by the new paths, with every old path kept as an alias; WGSL and GLSL digests unchanged; every deep import resolves; the two BA cycles gone, the mapper–focal cycle the only one left. 1c: `scripts/import-graph.mjs` finds no import that breaks the layering rule; the surface bake unchanged. |
| 2 One definition | 3–4 | `sfm/config` and the lab's `options` module hold each value once: the mapper defaults with the named per-entry differences kept (25 and 8), seed 17, the stall rules, the operator defaults shared by CPU and GPU operators, `observationKey` and `pairKey` with today's strings, the triangulation status enum, the two-view failure table, the host layout constants once per kernel. The lab preset sits in one module, placed by item 4 (`website/src/lib/` until then). Both workers, the page and the six scripts that copy it read it: `measure-focal.mjs`, `measure-multi-video.mjs`, `benchmark-sfm-pyramid-e2e.mjs`, `benchmark-rotation-tracking.mjs`, `benchmark-sift-clahe.mjs`, `measure-video-sfm-performance.mjs`. | Run record `options` identical; the output of `scripts/measure-focal.mjs` identical to phase 0's baseline; a test holds the preset to today's values. |
| 3 Record split and named policies | 3–4 | `run-record.ts` split into schema, validate, assemble and summary behind a barrel; `buildRunRecord` and `buildMultiRunRecord` move in with the browser globals as an `environment` argument; the probe record shares the validator helpers. Every inline lab rule becomes a named unit with its current value and unit: pair minimums (16 entries = 8 matches; 32 entries; 16 tracker matches), the feature minimum 8, the retention rule (0.5 s, 3, 3), the legacy and track decisions, the guided trigger, KLT cover. | Golden records byte-identical to phase 0's baseline with `createdAt` masked; v3 to v6 fixtures valid; transcripts identical; table tests copied from today's branches. |
| 4 Shared mapper parts | 3–4 | The pure parts the two mappers copy (masks, seed pick, initialization error, track index, resume adoption, ranking, failure and retry state, result assembly) move to `sfm/mapper/*`; both generators call them. One `drive` (sync and async) with an explicit error rule, the component driver keeping its current rule as a named option; one depth post-step; one bundle-stage telemetry builder; the execution-delta and report-merge helpers; the record packers with today's layouts. The mapper–focal cycle is broken by injection: `incrementalSteps` passes itself to `focalSearchSteps`, so `focal-search.ts` no longer imports `mapper.ts` at run time. | Phase 0 digests of both mappers; request traces identical; plan reports identical to phase 0's baselines (kernel names, dispatches, submissions); the cycle search finds no cycle. |
| 5 Worker by stage | 5–7, four pull requests | 5a the lab's pure helpers (audit lab, P1): budgets, feature cap, `medianMotionPx`, `trimStages`, the frame camera, colours, match lines and landmark colours, each with its variants kept (the worker's persistent colour cache, the job's per-call cache). 5b `SingleVideoJob` over the stage modules (as `MultiVideoJob` is), the worker keeping intake and the device. 5c `lab-device` for the five open sites and four loss handlers. 5d `lab-protocol`, `start(message)` instead of 23 positional parameters, typed `request<T>()` on the page. | Transcripts identical; table tests for each helper of 5a; the white-box hooks of `dense-gpu-mode-worker.test.mjs` and the source-regex test of `working-budget.test.mjs` rewritten to check the same thing (item 2); the browser tests in CI; new Node tests of `SingleVideoJob` with a fake port. |
| 6 Long functions | 3–4 | The GPU host functions split without a change to the device steps: request checks, step generators and unpack of `checkPairEstimation`, `estimatePosePnPGpu`, `triangulateTracksGpu`, `evaluateSceneObservationsGpu`; the host half of the two BA routes once where it is verbatim; `IncrementalSfmSession.add` and `update` reduced to pure policies (`evictWindow`, `chooseAnchors`, `carryAnchorEdges`, `archiveSolvePlan`) with unit tests. | Step-name sequences and plan reports identical to phase 0's baselines; WGSL unchanged; session digests on the CPU device and, in the CI job, on SwiftShader. |
| 7 Page controller and options schema | 4–5 | `lab-run` takes the run control that `runSfM`, `runMultiVideo` and `runBenchmark` copy (React keeps the state setters). `run-options` replaces the nine `*Override()` helpers (eight in `SfmLab.tsx` and `decodeOverride` in `decode-protocol.ts`) and the scattered parser calls with one table; the values the page sends stay the same. | Table test over about 40 query strings: start payloads and record options identical to phase 0's baseline; a Node test of the controller against a scripted fake worker (audit lab, P5); transcripts and golden records identical. |
| 8 Boundaries (owner's go-ahead) | 3–5, two pull requests | Behaviour; outside rule 1. **8a Mapper boundary** (items 3 and 16): BA window, subset, partition and depth-filter policy move from `HostSceneDriver` into the mapper. This changes the accepted `bundle` request of `gpu-resident-scene.md`: the mapper decides the subset, and the result stops reporting it. So 8a carries that document's amendment. `sceneSteps` reaches parity with `incrementalSteps` (the six missing features and `verifiedPairs`) from the phase 4 parts; a typed request-to-result map. **8b Lab placement** (item 4): the lab's stage modules move to the area that item 4 names, exported from `src/gpu/internal.ts` only (Amendment 6); the scripts that copy the pipeline are ported to the stages, so their outputs change (audit lab, P7). | Parity gates, each behaviour change with its own record. 8a: `sceneSteps` with `HostSceneDriver` equals `incrementalSteps` with the CPU driver bit for bit for every phase 0 option set; the `incrementalSteps` digest unchanged; the `reconstructIncrementalScene` digests of the option sets that use the six features renewed once, in 8a's record; `test:shade` and `test:webgpu`. 8b: transcripts identical; a CPU-tier run of `SingleVideoJob` on a tiny synthetic sequence compared with WebGPU under Rule 1.2; each ported script's new output recorded. |
| 9 Kernels, no compiler change | 2–3, one pull request per step | The conversions of audit kernel style, section 7, phase 1, narrowed by A1. An entry converts only if it has no per-lane loop and does not use the clamp-and-mask pattern today; the conversion then keeps its idle-lane shape. Checked in the kernels at `5e1c95c`: `planarPropose`, `planarVerify`, `planarFinish` and `claheApply` qualify, since they return early and run no loop. Every SLIC and mesh entry, `planarFit`, `kltEigenMax`, `houghVote` and `houghCollect` loop or clamp and mask, so they wait for C7 (phase 10). The compiler's map lowering emits the early return that A1 forbids (`kernel-lower.ts:756-767`), so a loop-carrying map cannot convert before C7. Further steps: the non-sparse kernels export the constants their hosts copy (`hough.shade.ts`: `MAX_SEGMENTS`, `SEGMENT_STRIDE`, `TIE`; `tsdf.shade.ts`: `EMPTY`), and the hosts import them; the three `chunkedPass` copies become one helper in `device`, with the TSDF floor and the planar split as parameters. Phase 9 touches no sparse-scene kernel: the inverse-depth conversion and the four test-only triangulation entries move to phase 10, after R03. | CPU-tier outputs equal before and after on the family's inputs (`test:shade`); the family's WebGPU test unchanged (`test:webgpu`); no new readback and no new submission in the plan report; an esbuild bundle check that a host's import of a shader constant resolves; a run on the RTX 5060 Ti while A1 is open (item 11). |
| 10 Kernels, after the compiler changes | — | Behaviour; outside rule 1; one pull request per step, each after its compiler change. A step that edits a sparse-scene kernel also waits for R03 (rule 6). **C1:** one BA module, after R03, in the shared-module form that compiles today or with C1's storage parameter (section 6, amendment 1). **C3:** the hand-written compactions of SIFT, ORB, KLT, matching and MVS become `push` where the compiler holds the count on the device, and so does the pair-preparation copy of the scan after R03; `scan.shade.ts` stays the device layer's scan until an amendment of `engine-v2.md` retires it (item 14). **C2:** the 18 oracle twins go only if item 10 is decided yes, the 12 sparse ones after R03. **C6 and C7** (amendment 2 corrects this step from proposal 0095, "TypeCV's evidence"). After C7 alone, these maps convert, since R3 accepts their writes today: `siftDescribe` (`k * 128 + j`), the global descriptor (`globalWords[i]`), the scalar matchers (`query * 3 + c`, `query * 2 + c`) and `matchBrief` (`i * 3 + c`). `siftDescribe` also writes its `return` as `continue`. The entries that phase 9 left convert after C7 too: SLIC, mesh, `planarFit`, the KLT maximum, the Hough vote and collect. `planarFit` writes `t * 14 + c`, a form R3 accepts today, so planar depth needs no C6. After C6, the maps that write at a base plus the index convert, with C7 where they loop or clamp and mask. They are in SIFT blur (`level.outOffset + i`), ORB (`lv.offset + i`), KLT (`kltLevel.offset + k`, `2 * (o + k) + c`) and MVS (`view * P + i`). The triangulation flags (`count + h`) also wait for R03. C6 covers neither `siftOrient` (`k * 2 + written`, a carried local) nor k-NN (`knnMean[visitOrder[lane]]`, a permutation). So both stay entries, and the kernel audit marks k-NN intentional. No TypeCV kernel writes several fields at a runtime stride. **Sentinel → optional value** (amendment 2; the paragraphs after this table). After a pin that carries change 0088, the sentinel results of function-scope helpers become `T \| undefined`. After a pin that carries 0093, so do the flags in struct fields. The sentinels held in storage stay. **After R03:** the inverse-depth conversion (struct arrays drop the host packing), the sparse kernels' exported constants, and the four test-only triangulation entries removed or kept with a reason (item 12). **Entry points:** they go live when the owner says so (items 9 and 13). R03 itself retires the object-model protocol and `runIncrementalGpu`, under its own records. | C1: every BA digest and WebGPU BA test unchanged. C3: row order and counts bit-identical; no new readback. C2: WebGPU unchanged; the CPU digests renewed once, in item 10's record. C6 and C7: phase 9's gate plus a hardware run. Sentinel → optional value: a behaviour change under Rule 2.2, with its own record. The CPU-tier digests of the affected kernels are renewed once, in that record. Each family's WebGPU test keeps its verdicts. After R03: phase 9's gate; the tests of removed entries named in the pull request. Entry points: the surface bake of `typecv` and `typecv/gpu` unchanged; each new entry's surface baked; every deep path still resolves. |
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

**The pin** (amendment 2). **Fact** (TypeShade, `git rev-list --count d36ac502..b920aea5`, run
on 2026-10-26). TypeCV's pin `d36ac502` is 44 commits behind TypeShade `main` at `b920aea`.
**Fact** (`git diff --stat d36ac502 b920aea5 -- src`). Their `src/` differs in 110 files. The range holds the
implementing commits of changes 0085, 0086 and 0088 (their `Change:` trailers). The rest are fixes,
documents and proposal acceptances. **Fact** (TypeShade `main` at `ff51e8f8`). Three commits
followed `b920aea`: typeshade/typeshade#773 implements change 0093, typeshade/typeshade#776 adds a
performance gate and typeshade/typeshade#777 adds a document. **Proposal.** The pin bump is its own
pull request. Its gate is phase 0's digests and the probe suite's pin test. **Fact**
(`tests/typeshade/compiler-probes.test.mjs`, `MEASURED_AT`). The pin test fails on every bump by
design, so each probe is read again. **Inference.** Some digests may change, since the compiler's
`src/` changes in 110 files.

**The pin bump** (amendment 3). **Fact** (#246, merged as `c1bfe38` on 2026-10-26). The pin is
`725b2969`: TypeShade `main` when the work of #246 started, which is `ff51e8f8` plus
typeshade/typeshade#778, a commit that adds only proposal 0097. **Fact** (TypeShade,
`git rev-list --count d36ac502..725b2969` and `git diff --stat d36ac502 725b2969 -- src`). The
range holds 48 commits, 33 of them not merges, and `src/` differs in 120 files. It implements
changes 0086, 0088 and 0093, and six stages of 0085, which stays `accepted`. It holds the fixes
typeshade/typeshade#760 and typeshade/typeshade#762. Proposals 0087, 0089, 0090 to 0092 and 0094
to 0097 are accepted and not implemented at `725b2969`.

**Fact** (#246, "What changed in TypeCV's fixtures, and why"). `tests/digests/kernels.json`
changes the `glsl` digest of 18 of the 30 kernel modules, from commit `4dedb2f6` of
typeshade/typeshade#760. Every WGSL digest and every count stay the same. So do the CPU-tier
digests, the surface bake, the WebGPU digests and the lab transcripts. `MEASURED_AT` of
`tests/typeshade/compiler-probes.test.mjs` moves, and no probe changes its verdict. **Fact** (#246,
"Product code"). `dist/gpu` changes shape: `index.js` now loads a shared chunk and a `fuse-plan`
chunk, and `dist/internal` likewise. **Inference** (#246). The cause is the automatic fusion
loading of change 0085 (typeshade/typeshade#751). **Fact** (the final report of the agent of #246:
its local gate on Node 22.22.0, with WebGPU on SwiftShader). `npm run test:website` passed 306 of
306. `npm run test:webgpu` passed 127 of 129 with 2 opt-in skips, the counts at the old pin. CI
run 40356667841 was green on every job on `fab38b9`, and its `webgpu` job ran on Node 24.21.0.

**Deviation** (#246, "Deviations from the brief"). The pull request records four:

1. The pin is `725b2969`, not `ff51e8f8`, since TypeShade `main` had moved when the work started.
2. Change 0085 is `accepted` at `725b2969`, not `implemented`; six of its stages are implemented.
3. The shader change is attributed by bisection over `git archive` trees, not by installing
   intermediate pins.
4. `compiler-changes.md` gains no entry: 0085's TypeCV work is not done, and 0086, 0088 and 0093
   name no TypeCV work.

**Disposition.** Each is accepted, by the owner's approval of #246 and its merge. The open items
of #246:

1. The red `digests (24)` of `ba5cb5f` is recorded in section 4, rule 2.
2. 0085's TypeCV product work is its own pull request. It covers automatic device selection, the
   registration-only `import 'typeshade/fuse'` of `src/gpu/device.ts` and explicit fp64 overrides.
   It also covers device ownership, execution reports, the chunks and the record in
   `compiler-changes.md`.
3. The sentinel step waits for R03 (the next paragraphs).

**The sentinel step** (amendment 2). **Fact** (TypeShade change 0088, `status: implemented`,
typeshade/typeshade#769, on TypeShade `main` at `6c6a1d1`). A local, a parameter and a function
result may hold `T | undefined`. The compiler lowers it to the value and a presence flag. The
lowering is the same on WGSL, GLSL ES 3.00 and the CPU oracle. Change 0088 excludes struct fields,
array elements, bindings, host values and `x?: T`. **Fact** (TypeShade proposal 0093). It extends
0088 to struct fields, array elements and optional parameters, where no binding layout holds them.
Its implementation, typeshade/typeshade#773, merged on 2026-10-26 as `19d3fbc`. Neither change
admits a binding or a workgroup variable.

**Fact** (TypeCV `main` at `917dd8a`, `grep -E '1e20|1e30' src/gpu/*.shade.ts`). 32 lines hold
`1e20` or `1e30`, 37 occurrences in all, the same as at `5e1c95c`. Proposal 0088's evidence counts
32. Some are bounds, such as `abs(alpha) < 1e20`, and two are in comments. The sentinels and flags
are these:

- helper results in function scope: `return 1e30;` in `error` (`consensus.shade.ts:96` and `:101`)
  and in `radialLimit` (`numerics.shade.ts:82` and `:87`);
- flags in struct fields: the `ok: u32` of `Projection`, `Linear3` and `Linear6`
  (`numerics.shade.ts:11-24`), tested as `projection.ok === 0` and `solution.ok === 0` in
  `bundle.shade.ts` and `joint-focal.shade.ts`;
- storage: `EMPTY: u32 = 0xffffffff` marks a free slot of `tsdfTable` (`tsdf.shade.ts:117`) and of
  `voxelTable` (`cloud-voxel.shade.ts:33`); `NO_VALUE: f32 = 1e30` (`dense-mesh.shade.ts:68`) is
  the empty value of the storage field `meshSdf` (line 20);
- workgroup memory: `let valid: u32` (`features.shade.ts:378`) is stored into the workgroup array
  `briefTileValid` (line 380).

**Proposal.** After a pin that carries 0088, the helper results in function scope become
`T | undefined`: `error` and `radialLimit`. After a pin that carries 0093, the flags of
`Projection`, `Linear3` and `Linear6` become optional values. The sentinels held in storage stay,
since both changes exclude bindings. **Inference.** Under 0088 alone, `project` could return
`Projection | undefined` instead; the step's record chooses. The `valid` flag of
`features.shade.ts` stays too, since it reaches a workgroup array. **Inference.** Every module that
imports `numerics.shade.ts` is a sparse-scene kernel, and so are `consensus.shade.ts`,
`bundle.shade.ts` and `joint-focal.shade.ts`. So this step also waits for R03 (section 4, rule 6).

**Proposal** (the gate). The CPU-tier digests of the affected kernels may change, since a sentinel
comparison becomes a flag. So the step is a behaviour change under Rule 2.2, with its own record and
a renewed digest. Each family's WebGPU test stays unchanged in its verdicts. **Fact** (change 0088,
"The proposal in one paragraph" and "The problem"). The step buys three things. The compiler catches
a read that forgets to test for the value. Nothing relies on NaN. One spelling serves every tier.

**The sentinel step at the pin** (amendment 3). **Fact** (#246, open item 3). The pin `725b2969`
carries changes 0088 and 0093, both `status: implemented`. So the compiler side of the step is
met. **Inference.** The step edits sparse-scene kernels only, so it still waits for R03 (section 4,
rule 6). It needs its own record under Rule 2.2.

**Status of the programme** (amendment 3, 2026-10-26). **Fact** (`main` at `c1bfe38`, and the pull
requests named).

| Step | Record | State on 2026-10-26 |
| --- | --- | --- |
| Phase 0 Characterize | #242, merged as `091a3e9` | done |
| Phase 1a Build selection | #247, merged as `3e2777b` | done; open item 1 closed by this amendment, item 2 moved to 1b (the phase 1 row) |
| The pin bump to `725b2969` | #246, merged as `c1bfe38` | done; open: 0085's TypeCV product work and the sentinel step (the paragraphs above) |
| The Node 24 gate change | #249 (section 4, rule 2) | a draft that waits for this amendment; its deviation open until 32 green `digests (24)` jobs |
| Phases 1b, 1c and 2 to 11 | none | no pull request |

## 6. Compiler changes this programme asks for

Each is a proposal in TypeShade's `changes/`, by Rule 3.3, or an issue where the row says so. C13
may instead be a change of TypeCV's device (audit lab, L8). The line counts are the kernel
audit's estimates (**Inference**) of what each change removes from TypeCV.

The compiler probes of the kernel audit (section 0.1) cover C1 (P4), C3 (P6), C6 (P9b, P20, P30),
C8 (P1) and C9 (P11, P20). C2, C4, C5 and C7 rest on the compiler's rules and source
without a probe: Rule 8.24, change 0074, surface §65 and `kernel-lower.ts:756-767`. C10 to C13
rest on the lab audit's L1, L3, L4 and L8, also without a probe. C14 (amendment 1) rests on the
measurements of TypeShade proposal 0090. **Fact** (amendment 1; #243,
`tests/typeshade/compiler-probes.test.mjs`). TypeCV's probe suite now probes C2 (S1), C5 (S5) and
C7 (S2) too. C4 and C10 to C13 stay without a probe: each needs a WebGPU device or an API that
does not exist yet.

| # | Limit (**Fact**) | Effect in TypeCV | Change asked | Lines |
| --- | --- | --- | --- | --- |
| C1 | `TS8020`: a storage array cannot be a helper's parameter, so a helper that reads a binding belongs to the file that declares it | the fixed and shared-focal BA kernels repeat about 900 of their lines with different record sizes | a helper that takes a storage array by reference, or a module parameterized by constants that a file instantiates twice; without a compiler change, a shared bindings module with one control struct (a layout change), within the 8 storage buffers per stage that WebGPU guarantees (audit sparse, L4) | ~900 |
| C2 | the CPU tier runs one invocation per call and cannot run a barrier (Rule 8.24), although the compiler's own oracle `dispatch` can | 18 `...Oracle` twins, the MVS splits and 29 backend branches; the oracle proves the twin, not the tiled entry, and 10 twins add in another order than the GPU's tree | the host CPU tier of an entry runs barrier entries in passes, as `dispatch` does; it then gives the GPU's bits for integer and `f32` folds, but not for `f64` folds (amendment 1) | ~250 |
| C3 | a kernel function's trip count is computed on the host, and a reduction result or a `push` count reaches only the host | `scan.shade.ts`, its copy, and about 25 flag, scatter, advance and mark entries keep counts on the device by hand | a `push` that appends after a count held in a `Resident` and leaves the new count there; a reduction into a `Resident`; a loop bound read from a `Resident`, lowered to an indirect dispatch | 300–400 |
| C4 | the compiler does not cut a loop into timed chunks (change 0074) | `chunkedPass` three times, `cappedPass`, and per-module chunk generators | a kernel call that splits a proved loop into chunks under `timeBudgetMs` | ~100 |
| C5 | `write(value)` replaces a whole `Resident` | `mvsCopyF32`, `mvsCopyU32` and their staging buffers copy a view into a slot | `write(value, { offset })` on the queue | ~40 |
| C6 | the proof refuses `out[base + i]` with a loop-invariant `base`, several fields at a runtime stride, a nested affine record and two record forms in one loop (rule R3 of 8.22) | several maps cannot convert as written | accept an injective offset, offsets `c < s` checked at the call as `a·i + c` is, and nested affine records | enabler |
| C7 | the map lowering starts every invocation with an early return past the trip count; A1 (`hardware-observations.md`, TypeShade #703) forbids that return on the RTX 5060 Ti | loop-heavy maps must stay entries written clamp-and-mask | an iteration scaffold that clamps the index and masks the writes | enabler |
| C8 | a kernel call takes no workgroup argument and no indirect count | 12 entries dispatch indirectly and cannot become kernel functions | a kernel call whose trip count is read from a `Resident` (shares C3's lowering) | enabler |
| C9 | surface §65 lists `min` and `max` among the scatter forms; the compiler refuses `a[k] = min(a[k], e)` | a conversion that needs it fails; no step of section 5 needs it (amendment 1: `kltEigenMax` is a `max` reduction and `houghVote` an `atomicAdd` scatter, both accepted today) | a TypeShade issue: the form the surface states, or a surface correction | bug |
| C10 | timestamp profiling is fixed when a device is created | the worker swaps the device for the dense stage and merges statistics across devices | a per-plan or per-scope switch for timestamp writes | lab |
| C11 | a device loss names no in-flight work | the lab tracks the last kernel itself | a loss record that names the plan or kernel submitted and not completed | lab |
| C12 | chunk rates live as long as the device | a record on a reused device includes earlier runs | a per-run epoch or snapshot of `chunkRates` | lab |
| C13 | GPU time cannot be attributed to stages inside one library call | the job diffs kernel statistics; view-graph stages carry no `gpuMs` | a named profiling scope that labels plan reports, in the TypeShade runtime or in TypeCV's device | lab |
| C14 | every TypeShade tool selects a shader module by its name in any directory, but `hostViewPath` writes its host view beside it, and `tsc` reads that view only under `moduleSuffixes` | the build split of section 2: GPU code selected by the directory `src/gpu`, generated views deleted by `scripts/build-gpu.mjs`, 27 kernel paths in `.prettierignore` | kernel modules compile in place: a shader module in any directory beside its host module, and the host views in a build directory outside the source tree | build |

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

**Status of the rows** (amendments 1 to 3, 2026-10-26). **Fact** (TypeShade `main` at `b920aea`,
and issues typeshade/typeshade#761 and typeshade/typeshade#775). Each proposal below is accepted as
a plan. No implementation of proposals 0090 to 0096 exists at `b920aea`. So TypeCV's pin `d36ac502`
carries none of them. **Fact** (TypeShade `main` at `ff51e8f8`). After `b920aea`,
typeshade/typeshade#773 implemented change 0093, which no row names. The proposals of the rows stay
unimplemented there. **Fact** (amendment 3; #246, and TypeShade at `725b2969`). TypeCV's pin is
`725b2969` since `c1bfe38`. Every proposal of the rows is `accepted` and not implemented there. So
the pin carries none of them, and no row changes its state.

| Row | TypeShade record | State on 2026-10-26 |
| --- | --- | --- |
| C14 | proposal 0090 (typeshade/typeshade#763) | accepted 2026-10-26 |
| C1 | proposal 0091 (typeshade/typeshade#767) | accepted 2026-10-26 |
| C2 | proposal 0092 (typeshade/typeshade#768) | accepted 2026-10-26 |
| C9 | issue typeshade/typeshade#761 | open |
| C3 with C8 | proposal 0094 (typeshade/typeshade#771) | accepted 2026-10-26 (amendment 2) |
| C6 | proposal 0095 (typeshade/typeshade#772) | accepted 2026-10-26 (amendment 2); its implementation follows the fix of issue typeshade/typeshade#775 (0095's D2) |
| C7 | proposal 0096 (typeshade/typeshade#774) | accepted 2026-10-26 (amendment 2); a run on the A1 device gates its first implementing pull request (0096's D5; section 7, item 11) |
| C4, C5, C10 to C13 | none | not opened (section 7, item 7) |

**Fact** (amendment 2; the "Approval and plan record" of each proposal). The owner's go-ahead
took the decisions that each proposal asked to settle before acceptance. The rest stay open:

- 0094 took D1, D2 and D6. Its first implementing pull request decides D3, D4, D5, D7 and D8.
  They are fusion, the report's `indirect` field, a held length with no push, a function with no
  loop and the scan's speed.
- 0095 took D1 and D2. Its first implementing pull request decides D3, D4 and D5. They are the
  text of `TS8070`, a negative stride and WebGL2.
- 0096 took D1, D2 and D5. Its first implementing pull request decides D3, D4 and D6. They are
  WebGL2, the entries an author writes and the face.

**Fact** (issue typeshade/typeshade#775, open). Today's `scaled` and `row-major` writes of a kernel
loop run unchecked at a stride of 0 and past an array's end. Decision D2 of 0095 makes its fix
land before 0095's implementation. **Fact** (proposal 0096, D5). The gate of 0096's first
implementing pull request is a run on the A1 device. The run uses a kernel function whose last
workgroup is partial. Section 7, item 11 names who runs it, and that item is open.

**Fact** (proposal 0090, "What it owes downstream"). The proposal lists TypeCV's expected work
after a pin that carries it. That work revises phase 1a and the rule "GPU code in `gpu/`
subdirectories" of section 3.1. **Inference.** Until such a pin, phase 1a stays as written.

**Fact** (amendment 3; #247, open item 1). Row C14's third column states the configuration before
phase 1a. Since `3e2777b`, the build selects `src/gpu`, `src/*/gpu` and `src/*/*/gpu`.
`scripts/build-gpu.mjs` deletes the host views in each of them. `.prettierignore` holds one glob
with three negated lines (section 2, "The delivered glob"). **Inference.** The host views are
still written beside the kernels, and `tsc` still reads them only under `moduleSuffixes`. So the
limit of row C14 and its request stand.

**Fact** (proposal 0091, "What TypeCV can do today"). Two entry files compiled with one shared
module that declares a storage binding and a workgroup tile. Its generic helper took each file's
own control struct. `compile()` gave no diagnostic; the WGSL was not given to Tint. The 13
storage bindings that `bundle.shade.ts` and `joint-focal.shade.ts` both declare are identical.
**Inference** (the proposal's estimate). A shared module of the 13 bindings and generic helpers
shares about 700 of the ~900 lines of row C1 today, with no compiler change. It changes no
layout. C1's compiler change then adds tens of lines for TypeCV, and its value is general. This
conversion edits sparse-scene kernels, so it waits for R03 (section 4, rule 6).

**Fact** (amendment 2; proposal 0091's front matter and TypeShade `main` at `ff51e8f8`). Proposal
0091 declares a new surface section 76. Change 0088 (typeshade/typeshade#769) took §76 first, for
optional values, and §76 is the last section on `main`. **Inference** (the TypeShade proposals
agent's re-scan of 2026-10-26). So 0091's implementation records §77, or the next free number, as
a deviation. No number is assigned yet.

**Inference** (proposal 0092, "The fold order and the bits"). With C2, the CPU tier adds the
lanes in the tree's order. It then gives the GPU's bits for integer and `f32` folds. For `f64`
folds it gives the GPU's order but not its bits, since WebGPU computes the emulated double. So
row C2's effect is narrowed. For the `f64` folds, one answer on every tier holds only within a
stated bound (Rule 1.2).

**Fact** (#243, `tests/typeshade/compiler-probes.test.mjs`, probes S6 and P20a). A `max`
reduction and an `atomicAdd` scatter are accepted forms at `d36ac502`. **Inference** (#243, from
the kernels' source). `kltEigenMax` is a `max` reduction and `houghVote` an `atomicAdd`
scatter. So C9 gates no step of section 5. Their conversions wait for C7 (phase 10), as the
phase 9 row says.

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
   **Result (amendment 1, 2026-10-26).** **Fact** (TypeShade `main` at `fc234dbc`). TypeShade
   accepted proposal 0091 for C1 (typeshade/typeshade#767) and 0092 for C2
   (typeshade/typeshade#768). It also accepted 0090 for C14 (typeshade/typeshade#763). Issue
   typeshade/typeshade#761 (C9) is open. The proposals for C3 with C8, C6 and C7 are in
   preparation (section 6, status of the rows). C4, C5 and C10 to C13 are not opened.
   **Result (amendment 2, 2026-10-26).** **Fact** (TypeShade `main` at `b920aea`). TypeShade
   accepted proposal 0094 for C3 with C8 (typeshade/typeshade#771), 0095 for C6
   (typeshade/typeshade#772) and 0096 for C7 (typeshade/typeshade#774). So every proposal that this
   item opened is accepted, and none is implemented. Issue typeshade/typeshade#761 (C9) is open.
   C4, C5 and C10 to C13 are not opened.
8. **Freeze or sequence.** Functional work on the files of a phase stops while the phase is open,
   or follows it.
   **Decision (2026-10-26).** Open. Decided before phase 1 opens.
   **Operating default (amendment 2, 2026-10-26).** **Fact.** On 2026-10-26 the owner instructed
   the coordinator to proceed through phase 1 and merge. **Inference** (the coordinator's reading
   of that instruction, not a decision of the owner). The coordinator applies "sequence" until the
   owner states otherwise. Functional work on a phase's files follows the phase, and no freeze is
   announced. The item stays open.
   **Operating default (amendment 3, 2026-10-26).** **Fact.** It continued through phase 1a (#247)
   and the pin bump (#246), and the item stays open.
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
  tier. It also runs the browser tests where the container has Chromium (section 4, rule 8,
  amendment 1). The CI job of phase 0 covers the GPU tiers:
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
- 2026-10-26: amendment 1 (TypeShade proposals 0090 to 0092 accepted, issue #761, the kernel
  alias, the Node-major binding of the digests, the container's Chromium), pull request #244.
- 2026-10-26: amendment 2 (TypeShade proposals 0094 to 0096 accepted, issues #761 and #775, the
  phase 10 corrections of 0095's reading, phase 0 deviation 2, the sentinel-to-optional step, the
  pin range), pull request #245.
- 2026-10-26: amendment 3 (phase 1a delivered in #247 with its `.prettierignore` deviation, the pin
  bump of #246 to `725b2969`, the Node 24 investigation of 2026-10-26 and its gate change in
  #249, the `focalGrid` fact of rule 2, the list that change 0094 asks of section 3.6, the status
  of the programme), pull request #248.
