# GPU throughput: measured bottlenecks and the plan to remove them

Status: accepted, 2026-10-21, as the order of work (owner request: maximum parallel throughput). Each item is verified by its own pull request. No kernel changes in this document's pull request. It adds the
measurement script `scripts/profile-kernels.mjs` and this record. Each item in "The plan" is its
own pull request with its own evidence.

Request source: the owner, 2026-10-21 (translated): more memory only lets a stage use more within
its limit. What matters more is to accelerate, to process with the maximum possible parallelism,
to split work finer where it can be split, and to put data in workgroup memory and the other
faster tiers. Moving memory up and down is a bottleneck. Recent papers, techniques from games,
other libraries and compression are all open.

Context: pull request #161 (dense MVS residency, `docs/dense-mvs.md` Amendment 3) and #167 (BA and
triangulation streaming) made every stage run within any memory budget. They did not make it
faster. This record asks where the time goes on the owner's GPU and what changes it.

Baseline: `main` at `4b16cd5`, compiler pin `3c6e964`. TypeShade documents are those of the pin
(`node_modules/typeshade/AUTHORING.md`, `docs/use-typeshade-surface.md`, `changes/`).

The 2026-10-22 audit in section 2 qualifies the historical GPU totals used here: composite
reports were counted twice and multi-video omitted Dense's device. The wall-time evidence
still locates expensive stages; the older kernel sums are not a corrected GPU-time budget.

## 1. Summary

Facts (section 3):

1. On the owner's RTX 5060 Ti, five stages hold 80 to 90 percent of a run: guided matching,
   pair verification, bundle adjustment, dense photometric propagation and dense geometric
   propagation. Triangulation and PnP hold another 11 to 14 percent.
2. The GPU is mostly idle in the sparse stages. They run 6 to 40 ms per dispatch and 0.5 ms per
   dispatch in pair verification, with 8-lane and 64-lane workgroups of tiny grids. This is
   latency and launch bound, not arithmetic or bandwidth bound.
3. The BA's PCG iteration costs about 7 ms (newest pin) to 17 ms (older pins). The bytes it moves
   would take about 25 microseconds at the card's memory bandwidth. The iteration is over 100
   times above its bandwidth floor.
4. Chunk timing measures the fence, not the kernel. Many timed chunks end at 3 to 8 ms whatever
   their size. The learned rate then makes chunks too small, and the GPU never fills.
5. Dense propagation is f32 and holds 97 percent of the dense GPU time on SwiftShader. It rereads
   the reference window and recomputes a weight and a relative pose for every hypothesis and
   every source.

Plan (section 7), by measured gain per effort. Estimates are inferences, and each pull request
measures its own gain.

| #   | Item                                                      | Stage hit                             | Expected gain on the 5060 Ti       |
| --- | --------------------------------------------------------- | ------------------------------------- | ---------------------------------- |
| 1   | Per-kernel GPU time and adapter features in the record    | all (measurement)                     | none by itself, decides the rest   |
| 2   | Fill the device before splitting a chunk                  | triangulation, PnP, pair verification | 8 to 12 s of 75 s                  |
| 3   | The guide of guided matching estimates E only             | guided matching                       | 8 to 12 s of 75 s                  |
| 4   | Dense propagation: hoist, then pack the gray              | dense photometric and geometric       | 25 to 40 percent of the dense time |
| 5   | BA: one workgroup per camera row group in the PCG product | bundle                                | 1.5 to 3 times on the BA stage     |
| 6   | BA: explicit reduced camera system and tiled Cholesky     | bundle                                | 3 to 8 times on the BA stage       |
| 7   | Hardware bilinear lookups for the source images           | dense propagation                     | 1.5 to 2.5 times on propagation    |
| 8   | Multi-resolution propagation (ACMM style)                 | dense photometric                     | 2 to 3 times, needs a quality gate |

Open decisions for the owner are in section 9.

## 2. Method and configurations

### Run-record accounting and actual IR audit (2026-10-22)

The owner requested analysis of the existing collected data before adding instrumentation.
Export run [39766639634](https://github.com/typeshade/typecv/actions/runs/39766639634) retrieved
the newest 50 records. Six carry v6 kernel timestamps; 19 have Dense results, 14 completed.
The latest stored run is `20261022025444698-40ee95c6`, from `b2d27fe`. The export therefore
contains no hardware measurement of the newer weighted TSDF visibility or #197's overlapping
readbacks. The [audit JSON](experiments/gpu/lab-bottleneck-audit-20261022.json) preserves the
archive hash, selected record fields and compiler/source identities without notes or input files.

Recorded wall times, in seconds (the Dense interval is separate from the sparse interval):

| Input on `b2d27fe` | Sparse BA / reconstruction | Pair verification | Dense total | Dense fusion |
| --- | ---: | ---: | ---: | ---: |
| Desert arch, 640 | BA 23.1 | 14.3 | 13.9 | 9.4 (68.0%) |
| Water tower, 1280 | BA 24.0 | 14.5 | 159.3 | 136.5 (85.7%) |
| Chimney rock, 4 videos, 1280 | Reconstruction 80.9 | 71.6 | 242.9 | 149.8 (61.7%) |
| Chimney rock, 15 videos, 640 | Reconstruction 513.0 | 247.8 | 146.0 | 54.6 (37.4%) |

Two confirmed TypeCV accounting defects limit the kernel rows:

1. `ShadeDevice.plan` recorded every runtime report, then `planSteps` / `planStepsMany`
   recorded their composite groups again. Groups, dispatches and nanoseconds were counted
   again without extra GPU work. Collection now happens only at `plan`; composite reports
   remain in history and returned execution reports. Deterministic timestamp regressions
   fail before the fix and pass afterward for serial, parallel and nested step streams.
2. Multi-video closes its sparse device before opening the Dense device. Its `result` held
   only the sparse table and `denseDone` carried none of the new device's table. The final
   Dense table is now sent once and added to the sparse table when the page builds its record.
   Per-component cumulative snapshots are not added together. A separate-device regression
   and an exported-record browser check cover this boundary.

Do not correct all historical rows by one factor: direct plans and composite plans have
different multiplicities. Stage elapsed times remain useful. GPU sums do not identify the
CPU share of a stage; queue gaps, transfer waits, compilation and sampled/untimed groups need
a timeline. Multi-video's zero application-transfer fields currently mean unavailable
accounting. Neither record kind stores CPU stacks, IR, read waves or full snapshot bytes.

Actual TypeShade IR and generated WGSL were inspected on source base `6cdd70e` with compiler
`94203d3`, using public `typeshade/dev` APIs. Reproduce from the repository root:

```sh
node scripts/profile-kernel-ir.mjs --out .typecv-build/kernel-ir
# A single entry, including its imported shader helpers:
node scripts/profile-kernel-ir.mjs --kernel 'src/gpu/mvs.shade.ts#mvsPropagate'
```

The tool writes authored IR, O0/O2 lowered IR, generated WGSL, dependency/code hashes and
emit-pass timings. It keeps the compiler out of the production bundle. Each count below
visits each reachable helper once; it is not a loop-weighted count, hardware instruction
count or GPU duration. Emit timings describe this host's build-time lowering, not browser
pipeline compilation.

| Entry | O0 arithmetic + call sites | O2 sites | Structure retained after optimization |
| --- | ---: | ---: | --- |
| `bundleSchurCamera` (shared focal) | 166 | 161 | One invocation walks a camera row's observations serially using emulated f64 |
| `appendPairModelRows` | 286 | 221 | QR row updates stay serial; `appendRow` takes and returns an 81-element f64 array by value |
| `mvsPropagate` | 599 | 581 | `sourceCost` recomputes relative pose and rereads the reference patch per hypothesis/source |
| `tsdfCarve` (legacy visibility) | 453 | 447 | Ray traversal, hash probes and atomic updates |
| `tsdfViewCarve` (current visibility) | 207 | 207 | Camera-by-voxel traversal and group evidence; no old-record timing for this path |

The three BA/pair/MVS source files are unchanged from `b2d27fe`, but that site's compiler pin
was `3c6e964`; TSDF source also changed. These IR observations are candidates to investigate
on the current compiler, not a measured runtime comparison against that site. O2 emits
`_licm` and CSE temporaries: optimization is active. An unchanged count does not establish a
missed optimization. WGSL array values do not prove actual spills or physical copies; the
driver may inline or eliminate them.

Ownership and next experiments:

| Question supported by the audit | Owner | Experiment needed before a speed claim |
| --- | --- | --- |
| Fusion takes most of Dense's wall time in two single-video runs | TypeCV scheduling/dataflow first; runtime if host profiling points there | Current-version fixed-depth replay; compare timestamp-covered kernel time, chunk counts, upload/read waves and CPU stacks |
| Shared-focal Schur product has long serial observation walks | TypeCV kernel / solver | Same observations with cooperative reduction or a different Schur solve; compare final cost and camera/point error as well as GPU time |
| QR repeatedly carries a large f64 array through `appendRow` | TypeCV representation; compiler only if a safe generic lowering helps | Same QR inputs, inspect current inline/reference facilities and generated code; measure native GPU time and verify numerical results |
| MVS recalculates reference/source work across hypotheses | TypeCV precomputation / caching; compiler for proven alias-safe common work | Same depths/sources/iterations with cached patch or relative pose, bounded memory and identical/toleranced results |
| Generic CSE/LICM/fusion may leave work | TypeShade only after a minimal reproducer | Toggle one pass on the same IR, inspect emitted code, then benchmark the same input on native hardware |

No new TypeShade compiler performance defect or native-GPU speedup is established by this
audit. The confirmed defects are TypeCV measurement/record assembly. Correct them before
using a fresh record to select the next kernel optimization; #197 already overlaps Dense
readbacks and must remain part of the new baseline.

### SfM workgroup specialization experiment (2026-10-22)

`nearestFloatGuided`, the actual scalar epipolar-guided matcher in
`src/gpu/kernels/float-matching.shade.ts`, now declares `GUIDED_WG: override<u32> = 64`.
The experiment imports its compiled manifest through the TypeShade Vite plugin, loads it with
the program runtime and selects 64, 128 or 256 through
`program.compute('nearestFloatGuided', { constants: { GUIDED_WG: size } })`.
Dispatch counts come from the returned `pipeline.workgroupShape` and are checked against the
device limits. Shader source is authored once. The browser bundle contains no compiler.

The existing SfM direct-call path retains the declared default of 64. The pinned TypeShade
direct-entry API, `entry(bindings, workgroups)`, accepts no per-call constants. This experiment
uses the program runtime; it does not add an automatic selector or change the Lab's execution
path. A production integration must preserve the device owner, resident buffers, chunk timing
and per-kernel reports when connecting a specialized pipeline to that path.

Run on a native GPU:

```sh
TYPESHADE_CHROMIUM=/path/to/chrome npm run benchmark:sfm:workgroups -- --hardware
```

Without `--hardware` the browser harness uses SwiftShader. Native runs reject software and
fallback adapters and an unidentified adapter. Default cases use 257 and 1025 queries, 1025
candidates, 128-dimensional deterministic descriptors, and epipolar bands of 2 and 10000 px.
`--queries`, `--candidates`, `--samples` and `--warmup` control the measurement. `--fixture file.json`
accepts one raw pair with `name`, `queries`, `candidates`, `dimensions`, `queryPoints`,
`candidatePoints`, `matrix` (row-major F or F-transpose for that direction) and `maxLinePx`.

Compile each specialization separately, keep its input buffers resident, warm it twice, then
rotate/reverse the variant order between samples. Fence before and after each dispatch. Where
`timestamp-query` is available, GPU time covers the compute pass; wall time covers encoding,
submission and queue completion. Compilation/pipeline preparation, descriptor uploads, output
readback and timestamp mapping are outside the samples. Uniform binding/packing remains in
the wall measurement. Each variant's indices, invalid flags and f32 distance bits must match
the existing direct 64-lane entry, both after warm-up and after measurements. Repeated requests
for the same override must return the cached pipeline. The separate correctness regression
covers tail queries, ties, nonfinite descriptors, empty bands, sole candidates and degenerate
lines; the existing SIFT regression also exercises all three sizes on six extracted-image
descriptor pairs. The timestamp-absent path is exercised by that SIFT regression.

Local measurement on 2026-10-22, based on `44b8800`, TypeShade `94203d3`, headless Chromium
with Google SwiftShader in the shared container. Command:

```sh
TYPESHADE_CHROMIUM=/usr/bin/chromium node scripts/benchmark-sfm-workgroups.mjs --queries 257,1025 --samples 5
```

Median GPU milliseconds, five samples after two warm-ups:

Raw samples and adapter identity: [sfm-guided-workgroups-software.json](experiments/gpu/sfm-guided-workgroups-software.json).

| Queries | Candidates | Band (px) | 64 lanes | 128 lanes | 256 lanes |
| ------- | ---------- | --------- | -------- | --------- | --------- |
| 257 | 1025 | 2 | 11.44 | 19.71 | 32.59 |
| 257 | 1025 | 10000 | 148.65 | 257.73 | 496.14 |
| 1025 | 1025 | 2 | 34.17 | 37.55 | 44.63 |
| 1025 | 1025 | 10000 | 597.33 | 604.34 | 609.54 |

All variants matched the direct entry bit for bit. No speedup is established by this
measurement, and these software timings do not select a native-GPU winner. The partial last
workgroup is a substantial fraction of the 257-query case at larger sizes. A physical GPU's
occupancy and scheduling differ; measure it before choosing a new default. These are isolated
match-kernel measurements, not complete guided matching (which includes RANSAC), or a full
SfM reconstruction time. No full-scene hardware quality/performance claim is made.

### Current-code performance audit (2026-10-22)

Scope: `44b8800` plus the workgroup experiment above. The older hardware records in section 3
identify candidate stages; their seconds and the estimates in section 7 are not measurements
of this revision. The following findings were checked against the current source. Only the
workgroup experiment above was newly timed in the audit. Readback overlap, BA camera reduction
and the Dense relative-pose cache were subsequently implemented and verified as recorded below.
Reference-sample/weight caching and the other remaining items are proposals.

| Priority | Candidate and current evidence | Proposed change and verification |
| -------- | ------------------------------ | -------------------------------- |
| 1, sparse SfM | BA still runs `bundleSchurCameraRow` with one lane per row, serially walking a camera's observations (`src/gpu/bundle.shade.ts`, `src/gpu/joint-focal.shade.ts`). `src/gpu/joint-focal.ts` enables its existing Schur cache only when explicitly requested, resident, unbounded, at most 16 cameras, and within a structural work limit. More than 16 cameras cannot use it. | First compare a workgroup reduction over each camera's observations. Then evaluate a budget-based extension of the cache. Changing reduction order needs per-iteration vector/cost checks and final reprojection/registration checks. Do not remove the 16-camera/work guards without bounding the assembly kernel. A direct Cholesky solve is a later, larger change. |
| 1, Dense | `referenceWindow` computes reference samples/weights and `sourceCost` reads/recomputes them inside every hypothesis/source evaluation. `sourceCost` and `relativePose` also reconstruct view-pair transforms repeatedly (`src/gpu/mvs.shade.ts`). There is no resident relative-pose cache. | Cache reference values/weights per active pixel and relative poses per reference/source window. Start with transformations of invariant calculations, preserving operation order. Compare depth/normal bits and final geometry. Benchmark cache variants: a large per-lane array can increase register pressure and make a larger workgroup slower. Handle all supported window sizes, not just the default 36 samples. |
| 2, small implementation | `readState` awaits depth, normal and optional cost one after another (`src/gpu/mvs.ts`). TSDF extraction similarly awaits positions then attributes (`src/gpu/tsdf.ts`). These reads have no data dependency after the producing kernels finish. | Overlap independent reads with `Promise.all`; measure `readWaves`, actual queue submissions and wall time. Parallel promises do not themselves combine staging buffers or guarantee one GPU submission. Keep terminal count reads before allocation, and copy/transfer outputs with their current ownership rules. Numerical arrays must remain identical. |
| 2, multi-video | `DescriptorStore.view().floatDescriptors()` dequantizes 8-bit descriptors on demand. `gpuViewGraphBackend.match()` uploads the first keyframe once for its candidate batch but dequantizes/uploads each candidate again when it appears in another batch (`src/reconstruction/view-graph.ts`, `src/gpu/view-graph.ts`). | Add a bounded LRU of decoded/resident descriptors keyed by immutable view id. Measure decode time, uploads, cache hits, peak active bytes and candidate-pair time. Reserve cache bytes against the working budget and release them before mapping. Reuse the same decoded values and preserve candidate order and exact match results. |
| 2, interactive Dense | Photometric iteration previews read the full depth image after iterations 1 and 3, then `depthPreview` reduces 4 x 4 blocks on the host (`src/gpu/mvs.ts`). Final depth/normal reads are separately required by the current fusion path. | Reduce preview-only images on the GPU before readback: about 1/16 as many depth values for these events, with ceil-sized edges. Preserve positive-depth mean semantics and verify previews within an explicit float tolerance. This does not remove final depth-map reads. Compare final geometry unchanged and page/worker activity with previews enabled. |
| 3, new TSDF visibility | The view-weighted mode introduced in #195 dispatches `tsdfViewCarve` over every voxel of every box for each observation view; off-screen samples return inside the shader (`src/gpu/tsdf.ts`, `src/gpu/tsdf.shade.ts`). | Investigate conservative block/frustum culling before per-voxel projection, then compact/indirect dispatch if it pays. Include truncation and voxel extent, near-plane crossings and every grouped camera. All retained points and evidence weights must equal the unculled route. First profile the new mode: no current native-GPU share is known. |
| Later, quality-changing | `sourceAt` loads four f32 gray values and mixes them manually. The depth stage initializes at one resolution (`src/gpu/mvs.shade.ts`, `src/gpu/mvs.ts`). | Hardware-filtered texture sampling and coarse-to-fine propagation are candidates after the invariant caches. They change interpolation or the search trajectory. Require accuracy AND completeness/floor coverage against the baseline, not just point count or runtime. Keep the no-data mask and budget/residency behavior. |

Already delivered and not counted as new opportunities: GPU-time/adapter telemetry, chunk
occupancy floors with GPU-time feedback, essential-only guided RANSAC, and reusing focal-search
pair solutions (section 8, #178/#180 and related records). `ShadeDevice.observe` currently learns
GPU time only for chunks with an occupancy floor on hardware. Extending that policy to other
long kernels needs measured evidence that their current fence-derived rates underfill the GPU;
it is not a reason to remove the dispatch-time ceiling.

Recommended order: overlap independent readbacks as a bounded first patch; prototype BA camera
reductions for sparse SfM and invariant caches for Dense as separate changes; use the new
workgroup probe as an additional parameter within a measured kernel. Keep descriptor caching
conditional on multi-video profiles. Defer changing multiscale search or filtering until a
quality gate includes the sparse-texture floors that motivated #195. Collect fresh native-GPU
kernel and stage timings before assigning percentage speedups or choosing a different default.

### Independent readbacks, first implementation (2026-10-22)

The first patch following the audit overlaps depth/normal/optional-cost reads in `readState`,
positions/attributes in pixel fusion, and positions/attributes in TSDF extraction. Terminal
count reads still finish before output allocation or readback. Each completed read immediately
copies its live prefix; the final arrays and callback transfer ownership remain as before.
`readTogether` waits for all sibling reads even if one fails, so the caller's `finally` block
cannot release a resident buffer while a sibling read is pending. This changes no kernel or
sample-validation rule and adds no page setting.

Compared with the identical kernels whose read requests are forced serial, on the CPU tier and
Chromium/SwiftShader WebGPU. The synthetic MVS scene has three 8 x 8 views; the TSDF fixture is
the existing exact-plane feedback case. Both modes warm the kernels first. Counts:

| Operation | Read calls, both modes | Read waves, serial | Read waves, concurrent | Requested bytes, both modes |
| --------- | --------------------- | ------------------ | ---------------------- | --------------------------- |
| Photometric snapshots, including cost | 9 | 9 | 3 | 2304 |
| Depth stage and pixel fusion | 21 | 21 | 12 | 3096 |
| TSDF count and extraction | 4 | 4 | 3 | 99904 |
| Pixel count and extraction from supplied valid depth maps | 9 | 9 | 6 | 1144 |

Depth, normal, cost, pixel/TSDF point arrays and all point attributes match exactly within each
tier. The supplied-depth pixel case and the TSDF case both contain real output points. Full
Resident snapshot bytes also match: 2304, 6936, 99904 and 3864 respectively. The difference
between requested and full snapshot bytes in the pixel path is existing allocation capacity;
the live prefixes are copied before returning them. No resident arrays remain live after each
case. The failure regression holds a sibling read pending and checks that the caller cannot
observe the rejection until that sibling settles, with the original error preserved.

Reproduce with `node --test tests/readback.test.ts tests/typeshade/readback.test.mjs` and
`TYPESHADE_CHROMIUM=/path/to/chrome node --test tests/typeshade/readback-webgpu.test.mjs`, after
`npm run build`. The tests record read waves and bytes; these are not physical GPU submission
counts or a measured full-run speedup. The number of data copies is unchanged. Native GPU wall
time remains to be measured. The next implementation follows below.

### BA camera reduction and Dense relative-pose cache (2026-10-22)

The BA camera product now has a cooperative entry: one 64-lane workgroup per reduced row,
512 bytes of workgroup scratch, strided observation gathers and a tree reduction. Fixed and
shared focal, bounded and unbounded submission, streamed observation windows and the cached
Schur certificate all use the same selection. Each streamed window starts or continues its
row as before. Inactive PCG workgroups still reach all barriers and leave the output alone.
No resident buffer or working-budget byte is added. The existing dense Schur cache and its
eligibility/work limits are unchanged.

Automatic selection keeps CPU, software adapters, short lists and problems beyond the
one-workgroup-per-row dispatch limit on the serial entry. With hardware occupancy scheduling,
at least 128 observations per camera on average selects cooperative rows. This is a structural
heuristic, not a threshold established by a native-GPU benchmark. `schurReduction` is a library
diagnostic override for comparisons; the lab requires no new setting. Native measurements
should check this threshold on the production adapters.

Dense depth kernels now calculate each resident source's relative rotation and translation
once per workgroup. A 448-byte table holds the window's at most seven slots; a uniform prologue
publishes it before pixel-dependent branches. It uses the same GPU f32 operations and order as
the scalar calculation. Photometric candidates, geometric reprojection and filtering reuse the
table; slot changes in streamed mode rebuild it automatically on the next dispatch. Tail lanes
still use the clamp-and-mask rule. CPU entries share the pixel implementation but retain the
barrier-free scalar oracle. There is no resident allocation, transfer, sample-count, filtering
or UI change. `relativePoseCache: false` is an internal comparison switch. Reference grayscale
samples and bilateral weights are still calculated as before: caching their larger arrays
requires a separate register/shared-memory tradeoff experiment.

Differential regressions on Chromium/SwiftShader:

- Fixed and shared focal at four and seventeen cameras, outliers, local streamed BA and the
  existing cached focal certificate all accept steps without solver failures. Every active
  product in the bounded four-camera runs is checked against scalar JS algebra from the same
  GPU point product and direction, and each LM cost is compared between reductions. Maximum
  scaled product discrepancy is below `1.2e-11`; final pose/point/focal differences are below
  `2e-10`. CPU forced-cooperative requests retain identical scalar results. Automatic selection
  and the cooperative dispatch-limit fallback are exercised separately.
- Dense photometric depth, normal and cost bits match after initialization and every iteration,
  including six sources, the default 36-sample patch, a 121-sample bilateral patch, 65-element
  tail chunks and streamed sources. Every photometric/geometric/filter array and final pixel
  cloud matches bit for bit in full window, grouped bilateral and budget-limited runs. The
  49 x 37 budget fixture keeps nonempty geometry, streams geometric sources through two slots
  and 64-element chunks, and peaks at 72,524 bytes within its 75,000-byte allowance.

Reproduce after `npm run build`:

```sh
node --test tests/typeshade/bundle-reduction.test.mjs
TYPESHADE_CHROMIUM=/path/to/chrome node --test --test-concurrency=1 tests/typeshade/bundle-reduction-webgpu.test.mjs tests/typeshade/mvs-pose-cache-webgpu.test.mjs
TYPESHADE_CHROMIUM=/path/to/chrome npm run benchmark:gpu:invariants -- --samples 5 --warmup 2
# On the production machine, reject software/fallback adapters:
TYPESHADE_CHROMIUM=/path/to/chrome npm run benchmark:gpu:invariants -- --hardware
```

The benchmark alternates variant order, after two complete warm-ups per variant. It measures
complete four-iteration BA or two-iteration photometric stages, including allocation, uploads,
submissions and terminal readback. Pipeline warm-up is excluded. GPU time is the sum of pass
timestamps only when every group has one. It is not an isolated camera-product time, full SfM
time or full Dense time. Transfer counters cover `ShadeDevice` hooks, not direct
`Resident.write` uploads; Dense's zero tracked upload bytes do not mean it uploads no images.
Raw samples, adapter identity, input hashes, shader/fixture hashes and
TypeShade pin: [ba-dense-invariants-software.json](experiments/gpu/ba-dense-invariants-software.json).
Base `6cdd70e` plus this implementation, shared container, Chromium/SwiftShader:

| Stage | Scalar wall / GPU median ms | Optimized wall / GPU median ms |
| ----- | --------------------------- | ------------------------------ |
| Fixed BA, 4 cameras x 256 points | 195.90 / 119.97 | 315.30 / 234.75 |
| Shared-focal BA, 17 cameras x 256 points | 714.60 / 619.01 | 1010.90 / 900.22 |
| Photometric Dense, 128 x 96, 4 sources | 1028.70 / 987.23 | 1013.00 / 974.43 |

Cooperative BA is slower on this software adapter, which automatic selection excludes. The
shared-focal variants also take 101 versus 100 CG iterations, affecting dispatch counts and
these whole-stage times; their final costs agree within `2e-10`. Dense's small software timing
difference does not establish a native-GPU speedup. No native-GPU or full-scene performance
claim follows from these records. Descriptor LRU, GPU preview reduction, conservative TSDF
culling and the larger reference-patch cache remain follow-ups in the audit's order.

### Historical run records

Run records. The owner's collector (Cloudflare KV namespace `1c8702ece02640b7bfdf7dca1af2e6e2`,
keys `record/<ts>-<8hex>`) was read on 2026-10-21 with `wrangler kv key get`. A record holds
`timing.stages` (wall time, calls, dispatches and submissions per stage), `timing.chunks`,
`timing.kernels` (the kernel names in first-use order, no times), `gpu`, `mainThread`, `transfers`
and `dense`. A record holds no time for a single kernel. Section 4 says what that costs.

| Record     | Site commit | Adapter                | Input                                   | Total    |
| ---------- | ----------- | ---------------------- | --------------------------------------- | -------- |
| `ed932d77` | `351fe7e`   | NVIDIA Blackwell       | pyramid video, 22 of 43 frames, dense   | 88.5 s   |
| `28dcfe16` | `351fe7e`   | NVIDIA Blackwell       | pyramid video, 24 of 47 frames, dense   | 88.9 s   |
| `0de918f8` | `a121ae0`   | NVIDIA Blackwell       | COLMAP south-building, 24 photos, dense | 78.2 s   |
| `4a8aa2ff` | `d89ef71`   | NVIDIA Blackwell       | pyramid video, 22 of 87 frames, dense   | 101.3 s  |
| `f1edec5c` | `53d3a7c`   | NVIDIA Blackwell       | Chimney Rock video, 27 frames, 1280x720 | 75.1 s   |
| `c39246e7` | `5200253`   | NVIDIA Turing (RTX 20) | COLMAP south-building, 24 photos, dense | 318.0 s  |
| `19f18274` | `edb0d35`   | NVIDIA Blackwell       | multi-video, 354 frames, 93 registered  | 628.8 s  |
| `fad31ad6` | `f2473c9`   | NVIDIA Blackwell       | the same multi-video input              | 169.5 s  |
| `d40f8950` | `f2473c9`   | NVIDIA Blackwell       | multi-video, 839 frames, failed         | no stage |

Local profile. `scripts/profile-kernels.mjs` wraps the TypeShade device's `fuse()` and sums, for
each kernel name, the `gpuTime` that `timestamp-query` reports. It ran on 2026-10-21 in headless
Chromium 1194 with SwiftShader, on a shared 4-core container, at `4b16cd5`:

```sh
npm run build
node scripts/profile-kernels.mjs --case mvs    --scene south-building-24 --ids 0,1,2,3,4,5 --scale 2
node scripts/profile-kernels.mjs --case bundle --scene south-building-24 --scale 1
node scripts/profile-kernels.mjs --case sift   --scene south-building-24 --ids 0,1,2,3
node scripts/profile-kernels.mjs --case pair   --scene south-building-24 --scale 1
```

Warning about SwiftShader. It is a CPU emulation. It ranks kernels by arithmetic and by barrier
cost, and it has no launch latency, no occupancy and no memory hierarchy worth the name. A
SwiftShader share is evidence of where arithmetic goes. It is not evidence of where a
physical GPU waits. The earlier comparison in `docs/lab-run-records.md` ("SwiftShader against the
RTX 2080") shows the gap: features ran 111 times faster on the card, pair verification 1.4 times.

Adapter facts. SwiftShader in this container offers the features `subgroups`, `timestamp-query`,
`float32-filterable`, `indirect-first-instance` and the texture-compression set. It does not offer
`shader-f16`. Its limits: 32768 bytes of workgroup storage, 256 invocations per workgroup, 10
storage buffers per stage. The run record keeps neither the features nor the limits of the
owner's adapter, so this record cannot say whether the 5060 Ti under Chrome 153 offers `shader-f16`
and `subgroups`. Chrome ships `subgroups` since version 134
([new in WebGPU 134](https://developer.chrome.com/blog/new-in-webgpu-134)), so the owner's browser
is expected to offer it. Item 1 records it.

## 3. Measured baseline

### 3.1 Time per stage on the owner's GPU

Seconds, with the share of the run's total. Source: `timing.stages` and `dense.stageMs` of the
records. Blank means the record has no such stage. The two multi-video records are in 3.4.

| Stage             | `ed932d77` | `28dcfe16` | `0de918f8` | `4a8aa2ff` | `f1edec5c` | `c39246e7` (Turing) |
| ----------------- | ---------- | ---------- | ---------- | ---------- | ---------- | ------------------- |
| features          | 1.3 (2%)   | 1.4 (2%)   | 1.6 (2%)   | 1.5 (1%)   | 1.7 (2%)   | -                   |
| matches           | 0.8 (1%)   | 0.8 (1%)   | 1.4 (2%)   | 0.7 (1%)   | 1.0 (1%)   | -                   |
| guided matching   | 17.5 (20%) | 18.1 (20%) | 13.2 (17%) | 17.5 (17%) | 17.6 (23%) | -                   |
| pair verification | 14.6 (16%) | 17.1 (19%) | 10.0 (13%) | 15.3 (15%) | 16.1 (21%) | 68.0 (21%)          |
| triangulation     | 5.4 (6%)   | 5.0 (6%)   | 5.3 (7%)   | 5.7 (6%)   | 5.4 (7%)   | 11.4 (4%)           |
| bundle            | 39.3 (44%) | 35.9 (40%) | 38.3 (49%) | 37.9 (37%) | 11.0 (15%) | 150.2 (47%)         |
| observations      | 1.5 (2%)   | 1.8 (2%)   | 1.4 (2%)   | 1.5 (1%)   | 1.9 (3%)   | 4.6 (1%)            |
| PnP               | 5.0 (6%)   | 5.0 (6%)   | 5.3 (7%)   | 5.5 (5%)   | 5.0 (7%)   | 10.5 (3%)           |
| dense photometric | 15.6 (18%) | 7.3 (8%)   | 26.3 (34%) | 17.9 (18%) | -          | 66.9 (21%)          |
| dense geometric   | 4.5 (5%)   | 3.3 (4%)   | 3.6 (5%)   | 4.4 (4%)   | -          | 13.9 (4%)           |
| dense fusion      | 0.6 (1%)   | 0.6 (1%)   | 0.7 (1%)   | 0.6 (1%)   | -          | 1.3 (0%)            |
| total             | 88.5       | 88.9       | 78.2       | 101.3      | 75.1       | 318.0               |

The sum of the rows is below the total by 5 to 10 percent: frame decode, upload, mapper control
on the host and the viewer.

Observation (fact). The bundle row falls from 36 to 39 s in four records to 11 s in `f1edec5c`
(27 views and 33458 observations, more than `28dcfe16`'s 24 views and 27062). Between them the pin
moved to `3c6e964` (#164, the removal of the typeshade/typeshade#717 stopgap) and #152 cut the host
cost of pair verification. Inference: the older BA was bound by the host cost of checking bound
arrays before each dispatch, and #164 removed most of it. The records do not isolate the cause. The
same inference fits the multi-video pair (3.4).

### 3.2 Dispatches, submissions and the time of one dispatch

Source: `timing.stages`. "ms per dispatch" is the stage's wall time divided by its dispatch count.
It includes everything: the host, the queue, the fence and the GPU.

| Stage (`f1edec5c`, older record in brackets) | Calls | Dispatches | Submissions | ms per call | ms per dispatch |
| -------------------------------------------- | ----- | ---------- | ----------- | ----------- | --------------- |
| features                                     | 27    | 4563       | 27          | 64          | 0.38            |
| matches                                      | 98    | 588        | 98          | 10          | 1.7             |
| guided matching                              | 98    | 588        | 98          | 180         | 30              |
| pair verification                            | 98    | 19020      | 3460        | 165         | 0.85            |
| triangulation                                | 77    | 159        | 159         | 70          | 34              |
| bundle                                       | 27    | 21590      | 403         | 406         | 0.51 (3.0)      |
| observations                                 | 69    | 207        | 69          | 28          | 9.3             |
| PnP                                          | 25    | 2563       | 208         | 199         | 1.9             |

Readings (inferences from these counts):

- Guided matching runs 6 dispatches and one submission per pair, yet costs 180 ms. The plain
  matcher runs the same 6 dispatches for 10 ms. `docs/lab-run-records.md` shows the scalar guided
  kernel runs about 3 percent of the plain matcher's sums. So the 170 ms difference is not the
  match kernels. The stage also runs the guide: a full `verifyImagePairGpu` (50 hypotheses) on
  the same correspondences (`guidedMatches` in `website/src/workers/sfm.worker.ts`). Pair
  verification then runs the full 500-hypothesis search on the same pair again.
- Pair verification runs about 195 dispatches and 35 submissions per pair. Its kernels are
  8-lane workgroups (`preparePairModels`, `appendPairModelRows`, `solvePairModelNull`,
  `finishPairModels`) with one model per lane. A batch of 32 models is 4 workgroups. The batch
  size is the "32 trials up" floor of the time budget (`docs/gpu-geometry.md`).
- Triangulation and `observations` run 9 to 42 ms per dispatch. These are single dispatches
  that wait for the GPU (`submissions` equals `dispatches`). Their cost is the fence plus a
  per-lane serial solve, 16 lanes to a workgroup.

### 3.3 The chunk rates show a floor of 3 to 8 ms

`timing.chunks` of `ed932d77`, `28dcfe16`, `0de918f8`, `4a8aa2ff`, `f1edec5c`:

| Key                 | Units of the last chunk | Longest chunk (ms) | Learned ms per unit | The cost of a unit if the longest chunk is the fence |
| ------------------- | ----------------------- | ------------------ | ------------------- | ---------------------------------------------------- |
| `scorePairModels`   | 1                       | 3.0 to 5.4         | 1.0 to 3.6          | one model costs microseconds, not 1 to 3 ms          |
| `scorePnP`          | 500                     | 4.1 to 8.2         | 0.005 to 0.010      | 500 trials in 4 ms is the same floor                 |
| `triangulateTracks` | 64 to 284               | 4.6 to 5.8         | 0.02 to 0.25        | a chunk of 118 tracks is 8 workgroups                |
| `mvsInitialize`     | 1536                    | 2.1 to 6.7         | 0.001 to 0.003      | the same floor                                       |
| `mvsPropagate`      | 1536                    | 3.6 to 6.1         | 0.001 to 0.004      | the same floor                                       |

Fact: the longest chunk of nearly every key is 3 to 8 ms, whatever its units. The one exception
class is `bundleCg` (57 to 182 ms), the only key whose chunks do real work per unit.

Inference: a chunk is timed with a queue fence (`device.wait()`), and a fence on this stack costs
about 3 to 5 ms. The learned rate adds that latency to the first timed chunks. The rate is then
too high, the planner picks small chunks, and each chunk fills a small part of the GPU. The
time budget is 250 ms (`timing.budgetMs`), but the chunks run at 3 to 8 ms. An RTX 5060 Ti has about
36 streaming multiprocessors and holds about 55000 resident threads. A 118-track chunk of 16-lane
workgroups does not fill one tenth of it. For a latency-bound kernel the cost of 118 and of 6341
tracks is nearly the same. Item 2 tests this. The test is a SwiftShader-independent one: the
timestamp of the chunk (item 1) against its fence time.

### 3.4 Multi-video (host-bound)

| Stage             | `19f18274` (`edb0d35`) | `fad31ad6` (`f2473c9`) |
| ----------------- | ---------------------- | ---------------------- |
| features          | 5.1 s                  | 5.0 s                  |
| pair matching     | 8.4 s                  | 9.1 s                  |
| pair verification | 176.8 s (882 pairs)    | 69.5 s                 |
| reconstruction    | 443.3 s                | 90.7 s                 |
| total             | 628.8 s                | 169.5 s                |

Fact: the same input ran 3.7 times faster between two site builds that changed no kernel (#152
host-cost cuts, #164 pin move). Pair verification is 79 ms per pair after the change. Inference:
for this input the host, not the GPU kernels, held most of the time. The record's `mainThread`
shows frame gaps of 41 s (`19f18274`), 15 s (`fad31ad6`) and 33 s with 164 long tasks (`d40f8950`,
a failed 839-frame run). The page stalls even when the worker does the work. This record leaves the
page's own stalls to the multi-video record (`docs/multi-video.md`).

### 3.5 Transfers

`transfers` of the records: application uploads 204 to 584 MB per run, readbacks 7 to 11 MB.
`applicationUploadBytes` counts the values given to `allocate*()` and also the zeros of every new
`f64` array (`ShadeDevice.statistics`). Much of the upload is therefore zero fill. At 10 GB/s
effective, 584 MB is 60 ms in a 75 s run. Fact: the volume is small in time. Transfer is not a
bottleneck in these runs. A GPU-side clear would still save the host copy and the allocation
(item 9).

### 3.6 Per-kernel profile on SwiftShader

Warning: SwiftShader. Read as the distribution of emulated arithmetic.

Dense propagation (`--case mvs`, 6 views of south-building-24 at 320x240, default iterations,
`gpuTime` sum 315 s):

| Kernel                                       | Groups | GPU ms     | Share |
| -------------------------------------------- | ------ | ---------- | ----- |
| `mvsPropagate` (photometric and geometric)   | 1022   | 305684     | 96.9% |
| `mvsInitialize`                              | 27     | 3350       | 1.1%  |
| `mvsGeometricInitialize`                     | 23     | 2543       | 0.8%  |
| `mvsFilter`                                  | 29     | 2535       | 0.8%  |
| `mvsScatterPoints`, `mvsFuseFlags`, the rest | 2500   | under 1000 | 0.4%  |

Bundle adjustment (`--case bundle`, 24 views, `refineSharedFocal`, 4 LM iterations, `gpuTime`
sum 9522 ms; 401 PCG iterations):

| Kernel                                                                                                   | Groups | GPU ms | Share | Threads of one dispatch                              |
| -------------------------------------------------------------------------------------------------------- | ------ | ------ | ----- | ---------------------------------------------------- |
| `bundleSchurCamera`                                                                                      | 401    | 4816   | 50.6% | 6C + 1 rows, 145 for 24 cameras (3 workgroups of 64) |
| `bundleSchurPoint`                                                                                       | 401    | 3706   | 38.9% | one per landmark, a few thousand                     |
| `bundleCameraRows`                                                                                       | 4      | 322    | 3.4%  | per camera row                                       |
| `bundleCandidatePartials`, `bundlePointBlocks`, `bundleFocalBlock`, `linearizeBundle`, `advanceBundleCg` |        | 555    | 5.8%  |                                                      |

Pair verification (`--case pair`, 23 consecutive pairs, `gpuTime` sum 6120 ms):
`scorePairModels` 55.5%, `appendPairModelRows` 16.6% (1229 groups, 53 per pair),
`triangulatePoseCandidates` 10.1%, `solvePairModelNull` 6.6%, the rest 11%.

SIFT (`--case sift`, 4 frames at 640x480, first octave -1, `gpuTime` sum 25 s):
`scanFlags` 49.1%, `siftBlurV` and `siftBlurH` 21.3%, `siftScatterExtrema` and `siftFlagExtrema`
18.6%, `siftDescribe` 6.8%. `scanFlags` is a 64-lane Hillis-Steele scan with 12 barriers.
SwiftShader pays dearly for barriers, a card does not. Do not act on this row.

### 3.7 What the BA numbers say about the card

`bundleCg` learned 0.0007 ms per unit on `ed932d77` and 0.0002 on `f1edec5c`. A unit is one row of
a PCG iteration (`M + 6C`, `docs/gpu-sfm.md`). For `f1edec5c` that is
0.0002 x (33458 + 162) = 6.7 ms per PCG iteration, and for `ed932d77` 0.0007 x 24312 = 17 ms.

The data one iteration reads, assuming 8 bytes per emulated `f64` and 21 stored cross-block words
per observation (`crossAt` in `src/gpu/joint-focal.shade.ts`): 33458 x 168 B = 5.6 MB per pass, two
passes, about 12 MB with the vectors. At about 450 GB/s that is 25 microseconds. The arithmetic is
about 33458 x 21 MACs x 2 passes x 20 f32 operations per emulated MAC = 28 MFLOP, below 2
microseconds at 20 TFLOP/s. Inference: the iteration is more than 100 times above its roofline, and
the cause is the critical path, not the volume. In `bundleSchurCameraRow`, one invocation walks all
observations of its camera, about 1200 for `f1edec5c`, with four dependent loads per step
(`cameraObservation`, `index`, `blocks`, `vectors`), and only 6C + 1 invocations exist.

## 4. What the records cannot tell, and how to close it

1. No kernel time. `timing.kernels` lists names only. Item 1 adds `gpuMs`, `groups` and `dispatches`
   per kernel, summed from `FuseReport.groups[].gpuTime`, and the wall time of the plans that held
   them. The difference of a stage's wall time and its summed GPU time is its host and fence time.
   This one table decides the order of items 2 to 8 on the owner's card.
2. No adapter features. `gpu` holds two limits. Item 1 adds `features` and the compute limits
   (`maxComputeWorkgroupStorageSize`, `maxComputeInvocationsPerWorkgroup`,
   `maxStorageBuffersPerShaderStage`), and `shader-f16` and `subgroups` in particular.
3. `mvsPropagate` does not separate the photometric and geometric pass in `dense.stageMs`'s
   kernels. The stage times separate them (3.1). Item 1 keeps the kernel's `pass` in the key.
4. Guided matching has no split between the guide and the match kernels. Item 1 times them.

## 5. Research

Each row: the method, the source, what it removes here and the fit to WebGPU. The WebGPU limits
that decide fit: no native `f64`; `subgroups` is optional and absent from many adapters; 16 KiB of
workgroup memory is the guaranteed minimum (`maxComputeWorkgroupStorageSize`), 32 KiB is common;
8 storage buffers per stage is the guaranteed minimum; no device-wide barrier inside a dispatch;
`timestamp-query` is optional; a forward-progress guarantee between workgroups does not exist.

### 5.1 Bundle adjustment

| Method                                                                                                                        | Source                                                                                                                                                           | Fit here                                                                                                                                                                                                                                                                                            |
| ----------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Implicit Schur complement with PCG and a block-Jacobi preconditioner; a thread per point and per camera row; Jacobians cached | Wu, Agarwal, Curless, Seitz, "Multicore Bundle Adjustment", CVPR 2011, [project](https://grail.cs.washington.edu/projects/mcba/)                                 | This is what `bundleSchurPoint` and `bundleSchurCamera` do. The paper targets thousands of cameras, where a thread per camera row is enough. At 20 to 100 cameras it is not.                                                                                                                        |
| GPU BA library: batched Jacobian by forward-mode AD, PCG with distributed Schur elimination, all kernels over observations    | Ren et al., "MegBA: A GPU-Based Distributed Library for Large-Scale Bundle Adjustment", ECCV 2022, [arXiv:2112.01349](https://arxiv.org/abs/2112.01349)          | Reports up to 41x over Ceres. Its parallel axis is the observation, not the camera. Adopt that axis for the camera-side product (item 5). The distribution part has no use here.                                                                                                                    |
| Landmark elimination by QR nullspace marginalization, single precision as accurate as double Schur on BAL                     | Demmel, Sommer, Cremers, Usenko, "Square Root Bundle Adjustment for Large-Scale Reconstruction", CVPR 2021, [arXiv:2103.01843](https://arxiv.org/abs/2103.01843) | Good numerics in f32, which removes the emulated `f64` of the Jacobian blocks. It needs per-landmark QR with a larger working set. Candidate for a later item. Single precision there still needs care at our depth range.                                                                          |
| Power series of the inverse Schur complement as a preconditioner, fewer PCG iterations at high accuracy                       | Weber, Demmel, Chan, Cremers, "Power Bundle Adjustment for Large-Scale 3D Reconstruction", [arXiv:2204.12834](https://arxiv.org/abs/2204.12834) (v4 2023)        | Reduces the iteration count, so it multiplies with every kernel gain. It adds one product with the point blocks per iteration. Fit: good, small kernels. Its gain is large for ill-conditioned, many-camera problems and unknown for our small ones.                                                |
| Dense reduced camera system, assembled per camera pair, factored by Cholesky                                                  | Ceres `DENSE_SCHUR` ([docs](http://ceres-solver.org/nnls_solving.html)); Agarwal et al., "Bundle Adjustment in the Large", ECCV 2010                             | Best at 20 to 200 cameras: the reduced system is 6C square (162 x 162 for 27 views). One factorization replaces 20 to 100 PCG iterations of 3 dispatches. Fits workgroup memory (tiles) and needs no atomics. This repository's `schurCache` already assembles the dense matrix (limit 16 cameras). |
| Mixed precision: low-precision products with compensated accumulation, refinement in high precision                           | Haidar et al., "Harnessing GPU Tensor Cores for Fast FP16 Arithmetic to Speed up Mixed-Precision Iterative Refinement Solvers", SC 2018                          | Store Jacobian and cross blocks as f32 (halves bytes, removes the emulated-`f64` multiplies), accumulate in emulated `f64` or in a compensated f32 pair, keep the LM step acceptance in `f64`. Needs a tolerance test, not bit equality.                                                            |
| Batching many small local BAs                                                                                                 | MegBA batches observations across the problem. The local BAs of `localBundleSubset` are separate problems today.                                                 | Each local BA is 4 to 8 LM iterations. A batch of independent local problems in one dispatch amortizes the launch chain. High effort. Defer.                                                                                                                                                        |

### 5.2 Features and matching

| Method                                                                                                  | Source                                                                                                                                                                                  | Fit here                                                                                                                                                                                                                                               |
| ------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| GPU SIFT with separable Gaussian passes through workgroup memory, one workgroup per image tile and halo | Wu, "SiftGPU" ([project](http://cs.unc.edu/~ccwu/siftgpu/)); AliceVision, [PopSift](https://github.com/alicevision/popsift), described in "AliceVision Meshroom", ACM MMSys 2021        | `siftBlurH` and `siftBlurV` read the image from storage once per tap. A tile with halo in 16 KiB of workgroup memory removes the repeated reads. Features are 1 to 3 percent of a run, so low priority.                                                |
| Brute-force kNN as tiled GEMM with a fused top-2 selection                                              | Johnson, Douze, Jegou, "Billion-scale similarity search with GPUs", IEEE TBD 2021, [arXiv:1702.08734](https://arxiv.org/abs/1702.08734); Garcia, Debreuve, Barlaud, CVPR Workshops 2008 | This repository already has a tiled L2 matcher (`nearestFloatForward`, tiles of the plain matcher). Matching is 1 to 2 percent of a run. Int8 descriptors with `packed4x8Dot` (a derived capability of the pin) would quarter the bytes. Low priority. |
| RootSIFT (Hellinger kernel)                                                                             | Arandjelovic, Zisserman, "Three things everyone should know to improve object retrieval", CVPR 2012                                                                                     | Already used (`rootSift: true` default).                                                                                                                                                                                                               |
| Epipolar-guided matching with the band from a robust E                                                  | COLMAP `guided_matching`, Schonberger and Frahm, "Structure-from-Motion Revisited", CVPR 2016                                                                                           | The cost here is the guide's RANSAC, not the band test (3.2). See item 3.                                                                                                                                                                              |

### 5.3 Dense MVS

| Method                                                                                    | Source                                                                                                                                        | Fit here                                                                                                                                                                                                                     |
| ----------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| PatchMatch stereo: slanted planes, random init, propagation, refinement                   | Bleyer, Rhemann, Rother, BMVC 2011                                                                                                            | The algorithm of `mvsPropagate`.                                                                                                                                                                                             |
| Red-black (checkerboard) propagation for massively parallel GPU MVS                       | Galliani, Lasinger, Schindler, "Massively Parallel Multiview Stereopsis by Surface Normal Diffusion", ICCV 2015 (Gipuma)                      | Already used (`mvsControl.color`).                                                                                                                                                                                           |
| Multi-hypothesis joint view selection, adaptive window, structured region information     | Xu, Tao, "Multi-Scale Geometric Consistency Guided Multi-View Stereo", CVPR 2019, [arXiv:1904.08103](https://arxiv.org/abs/1904.08103) (ACMM) | Coarse-to-fine pyramid: converge at 1/4 and 1/2 resolution, then refine at full resolution with fewer full-resolution iterations. 2 to 3 times fewer full-resolution pixels. Changes results, needs a quality gate (item 8). |
| Planar priors from sparse credible points for textureless areas                           | Xu, Tao, "Planar Prior Assisted PatchMatch Multi-View Stereo", AAAI 2020, [arXiv:1912.11744](https://arxiv.org/abs/1912.11744) (ACMP)         | Quality, not speed. Our sparse landmarks are the credible points. Note for a later quality record.                                                                                                                           |
| Pixelwise view selection with a probabilistic occlusion model; geometric consistency term | Schonberger, Zheng, Pollefeys, Frahm, "Pixelwise View Selection for Unstructured Multi-View Stereo", ECCV 2016 (COLMAP)                       | The design of `docs/dense-mvs.md`.                                                                                                                                                                                           |
| Zero-mean NCC from running sums (box sums, integral images)                               | Lewis, "Fast Normalized Cross-Correlation", Vision Interface 1995                                                                             | The window is sampled with step 2 on a homography-warped grid, so the source side cannot use a box filter. The reference side (mean, variance, weights) can be computed once per pixel and held in registers.                |
| Hardware texture filtering for bilinear lookups; 2D-local cache                           | Standard GPU practice (texture objects in CUDA PatchMatch implementations)                                                                    | `sourceAt` does four storage loads and three mixes per sample. A filtered texture does one instruction. Hardware weights have limited fractional precision, so results differ from the CPU oracle by a tolerance.            |
| Reduced-precision storage of depth, normal, cost, gray                                    | Octahedral normals: Cigolle et al., JCGT 2014, [paper](https://jcgt.org/published/0003/02/01/)                                                | Normals are already octahedral `u32`. Gray as `unorm8` quarters the source bytes. Depth as `f16` loses range, keep `f32`.                                                                                                    |

### 5.4 Game-engine and library techniques

| Technique                                                                        | Source                                                                                                                                                                                                          | Fit here                                                                                                                                                                                                                                                |
| -------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| GPU-driven pipelines: the GPU writes the counts and the next dispatch reads them | Haar, Aaltonen, "GPU-Driven Rendering Pipelines", SIGGRAPH 2015 Advances course                                                                                                                                 | Already used: indirect dispatch, TypeShade change 0075. Extend it from batch loops to the guide and the PnP chain (item 2).                                                                                                                             |
| Async upload with ring buffers and double buffering to overlap copy and compute  | Standard in engines (staging rings)                                                                                                                                                                             | Uploads are 60 ms of 75 s (3.5). No gain now. Reopen when images grow to 4K.                                                                                                                                                                            |
| Virtual texturing and sparse residency                                           | Barrett, "Sparse Virtual Textures", GDC 2008                                                                                                                                                                    | #161's windowed residency does this by view window. A page-table version helps only at memory limits, not speed.                                                                                                                                        |
| Tile binning and a per-tile workgroup for cooperative work                       | Kerbl et al., "3D Gaussian Splatting for Real-Time Radiance Field Rendering", SIGGRAPH 2023, [arXiv:2308.04079](https://arxiv.org/abs/2308.04079); gsplat, [arXiv:2409.06765](https://arxiv.org/abs/2409.06765) | The idea is a tile of pixels sharing one workgroup memory fill. It is the shape of the reference-window tile in item 4. A radix sort for binning (Onesweep, Adinets, Merrill, [arXiv:2206.01784](https://arxiv.org/abs/2206.01784)) is not needed here. |
| Single-pass scan with decoupled look-back                                        | Merrill, Garland, NVIDIA TR NVR-2016-002                                                                                                                                                                        | Needs forward progress between workgroups. WebGPU does not guarantee it. Do not use. Keep the two-level scan (`scanFlags`, `scanFlagBlocks`).                                                                                                           |
| Parallel reduction by tree in workgroup memory, then subgroup operations         | Harris, "Optimizing Parallel Reduction in CUDA", NVIDIA 2007                                                                                                                                                    | The tree is portable (`bundleMetricPartials` uses it). Subgroup operations need the `subgroups` feature and compiler intrinsics the pin lacks (gap G2).                                                                                                 |
| GPU-side convergence tests instead of readbacks                                  | Standard (device-resident flags)                                                                                                                                                                                | Already used: `snapshotBundleActivity` reads 8 bytes per chunk.                                                                                                                                                                                         |
| Kernel fusion                                                                    | TypeShade `fuse`, changes 0058 and 0069                                                                                                                                                                         | Already used. The remaining fusions are across the Schur point and camera kernels, which need a grid-wide barrier (not available). Do not fuse them.                                                                                                    |
| Data compression: f16 storage, quantized positions, block compression            | Standard                                                                                                                                                                                                        | TypeShade has `pack2x16float`, `unpack2x16float`, `pack4x8unorm`. It has no `f16` type (gap G1). Bytes are not the bottleneck now (3.5, 3.7), so use compression only where an item above is bandwidth bound.                                           |

## 6. TypeShade capabilities for each technique

Source: `node_modules/typeshade` at the pin (`AUTHORING.md`, `docs/use-typeshade-surface.md`,
`changes/`). Status "available" means the surface documents it. Nothing here was run on a card.

| Needed                                           | Status at the pin                                                                                                                                                      |
| ------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Workgroup memory and barriers                    | Available. `let tile: workgroup<array<f32, 64>>`, `workgroupBarrier()` (surface section 24, 25). Used in `scan.shade.ts`, `joint-focal.shade.ts`.                      |
| Barrier under divergent control flow             | Refused (`TS8052`). An item must keep every lane at each barrier. The idle-lane rule of `docs/gpu-sfm.md` already says so.                                             |
| CPU tier for a barrier kernel                    | Not available. "The oracle runs one invocation per call" (surface section 25). Every tiled kernel needs an `Oracle` sibling, as `bundleMetricPartialsOracle` does. G4. |
| Integer atomics                                  | Available (`atomic<u32>`, `atomic<i32>`). No `f32` atomics, as in WGSL. Items below use none.                                                                          |
| Indirect dispatch                                | Available (change 0075, used in `pair-models.shade.ts`).                                                                                                               |
| Workgroup size override                          | Available (change 0057, `@compute([N])` and overrides).                                                                                                                |
| `f16` type and arithmetic                        | Not available. The capability `f16` exists, but no `f16` scalar type is authorable (`AUTHORING.md`, "Capabilities"). Storage packing with `pack2x16float` works. G1.   |
| Subgroup intrinsics                              | Not available. The built-ins `subgroup_invocation_id` and `subgroup_size` are; `subgroupAdd` and the rest are absent (`AUTHORING.md`). G2.                             |
| Sampled texture and sampler in a compute entry   | Documented: `textureSampleLevel` is accepted in any stage (surface table at line 5011). Not run in this repository. The CPU oracle's bilinear is not the GPU's. G3.    |
| Texture written by a kernel function             | Accepted, not implemented at the pin (change 0077, `accepted`).                                                                                                        |
| `dot4I8Packed`, `packed4x8Dot`                   | Derived capability `packed4x8Dot` exists (`AUTHORING.md` table).                                                                                                       |
| Timestamp per fuse group                         | Available (change 0066, `FuseReport.groups[].gpuTime`). One value for a group, not for one entry call in a fused chain. Not used by the time budget. G5.               |
| Chunk learner on GPU time, fixed latency removed | Not available. Change 0074 times the chunk with a queue fence. G5.                                                                                                     |
| Large workgroup memory                           | Limited by the adapter (`maxComputeWorkgroupStorageSize`, 16 KiB minimum, 32 KiB on SwiftShader). The kernel must read the limit; G6 asks for a host-readable limit.   |
| Device-wide barrier, persistent kernel           | Not available in WebGPU. Do not plan on it.                                                                                                                            |
| Async readback pipeline                          | `device.read` is one round trip. TypeCV reads flags only every fourth batch and per chunk; no further gap.                                                             |

### Draft issues for typeshade/typeshade (not filed)

G1. Title: An `f16` scalar type for `"use typeshade"` kernels, so that half-precision storage and
arithmetic are authorable.
Body: `AUTHORING.md` lists `f16` as a capability (`enable f16;`, host feature `shader-f16`) and says
no `f16` scalar type exists. TypeCV's dense MVS and BA store gray, depth, cost and Jacobian blocks
in `f32`. It can pack two halves with `pack2x16float` but cannot compute on them. Expected: `f16`
scalar and `vec2h`, `vec4h` types, with the CPU oracle rounding to binary16 after each operation
(as `quantizeToF16` does) and the WebGL2 backend refusing them. Tests owed: a kernel with `f16`
storage and `f32` accumulation against the oracle; the `SD0030` refusal on GLSL.
Pinned commit: `3c6e964`.

G2. Title: Subgroup intrinsics (`subgroupAdd`, `subgroupMax`, `subgroupShuffle`, `subgroupBallot`)
with a CPU oracle meaning.
Body: The pin accepts `@builtin("subgroup_invocation_id")` and `@builtin("subgroup_size")` and
derives the `subgroups` capability, but no intrinsic. TypeCV's reductions (`bundleMetricPartials`,
the PCG dot products, the top-2 selection of matching) use a workgroup-memory tree with several
barriers. A subgroup reduction needs none. Expected: the intrinsics behind the `subgroups`
capability, with the oracle running a declared subgroup size of 1 or a stated simulated size, and a
refusal on WebGL2. Tests owed: a reduction against the tree reduction on SwiftShader (which offers
`subgroups`).

G3. Title: A sampled texture with a filtering sampler read from a compute entry, with a stated
tolerance class for the CPU oracle.
Body: The surface table lists `textureSampleLevel` as valid in any stage. TypeCV would store the
gray image of each view as an `r16float` or `r8unorm` texture and read bilinear samples in
`mvsPropagate`. Hardware bilinear weights have fewer than 32 bits of fraction, so the CPU oracle
cannot match bit for bit. Expected: documented behavior of `textureSampleLevel` in a `@compute`
entry on WebGPU and the CPU tier, and a tolerance class in the surface's accuracy table for the
filtered result (a bound in units of the texel's last place). Tests owed: a compute entry
that samples at fractional coordinates, compared with a four-texel oracle within the bound.

G4. Title: The CPU tier runs a tiled entry (workgroup memory and barriers) as a whole workgroup.
Body: Surface section 25 says the oracle runs one invocation per call, so a barrier has no
partner. TypeCV keeps a sequential `...Oracle` sibling for each tiled entry (`scanPairRowsOracle`,
`bundleMetricPartialsOracle`, `scorePairModelsOracle`). Each pair is a second kernel to write and
to hold to the first. Expected: a CPU tier that runs the lanes of a workgroup between barriers
(a generator or a thread pool), so that one source runs on every tier, as change 0069 allows for a
tiled entry in a fuse scope. Tests owed: the existing tile examples on the CPU tier.

G5. Title: The chunk learner of a device fits `time = latency + rate x units` from GPU
timestamps when the adapter has `timestamp-query`.
Body: Change 0074 times a chunk from its plan start to the queue fence. On the owner's RTX 5060 Ti
the longest chunk of nearly every key is 3 to 8 ms whatever its size (TypeCV record `ed932d77`:
`scorePairModels` 1 unit in 3.0 ms, `scorePnP` 500 units in 4.1 ms, `triangulateTracks` 118 units
in 5.4 ms). The fence latency enters the rate, the rate sizes small chunks, and small chunks leave
the GPU empty. Expected: when `FuseReport.groups[].gpuTime` exists, learn the rate from it and keep
the fence time only as a watchdog check. Without timestamps, fit two chunk sizes to separate the
latency. Also expose a per-entry `gpuTime` for the entries of one plan. Tests owed: a synthetic
kernel with a known latency floor, and a check that the chosen chunk grows to the budget.

G6. Title: A host-readable workgroup-memory limit and a way to size a tile from it.
Body: A tile kernel must fit `maxComputeWorkgroupStorageSize` (16 KiB guaranteed). The device
owner exposes limits (change 0051). Expected: a documented read of that limit in the program
runtime, and an `override` constant that a kernel can size its tile from (change 0057 covers
the workgroup size). Tests owed: one kernel built at two tile sizes.

## 7. The plan

Order: measured gain per effort, on the owner's card. Gains are inferences until the pull request
measures them. Each pull request states its before and after numbers: SwiftShader with
`scripts/profile-kernels.mjs` for the work done (dispatches, groups, GPU ms), and, by request,
the owner's GPU through a run record (item 1 makes that table).

Verification classes:

- Exact. The pull request shows equal output on the CPU tier and on SwiftShader (the tests
  named in each item). A pull request in this class changes order of loop iterations only
  where it says so.
- Tolerance. A kernel reorders a floating-point sum or reads a filtered texel. The pull request
  states the bound, shows the CPU oracle held to it, and shows the end-to-end figures
  (landmarks, mean reprojection error, fused points) within the run-to-run spread of the
  baseline.

### Item 1. Per-kernel GPU time and adapter features in the run record

- Stage: all. Technique: measurement. Cost: small (`website/src/lib/run-record.ts`,
  `website/src/workers/sfm.worker.ts`, `docs/lab-run-records.md`, record test).
- Change: record format v6. `timing.gpu` is a list of `{ name, groups, dispatches, gpuMs }` from
  the summed `FuseReport.groups` (a wrapper on the TypeShade device's report, as in
  `scripts/profile-kernels.mjs`), plus `timing.stages[*].gpuMs`. `gpu.features` holds the adapter's
  feature list and `gpu.limits` the compute limits. The cap is 120 kernel rows.
- Gain: none directly. It turns sections 3.2 and 3.3 from inference into fact on the owner's card
  and decides items 2 to 8.
- Risk: `timestamp-query` rounds timestamps coarsely in some browsers; a group may carry no
  `gpuTime`. A kernel row keeps `gpuMs: null` then. Timestamp queries cost a little per plan.
- Verification: record validator test (`website/src/lib/run-record.ts` tests), a record sent from
  SwiftShader carries the rows, v5 records still validate. Exact class (no kernel changes).
- TypeShade gaps: none (G5 later improves it).

### Item 2. Fill the device before splitting a chunk

- Stage: triangulation (5.4 s), PnP (5.0 s), pair verification (16 s, scoring and model batches),
  and the same rule for `observations`. Technique: occupancy-aware chunks. A chunk key gets a
  minimum number of units that fills the device (for example 2048 workgroups), and the rate
  learned from GPU time (item 1) replaces the fence time as soon as the pin offers it (G5).
  Until G5, TypeCV computes the rate itself from `gpuTime` of the chunk's report.
- Cost: medium-small. Files: `src/gpu/device.ts` (`planSteps`), `src/gpu/triangulation.ts`,
  `src/gpu/pnp.ts`, `src/gpu/pair-estimation.ts` (the `trials` expression, 32 floor),
  `src/gpu/observations.ts`.
- Expected gain: triangulation 5.4 s to about 1 s (a call of 6341 tracks as one 400-workgroup
  dispatch, because a latency-bound serial solve costs the same for 118 and 6341 tracks until
  the card fills); PnP 5 s to about 2 s; pair verification 16 s to about 9 s (a batch of 256 to
  512 models instead of 32 cuts the 195 dispatches per pair to about 60). Sum 8 to 12 s of 75 s.
- Risk: the Windows watchdog (TDR) and the earlier Blackwell hang in `triangulateTracks`
  (`docs/hardware-observations.md`, A1). Mitigation: a ceiling in measured GPU milliseconds (for
  example 100 ms) per dispatch, taken from `gpuTime`, and an upper bound of the first dispatch of
  each key at the old chunk size times four. The idle-lane rule stays.
- Verification: exact. `tests/typeshade/pnp-batch.test.mjs`, `pair-batch.test.mjs`,
  `tests/typeshade/streaming.test.mjs`, `plan-steps.test.mjs` pass unchanged; the RANSAC search
  stops "where a sequential RANSAC would, whatever the batch" (`docs/gpu-sfm.md`), so the models and
  inliers are equal. The pull request adds a test that a call of N tracks plans at most
  `ceil(N / floor)` dispatches. Benchmark: dispatch and submission counts per stage on SwiftShader
  (must fall), GPU ms from the profile script, and the owner's run record (`timing.stages`).
- TypeShade gaps: G5 (rate from GPU time). Not blocking.

Step record (PR 2, 2026-10-21; head of `perf/occupancy-floors`, based on `main` at `aa406fa`).

- Change: `GpuChunk.minUnits` and `ShadeDevice.chunkSize(key, max, min, minUnits)`. Floors:
  `triangulateTracks` 16384 tracks (16 lanes x 1024 workgroups), `scorePnP` 512 trials,
  `scorePairModels` 256 models. The floor of 32 in `src/gpu/pair-estimation.ts` and `src/gpu/pnp.ts`
  is the `min` argument still, and the floor is the new fourth argument. `observations` is one plan
  of three dispatches over all its units, so its floor is met already, and the step changes nothing
  there. The refit's one-model scoring declares no floor.
- Rate from GPU time: a timed chunk that declares `minUnits` teaches the device `ms per unit` from
  the sum of its groups' `gpuTime` (the collector of PR 1 reads the same field). A chunk with any
  group without `gpuTime` teaches nothing, and the fence rate stays. A GPU rate also holds the chunk
  to 80 percent of 100 GPU ms.
- Safety (three bounds, in the order they act): a fresh key starts at `min`, below the old chunk
  size times four, and grows at most fourfold a chunk (triangulation: 16, 64, 256, 1024, 4096,
  16384), so the first dispatch is
  never the whole floor; one dispatch over 100 GPU ms (`GPU_CEILING_MS`) halves the key's floor,
  again at each overrun; a settled key with a floor is timed again every 16th chunk
  (`RETIME_EVERY`), so that a larger input than the one the rate came from shows its cost. The
  third bound is an addition to the text of item 2: without it a settled key would never measure
  a dispatch again, and the ceiling could not act. Disposition: made part of the approved text by
  this record.
- Tests: `tests/typeshade/occupancy-floor.test.mjs` (new, in `test:shade`: the ramp, N units give
  `ceil(N / floor)` chunks on a settled key, the old arithmetic without a floor, the GPU rate, the
  fence fallback, the ceiling halving, the re-timing). `south-building-triangulation-webgpu.test.mjs`
  gains "a settled triangulation key plans ceil(N / floor) chunks" (1672 tracks and 16720 tracks on
  a device with `chunkTimingSamples: 1`: 1 + 1 and 1 + 2 dispatches). `pnp-batch`, `pair-batch`,
  `streaming` and `plan-steps` pass unchanged.
- Measurement. Facts, measured on SwiftShader (headless Chromium, `TYPESHADE_CHROMIUM`), 8-frame
  pyramid sample, `node scripts/benchmark-sfm-pyramid-e2e.mjs --max-models-per-batch=1024
  --occupancy-floors=1` (new options; the default stays 32, which the script pinned before and which caps a pair batch at 32,
  so the default run shows no change: 2648 dispatches and 632 submissions in pair verification
  before and after). The last warm run of each build, builds alternated, run alone from this
  session (the host was shared with other agents: load average 2 to 8 at the start of the runs):

  | Stage (`--max-models-per-batch=1024`) | Dispatches before | Dispatches after | Submissions before | Submissions after |
  | ------------------------------------- | ----------------- | ---------------- | ------------------ | ----------------- |
  | pair verification (16 calls)          | 2320 and 2352     | 1456 and 1456    | 544 and 552        | 304 and 304       |
  | triangulation (1 call)                | 2                 | 2                | 2                  | 2                 |
  | bundle, observations                  | 557 and 12        | 557 and 12       | 24 and 4           | 24 and 4          |

  Larger run (`--max-features=2000 --feature-mode=pyramid --warm-runs=1`, 5 registered views): pair
  verification 2516 to 1860 dispatches and 558 to 394 submissions; triangulation (7 calls) 16 to
  14; PnP (3 calls) 358 to 343 dispatches and 32 to 26 submissions; BA and observations equal.
- Software adapters. The CI job `website/site.test.mjs` "public UI runs real CPU/WebGPU
  verification" timed out at 240 s on the first head of this pull request in two local runs. It was
  not slower in a comparison made alone, one after the other, on a host at load average 7 to 9
  (the test total, with the Pyramid lab run inside it): `main` at `ffd289a` 664 and 666 s, the first
  head with floors on SwiftShader (`e2736a7`) 678 s, this head 632 s. Fact: no slowdown in this
  test. Inference: the two timeouts came from host load. A separate fact stays: on SwiftShader
  a batch of 256 or 512 models costs the work of 256 or 512 models, and a search that stops early
  scores the models past the stop for nothing (the next row: pair verification two to three times
  slower in the benchmark). A software adapter has no idle lanes to fill, so `ShadeDevice.open` takes
  `occupancyFloors`, true on a hardware adapter and false on a software one (a fallback adapter, or
  SwiftShader, llvmpipe or "software" in the adapter description). With it false the device keeps the
  old arithmetic. `--occupancy-floors=1` of the benchmark and `occupancyFloors: true` of the WebGPU
  triangulation test turn the floors on for SwiftShader.
- Time on SwiftShader: pair verification 716 and 831 ms before, 1602 and 2841 ms after
  (mapper 0.8 to 0.9 s before, 1.7 and 3.0 s after). A rerun on the final build under a load average
  of 14 gave 3687 and 2992 ms with the same counts (1456 dispatches, 304 submissions). It rises, because SwiftShader runs the lanes
  on the CPU and a batch of 256 scores models that a batch of 32 would skip after the search
  stopped. This is the case the open item "SwiftShader cannot show latency or occupancy gains"
  names. The gain item 2 expects is the fence and launch latency of a physical GPU, and it is not
  measured here.
- Exactness (fact): the registered views, the registration order, the initial pair, the landmark
  count (14, and 520 on the larger run), the observation count, the mean reprojection error
  (0.14519108498256017 px, and 0.28425065341709826 px) and `hypothesesScored` of every stage are
  equal before and after.
- Not run: the owner's run record against `f1edec5c` (no physical GPU here), the effect of the GPU
  rate (SwiftShader reports no `gpuTime`, so that path runs in the unit test with a stand-in
  device only), and the PnP floor at its full 512 on a scene with enough correspondences.
- Open: the owner's run on the 5060 Ti decides whether the floors of 16384, 512 and 256 are the
  right sizes; the ceiling and the re-timing bound a wrong one.

### Item 3. The guide of guided matching estimates E only

- Stage: guided matching (17.6 s, 23 percent). Technique: the guide needs a band, not a pose. Run
  the essential-matrix RANSAC alone (`estimateEssentialRansacGpu`, `src/gpu/two-view.ts`) in one
  batch of 64 to 128 hypotheses and build the band from E directly. Skip pose recovery,
  triangulation and refinement (`triangulatePoseCandidates` is 10 percent of pair verification
  GPU time on SwiftShader) and the `essentialFromPose` round trip.
- Cost: small to medium. Files: `website/src/workers/sfm.worker.ts` (`guidedMatches`),
  `website/src/lib/guided-matching.ts`, maybe `src/gpu/two-view.ts`.
- Expected gain: the guide drops from about 170 ms to about 40 to 60 ms per pair: 17.6 s to 6 to 8 s.
  A further step, reuse of the mapper's E (verify first, then rematch), would remove the guide
  entirely but reorders the worker. Record it as a follow-up.
- Risk: the band differs slightly (E unscaled against `[t]x R`). The mapper still verifies the
  chosen matches, so a wrong band costs matches, not correctness.
- Verification: tolerance. `tests/typeshade/guided-matching-webgpu.test.mjs` and the end-to-end
  figures. The pull request shows the guided match count per pair on the pyramid and south-building
  samples within 2 percent of the baseline, and equal landmarks within the baseline spread.
- TypeShade gaps: none.

Step record (PR 3, 2026-10-21; head of `perf/guide-essential-only`, based on `main` at `e6a35ad`).
Facts, measured on SwiftShader (headless Chromium, `TYPESHADE_CHROMIUM`), 8-frame pyramid sample,
`node scripts/measure-orb.mjs --sample="Pyramid benchmark" --query="octave=0&guided=1"`. The host
was shared with other agents (load average 8 to 11), so absolute times carry that noise. Runs of
the old and the new build alternated.

- Change: `guidedMatches` calls `estimateEssentialRansacGpu` (50 hypotheses, fixed draws, no
  adaptive stop) and passes its E to the matcher. The pose round trip is gone: `essentialFromPose`
  and `guideVerificationOptions` are removed, and `guideEssentialOptions` and `guideEssential`
  replace them. `GUIDE_RANSAC_ITERATIONS` stays 50.
- Guided stage time (`guidedMatches`, sum of 22 pairs): old 17.5, 20.8, 15.5 s (also 12.7, 16.5, 18.0
  s in three earlier runs); new 7.1, 8.0, 6.8 s. About 2.3 times faster. The total run
  is within the noise of the load (old 99.6, 99.9, 81.9 s; new 94.8, 97.8, 86.5 s).
- `--case guided` of `scripts/profile-kernels.mjs` (south-building-24, ids 0 to 5, scale 4, same
  loaded host, 2 pairs with 16 or more matches): guide 12.2 s with `--guide verify`, 4.0 s with
  `--guide essential`; the guided match counts are equal (64).
- Spread of the old build: six runs: 8 registered views, 694
  landmarks, mean error 0.5506 px, the same matches in every pair. The spread of the outputs is
  zero, because the GPU RANSAC is seeded.
- Guided matches per pair after the change: 21 of 22 pairs equal the old count. Pair 4-6 has 134
  against 128 (+4.7 percent; all 134 are inliers). The sum over the pairs is 3310 against 3304
  (+0.18 percent). Guided gain: 111 added matches, old 105.
- Result: 8 registered views (equal), mean error 0.5443 px (old 0.5506 px), landmarks 675
  (old 694, minus 2.7 percent).
- Deviation (open): the landmark count differs from the old build by 19, and the old spread is
  zero. Inference: a different E changes the guided matches of one pair, and the tracks change
  with it. Check: the same build with 100 hypotheses changes the matches of 10 pairs and gives 684
  landmarks and 0.5244 px. So a change of the guide's E moves the landmark count by 10 to 20 in
  either direction on this sample, and the error does not get worse. The acceptance bar "equal
  landmarks within the baseline spread" is not met as written. Next action: the owner decides
  whether a bar of a few percent applies to a deterministic spread of zero.
- Not run: the south-building end-to-end comparison and the owner's `guidedMatches` stage against
  `f1edec5c` (no physical GPU here).

### Item 4. Dense propagation: hoist the loop-invariant work, then pack the gray

- Stage: dense photometric (7 to 26 s) and geometric (3 to 5 s). On SwiftShader `mvsPropagate` is
  97 percent of the dense GPU time. Technique, in two exact steps and one tolerance step:
  1. Reference window cache (exact). `referenceWindow` and `sourceCost` read the reference window
     again for each of up to 10 hypotheses and 6 sources per pixel and compute an `exp` bilateral
     weight for each of 36 samples each time. Read the 36 reference values and weights once into
     `array<f32, 36>` registers per pixel and iteration. This removes about 2000 global loads and
     2000 `exp` per pixel and iteration.
  2. Relative pose cache (exact). `sourceCost` rebuilds `R_rs` and `t_rs` (about 100 flops and 12
     loads) for each hypothesis and source. They depend on the reference and source only. Compute
     them once per pass into a small storage array (`mvsRelative`, 6 x 16 floats), the way
     `mvsCameras` is held.
  3. Gray as `unorm8` (tolerance). The gray comes from 8-bit RGB. Store the undistorted gray with
     `pack4x8unorm` (4 samples per word). Source bytes fall 4 times. Quantization error is at most
     0.5/255 of the range per sample and enters NCC as noise below the cost's own noise.
- Cost: step 1 and 2 small (`src/gpu/mvs.shade.ts`, `src/gpu/mvs.ts`); step 3 medium
  (`mvsUndistort`, `mvsGray` layout in `docs/dense-mvs.md` 4.1 and the residency plan of
  Amendment 3).
- Expected gain: steps 1 and 2: 20 to 30 percent of the propagate time (a count of the loads and
  transcendental calls, to be measured on SwiftShader first); step 3: a further 10 to 20 percent
  where sampling is bandwidth bound. Together 25 to 40 percent of the dense time (4 to 12 s per run).
- Risk: more registers per lane (36 values and 36 weights), which can lower occupancy. The
  pull request measures register-bound effects with a second variant (weights recomputed
  from 36 offsets, values cached).
- Verification: steps 1 and 2 exact: `tests/typeshade/mvs.test.mjs`, `mvs-geometric.test.mjs` and
  `mvs-webgpu.test.mjs` compare depth and normal bit for bit; the fused point count equals the
  baseline. Step 3 tolerance: the fused point count and the 99th percentile depth change within
  1 percent on south-building-24 and the pyramid sample. Benchmark: `mvsPropagate` GPU ms before
  and after with the profile script, same views and iterations.
- TypeShade gaps: none for steps 1 and 2. `unorm8` uses `pack4x8unorm` (available).

### Item 5. BA: one workgroup per camera in the Schur camera product

- Stage: bundle (11 to 39 s). Technique: parallelize over observations, as MegBA does, and reduce in
  workgroup memory. Today `bundleSchurCamera` runs 6C + 1 invocations, each walking its camera's
  whole observation list with four dependent loads per step. New: one workgroup of 64 lanes per
  camera, lanes stride over that camera's observations, each lane keeps six `f64` partial sums, and a
  tree reduction in `workgroup<array<f64, 64>>` finishes the six rows (512 B per row, 3 KiB for six rows,
  well inside 16 KiB). Sort the camera's observation
  index so lane loads are contiguous. Keep the existing kernel as the CPU tier's `Oracle`.
- Cost: medium. Files: `src/gpu/joint-focal.shade.ts` (`bundleSchurCamera`, the cached variants),
  `src/gpu/bundle.shade.ts` (fixed-focal route), `src/gpu/joint-focal.ts`, `src/gpu/bundle.ts`,
  the oracle siblings, tests. The camera count per workgroup stays small, so grid size is
  C workgroups: at 27 cameras still fewer than a card's SMs. Item 6 is the answer to that; this
  item cuts the critical path from about 1200 serial steps to about 20.
- Expected gain: the critical path of the camera product falls about 20 to 50 times. The PCG
  iteration is the camera product plus the point product plus launches, so the iteration falls
  1.5 to 3 times (from 6.7 ms toward 2 to 4 ms). BA stage: 11 s to 4 to 7 s on the newest pin.
  SwiftShader cannot show this gain (it has no latency); the owner's record decides.
- Risk: reduction order changes the `f64` sum, so results differ in the last bits. LM
  acceptance is a threshold test and tolerates this; the CPU oracle keeps the sequential order.
  Barriers need uniform control flow (`TS8052`): every lane runs every step, with masked loads.
- Verification: tolerance, relative 1e-9 on the PCG vectors per iteration against the oracle and
  equal LM iteration counts on the fixtures of `tests/typeshade/bundle-checkpoint.test.mjs`,
  `solver-checkpoint.test.mjs`, `schur-cache.test.mjs`, `bundle-camera-webgpu.test.mjs`; final cost
  and mean reprojection error within 1e-6 relative. Benchmark: `profile-kernels.mjs --case bundle`
  (dispatch counts equal; GPU ms for the kernel may not fall on SwiftShader, state it) and the
  owner's `bundleCg` ms per unit (`timing.chunks`), which must fall.
- TypeShade gaps: G4 (a CPU twin per tiled kernel). Subgroup intrinsics G2 would shorten the tree.

### Item 6. BA: explicit reduced camera system and tiled Cholesky

- Stage: bundle. Technique: assemble the 6C x 6C reduced camera system once per LM iteration
  (one workgroup per camera pair that shares landmarks, accumulating over their common landmarks in a
  fixed order, no atomics) and factor it with a blocked Cholesky in workgroup memory (Ceres
  `DENSE_SCHUR` on a GPU). At C = 27 the matrix is 162 x 162, 26244 entries, about 210 KB in
  `f64` pairs. This is the `schurCache` assembly (`cacheBundleSchurCross`, `cacheBundleSchurRows`),
  lifted from its 16-camera guard and followed by a direct solve in place of PCG.
- Cost: large. Files: `src/gpu/joint-focal.shade.ts`, `src/gpu/joint-focal.ts`, a new
  `src/gpu/cholesky.shade.ts` (blocked, 16 x 16 tiles, 2 KiB each in `f64`), tests, and a record in
  `docs/gpu-geometry.md`. A direct solve also needs a damping retry loop for LM.
- Expected gain: 20 to 100 PCG iterations (60 to 300 dispatches) per LM step become about 10
  dispatches. A Cholesky of 162 costs 1.4 M emulated MACs, tens of microseconds of arithmetic on a
  card with a latency-bound chain of 162 column steps, so about 1 to 3 ms. The recorded
  `schurCache` result (RTX 2080, 2.4 times on the final BA, `docs/gpu-sfm.md`) is the lower
  evidence. Estimate 3 to 8 times on the BA stage. The solver also stops depending on the
  preconditioner and on `maxCgIterations`.
- Risk: memory grows as (6C)^2 (a 300-camera problem is 25 MB in `f64` pairs, fine; 1000 cameras
  is 288 MB, so keep PCG above a camera limit chosen from `device.workingBytes`). Cholesky in
  emulated `f64` needs a pivot guard for gauge-free directions.
- Verification: tolerance. The step equals the PCG step to the PCG tolerance (`cgRelativeTolerance`)
  on every BA fixture; LM trajectories compared per iteration on the CPU tier; end-to-end figures as
  item 5. Benchmark: dispatch count per BA call (must fall 5 to 10 times), GPU ms on SwiftShader, and the
  owner's `bundle` stage.
- TypeShade gaps: G4 (CPU twin), G6 (tile size from the limit). Subgroups (G2) are optional.

### Item 7. Hardware bilinear lookups for the source images

- Stage: dense propagation. Technique: upload each undistorted gray as an `r16float` or `r8unorm`
  texture; `sourceAt` becomes one `textureSampleLevel`. This keeps the 2D locality of the warped
  window in the texture cache, and the four loads and three mixes become one instruction. Needs
  the device owner to make textures (change 0077 for a kernel-written texture; an uploaded
  texture is available).
- Cost: medium. Files: `src/gpu/mvs.shade.ts`, `src/gpu/mvs.ts`, `src/gpu/mvs-residency.ts`
  (bytes per view change), `docs/dense-mvs.md`.
- Expected gain: 1.5 to 2.5 times on propagation where it is texture-bound. Unproven: it depends
  on the card (item 1 tells whether propagation is bandwidth or ALU bound).
- Risk: the CPU oracle's bilinear differs from the filter hardware (weights are 8-bit fractions).
  The "no data" sentinel (`-1`) cannot ride through filtering: the validity check moves to a
  mask texture or to `textureLoad` at the four corners.
- Verification: tolerance (G3 sets the class). Fused points within 1 percent, depth median
  within 0.1 percent on the two scenes.
- TypeShade gaps: G3.

### Item 8. Multi-resolution propagation (ACMM style)

- Stage: dense photometric. Technique: run 2 to 3 iterations at 1/4 and 1/2 resolution first, upsample
  the depth and normal as the full-resolution initialization, and run 2 to 3 iterations at full
  resolution instead of 8. Estimated 2 to 3 times on the photometric stage.
- Cost: large (`src/gpu/mvs.ts` plan, `mvs-residency.ts`, a design amendment to
  `docs/dense-mvs.md` 3.5 and 4.x). Result changes; needs a quality gate (ETH3D-style
  accuracy or the repository's TUM and south-building comparisons).
- Verification: tolerance with a quality gate against `scripts/compare-mvs-tum.mjs`. Own design
  amendment first.
- TypeShade gaps: none.

### Items later or not worth it now

| Candidate                                     | Why not now                                                                                                  |
| --------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Tiled SIFT blur, int8 matching, tiled GEMM    | Features and matching are 3 percent of a run (3.1). The SwiftShader `scanFlags` share is a barrier artifact. |
| Ring-buffer uploads, compression of transfers | Uploads are 60 ms of 75 s (3.5).                                                                             |
| Square-root BA, mixed-precision BA blocks     | Gain overlaps items 5 and 6. Revisit after item 6 with the owner's kernel table.                             |
| Power BA preconditioner                       | Item 6 removes PCG for the common case. Keep for the large-camera path.                                      |
| Decoupled look-back scan                      | Needs forward progress that WebGPU does not guarantee.                                                       |
| Zero-filled `f64` arrays by `clearBuffer`     | Saves host copies (item 9), under 1 percent of time.                                                         |
| Page stalls (`mainThread`, up to 41 s gaps)   | Not a kernel. Belongs to the multi-video record.                                                             |

### Item 9. Small items for the same pull request series

- Clear new `f64` arrays on the GPU instead of uploading zeros (`ShadeDevice.allocate*`).
- Add the `pass` (photometric or geometric) to the kernel key of item 1.

## 8. The first three pull requests

### PR 1. Run record v6: GPU time per kernel, adapter features, the profile script

1. Move `scripts/profile-kernels.mjs` into the pull request (it exists in the branch of this record).
   Keep its wrapper of `device.shade.fuse` as the model.
2. In `src/gpu/device.ts`, add a small collector in `ShadeDevice.plan`: for each report, add
   `groups` and `gpuTime` to a map keyed by `calls.join('+')`. Expose it as `statistics.kernels`.
   Do not keep more than 160 keys.
3. In `website/src/workers/sfm.worker.ts`, add the collector's rows to the record's `timing.gpu`
   and a per-stage `gpuMs`. `website/src/lib/run-record.ts` validates them (v6; v5 stays valid).
4. In `website/src/lib/send-record.ts` and the worker, add `adapter.features` and the three
   compute limits to `gpu`.
5. Update `docs/lab-run-records.md` ("What a record holds").
6. Tests: the validator accepts a v6 record and a v5 record, and rejects a negative `gpuMs`;
   `website/worker.test.mjs` sees the rows.
7. Ask the owner for one run of the pyramid sample with Dense MVS on the 5060 Ti. The table of
   section 3.2 then gets a GPU-ms column.

Checks: `npm run format:check`, `npm run typecheck`, `npm test`, `npm run test:website`,
`npm run test:shade`.

### PR 2. Fill the device before splitting a chunk (item 2)

1. Add a `minUnits` option to `GpuChunk` in `src/gpu/device.ts` and use it in `ShadeDevice.chunkSize`
   so that a key never returns less than `minUnits`, unless a measured dispatch exceeded the
   ceiling.
2. Set `minUnits` from workgroups: triangulation 16 lanes x 1024 workgroups, `scorePnP` 512 trials,
   `scorePairModels` 256 trials, `observations` all units. Replace the `Math.min(batchSize, 32)`
   floor in `src/gpu/pair-estimation.ts` and the equal floor in `src/gpu/pnp.ts`.
3. Teach `ShadeDevice` to read the chunk's `gpuTime` and to learn the rate from it when it
   exists (`device.shade.chunkRates` stays the fence-based fallback). Add a ceiling of 100 GPU
   ms per dispatch that halves the key's `minUnits` after one overrun.
4. Tests: a planner test that N tracks give `ceil(N / floor)` dispatches; the existing batch tests
   unchanged.
5. Measure: dispatches and submissions per stage on SwiftShader through
   `scripts/benchmark-sfm-pyramid-e2e.mjs`; the owner's record against `f1edec5c`.

### PR 3. The guide estimates E only (item 3)

1. In `website/src/workers/sfm.worker.ts` `guidedMatches`, replace `verifyImagePairGpu` with
   `estimateEssentialRansacGpu` at 64 to 128 hypotheses (read `src/gpu/two-view.ts` for the
   options), and build `E` from its result.
2. Remove the pose round trip (`essentialFromPose`) and `guideVerificationOptions`'s unused
   options; keep `GUIDE_RANSAC_ITERATIONS` as the hypothesis count.
3. Tests: `website/guided-matching.test.mjs` for the new helpers;
   `tests/typeshade/guided-matching-webgpu.test.mjs` unchanged.
4. Measure: guided stage time on SwiftShader through the pyramid benchmark, the guided match
   count per pair against the baseline (2 percent), landmarks and mean error against the
   baseline spread, and the owner's `guidedMatches` stage against `f1edec5c`.

### PR 4. The focal search verifies each pair once (owner record `66f7f868`)

Evidence (owner run record `20261021145305440-66f7f868`, Desert arch video sample, 48 frames at
1280, NVIDIA Turing, site `4f6b49a`): `pairVerification` ran 222 calls, 131.0 s and 52,828
dispatches. `appendPairModelRows` took 160.2 s of 265 s GPU time in 53,478 dispatches.

1. Cause (fact, `scripts/measure-focal-search-cost.mjs`): the focal search of #162 verified the
   10 pairs of its first 5 views at each of 17 grid points and once more at the refined focal.
   On the pyramid sample that is 180 of 208 calls.
2. Change (#180): `src/reconstruction/focal-pairs.ts` keeps the eight-point solution of each trial
   of a pair, which does not depend on the focal, and derives each grid point's verification on
   the host. Details, accuracy and measurements: `docs/auto-focal.md`, section 8.
3. Result (SwiftShader): pair verifications of the search 180 to 0; pyramid run 208 to 28 calls,
   84,379 to 68,774 dispatches, 80.3 to 49.4 s GPU time. The focal estimates equal the earlier
   ones on the four COLMAP samples and the pyramid (zero distortion).
4. Not changed: `appendPairModelRows`. The order of its row appends is the order of one QR update,
   so it is not a free choice. #178 owns the occupancy floors.
5. Open: the owner's time on the card. Expected, as an inference: the stage falls from 131 s to
   about 30 s.

After these, order: item 4 (steps 1 and 2 are exact and small), item 5, item 6, then 7 and 8. If the
owner's kernel table (PR 1) shows dense propagation or BA holds the GPU time, move items 4 to 6
before PR 2 and PR 3.

## 9. Open items

| Item                                                   | Reason                                                                                       | Next action                                                |
| ------------------------------------------------------ | -------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| The fence-latency reading of 3.3 is an inference       | Records hold no kernel time.                                                                 | PR 1, then one owner run.                                  |
| The bundle drop (36 to 11 s) is not isolated           | #152 and #164 landed in one interval.                                                        | PR 1's table; an owner run on `ed932d77`'s input.          |
| `shader-f16` and `subgroups` on the owner's card       | The record has no feature list.                                                              | PR 1.                                                      |
| SwiftShader cannot show latency or occupancy gains     | CPU emulation.                                                                               | Each item states its physical-GPU measure.                 |
| Guided matching and matching kernels were not profiled | The scripts cover MVS, BA, SIFT and pair verification only.                                  | Done in PR 3 (`--case guided`).                            |
| TypeShade gaps G1 to G6                                | Drafted here, not filed. The lab files them on typeshade/typeshade.                          | The owner or the next session files them, as the log says. |
| Item 6 camera limit and memory                         | Depends on the working-byte policy.                                                          | Decide in item 6's own record.                             |
| Quality gate for item 8                                | The repository has TUM and south-building comparisons; no accuracy bar for multi-resolution. | Owner decision before item 8.                              |

## 10. References

- Wu, Agarwal, Curless, Seitz. Multicore Bundle Adjustment. CVPR 2011. https://grail.cs.washington.edu/projects/mcba/
- Ren, Liang, Yan, Mai, Liu, Liu. MegBA: A GPU-Based Distributed Library for Large-Scale Bundle Adjustment. ECCV 2022. https://arxiv.org/abs/2112.01349
- Demmel, Sommer, Cremers, Usenko. Square Root Bundle Adjustment for Large-Scale Reconstruction. CVPR 2021. https://arxiv.org/abs/2103.01843
- Weber, Demmel, Chan, Cremers. Power Bundle Adjustment for Large-Scale 3D Reconstruction. arXiv 2022 (v4 2023). https://arxiv.org/abs/2204.12834
- Agarwal, Snavely, Seitz, Szeliski. Bundle Adjustment in the Large. ECCV 2010. https://grail.cs.washington.edu/projects/bal/
- Haidar, Tomov, Dongarra, Higham. Harnessing GPU Tensor Cores for Fast FP16 Arithmetic to Speed up Mixed-Precision Iterative Refinement Solvers. SC 2018.
- Bleyer, Rhemann, Rother. PatchMatch Stereo. BMVC 2011.
- Galliani, Lasinger, Schindler. Massively Parallel Multiview Stereopsis by Surface Normal Diffusion. ICCV 2015.
- Xu, Tao. Multi-Scale Geometric Consistency Guided Multi-View Stereo. CVPR 2019. https://arxiv.org/abs/1904.08103
- Xu, Tao. Planar Prior Assisted PatchMatch Multi-View Stereo. AAAI 2020. https://arxiv.org/abs/1912.11744
- Schonberger, Zheng, Pollefeys, Frahm. Pixelwise View Selection for Unstructured Multi-View Stereo. ECCV 2016.
- Schonberger, Frahm. Structure-from-Motion Revisited. CVPR 2016.
- Johnson, Douze, Jegou. Billion-scale similarity search with GPUs. IEEE Transactions on Big Data 2021. https://arxiv.org/abs/1702.08734
- Arandjelovic, Zisserman. Three things everyone should know to improve object retrieval. CVPR 2012.
- AliceVision PopSift. https://github.com/alicevision/popsift
- Cigolle, Donow, Evangelakos, Mara, McGuire, Meyer. A Survey of Efficient Representations for Independent Unit Vectors. JCGT 2014. https://jcgt.org/published/0003/02/01/
- Kerbl, Kopanas, Leimkuehler, Drettakis. 3D Gaussian Splatting for Real-Time Radiance Field Rendering. SIGGRAPH 2023. https://arxiv.org/abs/2308.04079
- Ye et al. gsplat: An Open-Source Library for Gaussian Splatting. https://arxiv.org/abs/2409.06765
- Adinets, Merrill. Onesweep: A Faster Least Significant Digit Radix Sort for GPUs. https://arxiv.org/abs/2206.01784
- Merrill, Garland. Single-pass Parallel Prefix Scan with Decoupled Look-back. NVIDIA TR NVR-2016-002.
- Haar, Aaltonen. GPU-Driven Rendering Pipelines. SIGGRAPH 2015 Advances in Real-Time Rendering.
- Barrett. Sparse Virtual Textures. GDC 2008.
- Harris. Optimizing Parallel Reduction in CUDA. NVIDIA 2007.
- Lewis. Fast Normalized Cross-Correlation. Vision Interface 1995.
- Chrome for Developers. What's new in WebGPU (Chrome 134), subgroups. https://developer.chrome.com/blog/new-in-webgpu-134

Citations were checked for title, authors and venue only where a link is given. The others are
cited from the authors' publications and need a check before a pull request relies on them.
