# TypeCV's use of TypeShade: compute entries versus kernel functions

Status: done. Read-only audit, 2026-10-26. No file of either repository was changed.

Record: a read-only audit written on 2026-10-26 by a Claude Code sub-agent at the coordinator's request, kept as the evidence of [`../library-structure.md`](../library-structure.md) section 3.6 and section 6. The numbers are those of `main` at `5e1c95c` with the compiler pin `d36ac502`.
Each statement carries a label: **Fact** (read in a source, counted by a script, or measured by
a compiler probe) or **Inference** (a conclusion from facts). A file:line names the evidence.

## 0. Configuration and method

- **Fact.** TypeCV: `bac4261` on branch `claude/3d-depth-gradient-collapse-ab4nf7`. `src/`,
  `tests/`, `docs/` and `package.json` have no local change (`git status`). The `scripts/` files of
  pull request #239 were in progress beside this audit; the audit did not read them.
- **Fact.** TypeShade pin: `d36ac502` (`package.json`), installed as `node_modules/typeshade`.
  `kernel-lower.ts`, `parallel-loop.ts`, `host-kernel.ts`, the surface document and the design
  rules there are byte-identical to that commit.
- **Fact.** The TypeShade checkout is `e03ff7e`, 31 commits after the pin. It adds 0085
  (`accepted`) and 0086 (`implemented`). `kernel-lower.ts` and `parallel-loop.ts` did not change
  after the pin, so the proof and lowering facts below also hold at `e03ff7e`.
- **Fact.** Method: source reading; counting scripts; 12 compiler probes (§0.1). The probes
  compiled scratch sources with the pinned compiler (an esbuild 0.25.12 bundle of
  `node_modules/typeshade/src`, Node v22.22.0, TypeScript 5.9.3), outside the repository.
- **Fact.** Not run: no TypeCV test, no WebGPU run, no hardware run. No performance number is
  claimed. The probes report the compiler's verdict (diagnostics and the host view's comment).

### 0.1 Compiler probes (pinned compiler, 2026-10-26)

| Probe | Source shape (kernel function unless stated) | Result |
| --- | --- | --- |
| P1 | `rgbaToGray` as a loop over `offset..offset+count` | GPU loop |
| P2 | loop calls a helper that reads a module `storage` binding | CPU: "it reaches \"lut\", which the call does not pass" |
| P3 | a local arrow function captures the kernel's `array<f32>` | GPU loop |
| P4 | a plain helper takes `array<f32>` | `TS8020` |
| P5 | one-iteration loop writes `state[0]`; variant writes `state[i * 2 + c]` | `TS8070` R3; variant GPU |
| P6 | trip count read from `counts[0]` before the loop; guard `if (i >= counts[0]) continue` | CPU; guard form GPU |
| P8 | `workgroup<…>` and `workgroupBarrier()` in the loop | `TS8070` R6: "Write a @compute entry to use it." |
| P9 | `out.push(i)` with `return out.length`; nested 2D loops; record `out[i * 4 + k]` | GPU (push, grid, record) |
| P9b | record at `(i * slots + s) * 4 + c` | `TS8070` R3 |
| P11 | `array<f64>` map; `best[k] = min(best[k], v)` on `u32`; `f32` scatter `+=` | f64 GPU; min scatter `TS8070` R3; f32 scatter CPU |
| P13 | a kernel function calls a kernel function; `@workgroup_size([16])`; struct parameter | `TS8099`; the other two accepted |
| P20 | `bins[keys[i]] += 1`; `out[stride * i]`; `out[stride * i]` and `[stride * i + 1]`; `out[base + i]`; `findIndex`; `reduce` | scatter GPU; one strided write GPU; two strided writes, and `base + i`, `TS8070` R3; methods GPU |
| P30 | `array<S>` of f64 fields in and out, `continue`; record with two write forms | struct form GPU; mixed record `TS8070` R3 |

- **Fact.** Surface §65 lists `min` and `max` among the scatter forms. P11 and P20 measured that
  `best[k] = min(best[k], v)` is refused. **Inference:** a compiler defect or a surface error.

## 1. The answer in short

- **Fact.** 30 `.shade.ts` files, 13,186 lines, 251 `@compute` entries, 223 distinct entry names
  (28 names exist in both BA modules). There are 0 array methods inside kernels.
- **Fact (correction to the baseline).** Rule 8.22 makes a kernel function of any exported
  non-entry function that takes `array<T>`. `src/gpu/epipolar.shade.ts:3-133` holds 8 of them.
  None has `@workgroup_size`, which the baseline's count required. `EpipolarSession`
  (`src/gpu/epipolar.ts:162-176`) runs them in one plan with `scope.scratch`. Only tests and
  `scripts/benchmark-gpu.mjs` call it; the SfM pipeline does not.
- **Fact.** Entry kinds (script over the entry bodies): 56 single-invocation `@compute([1])` steps;
  29 cooperative entries (workgroup memory, barriers or local ids); 18 `...Oracle` siblings;
  12 entries with atomics; 136 one-invocation-per-element maps.
- **Inference.** Most entries are intentional. Four needs force them today:
  1. Device-side control without a host round trip: the 56 single-invocation steps, counters
     read by later dispatches, and 11 entries that dispatch indirectly. A kernel function computes its trip
     count on the host and returns its reductions and its push count to the host.
  2. Workgroup memory and barriers (29 entries). Rule 8.22 R6 refuses both in a kernel loop (P8).
  3. Atomics with a returned value or compare-exchange (TSDF and cloud hashing). The proof
     accepts only `a[k] op= e` on an integer array (R4 refuses a read atomic result).
  4. The idle-lane hang A1 (`docs/hardware-observations.md:49-70`). The kernel lowering emits
     the early return that A1 forbids (`kernel-lower.ts:756-767`).
- **Inference.** TypeCV did not take TypeShade's advantages in four places:
  1. About 60 barrier-free per-element maps in the image families express directly as kernel
     loops. The compiler would own the workgroup count, the 65,535-workgroup split and the length
     checks, and `fuse` could join the chains that read at `i`.
  2. `scan.shade.ts`, its copy in `pair-preparation.shade.ts` and about 25 flag, scatter, advance
     and mark entries re-implement change 0078 `push`.
  3. Exact reductions (a maximum, integer counts) and histograms re-implement kernel reductions
     and the integer scatter.
  4. 10 of the 18 oracle siblings add floating-point lanes in order where the GPU entry uses a
     64-lane tree, so the CPU tier does not give the GPU's bits (§5). The 2 scan oracles are exact
     integer sums. The 6 MVS oracles drop the per-workgroup pose cache. A kernel reduction gives
     one tree order on every tier (Rule 7.2).
- **Inference.** The compiler gaps of §6 decide the rest. With them closed, about 1,600 to 1,700
  lines of TypeCV could go. Phase 1 of §7 removes about 300 more with no compiler change.

## 2. The kernel files

**Fact** (scripts over `bac4261`). Entries: S single-invocation, M map, C cooperative, A atomic,
O oracle. KF = kernel functions. WG = workgroup variables. At = atomic operations. Bar =
barriers. Builtins: g = `global_invocation_id`, l = `local_invocation_id`, w = `workgroup_id`.
No entry reads `.y` or `.z` of a builtin, so every dispatch is 1D. No file has a texture. Host:
P = `device.plan`, PS = `planSteps`/`planStepsMany`, Ch = chunk steps (`units:`), I = indirect
counts, K = `{ constants }`. Every host path ends in `ShadeDevice.plan` → `shade.fuse`
(`src/gpu/device.ts:747-753`).

| Module (`src/gpu/`) | Lines | S/M/C/A/O | KF | WG | At | Bar | Builtins | Overrides | f64 | Host |
| --- | --: | --- | -: | -: | -: | -: | --- | --- | --- | --- |
| `bundle.shade.ts` | 1,125 | 8/13/6/0/2 | 0 | 6 | 0 | 10 | g,l,w | 3 | yes | PS, Ch, K |
| `joint-focal.shade.ts` | 1,383 | 15/17/5/0/2 | 0 | 5 | 0 | 8 | g,l,w | 1 | yes | PS, Ch, K |
| `mvs.shade.ts` | 1,821 | 0/10/6/0/6 | 0 | 1 | 0 | 1 | g,l | 0 | no | PS, Ch |
| `tsdf.shade.ts` | 871 | 0/1/0/6/0 | 0 | 0 | 27 | 0 | g | 0 | no | PS, Ch |
| `recovery.shade.ts` | 675 | 8/6/1/0/1 | 0 | 1 | 0 | 2 | g,l,w | 0 | yes | P, PS, Ch |
| `klt.shade.ts` | 649 | 1/9/0/0/0 | 0 | 0 | 0 | 0 | g | 0 | no | PS, Ch |
| `pair-models.shade.ts` | 548 | 3/5/0/0/0 | 0 | 0 | 0 | 0 | g | 0 | yes | PS, Ch, I |
| `sift.shade.ts` | 497 | 2/10/0/0/0 | 0 | 0 | 0 | 0 | g | 0 | no | P |
| `orb.shade.ts` | 482 | 3/9/0/0/0 | 0 | 0 | 0 | 0 | g | 0 | no | P |
| `kernels/float-matching.shade.ts` | 480 | 0/4/3/0/0 | 0 | 7 | 0 | 9 | g,l,w | 1 | no | PS |
| `triangulation.shade.ts` | 464 | 1/4/1/0/1 | 0 | 2 | 0 | 2 | g,l,w | 0 | yes | PS, Ch |
| `pnp.shade.ts` | 455 | 10/4/2/0/2 | 0 | 3 | 0 | 4 | g,l,w | 0 | yes | PS, Ch, I |
| `features.shade.ts` | 427 | 0/12/1/0/0 | 0 | 2 | 0 | 2 | g,l | 0 | no | P, PS, Ch |
| `hough.shade.ts` | 369 | 0/4/0/5/0 | 0 | 0 | 9 | 0 | g | 0 | no | PS |
| `p3p.shade.ts` | 361 | 0/1/0/0/0 | 0 | 0 | 0 | 0 | g | 0 | yes | P, PS, I |
| `planar-depth.shade.ts` | 311 | 0/4/0/0/0 | 0 | 0 | 0 | 0 | g | 0 | no | PS |
| `dense-mesh.shade.ts` | 285 | 0/4/0/0/0 | 0 | 0 | 0 | 0 | g | 0 | no | PS |
| `numerics.shade.ts` | 281 | helpers only | 0 | 0 | 0 | 0 | – | 0 | yes | – |
| `consensus.shade.ts` | 222 | 2/3/1/0/1 | 0 | 2 | 0 | 2 | g,l,w | 0 | yes | PS, Ch, I |
| `projective.shade.ts` | 217 | helpers only | 0 | 0 | 0 | 0 | – | 0 | yes | – |
| `slic.shade.ts` | 208 | 0/4/0/0/0 | 0 | 0 | 0 | 0 | g | 0 | no | PS |
| `pair-preparation.shade.ts` | 179 | 2/4/1/0/1 | 0 | 1 | 0 | 3 | g,l,w | 0 | yes | P, PS |
| `cloud-knn.shade.ts` | 162 | 0/1/0/0/0 | 0 | 0 | 0 | 0 | g | 0 | no | PS, Ch |
| `clahe.shade.ts` | 161 | 0/3/0/0/0 | 0 | 0 | 0 | 0 | g | 0 | no | P |
| `observation-evaluation.shade.ts` | 138 | 0/2/1/0/1 | 0 | 2 | 0 | 2 | g,l,w | 0 | yes | PS |
| `epipolar.shade.ts` | 133 | – | 8 | 0 | 0 | 0 | – | 0 | no | P |
| `inverse-depth.shade.ts` | 81 | 0/1/0/0/0 | 0 | 0 | 0 | 0 | g | 0 | yes | PS |
| `cloud-voxel.shade.ts` | 78 | 0/0/0/1/0 | 0 | 0 | 1 | 0 | g | 0 | no | PS, Ch |
| `scan.shade.ts` | 72 | 1/0/1/0/1 | 0 | 1 | 0 | 3 | l,w | 0 | no | via `scan.ts` |
| `global-descriptor.shade.ts` | 51 | 0/1/0/0/0 | 0 | 0 | 0 | 0 | g | 0 | no | PS, Ch |
| **Total** | 13,186 | 56/136/29/12/18 | 8 | 33 | 37 | 48 | | 5 | 13 files | |

- **Fact.** The 11 indirect entries: `preparePairModels`, `appendPairModelRows`,
  `solvePairModelNull`, `finishPairModels`, `scorePairModels`, `scorePairModelsOracle`
  (`src/gpu/pair-estimation.ts:345-441`), and `samplePnP`, `solveP3p`, `scorePnP`,
  `scorePnPOracle`, `selectPnP` (`src/gpu/pnp.ts:258-305`).
- **Fact.** `{ constants }` (0084) is passed at `src/gpu/bundle.ts:313,593` and
  `src/gpu/joint-focal.ts:231`.
- **Fact.** In a plan, an entry call is a group of its own (surface §67; §71 fact F8). So TypeCV's
  plans give one submission per plan but never a fused dispatch. Only `EpipolarSession` can fuse.
- **Fact.** Four entries have no production caller, only `tests/typeshade/indexed-observations-probe.mjs`:
  `projectObservations`, `observationPartials`, `observationPartialsOracle` and
  `filterObservationDepth` (`src/gpu/triangulation.shade.ts:82-131,281-297`).

## 3. The kernel families

Needs (only an entry gives today): **DC** device-side count or state word read by a later
dispatch with no readback; **1I** a single-invocation step; **WG** workgroup memory and barriers;
**AT** an atomic with a returned value or compare-exchange; **IN** indirect dispatch; **OV**
override constants; **MAP** a specific invocation-to-work mapping; **A1** a long loop per lane
under the idle-lane rule. "Lines" is the **Inference** of the lines a conversion deletes, after
the change it names (§6); "now" means with no compiler change.

| Family | Entries: work per invocation | Needs | Verdict | Lines |
| --- | --- | --- | --- | --- |
| SIFT (`sift.shade.ts`) | 8 maps: one pixel (resample, blur tap loop, DoG, extremum flag) or one keypoint (36-bin orientation, 128-D descriptor); `siftScatterExtrema`, `siftScatterOriented` write row = device count + scan offset; `siftAdvance`, `siftMark` move counters | DC, 1I | **mixed**: maps convertible (`for` over the range; `blurOut[outOffset + i]` needs a shifted range, P20; `siftOrient`'s `k * 2 + written` and `siftDescribe`'s `return` need a rewrite); compaction convertible to `push` after G-C | ~80 after G-C |
| ORB (`orb.shade.ts`) | maps: gray, resize, FAST, Harris, describe one corner; flag and scatter corners; `orbPickTop` ranks a few thousand entries in one invocation (`orb.shade.ts:250-292`) | DC, 1I | **mixed**: maps convertible; `orbPickTop` intentional (an order-free ranking in one lane); compactions to `push` after G-C | ~60 after G-C |
| KLT (`klt.shade.ts`) | maps: pyramid, Scharr, `kltTrack` (all levels and iterations of one track), check, eigen map, select, suppress; `kltEigenMax` + `kltEigenMaxFinal` two-level max (`klt.shade.ts:526-548`); `kltCompact` | DC (`kltSelect` reads `kltMax[0]`, `klt.shade.ts:582`) | **convertible**: max as one kernel reduction (exact); the host reads the max before the selection anyway (`klt-tracker.ts:672-674`), so it can pass it to `kltSelect` as a scalar; `kltCompact` to `push` | ~35 now, ~15 after G-C |
| FAST/BRIEF and matching (`features.shade.ts`, `kernels/float-matching.shade.ts`) | maps per pixel or per query; `matchBriefTiles`, `nearestFloatTiles`, `…Wide`, `…GuidedTiles` load candidate slabs into workgroup memory; scalar twins `matchBrief`, `nearestFloatScalar`, `nearestFloatGuided`; `selectTileCorners` writes `scores[bestIndex] = -1` | WG, OV (`GUIDED_WG`) | **mixed**: tiled matchers intentional; scalar matchers and maps convertible (P1 form); `selectTileCorners` refused (R3) without a rewrite; three compactions to `push` after G-C | ~50 after G-C |
| Pair preparation, consensus, pair models (`pair-preparation`, `consensus`, `pair-models`, `epipolar`) | f64 maps per match; per-model QR row appends and null solve; 64-lane score tree per (model, block) with oracle; `selectPairModel` walks trials in order and shrinks the adaptive budget on device; batch counts for indirect dispatch | DC, 1I, WG, IN | **intentional**; the scan copy (`pair-preparation.shade.ts:86-145`, a second `scan.shade.ts` for another state layout) can go after G-C; the 8 kernel functions of `epipolar.shade.ts` serve only the f32 `EpipolarSession` | ~60 after G-C |
| Two-view recovery (`recovery.shade.ts`) | LM state machine: 8 single-invocation steps, 6 maps over matches or candidates, a normal-equation tree with oracle | DC, 1I, WG | **intentional** | ~15 after G-A |
| P3P / PnP (`p3p.shade.ts`, `pnp.shade.ts`) | one Grunert problem per lane with all roots (`p3p.shade.ts:195`); PnP RANSAC and LM on device: sampling, scoring tree, selection, pose steps | DC, 1I, WG, IN, A1 | **intentional**; `solveP3p` is a pure map, but PnP dispatches it indirectly (`pnp.ts:268`) | ~30 after G-A |
| Triangulation (`triangulation.shade.ts`) | `triangulateTracks`: one track per lane, Jacobi SVD up to 160 sweeps; prepare and filter maps; summary | A1, 1I | **intentional** (A1 was found on this kernel) | ~70 test-only entries (§2) |
| Observations (`observation-evaluation.shade.ts` + `observationSummary`) | project and depth-filter maps; 64-lane tree of (error, count); summary in one lane | WG (only to reduce) | **convertible**: one kernel function returning `(error, count)`; `observations.ts:208-211` reads the summary at once, so a host result costs no extra round trip; the f64 error sum changes order (§5) | ~45 (owner decision, §5) |
| BA fixed and joint focal (`bundle`, `joint-focal`) | linearize per observation; point and camera blocks; segmented cooperative Schur products; PCG and LM steps in one lane; activity checkpoints | DC, 1I, WG, OV | **intentional**; the copy (§6 G-B) and 4 oracles are compiler gaps | ~900 after G-B, ~30 after G-A |
| Inverse depth (`inverse-depth.shade.ts`) | one observation per lane: f64 linearization into a 23-word record | none | **convertible**: `array<S>` in and out (P30) lets the runtime pack; early returns become `continue` | ~50 host now (`inverse-depth.ts:47-105`) |
| MVS (`mvs.shade.ts`) | undistort per pixel; initialize, seed, red-black propagate, geometric start, filter, stream costs, each with a per-workgroup relative-pose cache and a barrier (`mvs.shade.ts:122,610-620`); fusion flags and scatter; `mvsCopyF32/U32` copy a view into a slot | WG (cache), MAP (red-black), DC | **mixed**: cache and mapping intentional; 6 oracles are G-A; copies are a runtime gap (G-J); fusion compaction to `push` after G-C | ~60 after G-A, ~20 after G-J |
| Planar depth (`planar-depth.shade.ts`) | tile plane fit (record `t * 14 + c`), propose, verify, finish per pixel | none | **convertible** (affine writes) | ~5 host |
| TSDF (`tsdf.shade.ts`) | block claims by compare-exchange (`tsdf.shade.ts:433`); integrate by fixed-point `atomicAdd` (`:542-546`); carve; extract appends with `atomicAdd` (`:855-858`), then the host sorts (`tsdf.ts:724-734`) | AT | **intentional** for the hash table; integrate and carve are integer scatters a kernel could do; extract to `push` | ~0 recommended |
| Cloud clean (`cloud-knn`, `cloud-voxel`) | k-NN per sorted point, written through `visitOrder` (locality); voxel claim by compare-exchange (`cloud-voxel.shade.ts:61`) | MAP, AT | **intentional** | 0 |
| Mesh (`dense-mesh.shade.ts`) | node field per node; classify and emit 16-word rows after the scan; vertices per welded key | DC | **convertible**: the host reads the cube count anyway (`dense-mesh.ts:156`), so `push` adds no round trip | ~40 now |
| SLIC (`slic.shade.ts`) | Lab per pixel; seeds and center update per cluster in a fixed gather order; assign per pixel | none | **convertible**, writes at `5 * k + c` and `i` | ~10 host |
| Hough (`hough.shade.ts`) | gray, blur, gradient, suppression, peaks, segments per element; `houghGrow` relaxes in place with a change counter; `houghCollect` appends edges with `atomicAdd`; `houghVote` adds votes with `atomicAdd` | AT (grow) | **mixed**: vote = integer scatter (P20), collect = `push` (the host reads the count, `hough.ts:472`), maps convertible; grow intentional | ~40 now |
| CLAHE (`clahe.shade.ts`) | histogram of one tile in one lane (`clahe.shade.ts:39-55`); clip and map of one tile; apply per pixel | none | **mixed**: histogram as a scatter over pixels (zeroing in a separate call, since a loop that writes in place and a scatter cannot share a function); map stays; apply convertible | ~0 net |
| Global descriptor (`global-descriptor.shade.ts`) | nearest word of one descriptor, sum in order | A1 (long loop) | **convertible** after G-F | ~5 host |
| Scan (`scan.shade.ts`, `scan.ts`) | 64-lane Hillis-Steele block scan with barriers, its oracle, then one lane walks every block (`scan.shade.ts:60-72`) | WG, DC | **convertible** to `push` after G-C; used by 9 host modules | ~100 after G-C |

- **Fact.** `docs/gpu-sfm.md:684-699` forbids an idle lane to return before a loop. Its cause is
  A1: workgroups with idle lanes of `triangulateTracks` hung an RTX 5060 Ti
  (`DXGI_ERROR_DEVICE_HUNG`, `docs/hardware-observations.md:49-70`). TypeShade issue #703 is
  recorded as open (`docs/hardware-observations.md:345`).
- **Fact.** The kernel lowering starts every map invocation with `if (_k >= _n) { return; }`
  (`node_modules/typeshade/src/core/passes/kernel-lower.ts:756-767`).
- **Inference.** So a conversion of a loop-heavy map (SIFT describe and orient, global descriptor,
  P3P, k-NN, `kltTrack`) waits for G-F or for a hardware gate. `klt.shade.ts:20-22` already
  records that `kltTrack` tolerates a returning tail lane.

## 4. Host-side work

- **Fact.** About 70 host expressions compute workgroup counts (`Math.ceil(n / 64)` and the like;
  the most in `features.ts` 11, `float-matching.ts` 10, `pair-estimation.ts` 8, `pnp.ts` 7).
  37 lines in 17 host files read `device.maxKernelElements`, which is 65,535 × 64
  (`device.ts:371`).
  `hough.ts:107,237-251` splits passes at that limit (`cappedPass`).
- **Fact.** A kernel loop numbers its invocation as `gid.x + gid.y * (nwg.x * wg)`, so a dispatch
  past 65,535 workgroups spills into y (`node_modules/typeshade/src/core/kernel-workgroup.ts`).
  **Inference:** for a converted map the counts, the limit checks and `cappedPass` go away.
- **Fact.** `chunkedPass` exists three times: `cloud-clean.ts:46`, `tsdf.ts:147`,
  `planar-depth.ts:10`. **Inference:** a host helper in one file removes two copies with no
  compiler change; automatic chunking (G-I) would remove the third.
- **Fact.** Compaction runs on the device, but the host plans it: `planFlagScan` (`scan.ts:20-26`)
  and `scanBlockCount` sizing in `features`, `float-matching`, `klt-tracker`, `mvs`,
  `mvs-residency`, `orb`, `sift`, `dense-mesh`. This re-implements 0078 `push`.
- **Fact.** Host work between dispatches:
  - Hough: a full gradient readback and quantile thresholds on the CPU (`hough.ts:340-346`); one
    readback of `changed` per hysteresis round (`hough.ts:361-376`); a readback of the whole
    accumulator and peak map, then a sort (`hough.ts:494-495,548-560`).
  - Mesh: the host flattens the rows and welds the keys by sort and unique
    (`dense-mesh.ts:179-187`, `src/reconstruction/dense-mesh.ts:437`).
  - Cloud clean: the host sorts the points into a grid (`buildPointGrid`, `cloud-clean.ts:80`),
    then un-permutes the result (`cloud-clean.ts:109-110`) and reduces voxels in double
    (`reduceVoxels`, `cloud-clean.ts:204`).
  - TSDF: the host sorts the extracted points, because the device appends in scheduling order
    (`tsdf.ts:724-734`).
  - KLT: every frame reads all checked tracks to update the host track table
    (`klt-tracker.ts:658`) and reads the maximum eigenvalue (`klt-tracker.ts:673`).
  - Inverse depth: the host packs 34 f64 per measurement and unpacks 23 per result
    (`inverse-depth.ts:47-105`).
- **Fact.** Readbacks inside loops are checkpoints by design: the RANSAC budget every fourth
  batch (`pair-estimation.ts:497`, `pnp.ts:321`), the LM and PCG activity (`pnp.ts:345`,
  `bundle.ts:625`, `joint-focal.ts:595,815`), the hysteresis change word.
- **Fact.** Constants are copied, not shared: `hough.ts:108-110` repeats `MAX_SEGMENTS`,
  `SEGMENT_STRIDE`, `TIE` of `hough.shade.ts:67-70`; `tsdf.ts:121` repeats `EMPTY`; seven host
  files declare `LANES`, six of them as the literal 64. **Inference:** surface §64 already lets a
  host import an exported shader constant, so this is a facility not taken, not a gap.
- **Fact.** `typeshade/algorithms` (`histogram`, `exclusiveScan`, `compact`) is not used.
  **Inference:** its scan reads about `2 n sqrt(n)` elements (surface §72), so `push` is the
  better target.

## 5. Bit-exactness

The references: TypeCV Rule 1.1 (one answer on every tier) and Rule 1.2 (proved bit for bit
where one answer exists, else within a stated bound) (`docs/design-rules.md`). TypeShade Rule
7.2 (a kernel reduction folds in the 256-wide tree order on every tier) and Rule 11.13 (a fused
plan gives the bits of the calls run apart).

- **Fact.** The GPU entries reduce in a 64-lane tree (stride 32 to 1); the oracle siblings add the
  64 lanes in order (`bundle.shade.ts:319-354`, `consensus.shade.ts:116-147`). The
  single-invocation finish steps then add the blocks in order (`bundle.shade.ts:356-377`).
- **Inference.** So the CPU tier and WebGPU differ in the last bits of each floating-point partial
  sum today. The tests accept this: `bundle-reduction-webgpu.test.mjs` holds product errors below
  2e-10 and costs within 1e-6; the CPU test holds the CPU modes equal to each other only.

| Conversion | Effect on the bits | Proof it needs |
| --- | --- | --- |
| A map, one element per iteration, sequential arithmetic in the body | **Inference:** none; the same IR runs, the CPU tier is the same generated code | CPU-tier outputs of old and new equal (`deepEqual`); the family's WebGPU test unchanged |
| An exact fold: integer count, `max`/`min` of non-NaN values, `|`, `&` | **Inference:** none in any order. A kernel `max` starts from the lowest finite f32 and then combines the variable's own start, so `let m = 0.` keeps today's 0 floor | as above; add a case with ties and with zeros |
| Compaction by `push` instead of flag, scan and scatter | **Inference:** none; both keep the sequential order | as above; the counts and the row order equal |
| Atomic append to `push` (TSDF extract, Hough collect) | **Inference:** device order becomes sequential; the host sort and the vote counts keep the final results | as above, plus the readback count |
| A floating-point sum (BA metrics, PnP and consensus costs, observation errors) | **Inference:** changes on both tiers (tree of 256 against 64-tree plus blocks); afterwards CPU = WebGPU bit for bit. Ties in `c === count && e < cost` (`consensus.shade.ts:182`) can pick another model | a behaviour change: owner decision under TypeCV Rule 2.2; new digests for `PREVIOUS_RESULTS` (`tests/typeshade/pair-batch.test.mjs:82`, `pnp-batch.test.mjs:44`) |
| Oracle sibling removed by G-A (CPU runs the tree entry in passes) | **Inference:** the CPU tier then gives the GPU's 64-tree bits; CPU digests change, WebGPU does not | owner decision; CPU digests renewed once |
| Chunking a kernel loop | **Fact:** independent iterations give the same result for any split (R1-R6). **Inference:** a reduction split over calls changes its tree | keep reductions in one call, or fold per chunk in a fixed order |
| Helpers on the CPU | **Fact:** a host call of a helper runs at f32 (Rule 11.7), as the CPU tier of an entry does | no change |

- **Fact.** TypeCV opens only `webgpu` or `cpu` (`device.ts:17`), so WebGL2 and GLSL
  (TypeShade Rule 1.2) do not enter these conversions.
- **Fact.** The tests that hold numbers: CPU-tier digests (`pair-batch`, `pnp-batch`, and the
  `sha256` cases of `bundle-fixed-jacobian-cases`, `tsdf-staging-cases`,
  `tsdf-inline-votes-cases`, `multi-video-combined-cases`, `cpu.test.mjs`); WebGPU equality
  (`deepEqual`) in `orb-webgpu`, `klt-webgpu`, `slic-webgpu`, `clahe-webgpu`,
  `global-descriptor-webgpu`, `tsdf-webgpu`, `dense-mesh-webgpu`, `hamming-match-webgpu`,
  `float-matching-webgpu`, `inverse-depth-webgpu`, `planar-depth-webgpu`, `hough-webgpu`.

## 6. Compiler gaps, ranked by the TypeCV lines they would remove

**Fact** for each rule, code and file; **Inference** for each line count.

| Rank | Gap | Today (rule, code, TypeCV workaround) | What a compiler change must allow | Lines |
| -: | --- | --- | --- | --: |
| 1 | G-B: share code across modules with different layouts | `TS8020` refuses `array<T>` as a helper parameter (P4; Rules 8.22-8.23). Helpers read bindings by name. `bundle.shade.ts` and `joint-focal.shade.ts` differ in record sizes (18 against 21 cross words, `6C` against `6C + 1` vectors, `array<f64, 6>` against `<f64, 7>` tiles) and repeat 896 of 1,125 lines (sibling audit; difflib here: 955) | a helper that takes a storage array by reference (WGSL `ptr<storage, …>`; monomorphized per binding where `unrestricted_pointer_parameters` is absent), or a module parameterized by constants that a file instantiates twice | ~900 |
| 2 | G-C: device-resident results of a kernel function | the trip count is computed on the host (`kernel-lower.ts:10-12`; P6); a reduction result and a `push` count reach only the host (surface §71 "await sum after fuse"; 0078 §5 "read back"). TypeCV keeps counts on the device: `scan.shade.ts`, the pair scan copy, about 25 flag, scatter, advance and mark entries, single-lane finish steps | a `push` that appends after a count held in a `Resident<Uint32Array>` and leaves the new count there; a reduction into a `Resident`; a loop bound read from a `Resident`, lowered to an indirect dispatch whose counts a compiler-made entry writes | 300-400 |
| 3 | G-A (TypeCV's G4): barriers on the CPU tier | Rule 8.24: an entry that reaches a barrier needs WebGPU or WebGL2; "the runtime does not ship" the lockstep. The oracle already runs it (`compileModule(m).dispatch`, surface §25). TypeCV keeps 18 `...Oracle` siblings, the `mvs…At` splits and the backend branches (29 per the sibling audit; 43 host lines name an oracle) | the host CPU tier of `entry(bindings, workgroups)` runs barrier entries in passes, as `dispatch` does | ~250 |
| 4 | G-I: automatic chunking of proved loops | 0074: "the compiler does not cut a loop". TypeCV writes `chunkedPass` three times, `cappedPass`, and per-module chunk generators | a kernel call that splits a loop the proof accepted into timed chunks under `timeBudgetMs` | ~100 |
| 5 | G-J: a partial write into a `Resident` | `write(value)` replaces the whole array (surface §65). `mvsCopyF32`, `mvsCopyU32` (`mvs.shade.ts:1418-1433`) and their staging buffers copy a view into a slot | `write(value, { offset })` on the queue | ~40 |
| 6 | G-E: proof forms | R3 refuses `out[base + i]` with a loop-invariant `base`, several fields at a runtime stride `s * i + c`, a nested record `(i * S + s) * K + c`, and two record forms in one loop (P9b, P20, P30) | accept an injective offset (`i + base`), offsets `c < s` checked by the call as `a * i + c` is, and nested affine records | enabler |
| 7 | G-F: the idle-lane scaffold | the map entry returns past the trip count before the body (`kernel-lower.ts:756-767`); TypeCV rule `gpu-sfm.md:684-699` and A1 | an iteration scaffold that clamps the index and masks the writes, as TypeCV's rule does | enabler |
| 8 | G-D: indirect counts for kernel calls | a kernel call takes no workgroup argument (host view of P1); 11 TypeCV entries dispatch indirectly | a kernel call whose trip count is read from a `Resident` (shares G-C's lowering) | enabler |
| 9 | min and max scatter | surface §65 lists them; P11 and P20 refuse `a[k] = min(a[k], e)` | the form the surface states, or a surface correction | bug |
| 10 | G-H: device-resident hashing | 0080 `Map` and `Set` apply records on the host (`kernel-lower.ts` `lowerCollections`); TSDF and cloud claim slots by compare-exchange | not recommended now; the entries are correct | 0 |
| 11 | Sort and unique on the device | `typeshade/algorithms` lists sort as "Not yet" (surface §72). Mesh weld, Hough peaks, TSDF points and the cloud grid sort on the host | a stable sort and a unique in the common module | 0 (transfers) |

Not gaps (**Fact**): 2D or 3D kernel loops (TypeCV uses only `.x`); f64 in kernel functions
(P11, P30); integer scatter atomics (P20); exported constants for the host (§64); workgroup memory
in kernel functions (R6, and the compiler itself names the entry as the remedy, P8);
kernel-to-kernel calls (Rule 8.6, `TS8099`, P13), since `fuse` composes calls on the host.

## 7. Recommendation for the refactoring proposal

The proposal changes no result unless a step says so. Each step is one pull request.

**Phase 1: no compiler change, no change of bits.**

1. Convert the barrier-free maps to kernel functions: SLIC (4), planar depth (4), CLAHE apply
   (1), mesh field and vertices (2), inverse depth (1, with struct arrays).
2. Replace `kltEigenMax` and `kltEigenMaxFinal` with one kernel `max` reduction.
3. Replace `houghVote` with an integer scatter. Replace `houghCollect` with `push`.
4. Replace mesh classify and emit with `push`.
5. Export the shader constants and import them in the host. Merge the three `chunkedPass` copies.
6. Remove the four test-only triangulation entries, or record why they stay.

Gate for each step: CPU-tier outputs equal before and after on the family's test inputs
(`npm run test:shade`); the family's WebGPU test unchanged (`npm run test:webgpu`); the plan
report shows no more readbacks (`device.statistics.reads`) and no more submissions. Steps 1 to 4
also need a hardware run on the RTX 5060 Ti while A1 is open.

**Phase 2: after compiler changes.**

1. G-B: one BA module. Gate: every BA digest and WebGPU BA test unchanged (a pure refactor).
2. G-C: replace `scan.shade.ts`, the pair scan copy and the compactions of SIFT, ORB, KLT,
   matching and MVS with `push`. Gate: row order and counts bit-identical; no new readback.
3. G-A: delete the 18 oracle siblings. Gate: WebGPU unchanged; the CPU digests change once, by
   the owner's decision (TypeCV Rule 2.2).
4. G-F with G-E: convert the loop-heavy maps (SIFT orient and describe, global descriptor, scalar
   matchers, k-NN). Gate: Phase 1 gate plus a hardware run.

**Keep as entries.** Tiled matchers, BA Schur products, PCG and LM steps, pair-model and PnP
batches (IN, DC), TSDF and cloud hashing (AT), MVS propagation and its pose cache, Hough
hysteresis, `orbPickTop`, `triangulateTracks` until G-F.

**Owner decision, not proposed here.** Floating-point reductions as kernel reductions (observation
summary, BA metrics, scoring) would give CPU = WebGPU bits but change today's numbers (§5).

## 8. Open items

- The `min`/`max` scatter mismatch (§0.1, §6 rank 9) needs a TypeShade issue. Not filed.
- The family line counts are estimates from reading; a branch that converts measures them.
- `EpipolarSession` is the only kernel-function path. Whether it stays public is the owner's
  decision.
- 0085 (owner inference, fusion facts by default) is after the pin. It makes kernel calls on
  device arrays easier, but it changes no verdict here.
