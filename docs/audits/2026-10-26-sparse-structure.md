# Sparse SfM structure audit (TypeCV)

Record: a read-only audit written on 2026-10-26 by a Claude Code sub-agent at the coordinator's request, kept as the evidence of [`../library-structure.md`](../library-structure.md). It changed no file and proposes no behaviour change. It read the code on the branch of #239 at `bac4261`, whose `src/` and `website/src/` equal `main` at `5e1c95c`. Its counts over `tests/` and `scripts/` are those of `5e1c95c`.

Revised on 2026-10-26 after the review of #241. The revision corrects the configuration below, the copies of the lab's mapper values (summary item 6, section 5) and the importers of focal-search.ts (1.1). It also corrects the samples of P0, the cycles that P1 breaks and the scripts of P2 (section 8).


## Configuration and method

- Repository `typeshade/typecv`, branch `claude/3d-depth-gradient-collapse-ab4nf7` (the branch of #239), head `bac4261f`. `src/` and `website/src/` are identical to `main` at `5e1c95c` (`git diff 5e1c95c HEAD -- src website/src` is empty). `tests/` and `scripts/` are not: #239 adds three test files and eight scripts. The counts below that they changed are restated for `5e1c95c`. Compiler pin `typeshade` `d36ac502`. Audit date 2026-10-26.
- Method: reading; static analysis with throwaway scripts (import graph from `import` statements, Tarjan cycle search, function lengths with the TypeScript compiler API, verbatim clone search: runs of ≥ 6 whitespace-normalized identical lines); `codebase-memory-mcp` 0.12.1 outlines; Node 22.22.0. The clone search undercounts near-copies with renamed identifiers.
- Not run: `npm test`, `npm run test:shade`, `npm run test:webgpu`, builds, scripts. No file in the repository was changed.
- Labels: **F** = Fact (seen in the code at the cited place, or a count with its method). **I** = Inference (a reading not proved by a run).
- Scope size (**F**): `src/reconstruction` in scope 27 files, 8,487 lines; `src/gpu` drivers 33 files, 10,746 lines; sparse kernels 16 files, 6,385 lines; `src/geometry` + `src/sfm` 15 files, 2,924 lines.

## Summary

1. **F** Two mapper state machines exist: `incrementalSteps` (mapper.ts:840-1429, 590 lines, object model, the product path) and `sceneSteps` (scene-mapper.ts:127-535, 409 lines, index protocol, the R03 path). 151 lines are verbatim copies; most of the rest is a near copy. **F** `sceneSteps` has no code for the options `preVerified`, `focalSearch`, `focalGroups`, for weak registration (`registerWeak`) or for the session parameters `finalize` and `completeRegisteredObservations` (grep), yet accepts the same `IncrementalOptions`.
2. **F** The GPU tier does not copy the mapper. `IncrementalSfmSession` re-runs `incrementalSteps` for every added keyframe once its first-map gate passes (gpu/mapper.ts:456-465, 713) and adds a second policy layer (window, anchors, archive) inside two methods of 161 and 208 lines.
3. **F** The two GPU BA routes are copies: 538 of 1,043 lines of `gpu/bundle.ts` and 896 of 1,125 lines of `bundle.shade.ts` recur verbatim in `gpu/joint-focal.ts` and `joint-focal.shade.ts`.
4. **F** Single-function monoliths: `bundleAdjustJointFocalGpu` 989 lines, `bundleAdjustGpuFixed` 846, `checkPairEstimation` 768, `expandObservationsGpu` 732, `incrementalSteps` 590.
5. **F** Three runtime import cycles: mapper.ts ↔ focal-search.ts, bundle-adjustment.ts ↔ joint-focal.ts, gpu/bundle.ts ↔ gpu/joint-focal.ts.
6. **F** The lab's six mapper values (`LAB_OPTIONS`, website/src/workers/sfm.worker.ts:125-157) are written in full in 9 places outside the tests, and in part in SfmLab.tsx (section 5). Library defaults recur across tiers (seed 17 in 8 places, LM stall 1e-9 in 4, BA iterations 20 / Huber 2 in 4). `finalBundleIterations` has two library defaults (25 and 8).
7. **F** The sparse kernels hold an 896-line BA copy and 12 `...Oracle` sibling entries. **I** Both follow from TypeShade rule TS8020 (no storage array as a helper parameter) and gap G4 (no CPU tier for barriers). **I** Host mirrors of kernel layout words are not forced at the pin: the host view can import kernel constants and enums.

## 1. Inventory

Columns: lines; responsibilities the file actually holds; main exports; importers as `src · website · scripts · tests` (graph count, **F**).

### 1.1 `src/reconstruction`

| File (lines) | Responsibilities | Main exports | Importers |
|---|---|---|---|
| mapper.ts (1448) | 9: option types; errors; `ReconstructionJob` protocol; CPU job executor; CPU observation evaluation (numerics, 269-319); defaults + validation (470-581); pair verdict and verification (647-838); mapper state machine; model copy | `incrementalSteps`, `reconstructIncremental`, `refineIncremental`, `verifyPairs`, `judgePair`, `incrementalControls`, `executeReconstructionJob`, `copyReconstruction`, `globalBundleDue`, errors | 10 · 1 · 2 · 11 |
| scene-mapper.ts (773) | `SceneRequest` protocol; second mapper `sceneSteps`; CPU driver with BA and depth-filter policy (`HostSceneDriver`, 538-733); entry | `sceneSteps`, `HostSceneDriver`, `reconstructIncrementalScene` | 0 · 0 · 0 · 6 |
| scene.ts (380) | host typed-array scene: append, flags, CSR orders, compaction, snapshot | `HostScene`, strides, flags | 1 · 0 · 0 · 1 |
| component-mapper.ts (97) | view-graph component → mapper input; CPU driver loop | `componentInput`, `reconstructComponent` | 1 · 0 · 0 · 1 |
| tracks.ts (122) | input validation; conflict-safe track union | `validateFeatures`, `buildTracks` | 8 · 0 · 1 · 1 |
| tracking.ts (138) | rotation-tracking controls, row merge, evidence | `rotationTrackingControls`, `trackingEvidence` | 6 · 0 · 0 · 0 |
| observation-selection.ts (167) | per-camera feature masks → filtered pairs | `selectCameraObservations`, `snapshotCameraMasks` | 7 · 0 · 0 · 2 |
| observation-paths.ts (64) | optical-path masks → camera masks (MVS user) | `cameraMaskFromPixelPaths` | 2 · 0 · 0 · 1 |
| focal-search.ts (684) | focal grid, components, scoring, parabola, runs of `incrementalSteps` | `focalSearchSteps`, `focalSearchControls`, `applyFocal` | 4 · 1 · 2 · 6 |
| focal-pairs.ts (150) | focal-independent pair base for the search | `pairBase`, `verificationFromBase` | 1 · 0 · 0 · 1 |
| bundle-adjustment.ts (379) | CPU fixed BA; validation; metrics; stall rule; dispatch to joint focal | `bundleAdjust`, `validateReconstruction` | 12 · 0 · 0 · 9 |
| joint-focal.ts (286) | CPU shared/grouped-focal BA (dense Schur reference) | `bundleAdjustJointFocal` | 1 · 0 · 0 · 0 |
| bundle-directions.ts (135) | bearing constraints inside BA | `appendBundleDirections`, … | 6 · 0 · 0 · 2 |
| directional-tracks.ts (123) | bearing fit, support gate, pending directions | `fitBundleDirection`, `pendingBundleDirections` | 2 · 0 · 0 · 1 |
| local-bundle.ts (208) | BA partitions (components, held poses); local subset extract/merge | `planConnectedBundles`, `extractLocalBundle` | 4 · 0 · 1 · 2 |
| sparse-schur.ts (208) | sparse symmetric Schur factor (depth BA) | `factorSparseSchur` | 1 · 0 · 0 · 1 |
| history-refinement.ts (155) | archive metrics; archive BA with injected solver; write-back | `refineHistory`, `measureHistory` | 1 · 0 · 0 · 1 |
| map-history.ts (95) | archive of evicted views/points, stable ids | `MapHistory` | 1 · 1 · 0 · 1 |
| view-graph.ts (862) | 5: descriptor store; CPU matcher; candidate selection; verification (through mapper.ts); components; build loop | `buildViewGraph`, `selectCandidates` | 6 · 3 · 3 · 8 |
| types.ts (41) | object model | `SparseReconstruction`, `ImageFeatures`, … | 33 · 0 · 1 · 12 |
| identity.ts (106) | content ids for multi-video; pair RANSAC seed | `assignViewIds`, `pairSeed` | 3 · 3 · 1 · 4 |
| posed-evidence.ts (329) | posed pair-proof graph | `preparePosedEvidence` | **0** · 0 · 0 · 1 |
| depth-tracks.ts (124) | persistent depth-track history | `DepthTrackHistory` | 1 · 0 · 0 · 1 |
| depth-record.ts (46) | depth sidecar record | `inverseDepthRecord` | 1 · 0 · 0 · 1 |
| depth-publication.ts (90) | mapper ↔ depth-tracker glue; mean error again | `publishDepth`, `recordDeferredDepth` | 3 · 0 · 0 · 0 |
| inverse-depth-tracker.ts (556) | persistent inverse-depth state machine | `InverseDepthTracker` | 8 · 0 · 0 · 3 |
| inverse-depth-bundle.ts (721) | inverse-depth BA generator + CPU driver | `inverseDepthBundleSteps`, `bundleAdjustInverseDepth` | 5 · 0 · 1 · 2 |

### 1.2 `src/gpu` drivers and device layer

| File (lines) | Responsibilities | Main exports | Importers |
|---|---|---|---|
| mapper.ts (973) | 6: float-descriptor matching adapter (163-249); GPU job driver + verification cache + seed rewrite (318-526); session window/anchor policy (`add`); archive solve policy (`update`); telemetry; depth wrapper | `IncrementalSfmSession`, `reconstructIncrementalGpu`, `refineIncrementalGpu`, `reconstructIncrementalFloatGpu` | 3 · 0 · 0 · 0 (+dist) |
| pair-estimation.ts (1068) | resident pair pipeline: undistortion/compaction, E/H model state machine, scoring, refits, recovery, H verification, multi-pair batching | `estimatePairGpu`, `estimatePairsGpu`, `undistortCorrespondencesGpu` | 4 · 0 · 0 · 1 |
| image-pair.ts (216) | pair verification facade: checks, cheirality retry, unpack | `verifyImagePairGpu`, `verifyImagePairsGpu` | 4 · 0 · 0 · 0 |
| two-view.ts (161) | E RANSAC and two-view facades; failure strings; `info[...]` unpack | `estimateEssentialRansacGpu`, `recoverySummary` | 3 · 0 · 0 · 0 |
| homography.ts (45), epipolar.ts (194), p3p.ts (121) | H facade; resident epipolar session; batch P3P | `estimateHomographyGpu`, `EpipolarSession`, `solveP3pBatchGpu` | 1 each (index) |
| pnp.ts (460) | PnP RANSAC + LM driver; **also** shared GPU bounds checks and the execution type used by 8 drivers | `estimatePosePnPGpu`, `validateGpuCamera`, `finiteGpuArray`, `GpuGeometryExecution` | 9 · 0 · 0 · 0 |
| triangulation.ts (274) | track packing (20 doubles/observation), budget chunks, readback | `triangulateTracksGpu` | 2 · 0 · 0 · 0 |
| observations.ts (243), observation-layout.ts (28) | projection/filter/depth flags; continuation variant; byte layout | `evaluateObservationsGpu`, `evaluateContinuedObservationsGpu` | 2 · 0 · 0 · 2 |
| observation-expansion.ts (932), -append (310), -consensus (264), -surface (186) | KLT measurement of missing observations; append; witness corroboration; local surface fit | `expandObservationsGpu`, `appendExpandedObservations` | product callers 0 (tests, 1 script) |
| bundle.ts (1043) | fixed-route BA: validation, packing, residency, camera products, LM/PCG steps, readback; route dispatcher; shared route policies | `bundleAdjustGpu`, `cooperativeBundleSchur`, `bundleStallMonitor` | 3 · 0 · 0 · 0 |
| joint-focal.ts (1051) | shared-focal route: copy of the fixed route + focal block + Schur cache | `bundleAdjustJointFocalGpu` | 1 · 0 · 0 · 0 |
| bundle-camera-product.ts (69), bundle-cg-schedule.ts (53), budget-plan.ts (188) | pure plans: camera-product tasks, PCG chunks, triangulation chunks and BA windows | `planCameraProduct`, `BundleCgSchedule`, `planTriangulationChunks` | 2–3 · 0 · 0 · 1–2 |
| inverse-depth.ts (150) | GPU linearization for the depth BA | `bundleAdjustInverseDepthGpu` | 3 · 0 · 0 · 0 |
| view-graph.ts (109), view-graph-descriptors.ts (181) | GPU backend of the view graph; resident descriptor LRU | `gpuViewGraphBackend` | 2 · 0 · 0 · 1 |
| global-descriptor.ts (275), -gpu (175), -candidates (102) | VLAD/BoW vocabulary and CPU descriptors; GPU assignment; k-means (scripts/tests only) | `globalDescriptorGpu`, `parseVocabulary` | 4 / 3 / 0 |
| multi-video.ts (401), multi-video-lifetime.ts (83) | ids → view graph → components → GPU mapper per component; owned lifetimes | `reconstructMultiVideoGpu` | 1 · 0 · 0 · 0 / 1 · 1 · 1 · 1 |
| device.ts (1029) | device owner: open, fp64 flavor, time budget and chunk sizing, statistics, queue, pools and working bytes, `plan`/`planSteps`/`planStepsMany`, read | `ShadeDevice`, `GpuStep`, `ShadePlanReport` | 38 · 0 · 0 · 7 |
| readback.ts (14), scan.ts (26) | read helper; flag scan plan (dense and front ends only) | `readTogether`, `planFlagScan` | 3 / 7 (no sparse user) |
| index.ts (216), internal.ts (106) | public `typecv/gpu`; internal test/lab entry | — | via `dist/` (42 + 42 test imports) |

### 1.3 `src/geometry` and `src/sfm` (oracle primitives)

| File (lines) | Responsibilities | Importers (src) |
|---|---|---|
| geometry/camera.ts (109) | `Camera` type (extends `CameraIntrinsics`, which sfm/two-view.ts declares; camera.ts:1, 4), radial projection/unprojection, validation | 33 |
| geometry/pose.ts (91), numerics.ts (156), errors.ts (7) | pose algebra; nullspace/solve/RNG/RANSAC budget; `GeometryEstimationError` | 23 / 12 / 6 |
| geometry/multiview.ts (300) | track triangulation with status codes 0/1/3/4/5, depth sensitivity | 7 |
| geometry/pnp.ts (317), p3p.ts (203), homography.ts (129), rotation.ts (210), inverse-depth.ts (248) | PnP RANSAC+LM; P3P; H RANSAC; relative rotation RANSAC; inverse-depth math | 4 / 1 / 4 / 4 / 5 |
| geometry/image-pair.ts (169) | pair verification: invertible rows, H support, E two-view, cheirality retries | 6 |
| sfm/two-view.ts (612), refinement.ts (198), triangulation.ts (32), linear.ts (143) | E estimation and two-view recovery; refinement; 2-view DLT; 3×3 SVD/eigen used by 5 geometry modules | 7 / 1 / 3 / 9 |

Further facts:

- **F** mapper.ts is a hub for small utilities: `copyReconstruction` (map-history.ts:2, history-refinement.ts:3), `verifyPairs`/`incrementalControls`/`executeReconstructionJob` (view-graph.ts:10-13), errors (gpu/multi-video.ts:12).
- **F** Layer inversion: geometry imports sfm (`CameraIntrinsics`, camera.ts:1; `sfm/linear.ts` from pose, p3p, pnp, rotation, homography) and sfm imports geometry (numerics, errors). Two linear-algebra modules (geometry/numerics.ts, sfm/linear.ts) and two triangulators (sfm/triangulation.ts, geometry/multiview.ts) exist.
- **F** 51 in-scope files are imported by deep path from outside `src` (tests, scripts, website). Website deep imports: `website/src/lib/initialization-failure.ts:1-2` (mapper.ts, focal-search.ts types), `multi-video-job.ts`/`multi-video-activity.ts` (view-graph.ts), `multi-video-input.ts` (identity.ts), `website/result-load.test.mjs:8` (map-history.ts). A file move breaks these without an alias.

## 2. Control flow of one registration

### 2.1 CPU tier (oracle): `reconstructIncremental` → `executeCpuMapper` (mapper.ts:399-434) → `incrementalSteps`

D = decides, C = computes, P = packs or unpacks. All lines in mapper.ts unless named.

1. D masks: `selectCameraObservations` (850). D focal search branch (863-899): `focalSearchSteps` runs `incrementalSteps` once per grid point (focal-search.ts:480-506, call at 493).
2. D `incrementalControls` (470-581): 19 defaults and their validation.
3. D/P `verifyPairs` (708-838): P `pixels()` (745-762); request `verifyPair` with derived controls (790-795); D `judgePair` (647-703): homography dominance 0.85/0.9 (670), parallax and positive-depth gates, seed score (699, again 775).
4. D seed pick (929-935). C `buildTracks` (951). P per-view track index (955-962).
5. Initial map: two views (969-981); request `triangulation` (1018-1052; gates: no point, `triedAt`, ≥ 2 registered observations); D `minInitial` (1207); `adjust` (1120-1196): D window (1125-1133), D subset (1134), D partitions (1136), request `bundle` per part (1155), merge (1168-1176), request `observations` stage `filter` (1098-1108).
6. Loop (1318-1357): D rank by 2D–3D count (1229-1239); D `minPnP` gate (1323); D retry rule (1333-1336, `MAX_REGISTRATION_ATTEMPTS` = 3 at 437); P + request `pnp` (1240-1257, seed `seed + order.length`); D a `GeometryEstimationError` marks the view failed (1340-1347); `addView` (1258-1270) → request `observations` stage `continuation` (1054-1079) → `triangulation`; `settle` (1271-1277): D `globalBundleDue` (598-609) selects global or local BA.
7. D recovery once per map size (1358-1366), then `registerWeak` (1284-1317): half consensus, keep only with ≥ `minPnP` new points, roll back by full map copies.
8. Final triangulation, BA, `observations` stage `final` (1368-1396); result (1397-1428). Driver: `executeReconstructionJob` (321-334); depth post-step (424-433).

### 2.2 GPU tier (lab): worker `session.add` (sfm.worker.ts:1260) → `IncrementalSfmSession.add` (gpu/mapper.ts:593-753) → `update` (765-972) → `runIncrementalGpu` (318-526) → the same `incrementalSteps`

1. D window (625-706): keep `windowSize − 2` recent views (627) and two anchors; D anchors = farthest registered pair with ≥ 12 shared points (654); P carry anchor edges with ≥ 16 shared observations (663-686); trim model; clear verification cache (705).
2. D first-map gate (713). `update` overrides (780-797): final BA 0 until `finish`, `refineSharedFocal` only at `finish`, focal search once.
3. P input snapshots (332-351). `incrementalSteps` resumes `session.model` with `session.schedule`, `finalize = false` (456-465). **F** The whole generator runs again per keyframe; pair verification hits the cache.
4. `execute` (355-454): `verifyPair` (358-384) → D digest cache (290-309, 360-366) and D seed rewrite `(seed ?? 17) + cache size` (372) → `verifyImagePairGpu` (image-pair.ts:145-168: P check/pack, C `estimatePairGpu`, D cheirality retry, P unpack); `pnp` → `estimatePosePnPGpu` (pnp.ts:90-460); `triangulation` → `triangulateTracksGpu` (P 20 doubles per observation, triangulation.ts:66-92; D budget chunks, 93-120); `observations` → `evaluate(Continued)ObservationsGpu` (P 18-double camera rows, observations.ts:102-135); `bundle` → `bundleAdjustGpu` (D route, bundle.ts:1007-1043; P scene 12C + 3P + 6C + 2M, 266-297; D residency, camera product, cooperative Schur; C LM/PCG chunks; P readback 938-1003).
5. After the run: depth update with GPU inverse-depth BA (504-512); archive `MapHistory.update` (816-818); at `finish`, D archive policy (820-846), `refineHistory` with GPU BA (847-917), write-back of cameras and focal (944-955).

### 2.3 Duplicated logic

| Fragment | Copies | Kind (**F**) |
|---|---|---|
| controls destructure + `keepTracking` | mapper.ts:903-926 · scene-mapper.ts:137-160 | verbatim, 24 lines |
| seed pick + initialization error | mapper.ts:928-953 · scene-mapper.ts:161-186 | verbatim, 26 |
| track index + `pixelOf` | mapper.ts:955-966 · scene-mapper.ts:187-198 | verbatim, 12 |
| resume adoption of landmarks | mapper.ts:987-1005 · scene-mapper.ts:222-247 | near copy |
| `triangulate`, `continueTracks`, `adjust` policies | mapper.ts:1018-1196 · scene-mapper.ts:265-400 | same policy, other data |
| initial triangulation + `minInitial` error | mapper.ts:1208-1223 · scene-mapper.ts:410-425 | verbatim, 16 |
| registration loop, failure and retry state | mapper.ts:1318-1366 · scene-mapper.ts:429-497 | near copy (scene lacks `registerWeak`) |
| result assembly | mapper.ts:1397-1428 · scene-mapper.ts:504-534 | near copy (scene lacks `verifiedPairs`) |
| driver loop + depth wrapper | mapper.ts:399-434 · scene-mapper.ts:735-773 (verbatim 19) · gpu/mapper.ts:455-525 · component-mapper.ts:93-96 | 4 loops, 3 error rules |
| rotationTracking × focalSearch refusal | mapper.ts:536-541 · gpu/mapper.ts:569-572 | same message |
| input snapshots | gpu/mapper.ts:168-191, 332-351, 551-557, 600-608 | 4 copies in one file |
| bundle stage telemetry | gpu/mapper.ts:424-452 · 866-899 | near copy |
| BA problem validation (gauge, window, connectivity) | bundle-adjustment.ts:160-215 · joint-focal.ts:49-99 · gpu/bundle.ts:173-224 · gpu/joint-focal.ts:77-128 | 4 copies, tier-specific messages |
| GPU BA host code | gpu/bundle.ts · gpu/joint-focal.ts | 538 lines verbatim (largest runs 176-224, 250-297, 393-438) |
| host reprojection-error loops | mapper.ts:269-319 · scene-mapper.ts:706-732 · history-refinement.ts:13-28 · depth-publication.ts (`publishDepth`) | 4+ copies, different populations |
| two-view failure strings | sfm/two-view.ts:352, 452, 587, 591 · gpu/two-view.ts:60, 136-139 | `isCheiralityFailure` (geometry/image-pair.ts:168) depends on the text |
| execution delta (`after − before` bytes) | bundle, joint-focal, observations, pnp, triangulation, observation-expansion, pair-estimation (×2) | 8 copies |
| camera/pose record packers | observations.ts:118-120 · triangulation.ts:75-85 · bundle.ts:271-273 · joint-focal.ts:193-195 · pnp.ts:195 · inverse-depth.ts:58-67 · scene.ts:125-136 | 7 packers of one record |
| pair key / observation key strings | ≥ 7 pair-key sites, 3 spellings (mapper.ts:550, 611; view-graph.ts:441; gpu/mapper.ts:216, 643, 665, 671) · ≥ 22 `viewId:featureId` sites in 11 files | ad hoc |

- **F** The doc claims "the same control flow as `incrementalSteps`" (scene-mapper.ts:122-126). **F** The six test files that run both mappers (tests/scene-mapper.test.ts, depth-mapper, direction-only-observations, rotation-tracking, structure-quality, bundle-partitions) pass none of the six missing options to the scene mapper; the tests that set `preVerified` or `focalSearch` call `reconstructIncremental` only (e.g. tests/rotation-tracking.test.ts:254-257, tests/direction-only-observations.test.ts:94-96). **I** No test can see the drift.

### 2.4 Constants and thresholds defined more than once (**F**)

| Value | Definitions |
|---|---|
| seed 17 | mapper.ts:476 · gpu/mapper.ts:372 · geometry/pnp.ts:207 · gpu/pnp.ts:102 · sfm/two-view.ts:306 · geometry/rotation.ts:41 · geometry/homography.ts:57 · gpu/homography.ts:34 |
| maxErrorPx 3 | mapper.ts:478 · gpu/mapper.ts:861 · focal-search.ts:546, 613 · multiview.ts:212 · gpu/triangulation.ts:40 |
| Huber = maxError / 2 | mapper.ts:1140 · scene-mapper.ts:380 · gpu/mapper.ts:861 · focal-search.ts:546, 613; inverted as `2 * job.huberPx` (scene-mapper.ts:666) |
| finalBundleIterations | 25 (mapper.ts:496) · 8 (gpu/mapper.ts:792, 850) |
| bundleIterations 8 | mapper.ts:495 · focal-search.ts:498 · history-refinement.ts:51 |
| windowSize 32 | gpu/mapper.ts:558, 625 |
| LM stall 1e-9 | bundle-adjustment.ts:46 · gpu/bundle.ts:73 · bundle.shade.ts:1089 · joint-focal.shade.ts:1096 (mapper's own 1e-4 at mapper.ts:586) |
| BA iterations 20, Huber 2 | bundle-adjustment.ts:160-161 · joint-focal.ts:49-50 · gpu/bundle.ts:173-174 · gpu/joint-focal.ts:77-78; CG tolerance 1e-6 at gpu/bundle.ts:176, gpu/joint-focal.ts:80 |
| PnP 1000 / 3 / 0.999 / 20 | geometry/pnp.ts:204-209 · gpu/pnp.ts:99-104 |
| triangulation 3 / 0.5 / 32 | multiview.ts:212-214 · gpu/triangulation.ts:40-42 |
| H iterations min(iterations, 500) | mapper.ts:795 · geometry/image-pair.ts:43 |
| triangulation status 0/1/3/4/5 | multiview.ts:100-101, 230-292 · triangulation.shade.ts:358-456 · mapper.ts (`status === 5`) · scene-mapper.ts (`-5`) |
| indirect-count offsets 0/3/6/9 | pnp.ts:84-87 vs pnp.shade.ts:269-272 · pair-estimation.ts:132-135 vs pair-models.shade.ts |
| `PROGRESS_INTERVAL` 4 | pnp.ts:80 · pair-estimation.ts:96, 124 |
| scene strides 12 / 6 / 3 / 2 | scene.ts:8-14 · bundle.ts:266-297 · bundle.shade.ts:50-79, 112-121 · joint-focal.shade.ts ditto |

## 3. Separation problems

### 3.1 Functions over about 80 lines (**F**, TypeScript AST scan; nested closures included in the outer count)

| Function (lines) | Distinct concerns inside |
|---|---|
| `bundleAdjustJointFocalGpu` gpu/joint-focal.ts:63-1051 (989) | validation, packing, residency plan, camera products, Schur cache, LM/PCG step generators, readback, unpack |
| `bundleAdjustGpuFixed` gpu/bundle.ts:159-1004 (846) | same set; one `enqueue` closure of 670 lines with an inner 410-line closure |
| `checkPairEstimation` gpu/pair-estimation.ts:137-904 (768) | request checks; model search state machine (`consensus` 222, its `step` 124); recovery; verification; result `finish` |
| `expandObservationsGpu` gpu/observation-expansion.ts:201-932 (732) | seeding, KLT dispatch, `measure` (207), ambiguity checks, lineage |
| `incrementalSteps` mapper.ts:840-1429 (590) | masks, focal branch, controls, verification, seed, track index, resume, 11 nested generators (`triangulate`, `continueTracks`, `applyEvaluation`, `filter`, `deferSensitive`, `adjust` 77, `rank`, `estimatePose`, `addView`, `settle`, `registerWeak`), loop, recovery, final pass, result |
| `inverseDepthBundleSteps` inverse-depth-bundle.ts:190-599 (410) | linearization jobs, Schur, LM, covariance |
| `sceneSteps` scene-mapper.ts:127-535 (409) | as `incrementalSteps` minus six features |
| `estimatePosePnPGpu` gpu/pnp.ts:90-460 (371) | validation, packing, batch plan, indirect counts, LM, readback |
| `focalSearchSteps` focal-search.ts:377-679 (303) | verification cache, failure memory, runner, scoring, component loop, directional branch, parabola refinement written twice (597-606, 645-657) |
| `bundleAdjustJointFocal` joint-focal.ts:44-286 (243); `bundleAdjustFixedIntrinsics` bundle-adjustment.ts:155-366 (212) | validation, normal equations, LM, metrics |
| `recoverTwoViewGeometry` sfm/two-view.ts:377-612 (236); `buildViewGraphWithBackend` view-graph.ts:630-862 (233) | recovery+refinement+classification; candidate rounds, verification, components |
| `runIncrementalGpu` gpu/mapper.ts:318-526 (209); `IncrementalSfmSession.update` 765-972 (208); `.add` 593-753 (161) | driver + cache + telemetry + depth; archive policy + telemetry + state; window policy + masks + state |
| `evaluateSceneObservationsGpu` observations.ts:57-243 (187); `HostSceneDriver.execute` scene-mapper.ts:543-704 (162) | validation, packing, plan, unpack; 11 request kinds incl. BA policy |
| `InverseDepthTracker.prepare` (156); `verifyPairs` mapper.ts:708-838 (131); `triangulateChunk` (128) and `triangulateTracksGpu` (111); `incrementalControls` mapper.ts:470-581 (112) | — |
| also ≥ 80: `appendExpandedObservations` 247, `corroborateObservations` 193, `reconstructMultiVideo` 167, `createObservationSurfacePredictor` 155, `createViewGraphDescriptorCache` 143, `preparePosedEvidence` 125, `estimatePosePnP` 122, `factorSparseSchur` 120, `p3p` 116, `fitPose` 105, `refineHistory` 103, `linearizeBatches` 101, `device.open` 96, `planSteps` 95, `triangulateTrackDetailed` 94, `planConnectedBundles` 92, `estimateRelativeRotation` 90, `gpuViewGraphBackend` 89, `estimateHomography` 87, `reconstructIncrementalFloatGpu` 87, `buildTracks` 85, `verifyImagePairInternal` 83, `DepthTrackHistory.update` 82, `planStepsMany` 81 | — |

### 3.2 Files that mix policy and numerics (**F** unless marked)

- mapper.ts: policy (seed, order, retries, BA schedule) with numerics (`evaluateCpuObservations`, 269-319), validation, protocol and utilities.
- scene-mapper.ts: the driver holds policy. `bundle` (642-703) extracts the local subset, plans partitions and fits bearings; `filter` (598-640) assesses depth on a host snapshot. **I** A `GpuScene` driver must re-implement this policy, or the policy must move to the mapper first.
- gpu/mapper.ts: matching adapter, GPU driver, cache, window policy, archive policy and telemetry in one file.
- gpu/bundle.ts: device route heuristics (`cooperativeBundleSchur` 76-92, `cameraAssemblyWorkgroup` 97-110) beside the solver; gpu/joint-focal.ts imports them back (cycle).

### 3.3 State in closures and module variables

- **F** `incrementalSteps` keeps the map in closure variables (`views`, `registered`, `landmarks`, `triedAt`, `failure`, `recoveredAt`, `last`, `depthDeferred`, `order`, reports). Eleven nested generators mutate them. `registerWeak` rolls back by copying whole maps (1295-1316).
- **F** The `schedule` argument is mutated in place (comment at 845; writes at 1192-1195, 1198-1202). `IncrementalSfmSession` relies on it (`SessionState.schedule`, gpu/mapper.ts:311-316).
- **F** `IncrementalSfmSession` has 12 private fields; `add` rewrites `this.options` (617-627, 731-738).
- **F** Module-level state is small: view-graph.ts:299 (`WeakMap`), directional-tracks.ts:8 (constant computed at load).

### 3.4 Parameters threaded through layers

- **F** `finalize`, `minimumFocalViews`, `completeRegisteredObservations` are positional through `update` → `runIncrementalGpu` (323-325) → `incrementalSteps` (848-852) → `focalSearchSteps` (384) and `adjust` (1369). Call with bare literals: `incrementalSteps(images, inputPairs, controls, model, undefined, true, 2, true)` (mapper.ts:361).
- **F** Raw `options` and validated `controls` are both read inside `incrementalSteps` (`options.initialPair` 929-935, `options.focalGroups` 1143, `options.refineSharedFocal` 1369). `refineSharedFocal`, `focalGroups` and `onReconstruction` are not validated by `incrementalControls`.

### 3.5 Options objects that mix concerns (**F**)

- `IncrementalOptions` (mapper.ts:68-139): 28 fields in 10 concerns: a stateful tracker object (`depthTracking`), bearings, masks, two-view refinement, rotation tracking, a UI callback (`onReconstruction`), RANSAC, thresholds (two inherited from `DepthSensitivityOptions`), BA schedule (7 fields), focal calibration (3), seeding (`initialPair`) and a verdict cache (`preVerified`).
- `GpuIncrementalOptions` (gpu/mapper.ts:63-84) adds session (`windowSize`, `historyBundleIterations`), telemetry (`onStage`, `onProgress`), solver route (`schurCache`) and device scheduling (`pnpBatchSize`).

### 3.6 Import cycles (**F**, Tarjan over the graph; edges checked by hand)

| Cycle | Runtime edges | Type-only edges |
|---|---|---|
| mapper.ts ↔ focal-search.ts | both ways (mapper.ts:50-57; focal-search.ts:1-12) | — |
| bundle-adjustment.ts ↔ joint-focal.ts (+ bundle-directions.ts) | bundle-adjustment.ts:1; joint-focal.ts:1-8 | bundle-directions.ts:4 |
| gpu/bundle.ts ↔ gpu/joint-focal.ts | bundle.ts:18; joint-focal.ts:55-60 (`bundleStallMonitor`, `cooperativeBundleSchur`) | — |
| mapper.ts ↔ depth-publication.ts; sfm/two-view.ts ↔ sfm/refinement.ts | one way each | depth-publication.ts:1; refinement.ts |

## 4. The generator protocol (`ReconstructionJob` and its drivers)

Good (**F**): five request kinds (mapper.ts:223-245) carry all numerical work. Drivers swap without a change to the mapper (CPU sync, GPU async). Tests drive the generator directly. `refineHistory` takes its solver as an argument (history-refinement.ts:34-41), a clean seam.

Leaks across the boundary:

1. **F** Requests carry the object model: `bundle` takes a whole `SparseReconstruction`; `observations` takes a model (234-244). **F** Each GPU request packs and uploads the map again (docs/gpu-resident-scene.md, Problem).
2. **F** Request types are CPU signatures: `args: Parameters<typeof verifyImagePair>` (224). The GPU driver must accept CPU option types and injects its own (`pnpBatchSize`, `schurCache`; gpu/mapper.ts:389, 427).
3. **F** Results are untyped: `request<T>` casts `(yield job) as T` (256-260). The GPU driver reshapes triangulation results by hand (gpu/mapper.ts:406-409).
4. **F** Result shapes differ by driver: a GPU continuation returns one mask entry per point (observations.ts:74, 216-220); the CPU returns full masks (mapper.ts:279-313). The mapper reads only the last entry (1077).
5. **F** The GPU session changes requests: digest cache and seed rewrite (gpu/mapper.ts:290-309, 360-375). **I** The CPU oracle cannot replay the request stream of a session; a session result is not comparable request for request.
6. **F** The error rule lives in each driver: the GPU wraps all but `GeometryEstimationError` (gpu/mapper.ts:476-481); the CPU passes every error back (mapper.ts:412-417); `reconstructComponent` passes none (component-mapper.ts:95, no `machine.throw`). **I** On the CPU component path a PnP failure ends the run instead of marking the view failed.
7. **F** Side channels: `onReconstruction` runs inside the generator (`publish`, 1007); progress labels are request kinds (gpu/mapper.ts:469-471), shown verbatim by the lab (sfm.worker.ts:764).
8. **F** The depth-tracker post-step is outside the protocol and is written three times (mapper.ts:424-433, scene-mapper.ts:763-772, gpu/mapper.ts:504-524).
9. **F** Hints without contract: `stage` selects a GPU kernel path (gpu/mapper.ts:414-415); `trackIds` is "for diagnostics; drivers ignore it" (230).
10. **F** `SceneRequest` (scene-mapper.ts:67-113) fixes 1 (indices and small typed arrays; the model crosses only in `load` and `snapshot`) and 7 for snapshots (a `snapshot` request). It keeps 3 (untyped `request<T>`, scene-mapper.ts:115-117) and 6, reuses the CPU-typed `PairVerificationJob`, and moves policy into the driver (3.2).

## 5. Configuration of the sparse pipeline

| Tunable | Library default | Lab (`LAB_OPTIONS`, sfm.worker.ts:125-157) | Other definitions and copies (**F**) |
|---|---|---|---|
| `ransacIterations` | 750 (mapper.ts:477) | 500 | workers/multi-video.ts:60 · SfmLab.tsx:687 · scripts/measure-focal.mjs:88 · measure-multi-video.mjs:94 · benchmark-sfm-pyramid-e2e.mjs:156, 402 · benchmark-rotation-tracking.mjs:294 · benchmark-sift-clahe.mjs:123 · measure-video-sfm-performance.mjs:157 · benchmark-observation-expansion.mjs:216 (512) |
| `maxErrorPx` | 3 (478) | 3 | same copies; 2.4 |
| `minInitialAngleDeg` / `minTriangulationAngleDeg` | 1 / 0.5 (491-492) | 1 / 0.5 | same copies (SfmLab.tsx omits them) |
| `bundleIterations` | 8 (495) | 4 | same copies; focal-search.ts:498, history-refinement.ts:51 |
| `finalBundleIterations` | 25 (496); 8 in a session (gpu/mapper.ts:792, 850) | 8 | same copies |
| `refineSharedFocal` | false | true; photo false (`PHOTO_PROFILE`, 163-165) | multi-video.ts:66 · SfmLab.tsx:691 · measure-multi-video.mjs:100 |
| `windowSize` | 32 (gpu/mapper.ts:558, 625) | 32 | SfmLab.tsx:686 (0) |
| `seed` | 17 (476) | default | scripts use 42 or 7; 2.4 |
| `minPnPInliers`, `minInitialPoints` | 12, 12 (493-494) | default | weak minimum `max(6, ⌈minPnP/2⌉)` (1285); anchor ≥ 12 (gpu/mapper.ts:654) |
| BA schedule | local size 6, subset true, ratios 1.1, frequency 0, stall 1e-4 (497-586) | default | solver stall 1e-9 (2.4) |
| pair controls | derived (790-795), dominance 0.85/0.9 (670), score cap 10 (699, 775) | — | — |
| focal search | `FOCAL_SEARCH_DEFAULTS` (focal-search.ts:128-134) | `focalSearch` per run | — |
| archive solve | `max(final, 32)` with several focal seeds (gpu/mapper.ts:855-860) | — | — |
| GPU scheduling | floors 512 / 256 / 16 384 (pnp.ts:82, pair-estimation.ts:130, triangulation.ts:31); batch 512 (pnp.ts:106, pair-estimation.ts:150); `PNP_BATCH` 64 (multi-video.ts:186); refit 128/512 (pair-estimation.ts:102-119) | device | — |

- **F** The lab reports `LAB_OPTIONS` in every run record (sfm.worker.ts:124). A move of these values must keep the record's `options` field identical.
- **F** The six lab mapper values (500, 3, 1, 0.5, 4, 8) are written in full in 9 places outside the tests: `LAB_OPTIONS`, `MAPPER` (workers/multi-video.ts:59-67) and 7 copies in 6 scripts (measure-focal.mjs:87-94, measure-multi-video.mjs:94, benchmark-sfm-pyramid-e2e.mjs:156 and 402, benchmark-rotation-tracking.mjs:294, benchmark-sift-clahe.mjs:123, measure-video-sfm-performance.mjs:157). SfmLab.tsx:684-697 holds four of them; tests/focal-search.test.ts:23-31 holds all six. `scripts/measure-focal-sweep.mjs`, which #239 adds, reuses `LAB` of measure-focal.mjs.

## 6. Kernel organization

| Kernel (lines) | Entries | `...Oracle` siblings | Imports | Drivers |
|---|---|---|---|---|
| bundle.shade.ts (1125) | 29 | 2 (347, 1063) | numerics | gpu/bundle.ts |
| joint-focal.shade.ts (1383) | 39 | 2 (386, 1070) | numerics | gpu/joint-focal.ts |
| pnp.shade.ts (455) | 18 | 2 (175, 346) | numerics | pnp.ts |
| recovery.shade.ts (675) / pair-models (548) / consensus (222) / pair-preparation (179) | 16 / 8 / 7 / 8 | 1 / 0 / 1 / 1 | numerics, projective | pair-estimation.ts |
| triangulation.shade.ts (464) | 7 | 1 (125) | numerics | triangulation.ts, observations.ts |
| observation-evaluation.shade.ts (138) | 4 | 1 (131) | numerics | observations.ts |
| p3p (361), inverse-depth (81), global-descriptor (51), scan (72) | 1 / 1 / 1 / 3 | 0 / 0 / 0 / 1 | — | p3p.ts + pnp.ts / inverse-depth.ts / global-descriptor-gpu.ts / scan.ts |
| numerics (281), projective (217), epipolar (133) | helpers 11 / 5 / 8 | — | — | 9 kernels / 2 kernels / epipolar.ts |

Sharing and copies (**F**):

- Value-only helpers are shared by import (`project`, `transform`, `solve6`, `movePose`, … from numerics.shade.ts).
- Helpers that read bindings or workgroup tiles are copied: BA kernels (896 lines; largest runs bundle.shade.ts:398-489, 646-710, 1038-1100); `PairControl` (consensus.shade.ts:4-19 = pair-models.shade.ts:8-23); 64-lane partial sums (observation-evaluation.shade.ts:111-128 = triangulation.shade.ts:105-122); 44-wide normal reduction (pnp.shade.ts:335-344 = recovery.shade.ts:332-341); scan block (pair-preparation.shade.ts:96-108 = scan.shade.ts:23-35). Tree-reduction loops occur in 9 sparse kernel files.
- One kernel lives in `src/gpu/kernels/` (float-matching.shade.ts); all others sit in `src/gpu/`.

Limitations that shaped the structure (for the owner, who can change the compiler):

- **L1. TS8020, no storage array as a helper parameter.** Rule (**F**): "A parameter or a result that holds [an unsized array] … is refused (TS8020) … read the binding by its name instead" (node_modules/typeshade/docs/use-typeshade-surface.md:2034-2038). Effect (**I**): a helper that reads `scene`, `trial`, `blocks` or `state` belongs to the file that declares them; the fixed and shared-focal BA declare the same 13 storage bindings with different control structs, so the helpers are copied (896 lines). Workaround today: copy. Options: (a) compiler: storage-array reference parameters (the pin already passes storage elements through `unrestricted_pointer_parameters`, surface §70); (b) no compiler change: one bindings module that both kernels import (imports bring bindings along, AUTHORING.md:688-692) with one control struct; this changes the fixed route's uniform layout.
- **L2. Gap G4, no CPU tier for barriers.** Rule (**F**): the CPU tier runs one invocation per call; TypeCV keeps a sequential sibling per tiled entry (docs/gpu-throughput.md:853-859, draft issue G4). Effect (**F**): 12 `...Oracle` entries in sparse kernels and 29 `device.backend` branches in pnp.ts, pair-estimation.ts, bundle.ts, joint-focal.ts and observations.ts (some are route policy). **I** For Rule 1.2 the CPU tier proves the sibling, not the tiled entry.
- **L3. Host mirrors of kernel layout words.** **F** The pinned host view exposes exported constants and enums of a shader module (use-typeshade-surface.md:7236-7250). **F** No TypeCV host file imports a constant from a `.shade.ts`. **I** The mirrors (count offsets, BA state words such as `values[6]`, `values[12]`, `values[base + 14]` in gpu/bundle.ts:138, 143, 940; recovery words `info[44..70]` in gpu/two-view.ts:145-160; status codes) are a convention, not a limit. The esbuild build must be checked for this import.
- **L4. WebGPU limits, not TypeShade.** **F** 8 storage buffers per stage and 16 KiB of workgroup memory are the guaranteed minimum (docs/gpu-throughput.md:744-747). **I** This drives the packing into a few f64 arrays with hand offsets (bundle.shade.ts:50-79), mirrored by host packers.
- **Not limits (F):** kernels import kernels (numerics.shade.ts has 9 importers); a file holds up to 39 entries; `extends` exists at the pin (surface:196, 2662), so the duplicated control structs are not forced. **I** A shared tree reduction over a workgroup tile may compile as an `@inout` module-variable argument (surface §70); not verified.

## 7. Target structure (fits docs/engine-v2.md)

Principle: one mapper policy (a state machine over indices), one request protocol, two drivers of it (host f64 = oracle; resident GPU scene = product), one device layer. Area entry points (`typecv/sfm`, `typecv/geometry`, …) come with the engine; old paths stay as aliases.

| Module (proposed path) | Responsibility | Contract |
|---|---|---|
| `sfm/model` | object model, copy, validation, `observationKey`, `pairKey`, content ids | pure; no tier |
| `sfm/config` | every default once; `resolveMapperControls`; session and focal controls; presets (lab) | pure; throws `RangeError`; returns frozen controls |
| `sfm/tracks` | track union, rotation evidence, observation masks and paths | pure |
| `sfm/pairs` | pair verdict, seed score, pair seeds, verification order, pre-verified index | pure policy + requests `verifyPair` |
| `sfm/mapper` | initialization, ranking, registration gates and retries, weak registration, triangulation/continuation rules, filter rules, BA schedule and window | generator over `SceneRequest`; no numerics, no callbacks, no packing beyond small typed arrays |
| `sfm/protocol` | request and result types (typed map), errors, `drive` (sync/async) with one error rule, depth post-step | the only place of the error rule |
| `sfm/scene` | `HostScene`, layout constants shared with kernels, `HostSceneDriver` (oracle), `GpuSceneDriver` (R03) | executes one request; no policy |
| `sfm/bundle` | CPU solvers, bearing constraints, BA problem plan and shared validation | pure numerics; tier-specific messages kept |
| `sfm/focal` | focal search, pair base, history seeds | generator over the protocol |
| `sfm/depth` | inverse-depth tracker, depth BA, publication, record, sparse Schur | own generator (`inverseDepthBundleSteps`) |
| `sfm/session` | window eviction, anchors, carried edges, archive (`MapHistory`), archive solve plan | pure functions + one orchestrator |
| `sfm/multiview` | view graph, components, global descriptors (host part) | pure + backend interface |
| `geometry` (CPU oracle) | camera (owns `CameraIntrinsics`), pose, linear algebra (sfm/linear + numerics), two-view, homography, P3P/PnP, triangulation, rotation, inverse depth, image pair, failure-reason table, status enums | pure f64 |
| `geometry/gpu` | GPU operators (pair, two-view, H, P3P, PnP, triangulation, observations); shared bounds checks, execution delta, report merge, record packers, tier selection | validate → pack → plan → read; no mapper policy |
| `device` | `ShadeDevice`, plans, budget plans, readback, scan | kept as is (engine-v2: "Kept. They are the device layer.") |

Mapping from today's files:

| Today | Target |
|---|---|
| mapper.ts: 68-245 types/errors/protocol · 269-334 CPU executor · 470-609 controls/schedule · 610-838 pair verdict/verification · 840-1429 mapper · 1432-1448 copy | `sfm/protocol` · `sfm/scene` legacy driver · `sfm/config` + `sfm/mapper` · `sfm/pairs` · `sfm/mapper` · `sfm/model` |
| scene-mapper.ts: 43-120 · 127-535 · 538-733 · 735-773 | `sfm/protocol` · merged into `sfm/mapper` · `sfm/scene` (policy moved to `sfm/mapper`) · `sfm/protocol.drive` |
| scene.ts | `sfm/scene` |
| tracks.ts, tracking.ts, observation-selection.ts, observation-paths.ts | `sfm/tracks` (`validateFeatures` → `sfm/model`) |
| focal-search.ts, focal-pairs.ts | `sfm/focal` (`cameraWidth`, `cameraAtRatio` → `geometry/camera`) |
| bundle-adjustment.ts, joint-focal.ts, bundle-directions.ts, directional-tracks.ts, local-bundle.ts | `sfm/bundle` |
| depth-*.ts, inverse-depth-*.ts, sparse-schur.ts | `sfm/depth` |
| map-history.ts, history-refinement.ts, gpu/mapper.ts `add`/`update` policy | `sfm/session` |
| view-graph.ts, component-mapper.ts, identity.ts, gpu/global-descriptor*.ts, gpu/view-graph*.ts, gpu/multi-video*.ts | `sfm/multiview` (ids → `sfm/model`) |
| gpu/mapper.ts `execute` (355-454), `IncrementalSfmSession` | legacy GPU driver, replaced by `GpuSceneDriver` + `sfm/session` (engine-v2) |
| gpu/pair-estimation.ts, image-pair.ts, two-view.ts, homography.ts, p3p.ts, pnp.ts, triangulation.ts, observations.ts, observation-layout.ts | `geometry/gpu` (pnp.ts helpers → `geometry/gpu/bounds`, `execution`) |
| gpu/bundle.ts, joint-focal.ts, bundle-camera-product.ts, bundle-cg-schedule.ts | `sfm/bundle/gpu` (shared host half once) |
| geometry/*, sfm/two-view.ts, refinement.ts, triangulation.ts, linear.ts | `geometry` |
| gpu/device.ts, budget-plan.ts, readback.ts, scan.ts | `device` |
| posed-evidence.ts, gpu/observation-expansion/-append/-consensus/-surface | owner decision (§9, item 8) |

## 8. Phased plan (structure only; one pull request per phase)

Common gate for every phase (**I**, proposed): `npm run typecheck`, `npm run format:check`, `npm test`, `npm run build`, `npm run test:shade`, `npm run test:webgpu`, `npm run build:website`; same test count; public declarations of `typecv` and `typecv/gpu` unchanged (diff of `dist/**/*.d.ts`); the Phase 0 digests unchanged. Each phase also names its own gate below.

| Phase (days) | Change | Files | Own gate |
|---|---|---|---|
| P0 (1–2) | Characterization tests only. Canonical digest (Float64 bit patterns) of the full output of `reconstructIncremental`, `reconstructIncrementalScene`, `refineIncremental`, `reconstructComponent` and the focal search on tests/scene.ts scenes and the four lab samples that scripts/measure-focal.mjs reads (`tests/fixtures/lab-images/<sample>/colmap-sift.bin`); same for `IncrementalSfmSession` (add per frame, finish) on the CPU device and on SwiftShader; digest of emitted WGSL/GLSL per sparse kernel module; per-request trace of each driver (kind, sizes) | new tests only | digests recorded; SwiftShader digests compared baseline vs candidate on the same runner |
| P1 (1–2) | Pure moves with alias re-exports: break the two BA cycles (`validateReconstruction`, `cloneBundleReconstruction`, `bundleMetrics` and `bundleStallTolerance`, which joint-focal.ts:1-8 imports, out of bundle-adjustment.ts; `bundleStallMonitor`/`cooperativeBundleSchur` out of gpu/bundle.ts); `judgePair`/`incrementalControls`/errors/`copyReconstruction` out of mapper.ts; pnp.ts helpers to their own module; `CameraIntrinsics` to geometry/camera.ts; `sfm/linear.ts` beside geometry. The mapper.ts ↔ focal-search.ts cycle stays: `incrementalSteps` and `focalSearchSteps` call each other (mapper.ts:868, focal-search.ts:493), so only an injected generator breaks it (phase 4 of the proposal) | mapper.ts, focal-search.ts, bundle-adjustment.ts, joint-focal.ts, gpu/bundle.ts, gpu/joint-focal.ts, gpu/pnp.ts and its 8 importers, geometry/camera.ts, sfm/* | the two BA cycles gone, the mapper–focal cycle the only one left; WGSL digests unchanged; deep imports from website/scripts/tests still resolve |
| P2 (2–3) | One definition for each value: `observationKey`/`pairKey` (same strings); triangulation status enum (host side); two-view failure table; operator defaults shared by CPU and GPU operators; mapper defaults table with the per-entry differences named (25 and 8 kept); host layout constants once per kernel; lab preset in one module read by the worker, the multi-video worker, SfmLab.tsx and the scripts | the sites in 2.4 and 5; website/src/workers/sfm.worker.ts, workers/multi-video.ts, components/SfmLab.tsx; scripts/measure-focal.mjs, measure-multi-video.mjs, benchmark-sfm-pyramid-e2e.mjs, benchmark-rotation-tracking.mjs, benchmark-sift-clahe.mjs, measure-video-sfm-performance.mjs | run-record `options` identical; `scripts/measure-focal.mjs` stdout identical |
| P3 (3–4) | Extract the shared pure parts of the two mappers (masks, seed pick, initialization error, track index, resume adoption, ranking, failure and retry state, result assembly) into `sfm/mapper/*`; both generators call them | mapper.ts, scene-mapper.ts | P0 digests of both mappers; scene parity tests |
| P4 (2–3) | Driver plumbing: one sync and one async `drive` with an explicit error rule (the component driver keeps its current rule as an explicit option until the owner decides); one depth post-step; one bundle-stage telemetry builder; execution-delta and report-merge helpers; record packers with today's layouts | mapper.ts, scene-mapper.ts, component-mapper.ts, gpu/mapper.ts, 8 GPU drivers | request traces identical; plan reports identical (kernel names, dispatches, submissions) |
| P5 (3–4) | Split long GPU host functions without changing the device steps: request checks, step generators, unpack (`checkPairEstimation`, `estimatePosePnPGpu`, `triangulateTracksGpu`, `evaluateSceneObservationsGpu`); one host half for both BA routes where it is verbatim (validation with tier messages kept, packing, readback) | pair-estimation.ts, pnp.ts, triangulation.ts, observations.ts, bundle.ts, joint-focal.ts | step-name sequence and plan reports identical; WGSL unchanged; GPU digests |
| P6 (2–3) | Split `IncrementalSfmSession.add`/`update` into pure policies (`evictWindow`, `chooseAnchors`, `carryAnchorEdges`, `archiveSolvePlan`) with unit tests | gpu/mapper.ts, new `sfm/session/*` | session digests (CPU device, SwiftShader); tests/typeshade/focal-search.test.mjs, streaming.test.mjs |
| P7 (3–5, owner go-ahead) | Boundary: move BA window/subset/partition and depth-filter policy from `HostSceneDriver` into the mapper; bring `sceneSteps` to parity (the six missing features and `verifiedPairs`) from P3 parts; typed request→result map | scene-mapper.ts, mapper.ts | new parity tests: `sceneSteps` + `HostSceneDriver` equal `incrementalSteps` + CPU driver bit for bit for every P0 option set; `incrementalSteps` digest unchanged |
| After R03 | Kernel work: shared BA bindings module or compiler change (L1); shared reductions; pair-preparation on scan.shade.ts; layout constants exported from kernels (L3); retire the object-model protocol and `runIncrementalGpu`; area entry points | kernels, gpu/mapper.ts, package.json | reviewed WGSL diff; CPU tier and SwiftShader bit-identical; native records |

Do not do before the engine-v2 scene work lands (**I**, from docs/engine-v2.md and docs/gpu-resident-scene.md):

- Do not edit kernel files (`*.shade.ts`): engine-v2 rewrites them "to bind the resident scene and f32 arrays", and a kernel edit can change GPU bits.
- Do not restructure `IncrementalSfmSession` or `runIncrementalGpu` beyond P6 extraction: engine-v2 replaces them.
- Do not extend `ReconstructionJob`; new requests go to `SceneRequest`.
- Do not add package entry points or remove deep paths: engine-v2 introduces entry points with the engine and keeps aliases.
- Do not fix a behaviour difference inside a refactor pull request (component driver error rule, default 25 vs 8, session seed rewrite, GPU focal groups refused at gpu/bundle.ts:1037-1041). Each needs its own record.
- Do not change a lab preset value while it is centralized.

## 9. Risks and open questions for the owner

1. **Which mapper survives?** Converge on `sceneSteps` (index protocol, the R03 path) and retire `incrementalSteps`, or keep both until R03? P7 needs this decision.
2. **Component driver error rule** (component-mapper.ts:95): intended, or a defect? Fixing it changes behaviour, so it is a separate pull request.
3. **Two defaults for one option**: `finalBundleIterations` 25 (library) and 8 (session); stall 1e-4 (mapper) and 1e-9 (solver). Keep as named per-entry defaults, or unify (a behaviour change)?
4. **Where the lab preset lives**: in `src` (shared by the worker and the scripts) or in `website/`. The run-record format depends on it.
5. **Compiler changes**: storage-array helper parameters (L1, removes about 900 kernel lines); a CPU tier that runs barriers (L2, removes 12 siblings and the tier branches); confirmation that host code may import kernel constants and enums in this build (L3).
6. **Proof on GPU tiers**: bit identity across machines is not established for SwiftShader or native GPUs (docs/engine-v2.md: 8 of 8 vs 5 of 8 views). Gates must compare baseline and candidate on the same runner; native GPUs are not in CI.
7. **Deep-path consumers**: 51 in-scope files have importers outside `src`. Alias shims are needed until the next major version; the removal date is open.
8. **Code without a product caller**: posed-evidence.ts (329 lines) and the observation-expansion family (1,692 lines, exported from `typecv/gpu`). Keep, move to `experimental/`, or delete?
