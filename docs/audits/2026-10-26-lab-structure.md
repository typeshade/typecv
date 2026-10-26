# Audit: structure of the lab's reconstruction orchestration

Record: a read-only audit written on 2026-10-26 by a Claude Code sub-agent at the coordinator's request, kept as the evidence of [`../library-structure.md`](../library-structure.md). It changed no file and proposes no behaviour change. The numbers are those of `main` at `5e1c95c`.


Read-only structural audit. No repository file was changed. It proposes no behaviour change.

**Configuration.** Repository `/home/user/typecv`, branch `claude/3d-depth-gradient-collapse-ab4nf7`.
The audit started at `d944125`. HEAD moved to `bac4261` during the audit: another actor committed
`350a6fe` and `bac4261` (scripts/record_analysis, docs/chimney-rock-58532c69.md).
`git diff d944125 bac4261 -- website src package.json` is 0 lines. The audited files are the same at both commits.
Tools: Node 22.22.0, `codebase-memory-mcp` (index, outlines, `trace_path`), grep and sed. Date: 2026-10-26.

**Checks that ran (observed results).**
- `npm test` (tests/*.test.ts): 445 tests, 445 pass, 0 fail, 115 s.
- The 36 website test files that need no browser (`node --test` on the `test:website` list without the
  `*-browser`, `site` and `result-page` files): 249 tests, 249 pass, 0 fail, 102 s.
- `node --experimental-strip-types -e "import('website/src/workers/sfm.worker.ts')"` fails with
  `ERR_MODULE_NOT_FOUND ... website/src/lib/video-sampling`. The same import of `lib/multi-video-job.ts` and
  `lib/dense-run.ts` succeeds.

**Checks that did not run.** The 10 browser files of `npm run test:website`: this container has no Chromium,
no Playwright cache and `TYPESHADE_CHROMIUM` is unset. Also not run: `test:shade`, `test:webgpu`, `format:check`
and `typecheck`. CI job `webgpu` (`.github/workflows/ci.yml:30-46`) is the place where `test:website` runs with Chromium.

Labels: **F** = fact, read in the code or a document, or observed. **I** = inference.

## 1. Inventory

F, unless a cell says (I). Lines from `wc -l`. "Importers" are production importers first, then tests and scripts.

| File | Lines | Responsibilities it actually holds | Exported API | Importers |
|---|---|---|---|---|
| `website/src/workers/sfm.worker.ts` | 1635 | Message intake and queue (1571-1635); device open, reuse and loss (567-809, 1490-1535); options assembly (689-714); frame pipeline (811-1109); guided matching (1184-1244); registration and descriptor retention (1247-1301); KLT driver (431-565); snapshots, scene colours, match lines (304-395, 1117-1152); stage ledger (327-374); outcome and failure (1307-1454); dense hand-off (1455-1487, 1538-1569); forwards `multi*` messages | `LAB_OPTIONS`, `PHOTO_PROFILE`, `Profile`, `trimStages`. No module imports them. | `SfmLab.tsx` (`new Worker`, 2453, 3117). Tests that bundle it with esbuild: `dense-gpu-mode-worker`, `sfm-worker-loss-browser`, `dense-quality-browser`, `gpu-preparation-browser`. `working-budget.test.mjs` regex-matches its source. |
| `website/src/workers/multi-video.ts` | 268 | Multi-video wiring: loads `/runtime/internal/gpu.js` and the vocabulary, opens a sparse device and a dense device, has two loss handlers, `MAPPER` | `MultiRequest`, `isMultiRequest`, `handleMultiVideo`, `cancelMultiDense`, `closeMultiVideo` | `sfm.worker.ts`; `working-budget.test` reads its source |
| `website/src/workers/decode.worker.ts` | 54 | A thin decode worker over a typed protocol | — | `sfm-video.ts` |
| `website/src/workers/probe.worker.ts` | 580 | Triangulation probe with its own `ShadeDevice` (`probe` 119-573) | — | `TriangulationProbe.tsx` |
| `lib/multi-video-job.ts` | 979 | `MultiVideoJob`: per-frame gate, SIFT and global descriptor (301-402); cap thinning (409-428); `finish` (431-751): graph events to messages, mapper stage totals, record summary, result; dense per component (872-937); host retention; `Snapshots` (210-244) | `MultiVideoJob`, `Snapshots`, `SNAPSHOT_INTERVAL_MS`, `Job*` types | `workers/multi-video.ts`; `multi-video-job.test`, `multi-video-lifetime.test` |
| `lib/multi-video.ts` | 253 | Page helpers: `MULTI_VIDEO_DEFAULTS` (15-24), query parser (35-48), thinning, tags, matrix, record options (177-195), partial videos | the listed functions and types | `SfmLab`, `MultiVideoPanel`, `MultiVideoActivity`, the job; tests |
| `lib/multi-video-stages.ts`, `lib/multi-video-activity.ts`, `lib/multi-video-input.ts` | 226, 133, 30 | Stage board model; view-graph activity model; file `videoId` and duration probe (uses the DOM) | board and activity functions | `SfmLab`, panels, the job; tests |
| `lib/dense-run.ts` | 545 | Dense defaults (11-61), `DenseJob`, `denseWorkingBytes` (115-117), `keepFrame`, `runDense` (263-545) through the `DenseRunContext` port (189-224), `endDense` (227-261) | `DENSE_LAB_OPTIONS`, `DENSE_LAB_FUSION`, `DENSE_CLEAN_OPTIONS`, `denseCleanDefault`, `runDense`, `endDense`, … | `sfm.worker`, `workers/multi-video`, the job; `dense-run.test`, `gpu-loss-context.test` |
| `lib/dense-activity.ts`, `dense-cloud.ts`, `dense-geometry.ts`, `dense-quality.ts`, `dense-reason.ts`, `depth-archive.ts` | 144, 257, 14, 18, 73, 227 | Activity text; PLY writing, budgets and previews; three.js geometry; `?denseBoundary`/`?denseFiniteGain` parser; reason text; tar.gz depth artifacts | pure helpers, except `dense-geometry` (three.js) | `SfmLab`, `SfmViewer`, `DenseActivity`, `dense-run`; tests |
| `lib/run-record.ts` | 1885 | The record format: constants (1-69), types (71-651), validators (653-1455), builders (1461-1834), text summaries (1572-1585, 1838-1885). Section 5 has the detail. | `RUN_RECORD_FORMAT(S)`, `RunRecord*`, `runRecordProblem`, `gpuRecord`, `chunkRecord`, `denseRecord`, `sumDenseRecords`, `reportGpuMs`, `kltLine`, `denseLine`, … | 11 `website/src` modules (incl. `SfmLab.tsx` and the worker), the collector `website/worker.mjs:4-9`, 2 scripts, 11 test files and `record-fixture.mjs` |
| `lib/send-record.ts`, `result-export.ts`, `result-file-sources.ts` | 316, 89, 43 | Result-file plan, upload and record post; `typecv-reconstruction-v1` JSON and PLY; files for each component | — | `SfmLab`, `TriangulationProbe`; tests |
| `lib/probe-plan.ts`, `lib/probe-record.ts` | 88, 288 | Probe plan in `localStorage`; probe format and validator. Its validator helpers `finite`, `count` and `text` (133-138) copy those of `run-record.ts:653-658`. | — | probe worker and page, collector |
| `lib/keyframe-gate.ts` | 135 | `PhaseGate`, `PHASE_GATE_OPTIONS`, `?gate=` parser. No DOM. | yes | `sfm.worker`, the job, `SfmLab`, `scripts/measure-keyframe-gate.mjs`, `scripts/measure-klt-agreement.mjs` |
| `lib/klt-join.ts`, `lib/klt-gate.ts` | 194, 150 | Tracker-pair merge, verdicts, anchors, median displacement, `MIN_PAIR_MATCHES`/`KLT_*` constants; lab gate over two records | yes | `sfm.worker`; `scripts/klt-lab-gate.mjs`; `klt-join.test` |
| `lib/guided-matching.ts`, `lib/sift-quality.ts` | 66, 128 | Guide RANSAC options, E wrapper, `preferGuided`; query parsers (octave, guided, clahe, features, ORB, KLT) | yes | `sfm.worker`, `SfmLab`; tests |
| `lib/video-sampling.ts`, `video-range.ts`, `sfm-input.ts`, `sfm-video.ts`, `decode-protocol.ts` | 47, 38, 100, 291, 117 | fps default and parser, timestamps; ranges; prepared-JSON parser; decode paths (mediabunny, browser); decode protocol | yes | `SfmLab`, `decode.worker`, `sfm.worker` (`VIDEO_SAMPLE_FPS`); tests |
| `lib/main-thread.ts`, `working-budget.ts`, `gpu-preparation.ts`, `video-connection-scene.ts` | 127, 14, 22, 370 | Long-task watch (browser); budget formula; preparation phases; three.js view of the video graph (presentation) | yes | `SfmLab`, both workers, components |
| `components/SfmLab.tsx` | 5441 | Run control: URL overrides (271-434), record builders `buildRunRecord` (436-603) and `buildMultiRunRecord` (610-775), reference comparison (809-832), `runBenchmark` (1914-1972), `runMultiVideo` (2067-2779), `runSfM` (2781-3360). `SfmLabContent` (1031-5441) has 95 `useState`, 35 `useRef` and 15 `useEffect`. The JSX starts at 4035. | default `SfmLab` | `pages/lab.astro`, `pages/ko/lab.astro`; `frame-navigation-browser` and `gpu-preparation-browser` bundle it |
| `components/SfmViewer.tsx` | 2404 | Viewer. Only its interface was read (scene, `videoColors`, dense layers). | — | `SfmLab` |
| `src/gpu/multi-video.ts` | 401 | View ids, view graph on the GPU backend, components, `reconstructIncrementalGpu` for each component (`PNP_BATCH` 64 at 186, `focalPerVideo: false` at 297), the owned variant with release callbacks (214-380), `describeKeyframeGpu` (386-401) | exported only through `src/gpu/internal.ts:91-103` | job (through the runtime bundle), tests, `scripts/measure-multi-video*.mjs` |
| `src/gpu/multi-video-lifetime.ts`, `src/gpu/internal.ts` | 83, 106 | Owned feature store and `backingBytes`; the internal entry, "a bundle of its own with its own copy of the runtime" (internal.ts:2-4), shipped as `/runtime/internal/gpu.js` | yes | `src/gpu/multi-video.ts`; `lib/multi-video-job.ts:26` by relative source path |

Context (F): `IncrementalSfmSession` (`src/gpu/mapper.ts:532-973`) is the single-video mapper. Its window is
8 to 64 views, 32 by default (558), and it evicts in 626-705. It is exported publicly (`src/gpu/index.ts:29`).
engine-v2 retires it in favour of the scene state machine (`docs/engine-v2.md`, "What is kept").

## 2. One video run as the worker executes it

F for the sequence and the lines. Mark **H**: the policy sits as a literal inside control flow. Mark **N**: the
policy is a named, exported and tested unit. Mark **N\***: the constant is named, but the rule that uses it is inline.

| # | Stage | Function and lines | Decisions, thresholds, retries, budgets |
|---|---|---|---|
| 0 | Intake | `self.onmessage` 1571-1635 | `multi*` goes to `workers/multi-video.ts` (1573-1578). `cancelDense` bypasses the queue (1581-1585). Other requests are serialized on `queue`. `start` maps 23 fields positionally onto `start()` (1594-1618). An error leads to `failJob`, `releaseFeatures` and disposal of the device (1624-1630). |
| 1a | Runtime and adapter | `start` 596-607 | `import(new URL('/runtime/gpu.js'))` (596). A preflight `navigator.gpu.requestAdapter()` runs once per worker (602). It sizes the budget and gives the adapter facts of the record (`ready`, 808; `outcome`, 1372). |
| 1b | Budgets | 621-629 | `scale = edge²/640² × (octave −1 ? 4 : 1)`, `workingBytes = max(64 MiB × scale, maxBufferSize/2)`, `poolBytes = 32 MiB × scale` (**H**) |
| 1c | Device reuse | 630-645 | Reuse requires 8 conditions together: device, availability, timestamp mode, bounded mode, fp64 flavour, budget, working bytes, adapter limits (**H**) |
| 1d | Device open | 648-684 | `maxModelsPerBatch: bounded ? 64 : 1024` (658, **H**). The progress post is throttled to 100 ms (669) and the "cold" window is 2000 ms (677) (**H**). No progress is posted while the dense stage runs. |
| 1e | Options | 689-714 | `maxFeatures = min(16384, 4096 × scale)` (694, **H**). KLT runs only for the video profile with SIFT (704). ORB overrides from `ORB_LAB_*` (159-161) (**H**). |
| 1f | Job and session | 717-760 | `resetKernelStatistics?.()` (717). `PhaseGate` only for the phase gate on a video (740). `createDenseJob` (748). `IncrementalSfmSession` with 9 controls and the callbacks `record`, `publish` and progress (751-759). |
| 1g | Loss handler | 770-807 | A loss during the dense stage ends the dense stage only. Otherwise `failJob` runs with the operation, the kernel, the bounded mode and the fp64 flavour. |
| 2a | Frame book-keeping | `processFrame` 811-821 | Only the 24 newest frames keep their image card (818, **H**) |
| 2b | Phase gate | 825-855 | **N** (`PhaseGate.decide`, `keyframe-gate.ts:91-123`). A skip discards the frame, pushes RGBA to KLT and posts `frameComplete`. `fallback` hands the frame to the legacy rule (854). |
| 2c | KLT track gate | 859-891 | Only `?klt=1&kltGate=1` without the phase gate. The median track displacement must be below `keyframeMinMotionPx × width/640` and the gap below `keyframeMaxGapSeconds` (865-867). **N\***: `medianTrackDisplacement` and `KLT_GATE_MIN_TRACKS` are named; the comparison is inline. |
| 2d | Features | 892-932 | SIFT (896) with `grayForKlt`, ORB (906) or the corner pyramid (909). Keypoint readback (920). `sampleColors` (keeps colours of the last 64 frames, 1128, **H**). `keepPixels` for the dense stage (922). Stage record (927). |
| 2e | Feature minimum | 933-946 | `count < 8` rejects the frame with "Too few detected features" (**H**; the same literal is in `multi-video-job.ts:359`) |
| 2f | Candidate set and descriptor retention | 949-961 with `registerFrame` 1249-1259 | Drop the features of every view that is not in `session.activeViewIds` (949-955, again 1088-1093). The last keyframe is matched first (960). After registration, keep the descriptors of the references (a new one when ≥ 0.5 s after the last, keep the first and 2 recent: 1251-1253) and of the 3 newest keyframes (1254) (**H**) |
| 2g | KLT tracks matching | 962-974 | `?kltMatching=tracks` on a frame already kept: push and early join. A pair with ≥ `KLT_COVER_MATCHES` (64) tracker matches skips descriptor matching (973, **N\***) |
| 2h | Pair matching | 975-1003 | SIFT: `matchFloatDescriptorsGpu`, ratio 0.8, mutual (991). Corner or ORB: `matchResidentFeaturesGpu` (995). Stage record (1000). |
| 2i | Legacy keyframe rule | 1004-1035 | Only on the last keyframe: `matches.length >= 16` (array entries, so 8 matches), median motion `< 8 × width/640` and gap `< 1 s`. Then the frame is not a keyframe: dispose, push to KLT, drop the dense image, `featureCounts.pop()` (1007-1018, **N\*** for the constants, **H** for the rule) |
| 2j | Guided matching | 1037-1048, `guidedMatches` 1184-1244 | Runs when SIFT, `guidedMatching` and ≥ 16 entries hold (**H**). The guide is a short E RANSAC (**N** `guideEssentialOptions`) with seed `101 + frame.id` (1211, **H**). Every keypoint of both frames is normalized on the host with `api.unprojectPixel` for each pair (1215-1224). The line bound is `GUIDED_LINE_PX` 4 (1175), a worker-local constant. `preferGuided` decides (1047, **N**). |
| 2k | Pair admission | 1050-1058 | `matches.length >= 16` entries, so 8 matches (**H**). The precomputed path needs 32 entries, so 16 matches (1099, **H**). A tracker-only pair needs `MIN_PAIR_MATCHES` = 16 matches (`klt-join.ts:30`, **N**). |
| 2l | KLT join and merge | 1062-1072, `kltJoin` 502-523 | **N** `mergeTrackerPairs`. Counters are kept in the job. |
| 2m | Registration | `registerFrame` 1260-1300 | `session.add` with `frameCamera` (1158-1172). Keyframe accepted, `lastKeyframe` set, a forced snapshot (1277), stage `keyframe` (1280), `frameComplete` with `verifiedInliers` (1283-1299) |
| 2n | KLT verdict | `kltFeedback` 529-540 | **N** `trackerVerdicts` and `kltReject` |
| 3 | Finish | `finish` 1455-1487 | `session.finish`, `releaseFeatures`, scene colours, `result` with `outcome()` |
| 4 | Dense hand-off | `prepareDenseDevice` 1490-1535, `denseContext` 1538-1569, `runDense` (dense-run.ts 263-545) | Replace the device when the timestamp mode differs (1491-1497). The dense stage then runs as photometric, geometric, fusion, clean (only with `dense.clean`), mesh (only with `dense.mesh`), archive and `denseResult`. Activity is throttled to 200 ms (dense-run.ts:304, **H**). |
| 5 | Snapshots | `publish` 376-395 | At most one snapshot every 500 ms (`LAB_OPTIONS.snapshotIntervalMs`). It posts landmark objects (385). The multi-video path posts packed arrays instead (`multi-video-job.ts:942`). |
| 6 | Stage records and run record | `record` 327-359, `trimStages` 365-374, `outcome` 1333-1396, `failJob` 1398-1454 | Stage totals by operation. The detailed list keeps at most 400 entries and keeps `keyframe` entries. Section 5 has the detail. |

F: none of `medianMotionPx` (397-418), `trimStages`, `frameCamera`, the retention rule (references and adjacent keyframes), the legacy rule
or the budget arithmetic has a unit test. A grep of `website/*.test.mjs` and `tests/` finds no test that names them.
F: no website test drives the worker with `gate=legacy`, `klt=1`, `features=orb`, `guided=tiled`, `bounded`,
`fp64` or `budget`. The four worker tests send at most one `frame` per run, and that frame is `precomputed` (sfm-worker-loss-browser.test.mjs:111-123, dense-quality-browser.test.mjs:70-82). So none of these four tests reaches stages 2b to 2k. Only `site.test.mjs` does, end to end in Chromium (CI). It drives the real lab with
`?octave=0&guided=0`, `?fps=1`, `?dense=1…` and `?decode=main` (236, 1014, 1188, 1424).

## 3. Separation problems

**3.1 Functions over about 80 lines** (F, `get_file_outline`):

| Symbol | Lines | Concerns mixed in one body |
|---|---|---|
| `sfm.worker.ts` `start` | 567-809 (243) | runtime import, adapter preflight, budgets, device reuse policy, open options and progress protocol, option derivation, job (36 fields), session, loss handler, `ready` |
| `sfm.worker.ts` `processFrame` | 811-1076 (266) | 3 gates, 3 front ends, colours, dense pixels, retention, ordering, KLT, matching, guided, admission, overlay messages, registration. It posts `frameComplete` from 4 exits plus `registerFrame`, with different fields (843, 870, 939, 1026, 1283). |
| `MultiVideoJob.finish` | 431-751 (321) | bytes accounting, two throttles, event-to-message translation, mapper stage totals, release callbacks, graph summary, matrix, model posts, dense input, record summary, result, memory release |
| `MultiVideoJob.frame` | 301-402 (102) | run cap 800, gate, `frameKeyOf`, SIFT, GPU time by diffing `kernelStatistics`, minimum, describe, messages |
| `dense-run.ts` `runDense` | 263-545 (283) | image selection, skip, archive, activity throttle, stage times, previews and block colours, memory peaks, clean, mesh, performance record, result, cancel and failure |
| `run-record.ts` `runRecordProblem`, `sumDenseRecords`, `denseRecord`; interfaces `RunRecord`, `RunRecordDense` | 209, 99, 78; 224, 103 | the validator of every section; sums over components |
| `SfmLab.tsx` `runMultiVideo`, `runSfM`, single-video `onWorkerMessage` | 713, 580, 167 | request correlation, watchdog, worker life, decode feed, React state, status text, dense wait, record building, debug hook |
| `SfmLab.tsx` `buildRunRecord`, `buildMultiRunRecord` | 168, 166 | record assembly that reads `navigator`, `screen` and `devicePixelRatio`. The multi-video builder makes a pseudo single-video result with literal mapper numbers. |
| `src/gpu/multi-video.ts` `reconstructMultiVideo` | 214-380 (167) | two result shapes chosen by an `owned` flag. The edge field list is written twice (type 137-148, literal 239-264). |
| `probe.worker.ts` `probe`, `step` | 455, 175 | outside the main scope; a third device-owning worker |

**3.2 The mutable `job` object** (F): its type is `sfm.worker.ts:174-263` (90 lines), with 36 top-level fields and an
`options` object of 30 fields (21 from `LAB_OPTIONS`, 9 added in `start`). Writers, from an awk and grep pass:
`processFrame`: frames, featureCounts, features, matchedPairs, rejections, skippedFrames, operation, the dense
images, phaseGate. `processPrecomputedFrame`: featureCounts, features, matchedPairs. `registerFrame`: references, features, keyframes, lastKeyframe, frameMs, stages.
`record`: summary, stages. `scene`: latest, pointColors. `sampleColors`: colors. `publish`: snapshot.\*.
`guidedMatches`: guidedPairs, guidedGain. `kltPush`, `kltJoinTracks` and `kltJoin`: klt, klt.anchor, klt.counters.
`releaseFeatures`: features, klt.final. `start` (device `onProgress`): kernels, lastKernel. Session progress: operation.
`failJob`: failed. `prepareDenseDevice`: priorDeviceGpu. `onmessage`: operation, dense.abort. `finish`: operation.
I: 18 functions and callbacks write one record with untyped parts (`session: any`, `stages: any[]`). No stage owns its state, so no stage can run alone.

**3.3 Module-level state** (F): `sfm.worker.ts` has 12 `let` bindings (168-173, 264-270): apiPromise, device,
deviceOptions, deviceTimestampRequest, adapter, deviceUnavailable, job, queue, lastProgress, lastCompile,
lastRequestId, requestContext. `workers/multi-video.ts` has 4 (71-83): apiPromise, active, lastProgress, lastDenseKernel.
I: after a single-video run, the next multi-video run in the same worker opens a second device. Reasons: `SfmLab.tsx`
reuses `workerRef` (2451-2453), and nothing disposes the module-level `device` of `sfm.worker.ts`. This was not observed.

**3.4 Duplicated logic** (F, with locations):

| What | Copies |
|---|---|
| Mapper controls (500, 3, 1, 0.5, 4, 8, true) | `LAB_OPTIONS` (sfm.worker.ts:125-157); `MAPPER` (workers/multi-video.ts:59-67); the record's pseudo options (SfmLab.tsx:684-697); `LAB` (scripts/measure-focal.mjs:87-94, "LAB_OPTIONS in website/src/workers/sfm.worker.ts"); partial copies in 9 more scripts, e.g. `measure-video-sfm-performance.mjs:157-162`, `benchmark-sfm-pyramid-e2e.mjs:156-161, 402-407` |
| Analysis scale and budgets | sfm.worker.ts:621-629; workers/multi-video.ts:108-124 and 194-198. `workingBytesFromAdapter` (working-budget.ts:9-14) and `denseWorkingBytes` (dense-run.ts:115-117) are the same formula under two names. |
| Feature cap `min(16384, 4096 × scale)` | sfm.worker.ts:694; workers/multi-video.ts:142; the 16384 bound again in multi-video.ts:46 |
| Device open and loss handling | 5 `ShadeDevice.open` sites (sfm.worker.ts:683, 1517; workers/multi-video.ts:114, 189; probe.worker.ts:208) and 4 loss handlers (sfm.worker.ts:770, 1522; workers/multi-video.ts:133, 215). The adapter preflight runs in sfm.worker.ts:600-607 and workers/multi-video.ts:85-89 (called twice). |
| Camera of a frame | `frameCamera` (sfm.worker.ts:1158-1172) and `cameraOf` (multi-video-job.ts:169-183): the same body, with a different focal source |
| Colours under keypoints, landmark colours | `sampleColors` (sfm.worker.ts:1117-1130, floats 0..1, 64-frame window) and (multi-video-job.ts:185-194, bytes); `scene` (sfm.worker.ts:304-326) and `MultiVideoJob.scene` (946-978) |
| Snapshot throttle | `publish` (sfm.worker.ts:376-395) and `Snapshots` (multi-video-job.ts:210-244); 500 ms twice (`LAB_OPTIONS.snapshotIntervalMs`, `SNAPSHOT_INTERVAL_MS` 197) |
| Stage and GPU accounting | `record` (sfm.worker.ts:327-359) and `onStage` (multi-video-job.ts:539-550) with `stage()` (680-686); `gpuFields` (sfm.worker.ts:1307-1330, merges an earlier device) and `MultiVideoJob.gpuFields` (765-772); the page merges two device tables (SfmLab.tsx:2639-2640); the chunk-rate read is copied in `outcome` (1340-1349) and `finalFields` (1551-1558) |
| Dense context | `denseContext` (sfm.worker.ts:1538-1569) and the context in `MultiVideoJob.runDense` (881-908) |
| Gate, SIFT and minimum | `processFrame` 825-946 and `MultiVideoJob.frame` 317-360. The meaning differs: the job keeps a `fallback` frame because it has no legacy rule (339-340). |
| Progress throttle 100 ms; activity throttle 200 ms | sfm.worker.ts:669 and workers/multi-video.ts:127; dense-run.ts:304 and multi-video-job.ts:456 |
| Page run control | `runSfM` and `runMultiVideo` copy fail, `armTimeout` and `request` (2174-2197 and 2854-2877), worker creation and `onerror` (2451-2466 and 3115-3130), the one-frame lookahead `last` (2493-2511 and 3196-3217), the dense wait (2626-2645 and 3229-3246) and the failure record (2716-2765 and 3305-3348) |
| Frame and profile types | `Frame` in sfm.worker.ts:39-61, SfmLab.tsx:196-213 and `JobFrame` (multi-video-job.ts:38-51); `Profile` in sfm.worker.ts:63 and SfmLab.tsx:215 |
| Pair admission `>= 16` in scripts | `benchmark-rotation-tracking-rgb.mjs:183`, `measure-video-sfm-performance.mjs:146`, `probe-general-video-rgb.mjs:145`. They re-implement a simplified pipeline: every earlier frame, no gate, no guided matching, no retention. |

Worker against `src/gpu/multi-video.ts` (F): the two paths use two matching policies on the same SIFT features.
The worker uses a sequential window, ratio 0.8, mutual matching and ≥ 8 matches. The library uses the view graph,
ratio 0.8 and `minMatches` 30 (`src/reconstruction/view-graph.ts:253-261`, and again in `MULTI_VIDEO_DEFAULTS`).
They also use two mapper drivers: a streaming session, and a batch `reconstructIncrementalGpu` with `pnpBatchSize`
64 that the single-video path does not set. `focalPerVideo: false` is decided in the library (297) and stated again by the page (multi-video.ts:192).

Worker against `IncrementalSfmSession` (F): two retention policies are stacked. The session keeps its window and
anchors (mapper.ts:626-705). The worker adds references and adjacent views on top (1249-1259) and prunes to
`activeViewIds` (949-955). The focal has two owners: the session applies `focal.ratio` to added images (601),
and the worker builds cameras from `session.focalSearch.ratio` (1162). Unregistered reasons live in the session
map and in the worker's `rejections`, merged at each post. The 16-entry literal is also in mapper.ts:682.

**3.5 Protocol mixed with computation** (F): the worker posts from inside computation: 13 `post({…})` and 4
`self.postMessage` calls. The job has 14 `this.post(`, `dense-run.ts` has 9 `context.post(` and `workers/multi-video.ts`
has 7 `self.postMessage`. No typed protocol module exists for the SfM worker; `decode-protocol.ts` exists for the decode worker.
The page sends untyped objects (`request(type: string, message: object)`, SfmLab.tsx:2190, 2871) and switches
on strings over `any`. Counts of `any`: 41 in the worker, 47 in the job, 34 in SfmLab.tsx.

**3.6 Pipeline logic trapped in the browser worker** (F): `sfm.worker.ts` uses imports without extensions
(2-37), `self` at module top level (289-303, 1571) and the runtime only by URL (596). A Node import fails (observed above).
So `LAB_OPTIONS`, `trimStages`, `medianMotionPx`, the gates, the retention and admission rules, the budgets,
`outcome` and the guided orchestration cannot be reused or unit-tested from Node. Consequences: `scripts/measure-focal.mjs`
copies `LAB`, and other scripts re-implement the pipeline (3.4). The lib modules that use `.ts` imports and no globals
are imported from Node today: `keyframe-gate.ts` by two scripts, `multi-video-job.ts` and `dense-run.ts` by tests.

**3.7 Drift markers** (F): `sfm.worker.ts:608-609` says "8 models per batch", but 658 sets 64. `sift-quality.ts:104`
says the KLT capacity default "is 1024", but the lab uses 512 (`LAB_KLT_CAPACITY`, 449); 1024 is the library value
(`src/gpu/klt.ts:67`). `sfm.worker.ts:80` says the edge is "640 by default", but the page default is 1280 (SfmLab.tsx:1156).
Doc comments in SfmLab.tsx:263-269 and 276-278 are detached from the functions they describe (`buildRunRecord` at 436, `budgetOverride` at 289).
`klt-tracking.md` 3.3 says a tracker pair needs "at least 16 matches, as a descriptor pair does", but descriptor pairs need 8 (2k).

## 4. Configuration

F. "Recorded" means the run-record path. "—" means not recorded.

| Flag | Parsed | Default (where) | Applied | Recorded |
|---|---|---|---|---|
| `?fps=` | `video-sampling.ts:24` (0 < v ≤ 120) via SfmLab.tsx:285 | 4, `VIDEO_SAMPLE_FPS` (video-sampling.ts:18). `LAB_OPTIONS.videoSampleFps` (sfm.worker.ts:149) also returns 4 in `result.options`, and the record ignores it. | page decode (3205, 2502) | `options.videoSampleFps` (SfmLab.tsx:526), page value |
| `?gate=` | `keyframe-gate.ts:41` via SfmLab.tsx:280 | `'phase'` on the page (keyframe-gate.ts:38); `'legacy'` in the worker (sfm.worker.ts:581) | worker 740 | `options.keyframeGate`; multi-video: the literal `'phase'` (SfmLab.tsx:697) |
| `?octave=` | `sift-quality.ts:27` | −1 at edge ≤ 640, else 0 (page); 0 (worker 575) | SIFT and budget scale | `options.siftFirstOctave` |
| `?guided=` `1/0/tiled` | `sift-quality.ts:27, 47` | on and scalar (page); off (worker 576) | worker 1037 | `guidedMatching`, `guidedKernel` (tiled only) |
| `?clahe=1` | `sift-quality.ts:35` | off | worker SIFT; multi start (SfmLab.tsx:2473) | single: `options.clahe`; multi: literal `false` (SfmLab.tsx:696) |
| `?features=orb`, `?orbFeatures/orbRatio/orbDistance=` | `sift-quality.ts:58, 74` | SIFT; `ORB_LAB_*` 4000, 0.95, 256 (sfm.worker.ts:159-161) | worker 706-712 | `features`, `frontEnd: corner`, maxFeatures, matchRatio, maxDistance |
| `?klt=1`; `?kltCapacity/kltFill/kltMatching/kltGate=` | `sift-quality.ts:96, 119` (capacity 1 to 8192; the literal bound equals `KLT_MAX_CAPACITY`, src/gpu/klt.ts:69) | off; 512 (worker 449) against 1024 (library) | worker, sampled video only (SfmLab.tsx:3178) | `options.klt`, `options.kltSetup`, `result.klt` |
| `?dense=1` | SfmLab.tsx:297 | on: `denseEnabled` is true (SfmLab.tsx:1204) | worker finish | presence of the `dense` block |
| `?denseFusion=`, `?denseClean=` | SfmLab.tsx:326, 319 | `'tsdf'` and clean = (fusion = pixel), written in SfmLab.tsx:1208-1209 and in `DENSE_LAB_FUSION`/`denseCleanDefault` (dense-run.ts:28, 52); the worker defaults (748) are unused by the page | preset to worker | `dense.options.fusion`, `denseClean` |
| `?denseMesh=1`, `?gpuTimestamps=1` | SfmLab.tsx:309, 303 | off; dense timestamps = bounded mode (sfm.worker.ts:1491), multi: false (workers/multi-video.ts:191) | dense stage, dense device | `dense.options.denseMesh`; dense timestamp fields |
| `?denseBoundary=`, `?denseFiniteGain=` | `dense-quality.ts:7` | absent | MVS options | dense options |
| `?bounded=1` | SfmLab.tsx:271 | off; single-video only | device 1c/1d | `options.boundedSubmissions` |
| `?fp64=`, `?budget=` | SfmLab.tsx:430, 289 (1 to 60000) | `'auto'`; the device's 250 ms | both workers' devices | `options.fp64Flavor`; `timing.budgetMs` |
| `?decode=` | `decode-protocol.ts:60` | feature detection | `sfm-video.ts:279` | `options.videoDecode` |
| `?canonical=`, `?maxKeyframes=`, `?maxFeatures=` (multi) | `multi-video.ts:35-48` | true; 60; `min(16384, 4096 × scale)` (stage 1e) | job and library | `options.multiVideo.*`, `options.maxFeatures` |

UI tunables (F): the analysis edge is 1280 on the page (SfmLab.tsx:1156) and 640 in the worker (574). The focal is
0.8 (`DEFAULT_VIDEO_FOCAL`, 261) with auto mode on (1076). The benchmark uses `BENCHMARK_TIMES` (217), 2 passes and edge 640 (1914-1972).
The watchdog is 120 s (single, 2845), 240 s (multi, 2127) and 600 s for dense (2628, 3231).
Not recorded (F, from a grep of `run-record.ts`): `keyframeMinMotionPx`, `keyframeMaxGapSeconds`, `minInitialAngleDeg`,
`minTriangulationAngleDeg`, `PHASE_GATE_OPTIONS`, the reference and retention rule, the 8, 16 and 32 minimums,
`KLT_COVER_MATCHES`, `GUIDE_RANSAC_ITERATIONS`, `GUIDED_LINE_PX`, the default KLT capacity, the analysis edge
(only as `input.width/height`), `MULTI_VIDEO_DEFAULTS.maxKeyframes`.
I: a record therefore does not identify the gate and matching configuration of the run. Two records from builds with different values of these constants look alike.

## 5. The run record

**Growth** (F, from `docs/lab-run-records.md` and the validator): v3 is the base and accepted from old pages.
v4 adds `timing.chunks` and `timing.budgetMs`. v5 adds the `dense` block (`docs/dense-mvs.md` 5.3). v6 adds
`timing.gpu`, `gpuTimestamps`, `gpuDropped`, `stages[*].gpuMs`, `gpu.features` and 3 compute limits.
`RUN_RECORD_FORMATS` keeps v3 to v6 (run-record.ts:18-23). Version branches sit inline at 1144, 1244, 1252-1256 and 1299.
Many additions came without a version change: `RunRecord` has 44 optional fields and `RunRecordDense` 13 (grep `?:`).
Examples: keyframeGate, keyframes, videoSource, range, klt and kltSetup, focalSearch, mainThread, all multi-video
fields, fusion, clean, mesh, files, hostRetention, descriptorCache and performance.

**Contents of its 1,885 lines** (F, outline): constants and result-file names 1-69. Types 71-651 (`RunRecord`
alone 428-651). Validators 653-1455, with builders interleaved at 685-730 and 906-957. Builders and normalizers
1461-1834 (`reportGpuMs`, `gpuRecord`, `summarizeStages`, `chunkRecord`, `denseRecord`, `sumDenseRecords`,
`cleanRecord`, `meshRecord`). Text summaries `chunkLines`, `kltLine` and `denseLine` (1572-1585, 1838-1885).
The types and the hand-written validator are two sources of truth. `runRecordProblem` reports only the name of the first bad section (1111-1319).
The module runs in three places: the page, the Cloudflare collector (`website/worker.mjs:4-9`) and Node tests and
scripts. It has no DOM dependency today.

**Where a record is assembled** (F). Four places:
- The worker: `record`, `outcome`, `gpuFields`, `kltRecord` and the partial outcome in `failJob`.
- The job: `MultiVideoSummary` (637-679), then the stage rows and the `result` message (680-747).
- The page: `buildRunRecord` and `buildMultiRunRecord` (SfmLab.tsx:436-775), which read `navigator`, `screen` and
  `devicePixelRatio`. The multi-video builder wraps a pseudo single-video result with literal options (671-711).
- `run-record.ts`: the builders.

**Separable?** I: yes. The worker part is a ledger fed by callbacks. It needs the device only for counters
(`statistics`, `kernelStatistics`, `chunkRates`), so it can take a small reader interface. The page part needs the
browser only for the `client` and `gpu` blocks, which can be an `environment` argument. Then `buildRunRecord` is pure
and testable in Node. The collector keeps importing the same validator through a barrel `run-record.ts`.

## 6. Target structure (fits `docs/engine-v2.md`)

This section is a proposal: the principles, the tree and the mapping.

Principles (proposal):
1. Each stage module is pure orchestration: no `self`, `navigator`, `location` or DOM, and `.ts` import specifiers.
   It does device work through an injected port. Today's precedents are `JobApi` (multi-video-job.ts:64-95) and
   `DenseRunContext` (dense-run.ts:189-224).
2. A stage emits typed events. Only the worker turns events into `postMessage`.
3. One options schema holds the defaults, the URL parsers and the record mapping. The page, both workers, the record and the scripts import it.
4. The record is built from a ledger plus an environment by pure functions.

I: then the same pipeline runs in the worker, in Node with a fake port, and in Node on `ShadeDevice.open({ backend: 'cpu' })`
(already used in `tests/typeshade/bundle-admission.test.mjs:68`). That serves Rules 1.1 and 1.2 and engine-v2 decision 4 (the oracle).

```
src/pipeline/ (name and area for the owner, risk 6; internal entry only)  website/src/lib/
  options.ts       schema, defaults, derive(scale)            lab-protocol.ts  typed page <-> worker messages
  schedule.ts      timestamps, fps, range, `last` look-ahead  lab-device.ts    open, reuse, loss, swap (both workers)
  keyframe-gate.ts phase + track + legacy rule as one unit    run-options.ts   URL and UI -> options (one parser)
  features.ts      front end via port, minimum, colours       run-record/      schema, validate, build, summary + barrel
  matching.ts      candidates, retention, admission, guided   lab-run.ts       page run controller (request, watchdog, feed, dense wait)
  klt.ts           tracker life, join, merge, verdict         workers/sfm.worker.ts  about 150 lines: messages <-> job
  registration.ts  driver over IncrementalSfmSession (scene engine later)
  single-video-job.ts / multi-video-job.ts / dense-driver.ts / ledger.ts
```

| Today | Target |
|---|---|
| `LAB_OPTIONS`, `PHOTO_PROFILE`, `ORB_LAB_*`, `LAB_KLT_CAPACITY`, `GUIDED_LINE_PX` (sfm.worker.ts); `MAPPER`, vocabulary (workers/multi-video.ts); `MULTI_VIDEO_DEFAULTS`; `DENSE_LAB_*`; the pseudo options (SfmLab.tsx:684-697); `LAB` (measure-focal.mjs) | `options.ts` |
| budgets (sfm.worker.ts:621-629; workers/multi-video.ts:108-124, 194-198), `working-budget.ts`, `denseWorkingBytes`, feature cap | `options.ts` `derive()` and `lab-device.ts` |
| `video-sampling.ts`, `video-range.ts`, the page `last` look-ahead (2 copies) | `schedule.ts` |
| `keyframe-gate.ts`, inline rules (sfm.worker.ts:859-891, 1004-1035), `medianMotionPx` | `keyframe-gate.ts` |
| extraction and minimum (sfm.worker.ts:892-946; multi-video-job.ts:343-360), `sampleColors` (2 copies) | `features.ts` |
| retention and candidates (949-961, 1088-1094, 1249-1259), matching loop (975-1058), `guidedMatches`, `frameCamera`/`cameraOf`, `guided-matching.ts` | `matching.ts`, `registration.ts` |
| `kltPush`, `kltJoinTracks`, `kltJoin`, `kltFeedback`, `kltRecord` (431-565), `klt-join.ts` | `klt.ts` |
| `processFrame`, `processPrecomputedFrame`, `registerFrame`, `finish` | `single-video-job.ts` |
| `lib/multi-video-job.ts`; `lib/dense-run.ts` | moved, otherwise unchanged; `dense-driver.ts` |
| `record`, `trimStages`, `outcome`, `gpuFields`, the failure partial; the job's stage totals | `ledger.ts` |
| `scene`, `publish`, `postMatchLines`; `Snapshots`, the job's `scene` | one throttle and colouring helper inside the jobs |
| `start` device code, `prepareDenseDevice`, loss handlers (4), adapter preflight (2) | `lab-device.ts` |
| `Request`, `MultiRequest`, the page's untyped `request()` and string switches | `lab-protocol.ts` |
| SfmLab.tsx 271-434 overrides and parser calls (3142-3181, 2468-2482) | `run-options.ts` |
| `run-record.ts`, `buildRunRecord`, `buildMultiRunRecord`, probe validator helpers | `run-record/` |
| `runSfM`, `runMultiVideo`, `runBenchmark` run control | `lab-run.ts` (React keeps the state setters as callbacks) |

## 7. Phased plan (each phase is pull requests that change structure only)

This section is a proposal. The day counts are estimates (I).

Common gate for every phase (F for what exists today):
- `npm test`: 445/445 on this configuration.
- The 36 browser-free website test files: 249/249.
- CI `check` (`npm run check`, website `astro check`, `build:website`) and CI `webgpu` (`test:webgpu`, `test:website` with Chromium).
- Run-record format checks: `runRecordProblem` through `website/worker.test.mjs` (2125 lines), `record-fixture.mjs`,
  `site.test.mjs` (477, 540, 626, 780) and `frame-navigation-browser.test.mjs` (405, 488, 676).
- Document audit: the moved symbols are named in the docs. `sfm.worker.ts` has 23 mentions in 11 docs,
  `run-record.ts` 10 in 5, `SfmLab.tsx` 11 in 5, `processFrame` 5 in 3, `dense-run.ts` 8 in 2.

| Phase (days) | Change | Files | Phase gate and new tests |
|---|---|---|---|
| P0 Characterize (1–2; test only) | Add a transcript test. It bundles the worker with esbuild into `vm` (as `dense-gpu-mode-worker.test.mjs:12-48`, but black-box through `onmessage` only) with a deterministic fake runtime. It records every message and every `session.add` input (pairs, cameras) for: phase skip and fallback, legacy, too few features, guided on and off, KLT gate and tracks matching, ORB, precomputed, dense with a fake MVS, failure mid-frame, loss idle and active. Timings are masked. Do the same for `MultiVideoJob.finish`. | new `website/sfm-worker-transcript.test.mjs`, fixtures | Passes on unchanged code. It is the identity gate for P1 to P6. |
| P1 Constants and pure helpers (2–3) | Move the defaults verbatim into one options module; `MAPPER` becomes a pick of it. Extract budgets, feature cap, `medianMotionPx`, `trimStages`, camera, colours, match lines and landmark colours. Keep both variants where they differ (the persistent colour cache in the worker, the per-call cache in the job). Give the worker imports `.ts` extensions. Point `scripts/measure-focal.mjs` at the module (same subset). Keep the record literals (`clahe: false`). | sfm.worker.ts, workers/multi-video.ts, multi-video-job.ts, dense-run.ts, working-budget.ts, SfmLab.tsx (pseudo options read only), measure-focal.mjs | Transcripts identical. Rewrite `working-budget.test.mjs:21-27` (source regex) as a behaviour test. New table tests for each helper. A test holds `LAB_OPTIONS`, `MAPPER` and the script `LAB` to today's values. |
| P2 Record split (2–3) | Split `run-record.ts` into schema, validate, build and summary behind a barrel, so its importers do not change. Move `buildRunRecord`/`buildMultiRunRecord` into `run-record/assemble.ts`. The only edit: the browser globals become an `environment` parameter that the page fills with the same values. Share the validator helpers with `probe-record.ts`. | run-record.ts, SfmLab.tsx, probe-record.ts | Golden records for single, multi, failed and dense runs from fixed inputs, byte-identical with `createdAt` masked. v3 to v6 fixtures stay valid. |
| P3 Policy units (3) | Name each inline rule with its current unit: `PAIR_MIN_MATCH_ENTRIES` 16 (= 8 matches), `PRECOMPUTED_MIN_MATCH_ENTRIES` 32, `MIN_FEATURES` 8, the reference rule (0.5 s, 3, 3), the legacy and track decisions, the guided trigger, KLT cover. The worker and the job call them. | sfm.worker.ts, multi-video-job.ts, new policy modules, klt-join.ts | Transcripts identical. Table-driven unit tests copied from today's branches. |
| P4 Worker by stage (4–5; 3 PRs) | 4a: a `SingleVideoJob` class (port-based, like `MultiVideoJob`) takes processFrame, registerFrame, guided, KLT, finish and outcome; the worker keeps intake and the device. 4b: `lab-device.ts` for the 5 open sites and 4 loss handlers. 4c: `lab-protocol.ts`; `start(message)` replaces 23 positional parameters; the page `request<T>()` gets types. | sfm.worker.ts, workers/multi-video.ts, SfmLab.tsx (types only), new modules | Transcripts identical. Update the hooks of `dense-gpu-mode-worker.test.mjs` (they name `job`, `device`, `queue`, `denseContext`, `releaseFeatures`, …). CI browser tests `sfm-worker-loss-browser`, `dense-quality-browser`, `gpu-preparation-browser`, `frame-navigation-browser`, `dense-activity-browser`, `site`. New Node tests of `SingleVideoJob` with a fake API. |
| P5 Page run controller (3–4) | Move the run-control copies of `runSfM`, `runMultiVideo` and `runBenchmark` into `lab-run.ts`. React keeps the setters. | SfmLab.tsx, new lab-run.ts | Same browser tests in CI. New Node test of the controller against a scripted fake worker. |
| P6 One options schema (3–4) | `run-options.ts`: a table of key, flag, parser, default, applies-to and record path. It replaces the 10 `*Override()` helpers and the scattered parser calls. Today's parsers stay as implementations. Effective values stay those the page sends now. | SfmLab.tsx, sfm.worker.ts, workers/multi-video.ts, run-record/assemble.ts, sift-quality.ts, keyframe-gate.ts, video-sampling.ts, dense-quality.ts, multi-video.ts, decode-protocol.ts, scripts | Table test over about 40 query strings: the old and the new start payloads and record options are identical. Transcripts and golden records identical. |
| P7 Library placement (3–5; after the owner decides) | Move the Node-importable stage modules to the chosen `src/` area, exported only from `src/gpu/internal.ts` (precedent: `multi-video.md` Amendment 6, decision 1). Add a CPU-tier run of `SingleVideoJob` on a tiny synthetic sequence, compared with WebGPU (Rule 1.2). Port the scripts that copy the pipeline. | src/…, internal.ts, scripts | Plus `test:shade` and `test:webgpu` (CI). |

## 8. TypeShade and platform limitations that shaped this code

| # | Limitation and evidence | Kind | What would lift it |
|---|---|---|---|
| L1 | Timestamp profiling is chosen at device creation (`requiredFeatures`, src/gpu/device.ts:251-253). The worker therefore replaces the device to run the dense stage unprofiled (sfm.worker.ts:1490-1535). The multi-video worker always opens a second device (workers/multi-video.ts:180-229). Both sides merge statistics across devices (`priorDeviceGpu`, `gpuFields`; SfmLab.tsx:2639-2640). The reason for the swap: on a SwiftShader synthetic case the unprofiled mode took 1,991.2 ms against 2,552.9 ms with timestamps, with the same 585 submissions (docs/large-run-recovery.md:281-287, 302-309). | F | A TypeShade runtime option to turn timestamp writes off per plan or per scope on a device that has the feature. It goes to TypeShade's `changes/` (Rule 3.3). |
| L2 | WebGPU reports no memory size. The budget is `max(64 MiB × scale, maxBufferSize/2)`, computed in the lab from a preflight adapter (sfm.worker.ts:600-629; workers/multi-video.ts:85-124). `ShadeDevice.open` requests its own adapter (device.ts:226-254), and `compiler-changes.md` (0083) records that the preflight remains. | F | TypeCV's device can derive `workingBytes` from the adapter it requests. No compiler change is needed. |
| L3 | A device loss names no work in flight. The lab tracks `lastKernel` from `onProgress` and keeps a `GpuRequestContext` (sfm.worker.ts:659-664, 770-807; gpu-loss-context.ts). `destroyRequested` is null for a GPUDevice that TypeShade owns (device-diagnostics.ts:4-5). | F | A runtime loss record of the plan or kernel submitted and not completed (TypeShade). |
| L4 | Chunk rates live as long as the device (change 0074). On a reused device `timing.chunks` includes earlier runs, and the lab cannot take a per-run delta. Cancellation waits for a chunk (dense-mvs.md 5.3 item 6). The GPU watchdog is a platform limit. | F | A per-run epoch or snapshot of `chunkRates` (TypeShade). The watchdog cannot be lifted. |
| L5 | There are two runtime copies, `/runtime/gpu.js` and `/runtime/internal/gpu.js` (internal.ts:2-4). The multi-video job opens its own device from the second copy (workers/multi-video.ts:97-114). | F; I: a device of one copy cannot be passed to the other | An owner decision on the export (Amendment 6, decision 1). This is not a compiler limit. |
| L6 | The host packs each dispatch (#717, fixed by typeshade#720; struct Residents are still dry-packed). This shaped the batch sizes (`maxModelsPerBatch` 64/1024, sfm.worker.ts:658; `PNP_BATCH` 64, src/gpu/multi-video.ts:186). | F | The remaining compiler limit concerns struct Residents only, and TypeCV declares none. |
| L7 | Emulated f64 flavours (0073) enter `?fp64=`, the device-reuse condition (sfm.worker.ts:635) and `options.fp64Flavor`. | F | engine-v2 decision 3 (f32) retires it. This is a TypeCV change. |
| L8 | GPU time cannot be attributed to stages inside one library call. The job diffs `kernelStatistics` sums (multi-video-job.ts:343-358, 754-762), and the view-graph stages carry no `gpuMs` (lab-run-records.md). | F | A named profiling scope on the device that labels plan reports (TypeShade runtime or TypeCV device). |
| L9 | The runtime is loaded by URL (sfm.worker.ts:596), so the API is `any` (41 uses in the worker). | F | A type-only import of `src/gpu/index.ts` (TypeCV build). No compiler change. |
| L10 | Reads after disposal or loss are not a stated contract, so the lab wraps them in try/catch (sfm.worker.ts:1309-1314, 1335-1349, 1505-1509, 1553-1558; multi-video-job.ts:754-772, 898-905). At pin `d36ac502` the TypeShade getters `statistics` and `chunkRates` do not check usability (node_modules/typeshade/src/runtime/device.ts:310-316, 424-426); only `ShadeDevice.info` states that it stays readable (device.ts:583). | F; I: the guards protect against a throw this pin does not raise. The tests fake that throw. | A documented contract that statistics stay readable (TypeCV device and TypeShade). |
| L11 | Keypoints are read back every frame. Guided matching normalizes keypoints on the host for every pair (sfm.worker.ts:920, 1215-1224). The multi-video path reads back full descriptors (src/gpu/multi-video.ts:391-395). | F | TypeCV design. engine-v2 says "descriptors stay resident". No compiler change. |

No `fuse` call exists in the orchestration files (F). Fusion reaches the record only through the report names
(`calls.join('+')`) and the accounting of composite reports (lab-run-records.md:34-52).

## 9. Risks and open questions for the owner

1. **Identity proof.** Behaviour inside the worker has thin coverage (section 2). The browser tests do not run in
   this container. Open question: do you accept a test-only P0 pull request (transcripts with a fake runtime) as the
   identity gate before any move?
2. **White-box and source-shape tests.** `dense-gpu-mode-worker.test.mjs:12-48` appends hooks that name nine internal
   bindings of the worker. `working-budget.test.mjs:21-27` matches source text with regexes. A pure move breaks both.
   Open question: may the structure pull requests rewrite these tests, with no change of what they check?
3. **Inconsistent pair minimums, preserved on purpose.** Descriptor pairs need 16 entries (8 matches). Tracker-only
   pairs need 16 matches. Precomputed pairs need 32 entries (16 matches). `klt-tracking.md` 3.3 claims the first two are equal.
   Open question: which value is intended? Any correction is a separate behaviour pull request.
4. **Record fidelity.** Multi-video records hard-code `clahe: false`, `keyframeGate: 'phase'` and the mapper numbers
   (SfmLab.tsx:684-697), while the run sends `quality.clahe` (2473). I: a `?clahe=1` multi-video record says `false`.
   The schema of P6 would correct this, and that is a change to the record.
   Open question: fix it in its own pull request with a record note?
5. **Defaults defined twice with different values.** gate (legacy in the worker, phase on the page), edge (640, 1280),
   guided (off, on), octave (0, −1 at ≤ 640), KLT capacity (512 in the lab, 1024 in the library), dense fusion and clean.
   The page always sends explicit values, so only tests and scripts see the worker defaults.
   Open question: which default is canonical in the single schema?
6. **Placement and API.** The stage modules can go to `src/` (internal entry) or stay in `website/src/`. A public
   export needs your approval (dense-mvs.md 5.4). engine-v2 replaces `IncrementalSfmSession`.
   Open question: how much should the programme invest in a registration driver around the session now?
7. **Concurrent edits.** Two commits landed on this branch during the audit. In the shallow history of 198 commits,
   `SfmLab.tsx` has 22 commits, `run-record.ts` 11 and `sfm.worker.ts` 9.
   Open question: do you freeze functional work on the files of each phase, or sequence it after the phase?
8. **Two devices in one worker** (I, 3.3). A multi-video run after a single-video run can hold two GPU devices.
   Open question: is this a bug for a behaviour pull request, or the intended warm reuse?
