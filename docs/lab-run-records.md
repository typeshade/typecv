# Lab run records

The SfM lab at `/lab/` can send the measurement of one run to the site's collector,
so that runs on hardware the maintainers do not own can be compared. Sending is a
choice the visitor makes with the consent checkbox. Nothing is sent without it. With
the checkbox selected before a run, the record is sent as soon as the run completes
or fails; selected after a run, the **Send record** button sends it.
[Hardware observations](hardware-observations.md) gathers what these records showed on each of
the owner's devices, with the remedy of each observation.

## What a record holds

`website/src/lib/run-record.ts` defines the record (`typecv-run-record-v6`) and its
validator. One record is one run, completed or failed:

| Field       | Content                                                                                                                                                                               |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `site`      | The site commit (`/version.json`) and the runtime path                                                                                                                                |
| `client`    | Browser user agent, platform, language, core count, device memory class, screen size and DPR                                                                                          |
| `gpu`       | WebGPU adapter vendor, architecture, device, description, fallback flag, the buffer and compute limits and the feature list (v6, below)                                               |
| `input`     | Kind (sample, prepared JSON, video), frame count, analysis and source resolution, duration                                                                                            |
| `options`   | The front end (`corner` or `sift`), the lab's numerical controls, the assumed focal ratio and the submission mode                                                                     |
| `result`    | Registered views, landmarks, observations, mean reprojection error, matched pairs, features                                                                                           |
| `timing`    | Total wall time, device reuse (cold or warm), per-operation totals, per-frame times, kernels in first-use order, GPU time per kernel (v6), chunk rates and main-thread stalls (below) |
| `transfers` | Application upload and readback bytes                                                                                                                                                 |
| `benchmark` | `pyramid-8` and `cold` or `warm` when the standard benchmark produced the run, else null                                                                                              |
| `reference` | Agreement with a sample's reference cameras (below), else null                                                                                                                        |
| `failure`   | Null for a completed run; else the error, the operation, the kernel that started last, the device-loss reason and the frames processed                                                |
| `note`      | Up to 500 characters the visitor typed                                                                                                                                                |
| `dense`     | Present only for a run with the dense MVS (the run option "Dense MVS" or `?dense=1`, v5): its status, options, views, points, time per stage and device bytes (below)                 |

`timing.gpu` (v6) holds the GPU time of each fused call list the run executed. A row is
`{ name, groups, dispatches, timedGroups, gpuMs }`. `name` is the kernel names of the list joined
by `+`, as the fuse report names a group (`calls.join('+')`). `groups` counts the groups of that
list the run executed. `dispatches` counts their dispatches (the entries of `workgroupShapes`).
`gpuMs` sums the `gpuTime` of the `timedGroups` groups that carry one, in ms with
microsecond precision. The rows are in first-use order. They come from
`ShadeDevice.statistics.kernels`, which the device fills from every plan report and which the
worker empties at the start of a run. The table keeps at most 160 call lists
(`RUN_RECORD_MAX_GPU_ROWS`). `timing.gpuDropped` counts the lists that did not fit. For a run
with the dense MVS the rows cover the sparse run and the MVS together.

A timestamp query is optional on a device. A device without the feature `timestamp-query` has
`timedGroups` 0 and `gpuMs` null in every row, and `timing.gpuTimestamps` is false. The rows
still count groups and dispatches. A row never holds a 0 for a time that was not measured.
`timing.gpuTimestamps` is true when at least one group of the run carried a time. A group of a
plan can lack a time when the browser fails the read of its timestamps. The browser may also
round each timestamp to a coarser step, so read a row of a few microseconds as a count, not as a
time.

`timing.stages[*].gpuMs` (v6) is the same sum over the plans of one stage, from the stage's fuse
report. It is absent when no group of the stage carried a time. The difference of
`elapsedMs` and `gpuMs` is the host, fence and launch time of the stage. A multi-video record
has `gpuMs` on `features`, `globalDescriptor` and `reconstruction`. The view-graph stages run
inside one library call, so they carry none, and the rows of `timing.gpu` hold their kernels.

`gpu.features` (v6) is the adapter's feature names, sorted (`shader-f16`, `subgroups`,
`timestamp-query` and the others), at most 64. `gpu.limits` gains
`maxComputeWorkgroupStorageSize`, `maxComputeInvocationsPerWorkgroup` and
`maxStorageBuffersPerShaderStage`, 0 when the page could not read one. Both come from the
adapter, not from the device the run opened. The device requests `timestamp-query` when the
adapter offers it.

A v6 record has all of these fields. A `typecv-run-record-v5` record has none of them, and the
validator rejects one that has any. The validator rejects a negative or non-finite time. The cost of the collector is none beyond the TypeShade device's own: the device already
measures each plan's groups when it has `timestamp-query`, and the collector only reads the
report (`ShadeDevice.collect`).

`timing.chunks` holds, for each chunk key of the time budget (`scorePairModels`,
`scorePnP`, `bundleCg`, `bundleCachedCg`, `triangulateTracks`), what `ShadeDevice.chunkRates` reports when the run
completes or fails: `msPerUnit` (the learned time of one unit), `chunks` (the timed
chunks), `longestMs` (the longest timed chunk) and `lastUnits` (the units of the last
timed chunk). A device times at most eight chunks of each key, and the later ones run
untimed (`docs/gpu-sfm.md`, "The time budget"), so `chunks` is at most 8 from site
commits after 2026-10-20 on.
`timing.budgetMs` is the device's `timeBudgetMs`, null when no device opened. The
device keeps the rates over its life, so on a warm run they include the earlier runs on
the same device. The page shows one line per key under the record's summary and in
the stage log, for example `scorePairModels: 37 chunks, longest 212 ms, 0.41 ms/unit`.
The validator still accepts a `typecv-run-record-v3` record, which has neither field,
from a page loaded before v4 was deployed.

`timing.mainThread` (absent from a page deployed before 2026-10-21) is what the page's
own main thread did during the run (`website/src/lib/main-thread.ts`): `longTasks` and
`longTaskMs` (tasks over 50 ms, the Long Tasks API), `longestTaskMs` and
`longestTaskStage` (the longest one and the lab's stage when it ended), and the same
four for frame gaps (`requestAnimationFrame` more than 100 ms apart). The SfM job runs
in a worker, so a long task is the page's own work (the 3D viewer, the record, React)
or the browser's, not a kernel. A frame gap without a long task is the browser not
painting: the GPU process was busy, or the tab was hidden. `hiddenMs` (absent before 2026-10-21) is the
time the tab was hidden during the run. A hidden tab paints no frames, so the watch does not count a
gap that overlaps it. The page shows one line under
the record's summary, for example
`Main thread: 4 long tasks, longest 2851 ms (pnp · scorePnP (start)); 3 frame gaps, longest 2854 ms (…)`,
and the export's summary line shows the longest task and the longest gap. On SwiftShader
on 2026-10-20 the longest task was the viewer's first shader compile (2.8 s); the viewer
now compiles its programs before it reports ready.

`dense` (v5, present only when the run asked for the dense MVS with the run option "Dense MVS" or `?dense=1`) is the
measurement of `docs/dense-mvs.md` 5.3. `status` is `completed`, `cancelled` (the
visitor stopped it, and `points` counts the points fused before), `failed` (`failure`
names the error, the stage, the kernel and the device-loss reason) or `skipped`
(`skipped` says why: no registered view kept its image. Before Amendment 3 of
`docs/dense-mvs.md` it also said: more keyframes than the dense layout holds). It holds
the MVS options, the views with a depth map and those without one, the fused points,
the wall time of the MVS and of its three stages (`photometric`, `geometric`,
`fusion`), and the device's bytes: `liveBytesBefore` and `pooledBytesBefore` before
the first MVS allocation, the active bytes at the depth stage and at the fusion stage,
and the depth stage's formula: (16 N + 4) P before Amendment 3, and the bytes of the depth
stage's residency plan after it. The MVS's chunk keys join `timing.chunks`.

`options.keyframeGate` (`phase` or `legacy`, the page's `?gate=`) names the keyframe gate of the run
and `options.keyframes` counts the frames it kept. Both are optional and absent in records before
2026-10-21, which ran the legacy gate. The format number is unchanged.

`options.videoSource` (`source`, `frames` or `mixed`) names where a video sample's frames came
from. `source` is the sample's source video, decoded by byte range at the analysis edge. `frames`
is its 640 x 360 JPEG frame files, the fallback when the browser cannot decode the codec or the
video does not answer. `mixed` is a multi-video run whose videos took different inputs. The field
is optional. It is absent for an uploaded video, a photo set, a sample that stores no source video
and a record before 2026-10-21. The format number is unchanged.

A run of two or more videos (`docs/multi-video.md`, 6.4) adds optional fields. The format number is
unchanged, and a record without them stays valid. `input.videos` lists the videos: `videoId` (16 hex
digits, the content hash of 2.4), `label` (the sample id, or null for an upload, since a typed name is
never sent), `durationSeconds`, `keyframes`, `registeredViews`, `frames` (the frames read, kept or not), `width`, `height` (the analysis size), `focalRatio`, `component` (0-based, or
null) and `tags` (`component`, `alone`, `not connected`, `rotation`, `duplicate`). `options.multiVideo`
holds `descriptor` (`vlad-32-v1`), `k`, `minMatches`, `minEdgeInliers`, `maxKeyframesPerVideo`,
`growth` (`incremental`), `canonical` (false for `?canonical=0`) and `focalPerVideo`. `result.components`
lists, for each component, its keyframes, registered views, landmarks, videos and mean reprojection
error. `result.crossVideoEdges` counts the accepted edges between two videos and sums their inliers.
`result.viewGraph` gives the nodes, candidate pairs, matched pairs and verified edges, and the pair count
of exhaustive matching. `result.notConnected` lists the `videoId`s that join no component of two or more
videos. `timing.stages` adds `features`, `globalDescriptor`, `retrieval`, `pairMatching`,
`pairVerification`, `components` and `reconstruction`. The lists hold at most 64 entries.

A failed multi-video run has no worker summary, and its record still carries what the run did.
`input.videos` lists each video that got a `videoId`, with `frames`, `width`, `height`, the
`keyframes` known so far, the `registeredViews` of the maps that existed at the failure, and the
`component` of those maps. Its `tags` are empty, since the tags need the finished graph. `result`
counts the views, landmarks and observations of those maps: the final map of a component, or else
its last partial map. `timing.stages` holds one entry for each stage that started (`frames`,
`candidates`, `matching`, `verification`, `components`, `reconstruction`) with its time in
`elapsedMs`, and `timing.totalMs` is the time of the run up to the failure.

`options.denseClean` says whether the cleanup of the fused cloud ran (on by default,
`?denseClean=0` turns it off; `docs/dense-mvs.md` 3.12). `clean` is present when it ran or
failed: the tier, the points before it and after the outlier, voxel and cluster stages, the
outlier threshold, the voxel size, the DBSCAN radius, the clusters found and kept, the time of
each stage in ms, and `error` (empty unless the cleanup failed, when the raw cloud stays).
`points` stays the number of raw fused points. Both fields are optional, so an earlier v5 record
stays valid.
`options.denseMesh` says whether the mesh of the final cloud was on (off by default, `?denseMesh=1`
turns it on; `docs/dense-mvs.md` 3.13). `mesh` is present when the mesh ran or failed: the tier,
the points meshed, the lattice spacing and the truncation radius, the nodes and cubes, the
vertices and faces, the time of the lattice, field, cube and vertex stages in ms, and `error`
(empty unless the mesh failed, when the points stay). Both fields are optional and the record
stays v5.
A dense failure leaves `failure` null, since the sparse result completed. The
validator still accepts a `typecv-run-record-v4` record, which has no `dense`, and a v5 record, which has no GPU time.

A record holds no video, image, file name or network address. The page shows the
exact JSON before it is sent, and offers the same JSON for download or copy when the
visitor prefers to share it another way. The collector adds `server.id`,
`server.receivedAt`, the Cloudflare country code and colo. The sender's address is
hashed with the hour for rate limiting only and is not stored.

The collector also stores probe records of the triangulation probe (Probe, below).
`website/src/lib/probe-record.ts` defines the probe record
(`typecv-triangulation-probe-v1`) and its validator. A probe record holds the site
commit, the page address with its switches, the browser user agent, the adapter
strings, the fixture's cameras and poses, every step with its outcome, timings and
chunk measurements, and the bisection's hanging tracks with their two pixels. It holds
at most 64 steps, and the number of page reloads it took (`reloads`). It holds no image,
file name or IP address.

## The standard benchmark

The lab's **Standard benchmark** button loads the eight-frame pyramid sample and runs
it twice in one worker: a cold pass that includes shader compilation, then a warm pass
on the retained device. Each pass produces one record with `benchmark.pass` set. With
the consent box checked before the benchmark starts, both records are sent as they
complete.

The benchmark assumes a focal length of 0.8 × the image width, the lab's default for an
uncalibrated video. Before 2026-10-20 the default was 1.2, which reconstructed the pyramid
upside down (the apex twice as far from the camera as the ground); `options.focalRatio` in
each record tells the two configurations apart, and the pass times are not comparable
across them.

The SIFT front end runs at its lab defaults (`website/src/lib/sift-quality.ts`): the first
octave of -1 at 640 px and below, and guided matching. Records before 2026-10-20 ran octave 0
without guidance, and carry neither `options.siftFirstOctave` nor `options.guidedMatching`.
Their pass times are not comparable with later ones: octave -1 takes three to six times the
time on a software GPU (#84). `?octave=0&guided=0` on the lab's address runs the earlier
configuration.

`?guided=tiled` on the lab's address keeps guided matching on and runs its tiled WebGPU kernel
("Guided matching kernels"). A record of such a run carries `options.guidedKernel: "tiled"`.
Other records ran the scalar kernel.

`?clahe=1` on the lab's address runs CLAHE on each frame's gray image before the SIFT pyramid.
A record of such a run carries `options.clahe: true`. Other records carry `false` or no field.

`?features=orb` on the lab's address runs the ORB front end instead of SIFT (`docs/gpu-features.md`,
"ORB"). A record of such a run carries `options.features: "orb"` and `options.frontEnd: "corner"`,
since ORB is a binary descriptor matched by Hamming distance. A record without `options.features`
ran SIFT, or the corner front end. The format number is unchanged. For a measurement,
`?orbFeatures=N` (1 to 8000), `?orbRatio=R` (above 0.1, below 1) and `?orbDistance=D` (1 to 256)
override the lab's ORB defaults of 4000 features, ratio 0.95 and no Hamming bound. The record
shows them as `options.maxFeatures`, `options.matchRatio` and `options.maxDistance`.

`?klt=1` on the lab's address tracks points between the sampled frames of an uploaded video
(`docs/klt-tracking.md`, 5.2). A record of such a run carries `options.klt: true` and `result.klt`.
`options.klt` is `false` for a run that asked for no tracking, and absent in earlier records. The
block `result.klt` holds: `pushes` and `joins`; the tracks `alive` at the end; `ended`, the tracks
that ended by reason (index 0 alive, then out of frame, low structure, no convergence,
forward-backward, residual, epipolar, age); `joined`, `seeded`, `filledIn` and `filtered`; the
matches the joins emitted (`matches`) and what the merge with the descriptor matches did with them
(`pairMatches` added, `duplicateMatches`, `conflictingMatches`); the `pairs` that hold tracker
matches and the `trackerOnlyPairs`; the tracks `rejected` on the verdict of the pair verification;
`lengths`, the tracks by frames lived (under 2, 2, 3 to 4, 5 to 8, 9 to 16, 17 or more) with their
mean `meanLived`; and the times `msPush`, `msSelect` and `msJoin` in ms. `timing.stages` holds
`kltPush` and `kltJoin` beside the other operations. The format number is unchanged.

### Physical-GPU runs at the new defaults

Two records from an NVIDIA RTX 2080 (Turing, Windows, Chrome 155) on site `363a1f2`, the
first deploy with the new SIFT defaults and the `scorePnP` warm-up (#89). Facts from the
records:

| Record                       | Input                               | Pass | Total   | Views    | Landmarks | Mean error | Longest chunk |
| ---------------------------- | ----------------------------------- | ---- | ------- | -------- | --------- | ---------- | ------------- |
| `20261020115652533-2744bdb5` | pyramid, 8 × 640 × 360              | cold | 100.6 s | 8 of 8   |           |            | 140 ms        |
| `20261020115721960-944372ed` | pyramid, 8 × 640 × 360              | warm | 28.4 s  | 8 of 8   | 2,295     | 0.31 px    | 140 ms        |
| `20261020121308655-a6fb6f3b` | video, 566 × 1280 × 720 (4K source) | cold | 440 s   | 27 of 27 | 4,631     | 0.35 px    | 196 ms        |

Stage times in seconds:

| Record     | Features | Matching | Guided matching | Pair verification | Bundle | PnP  |
| ---------- | -------- | -------- | --------------- | ----------------- | ------ | ---- |
| `944372ed` | 0.53     | 0.25     | 7.43            | 8.08              | 10.4   | 0.61 |
| `a6fb6f3b` | 45.0     | 5.44     | 72.5            | 56.7              | 102    | 11.2 |

- Guided matching costs about 29 times the plain matching. The first reading was that
  `nearestFloatGuided` is the scalar kernel, not the tiled one. The next section measures
  that reading and does not support it.
- The longest `scorePnP` chunk is 16.4 ms and 18.5 ms. On the RTX 5060 Ti before #89 it
  was 581 ms. The devices differ, so this is not a direct comparison.
- The video run spent 87.8 s in 1,010 main-thread long tasks (longest 151 ms) and 73.4 s in
  543 frame gaps (longest 222 ms). The pyramid runs had none.

### SwiftShader against the RTX 2080

This table gives a rough rate of each stage on a physical GPU from a run on a software GPU. It
holds one run on each side, so read each ratio as an estimate, not a measurement.

Configuration: the standard benchmark, warm pass, on site build `0026f39` (main before #113),
in headless Chromium 1194 with SwiftShader, on a shared 4-core container with no physical GPU.
The load average was 10 to 14 during the run, because other jobs ran at the same time. Run on
2026-10-20. Result: 8 of 8 views, 2,252 landmarks, 0.31 px. The RTX 2080 record is `944372ed`
on site `363a1f2`. Between the two builds, only #97 (guided matching) changed the sparse
pipeline of this sample. #102 changes only uploaded video.

| Stage             | SwiftShader (s) | RTX 2080 (s) | Ratio | Dispatches | Submissions |
| ----------------- | --------------- | ------------ | ----- | ---------- | ----------- |
| Features          | 58.8            | 0.53         | 111   | 1,696      | 8           |
| Matching          | 159.4           | 0.25         | 638   | 132        | 22          |
| Guided matching   | 12.3            | 7.43         | -     | 132        | 22          |
| Pair verification | 11.3            | 8.08         | 1.4   | 6,875      | 726         |
| Bundle            | 19.9            | 10.4         | 1.9   | 3,502      | 158         |
| PnP               | 3.57            | 0.61         | 5.9   | 800        | 112         |
| Total             | 268.6           | 28.4         | 9.5   |            |             |

- Guided matching has no ratio. #97 cut the guide's RANSAC from 500 to 50 iterations between
  the two builds.
- Inference: features and matching are bound by arithmetic. A physical GPU runs them 100 to 600
  times faster than SwiftShader.
- Inference: pair verification and bundle are bound by the round trips of their many small
  submissions and reads, not by arithmetic. The RTX 2080 runs them only 1.4 to 1.9 times
  faster. A change that cuts their submissions or reads saves time on a physical GPU. A change
  that only cuts arithmetic saves little there. Amendment 8 of `docs/multi-video.md` corrects
  this reading: a CPU profile puts most of the host time in the check of every bound array before
  each dispatch (typeshade/typeshade#717), not in the round trips.
- Estimate for #113 (bundle schedule) on the RTX 2080: its SwiftShader bundle time fell from
  32.3 s to 18.0 s, and its dispatches from 3,613 to 2,804. If the bundle stays bound by round
  trips, the RTX 2080 bundle falls from 10.4 s to about 8 s. If it scales with the SwiftShader
  time, it falls to about 6 s. The owner's run on a physical GPU decides.

### Guided matching kernels

On WebGPU, guided matching can run `nearestFloatGuided` (the scalar kernel, one query per lane)
or `nearestFloatGuidedTiles` (the tiles of the plain matcher, with the guide's line test).
`GpuEpipolarGuide.kernel` picks one, and the lab's `?guided=tiled` selects the tiled kernel.
Both kernels give the same bits: `tests/typeshade/guided-matching-webgpu.test.mjs` compares
every index, flag and distance bit on SwiftShader (123 random fixtures and 6 SIFT cases), and
`tests/typeshade/sift.test.mjs` holds the scalar kernel on the CPU tier. The CPU tier
always runs the scalar kernel.

Configuration: the standard benchmark (cold pass, one run per row) and the 12-photo COLMAP
south-building sample, on this branch (`perf/guided-tiled-2`, built from main `9d36984`),
headless Chromium with SwiftShader, on a shared 4-core container with no physical GPU. The load
average was 10 to 14. Run on 2026-10-20 with `scripts/measure-orb.mjs`. The stage
`guidedMatches` includes the guide's short RANSAC (`verifyImagePairGpu`) and both match
directions, as in the table above.

| Sample    | Kernel | Stage (s) | Stage without the first pair (s) | Per pair, mean without the first (s) | Dispatches | Submissions | Landmarks | Mean error |
| --------- | ------ | --------- | -------------------------------- | ------------------------------------ | ---------- | ----------- | --------- | ---------- |
| Pyramid   | scalar | 20.9      | 10.1                             | 0.48                                 | 132        | 22          | 2,253     | 0.309 px   |
| Pyramid   | tiled  | 50.7      | 38.9                             | 1.85                                 | 132        | 22          | 2,253     | 0.309 px   |
| COLMAP 12 | scalar | 59.7      | 49.5                             | 1.41                                 | 216        | 36          | 3,346     | 0.280 px   |
| COLMAP 12 | tiled  | 261.1     | 242.5                            | 6.93                                 | 216        | 36          | 3,346     | 0.280 px   |

- Facts: the two kernels reach the same landmarks, mean error, matches per pair and guided
  gain (1,045 added matches on the pyramid, 2,092 on COLMAP).
- Facts: the dispatches (6 for each pair) and the submissions (1 for each pair) are equal. The
  tiled kernel changes neither count.
- Facts: the tiled kernel is 3.9 times (pyramid) and 4.9 times (COLMAP) slower than the scalar
  kernel on SwiftShader. The first pair of each run takes 10 to 19 s in both kernels, because
  it compiles the shaders. The whole-run totals differ by less than the load noise.
- Cause on SwiftShader (inference): a software GPU spends most of a tiled kernel's time in
  barriers, and the scalar kernel runs no sum for the candidates off the epipolar line (about
  97 % of them). The earlier probe of 1,600 x 1,600 descriptors measured the same order
  (`1e21752`).
- Estimate for the RTX 2080 (inference, not a measurement). The 7.43 s of the stage in record
  `944372ed` is the guide's RANSAC at 500 iterations (#97 cut it to 50) and 22 round trips, and
  the match kernels. The scalar guided kernel runs about 3 % of the plain matcher's sums, and
  the plain matcher takes 0.25 s for the whole pyramid on this card. So the match kernels of the
  stage cannot cost more than a fraction of a second, even if the scalar kernel is several
  times slower than the tiled one per sum. A tiled kernel would save at most about 0.3 s of the
  28.4 s pass. The claim that the kernel explains the 29 times is not supported. The owner's run
  of `?guided=tiled` against the default on a physical GPU decides.

Decision: the scalar kernel stays the default. The tiled kernel is bit-exact but slower on
SwiftShader, and its possible gain on a physical GPU is small and unmeasured. It stays opt-in,
through `GpuEpipolarGuide.kernel: 'tiled'` and `?guided=tiled`. A record of such a run carries
`options.guidedKernel: "tiled"`. Reopen the decision if a physical-GPU run of `?guided=tiled`
shows a saving.

## The COLMAP sample

The lab's two **COLMAP south-building** buttons load twelve photographs of the
[COLMAP demo dataset](https://demuc.de/colmap/datasets/) south-building (image credit
Christopher Zach), resized to 640 × 480, with the dataset's refined intrinsics scaled to
that size and the poses of its bundled sparse model
(`website/public/sample/south-building/`, derived by
`scripts/prepare-lab-colmap-sample.py`). The first button runs the whole GPU pipeline
on the photographs with the lab's front end, scale-space SIFT
(`extractSiftGpuResident`) and the tiled L2 matcher with a 0.8 ratio test, and the
calibrated intrinsics held fixed (`PHOTO_PROFILE` in `sfm.worker.ts`). The second feeds the dataset's own SIFT keypoints and raw matches
(`colmap-sift.bin`, the matched keypoints only) to the GPU mapper, so the mapper is
measured apart from the lab's feature front end.

After each snapshot and at the end, the page similarity-aligns the registered cameras
to the COLMAP poses on their common views (`website/src/lib/pose-alignment.ts`, the
method of `colmap_dataset.py compare`) and shows the common view count, the median and
maximum rotation difference in degrees, and the median and maximum camera center
difference as a fraction of the reference trajectory's radius. The record's `reference`
field carries the same numbers. The COLMAP model is a reference, not ground truth: both
systems hold the same intrinsics fixed, and with the second button they also share the
same features and matches.

## Chunk sizes after the rate settles

Pull request #63 sizes each chunk of a key from the rate the device measured, and stops
timing the key after eight timed chunks (it "settles"). Nothing corrects the rate after
that. The open question: does a settled key overrun its budget when the cost of a unit
grows later in a run (more landmarks, more tracks, a bigger PnP set)?

Method (2026-10-20, site commit `fc070a8`, headless Chromium on SwiftShader, a CPU
renderer and not a physical GPU, with another CPU-heavy job running on the same
machine). The two COLMAP 12-photo samples ran at 640 px with the defaults (budget
250 ms), the GPU front end, one cold run each. The run record does not hold the data:
`timing.chunks` has one row per key (`msPerUnit`, `chunks`, `longestMs`, `lastUnits`)
and only the timed chunks, so no settled chunk appears in it. The measurement used a
local, uncommitted patch of `planSteps` that posted the key, the units and the elapsed
time of every chunk, and fenced the untimed ones with `wait()` to read their time.

Measured (the budget is 250 ms; times in ms; "max/budget" is the longest settled chunk
over the budget). The table counts the settled chunks that the host marks as timed.

| Sample, key         | Settled chunks | Settled units (median) | Median | p95 | Max | Max/budget | Over 1.0 | Over 1.5 |
| ------------------- | -------------- | ---------------------- | ------ | --- | --- | ---------- | -------- | -------- |
| south-building      |                |                        |        |     |     |            |          |          |
| `bundleCg`          | 104            | 122010                 | 240    | 318 | 477 | 1.91       | 28       | 2        |
| `triangulateTracks` | 30             | 60                     | 9      | 29  | 31  | 0.13       | 0        | 0        |
| `scorePnP`          | 2              | 150                    | 149    | 169 | 169 | 0.68       | 0        | 0        |
| `scorePairModels`   | 706            | 1                      | 4      | 19  | 57  | 0.23       | 0        | 0        |
| gerrard-hall        |                |                        |        |     |     |            |          |          |
| `bundleCg`          | 110            | 103766                 | 194    | 256 | 350 | 1.40       | 9        | 0        |
| `triangulateTracks` | 29             | 57                     | 9      | 42  | 65  | 0.26       | 0        | 0        |
| `scorePnP`          | 2              | 162                    | 300    | 333 | 333 | 1.33       | 2        | 0        |
| `scorePairModels`   | 737            | 1                      | 4      | 25  | 69  | 0.28       | 0        | 0        |

The two runs registered 12 of 12 views (3360 and 2877 landmarks, 0.28 and 0.27 px) in
963 s and 582 s. 504 and 536 more `scorePairModels` chunks and 30 and 27 more
`scorePnP` chunks ran with `timed: false` (indirect dispatch), so the device never
timed them. Their times are small: median 2 to 5 ms, p95 22 to 33 ms, max 50 to 153 ms.
They do not change the table.

Facts (measured):

- `bundleCg` is the only key whose settled chunks reach the budget. The learned rate was
  0.00155 ms per unit (south-building) and 0.00190 (gerrard-hall). The settled chunks ran
  at about 0.0019 to 0.0020. Chunk sizes stayed at about 122000 and 104000 units, which is
  `0.8 * budget / rate`, so the median chunk is 0.96 and 0.78 of the budget.
- In south-building the median of the first ten settled `bundleCg` chunks is 208 ms and of
  the last ten 243 ms. The ms per unit by thirds of the run is 0.0018, 0.0020, 0.0020. In
  gerrard-hall the same medians are 175 ms and 184 ms, and the thirds are 0.0020, 0.0019,
  0.0018. So the ratio does not grow steadily over the run.
- The largest `bundleCg` chunks (477, 426 and 346 ms in south-building, 350 ms in
  gerrard-hall) are isolated peaks between chunks of 230 to 260 ms.
- `scorePnP` has only two settled chunks per run. The learned rate is 1.33 and 1.23 ms per
  unit. The two settled chunks of south-building ran at 0.86 and 1.13 ms per unit (129 and
  169 ms). The two of gerrard-hall ran at 1.65 and 2.05 ms per unit (268 and 333 ms), 1.3
  to 1.7 times the learned rate.
- `triangulateTracks` and `scorePairModels` stay at or below 0.28 of the budget.
- No chunk is over 1.5 times the budget except two `bundleCg` chunks of south-building.

Inferences:

- The risk is real but small on these inputs. A settled `bundleCg` chunk ran 25 % above the
  learned rate in south-building, and a settled `scorePnP` chunk ran up to 1.7 times above
  it in gerrard-hall, so a settled chunk can pass the budget. The worst overrun is 1.9
  times, in one key of two runs. The data do not show a steady growth of the ratio over a
  run. Two runs of 12 photos are too few to exclude it on a larger scene.
- The two peaks above 1.5 times are probably noise from the other CPU job on the same
  machine, since they sit between chunks of normal time. One run each cannot separate
  noise from drift.
- A physical GPU may differ in both directions: SwiftShader time is CPU time. The
  watchdog limit of a physical GPU is not measured here.
- `scorePnP` is the likeliest to overrun on a larger scene: it settles after few, small
  calls and its two settled chunks per run were 0.86 to 2.05 ms per unit.

Next action (proposal, not done): the smallest fix is to measure again one settled chunk
in sixteen, with the two fences of a timed chunk, and to feed that time to the rate. A
chunk that then exceeds 1.5 times the budget would also reset the key to timed chunks.
Repeat this table after the change, and on a physical GPU and a larger input.

## A lost device

When the browser loses the WebGPU device during a run (on Windows,
`DXGI_ERROR_DEVICE_HUNG` is the GPU watchdog resetting a device whose command stream ran
too long), the page reports the operation and the kernel that started last, and builds
a failure record with everything measured before the loss: the counts of the last map
snapshot, the per-operation totals, the kernels in first-use order and the `failure`
field. With consent given before the run, the record is sent at once. Open the lab
with `?bounded=1` to run the same job with fenced submissions: the worker waits for the
queue after each group of dispatches and verifies at most 64 models per batch
(`ShadeDevice.open({ boundedSubmissions: true, maxModelsPerBatch: 64 })`). The run is
slower. The record's `options.boundedSubmissions` says which mode ran.

Observed on the owner's iPhone (iOS 18.7, Safari 27) on 2026-10-19: every unbounded run
lost the device during pair verification, under either flavor of the emulated double,
with no reason from Safari; the first bounded run under the `'integer'` flavor registered
the pyramid sample's eight views (`20261019140521190-07c4d3e5`, 691 landmarks, 0.400 px,
116 s). A bounded run can still lose the device in one long dispatch
(`sweepPairModelSvd`, `20261019140528615-057b3152`). Fencing is a diagnostic, not the
remedy: the time budget of `docs/gpu-resident-scene.md` step 2 chunks the long kernels
themselves, and is the fix.

Observed on the same iPhone on 2026-10-19 22:52 UTC, on the site at `7eef646` (the time
budget, the fixed-cost null-vector solve, the batch of up to 512 trials and the budget
moved into the compiler, pull requests #49, #50, #53 and #54): the first unbounded run
under the `'integer'` flavor (`20261019225221735-85ed0ae6`, the cold pass of the standard
benchmark with the SIFT front end) passed pair verification for all 22 pairs and
registered the eight views (640 landmarks, 1979 observations) in 40.2 s, then lost the
device in the final bundle adjustment (`bundle #25`, last kernel
`snapshotBundleActivity`, no reason from Safari). Pair verification took 25.0 s of the
40.2 s (22 pairs, 688 submissions, 2697 dispatches); the bundle adjustments before the
loss took 6.1 s (7 calls, 42 submissions). The loss moved from pair verification to the
one long dispatch that remains on this path, the shared-focal route's `bundlePersistentCg`
of `src/gpu/joint-focal.shade.ts`: the final bundle runs with `refineSharedFocal: true`,
and on WebGPU without bounded submissions one workgroup runs the whole conjugate-gradient
loop of a bundle iteration in one dispatch. This is an inference from the code and from
the kernels the record names first in that bundle (`bundleFocalCameraRows`,
`bundleFocalBlock`, which only the shared-focal route runs). The remedy is the same as for
pair verification: run the conjugate-gradient iterations as chunk steps under the time
budget. Pull request #56 removes the fixed-focal route's kernel of the same name
(`src/gpu/bundle.shade.ts`), and pull request #58
removes the shared-focal one and the Schur cache's `bundleCachedPersistentCg`
(`docs/gpu-sfm.md`, "The BA's PCG loop").

Observed on the owner's Windows computer (Windows 10/11, an NVIDIA Blackwell GPU
(a GeForce RTX 5060 Ti)) on 2026-10-20 01:20 UTC, on the site at `329043c`
(`20261020012028865-b2ca931e`, `typecv-run-record-v4`, the cold pass of the standard
benchmark with the SIFT front end, unbounded submissions, the `'float'` flavor,
`refineSharedFocal: true` and `solverCheckpoints: true`, the lab's default): pair
verification passed for all 10 pairs of the 5 frames processed, in 1265 ms (330
submissions, 2166 dispatches; key `scorePairModels`: 90 chunks, the longest 4.1 ms,
0.77 ms/unit, budget 250 ms). Then the device was lost in `triangulation #11`, last
kernel `triangulateTracks`, with `DXGI_ERROR_DEVICE_HUNG` (`0x887A0006`) from
`ID3D12Device::GetDeviceRemovedReason` (5 frames processed, 10 matched pairs, 102920
application readback bytes). Every Windows record in the collector lost the device in a
triangulation, none in pair verification: the first, `20261019095412032-e5e0f937`
(2026-10-19, site `6c07fff`, 2 frames), was lost in `triangulation #2` (the export of
2026-10-20 05:19 UTC, 30 records). The same computer ran every Windows record. The model, a
GeForce RTX 5060 Ti, is the owner's statement of 2026-10-20; the adapter reports only
`nvidia` and `blackwell`. The owner's iPhone spent 1.0 s in triangulation over 17 calls of
record `20261019225221735-85ed0ae6` without a loss.

Inference, from the code and from the kernel the record names: at `329043c`
`triangulateTracksGpu` dispatched `triangulateTracks` once over all the tracks of a
call. Each invocation runs up to `maxPairHypotheses + 1` hypotheses, each with a Jacobi
SVD of up to 160 sweeps, then a point LM of up to `refinementIterations` steps, in
emulated f64. That one dispatch ran past the Windows watchdog of about two seconds. The
remedy is the time budget: `triangulateTracks` now runs as chunks of contiguous tracks
(key `triangulateTracks`), each a submission that the TypeShade device fences and times.
The first chunk of a device is one workgroup of 16 tracks, so the rate is learned from a
short dispatch, and each later chunk is sized to fit the budget
(`docs/gpu-sfm.md`, "Triangulation"). A chunk is never smaller than one workgroup, so a
single workgroup of 16 tracks that takes longer than the watchdog still loses the device.

Observed on the same Windows computer on 2026-10-20 02:07 UTC, on the site at `a3041af`
(the chunked `triangulateTracks`, pull request #60; `20261020020739347-d7d25f91`, the cold
pass of the standard benchmark, unbounded submissions, the `'float'` flavor): pair
verification passed for all 10 pairs of the 5 frames processed, in 10.9 s (330
submissions, 2166 dispatches; key `scorePairModels`: 90 chunks, the longest 162.7 ms,
0.80 ms/unit). The device was lost again in `triangulation #11`, last kernel
`triangulateTracks`, with `DXGI_ERROR_DEVICE_HUNG`. The index `#11` counts every
operation of the GPU mapper (`src/gpu/mapper.ts`), so this is the first triangulation of
the run in both records. Before the loss, 8 chunks of key `triangulateTracks` completed:
the longest took 8763 ms, the rate was 4.68 ms per track, and the last chunk was 16
tracks, the least chunk. Inference: a chunk of one workgroup of 16 tracks ran past the
watchdog, which the chunking cannot prevent (above). The 8763 ms chunk completed, so its
time is not GPU time alone: the first chunk of a key includes the pipeline creation.
Pair verification took 8.6 times the 1265 ms of `b2ca931e` for the same 10 pairs, and the
cause is not known. The next measurements, each a run on the same computer: the lab with
`?fp64=integer` (whether the `'float'` flavor's lowered kernel is the one that runs
long), the lab with `?bounded=1`, and the three variants of
`tests/typeshade/south-building-triangulation-webgpu.test.mjs` (below), which name the
loop.

Observed on the same Windows computer between 02:10 and 02:25 UTC on 2026-10-20, on the
site at `a3041af`, five runs, the browser restarted between the later ones:

- `20261020021036188-ba3c638d` (`'float'`, unbounded, the standard benchmark, 7.9 s):
  pair verification 1171 ms for the 10 pairs. Two chunks of `triangulateTracks`
  completed (16 and 64 tracks, the longest 31 ms, 1.0 ms per track). The third chunk
  lost the device in `triangulation #11`, `DXGI_ERROR_DEVICE_HUNG`.
- `20261020021122316-8f3753de` (`'float'`, unbounded, 2 frames, `refineSharedFocal`
  off, 6.7 s): one pair, two chunks completed (16 and 64 tracks, the longest 22 ms),
  the third chunk lost the device in `triangulation #2`. Every track of this job has
  two observations.
- `20261020022307924-efd20d5f` (`'integer'`, unbounded, 2 frames, 58.1 s): the features
  and the match ran (0.5 s). `verifyPair #1` ran 57 s with no pair model kernel
  finishing, then the device was lost at `finishPairModels` with Dawn's "A valid
  external Instance reference no longer exists", not a `DXGI` reason. Inference: the
  GPU process ended under the browser's watchdog, and the time went to the compile of
  the integer-flavor pair kernels, not to a dispatch (the same kernels under the
  `'float'` flavor verify 10 pairs in 1.2 s warm). Not measured.
- `20261020022404777-4881d87d` (`'float'`, unbounded, 31.6 s): pair verification 16.0 s
  (cold pipelines), `triangulateTracks` 8 chunks, the longest 9095 ms (the first, with
  the compile), 4.7 ms per track, the last chunk 16 tracks, then the loss in
  `triangulation #11`.
- `20261020022502483-57d0b9cd` (`'float'`, `?bounded=1`, 26.5 s): pair verification
  14.0 s (2166 submissions), `triangulateTracks` 7 chunks, the longest 5013 ms, the last
  chunk 16 tracks, then the loss in `triangulation #11`, `DXGI_ERROR_DEVICE_HUNG`.

Inferences from the five runs and `d7d25f91`. The compile of the `'float'` pair kernels
costs 10 to 15 s and that of `triangulateTracks` about 9 s on this computer; a second run
in the same browser reuses the pipelines (1.2 s and 31 ms). The hang is not the compile:
in `ba3c638d` and `8f3753de` the kernel had run two chunks before a chunk hung. The hang
is not the chunk's size alone: 64 tracks ran in 31 ms, and the next chunk of at most 256
tracks ran past the watchdog. The hang is in some track's work, in a job whose tracks have
two observations each, so the per-track loops (a Jacobi SVD of up to 160 sweeps on four
rows, the hypotheses, the refinement) are small by count. Bounded submissions do not
prevent it. The `'integer'` flavor does not reach triangulation on this computer. The
owner cannot run the Node test on this computer, so the lab owes a probe page that runs
the fixture's variants in the browser.

Observed on the same Windows computer on 2026-10-20 (NVIDIA Blackwell, Chrome 155 on
D3D12, the site at `cd97b62`). The probe page ran `triangulateTracksGpu` on
`tests/fixtures/south-building-pair-triangulation.json` with
`{ refinementIterations: 0, maxPairHypotheses: 1 }`. Five probe records give these
results:

- `e0c75517` and `41d3ba90`: the chunks of 16, 64, 256 and 1024 tracks completed, the
  longest in 5 ms. The chunk of tracks 1360..1671 then lost the device with
  `DXGI_ERROR_DEVICE_HUNG`. That chunk is 312 tracks: 19 full workgroups of 16 and one
  workgroup with 8 active lanes.
- `02ed0bfc`: tracks 1360..1663 (304 tracks, 19 full workgroups) completed in all three
  variants. 303 tracks were accepted of 303 expected, and the statuses were bit for bit
  those of the CPU tier.
- `14349a11`: tracks 1664..1671 alone (one workgroup with 8 active lanes) lost the
  device.
- `e37f29b0` (04:52 UTC): tracks 1656..1671 (one full workgroup that holds the 8 tracks of
  `14349a11`) completed in all three variants. 16 tracks were accepted of 16, the
  statuses were bit for bit those of the CPU tier, and the longest chunk took 6.7 ms.

Inference: the data of tracks 1664..1671 is not the cause. On this driver a workgroup of
`triangulateTracks` in which some lanes return before the loops runs past the watchdog.
A workgroup whose 16 lanes all run the track work completes. Each earlier Windows loss in
these records (`b2ca931e`, `d7d25f91`, `ba3c638d`, `8f3753de`, `4881d87d`, `57d0b9cd`)
was in a single dispatch or in a chunk whose track count was not a multiple of 16. The
branch `fix/triangulation-full-workgroups` removes the early return of the idle lanes
(`docs/gpu-sfm.md`, "Triangulation"). A run of the lab on that kernel is not recorded
yet.

Observed on the same computer on 2026-10-20 05:46 to 05:48 UTC, on the site at `a67e2df`
(pull request #67, the idle lanes compute and write nothing): the probe completed tracks
1664..1671 in all three variants (`20261020054650780-4b85f996`, 8 of 8 accepted) and all
1672 tracks (`20261020054723134-472d5020`, 5 chunks, the longest 5.9 ms, 1666 of 1666). The
standard benchmark then completed both passes, the first Windows runs to do so: cold
(`20261020054818902-2a5988bb`) 8 views, 653 landmarks, 2015 observations, 0.581 px in
26.4 s; warm (`20261020054827516-209bc7c5`) 8 views in 7.5 s, pair verification 2.6 s,
triangulation 0.18 s, bundle adjustment 3.6 s, PnP 0.45 s. The remedy is confirmed on this
device. `docs/hardware-observations.md` (A1) holds the digest.

Observed on the owner's iPhone on 2026-10-20 02:09 UTC, on the site at `a3041af`: the
standard benchmark completed both passes for the first time, with unbounded submissions
under the `'integer'` flavor. The cold pass (`20261020020913972-527f815b`) registered the
eight views (640 landmarks, 1979 observations, 0.520 px) in 57.8 s: pair verification
24.4 s (22 pairs, 726 submissions), bundle adjustment 24.1 s (8 calls, 155 submissions,
4858 dispatches), triangulation 4.7 s, PnP 2.8 s. The warm pass
(`20261020020957239-1d174b95`) gave the same map (0.520 px) in 41.9 s: pair verification
21.7 s, bundle adjustment 15.1 s, triangulation 1.2 s. The chunks of the cold pass:
`scorePairModels` 216 chunks, the longest 7 ms, 1.73 ms/unit; `triangulateTracks` 29
chunks, the longest 3011 ms, 2.33 ms/unit; `bundleCg` 67 chunks, the longest 953 ms,
0.0136 ms/unit, the last chunk 12198 units. The device's chunk rates persist across the
two passes of one worker, so the warm record's `chunks` counts and `longestMs` include the
cold pass (432, 51 and 132 chunks, the same longest values). The loss of `85ed0ae6` in
the final bundle adjustment (above) did not recur: the shared-focal CG runs as chunk steps
since pull request #58.

After a device loss Safari may also drop the WebGL context, so the 3D viewer fails to
initialize on the next run. The run goes on without the viewer, and the map stays empty
until the page is reloaded.

## The flavor of the emulated double

TypeShade computes `f64` as two `f32`s, under one of two flavors (TypeShade change 0073).
The `'float'` flavor binds the `_fp64` guard; the `'integer'` flavor does the same
arithmetic through integer primitives, which a fast-math shader compiler cannot
reassociate. Apple's Metal compiler collapses the float flavor, which the iPhone records
of 2026-10-19 showed as slow pair verification and no stable initial pair. The device
picks the flavor from the adapter's info and the user agent (`ShadeDevice.open`,
`fp64Flavor: 'auto'`): `'integer'` on an Apple GPU, `'float'` elsewhere. Open the lab with
`?fp64=float` or `?fp64=integer` to force one for a measurement. The record's
`options.fp64Flavor` says which flavor ran, and the export's summary line shows it.
`?budget=<ms>` (1 to 60000) sets the device's time budget per submission (TypeShade
change 0074; 250 ms by default), to measure how the slice length affects the page's
responsiveness on a device: the record's `timing.budgetMs` says which ran, with
`timing.mainThread`.

To run the suspect kernel on the machine that lost the device, with a checkout of the
repository, Node 24 and `npm ci && npm run build && npx playwright install chromium`:

```powershell
$env:TYPECV_HARDWARE_GPU = '1'
node --test tests/typeshade/south-building-triangulation-webgpu.test.mjs
node --test --test-concurrency=1 tests/typeshade/webgpu.test.mjs
```

The first file runs the triangulation job of the lost run (`tests/fixtures/south-building-pair-triangulation.json`)
in three variants, each in its own browser, from the simplest to the full kernel. The
variant that hangs or loses the device names the loop. The second file is the general
WebGPU suite.

### Probe

The page `/lab/probe/` runs the same job in the browser, for a computer that cannot run
Node. A Web Worker loads the fixture from
`website/public/sample/probe/south-building-pair-triangulation.json` and runs
`triangulateTracksGpu` on it. It runs the three variants of the test file in the same
order, each on a new `ShadeDevice`. For each variant it records the outcome (completed or
lost), the elapsed time, the chunk rate of key `triangulateTracks`, the accepted tracks
and the dispatches. A completed variant also says whether its accepted count is within 8
of the fixture's `expectedStatus`.

When a variant loses the device, the probe bisects the tracks with that variant's
options, before it runs the remaining variants. The first variant that loses is the
simplest one that hangs, and `bisection.override` names its options. The bisection first
runs the chunk that was in flight at the loss. Then it runs the two halves of each set
that lost the device, each half on a new device. It continues down to single tracks, or
until it has run 40 steps. The result names each hanging track with its index and its
two pixels, and holds the fixture's cameras and poses, so the job can be replayed on the
CPU tier. A set that lost the device while neither of its halves did is recorded as a
minimal set.

On D3D12 a lost device takes the adapter with it for the page, and the next
`ShadeDevice.open` fails. The probe therefore keeps its state in `sessionStorage` after
every step (`website/src/lib/probe-plan.ts`): the steps so far, the plan of the
remaining steps, the consent and the log. When no device opens after a loss, the page
saves the state and reloads itself. After the reload it goes on from the saved plan
without a click, and the log says `Resumed after reload N`. Each reload compiles the
pipelines again. The probe takes at most 40 reloads, and at most 3 in a row that open no
device. Past either limit it aborts. The record's `reloads` field counts the reloads.
**Run probe** starts a new probe and discards a saved one.

The page takes the lab's switches `?bounded=1` and `?fp64=`. The switch
`?probe-fail=<track>` is a test hook: it simulates a loss for the chunk that holds that
track, and the next device then fails to open once, so that the page reloads. The
switch `?tracks=<first>-<last>` runs every step on those tracks only.

The page has the lab's consent checkbox. With the checkbox selected before the probe
ends, the result is sent to the collector once, as soon as the probe completes or
aborts. Selected after, the **Send result** button sends it. The page shows the
record's id when the collector has stored it. Nothing is sent without the checkbox.

To run the probe on the computer that lost the device:

1. Close the other tabs that use the GPU.
2. Open `https://cv.typeshade.dev/lab/probe/` in Chrome.
3. Select the checkbox "I agree to send the probe result to the TypeCV developers".
4. Select **Run probe**.
5. Leave the tab open. The page reloads itself after each device loss and goes on.
6. Wait until the line under **Send result** says `Sent:` and an id. Each device loss
   costs 2 to 7 s, and each reload about 10 s more.
7. Give the id to the person who asked for the probe.

If the line says `Not sent:`, select **Copy JSON** or **Download JSON**, and give the JSON
to the person who asked for the probe.

When the probe aborts, a line of the log starts with `Aborted`. The JSON holds every
step up to that point. Its field `resume` holds a query for the smallest set that lost
the device. To go on from that set:

1. Make sure that the result was sent, as in step 6.
2. Restart Chrome.
3. Open the page with the query of `resume`, for example `/lab/probe/?tracks=1360-1671`.
4. Do steps 3 to 7 again.

## Collector

`website/worker.mjs` runs in front of the static assets for `/api/*` only
(`wrangler.jsonc`: `run_worker_first`). Routes:

| Route                                | Behavior                                                                                                       |
| ------------------------------------ | -------------------------------------------------------------------------------------------------------------- |
| `POST /api/records/`                 | Same-origin only; body at most 256 KiB; validated; 30 per address per hour; stored in KV                       |
| `GET /api/records/`                  | `Authorization: Bearer <RECORDS_READ_TOKEN>`; lists `limit` (at most 100) records with metadata and a `cursor` |
| `GET /api/records/<id>/`             | Same token; returns one stored record                                                                          |
| `GET /api/config/`                   | No token; `{ "files": true }` when the result-file store is bound                                              |
| `PUT /api/records/<id>/files/<name>` | Same-origin only; stores one result file (see Result files)                                                    |
| `GET /api/records/<id>/files/`       | Same token as the list; lists the stored files of a record                                                     |
| `GET /api/records/<id>/files/<name>` | Same token; returns the bytes of one stored file                                                               |

The host answers every path without a trailing slash with a 301 to the path with one,
and a browser follows that redirect as a GET. The page therefore posts to
`/api/records/`, and the Worker routes both spellings alike.

Records live in the KV namespace bound as `RECORDS` under `record/<id>`, with
listing metadata (received time, GPU, input kind, frames, registered views, total ms,
device reuse, the fp64 flavor, the time budget and the longest chunk of any key,
benchmark name, the GPU time of the run in ms (`gpuMs`, v6, null without timestamps), and for a failed run the kernel or operation). `website/worker.test.mjs` tests the handler with an
in-memory KV.

The Worker validates a record with `format: "typecv-triangulation-probe-v1"` as a probe
record, and every other record as a run record. A probe record is stored under the
same prefix. Its listing metadata names it as a probe: `kind: 'probe'`, the received
time, the GPU, the fp64 flavor, the submission mode, the outcome of each variant, the
number of steps, the number of hanging tracks with the first 8 indices, and whether
the probe aborted.

### Result files

The lab can send the result files of a run with its record. The visitor opts in with the box
"Also send the result files" next to "Send record". The box is off by default, because the
data may come from the visitor's own photos or video. It is disabled when `GET /api/config/`
says `files: false`.

The files are the lab's own exports: `typecv-reconstruction.json`, `typecv-reconstruction.ply`,
`typecv-dense.ply` and `typecv-mesh.ply`. Only the files that exist for the run are sent.
They open in the lab with "Open result files".

1. The page computes the size and the SHA-256 of each file.
2. The page posts the record with a `files` list: `name`, `bytes`, `sha256`, and `skipped` (the
   reason) for a file that it does not upload. A file over 50 MB is skipped.
3. The page reads the record's `id` and sends `PUT /api/records/<id>/files/<name>` for each file
   that is not skipped, one after the other. The button shows the count.

The Worker stores a file in the private R2 bucket `typecv-records` (binding `RECORD_FILES`) under
`records/<id>/<name>`. It refuses a `PUT` in these cases:

- The origin is not the site (403).
- The name is not one of the four (400), or the record did not declare it or marked it
  `skipped` (400).
- The record does not exist (404) or is older than 10 minutes (403).
- The body is over 50 MB (413), or its size or SHA-256 differs from the record's entry (400).
- The file is stored already (409). A stored file is never replaced.
- The address sent more than 120 files in the hour (429). This count is apart from the
  30 records an hour.

`files` is an optional field of the run record, and the format name does not change. A file in
`files` without `skipped` that the bucket does not list was not stored: the upload failed or the
visitor left the page. The bucket is the record of what was stored.

With "Also send the result files" on, the lab does not send a record by itself at the end of
a run. The visitor presses "Send record" when the run is done.

## Setup

The deploy workflow (`.github/workflows/deploy.yml`) runs `scripts/ensure-run-records.mjs`
before `wrangler deploy`. The script lists the account's KV namespaces with the deploy
token, creates `typecv-records` when it is missing, writes the namespace id into
`wrangler.jsonc` for that deploy, and stores the `RECORDS_READ_TOKEN` Actions secret as
the Worker's secret when the repository has one. The deploy token
(`CLOUDFLARE_API_TOKEN`) needs **Workers KV Storage: Edit** for this; without it the
script removes the binding, warns, and the site deploys with the collector answering 503. `scripts/setup-run-records.ps1` does the same steps by hand from the owner's
machine. The read routes stay closed until a `RECORDS_READ_TOKEN` secret exists.

The result files need the private R2 bucket `typecv-records`. The deploy runs
`scripts/ensure-record-files-bucket.mjs`, which creates the bucket and keeps the `RECORD_FILES`
binding of `wrangler.jsonc`. The token needs **Workers R2 Storage: Edit**, as it does for
`typecv-samples`. Without it the script removes the binding, the file routes answer 503, and the
lab disables the box. The bucket is separate from the public `typecv-samples` and has no public
access.

## Reading the records

The **Export lab run records** workflow (`.github/workflows/lab-records.yml`, run by
hand from the Actions tab or by an agent through the API) reads the KV namespace with
the deploy's `CLOUDFLARE_API_TOKEN` (`scripts/export-run-records.mjs`, the Cloudflare
REST API) and needs no read token. It prints one summary line per record to the job
log, prints the failed runs in full (`print: all` prints every record), and uploads the
JSON files as the `lab-records` artifact for 14 days. The summary line of a v4 or later record
shows `longest chunk <ms>/<budget> ms`, and a v6 record with timestamps adds `gpu <ms> ms`. A v5 record with the dense MVS adds
`dense <status>, <points> points`, and `print: failures` also prints a run whose dense
MVS failed. A probe record has a line of its own: the GPU,
`probe`, the fp64 flavor, `bounded` or `unbounded`, the outcomes of the variants, the
hanging tracks, the number of steps, and `ABORTED` when the probe aborted. Under
`print: failures` the export also prints in full each probe record that found a hanging
track or aborted. The inputs are `limit` (newest
records, default 50) and `print` (`failures`, `all` or `none`).

With a `RECORDS_READ_TOKEN` secret, the collector's read routes answer too:

```sh
curl -H "Authorization: Bearer $TOKEN" 'https://cv.typeshade.dev/api/records/?limit=50'
curl -H "Authorization: Bearer $TOKEN" 'https://cv.typeshade.dev/api/records/<id>/'
curl -H "Authorization: Bearer $TOKEN" 'https://cv.typeshade.dev/api/records/<id>/files/'
curl -H "Authorization: Bearer $TOKEN" -o typecv-reconstruction.json \
  'https://cv.typeshade.dev/api/records/<id>/files/typecv-reconstruction.json'
```

Or on the Cloudflare dashboard: Workers & Pages → KV → `typecv-site-typecv-records`.
