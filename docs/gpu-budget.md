# GPU working budget: bundle adjustment and triangulation run within any device budget

Status: accepted, 2026-10-21, under the owner's approval of every merge of this session and of the rule of issue #160. Implemented in pull request #167. The open items of sections 3.6, 7 and 8 stay open under #160.
Issue: typeshade/typecv#160 (owner rule, 2026-10-21: size changes speed, not
success). The dense MVS did this in Amendment 3 of `docs/dense-mvs.md`. This record does it for
the bundle adjustment (global and local, fixed and shared focal) and for the triangulation. It
lists the limits of pair estimation and float matching, which a later pull request removes.

## 1. The failure

Run record `d40f8950` (multi-video sample set `chimney-rock-b`, 839 frames at 640 x 360, NVIDIA
Blackwell, adapter `maxBufferSize` 2 GiB, site `f2473c9`) failed with
`RangeError: GPU BA exceeds working/storage/dispatch budget`. The error came from
`src/gpu/bundle.ts` (`runIncrementalGpu` in `reconstructMultiVideoGpu`), stage
`prepareTriangulation (start)`, after the partial maps had 53,323 points. Two causes:

1. The working budget is fixed: `64 MiB x (edge / 640)^2` (`website/src/workers/multi-video.ts`,
   `website/src/workers/sfm.worker.ts`, the default of `ShadeDevice.open`). The adapter allows 2 GiB.
2. `bundle.ts`, `joint-focal.ts` and `triangulation.ts` throw when the whole problem does not
   fit. They do not split it.

## 2. The budget comes from the adapter

- `ShadeDeviceOptions.adapterLimits` (Amendment 3 of `docs/dense-mvs.md`) requests the adapter's
  `maxBufferSize` and `maxStorageBufferBindingSize`. `ShadeDevice.bindingBytes` and
  `ShadeDevice.bufferBytes` then read them. Both workers set it. Before this record only the
  dense run did.
- The working budget of a worker is `workingBytesFromAdapter(scaleBytes, maxBufferSize)` =
  `max(scaleBytes, floor(maxBufferSize / 2))`, where `scaleBytes` is the old
  `64 MiB x (edge / 640)^2` (octave -1: times 4). It is the same rule as `denseWorkingBytes`.
  The fixed scale stays as a lower bound for an adapter that reports no limit and for the CPU
  tier. Half of `maxBufferSize` leaves the other half for the textures, staging buffers and the
  pool that the budget does not count.
- A test keeps that a small `workingBytes` is still honored: the stages split to fit it.

## 3. Bundle adjustment

### 3.1 The arrays, in bytes

`C` cameras, `P` points, `M` observations. Fixed focal (`src/gpu/bundle.ts`), f64 = 8 bytes, u32 =
4 bytes. `ceil(M/64)` is the count of 64-observation blocks.

| Array           | Elements                | Bytes | Per what                             |
| --------------- | ----------------------- | ----- | ------------------------------------ |
| `scene`         | `18C + 3P + 2M`         | `8 x` | 144 / camera, 24 / point, 16 / obs   |
| `trial`         | `12C + 3P`              | `8 x` | 96 / camera, 24 / point              |
| `index`         | `3M + P + 2C + 2` (u32) | `4 x` | 12 / obs, 4 / point, 8 / camera      |
| `blocks`        | `42M + 13P + 78C`       | `8 x` | 336 / obs, 104 / point, 624 / camera |
| `cameraScratch` | `36C` (WebGPU)          | `8 x` | 288 / camera                         |
| `vectors`       | `30C + 3P`              | `8 x` | 240 / camera, 24 / point             |
| `state`         | 26                      | 208   |                                      |
| `partials`      | `3 ceil(M/64)`          | `8 x` | 0.375 / obs                          |
| `output`        | `12C + 3P + 26`         | `8 x` | 96 / camera, 24 / point              |

Total: 364.4 bytes per observation, 204 per point and 1,496 per camera. The shared-focal route
(`src/gpu/joint-focal.ts`) adds 3M to `blocks` (the focal cross block, 21 per observation), `2M +
6C + 6` to `focal` (the focal Jacobian of each observation, the direct camera/focal blocks), and
the `schur` and `schurCross` arrays of the opt-in Schur cache (at most 16 cameras: these stay
all-resident and the cache is off when the problem streams).

The first array to break is `blocks`: `42 M x 8` bytes must fit the binding limit (128 MiB without
adapter limits, 134,217,728 bytes) and the budget.

### 3.2 The case of run record d40f8950 (estimate)

The record does not give `M`. Facts: `C` = 839 (frames), `P` = 53,323. Assumption: `M` = 8 P =
426,584 (a track of 8 views). Marked as an estimate; the table gives `M` = 4 P and 12 P too.

| `M`            | `blocks` | all arrays | against 64 MiB (67.1 MB) | all-resident floor (arrays except the observation/point window of `blocks`) |
| -------------- | -------- | ---------- | ------------------------ | --------------------------------------------------------------------------- |
| 213,292 (4 P)  | 77.7 MB  | 89.9 MB    | over by 22.8 MB          | 12.6 MB                                                                     |
| 426,584 (8 P)  | 149.4 MB | 167.6 MB   | over by 100.5 MB         | 18.7 MB                                                                     |
| 639,876 (12 P) | 221.1 MB | 245.3 MB   | over by 178.2 MB         | 24.7 MB                                                                     |

At 8 P the all-resident BA needs 167.6 MB (159.8 MiB). It does not fit the fixed 64 MiB. Its
`blocks` array (149.4 MB) does not fit the 128 MiB default binding limit either. With the
adapter's 2 GiB, the budget is 1 GiB and the binding limit is 2 GiB: the whole problem fits
all-resident with a factor of 6. The map grows with the video and has no bound, so the rule needs
a streamed path whatever the budget.

### 3.3 The residency plan

`planBundleResidency` (`src/gpu/budget-plan.ts`) takes `P`, `M`, the offsets of the points into
the observations, the longest track, the budget (`workingBytes` less the arrays the device holds
now), `bindingBytes`, `maxKernelElements` and the byte counts of the route's arrays. It returns
`resident` or `streamed`, the window sizes `obsCapacity` and `pointCapacity`, the windows
themselves and the bytes of the plan. Both routes call it (`bundle.ts`, `joint-focal.ts`).

- Always resident (the floor): `scene`, `trial`, `index`, `cameraScratch`, `vectors`, `state`,
  `partials`, `output` and the camera part of `blocks` (damped camera blocks, inverse
  preconditioner, reduced RHS: `78C`). Shared focal adds the camera/focal part of `focal`
  (`6C + 6`). The floor is 1,496 C + 204 P + 28.4 M bytes less the window.
- Window: the observation part of `blocks` (J, residual, weight: 24 per observation; the cross
  blocks: 18) and the point part (13 per point), for `obsCapacity` observations and `pointCapacity`
  points. Window bytes = `336 x obsCapacity + 104 x pointCapacity`. Shared focal: the cross
  blocks are 21 and the focal Jacobian adds 16, so `376` per observation.
- `resident` when every array fits the budget, the binding limit and the dispatch limit. The
  layout and the dispatches are those of the code before this record, in the same order.
- `streamed` otherwise. `obsCapacity` is the largest count whose window fits the budget less the
  floor (by bisection), at most `maxKernelElements`, at least the longest track.
  `pointCapacity` = `ceil(obsCapacity x P / M)`. The windows are cut greedily: whole points, until
  the observations or the points of the window reach the capacity.
- The cost sweeps (the first cost and the cost of each candidate) read the scene, not the window
  (`metricLane` computes what `linearizeBundle` writes, with the same expressions), so they are
  cut only by the dispatch limit, at multiples of 64 observations.
- The opt-in Schur cache of the shared-focal route (at most 16 cameras) needs the whole problem
  resident. A run that streams does without it.

The plan reports `fits: false` when the floor and the least window (one track) exceed the budget
(3.6).

### 3.4 The streamed form is exact

The reduced camera system is `6C` (fixed focal) or `6C + 1` (shared focal) and stays resident. The
observations are ordered by point, so a contiguous range of points owns a contiguous range of
observations. The points are cut into windows by `obsCapacity` and `pointCapacity`. Each kernel
takes the window in its control (`obsBase`, `obsEnd`, `pointBase`, `pointEnd`, `obsCapacity`,
`pointCapacity`, `first`, `last`, `sweep`, `streamed`, `base`). A resident run is one window that
holds everything.

Each LM iteration runs sweeps over the windows, in ascending order:

1. Assembly sweep. For each window: `linearizeBundle` (its observations), `bundlePointBlocks` (its
   points), `checkBundlePoints` (a point whose 3 x 3 system is singular sets the failed-solve flag `state[20]`),
   `bundleCameraRows` (continues the camera rows; the last window adds the damping and
   `sRow += aRow`). Then `invertBundleCameraColumns` and `finalizeBundleCameraInverse` as today.
   Shared focal: `bundleFocalCameraRows` and `bundleFocalObservations` continue per window, then
   `dampBundleFocal` once, then a second pass over the windows (`linearizeBundle`,
   `bundlePointBlocks`, `bundleFocalPoints`) subtracts the point terms.
2. PCG sweeps. One per PCG iteration. For each window: `linearizeBundle`, `bundlePointBlocks`,
   `bundleSchurPoint`, `bundleSchurCamera` (continues the camera rows, and on the shared-focal
   route the focal row). Then `advanceBundleCg` once. The kernels of a window do nothing once PCG
   ended (`sweep` 1).
3. Update sweep. For each window: `linearizeBundle`, `bundlePointBlocks`, `updateBundleCandidate`
   (its points; the first window also moves the cameras and the focal scale).
4. Cost sweeps. `bundleCandidatePartials` over windows of the dispatch limit, then
   `acceptBundleCandidate`. The first cost uses `bundleMetricPartials` the same way.

A dispatch over more elements than the dispatch limit (`commitBundleCandidate`,
`normalizeBundleCandidate`, `packBundleResult`) runs in pieces with `base` set.

Why the result equals the resident run bit for bit:

- A camera row is a sum over the camera's observations in ascending observation order, kept in the
  resident accumulators (`blocks` A row, `cameraScratch` S row, RHS, `vectors` for S p). A window
  continues the sum where the previous window stopped. The sequence of additions is the sequence of
  the resident kernel. The camera's observation list is ascending, so each window reads a
  contiguous slice of it (found by binary search).
- Every per-observation and per-point value is a function of the scene, the damping and the
  window-independent inputs only. A recomputed value equals the first one.
- The sums over 64-observation blocks (`partials`) use the same blocks: the cost windows start at
  multiples of 64.
- The focal sums (`bundleFocalObservations`, `bundleFocalPoints`, the focal row of the Schur
  product) keep their running value in `focal[6C + 5]`, `focal[6C + 2]` and `vectors`, and add the
  terms in the order of the resident kernel.
- The scalar steps of PCG (`advanceBundleCg`) are unchanged.

The CPU tier's resident run uses the per-camera kernel `bundleCameraBlocks`. A streamed run, on
every tier, uses the per-row kernel `bundleCameraRows` of the WebGPU route. The two give the same
bits (the tests compare a resident CPU run with a streamed CPU run).

What changes: the time. A PCG iteration recomputes the linearization of every observation and the
point blocks. Section 6 measures it: 2.0 to 2.3 times the time of the resident run at 3,000 points on
SwiftShader. Nothing else changes.

### 3.5 Block-partitioned BA (the fallback, not in this record's code)

If the floor itself does not fit, the problem splits by cameras: local BA windows over the
covisibility graph, each with its cameras variable and the neighbors fixed. This is a different
optimizer: block coordinate descent over camera windows. It converges to a stationary point of the
same cost, not in the same iterates. The verification would be the final cost and the reprojection
error against the exact run on a problem that fits both ways. This record does not implement it
(open item 1).

### 3.6 What still refuses (open items)

- The floor. `1,496 C + 204 P + 28.4 M` bytes must fit the working budget, and the single arrays
  (`scene` is `8 x (18 C + 3 P + 2 M)`, `index` is `4 x (3 M + P + 2 C + 2)`) must fit the binding
  limit. With a 2 GiB adapter (budget 1 GiB) and 8 views to a point, the floor holds about 20
  million observations. The run throws `GPU BA needs N bytes for the arrays that stay resident ...`.
- `6 C` (`6 C + 1` with shared focal) must not exceed `maxKernelElements` (4,194,240 at the WebGPU
  default): 699,040 cameras. The run throws a `RangeError` that names the dispatch limit.
- A budget below the floor and one track throws the same message as the first item.

These break the owner's rule when they happen. Their next action is the block-partitioned BA of
3.5, and a split of `scene` and `index` into arrays that stay under the binding limit.

## 4. Triangulation

`triangulateTracksGpu` (`src/gpu/triangulation.ts`) keeps, per observation, `raw` (20 f64 = 160
bytes), `geometry` (6 f64 = 48) and `flags` (4), 212 bytes. Per track it keeps `offsets` (4),
`result` (5 f64 = 40) and `statuses` (4), 48 bytes. The tracks do not interact, so a chunk is a
contiguous range of tracks. The plan (`planTriangulationChunks`) cuts the tracks into chunks of at
most `chunkBytes` (the budget less the arrays the device holds, less the resident total of the
chunk), of at most `bindingBytes / 160` observations and `maxKernelElements / 4` tracks. A chunk
runs `prepareTriangulation` and `triangulateTracks` on its own arrays and reads its results back.
The chunk's arrays are released before the next chunk. The existing chunking by time stays inside
a chunk. A track longer than a chunk is not possible (a track has at most 64 observations). The
results equal one dispatch over all tracks bit for bit, because a track reads only its own
observations.

## 5. Tests

1. `tests/budget-plan.test.ts`: the planner (modes, windows tile the points, window sizes against
   the budget, the 128 MiB binding limit, the dispatch limit, the d40f8950 case) and the
   triangulation chunks.
2. `tests/typeshade/bundle-streamed.test.mjs` (CPU tier): a BA problem run all-resident and with
   budgets that force windows of 24 and 300 observations gives the poses, the points, the
   diagnostics and the CG counts bit for bit. Fixed and shared focal. A local window
   (`variableViewIds`) and a shifted gauge. An outlier. A budget of the floor and 64 observations
   (a fifth of the resident bytes) completes. A budget below the floor throws.
3. `tests/typeshade/bundle-admission.test.mjs`: the tests that expected a `RangeError` at the
   working budget, the binding limit and the dispatch limit now expect a completed run with the
   resident result. The mocks keep the integer index bounds.
4. `tests/typeshade/triangulation-streamed.test.mjs` (CPU tier): chunks of a few tracks, of 7
   observations and of one track give the digest of one chunk. A batch ten times the budget no
   longer throws.
5. `tests/typeshade/bundle-streamed-webgpu.test.mjs` (SwiftShader): the BA streamed and resident,
   fixed and shared focal, and the triangulation in chunks, equal bit for bit.
6. `website/working-budget.test.mjs`: the budget from the adapter limits, and that both workers use
   it.
7. The tests of the BA that exist (`bundle-checkpoint`, `schur-cache`, `focal`, `solver-checkpoint`,
   `bundle-camera-webgpu`, `focal-webgpu`, `schur-cache-webgpu`) pass unchanged: a resident run is
   what it was.

## 6. Measurements

`scripts/measure-ba-budget.mjs`: the south-building-24 sample (24 cameras, COLMAP's poses, the
dataset's SIFT matches triangulated, the 3,000 landmarks with the most observations: 12,706
observations), poses and points perturbed by 0.004, 4 LM iterations, headless Chromium with
SwiftShader (no physical GPU), `ShadeDevice.open({ workingBytes })`. The records are
`docs/experiments/gpu-budget/south-building-24-ba-fixed.json` and `-shared.json`. The peak is the
largest `liveBytes - pooledBytes` sampled at each kernel.

| Route        | `workingBytes` | Mode     | Windows per LM iteration | Time (ms) | Peak active bytes | Dispatches | Result (SHA-256) |
| ------------ | -------------- | -------- | ------------------------ | --------- | ----------------- | ---------- | ---------------- |
| fixed focal  | 512 MiB        | resident | 1                        | 6,741     | 5,278,096         | 1,048      | 2372666c4c8f     |
| fixed focal  | 2,500 KiB      | streamed | 3                        | 14,995    | 2,559,872         | 4,244      | 2372666c4c8f     |
| fixed focal  | 1,400 KiB      | streamed | 7                        | 13,614    | 1,433,560         | 9,402      | 2372666c4c8f     |
| shared focal | 512 MiB        | resident | 1                        | 8,938     | 5,787,584         | 1,107      | 051e2679a137     |
| shared focal | 2,500 KiB      | streamed | 4                        | 17,829    | 2,559,968         | 5,810      | 051e2679a137     |
| shared focal | 1,400 KiB      | streamed | 8                        | 20,633    | 1,433,600         | 11,335     | 051e2679a137     |

Reading (the observed results, then the inference):

- Observed: the result hash (poses, points, diagnostics, CG iterations) is the same in every run
  of a route, resident and streamed. The 304 CG iterations (317 with shared focal) are the same.
  The RMSE goes from 1.45 px to 0.158 px in every run.
- Observed: the peak of active bytes follows the budget (2.56 MB and 1.43 MB against 5.28 MB) and
  stays within it: 2,559,872 B within 2,560,000 B, and 1,433,600 B within 1,433,600 B.
- Observed: streaming costs 2.0 to 2.3 times the time on SwiftShader. A smaller window (7 windows against 3) costs no more in the fixed-focal run (13.6 s against 15.0 s, within the noise of a shared host). The recomputation, not the number of dispatches, is the cost.
- Inference: on a physical GPU the recomputation of `linearizeBundle` and `bundlePointBlocks` per
  PCG iteration is also the extra cost. This is not measured here (no physical GPU on the host).

CPU tier (`tests/typeshade/bundle-streamed.test.mjs`, 8 cameras, 120 points, 960 observations, 6
iterations): resident 4.2 s, streamed with a window of 500 observations 4.3 s, of 100 observations
5.0 s, of 64 observations 7.6 s. The results equal bit for bit.

### 6.1 The d40f8950 case, in bytes

`C` = 839, `P` = 53,323, `M` = 8 P = 426,584 (estimate, 3.2). All arrays: 167.6 MB (fixed focal).
The fixed 64 MiB (67.1 MB) budget refuses it: that is the failure. Output of `planBundleResidency`
for this problem:

| Budget  | Binding limit | Mode     | Windows | Window (obs / points) | Bytes of the plan |
| ------- | ------------- | -------- | ------- | --------------------- | ----------------- |
| 1 GiB   | 2 GiB         | resident | 1       | 426,584 / 53,323      | 167,570,036       |
| 160 MiB | 128 MiB       | streamed | 2       | 383,077 / 47,885      | 152,386,132       |
| 64 MiB  | 128 MiB       | streamed | 4       | 138,729 / 17,342      | 67,108,732        |
| 24 MiB  | 128 MiB       | streamed | 24      | 18,548 / 2,319        | 25,165,524        |

- With the adapter's limits (worker budget 1 GiB, binding limit 2 GiB) the problem is resident and
  takes 16 % of the budget.
- With the old fixed 64 MiB budget the same run streams in 4 windows and completes.
- The `blocks` array (149.4 MB) does not fit the default 128 MiB binding limit, so a device without
  the adapter's limits streams even with a budget of 160 MiB.

## 7. Limits of the stages left for a later pull request

These stages throw at a limit and break the rule. This record does not change them.

| Site                                                                                       | Limit                                                        | Direction (issue #160)                |
| ------------------------------------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------- |
| `src/gpu/pair-estimation.ts` (admission, 183)                                              | `working/storage/dispatch budget` for the hypothesis batches | split the batches of pairs and models |
| `src/gpu/pair-estimation.ts` (undistortion, 947)                                           | the same, for the undistorted image                          | tile the image                        |
| `src/gpu/float-matching.ts` (137)                                                          | the same, for the descriptor sets                            | match in blocks of the second set     |
| `src/gpu/features.ts`, `tiled-features.ts`, `orb.ts`, `global-descriptor-gpu.ts`, `p3p.ts` | see the table of issue #160                                  | issue #160, items 4 and 5             |

## 8. Open items

1. The block-partitioned BA for a floor that does not fit (3.5, 3.6).
2. `scene` and `index` are single bindings. A split removes the binding bound on `M`.
3. The Schur cache of the shared-focal route stays all-resident (at most 16 cameras).
4. The estimate of `M` for run record d40f8950 (3.2). The next lab run records `M` in its record.
