# GPU working budget: bundle adjustment and triangulation run within any device budget

Status: proposed. Issue: typeshade/typecv#160 (owner rule, 2026-10-21: size changes speed, not
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

`planBundleResidency` (`src/gpu/bundle-residency.ts`, new) takes `C, P, M`, the longest track, the
budget (`workingBytes` less the arrays the device holds now), `bindingBytes` and
`maxKernelElements`. It returns `resident` or `streamed`, and for `streamed` the window sizes
`obsCapacity` and `pointCapacity`.

- Always resident (the floor): `scene`, `trial`, `index`, `cameraScratch`, `vectors`, `state`,
  `partials`, `output` and the camera part of `blocks` (damped camera blocks, inverse
  preconditioner, reduced RHS: `78C`). The floor is 1,496 C + 204 P + 28.4 M bytes less the window.
- Window: the observation part of `blocks` (J, residual, weight: 24 per observation; the cross
  blocks: 18) and the point part (13 per point), for `obsCapacity` observations and `pointCapacity`
  points. Window bytes = `336 x obsCapacity + 104 x pointCapacity`. The shared-focal route adds 24
  per observation to the window (`cross` 21 instead of 18, and the focal Jacobian 16).
- `resident` when the floor plus the window for all `M` and `P` fits the budget and the binding
  limit. The layout and the dispatches are those of today, bit for bit.
- `streamed` otherwise. `obsCapacity` is the largest count whose window fits the budget less the
  floor, at most the dispatch limit `maxKernelElements`, at least the longest track. `pointCapacity`
  = `ceil(obsCapacity x P / M)`.

The plan also checks the floor against the budget and the binding limit (3.6).

### 3.4 The streamed form is exact

The reduced camera system is `6C` (fixed focal) or `6C + 1` (shared focal) and stays resident. The
observations are ordered by point, so a contiguous range of points owns a contiguous range of
observations. The points are cut into windows (point chunks) by `obsCapacity` and `pointCapacity`.

Each LM iteration runs sweeps over the windows, in ascending order:

1. Assembly sweep. For each window: `linearizeBundle` (the window's observations),
   `bundlePointBlocks` (its points), `checkBundlePoints` (a point whose 3 x 3 system is singular
   sets a flag), `bundleCameraRows` (continues the camera rows). `bundleCameraRows` of the last
   window adds the damping and `sRow += aRow`. Then `invertBundleCameraColumns` and
   `finalizeBundleCameraInverse` as today.
2. PCG sweeps. One per PCG iteration. For each window: `linearizeBundle`, `bundlePointBlocks`
   (recomputed: the window holds only the window, and the values are those of the assembly sweep,
   because the inputs are the same), `bundleSchurPoint`, `bundleSchurCamera` (continues the camera
   rows). Then `advanceBundleCg` once.
3. Update sweep. For each window: `linearizeBundle`, `bundlePointBlocks`, `updateBundleCandidate`
   (the window's points; the first window also updates the cameras).
4. Cost sweeps. `bundleCandidatePartials` over windows of observations aligned to 64, then
   `acceptBundleCandidate`. The initial metrics sweep (`linearizeBundle`, `bundleMetricPartials`)
   uses the same aligned windows.

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
- The scalar and the PCG steps (`advanceBundleCg`) are unchanged.

For the shared-focal route the scalar focal row adds sums over all observations
(`bundleSchurCameraRow` of the focal row, `bundleFocalBlock`, `bundleFocalCameraRows`). They
continue over the windows in the same way. `bundleFocalBlock` first sums the observation terms,
then damps, then subtracts the point terms. It needs a second assembly sweep for the point terms
(`linearizeBundle`, `bundlePointBlocks`, `accumulateBundleFocalPoints`).

What changes: the time. A PCG iteration recomputes the linearization of every observation. The
streamed run is slower than the resident one by the cost of one `linearizeBundle` and one
`bundlePointBlocks` per PCG iteration (Section 5 measures it). Nothing else changes.

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
  limit. With a 2 GiB adapter (budget 1 GiB) the floor holds about 36 million observations.
- `6 C` must not exceed `maxKernelElements` (4,194,240 at the WebGPU default): 699,040 cameras.
- A single track longer than the window (the window is at least the longest track).

These break the owner's rule when they happen. Their next action is the block-partitioned BA of
3.5, and a split of `scene` and `index` into the arrays that stay under the binding limit.

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

1. `tests/typeshade/bundle-streamed.test.mjs`: on the CPU tier, a BA problem of 12 cameras run
   all-resident and with budgets that force windows of 1, 7 and 64 observations (and one with a
   window of one point): the poses, the points, the diagnostics and the CG counts are bit for bit
   those of the resident run. Fixed focal and shared focal. With a local window (`variableViewIds`).
2. The same bundle problem with a tiny `workingBytes` that used to throw now completes.
   `tests/typeshade/bundle-admission.test.mjs` changes: the "rejects the next larger scene" test
   becomes "streams the next larger scene".
3. `tests/typeshade/triangulation-streamed.test.mjs`: the tracks of a scene run in one chunk and
   in chunks of 1 track and of a budget: the results are bit for bit equal.
4. `tests/bundle-residency.test.ts`: the planner (modes, window sizes, the floor, the limits).
5. WebGPU (SwiftShader): the BA of the south-building-24 sample at the default budget and at a
   budget that streams (`tests/typeshade/bundle-streamed-webgpu.test.mjs`).
6. `website/` : the worker budget from the adapter limits (`workingBytesFromAdapter`) and that a
   small budget is honored.

## 6. Measurements

(Filled when the implementation is measured.)

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
