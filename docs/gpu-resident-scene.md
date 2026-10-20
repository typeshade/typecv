# GPU-resident scene: design

Status: accepted, 2026-10-19 (the owner's go-ahead in the session). This document
describes a change to the mapper's execution contract. Each step below is its own
pull request with its own evidence.

## Problem

Every numerical mapper operation today rebuilds its device state from the host
model and reads the whole result back:

- `bundleAdjustGpu` packs poses, points, intrinsics, pixels and the index arrays
  of the entire model into fresh resident arrays, runs LM, reads the full result
  and rebuilds a host `SparseReconstruction`.
- `evaluateObservationsGpu` uploads 20 doubles per observation (pose, intrinsics
  and pixel repeated for each observation) to filter the model.
- `triangulateTracksGpu` uploads the same 20 doubles per observation per track.
- `estimatePosePnPGpu` uploads its correspondences; the host gathers them.
- `IncrementalSfmSession` copies the model after every registration, archives a
  second copy and posts a third to the UI.

The recorded eight-image software run (`docs/experiments/gpu/resident-pair-verification-software.json`)
uploads 22.8 MB of application data, submits 300 times and dispatches 24,918
kernels. The first 16 south-building images on the CPU path spent 13–17% of
their time in garbage collection before the persistent-landmark mapper, and the
32-image run peaks at 670 MB of host memory. The host model is the bottleneck of
both memory and transfer, and it grows with the map.

## Goal

Keep the map on the device for the whole reconstruction, as TypeNN keeps its
tensors: one owner (`ShadeDevice`), resident arrays with capacity headroom,
`fuse` scopes for the work of one mapper step, bounded pools, and explicit
reads. A registration appends. A filter compacts on the device. The host
reads small summaries and, at the end or on request, the model.

## Resident arrays

All arrays are resident on the device with a capacity and a count. Growth
reallocates at twice the capacity and copies on the device. Layout keeps the
current BA kernel layout (`bundle.shade.ts`) so those kernels bind the scene
directly.

| Array          | Element                                                                | Count    |
| -------------- | ---------------------------------------------------------------------- | -------- |
| `cameraPoses`  | f64 × 12: row-major R, then t                                          | C        |
| `cameraModels` | f64 × 6: fx, fy, cx, cy, k1, k2                                        | C        |
| `cameraFlags`  | u32: bit 0 variable in the current BA window                           | C        |
| `points`       | f64 × 3                                                                | P        |
| `pointFlags`   | u32: bit 0 alive                                                       | P        |
| `obsCamera`    | u32 camera index                                                       | M        |
| `obsPoint`     | u32 point index                                                        | M        |
| `obsPixel`     | f64 × 2                                                                | M        |
| `obsFlags`     | u32: bit 0 alive                                                       | M        |
| `pointOrder`   | u32 permutation of observations, point-major, and `pointStart` (P + 1) | M, P + 1 |
| `cameraOrder`  | u32 permutation, camera-major, and `cameraStart` (C + 1)               | M, C + 1 |

The host `Scene` (`src/reconstruction/scene.ts`, `HostScene`) holds the same arrays
as typed arrays, with two host-only columns the device never reads: `obsFeature`
(u32 feature id of each observation, which the snapshot needs) and
`pointObservations` (u32 alive observations of each point, kept current by the
observation writes, which the filter's second pass reads). `cameraIds` (view id)
and `pointTracks` (track id) name each camera and point; two maps give the
reverse lookups.

Observations are append-only in arrival order. The two CSR orders are derived
on the device by counting sort (histogram with atomics, 64-lane scan over the
block sums, scatter) when a step needs them. A compaction after filtering
rewrites the three observation arrays in place through the same scan and
scatter. Host-side state is small and index-based: view id to camera index,
track id to point index, the track graph and the per-view feature-to-track
index.

## Operations

1. **Register a view.** Write 18 doubles and one flag at index C; C += 1.
2. **Continue tracks.** The host sends the candidate list `(pointIndex, featureId,
pixel)` of the new view. One kernel projects each candidate with the resident
   pose and point, flags the ones within `maxErrorPx`, and the scan/scatter
   appends the accepted ones as observations.
3. **Triangulate.** The host sends the observation indices of each completed
   track (it knows them from the track graph) as a group list. The existing
   `triangulateTracks` kernel reads cameras and pixels from the scene instead of
   a 20-double record; accepted points are appended, and their observations
   appended with them.
4. **Filter.** One kernel projects every alive observation and clears the flag
   of each one beyond `maxErrorPx`; a second pass clears points left with fewer
   than two observations; scan and scatter compact the observation arrays. The
   host reads one u32 pair: observations removed, points removed.
5. **Register (PnP).** The host sends the new view's candidate `(pointIndex,
pixel)` list; a gather kernel builds the PnP problem from resident points. The
   current PnP plan runs unchanged on that resident problem.
6. **Bundle adjust.** The BA kernels bind the scene arrays and the derived
   orders directly. The LM result stays in the scene. The host reads the 26-double
   diagnostics block only.
7. **Read.** On request, download the alive cameras, points and observations and
   build a `SparseReconstruction`. The lab chooses its snapshot cadence; a
   snapshot is no longer a side effect of every registration.

## Mapper protocol

`incrementalSteps` keeps its control flow and gains an index-based request set.
A job carries indices and small typed arrays, never a `SparseReconstruction`:

| Request       | Payload                                                             | Result                                                  |
| ------------- | ------------------------------------------------------------------- | ------------------------------------------------------- |
| `load`        | a model (on resume only)                                            | — (camera i is view i, point j is landmark j)           |
| `register`    | view id, camera, pose                                               | camera index                                            |
| `continue`    | camera index, candidates (point index, feature id, pixel)           | per-candidate accepted flags                            |
| `triangulate` | groups of registered observations (camera index, feature id, pixel) | point index of each group, or -1                        |
| `filter`      | `maxErrorPx`                                                        | removed observations, removed points, point index remap |
| `pnp`         | camera, candidates (point index, pixel)                             | pose, inlier flags                                      |
| `bundle`      | iterations, gauge camera indices, variable camera indices           | diagnostics, whether the local subset was used          |
| `measure`     | `maxErrorPx`                                                        | mean reprojection error of the alive observations       |
| `snapshot`    | —                                                                   | `SparseReconstruction`                                  |

`src/reconstruction/scene-mapper.ts` holds the protocol (`SceneRequest`), the mapper
(`sceneSteps`) and the CPU driver (`HostSceneDriver`, `reconstructIncrementalScene`).
A triangulation group names observations that do not exist yet: the driver appends the
point and its inlier observations together. A filter compacts, so it returns the remap
of the point indices (a scan over the alive flags, which the device compaction derives
anyway); the host keeps the track-to-point map current from it. The host sends
pixels with each candidate until the keypoints are resident too.

The CPU driver implements the same protocol over a host `Scene` of typed arrays
with the same layout. The CPU path therefore sheds its per-call object cloning
too, and both drivers are compared on the same requests. Pair verification stays
as it is: it runs before the scene exists and already keeps one resident plan per
pair.

## Precision

Scene arrays hold emulated f64. Projection, filtering, triangulation and PnP
scoring are candidates for f32 after the precision split records their error
against the CPU oracle on the south-building harness. The BA normal equations
stay f64 until a mixed-precision residual certificate exists
(`docs/gpu-sfm-research.md`, Memory and precision contracts).

## Limits

| Quantity          | Bound                                                                                                                             |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| Observations      | `M × 16 bytes` (pixel) and `M × 4 bytes` (each index) within `maxStorageBufferBindingSize`; 256 MiB gives about 16 M observations |
| Cameras in one BA | 2..64; the PCG loop of both BA routes runs as chunks of the time budget                                                           |
| Capacity growth   | Reallocate ×2, copy on device; the host never holds the arrays                                                                    |
| Snapshot          | One full download; its size is the model                                                                                          |

## Evidence gates

Each step is accepted when:

- The south-building 16- and 32-image models (`docs/experiments/colmap-datasets/`)
  match the previous step's models within the BA tolerance (same registered
  views, landmark count within 1%, median rotation agreement unchanged to 0.01°).
- Upload bytes per registration grow with the new view's data, not with the map.
- Submissions per registration are bounded by a constant; dispatches grow only
  with the kernels that are O(M) by nature.
- Host peak memory on the 32-image run falls below the current 670 MB.

## Steps

1. Host `Scene` of typed arrays and the index-based request protocol; the CPU
   driver only. Same models as before on the harness. Done: `HostScene`,
   `SceneRequest`, `sceneSteps`, `HostSceneDriver`. `tests/scene-mapper.test.ts`
   holds the two mappers to the same poses, points and observations, bit for
   bit, on the synthetic scenes (global and local-subset bundles, shared focal,
   a refused registration with recovery, and a resumed model).
2. Resident `GpuScene`: register, continue, triangulate, filter with device
   compaction. `evaluateObservationsGpu` and `triangulateTracksGpu` keep their
   public form as thin wrappers over a temporary scene. Pair verification of
   independent pairs is batched in one submission with one progress read, each
   pair keeping its own seed, model budget and result order (the idea of the
   closed draft #15, redone on the scene driver).
   Every long kernel of the scene (pair RANSAC, the DLT and Jacobi sweeps of
   triangulation, the BA solver) runs under a time budget. WebGPU has one
   queue and no priority, so the protection against a driver timeout (the
   `DXGI_ERROR_DEVICE_HUNG` of the 2026-10-19 lab records: a submission that
   runs past about two seconds on Windows loses the device) is to keep every
   submission short. A kernel that iterates (RANSAC trials, sweeps, solver
   iterations) keeps its cursor and its done flag in a small storage buffer,
   the state machine, and each submission advances it by one chunk: a grid of
   blocks sized from the previous chunk's measured time (`timestamp-query`
   where the device has it, else the wall time around
   `queue.onSubmittedWorkDone()`), halved after a chunk that ran past the
   budget and doubled while chunks finish early. The host control loop waits
   on `onSubmittedWorkDone`, reads the state buffer once per chunk (a few
   bytes), and submits the next chunk or stops; nothing else is read back
   until the kernel is done. The budget is a scene option (default 250 ms per
   submission), and the lab records the chunk count and the longest chunk of
   each chunk key in `timing.chunks` beside the kernel names of `timing.kernels`.
   The CPU driver runs the same state machine in one chunk, so the two
   mappers stay bit for bit.
   Delivered so far: `ShadeDevice.open({ timeBudgetMs })`, the chunk steps of
   `planSteps` with their learned rates, and pair model generation as a state
   machine of short dispatches whose scoring is timed (`docs/gpu-sfm.md`, "The
   time budget"). The budget, the fences, the timing and the rates moved into
   the compiler's device (TypeShade change 0074, pin `41a140a`):
   `createDevice({ timeBudgetMs })`, `device.fuse(body, { chunk })`,
   `device.chunkSize()` and `device.chunkRates`, and `ShadeDevice` passes them
   through. Its longest kernel, the Jacobi SVD of each model, is gone: a
   fixed-cost solve of the triangular factor took its place
   (`docs/gpu-initialization.md`, "Numerical path"), which is the other half of
   the remedy: a kernel that does not iterate to convergence needs no chunking.
   The search stops where a sequential RANSAC would, so the batch is a
   performance choice, and the budget sizes it from the scoring rate.
   The BA's PCG loop runs as chunks too, on both routes (key `bundleCg`, units
   of `M + 6C` rows per iteration on the fixed-focal route and `M + 6C + 1` on
   the shared-focal route): the one-dispatch persistent kernels
   `bundlePersistentCg` of `bundle.shade.ts` and of `joint-focal.shade.ts` are
   removed, and the host reads the two activity flags after each chunk on every
   device (`docs/gpu-sfm.md`, "The BA's PCG loop"). The shared-focal route ran
   the final bundle that lost the device on the owner's iPhone in lab record
   `20261019225221735-85ed0ae6`. Its opt-in Schur cache runs its loop as chunks
   of its own key, `bundleCachedCg`, in place of `bundleCachedPersistentCg`.
   The pair search reads its state once per four batches, not once per
   chunk: the GPU writes each batch's workgroup counts and the batch's
   entries dispatch indirectly (TypeShade change 0075, pin `d70b539`), so a
   batch past the adaptive budget dispatches nothing, and only a scoring the
   host knows to run is timed.
   The budget is a device option, not yet a scene option; the chunk count and
   the longest chunk of each chunk key are in the plan report and in the lab
   record (`timing.chunks` and `timing.budgetMs`, `docs/lab-run-records.md`); the
   chunk grows at most fourfold and shrinks to what fits, rather than halving
   and doubling. A device times the first eight chunks of each key and runs
   the later ones untimed, with no fence, sized by the settled rate; before
   the first timed chunk of a key it makes the key's pipelines in a warm-up
   plan of zero workgroups, so that no timed chunk includes their creation
   (`docs/gpu-sfm.md`, "The time budget"). Triangulation runs as chunks of contiguous tracks (key
   `triangulateTracks`, one unit a track, at least one workgroup of 16 tracks):
   each track's hypotheses, sweeps and point LM stay in one invocation, and the
   chunk boundaries change no result (`docs/gpu-sfm.md`, "Triangulation").
   What remains is PnP. Its kernels loop at a fixed cost, not to convergence:
   `selectPnP` walks the `4 × trials` models of a batch over their score
   blocks in one invocation, a cost the batch fixes, and the pose refinement
   dispatches each LM step on its own. They run under
   `boundedSubmissions` alone.
   The same measurements predict the time a run takes. Each request carries
   its work in units (trials, sweeps, iterations, pairs, tracks), and the
   scheduler keeps, per kernel and per device, the ms one unit took in the
   chunks so far. While a kernel runs, the remaining time is the remaining
   units times that rate, updated after each chunk; a kernel that stops
   early (RANSAC) reports its bound, the maximum trials, and narrows it as
   the trials go. Before a run, the units are counted from the input (frames,
   features, pairs, tracks, the iteration caps) and the rate comes from this
   device's earlier records, or, with none, from the collector's table of
   rates by GPU vendor and architecture, replaced by this device's own rate
   after the first chunks. Pipeline creation is measured apart from compute,
   since it dominates a small run. A progress event carries `etaMs` for the
   lab's bar, and a run record stores `predictedMs` beside `totalMs`, so the
   collector shows the error of the prediction by device class. The counts
   an adaptive step decides (registration attempts, BA iterations to
   convergence) are predicted as a mean and a cap, never as one number.
3. BA binds the scene; diagnostics-only readback.
4. PnP gathers from the scene.
5. Lab snapshot cadence and session archive from device reads.

The serial single-invocation kernels (`initializeBundleMetrics`,
`acceptBundleCandidate`, `compactFeatures`, `compactMutualBrief`, the two-view
sums) are replaced by 64-lane partial reductions and the shared scan before step
2, since the compaction and the O(M) sums of this design depend on them.
