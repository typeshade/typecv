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

| Request       | Payload                                       | Result                               |
| ------------- | --------------------------------------------- | ------------------------------------ |
| `register`    | pose, camera                                  | camera index                         |
| `continue`    | camera index, candidates (point index, pixel) | accepted count, per-candidate flags  |
| `triangulate` | groups of observation indices                 | per-group status, point index        |
| `filter`      | `maxErrorPx`                                  | removed observations, removed points |
| `pnp`         | camera, candidates (point index, pixel)       | pose, inlier flags                   |
| `bundle`      | variable camera indices, iterations, gauge    | diagnostics                          |
| `snapshot`    | —                                             | `SparseReconstruction`               |

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
| Cameras in one BA | The persistent-CG route keeps `M × cgIterations ≤ 262,144`; larger problems use the checkpointed route or a local window          |
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
   driver only. Same models as before on the harness. The container is done
   (`HostScene`: append, flag, compact, the two orders, snapshot); the protocol
   and the driver follow.
2. Resident `GpuScene`: register, continue, triangulate, filter with device
   compaction. `evaluateObservationsGpu` and `triangulateTracksGpu` keep their
   public form as thin wrappers over a temporary scene.
3. BA binds the scene; diagnostics-only readback.
4. PnP gathers from the scene.
5. Lab snapshot cadence and session archive from device reads.

The serial single-invocation kernels (`initializeBundleMetrics`,
`acceptBundleCandidate`, `compactFeatures`, `compactMutualBrief`, the two-view
sums) are replaced by 64-lane partial reductions and the shared scan before step
2, since the compaction and the O(M) sums of this design depend on them.
