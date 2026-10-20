# Pyramidal Lucas–Kanade (KLT) feature tracking for the lab's video input: design

Status: accepted, 2026-10-20, under the owner's approval of all merges for that session. No step
is implemented. Decision by default: the design does not wait on `feat/phase-gate`. Each step
works with the keyframe gate that is on `main` when the step starts.
Each step in "Steps" below is its own pull request with its own evidence.

This document is written for an engineer or an agent who has not read the conversation that
led to it. It names files, symbols, buffer layouts in bytes, kernels, tests and the numbers
that accept each step. A name marked _(new)_ does not exist yet, and so does every name in a
list of proposed changes. Every other path, symbol and option of this repository named here
exists on `main` at `281dbb5`.

## 1. Problem and goal

The lab (`/lab/`) takes a video in two ways. The page decodes it at a fixed rate
(`VIDEO_SAMPLE_FPS` = 4, `website/src/lib/video-sampling.ts`), and the worker
(`website/src/workers/sfm.worker.ts`, `processFrame`) handles each frame like this:

1. It extracts GPU SIFT features of the frame (`extractSiftGpuResident`).
2. It matches the frame against every earlier keyframe that is still active
   (`matchFloatDescriptorsGpu`, ratio 0.8, mutual), the last keyframe first.
3. If the median motion against the last keyframe is below `keyframeMinMotionPx` (8 px at
   640 px) and the gap is below `keyframeMaxGapSeconds` (1 s), the frame is dropped. Its
   features and its matching are discarded.
4. Otherwise it runs guided matching (`guidedMatches`, `nearestFloatGuided`) and hands the
   pairs to the mapper session.

Another branch, `feat/phase-gate` (not on `main`), adds a phase-correlation gate that decides
the keyframe before the SIFT work. This design does not depend on that branch. Section 3.6
says how the two combine.

Three costs and one gap follow from this.

- **Every sampled frame pays for SIFT.** A dropped frame has paid for features and for
  matching against the last keyframe. The video record `20261020121308655-a6fb6f3b`
  (RTX 2080, 566 decoded frames of a 4K source, 27 keyframes, 1280 × 720) spent 45.0 s on
  features.
- **Matching cost grows with the number of keyframes.** Each new frame matches all earlier
  active keyframes. The same record spent 5.44 s on plain matching and 72.5 s on guided
  matching. Guided matching costs about 29 times the plain matching, because
  `nearestFloatGuided` is the scalar kernel (`docs/lab-run-records.md`, "Physical-GPU runs at
  the new defaults").
- **SwiftShader is arithmetic-bound.** The 8-view pyramid benchmark spent 58.8 s on features
  and 159.4 s on matching on SwiftShader, against 0.53 s and 0.25 s on the RTX 2080
  (`docs/lab-run-records.md`, "SwiftShader against the RTX 2080"). Every SIFT-side change is
  most visible there.
- **Tracks are short.** The mapper links pairwise matches into tracks. A feature that SIFT
  misses in one keyframe (a blur, a viewpoint change past the descriptor's range) cuts the
  track. The video record has 4,631 landmarks for 27 views, and the 8-view pyramid benchmark
  has 2,295 landmarks.

Goal: track points from frame to frame with a pyramidal Lucas–Kanade tracker on the GPU, and
use the tracks to give the mapper (a) correspondences between keyframes that are several
keyframes apart, (b) longer tracks with more observations per landmark, and (c) matches that
cost less than guided SIFT matching, in the video path only.

Non-goals: replacing SIFT. SIFT stays the keyframe descriptor, the source of the initial
pair, and the way a lost track is found again (3.3). A photographs or pyramid input does not
use KLT, because those frames are not a sequence. Optical flow for dense motion, rolling
shutter correction and a learned tracker are out of scope.

Inference, not measured: a track step costs a small fraction of a SIFT extraction, because it
reads a 15 × 15 window around at most a few thousand points, and SIFT builds a full
scale space. Step 3 measures this on SwiftShader and the RTX 2080 (section 7).

## 2. Method

### 2.1 The tracker

For two gray frames I (template) and J (target) and a point u in I, find d such that
J(u + d) ≈ I(u) over a window W around u. The sum of squared differences

ε(d) = Σ over x in W of (I(x) − J(x + d))²

is minimized by Gauss–Newton steps (Lucas and Kanade 1981; Tomasi and Kanade 1991). With the
inverse compositional form (Baker and Matthews 2004), the gradient and the 2 × 2 matrix are
taken from the template, so they are computed once for each point and level:

G = Σ over W of [Ix², IxIy; IxIy, Iy²], b(d) = Σ over W of (I(x) − J(x + d)) · [Ix; Iy], and the
step is δ = G⁻¹ b, with d ← d + δ.

The pyramid (Bouguet 2000) makes large motions reachable. Level L has the frame blurred and
decimated L times. The search starts at the top level with d = 0, and the result of level
L + 1, doubled, is the start at level L. The motion that one level resolves is a small
number of pixels, so L levels resolve about 2^L times that motion.

Sub-pixel sampling is bilinear. A window sample at a non-integer position reads the four
neighbours of J. The gradients Ix and Iy of the template are Scharr derivatives, stored per
level (4.1) and sampled at integer window positions around the template point. Because the
template point may be non-integer, the template window is sampled bilinearly too, as
OpenCV does.

### 2.2 Selection: the minimum eigenvalue

The 2 × 2 matrix G of a window tells where a track can be found. If both eigenvalues are
large the window has structure in two directions (a corner). If one is small the window has an
edge and the motion along the edge is not determined (the aperture problem). Shi and Tomasi
(1994) select a point when λmin(G) is large, and so does this design:

λmin = ((Gxx + Gyy) − sqrt((Gxx − Gyy)² + 4 Gxy²)) / 2.

Two uses:

1. **Selection of new points** (3.4), over a block of `blockSize` = 3 pixels, with a quality
   threshold of `qualityLevel` times the largest λmin of the frame, as `goodFeaturesToTrack`
   does.
2. **Rejection while tracking**, over the tracking window at the finest level: a point whose
   λmin per window pixel is below `minEigThreshold` is dropped, as `calcOpticalFlowPyrLK`
   does with its flag `OPTFLOW_LK_GET_MIN_EIGENVALUES` unset (the default drops points whose
   normalized λmin is below `minEigThreshold`).

### 2.3 Forward–backward check

Each point is tracked from I to J, then the result is tracked back from J to I. The distance
between the start and the back-tracked point is the forward–backward error (Kalal,
Mikolajczyk and Matas 2010, "Forward–Backward Error"). A point whose error is above
`fbThresholdPx` is dropped. This catches a window that converged to a wrong but self-similar
place, which the residual of one direction does not.

### 2.4 Defaults chosen

| Parameter           | OpenCV `calcOpticalFlowPyrLK` / `goodFeaturesToTrack` | scikit-image `optical_flow_ilk` | This design                                                  |
| ------------------- | ----------------------------------------------------- | ------------------------------- | ------------------------------------------------------------ |
| Window              | `winSize` 21 × 21                                     | `radius` 7 (15 × 15), Gaussian  | `halfWindow` 7 (15 × 15) at 640 px, 10 (21 × 21) at 1280 px  |
| Pyramid levels      | `maxLevel` 3 (4 levels)                               | no pyramid; `num_warp` 10       | `levels` 4 at 640 px, 5 at 1280 px                           |
| Iterations          | `criteria` 30 iterations or 0.01 px                   | `num_iter` 10 (inner)           | `maxIterations` 20 per level, `epsilonPx` 0.03               |
| Window weighting    | box                                                   | Gaussian (`gaussian=False` off) | box                                                          |
| Minimum eigenvalue  | `minEigThreshold` 1e-4                                | not applicable                  | `minEigThreshold` 1e-4 in OpenCV's units; step 3 fixes ours  |
| Residual check      | `err` output, no automatic drop                       | not applicable                  | drop when the zero-mean NCC of the final window is below 0.8 |
| Forward–backward    | none (the caller does it)                             | not applicable                  | `fbThresholdPx` 0.5 at level 0                               |
| Selection quality   | `qualityLevel` 0.01 (typical value in the docs)       | not applicable                  | `qualityLevel` 0.01, `blockSize` 3, `minDistancePx` 8        |
| Derivative operator | Scharr                                                | image gradient of the warp      | Scharr, 3 × 3                                                |

Reasons for the differences. The window of 15 × 15 holds 225 samples, half of the 441 of
OpenCV, and a track step has no cache to hide behind: each sample is a bilinear read of four
global values. The 1280 px size gets a larger window and one more level because motion in
pixels doubles with the frame width. The iteration limit of 20 has a fixed loop count
(4.3), so a smaller limit than OpenCV's 30 bounds the work per point. The NCC check is an
addition: OpenCV reports the residual in `err` and leaves the decision to the caller, and the
mapper has no caller to ask. scikit-image's `optical_flow_ilk` solves a dense flow field for
every pixel with a Gaussian window and warps the whole image, so it is a verification
reference (section 7) and not a model for the kernel.

Inference: the defaults are a starting point from the references. Step 2 measures the track
loss and the error on a synthetic sequence at each value and may change them. Any change goes
into this table.

Sources:

- B. D. Lucas and T. Kanade, "An Iterative Image Registration Technique with an Application to
  Stereo Vision", IJCAI 1981.
- C. Tomasi and T. Kanade, "Detection and Tracking of Point Features", CMU-CS-91-132, 1991.
- J. Shi and C. Tomasi, "Good Features to Track", CVPR 1994.
- J.-Y. Bouguet, "Pyramidal Implementation of the Affine Lucas Kanade Feature Tracker:
  Description of the Algorithm", Intel Corporation, 2000.
- S. Baker and I. Matthews, "Lucas–Kanade 20 Years On: A Unifying Framework", IJCV 56(3), 2004.
- Z. Kalal, K. Mikolajczyk and J. Matas, "Forward–Backward Error: Automatic Detection of
  Tracking Failures", ICPR 2010.
- OpenCV documentation: `cv::calcOpticalFlowPyrLK` and `cv::goodFeaturesToTrack` (module
  `video` and module `imgproc`), and `cv::buildOpticalFlowPyramid`.
- scikit-image documentation: `skimage.registration.optical_flow_ilk` (iterative Lucas–Kanade,
  after Le Besnerais and Champagnat 2005).

## 3. How the tracks feed the mapper

### 3.1 What the mapper takes today

The worker hands the mapper session a list of pairs:
`{ first: frameId, second: frameId, matches: Uint32Array of feature index pairs, distances }`
for the keyframe against each earlier keyframe. The mapper links the matches of all pairs into
tracks (`src/gpu/mapper.ts`, `src/gpu/observations.ts`), verifies each pair with the epipolar
model (`verifyPair`), and triangulates. KLT adds a second source of pairs in this same shape.
It changes no mapper contract.

### 3.2 Track ids and the chain

A KLT track has an id `trackId`, a `u32` that is the seed's index in the track table (4.1). A
track lives from the frame where it was seeded to the frame where it ends. It has a position
at each sampled frame in between.

Seeds are the SIFT keypoints of a keyframe K (up to `maxFeatures` = 4096) that pass the
minimum-eigenvalue test, plus Shi–Tomasi corners (3.4). The tracker advances all alive tracks
one sampled frame at a time (time step Δt = 0.25 s at 4 frames per second). Tracking between
adjacent sampled frames keeps the motion inside the pyramid's reach. A direct track from one
keyframe to the next, up to 1 s apart, would need a larger motion range and would lose points
that the chain keeps.

### 3.3 Joining SIFT keyframes (SIFT stays the keyframe descriptor)

When a frame becomes a keyframe K2, the worker has its SIFT keypoints (`points`). For each
track alive at K2 that was seeded at an earlier keyframe K1 (K1 < K2), the tracker holds its
position p in K2. The join is:

1. Find the nearest SIFT keypoint of K2 to p within `joinRadiusPx` = 2 (at 640 px, scaled by
   frame width).
2. If one exists and no other track already claims it, emit the match
   (K1 seed keypoint index, K2 keypoint index) in the pair (K1, K2).
3. Otherwise, the track ends at K2 without a match. Its position stays in the table until the
   track dies, so a later keyframe K3 may still join it, and the pair (K1, K3) gets the match.
4. A keypoint of K2 that no track claimed becomes a seed for the next interval.

The pairs (K1, K2) built this way sit beside the pairs from descriptor matching. The mapper
treats both as matches. A match found by both sources is one match: the worker merges by
(first index, second index) and keeps the descriptor match's distance. A KLT-only match
carries a distance of 0 for the mapper's sort, which the worker marks with a flag in the pair
record (`source: 'klt'`) _(new)_.

The effect: a track seeded at K1 that survives to K4 produces matches in the pairs (K1, K2),
(K1, K3) and (K1, K4). Today a pair (K1, K4) needs a descriptor match across 3 s of motion,
which the 0.8 ratio test often misses. The mapper then has more observations on the same
landmark and a wider baseline.

SIFT stays the keyframe descriptor for these reasons: a lost track must be found again with no
motion model (a loop closure, a long occlusion), the initial pair and the resection need
matches across a large baseline, and KLT has no descriptor.

### 3.4 Selection and refresh

KLT needs points to track that are not SIFT keypoints at all, for two reasons: SIFT's
difference-of-Gaussians extrema are blobs and some are poor trackers (low λmin), and a region
may have no keypoint.

At each keyframe (and at the first frame), after the SIFT seeds are in the table:

1. Compute the λmin map of the frame at level 0 (`kltEigenMap`, 5.1).
2. Mark the cells of `minDistancePx` = 8 that already hold an alive track.
3. In each empty cell, take the pixel of greatest λmin if it is at least `qualityLevel` ×
   the frame's largest λmin.
4. Compact the chosen pixels with the scan (`src/gpu/scan.shade.ts`) and append them to the
   table, up to `capacity` (4.1).

A fill-in track has no SIFT keypoint at its seed. It joins the mapper only at a later keyframe
through step 1 of 3.3, as the match (nearest SIFT keypoint of K1, K2 keypoint) when K1 has a
keypoint within `joinRadiusPx` of the seed. If K1 has none, the track serves only to check
that the geometry of the neighbourhood is consistent (3.5) and has no match to give.
Inference: fill-in tracks add little to the first step's gain, and step 6 measures them
against the SIFT-seeded tracks alone (section 7). If they add nothing, the refresh is
removed from the design.

### 3.5 Outlier rejection and termination

A track ends when any of these holds, at the step where it fails. The reason is stored in the
flag word (4.1) for the record.

1. **Out of frame.** The point is within `halfWindow` pixels of the border at level 0.
2. **Low structure.** λmin per window pixel is below `minEigThreshold`.
3. **No convergence.** The iteration limit is reached with the last step above `epsilonPx` at
   level 0.
4. **Forward–backward error** above `fbThresholdPx` (2.3).
5. **Residual.** NCC below 0.8.
6. **Epipolar outlier.** At a keyframe, the pair's verification (`verifyPair`, the fundamental
   or essential matrix RANSAC that the pipeline already runs) marks the match an outlier. The
   worker passes the inlier mask back to the tracker (`kltReject`), and the track ends.
7. **Age.** The track is older than `maxAgeFrames` = 16 sampled frames (4 s at 4 frames per
   second). Drift grows with the number of steps, since each step adds a sub-pixel error and
   nothing re-anchors the point to the first frame. A keyframe join (3.3) re-anchors the point
   to a SIFT keypoint, so the age counts from the last join.

The KLT matches enter the same verification as descriptor matches. A pair has no special
path, so the existing 50-iteration to 500-iteration RANSAC in `guidedMatches` and the pair
model in `verifyPair` apply to them. No fundamental-matrix code is added.

### 3.6 Combining with the keyframe gate

Today the gate needs features and a match. Steps in this design leave the gate alone until
step 6. After step 6, `kltTrack` runs on every sampled frame, and the median displacement of the
alive tracks from the last keyframe replaces the match-based motion in the gate when at least
64 tracks are alive. SIFT then runs only on frames that pass the gate, which is the saving in
section 1. If the phase-correlation gate of `feat/phase-gate` is merged by then, it stays the
first test. The track-based displacement is the second test, and it is a fallback if the
phase gate is off. The order is set in step 6 with the code of that day.

## 4. GPU data layout

### 4.1 Resident arrays

Sizes are in bytes. W and H are the analysis frame size. P is `capacity` = 8192 tracks (4096
SIFT seeds and 4096 fill-in). Frame slots: two, `prev` and `cur`, swapped after each step. The
sizes of a pyramid with `levels` levels are the sum of the levels, which is
(4/3)(1 − 4^(−levels)) · W · H pixels, about 1.33 W · H.

| Array (new)  | Element                                               | Count              | Bytes at 640 × 360 | Bytes at 1280 × 720 |
| ------------ | ----------------------------------------------------- | ------------------ | ------------------ | ------------------- |
| `kltGray`    | f32, all levels of one slot, level offsets in control | 2 × 1.33 W · H     | 2,457,600          | 9,830,400           |
| `kltGrad`    | f32 pairs (Ix, Iy), same levels and slots             | 2 × 2 × 1.33 W · H | 4,915,200          | 19,660,800          |
| `kltEig`     | f32, level 0 only, the λmin map                       | W · H              | 921,600            | 3,686,400           |
| `kltTracks`  | 32 B records, layout below                            | P                  | 262,144            | 262,144             |
| `kltNext`    | 16 B records: x, y, error, status                     | P                  | 131,072            | 131,072             |
| `kltBack`    | 16 B records, the backward pass                       | P                  | 131,072            | 131,072             |
| `kltControl` | uniform: sizes, level offsets, options                | 1                  | 256                | 256                 |
| `kltCells`   | u32 per cell of `minDistancePx`, selection scratch    | (W/8) · (H/8)      | 14,400             | 57,600              |
| Total        |                                                       |                    | about 8.8 MB       | about 33.8 MB       |

The gray image of the frame comes from `rgbaToGray` (`src/gpu/features.shade.ts`), the same
conversion SIFT reads, so the worker uploads the RGBA of a frame once and the two front ends
share it. Whether SIFT's resident gray buffer can serve as `kltGray` level 0 is an
optimization for step 3 (it removes the copy, and ties the two lifetimes together).

A track record, 32 bytes, as `array<f32>` words, because the repository keeps such tables as
flat arrays (`mvsSeeds` is one):

| Word | Type | Meaning                                                                               |
| ---- | ---- | ------------------------------------------------------------------------------------- |
| 0    | f32  | x at level 0 in the `cur` frame                                                       |
| 1    | f32  | y at level 0                                                                          |
| 2    | f32  | x at the seed frame                                                                   |
| 3    | f32  | y at the seed frame                                                                   |
| 4    | u32  | seed keyframe id                                                                      |
| 5    | u32  | seed keypoint index, or 0xFFFFFFFF for a fill-in track                                |
| 6    | u32  | age in sampled frames since the last join                                             |
| 7    | u32  | flags: bit 0 alive, bits 1 to 4 the end reason of 3.5, bits 8 to 15 the level reached |

The table is compacted on the host after each keyframe join (3.3), when dead tracks are
removed and the survivors are re-uploaded. Inference: 8192 records are 256 KiB, so the
re-upload is not a cost that needs a device-side compaction. A keyframe is rare (1 in about
21 decoded frames in the video record).

The memory is released when the video ends, with the other buffers of the run
(`liveBytes` in the run record). It adds 33.8 MB at 1280 px to a run whose MVS stage needs
229.7 MiB (`docs/dense-mvs.md`, 4.2), but the two do not run together: KLT frees its arrays
before `finish()`.

### 4.2 Kernels

All kernels have a workgroup of 64 lanes (`@compute([64])`), as every kernel of
`src/gpu/*.shade.ts`. The 2D kernels map a lane to one pixel by `gid.x` over W · H (a flat
index, as `rgbaToGray` does). Names are _(new)_ and live in `src/gpu/klt.shade.ts`.

| Kernel        | One lane does                                                                | Lanes                 | Notes                                                                           |
| ------------- | ---------------------------------------------------------------------------- | --------------------- | ------------------------------------------------------------------------------- |
| `kltPyrDown`  | one output pixel of level L + 1: a 5-tap binomial blur of level L, decimated | pixels of level L + 1 | 1 4 6 4 1 / 16 in x then y, two passes with a scratch level; border by clamping |
| `kltScharr`   | one pixel of one level: Ix and Iy                                            | pixels of the level   | 3 × 3 Scharr (3, 10, 3), scaled by 1/32; border clamped                         |
| `kltEigenMap` | one pixel at level 0: λmin of a `blockSize` window                           | W · H                 | for selection (2.2)                                                             |
| `kltEigenMax` | partial max of 64 values, then a second pass                                 | W · H / 64            | workgroup memory and barriers; or the host reads the map once                   |
| `kltSelect`   | one cell of `minDistancePx`: the best pixel of the cell                      | number of cells       | writes the pixel or a sentinel; the scan compacts them                          |
| `kltTrack`    | one track, all levels and iterations                                         | tracks of the chunk   | the core kernel (4.3). Direction in `kltControl`: forward or back               |
| `kltCheck`    | one track: the forward–backward error, NCC, the flags                        | tracks                | a few operations per track, no loop                                             |
| `kltReject`   | one track: clear alive from a host mask                                      | tracks                | applies the epipolar verdict of 3.5, item 6                                     |

Split of the pyramid and the gradient into two kernels follows the SIFT kernels, which keep
one operation per kernel and read a level at a time. Their dispatches count in the plan, and
section 4.4 gives the number.

### 4.3 The tracking kernel

`kltTrack` is one lane per track, because the work of one track is a short sequence of window
reads with a data-dependent iteration count, and a lane-per-track form keeps the whole path in
registers with no workgroup memory and no barrier. This is the form of `mvsPropagate` (one
invocation per element, no workgroup memory). The pseudocode, per lane:

```
for level = levels - 1 down to 0:
  // template window (center = u / 2^level) from kltGray[prev], kltGrad[prev]
  // G from the window, once per level; if det(G) is too small the point is lost
  for it in 0 .. maxIterations - 1:        // fixed bound, masked after convergence
    b = sum over window of (I - J(x + d)) * [Ix; Iy]     // bilinear read of J
    step = G^-1 b ; d += step
    if |step| < epsilonPx: done = true                   // later iterations change nothing
  d *= 2 when moving to the next finer level
```

Rules that apply to this kernel and to every kernel of this design:

- **An idle lane does not return before a loop or a barrier.** A lane past the chunk takes the
  chunk's last track, runs the same control flow and writes nothing. This is the clamp-and-mask
  form of `docs/hardware-observations.md`, A1, and the form `mvsPropagate` follows
  (`docs/dense-mvs.md`, 3.6). The exit of a converged lane is a flag, not a `break` or a
  `return`: the loop runs to `maxIterations` and a converged lane stops updating `d`. The
  loop bound is the same for every lane.
- **No `0u` literal.** The kernels write a `u32` zero as `0` where the type is known from
  context, or as `u32(0)`, and do the same for every unsigned constant. The spelling of
  `src/gpu/mvs.shade.ts` is the model (`u32(fx)`, `min(…, W - 1)`). The compiler rule is the
  owner's instruction for this repository, and the step 2 test is the first place it can break.
- **f32 only.** No emulated double. The kernels need no f64, so the flavor of
  `docs/hardware-observations.md`, B1, does not apply.
- **Border.** A window sample outside the level clamps to the border, and the lane marks the
  track as out of frame. It does not branch around the loop.
- **Bilinear read.** A helper `kltSample(level, x, y)` reads four values of `kltGray[cur]` with
  the level's offset and width from `kltControl`.

### 4.4 Chunks, the watchdog budget and warm steps

Kernels that loop over a variable amount of work run as chunks of the time budget through
`device.planSteps`, as `triangulateTracksGpu` and the MVS keys do
(`docs/gpu-sfm.md`, "The time budget"; `docs/dense-mvs.md`, 5.1). The host sizes each chunk with
`device.chunkSize(key, remaining, 64)`. Each chunk step carries `warm`, its entries over zero
workgroups, so the pipeline is created outside the first timed chunk.

| Chunk key (new) | Kernel        | One unit                   | Work per unit                                  |
| --------------- | ------------- | -------------------------- | ---------------------------------------------- |
| `kltTrack`      | `kltTrack`    | one track in one direction | `levels` × up to `maxIterations` × 225 samples |
| `kltEigenMap`   | `kltEigenMap` | one pixel                  | 2 × 9 gradient reads                           |

The forward and the backward pass share the key `kltTrack`, because the work per unit is the
same. The other kernels do a few operations per element and run as one plan with no chunk
(`kltPyrDown`, `kltScharr`, `kltSelect`, `kltCheck`, `kltReject`). The CPU tier runs each key as
one chunk (`chunkSize` returns `max`).

A chunk splits at any track boundary with no change of result, because the tracks do not read
each other's state: a track reads `prev` and `cur` and writes only its own `kltNext` record.
Step 2's test checks this bit for bit.

Budget estimate (inference, not measured). Per track, per frame pair, at most
4 levels × 20 iterations × 225 samples = 18,000 bilinear samples; at 8192 tracks, forward and
backward: 295 M samples. Most tracks converge in fewer than 5 iterations, so the typical load is
about a quarter of that, 74 M samples. An arithmetic-bound software device handles this in the
order of a second per frame pair at the worst case. The SIFT matching it replaces took 159.4 s for
8 views on SwiftShader. Step 3 records the real rate of `kltTrack` in ms per unit, and
`timing.chunks` carries it in the run record.

Dispatch count per frame, inference: the pyramid takes 2 dispatches per level (blur in x and y)
plus 1 for the gradient, so about 3 × `levels` = 12 at 640 px. The tracking takes 2 plans
(forward, back) with a chunk each, so about 2 to 6 in a typical run. The check takes 1. In total
about 20 dispatches and 1 to 3 submissions per frame, well below SIFT's 1,696 dispatches for
8 views (`docs/lab-run-records.md`, SwiftShader table). On a physical GPU the round trips of
submissions bound the time (`docs/lab-run-records.md`, same table), so the design keeps the
submissions per frame at 3 or fewer and reads back the track table once per keyframe, not once
per frame.

### 4.5 CPU reference

`src/gpu/klt.ts` _(new)_ holds the host API and a plain TypeScript reference
(`trackKltReference`) that runs the same arithmetic on typed arrays: the same pyramid, the
same Scharr, the same iteration order, the same f32 rounding with `Math.fround`. The reference
is the oracle for the kernels on the CPU tier (`tests/typeshade/cpu.test.mjs` is the model) and
for WebGPU (`tests/typeshade/*-webgpu.test.mjs`).

## 5. API and worker

### 5.1 API

```ts
// src/gpu/klt.ts (new). Shapes only.
export interface KltOptions {
  levels;
  halfWindow;
  maxIterations;
  epsilonPx;
  minEigThreshold;
  fbThresholdPx;
  qualityLevel;
  minDistancePx;
  maxAgeFrames;
  capacity;
}
export interface KltTracker {
  push(frame): KltStep;
  join(keyframe): KltPairs;
  reject(mask): void;
  dispose(): Promise<void>;
}
```

`push(frame)` uploads or receives the gray image, builds the pyramid, tracks the alive tracks
from the previous frame and runs the checks. `join(keyframe)` implements 3.3 and returns the
matches by pair. `reject(mask)` applies the epipolar verdict. The tracker owns its buffers and
releases them in `dispose()`. The export is added to `src/gpu/index.ts` in step 4.

### 5.2 Worker

`processFrame` in `website/src/workers/sfm.worker.ts` calls `tracker.push(frame)` for every
sampled frame of a video input, behind the page option `?klt=1` _(new)_ in steps 4 and 5, and by
default after step 7. The option is also a field of the run record (`options.klt`, v6). The
run record gains the per-stage times `klt` (pyramid, track, check, join), the counts of alive
tracks and of ends by reason, and the track-length histogram. These are the numbers of section 7.
The record's version number and the collector change in the same pull request as the first
field (`docs/lab-run-records.md`).

## 6. The lab

The viewer is not changed. The summary line of a run shows the mean track length and the
number of KLT matches beside the SIFT matches. A new switch `?klt=0` or `1` follows the pattern of `?guided=`
and `?octave=` in `website/src/lib/sift-quality.ts`.

## 7. Verification plan

### 7.1 Synthetic sequences on the CPU tier (CI)

Tests generate a textured image (a sum of random Gaussians and a checker, so λmin is large
everywhere) and a second frame by a known warp, with bilinear resampling:

1. **Translations.** Shifts of (0.3, 0.2), (3.7, −2.1), (12.5, 9.0) and (30, −24) px at 640 px.
   The reference and the kernel must recover each shift within 0.05 px for shifts below 12 px
   and within 0.1 px above. The 30 px shift must need the pyramid: with `levels` 1 the test
   must fail (proving the instrument).
2. **Rotations and scale.** Rotations of 2°, 5° and 10° about the image center, and a scale of
   1.05. The flow at a point is known in closed form. The error is allowed to grow with the
   distance from the center, but must stay under 0.2 px for 5°.
3. **Brightness change.** A gain of 1.1 and an offset of 0.05 on the second frame. The NCC check
   must keep the points, and the error must stay under 0.3 px. This documents a known limit
   of the plain SSD form (open question in section 9).
4. **Failure cases.** A textureless patch (λmin below the threshold), a one-pixel line (the
   aperture problem), a point at the border, and an occlusion (the window covers a second
   texture). Each track must end with the reason of 3.5.
5. **Chunk invariance.** The same tracking in one chunk and in chunks of 3 tracks gives the same
   bits.
6. **Forward–backward.** A repeating texture with a period of 8 px, shifted by 4 px, makes
   a forward track that converges to a wrong place. The check must drop at least 90 % of those.

Files: `tests/typeshade/klt.test.mjs` and `tests/typeshade/klt-cases.mjs` _(new)_, in the
repository's pattern (`mvs.test.mjs`, `mvs-cases.mjs`). They run with `npm run test:shade`.

### 7.2 The same sequences on WebGPU (CI)

`tests/typeshade/klt-webgpu.test.mjs` _(new)_ runs the cases of 7.1 on the WebGPU device and
compares with the CPU reference: at least 99 % of the tracks within 1e-3 px of the reference,
as the MVS test does. It also records the rate of `kltTrack` on SwiftShader. It runs with
`npm run test:webgpu`.

### 7.3 Comparison with OpenCV

If OpenCV is available (`opencv-python` in a tooling environment, not a dependency of this
repository), `scripts/klt-opencv-compare.py` _(new)_ tracks the same points of a fixed
frame pair with `cv2.calcOpticalFlowPyrLK` (21 × 21, `maxLevel` 3, 30 iterations, 0.01) and
the reference of this design, and prints the median and the 95th percentile of the distance
between the two flows, and the fraction of points that each keeps. Acceptance: the median
distance is below 0.1 px on points that both keep. The script is an offline record, like the
COLMAP comparison in `docs/dense-mvs.md`, 7.4. It is not a CI gate, because the environment
does not install OpenCV. If OpenCV is not available, the result is "not run", and it is
recorded so. The same frame pair goes through `skimage.registration.optical_flow_ilk` and the
dense flow is read at the points, as a second reference with a different method.

### 7.4 A lab gate on a video sample

Videos: the 4K video of record `a6fb6f3b` (27 keyframes, 4,631 landmarks, 0.35 px) and the
pyramid sample (8 views, 2,295 landmarks, 0.31 px). The gate compares `?klt=0` and `?klt=1`
on the same input, on the same build, and records:

| Number                           | Source                     | Gate (to be set by step 6)                                     |
| -------------------------------- | -------------------------- | -------------------------------------------------------------- |
| Registered views                 | `result` of the run record | not less than `klt=0`                                          |
| Mean track length (observations) | `result`                   | greater than `klt=0` by a margin the record sets               |
| Landmarks and observations       | `result`                   | observations per landmark up                                   |
| Mean reprojection error          | `result`                   | not more than `klt=0` plus 0.05 px                             |
| Stage times                      | `timing`                   | features plus matching plus KLT less than `klt=0` on the video |
| Longest chunk                    | `longestChunk`             | not above the time budget                                      |
| Tracks alive, ended by reason    | `klt` (new)                | recorded, no gate                                              |

The first run sets the margins, as the dense MVS gates did. Inference: the video input gains
more than the pyramid input, because its keyframes are close and a track survives many
frames. The pyramid sample is a set of photographs with larger gaps than the video's, and the gain there is unknown.

### 7.5 Gates

| Gate                        | Number                                                                                | Where                 |
| --------------------------- | ------------------------------------------------------------------------------------- | --------------------- |
| Synthetic translations, CPU | shift error ≤ 0.05 px below 12 px, ≤ 0.1 px above; `levels` 1 fails at 30 px          | `npm run test:shade`  |
| Chunk invariance            | bit for bit                                                                           | `npm run test:shade`  |
| Synthetic, WebGPU           | ≥ 99 % of tracks within 1e-3 px of the CPU tier                                       | `npm run test:webgpu` |
| OpenCV comparison           | median flow difference ≤ 0.1 px (offline; "not run" when OpenCV is absent)            | offline record        |
| Video sample, lab           | the table of 7.4                                                                      | lab, run record       |
| Physical devices            | the video sample completes on devices A and B, no device loss, longest chunk ≤ budget | lab, run records      |

## 8. Steps

Each step is one pull request, in this order. Each names this document in its commits.

| Step | Pull request content                                                                                                                               | What it proves                                                                                           |
| ---- | -------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| 1    | `src/gpu/klt.ts` `trackKltReference`: pyramid, Scharr, the inverse compositional solver, the checks. `tests/klt-reference.test.ts`.                | The reference recovers the known shifts and rotations of 7.1. The defaults of 2.4 are right.             |
| 2    | `src/gpu/klt.shade.ts`: `kltPyrDown`, `kltScharr`, `kltTrack`, `kltCheck`, CPU tier. `tests/typeshade/klt.test.mjs`.                               | The kernels equal the reference. Chunk invariance. The idle-lane rule and the literal rule compile.      |
| 3    | The chunk keys and `warm` steps. `tests/typeshade/klt-webgpu.test.mjs`. The rate of `kltTrack`.                                                    | WebGPU agrees with the CPU tier. The ms per unit on SwiftShader, and the section 4.4 estimates.          |
| 4    | `kltEigenMap`, `kltSelect`, the scan, `KltTracker.push`. The export in `src/gpu/index.ts`.                                                         | Selection finds the corners of the synthetic scene and keeps `minDistancePx`. The memory formula of 4.1. |
| 5    | `join` and `reject`: the joining of 3.3, the pair shape, the merge with descriptor matches. `?klt=1` in the worker, off by default. Run record v6. | KLT matches reach the mapper, pass `verifyPair`, and do not change `?klt=0` runs.                        |
| 6    | The lab gate of 7.4 on the video sample. The use of track displacement in the keyframe gate (3.6). The OpenCV record of 7.3.                       | Longer tracks, more observations per landmark, same or lower error. The saving in SIFT stages.           |
| 7    | The default on for videos. Physical-device runs and the gate rows of 7.5.                                                                          | No device loss on devices A and B. The physical rates.                                                   |

Steps 1 and 2 need no change to the lab. Step 6 can wait for the phase gate, since 3.6 says
how the two combine.

## 9. Risks and open questions

**Drift.** Each chained step adds a sub-pixel error, and a long track drifts from the first
position. The design bounds the age (3.5, item 7) and re-anchors at SIFT keypoints. Open:
whether 16 sampled frames is right. Step 6 measures reprojection error against the track age.

**Brightness change.** Plain SSD assumes constant brightness. Auto exposure in a video breaks
that. Options: a per-window gain and bias (as in Jin, Favaro and Soatto 2001), or mean
subtraction of both windows. The NCC drop of 3.5 hides some failures but does not fix the
estimate. Open: whether step 2 needs the mean-subtracted form by default. The test of 7.1, item
3 decides.

**Rolling shutter and motion blur.** A phone video has both. Blur makes λmin low and the tracks
end. Rolling shutter bends straight edges and adds a small inconsistency to the epipolar
check. No part of this design models either. Step 6 reports the fraction of tracks ended by
low structure.

**Parallelism of one lane per track.** 8192 lanes occupy a small fraction of a large GPU, and
each lane's loop is long. A workgroup per track with a barrier-based reduction over the window
would use more lanes, but it adds workgroup memory and barriers, the construct that
`docs/hardware-observations.md`, A1, ties to a device hang. Open: whether the lane-per-track
form is fast enough on the RTX 2080 and on mobile. Step 3 measures it. The barrier form is a
later step only if the number demands it.

**Selection spacing.** One best pixel per cell of 8 px enforces a spacing of at least 1 px but
not 8 px, and two strong pixels in adjacent cells can sit close. OpenCV's `minDistance` is
exact, with a greedy pass over sorted corners. Open: whether the cell form is good enough. Step
4 counts the pairs closer than `minDistancePx`.

**Joining by nearest keypoint.** A radius of 2 px can join a track to a wrong keypoint in
dense texture. The epipolar check of 3.5 catches the match as an outlier, but a good track is
lost. Open: whether the join needs a descriptor distance bound too. The mapper does not use
KLT matches' distances today (3.3).

**Mapper effects.** More matches on a pair raise the cost of `verifyPair` (pair verification
took 56.7 s in record `a6fb6f3b`, a round-trip-bound stage). Open: whether more, longer tracks
make the bundle and the verification slower than the saving. Step 6 reports the whole run time,
not only the SIFT stages.

**Interaction with `feat/phase-gate`.** The phase gate is another agent's work and not on
`main`. If it changes the keyframe rule or the worker's `processFrame`, section 3.6 and step 5
change with it. Decision for the owner: whether this design waits for that branch.

**Sampling rate.** At 4 frames per second a fast camera moves farther than the pyramid reaches
(about 2^levels × the per-level range, 30 px or more at 640 px). Open: whether KLT needs a
higher rate (`?fps=` accepts up to 120) than SIFT does. A higher rate costs the decode and the
upload of more frames, and not more SIFT, since the gate runs first. Step 6 sweeps the rate.

**Image size.** The 1280 px case doubles the motion in pixels and the window. The defaults for
1280 px in 2.4 are inferred from the 640 px ones, and step 3 checks them.
