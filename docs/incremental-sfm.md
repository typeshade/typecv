# Incremental calibrated sparse reconstruction

This guide covers TypeCV's incremental SfM module within geometry and 3D vision.
See the [SfM quickstart](sfm-quickstart.md) for usage and the
[project documentation](README.md) for other capability areas.

The root `reconstructIncremental` API below is synchronous CPU execution.
`typecv/gpu` also exports `reconstructIncrementalGpu`, using the same control
flow with device E/H/relative-pose initialization, PnP, triangulation, filtering
and joint BA. See [GPU geometry](gpu-geometry.md) and
[GPU initialization](gpu-initialization.md) for asynchronous contracts and the
host packing/graph boundary.

The `rotationTracking` mode, enabled by default without automatic `focalSearch`, retains calibrated rotation-compatible observations
and exposes pending tracks even when structure cannot initialize. It supplies relative
rotation evidence without inventing translation or depth. See [rotation tracking](rotation-tracking.md)
for its calibration requirements, CPU execution boundary and active-window limits, and the
[comparison](rotation-tracking-comparison.md) for measured benefits and costs.
Set `rotationTracking: false` for the original graph. Automatic focal search keeps it disabled.

## Data and coordinate contracts

`ImageFeatures` holds a unique nonnegative safe-integer image `id`, fixed `Camera`
parameters and finite packed `Float64Array` keypoints `[x,y,...]`. Coordinates use
pixel centers at integer locations, following the optional OpenCV adapter.
`PairMatches` holds distinct `first`/`second` image ids and `Uint32Array` feature
indices `[i,j,...]`. Optional finite nonnegative `distances` give one descriptor
distance per match; smaller distances take priority when tracks conflict.
The mapper accepts raw matches and geometrically verifies them. Duplicate
unordered image pairs are rejected; combine their matches before calling.

`Pose` always maps world to camera: **q = R X + t**. R is a proper row-major 3×3
rotation. Camera center is `-Rᵀt`. Multi-view translations are not individually
normalized. The first registered pose fixes the world frame, and the distance
between the initial camera centers fixes the arbitrary common scale. There is
no metric scale without external information. An explicit `initialPair` selects
an unordered pair; the actual ordered world-frame anchor is reported in
`diagnostics.initialPair`.

`Camera` contains `fx,fy,cx,cy,k1?,k2?`. Distortion is
`(xd,yd)=(x,y)*(1+k1*r²+k2*r⁴)` followed by pixel intrinsics. Inversion uses
backtracked Newton iterations on the injective branch connected to the optical
axis. Projection/inversion rejects lens folds and invalid depth. Tangential,
fisheye, rolling-shutter and per-frame stabilization models are not implemented.
Camera parameters are fixed during reconstruction/BA.

`SparseReconstruction` contains registered views and landmarks. Each landmark has
one `trackId`, world XYZ and its accepted observations with original pixels.
There may be at most one observation per image per landmark, and each
`(viewId,featureId)` belongs to at most one landmark. `validateReconstruction`
checks these ownership and dimension contracts. A BA problem must form one
connected observation graph and have a nonzero gauge baseline.

## Pipeline

1. Invert distortion and robustly estimate pair geometry using normalized
   eight-point Essential estimation. Separately fit homographies using normalized
   DLT and symmetric transfer error. Projective null-space fits use direct
   one-sided Jacobi SVD, avoiding the condition-number squaring of `AᵀA`.
2. Build tracks from verified Essential consensus edges. Deterministic union-find
   refuses a merge that would create two features from the same image. This
   enforces ownership, but does not prove that a track is a correct physical point.
3. Select a seed with enough positive-depth points and median parallax, rejecting
   homography dominance. The current heuristic rejects seeds when homography
   consensus covers at least 85% of inputs and at least 90% of Essential consensus.
   Score eligible pairs by accepted points times capped median angle. This is
   conservative heuristic model selection, not a calibrated statistical test.
4. Triangulate the seed pair's tracks into one frame. Fit all-observation and
   pair hypotheses, select positive-depth pixel-error consensus, refine a fixed
   consensus and require sufficient triangulation angle. A bounded number of pair
   hypotheses is tested (32 by default), ranked by camera baseline.
5. Rank unregistered images by the number of landmarks their tracks already have.
   P3P RANSAC generates positive-depth poses, then a fourth/additional point
   disambiguates branches. Refit sufficiently large consensus using normalized DLT
   or planar homography, and refine pose with analytical Jacobians and
   Huber-weighted LM. An image whose registration failed is retried once each time
   the map has grown, at most three times. When no image registers at a map size,
   one recovery pass retriangulates every track and refines the map; if that adds
   landmarks, the failed images get one more attempt. If the map still stalls, a weak
   registration pass tries each failed image with half the PnP consensus
   (`minPnPInliers / 2`, at least 6). Images that match each other strongly but reach the
   map through few landmarks need this: the tracks they share have one registered image, so
   nothing triangulates them until one of them registers. The pass keeps a registration only
   when the image's tracks then triangulate at least `minPnPInliers` new landmarks, and
   rolls it back otherwise. The pass runs only when the map would stop, so a model that
   registers every image without it is unchanged. Images that cannot be
   registered are reported with the last reason.
6. Landmarks persist. A newly registered image continues each landmark whose
   track it observes when that observation reprojects within `maxErrorPx`, and
   triangulates only the tracks it completes. Tracks that failed to triangulate are
   retried when the map has grown: before each global pass and at the end. Then
   one BA refines the retained points and poses: a global BA when the model has
   grown enough since the last global BA, a local camera-window BA with fixed
   boundary poses otherwise (see "BA schedule" below). Large reprojection errors are filtered explicitly
   outside BA; a landmark left with fewer than two observations is dropped and may
   be re-triangulated later. Finish with retriangulation, global BA and filtering.

Initialization failure throws `ReconstructionInitializationError`, which exposes
`pairVerification` reasons. Incomplete reconstruction returns `unregistered`
image ids and reasons. It does not invent poses for disconnected images or
silently create multiple models. Full pair geometry, registration order, track
conflicts and BA before/after diagnostics remain inspectable.

## P3P and numerical safeguards

Three world points and unit camera bearings give the three cosine-law distance
equations. Grunert's distance ratios reduce them to a quartic. Derivative critical
points isolate real roots; safeguarded bisection and distance Newton refinement
recover candidate branches. Positive lengths, rigid-transform consistency and
cheirality validate candidates. RANSAC scores all validated branches.

Collinear world points, repeated rays and extremely narrow ray cones (maximum
bearing separation below 0.001 in unit-ray chord distance) are rejected. Near
multiple roots can be numerically unresolved; more than four surviving distinct
branches causes rejection of that sample. This is an f64 solver with tested
numerical guards, not symbolic exact arithmetic. The tests include 100 generated
poses over three scales, symmetric/repeated-root configurations and degeneracies.

## Bundle adjustment

Each observation uses a two-dimensional Euclidean Huber loss. Analytical
Jacobians cover radial projection, XYZ and a left pose update
`q' = exp(omega) q + v`. LM forms point 3×3 and camera blocks, then eliminates
points:

`S = A - W V⁻¹ Wᵀ`, `rhs = b_camera - W V⁻¹ b_point`.

The reduced camera matrix is currently **dense**, solved with pivoted elimination.
Storage is quadratic and solve cost cubic in active cameras. This implementation
is intended for small scenes/local windows; it is not a large-scale sparse BA
backend. Point blocks are sparse with respect to their observing cameras.

Global BA fixes the anchor pose and normalizes candidate geometry by a common
similarity to preserve the initial baseline. This changes no pixel projections.
Local BA exposes `variableViewIds`; other poses remain fixed, and the scale camera
must be fixed when multiple boundary cameras are fixed. Landmarks still use all
their retained observations. Input geometry is copied. No observation is removed
inside BA. Steps must lower the robust objective on the same observation set;
raw RMSE is reported but is not the robust acceptance objective.

The mapper's `localBundleSubset: true` (the default) also restricts local point work.
It selects landmarks observed by the variable camera window, keeps every retained
observation of those landmarks, and includes all their observing cameras. Two
fixed boundary cameras with the largest baseline define the local gauge; isolated
seed cameras are not inserted into the problem. If the selected graph is
disconnected, lacks two distinct fixed camera centers, or leaves a requested
variable camera unobserved, the mapper uses its existing full-model camera-window
route. Intrinsics remain fixed during local passes.

After a successful subset solve, only selected XYZ coordinates and variable poses
are merged into the full map. Fixed cameras, unrelated landmarks and all existing
observations are retained. These subset passes do not delete observations.
Scheduled and final global passes still retriangulate/refine and filter using the
existing error gates, including optional final shared-focal refinement. The option
defaults to `true`; `false` keeps every landmark variable in a local pass, the
default before the BA schedule below. In one same-source native 12-frame comparison
with the cache disabled in both runs, enabling `localBundleSubset` reduced mean warm whole-pipeline
time from 5,776.35 ms to 5,635.70 ms (1.025×, about 2.44%). All five paired final
models matched exactly across 14,186 numeric leaves, and BA observation work fell
from 16,129 to 14,378. This is a single-scene result, not a general speed claim;
see the [GPU comparison record](experiments/gpu/sfm-local-schur-cache-rtx2080.json).

The standalone mapper's global filtering above is distinct from a rolling GPU
`IncrementalSfmSession`: it first finalizes its native local window, then may run a
complete-observation archive BA while preserving those observations. Its final mean
covers the full exported archive, rather than only local inliers. See
[final archive consistency](archive-consistency.md) for this session-specific path.

### BA schedule

Each registration is followed by one BA. It is global when the model has grown
since the last global BA by `globalBundleViewRatio` in registered views or by
`globalBundlePointRatio` in landmarks (both 1.1 by default, COLMAP's
`ba_global_images_ratio` and `ba_global_points_ratio`). It is a local BA of the
`localBundleSize` most recent cameras otherwise. The initial pass, a recovery pass
and the final pass are always global. `globalBundleFrequency` adds a global BA at
each multiple of its view count; it was the whole schedule before (every third
view) and is now off by default. `Infinity` turns a ratio off. `globalBundleDue` in
[`src/sfm/mapper/mapper.ts`](../src/sfm/mapper/mapper.ts) holds the rule, and
both mappers (`incrementalSteps`, `sceneSteps`) apply it. `IncrementalSfmSession`
carries the size at the last global BA across its runs; when its window evicts
views, the ratios hold against the smaller map.

`bundleStallTolerance` (default 1e-4) ends the LM loop of every mapper BA when an
accepted step lowers the robust cost by less than that share of the cost.
`bundleAdjust` and `bundleAdjustGpu` take it as `stallTolerance` (default 1e-9,
the earlier convergence rule). The GPU routes read the cost on the host after each
LM iteration when the tolerance is above 1e-9; the shaders keep their own 1e-9 rule.

Recorded 2026-10-20 in this container (4 cores, no physical GPU, other jobs
running), main 9633311 against this change, COLMAP south-building with the
dataset's features and matches (`maxErrorPx` 4, seed 42), compared with
pycolmap 4.2.1's mapper on the same matches:

| Run                                        | Registered    | Mean error px   | Median rotation vs mapper | Median center vs mapper | BA time       | BA work or dispatches       |
| ------------------------------------------ | ------------- | --------------- | ------------------------- | ----------------------- | ------------- | --------------------------- |
| 24 images, CPU reference, default options  | 24/24 → 24/24 | 0.3431 → 0.3433 | 0.0240° → 0.0226°         | 0.026% → 0.026%         | 34.2 → 14.7 s | 4.02M → 1.33M obs × LM step |
| 16 images, CPU reference, default options  | 16/16 → 16/16 | 0.3653 → 0.3650 | 0.0207° → 0.0234°         | 0.037% → 0.041%         | 16.1 → 7.3 s  | 1.57M → 0.64M obs × LM step |
| 16 images, lab session, TypeShade CPU tier | 16/16 → 16/16 | 0.3570 → 0.3570 | 0.0161° → 0.0152°         | 0.032% → 0.030%         | 546 → 279 s   | —                           |
| 8 images, lab session, SwiftShader WebGPU  | 8/8 → 8/8     | 0.3562 → 0.3562 | 0.0065° → 0.0078°         | 0.041% → 0.042%         | 32.3 → 18.0 s | 3,613 → 2,804 dispatches    |

The lab session is `IncrementalSfmSession` with the lab's photo controls
(`bundleIterations` 4, `finalBundleIterations` 8, `windowSize` 32). On the
24-image run, the schedule and the local subset alone lower the BA work by 21 %,
the stall rule alone by 55 %, and both by 67 %. Up to about 30 views the 1.1
ratios make most passes global (17 of 24 on that run, against 10 before), so the
gain there comes from the stall rule and the bounded local passes. Times are
single observations on a shared software container, not a physical-GPU claim.

## Options

| Mapper option              | Default   | Meaning                                                                          |
| -------------------------- | --------- | -------------------------------------------------------------------------------- |
| `seed`                     | 17        | Deterministic PRNG seed                                                          |
| `ransacIterations`         | 750       | Trial cap for pair geometry and PnP                                              |
| `maxErrorPx`               | 3         | Pixel-error gates; approximate normalized Sampson conversion during pair fitting |
| `minInitialAngleDeg`       | 1         | Minimum seed median parallax                                                     |
| `minTriangulationAngleDeg` | 0.5       | Minimum accepted track angle                                                     |
| `minPnPInliers`            | 12        | Minimum 2D–3D consensus for registering an image                                 |
| `minInitialPoints`         | 12        | Minimum initial reliable points                                                  |
| `bundleIterations`         | 8         | LM step cap during incremental construction                                      |
| `finalBundleIterations`    | 25        | Final global LM step cap                                                         |
| `refineSharedFocal`        | `false`   | Jointly refine one positive shared focal scale with final poses and points       |
| `focalSearch`              | off       | Estimate one shared focal before mapping ([automatic focal](auto-focal.md))      |
| `localBundleSize`          | 6         | Most recently registered variable cameras, excluding the seed pair               |
| `localBundleSubset`        | `true`    | Restrict local BA to incident landmarks and all their retained observations      |
| `globalBundleViewRatio`    | 1.1       | Global BA when registered views reach this multiple of the last global BA's      |
| `globalBundlePointRatio`   | 1.1       | Global BA when landmarks reach this multiple of the last global BA's             |
| `globalBundleFrequency`    | off       | Also run global BA at this registered-camera count interval (was 3)              |
| `bundleStallTolerance`     | 1e-4      | End a BA's LM loop when an accepted step lowers the cost by less than this share |
| `initialPair`              | automatic | Requested unordered seed; must pass all checks                                   |

Standalone `estimatePosePnP` accepts packed XYZ/pixels, requires at least four
correspondences and defaults to 1,000 trials, 3px threshold, 0.999 confidence,
20 refinement steps and `min(6,N)` inliers. RANSAC confidence estimates are not
pose-correctness guarantees. `buildTracks` is separately available for callers
with already verified edges. `triangulateTrack` and `bundleAdjust` are also public.

## Remaining limitations

Essential estimation remains eight-point, not calibrated five-point. Homography
selection is heuristic; planar/noisy/low-baseline scenes may fail or be wrongly
accepted. Optional shared focal calibration jointly estimates one positive scale
for the supplied focal lengths; principal point and radial distortion stay fixed.
Alone it moves the focal only a little from its start; `focalSearch` finds the
start first ([automatic focal](auto-focal.md)).
There is no keyframe selection, relocalization, multiple-component merging or
large-scale sparse camera solve.
A bounded triangulation hypothesis set may miss good pairs in long/outlier-heavy
tracks. The mapper keeps its landmarks across registrations; tracks are a fixed
union-find partition of the verified matches, so there is no track merging or
splitting after a wrong match survives verification. Native feature extraction and TypeShade kernels
are not used by this CPU core. Dense depth, fusion and surface reconstruction
are separate future work. Passing tests and matching a single reference example
are insufficient evidence of production-level reconstruction accuracy.
