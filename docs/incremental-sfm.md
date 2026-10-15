# Incremental calibrated sparse reconstruction

This guide covers TypeCV's incremental SfM module within geometry and 3D vision.
See the [SfM quickstart](sfm-quickstart.md) for usage and the
[project documentation](README.md) for other capability areas.

The root `reconstructIncremental` API below is synchronous CPU execution.
`typecv/gpu` also exports `reconstructIncrementalGpu`, using the same control
flow with device PnP, triangulation, filtering and joint BA. See
[GPU geometry](gpu-geometry.md) for its asynchronous contract and remaining
hybrid two-view initialization boundary.

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
4. Triangulate registered observations into one frame. Fit all-observation and
   pair hypotheses, select positive-depth pixel-error consensus, refine a fixed
   consensus and require sufficient triangulation angle. A bounded number of pair
   hypotheses is tested (32 by default), ranked by camera baseline.
5. Rank unregistered images by existing 2D–3D support. P3P RANSAC generates
   positive-depth poses, then a fourth/additional point disambiguates branches.
   Refit sufficiently large consensus using normalized DLT or planar homography,
   and refine pose with analytical Jacobians and Huber-weighted LM. Retry failed
   images after the map grows; report images that cannot be registered.
6. Retriangulate/complete tracks, run camera-window BA with fixed boundary poses,
   and periodic global BA. Filter large reprojection errors explicitly outside BA.
   Finish with retriangulation, global BA and final filtering.

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
| `localBundleSize`          | 6         | Most recently registered variable cameras, excluding the seed pair               |
| `globalBundleFrequency`    | 3         | Run global instead of local BA at this registered-camera count interval          |
| `initialPair`              | automatic | Requested unordered seed; must pass all checks                                   |

Standalone `estimatePosePnP` accepts packed XYZ/pixels, requires at least four
correspondences and defaults to 1,000 trials, 3px threshold, 0.999 confidence,
20 refinement steps and `min(6,N)` inliers. RANSAC confidence estimates are not
pose-correctness guarantees. `buildTracks` is separately available for callers
with already verified edges. `triangulateTrack` and `bundleAdjust` are also public.

## Remaining limitations

Essential estimation remains eight-point, not calibrated five-point. Homography
selection is heuristic; planar/noisy/low-baseline scenes may fail or be wrongly
accepted. There is no shared-intrinsic calibration, keyframe selection,
relocalization, multiple-component merging or large-scale sparse camera solve.
A bounded triangulation hypothesis set may miss good pairs in long/outlier-heavy
tracks. The mapper currently reprocesses tracks after registrations; it prioritizes
inspectability over scalability. Native feature extraction and TypeShade kernels
are not used by this CPU core. Dense depth, fusion and surface reconstruction
are separate future work. Passing tests and matching a single reference example
are insufficient evidence of production-level reconstruction accuracy.
