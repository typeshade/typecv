# Calibrated two-view SfM

## Inputs and options

Pass finite packed `[x1, y1, x2, y2, ...]` pixel correspondences, and camera
intrinsics `{ fx, fy, cx, cy }` with positive finite focal lengths. You need at
least eight independent correspondences, spatial coverage and adequate baseline.
The eight-point method is not a calibrated five-point minimal solver.

| Option | Default | Meaning |
| --- | --- | --- |
| `iterations` | `500` | Hard cap on RANSAC trials |
| `minIterations` | `min(64, iterations)` | Minimum trials before adaptive stopping |
| `confidence` | `0.999` | Estimated probability of sampling an all-inlier eight-point set |
| `threshold` | `0.002` | Sampson distance in normalized camera coordinates |
| `seed` | `17` | Deterministic integer PRNG seed |
| `refinementIterations` | `15` | Pose LM step cap; `0` disables pose and point refinement |
| `huberPx` | `2` | Huber transition per pixel residual component during pose fitting |
| `minParallaxDeg` | `0.1` | Minimum triangulation angle in degrees |
| `maxReprojectionErrorPx` | `4` | Maximum Euclidean error in each camera for accepted points |

A one-pixel Sampson threshold is approximately `1/fx` for similar, isotropic
intrinsics. This is not an exact conversion for anisotropic/different cameras.
Adaptive confidence is an estimate assuming independent sampling and a valid
inlier model; it does not guarantee correct pose recovery. Sampling and local
refitting use f64 CPU arithmetic.

## Refinement

The five pose degrees of freedom are rotation and two translation tangent-plane
coordinates. Rotation updates use Rodrigues' formula and translation stays unit
length. A variable-projection optimizer re-triangulates and refines each 3D point
for candidate poses. It solves damped least squares using numerical Jacobians
and Huber weights. Point refinement minimizes squared pixel residuals with the
pose fixed. This implementation favors a small dependency-free two-view solver;
it is not a scalable multi-view bundle adjustment implementation.

Each LM step must decrease its objective. The final refined pose is accepted
only when mean per-view Euclidean pixel error on the SAME initial positive-depth
observations does not increase. Points are held to the same observation set for
this comparison: cost cannot improve simply by discarding difficult matches.
An accepted pose regenerates `E = [t]x R` (normalized to unit Frobenius norm),
then recomputes epipolar consensus against that E.

## Results

| Field | Meaning |
| --- | --- |
| `E` | Row-major essential matrix, `x2^T E x1 = 0`, for normalized camera coordinates |
| `R`, `t` | Relative pose, `X2 = R X1 + t`, with unit translation |
| `inliers`, `inlierCount` | Epipolar consensus for the final E |
| `validPoints` | Final mask: consensus, positive depth, adequate angle and acceptable pixel errors |
| `points` | Camera-1 XYZ triplets, NaN for rejected points |
| `positiveDepthCount` | Number of final accepted points, after all quality filters |
| `meanReprojectionErrorPx` | Mean per-view Euclidean pixel error over the final accepted points |
| `diagnostics.ransacIterations` | Actual RANSAC trial count |
| `diagnostics.refinementApplied` | Whether a refined solution was accepted (including an already-optimal solution) |
| `diagnostics.medianParallaxDeg` | Median angle over positive-depth final consensus, before quality filtering |
| `diagnostics.meanReprojectionErrorBeforePx` | Initial DLT error on the optimization observation set |
| `diagnostics.meanReprojectionErrorAfterPx` | Accepted refined error on that same set (or initial error on fallback) |

The optimizer diagnostics and final cloud error can involve different observation
sets. Use `validPoints` when consuming XYZ. A Sampson inlier alone is not evidence
of reliable triangulation. Reconstruction throws if median parallax is too small
or fewer than six points survive quality filtering.

## Degeneracy and limitations

An eight-point rank check rejects ambiguous exact systems, including the tested
noiseless planar/collinear scenes. Low median triangulation angle also rejects
insufficient baseline. These checks do not implement homography-versus-essential
model selection: noisy planar scenes and pure rotation with noise can remain
ambiguous. Noisy data passing these checks is not proof of observable translation.

The current scope excludes feature detection/tracking, distortion correction,
five-point estimation, homography model selection, multi-view tracks, PnP,
multi-view bundle adjustment, loop closure, scale anchoring and intrinsic fitting.

## Validation

The 15 deterministic CPU tests cover clean/noisy reconstruction, outliers,
multiple RANSAC seeds, different anisotropic intrinsics, stopping bounds,
repeatability, SO(3)/unit-translation/E consistency, same-observation refinement
improvement, tiny baseline, exact rotation/planar/collinear degeneracies, masks
and invalid controls. Type checking and ESM/declaration generation are also
checked. CI runs these checks on Node 22 and 24.

In the 160-correspondence fixture with 0.28-pixel uniform noise amplitude,
18% outlier probability and RANSAC seed 71:

| Metric | Without refinement | With refinement |
| --- | --- | --- |
| Mean reprojection error | 0.52115 px | 0.10645 px |
| Accepted points | 136 | 136 |

With refinement, rotation error is 0.06043 degrees and translation-direction
cosine is 0.999984. Adaptive RANSAC takes 64 trials against a cap of 650. This
fixture is not a runtime benchmark or real-camera accuracy evaluation.
