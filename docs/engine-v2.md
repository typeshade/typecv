# Engine v2: direction

Status: accepted, 2026-10-19 (the owner's go-ahead in the session, #32). This
document decides the shape of the next reconstruction engine. It supersedes the incremental plan where the two differ.
The owner has said that the current implementation binds nothing: a better
structure replaces it.

## What the measurements say

The measurements of 2026-10-19 (`docs/experiments/colmap-datasets/`, the lab's
COLMAP sample, the lab's run records) locate the limits of the current engine:

| Stage                        | Observation                                                                                                                                                                                                                                                                                                             | Consequence                                                                                                                                 |
| ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| Front end (corner + patch)   | On twelve south-building photographs, consecutive pairs give 12 to 43 matches with the video controls and 75 to 165 with the photo controls; a pair two frames apart verifies to zero inliers. Four of twelve views register. COLMAP's SIFT on the same images gives 923 to 1785 matches per pair and twelve of twelve. | The descriptor is the first bottleneck. No mapper change recovers views whose features do not match.                                        |
| Mapper (CPU generator + GPU) | The registered four views agree with COLMAP to 0.07° to 0.23° and 0.8% to 2.8% of the trajectory radius. The CPU path on 32 images: 84 s, 670 MB of host memory. The host model is copied after each registration.                                                                                                      | The geometry is right. The structure (host model, per-job upload and readback) is the memory and time cost.                                 |
| Numerics (emulated f64)      | The eight-frame pyramid sample registers 8 of 8 in one container and 5 of 8 on the CI runner, with identical features and pair inliers; PnP finds 17 to 19 inliers here and fewer than 12 there on the same 29 correspondences.                                                                                         | Double-float emulation on f32 hardware differs across instruction sets, and thresholded decisions amplify the difference. Fragile and slow. |
| Verification                 | Pair verification and the CPU reference agree across machines; the GPU-only refinement path does not.                                                                                                                                                                                                                   | The CPU f64 path is a good oracle and a poor product path.                                                                                  |

## Decision

Build the engine as a GPU-resident program with f32 arithmetic, a scale-space
front end and a CPU oracle:

1. **Front end: GPU SIFT.** Gaussian pyramid with octaves, difference of
   Gaussians, 3 × 3 × 3 extrema with contrast and edge rejection, sub-pixel and
   sub-scale refinement, dominant orientation from a gradient histogram, a 4 × 4 × 8
   descriptor with trilinear binning, normalization and clipping. Descriptors stay
   resident; matching is the existing tiled L2 nearest-neighbour kernel with the
   ratio test. The COLMAP SIFT keypoints and matches of the lab sample are the
   reference for repeatability and match counts, image by image.
2. **Scene: resident and index-based.** `docs/gpu-resident-scene.md` stands: one
   `ShadeDevice` owns cameras, points, observations and the two CSR orders;
   registration appends; filtering compacts on the device; the host reads
   summaries. The mapper's control flow is a small state machine over indices.
   Candidate views are scored together in one dispatch; tracks are triangulated
   together in one dispatch.
3. **Numerics: f32 first.** Coordinates are normalized (pixels by the focal length,
   the world by the map's centroid and radius) so that f32 carries them. Residual
   sums, normal-equation accumulation and the scalar products of the conjugate
   gradient use compensated (two-float) accumulation. Everything else is f32. A
   decision that a threshold makes (an inlier, a filter, a consensus minimum)
   carries a margin that the oracle comparison measures; the margin is a recorded
   number, not a hope.
4. **Oracle: the CPU f64 path.** `src/reconstruction` and `src/geometry` stop being
   the product path and become the oracle: the same requests, run on the host in
   double precision, compared on the same inputs. The differential gate runs on
   real data (the lab sample's SIFT features) as well as on the synthetic scenes.
5. **Measurement first.** The lab, its run records and the COLMAP harness stay.
   Each step is accepted by a number from them.

## What is kept, replaced and retired

| Part                                                                         | Disposition                                                                              |
| ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| `ShadeDevice`, pools, `fuse` plans, the shared scan and partial reductions   | Kept. They are the device layer.                                                         |
| Pair verification kernels (E, H, relative pose)                              | Kept; moved to f32 with normalized coordinates when the precision gate passes.           |
| P3P, triangulation, Schur/PCG BA kernels                                     | Kept as algorithms; rewritten to bind the resident scene and f32 arrays.                 |
| Corner detector and patch descriptor (`features.shade.ts`, `float-matching`) | Retired from the pipeline once SIFT registers the lab sample; the matching kernel stays. |
| `incrementalSteps`, `IncrementalSfmSession`, `src/gpu/mapper.ts`             | Replaced by the scene state machine. The CPU generator stays as the oracle driver.       |
| Lab, run records, COLMAP sample, dataset harness                             | Kept. They are the gates.                                                                |
| Public API (`compile`-side types, `Scene`, diagnostics)                      | Kept in shape; versioned when a field changes.                                           |

## Evidence gates

| Gate                                                                   | Number                                                                                                                                        |
| ---------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| SIFT front end on the twelve south-building previews                   | Consecutive-pair verified inliers within 2× of COLMAP's; twelve of twelve views register with the existing mapper                             |
| Scene engine on south-building 16 and 32 images with COLMAP's features | Same registered views as the CPU oracle; median rotation vs the pycolmap mapper ≤ 0.1°; upload bytes per registration independent of map size |
| f32 numerics on the same runs                                          | Median rotation and center agreement within the oracle's own run-to-run spread; the pyramid sample 8 of 8 on CI and in the container          |
| Whole pipeline on a physical GPU (collector records)                   | The twelve-image sample in under 2 s after the cold start on an integrated GPU of 2024                                                        |
| Memory                                                                 | Host peak on the 32-image run below 150 MB; device bytes grow with observations only                                                          |

## Order

1. Finish the lab measurement work in flight (#27, #28, #29, #30): it is the
   instrument.
2. GPU SIFT (`src/gpu/sift.shade.ts`, `sift.ts`), with the lab's photo button as the
   gate. The mapper is unchanged in this step.
3. Scene engine, steps 1 to 4 of `docs/gpu-resident-scene.md`, in f32 with
   normalized coordinates from the start; the oracle comparison on the SIFT
   features of the lab sample runs in CI.
4. Retire the corner front end and the host-model mapper from the product path.
5. Dense reconstruction waits on this engine.

## Decisions recorded with the owner

Answers given in the session on 2026-10-19, after #32 was proposed:

| Question                                           | Decision                                                                                                                                                         |
| -------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Repository structure as TypeCV grows beyond SfM    | One npm package with one entry point per area (`typecv/features`, `typecv/geometry`, `typecv/sfm`, `typecv/image`, …) over one shared device layer. No monorepo. |
| The four decisions above                           | Accepted as written. The CPU f64 path stays as the oracle.                                                                                                       |
| Order of the next work                             | SIFT first, then the resident scene engine. Image operators and physical-GPU measurement follow.                                                                 |
| The COLMAP south-building sample in the repository | Included, with the credit and the README; the dataset page names the source and no licence.                                                                      |

The present layout (`src/gpu`, `src/reconstruction`, `src/geometry`, `src/sfm`)
binds nothing: the area entry points are introduced as the engine lands, and the
old paths stay as aliases until the next major version.

## What this does not decide

- The learned-feature path (TypeNN interoperability) stays a direction, not a
  step.
- Camera models beyond pinhole with k1, k2 are out of scope until a dataset in
  the harness needs them.
