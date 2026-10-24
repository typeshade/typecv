# Image-based expansion of sparse observations

Status: implementation in progress. Baseline: `e4e5046`.

The existing `ImageFeatures.keypoints` accepts continuous measured coordinates;
the core mapper does not require independent detector output. Existing KLT has
pyramids, forward/backward checking, NCC, spatial fill-in and fresh SIFT seeds at
joins. Its current join emits pairs only when a track reaches an existing SIFT
keypoint. Fill-in therefore does not by itself register detector-missed coordinates
as SfM observations. The pair LK API also starts its search at zero displacement.

This stage reuses those kernels and the mapper's geometric verification. It adds
an initial-target option and an explicit batch adapter that starts tracks from
multiple input frames, searches both earlier and later neighbours, and appends
image-measured coordinates while preserving all original feature indices.
Estimated rotations or finite geometry may initialize a search. Predictions are
never emitted directly as observations. Patch/round-trip checks remain mandatory;
the mapper must subsequently verify new pair candidates before accepting structure.

The first adapter uses a bounded frame neighbourhood, existing per-frame seeds,
and the existing translation-patch LK model. General affine patch warping,
occlusion inference, learned tracking, arbitrary track merging, and streaming
retroactive session updates are separate work. This does not implement Dense-SfM
or claim equivalent results. PR #148 owns photo pair selection and remains separate.

## Tasks

- [x] E1: audit existing KLT, join and mapper ingestion; compare numerical-only
  refinement with missing-observation recovery and identify the gap.
- [x] E2: add predicted initial coordinates to existing CPU/GPU LK, preserve the
  default path, and test that image evidence overrides or rejects predictions.
- [ ] E3: add multi-source, forward/backward observation expansion with append-only
  indices, ownership/duplicate checks, provenance and explicit candidate geometry.
- [ ] E4: compare original/new observation counts, track lengths, distinct landmarks,
  coverage and accuracy on RGB-derived input; test occlusion, textureless regions,
  wrong predictions, rotation and insufficient parallax on CPU and WebGPU.
- [ ] E5: finish documentation, execution/memory evidence and task/PR updates.

No source pose/depth truth is supplied to RGB-only reconstruction. Known-K and
estimated-K experiments must remain distinct. A longer track is not a new landmark,
and a point propagated under pure rotation still has unresolved depth.

E2 validation: 15 scalar-reference tests, 15 TypeShade CPU tests (including default
bit equality), and one SwiftShader WebGPU prediction/chunk-tail test pass. A one-level
tracker recovers a 30 px translation from an inexact prediction; flat pixels and
out-of-frame predictions are rejected. Forward/backward buffers also hold the
initial guesses, so no additional resident point array is allocated.
