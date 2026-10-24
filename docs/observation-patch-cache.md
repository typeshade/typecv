# Observation expansion: patch geometry and resident image reuse

Follow-up to the first batch observation-expansion stage at `6cd3ff0`.
This stage separates three questions: measured-coordinate precision under rotation/zoom,
unchanged measurements under cache reuse, and geometry quality of the same landmark cohort.

- [x] U1: Audit centre-only patch bias, repeated pyramid uploads/readbacks and changing evaluation populations.
- [x] U2: Add local affine patch tracking, independent image-quality checks and calibrated rotation/zoom regressions.
- [x] U3: Add bounded resident pyramid reuse, selective readback, parity and lifetime/budget tests.
- [ ] U4: Compare identical RGB inputs, retained/new landmark cohorts, degeneracy and real/video samples.
- [ ] U5: Record the defaults decision, costs, limits and final checks.

## Model and reuse

The existing inverse-compositional KLT solver is reused. A supplied 2×2 matrix A
maps source patch offsets into the target patch. Displacement is solved in source
patch coordinates: the target is sampled at `sourceCentre + A(offset + d)`, and the
measured target centre is `sourceCentre + A d`. Source gradients therefore remain
the derivatives with respect to d; treating target-pixel displacement as source
pixel displacement without the affine chain rule would be incorrect.

Forward and backward tracking use A and its inverse, respectively. NCC is measured
on those same image patches, including the host's competing-peak check. Target
footprints must fit the image and caller's valid mask. Translation-only calls keep
the old arithmetic path. Matrices must be finite, bounded, invertible and orientation
preserving. Invalid projections are deferred; no projected coordinate becomes a
measured observation on its own.

`geometryPatchWarp` derives a local Jacobian from the estimated relative rotation
and the two calibrated projection models, with radial distortion included. It uses
one-pixel centred differences. This accounts for local roll/zoom and perspective
variation of rotation; it does not infer a translated surface's normal, depth
gradient or a general scene homography. Strong translation across tilted surfaces,
parallax boundaries, deformation and an incorrect calibration can still violate
that local model. Each new observation can retain the matrix actually used.

The internal `createKltPairCache` keeps at most three frame pyramids/gradients in an
LRU. Dimensions, pixel-array identity and pyramid levels determine reuse. Image
contents must stay immutable while borrowed. Replacement arrays invalidate entries.
Eviction also respects actual device working-storage admission; unprotected frames
are released before a new image plus point scratch would exceed the budget. Two
images needed by a pair remain protected. Insufficient capacity for that pair is
an explicit error, not a fallback to another backend.

The batch adapter owns and disposes its cache in `finally`; original frame evidence
stays on the host. The streaming tracker is unchanged. `pyramidCache:false` selects
the stateless comparison path. The adapter skips the backward output read, since
its quality checks need the checked records and forward NCC. Default standalone
`trackKltGpu` readouts remain unchanged.

Affine matrices are packed in tails of the existing start and forward arrays;
there is no additional storage binding. The backward input's tail contains the
inverse warp. Reading the forward Resident array snapshots its tail as well, so
application prefix bytes and actual Resident snapshot bytes are reported separately.
With affine tails enabled, selective readback alone does not imply a one-third
reduction in total snapshot bytes.

## Evaluation contract

Existing overall camera/depth summaries remain available. The follow-up evaluator
also matches landmarks by original observation ownership and reports retention,
losses, merges and splits. Retained depth estimates are compared at the same original
baseline frame/pixel, using the same whole-map gauge. New landmark tracks are scored
separately. Shared-frame camera errors use a fixed map alignment; they do not benefit
from refitting a separate alignment after refinement. See
[the evaluator contract](observation-expansion-evaluation.md).

Ground-truth pose/depth never enters RGB reconstruction. Synthetic known-K,
COLMAP-supplied calibration and the video window's assumed K/manual mask remain
separate experiment modes. A count gain, image similarity or lower reprojection
error alone is not a geometry-accuracy claim.

U2/U3 local verification: 27 TypeShade CPU tests and eight SwiftShader WebGPU tests
pass, including unchanged legacy tracking, affine/reference parity, cache parity,
budget-driven eviction and cleanup. The core suite has 312 passing tests.
