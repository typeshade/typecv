# Observation expansion: patch geometry and resident image reuse

Follow-up to the first batch observation-expansion stage at `6cd3ff0`.
This stage separates three questions: measured-coordinate precision under rotation/zoom,
unchanged measurements under cache reuse, and geometry quality of the same landmark cohort.

- [x] U1: Audit centre-only patch bias, repeated pyramid uploads/readbacks and changing evaluation populations.
- [x] U2: Add local affine patch tracking, independent image-quality checks and calibrated rotation/zoom regressions.
- [x] U3: Add bounded resident pyramid reuse, selective readback, parity and lifetime/budget tests.
- [x] U4: Compare identical RGB inputs, retained/new landmark cohorts, degeneracy and real/video samples.
- [x] U5: Record the defaults decision, costs, limits and final checks.

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


## Measured results and defaults

The numerical implementation is `c061178`; the comparison base is the previous
batch implementation at `6cd3ff0`. All GPU runs identify Google SwiftShader,
`fallback=true`. They do not measure physical GPU performance.

Controlled patch tests supply an affine model and deliberately inexact initial
centres to both methods. These are tracking tests, not RGB-only SfM claims.
The options are two pyramid levels, half-window 5, epsilon0.005 target pixels and
30 iterations. Forward/backward checks and image NCC remain active.

| Controlled warp | Translation-only accepted /12 | Affine accepted /12 | Translation max error | Affine max error |
| --- | --- | --- | --- | --- |
| {'degrees': 12} | 12 | 12 | 10.7940 px | 0.0182 px |
| {'degrees': 25} | 1 | 12 | 0.6715 px | 0.0261 px |
| {'scale': 1.2} | 12 | 12 | 0.6530 px | 0.0321 px |
| {'degrees': 12, 'scale': 1.15} | 12 | 12 | 0.6482 px | 0.0285 px |

The 12-degree translation-only maximum includes a repeated-pattern false match
that passed its round-trip check. Tight convergence and a good centre guess alone
do not correct the wrong patch model. Maxima describe accepted sets; the 25-degree
unwarped set has only one point, so these are not equal-population accuracy scores.
All twelve warped points meet the absolute 0.05 px tracking criterion.

For RGB, reference/resident/affine modes run the same original images and supplied
calibration. The first two differ only in resident pyramid lifetime; the final
mode additionally enables geometric patch warps.

| RGB input | Prior expanded points | With affine patches | Prior finite observations | With affine patches |
| --- | --- | --- | --- | --- |
| render-translation | 255 | 250 | 606 | 597 |
| render-translation-omission | 170 | 169 | 399 | 397 |
| render-rotation | 0 | 0 | 0 | 0 |
| render-low-parallax | 0 | 0 | 0 | 0 |
| graham-hall-cached-rgb-features | 2471 | 2471 | 5899 | 5893 |

For rendered translation, the **same 141 retained baseline tracks**, evaluated at
the same original pixels, have median relative depth error 1.8021% with the prior
expansion versus 1.7251% with affine patches. Shared-frame camera-centre RMSE is
0.006605 → 0.006370 scene units. The unexpanded seed map's retained-track median is
1.5637%, so neither expanded model improves every metric against unexpanded SfM.

Under controlled detector omissions, the same 21 retained tracks have depth median
1.2548% → 1.2885%, a small regression. Pure rotation and low-parallax RGB retain
zero finite landmarks and deferred initialization in every tested mode. Graham
Hall retains 2,471 landmarks in both expanded modes; its fixed-alignment camera
RMSE changes only 0.575067 → 0.575040 COLMAP units. Its unexpanded baseline is
0.574956, so this is not evidence of a general accuracy gain over plain SfM.

The six-frame Shorts window uses the same manual background mask and assumed K
as the first stage. It retains all 202 baseline tracks. Prior expansion gives 471
finite landmarks / 1,253 observations; affine expansion gives 470 / 1,252. No pose or
depth truth exists for this clip. The new patch model does not establish improved
3D accuracy merely by changing its support count.

### Transfer and storage tradeoffs

On Graham Hall, resident reuse gives **identical** measured/reconstructed results
to stateless tracking: 2,471 points and 5,899 observations. It builds 12 pyramids
instead of 44 and reduces application upload 64,052,752 → 17,563,152 bytes (72.6%).
It retains a third image: peak pyramid storage 13,075,200 bytes versus roughly
8.7MB for the stateless pair, plus per-pair scratch. Budget admission may evict an
unprotected frame sooner; tests exercise a two-frame budget with a three-frame cap.

The single cold SwiftShader expansion timing is 4.857 s stateless versus 5.391 s
resident. This is a latency regression despite fewer transfers, so **pyramid reuse
stays opt-in**. No hardware speedup is inferred from upload counts.

The adapter now omits only the unused backward *output read*. The GPU still runs
backward tracking and the full round-trip check. With translation-only patches,
Graham Hall result snapshots/read calls reduce 129,552bytes/66calls in the prior
stage to 86,368bytes/44calls; measurements remain unchanged. This omission is the
only newly active adapter behavior. Standalone tracker readouts keep their defaults.

Affine tails raise the actual forward snapshot size: the affine Graham Hall run
reads 86,368 application-prefix bytes but 129,552 full Resident snapshot bytes.
Counters are not physical bus measurements. Host images, browser/driver memory,
staging and CPU geometry peak memory are not included in device peaks.

**`geometryPatchWarp` and `pyramidCache` both default to false.** The first needs
reliable calibration/rotation and more consistent map-quality evidence; the
second needs hardware/latency evidence and has a memory tradeoff. The Lab's
streaming/default reconstruction path is not changed. Unsupported or ill-conditioned
Jacobians fall back to an image-verified identity patch, whose actual matrix is
recorded; strong zoom, projection boundaries and translated tilted surfaces are
not claimed as solved.

### Reproduction and validation

```sh
npm run build
TYPESHADE_CHROMIUM=/usr/bin/chromium node scripts/benchmark-observation-expansion.mjs --pyramid-cache=0 --patch-warp=0 --output=/path/reference.json
TYPESHADE_CHROMIUM=/usr/bin/chromium node scripts/benchmark-observation-expansion.mjs --pyramid-cache=1 --patch-warp=0 --output=/path/resident.json
TYPESHADE_CHROMIUM=/usr/bin/chromium node scripts/benchmark-observation-expansion.mjs --pyramid-cache=1 --patch-warp=1 --output=/path/affine.json
TYPESHADE_CHROMIUM=/usr/bin/chromium node scripts/measure-klt-patch.mjs
```

The existing video flags reproduce the manually masked window. Uploaded pixels
remain local; reports contain hashes and numeric results. Small CPU/WebGPU tests
cover affine/legacy parity, target-pixel convergence, chunk tails, masks, cache
replacement, budget admission, allocation failures and queued/idempotent disposal.
The RGB WebGPU regression now enables affine patches and checks pose/depth,
registration and deferred-depth outcomes on four rendered cases.

Final checks: 312 core tests (including 8 cohort-evaluator tests), 27 affected
TypeShade CPU tests, 8 KLT/cache/expansion WebGPU tests, and 1 four-case affine RGB
WebGPU test pass. Core typecheck, declarations/build, formatting, TypeShade runtime
audit and a 93-page site build pass. The validation receipt records source/report
hashes and measured limitations. Current combined PR222/223 heads, physical GPUs,
full-clip automatic processing and general dynamic reconstruction remain untested
in this bounded follow-up.
