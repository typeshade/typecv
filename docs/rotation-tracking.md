# Calibrated rotation evidence and deferred structure

This is the first implementation after the [gameplay/cinematic audit](general-video-reconstruction.md).
It adds an **opt-in** path for relative rotation and 2D observations when a pair cannot seed
structure. The baseline is `737f0c5`; the source and runtime hashes of this working-tree change
are recorded in [validation](experiments/general-video/rotation-tracking-validation.json).
Default mapper behavior and the Lab UI remain unchanged.

## What a rotation result means

`estimateRelativeRotation(matches, firstCamera, secondCamera, options?)` takes calibrated
pixel correspondences `[x1,y1,x2,y2]` in a `Float64Array`. It returns a rotation-compatible
hypothesis or `undefined` when consensus or image coverage is insufficient.

- `R` is row-major and maps bearings in the first camera to bearings in the second.
- `inlierRows` addresses the original raw rows, including gaps from failed radial inversion.
- `translation: 'unobservable'` and `depth: 'not-estimated'` describe this rotation model.
  There is no `t`, camera center, XYZ, depth value or world trajectory in this result.
- A good fit also occurs with sufficiently distant structure or small translation. It is
  **not a classifier proving that the camera center was stationary**. Homography dominance
  alone is never used as proof of pure rotation.

The implementation reuses `unprojectPixel`, the existing proper-rotation `svd3`, seeded
sampling and RANSAC budget math. Three bearing pairs propose SO(3); all consensus bearings
refine it. Both forward and backward projections must have positive Z and remain within
the error threshold. Returned error is the mean of the larger directional pixel error per
inlier, in ideal pinhole pixels. No external library or new compute layer is introduced.

Defaults are 128 trials, 3px directional error, 12 inliers, 60% of invertible rows, 0.999
sampling confidence and 0.5° minor-axis bearing spread. Spread is `atan(stddev)` of the
calibrated image coordinates' minor covariance axis; it is a coverage guard, **not a pose
uncertainty or calibrated confidence**. Poorly distributed matches can remain unresolved.

## Mapper and session use

```ts
import { reconstructIncremental, ReconstructionInitializationError } from 'typecv';

try {
  const result = reconstructIncremental(images, pairs, { rotationTracking: true });
  // result.reconstruction contains only fully registered poses and accepted landmarks.
  // result.tracking holds separate 2D/relative-rotation evidence and pending track ids.
} catch (error) {
  if (!(error instanceof ReconstructionInitializationError)) throw error;
  // error.tracking can contain valid relative rotations and 2D tracks, without a map.
}
```

`rotationTracking` also accepts `RelativeRotationOptions`. It defaults to disabled.
For each seed-ineligible pair, the mapper fits calibrated rotation **in addition to** its
existing E/H verdict. It does not promote that pair's `seedEligible` status. `PairVerification`
adds `rotationInliers` and, for an accepted rotation hypothesis, `rotationErrorPx`.
Pre-verified E/H verdicts are reused; the opt-in rotation fit still examines their raw rows.

`tracking.pairs` unions epipolar and rotation-compatible observations, preserving feature
ownership and distances. The existing conflict-safe `buildTracks` consumes these edges.
`verifiedPairs` retains its original epipolar-only meaning. Track construction is not proof
that a point belongs to the static background. A large moving foreground can still dominate.

`tracking.rotations` contains pair ids, the supplied camera snapshots used for each fit,
and the relative rotation hypothesis. These calibrations precede any later focal BA;
the metadata is not silently reinterpreted using refined intrinsics. `tracking.tracks`
contains the active 2D graph. `untriangulatedTrackIds` names tracks with no accepted landmark.
It does not assign a reason-specific depth estimate, variance or probability.

```ts
import { IncrementalSfmSession } from 'typecv/gpu';

const session = new IncrementalSfmSession(device, { rotationTracking: true });
const mapUpdate = await session.add(image, newPairs);
const evidence = session.tracking; // Independent snapshot, also available before a first map.
```

`add()` still returns `undefined` until a stable map exists. A rotation-only `finish()` still
throws `ReconstructionInitializationError`, now with optional tracking evidence. A later
baseline can seed a map through the retained observations. Existing PnP registers views
from real map points; existing triangulation checks angle, positive depth and reprojection.
The relative-rotation hypothesis is never passed to triangulation as a full pose.

After motion, a point must actually be reobserved to acquire new depth evidence. A different
point becoming triangulated does not determine the old point's depth. Returned pending ids
are **local graph ids**, which can change as edges arrive; identify continuity through
`(viewId, featureId)` observations. Session evidence is limited to `activeViewIds` and the
existing 8–64-view window. This does not archive unresolved 2D observations indefinitely.
Returned arrays and calibration snapshots cannot mutate the next session update.

## Execution boundary and verification

The rotation fit uses the same existing f64 CPU math in the synchronous mapper, scene
mapper and GPU driver. GPU execution explicitly reports this in `execution.cpuStages`.
The fit consumes raw host match coordinates already available to mapper control; it adds
no GPU upload or readback of its own. Retaining useful tracks may lead to more legitimate
PnP/triangulation/BA work later. This is not a GPU rotation kernel or acceleration claim.

Core regressions cover noisy/outlier correspondences, radial inversion, known zoom,
incorrect calibration, poor coverage, translated planes and distant structure; rotation-only
deferral; a rotation prefix followed by a baseline; no-reobservation versus reobservation;
pre-verified rejection and default-mode compatibility. TypeShade CPU and browser WebGPU
tests compare relative rotations exactly with the f64 oracle, exercise session recovery,
snapshot ownership, verification caching, window eviction and buffer disposal.

The [RGB probe](experiments/general-video/rotation-tracking-rgb-software.json) uses only
PNG RGB and explicitly known per-frame K as inference inputs. Its evaluation-only poses
remain outside the browser arguments. It retains the audit probe's pyramid FAST/BRIEF
settings; it does not replace the Lab's SIFT/KLT frontend. It obtains:

| Static analytic room input | Registered frames / new points | Relative rotation hypotheses | Pending tracks | Max relative rotation error |
| --- | --- | ---: | ---: | ---: |
| Translation | 5/5 / 234 | 0 | 1 | unavailable |
| Fixed-center rotation | 0/5 / 0 | 9 | 357 | 0.162° |
| Fixed-center rotation + known zoom | 0/5 / 0 | 4 | 141 | 0.074° |

These are diagnostic synthetic results on **Google SwiftShader software WebGPU**. They
do not establish hardware performance or real gameplay accuracy. End-to-end wall times,
stage timings, transfer counters, managed allocation peaks and disposal state are in the
raw record. CPU validation ran concurrently; timings are not speed comparisons.

The completed checks are 273 core tests (including eight new tests), 12 affected TypeShade
CPU tests, the new WebGPU session test and the existing default-mode GPU mapper test.
Type/format checks, core/GPU builds, Astro check and the 79-page website build plus static
links pass. Full TypeShade/WebGPU suites and the full UI suite were not rerun; the exact
commands and limits are in the validation record.

```sh
npm run build
node --test tests/rotation-tracking.test.ts
node --test tests/typeshade/rotation-tracking.test.mjs
TYPESHADE_CHROMIUM=/usr/bin/chromium node --test tests/typeshade/rotation-tracking-webgpu.test.mjs
node scripts/benchmark-general-video.mjs --render-dir=data/general-video-fixtures --output=/tmp/general-video-geometry.json
TYPESHADE_CHROMIUM=/usr/bin/chromium node scripts/probe-general-video-rgb.mjs \
  --rotation-tracking=true \
  --output=docs/experiments/general-video/rotation-tracking-rgb-software.json
```

## Remaining boundaries

Automatic focal search and `rotationTracking` cannot be combined in this stage: supplied
calibration is required, and a wrong supplied K can bias or reject the hypothesis. Known
per-view K changes are supported; estimating unknown continuous zoom or scope projections
is not. The relative hypotheses do not add a global orientation optimizer, uncertainty
calibration, automatic cuts, relocalization, dynamic-object classification or nonrigid
reconstruction. No native GPU or real gameplay/cinematic clip was tested. These remain
later stages of the audit plan, including archival deferred observations beyond a window.
