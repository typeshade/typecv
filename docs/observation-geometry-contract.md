# Observation geometry and optical paths

Depth observability and optical path are independent. A direct mountain track may constrain
rotation while its depth is unknown. A reflected mountain track can have exactly the same
observability, without becoming evidence for a physical mountain below the water surface.
Neither low NCC cost nor low reprojection error identifies the optical path.

## Current implementation and boundaries

| Requirement | Implemented path | Remaining work |
| --- | --- | --- |
| Keep untriangulated observations | `ReconstructionTracking.tracks`, pending/depth-deferred ids | Persistent application identities should use `(viewId, featureId)`, not local track ids |
| Rotation without fabricated translation | Calibrated `estimateRelativeRotation`, `PairRotationEvidence`; explicit `translation: unobservable` | Focal-search integration and directional BA are #223's scope |
| Reconsider depth after later baseline | Pending tracks, conditional triangulation sensitivity and re-triangulation | A maintained anchored inverse-depth posterior with pose/K uncertainty is **not implemented here** |
| Separate reflection/refraction from ordinary geometry | Explicit `ObservationPathMask`, camera selection and Dense/planar exclusion | Automatic masks, extraction-time descriptor rejection and application/report plumbing |
| Preserve rendering evidence | RGB and original feature ids remain unchanged; separated pair rows retain ownership | Radiance export of optical paths, uncertainty and reflection planes is **not implemented here** |

The existing depth sensitivity is conditional on fixed poses and intrinsics. It is not a
calibrated full SfM covariance. The MVS directional-cost comparison withholds an ambiguous
finite depth; it neither creates a bearing landmark nor maintains an inverse-depth posterior.
No item in this PR should be described as completing the entire architecture above.

## One explicit optical-path policy for SfM and Dense

`ObservationPathMask.values` uses `0` for direct/static, `1` for reflection, `2` for refraction
and `3` for excluded/unknown. These are supplied labels with `source` provenance, not detector
output inferred from a small residual. Labels occupy the **undistorted analysis-pixel lattice**.
A mask from the original distorted RGB must be resampled conservatively to that lattice first.
Padding for uncertain boundaries must be included in the supplied pixel mask. Camera descriptor
support can additionally be excluded with `paddingPx` in `cameraMaskFromPixelPaths`.

```ts
const observationMask = { width, height, values, source: 'manual' as const };
const cameraObservationMasks = features.map((image) =>
  cameraMaskFromPixelPaths(image, observationMask, descriptorPaddingPx),
);
// Pass cameraObservationMasks into reconstructIncremental / the streaming session.
// Pass the same per-view observationMask into its MvsImage for reconstructDenseGpu.
```

Each view needs its own aligned mask. The caller must pass it to **both** stages; a feature mask
alone does not automatically create a Dense mask. `cameraMaskFromPixelPaths` undistorts feature
coordinates using that image's calibration, retains original feature ids and returns the existing
camera-selection mask. Selection occurs before E/H/R verification, tracks, PnP and BA; excluded
pair rows remain separately available. It does not remove descriptors already extracted from RGB.

Dense snapshots labels before queued work. An explicit mask invalidates gray samples after
undistortion; source bilinear interpolation already rejects any invalid tap. Photometric and
geometric host maps, diagnostics and supplied filtered maps exclude those pixels before their
use as geometric evidence. Planar completion uses only frozen, masked source maps and clears
forbidden fills before reporting or fusion. Fills never validate other fills. Original RGBA is
preserved for archival and a future optical renderer. An explicit mask adds one dispatch per
uploaded view, uses the existing float staging buffer and adds no image-sized GPU allocation;
the input snapshot and float upload do consume host memory. Absent masks keep the original path.

The synthetic reflection regression deliberately supplies ground-truth labels and matching
identities. A fixed y=0 mirror maps `(x,y,z)` to `(x,-y,z)`; the virtual points have positive
camera depth and tiny ordinary triangulation residuals, and an unmasked incremental map accepts
them. Explicit masks retain the direct map and separate reflected rows. This proves why ordinary
geometric consistency alone is insufficient. It is not an RGB-only water detector or a real-video
reflection result. CPU and WebGPU tests also cover mask propagation, grouped depth processing,
RGB preservation and forbidden planar fills.

## Constraints for the next inverse-depth state

Anchor a track to a calibrated unit bearing `b` and use inverse **range** `rho=1/D`, with history
and uncertainty. If using the existing pinhole ray with z=1, its inverse optical-axis depth
`1/Z` differs by the ray norm; the two must not be silently interchanged. At `rho≈0`, a bearing
may constrain rotation, but unknown depth is not an assertion that the point stays at infinity.
Do not encode unknown depth as a huge finite XYZ or unknown translation as a zero camera center.

Evaluate rotation-compensated bearing discrepancies against feature, rotation and calibration
uncertainty, not raw optical flow. Test rotation, essential and planar hypotheses separately;
a strong homography alone does not prove pure rotation. Rotation-compatible evidence must not
authorize a 5-point translation or a seed baseline. Existing reliable finite landmarks can
still support PnP during a rotation segment. A rotation loop constrains accumulated orientation,
without asserting that camera centers coincide.

Use adjacent observations for tracking and longer overlapping tracks for informative baseline.
Reconsider deferred depth as new observations arrive; promote only after positive depth,
multi-view consistency, parallax and uncertainty checks. Joint BA should retain the uncertainty
and allowed degrees of freedom rather than permanently fixing rotation from far points. Pure
rotation must preserve directions and unknown depth; later translation must recover depth from
the same retained observation identities. #223 and this change require a four-way old/new SfM ×
old/new MVS replay before claiming combined RGB quality gains.

Fixed planar reflections can later use `projection(P * S_plane * X)` with explicit optical paths.
The reflection transform has determinant -1 and must never be treated as a quaternion/SE(3)
rotation. Do not fit the water plane to virtual reflected structure or assume the reflected
scene is planar because the mirror is planar. Waves, refraction, SSR/capture approximations,
rolling shutter, time alignment, gimbal transforms and changing crop/focal models require
separate validated models. No threshold in this PR supplies those missing observations.
