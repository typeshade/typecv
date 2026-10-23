# Directional support during automatic focal calibration

Status: implementation in progress. Baseline: `e4e5046` (PR #221).

The opt-in `separateDirectionalObservations` mode can keep rotation-compatible
observations out of finite two-view structure. The focal search and subsequent
bundle adjustment still use finite landmarks only. At a plausible calibration,
unresolved directions therefore count as dropped observations; at a wrong
calibration, fabricated finite points can instead increase the candidate's
support. Joint focal refinement cannot use the missing directional residuals.

## Reproduction and scope

The committed [direction-depth experiment](experiments/general-video/direction-depth-cpu.json)
contains `estimated-focal-noise0.2`: 60 finite tracks and 60 direction-only tracks,
five views, 0.2 px synthetic noise, supplied correspondences, and an input focal
guess of 512 px rather than the evaluator's 480 px. With the experimental mode,
all five cameras register but all 60 direction-only tracks become finite points;
the maximum focal error is 28.2%. Ground-truth poses, depths and track labels are
only evaluator inputs. This is a geometry experiment, not an RGB benchmark.

A fresh baseline grid probe confirms that 0.694 and 0.798 times image width keep
60 finite points, while 0.918 keeps 120. Simply adding a post-hoc direction error
to the score does not fix this: the poses were optimized without those factors,
so the nearest grid candidate has worse directional residuals than the wrong
finite reconstruction. Direction factors must participate in the optimization;
changing the score alone is insufficient.

This stage will reuse the existing CPU and TypeShade bundle solvers. Finite
structure and unresolved directional observations must remain separate at API
and export boundaries. Direction factors may constrain orientation/calibration;
they must have zero translation derivative and cannot establish a camera center,
depth, metric scale or exact infinity. Observation compatibility remains
conditional on calibration, poses, noise and the static-scene assumption.

## Tasks and completion gates

- [x] T1: inspect the merged focal/geometry paths and existing focal PRs; reproduce
  the mixed-direction calibration failure and record the bounded design.
- [ ] T2: add a separate unresolved-direction factor path to existing bundle
  optimization, preserving finite XYZ behavior and translation observability.
  Validate the CPU reference and the TypeShade execution path.
- [ ] T3: carry compatible direction observations into focal candidate scoring
  and refinement without rewarding invented finite observations. Preserve
  pending depth and reject unsupported calibration/structure.
- [ ] T4: run regression and paired baseline comparisons, including ordinary
  samples, pure rotation, far structure, finite-only structure and noisy mixed
  observations. Record CPU/GPU backend identity, transfer costs and limitations.
  Decide defaults from measured regressions as well as improvements.
- [ ] T5: document the implemented conditions and remaining limits, finish
  validation, and update the PR's task checklist with the actual results.

Tasks are checked only after their implementation and stated validation finish.
Each completed task is committed and reflected in the PR. This stage does not
add semantic sky detection, variable zoom calibration, dynamic-object geometry,
shot/map merging, dense MVS or dynamic Radiance rendering.
