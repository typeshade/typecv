# Persistent anchored inverse depth

Follow-up to #223. This branch starts at `c1465cb` and does not include #222's Dense changes.
Depth observability and explicit optical-path selection remain independent.

## Tasks

- [x] T1: Stable observation-owned track identity, deduplicated history and independent snapshots.
- [x] T2: Anchored inverse-depth projection/Jacobians and CPU multi-view fitting, including rotation degeneracy.
- [ ] T3: Joint pose/point/focal refinement and undamped, gauge-fixed marginal uncertainty.
- [ ] T4: Persistent updates, promotion/demotion, reanchoring/scale and CPU/GPU mapper/session integration.
- [ ] T5: TypeShade measurement linearization and CPU/WebGPU agreement; no unreported CPU fallback.
- [ ] T6: Diagnostics/export contracts, negative/uncertainty tests, builds and validation receipt.

## Contract

Unknown depth is an explicit state, never a huge XYZ or a zero-variance infinity hypothesis.
Unknown translation is never replaced with a zero camera center. Repeated observation identities
are evaluated once; relinearizing a BA result does not add the same measurements as new evidence.
Raw observations, not successive correlated point estimates, are the estimator's evidence.

The point solver uses a fixed reference chart `(R_ref, C_ref)` and parameters `(a,b,rhoZ)`.
The camera projection is evaluated without division by inverse depth:

`q_i = R_i R_ref^T [a,b,1] + rhoZ (R_i C_ref + t_i)`.

The chart is a coordinate choice, not an extra pose measurement. The view originally supplying
the chart remains free to move in joint BA. At rhoZ=0 translation derivatives are exactly zero.
Public states distinguish inverse optical-axis depth from inverse range. Only validated finite
states may be materialized as XYZ or passed to PnP/MVS; bearing constraints remain separate.

Uncertainty must state its conditioning. A point-only covariance with fixed poses/K is not total
SfM uncertainty. Joint estimates use the undamped robust linearization after fixing the world
gauge and eliminating nuisance point/camera/focal parameters. These remain local Gaussian
approximations under the selected inliers and image-noise model, not calibrated probabilities.
Singular information remains unobservable; numerical damping does not count as measured evidence.

Acceptance includes pure rotation with no finite-depth promotion, later-baseline promotion of the
same tracks, duplicate-observation idempotence, calibration/pose relinearization, gauge/scale
consistency, explicit optical exclusion and numerical backend agreement. Full real-video RGB
improvements, automatic reflection detection and physical reflective/refraction models require
separate measurement and are not inferred from synthetic tests.
