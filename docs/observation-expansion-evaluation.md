# Comparing observation expansion without changing the evaluation population

`benchmark-observation-expansion.mjs` keeps its original aggregate camera/depth
metrics and adds `cohortComparison`. This additional evaluation runs in Node after
reconstruction. Renderer labels, reference poses and rendered depths are never
sent into the browser's feature extraction, tracking, admission or SfM calls.

Aggregate depth medians alone compare different populations: expansion can improve
old geometry while adding weak new landmarks, or drop difficult old points while
leaving an apparently better aggregate. Registration can also grow from two
cameras to four, making two camera-RMSE values incomparable without shared frames.
The evaluator reports these effects separately.

## Landmark correspondence

`landmarkCohorts` identifies an observation by its stable `viewId:featureId` key.
Track IDs may change during reconstruction; nearest XYZ is not used to match
points. Each expanded landmark is linked to the baseline landmarks whose original
observations it contains.

- A retained pair must have exactly one link in each direction.
- A baseline landmark with no surviving observation ownership is counted as lost.
- Expanded landmarks with no baseline finite-landmark observations form the new
  cohort. They can originate in formerly pending tracks; this is not proof of a
  previously unseen physical point under repeated appearances.
- Splits and merges are reported as ambiguous ownership. They are excluded from
  paired depth statistics, not silently matched by their best geometric fit.

For every retained pair, both estimated XYZ values are evaluated at the **same
original baseline frame and pixel**. The anchor is the baseline observation with
smallest `(viewId, featureId)`, irrespective of array ordering, photometric quality,
or estimated error. Even if the expanded map dropped that particular observation,
the same original reference pixel is retained for evaluation. Reference depths
that cannot be evaluated remain explicitly unavailable.

Depth is camera +Z, sampled at the nearest rendered pixel. Statistics include
count, invalid/unavailable counts, median, p90, RMSE and the fraction exceeding
10% relative depth error. The retained cohort includes before/after statistics
and per-pair better/worse/equal counts. New and lost cohorts have separate depth
statistics. Quantization and surface-boundary errors remain included; this is not
source point identity or a calibrated uncertainty model.

## Camera comparison

`cohortComparison.pose.sharedViewIds` is the intersection of registered camera
IDs. Both pose errors use that exact population. Added and lost camera IDs are
reported separately, so newly registered difficult cameras do not silently change
the before/after accuracy population.

For rendered scenes, each map uses one fixed truth seed pose and baseline gauge,
matching the original evaluation convention. No per-camera or short-segment
alignment is fitted.

For reference-photo scenes, the evaluator fits one Sim(3) to the baseline's whole
registered map, using camera orientations for rotation and camera centres for
scale/translation. When refinement retains the same seed tuple and first anchor pose and baseline magnitude
(within `1e-8`), that **same baseline transformation** is frozen for both models.
The reference trajectory radius is also held fixed. Post-refinement fitting
cannot absorb a new drift or position error. The fitted baseline view IDs are
recorded as `alignmentFittedViewIds`.

The explicit rebuild control can choose a different gauge. It receives one whole
map fit per model, then evaluates shared cameras, and labels this distinction in
the result. It is not presented as a frozen-gauge comparison. Original separately
aligned aggregate `poseReference` metrics remain available for continuity with
previous reports; their interpretation differs from the frozen comparison.

Unlabelled video reports ownership and registration populations but leaves depth
and reference-pose accuracy unavailable. An absent finite model is likewise an
unavailable comparison, not zero error.

## Tracking controls and verification

The benchmark forwards explicit `--patch-warp=0|1`, `--pyramid-cache=0|1`,
`--surface-warp=0|1` and `--multi-view=0|1` values
to `geometryPatchWarp` and `pyramidCache`, and records them in `trackingOptions`.
Defaults are `--patch-warp=0 --pyramid-cache=0 --surface-warp=0 --multi-view=0`; comparisons should change one
variable at a time and write to a new `--output` path to preserve earlier evidence.
These switches do not alter how renderer/reference labels are isolated.

`tests/observation-expansion-evaluation.test.ts` verifies fixed original pixels,
cohort separation, split/merged ownership, shared-frame camera populations,
frozen reference alignment under a nontrivial Sim(3), distinct rebuild gauges,
and unavailable labels. The numerical assertions expose an inaccurate added
camera and weak new points without contaminating the retained cohorts.

Summary medians use the upper sample median for an even population. The older
`poseReference` summary retains its existing averaged-median convention; compare
the before/after fields within one evaluator rather than mixing those conventions.
