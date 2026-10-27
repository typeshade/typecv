# Automatic shared focal for uncalibrated input: design and measurements

Status: accepted (2026-10-21), under the owner's approval of every merge of this session. The
implementation is in the same pull request (#162).

This record is written for an engineer or an agent who has not read the conversation that led to
it. Every path and symbol named here exists on the branch `feat/auto-focal`. Every number was
measured on 2026-10-21 in a container with Node.js 22.22.0 and no GPU, unless the row says
otherwise.

Labels used below. **Fact**: read in the code or measured. **Inference**: reasoned, not measured.
**Decision**: a choice this record makes. **Open**: a question still to answer.

## 1. Problem

Fact: for an uncalibrated video the lab asked the visitor for "focal length / image width"
(`DEFAULT_VIDEO_FOCAL` = 0.8 in `website/src/components/SfmLab.tsx`). The final bundle adjustment
refines one shared focal scale (`refineSharedFocal`, `src/sfm/bundle/joint-focal.ts` and
`src/sfm/bundle/gpu/joint-focal.ts`). On the owner's record `20261021062057594-4a8aa2ff` (the pyramid video
sample, 640 × 360, 22 registered views) the focal moved from 0.80 to 0.81.

Goal (the owner, 2026-10-21): the system estimates the focal by itself, independent of the start
value. The typed value stays only as an advanced override.

## 2. Why the refinement barely moves

### 2.1 Method

`scripts/measure-focal.mjs` runs the CPU mapper (`reconstructIncremental`) with the lab's
controls (the lab preset, `LAB_MAPPER_NUMBERS` of `src/sfm/config/lab-preset.ts`, which
`LAB_OPTIONS` of `website/src/lib/lab-options.ts` holds: 500 RANSAC trials, 3 px, 4 BA
iterations during mapping, 8 in the final BA) on the lab's samples:

- `south-building`, `gerrard-hall`, `person-hall`, `graham-hall` (`website/public/sample/`): 12
  photographs each, COLMAP's own SIFT keypoints and raw matches (`colmap-sift.bin`), COLMAP's
  calibration as the truth (`sample.json`). Every view starts at `fx = fy = start × width`. The
  principal point is COLMAP's. The distortion is COLMAP's (column "dist. known") or zero (column
  "zero dist.", the lab's condition for a video).
- `pyramid`: the 8 frames of the standard benchmark, SIFT on the CPU tier and mutual ratio-0.8
  matches of every pair (`tests/fixtures/pyramid-sift-cpu.json`, first octave 0, 353 to 443
  keypoints a frame). Its focal is unknown. The relief (`pyramidRelief`) is the median depth of
  the points seen more than 220 px from the apex over those seen within 90 px of it, minus 1, in
  frame 7. Above 0 is the right way up.
- Record `4a8aa2ff`: the tracks of the owner's result file (`typecv-reconstruction.json`, 22
  views, 5729 landmarks), turned back into keypoints and pair matches (226 pairs of 32 or more
  matches). Caveat: these tracks survived a run at 0.8, so they favour 0.8. The owner's files are
  not in the repository. Only the numbers are.

### 2.2 The refinement from three starts (before this change)

Final focal / width after the lab's run, `refineSharedFocal` on, 8 final iterations.

| Sample         | Truth | Start 0.5 | Start 0.8 | Start 1.2 | Start 0.5, zero dist. | Start 0.8, zero dist. | Start 1.2, zero dist. |
| -------------- | ----- | --------- | --------- | --------- | --------------------- | --------------------- | --------------------- |
| south-building | 0.833 | 0.568     | 0.832     | 1.112     | 0.570                 | 0.834                 | 1.109                 |
| gerrard-hall   | 0.683 | 0.578     | 0.750     | 1.077     | 0.559                 | 0.749                 | 1.078                 |
| person-hall    | 0.684 | 0.580     | 0.731     | 1.143     | 0.536                 | 0.750                 | 1.091                 |
| graham-hall    | 0.692 | 0.572     | 0.722     | 1.032     | 0.579                 | 0.743                 | 1.018                 |
| pyramid        | ?     | 0.564     | 0.786     | 1.287     |                       |                       |                       |
| `4a8aa2ff`     | ?     | 0.516     | 0.812     | 1.090     |                       |                       |                       |

Fact: the result follows the start. It moves 3 to 15 % towards the truth and stops.

### 2.3 Findings

1. **Too few LM steps for a slow valley.** Fact: on `south-building` from 1.2, the final BA alone
   (the map built with no final BA, then `bundleAdjust` with `refineSharedFocal`) reaches 1.112
   after 8 iterations, 0.941 after 32 and 0.835 (the truth) after 128. Each accepted step moves
   the focal by a few per cent. Inference: the focal and the depths are almost interchangeable
   (a larger focal and proportionally deeper points reproject nearly the same), so the damped
   Gauss-Newton step follows a long, curved valley.
2. **The stall rule is not the cause.** Fact: the same runs with `stallTolerance` 0 give the same
   focal at 8 and 32 iterations.
3. **The map built at a wrong focal is a different local minimum.** Fact: on `south-building`
   from 0.5, the final BA stops at 0.728 after 128 and after 512 iterations (RMS 0.300 px against
   0.145 px at the truth). The incremental mapping at 0.5 kept 5307 landmarks against 5769 at 0.8:
   triangulation and the observation filter at the wrong focal dropped the observations that would
   pull the focal back.
4. **Refining the focal during mapping does not fix it.** Fact: with the focal refined in every
   global BA of the mapping (a prototype, not kept), the results were still start-dependent and
   worse (`gerrard-hall` 1.2 → 1.254, `graham-hall` 0.8 → 0.799, and 10 of 12 views registered
   from 0.5). The focal of the first two or three views is poorly determined, and the map grows
   around it.
5. **The gauge is not the cause.** Fact: the joint BA fixes the anchor view and the baseline to
   the scale view, and the log-focal is a separate coordinate (`bundleAdjustJointFocal`).
6. **Near-planar drone scenes observe the focal weakly.** Fact: on the pyramid, the final RMS
   error of fixed-focal runs is 0.49 to 0.55 px for every focal from 0.64 to 1.04. A start at 1.2
   falls into a second solution with an inverted relief (RMS 0.79 px), and the refinement moves
   it further, to 1.29 (relief −0.49). Starts of 0.5 and 0.8 end at 0.56 (relief 0.93) and 0.79
   (relief 0.30): the same scene, deeper or flatter.

Conclusion (inference): the refinement is a local method. It needs a start in the right basin,
and the start must be close enough for 8 steps.

## 3. Candidates, measured

### 3.1 Focal from fundamental matrices

For each verified pair, the fundamental matrix was fitted to the inliers (normalized eight-point,
pixels centred on the principal point). The focal that makes `K F K` closest to two equal singular
values (Mendonça and Cipolla) was searched on a 1 % grid from 0.2 to 4. The median over pairs:

| Sample         | Truth | Inliers verified at 0.5 | Inliers verified at 1.2 | Spread (quartiles, at 0.5) |
| -------------- | ----- | ----------------------- | ----------------------- | -------------------------- |
| south-building | 0.833 | 0.729                   | 0.847                   | 0.616 to 0.838             |
| gerrard-hall   | 0.683 | 0.641                   | 0.715                   | 0.452 to 0.813             |
| person-hall    | 0.684 | 0.751                   | 0.766                   | 0.505 to 0.908             |
| graham-hall    | 0.692 | 0.647                   | 0.708                   | 0.552 to 0.838             |
| pyramid        | ?     | 0.378                   | 0.466                   | 0.304 to 0.490             |

Fact: errors of 5 to 15 %, a dependence on the start through the inlier sets, and a wide spread on
each pair. On the pyramid it points at 0.38 to 0.47, where the mapper's reprojection error is
higher than at 0.8 (2.2, item 6). Decision: not used.

### 3.2 A focal sweep on the first views

The mapper runs on the first `k` views at each focal of a grid, with the focal held. The full
`south-building` map (12 views) has its smallest final RMS error at 0.857 (0.157 px, against 0.6
px at 0.4 and 0.45 px at 1.5). The first four views alone give the same minimum (0.857, 0.133
px). On `gerrard-hall` the first four views give 0.709 (truth 0.683).

Fact: scored by the RMS error alone, the sweep fails on record `4a8aa2ff`. On its first five views,
0.60 scores 0.263 px with 3734 observations kept, and 0.92 scores 0.347 px with 5035. A run at a
wrong focal drops the observations it cannot fit, and the RMS error rewards the dropping. The full
22-view map prefers 0.857 (0.364 px, 5699 landmarks) to 0.586 (0.558 px, 4660 landmarks).

Decision: score each run by a truncated error (`scoreCandidates`). Among the runs that
registered the same number of views, `n` is the most observations any of them kept. A run that
kept `m` observations at RMS error `e` scores `(m·e² + (n − m)·τ²) / n`, with `τ` the BA's Huber
threshold (`maxErrorPx / 2`, 1.5 px in the lab). With it, record `4a8aa2ff` gives 0.85 (3.3).

## 4. Design

### 4.1 The search (`src/sfm/focal/focal-search.ts`)

1. Form match-connected components from pairs with at least eight correspondences. Try larger
   components first, preserving input order within them and breaking component ties by input
   order. Take their first `views` views (default 5) and the pairs between them. An explicit
   `initialPair` retains the original input ordering. Raw connectivity only selects evidence;
   every geometric and degeneracy check still runs.
2. For each selected focal of the grid, run the mapper on them (`incrementalSteps`), with the focal held,
   `refineSharedFocal` off and `bundleIterations` final iterations. The grid is `0.3 × 1.15^i`
   up to 3: 17 possible values (`focalGrid`). The adaptive schedule below chooses which to evaluate.
3. Score the runs on one common observation set (`commonSetScore`, `scoreCandidates`; section
   10). The set is the tracks of the subset's input matches (`commonTracks`), fixed before the
   sweep. Each run's poses triangulate every track, and each observation costs its truncated
   squared reprojection error. A run that does not build a map scores nothing. The directional
   (bearing) mode keeps the score of 3.2 with its complexity correction.
4. Pick the run with the most views, then the smallest score (`bestCandidate`).
5. Fit a parabola in `log ratio` through the best point and its two neighbours, held between the
   neighbours (`parabolaVertex`). This is `gridRatio`.
6. Run the mapper on the same views at `gridRatio` with `refineSharedFocal` and 25 final
   iterations. Keep its focal when it stays within one grid step of `gridRatio`. This is `ratio`.
7. When no run of step 2 builds a map, double the subset within that component and repeat.
   Try the next component if the whole component fails. If all fail, throw
   `ReconstructionInitializationError` with bounded focal-trial and pair diagnostics (section 9).

The standalone mapper then runs on every view at `ratio`, with the caller's options, and its final
BA refines the shared focal as before. The rolling GPU session's additional final calibration is
described in section 4.2 below.

- The focal is measured against the width `2·cx + 1` (`cameraWidth`), the width of an image whose
  principal point is its centre. `fy / fx` of each view is kept, and so are the principal point and
  the distortion (`cameraAtRatio`).
- The grid is fixed in absolute terms, so the estimate does not depend on the supplied focal. Fact:
  the tests compare the whole search result from starts 0.5 and 1.2 for equality.
- The search makes the mapper's own numerical requests (`verifyPair`, `pnp`, `triangulation`,
  `bundle`, `observations`). The CPU reference and the device driver run the same search, and
  Rule 1.2 of `design-rules.md` holds it as it holds the mapper: the session on the CPU tier lands
  within 1e-3 of the CPU reference (`tests/typeshade/focal-search.test.mjs`).

### 4.2 The API

- `IncrementalOptions.focalSearch?: boolean | FocalSearchOptions` (`views`, `minRatio`,
  `maxRatio`, `step`, `refineIterations`). Off by default. Ignored when a run resumes a map.
- `IncrementalResult.diagnostics.focalSearch?: FocalSearchResult` (`ratio`, `gridRatio`, `views`,
  `candidates`).
- `FOCAL_SEARCH_DEFAULTS` and the types `FocalSearchOptions`, `FocalSearchResult` and
  `FocalCandidate` from the root entry.
- `IncrementalSfmSession` (`src/sfm/gpu/mapper.ts`) with `focalSearch`: during `add`, the first map
  waits until one component registers at least `views` calibration views (or `finish`). Input
  count alone is insufficient. `finish` allows short input, whose focal can be poorly constrained.
  The run that builds it performs this initial subset search once. The initial map adopts its
  calibration, and later inputs use the current `ratio × width`. `session.focalSearch` retains the
  original subset/grid evidence and exposes the current jointly refined ratio.
  At `finish()`, a boundary-selected initial grid winner can trigger bounded joint pose/point/focal
  restarts when views beyond that original subset have arrived, even if every view still fits in
  the local window. The current estimate and at most two starts from the original grid are compared
  on the same complete retained observations and Huber objective; the initial feature/matching
  search is not repeated. Each start after the current estimate begins from a map rebuilt at its
  focal: the session maps the archive's tracks again (`historyTracks`) and triangulates each
  archived landmark with the rebuilt poses (section 10). An archive of more than 64 views reuses
  the archived map. The same initial subset without new views adds no restarts. Rolled
  archives also receive complete-observation refinement when supported. This does not force
  supplied or grouped intrinsics into one shared camera, and pending directional factors or an
  explicit conditional-depth admission policy defer the extra solve. Working-budget refusals
  preserve unsolved geometry rather than reporting a local inlier metric as a whole-map score.
  See [final archive consistency](archive-consistency.md) for iteration controls and deferred paths.

### 4.3 The lab

- One video: the focal field shows "자동" ("Auto"), and the estimate after the run, as before
  ("추정 0.78"). The worker passes `focalSearch` to the session for the `video` profile only.
  Calibrated photo samples keep their cameras.
- "고급 설정" ("Advanced") holds a switch "초점거리 직접 넣기" and the number. With it on, the run
  starts from the typed value and does not search, as before this change.
- The standard benchmark does not search. It keeps the start of 0.8, so that its pass times stay
  comparable with earlier records (`lab-run-records.md`).
- Multi-video runs keep their per-video focal rows. Open: a search per video (4.6).
- The run record carries `options.focalSearch` and `result.focalSearch` (`ratio`, `gridRatio`,
  `views`). The format number is unchanged. `lab-run-records.md` describes the fields.

### 4.4 Device budget (issue #160)

The search refuses no work on size. The default component starts with a five-view subset;
the full grid and optional interpolated refinement can require 18 mapper runs, or more after
subset growth or a failed component. The GPU session's verification cache keeps
the search's verdicts until the window evicts a view, as it keeps other verdicts.

### Adaptive grid schedule (2026-10-22)

The default search first evaluates alternate grid points, including both ends. It evaluates
both fine-grid neighbours of every coarse local minimum with the most registered views, then
brackets the current winner on the original grid. Each ratio runs at most once per subset.
The BA iterations, features, matches, score, parabolic interpolation and final focal refinement
are unchanged. There is no additional lab setting or URL parameter.

When the coarse pass cannot register the whole subset, its winner is at an edge, or adjacent
coarse scores are equal, the schedule evaluates every remaining grid point. A coarse pass with
no initialization therefore still finds a map supported only at an odd grid point. Short custom
grids also run in full. `FocalSearchResult.candidates` contains only evaluated points, sorted
by ratio; the winning parabola always uses adjacent points of the original fine grid.

This schedule is a heuristic. The fallback handles observed ambiguity; it cannot prove that an
unevaluated interval contains no narrow, better basin. The internal exhaustive mode remains a
measurement/test reference. New datasets should be checked against it before changing the
schedule further.

Reproduce the numerical gate with `node scripts/compare-focal-search.mjs`. It runs both schedules
on identical inputs and checks focal/error tolerances and equal registered views, landmarks and
observations. Measured here on 2026-10-22 with these changes on source base `f6d8587`,
the lab's numerical controls, zero distortion,
five search views and the committed feature/match fixtures:

| Sample | Candidates, full → adaptive | Search BA calls, full → adaptive | Registered views | Landmarks | Final RMS error (px) |
| --- | ---: | ---: | ---: | ---: | ---: |
| pyramid | 17 → 11 | 90 → 60 | 8 | 689 | 0.492651 |
| south-building | 17 → 11 | 90 → 60 | 12 | 5807 | 0.160695 |
| gerrard-hall | 17 → 13 | 75 → 60 | 12 | 7872 | 0.231046 |
| person-hall | 17 → 11 | 90 → 60 | 12 | 12276 | 0.277008 |
| graham-hall | 17 → 11 | 90 → 60 | 12 | 5952 | 0.320907 |

Both schedules gave identical final focals, view/landmark/observation counts and RMS errors in
these five runs. The pyramid retained relief 0.301242. Search BA calls fell 20–33%. This is a
CPU-reference work-count result, not a measured native iPhone speedup.

### 4.5 Cost

Fact (CPU reference, this container, shared with other jobs): a whole run with the search took
18 to 25 s on the 8-frame pyramid against 2 to 5 s without it, and 74 to 297 s on the COLMAP
samples against 12 to 59 s. The COLMAP pairs carry thousands of matches. Not measured: the time on a GPU. Row 11
of `physical-gpu-runs.md` asks for it.

### 4.6 Open items

1. The time on a physical GPU (4.5).
2. Lens distortion is not estimated. With zero distortion assumed, the strongly distorted lenses
   of `gerrard-hall` and `person-hall` (k1 = −0.11) land 5 to 9 % above COLMAP. The plain
   refinement from 0.8 lands there too (2.2).
3. The pyramid's true focal is unknown. The comment on `DEFAULT_VIDEO_FOCAL` called the drone
   lens "about 0.58". No measurement supports it: the mapper's error is lowest near 0.8 on both
   pyramid inputs, and the vanishing-point note (`research/vanishing-point-focal.md`) gave 0.77.
4. Multi-video runs (`docs/multi-video.md`) do not search yet.
5. On a calibration subset with little rotation, the grid's lower bound wins. Record
   `chimney-rock-58532c69.md` (section 3.1) measures it on a drone video: the RMS error is flat
   over the whole grid, the observations kept fall as the focal grows, and the truncated score
   therefore rises with the focal. The final restarts of 4.2 reuse the map built at the boundary
   and stop short of the reference focal. Done for the score and the restart (section 10, #267).
   The subset selection by parallax and rotation stays open (10.6).

## 5. Measurements with the search

`node scripts/measure-focal.mjs --search=1` (start 0.8; the result does not depend on the start).
Error = (estimate − truth) / truth.

| Sample         | Truth | Grid (dist. known) | Search (dist. known) | Final (dist. known) | Error  | Final (zero dist.) | Error  |
| -------------- | ----- | ------------------ | -------------------- | ------------------- | ------ | ------------------ | ------ |
| south-building | 0.833 | 0.881              | 0.834                | 0.834               | +0.1 % | 0.842              | +1.1 % |
| gerrard-hall   | 0.683 | 0.696              | 0.685                | 0.685               | +0.2 % | 0.720              | +5.4 % |
| person-hall    | 0.684 | 0.701              | 0.684                | 0.684               | +0.1 % | 0.744              | +8.8 % |
| graham-hall    | 0.692 | 0.686              | 0.675                | 0.676               | −2.1 % | 0.684              | −0.8 % |
| pyramid        | ?     | 0.697              | 0.768                | 0.784               |        |                    |        |
| `4a8aa2ff`     | ?     | 0.936              | 0.882                | 0.851               |        |                    |        |

- Fact: the grid is absolute, so the start does not enter the search. A run from 0.5 and from 1.2
  gave the same estimate on the four COLMAP samples (with an earlier score, the RMS error alone)
  and on the pyramid (with the score of 3.2, also held by `tests/focal-search.test.ts`).
- Fact: the pyramid's relief is the right way up at the estimate (0.30 from a start of 1.2,
  against −0.49 for the refinement alone from 1.2).
- Fact: on record `4a8aa2ff` the full 22-view map with a fixed focal has its smallest error at
  0.857 (0.364 px). The search gives 0.851.

Decision: the tolerance of the tests is 5 % of the truth with the lab's front end and zero
distortion (`graham-hall`, `tests/focal-search.test.ts`), and 2 % on the synthetic orbit of
`tests/typeshade/focal-search-cases.mjs`.

## 6. Verification

- `tests/focal-search.test.ts`: the grid, the controls, the score and the parabola; the pyramid
  from 0.5 and 1.2 gives one search result, an estimate from 0.65 to 0.95 and the relief the
  right way up, where the fixed start of 1.2 inverts it; `graham-hall` with the lab's front end
  lands within 5 % of COLMAP from a start of 1.2.
- `tests/typeshade/focal-search.test.mjs` (in `test:shade`): `IncrementalSfmSession` on the CPU
  tier finds the CPU reference's estimate within 1e-3 and lands within 2 % of the truth.
- `website/worker.test.mjs`: the run record validator takes the new fields and names a broken one.

## 7. Record of this document

- 2026-10-21: written with the implementation, proposed (#162).
- 2026-10-21: section 8 added: the pair base of the search (#180).
- 2026-10-26: open item 5 added from the diagnosis record `chimney-rock-58532c69.md` (#239).
- 2026-10-27: section 10 added: the common observation set of the score and the rebuilt restarts
  (#267); 4.1 step 3, 4.2 and open item 5 updated.

## 8. Step record: the pair base of the search (#180)

Authority: `docs/gpu-throughput.md` (accepted 2026-10-21), owner run record
`20261021145305440-66f7f868`. Configuration: `main` at `4f6b49a`, Node.js 22.22.0, headless
Chromium 1194 with SwiftShader on a shared 4-core container, no physical GPU.

### 8.1 Cause

Fact (owner record, 48 frames of the Desert arch video at 1280, NVIDIA Turing, 13 registered views):
the stage `pairVerification` ran 222 calls and 131.0 s of the 254.7 s total. The kernel
`appendPairModelRows` took 160.2 s of the 265 s of GPU time.

Fact (this change, `scripts/measure-focal-search-cost.mjs`, pyramid sample, 8 frames, 28 pairs):
a run without the search makes 28 `pairVerification` calls. A run with the search makes 208. The
difference is 180 = 18 mapper runs × 10 pairs: the 17 grid points and the joint refinement each
verify the 10 pairs of the first 5 views. The owner's 222 calls fit the same count: 180 for the
search and about 42 for the final mapper run.

Fact: the kernel appends the rows of a refit one after another into one triangular factor. A pair
refits 8 times, so its calls grow with the matches of the pair (`REFIT_ROW_CHUNK`, 128 rows per
call). The kernel is large because the search repeated the refits 18 times. On the pyramid
sample the kernel took 0.5 s of 11.6 s GPU time without the search and 4.4 s of 80.3 s with it.
The pairs of the owner's record carry more matches.

### 8.2 What depends on the focal

Fact: the essential-matrix search draws 8 rows at random and solves the normalized eight-point
system. Hartley's normalization removes the scale of the coordinates. The draw (the seed of the
pair) and the solution `F` are the same at every focal, up to rounding. The Sampson threshold
`maxError / f` in normalized units is `maxError` in pixels at every focal.

Only these steps depend on the focal `f`:

1. `E(f) = K(f)ᵀ F K(f)`, projected to the nearest essential matrix.
2. The scoring of each trial, the adaptive stop, the local refit and the pose recovery.
3. Nothing else: the matches and the homography of the pair do not depend on `f`.

### 8.3 Change

1. `pairBase` (`src/sfm/focal/focal-pairs.ts`) keeps, for each pair of the subset, the
   correspondences, the homography inlier count and the eight-point solution of each trial
   (drawn on demand, in the order of `reconstructTwoView`).
2. `verificationFromBase` scales a solution to the cameras of one grid point, projects it and
   runs the same consensus, local refit and pose recovery as `verifyImagePair`
   (`ImagePairOptions.hypotheses`, internal). The inlier rows and the geometry are the same as
   those of a full verification.
3. `focalSearchSteps` gives each mapper run these verdicts as `preVerified` rows. A pair that
   failed has `PreVerifiedPair.rejected`. `judgePair` (`sfm/pairs/pair-verdict.ts`) is the one function that
   turns a verification into a verdict, for `verifyPairs` and for the search.
4. The base runs on the host, so the CPU reference and the device driver use the same verdicts.
   A subset with lens distortion verifies each grid point in full as before: the ideal pixels
   depend on the focal there.
5. The mapper run on all views, after the search, verifies its pairs on the device as before.

Not changed: `appendPairModelRows`. Fact: the order of the row appends is the order of the QR
update of one model, so a different order changes the bits of the result. A larger chunk saves
host calls only. The occupancy floors of `pair-estimation.ts` are the work of #178.

### 8.4 Accuracy

The scores are equal, not bit-identical: the scaled solution differs from a solution fitted
in the coordinates of the grid point by rounding. Fact: on the CPU reference, the 17 candidates
of the pyramid agree with the earlier ones to the printed precision (4 digits), and the
estimates agree to 1e-11. `tests/focal-pairs.test.ts` holds the equality of the inlier rows,
the counts and the translation (1e-9) of a derived verification and a full one for 6 pairs at 5
focals. `tests/focal-search.test.ts` holds 8 pyramid scores of the earlier code.

`node scripts/measure-focal.mjs --search=1 --starts=0.8 --distortion=0` (zero distortion, the
lab's condition for a video), before and after:

| Sample         | Truth | Grid before | Grid after | Final before | Final after | Error after |
| -------------- | ----- | ----------- | ---------- | ------------ | ----------- | ----------- |
| pyramid        | ?     | 0.697       | 0.697      | 0.784        | 0.784       |             |
| south-building | 0.833 | 0.867       | 0.867      | 0.842        | 0.842       | +1.1 %      |
| gerrard-hall   | 0.683 | 0.732       | 0.732      | 0.720        | 0.720       | +5.4 %      |
| person-hall    | 0.684 | 0.751       | 0.751      | 0.744        | 0.744       | +8.8 %      |
| graham-hall    | 0.690 | 0.687       | 0.687      | 0.684        | 0.684       | −0.8 %      |

With the distortion known (the table of section 5) the cameras carry `k1`, the search verifies
in full, and the numbers are those of section 5 by construction. The device search is not
bit-identical to the earlier device search: the earlier one ran its trials on the GPU, the new
one on the host.

### 8.5 Cost on SwiftShader

`node scripts/measure-focal-search-cost.mjs --sample <name> --search 1` (`reconstructIncrementalGpu`,
the lab's controls, start 0.8). SwiftShader is a CPU emulation, and the container was shared with
other jobs, so wall times are noisy. Counts are exact.

| Run                                               | Pair verifications | Dispatches | Submissions | GPU time | Wall  |
| ------------------------------------------------- | ------------------ | ---------- | ----------- | -------- | ----- |
| pyramid, 8 frames, no search                      | 28                 | 10,028     | 1,155       | 11.6 s   | 70 s  |
| pyramid, search, before                           | 208                | 84,379     | 9,354       | 80.3 s   | 180 s |
| pyramid, search, after                            | 28                 | 68,774     | 3,282       | 49.4 s   | 135 s |
| south-building, 5 frames, 4 views, search, before | 118                | 67,247     | 6,367       | 186.1 s  | 280 s |
| south-building, 5 frames, 4 views, search, after  | 10                 | 39,200     | 2,367       | 126.4 s  | 214 s |

Fact: the pair-verification calls of the search fall to zero on the device. The kernels
`scorePairModels`, `appendPairModelRows`, `solvePairModelNull` and their neighbours leave the
search. The pair verification time on south-building falls from 107.7 s to 29.3 s (the final
run only).

### 8.6 Open items

1. The adaptive schedule in section 4.4 reduces the pyramid search from 90 to 60 BA calls,
   retaining its final focal and geometry on the committed fixture. Larger subsets and new
   inputs still need the exhaustive numerical comparison; the fallback retains the full-grid
   worst case.
2. The owner's later iPhone record `7b5f61ef` measured 84.8 s of BA before device loss on the
   fifth frame. It does not measure the adaptive schedule or a completed native before/after
   run. Next action: one owner run at the merged optimization commit.
3. The search verifies on the host. A large subset (thousands of matches) costs host time for
   the trial scoring and the local refit. Not isolated: the container was shared. Next action:
   time the host part on the owner's run.

## 9. Connected calibration evidence and initialization diagnostics (2026-10-23)

A chronological prefix can contain a disconnected rotation-only shot and only two views of a
later translating shot. The former cannot initialize structure; calibrating from the latter
can still return a focal that fits two views but distorts the later map. The search now prefers
the largest match-connected component. Streaming `add()` also waits for the configured number
of **registered calibration views**, rather than finalizing a focal as soon as that many
unrelated inputs arrive. A smaller component remains a fallback if larger ones are degenerate;
`finish()` can process a short clip. No camera calibration or geometric rejection thresholds
are relaxed. Components are not merged by this change.

If the interpolated joint refinement fails or registers fewer views than the best grid
candidate, retain the grid focal that actually built the map rather than the failed interpolation.

The CPU evaluator is reproducible after building both worktrees:

```sh
node scripts/evaluate-focal-components.mjs --baseline /path/to/baseline-worktree
```

[Measured results](experiments/focal-components-20261023.json) compare the previous mapper
with the new selection on supplied synthetic correspondences. Ground-truth cameras and points
are passed only to the evaluator, using one camera-center Sim(3) per map. The shared focal is
0.7 × width; the seed and ±0.25 px noise are deterministic. All seven cases retain all six
expected cameras and their points. All three already-connected controls preserve identical
geometry hashes. Rotation-only, planar and distant-scene controls reject initialization in
both versions; the new errors retain their rejection evidence.

For a three-view disconnected prefix followed by 48 orbit points, the search's focal error
falls from 6.61% to 0.16%; after final BA it falls from 1.74% to 0.15%. The final mean rotation
error falls from 1.58° to 0.24°, and median point error from about 0.287 to 0.028 fixture units.
The 64-point case also improves substantially. The 80-point and four-prefix-view cases have
slightly larger final focal and point errors, while their rotation errors improve. The complete
results include these counterexamples; the change is not a guarantee that every metric falls.
This small geometry corpus does not establish RGB registration success, real-video accuracy,
dense surface accuracy or native GPU speed. In particular, valid rejection of an unobservable
scene is not counted as a reconstruction success.

Failed searches now keep `ReconstructionInitializationError.pairVerification` and optional
`focalSearch` evidence. Up to 64 failed trials record the ratio, subset size, geometric pair
counts, eligible seeds, a representative pair, rejection counts and failure text; an omitted
count describes older trials. Pair diagnostics include positive-depth support when available.
The lab preserves bounded evidence as optional `failure.initialization` in v6 and calls a
`finish()` without a map `initializing 3D map`, rather than attributing it to final BA.
The diagnostic history is per search invocation. Cross-update caching, host phase timing,
distortion/zoom models and broader input validation remain separate work.

## 10. Step record: one observation set for the grid, and rebuilt restarts (#267)

Authority: candidates 1 and 2 of `chimney-rock-58532c69.md`, section 5 (bug B1 of the
coordinator's bug lane, 2026-10-27). Configuration: `main` at `02f4aed` (before) and the branch
`claude/fix-focal-search-bias` (after), Node.js 22.22.0, a shared container with no GPU. Every
number below is from the CPU reference.

### 10.1 Cause

Fact (`chimney-rock-58532c69.md` 3.1): on record 58532c69 the search ended at
`{ ratio: 0.5835, gridRatio: 0.3, views: 5 }`. The reference is about 1,242 px (ratio about
0.97).

Fact: the search scored each grid point on the observations that its own run kept. A smaller
focal makes every triangulation angle larger, so more tracks passed the 0.5° gate. The truncated
score of 3.2 charges each dropped observation τ², so it rewarded the smallest focal. On the
exported tracks of the five calibration views the grid's scores rose monotonically from 0.0754
px² at 0.3 to 0.7088 px² at 2.807, with the RMS error flat (0.258 to 0.275 px).

Fact (3.5 of the record): at `finish()` the restarts of `historyFocalSeeds` reused the poses and
points of the map built at the boundary. That map is a different local minimum (2.3, item 3).

### 10.2 What depends on it

- `focalSearchSteps`, through `scoreCandidates` and the adaptive schedule
  (`focalCandidateSteps`): the grid winner, `gridRatio` and the joint refinement's start.
- `IncrementalSfmSession.finish()` (`src/sfm/gpu/mapper.ts`): `historyFocalSeeds` and
  `refineHistory` (`src/sfm/session/history-refinement.ts`).
- The digests of the focal-search cases (`tests/digests/cpu-*.json`, `webgpu.json`).

### 10.3 Change

1. Candidate 1, the score. `commonTracks` forms the conflict-free tracks of the subset's input
   matches (`buildTracks`) once, before the sweep. `commonSetScore` triangulates each track with
   the poses of a run's map (`triangulateLinear`, then `refineTrackPoint`, no angle gate). Each
   observation costs `min(e², τ²)`; a track that does not triangulate costs τ² for each
   observation. `scoreCandidates` takes this score in place of the score on the kept
   observations. `FocalCandidate` keeps its fields; only the meaning of `score` changes. The
   directional mode keeps its own score.
2. Decision: the common set, not the angle gate held in image terms. Fact: the gate held in
   pixels (the angle gate scaled by `reference / ratio`) reversed the bias on the sample. The
   observations kept then rose with the focal, and the grid's upper bound 2.807 won at
   reference ratios 0.6 and 1.0. The common set gave a minimum inside the grid (10.4).
3. Candidate 2, the restarts. `refineHistory` takes an internal `rebuildAtRatio`. Each restart
   focal after the first (the current calibration) starts from a map rebuilt at that focal. The
   session maps the archive's tracks again (`historyTracks`, then `runIncrementalGpu` at that
   focal) and triangulates each archived landmark with the rebuilt poses. A rebuild that builds
   no map, or misses an archived view, skips that restart. A landmark that does not land in front of
   all its cameras is placed on the ray of one of its observations at the median depth of the
   others; without such a point, or when the solver refuses the start, the restart is skipped.
   An unsolved part of a rebuilt start keeps the rebuilt geometry. The least-cost rule on the archive's
   observations is unchanged. An archive of more than 64 views reuses the archived map.

### 10.4 Accuracy

Record 58532c69, the exported tracks (`typecv-reconstruction.json`, 47 views). The search runs
on the first five views by time (0, 5, 9, 13, 17); the final row is the standalone CPU mapper on
all 47 views at the search's ratio with `refineSharedFocal`. Scratch script on the public
internals (`focalSearchSteps`, `reconstructIncremental`), the lab's controls:

| Quantity | Before | After | Reference |
| --- | --- | --- | --- |
| Grid winner | 0.3 (lower bound) | interior (0.918 to 1.214) | |
| `gridRatio` | 0.3 | 1.107 | |
| Search `ratio` | 0.343 | 1.107 | |
| Final ratio, 47 views | 0.356 | 1.021 | about 0.97 (COLMAP 0.98) |
| Final RMS, 47 views | 0.786 px | 0.397 px | 0.348 px (COLMAP) |
| Registered views / landmarks | 47 / 18,081 | 47 / 16,928 | |

The adaptive schedule evaluated 11 grid points. Their common-set scores fall from 0.0715 px² at
0.3 to 0.0609 px² at 0.918, 1.055 and 1.214 (equal to four digits), and rise to 0.0660 px² at
2.807. Caveat: the tracks survived the lab's run at 0.58 (record 3.1).

Candidate 2 alone, on the same export: `refineHistory` on the CPU (`bundleAdjust`, 32
iterations, τ = 1.5 px) from the export's map (47 views, 90,391 observations, ratio 0.5835), with
the three seeds that `historyFocalSeeds` gave in the record (0.5835, 0.918, 2.807). Before, the
restarts reuse the archived map; after, each restart after the first rebuilds it with the CPU
mapper on `historyTracks` at its focal (a scratch replay; in the session the rebuild runs through
`runIncrementalGpu`):

| Restarts | Final ratio | Mean error | Robust cost | Rebuilds |
| --- | --- | --- | --- | --- |
| Reuse the archived map | 0.640 | 0.678 px | 74,968 | none |
| Rebuild at each restart | 0.967 | 0.227 px | 9,545 | 0.918: 47 views; 2.807: 41 views (skipped) |

The archive's cost before either solve is 77,353 (mean error 0.696 px). The first replay of the
rebuild found a landmark behind a rebuilt camera, which the bundle adjustment refuses; 10.3,
item 3 now places such a landmark in front of its cameras. Each rebuild took about 151 s on the
CPU reference.

The fixtures, `node scripts/measure-focal.mjs --search=1 --starts=0.8 --distortion=0` (zero
distortion, the lab's condition for a video), before and after:

| Sample | Truth | Grid before | Grid after | Search before | Search after | Final before | Final after |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pyramid | ? | 0.697 | 0.742 | 0.768 | 0.774 | 0.784 | 0.782 |
| south-building | 0.833 | 0.867 | 0.848 | 0.842 | 0.848 | 0.842 | 0.843 |
| gerrard-hall | 0.683 | 0.732 | 0.734 | 0.730 | 0.730 | 0.720 | 0.720 |
| person-hall | 0.684 | 0.751 | 0.747 | 0.744 | 0.747 | 0.744 | 0.747 |
| graham-hall | 0.690 | 0.687 | 0.697 | 0.682 | 0.682 | 0.684 | 0.684 |

The pyramid's relief is 0.301 before and 0.302 after. With the distortion known
(`--distortion=1`, the condition of the sample digests), the final ratios are:

| Sample | Truth | Final before | Final after |
| --- | --- | --- | --- |
| south-building | 0.833 | 0.834 (+0.1 %) | 0.831 (−0.3 %) |
| gerrard-hall | 0.683 | 0.685 | 0.685 |
| person-hall | 0.684 | 0.684 | 0.684 |
| graham-hall | 0.690 | 0.676 | 0.676 |

The synthetic scene `scene-90x6-focal-1.2` of `tests/characterization.ts` (truth 0.8845) gives
grid 0.8973 and final 0.8815 before and after, equal to 1e-8.

### 10.5 Cost

Fact: on the sample's five views, the search ran 432 mapper requests in 122 to 151 s before and
288 requests in 84 s after (one run each, a shared container). Before, the lower-bound winner
sent the adaptive schedule to the full grid; after, the interior winner keeps it adaptive. The
common set adds one host triangulation of each track per grid point. The search on graham-hall makes 70 bundle requests where it made 60 (the adaptive schedule
evaluates two more grid points; `tests/digests/cpu-graham-hall.json`); the other digest cases
keep their request counts. The fixture runs of 10.4 took about the same time before and after (pyramid 16 s and 18 s, south-building 69 s and 74 s,
graham-hall 53 s and 55 s, person-hall 204 s and 229 s).

Fact: on the CPU reference a rebuild of the sample's 47-view archive took about 151 s, and the
replay with two rebuilds took 461 s against 218 s without them (10.4). Not measured: the cost of
a rebuild in the session on a GPU or on SwiftShader. A rebuild is one
mapper run of the archive per restart focal, at most two, and it runs only when the grid winner
is a boundary point and later views have arrived.

### 10.6 Open items

1. The calibration subset is still the first views of the largest component in input order, not
   chosen by parallax and rotation (the second part of candidate 1). Fact: on the sample, the
   five first views turn 0.04° to 0.20° and the common set still finds 1.107. Not measured: a
   subset chosen by parallax. Next action: its own record.
2. Two scenes move away from their truth by less than 0.5 %. Person-hall with zero distortion
   moves from 0.744 to 0.747 (truth 0.684, +8.8 % to +9.3 %); with the distortion known it does
   not move. South-building with the distortion known moves from 0.834 to 0.831 (truth 0.833,
   +0.1 % to −0.3 %); with zero distortion from 0.842 to 0.843. Open for the owner's decision.
3. The directional (bearing) mode keeps the score on the kept observations, with its complexity
   correction. Not measured on the common set.
4. No session run of the sample with its images: the after-numbers come from the exported tracks.
   `refineHistory` with `rebuildAtRatio` runs in `tests/history-refinement.test.ts` on the CPU;
   the session's rebuild through `runIncrementalGpu` ran in no test of this change.
