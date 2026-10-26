# Diagnosis of run record `20261026043150261-58532c69` (Chimney Rock video B5 sample)

Status: draft record, 2026-10-26, in preparation for its pull request. No product code was changed. Labels: **Fact** (read or measured),
**Inference** (reasoned from facts), **Not run** (a check that did not run).

## 1. Configuration

| Item | Identifier |
| --- | --- |
| Run record | `20261026043150261-58532c69` (collector KV), site commit `5e1c95c46ac3a7dca291ab2a77f6f207be01e888` (= `main` of typecv at analysis time) |
| Input | Lab sample "Chimney Rock video B5 sample", 2560 x 1440 source, 1280 x 720 analysis, 34.2 s, 138 sampled frames (4 fps), 47 keyframes (phase gate) |
| Options | SIFT, 16384 features, guided matching, `focalSearch: true`, `refineSharedFocal: true`, `windowSize 32`, Dense: TSDF (`view-weighted`), `planarDepth: true`, `maxSources 6`, `bestSources 3`, window radius 5 step 2 |
| Result files | `typecv-reconstruction.json` (47 views, 19,123 landmarks, 90,391 observations), `typecv-dense.ply` (698,269 points), depth archives `typecv-depths-c000-0000..0061.tar.gz` + manifest (47 views x 4 stages, inputs included) |
| Reference | pycolmap 4.2.1 on the 47 archived 1280 x 720 keyframes: SIFT (CPU), exhaustive matching, incremental mapping, one shared camera, focal refined. Two runs: `SIMPLE_RADIAL` prior 1.2 x max side, `SIMPLE_PINHOLE` prior 0.8 x width. COLMAP is a reference, not ground truth. |
| Environment | Linux x86_64 container, Node 22.22.0, Python 3.13, numpy 2.5.3, scipy 1.18.1, no GPU. TypeCV CPU reference mapper from `src/` of the checkout at `5e1c95c`. |

Scripts (scratchpad `analysis/`): `sparse_stats.py`, `mvs_views.mjs` (replays `planMvsViews`), `dense_normals.py`,
`depth_view.py`, `seed_agreement.py`, `block_planes.py`, `colmap_ref.py`, `colmap_compare.py`,
`relief_compare.py`, `focal_sweep.mjs`.

## 2. The GPT analysis, checked

| Claim | Measured here | Verdict |
| --- | --- | --- |
| Focal 747 px (TypeCV) vs 1,242 px (COLMAP) | TypeCV `fx = 746.885` at 1280 width (ratio 0.5835; `gridRatio 0.3`). COLMAP: 1,255.1 px (`SIMPLE_RADIAL`, k1 = 0.011), 1,246.0 px (`SIMPLE_PINHOLE`, prior 0.8). Start-independent. | Confirmed |
| View direction change first to last 41.5° vs 61.6° | 41.5° (TypeCV) vs 61.5° / 61.8° (COLMAP) | Confirmed |
| Median observations per feature 3 vs 10 | TypeCV median 3, mean 4.73 (6,459 of 19,123 tracks have 2 views). COLMAP median 8, mean 12.1 | Confirmed (COLMAP median 8 here, not 10) |
| Around 30 s, 6 other images connected with meaningful parallax | View 120 (30.0 s): sources 128, 130, 112, 110, 132, 108 (2 to 3 s away), median triangulation angle 4.4° to 6.7° | Confirmed |
| Dense stage flattens the top/front face angle from 49° to 5° | Not reproduced. Dense depth equals the sparse depth at the landmark pixels (view 120: median difference -0.7 %, 67 % within 2 %, 98 % within 10 %). Per-block robust planes of the dense depth map agree with the sparse planes within 2° to 11° in 10 of 12 blocks. Block (1,1) vs (3,1) of view 120: sparse 50.1°, dense depth map 53.1°, dense PLY 76.7°. | Not reproduced; the dense surface follows the sparse structure |

## 3. Facts found here

### 3.1 The focal search stopped at the grid's lower bound

- **Fact.** `result.focalSearch = { ratio: 0.5835, gridRatio: 0.3, views: 5 }`. The grid is `0.3 x 1.15^i`, so 0.3 is its lowest point. The previous run of the same sample (`20261025012536402-fd99c8db`, Turing) ended at ratio 0.3 exactly; the Desert canyon run `20261024151031417-106d9185` at 0.3000. The Water tower run `20261024145805483-44bb5d97` ended at 0.908 (grid 0.963).
- **Fact.** The search subset is the first five keyframes in input order: frames 0, 5, 9, 13, 17 (0 to 4.24 s). On the exported tracks of these five views, the CPU reference mapper at a fixed focal gives (`focal_sweep.mjs`, 17 grid points):

| ratio | RMS px | observations kept | camera turn first to last | truncated score px² |
| --- | --- | --- | --- | --- |
| 0.30 | 0.273 | 11,407 | 0.20° | 0.075 |
| 0.46 | 0.263 | 10,839 | 0.19° | 0.178 |
| 0.60 | 0.263 | 10,355 | 0.16° | 0.270 |
| 0.80 | 0.258 | 9,897 | 0.13° | 0.356 |
| 0.92 | 0.263 | 9,612 | 0.12° | 0.412 |
| 1.21 | 0.259 | 9,037 | 0.09° | 0.521 |
| 2.81 | 0.262 | 8,060 | 0.04° | 0.709 |

  The RMS error is flat (0.258 to 0.273 px) over the whole grid. The observations kept fall monotonically with the ratio. The truncated score of `scoreCandidates` (`(m e² + (n - m) tau²) / n`, tau = 1.5 px) therefore rises monotonically, and the lowest grid point wins. Caveat: these tracks survived the run at 0.58, which favours that region; the monotonic trend is independent of that.
- **Inference.** In the first 4 s the camera translates and barely rotates (0.04° to 0.20° in every run), so the focal is unobservable from the subset. A smaller focal makes every triangulation angle larger (angle ~ disparity / f), so more tracks pass the 0.5° `minTriangulationAngleDeg` gate and more observations are kept. The score counts each dropped observation as a tau² misfit, so it rewards the smallest focal. This is a bias of the score, not evidence for 0.3.
- **Fact.** At `finish()`, a boundary winner triggers the restarts of `historyFocalSeeds` (current 0.3, the grid middle, the opposite end 3.0) in `refineHistory`; the accepted result was 0.5835. In `fd99c8db` it stayed at 0.3. `docs/auto-focal.md` 2.3 item 3 already records that a map built at a wrong focal is a different local minimum.

### 3.2 The sparse structure is a projectively distorted solution

- **Fact.** Similarity alignment of the 47 camera poses to COLMAP's (`colmap_compare.py`): rotation difference median 6.8°, max 14.2°; camera-centre difference median 0.31 and max 0.55 of the reference trajectory radius. Mean reprojection error: TypeCV 0.70 px, COLMAP 0.35 px.
- **Fact.** Fixed-focal runs of the CPU reference mapper on all 47 views and the exported tracks (`focal_sweep.mjs`): the total camera turn grows with the focal (table in section 3.5). The lab's 0.5835 gave 41.5°; COLMAP at 0.98 gives 61.5°.
- **Fact.** Depth relief on the same pixels (`relief_compare.py`: COLMAP's sparse depth at its feature pixels against TypeCV's dense `geometric-filtered` depth at those pixels):

| view (time) | common pixels | depth spread p95/p5, COLMAP | same, TypeCV | log-log slope TypeCV vs COLMAP | note |
| --- | --- | --- | --- | --- | --- |
| 0 (0.0 s) | 3,546 | 1.146 | 1.146 | +0.97 (MAD 0.004) | identical relief |
| 62 (15.5 s) | 8,370 | 1.196 | 1.094 | -0.15 | |
| 104 (26.0 s) | 5,669 | 1.390 | 1.280 | -0.45 | |
| 120 (30.0 s) | 5,700 | 2.164 | 1.555 | -0.35 | bottom third: COLMAP 1.28 x median depth, TypeCV 0.81 x; left third: 0.77 x vs 1.03 x |
| 137 (34.2 s) | 3,690 | 2.925 | 1.683 | -0.20 | right third: COLMAP 1.72 x, TypeCV 0.98 x |

  The first view agrees with COLMAP. From 15 s on, the depth ordering across the image is compressed and reversed relative to COLMAP. At 30 s the ground below the cliff (bottom right of the frame) is nearer than the mesa top in TypeCV's solution and farther in COLMAP's.
- **Inference.** This is the bas-relief family of solutions: less camera rotation, a smaller focal and a distorted relief reproject almost as well as more rotation, a larger focal and the true relief, when the tracks are short and the focal is fixed by a biased search. The owner's observation (a top face at 45°, faces perpendicular to the viewing direction, a flat background when turned 90°) is what such a solution looks like from any viewpoint but the cameras'.

### 3.3 The dense stage follows the sparse structure; it does not add a systematic collapse

- **Fact.** Dense depth at the landmark pixels (the MVS seeds) against the landmark depth: view 0 median 0.0 %, 91 % within 2 %; view 62 75 % within 2 %; view 120 67 % within 2 %, 98 % within 10 %; view 137 54 % within 2 %, 80 % within 10 %. Pixels without a dense depth at a landmark: 59 %, 25 %, 33 %, 20 %, 50 % for views 0, 62, 104, 120, 137.
- **Fact.** View 120, `geometric-filtered` map (64.4 % valid): angle between the depth surface's finite-difference normal and the viewing ray, median 55.8° (step 2 px) and 46.6° (median-filtered, step 6 px); the stored PatchMatch normal, median 53.4°. Shares below 10°: 1.3 % and 4.0 %. In the raw `photometric` map the finite-difference angle has a median of 82.8° (noise). The dense PLY normals have a median angle of 52° to the ray of every camera that sees the point. There is no fronto-parallel peak.
- **Fact.** View 120 per-block planes (4 x 3 blocks of 320 x 240 px): sparse against dense depth map normals differ by 2.0° to 11.3° in 10 blocks, 17.2° and 26.9° in two.
- **Inference.** The depth maps add noise and holes but keep the sparse shape. A fix in the dense stage cannot correct the shape; the shape is decided upstream.

### 3.4 Short tracks and near-only matching

- **Fact.** `registerFrame` in `website/src/workers/sfm.worker.ts` keeps descriptors of "a long-baseline first reference, two recent reference images and three adjacent frames" and disposes the rest, so each keyframe is matched against about four earlier keyframes: 176 matching calls and 176 matched pairs for 47 keyframes. COLMAP matched all 1,081 pairs. Median track length 3 against 8; mean 4.7 against 12.1. Longer links in the exported map (713 view pairs share 16 or more landmarks) exist only through chained tracks.
- **Fact.** MVS source selection (`planMvsViews`, score = sum over shared landmarks of g(angle)) picks temporal neighbours only: over all 282 source links, median time gap 3.0 s, median keyframe gap 4 (max 9), median triangulation angle 5.7°; 36 % of links have a median angle below 5°, 93 % below 10°.
- **Inference.** Both limits keep "parallax from elsewhere" out: the sparse stage never matches distant keyframes directly, and the dense stage never reads a source from the other side of the orbit. The first limit weakens the focal and the global shape; the second limits dense accuracy and completeness but is not the cause of the distortion.

### 3.5 The same tracks at a fixed focal near COLMAP's fit far better

- **Fact.** `focal_sweep.mjs`, CPU reference mapper (`reconstructIncremental`, lab controls, `refineSharedFocal: false`), all 47 views, the exported tracks (713 pairs with 32 or more matches, 90,391 keypoints), the focal held at each ratio:

| ratio | views | landmarks | observations | RMS px | mean px | camera turn first to last |
| --- | --- | --- | --- | --- | --- | --- |
| 0.30 | 47 | 17,859 | 83,569 | 0.792 | 0.605 | 21.1° |
| 0.5835 (the lab's result) | 47 | 18,131 | 84,444 | 0.738 | 0.556 | 41.0° (lab: 41.5°) |
| 0.80 | 47 | 16,293 | 80,563 | 0.604 | 0.435 | 65.2° |
| 0.97 | 47 | 16,658 | 83,818 | 0.317 | 0.227 | 61.2° |
| 1.30 | 47 | 15,840 | 77,855 | 0.643 | 0.474 | 57.1° |

  COLMAP on the same images: 0.98, 0.348 px, 61.5°.
- **Fact.** Depth relief of view 120 (median landmark depth per image third over the view's median): at 0.5835 rows top/mid/bottom 1.070 / 0.996 / 0.801 and columns left/centre/right 1.078 / 0.941 / 0.984; at 0.97 rows 0.831 / 1.069 / 1.331 and columns 0.742 / 1.008 / 1.154. COLMAP (its own features): rows 0.894 / 1.022 / 1.276, columns 0.770 / 1.019 / 1.134. View 137 at 0.97: columns 0.625 / 1.011 / 1.591 (COLMAP 0.757 / 1.122 / 1.721); at 0.5835: 1.243 / 0.991 / 0.974.
- **Inference.** Even on tracks selected under the 0.58 calibration, a map rebuilt at 0.97 halves the reprojection error, restores the camera rotation and restores COLMAP's depth ordering. The lab's 0.58 is a local minimum of the joint focal refinement that started from the map built at 0.3; the restart from the grid middle (`historyFocalSeeds`) reuses that map's poses and points, and a local solver cannot flip an inverted relief. A rebuild at the restart focal (candidate 2 in section 5) would have found the 0.97 basin on this sample.

## 4. What was not run

- No TypeCV run of this sample at a fixed focal near 0.98 with the real images (the lab's "초점거리 직접 넣기"), on a GPU or on SwiftShader. The CPU sweeps above reuse the exported tracks, which were selected at 0.58.
- No COLMAP dense reconstruction; no ground truth of the scene.
- The `fd99c8db` and `106d9185` depth archives were not analysed.

## 5. Candidate changes (owner decides; each needs its own record)

1. Focal search score: evaluate every grid point on one common observation set, or hold the triangulation-angle gate in image terms (disparity) during the search, so that a focal is not rewarded for passing more tracks through the 0.5° gate. Choose the calibration subset by parallax and rotation, not by input order.
2. Final calibration: when the grid winner is a boundary, rebuild the observations at each restart focal instead of reusing the map built at the boundary focal.
3. Matching reach for one video: keep descriptors (or a retrieval index) for more keyframes so that the orbit's far side is matched directly; measure track length and the focal against COLMAP on this sample.
4. MVS source selection: add an angle prior that prefers 5° to 15° and caps the weight of shared-landmark count, so that sources from the other side of the orbit enter when they exist.
