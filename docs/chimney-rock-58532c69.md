# Record 58532c69 (Chimney Rock): the relief collapse comes from the focal calibration

Status: diagnosis record, 2026-10-26 (pull request #239). It changes no product code and decides
nothing; section 5 lists candidate changes for the owner's decision, and each needs its own record.
Labels: **Fact** (read or measured), **Inference** (reasoned from facts), **Not run** (a check that
did not run).

The owner reported that the 3D model of this run looks right from the cameras' viewpoints only:
turned in the viewer, the top of the rock stands at about 45°, the cliff of the 30 s region does
not reach down, and the background is nearly a plane. This record measures where that comes from.

## 1. Configuration

| Item | Identifier |
| --- | --- |
| Run record | `20261026043150261-58532c69` in the collector's KV; site commit `5e1c95c46ac3a7dca291ab2a77f6f207be01e888` (`main` at the time of the analysis) |
| Input | Lab sample "Chimney Rock video B5 sample": 2560 × 1440 source, 1280 × 720 analysis, 34.2 s, 138 sampled frames at 4 fps, 47 keyframes (phase gate) |
| Options | SIFT with 16,384 features, guided matching, `focalSearch: true`, `refineSharedFocal: true`, `windowSize` 32; dense: TSDF fusion with `view-weighted` visibility, `planarDepth: true`, `maxSources` 6, `bestSources` 3, window radius 5 at step 2 |
| Result files | `typecv-reconstruction.json` (47 views, 19,123 landmarks, 90,391 observations), `typecv-dense.ply` (698,269 points), the depth archive `typecv-depths-c000.json` with parts `0000` to `0061` (47 views × 4 stages, inputs included) |
| Reference | pycolmap 4.2.1 on the 47 archived 1280 × 720 keyframes: SIFT on the CPU, exhaustive matching, incremental mapping with one shared camera and the focal refined. Two runs: `SIMPLE_RADIAL` with the prior 1.2 × the long side, `SIMPLE_PINHOLE` with the prior 0.8 × the width. COLMAP is a reference, not ground truth. |
| Environment | Linux x86_64 container without a GPU; Node 22.22.0; Python 3.13 with numpy 2.5.3, scipy 1.18.1, pycolmap 4.2.1; the CPU reference mapper of `src/` at `5e1c95c` |

Section 6 gives the commands. The scripts are `scripts/fetch-record-files.mjs`,
`scripts/measure-mvs-sources.mjs`, `scripts/measure-focal-sweep.mjs` and
`scripts/analyze-record.py` (the package `scripts/record_analysis/`).

## 2. Claims checked

The owner brought an earlier analysis of the same record. Each of its claims was measured again.

| Claim | Measured here | Verdict |
| --- | --- | --- |
| Focal 747 px (TypeCV) against 1,242 px (COLMAP) | TypeCV `fx` = 746.885 at the width 1280 (ratio 0.5835; `gridRatio` 0.3). COLMAP: 1,255.1 px (`SIMPLE_RADIAL`, k1 = 0.011) and 1,246.0 px (`SIMPLE_PINHOLE`, prior 0.8). The estimate does not depend on the prior. | Confirmed |
| View direction change, first to last view, 41.5° against 61.6° | 41.5° (TypeCV) against 61.5° and 61.8° (the two COLMAP runs) | Confirmed |
| Median observations per feature 3 against 10 | TypeCV median 3, mean 4.73 (6,459 of the 19,123 tracks have two views). COLMAP median 8, mean 12.1 | Confirmed; the reference's median is 8 here |
| Six other images connected around 30 s, with meaningful parallax | View 120 (30.0 s) has the sources 128, 130, 112, 110, 132 and 108, 2 to 3 s away, with median triangulation angles of 4.4° to 6.7° | Confirmed |
| The dense stage flattens the angle between the top and the front face from 49° to 5° | Not reproduced. The dense depth equals the sparse depth at the landmark pixels (view 120: median difference −0.7 %, 67 % within 2 %, 98 % within 10 %). The robust planes of the dense depth map agree with the sparse planes within 2° to 11° in 10 of 12 image blocks. Blocks (1,1) and (3,1) of view 120: sparse 50.1°, dense depth map 53.1°, dense PLY 76.7°. | Not reproduced: the dense surface follows the sparse structure (3.3) |

## 3. Facts found

### 3.1 The focal search stopped at the grid's lower bound

- **Fact.** `result.focalSearch` of the record is `{ ratio: 0.5835, gridRatio: 0.3, views: 5 }`. The
  grid is `0.3 × 1.15^i`, so 0.3 is its lowest point. The run of the same sample the day before
  (`20261025012536402-fd99c8db`, Turing) ended at 0.3 exactly, and the Desert canyon run
  `20261024151031417-106d9185` at 0.3000. The Water tower run `20261024145805483-44bb5d97` ended
  at 0.908 (grid 0.963).
- **Fact.** The calibration subset is the first five keyframes in input order: frames 0, 5, 9, 13
  and 17 (0 to 4.24 s). On the exported tracks of these five views the CPU reference mapper at a
  fixed focal gives (`measure-focal-sweep.mjs --views 5`, all 17 grid points; seven of them here):

| ratio | RMS px | observations kept | camera turn, first to last | truncated score px² |
| --- | --- | --- | --- | --- |
| 0.30 | 0.273 | 11,407 | 0.20° | 0.075 |
| 0.46 | 0.263 | 10,839 | 0.19° | 0.178 |
| 0.60 | 0.263 | 10,355 | 0.16° | 0.270 |
| 0.80 | 0.258 | 9,897 | 0.13° | 0.356 |
| 0.92 | 0.263 | 9,612 | 0.12° | 0.412 |
| 1.21 | 0.259 | 9,037 | 0.09° | 0.521 |
| 2.81 | 0.262 | 8,060 | 0.04° | 0.709 |

  The RMS error is flat over the whole grid (0.258 to 0.273 px). The observations kept fall
  monotonically as the ratio grows. The truncated score of `scoreCandidates`
  (`(m·e² + (n − m)·τ²) / n`, τ = 1.5 px) therefore rises monotonically, and the lowest grid point
  wins. Caveat: these tracks survived the lab's run at 0.58, which favours that region; the
  monotonic trend does not depend on it.
- **Inference.** In the first 4 s the camera translates and barely turns (0.04° to 0.20° in every
  run), so the subset does not observe the focal. A smaller focal makes every triangulation angle
  larger (the angle is about the disparity over the focal), so more tracks pass the 0.5°
  `minTriangulationAngleDeg` gate and more observations stay. The score counts each dropped
  observation as a τ² misfit, so it rewards the smallest focal. This is a bias of the score, not
  evidence for 0.3.
- **Fact.** At `finish()` a boundary winner triggers the restarts of `historyFocalSeeds` (the
  current 0.3, the grid's middle point and the opposite end 3.0) inside `refineHistory`; the
  accepted result was 0.5835. In `fd99c8db` it stayed at 0.3. `docs/auto-focal.md` 2.3, item 3,
  already records that a map built at a wrong focal is a different local minimum.

### 3.2 The sparse structure is a projectively distorted solution

- **Fact.** Similarity alignment of the 47 camera poses to the reference (`analyze-record.py
  colmap-compare`): rotation difference median 6.8°, maximum 14.2°; camera-centre difference
  median 0.31 and maximum 0.55 of the reference trajectory's radius. Mean reprojection error:
  TypeCV 0.70 px, COLMAP 0.35 px.
- **Fact.** Depth relief on the same pixels (`analyze-record.py relief`: COLMAP's sparse depth at
  its feature pixels against TypeCV's dense `geometric-filtered` depth at those pixels):

| view (time) | common pixels | depth spread p95 / p5, COLMAP | the same, TypeCV | log-log slope, TypeCV against COLMAP | note |
| --- | --- | --- | --- | --- | --- |
| 0 (0.0 s) | 3,546 | 1.146 | 1.146 | +0.97 (MAD 0.004) | the same relief |
| 62 (15.5 s) | 8,370 | 1.196 | 1.094 | −0.15 | |
| 104 (26.0 s) | 5,669 | 1.390 | 1.280 | −0.45 | |
| 120 (30.0 s) | 5,700 | 2.164 | 1.555 | −0.35 | bottom third of the frame: COLMAP 1.28 × the median depth, TypeCV 0.81 ×; left third: 0.77 × against 1.03 × |
| 137 (34.2 s) | 3,690 | 2.925 | 1.683 | −0.20 | right third: COLMAP 1.72 ×, TypeCV 0.98 × |

  The first view agrees with the reference. From 15 s on, the depth ordering across the image is
  compressed and reversed. At 30 s the ground below the cliff, at the bottom right of the frame, is
  nearer than the top of the rock in TypeCV's solution and farther in the reference's.
- **Inference.** This is the bas-relief family of solutions: less camera rotation, a smaller focal
  and a distorted relief reproject almost as well as more rotation, a larger focal and the true
  relief, when the tracks are short and a biased search fixes the focal. What the owner saw in the
  viewer is such a solution seen from anywhere but the cameras.

### 3.3 The dense stage follows the sparse structure; it adds no systematic collapse

- **Fact.** Dense depth at the landmark pixels, the seeds of the MVS, against the landmark depth
  (`analyze-record.py seeds`): view 0, median 0.0 % and 91 % within 2 %; view 62, 75 % within 2 %;
  view 120, 67 % within 2 % and 98 % within 10 %; view 137, 54 % within 2 % and 80 % within 10 %.
  Landmarks without a dense depth at their pixel: 59 %, 25 %, 33 %, 20 % and 50 % for the views 0,
  62, 104, 120 and 137.
- **Fact.** View 120, `geometric-filtered` map, 64.4 % valid (`analyze-record.py depth-view`): the
  angle between the finite-difference normal of the depth surface and the viewing ray has the
  median 55.8° (step 2 px) and 46.6° (median-filtered, step 6 px); the stored PatchMatch normal,
  53.4°. The shares below 10° are 1.3 % and 4.0 %. In the raw `photometric` map the
  finite-difference angle has the median 82.8°, which is noise. The normals of the dense PLY have
  a median angle of 52° to the ray of every camera that sees the point (`analyze-record.py
  normals`). There is no fronto-parallel peak.
- **Fact.** View 120 in 4 × 3 blocks of 320 × 240 px (`analyze-record.py blocks`): the robust
  plane normals of the sparse landmarks and of the dense depth map differ by 2.0° to 11.3° in ten
  blocks, and by 17.2° and 26.9° in two.
- **Inference.** The depth maps add noise and holes and keep the sparse shape. A change in the
  dense stage cannot correct the shape; the shape is decided upstream.

### 3.4 Short tracks and near-only matching

- **Fact.** `registerFrame` in `website/src/workers/sfm.worker.ts` keeps the descriptors of "a
  long-baseline first reference, two recent reference images and three adjacent frames" and
  disposes the rest, so each keyframe is matched against about four earlier keyframes: 176
  matching calls and 176 matched pairs for 47 keyframes. COLMAP matched all 1,081 pairs. The
  median track length is 3 against 8, the mean 4.7 against 12.1. Longer links in the exported map
  (713 view pairs share 16 or more landmarks) exist through chained tracks only.
- **Fact.** The MVS source selection (`planMvsViews`; the score is the sum over the shared
  landmarks of g(angle)) picks temporal neighbours only (`measure-mvs-sources.mjs`): over all 282
  source links, the median time gap is 3.0 s, the median keyframe gap 4 positions in time order
  (the largest 9), the median triangulation angle 5.7°; 36 % of the links have a median angle below
  5° and 93 % below 10°.
- **Inference.** Both limits keep the parallax of the other side of the orbit out: the sparse stage
  never matches distant keyframes directly, and the dense stage never reads a source from there.
  The first limit weakens the focal and the global shape; the second limits dense accuracy and
  completeness and does not cause the distortion.

### 3.5 The same tracks at a fixed focal near the reference fit far better

- **Fact.** `measure-focal-sweep.mjs` on all 47 views and the exported tracks (713 pairs with 16
  or more matches, `--min-matches 16`; 90,391 keypoints), the CPU reference mapper with the lab's
  controls and `refineSharedFocal: false`, the focal held at each ratio:

| ratio | views | landmarks | observations | RMS px | mean px | camera turn, first to last |
| --- | --- | --- | --- | --- | --- | --- |
| 0.30 | 47 | 17,859 | 83,569 | 0.792 | 0.605 | 21.1° |
| 0.5835 (the lab's result) | 47 | 18,131 | 84,444 | 0.738 | 0.556 | 41.0° (the lab: 41.5°) |
| 0.80 | 47 | 16,293 | 80,563 | 0.604 | 0.435 | 65.2° |
| 0.97 | 47 | 16,658 | 83,818 | 0.317 | 0.227 | 61.2° |
| 1.30 | 47 | 15,840 | 77,855 | 0.643 | 0.474 | 57.1° |

  The script's default of 32 matches keeps 623 pairs and moves these numbers in the fourth digit
  (0.5835: RMS 0.7388 px, 18,133 landmarks, 40.99°; 0.97: 0.3169 px, 16,660, 61.19°). COLMAP on the
  same images: ratio 0.98, 0.348 px, 61.5°.
- **Fact.** The depth relief of view 120 (`analyze-record.py sweep-relief`: the median landmark
  depth of each image third over the view's median depth): at 0.5835, rows top / middle / bottom
  1.070 / 0.996 / 0.801 and columns left / centre / right 1.078 / 0.941 / 0.984; at 0.97, rows
  0.831 / 1.069 / 1.331 and columns 0.742 / 1.008 / 1.154. The reference, on its own features:
  rows 0.894 / 1.022 / 1.276, columns 0.770 / 1.019 / 1.134. View 137 at 0.97: columns
  0.625 / 1.011 / 1.591 (reference 0.757 / 1.122 / 1.721); at 0.5835: 1.243 / 0.991 / 0.974.
- **Inference.** Even on tracks selected under the 0.58 calibration, a map rebuilt at 0.97 halves
  the reprojection error, restores the camera rotation and restores the reference's depth ordering.
  The lab's 0.58 is a local minimum of the joint focal refinement that started from the map built
  at 0.3; the restart from the grid's middle (`historyFocalSeeds`) reuses that map's poses and
  points, and a local solver cannot turn an inverted relief over. A rebuild at the restart focal
  (candidate 2 in section 5) would have found the 0.97 basin on this sample.

## 4. What was not run

- No TypeCV run of this sample at a fixed focal near 0.98 with the real images (the lab's manual
  focal), on a GPU or on SwiftShader. The sweeps above reuse the exported tracks, which were
  selected at 0.58.
- No COLMAP dense reconstruction, and no ground truth of the scene.
- The depth archives of `fd99c8db` and `106d9185` were not analysed.

## 5. Candidate changes

The owner decides. Each needs its own record.

1. The focal search score: evaluate every grid point on one common observation set, or hold the
   triangulation-angle gate in image terms (disparity) during the search, so that a focal is not
   rewarded for passing more tracks through the 0.5° gate. Choose the calibration subset by
   parallax and rotation, not by input order.
2. The final calibration: when the grid winner is a boundary, rebuild the observations at each
   restart focal instead of reusing the map built at the boundary focal.
3. The matching reach of one video: keep descriptors, or a retrieval index, for more keyframes, so
   that the far side of the orbit is matched directly; measure the track length and the focal
   against the reference on this sample.
4. The MVS source selection: an angle prior that prefers 5° to 15° and a cap on the weight of the
   shared-landmark count, so that sources from the other side of the orbit enter when they exist.

## 6. Reproduction

The commands below ran on 2026-10-26 in the environment of section 1, from the repository root.
`CLOUDFLARE_API_TOKEN` must reach the account of `wrangler.jsonc`; behind a proxy, wrangler's fetch
needs `NODE_USE_ENV_PROXY=1`. The Python requirements are
`scripts/requirements-record-analysis.txt` (Python 3.12 or later). The unit tests of the package run
with `python3 -m unittest discover -s scripts -p 'test_*.py'`.

```sh
node scripts/export-run-records.mjs --out lab-records --limit 120 --print none
node scripts/fetch-record-files.mjs --record lab-records/20261026043150261-58532c69.json --out files-58532c69
python3 scripts/analyze-record.py extract --manifest files-58532c69/typecv-depths-c000.json \
  --out files-58532c69/extracted
python3 scripts/analyze-record.py frames --archive files-58532c69/extracted --out frames \
  --png-views 0,62,104,120,137
node scripts/measure-mvs-sources.mjs --model files-58532c69/typecv-reconstruction.json
node scripts/measure-focal-sweep.mjs --model files-58532c69/typecv-reconstruction.json --views 5 \
  --ratios 0.3,0.345,0.3968,0.4563,0.5247,0.6034,0.6939,0.798,0.9177,1.0554,1.2137,1.3957,1.6051,1.8459,2.1228,2.4412,2.8074
node scripts/measure-focal-sweep.mjs --model files-58532c69/typecv-reconstruction.json \
  --ratios 0.3,0.5835,0.8,0.97,1.3 --min-matches 16 --models sweep-models
python3 scripts/analyze-record.py sparse --export files-58532c69/typecv-reconstruction.json
python3 scripts/analyze-record.py seeds --archive files-58532c69/extracted \
  --export files-58532c69/typecv-reconstruction.json --views 0,62,104,120,137
python3 scripts/analyze-record.py normals --ply files-58532c69/typecv-dense.ply \
  --export files-58532c69/typecv-reconstruction.json
python3 scripts/analyze-record.py depth-view --archive files-58532c69/extracted --view 120 --out depth-view
python3 scripts/analyze-record.py blocks --archive files-58532c69/extracted \
  --export files-58532c69/typecv-reconstruction.json --ply files-58532c69/typecv-dense.ply \
  --view 120 --pair 1,1 3,1
python3 scripts/analyze-record.py colmap-run --images frames/jpg --out colmap-radial \
  --camera-model SIMPLE_RADIAL --focal-prior 1.2
python3 scripts/analyze-record.py colmap-run --images frames/jpg --out colmap-pinhole \
  --camera-model SIMPLE_PINHOLE --focal-prior 0.8
python3 scripts/analyze-record.py colmap-compare --export files-58532c69/typecv-reconstruction.json \
  --colmap colmap-radial/sparse/0 --views 0,62,120
python3 scripts/analyze-record.py relief --archive files-58532c69/extracted \
  --colmap colmap-radial/sparse/0 --views 0,62,104,120,137
python3 scripts/analyze-record.py sweep-relief files-58532c69/typecv-reconstruction.json \
  sweep-models/ratio-0.5835.model.json sweep-models/ratio-0.97.model.json --views 120,137
```

## 7. Record of this document

- 2026-10-26: written from the analysis of the record (#239); `docs/auto-focal.md` gains open
  item 5 from section 3.1.
