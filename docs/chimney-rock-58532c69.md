# Record 58532c69 (Chimney Rock): the relief collapse comes from the focal calibration

Status: diagnosis record, 2026-10-26 (pull request #239). It changes no product code and decides
nothing. Section 5 lists candidate changes for the owner's decision; each needs its own record.
Labels: **Fact** (read or measured), **Inference** (reasoned from facts), **Not run** (a check that
did not run).

The owner reported that the 3D model of this run looks right only from the cameras' viewpoints.
Turned in the viewer, the top of the rock stands at about 45°. The cliff of the 30 s region does
not reach down, and the background is nearly a plane. This record measures where that comes from.

## 1. Configuration

| Item | Identifier |
| --- | --- |
| Run record | `20261026043150261-58532c69` in the collector's KV; site commit `5e1c95c46ac3a7dca291ab2a77f6f207be01e888` (`main` at the time of the analysis) |
| Input | Lab sample "Chimney Rock video B5 sample": 2560 × 1440 source, 1280 × 720 analysis, 34.2 s, 138 sampled frames at 4 fps, 47 keyframes (phase gate) |
| Options | SIFT with 16,384 features, guided matching, `focalSearch: true`, `refineSharedFocal: true`, `windowSize` 32; dense: TSDF fusion with `view-weighted` visibility, `planarDepth: true`, `maxSources` 6, `bestSources` 3, window radius 5 at step 2 |
| Result files | `typecv-reconstruction.json` (47 views, 19,123 landmarks, 90,391 observations), `typecv-dense.ply` (698,269 points), the depth archive `typecv-depths-c000.json` with parts `0000` to `0061` (47 views × 4 stages, inputs included) |
| Reference | pycolmap 4.2.1 on the 47 archived 1280 × 720 keyframes: SIFT on the CPU, exhaustive matching, incremental mapping with one shared camera and the focal refined. Two runs: `SIMPLE_RADIAL` with the prior 1.2 × the long side, `SIMPLE_PINHOLE` with the prior 0.8 × the width. COLMAP is a reference, not ground truth. |
| Environment | Linux x86_64 container without a GPU; Node 22.22.0; Python 3.13 with numpy 2.5.3, scipy 1.18.1, Pillow 12.3.0 (its JPEG encoder writes COLMAP's input) and pycolmap 4.2.1; the CPU reference mapper of `src/` at `5e1c95c` |

Section 6 gives the commands. The scripts are `scripts/fetch-record-files.mjs`,
`scripts/measure-mvs-sources.mjs`, `scripts/measure-focal-sweep.mjs` and
`scripts/analyze-record.py` (the package `scripts/record_analysis/`).

## 2. Claims checked

The owner brought an earlier analysis of the same record. Each of its claims was measured again.

| Claim | Measured here | Verdict |
| --- | --- | --- |
| Focal 747 px (TypeCV) against 1,242 px (COLMAP) | TypeCV `fx` = 746.885 at the width 1280 (ratio 0.5835; `gridRatio` 0.3). COLMAP: 1,255.1 px (`SIMPLE_RADIAL`, k1 = 0.011, prior 1.2) and 1,246.0 px (`SIMPLE_PINHOLE`, prior 0.8). The two COLMAP runs change the camera model and the prior together, and differ by 0.7 %. | Confirmed |
| View direction change, first to last view, 41.5° against 61.6° | 41.5° (TypeCV) against 61.5° and 61.8° (the two COLMAP runs) | Confirmed |
| Median observations per feature 3 against 10 | TypeCV median 3, mean 4.73 (6,459 of the 19,123 tracks have two views). COLMAP median 8, mean 12.1 | Confirmed; the reference's median is 8 here |
| Six other images connected around 30 s, with meaningful parallax | View 120 (30.0 s) has the sources 128, 130, 112, 110, 132 and 108, 2 to 3 s away. Their median triangulation angles are 4.4° to 6.7°. | Confirmed |
| The dense stage flattens the angle between the top and the front face from 49° to 5° | View 120 in 4 × 3 blocks. The robust planes of the depth map differ from the sparse planes by 2.0° to 11.3° in 10 of 12 blocks. Blocks (1,1) and (3,1) meet at 50.1° (sparse) and 53.1° (depth map). The fused PLY differs from the sparse planes by more than 10° in 8 of 12 blocks (up to 55.6°). It gives 76.7° for that pair. At the landmark pixels the dense depth equals the sparse depth (view 120: median −0.7 %), partly by construction (3.3). | Not reproduced for the depth maps (3.3). The fusion into the PLY was not analysed (section 4). |

## 3. Facts found

### 3.1 The focal search stopped at the grid's lower bound

- **Fact.** `result.focalSearch` of the record is `{ ratio: 0.5835, gridRatio: 0.3, views: 5 }`. The
  grid is `0.3 × 1.15^i`, so 0.3 is its lowest point. The run of the same sample the day before
  (`20261025012536402-fd99c8db`, Turing) ended at 0.3 exactly. The Desert canyon run
  `20261024151031417-106d9185` ended at 0.3000. The Water tower run `20261024145805483-44bb5d97`
  ended at 0.908 (grid 0.963).
- **Inference.** The calibration subset is the first five keyframes: frames 0, 5, 9, 13 and 17
  (0 to 4.24 s). The record holds only `views: 5`. The session (`IncrementalSfmSession.add` in
  `src/gpu/mapper.ts`) runs the search when its fifth keyframe arrives.
- **Fact.** The CPU reference mapper ran at a fixed focal on the exported tracks of these five
  views, in time order (`measure-focal-sweep.mjs --views 5`). It ran at all 17 grid points; the
  table shows seven:

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
  wins.

  These runs use the lab's `finalBundleIterations` of 8 (`--final 8`, the script's default). The
  search runs its grid points differently: with `bundleIterations` (4) as the final iterations,
  `rotationTracking: false` and one shared pair verification per pair. With `--final 4` and the
  other two differences left, the scores at 0.3, 0.798 and 2.807 are 0.0754, 0.3556 and 0.7088
  px². The trend holds.

  Caveat: these tracks survived the lab's run at 0.58, which favours that region. The trend was
  not measured on other tracks (section 4).
- **Inference.** In the first 4 s the camera translates and barely turns: 0.04° to 0.20° in every
  run. The subset therefore does not observe the focal. A smaller focal makes every triangulation
  angle larger (the angle is about the disparity over the focal). More tracks then pass the 0.5°
  `minTriangulationAngleDeg` gate, and more observations stay. The score counts each dropped
  observation as a τ² misfit, so it rewards the smallest focal. This is a bias of the score, not
  evidence for 0.3.
- **Fact.** At `finish()` a boundary winner makes `historyFocalSeeds` pass three starting focals to
  `refineHistory`. They are the current focal, the grid's middle point 0.918 and the last grid
  point 2.807. The current focal is not in the record. `refineHistory` keeps the solution of least
  cost, and the result was 0.5835. In `fd99c8db` it stayed at 0.3. `docs/auto-focal.md` 2.3, item
  3, already records that a map built at a wrong focal is a different local minimum.

### 3.2 The sparse structure is a projectively distorted solution

- **Fact.** Similarity alignment of the 47 camera poses to the reference (`analyze-record.py
  colmap-compare`): the rotation difference has the median 6.8° and the maximum 14.2°. The
  camera-centre difference has the median 0.31 and the maximum 0.55 of the reference trajectory's
  radius. The mean reprojection error is 0.70 px for TypeCV (`analyze-record.py sparse`,
  `meanReprojectionErrorPx` 0.6956) and 0.35 px for COLMAP (`colmap-run`, 0.348).
- **Fact.** Depth relief on the same pixels (`analyze-record.py relief`). It compares COLMAP's
  sparse depth at its feature pixels with TypeCV's dense `geometric-filtered` depth at those pixels:

| view (time) | common pixels | depth spread p95 / p5, COLMAP | the same, TypeCV | log-log slope, TypeCV against COLMAP | note |
| --- | --- | --- | --- | --- | --- |
| 0 (0.0 s) | 3,546 | 1.146 | 1.146 | +0.97 (residual scatter 0.004) | the same relief |
| 62 (15.5 s) | 8,370 | 1.196 | 1.094 | −0.15 | |
| 104 (26.0 s) | 5,669 | 1.390 | 1.280 | −0.45 | |
| 120 (30.0 s) | 5,700 | 2.164 | 1.555 | −0.35 | top third of the frame: COLMAP 0.89 × the median depth, TypeCV 1.04 ×; bottom third: 1.28 × against 0.81 ×; left third: 0.77 × against 1.03 × |
| 137 (34.2 s) | 3,690 | 2.925 | 1.683 | −0.20 | right third: COLMAP 1.72 ×, TypeCV 0.98 × |

  The residual scatter is 1.4826 × the median absolute deviation of the residuals from the
  Theil-Sen line (view 0: MAD 0.0029). A third's depth is its median over the common pixels,
  divided by the median of all the common pixels.
- **Inference.** The first view agrees with the reference. From 15 s on, the negative slopes mean
  that the depth order across the image is compressed and reversed. `relief.py` names this reading
  an inference too.
- **Inference.** The frame of view 120 shows the top of the rock in its upper part. It shows the
  ground below the cliff at the bottom right. Both are readings of the image. The reference puts
  the bottom third farther than the median and the top third nearer, as a camera above a cliff sees
  it. TypeCV's map reverses both.
- **Inference.** This is the bas-relief family of solutions. Less camera rotation, a smaller focal
  and a distorted relief reproject almost as well as more rotation, a larger focal and the true
  relief. This holds when the tracks are short and a biased search fixes the focal. What the owner
  saw in the viewer is such a solution seen from anywhere but the cameras.

### 3.3 The depth maps follow the sparse structure; the fusion was not analysed

- **Fact.** View 120 in 4 × 3 blocks of 320 × 240 px (`analyze-record.py blocks`). The robust plane
  normals of the sparse landmarks and of the `geometric-filtered` depth map differ by 2.0° to 11.3°
  in ten blocks. They differ by 17.2° and 26.9° in two.
- **Fact.** The planes of the fused PLY use its 170,419 points that agree with the depth map within
  5 %. They differ from the sparse planes by more than 10° in 8 of 12 blocks, up to 55.6° in block
  (0,2). Blocks (1,1) and (3,1) meet at 50.1° (sparse), 53.1° (depth map) and 76.7° (PLY).
- **Fact.** Dense depth at the landmark pixels against the landmark depth (`analyze-record.py
  seeds`). View 0: median 0.0 %, 91 % within 2 %. View 62: 75 % within 2 %. View 120: 67 % within
  2 % and 98 % within 10 %. View 137: 54 % within 2 % and 80 % within 10 %. Landmarks without a
  dense depth at their pixel: 59 %, 25 %, 33 %, 20 % and 50 % for the views 0, 62, 104, 120 and 137.
- **Fact.** This agreement is partly by construction. `mvsSeedAt` (`src/gpu/mvs.shade.ts`) writes
  each landmark's depth at its pixel before the PatchMatch iterations. The agreement measures
  whether the iterations and the filter kept the seeds. It is not independent evidence that the
  dense stage keeps the shape; the block planes are.
- **Fact.** View 120, `geometric-filtered` map, 64.4 % valid (`analyze-record.py depth-view`). The
  finite-difference normal of the depth surface makes a median angle of 55.8° with the viewing ray
  (step 2 px). After a 5 × 5 median filter (step 6 px) the median is 45.9°. The stored PatchMatch
  normal gives 53.4°. The shares below 10° are 1.3 %, 2.3 % and 4.0 % for these three normals. In
  the raw `photometric` map the finite-difference angle has the median 82.8°. The normals of the
  dense PLY have a median angle of 52° to the ray of every camera that sees the point
  (`analyze-record.py normals`). No distribution peaks near 0° (fronto-parallel).
- **Inference.** The 82.8° of the raw map is noise. The depth maps add noise and holes and keep the
  sparse shape. A change in the depth-map stage cannot correct the shape; the shape is decided
  upstream. The fused cloud that the viewer shows departs further from the sparse planes, and this
  record does not explain why (section 4).

### 3.4 Short tracks and a matching reach of one window

- **Fact.** `registerFrame` in `website/src/workers/sfm.worker.ts` keeps the descriptors of "a
  long-baseline first reference, two recent reference images and three adjacent frames". It
  disposes the rest. Each new keyframe is matched against every kept descriptor of a view in the
  session window (`windowSize` 32, `src/gpu/mapper.ts`).
- **Fact.** A replay of these rules over the 47 keyframe times gives 176 matching calls, the
  record's count (`timing.stages.matches.calls`). The record has 176 matched pairs
  (`result.matchedPairs`), so every call became a pair. In the replay, frame 0 stays the
  long-baseline reference and is matched directly with every later keyframe up to frame 110.
  Frame 110 is 27.5 s later, at a COLMAP turn of 35.0° from frame 0 (61.5° in all). After the
  window drops frame 0, frame 108 takes its place and is matched with every later keyframe. Apart
  from these, a keyframe meets two recent references and three adjacent keyframes.
- **Fact.** COLMAP matched all 1,081 pairs. The median track length is 3 against 8, the mean 4.7
  against 12.1. The export has 713 view pairs that share 16 or more landmarks. 173 of them were
  matched directly in the replay; the other 540 share landmarks through tracks that pass other
  keyframes. In 86 tracks the next observation after view 0 lies more than 3 keyframes later, more
  than after any other view (39 after view 5).
- **Fact.** The MVS source selection (`planMvsViews`; the score is the sum over the shared landmarks
  of g(angle)) picks temporal neighbours only (`measure-mvs-sources.mjs`). Over all 282 source
  links, the median time gap is 3.0 s and the median keyframe gap 4 positions in time order (the
  largest 9). The median triangulation angle is 5.7°. 35 % of the links have a median angle below
  5° and 93 % below 10°.
- **Inference.** The sparse stage links distant keyframes directly only through one view, the
  window's long-baseline reference, and never beyond the 32-view window. The dense stage never
  reads a source more than 9 keyframes away. The first limit keeps the tracks short and weakens the
  focal and the global shape. The second limits dense accuracy and completeness and does not cause
  the distortion.

### 3.5 The same tracks at a fixed focal near the reference fit far better

- **Fact.** `measure-focal-sweep.mjs` on all 47 views in time order and the exported tracks: 713
  pairs with 16 or more matches (`--min-matches 16`) and 90,391 keypoints. The CPU reference mapper
  runs with the lab's controls and `refineSharedFocal: false`, the focal held at each ratio:

| ratio | views | landmarks | observations | RMS px | mean px | camera turn, first to last |
| --- | --- | --- | --- | --- | --- | --- |
| 0.30 | 47 | 17,859 | 83,569 | 0.792 | 0.605 | 21.1° |
| 0.5835 (the lab's result) | 47 | 18,131 | 84,444 | 0.738 | 0.556 | 41.0° (the lab: 41.5°) |
| 0.80 | 47 | 16,293 | 80,563 | 0.604 | 0.435 | 65.2° |
| 0.97 | 47 | 16,658 | 83,818 | 0.317 | 0.227 | 61.2° |
| 1.30 | 47 | 15,840 | 77,855 | 0.643 | 0.474 | 57.1° |

  The script's default of 32 matches keeps 623 pairs and moves these numbers in the fourth digit.
  At 0.5835 it gives RMS 0.7388 px, 18,133 landmarks and 40.99°; at 0.97, 0.3169 px, 16,660 and
  61.19°. COLMAP on the same images: ratio 0.98, 0.348 px, 61.5°.
- **Fact.** The depth relief of view 120 (`analyze-record.py sweep-relief`): the median depth of the
  model's landmarks in each image third over the view's median landmark depth. At 0.5835: rows top
  / middle / bottom 1.070 / 0.996 / 0.801, columns left / centre / right 1.078 / 0.941 / 0.984. At
  0.97: rows 0.831 / 1.069 / 1.331, columns 0.742 / 1.008 / 1.154. View 137 at 0.97: columns
  0.625 / 1.011 / 1.591; at 0.5835: 1.243 / 0.991 / 0.974.
- **Fact.** The reference's thirds come from `relief` (3.2), on another set of pixels. They use
  COLMAP's depth at the 5,700 of its 8,645 feature pixels of view 120 where TypeCV's filtered map is
  valid. The median of that subset normalizes them. Rows 0.894 / 1.022 / 1.276, columns 0.770 /
  1.019 / 1.134. View 137 (3,690 of 7,527 pixels): columns 0.757 / 1.122 / 1.721.
- **Inference.** Even on tracks selected under the 0.58 calibration, a map rebuilt at 0.97 halves
  the reprojection error. It restores the camera rotation and the reference's depth ordering. The
  lab's 0.58 is a local minimum of the joint focal refinement. That refinement started from the map
  built near 0.3, and the restarts of `historyFocalSeeds` reuse that map's poses and points. A
  local solver cannot turn an inverted relief over.
- **Inference.** A rebuild at the restart focal 0.918 (candidate 2 in section 5) might reach the
  basin of the 0.97 map on this sample. This was not run: the sweep held 0.97, not 0.918, on tracks
  selected at 0.58 (section 4).

## 4. What was not run

- No TypeCV run of this sample with the real images at a fixed focal near 0.98, the lab's manual
  focal. Neither a GPU nor SwiftShader ran it. The sweeps above reuse the exported tracks, which
  were selected at 0.58.
- No rebuild of the map at the restart focal 0.918.
- No focal sweep on tracks that the lab's run did not select, for example COLMAP's tracks.
- The fusion of the depth maps into `typecv-dense.ply` was not analysed. The fused cloud differs
  from the sparse planes by more than 10° in 8 of 12 blocks of view 120 (3.3).
- No COLMAP dense reconstruction, and no ground truth of the scene.
- The depth archives of `fd99c8db` and `106d9185` were not analysed.

## 5. Candidate changes

The owner decides. Each needs its own record.

1. The focal search score. Evaluate every grid point on one common observation set, or hold the
   triangulation-angle gate in image terms (disparity) during the search. Then a focal is not
   rewarded for passing more tracks through the 0.5° gate. Choose the calibration subset by
   parallax and rotation, not by input order.
2. The final calibration. When the grid winner is a boundary, rebuild the observations at each
   restart focal. Do not reuse the map built at the boundary focal.
3. The matching reach of one video. Today a keyframe meets distant keyframes directly only through
   the window's long-baseline reference, and never beyond the 32-view window. Keep descriptors, or
   a retrieval index, for more keyframes, so that distant keyframes are matched directly. Measure
   the track length and the focal against the reference on this sample.
4. The MVS source selection. Add an angle prior that prefers 5° to 15°, and cap the weight of the
   shared-landmark count. Then sources from the other side of the orbit enter when they exist.

## 6. Reproduction

The commands below ran on 2026-10-26 in the environment of section 1, from the repository root.
They are POSIX `sh` lines. In PowerShell, write each command on one line without the `\` breaks,
and run `py` (or `python`) in place of `python3`.

`CLOUDFLARE_API_TOKEN` must reach the account of `wrangler.jsonc`; behind a proxy, wrangler's fetch
needs `NODE_USE_ENV_PROXY=1`. `--limit 120` exports the newest 120 records, which held this record
on 2026-10-26. After more records arrive, raise the limit until the export holds it. The Python
requirements are `scripts/requirements-record-analysis.txt` (Python 3.12 or later). The unit tests
of the package run with `python3 -m unittest discover -s scripts -p 'test_*.py'`.

```sh
node scripts/export-run-records.mjs --out lab-records --limit 120 --print none
node scripts/fetch-record-files.mjs --record lab-records/20261026043150261-58532c69.json --out files-58532c69
python3 scripts/analyze-record.py extract --manifest files-58532c69/typecv-depths-c000.json \
  --out files-58532c69/extracted
python3 scripts/analyze-record.py frames --archive files-58532c69/extracted --out frames \
  --png-views 0,62,104,120,137
node scripts/measure-mvs-sources.mjs --model files-58532c69/typecv-reconstruction.json
node scripts/measure-focal-sweep.mjs --model files-58532c69/typecv-reconstruction.json --views 5 \
  --ratios 0.3,0.345,0.3967,0.4563,0.5247,0.6034,0.6939,0.798,0.9177,1.0554,1.2137,1.3957,1.6051,1.8458,2.1227,2.4411,2.8073
node scripts/measure-focal-sweep.mjs --model files-58532c69/typecv-reconstruction.json --views 5 \
  --final 4 --ratios 0.3,0.798,2.8073
node scripts/measure-focal-sweep.mjs --model files-58532c69/typecv-reconstruction.json \
  --ratios 0.3,0.5835,0.8,0.97,1.3 --min-matches 16 --models sweep-models
node scripts/measure-focal-sweep.mjs --model files-58532c69/typecv-reconstruction.json \
  --ratios 0.5835,0.97
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
  --colmap colmap-radial/sparse/0 --depth-views 0,62,120
python3 scripts/analyze-record.py relief --archive files-58532c69/extracted \
  --colmap colmap-radial/sparse/0 --views 0,62,104,120,137
python3 scripts/analyze-record.py sweep-relief files-58532c69/typecv-reconstruction.json \
  sweep-models/ratio-0.5835.model.json sweep-models/ratio-0.97.model.json --views 120,137
```

The replay of 3.4 is not one of these scripts. It applies the two rules of `registerFrame` and of
the session window to the keyframe times of the export.

## 7. Record of this document

- 2026-10-26: written from the analysis of the record (#239); `docs/auto-focal.md` gains open
  item 5 from section 3.1.
- 2026-10-26: corrected after the review of #239. Section 3.4 and candidate change 3 now follow a
  replay of the matching rules. The dense verdict holds for the depth maps only, and the fusion is
  listed as not analysed. The labels, the search settings of 3.1 and the pixel sets of the thirds
  are stated. The median-filtered normal now leaves out zero depths: 46.6° became 45.9°. The
  sweeps now pass the views in time order, and no number of 3.1 or 3.5 changed.
