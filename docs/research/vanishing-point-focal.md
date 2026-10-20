# Vanishing-point focal length: feasibility measurement

Status: research note. It changes no product code.
Script: `scripts/research/vp_focal.py` (Python, numpy and PIL only).
Configuration: branch `research/vanishing-points` from `origin/main`, run on 2026-10-20.
Samples: `website/public/sample/` at the same commit, 640 px images.

## Question

Can vanishing points and the image of the absolute conic (IAC) estimate the camera focal length
from the lab's sample images? The lab assumes `DEFAULT_VIDEO_FOCAL` = 0.8 × image width for
uncalibrated video (PR #83). The earlier default of 1.2 × width reconstructed the pyramid upside
down.

## Method

Fact: OpenCV and SciPy are not installed in this environment, and the disk had no room to install
them. The script therefore carries its own detector.

1. Smooth the image with a 5-tap binomial filter. Compute central-difference gradients.
2. Detect line segments in the style of LSD. Grow regions of pixels with similar level-line angle
   (22.5° tolerance, gradient magnitude above 5 grey levels), starting from the strongest pixels.
3. Fit a line to each region by weighted PCA. Keep elongated regions (axis ratio below 0.18,
   length at least 15 px).
4. Find up to five vanishing points with sequential RANSAC (800 samples each). The vanishing
   point is the cross product of two segment lines. A segment is an inlier when its direction is
   within 2° of the direction from its midpoint to the vanishing point. The support of a vanishing
   point is the summed inlier length. Refit by weighted least squares, remove the inliers, repeat.
5. For each pair of vanishing points, solve `v1ᵀ ω v2 = 0` with square pixels, zero skew and the
   principal point at the image centre. In centred homogeneous coordinates
   `f² = −(x1·x2 + y1·y2) / (w1·w2)`. Reject pairs with no real f or f outside 0.25 to 4 × width.
6. Pick the pair with the best score: the support of the weaker vanishing point, plus a bonus
   when a third vanishing point is orthogonal to both and gives the same f within 15%. A
   triplet-consistent result reports the median of its three f values.
7. Confidence = min(1, 3 × score). Confidence 0.8 or more is called "gated" below.

Not modelled: lens distortion (the COLMAP references carry a small `k1`), the principal point
(taken at the centre), and any scene prior such as "one vertical direction".

Caveat (fact): the segment thresholds were changed once after a first run on `south-building`
(minimum length 25 to 15 px, gradient threshold 8 to 5), to get enough segments. The thresholds
were chosen for segment count, not for the error in f. Nothing else was tuned. The confidence
gate of 0.8 was chosen after seeing the results, so the gated numbers are optimistic.

## Result 1: COLMAP samples (known f)

The true f is the per-frame `fx` in `sample.json`. Error = (estimate − true) / true.
Each sample has 12 frames, 48 in total.

| Sample         | Median abs error, all frames | Frames within 10% | Frames within 20% | Median-of-frames f vs true | Gated frames (conf ≥ 0.8) | Gated median abs error |
| -------------- | ---------------------------- | ----------------- | ----------------- | -------------------------- | ------------------------- | ---------------------- |
| south-building | 25.9%                        | 3/12              | 4/12              | +22.0%                     | 1                         | 7.2%                   |
| gerrard-hall   | 35.7%                        | 3/12              | 3/12              | +35.7%                     | 3                         | 1.4%                   |
| person-hall    | 41.9%                        | 0/12              | 1/12              | +37.2%                     | 0                         | none                   |
| graham-hall    | 11.9%                        | 4/12              | 10/12             | +11.8%                     | 11                        | 11.4%                  |
| all 48 frames  | 28.3%                        | 10/48             | 18/48             |                            | 15                        | 9.7% (max 32.8%)       |

Per-frame signed errors (%), frame 00 to 11:

| Sample         | Errors                                             |
| -------------- | -------------------------------------------------- |
| south-building | -7, 23, 14, 208, -6, 277, 21, 29, 259, 4, -29, 35  |
| gerrard-hall   | 299, 22, 39, 54, 32, 31, 66, 2, 1, 0, 54, 247      |
| person-hall    | 445, -52, 38, 24, 41, 47, 225, 16, 28, 95, 37, -43 |
| graham-hall    | 3, 2, 25, 15, 33, 10, 12, 15, 5, 11, 11, 13        |

Inferences:

- The method works only on some images. Graham-hall is a clean, frontal building with a strong
  vertical and two horizontal families. It gives a usable f (+12% median, 10 of 12 within 20%).
- The other three give large errors, and the errors are mostly positive. The wide-error frames
  (+200% and more) have a near-degenerate pair: two vanishing points that are almost parallel
  to the image plane, where `x1·x2 + y1·y2` is close to zero and f blows up.
- Triplet consistency was the most reliable signal. 8 of 48 frames had a triplet-consistent
  answer. Five of the 8 were within 8% (three `gerrard-hall` frames at about 1%, one
  `graham-hall` frame at 2.5%, one `south-building` frame at -7%). The other three
  (`graham-hall`) were at +12%, +25% and +33%.
- The confidence is a weak predictor on its own. `person-hall` has frames with confidence 0.6 and
  errors of 40% or more.

## Result 2: pyramid (true f unknown)

| Frame | Segments | Estimated f (px) | f / width | Confidence |
| ----- | -------- | ---------------- | --------- | ---------- |
| 00    | 121      | 279              | 0.44      | 0.51       |
| 01    | 124      | 504              | 0.79      | 0.26       |
| 02    | 121      | 504              | 0.79      | 0.29       |
| 03    | 116      | 486              | 0.76      | 0.52       |
| 04    | 98       | 405              | 0.63      | 0.62       |
| 05    | 95       | 197              | 0.31      | 0.42       |
| 06    | 90       | 542              | 0.85      | 0.41       |
| 07    | 84       | 1389             | 2.17      | 0.42       |

- Fact: the median is 495 px (0.77 × width). The median absolute deviation is 14%. The range is
  197 to 1389 px. Five of eight frames are in 400 to 650 px. No frame is within 15% of 768 px.
- Fact: no frame reaches confidence 0.8. The highest is 0.62.
- Inference: the median is closer to 0.8 × width (512 px, -3%) than to 1.2 × width (768 px,
  -36%). It is weak evidence only. The frames are near-nadir, and the road and building grids
  give one orthogonal pair on a nearly planar scene. The measured spread on COLMAP frames of
  similar confidence (up to 40% error) is larger than the gap that decides between the two
  defaults. It also does not show that 0.8 is right: it shows that 1.2 is less likely.

## Recommendation (inferences)

1. A vanishing-point focal initializer is worth building as an optional hint, not as a
   replacement for the 0.8 default. The best case (a frontal building with several clean line
   families) gives about 10% accuracy. That is useful to choose between 0.8 and 1.2, but it
   is not tight enough to replace bundle adjustment.
2. Use the estimate only when all of these hold:
   - three vanishing points are found, and their three pairwise f values agree within 15%
   - each of the three has a large share of the segment length
   - the pair is not near-degenerate (the denominator of f² is far from zero)
   - the estimate agrees across several frames of the clip (median over frames, spread below
     about 15%)
3. Fall back to 0.8 × width on:
   - near-nadir or aerial scenes, and near-planar scenes (such as the pyramid)
   - natural scenes, foliage, and scenes with few straight edges
   - images with no triplet-consistent answer (40 of 48 COLMAP frames here)
   - clips that disagree between frames by more than about 15%
4. Before building, measure again on a larger set. This run used 4 scenes and 48 frames, a
   self-written detector, and no distortion model. An LSD implementation from OpenCV and a
   Manhattan-world prior (one vertical and two horizontal directions) are likely to help, and
   this note did not measure them.
5. A cheaper alternative to test first: score the 0.8 and 1.2 candidates by the vanishing-point
   orthogonality residual. This needs only a yes/no choice, not an exact f.

## Reproduce

1. Run `python3 scripts/research/vp_focal.py --json out.json` from the repository root.
2. Read one line per frame: sample, file, segment count, f, confidence, triplet flag, error %.

RANSAC uses a fixed seed, so a rerun on the same machine gives the same numbers.
