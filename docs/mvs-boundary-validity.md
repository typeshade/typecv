# MVS boundary support and finite-depth observability

Baseline: `e4e5046`. The experimental options are independent and disabled by default:

- `boundaryAware: true` centers an odd sample lattice on the reference pixel and uses
  bilateral weights with intensity sigma 0.1. For radius 5 and step 2 the offsets are
  `[-4, -2, 0, 2, 4]`; the legacy offsets remain `[-5, -3, -1, 1, 3, 5]`.
- `minFiniteGain: 0.02` requires each supporting source to fit the finite plane better than a
  valid rotation-only warp by this NCC-cost margin. A directional cost above the existing NCC
  bound (including invalid comparisons) does not veto geometric support. Two non-ambiguous
  sources must also pass the existing
  depth/reprojection/angle rules. This withholds ambiguous depth; it does not classify sky,
  assert exact infinity, lower the 3-degree rule or infer an unobserved surface.

The rotation-only homography is `K_s R_s R_r^T K_r^-1`. It uses no very large finite depth or
translation. Finite and directional costs use exactly the same reference samples and weights.
The window and grouped paths use the same observability predicate and allocate no additional
pixel-sized GPU buffers. `minFiniteGain: 0` retains the existing support-count rule.

## Tasks and acceptance

1. Freeze the supplied three reports and provide reproducible input/hash checks.
2. Verify centered support on a one-pixel foreground line and a finite textured surface.
3. Verify withholding of a rotation-compatible depth and identical window/grouped decisions.
4. Check CPU/WebGPU regressions, memory, chunks and default-disabled compatibility.
5. Compare all three recorded scenes, publish losses as well as benefits, then select defaults.

The PR task list is updated only when each task's implementation and validation are complete.
Quality acceptance must consider both retained foreground and withheld background; fewer points
by itself does not establish an improvement. Reflective water is an additional negative control,
not a claimed solution of this change. No lab default changes are made before that comparison.

## Frozen real-scene replay

The exported files stay outside the repository. Place the report JSON/depth manifest and all
`typecv-depths-c000-*.tar.gz` parts under `record-6c42e7d3`, `record-a4d735ce` and
`record-1e118cc5` in one directory. Python preparation requires numpy, Pillow and scipy.

```sh
# Build the baseline in a separate checkout, then retain its self-contained internal bundle.
git worktree add --detach /tmp/typecv-mvs-baseline e4e5046
cd /tmp/typecv-mvs-baseline
npm ci
npm run build
cp dist/internal/gpu.js /tmp/mvs-baseline.mjs
# Return to this PR's checkout.
cd -
npm ci
npm run build
python scripts/prepare-mvs-boundary-replay.py --records /path/to/records --out /tmp/mvs-inputs
TYPESHADE_CHROMIUM=/usr/bin/chromium node scripts/measure-mvs-boundary.mjs \
  --cases /tmp/mvs-inputs --baseline /tmp/mvs-baseline.mjs --out /tmp/mvs-results
```

The preparation streams original analysis RGBA from the tar parts and takes the recorded
reference and its recorded six source cameras. It adjusts intrinsics and observations with the
pixel-center convention for a 320x180 overview of each scene, plus a 320x320 tower crop retaining
its original pixel pitch. The reference/source set is fixed between variants. Seeds, depth ranges
and normalization are recomputed for that seven-view subset; this is a **controlled depth-stage
replay**, not a reproduction of every original full-resolution output or a final TSDF benchmark.
It does not rerun SfM, select new cameras or measure surface completeness against ground truth.

The source/crop/scene/mask hashes are in
[the input manifest](experiments/dense-mvs/boundary-replay-inputs.json). The measurement command
checks image/scene/mask hashes before running any shader. Output files contain per-ROI accepted
pixel counts, the actual selected source IDs, normalized depth scale, raw filtered depth/normals,
backend information and baseline/candidate bundle hashes. `--case` and `--modes` select subsets.
Iterations default to 4 photometric / 4 geometric; the supplied original runs used 8 / 4.

ROI masks are hand-annotated evaluation controls, never production masks. A sky-colored object
must not be removed because of its color. The finite-depth rule is deliberately color-independent;
a genuinely finite periodic or very low-parallax surface may still be ambiguous and withheld.
Software WebGPU timings are not physical GPU performance measurements. Concurrent runs must
not be interpreted as a speed comparison.

## Relationship to PR #223

PR #223 adds directional factors to focal calibration and bundle adjustment upstream. This PR
uses the resulting cameras for a different purpose: deciding whether dense finite depth beats
a rotation-only image explanation. It does not add a second focal optimizer or convert a
direction factor into a finite landmark. Accurate intrinsics/rotations matter to the dense test;
the two changes can complement each other, but combined quality has not been established.

The current replay freezes cameras from the supplied reports to isolate the dense change. Once
#223's focal-search integration is complete, integration acceptance should compare all four
combinations of old/new SfM and old/new MVS on the same decoded frames and matching settings.
Measure registration/calibration, finite foreground coverage, sky artifacts and thin structures
separately. Do not attribute an improvement from changed cameras to this MVS filter, and do not
enable defaults merely because both PRs independently pass synthetic fixtures.

## Initial behavioral checks

`tests/typeshade/mvs-boundary.test.mjs` runs the production shader through TypeShade's f64/f32
CPU generators. The calibrated fixture moves finite texture by ten pixels between cameras while
world-direction texture stays still. It verifies an observable finite surface and a one-pixel
foreground line, an ambiguous periodic direction, and a background pixel beside the line. The
old rule demonstrably accepts the ambiguous depth; the new rule withholds it. Both scalar and
streamed filter decisions and confidence values agree. Existing reference-patch and inverse-depth
sampling tests also pass. The completed [validation record](experiments/dense-mvs/boundary-validation.json) includes
CPU window/grouped/tail-chunk comparisons, WebGPU grouped comparisons, finite-scene accuracy,
and default-disabled complete map/TSDF hashes. On the 49x37 synthetic scene (5 views, 2/1
iterations), the combined option retains 66.19% and has 2.13% median relative depth error.
The baseline retains 60.67% with 3.57% median error. WebGPU agrees with CPU at median relative
depth difference 2.17e-7 over 1,200 common accepted pixels. These are fixture results, not
real-scene accuracy. All profiles have a planned peak of 165,400 array bytes and release live
arrays. No additional image-sized GPU buffers are introduced.

Default-disabled baseline/candidate full map hashes and 307-point TSDF hashes are identical
in window and grouped modes. The two new controls also reach the public typed GPU API. A real
browser worker test verifies enabled, absent and explicitly disabled settings across consecutive
runs; report validation preserves their values and rejects invalid values. Core (25), focused
shader (3), CPU maps (2), WebGPU maps (1), dense/quality unit (19), worker/multi-video (51), and
quality worker browser (1) checks pass. Typecheck, core/GPU build, Astro check (0 errors; four
existing hints) and the 89-page site build pass. The WebGPU backend is software SwiftShader.


## Explicit lab comparisons

The lab leaves both experiments off by default. After this PR is deployed, add
`?dense=1&denseFiniteGain=0.02` to test observability alone, `?dense=1&denseBoundary=1`
for boundary support alone, or both parameters for the combination. The same overrides reach
single-video and multi-video Dense runs. Missing parameters preserve legacy options; explicit
zero disables an experiment. Invalid values are ignored. Exported run records preserve the
actual options, including explicit false/zero, so a later report identifies what ran.

The three-scene comparison is complete. Both experiments remain explicit opt-ins because the
measured gains do not meet a scene-independent default-enabling gate.


## Three-scene result and default decision

The [measurement record](experiments/dense-mvs/boundary-three-scenes.json) preserves input,
bundle and output hashes, ROI counts and raw software-WebGPU latency. These are **accepted depth
pixels in fixed masks**, not final cloud points, surface accuracy or completeness against truth.
The reference and six source views are frozen. Tower keeps native pixel pitch in a crop; desert
and canyon use 320x180 overviews. All runs use 4 photometric / 4 geometric iterations. Replaying
these seven-view subsets does not reproduce every original 1280x720 production result.

| Scene / ROI | Baseline | Boundary only | Final observability only |
| --- | ---: | ---: | ---: |
| Tower: dark metal | 605 | 545 | 605 |
| Tower: sky 3–12 px from metal | 1,758 | 1,203 | 1,752 |
| Tower: separate sky in crop | 135 | 130 | 135 |
| Desert: sky | 1,544 | 1,581 | 541 |
| Desert: central rock | 4,030 | 4,029 | 4,030 |
| Desert: left distant terrain | 1,155 | 1,163 | 1,104 |
| Desert: right distant terrain | 384 | 392 | 381 |
| Canyon: left rock | 3,805 | 3,800 | 3,805 |
| Canyon: right rock | 283 | 375 | 202 |
| Canyon: reflected pool | 378 | 387 | 364 |
| Canyon: channel | 25 | 25 | 25 |

The final observability rule reduces desert sky support by 65.0%, preserving the central rock
and 99.2% of the baseline right-distant support. Left-distant support still drops 4.4%. It does
not meaningfully remove the tower halo. Boundary support removes 31.6% of the tower's near-metal
sky support but also 9.9% of the metal support. Its larger canyon-right count does not prove
that those depths are correct. The final filter also loses 28.6% of canyon-right support.
The final combined canyon run keeps left/right/pool/channel counts 3,800/262/364/21. Final
combined tower/desert replays have not been measured; no additive real-scene gain is claimed.

The initial stricter filter also rejected support when the rotation-only comparison was poor
or invalid. It reduced desert-right support to 310 and tower metal to 566. The final predicate
only vetoes support when the directional comparison itself fits within the existing NCC bound;
otherwise the geometric support and aggregate photometric rule remain responsible for the
pixel. The rejected initial measurements remain in the JSON so the final choice is reviewable.

**Decision:** retain `boundaryAware: false` and `minFiniteGain: 0` in core and preserve lab
settings unless an explicit query override is present. The options are implemented, reproducible
and independently testable, but the tower/terrain tradeoffs do not justify universal activation.
A finite periodic or low-parallax surface can be directionally ambiguous; a bad directional fit
cannot certify a finite depth. This change is not semantic sky segmentation or a reflective-water
solution. The next acceptance gate is boundary retention plus camera-calibrated/semantic
background handling, with #223's upstream camera changes assessed separately in the four-way
integration protocol above. Neither PR's independent tests establish an end-to-end RGB gain.

## Further refinement: native thin-structure acceptance

The next bounded target is to preserve the tower metal while reducing the attached background
surface, against the same frozen inputs and ROI masks. Task 6's
[numerical diagnosis](experiments/dense-mvs/boundary-native-precision-before.json) uses identical
f32 cameras/gray in TypeShade's f64/f32 generators. A fronto-parallel hypothetical model-Z=3
plane is a cost probe, not ground-truth depth or the original GPU hypotheses. In 5,364 near-metal
sky comparisons, 59 disagree on source-cost validity; 1,132 differ by more than 0.01. The metal's
median bilateral support mass is 0.266 of its spatial window, versus 0.997 for nearby sky.

The primary real-scene acceptance targets are fixed before implementation: preserve at least
the baseline's 605 accepted metal pixels while reducing the 1,758 accepted near-metal sky
pixels, and report depth distributions plus a frozen foreground-depth-band comparison to avoid
calling a far-depth metal pixel a recovered pole. Counts remain coverage/error proxies, not
true surface accuracy. Also check desert/canyon foreground retention, two-source geometric
requirements, memory/reproducibility, source-cost precision and runtime. If a candidate does
not meet both retention and background targets, record that outcome and keep it experimental.
The masks and thresholds must not be retuned to turn an unsuccessful candidate into a pass.

### NCC precision refinement

The opt-in paths now subtract a sampled source-center anchor before accumulating source moments,
and center reference-mean accumulation on its own center. NCC is translation invariant in gray
intensity; this avoids catastrophic cancellation in `E[b²] - E[b]²` for bright, low-contrast
patches. Scalar and cached source-cost functions use the same arithmetic. Default-disabled
arithmetic is retained. No epsilon, variance threshold or geometric-support requirement changed.

The [after record](experiments/dense-mvs/boundary-native-precision-after.json) repeats Task 6's
identical f32 inputs and hypothetical plane: near-metal sky's f64/f32 p95 cost difference drops
from 0.05024 to 0.00001770, cost differences above 0.01 drop from 1,132 to zero, and 59 validity
disagreements drop to zero. Metal's p95 difference is 0.00003178. This establishes numerical
agreement for the probe, not improved ground-truth depth. The independent brightness/contrast
fixture covers 27 aligned patches per precision with f32 maximum cost 0.00001109.

Two adaptive-support prototypes were evaluated: background-weight mixing and a dense step-1
centered lattice only when the pilot assigns minority support to its center. They retain 537
and 572 metal pixels respectively, below the frozen 605-pixel gate. Neither prototype ships in
the production shader. The stable, original centered lattice retains 553 metal pixels and
1,151 near-metal sky pixels. Its background reduction does not compensate for foreground loss;
boundary support remains opt-in. Completed implementation/experimentation tasks do not imply
that this real-scene acceptance gate passed.

The [observation geometry contract](observation-geometry-contract.md) distinguishes this dense
cost check from a maintained inverse-depth state, and documents the shared explicit optical-path
mask, its coordinate/provenance requirements and the remaining SfM/Radiance work.

The [refinement replay record](experiments/dense-mvs/boundary-refinement.json) includes five
shipped-candidate measurements, two rejected prototypes and depth-distribution checks. The
foreground band is the baseline accepted metal median ±5% (model Z 2.85885–3.15978), not truth.
Of 605 baseline metal depths, 602 lie in this band; stable centered support retains 552 in-band,
and the rejected dense-support prototype retains 571. Their near-metal sky counts in the same
band are 1,070 and 1,130 versus baseline 1,661. This confirms a foreground/background tradeoff
rather than a count inflated solely by far-depth metal pixels.

With stabilized observability, desert sky is 527 (baseline 1,544), central rock 4,030, left/right
terrain 1,101/381 (baseline 1,155/384). Canyon left/right/pool/channel are 3,805/206/364/25
(baseline 3,805/283/378/25). Stabilized boundary support gives desert sky 1,620, left/right
1,167/391, central 4,029; canyon 3,800/375/387/25. No optical-path masks were supplied in these
real-scene replays. Pool counts do not establish water-surface depth or a reflection fix.
SwiftShader runs overlapped; recorded wall times must not be used for speed comparisons.

### Completed refinement validation

The [refinement validation record](experiments/dense-mvs/boundary-refinement-validation.json)
records 304 core tests, eight focused CPU shader tests, two WebGPU tests, type/format checks,
library builds and the 91-page site build. Default-disabled full depth-map and 307-point TSDF
hashes remain identical to the original baseline in window and grouped modes, with zero live
arrays. The masked synthetic scene retains 850 direct pixels identically on CPU and WebGPU;
18 allowed planar holes fill and 18 forbidden holes remain empty. This is a supplied-mask test.

An isolated local integration of this code with #223 `c1465cb` resolves only test-list conflicts.
Its 35 focused CPU tests, type checks, core/GPU builds and explicit-mask WebGPU test pass. No
merge was pushed, and no combined RGB quality improvement was measured. Hosted CI was still
running when these local checks completed. The validation commits add records only; code is
the tested `f44a3f2`.
