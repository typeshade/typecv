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
sampling tests also pass. Full real-scene and WebGPU validation are tracked separately in the PR.


## Explicit lab comparisons

The lab leaves both experiments off by default. After this PR is deployed, add
`?dense=1&denseFiniteGain=0.02` to test observability alone, `?dense=1&denseBoundary=1`
for boundary support alone, or both parameters for the combination. The same overrides reach
single-video and multi-video Dense runs. Missing parameters preserve legacy options; explicit
zero disables an experiment. Invalid values are ignored. Exported run records preserve the
actual options, including explicit false/zero, so a later report identifies what ran.

These are experimental comparison parameters, not a claim that every scene benefits. The first
native tower comparison reduces background depth but also loses foreground line pixels; the
three-scene acceptance task remains open until those tradeoffs and the revised filter finish.
