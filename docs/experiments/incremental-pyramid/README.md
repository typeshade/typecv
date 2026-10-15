# Controlled-input incremental pyramid experiment

## Input and provenance

Use [Blanka OG's free Pexels pyramid aerial clip](https://www.pexels.com/video/drone-footage-of-a-pyramid-8134952/) under the
[Pexels License](https://www.pexels.com/license/), checked 2026-10-15. It permits free
use/modification with its restrictions; the footage is not public domain. Raw
media, decoded frames, keypoints and generated clouds remain local.

- Original file: 3840 × 2160, 24000/1001 fps, 23.606917 seconds.
- Source SHA-256: `c79321c99c8ba1898e858a8270292a51868f8c3ebef7c489b763e1ff111e5200`.
- Prepared JSON SHA-256: `c2d9d873413bae1834e28df77801dc808a1a84e3eacbece275eaf63bba91e92d`.
- Eight frames requested at 0, 1, 2, 4, 8, 12, 16, 20 seconds; resized to 960 × 540.
- SIFT (up to 4,000 features), mutual 0.75 ratio matching, 8 × 6 spatial bins,
  at most 12 matches/bin and 400/pair. All 28 image pairs are prepared.
- Fixed **assumed** camera: fx=fy=1152, cx=480, cy=270, zero distortion.
  These are not calibrated parameters. No ground-truth camera poses or geometry.

OpenCV supplies SIFT/matching only. TypeCV performs verification, tracks,
initialization, P3P, triangulation and BA. The independent baseline imports the
EXACT SAME feature identities and raw matches into pycolmap 4.2.1, then uses its
own geometry verification and mapper. It does not import TypeCV poses or points.
For COLMAP's half-pixel convention, both pixels and principal point receive
+0.5, preserving the camera rays.

Both use fixed intrinsics and aligned 3px reprojection, 1-degree initialization,
0.5-degree triangulation and 12-inlier registration thresholds where options
permit. Other algorithm/settings differ and are recorded in the snapshots.
This is **not COLMAP's default full feature-to-model pipeline**.

## Recorded result

| Metric                                                   | TypeCV  | COLMAP reference |
| -------------------------------------------------------- | ------- | ---------------- |
| Registered images                                        | 8/8     | 8/8              |
| Sparse landmarks                                         | 1,729   | 1,848            |
| Accepted observations                                    | 5,153   | 5,592            |
| Mean Euclidean pixel residual over accepted observations | 0.46278 | 0.49233          |

The residuals are calculated with the same definition, but on DIFFERENT accepted
observation sets. More points or a smaller residual does not prove higher
accuracy. COLMAP's native mean error (0.41524) uses its own aggregation and must
not be compared directly against TypeCV's per-observation Euclidean mean.

With a common similarity alignment based on all camera rotations and centers,
the median camera orientation difference is 0.06306 degrees.
Median center difference is 0.274% of the reference
trajectory's maximum radius about its centroid. These are **reference agreement**,
not ground-truth errors. Both systems can share systematic calibration/matching
errors. The initial normalized-DLT PnP version only registered 5/8 images; the
P3P implementation was added after reproducing this failure, and final scripts
now register 8/8 without manually overriding the initial pair.

[Comparison](comparison.json), [full TypeCV diagnostics](typecv-summary.json),
and [COLMAP options/poses/verification](colmap-summary.json) preserve the run.
Elapsed times are single local observations, not controlled performance claims.
Native COLMAP is substantially faster in this small experiment. Runtime scaling,
other real scenes and calibrated ground-truth data still need evaluation.

## Reproduce

Use Python 3.12 and Node 24 for the recorded environment (core supports Node
22.18+). Optional Python packages are pinned; the core has no runtime dependencies.

```sh
npm ci
npm run build
python -m venv .venv-sequence
.venv-sequence/bin/python -m pip install -r scripts/requirements-sequence.txt
mkdir -p data/pyramid
curl -fL 'https://videos.pexels.com/video-files/8134952/8134952-uhd_3840_2160_24fps.mp4' -o data/pyramid/source.mp4
.venv-sequence/bin/python scripts/prepare-sequence.py data/pyramid/source.mp4 data/pyramid/sequence
node scripts/reconstruct-sequence.mjs data/pyramid/sequence/sequence.json data/pyramid/typecv
.venv-sequence/bin/python scripts/compare-colmap.py data/pyramid/sequence/sequence.json data/pyramid/colmap
.venv-sequence/bin/python scripts/summarize-comparison.py data/pyramid/typecv data/pyramid/colmap data/pyramid/comparison.json
```

Use a fresh COLMAP output directory; the script refuses to overwrite an existing
database. Check both file hashes above before comparing. Source URLs may change;
use the original page if needed and verify the checksum. Decoder/library changes
may change keypoints. No external footage or benchmarks are downloaded by CI.
To supply known calibration, pass `--camera camera.json` to preparation; camera
parameters must apply to the resized resolution. An optional third JSON path to
the Node reconstruction script overrides mapper options and must be kept with
any reported result.
