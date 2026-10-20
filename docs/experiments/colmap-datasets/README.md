# COLMAP demo datasets as TypeCV benchmarks

The [COLMAP demo datasets](https://demuc.de/colmap/datasets/) (south-building,
gerrard-hall, person-hall, graham-hall) ship three things: the images, the COLMAP
database with SIFT keypoints, descriptors, raw matches and verified inlier matches,
and a bundled sparse model from COLMAP's full pipeline. `scripts/colmap-datasets/colmap_dataset.py`
turns one dataset into a controlled benchmark:

1. `export` writes the TypeCV sequence JSON from the database: the exact COLMAP
   SIFT keypoints and raw matches, with the bundled model's refined intrinsics held
   fixed. COLMAP's +0.5 pixel-center convention is shifted to TypeCV's integer
   centers for keypoints and principal point. A subset (`--images`, `--step`,
   `--window`) keeps a run inside the current mapper's limits.
2. `reference` summarizes the bundled model on the same image subset, and with
   `--run-mapper` runs pycolmap's incremental mapper on a copy of the same database
   with the same fixed intrinsics. That run isolates the mapper: both systems see the
   same features and the same putative matches.
3. `scripts/reconstruct-sequence.mjs` (CPU reference path) or
   `scripts/reconstruct-sequence-gpu.mjs` (TypeShade path in Chromium) reconstructs
   the sequence.
4. `compare` similarity-aligns the TypeCV cameras to each reference on their common
   images and reports rotation and center agreement, registration, point and
   observation counts and elapsed time.

`export` accepts the SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL and RADIAL camera models, and
OPENCV when its tangential terms move no point by half a pixel at the analysis size
(gerrard-hall: p1 1.2e-4, p2 3.0e-4, 0.27 px at 640 wide); the terms are then dropped
and recorded under the camera's `droppedTangential`.

The references are not ground truth. Both systems share the same features and
matches, so a systematic matching error appears in both. Counts and reprojection
errors use different accepted observation sets and are not accuracy claims. The
bundled model used COLMAP's own intrinsics refinement over all 128 images; TypeCV and
the mapper run hold those refined intrinsics fixed, which favors both.

## Reproduce

Use Python 3.12+ and Node 22.18+. `pycolmap` 4.2.1 is pinned in
`scripts/requirements-sequence.txt`.

```sh
npm ci && npm run build
python -m venv .venv-sequence
.venv-sequence/bin/python -m pip install -r scripts/requirements-sequence.txt
D=.venv-sequence/bin/python scripts/colmap-datasets/colmap_dataset.py
$D fetch south-building data/colmap
$D export data/colmap/south-building data/sb16/sequence.json --images 16
echo '{"maxErrorPx": 4, "ransacIterations": 750, "seed": 42}' > data/sb16/options.json
$D reference data/colmap/south-building data/sb16/reference.json \
  --sequence data/sb16/sequence.json --run-mapper data/sb16/colmap --fixed-camera
node scripts/reconstruct-sequence.mjs data/sb16/sequence.json data/sb16/typecv data/sb16/options.json
$D compare data/sb16/typecv data/sb16/sequence.json data/sb16/reference.json data/sb16/comparison.json
# TypeShade path (software WebGPU by default; TYPECV_HARDWARE_GPU=1 for a physical adapter):
node scripts/reconstruct-sequence-gpu.mjs data/sb16/sequence.json data/sb16/typecv-gpu data/sb16/options.json
$D compare data/sb16/typecv-gpu data/sb16/sequence.json data/sb16/reference.json data/sb16/comparison-gpu.json
```

`maxErrorPx` 4 matches COLMAP's default filter at the source resolution (3072 × 2304).
Use `--max-edge` on export to scale coordinates and intrinsics together when a run
must stay at a lower analysis resolution.

## Recorded results: south-building (2026-10-19, this container, no physical GPU)

All runs use the same COLMAP SIFT keypoints and raw matches and the bundled refined
intrinsics held fixed. `maxErrorPx` 4, 750 RANSAC trials, seed 42. The pycolmap
mapper is the CPU wheel. TypeCV CPU runs are the synchronous reference path on one
core; the GPU run is Chromium's SwiftShader software WebGPU, so its time measures
correctness and structure, not acceleration.

| Subset    | System                           | Registered | Landmarks | Observations | Median rotation vs mapper | Median center vs mapper | Elapsed             |
| --------- | -------------------------------- | ---------- | --------- | ------------ | ------------------------- | ----------------------- | ------------------- |
| 16 images | TypeCV CPU                       | 16/16      | 6,542     | 17,746       | 0.019°                    | 0.035% of radius        | 44.7 s              |
| 16 images | pycolmap mapper                  | 16/16      | 2,494     | 9,771        | —                         | —                       | 2.5 s               |
| 32 images | TypeCV CPU                       | 32/32      | 20,266    | 57,421       | 0.058°                    | 0.056% of radius        | 216.8 s, 639 MB RSS |
| 32 images | pycolmap mapper                  | 32/32      | 9,911     | 38,368       | —                         | —                       | 6.5 s               |
| 6 images  | TypeCV GPU (SwiftShader)         | 6/6        | 3,594     | —            | see record                | see record              | 76.8 s software     |
| 16 images | TypeCV CPU, persistent landmarks | 16/16      | 6,545     | 17,568       | 0.021°                    | 0.037% of radius        | 18.4 s              |
| 32 images | TypeCV CPU, persistent landmarks | 32/32      | 20,266    | 56,839       | 0.064°                    | 0.063% of radius        | 83.8 s, 670 MB RSS  |

Records: [south-building-16.json](south-building-16.json),
[south-building-32.json](south-building-32.json),
[south-building-6-gpu-software.json](south-building-6-gpu-software.json),
[south-building-16-persistent.json](south-building-16-persistent.json),
[south-building-32-persistent.json](south-building-32-persistent.json).

The persistent-landmark rows (the mapper keeps its landmarks across registrations
and triangulates only completed tracks, with a closed-form 3×3 inverse in CPU BA)
were timed while SwiftShader tests ran on the same container, so they are upper
bounds. Their profile puts the remaining CPU time in bundle adjustment (local
passes still carry every landmark) and in eight-point pair verification.

Observed: camera agreement with COLMAP's mapper is within 0.06° on both subsets.
TypeCV's CPU mapper is 18× slower at 16 images and 33× slower at 32 images, and its
time grows about 4.9× for 2× images. The landmark counts differ because the two
mappers accept different tracks (TypeCV's triangulation angle threshold is 0.5°, COLMAP's
default is 1.5°); they are not a quality ranking. The profile of the 16-image run puts
the time in repeated track triangulation (`nullspace`), bundle adjustment and garbage
collection: the mapper re-triangulates every track after each registration. That is
the first mapper change to make.
