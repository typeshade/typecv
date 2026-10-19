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

## Recorded result: south-building, first 16 images

See [south-building-16.json](south-building-16.json) for the full record.
