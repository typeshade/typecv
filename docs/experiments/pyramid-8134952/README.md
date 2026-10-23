# Pyramid aerial video: exploratory two-view experiment

The recorded results use the two-view CPU core at commit
`01789fbf9921fc3fa110e3bd77bab5e8140605c0`. Later numerical improvements can
change results when rerunning the scripts. This historical experiment is part
of TypeCV's [SfM subset](../../sfm-quickstart.md); see the
[incremental comparison](../incremental-pyramid/README.md) for subsequent work.

## Source and scope

- [Drone Footage of a Pyramid](https://www.pexels.com/video/drone-footage-of-a-pyramid-8134952/), by **Blanka OG**.
- [Pexels License](https://www.pexels.com/license/), checked 2026-10-15: free download/use and modification are allowed, subject to its restrictions. This is not public-domain footage.
- Downloaded file: 3840 × 2160, 24000/1001 fps, duration 23.606917 s, 68,501,042 bytes.
- SHA-256: `c79321c99c8ba1898e858a8270292a51868f8c3ebef7c489b763e1ff111e5200`.
- The original video remains local. The public TypeCV site includes eight resized 640 × 360 JPEG preview frames, sampled at 0, 1, 2, 4, 8, 12, 16 and 20 seconds for an in-browser SfM demonstration. The transformed frames are attributed under the Pexels license in `tests/fixtures/lab-images/pyramid/README.txt`.

This is an uncalibrated real-video feasibility experiment, **not an accuracy
benchmark**. No ground-truth camera poses, dimensions or calibrated intrinsics
are available. It does not reconstruct a dense or multi-view pyramid model.
The clip was selected as a free pyramid aerial source, not on reconstruction scores.

## Method

Resize decoded frames to 960 × 540. Compare reference 0s to requested timestamps
1, 2, 4, 8, 12, 16 and 20s (decoder seeks to available video frames). Extract up
to 4,000 SIFT features per frame, use bidirectional Lowe ratio matches (0.75),
then spatially cap reference-image bins (8 × 6, at most 12 per bin, 400 total).
OpenCV provides feature extraction/matching and a homography diagnostic only;
**TypeCV's CPU solver performs essential estimation, pose and triangulation**.
No OpenCV essential/PnP result or homography mask prefilters TypeCV's inputs.

Run all seven pairs at focal lengths 0.8, 1.2 and 1.6 times resized width, with
square pixels, centered principal point, fixed intrinsics and zero distortion.
These are sensitivity assumptions, not estimated/calibrated camera parameters.
Use 1,000 RANSAC trials maximum, seed 42, threshold 1.5/focal, 10 refinement
iterations and default quality controls. All 21 configurations are retained in
[results.json](results.json), including their poses, errors and timing.

## Observations

For the **middle focal assumption only** (f = 1152 px):

| Requested pair | Inputs | Accepted 3D points | Mean error (px) | Median angle (deg) | Homography fraction |
| -------------- | ------ | ------------------ | --------------- | ------------------ | ------------------- |
| 0-1s           | 352    | 298                | 0.412           | 0.300              | 90.6%               |
| 0-2s           | 328    | 321                | 0.153           | 1.479              | 86.0%               |
| 0-4s           | 303    | 295                | 0.236           | 3.423              | 82.5%               |
| 0-8s           | 194    | 164                | 0.308           | 4.605              | 72.2%               |
| 0-12s          | 141    | 107                | 0.296           | 20.286             | 53.9%               |
| 0-16s          | 116    | 68                 | 0.332           | 3.394              | 50.9%               |
| 0-20s          | 111    | 50                 | 0.370           | 2.426              | 53.2%               |

All 21 configurations produced numerical reconstructions, which does not imply
correct geometry. In the 0–8s pair, 162–166 points pass filters across the three
focal assumptions, with mean errors of 0.300–0.322 px. However, estimated
parallax and poses change with focal assumptions; longer temporal gaps also
show substantial sensitivity. Small pixel residuals can coexist with incorrect
intrinsics, ambiguous matching or geometry.

The 0–1s pair fits a homography at roughly 91% of selected matches (2px RANSAC
threshold); short gaps also have low parallax. The homography fraction is an
ambiguity diagnostic, not proof of a planar scene or a replacement for scored
homography-versus-essential model selection. Individual pairs each have their
own unit baseline and camera-1 frame; their PLY files must not be concatenated
as a common-scale multi-view reconstruction.

This experiment supports prioritizing camera calibration/focal estimation,
explicit homography-versus-essential selection and calibrated-data evaluation.
A ground-truth dataset or independent reconstruction comparison is needed before
claiming geometric accuracy. This run did not use TypeShade GPU kernels.

## Reproduce

From the repository root, with Python 3.11+ and Node 22.18+:

```sh
npm ci
npm run build
python -m venv .venv-video
.venv-video/bin/python -m pip install -r scripts/requirements-video.txt
mkdir -p data/pyramid
curl -fL 'https://videos.pexels.com/video-files/8134952/8134952-uhd_3840_2160_24fps.mp4' -o data/pyramid/source.mp4
.venv-video/bin/python scripts/prepare-video.py data/pyramid/source.mp4 data/pyramid/prepared
node scripts/evaluate-video.mjs data/pyramid/prepared/matches.json data/pyramid/results
```

Verify the source checksum in `prepared/matches.json` against the value above.
Pexels may change a media URL; if it does, download this clip via its source page
and keep the same checksum for comparable results. Output includes extracted
frames, all input matches, per-configuration PLY clouds, accepted matches and
results JSON. Nothing is downloaded by the scripts or CI automatically. Runtime
figures are local observations, not controlled performance benchmarks. Feature
matches can vary with decoder/OpenCV versions; this run used OpenCV 5.0.0.93,
NumPy 2.3.5 and Node 24.19.0 on Linux.
