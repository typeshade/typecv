# TypeCV public website

The public site at <https://cv.typeshade.dev/> uses the TypeNN site template:
Astro static pages, React interactive islands, Ant Design 5 components, Tailwind
CSS 4, and the TypeNN IBM Plex design system. Reference snapshots of the source
template are in `references/typenn/`; TypeCV product claims are maintained in
the root documentation.

```sh
npm ci
npm --prefix website ci
npm run build
npm run build:website
npm run test:website
npm --prefix website run dev
```

The Astro build generates bilingual overview, pair-verification playground,
SfM lab, and documentation routes. It renders Markdown in `docs/`, preserves
heading anchors, copies the recorded experiment data and compiled browser
runtime, and writes `version.json`, `robots.txt`, and `sitemap.xml`. The gallery
and app are served as public Cloudflare Workers Static Assets at
`cv.typeshade.dev`; unknown paths return a real 404.

The lab's run settings include a **Dense MVS combination** selector for TSDF or pixel
fusion, each with cleanup on or off. The recommended default is TSDF with cleanup off
to preserve more surface coverage. Cleanup can remove real surfaces as well as noise.
The same selection applies to single-video and multi-video runs. Existing
`denseFusion` and `denseClean` URL parameters initialize the selector; changes made in
the page take precedence for subsequent runs.

The TSDF combinations automatically use confidence-weighted visibility with one
vote per spatial camera group, and local planar depth completion verified by two
other camera groups. The summary reports completed depth pixels. Run records
preserve the selected visibility method, whether planar completion was enabled,
and the completed count; multi-video records sum it across components. These
changes have synthetic CPU/WebGPU regression coverage; gains on a particular
video still need a same-input comparison. See `docs/dense-mvs.md` section 10.11.

The pair playground calls the compiled TypeCV CPU reference or TypeShade
WebGPU implementation after an explicit backend selection. There is no silent
CPU fallback. The video lab decodes every actual frame of an uploaded video in
presentation order using Mediabunny and browser codecs, waiting for GPU analysis
without a total-frame ceiling. A frame becomes a keyframe when its image moved
at least 4 % of the frame since the last keyframe (phase correlation of a small
grayscale thumbnail, before feature extraction) or one second has passed; other
frames are counted as skipped. `?gate=legacy` restores the older rule (the median
match displacement reaches 8 analysis pixels). The source video follows each frame's
timestamp. The analysis edge defaults to 1280 pixels, with 640 available in the selector. Uploaded and
sample videos are decoded at that edge without upscaling smaller sources; a prepared JSON keeps the edge it was prepared at. A default 32-camera local optimization
window bounds numerical work while all registered cameras and point observations
are archived. Processing stages show real detections, match candidates and
camera-depth colors. The 3D cloud uses source-image RGB; point selection, a photo
plane, camera movement/rotation and synchronized video playback connect the image to geometry.
The lab puts input selection, settings and the run action above the video and 3D panes.
The Run icon becomes Stop during a reconstruction and returns to Run when it finishes, fails or
is stopped. Dense MVS is enabled by default and can be disabled in settings. Stopping during
Dense MVS keeps the sparse reconstruction. Video inputs use native playback/seeking and arrow
keys for analysed-frame stepping; prepared frames
retain their frame navigator. Detailed statistics and record sharing can be expanded below.
Following the selected camera eases the viewing position, orientation and field of view over
300 ms. Further selections continue from the displayed position; manual navigation or disabling
follow stops the move. Reduced-motion preferences use immediate moves. Reconstructed poses and
exports retain their original values.
The sample catalog (`public/sample/index.json`) offers original drone videos and video sets
from `samples/remote-videos.json`, plus saved results. Each source lives unchanged in the R2
bucket `typecv-samples` at `/sample/<id>/source.mp4`. The publisher uploads one display-only
`thumbnail.jpg` per video. A run opens the source with Mediabunny `UrlSource`, fetches byte ranges
and samples at 4 fps (`?fps=` overrides it) before keyframe selection. An unavailable source or
unsupported codec reports an error. Pre-extracted JPEG inputs and photograph examples are
excluded from the public site, and historical `frame-NN.jpg` routes return 404. Numerical image
fixtures live under `tests/fixtures/lab-images`.

The standard benchmark reads the original pyramid at eight fixed timestamps and 640 px before
its cold and warm SfM passes. Compare its site commit as well as its name when reading older
JPEG-based records. The original-video input differs from those earlier records.
Uploaded videos stay in the browser. Intrinsics are estimated and results depend on scene
texture and parallax. Prepared JSON remains available for controlled user inputs. Final
geometry refinement runs on the latest local window.

Deployment reads the existing organization Actions secret
`CLOUDFLARE_API_TOKEN`. The credential is used only by the GitHub deployment job and
never included in website assets. One Worker (`worker.mjs`) runs for `/api/*` and
`/sample/*`: it stores the lab's run records in the `RECORDS` KV namespace and serves the
remote samples from the `SAMPLES` R2 binding, falling back to the static assets;
`scripts/ensure-samples-bucket.mjs` creates the bucket at deploy time and drops the binding
when the token has no "Workers R2 Storage: Edit" permission; `docs/lab-run-records.md` describes the record, the
routes and the one-time setup (`scripts/setup-run-records.ps1`). The deploy token
needs the Workers KV Storage edit permission for that binding. The result files that a visitor
chooses to send with a record go to the private R2 bucket `typecv-records` (binding
`RECORD_FILES`, `scripts/ensure-record-files-bucket.mjs`); see "Result files" in
`docs/lab-run-records.md`.

Video sample browser regressions require `ffmpeg` to encode private image fixtures as source
videos. Install it before running `npm run test:website` from the repository root. These
regression images live under `tests/fixtures/lab-images` and are excluded from the public build.
