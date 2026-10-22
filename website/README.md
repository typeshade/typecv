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

The pair playground calls the compiled TypeCV CPU reference or TypeShade
WebGPU implementation after an explicit backend selection. There is no silent
CPU fallback. The video lab decodes every actual frame of an uploaded video in
presentation order using Mediabunny and browser codecs, waiting for GPU analysis
without a total-frame ceiling. A frame becomes a keyframe when its image moved
at least 4 % of the frame since the last keyframe (phase correlation of a small
grayscale thumbnail, before feature extraction) or one second has passed; other
frames are counted as skipped. `?gate=legacy` restores the older rule (the median
match displacement reaches 8 analysis pixels). The source video follows each frame's
timestamp. The analysis edge is 640 pixels, or 1280 by the selector, which decodes uploaded
video at that edge and raises the feature cap with the pixel count (4096 at 640, 16384 at
1280); a prepared JSON keeps the edge it was prepared at. A default 32-camera local optimization
window bounds numerical work while all registered cameras and point observations
are archived. Processing stages show real detections, match candidates and
camera-depth colors. The 3D cloud uses source-image RGB; point selection, a photo
plane, camera movement/rotation and synchronized video playback connect the image to geometry.
The lab puts input selection, settings and the run action above the video and 3D panes. Video
inputs use native playback/seeking and arrow keys for analysed-frame stepping; prepared frames
retain their frame navigator. Detailed statistics and record sharing can be expanded below.
The sample catalog
(`public/sample/index.json`, one entry per directory under `public/sample/`) offers
the attributed, resized Pexels pyramid frames, calibrated photo samples from the
COLMAP demo datasets (south-building, 12 and 24 photographs; gerrard-hall, person-hall and
graham-hall, 12 each) with the
datasets' SIFT features as an option, and 16 frames of a TUM RGB-D sequence with its
motion-capture poses as the reference; `scripts/prepare-lab-colmap-sample.py` and
`scripts/prepare-lab-tum-sample.py` derive them; the COLMAP samples also ship a
1280-pixel preview set under `1280/`, which the analysis resolution selector loads.
The eight-frame pyramid set stays in the repository as the standard benchmark's input. The
other drone videos (`samples/remote-videos.json`) live in the R2 bucket `typecv-samples`:
two frames per second over each whole clip, 24 to 48 frames (`src/lib/sample-frames.ts`).
The "Publish lab video samples" workflow (`.github/workflows/lab-samples.yml`) cuts and
uploads them with `scripts/lab-remote-samples.mjs publish`, and the Worker serves them under
`/sample/`. A card whose frames are not in the bucket yet reads "Not published yet".
The same workflow stores each clip's source video unchanged as `/sample/<id>/source.mp4`
(`publish --videos-only` stores only the videos, `--frames-only` only the frames) and records
its size, SHA-256 and `videoId` in the manifest. The Worker answers `Range` and `HEAD` requests
on `/sample/`, and the lab opens the source video with mediabunny `UrlSource`, so a run fetches
only the byte ranges it needs and decodes at the selected analysis edge. A browser that cannot
decode the codec uses the 640 pixel frames (`options.videoSource` of the run record).
Uploaded videos stay in the browser. Camera intrinsics are assumed, and results depend on scene texture
and parallax. Prepared JSON and the eight-frame sample use the same sequential
session. Final geometry refinement runs on the latest local window. Unsupported
codecs require conversion or prepared JSON. This does not
establish camera-rate SLAM or long-sequence robustness.

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
