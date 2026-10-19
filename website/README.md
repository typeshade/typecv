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
without sampling or a total-frame ceiling. The source video follows each frame's
timestamp. The analysis edge is 640 pixels. A default 32-camera local optimization
window bounds numerical work while all registered cameras and point observations
are archived. Processing stages show real detections, match candidates and
camera-depth colors. The 3D cloud uses source-image RGB; point selection, a photo
plane, camera movement/rotation and timeline replay connect the image to geometry.
The included
Pexels pyramid sample uses attributed, resized frames; uploaded videos stay in
the browser. Camera intrinsics are assumed, and results depend on scene texture
and parallax. Prepared JSON and the eight-frame sample use the same sequential
session. Final geometry refinement runs on the latest local window. Unsupported
codecs require conversion or prepared JSON. This does not
establish camera-rate SLAM or long-sequence robustness.

Deployment reads the existing organization Actions secret
`CLOUDFLARE_API_TOKEN`. The credential is used only by the GitHub deployment job and
never included in website assets. The site does not require R2 credentials or an R2
bucket. One Worker (`worker.mjs`) runs for `/api/*` and stores the lab's run records
in the `RECORDS` KV namespace; `docs/lab-run-records.md` describes the record, the
routes and the one-time setup (`scripts/setup-run-records.ps1`). The deploy token
needs the Workers KV Storage edit permission for that binding.
