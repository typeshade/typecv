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
CPU fallback. The video lab decodes local video in the browser and previews
eight sampled frames, then runs GPU feature extraction, BRIEF matching,
incremental SfM and a Three.js sparse reconstruction viewer. The included
Pexels pyramid sample uses attributed, resized frames; uploaded videos stay in
the browser. Camera intrinsics are assumed, and results depend on scene texture
and parallax.

Deployment reads the existing organization Actions secret
`CLOUDFLARE_API_TOKEN`; no additional secret registration is needed. The
credential is used only by the GitHub deployment job and never included in
website assets. The site does not require R2 credentials or an R2 bucket.
