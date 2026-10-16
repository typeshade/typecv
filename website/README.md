# TypeCV public website

The source for **https://cv.typeshade.dev/**, served publicly by the
`typecv-site` Cloudflare Worker using Workers Static Assets, matching TypeNN's
deployment model. There is no visitor authentication, server API, upload endpoint
or runtime credential. The site uses the IBM Plex fonts under their bundled license.

```sh
npm ci
npm run build
npm run build:website
npm run test:website
# Preview with the same routing/headers as production:
npx wrangler dev --port 8787
# Use a process-local CLOUDFLARE_API_TOKEN with Workers and zone permissions:
npm run deploy:website
```

The website build consumes the root library build. It renders the repository's
Markdown documents, copies the recorded JSON experiments and compiled TypeShade
runtime, and bundles the independent CPU reference. Documents stay in `docs/`;
editing a document updates the website on the next build. Local Markdown links
and experiment links are rewritten to public routes. Unknown paths receive a real 404. Build output in `website/dist/` and Wrangler local state are ignored.

The English overview and playground are at `/` and `/playground/`; separately
written Korean equivalents are at `/ko/` and `/ko/playground/`. Detailed project
documents are currently English. `version.json` records the source commit and
generated routes. The runtime license files are served under `/runtime/`.

The playground synthesizes calibrated correspondences and calls the actual
`verifyImagePairGpu` operator or the explicitly selected independent
`verifyImagePair` CPU reference. It accepts fixed numeric controls, never source
code. It does not silently retry on CPU when WebGPU is unavailable or fails.
Controls show real consensus/quality results and device dispatch/readback counts;
elapsed time is not advertised as a physical-GPU benchmark.

`wrangler.jsonc` binds only `cv.typeshade.dev`, disables `workers.dev` and preview
URLs, and uses the existing TypeShade account. The account ID is public deployment
metadata. **Never add API tokens, R2 access keys or `.env` files to the repository.**
This site does not require the supplied R2 credentials or an R2 bucket.

`.github/workflows/deploy.yml` builds and deploys `main`, with a repository or
organization Actions secret named `CLOUDFLARE_API_TOKEN`. Set that secret outside
the repository; it is used only by the deployment job and is never sent to visitors.
Without that secret the workflow builds the site and reports a warning without
deploying. The initial publication can be deployed directly with a process-local
Cloudflare token. The current GitHub integration cannot register Actions secrets;
automatic redeployment requires an administrator to add this secret through GitHub.
