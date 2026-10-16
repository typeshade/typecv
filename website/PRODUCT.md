# TypeCV website product contract

The website presents TypeCV as a general computer vision library implemented in TypeScript and built on TypeShade. SfM is one area within the broader CV scope. Current public browser functionality consists of compiled CPU/WebGPU two-view geometric verification and local video SfM with TypeShade GPU feature extraction, matching, incremental reconstruction, and Three.js visualization.

The synthetic playground constructs a known two-view scene and calls the library implementation selected by the user. It reports the actual inlier, pose-quality, and backend execution results. It does not silently fall back from WebGPU to CPU.

The video lab accepts a local file and decodes eight frames in the browser, then feeds those frames to GPU feature and geometric operators. An attributed Pexels pyramid frame sequence allows a reproducible real-footage run. Intrinsics remain assumed; the sample has no ground-truth camera poses or metric scale. Each run reports registration, landmark and reprojection results instead of promising success for all footage.

The website uses the Astro, React, Ant Design and Tailwind template described in `references/typenn/PRODUCT.md`. Deployment is public through Cloudflare Workers Static Assets on `cv.typeshade.dev`. The existing organization Actions secret `CLOUDFLARE_API_TOKEN` supplies deployment authorization.
