# TypeCV library repository

This repository owns the TypeCV computer vision library: its implementation in `src/`, library
examples, documentation, tests, build, and package release process. Keep browser applications,
website source, website-specific dependencies, and deployment configuration in `typeshade/web`.

Run `npm ci` to install dependencies and `npm run check` for the library checks. GPU tests run in
three shards in `.github/workflows/ci.yml`; keep `scripts/ci-shard.mjs` and its shard coverage
check aligned with that workflow.
