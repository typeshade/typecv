# Website source migration

The website and browser applications moved to `typeshade/web`. That repository owns the Astro
site, the TypeCV lab and playground, website tests, record and sample operations, previews, and
Cloudflare deployment. This repository now contains the TypeCV library, its documentation and
examples, library tests, build, and package release configuration.

The removal is based on TypeCV revision `db7216a4d0d763996b677bbf27deb0f3913c930c` and
reconciled with library changes through `715fd1b694508014315e1a505aa2c970051a7f3a`. The application
source and site tests were migrated into `web/apps/typecv`; the source checkout no longer needs the
Astro plugin to format library files. Record and sample workflows, Wrangler configuration, and
site-only deployment scripts moved with the application. The library CI retains all three
`test:webgpu` shards and removes the two website shards.

The lab's underlying computer vision implementation remains in `src/`. Library tests, GPU tests,
examples, documentation, builds, and npm package settings remain here. The global-descriptor
research tools also remain library-owned: their catalog and video-manifest inputs are frozen as
fixtures from `db7216a4d0d763996b677bbf27deb0f3913c930c`, and the serialized 248-frame list is
identical to the pre-migration result. The app repository reads pinned library sources; it does not
become a second editable library implementation.

This change prepares repository ownership only. It does not disable or replace production Workers,
move records or sample objects, change DNS, or establish a production cutover. See the shared
migration handoff in `typeshade/web/docs/repository-migration.md` for operational sequencing and
remaining configuration checks.

## Validation

- At the reconciled `715fd1b694508014315e1a505aa2c970051a7f3a` base, `npm run format:check`,
  `npm run typecheck`, `npm test` (477/477), `npm run build`, and `npm run test:surface` (6/6)
  passed. `npm pack --dry-run` passed and the tarball contains no `website/` paths. The corresponding
  app record and hardware-check tests passed 24/24 in Chromium on SwiftShader.
- The shard assignment check passed and all three library WebGPU shards remain in CI. The updated digest declaration guard from the merged base is preserved.
- The global-descriptor metadata move preserves all 248 serialized frames exactly. Retained
  dependency versions are unchanged; the lockfile removes 91 site-only entries.
- `test:shade` was still executing its existing CPU-intensive MVS/TSDF cases when the shared
  execution server lost the session. Its final output and exit status could not be recovered;
  it is not recorded as a pass. The full required suite remains enabled in CI.

The original dirty checkout was preserved. This cleanup was prepared in an isolated worktree;
the replacement application and its tests are reviewed in `typeshade/web` before cutover.
