# Lab run records

The SfM lab at `/lab/` can send the measurement of one run to the site's collector,
so that runs on hardware the maintainers do not own can be compared. Sending is a
choice the visitor makes for each record. Nothing is sent without the checkbox.

## What a record holds

`website/src/lib/run-record.ts` defines the record and its validator. One record is
one completed run:

| Field       | Content                                                                                      |
| ----------- | -------------------------------------------------------------------------------------------- |
| `site`      | The site commit (`/version.json`) and the runtime path                                       |
| `client`    | Browser user agent, platform, language, core count, device memory class, screen size and DPR |
| `gpu`       | WebGPU adapter vendor, architecture, device, description, fallback flag, two buffer limits   |
| `input`     | Kind (sample, prepared JSON, video), frame count, analysis and source resolution, duration   |
| `options`   | The lab's numerical controls and the assumed focal ratio                                     |
| `result`    | Registered views, landmarks, observations, mean reprojection error, matched pairs, features  |
| `timing`    | Total wall time, device reuse (cold or warm), per-operation totals, per-frame times          |
| `transfers` | Application upload and readback bytes                                                        |
| `benchmark` | `pyramid-8` and `cold` or `warm` when the standard benchmark produced the run, else null     |
| `reference` | Agreement with a sample's reference cameras (below), else null                               |
| `note`      | Up to 500 characters the visitor typed                                                       |

A record holds no video, image, file name or network address. The page shows the
exact JSON before it is sent, and offers the same JSON for download or copy when the
visitor prefers to share it another way. The collector adds `server.id`,
`server.receivedAt`, the Cloudflare country code and colo. The sender's address is
hashed with the hour for rate limiting only and is not stored.

## The standard benchmark

The lab's **Standard benchmark** button loads the eight-frame pyramid sample and runs
it twice in one worker: a cold pass that includes shader compilation, then a warm pass
on the retained device. Each pass produces one record with `benchmark.pass` set. With
the consent box checked before the benchmark starts, both records are sent as they
complete.

## The COLMAP sample

The lab's two **COLMAP south-building** buttons load twelve photographs of the
[COLMAP demo dataset](https://demuc.de/colmap/datasets/) south-building (image credit
Christopher Zach), resized to 640 × 480, with the dataset's refined intrinsics scaled to
that size and the poses of its bundled sparse model
(`website/public/sample/south-building/`, derived by
`scripts/prepare-lab-colmap-sample.py`). The first button runs the whole GPU pipeline
on the photographs with the `photo` feature profile (`PHOTO_PROFILE` in
`sfm.worker.ts`). The second feeds the dataset's own SIFT keypoints and raw matches
(`colmap-sift.bin`, the matched keypoints only) to the GPU mapper, so the mapper is
measured apart from the lab's feature front end.

After each snapshot and at the end, the page similarity-aligns the registered cameras
to the COLMAP poses on their common views (`website/src/lib/pose-alignment.ts`, the
method of `colmap_dataset.py compare`) and shows the common view count, the median and
maximum rotation difference in degrees, and the median and maximum camera center
difference as a fraction of the reference trajectory's radius. The record's `reference`
field carries the same numbers. The COLMAP model is a reference, not ground truth: both
systems hold the same intrinsics fixed, and with the second button they also share the
same features and matches.

## Collector

`website/worker.mjs` runs in front of the static assets for `/api/*` only
(`wrangler.jsonc`: `run_worker_first`). Routes:

| Route                   | Behavior                                                                                   |
| ----------------------- | ------------------------------------------------------------------------------------------ |
| `POST /api/records`     | Same-origin only; body at most 256 KiB; validated; 30 per address per hour; stored in KV   |
| `GET /api/records`      | `Authorization: Bearer <RECORDS_READ_TOKEN>`; lists `limit` (at most 100) records with metadata and a `cursor` |
| `GET /api/records/<id>` | Same token; returns one stored record                                                      |

Records live in the KV namespace bound as `RECORDS` under `record/<id>`, with
listing metadata (received time, GPU, input kind, frames, registered views, total ms,
device reuse, benchmark name). `website/worker.test.mjs` tests the handler with an
in-memory KV.

## Setup (owner, once)

1. Run `pwsh scripts/setup-run-records.ps1` after `npx wrangler login`. It creates the
   `typecv-records` namespace, prints its id and stores the `RECORDS_READ_TOKEN` secret.
2. Put the id into `wrangler.jsonc` (`kv_namespaces[0].id`) and commit it.
3. Give the deploy token (`CLOUDFLARE_API_TOKEN`) the **Workers KV Storage: Edit**
   permission, or `wrangler deploy` cannot bind the namespace.

Until step 2 is done, `wrangler deploy` fails on the placeholder id, and the page
reports "collector not configured" when a visitor presses Send.

## Reading the records

```sh
curl -H "Authorization: Bearer $TOKEN" 'https://cv.typeshade.dev/api/records?limit=50'
curl -H "Authorization: Bearer $TOKEN" 'https://cv.typeshade.dev/api/records/<id>'
```

Or on the Cloudflare dashboard: Workers & Pages → KV → `typecv-site-typecv-records`.
