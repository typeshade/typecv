# Create the Cloudflare resources the lab's run-record collector needs. Run once, on the
# owner's machine, after `npm ci` (wrangler is a devDependency) and `npx wrangler login`.
#
#   pwsh scripts/setup-run-records.ps1
#
# Steps:
#   1. Create the KV namespace that stores the records.
#   2. Print the id to paste into wrangler.jsonc ("kv_namespaces" -> "id").
#   3. Store the read token as a Worker secret. Readers send it as "Authorization: Bearer <token>".
# The GitHub Actions deploy token (CLOUDFLARE_API_TOKEN) must also carry
# "Workers KV Storage: Edit" for `wrangler deploy` to bind the namespace.
$ErrorActionPreference = 'Stop'

Write-Host 'Creating KV namespace typecv-records (no-op if it exists)...'
$created = npx wrangler kv namespace create typecv-records 2>&1 | Out-String
Write-Host $created
$id = [regex]::Match($created, '"id":\s*"([0-9a-f]{32})"').Groups[1].Value
if (-not $id) {
  $listed = npx wrangler kv namespace list | Out-String
  $id = [regex]::Match($listed, '"id":\s*"([0-9a-f]{32})"[^}]*"title":\s*"typecv-site-typecv-records"').Groups[1].Value
}
if (-not $id) { throw 'Could not read the KV namespace id; run `npx wrangler kv namespace list` and copy it by hand.' }
Write-Host ''
Write-Host "KV namespace id: $id"
Write-Host 'Replace REPLACE_WITH_RECORDS_KV_ID in wrangler.jsonc with this id and commit it.'
Write-Host ''
Write-Host 'Now set the read token (any long random string; keep it out of the repository):'
npx wrangler secret put RECORDS_READ_TOKEN
Write-Host ''
Write-Host 'Read the collected records later with:'
Write-Host '  curl -H "Authorization: Bearer <token>" https://cv.typeshade.dev/api/records/?limit=50'
Write-Host '  curl -H "Authorization: Bearer <token>" https://cv.typeshade.dev/api/records/<id>/'
