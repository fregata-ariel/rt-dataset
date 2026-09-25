# Web Viewer

The shared web viewer for RF-camera bundles (epic #26). This file covers the content-addressed
store, the deriver registry and cache, bundle kind detection and validation, and the HTTP backend
(FastAPI). The bundle format itself is defined in `docs/viewer_bundle.md`.

## Store format

Layout under `VIEWER_DATA`:

```text
<VIEWER_DATA>/
  index.sqlite          # SQLite index in WAL mode (+ `index.sqlite-wal`, `index.sqlite-shm`)
  store.lock            # exclusive `flock` around commit, delete and migrations
  staging/<uuid>/       # uploads, extraction, directory copies
  staging/deleting-<uuid>/  # transient target during deletes
  bundles/<digest>/raw/     # the bundle root, immutable: files `0444`, directories `0555`
  bundles/<digest>/derived/<member>/<deriver>/v<version>/<params>/<links>/  # derived outputs
```

Digest: for every regular file below the bundle root, one line
`"{relpath}\t{size}\t{sha256}\n"` where `relpath` is the POSIX path relative to the root,
`size` is the byte size and `sha256` is the lowercase hex of the content. Lines are sorted by the
UTF-8 bytes of the whole `relpath`; the digest is the hex `sha256` of the joined lines. Only regular
files count; empty directories, modes and mtimes do not affect the digest. The same content is stored
once under `bundles/<digest>/`. The bundle root is chosen by the rule in `docs/viewer_bundle.md` (a
single top-level directory is descended into once), so a wrapping top-level directory does not change
the digest.

Commit, delete and locking: `store.lock` is held with an exclusive `flock` around commit, delete and
migrations (a fresh fd per acquisition, so threads exclude each other). Commit renames staging to
`raw/` (chmodded read-only afterwards) then writes the index rows. Delete removes the index rows first,
then renames the bundle directory under `staging/` and removes it. A directory under `bundles/` without
an index row is treated as a crash leftover and replaced on the next commit of that digest.

### Tables

The index is SQLite in WAL mode.

`store_meta` (`key`, `value`): string metadata. Keys are `store_schema_version` (the schema version as
a decimal string), `created_by_viewer` (viewer version that created the store) and
`migrated_by_viewer` (viewer version that last migrated the store).

| Column | Meaning |
|---|---|
| `key` | metadata key (`store_schema_version`, `created_by_viewer`, `migrated_by_viewer`) |
| `value` | metadata value |

`bundles`: one row per stored bundle.

| Column | Meaning |
|---|---|
| `digest` | content digest (primary key, directory name under `bundles/`) |
| `name` | bundle name given at commit |
| `members_json` | JSON array of member objects |
| `created_at` | commit time, UTC ISO 8601 with `Z` suffix |
| `total_bytes` | sum of file sizes |
| `file_count` | number of files |
| `status` | bundle status (`ready`) |
| `error` | error text (NULL when ready) |
| `validated_with` | the reader version used for validation (re-validation in V1-15 (#60)) |

`files`: one row per file in every bundle, with an index on `sha256` used to link bundles such as
tomography runs to their source dataset.

| Column | Meaning |
|---|---|
| `digest` | owning bundle digest |
| `relpath` | POSIX path relative to the bundle root |
| `size` | file size in bytes |
| `sha256` | lowercase hex of the file content |

`derived`: derived-output records per bundle member.

| Column | Meaning |
|---|---|
| `digest` | owning bundle digest |
| `member` | member id |
| `deriver` | deriver name |
| `version` | deriver version |
| `params_key` | parameters key |
| `links_key` | links key |
| `status` | derivation status |
| `error` | error text (NULL on success) |
| `updated_at` | last update time, UTC ISO 8601 |

`jobs`: background job records (written by `plateau_rt.viewer.jobs`; see "Jobs").

| Column | Meaning |
|---|---|
| `job_id` | job id (primary key) |
| `kind` | job kind |
| `digest` | bundle digest (nullable) |
| `member` | member id (nullable) |
| `deriver` | deriver name (nullable) |
| `params_key` | parameters key (nullable) |
| `status` | job status |
| `stage` | job stage (nullable) |
| `done_bytes` | bytes done (nullable) |
| `total_bytes` | total bytes (nullable) |
| `error` | error text (nullable) |
| `created_at` | creation time, UTC ISO 8601 |
| `updated_at` | last update time, UTC ISO 8601 |

Versioning: `store_schema_version` (currently 1). `MIGRATIONS[v]` upgrades a store from `v - 1` to `v`,
each in one transaction, applied in order when a store is opened. A store newer than the viewer is
refused with `StoreVersionError` ("created by viewer X or later") and left untouched. Changing the
layout or a table requires a new migration and a version bump.

## Derivers

A **deriver** turns one member of a bundle into cached, named outputs (arrays, JSON, bytes) for a
requested parameter combination. `plateau_rt.viewer.derive` holds the registry, validation and cache;
`plateau_rt.viewer.testing` holds the determinism and golden helpers used by the test suite.

### Declaring a deriver

A deriver is any object with a `spec: DeriverSpec`, `param_space(ctx)` and `derive(ctx, params)`.
`DeriverSpec(name, version, kinds, params, eager)` fixes the deriver name (`[a-z][a-z0-9_]{0,63}`),
a `version >= 1`, the member `kinds` it supports (`rf_dataset`, `rf_partial`, `tomo_run`, `scene`)
and its `params`. Each `ParamSpec(name, kind, ...)` is one of:

| Kind | Validation | Role |
|---|---|---|
| `int` | `min`/`max` (ints) | range parameter, enumerated by callers |
| `float` | `min`/`max`/`step` (step divides the range) | range parameter |
| `enum` | non-empty unique `values` | space parameter |
| `view` / `bs` / `member_link` | none | space parameter, values come from the bundle |

**Range kinds** (`int`, `float`) are supplied by the request; **space kinds** (`enum`, `view`, `bs`,
`member_link`) form the allowed combinations returned by `param_space(ctx)`. `view`, `bs` and
`member_link` values are read from the bundle, so a request can only name views, base stations or
members that actually exist. `validate_params` canonicalises every value (e.g. `"+05"` -> `"5"`,
`"-3e1"` -> `"-30"`), rejects anything undeclared, missing, out of range or not in `param_space`, and
raises `BadParams`. A `BadParams` raised inside `derive` (a data-dependent limit) is treated the same:
nothing is written or recorded. An **eager** deriver has no range kinds and is enumerated from
`param_space` alone by `derive_eager`.

### Adding a deriver

1. Put a module under `plateau_rt.viewer.derive` that constructs its deriver and calls `register()`
   at import time (a private `_register` per module is fine).
2. Add one line `"plateau_rt.viewer.derive.<module>"` to `DERIVER_MODULES` in that package.
3. Record a golden with `pytest tests/test_viewer_golden.py --update-viewer-golden` and commit
   `tests/viewer_golden/derivers.json`.
4. **Bump `DeriverSpec.version` whenever an output's bytes can change** — including through shared
   domain code the deriver calls. The golden test fails otherwise ("bump DeriverSpec.version").

### Output rules

`derive` returns a non-empty `Mapping[str, value]` of output file names to values:

- `np.ndarray`: name ends in `.npy`; dtype must be one of `float16`, `float32`, `int32`, `uint8`,
  `uint32`, `bool`. Written little-endian and C-contiguous. Complex dtypes are rejected (split into
  magnitude and phase); object/structured/64-bit dtypes are rejected.
- `dict` / `list`: name ends in `.json`; written as canonical JSON (`sort_keys`, compact separators,
  UTF-8, no trailing newline). NaN becomes `null`; `inf` and non-string keys are rejected.
- `bytes`: name must not end in `.npy` or `.json`; written verbatim.

### Cache layout and keys

```text
bundles/<digest>/derived/<member>/<deriver>/v<version>/<params_key>/<links_key>/
```

`params_key` is `noparams` when there are no parameters, else the sorted `name=value` pairs joined
with `,`, with values percent-encoded, capped at 200 characters (`BadParams` when longer). `links_key`
is `nolink` when there are no links or all resolve to nothing; otherwise the link tokens joined with
`_`. A same-bundle link contributes `self`, a missing link `none`, and a linked bundle its digest;
overlong sequences become `links-<sha256>`. Every directory also holds `_meta.json` describing the
request, links and output files.

### Links and revalidation

`ctx.link(LinkTarget(member=..., sha256=..., kind=...))` resolves a link and allows the linked
bundle's raw root for the context's loaders. Resolution order: (1) a member id of the same bundle
(filtered by `kind`); (2) a file sha256 matched in the store index, preferring the same bundle, then
the earliest registered bundle, then digest/member/relpath; (3) otherwise no link. Before serving a
cache entry, every recorded link target is re-resolved against the current store and the resulting
`links_key` must still match, so deleting a linked bundle correctly falls back to the unlinked cache.
Each cache hit is also checked for the presence and size of every output file.

### The `derived` table

Each attempt upserts a row keyed by `(digest, member, deriver, version, params_key, links_key)`:
`status = ready` on success, `status = failed` with the `error` text when the deriver raises or
returns invalid outputs (no directory is left behind). `updated_at` uses the store's UTC timestamp.

### Safe loaders

`plateau_rt.viewer.safeio` is the **only** viewer module allowed to call `np.load` or import
`xml.etree` (enforced by an AST test). `resolve_inside(root, relpath)` rejects absolute paths, `..`
segments, NUL bytes and symlinks that leave the root; every path written in stored data is resolved
through it. The loaders check file sizes and declared array shapes before reading data, reject object
dtypes and non-`npy`/`npz` members, and parse XML with DTDs, entities and external references
forbidden.

### overview

The eager `overview` deriver (`plateau_rt.viewer.derive.overview`, version 1) runs over every
`rf_dataset` member with no parameters and writes `overview.json`: `member`, `schema_version`,
`mode`, `source_scene`, `num_views`, `num_bs`, `base_stations` (id, index, `position_m`,
`look_at_m`), `views` (id, index, `position_m`, `look_at_m`, `orientation_rad`), `frequency`
(`carrier_frequency_hz`, `bandwidth_hz`, `num_bins`, `bin_spacing_hz`, `delay_resolution_s`,
`unambiguous_delay_s` with manifest-then-derived fallbacks), `camera_model` (`fft_rows`/`fft_cols`
from `valid_mask`, `rx_rows`/`rx_cols`, spacings, `hemispheres`), `pairs` (view-major `view_id`,
`bs_id`, `bs_in_front_hemisphere`, `hemisphere_energy`, `total_energy`, `back_fraction`),
`contents` (`optical`, `optical_artifacts`, `transforms_json`, `path_gt`, `path_schema`,
`observations`, `partials`, `scene`, `placement`, `tomography_gt`) and `image_axes`. `image_axes`
uses the manifest's `camera_model.image_axes` when present, else the `default_m1` fallback
(`source: "default_m1"`).

## Kind detection and validation

`plateau_rt.viewer.kinds.detect_members` follows `docs/viewer_bundle.md`: with `bundle.json` it
checks the format version, member ids/kinds/paths, marker manifests, directory collisions, and
`for`/`source` links (a partial without `source` falls back to its manifest's `source_dataset`
joined lexically to its member directory); without it exactly one root marker manifest
(`dataset_manifest.json`, `partial_manifest.json`, `run_manifest.json`) gives the implicit member.
`validate_member` then checks each kind: `rf_dataset` re-reads the manifest and checks every
artifact (`camera_model`, per-view `pose`/`aperture_cfr`/`optical_*`, per-BS derivatives and
`observed.*`, path GT plus schema, optical `transforms.json`) for containment, existence, `.npy`
headers, `.npz` members, and the aperture CFR shape/dtype; `scene` resolves every
`scene_file_references` filename; `rf_partial` checks only the manifest envelope
(`schema_version` 1, `rf_camera_partial_observation` mode). Failures raise
`BundleValidationError(member_id, message)`, carrying the `ManifestError` text verbatim.
`tomo_run` is rejected until V3-2 (#71). All paths go through `resolve_inside` and are never
opened outside the bundle root; unsafe XML `filename` values are errors, not opens.

## Running the backend

```bash
uv sync --group viewer
python -m plateau_rt.viewer serve --host 127.0.0.1 --port 8765 --data /path/to/store
```

Run a **single worker**. Job state and request deduplication live inside the server process:
a second worker would run duplicate jobs and each worker would only see its own job status. For
that reason `serve` has no `--workers` option and the equivalent uvicorn form is pinned to one
worker:

```bash
VIEWER_DATA=/path/to/store uv run uvicorn --factory plateau_rt.viewer.api:create_app_from_env \
  --host 127.0.0.1 --port 8765 --workers 1
```

Bind to `127.0.0.1` by default and expose the viewer only behind a TLS reverse proxy (V0-D1, #27).
`python -m plateau_rt.viewer` also provides the `ingest` and `derive` commands (see "Command line").

## Jobs

A derivation is never computed inside a request when it is missing from the cache: the derive
route answers 202 and queues an in-process job. `plateau_rt.viewer.jobs.JobManager` runs each job
in a `spawn` child process so a crashing or memory-hungry deriver cannot take the server down.

- **Child process.** One `multiprocessing` `spawn` child per derivation. The child re-registers
  the deriver (the object is pickled to it) and calls `get_or_derive`, so the normal cache and
  `derived`-table behaviour applies.
- **Memory.** The child sets `RLIMIT_AS` to `VIEWER_DERIVE_MEM_BYTES`. The limit covers the whole
  child address space, including the interpreter and NumPy (about 0.7 GiB before any derivation
  data), so it must be set above that floor. A `MemoryError` anywhere in the exception chain is
  reported as `memory_limit`.
- **Timeout.** `VIEWER_DERIVE_TIMEOUT_S` starts when the child is started and therefore includes
  child start-up. A timeout kills the child and fails the job with `timeout`.
- **Concurrency.** `VIEWER_MAX_CONCURRENT_DERIVES` worker threads run one child each. Eager and
  lazy jobs share the same workers; a lazy job that arrives while eager jobs are pending is
  promoted ahead of them (the queue is lazy-first, FIFO within each priority).
- **Dedup.** Jobs are deduplicated by `(bundle digest, member, deriver, params_key)`. Running the
  same request again returns the active job instead of starting a second one.
- **Statuses and stages.** A job is `queued`, `running`, `done` or `failed`; its `stage` moves
  through `queued`, `starting`, `deriving` and `done`. The `error` column stores
  `"<reason>: <message>"`; `reason` is one of `timeout`, `memory_limit`, `restart`, `error` or
  `crashed` (a child that died without a final message).
- **Sticky failures.** A failed job is not retried by a plain GET: the failure is returned
  unchanged until the client calls the retry route. This keeps a broken deriver from being
  re-run on every page load. A `restart` failure is not sticky: the next GET queues a new job.
  The CLI `derive` command always re-runs failed derivations.
- **Restart recovery.** At startup every `queued` or `running` row is marked failed with reason
  `restart`, and a missing eager derivation is queued again for every stored bundle, so a
  derivation interrupted by a restart is recomputed.
- **Cancel on delete.** Deleting a bundle cancels its active jobs and waits briefly for the
  running children to stop before removing the files.
- **`jobs` table.** Each job is one row keyed by `job_id`, with `kind` (`eager`/`lazy`), the
  bundle `digest`, `member`, `deriver`, `params_key`, `status`, `stage`, `error`, `created_at`
  and `updated_at`. `done_bytes`/`total_bytes` are reserved for progress reporting.


## Configuration

`plateau_rt.viewer.settings.ViewerSettings.from_env` reads `os.environ`; `ENV_VARS` maps each field
to its variable. Byte sizes accept a decimal number with an optional binary suffix
(`KiB`/`MiB`/`GiB`/`TiB`, e.g. `4 GiB` or `4GiB`).

| Variable | Default | Meaning |
|---|---|---|
| `VIEWER_DATA` | `viewer_data` | Store directory (index, staging, bundles). |
| `VIEWER_MAX_UPLOAD_BYTES` | `4 GiB` | Maximum upload body size; larger bodies answer 413. |
| `VIEWER_MAX_EXTRACTED_BYTES` | `16 GiB` | Extraction cap, also the minimum free-space requirement. |
| `VIEWER_MAX_FILES` | `100000` | Maximum entries extracted from one archive. |
| `VIEWER_MAX_ARRAY_BYTES` | `1 GiB` | Largest `.npy`/`.npz` array accepted by validation and derivers. |
| `VIEWER_IMPORT_ROOTS` | empty | `os.pathsep`-separated absolute directories allowed for directory imports. |
| `VIEWER_DERIVE_TIMEOUT_S` | `120` | Per-derivation timeout, including child start-up. |
| `VIEWER_DERIVE_MEM_BYTES` | `4 GiB` | Per-derivation `RLIMIT_AS` for the child address space. |
| `VIEWER_MAX_CONCURRENT_DERIVES` | `2` | Worker threads shared by eager and lazy jobs (lazy first). |
| `VIEWER_ALLOWED_HOSTS` | `127.0.0.1,localhost` | `Host` allow-list (enforced by V0-9, #44). |
| `VIEWER_READ_ONLY` | `false` | Reported by `/api/health`; rejecting mutating operations is V1-12 (#59). |

## HTTP API

All routes are under `/api` except the static hook. Errors always use the envelope below.

| Method | Path | Success | Body / notes |
|---|---|---|---|
| `GET` | `/api/health` | 200 | `status`, `viewer_version`, `read_only`, `store_schema_version`. |
| `GET` | `/api/derivers` | 200 | `derivers`: name, version, kinds, eager, params. |
| `PUT` | `/api/bundles/upload?name=<name>` | 201 / 200 | `digest`, `created`, `members`; raw archive body, not multipart. |
| `GET` | `/api/bundles` | 200 | `bundles`: one summary per bundle. |
| `GET` | `/api/bundles/{digest}` | 200 | Summary plus `error`, `validated_with`, `derived` rows. |
| `DELETE` | `/api/bundles/{digest}` | 200 | `{"confirm": "<digest>"}` body; `digest`, `deleted`. |
| `GET` | `/api/bundles/{digest}/members/{member}/raw/{path}` | 200 | File bytes with the download security headers. |
| `GET` | `/api/bundles/{digest}/members/{member}/derived/{deriver}` | 200 / 202 | Reuses a cached derivation (200) or queues a job (202 `job_id`, `status_url`, `status`). |
| `POST` | `/api/bundles/{digest}/members/{member}/derived/{deriver}/retry` | 202 | Re-queues a failed derivation; 409 when it is ready or not failed. |
| `GET` | `/api/jobs/{job_id}` | 200 | One job row plus `status_url` and (when `done`) `result`. |
| `GET` | `/api/bundles/{digest}/status` | 200 | `digest`, `complete`, `eager` counts, `items` and `failures`. |
| `GET` | `.../derived/{deriver}/v{version}/{params_key}/{links_key}/{name}` | 200 / 304 | One derived file with `ETag`, immutable cache, conditional `If-None-Match`. |
| `GET` | `/` | 200 | `index.html` with `no-cache`, or a plain-text pointer when no frontend is installed. |
| `GET` | `/static/<build>/...` | 200 | One snapshotted frontend file with `public, max-age=31536000, immutable`. |
| `GET` | `/static/<path>` | 200 | Unversioned frontend file (tests and debugging) with `no-cache`. |

Upload rules. The body is the raw archive (`zip`, `tar`, `tar.gz`), **not** multipart:
`python-multipart` would spool the body to `/tmp`, double the disk use and only check the size
after the whole body was read. `name` must be 1..200 characters without control characters; a
present `Content-Type` must be `application/octet-stream`. `Content-Length`, when present, is
validated and compared with `VIEWER_MAX_UPLOAD_BYTES` **before** the body is read, so an oversized
declared body answers 413 without consuming a byte. The body is streamed into
`staging/<uuid>.upload` (never buffered whole); a chunked body that crosses the cap stops
immediately with 413, and a free space below `VIEWER_MAX_EXTRACTED_BYTES` answers 507 before
reading. The archive is extracted with `safe_extract`, validated, and committed; `created` is
`true` (201) for a new digest and `false` (200) for a duplicate. Every failure path removes the
staging file and staging directory, so a rejected upload leaves no trace. A successful commit of a
new bundle calls the `on_bundle_committed(store, digest)` hook once in the threadpool; the hook
queues the bundle's missing eager derivations and never fails the upload. Duplicates and failures
do not call it.

Raw serving. A member's raw directory is the `scene` XML's directory for `scene` members and the
member path for every other kind; the requested `path` is **relative to that member directory**, so
e.g. `.../members/dataset/raw/dataset_manifest.json`. Both the member directory and the target pass
through `safeio.resolve_inside`, so absolute paths, `..` and symlink escapes answer 400. Every file
is served as `application/octet-stream` with `X-Content-Type-Options: nosniff`,
`Content-Disposition: attachment`, `Content-Security-Policy: sandbox` and
`Cache-Control: private, max-age=0`: a stored `.html` or `.svg` must never execute as active
content in the viewer's origin.

Derived files. The served URL contains the deriver version, the parameter key and the links key, so
a cache entry can never be confused with another version or parameter combination; after a version
bump the old URLs answer 404. A file is served only when its derivation has a `ready` row in the
index and the file is listed in the cache directory's `_meta.json` with a matching size (so
`_meta.json` itself and any other name answer 404). The `ETag` is `"<sha256>"` from `_meta.json`,
the response is `Cache-Control: public, max-age=31536000, immutable`, and `If-None-Match`
(including `W/` prefixes and `*`) answers 304 without a body. Path segments of the URL are
percent-encoded (keeping `=` and `,`), so a `%` inside a `params_key` travels as `%25`.

Idempotent GET. The derive route reuses a cached derivation and answers 200. On a cache miss it
queues a lazy job and answers 202 with `job_id`, `status_url` and the initial `status`; the client
polls `GET /api/jobs/{job_id}` and reads the ready body from the job's `result` (or re-requests the
route). A failed derivation is sticky: the same GET returns the failed job rather than running it
again, and the client must `POST .../retry` to try once more. `GET /api/bundles/{digest}/status`
reports which eager derivations are `done` (cache ready), `queued`, `running`, `failed` or
`missing`, and lists every failed job of the bundle that does not have a ready cache.

Error envelope. Every error response contains exactly
`{"error": {"type": <type>, "member": <id or null>, "message": <text>}}` and
`Content-Type: application/json`; messages are passed through verbatim (newlines and HTML
included).

| `type` | Status | Raised for |
|---|---|---|
| `unsafe_archive` | 400 | `UnsafeArchiveError` from format, traversal, links, devices, duplicates, size or file caps. |
| `validation` | 400 | `BundleValidationError` (manifest text verbatim, `member` set). |
| `unknown_kind` | 400 | `UnknownKindError`: no supported member kind or an unsupported kind. |
| `too_large` | 413 | Body or delete body over its cap. |
| `insufficient_storage` | 507 | Free space below `max_extracted_bytes`, or `ENOSPC`/`EDQUOT`. |
| `not_found` | 404 | Unknown digest, member, deriver, file or ready derivation. |
| `bad_params` | 400 | Invalid query/name/confirm/Content-Type or `derive.BadParams`. |
| `derive_failed` | 500 | `derive.DeriveError` from a failing or invalid derivation. |
| `conflict` | 409 | Retry of a derivation that is already ready or has not failed. |

## Architecture

| Module | Role |
|---|---|
| `plateau_rt.viewer.settings` | `ViewerSettings` and environment parsing. |
| `plateau_rt.viewer.extract` | `safe_extract` and the archive rejection reasons. |
| `plateau_rt.viewer.store` | Content-addressed store, SQLite index, commit/delete and derived cache layout. |
| `plateau_rt.viewer.safeio` | Path containment and bounded XML/`.npy`/`.npz` loaders. |
| `plateau_rt.viewer.kinds` | Kind detection and member validation. |
| `plateau_rt.viewer.ingest` | HTTP-free `ingest_staged`/`ingest_archive` used by the API and later CLIs. |
| `plateau_rt.viewer.derive` | Deriver registry, parameter validation and versioned cache. |
| `plateau_rt.viewer.jobs` | Background job manager (`spawn` children, limits, dedup, recovery). |
| `plateau_rt.viewer.__main__` | Headless `serve`/`ingest`/`derive` command line. |
| `plateau_rt.viewer.api` | FastAPI app factory, error envelope, bundle, derived and job routes. |
| `plateau_rt.viewer.api.static_assets` | Build-hashed static snapshot served at `/` and `/static/...` (see "Frontend"). |
| `plateau_rt.viewer.static` | Frontend assets (V0-6, #41): `index.html`, `css/viewer.css`, `js/` modules, `vendor/`. |
| `plateau_rt.viewer.testing` | Determinism and golden helpers for tests. |

Data flow: upload (`PUT /api/bundles/upload`) -> streamed into `staging/<uuid>.upload` ->
`safe_extract` into a staging directory -> `detect_members`/`validate_member` -> `Store.commit`
(content digest, read-only `raw/`, index rows) -> `on_bundle_committed` hook -> the job manager
queues every missing eager derivation. A derive request on a cache miss queues a lazy job (or
promotes an already queued one); each job runs `get_or_derive` in a limited `spawn` child and
caches the outputs under `bundles/<digest>/derived/...`.

## Command line

`python -m plateau_rt.viewer` is the headless entry point (it is not routed through
`plateau_rt.cli.main`). Every command accepts `--data DIR`. `ingest` and `derive` print exactly one
JSON object to stdout with `indent=2` and sorted keys (the heavy CI parses it); diagnostics never
go to stdout. `serve` logs through uvicorn.

| Command | Behaviour |
|---|---|
| `serve [--host 127.0.0.1] [--port 8765] [--data DIR]` | Runs the uvicorn backend with exactly one worker (see "Running the backend"). |
| `ingest PATH [--name NAME] [--data DIR]` | Ingests a directory or archive; `--name` defaults to the path basename. Does not run derivations. |
| `derive DIGEST [--all-lazy] [--data DIR]` | Runs the eager derivations and, with `--all-lazy`, the lazy ones. |

`ingest` prints `{"digest", "created", "members"}` and exits 0, or
`{"error": {"type", "member", "message"}}` and exits 1 with type `unsafe_archive`, `unknown_kind`,
`validation`, `not_found` (missing path) or `error`. `derive` prints
`{"digest", "results", "failed"}` with one result per combination (`member`, `deriver`, `params`,
`params_key`, `status`, `cached`, `job_id`, `reason`, `error`) and exits 1 when any result failed or
the digest is unknown; it exits 0 when the only non-`done` results are skipped. A deriver with a
**range** parameter (`int`/`float`) cannot be enumerated, so `--all-lazy` reports it as `skipped`
rather than deriving it. Without `--all-lazy`, lazy derivers are not enumerated at all.

## Frontend

The browser frontend (V0-6, #41) is plain ES modules loaded directly by the browser: no bundler,
no build step, no framework, no Node toolchain. `index.html` references
`/static/__BUILD__/css/viewer.css` and `/static/__BUILD__/js/app.js` (a `type="module"` script);
the backend replaces `__BUILD__` with the build id, so every relative ES-module import resolves
under the same `/static/<build>/` prefix automatically. JavaScript never hard-codes `/static/`
paths: to address another static file it uses `new URL("../vendor/...", import.meta.url)`.

Files and roles:

| File | Role |
|---|---|
| `static/index.html` | Shell: `#app` root, stylesheet and module script, `noscript` fallback. |
| `static/css/viewer.css` | Neutral styles: layout, header, tabs, tables, status boxes, dropzone. |
| `static/js/app.js` | Hash router: header, route rendering, panel mounting, `window.__viewer` hook. |
| `static/js/api.js` | API wrapper: `ApiError`, requests, uploads, job polling, derivation helpers. |
| `static/js/npy.js` | Pure `.npy` parser (`float16`/`float32`/`int32`/`uint8`/`uint32`/`bool`). |
| `static/js/state.js` | Hash parsing/formatting and the router store. |
| `static/js/dom.js` | `h()` element builder, `escapeHtml`, and the shared status views. |
| `static/js/strings.js` | Every user-visible English string plus the `fmt` template helper. |
| `static/js/format.js` | Display formatting only (bytes, Hz, ns, numbers, energies, vectors). |
| `static/js/vendor.js` | `vendorUrl` plus the lazy three.js/Plotly loaders used by later panels. |
| `static/js/panels/registry.js` | Panel list in tab order, `panelsForKind` and `getPanel`. |
| `static/js/panels/bundles.js` | Home panel: upload box and bundle list with delete. |
| `static/js/panels/overview.js` | Overview panel for `rf_dataset` members. |

Hash format (shareable link, V0-D1 #27): home is `#/` (also for an empty hash or anything
unrecognised); a bundle is `#/b/<digest>[/<member>[/<panel>]]?v=1&...` where `<member>` and
`<panel>` are optional (default member/panel when absent). `digest` must be 64 lowercase hex
chars, member `[A-Za-z0-9_.-]{1,64}` and panel `[a-z][a-z0-9_-]{0,63}`; anything else is the
`invalid` route. The selection shared by all panels:

| Key | URL param | Type | Default |
|---|---|---|---|
| `view` | `view` | id string (1..200 chars, no control chars) | `null` |
| `bs` | `bs` | id string | `null` |
| `hemisphere` | `hemi` | `front`/`back` | `"front"` |
| `orientation` | `orient` | `rf`/`photo` | `"rf"` |
| `path` | `path` | integer >= 0 | `null` |
| `pixel` | `px` | `[row, col]` integers >= 0, URL form `row,col` | `null` |
| `delayBin` | `dbin` | integer >= 0 | `null` |
| `variant` | `variant` | id string | `null` |

`v=1` marks the format version; defaults are never written, invalid values keep their default and
unknown params are ignored. `parseHash(formatHash(route, state))` round-trips, and formatting a
parsed canonical hash returns it unchanged. A `v` other than 1 still parses but shows a notice
(compat is V1-11).

API wrapper (`js/api.js`). Failures throw `ApiError` with `{type, member, message}` from the error
envelope (plus `status`, and `reason`/`jobId` for failed jobs). Every request other than
`GET`/`HEAD` sends `X-Viewer-Request: 1`. Uploads use `XMLHttpRequest` with progress callbacks
(`uploading`, then `validating` once the body is in) and resolve with the parsed JSON on 200/201.
A 202 derivation answer is polled with `nextPollDelay` backoff (0.5 s growing to a 5 s cap) until
the job is `done` (returning its `result`) or `failed` (throwing `derive_failed` with the reason).
Failures are sticky: the derive route keeps pointing at the failed job until the client POSTs
`retry`, which the overview panel exposes as a Retry button.

DOM rules. Elements are built with `h()` from `dom.js`, which sets attributes, `dataset`, `style`
objects and `on*` listeners and turns every data string into a text node; `innerHTML`,
`outerHTML`, `insertAdjacentHTML`, `document.write`, `eval`, `new Function` and inline event
handlers are forbidden, and `index.html` has no inline `<script>` or `<style>`. Errors render with
`errorView`, which keeps `ManifestError` newlines verbatim in a `pre.message`. All user-visible
English strings live in `strings.js` (templates use `{name}` placeholders filled by `fmt`); panel
files contain no English sentences. `npy.js` parses `.npy` 1.0/2.0 headers with strict regexes (no
eval) into `float16` (decoded through a lookup table), `float32`, `int32`, `uint32`, `uint8` and
`bool` arrays, C order only. JavaScript never computes physics or conventions (orientation,
mirroring, hemisphere, markers, delay wrap, gauge, energies): it only draws backend values, plus
unit formatting (`12.5 ns`, `3.5 GHz`, `1.2 MiB`) through `format.js`. `window.__viewer` exposes
`{store, api, parseHash, formatHash, parseNpy}` as a test hook for browser smoke tests.

Caching. `/` serves the templated `index.html` with `Cache-Control: no-cache`.
`/static/<build>/...` serves the startup snapshot with
`Cache-Control: public, max-age=31536000, immutable` and an `ETag` (`If-None-Match` answers 304);
`build` is 12 lowercase hex chars of the sha256 over all static files, and `index.html` carries
the `__BUILD__` placeholder. The unversioned `/static/<path>` form is `no-cache` for tests and
debugging. Lookups are dictionary lookups only, so `..`, encoded dots, absolute paths and NUL
bytes can never reach the disk, and `index.html` is only served at `/`.

Vendor. `static/vendor/` holds third-party files served under `/static/<build>/vendor/`:
three.js 0.186.1 ships no minified build any more, so the unminified `three.module.js` plus
`three.core.js` are vendored; `OrbitControls.js` has its `'three'` import rewritten to
`./three.module.js` because an import map would need an inline script exception in the CSP;
Plotly uses the `plotly.js-strict-dist-min` 4.1.1 bundle, which avoids `eval`-style code for the
CSP. `VENDOR.json` records each file's `package`, `version`, `source_url`, `source_path`,
`sha256`, `upstream_sha256`, `license` and `license_file` (itself listed). To update a vendor
file: download the npm tarball at `source_url`, copy `source_path` to `path`, re-apply the
recorded `modification`, and refresh `sha256`/`upstream_sha256`.

### Adding a slice (frontend)

1. Write `static/js/panels/<id>.js` exporting the panel object (`{id, title, kinds, mount, update,
   unmount}`).
2. Add one line to `PANELS` in `panels/registry.js`.
3. Add one `(panel_id, state)` case to the browser smoke test of V0-8 (#43).
4. Put new UI strings in `strings.js`.

## Adding a slice (backend)

1. Write a deriver module under `plateau_rt.viewer.derive` (see "Adding a deriver" above).
2. Add one line `"plateau_rt.viewer.derive.<module>"` to `DERIVER_MODULES`.
3. Record its golden with `pytest tests/test_viewer_golden.py --update-viewer-golden`, and bump
   `DeriverSpec.version` whenever an output's bytes can change.

The generic derived endpoints (`/api/derivers`, the derive route and the versioned file route) then
serve the new deriver with no API change; no new route is needed per slice.
