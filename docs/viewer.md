# Web Viewer

The shared web viewer for RF-camera bundles (epic #26). The bundle format is defined in
`docs/viewer_bundle.md`; more sections (usage, operations, development) are added by later issues.

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

`jobs`: background job records (used by V0-5b (#40) and V1-10 (#56)).

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
