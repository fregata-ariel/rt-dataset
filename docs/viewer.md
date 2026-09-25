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
