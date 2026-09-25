# Viewer Bundle Format

A **bundle** is what is uploaded to the shared web viewer (epic #26): either an
archive (zip / tar / tar.gz) or a registered server directory. Format version 1,
decided in issue #31.

A dataset directory alone is not enough. `dataset_manifest.json`'s
`source_scene` is the scene XML path as the writer saw it (relative to the
writer's working directory, e.g. `ci-reports/mock_results/mock_building.city.xml`),
not relative to the dataset, so the Mitsuba XML and its PLY files must be carried
separately. A partial / summary dataset (`rf-camera-partial`, see
`docs/rf_camera_partial.md`) lives in its own directory; `partial_manifest.json`'s
`source_dataset` is the source dataset path relative to the partial output
directory (no sha256). Other partial-manifest paths (`source_manifest`,
`camera_model_source`, `path_geometry_gt.artifact`, `path_geometry_gt.path_schema`)
are also relative to the partial output directory and point INTO the source dataset
(they start with the `source_dataset` prefix). Tomography results (T16,
`rf-tomo-bench --out DIR`) are a separate directory with `run_manifest.json`,
`results.jsonl`, `recon/**/*.npz`; their exact contract is decided by V3-D1 (#69).
`placement/` (coverage placement, #16) and `tomography_gt.npz` (T17) live inside the
dataset directory. Observed variants (`rf-camera-observe`) also live inside the
dataset directory. Mitsuba XML `<string name="filename" value="..."/>` values are
relative to the XML file's directory (the scene compiler writes the PLY basename
next to the XML).

Consumers: V0-2b (#34) fixtures, V0-4b (#38) kind detection and validation, V1-1
(#45) `rf-dataset-bundle`, V1-9 (#55) partial links, V3-2 (#71) tomography results.
V0-4 (#37) provides `resolve_inside`.

## Options considered

| Option | Content | Outcome |
|---|---|---|
| A. zip the dataset directory as is (kind from the manifest alone) | dataset directory only | accepted as a fallback, not canonical: no scene, no partial link |
| B. self-contained bundle with `bundle.json` describing members and relations, built by the `rf-dataset-bundle` command (V1-1, #45) | dataset + scene + partials + runs + `bundle.json` | canonical format |
| C. viewer searches `source_scene` on the server file system | needs a shared file system, hurts reproducibility | not done |

Decision: B is the canonical format; A is also accepted (when there is no
`bundle.json`); C is not done.

## Layout

```text
<bundle root>/
  bundle.json          # optional; bundle_format_version: 1
  dataset/             # rf_dataset: dataset_manifest.json (+ placement/, tomography_gt.npz, observed variants)
  scene/               # scene: Mitsuba XML and the PLY files it references
  partials/<name>/     # rf_partial: partial_manifest.json
  runs/<name>/         # tomo_run: run_manifest.json
```

The directory names `dataset/`, `scene/`, `partials/`, `runs/` are the convention
written by `rf-dataset-bundle`; the viewer never relies on them — only `bundle.json`
(or the fallback detection) decides what is where. Files not belonging to any member
are allowed, ignored by the viewer, but part of the bundle digest (like `bundle.json`
itself: the same data with a different `bundle.json` is a different bundle).

## Member kinds

| Kind | `path` points to | Marker | Contents |
|---|---|---|---|
| `rf_dataset` | directory | `dataset_manifest.json` in that directory | schema v3 / v2 dataset read by `load_rf_dataset_manifest`; `placement/`, `tomography_gt.npz`, optical references, path GT and observed variants are part of it (no separate kinds) |
| `rf_partial` | directory | `partial_manifest.json` | partial / summary dataset |
| `tomo_run` | directory | `run_manifest.json` | tomography run output (contract decided by #69) |
| `scene` | file ending in `.xml` | the file itself (root element `<scene>`, parsed with defusedxml) | Mitsuba XML; the PLY files it references must be in the bundle |

Exactly these four kinds exist. In particular, placement and tomography_gt are NOT
kinds: they are files inside an `rf_dataset` member, never members of their own.

## Bundle root

The root is chosen the same way for archives AND registered directories, with or
without `bundle.json`: if the top level of the extracted archive (or the registered
directory) contains exactly one entry and it is a directory, the root is that
directory; this descends at most once. Every entry counts (no special-casing of
hidden files). Then: if `<root>/bundle.json` exists, use it (see
`## bundle.json keys`); otherwise use fallback detection (see
`## Detection without bundle.json`).

## bundle.json keys

```json
{
  "bundle_format_version": 1,
  "members": [
    {"id": "dataset", "kind": "rf_dataset", "path": "dataset"},
    {"id": "scene", "kind": "scene", "path": "scene/mock_building.city.xml", "for": "dataset"},
    {"id": "p0", "kind": "rf_partial", "path": "partials/p0", "source": "dataset"},
    {"id": "smoke", "kind": "tomo_run", "path": "runs/smoke", "source": "dataset"}
  ],
  "created_by": {"tool": "rf-dataset-bundle", "tool_version": 1}
}
```

| Key | Type | Required | Meaning |
|---|---|---|---|
| `bundle_format_version` | integer | required | must be `1`; any other value (or a non-integer, or a bool) is an error "unsupported bundle_format_version" — never guessed |
| `members` | array of objects | required | non-empty; at most 1024 entries |
| `members[].id` | string | required | full match `[A-Za-z0-9_.-]{1,64}`, not `.` or `..`; unique within the bundle (case-sensitive); used in URLs (`/members/{member}/`) and store paths (`derived/<member>/`) |
| `members[].kind` | string | required | one of `rf_dataset`, `rf_partial`, `tomo_run`, `scene` |
| `members[].path` | string | required | relative POSIX path from the bundle root (rules in "Paths"); a directory for `rf_dataset` / `rf_partial` / `tomo_run`, an `.xml` file for `scene` |
| `members[].for` | string | optional | only on `scene`: id of the `rf_dataset` member this scene belongs to. At most one scene per dataset. A scene without `for` is shown on its own |
| `members[].source` | string | optional | only on `rf_partial` and `tomo_run`: id of the `rf_dataset` member they were derived from / evaluated against; takes precedence over paths or hashes recorded in the data |
| `created_by` | object | optional | provenance, informational only; extra keys inside it are allowed and ignored |
| `created_by.tool` | string | optional | e.g. `rf-dataset-bundle` |
| `created_by.tool_version` | integer | optional | version of the tool's output |

The file must be a UTF-8 JSON object of at most 1 MiB. Unknown keys at the top level
or inside a member are errors (so a typo such as `sorce` never silently drops a
link). Future incompatible changes bump `bundle_format_version`. Relations between
members live only in `for` / `source`; manifests are never rewritten to express them.

## Detection without bundle.json

Fallback (option A): look at `<root>` only (no recursion). If exactly one of
`dataset_manifest.json`, `partial_manifest.json`, `run_manifest.json` exists there,
the bundle has one member of that kind whose path is the root itself. Implicit member
ids: `rf_dataset` -> `dataset`, `rf_partial` -> `partial`, `tomo_run` -> `run`. If two
or more of them exist, or none, it is an error. A lone scene (XML without a dataset)
is not detected in fallback mode; it needs `bundle.json`. In fallback mode an
`rf_partial`'s `source_dataset` normally points outside the root, so the partial is
shown standalone (see "Link resolution"); a `tomo_run` can still be linked by sha256.

Example: archive `mvp_dataset.tar.gz` containing `mvp_dataset/dataset_manifest.json`
(plus the dataset files) extracts to a single top-level directory, so the root is
`mvp_dataset/` and the bundle has one member with id `dataset`. There is no scene,
so the 3D view has no mesh.

## Validation errors

Every condition below rejects the whole bundle (nothing is registered; the message is
shown as is, like `ManifestError`):

1. `bundle.json` is not valid UTF-8 JSON, not an object, or larger than 1 MiB.
2. `bundle_format_version` missing or not `1`.
3. `members` missing, not a list, empty, or longer than 1024.
4. unknown key at top level or in a member; `created_by` not an object.
5. member `id` missing / invalid / duplicate.
6. member `kind` missing or not one of the four kinds.
7. member `path` invalid (see "Paths"), missing on disk, or of the wrong type
   (directory vs `.xml` file).
8. a directory member without its marker manifest (e.g. `kind: rf_partial` but no
   `partial_manifest.json`), or a `scene` whose root element is not `<scene>`.
9. two directory members with the same path, or one directory member nested inside
   another (`dataset` and `dataset/p0`) — each file belongs to at most one directory
   member. (A `scene` XML may live anywhere, including inside a dataset directory.)
10. `for` on a non-`scene` member, `source` on a member that is not `rf_partial` /
    `tomo_run`, a `for` / `source` naming an unknown id or a member that is not
    `rf_dataset`, or two scenes with the same `for`.
11. fallback mode: zero or several marker manifests at the root.
12. a member's own data fails the typed reader (`ManifestError` from
    `load_rf_dataset_manifest`, partial / run manifest validation), or a path written
    in the data is invalid (see "Paths inside the data").

Archive-level checks (symlinks, hard links, device files, absolute or `..` entry
names, size and file-count limits) happen earlier during extraction (V0-3, #35, limits
from #28) and are not repeated here.

## Paths

### Member paths

`path` in `bundle.json`: relative POSIX path; not empty; no leading `/`, no drive
letter or `\`, no NUL; no empty, `.` or `..` segments (so no trailing `/` and no
`./dataset`); compared as plain strings (no normalisation, which is why non-canonical
spellings are rejected rather than normalised); every component must be a real
directory / file, never a symlink; the path `.` (the root itself) is not allowed in
`bundle.json` (only the fallback's implicit member uses the root).

### Paths inside the data

Paths written inside member files: manifest artifact paths (resolved against the
manifest's directory), Mitsuba XML `filename` values (resolved against the XML file's
directory), partial-manifest paths and `source_dataset` (resolved against the partial
member directory). All are resolved with `resolve_inside(bundle_root, ...)` from V0-4
(#37): a path that is absolute or that leaves the bundle root is never opened; the
viewer never reads outside `raw/`.

| Path | Absolute or outside the bundle root |
|---|---|
| manifest artifact (`dataset_manifest.json`, `partial_manifest.json`, `run_manifest.json`) | error |
| Mitsuba XML `filename` | error |
| partial `source_dataset` | not an error: treated as "no link" |
| dataset `source_scene` | never resolved (informational only; the scene comes from a `scene` member) |

The containment boundary is the bundle root, not the member directory, so a partial
may legitimately reference files of an `rf_dataset` member in the same bundle.

## Link resolution

A link written in `bundle.json` always wins over paths or hashes in the data, and an
invalid link in `bundle.json` is an error (validation error 10), not a fallback.

### Scene to dataset

1. the scene's `for`.
2. otherwise the scene is unlinked and shown standalone.

The dataset's `source_scene` is never used to find the scene, and a dataset without a
linked scene is shown without a mesh.

### Partial to source dataset

1. the partial's `source` in `bundle.json`.
2. otherwise `source_dataset` from `partial_manifest.json`, joined to the partial
   member directory and normalised lexically; if it stays inside the bundle root and
   equals the `path` of an `rf_dataset` member of the same bundle, that member is the
   source.
3. otherwise the partial is shown standalone (its own artifacts only).

A link is only ever made within the same bundle (never to another stored bundle).

**Rebasing source paths.** `rf-dataset-bundle` copies directories without rewriting
manifests, so the recorded `source_dataset` (e.g. `../mvp_dataset`) usually no longer
matches the bundle layout (`../../dataset`). When the source was resolved (by step 1
or 2), every partial-manifest path whose lexical form starts with the recorded
`source_dataset` prefix (`source_manifest`, `camera_model_source`,
`path_geometry_gt.artifact`, `path_geometry_gt.path_schema`) is rebased: the prefix is
replaced by the linked dataset member's path, and the result must still pass
`resolve_inside`. Other paths (e.g. `element_mask.file`, `views[].artifacts`) resolve
against the partial member directory. When there is no source, prefixed paths are not
opened and the panels that need them (source camera model, path GT) say "source
dataset not in bundle". Consistency of the linked pair (view ids at `source_index`, BS
ids, schema version) is checked in V1-9 (#55); a mismatch is shown as a link error,
not silently ignored.

Worked example: partial member `partials/p0` records
`source_dataset: "../mvp_dataset"` and
`camera_model_source: "../mvp_dataset/camera_model.npz"`, and `bundle.json` sets
`source: "dataset"`; the viewer rebases the prefix to the linked dataset member's
path, giving `dataset/camera_model.npz` (relative to the bundle root). Step 2 alone
would not link this partial: `../mvp_dataset` joined to `partials/p0` is
`partials/mvp_dataset`, which is not a member path.

### Tomography run to source dataset

1. the run's `source` in `bundle.json` (same bundle).
2. otherwise the source-dataset sha256 recorded in `run_manifest.json` is matched
   against the per-file sha256 in the store index (`index.sqlite`): a match is an
   `rf_dataset` member whose recorded file (which file is hashed, and the key name and
   location in `run_manifest.json`, are decided by V3-D1, #69) has that sha256.
   Candidates in the same bundle win; otherwise among other stored bundles the earliest
   registered wins (ties broken by bundle digest, then member id), and the link records
   the chosen bundle digest (derived-data paths include linked digests, epic #26
   "Store").
3. otherwise the run has no source dataset.

This is valid: a `tomo_run` may come from an analytic L0 scene with no dataset (where
its GT lives is decided by #69).

## Invariants

- Raw data (manifests included) is never rewritten; relations live only in
  `bundle.json`.
- The viewer does not search the server for `source_scene` or other external paths
  (option C).
- A bundle is immutable once stored (content-addressed digest, epic #26).
- What the viewer reads is limited to the bundle root.

## Downstream issues

| Issue | Uses |
|---|---|
| V0-2b (#34) | fixture bundle layout |
| V0-4b (#38) | kind detection, validation errors, member ids |
| V1-1 (#45) | writes this layout and `bundle.json` (`created_by.tool = rf-dataset-bundle`) |
| V1-9 (#55) | partial link and rebasing |
| V3-D1 (#69) | run-manifest sha256 key and hashed file |
| V3-2 (#71) | tomo_run links |
