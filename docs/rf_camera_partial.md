# Partial / Summary RF-Camera Observations

`rf-camera-partial` derives a *partial* or *summary* dataset from a multi-view
RF-camera dataset written by `rf-camera-multiview`: keep a subset of views,
zero an aperture element mask, slice a contiguous frequency subband, and
optionally replace the raw CFR by a summary (per-element power, or one dominant
delay). `partial_manifest.json` records exactly what was kept so a partial
sample can later be paired with its full sample (foundation-model training,
issue #14).

The source dataset comes from a GPU trace that is not bit-reproducible, so the
tool never modifies the source: it only reads through the typed reader
(`plateau_rt.application.rf_dataset_manifest`, schema v3 and v2) and writes a
separate output directory.

## CLI

```bash
PYTHONPATH=./src python -m plateau_rt.cli.main rf-camera-partial DATASET_DIR OUT_DIR \
  --view-fraction 0.5 --element-mask checkerboard --subband 2:6 \
  --summary delay --mask-fraction 0.5 --seed 0 [--overwrite]
```

| Option | Meaning |
|---|---|
| `DATASET_DIR` | Existing `rf-camera-multiview` output directory |
| `OUT_DIR` | Fresh output directory (must not equal, contain, or be contained in `DATASET_DIR`) |
| `--view-fraction` | Fraction of views to keep, `(0, 1]`; count uses round-half-up (`floor(f*n+0.5)`, at least 1) |
| `--element-mask` | `none`, `random`, `every_other_row`, `every_other_col`, `checkerboard` (`True` = kept) |
| `--mask-fraction` | Fraction of elements kept for `random` (round-half-up, at least 1) |
| `--subband` | `START:STOP` slice (Python semantics, `STOP` exclusive); default is the full band |
| `--summary` | `none` (raw CFR), `power`, or `delay` (`delay` needs at least 2 subband bins) |
| `--seed` | Integer seed for the view draw and the random mask |
| `--overwrite` | Replace a previous partial output in `OUT_DIR` (otherwise a non-empty `OUT_DIR` fails) |

Mock shortcut (after `rf-camera-multiview-mock`):

```bash
make rf-camera-partial-mock
```

Errors (unknown schema, bad subband, unsafe paths, non-empty output without
`--overwrite`, missing files) exit non-zero with a one-line message and no
traceback.

## Output layout

`element_mask.npy` is the bool `[rows, cols]` mask used for every kept view.
Per summary kind (paths relative to `OUT_DIR`):

```text
rf_camera_partial/
  partial_manifest.json
  element_mask.npy
  views/
    <view_id>/rf/
      aperture_cfr.npy                        # summary=none: complex64 [bs, hemi, row, col, sub_freq]
      element_power.npy + hemisphere_power.npy  # summary=power: float32 [bs, hemi, row, col] + float64 [bs, hemi]
      <bs_id>/dominant_delay.json              # summary=delay: one file per (view, BS)
```

`summary=none` keeps view-level artifact `aperture_cfr` (always 5-D, also for
v2 input, where the reader synthesises `B = 1` as `bs_000`). `summary=power`
stores the two view-level power artifacts. `summary=delay` stores one JSON per
`(view, BS)` under `rf/<bs_id>/`, listed as `artifacts: {"dominant_delay": ...}`
in that view's `bs[]` entry (mirroring v3's per-BS derived artifacts under
`rf/bs_XXX/`); view-level `artifacts` is `{}` for `delay`.

## Partial manifest keys

`partial_manifest.json` has `schema_version: 1` and
`mode: "rf_camera_partial_observation"`.

- `source_dataset`: path of the source dataset relative to `OUT_DIR` (POSIX).
- `source_manifest`: path of the source `dataset_manifest.json` relative to `OUT_DIR`.
- `source_manifest_schema_version`: the source's schema (3 or 2).
- `options`: the CLI options used (`view_fraction`, `element_mask_kind`,
  `mask_fraction`, `subband`, `summary`, `seed`).
- `kept_view_indices` / `kept_view_ids`: kept views, sorted ascending.
- `element_mask`: `kind`, `fraction`, `seed`, `file`, `kept_count`, `total`.
- `subband`: `start`, `stop`, `num_bins`, `frequency_offsets_hz`,
  `absolute_frequencies_hz` (carrier + kept offsets), `delay_resolution_s`
  (`1/(n_sub*delta_f)`) and `unambiguous_delay_s` (`1/delta_f`, with `delta_f`
  the full-grid spacing; both `None` when fewer than 2 bins are kept).
- `summary`: `kind` plus `axis_order` (`none`: `{"aperture_cfr": [bs,
  hemisphere, row, col, frequency_offset]}`; `power`: `{"element_power":
  ["bs","hemisphere","row","col"], "hemisphere_power": ["bs","hemisphere"]}`;
  `delay`: `{}`). Per-file paths live in `views[]`, not here.
- `views[]`: `view_id`, `source_index` (index into the source `views[]`),
  `position_m`, `look_at_m`, `orientation_rad`, `artifacts` (view level), and
  `bs`: one entry per BS in BS order with `bs_id`, `bs_index`,
  `bs_direction_local`, `bs_in_front_hemisphere` (from the reader's
  `ViewBSEntry`) and `artifacts` (`{}` unless the summary is `delay`).
- `config`: a copy of the source config; `carrier_frequency_hz`;
  `hemispheres`; `base_stations`: `[{bs_id, index, position_m, look_at_m}]`
  (for v2 the reader synthesises `bs_000` from `tx_position`/`tx_look_at`).
- `camera_model_source`: source camera model path relative to `OUT_DIR`.
- `path_geometry_gt`: `None`, or `{artifact, path_schema, axis_order,
  view_index: "source_index"}` from the reader's `PathGeometryGT`. `artifact`
  and `path_schema` are relative to `OUT_DIR`; `path_schema` is `None` for
  datasets written before the canonical path schema (v2, or an older v3).
  `axis_order` is the stored axis order of `tau` (from `path_schema.json`, or
  the reader's legacy fallback); see `path_schema.json` for the other arrays.
- `raw_observation_axis_order` (only for `summary=none`): the reader's
  `APERTURE_CFR_AXIS_ORDER` as a list.

Every path in the manifest is relative; `(OUT_DIR / path)` exists.

## Pairing a partial sample with its full sample

- `source_dataset` points back at the source dataset directory.
- Each partial `views[]` entry's `source_index` is the index of its source view
  in the source manifest's `views[]`; the view id is preserved as `view_id`.
- The leading path-GT `view` axis is indexed by the same `source_index`
  (`path_geometry_gt.view_index == "source_index"`).
- Within a view, `bs_id` (e.g. `bs_000`) selects the BS slice of the source
  `aperture_cfr.npy` (`[bs, hemisphere, row, col, freq]`) and the matching
  partial artifact.

## Delay summary definition

For one (view, BS) pair, each kept front-hemisphere element CFR is IFFTed to a
delay profile, profiles are summed **incoherently** (power) over elements, and
the argmax gives the delay modulo `unambiguous_delay_s`. The reported delay
lies on the subband delay grid `k / (n_sub * delta_f)`: for an isolated
arrival it is within half a delay bin of its true delay (modulo the period);
closely spaced arrivals can merge or shift the peak. The JSON stores `delay_s`, `power`,
`delay_resolution_s`, `unambiguous_delay_s` and `valid`. `valid` is about
signal, not geometry: it only means "non-zero front-hemisphere signal"
(`power > 0` and finite), while `bs_in_front_hemisphere` is the geometric LoS
direction. A BS behind the camera can still give `valid: true` through a
front-hemisphere reflection, and a BS in front can give `valid: false` when
nothing reaches the front hemisphere. When not valid (e.g. the front CFR is
all zeros), `delay_s` is `null` in JSON (`NaN` in the domain dict) and the file
is written with `allow_nan=False` (strict JSON).

## Seeds and rounding

The view draw uses `np.random.default_rng([seed, VIEW_STREAM])` (`VIEW_STREAM
= 0`) and the random mask uses `np.random.default_rng([seed, MASK_STREAM])`
(`MASK_STREAM = 1`), so the two draws never share a generator state for the
same seed. Kept counts use round-half-up (`floor(fraction * n + 0.5)`, at
least 1), not Python's banker's `round()`.

## Safety rules

- **Source protection:** `dataset_dir` and `out_dir` are resolved
  (symlinks followed); the build refuses when they are equal, when `out_dir`
  is inside `dataset_dir`, or when `dataset_dir` is inside `out_dir`. The
  check runs before anything is created or written, so a rejected run leaves
  the source manifest and every source `.npy` byte-identical.
- **Reruns:** all validation (manifest load, source/out-dir check, options, subband, the
  `delay`-needs-2-bins rule, output check) runs before any write, and
  `partial_manifest.json` is written last. A non-empty `OUT_DIR` fails with
  `FileExistsError` unless `--overwrite` is given; with `--overwrite` only a
  directory whose top-level entries are all among `partial_manifest.json`,
  `element_mask.npy`, `views` is replaced (exactly those entries are removed;
  `OUT_DIR` itself is never removed), anything else fails naming the
  unexpected entries.
