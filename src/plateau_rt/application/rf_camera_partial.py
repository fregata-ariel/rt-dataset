"""Derive a partial/summary RF-camera dataset from a multi-view dataset (CPU only).

Reads a dataset written by ``rf-camera-multiview`` (``dataset_manifest.json``,
schema v3 or v2) via
:func:`plateau_rt.application.rf_dataset_manifest.load_rf_dataset_manifest`,
keeps a subset of views, applies a contiguous frequency subband and an
aperture element mask, and optionally replaces the raw complex CFR with a
per-(view, BS) summary (element power, or one dominant delay).

This module must not import Sionna: everything runs on stored ``.npy`` files.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from plateau_rt.application.rf_dataset_manifest import (
    APERTURE_CFR_AXIS_ORDER,
    ViewBSEntry,
    load_rf_dataset_manifest,
)
from plateau_rt.domain.rf_camera.imaging import uniform_frequency_spacing
from plateau_rt.domain.rf_camera.partial import (
    SUMMARY_KINDS,
    apply_element_mask,
    element_mask,
    element_power,
    hemisphere_total_power,
    parse_subband,
    select_subband,
    select_views,
    view_dominant_delay,
)

# Top-level entries a previous partial run may have left behind. ``overwrite``
# only removes exactly these; anything else aborts with FileExistsError.
PARTIAL_OUT_ENTRIES = frozenset({"partial_manifest.json", "element_mask.npy", "views"})


def _relative_to_out(target: Path, out_resolved: Path) -> str:
    """Return ``target`` as a POSIX path relative to the resolved ``out_dir``."""
    return Path(os.path.relpath(Path(target).resolve(), out_resolved)).as_posix()


def _bs_entry_payload(bs_entry: ViewBSEntry, artifacts: dict[str, str]) -> dict[str, Any]:
    """Return the partial-manifest ``bs[]`` entry for one (view, BS) pair."""
    return {
        "bs_id": bs_entry.bs_id,
        "bs_index": int(bs_entry.bs_index),
        "bs_direction_local": (
            list(bs_entry.bs_direction_local) if bs_entry.bs_direction_local is not None else None
        ),
        "bs_in_front_hemisphere": bs_entry.bs_in_front_hemisphere,
        "artifacts": artifacts,
    }


def build_partial_dataset(
    dataset_dir: Path,
    out_dir: Path,
    *,
    view_fraction: float = 1.0,
    element_mask_kind: str = "none",
    mask_fraction: float = 0.5,
    subband: str | None = None,
    summary: str = "none",
    seed: int = 0,
    overwrite: bool = False,
) -> Path:
    """Build a derived partial/summary dataset and return ``partial_manifest.json``."""
    # Typed reader: accepts schema 3 and schema 2 (read as B = 1 with
    # ``bs_000``), raises ManifestError for anything else (e.g. schema 1).
    dataset = load_rf_dataset_manifest(Path(dataset_dir))

    # Never overwrite the source dataset. Resolve symlinks and refuse when
    # the two directories are equal or one contains the other. This runs before
    # anything is created or written (including mkdir).
    dataset_resolved = Path(dataset_dir).resolve()
    out_resolved = Path(out_dir).resolve()
    if (
        out_resolved == dataset_resolved
        or out_resolved.is_relative_to(dataset_resolved)
        or dataset_resolved.is_relative_to(out_resolved)
    ):
        raise ValueError(
            f"out_dir ({out_resolved}) must not equal dataset_dir ({dataset_resolved}) "
            "nor be nested inside it in either direction"
        )

    if summary not in SUMMARY_KINDS:
        raise ValueError(f"unknown summary kind: {summary!r}")

    rows, cols = int(dataset.rx_rows), int(dataset.rx_cols)
    full_offsets = np.asarray(dataset.frequency_offsets_hz, dtype=np.float64)
    num_bins = int(dataset.num_frequency_bins)

    kept_indices = select_views(dataset.num_views, view_fraction, seed)
    start, stop = parse_subband(subband, num_bins)
    mask = element_mask(rows, cols, element_mask_kind, fraction=mask_fraction, seed=seed)
    n_sub = stop - start
    if summary == "delay" and n_sub < 2:
        raise ValueError(
            f"summary 'delay' needs at least 2 subband bins, got [{start}:{stop}] ({n_sub})"
        )

    # TODO: allow choosing an rf-camera-observe variant (the per-pair
    # ``observed.<name>.aperture_cfr`` artifacts, [row, col, freq]) as input
    # instead of the ideal two-hemisphere aperture_cfr.
    kept_offsets = full_offsets[start:stop]
    kept_absolute = np.asarray(dataset.absolute_frequencies_hz, dtype=np.float64)[start:stop]
    # A single-bin source has no frequency spacing; leave the delay fields
    # empty (None) instead of failing, so non-delay summaries still work.
    delta_f: float | None = uniform_frequency_spacing(full_offsets) if num_bins >= 2 else None
    if n_sub >= 2 and delta_f is not None:
        delay_resolution_s: float | None = 1.0 / (n_sub * delta_f)
        unambiguous_delay_s: float | None = 1.0 / delta_f
    else:
        delay_resolution_s = None
        unambiguous_delay_s = None

    # Link to the source path-level ground truth. Resolved here, before any
    # write, because reading the path schema can fail. ``axis_order`` is the
    # stored axis order of ``tau`` (from path_schema.json, or the reader's
    # legacy fallback); its leading view axis is indexed by each partial
    # view's ``source_index``.
    path_gt = dataset.path_geometry_gt
    if path_gt is None:
        path_geometry_gt: dict[str, Any] | None = None
    else:
        path_geometry_gt = {
            "artifact": _relative_to_out(path_gt.path, out_resolved),
            "path_schema": (
                None
                if path_gt.schema_path is None
                else _relative_to_out(path_gt.schema_path, out_resolved)
            ),
            "axis_order": list(path_gt.array_axes("tau")),
            "view_index": "source_index",
        }

    # Stale files on rerun. All validation happens before anything is
    # created or written; the out_dir state check is part of validation.
    out_path = Path(out_dir)
    out_non_empty = out_path.exists() and any(out_path.iterdir())
    if out_non_empty:
        if not overwrite:
            raise FileExistsError(
                f"out_dir {out_path} already exists and is not empty; "
                "pass overwrite=True to replace a previous partial output"
            )
        unexpected = sorted(
            entry.name for entry in out_path.iterdir() if entry.name not in PARTIAL_OUT_ENTRIES
        )
        if unexpected:
            raise FileExistsError(
                f"out_dir {out_path} contains unexpected entries {unexpected}; "
                "refusing to overwrite a directory that is not a previous partial output"
            )

    # All validation passed: with overwrite, remove exactly the previous
    # partial entries (never rmtree out_dir itself), then write.
    if out_non_empty and overwrite:
        for name in sorted(PARTIAL_OUT_ENTRIES):
            old = out_path / name
            if not old.exists() and not old.is_symlink():
                continue
            if old.is_dir() and not old.is_symlink():
                shutil.rmtree(old)
            else:
                old.unlink()
    out_path.mkdir(parents=True, exist_ok=True)
    mask_path = out_path / "element_mask.npy"
    np.save(mask_path, mask.astype(bool, copy=False))

    front_index = list(dataset.hemispheres).index("front")

    manifest_views: list[dict[str, Any]] = []
    for source_index in kept_indices.tolist():
        view_obj = dataset.views[int(source_index)]
        view_id = view_obj.view_id
        full_cfr = dataset.load_aperture_cfr(view_obj)
        partial_cfr, _ = select_subband(full_cfr, full_offsets, start, stop)
        partial_cfr = apply_element_mask(partial_cfr, mask)

        rf_dir = out_path / "views" / view_id / "rf"
        rf_dir.mkdir(parents=True, exist_ok=True)
        bs_entries: list[dict[str, Any]] = []
        if summary == "none":
            artifact_path = rf_dir / "aperture_cfr.npy"
            np.save(artifact_path, partial_cfr.astype(np.complex64, copy=False))
            artifacts = {"aperture_cfr": str(artifact_path.relative_to(out_path).as_posix())}
            for bs_entry in view_obj.bs:
                bs_entries.append(_bs_entry_payload(bs_entry, {}))
        elif summary == "power":
            power = element_power(partial_cfr)
            hemi = hemisphere_total_power(partial_cfr)
            power_path = rf_dir / "element_power.npy"
            hemi_path = rf_dir / "hemisphere_power.npy"
            np.save(power_path, power.astype(np.float32, copy=False))
            np.save(hemi_path, hemi.astype(np.float64, copy=False))
            artifacts = {
                "element_power": str(power_path.relative_to(out_path).as_posix()),
                "hemisphere_power": str(hemi_path.relative_to(out_path).as_posix()),
            }
            for bs_entry in view_obj.bs:
                bs_entries.append(_bs_entry_payload(bs_entry, {}))
        else:
            artifacts = {}
            for bs_entry in view_obj.bs:
                front = partial_cfr[bs_entry.bs_index, front_index]
                result = view_dominant_delay(front, kept_offsets, element_mask=mask)
                delay_path = rf_dir / bs_entry.bs_id / "dominant_delay.json"
                delay_path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "delay_s": None if not result["valid"] else float(result["delay_s"]),
                    "power": float(result["power"]),
                    "delay_resolution_s": float(result["delay_resolution_s"]),
                    "unambiguous_delay_s": float(result["unambiguous_delay_s"]),
                    "valid": bool(result["valid"]),
                }
                delay_path.write_text(
                    json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
                )
                bs_entries.append(
                    _bs_entry_payload(
                        bs_entry,
                        {"dominant_delay": str(delay_path.relative_to(out_path).as_posix())},
                    )
                )

        entry: dict[str, Any] = {
            "view_id": view_id,
            "source_index": int(source_index),
            "position_m": list(view_obj.position_m),
            "look_at_m": list(view_obj.look_at_m) if view_obj.look_at_m is not None else None,
            "orientation_rad": (
                list(view_obj.orientation_rad) if view_obj.orientation_rad is not None else None
            ),
            "artifacts": artifacts,
            "bs": bs_entries,
        }
        manifest_views.append(entry)

    if summary == "none":
        summary_axis_order: dict[str, list[str]] = {"aperture_cfr": list(APERTURE_CFR_AXIS_ORDER)}
    elif summary == "power":
        summary_axis_order = {
            "element_power": ["bs", "hemisphere", "row", "col"],
            "hemisphere_power": ["bs", "hemisphere"],
        }
    else:
        summary_axis_order = {}

    partial_manifest: dict[str, Any] = {
        "schema_version": 1,
        "mode": "rf_camera_partial_observation",
        "source_dataset": Path(os.path.relpath(dataset_resolved, out_resolved)).as_posix(),
        "source_manifest": _relative_to_out(dataset.manifest_path, out_resolved),
        "source_manifest_schema_version": int(dataset.schema_version),
        "options": {
            "view_fraction": float(view_fraction),
            "element_mask_kind": element_mask_kind,
            "mask_fraction": float(mask_fraction),
            "subband": subband,
            "summary": summary,
            "seed": int(seed),
        },
        "kept_view_indices": [int(i) for i in kept_indices.tolist()],
        "kept_view_ids": [v["view_id"] for v in manifest_views],
        "element_mask": {
            "kind": element_mask_kind,
            "fraction": float(mask_fraction),
            "seed": int(seed),
            "file": str(mask_path.relative_to(out_path).as_posix()),
            "kept_count": int(mask.sum()),
            "total": int(mask.size),
        },
        "subband": {
            "start": start,
            "stop": stop,
            "num_bins": int(n_sub),
            "frequency_offsets_hz": kept_offsets.tolist(),
            "absolute_frequencies_hz": kept_absolute.tolist(),
            "delay_resolution_s": delay_resolution_s,
            "unambiguous_delay_s": unambiguous_delay_s,
        },
        "summary": {
            "kind": summary,
            "axis_order": summary_axis_order,
        },
        "views": manifest_views,
        "config": dict(dataset.config),
        "carrier_frequency_hz": float(dataset.carrier_frequency_hz),
        "hemispheres": list(dataset.hemispheres),
        "base_stations": [
            {
                "bs_id": bs.bs_id,
                "index": int(bs.index),
                "position_m": list(bs.position_m),
                "look_at_m": list(bs.look_at_m),
            }
            for bs in dataset.base_stations
        ],
        "camera_model_source": _relative_to_out(dataset.camera_model_path, out_resolved),
        "path_geometry_gt": path_geometry_gt,
    }
    if summary == "none":
        partial_manifest["raw_observation_axis_order"] = list(APERTURE_CFR_AXIS_ORDER)

    out_manifest_path = out_path / "partial_manifest.json"
    out_manifest_path.write_text(json.dumps(partial_manifest, indent=2), encoding="utf-8")
    return out_manifest_path
