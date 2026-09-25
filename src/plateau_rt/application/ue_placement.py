"""Sionna-free radio-map persistence and manifest glue for coverage placement.

This module saves and loads the 2D path-gain map computed by
:mod:`plateau_rt.adapters.sionna.radio_map` as a content-addressed artifact and
builds the manifest ``placement`` section for the RF-camera dataset. Because
Sionna GPU tracing is not bit-reproducible, ``saved map + placement_seed`` is
the reproducibility contract: reusing the saved float32 arrays through
:func:`replan_from_manifest` reproduces the poses exactly.

It must stay importable without Sionna, Mitsuba, Dr.Jit or Matplotlib (see
``tests/test_rf_camera_boundaries.py``): only NumPy, the standard library and
the NumPy-only ``plateau_rt.domain`` / manifest reader are used.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from plateau_rt.application.rf_dataset_manifest import load_rf_dataset_manifest
from plateau_rt.domain.rf_camera.placement import (
    CoveragePlacement,
    CoveragePlacementSettings,
    RadioMapGrid,
    dilate_mask,
    plan_coverage_placement,
)

PLACEMENT_DIR = "placement"
RADIO_MAP_METADATA_FILE = "radio_map.json"
RADIO_MAP_PATH_GAIN_FILE = "radio_map_path_gain.npy"
RADIO_MAP_INDOOR_MASK_FILE = "radio_map_indoor_mask.npy"
RADIO_MAP_FORMAT_VERSION = 1


@dataclass(frozen=True)
class SavedRadioMap:
    """A radio map loaded from disk together with its parsed metadata."""

    path_gain: np.ndarray
    indoor_mask: np.ndarray
    grid: RadioMapGrid
    metadata: Mapping[str, Any]
    metadata_path: Path


def file_sha256(path: Path) -> str:
    """Return the hex sha256 digest of the bytes of ``path``."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_radio_map_arrays(
    path_gain: np.ndarray,
    indoor_mask: np.ndarray,
    grid: RadioMapGrid,
    num_base_stations: int,
) -> None:
    """Check the radio-map array shapes against the grid (ValueError otherwise)."""
    gain = np.asarray(path_gain)
    mask = np.asarray(indoor_mask)
    if gain.ndim != 3:
        raise ValueError(f"path_gain must have shape [num_bs, ny, nx], got shape {gain.shape}")
    if tuple(gain.shape[1:]) != grid.shape:
        raise ValueError(
            f"path_gain spatial shape {tuple(gain.shape[1:])} does not match "
            f"grid shape {grid.shape}"
        )
    if int(gain.shape[0]) != num_base_stations:
        raise ValueError(
            f"path_gain has {int(gain.shape[0])} base-station slices, "
            f"but {num_base_stations} base stations were given"
        )
    if mask.ndim != 2 or tuple(mask.shape) != grid.shape:
        raise ValueError(f"indoor_mask shape {mask.shape} does not match grid shape {grid.shape}")


def save_radio_map(
    output_dir: Path,
    *,
    path_gain: np.ndarray,
    indoor_mask: np.ndarray,
    grid: RadioMapGrid,
    solver: Mapping[str, Any],
    base_stations: Sequence[Mapping[str, Any]],
    carrier_frequency_hz: float,
    source_scene: str,
) -> Path:
    """Write a radio map under ``output_dir/placement/`` and return its json path.

    ``base_stations`` entries are ``{"bs_id", "position_m", "look_at_m"}`` in
    transmitter order. ``np.save`` output is byte-deterministic, so the
    recorded sha256 digests identify the stored array contents.
    """
    output_dir = Path(output_dir)
    grid.validate()
    _validate_radio_map_arrays(path_gain, indoor_mask, grid, len(base_stations))

    placement_dir = output_dir / PLACEMENT_DIR
    placement_dir.mkdir(parents=True, exist_ok=True)
    path_gain_path = placement_dir / RADIO_MAP_PATH_GAIN_FILE
    indoor_mask_path = placement_dir / RADIO_MAP_INDOOR_MASK_FILE
    np.save(path_gain_path, np.asarray(path_gain, dtype=np.float32))
    np.save(indoor_mask_path, np.asarray(indoor_mask, dtype=bool))

    metadata = {
        "format_version": RADIO_MAP_FORMAT_VERSION,
        "source_scene": str(source_scene),
        "carrier_frequency_hz": float(carrier_frequency_hz),
        "base_stations": [dict(entry) for entry in base_stations],
        "grid": grid.to_dict(),
        "solver": dict(solver),
        "path_gain_axis_order": ["bs", "y", "x"],
        "path_gain_scale": "linear",
        "artifacts": {
            "path_gain": RADIO_MAP_PATH_GAIN_FILE,
            "indoor_mask": RADIO_MAP_INDOOR_MASK_FILE,
        },
        "sha256": {
            "path_gain": file_sha256(path_gain_path),
            "indoor_mask": file_sha256(indoor_mask_path),
        },
    }
    metadata_path = placement_dir / RADIO_MAP_METADATA_FILE
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata_path


def _resolve_metadata_path(path: Path) -> Path:
    """Resolve ``path`` (json, placement dir or dataset dir) to a radio_map.json."""
    location = Path(path)
    if location.is_file():
        return location
    if location.is_dir():
        direct = location / RADIO_MAP_METADATA_FILE
        if direct.is_file():
            return direct
        nested = location / PLACEMENT_DIR / RADIO_MAP_METADATA_FILE
        if nested.is_file():
            return nested
    raise ValueError(
        f"no {RADIO_MAP_METADATA_FILE} found at {location} "
        f"(expected a json file, a placement/ directory or a dataset directory)"
    )


def load_radio_map(path: Path) -> SavedRadioMap:
    """Load and verify a saved radio map.

    ``path`` may be the ``radio_map.json`` file, a ``placement/`` directory
    containing it, or a dataset directory containing
    ``placement/radio_map.json``. Raises ValueError when the format version,
    the artifact sha256 digests or the array shapes are wrong.
    """
    metadata_path = _resolve_metadata_path(path)
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(
            f"radio map metadata could not be read from {metadata_path}: {exc}"
        ) from exc
    if not isinstance(metadata, Mapping):
        raise ValueError(f"radio map metadata {metadata_path} must be a JSON object")
    if metadata.get("format_version") != RADIO_MAP_FORMAT_VERSION:
        raise ValueError(
            f"unsupported radio map format_version {metadata.get('format_version')!r} in "
            f"{metadata_path}; expected {RADIO_MAP_FORMAT_VERSION}"
        )
    try:
        grid_record = metadata["grid"]
        artifacts = metadata["artifacts"]
        recorded = metadata["sha256"]
    except KeyError as exc:
        raise ValueError(f"radio map metadata {metadata_path} misses key {exc}") from None
    grid = RadioMapGrid.from_dict(grid_record)

    placement_dir = metadata_path.parent
    resolved: dict[str, Path] = {}
    for key in ("path_gain", "indoor_mask"):
        try:
            relative = artifacts[key]
        except (KeyError, TypeError):
            raise ValueError(
                f"radio map metadata {metadata_path} misses artifact {key!r}"
            ) from None
        if not isinstance(relative, str):
            raise ValueError(f"radio map artifact {key!r} must be a relative path string")
        artifact_path = placement_dir / relative
        actual = file_sha256(artifact_path)
        expected = recorded.get(key) if isinstance(recorded, Mapping) else None
        if actual != expected:
            raise ValueError(
                f"radio map artifact {artifact_path} sha256 mismatch: "
                f"recorded {expected!r}, actual {actual!r}"
            )
        resolved[key] = artifact_path

    path_gain = np.load(resolved["path_gain"], allow_pickle=False)
    indoor_mask = np.load(resolved["indoor_mask"], allow_pickle=False)
    path_gain = np.asarray(path_gain, dtype=np.float32)
    indoor_mask = np.asarray(indoor_mask, dtype=bool)
    _validate_radio_map_arrays(path_gain, indoor_mask, grid, int(path_gain.shape[0]))
    return SavedRadioMap(
        path_gain=path_gain,
        indoor_mask=indoor_mask,
        grid=grid,
        metadata=metadata,
        metadata_path=metadata_path,
    )


def copy_radio_map(saved: SavedRadioMap, output_dir: Path) -> Path:
    """Copy a saved radio map into ``output_dir/placement/`` byte-for-byte.

    Returns the copied json path. When ``saved`` already lives in that
    directory the source json path is returned unchanged.
    """
    output_dir = Path(output_dir)
    destination_dir = output_dir / PLACEMENT_DIR
    source_dir = saved.metadata_path.parent
    if source_dir.resolve() == destination_dir.resolve():
        return saved.metadata_path
    destination_dir.mkdir(parents=True, exist_ok=True)
    for name in (RADIO_MAP_METADATA_FILE, RADIO_MAP_PATH_GAIN_FILE, RADIO_MAP_INDOOR_MASK_FILE):
        shutil.copyfile(source_dir / name, destination_dir / name)
    return destination_dir / RADIO_MAP_METADATA_FILE


def check_radio_map_matches(
    saved: SavedRadioMap,
    *,
    carrier_frequency_hz: float,
    base_stations: Sequence[tuple[str, tuple[float, float, float], tuple[float, float, float]]],
    ue_height_m: float,
) -> None:
    """Check that a saved radio map fits the requested dataset geometry.

    Raises ValueError when the carrier, the base stations or the UE height
    differ from the saved map. ``base_stations`` is the output of
    :meth:`RFMultiViewConfig.resolve_base_stations`.
    """
    metadata = saved.metadata
    recorded_carrier = float(metadata.get("carrier_frequency_hz", float("nan")))
    if not math.isclose(recorded_carrier, float(carrier_frequency_hz), rel_tol=1e-9):
        raise ValueError(
            f"radio map carrier {recorded_carrier!r} does not match the requested "
            f"{float(carrier_frequency_hz)!r}"
        )
    recorded_bs = metadata.get("base_stations")
    if not isinstance(recorded_bs, (list, tuple)):
        raise ValueError("radio map metadata has no 'base_stations' list")
    if len(recorded_bs) != len(base_stations):
        raise ValueError(
            f"radio map has {len(recorded_bs)} base stations, but "
            f"{len(base_stations)} were requested"
        )
    for index, (bs_id, position, look_at) in enumerate(base_stations):
        entry = recorded_bs[index]
        if not isinstance(entry, Mapping) or entry.get("bs_id") != bs_id:
            raise ValueError(
                f"radio map base station {index} is not {bs_id!r} "
                f"(got {entry.get('bs_id') if isinstance(entry, Mapping) else entry!r})"
            )
        for name, wanted in (("position_m", position), ("look_at_m", look_at)):
            recorded = entry.get(name)
            if not isinstance(recorded, (list, tuple)) or len(recorded) != 3:
                raise ValueError(f"radio map base station {bs_id!r} has no valid {name!r}")
            for axis, (got, want) in enumerate(zip(recorded, wanted, strict=True)):
                if abs(float(got) - float(want)) > 1e-9:
                    raise ValueError(
                        f"radio map base station {bs_id!r} {name}[{axis}]={float(got)!r} "
                        f"does not match the requested {float(want)!r}"
                    )
    height = float(saved.grid.center_m[2])
    if abs(height - float(ue_height_m)) > 1e-9:
        raise ValueError(
            f"radio map UE height {height!r} does not match the requested {float(ue_height_m)!r}"
        )


def building_exclusion_mask(
    indoor_mask: np.ndarray, grid: RadioMapGrid, *, clearance_m: float
) -> np.ndarray:
    """Return the dilated building-interior mask (``clearance_m`` >= 0)."""
    return dilate_mask(indoor_mask, grid, clearance_m)


def placement_manifest_section(
    placement: CoveragePlacement,
    *,
    saved: SavedRadioMap,
    dataset_dir: Path,
    radio_map_source: str,
    radio_map_origin: str | None,
    building_clearance_m: float,
) -> dict[str, Any]:
    """Build the manifest ``placement`` section from a planned placement."""
    dataset_dir = Path(dataset_dir)
    metadata_relative = saved.metadata_path.resolve().relative_to(dataset_dir.resolve()).as_posix()
    settings = placement.settings
    record = placement.to_record()
    record["exclusion"] = {
        "indoor_mask": "upward_ray_hits_geometry",
        "building_clearance_m": float(building_clearance_m),
        "invalid_cells": "aggregated path gain <= 0 or non-finite",
        "min_bs_distance_m": float(settings.min_bs_distance_m),
        "min_ue_spacing_m": float(settings.min_spacing_m),
    }
    record["radio_map"] = {
        "source": radio_map_source,
        "origin": radio_map_origin,
        "metadata": metadata_relative,
        "artifacts": {
            "path_gain": f"{PLACEMENT_DIR}/{RADIO_MAP_PATH_GAIN_FILE}",
            "indoor_mask": f"{PLACEMENT_DIR}/{RADIO_MAP_INDOOR_MASK_FILE}",
        },
        "sha256": dict(saved.metadata["sha256"]),
        "grid": saved.grid.to_dict(),
        "solver": dict(saved.metadata["solver"]),
    }
    return record


def replan_from_manifest(dataset_dir: Path) -> CoveragePlacement:
    """Rebuild the coverage placement recorded in a dataset manifest.

    Loads the saved radio map referenced by the manifest, rebuilds the settings
    and exclusion mask, and re-runs the deterministic placement. This is the
    "saved map + placement_seed -> identical poses" contract.
    """
    dataset_dir = Path(dataset_dir)
    manifest = load_rf_dataset_manifest(dataset_dir)
    placement = manifest.placement
    if not isinstance(placement, Mapping) or placement.get("method") != "coverage":
        raise ValueError(f"{dataset_dir}/dataset_manifest.json has no coverage placement section")
    try:
        radio_map_record = placement["radio_map"]
        metadata_relative = radio_map_record["metadata"]
        settings = CoveragePlacementSettings.from_dict(placement["settings"])
        clearance = float(placement["exclusion"]["building_clearance_m"])
    except (KeyError, TypeError) as exc:
        raise ValueError(f"malformed placement section in {dataset_dir}: {exc}") from None
    saved = load_radio_map(dataset_dir / metadata_relative)
    exclusion = building_exclusion_mask(saved.indoor_mask, saved.grid, clearance_m=clearance)
    bs_positions = [tuple(bs.position_m) for bs in manifest.base_stations]
    return plan_coverage_placement(
        saved.path_gain,
        saved.grid,
        settings,
        exclusion_mask=exclusion,
        bs_positions=bs_positions,
    )
