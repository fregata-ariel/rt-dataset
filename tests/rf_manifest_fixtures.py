"""Synthetic RF-camera datasets for manifest reader tests.

Writes small, self-consistent schema v2/v3 datasets (manifest plus the
payload files the typed reader resolves) with deterministic random aperture
CFRs. This is a plain helper module, not a test file, and will be reused by
other test files later.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from plateau_rt.domain.rf_camera.calibration import geometric_los_source_direction_local
from plateau_rt.domain.rf_camera.camera import (
    HEMISPHERES,
    IMAGE_QUANTITY,
    PROJECTION,
    RFViewSpec,
    build_direction_cosine_camera_model,
    generate_ring_views,
    view_pose_payload,
)
from plateau_rt.domain.rf_camera.imaging import frequency_offsets
from plateau_rt.domain.rf_camera.paths import (
    PATH_GEOMETRY_GT_FILE_NAME,
    PATH_GT_MODE_CANONICAL,
    PATH_SCHEMA_FILE_NAME,
    build_path_schema,
)

CARRIER_HZ = 3.5e9
BANDWIDTH_HZ = 100e6
TARGET_M = (5.0, 5.0, 5.0)
BS_POSITIONS_M = ((-50.0, -50.0, 30.0), (60.0, 35.0, 25.0))
FFT_ROWS = FFT_COLS = 16


def centred_frequency_offsets(num_bins: int, bandwidth_hz: float = BANDWIDTH_HZ) -> np.ndarray:
    """Return the dataset writer's baseband grid (float32 values, as stored in the manifest)."""
    return frequency_offsets(bandwidth_hz, num_bins)


def _resolve_bs_positions(num_bs: int) -> list[tuple[float, float, float]]:
    """Return ``num_bs`` deterministic BS positions (first two match the mock)."""
    positions = list(BS_POSITIONS_M)
    while len(positions) < num_bs:
        index = len(positions)
        positions.append((float(20 * index), float(-30 + 10 * index), 25.0))
    return positions[:num_bs]


def _resolve_views(views: Sequence[RFViewSpec] | None, num_views: int) -> list[RFViewSpec]:
    """Return ``views`` or, when None, ``num_views`` mock-like ring views around the target."""
    if views is not None:
        return list(views)
    return generate_ring_views(target=TARGET_M, radius_m=30.0, ue_height_m=1.5, num_views=num_views)


def _random_aperture(shape: tuple[int, ...], seed: int) -> np.ndarray:
    """Return a deterministic random complex64 aperture CFR."""
    rng = np.random.default_rng(seed)
    real = rng.standard_normal(shape)
    imag = rng.standard_normal(shape)
    return (real + 1j * imag).astype(np.complex64)


def _hemisphere_energy(aperture_bs: np.ndarray) -> dict[str, float]:
    """Sum ``|aperture|^2`` over elements and frequencies per hemisphere."""
    return {
        name: float(np.sum(np.abs(aperture_bs[HEMISPHERES.index(name)]) ** 2))
        for name in HEMISPHERES
    }


def _write_camera_model(root: Path) -> None:
    """Write ``camera_model.npz`` with a 16x16 direction-cosine grid."""
    model = build_direction_cosine_camera_model(
        fft_rows=FFT_ROWS,
        fft_cols=FFT_COLS,
        horizontal_spacing_lambda=0.5,
        vertical_spacing_lambda=0.5,
    )
    np.savez_compressed(root / "camera_model.npz", **model)


def _write_path_geometry_gt(
    root: Path,
    *,
    num_views: int,
    num_bs: int,
    rows: int,
    cols: int,
    bs_ids: Sequence[str],
    view_ids: Sequence[str],
    write_schema: bool,
) -> None:
    """Write a small canonical ``path_geometry_gt.npz`` and, optionally, its schema."""
    arrays = {
        "valid": np.ones((num_views, num_bs, 2), dtype=bool),
        "tau": np.zeros((num_views, num_bs, 2), dtype=np.float32),
        "a_baseband": np.zeros((num_views, num_bs, 2, rows, cols, 2), dtype=np.complex64),
        "num_interactions": np.zeros((num_views, num_bs, 2), dtype=np.int32),
    }
    np.savez_compressed(root / PATH_GEOMETRY_GT_FILE_NAME, **arrays)
    if not write_schema:
        return
    schema = build_path_schema(
        arrays,
        mode=PATH_GT_MODE_CANONICAL,
        object_names=["mock_building"],
        carrier_frequency_hz=CARRIER_HZ,
        bs_ids=bs_ids,
        view_ids=view_ids,
    )
    (root / PATH_SCHEMA_FILE_NAME).write_text(json.dumps(schema, indent=2), encoding="utf-8")


def _write_placeholder_npy(path: Path) -> None:
    """Write a tiny placeholder for a derived per-BS artifact file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.zeros((2,), dtype=np.float32))


def _bs_geometry(bs_position: tuple[float, float, float], *, view: Any) -> tuple[list[float], bool]:
    """Return the geometric BS direction (UE-local) and front-hemisphere flag."""
    bs_local = geometric_los_source_direction_local(
        tx_position=bs_position,
        ue_position=view.position,
        ue_orientation=view.orientation,
    )
    return [float(v) for v in bs_local], bool(bs_local[0] >= 0.0)


def write_v3_dataset(
    root: Path,
    *,
    num_views: int = 2,
    num_bs: int = 2,
    rows: int = 2,
    cols: int = 3,
    bins: int = 4,
    seed: int = 0,
    views: Sequence[RFViewSpec] | None = None,
    source_scene: str = "mock_scene.xml",
    bs_positions: Sequence[Sequence[float]] | None = None,
    bs_look_at: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Write a synthetic schema v3 dataset under ``root`` and return its manifest."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    offsets = centred_frequency_offsets(bins)
    if bs_positions is None:
        positions: list[tuple[float, float, float]] = _resolve_bs_positions(num_bs)
    else:
        positions = [(float(p[0]), float(p[1]), float(p[2])) for p in bs_positions]
        num_bs = len(positions)
    look_at: tuple[float, float, float] = (
        (float(bs_look_at[0]), float(bs_look_at[1]), float(bs_look_at[2]))
        if bs_look_at is not None
        else TARGET_M
    )
    bs_ids = [f"bs_{index:03d}" for index in range(num_bs)]
    views = _resolve_views(views, num_views)
    num_views = len(views)

    _write_camera_model(root)
    _write_path_geometry_gt(
        root,
        num_views=num_views,
        num_bs=num_bs,
        rows=rows,
        cols=cols,
        bs_ids=bs_ids,
        view_ids=[view.view_id for view in views],
        write_schema=True,
    )

    manifest_views: list[dict[str, Any]] = []
    for view_index, view in enumerate(views):
        view_dir = root / "views" / view.view_id
        rf_dir = view_dir / "rf"
        rf_dir.mkdir(parents=True, exist_ok=True)
        (view_dir / "pose.json").write_text(
            json.dumps(view_pose_payload(view), indent=2), encoding="utf-8"
        )
        aperture = _random_aperture((num_bs, 2, rows, cols, bins), seed + view_index)
        np.save(rf_dir / "aperture_cfr.npy", aperture)

        bs_entries: list[dict[str, Any]] = []
        for bs_index, (bs_id, bs_position) in enumerate(zip(bs_ids, positions)):
            direction, in_front = _bs_geometry(bs_position, view=view)
            artifacts = {
                "angular_cfr_center": f"views/{view.view_id}/rf/{bs_id}/angular_cfr_center.npy",
                "angular_power_center": f"views/{view.view_id}/rf/{bs_id}/angular_power_center.npy",
                "phase_valid_mask": f"views/{view.view_id}/rf/{bs_id}/phase_valid_mask.npy",
                "dominant_delay_s": f"views/{view.view_id}/rf/{bs_id}/dominant_delay_s.npy",
                "dominant_delay_power": (
                    f"views/{view.view_id}/rf/{bs_id}/dominant_delay_power.npy"
                ),
                "debug_power_png": f"views/{view.view_id}/rf/{bs_id}/angular_power_center.png",
            }
            for name, relative in artifacts.items():
                if relative.endswith(".npy"):
                    _write_placeholder_npy(root / relative)
            bs_entries.append(
                {
                    "bs_id": bs_id,
                    "bs_direction_local": direction,
                    "bs_in_front_hemisphere": in_front,
                    "hemisphere_energy": _hemisphere_energy(aperture[bs_index]),
                    "artifacts": artifacts,
                }
            )
        manifest_views.append(
            {
                "view_id": view.view_id,
                "position_m": list(view.position),
                "look_at_m": list(view.look_at),
                "orientation_rad": list(view.orientation),
                "artifacts": {
                    "pose": f"views/{view.view_id}/pose.json",
                    "aperture_cfr": f"views/{view.view_id}/rf/aperture_cfr.npy",
                },
                "bs": bs_entries,
            }
        )

    manifest = {
        "schema_version": 3,
        "mode": "multibs_multiue_rf_camera_dataset",
        "source_scene": source_scene,
        "config": {
            "carrier_frequency_hz": CARRIER_HZ,
            "bandwidth_hz": BANDWIDTH_HZ,
            "num_frequency_bins": bins,
            "tx_positions": [list(position) for position in positions],
            "tx_look_at": list(look_at),
            "tx_look_ats": None,
            "rx_rows": rows,
            "rx_cols": cols,
            "vertical_spacing_lambda": 0.5,
            "horizontal_spacing_lambda": 0.5,
            "tx_pattern": "tr38901",
            "polarization": "V",
            "fft_rows": FFT_ROWS,
            "fft_cols": FFT_COLS,
            "phase_floor_db": -35.0,
            "max_depth": 5,
            "synthetic_array": True,
            "seed": seed,
        },
        "frequency_offsets_hz": offsets.tolist(),
        "absolute_frequencies_hz": (CARRIER_HZ + offsets).tolist(),
        "delay_resolution_s": 1.0 / BANDWIDTH_HZ,
        "unambiguous_delay_s": bins / BANDWIDTH_HZ,
        "base_stations": [
            {
                "bs_id": bs_id,
                "index": index,
                "position_m": list(position),
                "look_at_m": list(look_at),
            }
            for index, (bs_id, position) in enumerate(zip(bs_ids, positions))
        ],
        "raw_observation": {
            "artifact": "aperture_cfr",
            "axis_order": ["bs", "hemisphere", "row", "col", "frequency_offset"],
            "bs_ids": list(bs_ids),
            "hemispheres": list(HEMISPHERES),
            "rx_element_pattern": "rf_camera_split",
            "note": (
                "Front (local kx >= 0) and back (kx < 0) arrivals of a vertically "
                "polarized isotropic element; front + back equals the isotropic "
                "element. A finite front-to-back ratio g can be synthesized as "
                "front + g * back."
            ),
        },
        "camera_model": {
            "projection": PROJECTION,
            "forward_axis_local": "+x",
            "array_plane_local": "y-z",
            "ray_directions": "camera_model.npz",
            "developed_hemisphere": HEMISPHERES[0],
            "image_quantity": IMAGE_QUANTITY,
            "image_definition": (
                "A(ky, kz) = kx * U(ky, kz) with kx = sqrt(1 - ky^2 - kz^2): U is the "
                "calibrated angular spectrum of the front-hemisphere aperture CFR, A the "
                "complex amplitude per unit solid angle (camera_model.npz "
                "solid_angle_weight). Back-hemisphere arrivals are excluded, like light "
                "behind an optical camera."
            ),
        },
        "path_geometry_gt": PATH_GEOMETRY_GT_FILE_NAME,
        "path_schema": PATH_SCHEMA_FILE_NAME,
        "views": manifest_views,
    }
    (root / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def write_v2_dataset(
    root: Path,
    *,
    num_views: int = 2,
    rows: int = 2,
    cols: int = 3,
    bins: int = 4,
    seed: int = 0,
    views: Sequence[RFViewSpec] | None = None,
    source_scene: str = "mock_scene.xml",
) -> dict[str, Any]:
    """Write a synthetic schema v2 (single-BS) dataset under ``root``."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    offsets = centred_frequency_offsets(bins)
    bs_position = BS_POSITIONS_M[0]
    views = _resolve_views(views, num_views)
    num_views = len(views)

    _write_camera_model(root)
    _write_path_geometry_gt(
        root,
        num_views=num_views,
        num_bs=1,
        rows=rows,
        cols=cols,
        bs_ids=["bs_000"],
        view_ids=[view.view_id for view in views],
        write_schema=False,
    )

    manifest_views: list[dict[str, Any]] = []
    for view_index, view in enumerate(views):
        view_dir = root / "views" / view.view_id
        rf_dir = view_dir / "rf"
        rf_dir.mkdir(parents=True, exist_ok=True)
        (view_dir / "pose.json").write_text(
            json.dumps(view_pose_payload(view), indent=2), encoding="utf-8"
        )
        aperture = _random_aperture((2, rows, cols, bins), seed + view_index)
        np.save(rf_dir / "aperture_cfr.npy", aperture)
        direction, in_front = _bs_geometry(bs_position, view=view)
        artifacts = {
            "pose": f"views/{view.view_id}/pose.json",
            "aperture_cfr": f"views/{view.view_id}/rf/aperture_cfr.npy",
            "angular_cfr_center": f"views/{view.view_id}/rf/angular_cfr_center.npy",
            "angular_power_center": f"views/{view.view_id}/rf/angular_power_center.npy",
            "phase_valid_mask": f"views/{view.view_id}/rf/phase_valid_mask.npy",
            "dominant_delay_s": f"views/{view.view_id}/rf/dominant_delay_s.npy",
            "dominant_delay_power": f"views/{view.view_id}/rf/dominant_delay_power.npy",
            "debug_power_png": f"views/{view.view_id}/rf/angular_power_center.png",
        }
        for name, relative in artifacts.items():
            if relative.endswith(".npy") and name not in ("aperture_cfr",):
                _write_placeholder_npy(root / relative)
        manifest_views.append(
            {
                "view_id": view.view_id,
                "position_m": list(view.position),
                "look_at_m": list(view.look_at),
                "orientation_rad": list(view.orientation),
                "bs_direction_local": direction,
                "bs_in_front_hemisphere": in_front,
                "hemisphere_energy": _hemisphere_energy(aperture),
                "artifacts": artifacts,
            }
        )

    manifest = {
        "schema_version": 2,
        "mode": "1bs_multiue_rf_camera_dataset",
        "source_scene": source_scene,
        "config": {
            "carrier_frequency_hz": CARRIER_HZ,
            "bandwidth_hz": BANDWIDTH_HZ,
            "num_frequency_bins": bins,
            "tx_position": list(bs_position),
            "tx_look_at": list(TARGET_M),
            "rx_rows": rows,
            "rx_cols": cols,
            "vertical_spacing_lambda": 0.5,
            "horizontal_spacing_lambda": 0.5,
            "tx_pattern": "tr38901",
            "polarization": "V",
            "fft_rows": FFT_ROWS,
            "fft_cols": FFT_COLS,
            "phase_floor_db": -35.0,
            "max_depth": 5,
            "synthetic_array": True,
            "seed": seed,
        },
        "frequency_offsets_hz": offsets.tolist(),
        "absolute_frequencies_hz": (CARRIER_HZ + offsets).tolist(),
        "delay_resolution_s": 1.0 / BANDWIDTH_HZ,
        "unambiguous_delay_s": bins / BANDWIDTH_HZ,
        "raw_observation": {
            "artifact": "aperture_cfr",
            "axis_order": ["hemisphere", "row", "col", "frequency_offset"],
            "hemispheres": list(HEMISPHERES),
            "rx_element_pattern": "rf_camera_split",
            "note": (
                "Front (local kx >= 0) and back (kx < 0) arrivals of a vertically "
                "polarized isotropic element; front + back equals the isotropic "
                "element. A finite front-to-back ratio g can be synthesized as "
                "front + g * back."
            ),
        },
        "camera_model": {
            "projection": PROJECTION,
            "forward_axis_local": "+x",
            "array_plane_local": "y-z",
            "ray_directions": "camera_model.npz",
            "developed_hemisphere": HEMISPHERES[0],
            "image_quantity": IMAGE_QUANTITY,
            "image_definition": (
                "A(ky, kz) = kx * U(ky, kz) with kx = sqrt(1 - ky^2 - kz^2): U is the "
                "calibrated angular spectrum of the front-hemisphere aperture CFR, A the "
                "complex amplitude per unit solid angle (camera_model.npz "
                "solid_angle_weight). Back-hemisphere arrivals are excluded, like light "
                "behind an optical camera."
            ),
        },
        "path_geometry_gt": "path_geometry_gt.npz",
        "views": manifest_views,
    }
    (root / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
