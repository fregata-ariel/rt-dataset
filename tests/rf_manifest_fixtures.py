"""Synthetic RF-camera datasets for manifest reader tests.

Writes small, self-consistent schema v2/v3 datasets (manifest plus the
payload files the typed reader resolves) with deterministic random aperture
CFRs by default. Callers may instead supply the aperture CFRs, the derived
per-BS images and the path-geometry ground truth, so the same writer also
backs the physically consistent viewer fixtures. This is a plain helper
module, not a test file, and will be reused by other test files later.
"""

from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from plateau_rt.application.rf_dataset_manifest import PER_BS_ARTIFACT_KEYS
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

# A per-(view, BS) artifact producer: returns every ``PER_BS_ARTIFACT_KEYS``.
DeriveFn = Callable[[int, int, np.ndarray], Mapping[str, np.ndarray | bytes]]


def save_npz_deterministic(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write ``arrays`` as a compressed ``.npz`` whose bytes depend only on the arrays.

    ``numpy.savez_compressed`` stamps the current time into the zip entries, so
    two builds differ in bytes. This writer pins every entry timestamp to the
    zip epoch and serialises each array with ``numpy.lib.format.write_array``.
    """
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, value in arrays.items():
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.asanyarray(value), allow_pickle=False)
            archive.writestr(info, buffer.getvalue())


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


def _write_camera_model(root: Path, *, fft_rows: int = FFT_ROWS, fft_cols: int = FFT_COLS) -> None:
    """Write ``camera_model.npz`` with a deterministic direction-cosine grid."""
    model = build_direction_cosine_camera_model(
        fft_rows=fft_rows,
        fft_cols=fft_cols,
        horizontal_spacing_lambda=0.5,
        vertical_spacing_lambda=0.5,
    )
    save_npz_deterministic(root / "camera_model.npz", model)


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
    arrays: Mapping[str, np.ndarray] | None = None,
    object_names: Sequence[str] = ("mock_building",),
    carrier_hz: float = CARRIER_HZ,
) -> None:
    """Write canonical ``path_geometry_gt.npz`` (supplied or dummy) and its schema."""
    if arrays is None:
        resolved = {
            "valid": np.ones((num_views, num_bs, 2), dtype=bool),
            "tau": np.zeros((num_views, num_bs, 2), dtype=np.float32),
            "a_baseband": np.zeros((num_views, num_bs, 2, rows, cols, 2), dtype=np.complex64),
            "num_interactions": np.zeros((num_views, num_bs, 2), dtype=np.int32),
        }
    else:
        resolved = dict(arrays)
    save_npz_deterministic(root / PATH_GEOMETRY_GT_FILE_NAME, resolved)
    if not write_schema:
        return
    schema = build_path_schema(
        resolved,
        mode=PATH_GT_MODE_CANONICAL,
        object_names=list(object_names),
        carrier_frequency_hz=carrier_hz,
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


def _bs_artifact_paths(view_id: str, bs_id: str) -> dict[str, str]:
    """Return the relative artifact paths of one (view, BS) entry (writer layout)."""
    prefix = f"views/{view_id}/rf/{bs_id}"
    return {
        "angular_cfr_center": f"{prefix}/angular_cfr_center.npy",
        "angular_power_center": f"{prefix}/angular_power_center.npy",
        "phase_valid_mask": f"{prefix}/phase_valid_mask.npy",
        "dominant_delay_s": f"{prefix}/dominant_delay_s.npy",
        "dominant_delay_power": f"{prefix}/dominant_delay_power.npy",
        "debug_power_png": f"{prefix}/angular_power_center.png",
    }


def _write_derived_artifacts(
    root: Path,
    artifacts: Mapping[str, str],
    derived: Mapping[str, np.ndarray | bytes] | None,
) -> None:
    """Write per-BS derived artifacts, either placeholders or a supplied mapping."""
    if derived is None:
        # Placeholders cover the .npy artifacts only; the debug PNG is not written.
        for key in PER_BS_ARTIFACT_KEYS:
            if artifacts[key].endswith(".npy"):
                _write_placeholder_npy(root / artifacts[key])
        return
    missing = [key for key in PER_BS_ARTIFACT_KEYS if key not in derived]
    if missing:
        raise ValueError(f"derive mapping is missing artifact key(s): {missing}")
    for name in PER_BS_ARTIFACT_KEYS:
        path = root / artifacts[name]
        path.parent.mkdir(parents=True, exist_ok=True)
        value = derived[name]
        if name == "debug_power_png":
            if not isinstance(value, (bytes, bytearray)):
                raise ValueError(f"derive mapping key {name!r} must be bytes, got {type(value)!r}")
            path.write_bytes(bytes(value))
        else:
            np.save(path, np.asarray(value))


def _resolve_apertures(
    apertures: np.ndarray | None,
    *,
    num_views: int,
    expected_shape: tuple[int, ...],
    random_shape: tuple[int, ...],
    seed: int,
) -> np.ndarray:
    """Return the supplied aperture CFRs (validated) or deterministic random ones."""
    if apertures is None:
        return np.stack(
            [_random_aperture(random_shape, seed + view_index) for view_index in range(num_views)]
        )
    resolved = np.asarray(apertures)
    if resolved.shape != expected_shape:
        raise ValueError(f"apertures must have shape {expected_shape}, got {resolved.shape}")
    return resolved.astype(np.complex64, copy=False)


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
    bs_positions: Sequence[tuple[float, float, float]] | None = None,
    bs_look_at: tuple[float, float, float] = TARGET_M,
    carrier_hz: float = CARRIER_HZ,
    bandwidth_hz: float = BANDWIDTH_HZ,
    fft_rows: int = FFT_ROWS,
    fft_cols: int = FFT_COLS,
    apertures: np.ndarray | None = None,
    derive: DeriveFn | None = None,
    path_gt_arrays: Mapping[str, np.ndarray] | None = None,
    path_gt_object_names: Sequence[str] = ("mock_building",),
    write_path_gt: bool = True,
) -> dict[str, Any]:
    """Write a synthetic schema v3 dataset under ``root`` and return its manifest."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    offsets = centred_frequency_offsets(bins, bandwidth_hz)
    if bs_positions is not None:
        resolved_positions = [
            (float(position[0]), float(position[1]), float(position[2]))
            for position in bs_positions
        ]
        num_bs = len(resolved_positions)
    else:
        resolved_positions = _resolve_bs_positions(num_bs)
    bs_ids = [f"bs_{index:03d}" for index in range(num_bs)]
    views = _resolve_views(views, num_views)
    num_views = len(views)
    aperture = _resolve_apertures(
        apertures,
        num_views=num_views,
        expected_shape=(num_views, num_bs, 2, rows, cols, bins),
        random_shape=(num_bs, 2, rows, cols, bins),
        seed=seed,
    )

    _write_camera_model(root, fft_rows=fft_rows, fft_cols=fft_cols)
    if write_path_gt:
        _write_path_geometry_gt(
            root,
            num_views=num_views,
            num_bs=num_bs,
            rows=rows,
            cols=cols,
            bs_ids=bs_ids,
            view_ids=[view.view_id for view in views],
            write_schema=True,
            arrays=path_gt_arrays,
            object_names=path_gt_object_names,
            carrier_hz=carrier_hz,
        )

    manifest_views: list[dict[str, Any]] = []
    for view_index, view in enumerate(views):
        view_dir = root / "views" / view.view_id
        rf_dir = view_dir / "rf"
        rf_dir.mkdir(parents=True, exist_ok=True)
        (view_dir / "pose.json").write_text(
            json.dumps(view_pose_payload(view), indent=2), encoding="utf-8"
        )
        np.save(rf_dir / "aperture_cfr.npy", aperture[view_index].astype(np.complex64))

        bs_entries: list[dict[str, Any]] = []
        for bs_index, (bs_id, bs_position) in enumerate(zip(bs_ids, resolved_positions)):
            direction, in_front = _bs_geometry(bs_position, view=view)
            artifacts = _bs_artifact_paths(view.view_id, bs_id)
            aperture_bs = aperture[view_index, bs_index]
            derived = None if derive is None else derive(view_index, bs_index, aperture_bs)
            _write_derived_artifacts(root, artifacts, derived)
            bs_entries.append(
                {
                    "bs_id": bs_id,
                    "bs_direction_local": direction,
                    "bs_in_front_hemisphere": in_front,
                    "hemisphere_energy": _hemisphere_energy(aperture_bs),
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
            "carrier_frequency_hz": carrier_hz,
            "bandwidth_hz": bandwidth_hz,
            "num_frequency_bins": bins,
            "tx_positions": [list(position) for position in resolved_positions],
            "tx_look_at": list(bs_look_at),
            "tx_look_ats": None,
            "rx_rows": rows,
            "rx_cols": cols,
            "vertical_spacing_lambda": 0.5,
            "horizontal_spacing_lambda": 0.5,
            "tx_pattern": "tr38901",
            "polarization": "V",
            "fft_rows": fft_rows,
            "fft_cols": fft_cols,
            "phase_floor_db": -35.0,
            "max_depth": 5,
            "synthetic_array": True,
            "seed": seed,
        },
        "frequency_offsets_hz": offsets.tolist(),
        "absolute_frequencies_hz": (carrier_hz + offsets).tolist(),
        "delay_resolution_s": 1.0 / bandwidth_hz,
        "unambiguous_delay_s": bins / bandwidth_hz,
        "base_stations": [
            {
                "bs_id": bs_id,
                "index": index,
                "position_m": list(position),
                "look_at_m": list(bs_look_at),
            }
            for index, (bs_id, position) in enumerate(zip(bs_ids, resolved_positions))
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
        "views": manifest_views,
    }
    if write_path_gt:
        manifest["path_geometry_gt"] = PATH_GEOMETRY_GT_FILE_NAME
        manifest["path_schema"] = PATH_SCHEMA_FILE_NAME
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
    bs_position: tuple[float, float, float] | None = None,
    bs_look_at: tuple[float, float, float] = TARGET_M,
    carrier_hz: float = CARRIER_HZ,
    bandwidth_hz: float = BANDWIDTH_HZ,
    fft_rows: int = FFT_ROWS,
    fft_cols: int = FFT_COLS,
    apertures: np.ndarray | None = None,
    derive: DeriveFn | None = None,
    path_gt_arrays: Mapping[str, np.ndarray] | None = None,
    path_gt_object_names: Sequence[str] = ("mock_building",),
    write_path_gt: bool = True,
    write_path_schema: bool = False,
) -> dict[str, Any]:
    """Write a synthetic schema v2 (single-BS) dataset under ``root``."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    offsets = centred_frequency_offsets(bins, bandwidth_hz)
    source_position = BS_POSITIONS_M[0] if bs_position is None else bs_position
    resolved_position = (
        float(source_position[0]),
        float(source_position[1]),
        float(source_position[2]),
    )
    views = _resolve_views(views, num_views)
    num_views = len(views)
    aperture = _resolve_apertures(
        apertures,
        num_views=num_views,
        expected_shape=(num_views, 2, rows, cols, bins),
        random_shape=(2, rows, cols, bins),
        seed=seed,
    )

    _write_camera_model(root, fft_rows=fft_rows, fft_cols=fft_cols)
    if write_path_gt:
        _write_path_geometry_gt(
            root,
            num_views=num_views,
            num_bs=1,
            rows=rows,
            cols=cols,
            bs_ids=["bs_000"],
            view_ids=[view.view_id for view in views],
            write_schema=write_path_schema,
            arrays=path_gt_arrays,
            object_names=path_gt_object_names,
            carrier_hz=carrier_hz,
        )

    manifest_views: list[dict[str, Any]] = []
    for view_index, view in enumerate(views):
        view_dir = root / "views" / view.view_id
        rf_dir = view_dir / "rf"
        rf_dir.mkdir(parents=True, exist_ok=True)
        (view_dir / "pose.json").write_text(
            json.dumps(view_pose_payload(view), indent=2), encoding="utf-8"
        )
        aperture_view = aperture[view_index].astype(np.complex64)
        np.save(rf_dir / "aperture_cfr.npy", aperture_view)
        direction, in_front = _bs_geometry(resolved_position, view=view)
        artifacts = {
            "pose": f"views/{view.view_id}/pose.json",
            "aperture_cfr": f"views/{view.view_id}/rf/aperture_cfr.npy",
        }
        artifacts.update(_bs_artifact_paths(view.view_id, "bs_000"))
        derived = None if derive is None else derive(view_index, 0, aperture_view)
        _write_derived_artifacts(root, artifacts, derived)
        manifest_views.append(
            {
                "view_id": view.view_id,
                "position_m": list(view.position),
                "look_at_m": list(view.look_at),
                "orientation_rad": list(view.orientation),
                "bs_direction_local": direction,
                "bs_in_front_hemisphere": in_front,
                "hemisphere_energy": _hemisphere_energy(aperture_view),
                "artifacts": artifacts,
            }
        )

    manifest = {
        "schema_version": 2,
        "mode": "1bs_multiue_rf_camera_dataset",
        "source_scene": source_scene,
        "config": {
            "carrier_frequency_hz": carrier_hz,
            "bandwidth_hz": bandwidth_hz,
            "num_frequency_bins": bins,
            "tx_position": list(resolved_position),
            "tx_look_at": list(bs_look_at),
            "rx_rows": rows,
            "rx_cols": cols,
            "vertical_spacing_lambda": 0.5,
            "horizontal_spacing_lambda": 0.5,
            "tx_pattern": "tr38901",
            "polarization": "V",
            "fft_rows": fft_rows,
            "fft_cols": fft_cols,
            "phase_floor_db": -35.0,
            "max_depth": 5,
            "synthetic_array": True,
            "seed": seed,
        },
        "frequency_offsets_hz": offsets.tolist(),
        "absolute_frequencies_hz": (carrier_hz + offsets).tolist(),
        "delay_resolution_s": 1.0 / bandwidth_hz,
        "unambiguous_delay_s": bins / bandwidth_hz,
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
        "views": manifest_views,
    }
    if write_path_gt:
        manifest["path_geometry_gt"] = PATH_GEOMETRY_GT_FILE_NAME
    if write_path_gt and write_path_schema:
        manifest["path_schema"] = PATH_SCHEMA_FILE_NAME
    (root / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
