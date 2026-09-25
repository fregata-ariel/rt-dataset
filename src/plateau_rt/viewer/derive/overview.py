"""The eager overview deriver for rf_dataset members."""

from __future__ import annotations

import math
import posixpath
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from plateau_rt.application.rf_dataset_manifest import RFDatasetManifest
from plateau_rt.viewer import kinds
from plateau_rt.viewer.derive import DeriveContext, DeriverSpec, register
from plateau_rt.viewer.safeio import NpyInfo, UnsafePathError

DEFAULT_M1_IMAGE_AXES: dict[str, Any] = {
    "col": "+ky (camera left)",
    "row": "+kz (up)",
    "array_order": (
        "row index increases with kz, column index increases with ky (camera_model.npz grid)"
    ),
    "mirrored_vs_pinhole": True,
    "hemisphere_png_flipud": True,
}


def _is_finite_number(value: Any) -> bool:
    """Return True for a finite int/float that is not a bool."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _member_join(member_dir: str, relpath: str) -> str:
    """Join ``relpath`` onto a bundle-relative member directory."""
    if member_dir in (".", ""):
        return relpath
    return posixpath.join(member_dir, relpath)


def _bin_spacing_hz(offsets: np.ndarray) -> float | None:
    """Return the uniform frequency bin spacing, or None when not uniform."""
    values = [float(value) for value in np.asarray(offsets, dtype=np.float64).ravel()]
    count = len(values)
    if count < 2:
        return None
    diffs = [values[index + 1] - values[index] for index in range(count - 1)]
    first = diffs[0]
    if all(math.isclose(diff, first, rel_tol=1e-9, abs_tol=0.0) for diff in diffs):
        return float(first)
    return None


def _fft_shape(
    camera_model: Mapping[str, NpyInfo] | None, config: Mapping[str, Any]
) -> tuple[int | None, int | None]:
    """Return (fft_rows, fft_cols) from valid_mask or the config fallback."""
    if camera_model is not None and "valid_mask" in camera_model:
        shape = tuple(int(dim) for dim in camera_model["valid_mask"].shape)
        if len(shape) >= 2:
            return shape[0], shape[1]
        return None, None
    rows = config.get("fft_rows")
    cols = config.get("fft_cols")
    fft_rows = int(rows) if isinstance(rows, int) and not isinstance(rows, bool) else None
    fft_cols = int(cols) if isinstance(cols, int) and not isinstance(cols, bool) else None
    return fft_rows, fft_cols


def _spacing(value: Any) -> float | None:
    """Return a finite spacing number, or None."""
    if _is_finite_number(value):
        return float(value)
    return None


def build_overview(
    manifest: RFDatasetManifest,
    *,
    member: str,
    members: Sequence[Mapping[str, Any]],
    camera_model: Mapping[str, NpyInfo] | None,
    exists: Callable[[str], bool],
    is_dir: Callable[[str], bool],
) -> dict[str, Any]:
    """Build the overview.json payload for one rf_dataset member."""
    raw = manifest.raw if isinstance(manifest.raw, Mapping) else {}
    config = manifest.config if isinstance(manifest.config, Mapping) else {}
    member_dir = str(manifest.root)
    offsets = np.asarray(manifest.frequency_offsets_hz, dtype=np.float64).ravel()
    num_bins = int(offsets.shape[0])
    spacing = _bin_spacing_hz(offsets)
    bandwidth_raw = config.get("bandwidth_hz")
    bandwidth: float | None = None
    if (
        isinstance(bandwidth_raw, (int, float))
        and not isinstance(bandwidth_raw, bool)
        and math.isfinite(bandwidth_raw)
        and bandwidth_raw > 0
    ):
        bandwidth = float(bandwidth_raw)
    elif spacing is not None:
        bandwidth = float(num_bins) * float(spacing)
    if manifest.delay_resolution_s is not None:
        delay_resolution: float | None = float(manifest.delay_resolution_s)
    elif bandwidth is not None and bandwidth > 0:
        delay_resolution = 1.0 / float(bandwidth)
    else:
        delay_resolution = None
    if manifest.unambiguous_delay_s is not None:
        unambiguous_delay: float | None = float(manifest.unambiguous_delay_s)
    elif bandwidth is not None and bandwidth > 0:
        unambiguous_delay = float(num_bins) / float(bandwidth)
    else:
        unambiguous_delay = None
    fft_rows, fft_cols = _fft_shape(camera_model, config)
    pairs: list[dict[str, Any]] = []
    for view, entry in manifest.pairs():
        energy = {str(key): float(value) for key, value in entry.hemisphere_energy.items()}
        total = float(sum(energy.values()))
        if total > 0 and "back" in energy:
            back_fraction: float | None = float(energy["back"]) / total
        else:
            back_fraction = None
        pairs.append(
            {
                "view_id": str(view.view_id),
                "bs_id": str(entry.bs_id),
                "bs_in_front_hemisphere": (
                    None
                    if entry.bs_in_front_hemisphere is None
                    else bool(entry.bs_in_front_hemisphere)
                ),
                "hemisphere_energy": energy,
                "total_energy": total,
                "back_fraction": back_fraction,
            }
        )
    optical_keys: set[str] = set()
    for view in manifest.views:
        for key in view.artifacts:
            if key.startswith("optical_"):
                optical_keys.add(str(key))
    optical_reference = raw.get("optical_reference")
    transforms_name = "transforms.json"
    if isinstance(optical_reference, Mapping):
        pinhole = optical_reference.get("pinhole")
        if isinstance(pinhole, Mapping) and isinstance(pinhole.get("transforms"), str):
            transforms_name = str(pinhole["transforms"])
    path_gt = manifest.path_geometry_gt is not None and exists(str(manifest.path_geometry_gt.path))
    if manifest.path_geometry_gt is not None and manifest.path_geometry_gt.schema_path is not None:
        path_schema = exists(str(manifest.path_geometry_gt.schema_path))
    else:
        path_schema = False
    observations_raw = raw.get("observations")
    if isinstance(observations_raw, Mapping):
        observations = sorted(str(key) for key in observations_raw)
    else:
        observations = []
    partials = sorted(
        str(info["id"])
        for info in members
        if info.get("kind") == "rf_partial" and (info.get("links") or {}).get("source") == member
    )
    scene_id: str | None = None
    for info in members:
        if info.get("kind") == "scene" and (info.get("links") or {}).get("for") == member:
            scene_id = str(info["id"])
            break
    camera_raw = raw.get("camera_model")
    if isinstance(camera_raw, Mapping) and isinstance(camera_raw.get("image_axes"), Mapping):
        image_axes: dict[str, Any] = {
            "source": "manifest",
            "axes": dict(camera_raw["image_axes"]),
        }
    else:
        image_axes = {"source": "default_m1", "axes": dict(DEFAULT_M1_IMAGE_AXES)}
    return {
        "member": str(member),
        "schema_version": int(manifest.schema_version),
        "mode": None if manifest.mode is None else str(manifest.mode),
        "source_scene": None if manifest.source_scene is None else str(manifest.source_scene),
        "num_views": int(manifest.num_views),
        "num_bs": int(manifest.num_bs),
        "base_stations": [
            {
                "bs_id": str(station.bs_id),
                "index": int(station.index),
                "position_m": [float(value) for value in station.position_m],
                "look_at_m": [float(value) for value in station.look_at_m],
            }
            for station in manifest.base_stations
        ],
        "views": [
            {
                "view_id": str(view.view_id),
                "index": int(view.index),
                "position_m": [float(value) for value in view.position_m],
                "look_at_m": (
                    None if view.look_at_m is None else [float(value) for value in view.look_at_m]
                ),
                "orientation_rad": (
                    None
                    if view.orientation_rad is None
                    else [float(value) for value in view.orientation_rad]
                ),
            }
            for view in manifest.views
        ],
        "frequency": {
            "carrier_frequency_hz": float(manifest.carrier_frequency_hz),
            "bandwidth_hz": bandwidth,
            "num_bins": int(num_bins),
            "bin_spacing_hz": spacing,
            "delay_resolution_s": delay_resolution,
            "unambiguous_delay_s": unambiguous_delay,
        },
        "camera_model": {
            "fft_rows": fft_rows,
            "fft_cols": fft_cols,
            "rx_rows": int(manifest.rx_rows),
            "rx_cols": int(manifest.rx_cols),
            "horizontal_spacing_lambda": _spacing(config.get("horizontal_spacing_lambda")),
            "vertical_spacing_lambda": _spacing(config.get("vertical_spacing_lambda")),
            "hemispheres": [str(name) for name in manifest.hemispheres],
        },
        "pairs": pairs,
        "contents": {
            "optical": bool(optical_keys),
            "optical_artifacts": sorted(optical_keys),
            "transforms_json": exists(_member_join(member_dir, transforms_name)),
            "path_gt": bool(path_gt),
            "path_schema": bool(path_schema),
            "observations": observations,
            "partials": partials,
            "scene": scene_id,
            "placement": bool(is_dir(_member_join(member_dir, "placement")) or "placement" in raw),
            "tomography_gt": exists(_member_join(member_dir, "tomography_gt.npz")),
        },
        "image_axes": image_axes,
    }


class OverviewDeriver:
    """The eager overview deriver for rf_dataset members."""

    spec = DeriverSpec("overview", 1, ("rf_dataset",), (), eager=True)

    def param_space(self, ctx: DeriveContext) -> list[dict[str, str]]:
        """The overview deriver has no parameters."""
        return [{}]

    def derive(self, ctx: DeriveContext, params: Mapping[str, Any]) -> dict[str, Any]:
        """Derive overview.json for the member through the context loaders."""
        member_path = str(ctx.member_info.get("path", "."))
        manifest_relpath = (
            "dataset_manifest.json"
            if member_path == "."
            else posixpath.join(member_path, "dataset_manifest.json")
        )
        data = ctx.read_bytes(manifest_relpath, max_bytes=kinds.MAX_MANIFEST_BYTES)
        manifest = kinds.dataset_manifest_from_bytes(data, member_path)
        camera_model = ctx.check_npz(str(manifest.camera_model_path))

        def exists(relpath: str) -> bool:
            """Return True when a bundle-root relpath is an existing file."""
            try:
                return ctx.path(relpath).is_file()
            except (UnsafePathError, OSError):
                return False

        def is_dir(relpath: str) -> bool:
            """Return True when a bundle-root relpath is an existing directory."""
            try:
                return ctx.path(relpath).is_dir()
            except (UnsafePathError, OSError):
                return False

        payload = build_overview(
            manifest,
            member=ctx.member,
            members=ctx.members,
            camera_model=camera_model,
            exists=exists,
            is_dir=is_dir,
        )
        return {"overview.json": payload}


OVERVIEW = register(OverviewDeriver())
