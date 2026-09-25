"""Typed, Sionna-free reader for RF-camera multi-view dataset manifests.

Reads ``dataset_manifest.json`` files written by
``plateau_rt.adapters.sionna.rf_camera_dataset`` in the schema v3
(multi-BS) and schema v2 (single-BS) layouts and exposes them as frozen
dataclasses with absolute artifact paths.

NumPy is the only third-party dependency: this module must stay importable
without Sionna, Mitsuba, Dr.Jit or Matplotlib (see
``tests/test_rf_camera_boundaries.py``).
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from plateau_rt.domain.rf_camera.camera import HEMISPHERES

MANIFEST_FILE_NAME = "dataset_manifest.json"
SUPPORTED_SCHEMA_VERSIONS: tuple[int, ...] = (2, 3)
APERTURE_CFR_AXIS_ORDER: tuple[str, ...] = ("bs", "hemisphere", "row", "col", "frequency_offset")
V2_APERTURE_CFR_AXIS_ORDER: tuple[str, ...] = ("hemisphere", "row", "col", "frequency_offset")
PER_BS_ARTIFACT_KEYS: tuple[str, ...] = (
    "angular_cfr_center",
    "angular_power_center",
    "phase_valid_mask",
    "dominant_delay_s",
    "dominant_delay_power",
    "debug_power_png",
)
LEGACY_PATH_GT_KEYS: tuple[str, ...] = (
    "valid",
    "tau",
    "theta_t",
    "phi_t",
    "theta_r",
    "phi_r",
)
DEFAULT_PATH_GT_AXES: tuple[str, ...] = ("view", "bs", "path")
NATIVE_PATH_GT_AXES: tuple[str, ...] = ("view", "rx_ant", "bs", "tx_ant", "path")


class ManifestError(ValueError):
    """Raised when a manifest is malformed, unsupported or inconsistent."""


@dataclass(frozen=True)
class BaseStationInfo:
    """One base station: resolved id, index, position and look-at in metres."""

    bs_id: str
    index: int
    position_m: tuple[float, float, float]
    look_at_m: tuple[float, float, float]


@dataclass(frozen=True)
class ViewBSEntry:
    """Per-(view, BS) manifest entry with absolute-resolved artifact paths."""

    view_id: str
    bs_id: str
    bs_index: int
    bs_direction_local: tuple[float, float, float] | None
    bs_in_front_hemisphere: bool | None
    hemisphere_energy: Mapping[str, float]
    artifacts: Mapping[str, Path]

    def artifact(self, name: str) -> Path:
        """Return the absolute path of per-BS artifact ``name``.

        Raises ManifestError naming the view, BS and artifact when missing.
        """
        try:
            return self.artifacts[name]
        except KeyError as exc:
            raise ManifestError(
                f"view {self.view_id!r} bs {self.bs_id!r} has no artifact {name!r}; "
                f"known artifacts: {sorted(self.artifacts)}"
            ) from exc

    @property
    def total_energy(self) -> float:
        """Sum of ``hemisphere_energy`` values over all hemispheres."""
        return float(sum(self.hemisphere_energy.values()))


@dataclass(frozen=True)
class DatasetView:
    """One UE view: view-level artifacts plus one entry per base station."""

    view_id: str
    index: int
    position_m: tuple[float, float, float]
    look_at_m: tuple[float, float, float] | None
    orientation_rad: tuple[float, float, float] | None
    artifacts: Mapping[str, Path]
    bs: tuple[ViewBSEntry, ...]

    def artifact(self, name: str) -> Path:
        """Return the absolute path of view-level artifact ``name``.

        Raises ManifestError naming the view and artifact when missing.
        """
        try:
            return self.artifacts[name]
        except KeyError as exc:
            raise ManifestError(
                f"view {self.view_id!r} has no view-level artifact {name!r}; "
                f"known artifacts: {sorted(self.artifacts)}"
            ) from exc

    @property
    def pose_path(self) -> Path:
        """Absolute path of the ``pose`` artifact."""
        return self.artifact("pose")

    @property
    def aperture_cfr_path(self) -> Path:
        """Absolute path of the ``aperture_cfr`` artifact."""
        return self.artifact("aperture_cfr")

    def bs_entry(self, bs_id: str) -> ViewBSEntry:
        """Return the per-BS entry for ``bs_id``.

        Raises ManifestError naming the view and unknown BS id when missing.
        """
        for entry in self.bs:
            if entry.bs_id == bs_id:
                return entry
        raise ManifestError(
            f"view {self.view_id!r} has no BS entry {bs_id!r}; "
            f"known bs_ids: {[entry.bs_id for entry in self.bs]}"
        )


@dataclass(frozen=True)
class PathGeometryGT:
    """Path-geometry ground-truth artifact plus its optional schema file.

    ``synthetic_array`` is the writer's ``config.synthetic_array`` and only
    drives the legacy axis fallback used when the dataset has no
    ``path_schema`` (schema v2 or an older v3).
    """

    path: Path
    schema_path: Path | None
    synthetic_array: bool = True

    def load_schema(self) -> dict[str, Any]:
        """Load and validate the ``path_schema.json`` described by the manifest.

        Raises ManifestError when no schema is referenced, the file cannot be
        read, is not valid JSON, is not an object or has no ``arrays`` mapping.
        """
        if self.schema_path is None:
            raise ManifestError(f"path geometry GT {self.path} has no path_schema")
        try:
            text = self.schema_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ManifestError(
                f"path schema could not be read from {self.schema_path}: {exc}"
            ) from exc
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ManifestError(f"path schema {self.schema_path} is not valid JSON: {exc}") from exc
        if not isinstance(data, Mapping):
            raise ManifestError(f"path schema {self.schema_path} must be a JSON object")
        arrays = data.get("arrays")
        if not isinstance(arrays, Mapping):
            raise ManifestError(f"path schema {self.schema_path} has no 'arrays' mapping")
        return dict(data)

    def array_axes(self, name: str) -> tuple[str, ...]:
        """Return the stored axis names of path-GT array ``name``.

        Uses the schema when one is referenced. For a dataset without a
        ``path_schema`` only the six legacy geometry keys are described, with
        axes derived from ``synthetic_array``; anything else raises
        ManifestError.
        """
        if not isinstance(name, str) or not name:
            raise ManifestError(f"path GT array name must be a non-empty string, got {name!r}")
        if self.schema_path is not None:
            arrays = self.load_schema()["arrays"]
            if name not in arrays:
                raise ManifestError(
                    f"path schema has no array {name!r}; known arrays: {sorted(arrays)}"
                )
            entry = arrays[name]
            axes = entry.get("axes") if isinstance(entry, Mapping) else None
            if not isinstance(axes, list) or any(not isinstance(axis, str) for axis in axes):
                raise ManifestError(
                    f"path schema array {name!r} 'axes' must be a list of strings, got {axes!r}"
                )
            return tuple(axes)
        if name not in LEGACY_PATH_GT_KEYS:
            raise ManifestError(
                f"path geometry GT without a path_schema only describes "
                f"{list(LEGACY_PATH_GT_KEYS)}, not {name!r}"
            )
        return DEFAULT_PATH_GT_AXES if self.synthetic_array else NATIVE_PATH_GT_AXES

    def load_arrays(self) -> dict[str, np.ndarray]:
        """Load every array stored in ``path_geometry_gt.npz`` as a fresh dict.

        Raises ManifestError when the artifact cannot be loaded.
        """
        try:
            with np.load(self.path) as data:
                return {name: np.asarray(data[name]) for name in data.files}
        except OSError as exc:
            raise ManifestError(
                f"path geometry GT could not be loaded from {self.path}: {exc}"
            ) from exc


@dataclass(frozen=True, eq=False)
class RFDatasetManifest:
    """Typed view of a parsed ``dataset_manifest.json`` (schema v2 or v3)."""

    root: Path
    manifest_path: Path
    schema_version: int
    mode: str | None
    source_scene: str | None
    config: Mapping[str, Any]
    carrier_frequency_hz: float
    frequency_offsets_hz: np.ndarray
    absolute_frequencies_hz: np.ndarray
    delay_resolution_s: float | None
    unambiguous_delay_s: float | None
    hemispheres: tuple[str, ...]
    stored_aperture_axis_order: tuple[str, ...]
    rx_rows: int
    rx_cols: int
    base_stations: tuple[BaseStationInfo, ...]
    views: tuple[DatasetView, ...]
    camera_model_path: Path
    path_geometry_gt: PathGeometryGT | None
    raw: Mapping[str, Any]

    @property
    def num_views(self) -> int:
        """Number of views in the dataset."""
        return len(self.views)

    @property
    def num_bs(self) -> int:
        """Number of base stations (1 for schema v2)."""
        return len(self.base_stations)

    @property
    def placement(self) -> Mapping[str, Any] | None:
        """The raw ``placement`` section (coverage placement, #16), or None."""
        return self.raw.get("placement")

    @property
    def view_ids(self) -> tuple[str, ...]:
        """View ids in manifest order."""
        return tuple(view.view_id for view in self.views)

    @property
    def bs_ids(self) -> tuple[str, ...]:
        """Base-station ids in manifest order."""
        return tuple(bs.bs_id for bs in self.base_stations)

    @property
    def num_frequency_bins(self) -> int:
        """Number of frequency bins (length of ``frequency_offsets_hz``)."""
        return int(self.frequency_offsets_hz.shape[0])

    @property
    def aperture_cfr_axis_order(self) -> tuple[str, ...]:
        """Normalized aperture axis order (always with a leading ``bs`` axis)."""
        return APERTURE_CFR_AXIS_ORDER

    @property
    def aperture_cfr_shape(self) -> tuple[int, int, int, int, int]:
        """Expected normalized aperture shape ``(bs, hemi, rows, cols, bins)``."""
        return (
            self.num_bs,
            len(self.hemispheres),
            self.rx_rows,
            self.rx_cols,
            self.num_frequency_bins,
        )

    def view(self, view_id: str) -> DatasetView:
        """Return the view with id ``view_id``.

        Raises ManifestError naming the unknown view id when missing.
        """
        for view in self.views:
            if view.view_id == view_id:
                return view
        raise ManifestError(f"unknown view_id {view_id!r}; known view_ids: {list(self.view_ids)}")

    def base_station(self, bs_id: str) -> BaseStationInfo:
        """Return the base station with id ``bs_id``.

        Raises ManifestError naming the unknown BS id when missing.
        """
        for bs in self.base_stations:
            if bs.bs_id == bs_id:
                return bs
        raise ManifestError(f"unknown bs_id {bs_id!r}; known bs_ids: {list(self.bs_ids)}")

    def pairs(self) -> Iterator[tuple[DatasetView, ViewBSEntry]]:
        """Yield ``(view, bs_entry)`` pairs, view-major then in BS order."""
        for view in self.views:
            for entry in view.bs:
                yield view, entry

    def load_aperture_cfr(self, view: DatasetView | str) -> np.ndarray:
        """Load one view's aperture CFR in normalized axis order.

        Accepts a :class:`DatasetView` or a view id. For schema v2 (stored
        without a BS axis) a leading axis is added. Raises ManifestError
        naming the view when the normalized shape differs from
        :attr:`aperture_cfr_shape`.
        """
        view_obj = self.view(view) if isinstance(view, str) else view
        if not isinstance(view_obj, DatasetView):
            raise TypeError(f"view must be a DatasetView or view id, got {type(view)!r}")
        path = view_obj.aperture_cfr_path
        try:
            array = np.load(path)
        except OSError as exc:
            raise ManifestError(
                f"view {view_obj.view_id!r} aperture_cfr could not be loaded from {path}: {exc}"
            ) from exc
        array = np.asarray(array)
        if self.schema_version == 2:
            array = array[np.newaxis]
        if tuple(array.shape) != self.aperture_cfr_shape:
            raise ManifestError(
                f"view {view_obj.view_id!r} aperture_cfr shape {tuple(array.shape)} "
                f"does not match expected {self.aperture_cfr_shape} in axis order "
                f"{list(APERTURE_CFR_AXIS_ORDER)}"
            )
        return array


def _is_number(value: Any) -> bool:
    """Return True for real numbers (bools excluded)."""
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _require_vector(value: Any, *, name: str) -> tuple[float, float, float]:
    """Validate a length-3 vector of finite numbers, raising ManifestError."""
    if isinstance(value, (str, bytes)):
        raise ManifestError(f"{name} must be a length-3 sequence of finite numbers")
    try:
        items = list(value)
    except TypeError as exc:
        raise ManifestError(f"{name} must be a length-3 sequence of finite numbers") from exc
    if len(items) != 3 or any(
        not _is_number(item) or not math.isfinite(float(item)) for item in items
    ):
        raise ManifestError(f"{name} must be a length-3 sequence of finite numbers, got {value!r}")
    return (float(items[0]), float(items[1]), float(items[2]))


def _require_optional_vector(value: Any, *, name: str) -> tuple[float, float, float] | None:
    """Validate an optional 3-vector (None stays None), raising ManifestError."""
    if value is None:
        return None
    return _require_vector(value, name=name)


def _require_int(value: Any, *, name: str) -> int:
    """Validate an integer config value, raising ManifestError."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ManifestError(f"{name} must be an integer, got {value!r}")
    return int(value)


def _require_optional_float(value: Any, *, name: str) -> float | None:
    """Validate an optional finite float (None stays None), raising ManifestError."""
    if value is None:
        return None
    if not _is_number(value) or not math.isfinite(float(value)):
        raise ManifestError(f"{name} must be a finite number, got {value!r}")
    return float(value)


def _require_float_list(value: Any, *, name: str) -> np.ndarray:
    """Validate a non-empty list of finite numbers as a float64 array."""
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ManifestError(f"{name} must be a non-empty list of numbers, got {value!r}")
    items = list(value)
    if not items or any(not _is_number(item) or not math.isfinite(float(item)) for item in items):
        raise ManifestError(f"{name} must be a non-empty list of numbers, got {value!r}")
    return np.asarray([float(item) for item in items], dtype=np.float64)


def _require_energy_mapping(value: Any, *, name: str) -> dict[str, float]:
    """Validate a ``{hemisphere: energy}`` dict of finite numbers."""
    if not isinstance(value, Mapping):
        raise ManifestError(f"{name} must be a dict of finite numbers, got {value!r}")
    energies: dict[str, float] = {}
    for key, item in value.items():
        if not _is_number(item) or not math.isfinite(float(item)):
            raise ManifestError(f"{name}[{key!r}] must be a finite number, got {item!r}")
        energies[str(key)] = float(item)
    return energies


def _resolve_artifacts(value: Any, *, root: Path, name: str) -> dict[str, Path]:
    """Resolve an artifacts dict of relative paths against ``root``."""
    if not isinstance(value, Mapping):
        raise ManifestError(f"{name} must be a dict of relative paths, got {value!r}")
    resolved: dict[str, Path] = {}
    for key, item in value.items():
        if not isinstance(item, str):
            raise ManifestError(f"{name}[{key!r}] must be a relative path, got {item!r}")
        resolved[str(key)] = root / item
    return resolved


def _require_axis_order(value: Any, *, name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    """Validate a list of axis names (missing stays ``default``)."""
    if value is None:
        return default
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ManifestError(f"{name} must be a list of axis names, got {value!r}")
    items = list(value)
    if not items or any(not isinstance(item, str) for item in items):
        raise ManifestError(f"{name} must be a list of axis names, got {value!r}")
    return tuple(items)


def _require_schema_path(value: Any) -> str:
    """Validate a ``path_schema`` value as a non-empty relative path string."""
    if not isinstance(value, str) or not value:
        raise ManifestError(f"'path_schema' must be a non-empty string, got {value!r}")
    return value


def _parse_path_geometry_gt(
    value: Any, *, root: Path, config: Mapping[str, Any], schema_value: Any
) -> PathGeometryGT | None:
    """Parse the ``path_geometry_gt`` and ``path_schema`` manifest payloads.

    Accepts a plain string (v2 and current v3) and the transitional dict form
    ``{"artifact": ...}`` (only ``artifact`` is read; ``axis_order`` / ``note``
    are ignored). ``path_schema`` must be a string when present.
    """
    if value is None:
        if schema_value is not None:
            _require_schema_path(schema_value)
        return None
    if isinstance(value, str):
        artifact = value
    elif isinstance(value, Mapping):
        artifact = value.get("artifact")
        if not isinstance(artifact, str) or not artifact:
            raise ManifestError(
                "'path_geometry_gt' dict must have a non-empty string 'artifact' key, "
                f"got {value!r}"
            )
    else:
        raise ManifestError(
            f"'path_geometry_gt' must be a string or a dict with an 'artifact' key, got {value!r}"
        )
    schema_path = None
    if schema_value is not None:
        schema_path = root / _require_schema_path(schema_value)
    return PathGeometryGT(
        path=root / artifact,
        schema_path=schema_path,
        synthetic_array=bool(config.get("synthetic_array", True)),
    )


def _parse_base_stations_v3(value: Any) -> tuple[BaseStationInfo, ...]:
    """Parse and validate the v3 ``base_stations`` list."""
    if not isinstance(value, list) or not value:
        raise ManifestError(f"'base_stations' must be a non-empty list, got {value!r}")
    stations: list[BaseStationInfo] = []
    seen: set[str] = set()
    for index, entry in enumerate(value):
        if not isinstance(entry, Mapping):
            raise ManifestError(f"'base_stations'[{index}] must be a mapping, got {entry!r}")
        bs_id = entry.get("bs_id")
        if not isinstance(bs_id, str) or not bs_id:
            raise ManifestError(
                f"'base_stations'[{index}] 'bs_id' must be a non-empty string, got {bs_id!r}"
            )
        if bs_id in seen:
            raise ManifestError(f"duplicate bs_id {bs_id!r} in 'base_stations'")
        seen.add(bs_id)
        if entry.get("index") != index:
            raise ManifestError(
                f"'base_stations'[{index}] 'index' must equal {index}, got {entry.get('index')!r}"
            )
        position_m = _require_vector(
            entry.get("position_m"), name=f"base station {bs_id!r} 'position_m'"
        )
        look_at_m = _require_vector(
            entry.get("look_at_m"), name=f"base station {bs_id!r} 'look_at_m'"
        )
        stations.append(
            BaseStationInfo(bs_id=bs_id, index=index, position_m=position_m, look_at_m=look_at_m)
        )
    return tuple(stations)


def _parse_bs_entry_v3(
    value: Any, *, view_id: str, station: BaseStationInfo, root: Path
) -> ViewBSEntry:
    """Parse one v3 per-BS view entry against its base station."""
    name = f"view {view_id!r} bs {station.bs_id!r}"
    if not isinstance(value, Mapping):
        raise ManifestError(f"{name} entry must be a mapping, got {value!r}")
    hemisphere_energy = _require_energy_mapping(
        value.get("hemisphere_energy"), name=f"{name} 'hemisphere_energy'"
    )
    artifacts = _resolve_artifacts(value.get("artifacts"), root=root, name=f"{name} 'artifacts'")
    bs_direction_local = _require_optional_vector(
        value.get("bs_direction_local"), name=f"{name} 'bs_direction_local'"
    )
    in_front = value.get("bs_in_front_hemisphere")
    if in_front is not None and not isinstance(in_front, bool):
        raise ManifestError(f"{name} 'bs_in_front_hemisphere' must be a bool, got {in_front!r}")
    return ViewBSEntry(
        view_id=view_id,
        bs_id=station.bs_id,
        bs_index=station.index,
        bs_direction_local=bs_direction_local,
        bs_in_front_hemisphere=in_front,
        hemisphere_energy=hemisphere_energy,
        artifacts=artifacts,
    )


def _parse_view_v3(
    value: Any, *, index: int, base_stations: tuple[BaseStationInfo, ...], root: Path
) -> DatasetView:
    """Parse one v3 view (view-level artifacts plus an ordered ``bs`` list)."""
    if not isinstance(value, Mapping):
        raise ManifestError(f"'views'[{index}] must be a mapping, got {value!r}")
    view_id = value.get("view_id")
    if not isinstance(view_id, str) or not view_id:
        raise ManifestError(f"'views'[{index}] 'view_id' must be a non-empty string")
    position_m = _require_vector(value.get("position_m"), name=f"view {view_id!r} 'position_m'")
    look_at_m = _require_optional_vector(
        value.get("look_at_m"), name=f"view {view_id!r} 'look_at_m'"
    )
    orientation_rad = _require_optional_vector(
        value.get("orientation_rad"), name=f"view {view_id!r} 'orientation_rad'"
    )
    artifacts_raw = value.get("artifacts")
    if not isinstance(artifacts_raw, Mapping):
        raise ManifestError(f"view {view_id!r} 'artifacts' must be a mapping")
    for required in ("pose", "aperture_cfr"):
        if required not in artifacts_raw:
            raise ManifestError(
                f"view {view_id!r} 'artifacts' is missing required key {required!r}"
            )
    artifacts = _resolve_artifacts(
        dict(artifacts_raw), root=root, name=f"view {view_id!r} 'artifacts'"
    )
    bs_raw = value.get("bs")
    if not isinstance(bs_raw, list):
        raise ManifestError(f"view {view_id!r} 'bs' must be a list with one entry per base station")
    expected_ids = [station.bs_id for station in base_stations]
    got_ids = [entry.get("bs_id") if isinstance(entry, Mapping) else None for entry in bs_raw]
    if got_ids != expected_ids:
        raise ManifestError(
            f"view {view_id!r} 'bs' ids {got_ids} do not match base station order {expected_ids}"
        )
    entries = tuple(
        _parse_bs_entry_v3(entry, view_id=view_id, station=station, root=root)
        for entry, station in zip(bs_raw, base_stations)
    )
    return DatasetView(
        view_id=view_id,
        index=index,
        position_m=position_m,
        look_at_m=look_at_m,
        orientation_rad=orientation_rad,
        artifacts=artifacts,
        bs=entries,
    )


def _parse_view_v2(value: Any, *, index: int, root: Path) -> DatasetView:
    """Parse one v2 view, splitting per-BS artifacts into a synthesized entry."""
    if not isinstance(value, Mapping):
        raise ManifestError(f"'views'[{index}] must be a mapping, got {value!r}")
    view_id = value.get("view_id")
    if not isinstance(view_id, str) or not view_id:
        raise ManifestError(f"'views'[{index}] 'view_id' must be a non-empty string")
    position_m = _require_vector(value.get("position_m"), name=f"view {view_id!r} 'position_m'")
    look_at_m = _require_optional_vector(
        value.get("look_at_m"), name=f"view {view_id!r} 'look_at_m'"
    )
    orientation_rad = _require_optional_vector(
        value.get("orientation_rad"), name=f"view {view_id!r} 'orientation_rad'"
    )
    artifacts_raw = value.get("artifacts")
    if not isinstance(artifacts_raw, Mapping):
        raise ManifestError(f"view {view_id!r} 'artifacts' must be a mapping")
    for required in ("pose", "aperture_cfr"):
        if required not in artifacts_raw:
            raise ManifestError(
                f"view {view_id!r} 'artifacts' is missing required key {required!r}"
            )
    hemisphere_energy = _require_energy_mapping(
        value.get("hemisphere_energy"), name=f"view {view_id!r} 'hemisphere_energy'"
    )
    bs_direction_local = _require_optional_vector(
        value.get("bs_direction_local"), name=f"view {view_id!r} 'bs_direction_local'"
    )
    in_front = value.get("bs_in_front_hemisphere")
    if in_front is not None and not isinstance(in_front, bool):
        raise ManifestError(
            f"view {view_id!r} 'bs_in_front_hemisphere' must be a bool, got {in_front!r}"
        )
    bs_artifacts_raw = {
        key: item for key, item in artifacts_raw.items() if key in PER_BS_ARTIFACT_KEYS
    }
    view_artifacts_raw = {
        key: item for key, item in artifacts_raw.items() if key not in PER_BS_ARTIFACT_KEYS
    }
    entry = ViewBSEntry(
        view_id=view_id,
        bs_id="bs_000",
        bs_index=0,
        bs_direction_local=bs_direction_local,
        bs_in_front_hemisphere=in_front,
        hemisphere_energy=hemisphere_energy,
        artifacts=_resolve_artifacts(
            bs_artifacts_raw, root=root, name=f"view {view_id!r} bs 'bs_000' 'artifacts'"
        ),
    )
    return DatasetView(
        view_id=view_id,
        index=index,
        position_m=position_m,
        look_at_m=look_at_m,
        orientation_rad=orientation_rad,
        artifacts=_resolve_artifacts(
            view_artifacts_raw, root=root, name=f"view {view_id!r} 'artifacts'"
        ),
        bs=(entry,),
    )


def parse_rf_dataset_manifest(
    data: Mapping[str, Any], *, root: Path, manifest_path: Path | None = None
) -> RFDatasetManifest:
    """Parse an already-loaded manifest mapping into an :class:`RFDatasetManifest`.

    ``root`` is the dataset directory artifact paths are resolved against;
    ``manifest_path`` defaults to ``root / MANIFEST_FILE_NAME``. Raises
    ManifestError when the manifest is malformed, unsupported or inconsistent.
    """
    if not isinstance(data, Mapping):
        raise ManifestError(f"manifest must be a JSON object, got {type(data).__name__}")
    root = Path(root)
    manifest_path = Path(manifest_path) if manifest_path is not None else root / MANIFEST_FILE_NAME

    schema_version_raw = data.get("schema_version")
    if schema_version_raw not in SUPPORTED_SCHEMA_VERSIONS:
        raise ManifestError(
            f"unsupported 'schema_version' {schema_version_raw!r}: "
            f"supported versions are {list(SUPPORTED_SCHEMA_VERSIONS)}"
        )
    schema_version = int(schema_version_raw)

    config_raw = data.get("config")
    if not isinstance(config_raw, Mapping):
        raise ManifestError(f"'config' must be a mapping, got {config_raw!r}")
    config = dict(config_raw)
    for required in ("carrier_frequency_hz", "rx_rows", "rx_cols"):
        if required not in config:
            raise ManifestError(f"'config' is missing required key {required!r}")
    carrier_frequency_hz = _require_optional_float(
        config["carrier_frequency_hz"], name="'config' key 'carrier_frequency_hz'"
    )
    if carrier_frequency_hz is None:  # pragma: no cover - guarded by the check above
        raise ManifestError("'config' key 'carrier_frequency_hz' must be a finite number")
    rx_rows = _require_int(config["rx_rows"], name="'config' key 'rx_rows'")
    rx_cols = _require_int(config["rx_cols"], name="'config' key 'rx_cols'")
    if rx_rows < 1 or rx_cols < 1:
        raise ManifestError(
            f"'config' keys 'rx_rows'/'rx_cols' must be >= 1, got {rx_rows}/{rx_cols}"
        )

    frequency_offsets_hz = _require_float_list(
        data.get("frequency_offsets_hz"), name="'frequency_offsets_hz'"
    )
    if "num_frequency_bins" in config and config["num_frequency_bins"] != len(frequency_offsets_hz):
        raise ManifestError(
            f"'config' 'num_frequency_bins' ({config['num_frequency_bins']!r}) does not match "
            f"len('frequency_offsets_hz') ({len(frequency_offsets_hz)})"
        )
    if data.get("absolute_frequencies_hz") is not None:
        absolute_frequencies_hz = _require_float_list(
            data.get("absolute_frequencies_hz"), name="'absolute_frequencies_hz'"
        )
        if len(absolute_frequencies_hz) != len(frequency_offsets_hz):
            raise ManifestError(
                f"'absolute_frequencies_hz' length ({len(absolute_frequencies_hz)}) must equal "
                f"'frequency_offsets_hz' length ({len(frequency_offsets_hz)})"
            )
    else:
        absolute_frequencies_hz = carrier_frequency_hz + frequency_offsets_hz

    raw_observation = data.get("raw_observation")
    if raw_observation is None:
        raw_observation = {}
    if not isinstance(raw_observation, Mapping):
        raise ManifestError(f"'raw_observation' must be a mapping, got {raw_observation!r}")
    expected_axis = APERTURE_CFR_AXIS_ORDER if schema_version == 3 else V2_APERTURE_CFR_AXIS_ORDER
    stored_aperture_axis_order = _require_axis_order(
        raw_observation.get("axis_order"),
        name="'raw_observation' 'axis_order'",
        default=expected_axis,
    )
    if stored_aperture_axis_order != expected_axis:
        raise ManifestError(
            f"'raw_observation' 'axis_order' {list(stored_aperture_axis_order)} does not match "
            f"expected {list(expected_axis)} for schema_version {schema_version}"
        )
    hemispheres = _require_axis_order(
        raw_observation.get("hemispheres"),
        name="'raw_observation' 'hemispheres'",
        default=tuple(HEMISPHERES),
    )

    mode = data.get("mode")
    if mode is not None and not isinstance(mode, str):
        raise ManifestError(f"'mode' must be a string, got {mode!r}")
    source_scene = data.get("source_scene")
    if source_scene is not None and not isinstance(source_scene, str):
        raise ManifestError(f"'source_scene' must be a string, got {source_scene!r}")
    delay_resolution_s = _require_optional_float(
        data.get("delay_resolution_s"), name="'delay_resolution_s'"
    )
    unambiguous_delay_s = _require_optional_float(
        data.get("unambiguous_delay_s"), name="'unambiguous_delay_s'"
    )

    camera_model = data.get("camera_model")
    ray_directions = "camera_model.npz"
    if isinstance(camera_model, Mapping) and isinstance(camera_model.get("ray_directions"), str):
        ray_directions = camera_model["ray_directions"]
    camera_model_path = root / ray_directions

    views_raw = data.get("views")
    if not isinstance(views_raw, list) or not views_raw:
        raise ManifestError(f"'views' must be a non-empty list, got {views_raw!r}")

    if schema_version == 3:
        base_stations = _parse_base_stations_v3(data.get("base_stations"))
        if raw_observation.get("bs_ids") is not None:
            bs_ids = list(raw_observation["bs_ids"])
            expected_ids = [station.bs_id for station in base_stations]
            if bs_ids != expected_ids:
                raise ManifestError(
                    f"'raw_observation' 'bs_ids' {bs_ids} do not match base station "
                    f"order {expected_ids}"
                )
        views = tuple(
            _parse_view_v3(entry, index=index, base_stations=base_stations, root=root)
            for index, entry in enumerate(views_raw)
        )
    else:
        if config.get("tx_position") is None:
            raise ManifestError(
                "'config' is missing required 3-vector 'tx_position' for schema_version 2"
            )
        if config.get("tx_look_at") is None:
            raise ManifestError(
                "'config' is missing required 3-vector 'tx_look_at' for schema_version 2"
            )
        tx_position = _require_vector(config["tx_position"], name="'config' 'tx_position'")
        tx_look_at = _require_vector(config["tx_look_at"], name="'config' 'tx_look_at'")
        base_stations = (
            BaseStationInfo(bs_id="bs_000", index=0, position_m=tx_position, look_at_m=tx_look_at),
        )
        views = tuple(
            _parse_view_v2(entry, index=index, root=root) for index, entry in enumerate(views_raw)
        )

    seen_view_ids: set[str] = set()
    for view in views:
        if view.view_id in seen_view_ids:
            raise ManifestError(f"duplicate view_id {view.view_id!r} in 'views'")
        seen_view_ids.add(view.view_id)

    path_geometry_gt = _parse_path_geometry_gt(
        data.get("path_geometry_gt"),
        root=root,
        config=config,
        schema_value=data.get("path_schema"),
    )

    if "placement" in data and not isinstance(data["placement"], Mapping):
        raise ManifestError(f"'placement' must be a mapping, got {data['placement']!r}")

    return RFDatasetManifest(
        root=root,
        manifest_path=manifest_path,
        schema_version=schema_version,
        mode=mode,
        source_scene=source_scene,
        config=config,
        carrier_frequency_hz=carrier_frequency_hz,
        frequency_offsets_hz=frequency_offsets_hz,
        absolute_frequencies_hz=absolute_frequencies_hz,
        delay_resolution_s=delay_resolution_s,
        unambiguous_delay_s=unambiguous_delay_s,
        hemispheres=hemispheres,
        stored_aperture_axis_order=stored_aperture_axis_order,
        rx_rows=rx_rows,
        rx_cols=rx_cols,
        base_stations=base_stations,
        views=views,
        camera_model_path=camera_model_path,
        path_geometry_gt=path_geometry_gt,
        raw=data,
    )


def load_rf_dataset_manifest(path: Path | str) -> RFDatasetManifest:
    """Load and parse a dataset manifest from disk.

    ``path`` may be the dataset directory or the manifest file itself (as a
    string or a :class:`Path`). Raises ManifestError for malformed manifests.
    """
    location = Path(path)
    if location.is_dir():
        root = location
        manifest_path = location / MANIFEST_FILE_NAME
    else:
        manifest_path = location
        root = location.parent
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return parse_rf_dataset_manifest(data, root=root, manifest_path=manifest_path)
