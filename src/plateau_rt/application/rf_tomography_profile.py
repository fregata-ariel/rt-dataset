"""Profiles, pose bank, `tomography` manifest section and verification of the tomography dataset
profile; Sionna-free.
"""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from plateau_rt.application import rf_tomography_io
from plateau_rt.application.rf_dataset_manifest import load_rf_dataset_manifest
from plateau_rt.application.ue_placement import (
    SavedRadioMap,
    building_exclusion_mask,
    load_radio_map,
    placement_manifest_section,
)
from plateau_rt.domain.rf_camera.camera import RFViewSpec, look_at_orientation
from plateau_rt.domain.rf_camera.paths import synthesize_cfr
from plateau_rt.domain.rf_camera.placement import (
    AGGREGATIONS,
    LOS_REFERENCES,
    CoveragePlacementSettings,
    CoverageThreshold,
    RadioMapGrid,
    plan_coverage_placement,
)
from plateau_rt.domain.rf_tomography import antenna, bank, sync
from plateau_rt.domain.rf_tomography.bank import DEFAULT_VIEW_SUBSET_SIZES

PROFILE_SCHEMA = "rf_tomo_profile/1"
BANK_METHOD = "tomography_bank"
BANK_VERSION = 1
LOS_FREE_ARTIFACT = "aperture_cfr_los_free"
ELEVATED_RING_RADIUS_M = 30.0
TOMOGRAPHY_SECTION = "tomography"
ORACLE_RTOL: dict[str, float] = {"specular": 1e-4, "refraction": 1e-4, "diffraction": 3e-2}


def _finite_float(name: str, value: Any) -> float:
    """Return ``value`` as a finite Python float, else raise ``ValueError``."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a finite float, got {value!r}") from None
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def _point3(name: str, value: Any) -> tuple[float, float, float]:
    """Return ``value`` as 3 finite Python floats, else raise ``ValueError``."""
    try:
        items = tuple(value)
    except TypeError:
        raise ValueError(f"{name} must be a length-3 sequence of finite numbers") from None
    if len(items) != 3:
        raise ValueError(f"{name} must be a length-3 sequence of finite numbers")
    try:
        point = (float(items[0]), float(items[1]), float(items[2]))
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a length-3 sequence of finite numbers") from None
    if not all(math.isfinite(v) for v in point):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return point


def _pair(name: str, value: Any) -> tuple[float, float]:
    """Return ``value`` as 2 finite Python floats, else raise ``ValueError``."""
    try:
        items = tuple(value)
    except TypeError:
        raise ValueError(f"{name} must be a length-2 sequence of finite numbers") from None
    if len(items) != 2:
        raise ValueError(f"{name} must be a length-2 sequence of finite numbers")
    try:
        pair = (float(items[0]), float(items[1]))
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a length-2 sequence of finite numbers") from None
    if not all(math.isfinite(v) for v in pair):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return pair


def _check_sizes(name: str, value: Any) -> tuple[int, ...]:
    """Return ``value`` as positive strictly increasing ints, else raise ``ValueError``."""
    try:
        items = list(value)
    except TypeError:
        raise ValueError(f"{name} must hold positive strictly increasing ints") from None
    for item in items:
        if isinstance(item, bool) or not isinstance(item, (int, np.integer)) or int(item) < 1:
            raise ValueError(f"{name} must hold positive strictly increasing ints, got {value!r}")
    ints = [int(item) for item in items]
    if any(b - a <= 0 for a, b in zip(ints, ints[1:])):
        raise ValueError(f"{name} must hold positive strictly increasing ints, got {value!r}")
    return tuple(ints)


@dataclass(frozen=True)
class CoverageProfile:
    """Radio-map coverage fill of the tomography pose bank."""

    ue_height_m: float = 1.5
    rm_size_m: tuple[float, float] = (120.0, 120.0)
    rm_cell_size_m: tuple[float, float] = (1.0, 1.0)
    rm_max_depth: int = 5
    rm_samples_per_tx: int = 100_000_000
    rm_seed: int = 42
    threshold_mode: str = "relative_to_max_db"
    threshold_value: float = 50.0
    aggregation: str = "any"
    los_fraction: float | None = None
    los_reference: str = "any"

    def validate(self) -> None:
        """Check every coverage field (ValueError otherwise)."""
        CoverageThreshold(self.threshold_mode, self.threshold_value).validate()
        if self.aggregation not in AGGREGATIONS:
            raise ValueError(f"aggregation must be one of {AGGREGATIONS}, got {self.aggregation!r}")
        if self.los_reference not in LOS_REFERENCES:
            raise ValueError(
                f"los_reference must be one of {LOS_REFERENCES}, got {self.los_reference!r}"
            )
        if self.los_fraction is not None:
            fraction = _finite_float("los_fraction", self.los_fraction)
            if not 0.0 <= fraction <= 1.0:
                raise ValueError(
                    f"los_fraction must satisfy 0 <= los_fraction <= 1, got {self.los_fraction!r}"
                )
        size = _pair("rm_size_m", self.rm_size_m)
        cell = _pair("rm_cell_size_m", self.rm_cell_size_m)
        if not (size[0] > 0.0 and size[1] > 0.0):
            raise ValueError(f"rm_size_m must be > 0, got {self.rm_size_m!r}")
        if not (cell[0] > 0.0 and cell[1] > 0.0):
            raise ValueError(f"rm_cell_size_m must be > 0, got {self.rm_cell_size_m!r}")
        _finite_float("ue_height_m", self.ue_height_m)
        if isinstance(self.rm_max_depth, bool) or not isinstance(
            self.rm_max_depth, (int, np.integer)
        ):
            raise ValueError(f"rm_max_depth must be an int >= 0, got {self.rm_max_depth!r}")
        if int(self.rm_max_depth) < 0:
            raise ValueError(f"rm_max_depth must be >= 0, got {self.rm_max_depth!r}")
        if isinstance(self.rm_samples_per_tx, bool) or not isinstance(
            self.rm_samples_per_tx, (int, np.integer)
        ):
            raise ValueError(
                f"rm_samples_per_tx must be an int >= 1, got {self.rm_samples_per_tx!r}"
            )
        if int(self.rm_samples_per_tx) < 1:
            raise ValueError(f"rm_samples_per_tx must be >= 1, got {self.rm_samples_per_tx!r}")
        if isinstance(self.rm_seed, bool) or not isinstance(self.rm_seed, (int, np.integer)):
            raise ValueError(f"rm_seed must be an int, got {self.rm_seed!r}")

    def threshold(self) -> CoverageThreshold:
        """Return the coverage candidate threshold."""
        return CoverageThreshold(self.threshold_mode, self.threshold_value)

    def grid(self, target: Sequence[float]) -> RadioMapGrid:
        """Return the radio-map grid centred on the target at UE height."""
        point = _point3("target", target)
        return RadioMapGrid(
            center_m=(point[0], point[1], float(self.ue_height_m)),
            size_m=(float(self.rm_size_m[0]), float(self.rm_size_m[1])),
            cell_size_m=(float(self.rm_cell_size_m[0]), float(self.rm_cell_size_m[1])),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serializable coverage profile (declaration order)."""
        return {
            "ue_height_m": float(self.ue_height_m),
            "rm_size_m": [float(v) for v in self.rm_size_m],
            "rm_cell_size_m": [float(v) for v in self.rm_cell_size_m],
            "rm_max_depth": int(self.rm_max_depth),
            "rm_samples_per_tx": int(self.rm_samples_per_tx),
            "rm_seed": int(self.rm_seed),
            "threshold_mode": self.threshold_mode,
            "threshold_value": float(self.threshold_value),
            "aggregation": self.aggregation,
            "los_fraction": None if self.los_fraction is None else float(self.los_fraction),
            "los_reference": self.los_reference,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CoverageProfile:
        """Rebuild a coverage profile from :meth:`to_dict` (validated)."""
        if not isinstance(data, Mapping):
            raise ValueError(f"coverage record must be a mapping, got {data!r}")
        try:
            profile = cls(
                ue_height_m=float(data["ue_height_m"]),
                rm_size_m=(float(data["rm_size_m"][0]), float(data["rm_size_m"][1])),
                rm_cell_size_m=(
                    float(data["rm_cell_size_m"][0]),
                    float(data["rm_cell_size_m"][1]),
                ),
                rm_max_depth=int(data["rm_max_depth"]),
                rm_samples_per_tx=int(data["rm_samples_per_tx"]),
                rm_seed=int(data["rm_seed"]),
                threshold_mode=str(data["threshold_mode"]),
                threshold_value=float(data["threshold_value"]),
                aggregation=str(data["aggregation"]),
                los_fraction=None if data["los_fraction"] is None else float(data["los_fraction"]),
                los_reference=str(data["los_reference"]),
            )
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise ValueError(f"bad coverage record: {exc}") from None
        profile.validate()
        return profile


@dataclass(frozen=True)
class TomographyProfile:
    """One named tomography dataset profile (pose bank, tracing and splits)."""

    name: str
    target: tuple[float, float, float]
    bs_positions: tuple[tuple[float, float, float], ...]
    num_held_out_bs: int
    bs_subset_sizes: tuple[int, ...]
    num_poses: int | None
    rings: tuple[tuple[float, float], ...]
    ring_views: int
    ring_start_azimuth_deg: float
    look_at_jitter_deg: float
    holdout_fraction: float
    min_holdout: int
    order_seeds: tuple[int, ...]
    view_subset_sizes: tuple[int, ...] = DEFAULT_VIEW_SUBSET_SIZES
    snr_db: float = 30.0
    snr_levels_db: tuple[float, ...] = (30.0, 20.0, 10.0, 0.0)
    carrier_frequency_hz: float = 3.5e9
    bandwidth_hz: float = 100e6
    num_frequency_bins: int = 128
    max_depth: int = 5
    min_bs_distance_m: float = 10.0
    building_clearance_m: float = 2.0
    min_ue_spacing_m: float = 5.0
    coverage: CoverageProfile | None = None

    def validate(self) -> None:
        """Check every profile field (ValueError otherwise)."""
        if not isinstance(self.name, str) or not self.name:
            raise ValueError(f"name must be a non-empty string, got {self.name!r}")
        _point3("target", self.target)
        try:
            stations = list(self.bs_positions)
        except TypeError:
            raise ValueError("bs_positions must hold at least one 3-element position") from None
        if len(stations) == 0:
            raise ValueError("bs_positions must hold at least one 3-element position")
        for station in stations:
            _point3("bs_positions entry", station)
        if (
            isinstance(self.num_held_out_bs, bool)
            or not isinstance(self.num_held_out_bs, (int, np.integer))
            or not 0 <= int(self.num_held_out_bs) < len(stations)
        ):
            raise ValueError(
                f"num_held_out_bs must satisfy 0 <= num_held_out_bs < {len(stations)}, "
                f"got {self.num_held_out_bs!r}"
            )
        _check_sizes("bs_subset_sizes", self.bs_subset_sizes)
        _check_sizes("view_subset_sizes", self.view_subset_sizes)
        if self.num_poses is not None and (
            isinstance(self.num_poses, bool)
            or not isinstance(self.num_poses, (int, np.integer))
            or int(self.num_poses) < 1
        ):
            raise ValueError(f"num_poses must be None or an int >= 1, got {self.num_poses!r}")
        try:
            ring_list = list(self.rings)
        except TypeError:
            raise ValueError("rings must hold at least one (radius_m, height_m) pair") from None
        if len(ring_list) == 0:
            raise ValueError("rings must hold at least one (radius_m, height_m) pair")
        for entry in ring_list:
            try:
                radius, height = float(entry[0]), float(entry[1])
            except (TypeError, ValueError, IndexError):
                raise ValueError(
                    f"rings entries must be (radius_m, height_m), got {entry!r}"
                ) from None
            if not math.isfinite(radius) or not math.isfinite(height):
                raise ValueError(f"ring (radius_m, height_m) must be finite, got {entry!r}")
            if radius <= 0.0:
                raise ValueError(f"ring radius_m must be > 0, got {entry!r}")
        if (
            isinstance(self.ring_views, bool)
            or not isinstance(self.ring_views, (int, np.integer))
            or int(self.ring_views) < 1
        ):
            raise ValueError(f"ring_views must be an int >= 1, got {self.ring_views!r}")
        _finite_float("ring_start_azimuth_deg", self.ring_start_azimuth_deg)
        jitter = _finite_float("look_at_jitter_deg", self.look_at_jitter_deg)
        if not 0.0 <= jitter < 90.0:
            raise ValueError(
                "look_at_jitter_deg must satisfy 0 <= look_at_jitter_deg < 90, "
                f"got {self.look_at_jitter_deg!r}"
            )
        holdout = _finite_float("holdout_fraction", self.holdout_fraction)
        if not 0.0 <= holdout < 1.0:
            raise ValueError(
                "holdout_fraction must satisfy 0 <= holdout_fraction < 1, "
                f"got {self.holdout_fraction!r}"
            )
        if (
            isinstance(self.min_holdout, bool)
            or not isinstance(self.min_holdout, (int, np.integer))
            or int(self.min_holdout) < 0
        ):
            raise ValueError(f"min_holdout must be an int >= 0, got {self.min_holdout!r}")
        try:
            seeds = list(self.order_seeds)
        except TypeError:
            raise ValueError(
                "order_seeds must be a non-empty sequence without duplicates"
            ) from None
        if len(seeds) == 0 or len(set(int(s) for s in seeds)) != len(seeds):
            raise ValueError("order_seeds must hold distinct values without duplicates")
        for seed in seeds:
            if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
                raise ValueError(f"order_seeds must hold ints >= 0, got {self.order_seeds!r}")
        _finite_float("snr_db", self.snr_db)
        try:
            levels = list(self.snr_levels_db)
        except TypeError:
            raise ValueError("snr_levels_db must hold finite floats") from None
        for level in levels:
            _finite_float("snr_levels_db entry", level)
        if (
            isinstance(self.num_frequency_bins, bool)
            or not isinstance(self.num_frequency_bins, (int, np.integer))
            or int(self.num_frequency_bins) < 2
        ):
            raise ValueError(
                f"num_frequency_bins must be an int >= 2, got {self.num_frequency_bins!r}"
            )
        if not float(self.carrier_frequency_hz) > 0.0 or not math.isfinite(
            float(self.carrier_frequency_hz)
        ):
            raise ValueError(f"carrier_frequency_hz must be > 0, got {self.carrier_frequency_hz!r}")
        if not float(self.bandwidth_hz) > 0.0 or not math.isfinite(float(self.bandwidth_hz)):
            raise ValueError(f"bandwidth_hz must be > 0, got {self.bandwidth_hz!r}")
        if (
            isinstance(self.max_depth, bool)
            or not isinstance(self.max_depth, (int, np.integer))
            or int(self.max_depth) < 0
        ):
            raise ValueError(f"max_depth must be an int >= 0, got {self.max_depth!r}")
        for label in ("min_bs_distance_m", "building_clearance_m", "min_ue_spacing_m"):
            if _finite_float(label, getattr(self, label)) < 0.0:
                raise ValueError(f"{label} must be >= 0, got {getattr(self, label)!r}")
        if (self.coverage is None) != (self.num_poses is None):
            raise ValueError("coverage and num_poses must either both be set or both be None")
        if self.coverage is not None:
            if not isinstance(self.coverage, CoverageProfile):
                raise ValueError(f"coverage must be a CoverageProfile, got {self.coverage!r}")
            self.coverage.validate()

    def with_elevated_heights(self, heights_m: Sequence[float]) -> TomographyProfile:
        """Return a copy with one extra 30 m ring per elevated UE height."""
        try:
            heights = list(heights_m)
        except TypeError:
            raise ValueError(f"heights_m must hold finite heights > 0, got {heights_m!r}") from None
        extra: list[tuple[float, float]] = []
        for height in heights:
            try:
                value = float(height)
            except (TypeError, ValueError):
                raise ValueError(
                    f"heights_m must hold finite heights > 0, got {heights_m!r}"
                ) from None
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"heights_m must hold finite heights > 0, got {heights_m!r}")
            extra.append((ELEVATED_RING_RADIUS_M, value))
        return dataclasses.replace(self, rings=tuple(self.rings) + tuple(extra))

    def dataset_config_kwargs(self, variant: str, *, los_free_trace: bool = True) -> dict[str, Any]:
        """Return the ``RFMultiViewConfig`` fields of one mechanism variant."""
        try:
            mechanism = bank.MECHANISM_VARIANTS[variant]
        except KeyError:
            raise ValueError(
                f"unknown mechanism variant {variant!r}; "
                f"expected one of {list(bank.MECHANISM_VARIANTS)}"
            ) from None
        return {
            "carrier_frequency_hz": float(self.carrier_frequency_hz),
            "bandwidth_hz": float(self.bandwidth_hz),
            "num_frequency_bins": int(self.num_frequency_bins),
            "tx_positions": tuple(tuple(float(v) for v in p) for p in self.bs_positions),
            "tx_look_at": tuple(float(v) for v in self.target),
            "tx_look_ats": None,
            "rx_rows": 8,
            "rx_cols": 8,
            "tx_pattern": "tr38901",
            "polarization": "V",
            "max_depth": int(self.max_depth),
            "synthetic_array": True,
            "seed": 42,
            "specular_reflection": mechanism.specular_reflection,
            "refraction": mechanism.refraction,
            "diffraction": mechanism.diffraction,
            "los_free_trace": bool(los_free_trace),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serializable profile (declaration order, tuples as lists)."""
        return {
            "name": self.name,
            "target": [float(v) for v in self.target],
            "bs_positions": [[float(v) for v in p] for p in self.bs_positions],
            "num_held_out_bs": int(self.num_held_out_bs),
            "bs_subset_sizes": [int(v) for v in self.bs_subset_sizes],
            "num_poses": None if self.num_poses is None else int(self.num_poses),
            "rings": [[float(r), float(h)] for r, h in self.rings],
            "ring_views": int(self.ring_views),
            "ring_start_azimuth_deg": float(self.ring_start_azimuth_deg),
            "look_at_jitter_deg": float(self.look_at_jitter_deg),
            "holdout_fraction": float(self.holdout_fraction),
            "min_holdout": int(self.min_holdout),
            "order_seeds": [int(v) for v in self.order_seeds],
            "view_subset_sizes": [int(v) for v in self.view_subset_sizes],
            "snr_db": float(self.snr_db),
            "snr_levels_db": [float(v) for v in self.snr_levels_db],
            "carrier_frequency_hz": float(self.carrier_frequency_hz),
            "bandwidth_hz": float(self.bandwidth_hz),
            "num_frequency_bins": int(self.num_frequency_bins),
            "max_depth": int(self.max_depth),
            "min_bs_distance_m": float(self.min_bs_distance_m),
            "building_clearance_m": float(self.building_clearance_m),
            "min_ue_spacing_m": float(self.min_ue_spacing_m),
            "coverage": None if self.coverage is None else self.coverage.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> TomographyProfile:
        """Rebuild a profile from :meth:`to_dict` (validated)."""
        if not isinstance(data, Mapping):
            raise ValueError(f"profile record must be a mapping, got {data!r}")
        try:
            coverage_data = data["coverage"]
            profile = cls(
                name=str(data["name"]),
                target=(
                    float(data["target"][0]),
                    float(data["target"][1]),
                    float(data["target"][2]),
                ),
                bs_positions=tuple(
                    (float(p[0]), float(p[1]), float(p[2])) for p in data["bs_positions"]
                ),
                num_held_out_bs=int(data["num_held_out_bs"]),
                bs_subset_sizes=tuple(int(v) for v in data["bs_subset_sizes"]),
                num_poses=None if data["num_poses"] is None else int(data["num_poses"]),
                rings=tuple((float(r[0]), float(r[1])) for r in data["rings"]),
                ring_views=int(data["ring_views"]),
                ring_start_azimuth_deg=float(data["ring_start_azimuth_deg"]),
                look_at_jitter_deg=float(data["look_at_jitter_deg"]),
                holdout_fraction=float(data["holdout_fraction"]),
                min_holdout=int(data["min_holdout"]),
                order_seeds=tuple(int(v) for v in data["order_seeds"]),
                view_subset_sizes=tuple(int(v) for v in data["view_subset_sizes"]),
                snr_db=float(data["snr_db"]),
                snr_levels_db=tuple(float(v) for v in data["snr_levels_db"]),
                carrier_frequency_hz=float(data["carrier_frequency_hz"]),
                bandwidth_hz=float(data["bandwidth_hz"]),
                num_frequency_bins=int(data["num_frequency_bins"]),
                max_depth=int(data["max_depth"]),
                min_bs_distance_m=float(data["min_bs_distance_m"]),
                building_clearance_m=float(data["building_clearance_m"]),
                min_ue_spacing_m=float(data["min_ue_spacing_m"]),
                coverage=None
                if coverage_data is None
                else CoverageProfile.from_dict(coverage_data),
            )
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise ValueError(f"bad tomography profile record: {exc}") from None
        profile.validate()
        return profile


MOCK_CITY_TARGET = (0.0, 0.0, 8.0)
MOCK_CITY_BS_POSITIONS = (
    (-70.0, 5.0, 25.0),
    (60.0, -40.0, 20.0),
    (15.0, 75.0, 30.0),
    (55.0, 45.0, 15.0),
    (-45.0, -60.0, 20.0),
)
PROFILES: dict[str, TomographyProfile] = {
    "ci": TomographyProfile(
        name="ci",
        target=MOCK_CITY_TARGET,
        bs_positions=MOCK_CITY_BS_POSITIONS[:2],
        num_held_out_bs=0,
        bs_subset_sizes=(1, 2),
        num_poses=None,
        rings=((40.0, 1.5),),
        ring_views=8,
        ring_start_azimuth_deg=0.0,
        look_at_jitter_deg=0.0,
        holdout_fraction=0.25,
        min_holdout=2,
        order_seeds=(0,),
        coverage=None,
    ),
    "full": TomographyProfile(
        name="full",
        target=MOCK_CITY_TARGET,
        bs_positions=MOCK_CITY_BS_POSITIONS,
        num_held_out_bs=1,
        bs_subset_sizes=(1, 2, 4),
        num_poses=64,
        rings=((20.0, 1.5), (30.0, 1.5), (40.0, 1.5)),
        ring_views=8,
        ring_start_azimuth_deg=22.5,
        look_at_jitter_deg=15.0,
        holdout_fraction=0.25,
        min_holdout=4,
        order_seeds=(0, 1, 2, 3, 4),
        coverage=CoverageProfile(),
    ),
}


@dataclass(frozen=True)
class PoseBank:
    """One built pose bank: views, per-view placements and the manifest record."""

    views: list[RFViewSpec]
    view_placements: list[dict[str, Any]]
    record: dict[str, Any]


def _as_seed(name: str, value: Any) -> int:
    """Return ``value`` as an int >= 0, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an int >= 0, got {value!r}")
    if int(value) < 0:
        raise ValueError(f"{name} must be an int >= 0, got {value!r}")
    return int(value)


def build_pose_bank(
    profile: TomographyProfile,
    *,
    placement_seed: int,
    saved_radio_map: SavedRadioMap | None = None,
    dataset_dir: Path | str | None = None,
    radio_map_source: str = "computed",
    radio_map_origin: str | None = None,
) -> PoseBank:
    """Build the deterministic pose bank of ``profile`` (rings plus coverage fill)."""
    profile.validate()
    placement_seed = _as_seed("placement_seed", placement_seed)
    ring_list = bank.ring_poses(
        profile.target, profile.rings, profile.ring_views, profile.ring_start_azimuth_deg
    )
    if profile.coverage is not None:
        if saved_radio_map is None or dataset_dir is None:
            raise ValueError("a coverage profile needs saved_radio_map and dataset_dir")
        exclusion = building_exclusion_mask(
            saved_radio_map.indoor_mask,
            saved_radio_map.grid,
            clearance_m=profile.building_clearance_m,
        )
    else:
        exclusion = None

    kept: list[bank.RingPose] = []
    dropped: list[dict[str, Any]] = []
    bs_array = np.asarray(profile.bs_positions, dtype=np.float64)
    grid_shape: tuple[int, int] | None = None
    grid_origin: tuple[float, float] | None = None
    grid_cell: tuple[float, float] | None = None
    if exclusion is not None and saved_radio_map is not None:
        grid = saved_radio_map.grid
        ny, nx = grid.shape
        grid_shape = (ny, nx)
        grid_origin = (
            float(grid.center_m[0]) - float(grid.size_m[0]) / 2.0,
            float(grid.center_m[1]) - float(grid.size_m[1]) / 2.0,
        )
        grid_cell = (float(grid.cell_size_m[0]), float(grid.cell_size_m[1]))
    for pose in ring_list:
        position = np.asarray(pose.position, dtype=np.float64)
        reason: str | None = None
        if np.any(np.linalg.norm(bs_array - position[None, :], axis=1) < profile.min_bs_distance_m):
            reason = "too_close_to_bs"
        elif exclusion is not None:
            assert grid_shape is not None and grid_origin is not None and grid_cell is not None
            ny, nx = grid_shape
            ix = math.floor((float(position[0]) - grid_origin[0]) / grid_cell[0])
            iy = math.floor((float(position[1]) - grid_origin[1]) / grid_cell[1])
            if not 0 <= ix < nx or not 0 <= iy < ny:
                reason = "outside_grid"
            elif bool(exclusion[iy, ix]):
                reason = "excluded"
        if reason is None:
            kept.append(pose)
        else:
            dropped.append(
                {
                    "ring_radius_m": pose.radius_m,
                    "height_m": pose.height_m,
                    "azimuth_deg": pose.azimuth_deg,
                    "ring_index": pose.ring_index,
                    "reason": reason,
                }
            )
    n_ring = len(kept)
    if profile.coverage is None:
        n_cov = 0
        if profile.num_poses is not None and int(profile.num_poses) != n_ring:
            raise ValueError(
                f"ring-only bank has {n_ring} poses, but num_poses={profile.num_poses}"
            )
        cov_record: dict[str, Any] | None = None
        coverage_positions: list[tuple[float, float, float]] = []
    else:
        assert profile.num_poses is not None and saved_radio_map is not None
        assert dataset_dir is not None
        n_cov = int(profile.num_poses) - n_ring
        if n_cov < 1:
            raise ValueError(
                f"coverage fill needs num_poses ({profile.num_poses}) > ring poses ({n_ring})"
            )
        cov = profile.coverage
        assert cov is not None
        settings = CoveragePlacementSettings(
            num_views=n_cov,
            placement_seed=placement_seed,
            threshold=cov.threshold(),
            aggregation=cov.aggregation,
            min_bs_distance_m=profile.min_bs_distance_m,
            min_spacing_m=profile.min_ue_spacing_m,
            jitter_fraction=0.0,
            orientation_policy="look_at_target",
            target=(float(profile.target[0]), float(profile.target[1]), float(profile.target[2])),
            los_fraction=cov.los_fraction,
            los_reference=cov.los_reference,
        )
        centers = saved_radio_map.grid.cell_centers()[..., :2]
        ring_xy = np.asarray(
            [[pose.position[0], pose.position[1]] for pose in kept], dtype=np.float64
        )
        ring_block = np.zeros(saved_radio_map.grid.shape, dtype=bool)
        if ring_xy.shape[0]:
            dist_xy = np.linalg.norm(centers[:, :, None, :] - ring_xy[None, None, :, :], axis=-1)
            ring_block = np.any(dist_xy < profile.min_ue_spacing_m, axis=-1)
        placement = plan_coverage_placement(
            saved_radio_map.path_gain,
            saved_radio_map.grid,
            settings,
            exclusion_mask=(exclusion | ring_block) if exclusion is not None else ring_block,
            bs_positions=profile.bs_positions,
            los_mask=saved_radio_map.los_mask,
        )
        cov_record = placement_manifest_section(
            placement,
            saved=saved_radio_map,
            dataset_dir=Path(dataset_dir),
            radio_map_source=radio_map_source,
            radio_map_origin=radio_map_origin,
            building_clearance_m=profile.building_clearance_m,
        )
        cov_record["exclusion"]["ring_exclusion_radius_m"] = float(profile.min_ue_spacing_m)
        for j, entry in enumerate(cov_record["views"]):
            entry["view_id"] = f"ue_{n_ring + j:06d}"
        coverage_positions = [view.position for view in placement.views]
    bank_positions = np.asarray(
        [[*pose.position] for pose in kept] + [list(p) for p in coverage_positions],
        dtype=np.float64,
    )
    count = n_ring + n_cov
    if profile.look_at_jitter_deg > 0.0:
        jitter_rng: np.random.Generator | None = np.random.default_rng(
            np.random.SeedSequence([placement_seed, bank.JITTER_STREAM_TAG])
        )
    else:
        jitter_rng = None
    look_ats, orientations, jitter = bank.jitter_look_at(
        bank_positions, profile.target, max_deg=profile.look_at_jitter_deg, rng=jitter_rng
    )
    views_out: list[RFViewSpec] = []
    placements_out: list[dict[str, Any]] = []
    for i in range(count):
        pos = (
            float(bank_positions[i, 0]),
            float(bank_positions[i, 1]),
            float(bank_positions[i, 2]),
        )
        views_out.append(
            RFViewSpec(
                view_id=f"ue_{i:06d}",
                position=pos,
                look_at=look_ats[i],
                orientation=orientations[i],
            )
        )
        jitter_pair = [float(jitter[i, 0]), float(jitter[i, 1])]
        if i < n_ring:
            pose = kept[i]
            placements_out.append(
                {
                    "bank_index": i,
                    "source": "ring",
                    "ring_radius_m": pose.radius_m,
                    "height_m": pose.height_m,
                    "azimuth_deg": pose.azimuth_deg,
                    "ring_index": pose.ring_index,
                    "look_at_jitter_deg": jitter_pair,
                }
            )
        else:
            j = i - n_ring
            assert cov_record is not None
            entry = cov_record["views"][j]
            placements_out.append(
                {
                    "bank_index": i,
                    "source": "coverage",
                    "coverage_index": j,
                    "cell_index": entry["cell_index"],
                    "path_gain_db": entry["path_gain_db"],
                    "per_bs_path_gain_db": entry["per_bs_path_gain_db"],
                    "los_bs": entry["los_bs"],
                    "los": entry["los"],
                    "look_at_jitter_deg": jitter_pair,
                }
            )
    record = {
        "method": BANK_METHOD,
        "bank_version": BANK_VERSION,
        "placement_seed": placement_seed,
        "profile": profile.to_dict(),
        "target": [float(v) for v in profile.target],
        "num_views": count,
        "sources": {"ring": n_ring, "coverage": n_cov},
        "rings": {"requested": len(ring_list), "kept": n_ring, "dropped": dropped},
        "coverage": cov_record,
        "look_at_jitter": {
            "max_deg": float(profile.look_at_jitter_deg),
            "distribution": (
                "uniform in [-max_deg, max_deg] on the azimuth and elevation "
                "of the direction to the target"
            ),
            "rng": "SeedSequence([placement_seed, JITTER_STREAM_TAG])",
        },
    }
    json.dumps(record, allow_nan=False)
    return PoseBank(views=views_out, view_placements=placements_out, record=record)


def replan_pose_bank(dataset_dir: Path | str) -> PoseBank:
    """Rebuild the pose bank recorded in a dataset manifest (saved map + seeds)."""
    root = Path(dataset_dir)
    manifest = load_rf_dataset_manifest(root)
    placement = manifest.placement
    if not isinstance(placement, Mapping) or placement.get("method") != BANK_METHOD:
        raise ValueError(f"{root}/dataset_manifest.json has no {BANK_METHOD!r} placement section")
    try:
        profile = TomographyProfile.from_dict(placement["profile"])
        placement_seed = _as_seed("placement_seed", placement["placement_seed"])
        coverage_record = placement["coverage"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"malformed placement section in {root}: {exc}") from None
    saved: SavedRadioMap | None = None
    source = "computed"
    origin: str | None = None
    if profile.coverage is not None:
        if not isinstance(coverage_record, Mapping):
            raise ValueError(f"malformed placement section in {root}: no coverage record")
        try:
            radio_map_record = coverage_record["radio_map"]
            metadata_relative = radio_map_record["metadata"]
            source = str(radio_map_record.get("source", "computed"))
            raw_origin = radio_map_record.get("origin")
            origin = None if raw_origin is None else str(raw_origin)
        except (KeyError, TypeError) as exc:
            raise ValueError(f"malformed placement section in {root}: {exc}") from None
        saved = load_radio_map(root / metadata_relative)
    return build_pose_bank(
        profile,
        placement_seed=placement_seed,
        saved_radio_map=saved,
        dataset_dir=root,
        radio_map_source=source,
        radio_map_origin=origin,
    )


def _require_bank_placement(dataset_dir: Path | str) -> tuple[Any, Mapping[str, Any]]:
    """Load the manifest and return it with its validated bank placement record."""
    manifest = load_rf_dataset_manifest(dataset_dir)
    placement = manifest.placement
    if not isinstance(placement, Mapping) or placement.get("method") != BANK_METHOD:
        raise ValueError(
            f"{Path(dataset_dir)}/dataset_manifest.json has no {BANK_METHOD!r} placement section"
        )
    return manifest, placement


def build_tomography_section(
    dataset_dir: Path | str, *, profile: TomographyProfile, variant: str, split_seed: int
) -> dict[str, Any]:
    """Build the ``tomography`` manifest section by deterministic post-processing."""
    root = Path(dataset_dir)
    profile.validate()
    split_seed = _as_seed("split_seed", split_seed)
    try:
        mechanism = bank.MECHANISM_VARIANTS[variant]
    except KeyError:
        raise ValueError(
            f"unknown mechanism variant {variant!r}; "
            f"expected one of {list(bank.MECHANISM_VARIANTS)}"
        ) from None
    manifest, placement = _require_bank_placement(root)
    if manifest.num_bs != len(profile.bs_positions):
        raise ValueError(
            f"manifest has {manifest.num_bs} base stations, "
            f"but the profile has {len(profile.bs_positions)}"
        )
    for station, wanted in zip(manifest.base_stations, profile.bs_positions):
        for axis, (got, want) in enumerate(zip(station.position_m, wanted)):
            if abs(float(got) - float(want)) > 1e-9:
                raise ValueError(
                    f"manifest base station {station.bs_id!r} position_m[{axis}]={float(got)!r} "
                    f"differs from the profile {float(want)!r} by more than 1e-9 m"
                )
    config = manifest.config
    for key in ("specular_reflection", "refraction", "diffraction"):
        if key in config and bool(config[key]) != getattr(mechanism, key):
            raise ValueError(
                f"manifest config {key}={config[key]!r} does not match variant {variant!r}"
            )
    y_clean = np.stack([manifest.load_aperture_cfr(view) for view in manifest.views]).astype(
        np.complex128, copy=False
    )
    flags = [LOS_FREE_ARTIFACT in view.artifacts for view in manifest.views]
    y_los_free: np.ndarray | None = None
    if any(flags):
        if not all(flags):
            raise ValueError(
                f"partial {LOS_FREE_ARTIFACT!r} artifacts: "
                f"{sum(flags)}/{len(flags)} views carry the oracle trace"
            )
        los_free_stack = np.stack(
            [np.load(view.artifacts[LOS_FREE_ARTIFACT]) for view in manifest.views]
        ).astype(np.complex128, copy=False)
        if los_free_stack.shape != y_clean.shape:
            raise ValueError(
                f"{LOS_FREE_ARTIFACT!r} shape {los_free_stack.shape} does not match "
                f"aperture_cfr shape {y_clean.shape}"
            )
        y_los_free = los_free_stack
    los_visible, los_source = rf_tomography_io.los_visibility(manifest)
    num_views, num_bs = int(y_clean.shape[0]), int(y_clean.shape[1])
    train_v, held_v = bank.split_bank(
        num_views,
        holdout_fraction=profile.holdout_fraction,
        min_holdout=profile.min_holdout,
        seed=split_seed,
    )
    train_b, held_b = bank.bs_split(num_bs, profile.num_held_out_bs)
    subsets = bank.bs_subsets(train_b, profile.bs_subset_sizes)
    orders = bank.nested_training_orders(train_v, profile.order_seeds)
    sizes = bank.view_subset_sizes(len(train_v), profile.view_subset_sizes)
    capture_mask = (
        np.isin(np.arange(num_views), train_v)[:, None]
        & np.isin(np.arange(num_bs), train_b)[None, :]
    )
    noise = bank.noise_reference(
        y_clean, los_visible, capture_mask, profile.snr_db, Y_los_free=y_los_free
    )
    raw = manifest.raw
    raw_observation = raw.get("raw_observation") if isinstance(raw, Mapping) else None
    rx_pattern = None
    if isinstance(raw_observation, Mapping):
        rx_pattern = raw_observation.get("rx_element_pattern")
    base_stations = []
    for station in manifest.base_stations:
        pos = (
            float(station.position_m[0]),
            float(station.position_m[1]),
            float(station.position_m[2]),
        )
        tgt = (
            float(station.look_at_m[0]),
            float(station.look_at_m[1]),
            float(station.look_at_m[2]),
        )
        base_stations.append(
            {
                "bs_id": station.bs_id,
                "position_m": [pos[0], pos[1], pos[2]],
                "look_at_m": [tgt[0], tgt[1], tgt[2]],
                "orientation_rad": list(look_at_orientation(pos, tgt)),
                "world_from_local_rotation": antenna.bs_orientation(pos, tgt).tolist(),
            }
        )
    view_ids = [view.view_id for view in manifest.views]
    section = {
        "schema": PROFILE_SCHEMA,
        "profile": profile.name,
        "variant": variant,
        "mechanism": {
            **mechanism.to_dict(),
            "max_depth": int(config["max_depth"]),
            "synthetic_array": bool(config.get("synthetic_array", True)),
            "seed": int(config.get("seed", 42)),
        },
        "seeds": {
            "placement_seed": int(placement["placement_seed"]),
            "split_seed": split_seed,
            "order_seeds": [int(v) for v in profile.order_seeds],
        },
        "bank": {
            "num_views": num_views,
            "view_ids": view_ids,
            "sources": dict(placement["sources"]),
        },
        "splits": {
            "holdout_fraction": float(profile.holdout_fraction),
            "min_holdout": int(profile.min_holdout),
            "rng": (
                "SeedSequence([split_seed, SPLIT_STREAM_TAG]).permutation(num_views)[:n_hold] "
                "are held out"
            ),
            "train_views": [int(v) for v in train_v],
            "held_out_views": [int(v) for v in held_v],
            "train_bs": [int(v) for v in train_b],
            "held_out_bs": [int(v) for v in held_b],
            "bs_subsets": {str(size): [int(v) for v in subset] for size, subset in subsets.items()},
            "view_subset_sizes": [int(v) for v in sizes],
            "nested_view_orders": {str(s): [int(v) for v in order] for s, order in orders.items()},
            "gauge_references": {
                str(s): [int(order[0]), int(train_b[0])] for s, order in orders.items()
            },
        },
        "los_visible": [[bool(v) for v in row] for row in los_visible],
        "los_visible_source": los_source,
        "noise": {
            "snr_db": float(profile.snr_db),
            "definition": (
                "sigma2 = p_ref / 10**(snr_db / 10); p_ref = sync.capture_power "
                "(hemisphere powers summed, mean over row, col, bin) of the lower-median-power "
                "LoS-visible training capture (sync.reference_power)"
            ),
            "reference_set": "los_visible & train_views x train_bs",
            "p_ref": float(noise.p_ref),
            "c_ref": [int(noise.c_ref[0]), int(noise.c_ref[1])],
            "sigma2": float(noise.sigma2),
            "los_fallback": bool(noise.los_fallback),
            "sigma2_by_snr_db": {
                f"{float(level):g}": float(noise.p_ref) / 10.0 ** (float(level) / 10.0)
                for level in profile.snr_levels_db
            },
            "capture_power": noise.capture_power.tolist(),
            "expected_snr_db": noise.expected_snr_db.tolist(),
            "scatter_power": None if noise.scatter_power is None else noise.scatter_power.tolist(),
            "expected_scatter_snr_db": None
            if noise.expected_scatter_snr_db is None
            else noise.expected_scatter_snr_db.tolist(),
        },
        "antenna": {
            "tx_pattern": config["tx_pattern"],
            "tx_polarization": config["polarization"],
            "tx_array": "single element (1x1 PlanarArray)",
            "rx_pattern": rx_pattern,
            "rx_polarization": config["polarization"],
            "orientation_rule": "Transmitter(look_at=...): local +x toward look_at, roll 0",
            "base_stations": base_stations,
        },
        "oracle_los_free": None
        if y_los_free is None
        else {
            "artifact": LOS_FREE_ARTIFACT,
            "trace": "PathSolver(los=False) with the same mechanism flags, max_depth and seed",
            "definition": (
                "aperture_cfr without the direct (zero-interaction) paths, up to GPU tracing noise"
            ),
        },
        "hashes": {
            "algorithm": "sha256",
            "aperture_cfr": {
                view.view_id: rf_tomography_io.sha256_file(view.aperture_cfr_path)
                for view in manifest.views
            },
            "aperture_cfr_los_free": None
            if y_los_free is None
            else {
                view.view_id: rf_tomography_io.sha256_file(view.artifacts[LOS_FREE_ARTIFACT])
                for view in manifest.views
            },
            "path_geometry_gt": None
            if manifest.path_geometry_gt is None
            else rf_tomography_io.sha256_file(manifest.path_geometry_gt.path),
            "path_schema": None
            if manifest.path_geometry_gt is None or manifest.path_geometry_gt.schema_path is None
            else rf_tomography_io.sha256_file(manifest.path_geometry_gt.schema_path),
            "camera_model": rf_tomography_io.sha256_file(manifest.camera_model_path),
        },
    }
    return rf_tomography_io.to_jsonable(section)


def write_tomography_section(dataset_dir: Path | str, section: Mapping[str, Any]) -> Path:
    """Append (or replace) the ``tomography`` section of the manifest JSON."""
    root = Path(dataset_dir)
    manifest_path = root / "dataset_manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw[TOMOGRAPHY_SECTION] = rf_tomography_io.to_jsonable(dict(section))
    manifest_path.write_text(json.dumps(raw, indent=2, allow_nan=False), encoding="utf-8")
    return manifest_path


def load_tomography_section(dataset_dir: Path | str) -> dict[str, Any]:
    """Return the stored ``tomography`` section (ValueError when absent or stale)."""
    manifest = load_rf_dataset_manifest(dataset_dir)
    section = manifest.tomography
    if not isinstance(section, Mapping):
        raise ValueError(
            f"{Path(dataset_dir)}/dataset_manifest.json has no {TOMOGRAPHY_SECTION!r} section"
        )
    if section.get("schema") != PROFILE_SCHEMA:
        raise ValueError(
            f"{TOMOGRAPHY_SECTION!r} schema {section.get('schema')!r} "
            f"does not match {PROFILE_SCHEMA!r}"
        )
    return dict(section)


def _normalised(value: Any) -> Any:
    """Round-trip ``value`` through JSON for order-insensitive comparison."""
    return json.loads(json.dumps(rf_tomography_io.to_jsonable(value), allow_nan=False))


def verify_tomography_dataset(
    dataset_dir: Path | str, *, oracle_rtol: float | None = None
) -> dict[str, Any]:
    """Verify a tomography dataset; raise one ``ValueError`` listing every failure."""
    root = Path(dataset_dir)
    failures: list[str] = []
    try:
        section = load_tomography_section(root)
    except ValueError as exc:
        raise ValueError(str(exc)) from None
    manifest = load_rf_dataset_manifest(root)
    placement = manifest.placement
    if not isinstance(placement, Mapping) or placement.get("method") != BANK_METHOD:
        failures.append(f"placement method is not {BANK_METHOD!r}")
        profile = None
    else:
        try:
            profile = TomographyProfile.from_dict(placement["profile"])
        except ValueError as exc:
            failures.append(f"placement profile is invalid: {exc}")
            profile = None
    if profile is not None:
        try:
            recomputed = build_tomography_section(
                root,
                profile=profile,
                variant=section["variant"],
                split_seed=int(section["seeds"]["split_seed"]),
            )
        except (ValueError, KeyError, TypeError) as exc:
            failures.append(f"tomography section rebuild failed: {exc}")
        else:
            stored = _normalised(section)
            fresh = _normalised(recomputed)
            differing = [key for key in fresh if key not in stored or stored[key] != fresh[key]] + [
                key for key in stored if key not in fresh
            ]
            if differing:
                failures.append(
                    "stored tomography section differs from the recomputed one "
                    f"(keys: {sorted(differing)})"
                )
        try:
            replanned = replan_pose_bank(root)
        except ValueError as exc:
            failures.append(f"pose bank replan failed: {exc}")
            replanned = None
        else:
            raw_views = manifest.raw["views"] if isinstance(manifest.raw, Mapping) else []
            for index, view in enumerate(replanned.views):
                entry = raw_views[index] if index < len(raw_views) else {}
                for label, got, want in (
                    ("view_id", view.view_id, entry.get("view_id")),
                    ("position_m", list(view.position), entry.get("position_m")),
                    ("look_at_m", list(view.look_at), entry.get("look_at_m")),
                ):
                    if _normalised(got) != _normalised(want):
                        failures.append(f"replanned view {index} {label} differs from the manifest")
                got_ori = np.asarray(view.orientation, dtype=np.float64)
                want_ori = np.asarray(entry.get("orientation_rad"), dtype=np.float64)
                if got_ori.shape != (3,) or want_ori.shape != (3,):
                    failures.append(f"replanned view {index} orientation_rad is malformed")
                elif not bool(np.all(np.abs(got_ori - want_ori) <= 1e-12)):
                    failures.append(
                        f"replanned view {index} orientation_rad differs from the manifest"
                    )
                stored_placement = entry.get("placement")
                if _normalised(replanned.view_placements[index]) != _normalised(stored_placement):
                    failures.append(f"replanned view {index} placement differs from the manifest")
            if _normalised(replanned.record) != _normalised(placement):
                failures.append("replanned placement record differs from the manifest")
    try:
        resolved_rtol = (
            float(oracle_rtol)
            if oracle_rtol is not None
            else float(ORACLE_RTOL[section["variant"]])
        )
    except (KeyError, TypeError, ValueError):
        failures.append(f"unknown tomography variant {section.get('variant')!r}")
        resolved_rtol = float(ORACLE_RTOL["refraction"])
    if not math.isfinite(resolved_rtol) or resolved_rtol < 0.0:
        failures.append(f"oracle_rtol must be finite and >= 0, got {oracle_rtol!r}")
    num_views = manifest.num_views
    num_bs = manifest.num_bs
    los_visible, _ = rf_tomography_io.los_visibility(manifest)
    los_count = int(np.count_nonzero(los_visible))
    max_oracle_residual: float | None = None
    min_los_share: float | None = None
    oracle = section.get("oracle_los_free")
    if oracle is not None and profile is not None:
        y_clean = np.stack([manifest.load_aperture_cfr(view) for view in manifest.views]).astype(
            np.complex128, copy=False
        )
        flags = [LOS_FREE_ARTIFACT in view.artifacts for view in manifest.views]
        if not all(flags):
            failures.append(f"partial {LOS_FREE_ARTIFACT!r} artifacts in an oracle dataset")
        else:
            y_free = np.stack(
                [np.load(view.artifacts[LOS_FREE_ARTIFACT]) for view in manifest.views]
            ).astype(np.complex128, copy=False)
            y_los: np.ndarray | None = None
            try:
                if manifest.path_geometry_gt is None:
                    raise ValueError("dataset has no path_geometry_gt")
                arrays = manifest.path_geometry_gt.load_arrays()
                a_gt = np.asarray(arrays["a_baseband"])
                tau_gt = np.asarray(arrays["tau"])
                valid_gt = np.asarray(arrays["valid"], dtype=bool)
                num_gt = np.asarray(arrays["num_interactions"])
                los = valid_gt & (num_gt == 0)
                a = np.where(los[:, :, None, None, None, :], a_gt, 0)
                tau = np.where(los, tau_gt, -1.0)[:, :, None, None, None, :]
                y_los = synthesize_cfr(a, tau, manifest.frequency_offsets_hz)
            except (ValueError, OSError, KeyError) as exc:
                failures.append(f"oracle los-free: path geometry GT unavailable: {exc}")
                y_los = None
            if y_los is not None:
                denom = sync.capture_power(y_clean)
                numer = sync.capture_power(y_clean - y_free - y_los)
                los_power = sync.capture_power(y_los)
                residuals: list[float] = []
                shares: list[float] = []
                for v in range(num_views):
                    for b in range(num_bs):
                        if not denom[v, b] > 0.0:
                            continue
                        residual = float(numer[v, b] / denom[v, b])
                        residuals.append(residual)
                        if bool(los_visible[v, b]):
                            shares.append(float(los_power[v, b] / denom[v, b]))
                        if not residual <= resolved_rtol:
                            failures.append(
                                f"oracle los-free: capture ({v}, {b}) residual "
                                f"{residual:.3g} above oracle_rtol {resolved_rtol:.3g}"
                            )
                max_oracle_residual = max(residuals) if residuals else None
                min_los_share = min(shares) if shares else None
    if failures:
        raise ValueError("\n".join(failures))
    assert profile is not None
    train_views = [int(v) for v in section["splits"]["train_views"]]
    held_views = [int(v) for v in section["splits"]["held_out_views"]]
    train_bs = [int(v) for v in section["splits"]["train_bs"]]
    held_bs = [int(v) for v in section["splits"]["held_out_bs"]]
    noise_section = section["noise"]
    return rf_tomography_io.to_jsonable(
        {
            "dataset": str(root),
            "profile": section["profile"],
            "variant": section["variant"],
            "num_views": num_views,
            "num_bs": num_bs,
            "sources": dict(placement["sources"]) if isinstance(placement, Mapping) else None,
            "rings_dropped": len(placement["rings"]["dropped"])
            if isinstance(placement, Mapping)
            else None,
            "train_views": train_views,
            "held_out_views": held_views,
            "train_bs": train_bs,
            "held_out_bs": held_bs,
            "los_visible_captures": los_count,
            "captures": num_views * num_bs,
            "p_ref": noise_section["p_ref"],
            "c_ref": noise_section["c_ref"],
            "sigma2": noise_section["sigma2"],
            "oracle_rtol": resolved_rtol,
            "max_oracle_residual": max_oracle_residual,
            "min_los_share": min_los_share,
        }
    )
