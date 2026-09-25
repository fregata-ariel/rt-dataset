"""Coverage-map UE placement for multi-view RF-camera datasets (NumPy only).

The multi-view RF-camera dataset places UEs (RF cameras) either on a
deterministic ring (see :func:`plateau_rt.domain.rf_camera.camera.generate_ring_views`)
or, with this module, on cells of a 2D path-gain map (radio map) computed at UE
height. The coverage workflow is:

1. Describe the horizontal radio-map grid with :class:`RadioMapGrid` (the same
   convention as Sionna's ``PlanarRadioMap`` with orientation ``(0, 0, 0)``).
2. Select candidate cells with :func:`candidate_cells`: cells whose (aggregated)
   path gain passes a :class:`CoverageThreshold` and that are neither invalid
   (no ray reached the cell) nor excluded (e.g. inside/near buildings) nor too
   close to a base station.
3. Draw UE positions with :func:`sample_placements` (uniform without
   replacement, optional minimum spacing and intra-cell jitter).
4. Orient the views with :func:`orient_views` (look at a target, face a base
   station, or take a random yaw).
5. Or run all of the above at once with :func:`plan_coverage_placement`, which
   derives two independent RNG streams (positions, orientation) from a single
   ``placement_seed`` so that changing only the orientation policy leaves the
   positions unchanged.

Conventions:
    * Scene coordinates are in metres, z up. A radio map is a horizontal plane
      at height ``center_m[2]`` (the UE height).
    * Path gain is stored linear scale as ``[num_tx, ny, nx]`` (row index = y,
      column index = x), 0 where no ray reached the cell.
    * UE local frame: x forward, y left, z up (same as ``camera.py``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from plateau_rt.domain.rf_camera.camera import RFViewSpec, look_at_orientation

THRESHOLD_MODES: tuple[str, ...] = ("absolute_db", "relative_to_max_db", "percentile")
AGGREGATIONS: tuple[str, ...] = ("max", "sum", "all", "any")
LOS_REFERENCES: tuple[str, ...] = ("any", "all")
ORIENTATION_POLICIES: tuple[str, ...] = ("look_at_target", "face_bs", "random_yaw")
RNG_DERIVATION = "SeedSequence(placement_seed).spawn(2) -> [positions, orientation]"
# Version 2 draws the intra-cell jitter for every candidate before the greedy
# spacing scan, so ``min_spacing_m`` is enforced on the jittered positions.
# Version 1 jittered only the accepted cells after the scan, so two accepted
# UEs could end up closer than ``min_spacing_m``.
SAMPLER_VERSION = 2


def _as_finite_float(name: str, value: Any) -> float:
    """Return ``value`` as a finite Python float, else raise ``ValueError``."""
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a finite float, got {value!r}") from None
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def _as_nonnegative_float(name: str, value: Any) -> float:
    """Return ``value`` as a finite ``>= 0`` Python float, else raise ``ValueError``."""
    result = _as_finite_float(name, value)
    if result < 0.0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    return result


def _as_point(name: str, value: Any) -> tuple[float, float, float]:
    """Return ``value`` as a finite length-3 tuple of Python floats."""
    try:
        items = tuple(value)
    except TypeError:
        raise ValueError(f"{name} must be a length-3 sequence, got {value!r}") from None
    if len(items) != 3:
        raise ValueError(f"{name} must be a length-3 sequence, got {value!r}")
    try:
        point = (float(items[0]), float(items[1]), float(items[2]))
    except (TypeError, ValueError):
        raise ValueError(f"{name} must contain finite floats, got {value!r}") from None
    if not all(np.isfinite(point)):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return point


def _as_bs_positions(bs_positions: Sequence[Sequence[float]]) -> np.ndarray:
    """Return base-station positions as a finite float64 ``[M, 3]`` array."""
    try:
        arr = np.asarray(bs_positions, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError(
            f"bs_positions must be a sequence of length-3 positions, got {bs_positions!r}"
        ) from None
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] < 1:
        raise ValueError(
            "bs_positions must have shape [num_bs, 3] with at least one base station, "
            f"got shape {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("bs_positions must be finite")
    return arr


def _json_float(value: Any) -> float | None:
    """Return ``value`` as a plain Python float, or ``None`` when non-finite."""
    result = float(value)
    return result if np.isfinite(result) else None


@dataclass(frozen=True)
class RadioMapGrid:
    """Horizontal radio-map grid (Sionna PlanarRadioMap, orientation (0, 0, 0))."""

    center_m: tuple[float, float, float]
    size_m: tuple[float, float]
    cell_size_m: tuple[float, float]

    def validate(self) -> None:
        """Check that the grid definition is finite with positive sizes."""
        try:
            center = tuple(self.center_m)
            size = tuple(self.size_m)
            cell = tuple(self.cell_size_m)
        except TypeError:
            raise ValueError(
                "center_m, size_m and cell_size_m must be sequences, got "
                f"({self.center_m!r}, {self.size_m!r}, {self.cell_size_m!r})"
            ) from None
        if len(center) != 3:
            raise ValueError(f"center_m must have 3 entries, got {self.center_m!r}")
        if len(size) != 2:
            raise ValueError(f"size_m must have 2 entries, got {self.size_m!r}")
        if len(cell) != 2:
            raise ValueError(f"cell_size_m must have 2 entries, got {self.cell_size_m!r}")
        for name, values in (
            ("center_m", center),
            ("size_m", size),
            ("cell_size_m", cell),
        ):
            for entry in values:
                try:
                    number = float(entry)
                except (TypeError, ValueError):
                    raise ValueError(f"{name} must contain finite floats") from None
                if not np.isfinite(number):
                    raise ValueError(f"{name} must be finite, got {entry!r}")
        if not (float(size[0]) > 0.0 and float(size[1]) > 0.0):
            raise ValueError(f"size_m must be > 0, got {self.size_m!r}")
        if not (float(cell[0]) > 0.0 and float(cell[1]) > 0.0):
            raise ValueError(f"cell_size_m must be > 0, got {self.cell_size_m!r}")

    @property
    def shape(self) -> tuple[int, int]:
        """Return ``(ny, nx)`` with ``ceil(size / cell_size)`` per axis."""
        self.validate()
        size_x, size_y = float(self.size_m[0]), float(self.size_m[1])
        cell_x, cell_y = float(self.cell_size_m[0]), float(self.cell_size_m[1])
        return (int(np.ceil(size_y / cell_y)), int(np.ceil(size_x / cell_x)))

    def cell_centers(self) -> np.ndarray:
        """Return the float64 ``[ny, nx, 3]`` centres of all cells."""
        self.validate()
        center_x, center_y, center_z = (
            float(self.center_m[0]),
            float(self.center_m[1]),
            float(self.center_m[2]),
        )
        size_x, size_y = float(self.size_m[0]), float(self.size_m[1])
        cell_x, cell_y = float(self.cell_size_m[0]), float(self.cell_size_m[1])
        ny, nx = self.shape
        ix = np.arange(nx, dtype=np.float64)
        iy = np.arange(ny, dtype=np.float64)
        xs = center_x + (ix + 0.5) * cell_x - 0.5 * size_x
        ys = center_y + (iy + 0.5) * cell_y - 0.5 * size_y
        centers = np.empty((ny, nx, 3), dtype=np.float64)
        centers[:, :, 0] = xs[None, :]
        centers[:, :, 1] = ys[:, None]
        centers[:, :, 2] = center_z
        return centers

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serializable description of this grid."""
        ny, nx = self.shape
        return {
            "center_m": [float(v) for v in self.center_m],
            "size_m": [float(v) for v in self.size_m],
            "cell_size_m": [float(v) for v in self.cell_size_m],
            "orientation_rad": [0.0, 0.0, 0.0],
            "shape": [ny, nx],
            "axis_order": ["y", "x"],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RadioMapGrid:
        """Rebuild a grid from :meth:`to_dict` (extra keys are ignored)."""
        if not isinstance(data, Mapping):
            raise ValueError(f"grid record must be a mapping, got {data!r}")
        if "orientation_rad" in data:
            try:
                orientation = [float(v) for v in data["orientation_rad"]]
            except (TypeError, ValueError):
                raise ValueError(
                    f"orientation_rad must hold floats, got {data['orientation_rad']!r}"
                ) from None
            if len(orientation) != 3 or not all(v == 0.0 for v in orientation):
                raise ValueError(
                    f"orientation_rad must be all zero, got {data['orientation_rad']!r}"
                )
        try:
            center = data["center_m"]
            size = data["size_m"]
            cell = data["cell_size_m"]
        except KeyError:
            raise ValueError("grid record needs 'center_m', 'size_m' and 'cell_size_m'") from None
        try:
            grid = cls(
                center_m=(float(center[0]), float(center[1]), float(center[2])),
                size_m=(float(size[0]), float(size[1])),
                cell_size_m=(float(cell[0]), float(cell[1])),
            )
        except (TypeError, ValueError, IndexError):
            raise ValueError(
                "grid record needs length-3 'center_m' and length-2 'size_m'/'cell_size_m'"
            ) from None
        grid.validate()
        return grid


@dataclass(frozen=True)
class CoverageThreshold:
    """Candidate threshold on the (aggregated) path gain in dB."""

    mode: str = "relative_to_max_db"
    value: float = 50.0

    def validate(self) -> None:
        """Check the threshold mode and value ranges."""
        if self.mode not in THRESHOLD_MODES:
            raise ValueError(
                f"unknown threshold mode {self.mode!r}; expected one of {THRESHOLD_MODES}"
            )
        value = _as_finite_float("value", self.value)
        if self.mode == "relative_to_max_db" and value < 0.0:
            raise ValueError(f"relative_to_max_db threshold value must be >= 0, got {self.value!r}")
        if self.mode == "percentile" and not 0.0 <= value <= 100.0:
            raise ValueError(
                f"percentile threshold value must satisfy 0 <= value <= 100, got {self.value!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serializable description of this threshold."""
        return {"mode": self.mode, "value": float(self.value)}


def _threshold_from_dict(data: Mapping[str, Any]) -> CoverageThreshold:
    """Rebuild a :class:`CoverageThreshold` from :meth:`CoverageThreshold.to_dict`."""
    if not isinstance(data, Mapping):
        raise ValueError(f"threshold record must be a mapping, got {data!r}")
    try:
        threshold = CoverageThreshold(mode=data["mode"], value=float(data["value"]))
    except KeyError:
        raise ValueError("threshold record needs 'mode' and 'value'") from None
    except (TypeError, ValueError):
        raise ValueError(f"bad threshold record: {data!r}") from None
    threshold.validate()
    return threshold


def path_gain_db(path_gain: np.ndarray) -> np.ndarray:
    """Convert linear path gain to dB (``10 * log10``) with safe zeros.

    ``path_gain`` is ``[ny, nx]`` (one BS) or ``[B, ny, nx]``; the result is
    always float64 ``[B, ny, nx]``. Cells that are ``<= 0``, NaN or inf become
    ``-inf``.
    """
    gain = np.asarray(path_gain, dtype=np.float64)
    if gain.ndim == 2:
        gain = gain[None, :, :]
    elif gain.ndim != 3:
        raise ValueError(
            f"path_gain must have shape [ny, nx] or [B, ny, nx], got shape {gain.shape}"
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        gain_db = 10.0 * np.log10(gain)
    gain_db[(gain <= 0.0) | ~np.isfinite(gain)] = -np.inf
    return gain_db


def box_footprint(x_min: float, y_min: float, x_max: float, y_max: float) -> np.ndarray:
    """Return a counter-clockwise ``[4, 2]`` rectangle footprint."""
    x_min_f = _as_finite_float("x_min", x_min)
    y_min_f = _as_finite_float("y_min", y_min)
    x_max_f = _as_finite_float("x_max", x_max)
    y_max_f = _as_finite_float("y_max", y_max)
    if not x_min_f < x_max_f:
        raise ValueError(f"x_min must be < x_max, got ({x_min!r}, {x_max!r})")
    if not y_min_f < y_max_f:
        raise ValueError(f"y_min must be < y_max, got ({y_min!r}, {y_max!r})")
    return np.array(
        [
            [x_min_f, y_min_f],
            [x_max_f, y_min_f],
            [x_max_f, y_max_f],
            [x_min_f, y_max_f],
        ],
        dtype=np.float64,
    )


def _as_polygon(footprint: np.ndarray, index: int) -> np.ndarray:
    """Return one footprint as a finite float64 ``[K, 2]`` array (``K >= 3``)."""
    try:
        poly = np.asarray(footprint, dtype=np.float64)
    except (TypeError, ValueError):
        raise ValueError(f"footprints[{index}] must be a [K, 2] array") from None
    if poly.ndim != 2 or poly.shape[1] != 2 or poly.shape[0] < 3:
        raise ValueError(
            f"footprints[{index}] must have shape [K, 2] with K >= 3, got shape {poly.shape}"
        )
    if not np.all(np.isfinite(poly)):
        raise ValueError(f"footprints[{index}] must be finite")
    return poly


def _polygon_edge_distances(xs: np.ndarray, ys: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Return the horizontal distance of every cell centre to a polygon's edges."""
    num_vertices = poly.shape[0]
    best = np.full(xs.shape, np.inf, dtype=np.float64)
    for index in range(num_vertices):
        ax, ay = float(poly[index, 0]), float(poly[index, 1])
        bx, by = (
            float(poly[(index + 1) % num_vertices, 0]),
            float(poly[(index + 1) % num_vertices, 1]),
        )
        edge_x, edge_y = bx - ax, by - ay
        length_sq = edge_x * edge_x + edge_y * edge_y
        if length_sq == 0.0:
            dist_sq = (xs - ax) ** 2 + (ys - ay) ** 2
        else:
            t = ((xs - ax) * edge_x + (ys - ay) * edge_y) / length_sq
            t = np.clip(t, 0.0, 1.0)
            dist_sq = (xs - (ax + t * edge_x)) ** 2 + (ys - (ay + t * edge_y)) ** 2
        best = np.minimum(best, dist_sq)
    return np.sqrt(best)


def _points_in_polygon(xs: np.ndarray, ys: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Return True for cell centres inside a polygon (even-odd rule, edges included)."""
    inside = np.zeros(xs.shape, dtype=bool)
    num_vertices = poly.shape[0]
    for index in range(num_vertices):
        x1, y1 = float(poly[index, 0]), float(poly[index, 1])
        x2, y2 = (
            float(poly[(index + 1) % num_vertices, 0]),
            float(poly[(index + 1) % num_vertices, 1]),
        )
        if y1 == y2:
            continue
        crossing = ((y1 > ys) != (y2 > ys)) & (xs < (x2 - x1) * (ys - y1) / (y2 - y1) + x1)
        inside ^= crossing
    inside |= _polygon_edge_distances(xs, ys, poly) == 0.0
    return inside


def footprint_mask(
    grid: RadioMapGrid,
    footprints: Sequence[np.ndarray],
    *,
    clearance_m: float = 0.0,
) -> np.ndarray:
    """Return True where a cell centre is inside (or near) any footprint polygon.

    Each footprint is a ``[K, 2]`` array (``K >= 3``, implicitly closed) tested
    with the even-odd rule, vectorised over cells. With ``clearance_m > 0``,
    cells within that horizontal distance of any polygon edge are included too.
    """
    grid.validate()
    clearance = _as_nonnegative_float("clearance_m", clearance_m)
    try:
        polygons_in = list(footprints)
    except TypeError:
        raise ValueError("footprints must be a sequence of [K, 2] arrays") from None
    ny, nx = grid.shape
    centers = grid.cell_centers()
    xs, ys = centers[:, :, 0], centers[:, :, 1]
    covered = np.zeros((ny, nx), dtype=bool)
    for index, footprint in enumerate(polygons_in):
        poly = _as_polygon(footprint, index)
        distances = _polygon_edge_distances(xs, ys, poly)
        covered |= _points_in_polygon(xs, ys, poly)
        if clearance > 0.0:
            covered |= distances <= clearance
    return covered


def dilate_mask(mask: np.ndarray, grid: RadioMapGrid, radius_m: float) -> np.ndarray:
    """Return True where a cell centre is within ``radius_m`` of a True cell.

    Distances are horizontal Euclidean distances between cell centres
    (inclusive ``<=``). ``radius_m == 0`` returns a copy of ``mask``.
    """
    grid.validate()
    radius = _as_nonnegative_float("radius_m", radius_m)
    cells = np.asarray(mask, dtype=bool)
    if cells.ndim != 2 or cells.shape != grid.shape:
        raise ValueError(f"mask shape {cells.shape} does not match grid shape {grid.shape}")
    if radius == 0.0:
        return cells.copy()
    ny, nx = grid.shape
    cell_x = float(grid.cell_size_m[0])
    cell_y = float(grid.cell_size_m[1])
    dx_max = int(np.ceil(radius / cell_x))
    dy_max = int(np.ceil(radius / cell_y))
    dilated = np.zeros((ny, nx), dtype=bool)
    for dy in range(-dy_max, dy_max + 1):
        for dx in range(-dx_max, dx_max + 1):
            if np.hypot(float(dx) * cell_x, float(dy) * cell_y) > radius:
                continue
            y0, y1 = max(0, dy), ny + min(0, dy)
            x0, x1 = max(0, dx), nx + min(0, dx)
            dilated[y0:y1, x0:x1] |= cells[y0 - dy : y1 - dy, x0 - dx : x1 - dx]
    return dilated


def los_indicator(per_bs_los: np.ndarray, reference: str) -> np.ndarray:
    """Reduce a bool ``[N, B]`` per-BS LoS array to ``[N]`` (``any`` / ``all`` over BSs)."""
    if reference not in LOS_REFERENCES:
        raise ValueError(f"unknown los reference {reference!r}; expected one of {LOS_REFERENCES}")
    los = np.asarray(per_bs_los)
    if los.ndim != 2:
        raise ValueError(f"per_bs_los must have shape [N, B], got shape {los.shape}")
    los_bool = los.astype(bool)
    if reference == "all":
        return np.all(los_bool, axis=1)
    return np.any(los_bool, axis=1)


@dataclass(frozen=True)
class CandidateCells:
    """Cells that pass the validity, exclusion, BS-distance and threshold tests."""

    indices: np.ndarray
    positions_m: np.ndarray
    gain_db: np.ndarray
    per_bs_gain_db: np.ndarray
    mask: np.ndarray
    threshold_db: tuple[float, ...]
    counts: Mapping[str, int]
    per_bs_los: np.ndarray | None = None
    per_bs_passing: tuple[int, ...] | None = None

    @property
    def count(self) -> int:
        """Return the number of candidate cells."""
        return int(self.indices.shape[0])


def candidate_cells(
    path_gain: np.ndarray,
    grid: RadioMapGrid,
    *,
    threshold: CoverageThreshold,
    aggregation: str = "max",
    exclusion_mask: np.ndarray | None = None,
    bs_positions: Sequence[Sequence[float]] | None = None,
    min_bs_distance_m: float = 0.0,
    los_mask: np.ndarray | None = None,
) -> CandidateCells:
    """Select the radio-map cells that may host a UE.

    ``path_gain`` is LINEAR, ``[ny, nx]`` (one BS) or ``[B, ny, nx]``. A cell is
    invalid when its aggregated dB is ``-inf``, excluded when ``exclusion_mask``
    is True, and too close when its 3D centre distance to any BS position is
    ``< min_bs_distance_m``. Eligible cells pass the ``threshold``; for
    ``aggregation`` ``"all"`` the per-BS thresholds are combined by
    intersection and for ``"any"`` by union. The resolved threshold is computed
    over the eligible cells only.

    ``los_mask`` is an optional bool ``[ny, nx]`` (promoted to ``[1, ny, nx]``)
    or ``[B, ny, nx]`` geometric LoS indicator stored per candidate as
    ``per_bs_los``; it does not affect selection.
    """
    grid.validate()
    if not isinstance(threshold, CoverageThreshold):
        raise ValueError(f"threshold must be a CoverageThreshold, got {threshold!r}")
    threshold.validate()
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"unknown aggregation {aggregation!r}; expected one of {AGGREGATIONS}")
    gain = np.asarray(path_gain, dtype=np.float64)
    if gain.ndim == 2:
        gain = gain[None, :, :]
    elif gain.ndim != 3:
        raise ValueError(
            f"path_gain must have shape [ny, nx] or [B, ny, nx], got shape {gain.shape}"
        )
    ny, nx = grid.shape
    if gain.shape[1] != ny or gain.shape[2] != nx:
        raise ValueError(
            f"path_gain spatial shape {(gain.shape[1], gain.shape[2])} does not match "
            f"grid shape {(ny, nx)}"
        )
    num_bs = int(gain.shape[0])
    if num_bs < 1:
        raise ValueError("path_gain needs at least one base station")
    los: np.ndarray | None = None
    if los_mask is not None:
        los_array = np.asarray(los_mask)
        if los_array.ndim == 2:
            if tuple(los_array.shape) != (ny, nx):
                raise ValueError(
                    f"los_mask shape {los_array.shape} does not match grid shape {(ny, nx)}"
                )
            los_array = los_array[None, :, :]
        elif los_array.ndim != 3 or tuple(los_array.shape[1:]) != (ny, nx):
            raise ValueError(
                "los_mask must have shape [ny, nx] or [B, ny, nx] matching the grid, "
                f"got shape {los_array.shape}"
            )
        if int(los_array.shape[0]) != num_bs:
            raise ValueError(
                f"los_mask has {int(los_array.shape[0])} base-station slices, "
                f"but path_gain has {num_bs}"
            )
        los = los_array.astype(bool)
    min_distance = _as_nonnegative_float("min_bs_distance_m", min_bs_distance_m)
    if exclusion_mask is None:
        excluded = np.zeros((ny, nx), dtype=bool)
    else:
        excluded = np.asarray(exclusion_mask, dtype=bool)
        if excluded.ndim != 2 or excluded.shape != (ny, nx):
            raise ValueError(
                f"exclusion_mask shape {excluded.shape} does not match grid shape {(ny, nx)}"
            )
    bs_array: np.ndarray | None = None
    if bs_positions is not None:
        bs_array = _as_bs_positions(bs_positions)
    if min_distance > 0.0 and bs_array is None:
        raise ValueError("bs_positions is required when min_bs_distance_m > 0")

    gain_db = path_gain_db(gain)
    if aggregation in ("max", "any"):
        aggregated = np.max(gain_db, axis=0)
    elif aggregation == "sum":
        sanitized = np.where(np.isfinite(gain) & (gain > 0.0), gain, 0.0)
        total = np.sum(sanitized, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            aggregated = 10.0 * np.log10(total)
        aggregated[~(total > 0.0)] = -np.inf
    else:
        aggregated = np.min(gain_db, axis=0)
    valid = np.isfinite(aggregated)

    if bs_array is None or min_distance == 0.0:
        too_close = np.zeros((ny, nx), dtype=bool)
    else:
        centers_all = grid.cell_centers()
        too_close = np.zeros((ny, nx), dtype=bool)
        for bs in range(bs_array.shape[0]):
            diff = centers_all - bs_array[bs][None, None, :]
            too_close |= np.sqrt(np.sum(diff * diff, axis=-1)) < min_distance

    eligible = valid & ~excluded & ~too_close
    if not bool(np.any(eligible)):
        raise ValueError("no eligible radio-map cells after validity/exclusion screening")

    value = float(threshold.value)
    per_bs_passing: tuple[int, ...] | None = None
    if aggregation in ("all", "any"):
        resolved: list[float] = []
        for bs in range(num_bs):
            own = gain_db[bs][eligible & np.isfinite(gain_db[bs])]
            if own.size == 0:
                resolved.append(float("inf"))
            elif threshold.mode == "absolute_db":
                resolved.append(value)
            elif threshold.mode == "relative_to_max_db":
                resolved.append(float(np.max(own)) - value)
            else:
                resolved.append(float(np.percentile(own, value)))
        threshold_db = tuple(resolved)
        per_bs_passing = tuple(
            int(np.sum(eligible & (gain_db[bs] >= threshold_db[bs]))) for bs in range(num_bs)
        )
        if aggregation == "all":
            passes = np.ones((ny, nx), dtype=bool)
            for bs in range(num_bs):
                passes &= gain_db[bs] >= threshold_db[bs]
        else:
            passes = np.zeros((ny, nx), dtype=bool)
            for bs in range(num_bs):
                passes |= gain_db[bs] >= threshold_db[bs]
    else:
        own_all = aggregated[eligible]
        if threshold.mode == "absolute_db":
            resolved_single = value
        elif threshold.mode == "relative_to_max_db":
            resolved_single = float(np.max(own_all)) - value
        else:
            resolved_single = float(np.percentile(own_all, value))
        threshold_db = (resolved_single,)
        passes = aggregated >= resolved_single

    selected = eligible & passes
    indices = np.stack(np.nonzero(selected), axis=1).astype(np.int64)
    centers = grid.cell_centers()
    counts = {
        "cells": int(ny * nx),
        "invalid": int(np.sum(~valid)),
        "excluded": int(np.sum(valid & excluded)),
        "too_close_to_bs": int(np.sum(valid & ~excluded & too_close)),
        "below_threshold": int(np.sum(eligible & ~passes)),
        "candidates": int(np.sum(selected)),
    }
    return CandidateCells(
        indices=indices,
        positions_m=np.ascontiguousarray(centers[selected].astype(np.float64)),
        gain_db=np.ascontiguousarray(aggregated[selected].astype(np.float64)),
        per_bs_gain_db=np.ascontiguousarray(gain_db[:, selected].T.astype(np.float64)),
        mask=np.ascontiguousarray(selected),
        threshold_db=threshold_db,
        counts=counts,
        per_bs_los=None if los is None else np.ascontiguousarray(los[:, selected].T),
        per_bs_passing=per_bs_passing,
    )


@dataclass(frozen=True)
class SampledCells:
    """UE host cells drawn from :class:`CandidateCells`."""

    candidate_index: np.ndarray
    positions_m: np.ndarray


def sample_placements(
    candidates: CandidateCells,
    num_views: int,
    *,
    rng: np.random.Generator,
    grid: RadioMapGrid | None = None,
    min_spacing_m: float = 0.0,
    jitter_fraction: float = 0.0,
    los: np.ndarray | None = None,
    los_fraction: float | None = None,
) -> SampledCells:
    """Draw ``num_views`` UE host cells uniformly without replacement.

    The RNG consumption order is part of the contract: exactly one
    ``rng.permutation(count)`` call (always, first), then (only when
    ``jitter_fraction > 0``) exactly one ``rng.uniform(-0.5, 0.5, size=(count,
    2))`` call. The jitter of row ``i`` belongs to candidate ``i`` (not to
    permutation position ``i``) and is applied before the greedy scan, so the
    ``min_spacing_m`` test uses the jittered positions.

    With ``los_fraction`` and a bool ``los`` ``[count]`` the draw is stratified:
    ``n_los = floor(los_fraction * num_views + 0.5)`` views are taken from LoS
    candidates and ``num_views - n_los`` from NLoS candidates. The greedy scan
    walks the permutation once, accepts a candidate only while its stratum
    quota is left and its jittered position is ``>= min_spacing_m`` away from
    every accepted position (all strata together), and stops when both quotas
    are filled.
    """
    if (
        isinstance(num_views, bool)
        or not isinstance(num_views, (int, np.integer))
        or int(num_views) < 1
    ):
        raise ValueError(f"num_views must be >= 1, got {num_views!r}")
    num_views = int(num_views)
    if not isinstance(rng, np.random.Generator):
        raise TypeError(f"rng must be a np.random.Generator, got {type(rng).__name__}")
    spacing = _as_nonnegative_float("min_spacing_m", min_spacing_m)
    jitter = _as_finite_float("jitter_fraction", jitter_fraction)
    if not 0.0 <= jitter < 1.0:
        raise ValueError(
            f"jitter_fraction must satisfy 0 <= jitter_fraction < 1, got {jitter_fraction!r}"
        )
    if jitter > 0.0 and grid is None:
        raise ValueError("grid is required when jitter_fraction > 0")
    if grid is not None:
        grid.validate()
    count = int(candidates.count)
    los_bool: np.ndarray | None = None
    quotas: tuple[int, ...]
    if los_fraction is not None:
        fraction = _as_finite_float("los_fraction", los_fraction)
        if not 0.0 <= fraction <= 1.0:
            raise ValueError(
                f"los_fraction must satisfy 0 <= los_fraction <= 1, got {los_fraction!r}"
            )
        if los is None:
            raise ValueError("los is required when los_fraction is set")
        los_array = np.asarray(los)
        if los_array.ndim != 1 or int(los_array.shape[0]) != count:
            raise ValueError(f"los must have shape [{count}], got shape {los_array.shape}")
        los_bool = los_array.astype(bool)
        n_los = int(np.floor(fraction * num_views + 0.5))
        quotas = (n_los, num_views - n_los)
    else:
        # Without los_fraction there is no stratification; a given los is ignored.
        quotas = (num_views,)

    order = np.asarray(rng.permutation(count), dtype=np.int64)
    positions_all = np.asarray(candidates.positions_m, dtype=np.float64).copy()
    if jitter > 0.0:
        assert grid is not None
        offsets = rng.uniform(-0.5, 0.5, size=(count, 2))
        positions_all[:, 0] += offsets[:, 0] * jitter * float(grid.cell_size_m[0])
        positions_all[:, 1] += offsets[:, 1] * jitter * float(grid.cell_size_m[1])

    def stratum(index: int) -> int:
        if los_bool is None:
            return 0
        return 0 if bool(los_bool[index]) else 1

    remaining = list(quotas)
    accepted: list[int] = []
    for raw in order:
        index = int(raw)
        which = stratum(index)
        if remaining[which] <= 0:
            continue
        ok = True
        for other in accepted:
            dx = float(positions_all[index, 0]) - float(positions_all[other, 0])
            dy = float(positions_all[index, 1]) - float(positions_all[other, 1])
            if np.hypot(dx, dy) < spacing:
                ok = False
                break
        if ok:
            accepted.append(index)
            remaining[which] -= 1
        if all(value <= 0 for value in remaining):
            break

    if any(value > 0 for value in remaining):
        if los_bool is None:
            raise ValueError(
                f"could only place {len(accepted)}/{num_views} views from {count} "
                f"candidates with min_spacing_m={spacing}"
            )
        placed_los = quotas[0] - remaining[0]
        placed_nlos = quotas[1] - remaining[1]
        raise ValueError(
            f"could only place {placed_los}/{quotas[0]} LoS and {placed_nlos}/{quotas[1]} "
            f"NLoS views from {count} candidates with min_spacing_m={spacing}"
        )
    candidate_index = np.asarray(accepted, dtype=np.int64)
    positions = positions_all[candidate_index].copy()
    return SampledCells(candidate_index=candidate_index, positions_m=positions)


def orient_views(
    positions_m: np.ndarray,
    *,
    policy: str,
    rng: np.random.Generator | None = None,
    target: Sequence[float] | None = None,
    bs_positions: Sequence[Sequence[float]] | None = None,
    face_bs: int | str = "strongest",
    per_bs_gain_db: np.ndarray | None = None,
    pitch_deg: float = 0.0,
) -> tuple[list[RFViewSpec], list[int | None]]:
    """Orient UE positions and return the views plus the faced BS per view.

    Policies: ``"look_at_target"`` points every camera at ``target``;
    ``"face_bs"`` points each camera at one base station (fixed ``face_bs``
    index or ``"strongest"`` per-BS gain); ``"random_yaw"`` draws one yaw per
    view from ``rng`` at elevation ``pitch_deg``. Only ``"random_yaw"`` uses
    the RNG (exactly one ``rng.uniform`` call).
    """
    if policy not in ORIENTATION_POLICIES:
        raise ValueError(
            f"unknown orientation policy {policy!r}; expected one of {ORIENTATION_POLICIES}"
        )
    try:
        pitch = float(pitch_deg)
    except (TypeError, ValueError):
        raise ValueError(f"pitch_deg must be a finite float, got {pitch_deg!r}") from None
    if not np.isfinite(pitch):
        raise ValueError(f"pitch_deg must be finite, got {pitch_deg!r}")
    positions = np.asarray(positions_m, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"positions_m must have shape [K, 3], got shape {positions.shape}")
    if not np.all(np.isfinite(positions)):
        raise ValueError("positions_m must be finite")
    num = int(positions.shape[0])

    def _build(look_ats: list[tuple[float, float, float]]) -> list[RFViewSpec]:
        views: list[RFViewSpec] = []
        for index in range(num):
            position = (
                float(positions[index, 0]),
                float(positions[index, 1]),
                float(positions[index, 2]),
            )
            orientation = look_at_orientation(position, look_ats[index])
            yaw, pitch_v, roll = orientation
            views.append(
                RFViewSpec(
                    view_id=f"ue_{index:06d}",
                    position=position,
                    look_at=look_ats[index],
                    orientation=(float(yaw), float(pitch_v), float(roll)),
                )
            )
        return views

    if policy == "look_at_target":
        if target is None:
            raise ValueError("target is required for policy 'look_at_target'")
        point = _as_point("target", target)
        no_facing: list[int | None] = [None for _ in range(num)]
        return _build([point for _ in range(num)]), no_facing
    if policy == "face_bs":
        if bs_positions is None:
            raise ValueError("bs_positions is required for policy 'face_bs'")
        bs_array = _as_bs_positions(bs_positions)
        num_bs = int(bs_array.shape[0])
        if isinstance(face_bs, str):
            if face_bs != "strongest":
                raise ValueError("face_bs must be a BS index or 'strongest'")
            if per_bs_gain_db is None:
                raise ValueError("per_bs_gain_db is required when face_bs='strongest'")
            gains = np.asarray(per_bs_gain_db, dtype=np.float64)
            if gains.ndim != 2 or gains.shape != (num, num_bs):
                raise ValueError(
                    f"per_bs_gain_db must have shape {(num, num_bs)}, got shape {gains.shape}"
                )
            facing: list[int | None] = [int(np.argmax(gains[row])) for row in range(num)]
        else:
            if (
                isinstance(face_bs, bool)
                or not isinstance(face_bs, (int, np.integer))
                or int(face_bs) < 0
                or int(face_bs) >= num_bs
            ):
                raise ValueError(
                    f"face_bs must be a BS index in [0, {num_bs}) or 'strongest', got {face_bs!r}"
                )
            facing = [int(face_bs) for _ in range(num)]
        look_ats = [
            (float(bs_array[j, 0]), float(bs_array[j, 1]), float(bs_array[j, 2])) for j in facing
        ]
        return _build(look_ats), facing
    if rng is None or not isinstance(rng, np.random.Generator):
        raise TypeError(
            f"rng must be a np.random.Generator for policy 'random_yaw', "
            f"got {type(rng).__name__ if rng is not None else None}"
        )
    if not -90.0 < pitch < 90.0:
        raise ValueError(
            f"pitch_deg must satisfy -90 < pitch_deg < 90 for policy 'random_yaw', "
            f"got {pitch_deg!r}"
        )
    yaw = np.asarray(rng.uniform(0.0, 2.0 * np.pi, size=num), dtype=np.float64)
    elevation = float(np.deg2rad(pitch))
    look_ats = [
        (
            float(positions[index, 0]) + float(np.cos(elevation) * np.cos(yaw[index])),
            float(positions[index, 1]) + float(np.cos(elevation) * np.sin(yaw[index])),
            float(positions[index, 2]) + float(np.sin(elevation)),
        )
        for index in range(num)
    ]
    return _build(look_ats), [None for _ in range(num)]


@dataclass(frozen=True)
class CoveragePlacementSettings:
    """Full settings of one coverage-map UE placement."""

    num_views: int
    placement_seed: int
    threshold: CoverageThreshold = CoverageThreshold()
    aggregation: str = "max"
    min_bs_distance_m: float = 0.0
    min_spacing_m: float = 0.0
    jitter_fraction: float = 0.0
    orientation_policy: str = "face_bs"
    face_bs: int | str = "strongest"
    target: tuple[float, float, float] | None = None
    pitch_deg: float = 0.0
    los_fraction: float | None = None
    los_reference: str = "any"

    def validate(self) -> None:
        """Check every setting value and cross-field requirement."""
        if (
            isinstance(self.num_views, bool)
            or not isinstance(self.num_views, (int, np.integer))
            or int(self.num_views) < 1
        ):
            raise ValueError(f"num_views must be >= 1, got {self.num_views!r}")
        if (
            isinstance(self.placement_seed, bool)
            or not isinstance(self.placement_seed, (int, np.integer))
            or int(self.placement_seed) < 0
        ):
            raise ValueError(f"placement_seed must be an int >= 0, got {self.placement_seed!r}")
        if not isinstance(self.threshold, CoverageThreshold):
            raise ValueError(f"threshold must be a CoverageThreshold, got {self.threshold!r}")
        self.threshold.validate()
        if self.aggregation not in AGGREGATIONS:
            raise ValueError(
                f"unknown aggregation {self.aggregation!r}; expected one of {AGGREGATIONS}"
            )
        _as_nonnegative_float("min_bs_distance_m", self.min_bs_distance_m)
        _as_nonnegative_float("min_spacing_m", self.min_spacing_m)
        jitter = _as_finite_float("jitter_fraction", self.jitter_fraction)
        if not 0.0 <= jitter < 1.0:
            raise ValueError(
                "jitter_fraction must satisfy 0 <= jitter_fraction < 1, "
                f"got {self.jitter_fraction!r}"
            )
        if self.orientation_policy not in ORIENTATION_POLICIES:
            raise ValueError(
                f"unknown orientation_policy {self.orientation_policy!r}; "
                f"expected one of {ORIENTATION_POLICIES}"
            )
        if isinstance(self.face_bs, str):
            if self.face_bs != "strongest":
                raise ValueError("face_bs must be a BS index or 'strongest'")
        elif (
            isinstance(self.face_bs, bool)
            or not isinstance(self.face_bs, (int, np.integer))
            or int(self.face_bs) < 0
        ):
            raise ValueError(f"face_bs must be an int >= 0 or 'strongest', got {self.face_bs!r}")
        try:
            pitch = float(self.pitch_deg)
        except (TypeError, ValueError):
            raise ValueError(f"pitch_deg must be a finite float, got {self.pitch_deg!r}") from None
        if not np.isfinite(pitch):
            raise ValueError(f"pitch_deg must be finite, got {self.pitch_deg!r}")
        if self.orientation_policy == "random_yaw" and not -90.0 < pitch < 90.0:
            raise ValueError(
                "pitch_deg must satisfy -90 < pitch_deg < 90 for policy 'random_yaw', "
                f"got {self.pitch_deg!r}"
            )
        if self.orientation_policy == "look_at_target":
            if self.target is None:
                raise ValueError("target is required for policy 'look_at_target'")
            _as_point("target", self.target)
        elif self.target is not None:
            _as_point("target", self.target)
        if self.los_fraction is not None:
            fraction = _as_finite_float("los_fraction", self.los_fraction)
            if not 0.0 <= fraction <= 1.0:
                raise ValueError(
                    f"los_fraction must satisfy 0 <= los_fraction <= 1, got {self.los_fraction!r}"
                )
        if self.los_reference not in LOS_REFERENCES:
            raise ValueError(
                f"unknown los_reference {self.los_reference!r}; expected one of {LOS_REFERENCES}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-serializable description of these settings."""
        face_bs: int | str = (
            int(self.face_bs) if not isinstance(self.face_bs, str) else self.face_bs
        )
        return {
            "num_views": int(self.num_views),
            "placement_seed": int(self.placement_seed),
            "threshold": self.threshold.to_dict(),
            "aggregation": self.aggregation,
            "min_bs_distance_m": float(self.min_bs_distance_m),
            "min_spacing_m": float(self.min_spacing_m),
            "jitter_fraction": float(self.jitter_fraction),
            "orientation_policy": self.orientation_policy,
            "face_bs": face_bs,
            "target": [float(v) for v in self.target] if self.target is not None else None,
            "pitch_deg": float(self.pitch_deg),
            "los_fraction": None if self.los_fraction is None else float(self.los_fraction),
            "los_reference": self.los_reference,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CoveragePlacementSettings:
        """Rebuild settings from :meth:`to_dict`."""
        if not isinstance(data, Mapping):
            raise ValueError(f"settings record must be a mapping, got {data!r}")
        try:
            threshold_data = data["threshold"]
            target_data = data.get("target", None)
            settings = cls(
                num_views=data["num_views"],
                placement_seed=data["placement_seed"],
                threshold=_threshold_from_dict(threshold_data),
                aggregation=data["aggregation"],
                min_bs_distance_m=data["min_bs_distance_m"],
                min_spacing_m=data["min_spacing_m"],
                jitter_fraction=data["jitter_fraction"],
                orientation_policy=data["orientation_policy"],
                face_bs=data["face_bs"],
                target=tuple(target_data) if target_data is not None else None,
                pitch_deg=data["pitch_deg"],
                los_fraction=data.get("los_fraction", None),
                los_reference=data.get("los_reference", "any"),
            )
        except KeyError:
            raise ValueError(f"settings record misses required keys: {data!r}") from None
        settings.validate()
        return settings


@dataclass(frozen=True)
class CoveragePlacement:
    """One finished coverage-map UE placement (settings, grid, cells, views)."""

    settings: CoveragePlacementSettings
    grid: RadioMapGrid
    candidates: CandidateCells
    sampled: SampledCells
    views: list[RFViewSpec]
    facing_bs_index: list[int | None]

    def to_record(self) -> dict[str, Any]:
        """Return the JSON-serializable record of this placement.

        Every non-finite float becomes ``None`` so ``json.dumps(record,
        allow_nan=False)`` succeeds even when a chosen cell has ``-inf`` gain.
        """
        entries: list[dict[str, Any]] = []
        per_bs_los = self.candidates.per_bs_los
        for row, view in enumerate(self.views):
            chosen = int(self.sampled.candidate_index[row])
            iy, ix = (
                int(self.candidates.indices[chosen, 0]),
                int(self.candidates.indices[chosen, 1]),
            )
            facing = self.facing_bs_index[row]
            los_bs = None if per_bs_los is None else [bool(v) for v in per_bs_los[chosen, :]]
            if los_bs is None:
                los: bool | None = None
            elif self.settings.los_reference == "all":
                los = all(los_bs)
            else:
                los = any(los_bs)
            entries.append(
                {
                    "view_id": view.view_id,
                    "cell_index": [iy, ix],
                    "cell_center_m": [
                        float(self.candidates.positions_m[chosen, 0]),
                        float(self.candidates.positions_m[chosen, 1]),
                        float(self.candidates.positions_m[chosen, 2]),
                    ],
                    "position_m": [
                        float(view.position[0]),
                        float(view.position[1]),
                        float(view.position[2]),
                    ],
                    "look_at_m": [
                        float(view.look_at[0]),
                        float(view.look_at[1]),
                        float(view.look_at[2]),
                    ],
                    "orientation_rad": [
                        float(view.orientation[0]),
                        float(view.orientation[1]),
                        float(view.orientation[2]),
                    ],
                    "path_gain_db": _json_float(self.candidates.gain_db[chosen]),
                    "per_bs_path_gain_db": [
                        _json_float(v) for v in self.candidates.per_bs_gain_db[chosen, :]
                    ],
                    "facing_bs_index": None if facing is None else int(facing),
                    "los_bs": los_bs,
                    "los": los,
                }
            )
        if per_bs_los is None:
            los_section: dict[str, Any] | None = None
        else:
            reduced = los_indicator(per_bs_los, self.settings.los_reference)
            los_section = {
                "reference": self.settings.los_reference,
                "definition": "geometric shadow ray from the cell centre to each BS",
                "candidates_los": int(np.sum(reduced)),
                "candidates_nlos": int(reduced.size - np.sum(reduced)),
            }
        aggregation = self.settings.aggregation
        per_bs_passing = self.candidates.per_bs_passing
        return {
            "method": "coverage",
            "placement_seed": int(self.settings.placement_seed),
            "sampler_version": SAMPLER_VERSION,
            "rng": {
                "bit_generator": "PCG64",
                "derivation": RNG_DERIVATION,
                "streams": ["positions", "orientation"],
            },
            "settings": self.settings.to_dict(),
            "grid": self.grid.to_dict(),
            "threshold_db": [_json_float(v) for v in self.candidates.threshold_db],
            "candidate_count": int(self.candidates.count),
            "cell_counts": dict(self.candidates.counts),
            "multi_bs": {
                "aggregation": aggregation,
                "per_bs_threshold": aggregation in ("all", "any"),
                "combine": {"all": "intersection", "any": "union"}.get(aggregation),
                "per_bs_passing_cells": (
                    None if per_bs_passing is None else [int(v) for v in per_bs_passing]
                ),
            },
            "los": los_section,
            "views": entries,
        }


def plan_coverage_placement(
    path_gain: np.ndarray,
    grid: RadioMapGrid,
    settings: CoveragePlacementSettings,
    *,
    exclusion_mask: np.ndarray | None = None,
    bs_positions: Sequence[Sequence[float]] | None = None,
    los_mask: np.ndarray | None = None,
) -> CoveragePlacement:
    """Plan UE poses from a linear path-gain map ``[ny, nx]`` or ``[B, ny, nx]``.

    Candidate selection is deterministic; the UE draw and the orientation draw
    use two independent streams spawned from
    ``SeedSequence(placement_seed)`` (``[positions, orientation]``), so changing
    only the orientation policy leaves the positions unchanged. With
    ``los_mask`` and ``settings.los_fraction`` the UE draw is stratified into
    LoS / NLoS quotas using the geometric LoS indicator.
    """
    settings.validate()
    grid.validate()
    candidates = candidate_cells(
        path_gain,
        grid,
        threshold=settings.threshold,
        aggregation=settings.aggregation,
        exclusion_mask=exclusion_mask,
        bs_positions=bs_positions,
        min_bs_distance_m=settings.min_bs_distance_m,
        los_mask=los_mask,
    )
    if settings.los_fraction is not None and candidates.per_bs_los is None:
        raise ValueError("los_fraction needs a LoS mask (los_mask) to stratify the UE draw")
    position_seq, orientation_seq = np.random.SeedSequence(int(settings.placement_seed)).spawn(2)
    position_rng = np.random.default_rng(position_seq)
    orientation_rng = np.random.default_rng(orientation_seq)
    los_indices = (
        None
        if candidates.per_bs_los is None
        else los_indicator(candidates.per_bs_los, settings.los_reference)
    )
    sampled = sample_placements(
        candidates,
        int(settings.num_views),
        rng=position_rng,
        grid=grid,
        min_spacing_m=float(settings.min_spacing_m),
        jitter_fraction=float(settings.jitter_fraction),
        los=los_indices,
        los_fraction=settings.los_fraction,
    )
    views, facing = orient_views(
        sampled.positions_m,
        policy=settings.orientation_policy,
        rng=orientation_rng,
        target=settings.target,
        bs_positions=bs_positions,
        face_bs=settings.face_bs,
        per_bs_gain_db=candidates.per_bs_gain_db[sampled.candidate_index],
        pitch_deg=float(settings.pitch_deg),
    )
    return CoveragePlacement(
        settings=settings,
        grid=grid,
        candidates=candidates,
        sampled=sampled,
        views=views,
        facing_bs_index=facing,
    )


def views_from_placement_record(record: Mapping[str, Any]) -> list[RFViewSpec]:
    """Rebuild :class:`RFViewSpec` poses from a :meth:`CoveragePlacement.to_record` record."""
    if not isinstance(record, Mapping) or record.get("method") != "coverage":
        raise ValueError("record must be a coverage placement record with method 'coverage'")
    entries = record.get("views", None)
    if not isinstance(entries, (list, tuple)):
        raise ValueError("record entry 'views' must be a list")
    views: list[RFViewSpec] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError(f"malformed view entry: {entry!r}")
        try:
            view_id = entry["view_id"]
            position_m = entry["position_m"]
            look_at_m = entry["look_at_m"]
            orientation_rad = entry["orientation_rad"]
        except KeyError:
            raise ValueError(f"malformed view entry: {entry!r}") from None
        if not isinstance(view_id, str):
            raise ValueError(f"malformed view entry: {entry!r}")
        try:
            position = (float(position_m[0]), float(position_m[1]), float(position_m[2]))
            look_at = (float(look_at_m[0]), float(look_at_m[1]), float(look_at_m[2]))
            orientation = (
                float(orientation_rad[0]),
                float(orientation_rad[1]),
                float(orientation_rad[2]),
            )
        except (TypeError, ValueError, IndexError):
            raise ValueError(f"malformed view entry: {entry!r}") from None
        if len(tuple(position_m)) != 3 or len(tuple(look_at_m)) != 3:
            raise ValueError(f"malformed view entry: {entry!r}")
        if len(tuple(orientation_rad)) != 3:
            raise ValueError(f"malformed view entry: {entry!r}")
        views.append(
            RFViewSpec(
                view_id=view_id,
                position=position,
                look_at=look_at,
                orientation=orientation,
            )
        )
    return views
