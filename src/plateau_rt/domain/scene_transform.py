"""Scene-local coordinate transform (NumPy only, Sionna-free).

The scene builder recentres CityJSON vertices into scene-local coordinates by
subtracting a pure translation (no rotation or scaling). This module holds the
:class:`SceneTransform` value object describing that translation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from numpy.typing import ArrayLike

SCENE_TRANSFORM_DEFINITION: str = (
    "local_xyz = projected_xyz - origin_projected_xyz = projected_xyz + translation_xyz; "
    "origin x, y = centre of the CityJSON vertex bounding box and origin z = minimum "
    "vertex z, "
    "both after applying the CityJSON transform (scale, translate); no rotation or scaling"
)


def _require_real(value: Any, *, name: str) -> float:
    """Return ``value`` as float, rejecting bools and non-real numbers."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{name} must be a real number, got {value!r}")
    return float(value)


@dataclass(frozen=True)
class SceneTransform:
    """Pure translation between the source projected CRS and scene-local coordinates."""

    origin_projected_xyz: tuple[float, float, float]
    source_crs: str | None = None
    legacy: bool = False

    def __post_init__(self) -> None:
        try:
            items = tuple(self.origin_projected_xyz)
        except TypeError as exc:
            raise ValueError("origin_projected_xyz must be a length-3 tuple") from exc
        if len(items) != 3:
            raise ValueError("origin_projected_xyz must be a length-3 tuple")
        values = tuple(_require_real(v, name="origin_projected_xyz") for v in items)
        if not math.isfinite(values[0]) or not math.isfinite(values[1]):
            raise ValueError("origin_projected_xyz x and y must be finite")
        if self.legacy:
            if not math.isnan(values[2]):
                raise ValueError("legacy origin_projected_xyz z must be NaN")
        elif not math.isfinite(values[2]):
            raise ValueError("origin_projected_xyz z must be finite")
        if self.source_crs is not None and (
            not isinstance(self.source_crs, str) or not self.source_crs
        ):
            raise ValueError("source_crs must be None or a non-empty str")
        object.__setattr__(self, "origin_projected_xyz", values)
        object.__setattr__(self, "source_crs", self.source_crs)
        object.__setattr__(self, "legacy", bool(self.legacy))

    @property
    def translation_xyz(self) -> tuple[float, float, float]:
        """Scene translation ``(-x, -y, -z)`` added to projected coordinates."""
        return tuple(0.0 - v for v in self.origin_projected_xyz)  # type: ignore[return-value]

    def to_projected(self, local_xyz: ArrayLike) -> np.ndarray:
        """Map scene-local coordinates back to the source projected CRS.

        The z column becomes NaN for a legacy transform (unknown z offset).
        """
        arr = np.asarray(local_xyz, dtype=np.float64)
        if arr.shape == () or (arr.ndim >= 1 and arr.shape[-1] != 3):
            raise ValueError("local_xyz must have shape (..., 3)")
        origin = np.asarray(self.origin_projected_xyz, dtype=np.float64)
        return arr + origin

    def to_local(self, projected_xyz: ArrayLike) -> np.ndarray:
        """Map source projected coordinates into scene-local coordinates."""
        arr = np.asarray(projected_xyz, dtype=np.float64)
        if arr.shape == () or (arr.ndim >= 1 and arr.shape[-1] != 3):
            raise ValueError("projected_xyz must have shape (..., 3)")
        origin = np.asarray(self.origin_projected_xyz, dtype=np.float64)
        return arr - origin

    def to_payload(self) -> dict[str, Any]:
        """Return the JSON-serialisable build-manifest payload."""
        if self.legacy:
            raise ValueError("legacy SceneTransform has no payload (unknown z offset)")
        ox, oy, oz = self.origin_projected_xyz
        tx, ty, tz = self.translation_xyz
        return {
            "source_crs": self.source_crs,
            "origin_projected_xyz": [float(ox), float(oy), float(oz)],
            "translation_xyz": [float(tx), float(ty), float(tz)],
            "definition": SCENE_TRANSFORM_DEFINITION,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> SceneTransform:
        """Build a transform from a manifest payload mapping."""
        if not isinstance(payload, Mapping):
            raise ValueError("scene_transform payload must be a mapping")
        if "origin_projected_xyz" not in payload:
            raise ValueError("scene_transform payload is missing 'origin_projected_xyz'")
        raw_origin = payload["origin_projected_xyz"]
        try:
            origin_items = list(raw_origin)
        except TypeError as exc:
            raise ValueError("origin_projected_xyz must be 3 finite real numbers") from exc
        if len(origin_items) != 3:
            raise ValueError("origin_projected_xyz must be 3 finite real numbers")
        origin = tuple(_require_real(v, name="origin_projected_xyz") for v in origin_items)
        for v in origin:
            if not math.isfinite(v):
                raise ValueError("origin_projected_xyz must be 3 finite real numbers")
        source_crs: str | None = None
        if "source_crs" in payload and payload["source_crs"] is not None:
            raw_crs = payload["source_crs"]
            if not isinstance(raw_crs, str) or not raw_crs:
                raise ValueError("source_crs must be None or a non-empty str")
            source_crs = raw_crs
        if "translation_xyz" in payload and payload["translation_xyz"] is not None:
            raw_t = payload["translation_xyz"]
            try:
                t_items = list(raw_t)
            except TypeError as exc:
                raise ValueError("translation_xyz must be 3 finite numbers") from exc
            if len(t_items) != 3:
                raise ValueError("translation_xyz must be 3 finite numbers")
            trans = tuple(_require_real(v, name="translation_xyz") for v in t_items)
            for v in trans:
                if not math.isfinite(v):
                    raise ValueError("translation_xyz must be 3 finite numbers")
            for t_i, o_i in zip(trans, origin):
                if abs(t_i + o_i) > 1e-6 * max(1.0, abs(o_i)):
                    raise ValueError(
                        f"translation_xyz {list(trans)!r} is inconsistent with "
                        f"origin_projected_xyz {list(origin)!r}"
                    )
        origin3 = cast("tuple[float, float, float]", origin)
        return cls(origin_projected_xyz=origin3, source_crs=source_crs)

    @classmethod
    def from_legacy_center(cls, center_xy: Any) -> SceneTransform:
        """Build a legacy transform from an old manifest's ``center_lat_lon``."""
        try:
            items = list(center_xy)
        except TypeError as exc:
            raise ValueError("center_lat_lon must be a length-2 sequence") from exc
        if isinstance(center_xy, (str, bytes)) or len(items) != 2:
            raise ValueError("center_lat_lon must be a length-2 sequence")
        values = tuple(_require_real(v, name="center_lat_lon") for v in items)
        for v in values:
            if not math.isfinite(v):
                raise ValueError("center_lat_lon must be finite real numbers")
        return cls(
            origin_projected_xyz=(values[0], values[1], float("nan")),
            source_crs=None,
            legacy=True,
        )
