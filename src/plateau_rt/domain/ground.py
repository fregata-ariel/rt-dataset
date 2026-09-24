"""Flat ground-plane mesh helper (NumPy only)."""

from __future__ import annotations

import math

import numpy as np

GROUND_PLANE_ID = "ground_plane"
GROUND_PLANE_Z_M = -0.01
GROUND_ITU_MATERIAL = "itu_medium_dry_ground"
GROUND_VALID_CARRIER_RANGE_HZ: tuple[float, float] = (1.0e9, 10.0e9)


def ground_plane_mesh(
    size_m: float, z_m: float = GROUND_PLANE_Z_M
) -> tuple[np.ndarray, np.ndarray]:
    """Return vertices and faces of a square ground plane centred on the origin.

    Args:
        size_m: Side length of the square in metres (must be > 0).
        z_m: Height of the plane. Defaults to -0.01 m so the plane does not
            lie coplanar with building ``GroundSurface`` polygons at z = 0.

    Returns:
        Tuple of ``vertices`` with shape (4, 3) and dtype float64, and
        ``faces`` with shape (2, 3) and integer dtype. Both triangles are
        wound counter-clockwise seen from +z, giving a +z normal.
    """
    if not math.isfinite(size_m) or size_m <= 0:
        raise ValueError("size_m must be > 0")
    half = float(size_m) / 2.0
    vertices = np.array(
        [
            [-half, -half, z_m],
            [half, -half, z_m],
            [half, half, z_m],
            [-half, half, z_m],
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
    return vertices, faces


def check_ground_carrier_frequency(carrier_frequency_hz: float) -> None:
    """Fail fast when the carrier is outside the ground material's valid range.

    Args:
        carrier_frequency_hz: Carrier frequency in Hz.

    Raises:
        ValueError: If the carrier is not finite or lies outside 1-10 GHz.
    """
    lo_hz, hi_hz = GROUND_VALID_CARRIER_RANGE_HZ
    try:
        carrier_value = float(carrier_frequency_hz)
    except (TypeError, ValueError):
        carrier_value = float("nan")
    carrier_ghz = carrier_value / 1e9
    if not math.isfinite(carrier_value) or not (lo_hz <= carrier_value <= hi_hz):
        raise ValueError(
            f"ITU material '{GROUND_ITU_MATERIAL}' is only defined for carriers "
            f"from 1 to 10 GHz, got {carrier_ghz:.6f} GHz. "
            f"Rebuild without --ground-plane-size-m or use a carrier inside the range."
        )
