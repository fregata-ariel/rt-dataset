"""RF-camera poses and the direction-cosine camera model (NumPy only).

Each UE is an RF camera: local +x is the camera forward axis and the receive
aperture lies in the local y-z plane. A developed image is a direction-cosine
disk ``(ky/k, kz/k)``; front-hemisphere rays are reconstructed with
``kx/k = +sqrt(1 - (ky/k)^2 - (kz/k)^2)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from plateau_rt.domain.rf_camera.calibration import direction_cosine_axes, rotation_matrix
from plateau_rt.domain.rf_camera.delay import propagating_direction_mask

PROJECTION = "front_hemisphere_direction_cosine"


@dataclass(frozen=True)
class RFViewSpec:
    """One RF-camera pose."""

    view_id: str
    position: tuple[float, float, float]
    look_at: tuple[float, float, float]
    orientation: tuple[float, float, float]


def look_at_orientation(
    position: tuple[float, float, float],
    target: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Return Sionna Euler angles whose local +x points at ``target``."""

    p = np.asarray(position, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    direction = t - p
    distance = float(np.linalg.norm(direction))
    if distance == 0.0:
        raise ValueError("position and look-at target must differ")
    direction /= distance

    theta = float(np.arccos(np.clip(direction[2], -1.0, 1.0)))
    phi = float(np.arctan2(direction[1], direction[0]))
    return (phi, theta - np.pi / 2.0, 0.0)


def generate_ring_views(
    *,
    target: tuple[float, float, float],
    radius_m: float,
    ue_height_m: float,
    num_views: int,
    start_azimuth_deg: float = 0.0,
) -> list[RFViewSpec]:
    """Generate deterministic RF-camera poses around a target."""

    if radius_m <= 0.0:
        raise ValueError("radius_m must be > 0")
    if num_views < 1:
        raise ValueError("num_views must be >= 1")

    target_arr = np.asarray(target, dtype=np.float64)
    start = np.deg2rad(start_azimuth_deg)
    views: list[RFViewSpec] = []
    for index in range(num_views):
        azimuth = start + 2.0 * np.pi * index / num_views
        position = (
            float(target_arr[0] + radius_m * np.cos(azimuth)),
            float(target_arr[1] + radius_m * np.sin(azimuth)),
            float(ue_height_m),
        )
        orientation = look_at_orientation(position, target)
        views.append(
            RFViewSpec(
                view_id=f"ue_{index:06d}",
                position=position,
                look_at=tuple(float(v) for v in target),
                orientation=orientation,
            )
        )
    return views


def view_pose_payload(view: RFViewSpec) -> dict[str, Any]:
    """Return the JSON-serializable pose of one view (``pose.json``).

    A local RF-camera ray maps to world coordinates by
    ``world_from_local_rotation @ ray_local``.
    """
    rotation = rotation_matrix(view.orientation)
    return {
        "view_id": view.view_id,
        "position_m": list(view.position),
        "look_at_m": list(view.look_at),
        "orientation_rad": list(view.orientation),
        "world_from_local_rotation": rotation.tolist(),
        "camera_forward_axis_local": [1.0, 0.0, 0.0],
        "camera_forward_world": rotation[:, 0].tolist(),
        "projection": PROJECTION,
    }


def build_direction_cosine_camera_model(
    *,
    fft_rows: int,
    fft_cols: int,
    horizontal_spacing_lambda: float,
    vertical_spacing_lambda: float,
) -> dict[str, np.ndarray]:
    """Build front-hemisphere local rays for the direction-cosine image."""

    ky, kz = direction_cosine_axes(
        fft_rows=fft_rows,
        fft_cols=fft_cols,
        horizontal_spacing_lambda=horizontal_spacing_lambda,
        vertical_spacing_lambda=vertical_spacing_lambda,
    )
    valid = propagating_direction_mask(ky, kz)

    ky_grid = np.broadcast_to(ky[None, :], (fft_rows, fft_cols))
    kz_grid = np.broadcast_to(kz[:, None], (fft_rows, fft_cols))
    kx_sq = 1.0 - ky_grid**2 - kz_grid**2
    kx = np.sqrt(np.maximum(kx_sq, 0.0))

    rays = np.stack([kx, ky_grid, kz_grid], axis=-1).astype(np.float32)
    rays[~valid] = 0.0

    return {
        "ray_directions_local": rays,
        "valid_mask": valid.astype(bool),
        "ky_over_k": ky.astype(np.float32),
        "kz_over_k": kz.astype(np.float32),
    }
