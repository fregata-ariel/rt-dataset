"""RF-camera poses and the direction-cosine camera model (NumPy only).

Each UE is an RF camera: local +x is the camera forward axis and the receive
aperture lies in the local y-z plane. The aperture is recorded separately for
the front (``kx >= 0``) and back (``kx < 0``) hemispheres; the developed image
is the front hemisphere on a direction-cosine disk ``(ky/k, kz/k)``, with rays
reconstructed as ``kx/k = +sqrt(1 - (ky/k)^2 - (kz/k)^2)``.

Image quantity: the FFT of an isotropic aperture measures the plane-wave
spectrum ``U(ky, kz)`` (amplitude per unit direction-cosine area). A pixel
covers the solid angle ``dOmega = dky dkz / kx``, so the developed image is the
complex amplitude per unit solid angle ``A = kx * U``; ``|A|^2`` is power per
steradian, the quantity a renderer integrates over solid angle.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from plateau_rt.domain.rf_camera.calibration import direction_cosine_axes, rotation_matrix
from plateau_rt.domain.rf_camera.delay import propagating_direction_mask

PROJECTION = "front_hemisphere_direction_cosine"
HEMISPHERES = ("front", "back")
IMAGE_QUANTITY = "solid_angle_amplitude"


def image_axes_payload() -> dict[str, Any]:
    """Machine-readable [row, col] axis convention of every RF-camera image array."""
    return {
        "array_axes": ["row", "col"],
        "row": "+kz (up)",
        "col": "+ky (camera left)",
        "row_coordinate": "kz_over_k",
        "col_coordinate": "ky_over_k",
        "row_index_increases_toward": "+kz",
        "col_index_increases_toward": "+ky",
        "row_0": "-kz (bottom)",
        "col_0": "-ky (camera right)",
        "display_origin": "lower",
        "mirrored_vs_pinhole_photo": True,
        "to_pinhole_photo_orientation": "np.flip(image, axis=(0, 1))",
    }


def channel_gain_reference_payload() -> dict[str, Any]:
    """What the stored complex channel values are referenced to (unit transmit power)."""
    return {
        "reference": "unit_transmit_power",
        "tx_power_applied": False,
        "definition": (
            "aperture_cfr and the path_geometry_gt amplitudes are Sionna channel "
            "coefficients for unit transmit power (Paths.cfr(normalize=False)): "
            "dimensionless, including the Tx and Rx antenna patterns and all propagation "
            "losses, so |H|^2 is the path gain; the developed angular images are linear "
            "transforms of aperture_cfr. config.tx_power_dbm is recorded but not applied "
            "to any stored array: received power [W] = 10**((tx_power_dbm - 30) / 10) * |H|^2."
        ),
    }


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


def solid_angle_weight(ky_over_k: np.ndarray, kz_over_k: np.ndarray) -> np.ndarray:
    """Return ``kx/k`` on the ``[kz, ky]`` image grid (0 outside the propagating disk)."""
    ky = np.asarray(ky_over_k, dtype=np.float64)[None, :]
    kz = np.asarray(kz_over_k, dtype=np.float64)[:, None]
    return np.sqrt(np.maximum(1.0 - ky**2 - kz**2, 0.0))


def to_solid_angle_amplitude(
    angular_cfr: np.ndarray,
    ky_over_k: np.ndarray,
    kz_over_k: np.ndarray,
) -> np.ndarray:
    """Convert a calibrated angular spectrum ``U[kz, ky, ...]`` into ``A = kx * U``."""
    cfr = np.asarray(angular_cfr)
    weight = solid_angle_weight(ky_over_k, kz_over_k)
    return cfr * weight.reshape(weight.shape + (1,) * (cfr.ndim - 2))


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
    kx = solid_angle_weight(ky, kz)

    rays = np.stack([kx, ky_grid, kz_grid], axis=-1).astype(np.float32)
    rays[~valid] = 0.0

    return {
        "ray_directions_local": rays,
        "valid_mask": valid.astype(bool),
        "solid_angle_weight": np.where(valid, kx, 0.0).astype(np.float32),
        "ky_over_k": ky.astype(np.float32),
        "kz_over_k": kz.astype(np.float32),
    }
