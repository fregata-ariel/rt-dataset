"""NumPy model of the base-station antenna pattern (Sionna TR 38.901 element).

Implements the 3GPP TR 38.901 vertically polarised element power gain and the
complex field pattern of a base station with a look-at orientation. NumPy only.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from plateau_rt.domain.rf_camera.calibration import rotation_matrix
from plateau_rt.domain.rf_camera.camera import look_at_orientation

PATTERN_KINDS: tuple[str, ...] = ("tr38901", "iso")

TR38901_BEAMWIDTH_3DB_RAD = float(np.deg2rad(65.0))
TR38901_SLA_DB = 30.0
TR38901_A_MAX_DB = 30.0
TR38901_MAX_GAIN_DBI = 8.0


def tr38901_gain(theta: np.ndarray | float, phi: np.ndarray | float) -> np.ndarray:
    """Linear power gain (dimensionless, not dB) of the 3GPP TR 38.901 element, as in Sionna.

    ``theta`` is the local zenith angle and ``phi`` the local azimuth (wrapped to
    [-pi, pi)), both in rad; boresight is local +x (theta = pi/2, phi = 0).
    """
    theta_arr, phi_arr = np.broadcast_arrays(
        np.asarray(theta, dtype=np.float64), np.asarray(phi, dtype=np.float64)
    )
    phi_w = (phi_arr + np.pi) % (2.0 * np.pi) - np.pi
    a_v = -np.minimum(
        12.0 * ((theta_arr - np.pi / 2.0) / TR38901_BEAMWIDTH_3DB_RAD) ** 2,
        TR38901_SLA_DB,
    )
    a_h = -np.minimum(12.0 * (phi_w / TR38901_BEAMWIDTH_3DB_RAD) ** 2, TR38901_A_MAX_DB)
    a_db = TR38901_MAX_GAIN_DBI - np.minimum(-(a_v + a_h), TR38901_A_MAX_DB)
    return np.power(10.0, a_db / 10.0)


def bs_orientation(
    bs_position: Sequence[float] | np.ndarray, target: Sequence[float] | np.ndarray
) -> np.ndarray:
    """World-from-local rotation [3,3] of a BS whose local +x looks at ``target`` (roll 0)."""
    bs_flat = np.asarray(bs_position, dtype=np.float64).ravel()
    tgt_flat = np.asarray(target, dtype=np.float64).ravel()
    if bs_flat.size != 3 or tgt_flat.size != 3:
        raise ValueError("bs_position and target must each have exactly 3 entries")
    bs_tuple = (float(bs_flat[0]), float(bs_flat[1]), float(bs_flat[2]))
    target_tuple = (float(tgt_flat[0]), float(tgt_flat[1]), float(tgt_flat[2]))
    return np.asarray(rotation_matrix(look_at_orientation(bs_tuple, target_tuple)))


def bs_pattern(
    dir_world: np.ndarray, bs_orientation: np.ndarray, kind: str = "tr38901"
) -> np.ndarray:
    """Complex field pattern G_b of the BS along departure directions ``dir_world`` [..., 3].

    Field (not power) pattern for departure directions in the world frame. The
    directions are normalised inside. Polarisation mismatch is not modelled
    (scalar V-pol co-polar amplitude, phase exactly 0).
    """
    directions = np.asarray(dir_world, dtype=np.float64)
    if directions.ndim == 0 or directions.shape[-1] != 3:
        raise ValueError(f"dir_world must have shape [..., 3], got {directions.shape}")
    if not bool(np.all(np.isfinite(directions))):
        raise ValueError("dir_world must contain only finite entries")
    norms = np.linalg.norm(directions, axis=-1)
    if not bool(np.all(norms > 0.0)):
        raise ValueError("dir_world must not contain zero vectors")
    rotation = np.asarray(bs_orientation, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError(f"bs_orientation must have shape (3, 3), got {rotation.shape}")
    unit = directions / norms[..., None]
    local = unit @ rotation
    theta = np.arccos(np.clip(local[..., 2], -1.0, 1.0))
    phi = np.arctan2(local[..., 1], local[..., 0])
    if kind == "tr38901":
        return np.sqrt(tr38901_gain(theta, phi)).astype(np.complex128)
    if kind == "iso":
        return np.ones(directions.shape[:-1], dtype=np.complex128)
    raise ValueError(f"unknown pattern kind {kind!r}; expected one of {PATTERN_KINDS}")
