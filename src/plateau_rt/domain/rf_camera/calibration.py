"""Physical calibration of the RF-camera angular FFT (NumPy only).

The direct NumPy FFT of the receive-aperture matrix is useful for validating
Sionna array extraction, but two details must be corrected before
interpreting the complex spectrum as a physical angular field:

1. Sionna's PlanarArray rows run from +z to -z, so the FFT row coordinate has
   the opposite sign of physical local kz/k.
2. The physical aperture is centered on the UE, while ``np.fft.fft2`` treats
   matrix index (0, 0) as the spatial origin. This does not change power, but it
   adds a deterministic linear phase ramp to the angular spectrum.

This module also holds the UE-local geometry (Sionna's device rotation
convention and the geometric LoS direction) used to validate the image.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AngularCalibration:
    """A physically oriented angular field and its local direction axes."""

    cfr: np.ndarray
    ky_over_k: np.ndarray
    kz_over_k: np.ndarray


def direction_cosine_axes(
    *,
    fft_rows: int,
    fft_cols: int,
    horizontal_spacing_lambda: float,
    vertical_spacing_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the calibrated UE-local ``(ky/k, kz/k)`` axes of the angular image.

    Both axes increase toward the physical local +y and +z directions.
    """
    q_col = np.fft.fftshift(np.fft.fftfreq(fft_cols))
    q_row = np.fft.fftshift(np.fft.fftfreq(fft_rows))
    ky_over_k = q_col / horizontal_spacing_lambda
    kz_over_k = -q_row[::-1] / vertical_spacing_lambda
    return ky_over_k, kz_over_k


def calibrate_angular_cfr(
    raw_angular_cfr: np.ndarray,
    *,
    aperture_rows: int,
    aperture_cols: int,
    horizontal_spacing_lambda: float,
    vertical_spacing_lambda: float,
) -> AngularCalibration:
    """Convert the raw matrix FFT into a UE-local physical angular field.

    The input is the direct ``fftshift(fft2(aperture))`` from
    :func:`imaging.aperture_to_angular_fft`. The returned array has shape
    ``[kz, ky, frequency]`` with both coordinate arrays increasing in the
    physical local +z and +y directions.

    The phase correction follows directly from the centered PlanarArray
    positions

    ``y_j = d_h (j - (C-1)/2)``
    ``z_i = d_v ((R-1)/2 - i)``.

    It therefore preserves the complex phase that would be obtained by a
    Fourier sum using the actual centered antenna positions.
    """
    raw = np.asarray(raw_angular_cfr)
    if raw.ndim != 3:
        raise ValueError("raw_angular_cfr must have shape [fft_row, fft_col, frequency]")
    if aperture_rows < 1 or aperture_cols < 1:
        raise ValueError("aperture_rows and aperture_cols must be >= 1")
    if horizontal_spacing_lambda <= 0 or vertical_spacing_lambda <= 0:
        raise ValueError("antenna spacing must be > 0")

    fft_rows, fft_cols, _ = raw.shape
    q_row = np.fft.fftshift(np.fft.fftfreq(fft_rows))
    q_col = np.fft.fftshift(np.fft.fftfreq(fft_cols))

    # np.fft assumes sample positions i,j starting at zero. The actual Sionna
    # aperture is centered, so restore the phase origin to the UE center.
    phase_origin = np.exp(
        1j
        * 2.0
        * np.pi
        * (q_row[:, None] * (aperture_rows - 1) / 2.0 + q_col[None, :] * (aperture_cols - 1) / 2.0)
    )
    centered = raw * phase_origin[:, :, None]

    # PlanarArray row index increases toward local -z. Flip the FFT row axis so
    # the returned image increases toward physical local +z.
    calibrated = np.flip(centered, axis=0).copy()

    ky_over_k, kz_over_k = direction_cosine_axes(
        fft_rows=fft_rows,
        fft_cols=fft_cols,
        horizontal_spacing_lambda=horizontal_spacing_lambda,
        vertical_spacing_lambda=vertical_spacing_lambda,
    )

    return AngularCalibration(
        cfr=calibrated,
        ky_over_k=ky_over_k,
        kz_over_k=kz_over_k,
    )


def rotation_matrix(orientation: tuple[float, float, float]) -> np.ndarray:
    """Return Sionna's (z, y, x) device rotation matrix as a NumPy array.

    The columns are the device-local x, y and z axes expressed in world
    coordinates (``world_from_local``).
    """
    alpha, beta, gamma = orientation
    sa, ca = np.sin(alpha), np.cos(alpha)
    sb, cb = np.sin(beta), np.cos(beta)
    sg, cg = np.sin(gamma), np.cos(gamma)

    return np.array(
        [
            [ca * cb, ca * sb * sg - sa * cg, ca * sb * cg + sa * sg],
            [sa * cb, sa * sb * sg + ca * cg, sa * sb * cg - ca * sg],
            [-sb, cb * sg, cb * cg],
        ],
        dtype=np.float64,
    )


def geometric_los_source_direction_local(
    *,
    tx_position: tuple[float, float, float],
    ue_position: tuple[float, float, float],
    ue_orientation: tuple[float, float, float],
) -> np.ndarray:
    """Return the UE-local unit vector pointing from the UE toward the BS.

    Sionna's receive-array synthetic phase uses the arrival/source direction.
    The 2-D planar aperture observes only its local y-z projection; the sign of
    local x remains a front/back ambiguity for phase-only planar sampling.
    """
    tx = np.asarray(tx_position, dtype=np.float64)
    ue = np.asarray(ue_position, dtype=np.float64)
    world = tx - ue
    distance = float(np.linalg.norm(world))
    if distance == 0.0:
        raise ValueError("Tx and UE positions must differ")
    world /= distance

    rotation = rotation_matrix(ue_orientation)
    local = rotation.T @ world
    return local / np.linalg.norm(local)


def angular_peak_projection(
    angular_slice: np.ndarray,
    *,
    ky_over_k: np.ndarray,
    kz_over_k: np.ndarray,
) -> tuple[float, float, tuple[int, int]]:
    """Return the strongest angular bin as (ky/k, kz/k, (row, col))."""
    power = np.abs(np.asarray(angular_slice)) ** 2
    row, col = np.unravel_index(int(np.argmax(power)), power.shape)
    return float(ky_over_k[col]), float(kz_over_k[row]), (int(row), int(col))
