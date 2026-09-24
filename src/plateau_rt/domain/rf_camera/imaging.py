"""RF-camera image formation from a planar receive aperture (NumPy only).

The raw observation is a complex channel frequency response (CFR) sampled
across a planar receive aperture. A 2-D spatial FFT is the first
"development" step that turns it into an angular-spectrum image.

This package (:mod:`plateau_rt.domain.rf_camera`) must not depend on Sionna,
so it can be tested and reused for post-processing without a GPU scene.
"""

from __future__ import annotations

import numpy as np


def frequency_offsets(bandwidth_hz: float, num_bins: int) -> np.ndarray:
    """Return evenly spaced baseband offsets centered on DC."""
    if num_bins == 1:
        return np.array([0.0], dtype=np.float32)
    # Endpoint=False gives a regular FFT/OFDM-like grid centered around DC.
    spacing = bandwidth_hz / num_bins
    indices = np.arange(num_bins, dtype=np.float64) - num_bins // 2
    return (indices * spacing).astype(np.float32)


def reshape_planar_column_first(
    aperture_flat: np.ndarray,
    *,
    rows: int,
    cols: int,
) -> np.ndarray:
    """Restore Sionna PlanarArray's column-first antenna numbering.

    `aperture_flat` has shape [rx_ant, ...]. Antenna indices walk down all
    rows of the first column before moving to the next column.
    """
    aperture_flat = np.asarray(aperture_flat)
    if aperture_flat.shape[0] != rows * cols:
        raise ValueError(f"Expected {rows * cols} antenna samples, got {aperture_flat.shape[0]}")

    out = np.empty((rows, cols) + aperture_flat.shape[1:], dtype=aperture_flat.dtype)
    for antenna_index in range(rows * cols):
        row = antenna_index % rows
        col = antenna_index // rows
        out[row, col, ...] = aperture_flat[antenna_index, ...]
    return out


def aperture_to_angular_fft(
    aperture_cfr: np.ndarray,
    *,
    fft_rows: int,
    fft_cols: int,
) -> np.ndarray:
    """Create a first-look angular spectrum by spatial FFT of the UE aperture.

    The receive aperture is the local y-z plane. The returned image is kept in
    raw spatial-frequency coordinates; :func:`calibration.calibrate_angular_cfr`
    converts it into a physically oriented direction-cosine image.
    """
    aperture_cfr = np.asarray(aperture_cfr)
    if aperture_cfr.ndim != 3:
        raise ValueError("aperture_cfr must have shape [row, col, frequency]")
    if fft_rows < aperture_cfr.shape[0] or fft_cols < aperture_cfr.shape[1]:
        raise ValueError("FFT grid must not be smaller than aperture")

    spectrum = np.fft.fft2(aperture_cfr, s=(fft_rows, fft_cols), axes=(0, 1))
    return np.fft.fftshift(spectrum, axes=(0, 1))


def raw_spatial_frequency_axes(
    *,
    fft_rows: int,
    fft_cols: int,
    horizontal_spacing_lambda: float,
    vertical_spacing_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the (horizontal, vertical) axes of the uncalibrated FFT image.

    fftfreq values are cycles/sample. Dividing by d/lambda converts them to
    direction-cosine-like spatial coordinates k_axis/k for the planar array.
    Unlike the calibrated axes, the vertical axis keeps the matrix row sign.
    """
    horizontal = np.fft.fftshift(np.fft.fftfreq(fft_cols))
    vertical = np.fft.fftshift(np.fft.fftfreq(fft_rows))
    return horizontal / horizontal_spacing_lambda, vertical / vertical_spacing_lambda
