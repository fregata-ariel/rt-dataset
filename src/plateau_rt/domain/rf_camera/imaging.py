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


def uniform_frequency_spacing(frequency_offsets_hz: np.ndarray) -> float:
    """Return the grid spacing ``delta_f`` of a strictly increasing offset axis.

    ``frequency_offsets_hz`` must be a 1-D, finite, strictly increasing array
    with at least two bins; :func:`delay.angular_cfr_to_delay` sorts its axis
    before calling this helper.

    ``delta_f`` is taken as the exact endpoint slope ``(f[-1] - f[0]) / (N - 1)``
    in float64. That is more accurate than a median of quantised differences and
    is the value reproduced by a uniform grid.

    The uniformity tolerance includes a float32 term because the offsets are
    stored as float32: they are exactly the grid passed to Sionna's
    ``Paths.cfr``, so the sampled CFR is evaluated on that quantised grid. Above
    ``2**24`` Hz the float32 ulp reaches a few Hz and the endpoint slope can no
    longer be matched to better than ``4 * spacing(max |f|)``; a genuinely
    non-uniform grid still exceeds this tolerance.
    """
    frequencies = np.asarray(frequency_offsets_hz, dtype=np.float64)
    if frequencies.ndim != 1:
        raise ValueError("frequency offsets must be one-dimensional")
    if frequencies.size < 2:
        raise ValueError("at least two frequency bins are required")
    if not np.all(np.isfinite(frequencies)):
        raise ValueError("frequency offsets must be finite")
    if np.any(np.diff(frequencies) <= 0.0):
        raise ValueError("frequency offsets must contain distinct increasing bins")

    delta_f = float((frequencies[-1] - frequencies[0]) / (frequencies.size - 1))

    max_abs = np.max(np.abs(frequencies))
    tolerance = max(
        1e-3,
        1e-6 * abs(delta_f),
        4.0 * float(np.spacing(np.float32(max_abs))),
    )
    expected = frequencies[0] + np.arange(frequencies.size, dtype=np.float64) * delta_f
    if np.any(np.abs(frequencies - expected) > tolerance):
        raise ValueError("frequency offsets must be uniformly spaced")
    return delta_f


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


def split_pattern_axis(
    rx_ant_flat: np.ndarray,
    *,
    num_patterns: int,
    rows: int,
    cols: int,
) -> np.ndarray:
    """Split Sionna's fused receive axis into ``[pattern, row, col, ...]``.

    Sionna fuses the antenna-pattern axis with the array axis pattern-major:
    receive channel ``p * rows * cols + a`` is antenna ``a`` (column-first
    numbering) seen through pattern ``p``. The RF camera uses the two pattern
    slots for the front and back hemispheres.
    """
    rx_ant_flat = np.asarray(rx_ant_flat)
    size = rows * cols
    if rx_ant_flat.shape[0] != num_patterns * size:
        raise ValueError(
            f"Expected {num_patterns} x {size} receive channels, got {rx_ant_flat.shape[0]}"
        )
    patterns = rx_ant_flat.reshape((num_patterns, size) + rx_ant_flat.shape[1:])
    return np.stack([reshape_planar_column_first(p, rows=rows, cols=cols) for p in patterns])


def split_tx_pattern_axes(
    rx_cfr: np.ndarray,
    *,
    num_tx: int,
    num_patterns: int,
    rows: int,
    cols: int,
) -> np.ndarray:
    """Split one receiver's fused CFR into ``[tx, pattern, row, col, freq]``.

    ``rx_cfr`` is one receiver's ``cfr[rx, :, :, 0, 0, :]`` with shape
    ``[num_patterns*rows*cols, num_tx, freq]``. Each tx slice is split with
    :func:`split_pattern_axis` and stacked on a leading tx axis.
    """
    rx_cfr = np.asarray(rx_cfr)
    size = num_patterns * rows * cols
    if rx_cfr.ndim != 3 or rx_cfr.shape[0] != size or rx_cfr.shape[1] != num_tx:
        raise ValueError(f"Expected rx_cfr with shape [{size}, {num_tx}, freq], got {rx_cfr.shape}")
    return np.stack(
        [
            split_pattern_axis(rx_cfr[:, tx, :], num_patterns=num_patterns, rows=rows, cols=cols)
            for tx in range(num_tx)
        ]
    )


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
