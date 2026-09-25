"""Per-(view, BS) development of an RF-camera aperture CFR (NumPy only).

"Development" is the deterministic chain that turns one BS's compact
receive-aperture channel frequency response into the derived summaries stored
next to it: a spatial FFT of the local y-z aperture, the physical calibration
of the resulting angular spectrum, and the solid-angle amplitude ``A = kx * U``
(``kx = sqrt(1 - ky^2 - kz^2)``, so ``|A|^2`` is power per steradian). The
same grid then yields the center-frequency image and the angle-delay power
volume.

The writer, the web viewer and the test fixtures all share this module so the
combination of the individual NumPy steps is implemented once. Only NumPy and
sibling :mod:`plateau_rt.domain.rf_camera` modules are imported.

Hemisphere mapping: the planar y-z aperture can only resolve ``(ky, kz)``, so
the front (``kx >= 0``) and back (``kx < 0``) hemispheres are developed onto
the same grid. Pixel ``(ky, kz)`` is the UE-local direction
``(+sqrt(1 - ky^2 - kz^2), ky, kz)`` for the front hemisphere and
``(-sqrt(1 - ky^2 - kz^2), ky, kz)`` for the back hemisphere. ``+ky`` is
camera left, so the image is mirrored compared with a pinhole photo.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from plateau_rt.domain.rf_camera.calibration import calibrate_angular_cfr
from plateau_rt.domain.rf_camera.camera import to_solid_angle_amplitude
from plateau_rt.domain.rf_camera.delay import angular_cfr_to_delay, dominant_delay
from plateau_rt.domain.rf_camera.imaging import aperture_to_angular_fft


@dataclass(frozen=True)
class DevelopParams:
    """Geometry and trimming parameters of the RF-camera development."""

    fft_rows: int
    fft_cols: int
    rx_rows: int
    rx_cols: int
    horizontal_spacing_lambda: float
    vertical_spacing_lambda: float
    phase_floor_db: float

    def __post_init__(self) -> None:
        for name in ("fft_rows", "fft_cols", "rx_rows", "rx_cols"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.fft_rows < self.rx_rows or self.fft_cols < self.rx_cols:
            raise ValueError("FFT grid must not be smaller than the receive aperture")
        for name in ("horizontal_spacing_lambda", "vertical_spacing_lambda"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")
        if not np.isfinite(self.phase_floor_db):
            raise ValueError("phase_floor_db must be finite")


@dataclass(frozen=True)
class DevelopedImage:
    """Solid-angle angular image of one hemisphere and its local axes."""

    image: np.ndarray
    ky_over_k: np.ndarray
    kz_over_k: np.ndarray


@dataclass(frozen=True)
class CenterProducts:
    """Center-frequency summaries of a developed image."""

    center_cfr: np.ndarray
    center_power: np.ndarray
    phase_valid: np.ndarray


@dataclass(frozen=True)
class DelayProducts:
    """Per-direction dominant-delay summaries of a developed image."""

    dominant_delay_s: np.ndarray
    dominant_delay_power: np.ndarray


def _validate_aperture(aperture_hemi: np.ndarray, params: DevelopParams) -> np.ndarray:
    """Return ``aperture_hemi`` as an array or raise for a wrong shape."""
    aperture = np.asarray(aperture_hemi)
    if aperture.ndim != 3 or aperture.shape[:2] != (params.rx_rows, params.rx_cols):
        raise ValueError(
            "aperture_hemi must have shape [row, col, frequency] with "
            f"[row, col] == ({params.rx_rows}, {params.rx_cols})"
        )
    return aperture


def develop_hemisphere_image(
    aperture_hemi: np.ndarray,
    params: DevelopParams,
) -> DevelopedImage:
    """Develop one hemisphere's aperture CFR into a solid-angle image.

    ``aperture_hemi`` is complex ``[row, col, frequency]`` and must match the
    parameter aperture ``(rx_rows, rx_cols)``. The caller picks the hemisphere:
    the same function develops the front and the back hemisphere onto the same
    ``(ky, kz)`` grid, because the planar y-z aperture only sees ``(ky, kz)``.
    Pixel ``(ky, kz)`` is the UE-local direction
    ``(+sqrt(1 - ky^2 - kz^2), ky, kz)`` for the front hemisphere and
    ``(-sqrt(1 - ky^2 - kz^2), ky, kz)`` for the back hemisphere. ``+ky`` is
    camera left, so the image is mirrored compared with a pinhole photo.

    The image is not cast: a complex64 aperture yields a complex128 image
    because the calibration and solid-angle weights are float64.
    """
    aperture = _validate_aperture(aperture_hemi, params)
    calibration = calibrate_angular_cfr(
        aperture_to_angular_fft(aperture, fft_rows=params.fft_rows, fft_cols=params.fft_cols),
        aperture_rows=params.rx_rows,
        aperture_cols=params.rx_cols,
        horizontal_spacing_lambda=params.horizontal_spacing_lambda,
        vertical_spacing_lambda=params.vertical_spacing_lambda,
    )
    image = to_solid_angle_amplitude(calibration.cfr, calibration.ky_over_k, calibration.kz_over_k)
    return DevelopedImage(
        image=image,
        ky_over_k=calibration.ky_over_k,
        kz_over_k=calibration.kz_over_k,
    )


def center_frequency_products(
    image: np.ndarray,
    valid_mask: np.ndarray,
    phase_floor_db: float,
    *,
    freq_bin: int | None = None,
) -> CenterProducts:
    """Summarize one frequency bin of a developed image.

    ``image`` is ``[kz, ky, frequency]``; ``valid_mask`` is the boolean
    ``[kz, ky]`` propagating disk. ``freq_bin=None`` selects the writer's
    center bin ``frequency // 2``; an explicit bin must lie in
    ``[0, frequency)`` (negative indices are rejected). The float64 power picks
    the peak over ``valid_mask`` (floored at ``1e-30``) for the phase mask; the
    saved arrays are cast to complex64 / float32.
    """
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError("image must have shape [kz, ky, frequency]")
    mask = np.asarray(valid_mask)
    if mask.shape != image.shape[:2]:
        raise ValueError("valid_mask must have shape [kz, ky] matching the image")
    mask = mask.astype(bool, copy=False)

    num_frequencies = image.shape[-1]
    if freq_bin is None:
        freq_bin = num_frequencies // 2
    elif not 0 <= freq_bin < num_frequencies:
        raise ValueError(f"freq_bin must be in [0, {num_frequencies})")

    center = image[:, :, freq_bin]
    power = np.abs(center) ** 2
    peak = max(float(np.max(power[mask])), 1e-30)
    phase_valid = mask & (power >= peak * 10.0 ** (phase_floor_db / 10.0))
    return CenterProducts(
        center_cfr=center.astype(np.complex64, copy=False),
        center_power=power.astype(np.float32, copy=False),
        phase_valid=phase_valid,
    )


def angle_delay_power(
    image: np.ndarray,
    frequency_offsets_hz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(|cir|^2, delay_s)`` of a developed image's angle-delay volume.

    The complex ``[kz, ky, delay]`` volume is the NumPy ``ifft`` (backward) of
    the frequency axis, so by Parseval the sum over delay equals the MEAN over
    frequency bins of ``|image|^2`` (sum / N). The delay axis starts at 0 and
    has period ``N / B``. The maximum over delay per pixel is exactly
    :attr:`DelayProducts.dominant_delay_power` before the float32 cast.
    """
    volume = angular_cfr_to_delay(image, frequency_offsets_hz)
    return np.abs(volume.cir) ** 2, volume.delay_s


def delay_products(
    image: np.ndarray,
    valid_mask: np.ndarray,
    frequency_offsets_hz: np.ndarray,
) -> DelayProducts:
    """Return the per-direction dominant delay and power of a developed image.

    ``image`` is ``[kz, ky, frequency]`` and ``valid_mask`` its boolean
    ``[kz, ky]`` propagating disk. The delay is NaN and the power 0 wherever
    the pixel is outside the mask or carries no energy; outputs are float32.
    """
    image = np.asarray(image)
    if image.ndim != 3:
        raise ValueError("image must have shape [kz, ky, frequency]")
    mask = np.asarray(valid_mask)
    if mask.shape != image.shape[:2]:
        raise ValueError("valid_mask must have shape [kz, ky] matching the image")
    mask = mask.astype(bool, copy=False)

    power, delay_s = angle_delay_power(image, frequency_offsets_hz)
    _, dominant_delay_s, dominant_power = dominant_delay(power, delay_s)
    observed = mask & (np.asarray(dominant_power) > 0.0)

    dominant_delay_s = dominant_delay_s.astype(np.float32)
    dominant_delay_s[~observed] = np.nan
    dominant_power = dominant_power.astype(np.float32)
    dominant_power[~observed] = 0.0
    return DelayProducts(
        dominant_delay_s=dominant_delay_s,
        dominant_delay_power=dominant_power,
    )


def raw_spectrum_energy(aperture_hemi: np.ndarray, params: DevelopParams) -> float:
    """Return the raw aperture energy on the FFT grid (no calibration or weight).

    ``X = fft2(aperture)``; by Parseval
    ``sum |X|^2 / (fft_rows * fft_cols)`` equals ``sum |aperture_hemi|^2``,
    which is the manifest's ``hemisphere_energy``.
    """
    aperture = _validate_aperture(aperture_hemi, params)
    spectrum = np.fft.fft2(
        aperture.astype(np.complex128),
        s=(params.fft_rows, params.fft_cols),
        axes=(0, 1),
    )
    return float(np.sum(np.abs(spectrum) ** 2) / (params.fft_rows * params.fft_cols))
