"""Angle-delay development of a calibrated angular CFR (NumPy only).

An IFFT along the uniformly sampled baseband-frequency axis turns the
physically calibrated complex angular CFR into a tensor with axes

    [UE-local kz/k, UE-local ky/k, delay]

that keeps complex phase, so delay bins can be treated as image channels or
as a small RF volume.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SPEED_OF_LIGHT_M_S = 299_792_458.0


@dataclass(frozen=True)
class AngularDelayVolume:
    """Complex angle-delay response and its physical axes."""

    cir: np.ndarray
    delay_s: np.ndarray
    frequency_spacing_hz: float
    unambiguous_delay_s: float

    @property
    def delay_resolution_s(self) -> float:
        """Delay bin width, set by the total sampled bandwidth."""
        return 1.0 / (len(self.delay_s) * self.frequency_spacing_hz)


def angular_cfr_to_delay(
    angular_cfr: np.ndarray,
    frequency_offsets_hz: np.ndarray,
) -> AngularDelayVolume:
    """IFFT a centered, uniformly sampled CFR into positive modulo-delay bins.

    ``frequency_offsets_hz`` is expected to contain the baseband offsets used
    for ``Paths.cfr()``, ordered from negative to positive frequency. The
    resulting delay axis starts at zero and spans one unambiguous delay period
    ``1 / delta_f``. Absolute Sionna delays therefore appear modulo this period.
    """
    cfr = np.asarray(angular_cfr)
    frequencies = np.asarray(frequency_offsets_hz, dtype=np.float64)

    if cfr.ndim != 3:
        raise ValueError("angular_cfr must have shape [kz, ky, frequency]")
    if frequencies.ndim != 1 or frequencies.size != cfr.shape[-1]:
        raise ValueError("frequency_offsets_hz must match the CFR frequency axis")
    if frequencies.size < 2:
        raise ValueError("at least two frequency bins are required for delay imaging")

    order = np.argsort(frequencies)
    frequencies = frequencies[order]
    cfr = cfr[..., order]

    differences = np.diff(frequencies)
    delta_f = float(np.median(differences))
    if delta_f <= 0.0:
        raise ValueError("frequency offsets must contain distinct increasing bins")
    if not np.allclose(differences, delta_f, rtol=1e-6, atol=max(1e-3, abs(delta_f) * 1e-9)):
        raise ValueError("frequency offsets must be uniformly spaced")

    # The stored frequency axis is centered as [-B/2, ..., 0, ..., +B/2).
    # Move DC to index zero before using NumPy's inverse DFT convention.
    cir = np.fft.ifft(np.fft.ifftshift(cfr, axes=-1), axis=-1)

    num_bins = frequencies.size
    delay_resolution_s = 1.0 / (num_bins * delta_f)
    delay_s = np.arange(num_bins, dtype=np.float64) * delay_resolution_s

    return AngularDelayVolume(
        cir=cir,
        delay_s=delay_s,
        frequency_spacing_hz=delta_f,
        unambiguous_delay_s=1.0 / delta_f,
    )


def dominant_delay(
    power: np.ndarray,
    delay_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the strongest delay per direction as ``(bin, delay_s, power)``.

    ``power`` is an angle-delay power volume ``[kz, ky, delay]``.
    """
    dominant_bin = np.argmax(power, axis=-1)
    dominant_power = np.take_along_axis(power, dominant_bin[..., None], axis=-1)[..., 0]
    return dominant_bin, np.asarray(delay_s)[dominant_bin], dominant_power


def propagating_direction_mask(
    ky_over_k: np.ndarray,
    kz_over_k: np.ndarray,
    *,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """Return the far-field propagating disk for a planar y-z aperture.

    A real plane-wave direction must satisfy ``kx^2 + ky^2 + kz^2 = 1``.
    The planar aperture does not determine the sign of ``kx`` but it does
    determine whether a sampled y-z projection can correspond to a propagating
    wave at all.
    """
    ky = np.asarray(ky_over_k, dtype=np.float64)[None, :]
    kz = np.asarray(kz_over_k, dtype=np.float64)[:, None]
    return ky**2 + kz**2 <= 1.0 + tolerance


def geometric_los_delay_s(
    tx_position: tuple[float, float, float],
    ue_position: tuple[float, float, float],
) -> float:
    """Free-space geometric delay from BS to UE aperture center."""
    tx = np.asarray(tx_position, dtype=np.float64)
    ue = np.asarray(ue_position, dtype=np.float64)
    return float(np.linalg.norm(tx - ue) / SPEED_OF_LIGHT_M_S)


def circular_delay_error_s(value: float, reference: float, period: float) -> float:
    """Shortest delay error on a modulo-delay circle."""
    if period <= 0.0:
        raise ValueError("period must be > 0")
    difference = abs((value - reference) % period)
    return float(min(difference, period - difference))
