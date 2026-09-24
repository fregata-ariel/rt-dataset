"""Align CFR pairs up to an unknown common phase and timing offset (NumPy only).

An observed CFR ``h_obs`` is compared with a rendered reference ``h_ref``
through the "gauge" model

    h_obs[..., f] ~= exp(j * (phi - 2 * pi * f * tau)) * h_ref[..., f]

where ``f`` are baseband frequency offsets (last axis, in Hz) and the leading
axes are independent aperture elements. A single ``phi`` and a single ``tau``
are shared by every element. The sign convention matches Sionna's
``Paths.cfr()``: a positive ``tau`` means ``h_obs`` arrives later than
``h_ref``.

The alignment is the exact least-squares solution, so it needs no
approximation: the phase is closed form given the delay, and the delay is the
maximiser of a smooth single-variable function whose derivative is analytic.

This module must stay importable without Sionna/Mitsuba, so it only depends on
NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from plateau_rt.domain.rf_camera.imaging import uniform_frequency_spacing

__all__ = [
    "GaugeAlignment",
    "align_common_phase_and_delay",
    "gauge_aligned_nmse",
    "nmse",
]

_TWO_PI = 2.0 * np.pi


@dataclass(frozen=True)
class GaugeAlignment:
    """Least-squares gauge and the reference it produces.

    Attributes:
        phase_rad: Common phase ``phi``, wrapped to ``(-pi, pi]``.
        delay_s: Common delay ``tau``, wrapped to ``[-T/2, T/2)`` with
            ``T = 1 / delta_f``.
        aligned_ref: ``exp(j * (phi - 2 * pi * f * tau)) * h_ref`` as
            ``complex128`` with the same shape as ``h_ref``.
    """

    phase_rad: float
    delay_s: float
    aligned_ref: np.ndarray


def nmse(h_obs: np.ndarray, h_ref: np.ndarray) -> float:
    """Normalised mean squared error ``sum|h_obs - h_ref|^2 / sum|h_obs|^2``.

    Returns ``inf`` when ``sum|h_obs|^2 == 0`` (the relative error is undefined
    rather than a finite number). Inputs are cast to ``complex128``.
    """
    observed = np.asarray(h_obs, dtype=np.complex128)
    reference = np.asarray(h_ref, dtype=np.complex128)
    denominator = float(np.sum(np.abs(observed) ** 2))
    if denominator == 0.0:
        return float("inf")
    residual = float(np.sum(np.abs(observed - reference) ** 2))
    return residual / denominator


def align_common_phase_and_delay(
    h_obs: np.ndarray,
    h_ref: np.ndarray,
    frequency_offsets_hz: np.ndarray,
    *,
    oversample: int = 16,
) -> GaugeAlignment:
    """Find the shared ``(phi, tau)`` that best matches ``h_obs`` to ``h_ref``.

    Args:
        h_obs: Observed CFR; last axis is frequency, leading axes are elements.
        h_ref: Reference CFR with exactly the same shape as ``h_obs``.
        frequency_offsets_hz: 1-D, uniformly spaced baseband offsets in Hz,
            one per frequency bin. They are sorted before use.
        oversample: Number of coarse delay-grid points per frequency bin.

    Returns:
        A :class:`GaugeAlignment` whose ``aligned_ref`` minimises the squared
        error against ``h_obs``.

    Raises:
        ValueError: If the shapes differ, either CFR contains a non-finite
            value, the grid is not 1-D/finite/uniform, or fewer than two
            frequency bins are supplied.
    """
    if oversample < 1:
        raise ValueError("oversample must be >= 1")

    observed, reference, frequencies, delta_f, reference_caller, frequencies_caller = _prepare(
        h_obs, h_ref, frequency_offsets_hz
    )

    element_axes = tuple(range(observed.ndim - 1))
    cross_spectrum = np.sum(observed * np.conj(reference), axis=element_axes)

    # All-zero cross-spectrum (e.g. h_ref == 0): the gauge is unidentifiable.
    if not np.any(cross_spectrum):
        return GaugeAlignment(
            phase_rad=0.0,
            delay_s=0.0,
            aligned_ref=reference_caller.copy(),
        )

    period = 1.0 / delta_f
    num_bins = frequencies.size

    # Coarse search: evaluate |S(tau)| directly on one period. The true
    # frequencies (not bin indices) enter the exponent, so the recovered phase
    # is unbiased for the non-symmetric even-N grid.
    coarse_steps = oversample * num_bins
    coarse_step = period / coarse_steps
    tau_grid = (np.arange(coarse_steps) - coarse_steps // 2) * coarse_step
    coarse = np.exp(1j * _TWO_PI * np.outer(tau_grid, frequencies)) @ cross_spectrum
    peak = int(np.argmax(np.abs(coarse)))

    tau = _refine_delay(tau_grid[peak], coarse_step, frequencies, cross_spectrum)
    tau = _wrap_delay(tau, period)

    spectrum = np.exp(1j * _TWO_PI * tau * frequencies) @ cross_spectrum
    phi = _wrap_phase(float(np.angle(spectrum)))

    # Rebuild the aligned reference in the caller's frequency order.
    alignment = np.exp(1j * (phi - _TWO_PI * frequencies_caller * tau))
    return GaugeAlignment(
        phase_rad=phi,
        delay_s=float(tau),
        aligned_ref=alignment * reference_caller,
    )


def gauge_aligned_nmse(
    h_obs: np.ndarray,
    h_ref: np.ndarray,
    frequency_offsets_hz: np.ndarray,
    *,
    oversample: int = 16,
) -> float:
    """NMSE after aligning ``h_ref`` to ``h_obs`` with the common gauge."""
    aligned = align_common_phase_and_delay(
        h_obs, h_ref, frequency_offsets_hz, oversample=oversample
    )
    return nmse(h_obs, aligned.aligned_ref)


def _prepare(
    h_obs: np.ndarray,
    h_ref: np.ndarray,
    frequency_offsets_hz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """Validate inputs, cast to float64/complex128 and sort by frequency.

    Returns the frequency-sorted observed/reference/frequency arrays together
    with ``delta_f``, plus the caller-order reference and caller-order float64
    frequencies so ``aligned_ref`` can be rebuilt in the caller's order.
    """
    observed = np.asarray(h_obs, dtype=np.complex128)
    reference = np.asarray(h_ref, dtype=np.complex128)
    frequencies = np.asarray(frequency_offsets_hz, dtype=np.float64)

    if observed.shape != reference.shape:
        raise ValueError("h_obs and h_ref must have the same shape")
    if frequencies.ndim != 1 or frequencies.size != observed.shape[-1]:
        raise ValueError("frequency_offsets_hz must match the CFR frequency axis")
    if frequencies.size < 2:
        raise ValueError("at least two frequency bins are required for delay alignment")
    if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(reference)):
        raise ValueError("h_obs and h_ref must be finite")

    order = np.argsort(frequencies)
    frequencies_sorted = frequencies[order]
    delta_f = uniform_frequency_spacing(frequencies_sorted)

    return (
        observed[..., order],
        reference[..., order],
        frequencies_sorted,
        delta_f,
        reference,
        frequencies,
    )


def _spectrum_gradient(
    tau: float,
    frequencies: np.ndarray,
    cross_spectrum: np.ndarray,
) -> float:
    """Return ``d|S|^2/dtau`` at ``tau``.

    ``S(tau) = sum_f C(f) exp(j 2 pi f tau)`` and its derivative have closed
    forms, so the squared magnitude is differentiated analytically.
    """
    angular = _TWO_PI * frequencies
    exponential = np.exp(1j * tau * angular)
    spectrum = exponential @ cross_spectrum
    first = (1j * angular * exponential) @ cross_spectrum

    return 2.0 * float(np.real(np.conj(spectrum) * first))


def _refine_delay(
    tau0: float,
    step: float,
    frequencies: np.ndarray,
    cross_spectrum: np.ndarray,
) -> float:
    """Maximise ``|S(tau)|`` within ``[tau0 - step, tau0 + step]``.

    The coarse peak brackets the true maximum to within one coarse step. On
    that bracket ``|S|^2`` is unimodal, so bisecting its analytic derivative to
    floating-point resolution converges to the exact maximiser. This is exact
    even for off-grid delays, unlike parabolic interpolation of ``|S|``.
    """
    lower = tau0 - step
    upper = tau0 + step

    if _spectrum_gradient(lower, frequencies, cross_spectrum) <= 0.0:
        return lower
    if _spectrum_gradient(upper, frequencies, cross_spectrum) >= 0.0:
        return upper

    for _ in range(100):
        middle = 0.5 * (lower + upper)
        if middle == lower or middle == upper:
            break
        if _spectrum_gradient(middle, frequencies, cross_spectrum) > 0.0:
            lower = middle
        else:
            upper = middle

    return 0.5 * (lower + upper)


def _wrap_delay(tau: float, period: float) -> float:
    """Wrap ``tau`` into ``[-T/2, T/2)``."""
    return float((tau + 0.5 * period) % period - 0.5 * period)


def _wrap_phase(phi: float) -> float:
    """Wrap ``phi`` into ``(-pi, pi]``."""
    wrapped = float(np.angle(np.exp(1j * phi)))
    if wrapped <= -np.pi:
        wrapped = np.pi
    return wrapped
