"""Receiver impairments for the ideal two-hemisphere aperture CFR (NumPy only).

The ideal RF camera records a complex receive-aperture CFR separately for the
front (``local kx >= 0``) and back (``local kx < 0``) hemispheres of the UE
aperture. A real single-channel receiver collapses those hemispheres, applies
per-element complex gain errors, a common timing ramp and phase, and additive
noise. This module synthesizes that "observed" CFR and returns a JSON-safe
ground-truth record of the parameters that were actually applied.

The impairment order is fixed and the random variates are drawn in a fixed
order. Every Gaussian/uniform draw of the element errors, timing offset and
common phase is consumed even when its spread is zero, so their RNG stream
layout does not depend on the configuration values; the noise is drawn last and
only when the noise variance is positive. A zero-valued configuration without
noise reproduces the ideal front hemisphere exactly.

Noise is a **dataset-level** setting: a physical receiver has one noise floor,
not one per view. :class:`NoiseSpec` describes how that floor is chosen (an
absolute complex variance, or an SNR relative to a caller-supplied reference
power) while :func:`apply_impairments` only receives the already-resolved
noise variance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "ElementErrors",
    "ImpairmentConfig",
    "NoiseSpec",
    "apply_impairments",
    "draw_element_errors",
    "front_to_back_gain",
    "isotropic_mean_power",
    "resolve_noise_variance",
    "timing_phase_ramp",
]


@dataclass(frozen=True)
class ImpairmentConfig:
    """Configuration of the ordered single-channel receiver impairment chain."""

    front_to_back_db: float | None = None
    """Front-to-back ratio in dB; ``None`` means an ideal front-only receiver."""

    common_phase_deg: float = 0.0
    """Fixed UE common phase in degrees."""

    random_common_phase: bool = False
    """If ``True``, draw a uniform ``[0, 360)`` degree common phase instead."""

    timing_offset_ns: float = 0.0
    """Fixed timing offset in nanoseconds."""

    timing_offset_std_ns: float = 0.0
    """Standard deviation of an extra per-link Gaussian timing offset (ns)."""

    element_gain_std_db: float = 0.0
    """Per-element amplitude error standard deviation in dB."""

    element_phase_std_deg: float = 0.0
    """Per-element phase error standard deviation in degrees."""

    def __post_init__(self) -> None:
        for name in ("timing_offset_std_ns", "element_gain_std_db", "element_phase_std_deg"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and >= 0, got {value!r}")
        for name in ("timing_offset_ns", "common_phase_deg"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}")
        if self.front_to_back_db is not None and not math.isfinite(float(self.front_to_back_db)):
            raise ValueError(f"front_to_back_db must be finite, got {self.front_to_back_db!r}")
        if self.random_common_phase and float(self.common_phase_deg) != 0.0:
            raise ValueError(
                "random_common_phase=True requires common_phase_deg == 0.0, "
                f"got {self.common_phase_deg!r}"
            )


@dataclass(frozen=True)
class ElementErrors:
    """Per-element complex gain error of one UE receive array (constant in frequency)."""

    gain_db: np.ndarray
    """Amplitude error ``[row, col]`` in dB (float64)."""

    phase_rad: np.ndarray
    """Phase error ``[row, col]`` in radians (float64)."""

    @property
    def complex_gain(self) -> np.ndarray:
        """Complex gain ``10 ** (gain_db / 20) * exp(1j * phase_rad)``."""
        return (10.0 ** (self.gain_db / 20.0)) * np.exp(1j * self.phase_rad)


@dataclass(frozen=True)
class NoiseSpec:
    """Dataset-wide noise-floor definition (absolute variance or SNR)."""

    snr_db: float | None = None
    """SNR in dB relative to the caller-supplied dataset reference power."""

    noise_variance: float | None = None
    """Absolute complex noise variance per element per bin (same units as ``|CFR|^2``)."""

    def __post_init__(self) -> None:
        if self.snr_db is not None and self.noise_variance is not None:
            raise ValueError("snr_db and noise_variance are mutually exclusive")
        if self.snr_db is not None and not math.isfinite(float(self.snr_db)):
            raise ValueError(f"snr_db must be finite, got {self.snr_db!r}")
        if self.noise_variance is not None:
            value = float(self.noise_variance)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"noise_variance must be finite and >= 0, got {value!r}")

    @property
    def mode(self) -> str:
        """One of ``"snr_relative_to_reference"``, ``"absolute"`` or ``"none"``."""
        if self.snr_db is not None:
            return "snr_relative_to_reference"
        if self.noise_variance is not None:
            return "absolute"
        return "none"


def front_to_back_gain(front_to_back_db: float | None) -> float:
    """Return the linear front-to-back gain ``g``.

    ``None`` selects the ideal front-only receiver (``g = 0``). Otherwise
    ``g = 10 ** (-front_to_back_db / 20)`` so a positive dB ratio attenuates the
    back hemisphere relative to the front.
    """
    if front_to_back_db is None:
        return 0.0
    return float(10.0 ** (-float(front_to_back_db) / 20.0))


def timing_phase_ramp(frequency_offsets_hz: np.ndarray, tau_s: float) -> np.ndarray:
    """Return ``exp(-1j * 2*pi * f * tau)`` sampled on the frequency offsets."""
    frequencies = np.asarray(frequency_offsets_hz, dtype=np.float64)
    return np.exp(-1j * 2.0 * np.pi * frequencies * float(tau_s))


def isotropic_mean_power(pair_cfr: np.ndarray) -> float:
    """Mean ``|front + back|^2`` over ``(row, col, frequency)`` in float64.

    ``pair_cfr`` has shape ``[2, row, col, frequency]`` and ``front + back`` is
    the ideal isotropic element.
    """
    cfr = np.asarray(pair_cfr)
    if cfr.ndim != 4 or cfr.shape[0] != 2:
        raise ValueError("pair_cfr must have shape [2, row, col, frequency]")
    combined = cfr[0].astype(np.complex128) + cfr[1].astype(np.complex128)
    return float(np.mean(np.abs(combined) ** 2))


def resolve_noise_variance(spec: NoiseSpec, reference_power: float) -> float:
    """Resolve ``spec`` into an absolute complex noise variance.

    ``none`` returns ``0.0``, ``absolute`` returns the configured variance and
    ``snr_relative_to_reference`` returns ``reference_power / 10**(snr_db/10)``.
    """
    if spec.mode == "none":
        return 0.0
    if spec.mode == "absolute":
        variance = spec.noise_variance
        assert variance is not None  # guaranteed by NoiseSpec.mode
        return float(variance)
    snr_db = spec.snr_db
    assert snr_db is not None  # guaranteed by NoiseSpec.mode
    reference_power = float(reference_power)
    if not math.isfinite(reference_power) or reference_power <= 0.0:
        raise ValueError(
            "snr_db is relative to a dataset reference power, but reference_power "
            f"is {reference_power!r} (nothing to reference)"
        )
    return float(reference_power / (10.0 ** (float(snr_db) / 10.0)))


def draw_element_errors(
    config: ImpairmentConfig, rows: int, cols: int, rng: np.random.Generator
) -> ElementErrors:
    """Draw one view's per-element amplitude and phase errors.

    The amplitude errors are drawn before the phase errors, always, so the RNG
    stream layout does not depend on the standard deviations.
    """
    gain_db = rng.normal(0.0, float(config.element_gain_std_db), size=(rows, cols))
    phase_deg = rng.normal(0.0, float(config.element_phase_std_deg), size=(rows, cols))
    return ElementErrors(gain_db=gain_db, phase_rad=np.deg2rad(phase_deg))


def apply_impairments(
    aperture_cfr: np.ndarray,
    frequency_offsets_hz: np.ndarray,
    config: ImpairmentConfig,
    rng: np.random.Generator,
    *,
    element_errors: ElementErrors | None = None,
    noise_variance: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the ordered impairment chain and return ``(observed, gt)``.

    ``aperture_cfr`` has shape ``[hemisphere(2), row, col, frequency]``.
    ``observed`` is the single-channel ``complex64`` CFR with shape
    ``[row, col, frequency]`` and ``gt`` is a JSON-safe dict recording the
    applied parameters. ``element_errors`` are the per-view element errors,
    shared by all BSs of a view; when ``None`` they are drawn from ``rng``
    (standalone use). ``noise_variance`` is the already-resolved dataset-wide
    complex noise variance per element per bin.

    The chain order is:

    1. ``front + g * back``
    2. per-element complex gain ``G[row, col]`` (constant over frequency)
    3. timing ramp ``exp(-1j * 2*pi * f * tau)``
    4. common phase ``exp(1j * phi)``
    5. circular complex AWGN with the given ``noise_variance``

    From ``rng`` the extra timing offset, the uniform common phase and (only
    when ``noise_variance > 0``) the noise are drawn, in that order.
    """
    if not math.isfinite(float(noise_variance)) or float(noise_variance) < 0.0:
        raise ValueError(f"noise_variance must be finite and >= 0, got {noise_variance!r}")

    cfr = np.asarray(aperture_cfr)
    if cfr.ndim != 4 or cfr.shape[0] != 2:
        raise ValueError("aperture_cfr must have shape [2, row, col, frequency]")

    frequencies = np.asarray(frequency_offsets_hz)
    if frequencies.ndim != 1 or frequencies.size != cfr.shape[-1]:
        raise ValueError("frequency_offsets_hz must match the CFR frequency axis")

    rows, cols = cfr.shape[1], cfr.shape[2]

    if element_errors is None:
        element_errors = draw_element_errors(config, rows, cols, rng)
    elif element_errors.gain_db.shape != (rows, cols) or element_errors.phase_rad.shape != (
        rows,
        cols,
    ):
        raise ValueError(
            "element_errors shape does not match the aperture: "
            f"gain {element_errors.gain_db.shape}, phase {element_errors.phase_rad.shape}, "
            f"expected {(rows, cols)}"
        )

    # Draw every variate in a fixed order, including zero-spread ones, so the
    # RNG stream layout depends on the array shape but never on the config.
    extra_tau_s = float(rng.normal(0.0, float(config.timing_offset_std_ns) * 1e-9))
    uniform_common_phase_deg = float(rng.uniform(0.0, 360.0))

    # 1. Collapse the hemispheres with the front-to-back gain.
    g = front_to_back_gain(config.front_to_back_db)
    y = cfr[0] + g * cfr[1]

    # 2. Per-element complex gain, constant over frequency.
    element_gain = element_errors.complex_gain.astype(np.complex64)
    y = y * element_gain[:, :, None]

    # 3. Timing ramp.
    tau_s = float(config.timing_offset_ns) * 1e-9 + extra_tau_s
    ramp = timing_phase_ramp(frequencies, tau_s).astype(np.complex64)
    y = y * ramp

    # 4. Common phase.
    common_phase_deg = (
        uniform_common_phase_deg if config.random_common_phase else float(config.common_phase_deg)
    )
    common_phase_rad = float(np.deg2rad(common_phase_deg))
    y = y * np.exp(1j * common_phase_rad).astype(np.complex64)

    noiseless = y
    signal_power = float(np.mean(np.abs(noiseless) ** 2))

    # 5. Circular complex AWGN, whenever a positive noise floor is given.
    noise_variance = float(noise_variance)
    expected_snr_db: float | None = None
    achieved_snr_db: float | None = None
    if noise_variance > 0.0:
        noise_std = np.sqrt(noise_variance / 2.0)
        noise = rng.normal(0.0, noise_std, size=y.shape) + 1j * rng.normal(
            0.0, noise_std, size=y.shape
        )
        y = noiseless + noise
        if signal_power > 0.0:
            expected_snr_db = float(10.0 * np.log10(signal_power / noise_variance))
            realized_noise_power = float(np.mean(np.abs(y - noiseless) ** 2))
            if realized_noise_power > 0.0:
                achieved_snr_db = float(10.0 * np.log10(signal_power / realized_noise_power))

    observed = np.asarray(y).astype(np.complex64, copy=False)
    gt: dict[str, Any] = {
        "front_to_back_db": config.front_to_back_db,
        "g": g,
        "common_phase_rad": common_phase_rad,
        "timing_offset_s": tau_s,
        "element_gain_db": element_errors.gain_db.tolist(),
        "element_phase_rad": element_errors.phase_rad.tolist(),
        "signal_power": signal_power,
        "noise_variance": noise_variance,
        "expected_snr_db": expected_snr_db,
        "achieved_snr_db": achieved_snr_db,
    }
    return observed, gt
