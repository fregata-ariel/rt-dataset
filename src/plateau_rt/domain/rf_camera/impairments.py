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

Array-level tomography helpers
------------------------------
The tomography stack consumes a whole dataset as
``aperture_cfr[V, B, 2, R, C, N]`` (views, BSs, hemisphere, row, col,
frequency). :func:`apply_hardware_impairments` applies the gauge-free receiver
chain to every capture and returns ``[V, B, 1, R, C, N]`` plus per-``(v, b)``
ground truth; the per-capture clock gauge is added separately by
:func:`apply_gauge` (the same helper re-exported by ``rf_tomography.sync``).
:func:`calibration_capture` models a known boresight source through the same
input-referred chain, :func:`perturb_poses` reports jittered UE poses for the
solver, and :func:`capture_gt_record` flattens the per-``(v, b)`` ground truth
into a JSON-safe record.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "CalibrationCapture",
    "ElementErrors",
    "ImpairmentConfig",
    "NoiseSpec",
    "PosePerturbation",
    "apply_gauge",
    "apply_hardware_impairments",
    "apply_impairments",
    "calibration_capture",
    "capture_gt_record",
    "draw_element_errors",
    "front_to_back_gain",
    "gauge_factor",
    "isotropic_mean_power",
    "perturb_poses",
    "resolve_noise_variance",
    "rotation_from_vector",
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


def gauge_factor(phi: np.ndarray, tau: np.ndarray, freq_offsets: np.ndarray) -> np.ndarray:
    """Return the unit-modulus gauge ``[*S, N]`` for per-capture ``phi``/``tau``."""
    phi_arr = np.asarray(phi, dtype=np.float64)
    tau_arr = np.asarray(tau, dtype=np.float64)
    freq = np.asarray(freq_offsets, dtype=np.float64)
    if phi_arr.shape != tau_arr.shape:
        raise ValueError("phi and tau must have identical shapes")
    if freq.ndim != 1:
        raise ValueError("freq_offsets must be one-dimensional")
    if not np.all(np.isfinite(phi_arr)) or not np.all(np.isfinite(tau_arr)):
        raise ValueError("phi and tau must be finite")
    if not np.all(np.isfinite(freq)):
        raise ValueError("freq_offsets must be finite")
    return np.exp(1j * phi_arr[..., None]) * np.exp(-2j * np.pi * freq * tau_arr[..., None])


def apply_gauge(
    Y: np.ndarray, phi: np.ndarray, tau: np.ndarray, freq_offsets: np.ndarray
) -> np.ndarray:
    """Return ``Y`` multiplied by the per-capture gauge factor."""
    arr = np.asarray(Y, dtype=np.complex128)
    if arr.ndim != 6:
        raise ValueError("Y must have shape [V, B, H, R, C, N]")
    phi_arr = np.asarray(phi, dtype=np.float64)
    tau_arr = np.asarray(tau, dtype=np.float64)
    if phi_arr.shape != tau_arr.shape or phi_arr.shape != arr.shape[:2]:
        raise ValueError("phi and tau must have shape [V, B]")
    freq = np.asarray(freq_offsets, dtype=np.float64)
    if freq.shape != (arr.shape[-1],):
        raise ValueError("len(freq_offsets) must equal Y.shape[-1]")
    gauge = gauge_factor(phi_arr, tau_arr, freq)
    return (arr * gauge[:, :, None, None, None, :]).astype(np.complex128, copy=False)


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


def apply_hardware_impairments(
    aperture_cfr: np.ndarray,
    *,
    element_gain: np.ndarray | None = None,
    front_to_back_db: float | None = None,
    noise_var_abs: float = 0.0,
    noise: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the gauge-free receiver chain to a whole dataset.

    ``aperture_cfr`` has shape ``[V, B, 2, R, C, N]`` and the result is the
    observed single-channel ``[V, B, 1, R, C, N]``. The chain collapses the
    hemispheres (``front + g * back`` with ``g = front_to_back_gain``) and then
    applies the per-element complex gain ``element_gain`` (``[V, R, C]`` per
    view, shared by all BSs, or ``[R, C]`` broadcast; ``None`` means all ones).

    The noise is **input-referred**: it is added to the collapsed channel
    *before* the element gain, so a receive chain with gain error ``G`` outputs
    ``G * (s + w)`` and, after pre-calibration by ``G_hat``, the data are
    ``eps * (s + w)`` with ``eps = G / G_hat``. This differs on purpose from
    :func:`apply_impairments`, whose noise is added after the gain.
    ``noise_var_abs`` is the **absolute** complex variance ``sigma^2`` per
    element per bin (same units as ``|CFR|^2``) and never depends on the signal
    power. ``noise`` may supply a pre-drawn ``[V, B, R, C, N]`` realisation
    used verbatim. ``rng`` is required only when ``noise is None`` and
    ``noise_var_abs > 0``; otherwise it is ignored and not advanced.

    Returns ``(observed, gt)`` where ``gt`` records the applied parameters and
    the per-``(v, b)`` signal/noise powers and SNRs.
    """
    cfr = np.asarray(aperture_cfr, dtype=np.complex128)
    if cfr.ndim != 6 or cfr.shape[2] != 2:
        raise ValueError("aperture_cfr must have shape [V, B, 2, R, C, N]")
    if not np.all(np.isfinite(cfr)):
        raise ValueError("aperture_cfr must be finite")
    num_views, num_bs, _, num_rows, num_cols, num_bins = cfr.shape

    if element_gain is None:
        gain = np.ones((num_views, num_rows, num_cols), dtype=np.complex128)
    else:
        raw_gain = np.asarray(element_gain, dtype=np.complex128)
        if not np.all(np.isfinite(raw_gain)):
            raise ValueError("element_gain must be finite")
        if raw_gain.shape == (num_rows, num_cols):
            gain = np.broadcast_to(raw_gain, (num_views, num_rows, num_cols)).copy()
        elif raw_gain.shape == (num_views, num_rows, num_cols):
            gain = raw_gain.copy()
        else:
            raise ValueError("element_gain must have shape [V, R, C] or [R, C]")

    if front_to_back_db is not None and not math.isfinite(float(front_to_back_db)):
        raise ValueError(f"front_to_back_db must be finite, got {front_to_back_db!r}")
    g = front_to_back_gain(front_to_back_db)

    sigma2 = float(noise_var_abs)
    if not math.isfinite(sigma2) or sigma2 < 0.0:
        raise ValueError(f"noise_var_abs must be finite and >= 0, got {noise_var_abs!r}")

    w: np.ndarray
    if noise is not None:
        w = np.asarray(noise, dtype=np.complex128)
        if w.shape != (num_views, num_bs, num_rows, num_cols, num_bins):
            raise ValueError("noise must have shape [V, B, R, C, N]")
        if not np.all(np.isfinite(w)):
            raise ValueError("noise must be finite")
    elif sigma2 > 0.0:
        if not isinstance(rng, np.random.Generator):
            raise ValueError("rng is required when noise is None and noise_var_abs > 0")
        z = rng.standard_normal((2, num_views, num_bs, num_rows, num_cols, num_bins))
        w = math.sqrt(sigma2 / 2.0) * (z[0] + 1j * z[1])
    else:
        w = np.zeros_like(cfr[:, :, 0])

    collapsed = cfr[:, :, 0] + g * cfr[:, :, 1]
    gain6 = gain[:, None, :, :, None]
    observed = (gain6 * (collapsed + w))[:, :, None]
    signal = gain6 * collapsed
    noise_out = gain6 * w

    signal_power = np.mean(np.abs(signal) ** 2, axis=(2, 3, 4))
    noise_power = np.mean(np.abs(noise_out) ** 2, axis=(2, 3, 4))
    gain_power = np.mean(np.abs(gain) ** 2, axis=(1, 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        expected_snr_db = 10.0 * np.log10(signal_power / (sigma2 * gain_power[:, None]))
        achieved_snr_db = 10.0 * np.log10(signal_power / noise_power)

    gt: dict[str, Any] = {
        "front_to_back_db": front_to_back_db,
        "front_to_back_gain": g,
        "element_gain": gain,
        "noise_var": sigma2,
        "signal_power": signal_power,
        "noise_power": noise_power,
        "expected_snr_db": expected_snr_db,
        "achieved_snr_db": achieved_snr_db,
    }
    return observed.astype(np.complex128, copy=False), gt


@dataclass(frozen=True)
class CalibrationCapture:
    """Calibration capture of a known far-field source and the gain estimate."""

    element_gain: np.ndarray
    """True complex gains ``[..., R, C]`` (complex128 copy)."""

    capture: np.ndarray
    """Recorded calibration CFR ``[..., R, C, N]`` (complex128)."""

    element_gain_est: np.ndarray
    """Estimated gains ``[..., R, C]`` (complex128)."""

    snr_db: float
    """Per-sample SNR of the calibration capture."""

    noise_var: float
    """Its noise variance (the known source has unit power)."""

    @property
    def residual(self) -> np.ndarray:
        """Calibration residual ``element_gain / element_gain_est`` (``[..., R, C]``)."""
        return self.element_gain / self.element_gain_est


def calibration_capture(
    element_gains: np.ndarray, snr_db: float, rng: np.random.Generator, *, num_bins: int
) -> CalibrationCapture:
    """Measure a known unit-amplitude boresight source through the chain.

    The source is a flat unit far-field plane wave (steering vector 1 on every
    element and bin), so the recorded capture is ``G * (1 + w)`` through the
    same input-referred chain as :func:`apply_hardware_impairments`. The gain
    estimate is the mean of the capture over the frequency bins. The noise is
    always drawn, even at ``snr_db = +inf`` (where it is zero), so the RNG
    stream layout does not depend on the SNR.
    """
    gains = np.asarray(element_gains, dtype=np.complex128)
    if gains.ndim < 2 or not np.all(np.isfinite(gains)):
        raise ValueError("element_gains must have ndim >= 2 and be finite")
    snr = float(snr_db)
    if math.isnan(snr) or snr == -math.inf:
        raise ValueError("snr_db must not be NaN or -inf")
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be an np.random.Generator")
    if isinstance(num_bins, bool) or not isinstance(num_bins, (int, np.integer)):
        raise ValueError("num_bins must be an int >= 1")
    num_bins = int(num_bins)
    if num_bins < 1:
        raise ValueError("num_bins must be an int >= 1")

    noise_var = 0.0 if math.isinf(snr) else 10.0 ** (-snr / 10.0)
    z = rng.standard_normal((2, *gains.shape, num_bins))
    scale = math.sqrt(noise_var / 2.0) if noise_var > 0.0 else 0.0
    w = scale * (z[0] + 1j * z[1])
    capture = gains[..., None] * (1.0 + w)
    element_gain_est = capture.mean(axis=-1)
    return CalibrationCapture(
        element_gain=gains.copy(),
        capture=capture,
        element_gain_est=element_gain_est,
        snr_db=snr,
        noise_var=noise_var,
    )


def rotation_from_vector(rotation_vector: np.ndarray) -> np.ndarray:
    """Rodrigues: rotation matrices ``[..., 3, 3]`` for vectors ``[..., 3]`` (radians)."""
    omega = np.asarray(rotation_vector, dtype=np.float64)
    if omega.ndim < 1 or omega.shape[-1] != 3:
        raise ValueError("rotation_vector must have shape [..., 3]")

    theta = np.linalg.norm(omega, axis=-1, keepdims=True)
    safe = np.where(theta > 0.0, theta, 1.0)
    k = omega / safe
    kx, ky, kz = k[..., 0], k[..., 1], k[..., 2]
    skew = np.zeros((*k.shape[:-1], 3, 3), dtype=np.float64)
    skew[..., 0, 1] = -kz
    skew[..., 0, 2] = ky
    skew[..., 1, 0] = kz
    skew[..., 1, 2] = -kx
    skew[..., 2, 0] = -ky
    skew[..., 2, 1] = kx
    skew2 = skew @ skew

    scaled = theta[..., None]
    rotation = np.eye(3) + np.sin(scaled) * skew + (1.0 - np.cos(scaled)) * skew2
    identity = theta == 0.0
    return np.where(identity[..., None], np.eye(3), rotation)


@dataclass(frozen=True)
class PosePerturbation:
    """Reported (perturbed) UE poses and the applied errors (ground truth)."""

    positions: np.ndarray
    """``[V, 3]`` perturbed positions (m)."""

    rotations: np.ndarray
    """``[V, 3, 3]`` perturbed world-from-local rotations."""

    delta_position: np.ndarray
    """``[V, 3]`` ``positions - true positions`` (m)."""

    rotation_vector: np.ndarray
    """``[V, 3]`` world-frame rotation vector (rad)."""


def perturb_poses(
    positions: np.ndarray,
    rotations: np.ndarray,
    rng: np.random.Generator,
    *,
    position_std_m: float = 0.0,
    rotation_std_deg: float = 0.0,
) -> PosePerturbation:
    """Perturb true UE poses and return the reported poses plus the errors.

    The data are traced at the true poses; the perturbed poses are what a
    solver is told (the pose-jitter sweep, design section 4.3 / 5.3). Both
    Gaussian draws are always consumed, so the RNG stream layout does not
    depend on the standard deviations. The rotation is a world-frame (left)
    perturbation: ``R_new = R(omega) @ R_true``.
    """
    pos = np.asarray(positions, dtype=np.float64)
    rot = np.asarray(rotations, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3:
        raise ValueError("positions must have shape [V, 3]")
    if rot.ndim != 3 or rot.shape != (pos.shape[0], 3, 3):
        raise ValueError("rotations must have shape [V, 3, 3]")
    if not np.all(np.isfinite(pos)) or not np.all(np.isfinite(rot)):
        raise ValueError("positions and rotations must be finite")
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be an np.random.Generator")
    for name, value in (
        ("position_std_m", position_std_m),
        ("rotation_std_deg", rotation_std_deg),
    ):
        spread = float(value)
        if not math.isfinite(spread) or spread < 0.0:
            raise ValueError(f"{name} must be finite and >= 0, got {value!r}")

    num_views = pos.shape[0]
    delta = rng.normal(0.0, float(position_std_m), size=(num_views, 3))
    omega = np.deg2rad(rng.normal(0.0, float(rotation_std_deg), size=(num_views, 3)))
    new_positions = pos + delta
    new_rotations = rotation_from_vector(omega) @ rot
    return PosePerturbation(
        positions=new_positions,
        rotations=new_rotations,
        delta_position=delta,
        rotation_vector=omega,
    )


def _json_value(value: Any) -> Any:
    """Return ``value`` as JSON-safe nested lists, mapping non-finite to ``None``."""
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        scalar = float(array)
        return scalar if math.isfinite(scalar) else None
    return [_json_value(item) for item in array]


def capture_gt_record(
    hardware_gt: Mapping[str, Any],
    view: int,
    bs: int,
    *,
    gauge: tuple[np.ndarray, np.ndarray] | None = None,
    pose: PosePerturbation | None = None,
    calibration: CalibrationCapture | None = None,
) -> dict[str, Any]:
    """Flatten the per-``(v, b)`` hardware ground truth into a JSON-safe record.

    ``hardware_gt`` is the ``gt`` dict from :func:`apply_hardware_impairments`;
    ``gauge`` is ``(phi[V, B], tau[V, B])``; ``calibration`` holds ``[V, R, C]``
    gains (indexed by ``view``) or ``[R, C]`` gains (used as is). Every float is
    a Python float, every array a nested list, and every non-finite float is
    ``None`` so ``json.dumps(record, allow_nan=False)`` succeeds.
    """
    view = int(view)
    bs = int(bs)
    signal_power = np.asarray(hardware_gt["signal_power"], dtype=np.float64)
    num_views, num_bs = signal_power.shape
    if not 0 <= view < num_views or not 0 <= bs < num_bs:
        raise ValueError("view/bs lie outside the capture grid")

    element_gain = np.asarray(hardware_gt["element_gain"], dtype=np.complex128)[view]
    with np.errstate(divide="ignore", invalid="ignore"):
        element_gain_db = 20.0 * np.log10(np.abs(element_gain))
    element_phase_rad = np.angle(element_gain)

    front_to_back_db = hardware_gt["front_to_back_db"]
    if front_to_back_db is not None:
        front_to_back_db = _json_value(front_to_back_db)

    if gauge is None:
        phase_rad: Any = None
        delay_s: Any = None
    else:
        phase_rad = _json_value(np.asarray(gauge[0], dtype=np.float64)[view, bs])
        delay_s = _json_value(np.asarray(gauge[1], dtype=np.float64)[view, bs])

    if pose is None:
        pose_delta_position_m: Any = None
        pose_rotation_vector_rad: Any = None
    else:
        pose_delta_position_m = _json_value(pose.delta_position[view])
        pose_rotation_vector_rad = _json_value(pose.rotation_vector[view])

    if calibration is None:
        calibration_snr_db: Any = None
        element_gain_est_db: Any = None
        element_phase_est_rad: Any = None
    else:
        calibration_snr_db = _json_value(calibration.snr_db)
        gain_est = np.asarray(calibration.element_gain_est, dtype=np.complex128)
        if gain_est.ndim == 3:
            gain_est = gain_est[view]
        with np.errstate(divide="ignore", invalid="ignore"):
            element_gain_est_db = _json_value(20.0 * np.log10(np.abs(gain_est)))
        element_phase_est_rad = _json_value(np.angle(gain_est))

    return {
        "view": view,
        "bs": bs,
        "front_to_back_db": front_to_back_db,
        "front_to_back_gain": _json_value(hardware_gt["front_to_back_gain"]),
        "element_gain_db": _json_value(element_gain_db),
        "element_phase_rad": _json_value(element_phase_rad),
        "noise_variance": _json_value(hardware_gt["noise_var"]),
        "signal_power": _json_value(signal_power[view, bs]),
        "noise_power": _json_value(np.asarray(hardware_gt["noise_power"])[view, bs]),
        "expected_snr_db": _json_value(np.asarray(hardware_gt["expected_snr_db"])[view, bs]),
        "achieved_snr_db": _json_value(np.asarray(hardware_gt["achieved_snr_db"])[view, bs]),
        "phase_rad": phase_rad,
        "delay_s": delay_s,
        "pose_delta_position_m": pose_delta_position_m,
        "pose_rotation_vector_rad": pose_rotation_vector_rad,
        "calibration_snr_db": calibration_snr_db,
        "element_gain_est_db": element_gain_est_db,
        "element_phase_est_rad": element_phase_est_rad,
    }
