"""Synchronisation gauges and paired data tracks (tomography Phase 0, NumPy only).

Gauge convention
----------------
A capture recorded without a shared clock is::

    Y_obs[v, b, h, r, col, n] = exp(+1j * phi[v, b])
        * exp(-1j * 2 * pi * freq_offsets[n] * tau[v, b])
        * Y[v, b, h, r, col, n],

i.e. ``gauge[v, b, n] = exp(1j * phi) * exp(-2j * pi * f * tau)`` is unit
modulus and shared by all hemispheres and elements of the capture. A positive
``tau`` is a later arrival, the same convention as Sionna ``Paths.cfr`` and as
``gauge.align_common_phase_and_delay`` on the gauge-alignment branch.

Reference capture
-----------------
The gauge of the reference capture ``ref = (v0, b0)`` (default ``(0, 0)``) is
exactly ``(0.0, 0.0)`` in every mode.

Seeding
-------
Noise and hardware draws use per-capture streams
``SeedSequence([dataset_seed, v, b, realization])``; gauge draws use separate
streams ``SeedSequence([dataset_seed, realization, GAUGE_STREAM_TAG, k])`` for
``k = 0, 1, 2`` (N, N_sep, S_tau), which can never collide with a capture seed
because ``b < 2**32 - 1``.

Tracks
------
From one clean array six paired tracks share the same noise realisation:
``ideal-S``, ``ideal-N``, ``observed-S``, ``observed-N``, ``N-sep`` and
``S_tau``. N-type tracks are the S track times the unit-modulus gauge. The
observed track is built by :func:`impairments.apply_hardware_impairments` with
the calibration residual of :func:`impairments.calibration_capture`, and the
gauge helpers ``apply_gauge``/``gauge_factor`` live in
:mod:`plateau_rt.domain.rf_camera.impairments` (re-exported here).
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from plateau_rt.domain.rf_camera import impairments
from plateau_rt.domain.rf_camera.imaging import uniform_frequency_spacing
from plateau_rt.domain.rf_camera.impairments import apply_gauge as apply_gauge
from plateau_rt.domain.rf_camera.impairments import gauge_factor as gauge_factor

GAUGE_MODES: tuple[str, ...] = ("S", "N", "S_tau", "N_sep")
MODE_ALIASES: dict[str, str] = {"S_τ": "S_tau", "N-sep": "N_sep"}
TRACK_NAMES: tuple[str, ...] = ("ideal-S", "ideal-N", "observed-S", "observed-N", "N-sep", "S_tau")
DEFAULT_SIGMA_T: float = 10e-9
DEFAULT_SNR_DB: float = 30.0
GAUGE_STREAM_TAG: int = 2**32 - 1


@dataclass(frozen=True)
class HardwareConfig:
    """Observed-track hardware effects shared by S and N (design §2.5 step 2)."""

    element_gain_std_db: float = 0.5
    element_phase_std_deg: float = 5.0
    front_to_back_db: float = 20.0
    calibration_snr_db: float = 30.0

    def __post_init__(self) -> None:
        """Validate the hardware tolerances."""
        gain = float(self.element_gain_std_db)
        if not math.isfinite(gain) or gain < 0.0:
            raise ValueError("element_gain_std_db must be finite and >= 0")
        phase = float(self.element_phase_std_deg)
        if not math.isfinite(phase) or phase < 0.0:
            raise ValueError("element_phase_std_deg must be finite and >= 0")
        fb = float(self.front_to_back_db)
        if not math.isfinite(fb):
            raise ValueError("front_to_back_db must be finite")
        cal = float(self.calibration_snr_db)
        if math.isnan(cal) or cal == -math.inf:
            raise ValueError("calibration_snr_db must not be nan or -inf")


@dataclass(frozen=True)
class TrackSeeds:
    """Seeds of one paired realisation."""

    dataset_seed: int
    realization: int = 0

    def __post_init__(self) -> None:
        """Validate the seed pair."""
        for label, value in (
            ("dataset_seed", self.dataset_seed),
            ("realization", self.realization),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{label} must be an int >= 0")
            if int(value) < 0:
                raise ValueError(f"{label} must be an int >= 0")


def draw_gauges(
    num_views: int,
    num_bs: int,
    mode: str,
    sigma_t: float | str | None,
    rng: np.random.Generator,
    *,
    ref: tuple[int, int] = (0, 0),
    period: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw per-capture ``(phi, tau)`` gauges with the reference fixed to zero."""
    num_views = int(num_views)
    num_bs = int(num_bs)
    if num_views < 1 or num_bs < 1:
        raise ValueError("num_views and num_bs must be >= 1")
    canonical = MODE_ALIASES.get(mode, mode)
    if canonical not in GAUGE_MODES:
        raise ValueError(f"unknown gauge mode {mode!r}")
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be an np.random.Generator")
    try:
        v0, b0 = (int(value) for value in ref)
    except (TypeError, ValueError) as error:
        raise ValueError("ref must be a pair of integers") from error
    if not 0 <= v0 < num_views or not 0 <= b0 < num_bs:
        raise ValueError("ref lies outside the capture grid")

    tau_specs: tuple[str, float | None] | None = None
    if canonical in ("N", "N_sep"):
        tau_specs = _validate_sigma_t(sigma_t, period)
    if canonical == "S":
        phi = np.zeros((num_views, num_bs), dtype=np.float64)
        tau = np.zeros((num_views, num_bs), dtype=np.float64)
    elif canonical == "N":
        assert tau_specs is not None
        phi = np.asarray(rng.uniform(0.0, 2.0 * np.pi, size=(num_views, num_bs)), dtype=np.float64)
        tau = _draw_tau(rng, (num_views, num_bs), tau_specs)
        phi[v0, b0] = 0.0
        tau[v0, b0] = 0.0
    elif canonical == "S_tau":
        phi = np.asarray(rng.uniform(0.0, 2.0 * np.pi, size=(num_views, num_bs)), dtype=np.float64)
        tau = np.zeros((num_views, num_bs), dtype=np.float64)
        phi[v0, b0] = 0.0
    else:  # N_sep: each clock has spread sigma_t, so a capture tau has std sqrt(2)*sigma_t.
        assert tau_specs is not None
        phi_v = np.asarray(rng.uniform(0.0, 2.0 * np.pi, size=num_views), dtype=np.float64)
        phi_b = np.asarray(rng.uniform(0.0, 2.0 * np.pi, size=num_bs), dtype=np.float64)
        tau_v = _draw_tau(rng, (num_views,), tau_specs)
        tau_b = _draw_tau(rng, (num_bs,), tau_specs)
        phi_v[v0] = 0.0
        phi_b[b0] = 0.0
        tau_v[v0] = 0.0
        tau_b[b0] = 0.0
        phi = np.mod(phi_v[:, None] + phi_b[None, :], 2.0 * np.pi)
        tau = tau_v[:, None] + tau_b[None, :]

    phi = np.mod(np.asarray(phi, dtype=np.float64), 2.0 * np.pi)
    phi[phi == 2.0 * np.pi] = 0.0
    phi[v0, b0] = 0.0
    tau = np.asarray(tau, dtype=np.float64)
    tau[v0, b0] = 0.0
    return phi, tau


def capture_power(Y: np.ndarray) -> np.ndarray:
    """Return the per-capture power with the hemisphere powers summed."""
    arr = np.asarray(Y, dtype=np.complex128)
    if arr.ndim != 6:
        raise ValueError("Y must have shape [V, B, H, R, C, N]")
    return np.sum(np.abs(arr) ** 2, axis=2).mean(axis=(2, 3, 4)).astype(np.float64)


def reference_power(Y_clean: np.ndarray, los_visible: np.ndarray) -> tuple[float, tuple[int, int]]:
    """Return the power and index of the median-power LoS-visible capture."""
    arr = np.asarray(Y_clean, dtype=np.complex128)
    if arr.ndim != 6:
        raise ValueError("Y_clean must have shape [V, B, H, R, C, N]")
    mask = np.asarray(los_visible, dtype=bool)
    if mask.shape != arr.shape[:2]:
        raise ValueError("los_visible must have shape [V, B]")
    num_bs = arr.shape[1]
    powers = capture_power(arr)
    flat_visible = np.flatnonzero(mask.ravel(order="C"))
    if flat_visible.size == 0:
        raise ValueError("no LoS-visible capture")
    order = np.argsort(powers.ravel(order="C")[flat_visible], kind="stable")
    chosen = flat_visible[order[(flat_visible.size - 1) // 2]]
    v_ref, b_ref = int(chosen // num_bs), int(chosen % num_bs)
    return float(powers[v_ref, b_ref]), (v_ref, b_ref)


def make_tracks(
    Y_clean: np.ndarray,
    snr_db: float,
    p_ref: float,
    seeds: TrackSeeds,
    *,
    freq_offsets: np.ndarray,
    sigma_t: float | str = DEFAULT_SIGMA_T,
    ref: tuple[int, int] = (0, 0),
    hardware: HardwareConfig = HardwareConfig(),
    Y_los: np.ndarray | None = None,
    c_ref: tuple[int, int] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build the six paired tracks and their impairment ground truth."""
    clean = np.asarray(Y_clean, dtype=np.complex128)
    if clean.ndim != 6 or clean.shape[2] != 2:
        raise ValueError("Y_clean must have shape [V, B, 2, R, C, N]")
    if not np.all(np.isfinite(clean)):
        raise ValueError("Y_clean must be finite")
    num_views, num_bs, _, num_rows, num_cols, num_bins = clean.shape
    freq = np.asarray(freq_offsets, dtype=np.float64)
    if freq.ndim != 1 or freq.shape != (num_bins,):
        raise ValueError("len(freq_offsets) must equal Y_clean.shape[-1]")
    snr = float(snr_db)
    if math.isnan(snr) or snr == -math.inf:
        raise ValueError("snr_db must be finite or +inf")
    p_ref_f = float(p_ref)
    if not math.isfinite(p_ref_f) or p_ref_f <= 0.0:
        raise ValueError("p_ref must be finite and > 0")
    if not isinstance(seeds, TrackSeeds):
        raise ValueError("seeds must be a TrackSeeds")
    if not isinstance(hardware, HardwareConfig):
        raise ValueError("hardware must be a HardwareConfig")
    los_arr = np.asarray(Y_los, dtype=np.complex128) if Y_los is not None else None
    if los_arr is not None and los_arr.shape != clean.shape:
        raise ValueError("Y_los must have the same shape as Y_clean")
    if los_arr is not None and not np.all(np.isfinite(los_arr)):
        raise ValueError("Y_los must be finite")
    los: np.ndarray | None = los_arr
    try:
        v0, b0 = (int(value) for value in ref)
    except (TypeError, ValueError) as error:
        raise ValueError("ref must be a pair of integers") from error
    if not 0 <= v0 < num_views or not 0 <= b0 < num_bs:
        raise ValueError("ref lies outside the capture grid")

    if math.isinf(snr):
        sigma2 = 0.0
    else:
        sigma2 = p_ref_f / 10.0 ** (snr / 10.0)
    period: float | None = None
    if isinstance(sigma_t, str) and sigma_t == "uniform":
        period = 1.0 / uniform_frequency_spacing(np.sort(freq))

    noise = np.zeros((num_views, num_bs, 2, num_rows, num_cols, num_bins), dtype=np.complex128)
    gain = np.zeros((num_views, num_rows, num_cols), dtype=np.complex128)
    gain_est = np.zeros((num_views, num_rows, num_cols), dtype=np.complex128)
    for v in range(num_views):
        for b in range(num_bs):
            rng = np.random.default_rng(
                np.random.SeedSequence([int(seeds.dataset_seed), v, b, int(seeds.realization)])
            )
            z = rng.standard_normal((2, 2, num_rows, num_cols, num_bins))
            scale = math.sqrt(sigma2 / 2.0) if sigma2 > 0.0 else 0.0
            noise[v, b] = scale * (z[0] + 1j * z[1])
            if b == 0:
                errors = impairments.draw_element_errors(
                    impairments.ImpairmentConfig(
                        element_gain_std_db=hardware.element_gain_std_db,
                        element_phase_std_deg=hardware.element_phase_std_deg,
                    ),
                    num_rows,
                    num_cols,
                    rng,
                )
                gain[v] = errors.complex_gain
                cal = impairments.calibration_capture(
                    gain[v], hardware.calibration_snr_db, rng, num_bins=num_bins
                )
                gain_est[v] = cal.element_gain_est

    ideal_s = (clean + noise).astype(np.complex128, copy=False)

    eps = (gain / gain_est).astype(np.complex128)
    observed_s, hw_gt = impairments.apply_hardware_impairments(
        clean,
        element_gain=eps,
        front_to_back_db=float(hardware.front_to_back_db),
        noise_var_abs=sigma2,
        noise=noise[:, :, 0],
    )

    ds, realization = int(seeds.dataset_seed), int(seeds.realization)
    rng_0 = np.random.default_rng(np.random.SeedSequence([ds, realization, GAUGE_STREAM_TAG, 0]))
    rng_1 = np.random.default_rng(np.random.SeedSequence([ds, realization, GAUGE_STREAM_TAG, 1]))
    rng_2 = np.random.default_rng(np.random.SeedSequence([ds, realization, GAUGE_STREAM_TAG, 2]))
    phi_n, tau_n = draw_gauges(num_views, num_bs, "N", sigma_t, rng_0, ref=ref, period=period)
    phi_sep, tau_sep = draw_gauges(
        num_views, num_bs, "N_sep", sigma_t, rng_1, ref=ref, period=period
    )
    phi_st, tau_st = draw_gauges(num_views, num_bs, "S_tau", None, rng_2, ref=ref)

    zeros_phi = np.zeros((num_views, num_bs), dtype=np.float64)
    zeros_tau = np.zeros((num_views, num_bs), dtype=np.float64)
    tracks = {
        "ideal-S": ideal_s,
        "ideal-N": apply_gauge(ideal_s, phi_n, tau_n, freq),
        "observed-S": observed_s,
        "observed-N": apply_gauge(observed_s, phi_n, tau_n, freq),
        "N-sep": apply_gauge(ideal_s, phi_sep, tau_sep, freq),
        "S_tau": apply_gauge(ideal_s, phi_st, tau_st, freq),
    }

    noise_power = np.mean(np.abs(noise) ** 2, axis=(2, 3, 4, 5)).astype(np.float64)
    clean_power = capture_power(clean)
    with np.errstate(divide="ignore"):
        expected = 10.0 * np.log10(clean_power / sigma2)
        achieved = 10.0 * np.log10(clean_power / noise_power)
        if los is None:
            expected_scat = np.full((num_views, num_bs), np.nan, dtype=np.float64)
            achieved_scat = np.full((num_views, num_bs), np.nan, dtype=np.float64)
        else:
            scat_power = capture_power(clean - los)
            expected_scat = 10.0 * np.log10(scat_power / sigma2)
            achieved_scat = 10.0 * np.log10(scat_power / noise_power)

    gt: dict[str, Any] = {
        "sigma2": float(sigma2),
        "snr_db": float(snr),
        "p_ref": float(p_ref_f),
        "c_ref": None if c_ref is None else (int(c_ref[0]), int(c_ref[1])),
        "ref": (v0, b0),
        "sigma_t": sigma_t,
        "seeds": (ds, realization),
        "hardware": dataclasses.asdict(hardware),
        "gauges": {
            "ideal-S": (zeros_phi.copy(), zeros_tau.copy()),
            "ideal-N": (phi_n.copy(), tau_n.copy()),
            "observed-S": (zeros_phi.copy(), zeros_tau.copy()),
            "observed-N": (phi_n.copy(), tau_n.copy()),
            "N-sep": (phi_sep.copy(), tau_sep.copy()),
            "S_tau": (phi_st.copy(), tau_st.copy()),
        },
        "noise_power": noise_power,
        "expected_snr_db": expected.astype(np.float64),
        "achieved_snr_db": achieved.astype(np.float64),
        "expected_scatter_snr_db": expected_scat.astype(np.float64),
        "achieved_scatter_snr_db": achieved_scat.astype(np.float64),
        "observed_achieved_snr_db": hw_gt["achieved_snr_db"],
        "element_gain": gain,
        "element_gain_est": gain_est,
        "calibration_residual": eps,
        "front_to_back_gain": hw_gt["front_to_back_gain"],
    }
    return tracks, gt


def _validate_sigma_t(
    sigma_t: float | str | None, period: float | None
) -> tuple[str, float | None]:
    """Return the validated ``("gaussian", value)`` or ``("uniform", period)`` spec."""
    if isinstance(sigma_t, str):
        if sigma_t != "uniform":
            raise ValueError("sigma_t string must be 'uniform'")
        if period is None or not math.isfinite(float(period)) or float(period) <= 0.0:
            raise ValueError("'uniform' sigma_t requires a positive finite period")
        return ("uniform", float(period))
    if sigma_t is None or isinstance(sigma_t, bool):
        raise ValueError("sigma_t must be a float >= 0 or 'uniform'")
    try:
        value = float(sigma_t)
    except (TypeError, ValueError) as error:
        raise ValueError("sigma_t must be a float >= 0 or 'uniform'") from error
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("sigma_t must be a float >= 0 or 'uniform'")
    return ("gaussian", value)


def _draw_tau(
    rng: np.random.Generator, shape: tuple[int, ...], spec: tuple[str, float | None]
) -> np.ndarray:
    """Return one tau draw of ``shape`` from the validated ``spec``."""
    kind, value = spec
    assert value is not None
    if kind == "uniform":
        half = value / 2.0
        return np.asarray(rng.uniform(-half, half, size=shape), dtype=np.float64)
    return np.asarray(rng.normal(0.0, value, size=shape), dtype=np.float64)
