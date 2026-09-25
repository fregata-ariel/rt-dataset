"""Nuisance atoms and gauge solvers (design §2.4/§3.3/§4.2 rows 8, 10, 12, 14).

Implements the C13 nuisance atoms of §3.3 (the LoS atom with its phase fixed by
the model and a free complex ground-image atom), the LoS-anchored and
power-cross-correlation delay solvers of §4.2 rows 8/12, the alternating
self-calibration of rows 10/12/14 and the profiled VarPro cost/gradient of row 14.

Gauge convention
----------------
A capture ``c = (v, b)`` recorded without a shared clock is::

    Y_obs[v, b, h, r, col, n] = exp(+1j * phi[v, b])
        * exp(-2j * pi * freq_offsets[n] * tau[v, b])
        * Y[v, b, h, r, col, n],

exactly :func:`sync.gauge_factor`.  This is the same sign convention as
:func:`rf_camera.gauge.align_common_phase_and_delay`, whose least-squares output
``(phase_rad, delay_s)`` therefore enters with no sign conversion.  Phases are
wrapped to ``(-pi, pi]`` and delays to ``[-T/2, T/2)`` with ``T`` the delay
period.  Every estimator fixes the reference capture ``ref = (v0, b0)`` to
``phi = 0`` and estimates all delays.

NumPy/SciPy only: nothing here may import Sionna, Mitsuba or Dr.Jit.  All
lengths are metres, angles radians, frequencies hertz.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar, nnls

from plateau_rt.domain.rf_camera.gauge import align_common_phase_and_delay
from plateau_rt.domain.rf_tomography.forward_exact import _capture_index, atom_cfr
from plateau_rt.domain.rf_tomography.forward_sep import SeparableOperator
from plateau_rt.domain.rf_tomography.geometry import CaptureGeometry, mirror_point
from plateau_rt.domain.rf_tomography.observables import angle_delay_volume
from plateau_rt.domain.rf_tomography.sync import apply_gauge, gauge_factor

FIT_MODES: tuple[str, ...] = ("complex", "power")
DEFAULT_OVERSAMPLE: int = 16
POWER_OVERSAMPLE: int = 4
POWER_DELAY_OVERSAMPLE: int = 4
SELF_CAL_ITERATIONS: int = 10
SELF_CAL_RTOL: float = 1e-4


def _wrap_phases(phi: np.ndarray) -> np.ndarray:
    """Wrap an array of phases into ``(-pi, pi]`` elementwise."""
    wrapped = np.angle(np.exp(1j * np.asarray(phi, dtype=np.float64)))
    return np.where(wrapped <= -np.pi, np.pi, wrapped).astype(np.float64, copy=False)


def _wrap_delay(tau: float, period: float) -> float:
    """Wrap a delay into ``[-T/2, T/2)``."""
    return float((float(tau) + 0.5 * period) % period - 0.5 * period)


def nuisance_points(geom: CaptureGeometry, b: int, ground_height: float | None = 0.0) -> np.ndarray:
    """Return the ``[K, 3]`` virtual sources of BS ``b``.

    Row 0 is the LoS source ``geom.bs_pos[b]``.  When ``ground_height`` is not
    ``None``, row 1 is the BS mirrored in the horizontal plane ``z =
    ground_height``, so ``K = 2``; otherwise ``K = 1``.
    """
    b = _capture_index(b, geom.num_bs, "b")
    if ground_height is None:
        return geom.bs_pos[b][None, :].astype(np.float64, copy=True)
    height = float(ground_height)
    if not np.isfinite(height):
        raise ValueError("ground_height must be finite or None")
    image = mirror_point(geom.bs_pos[b], (0.0, 0.0, height), (0.0, 0.0, 1.0))
    return np.stack([geom.bs_pos[b], image], axis=0).astype(np.float64, copy=False)


def _residual_stats(Y_c: np.ndarray, model: np.ndarray) -> float:
    """Return the relative residual ``||Y_c - model||^2 / ||Y_c||^2``."""
    residual = float(np.sum(np.abs(Y_c - model) ** 2))
    reference = float(np.sum(np.abs(Y_c) ** 2))
    return float("inf") if reference == 0.0 else residual / reference


def _median_noise(residual_power: np.ndarray) -> float:
    """Robust per-sample noise power: ``max(0, median(residual_power) / ln 2)``.

    For circular complex Gaussian noise of variance ``sigma^2`` each angle-delay
    cell power is exponential with median ``sigma^2 ln 2``.
    """
    return max(0.0, float(np.median(residual_power) / np.log(2.0)))


def _fit_complex(
    Y_c: np.ndarray,
    cols: list[np.ndarray],
    f: np.ndarray,
    period: float,
    *,
    with_gauge: bool,
    free_los_phase: bool,
    oversample: int,
) -> dict[str, Any]:
    """Profiled complex LS fit of the LoS/ground nuisance atoms of one capture."""
    num_atoms = len(cols)
    num_bins = f.size
    A = np.stack([col.reshape(-1, num_bins) for col in cols], axis=0)
    Yf = Y_c.reshape(-1, num_bins)
    gram = np.einsum("ken,len->kl", np.conj(A), A)
    corr = np.einsum("ken,en->kn", np.conj(A), Yf)

    if with_gauge:
        num_grid = oversample * num_bins
        step = period / num_grid
        tau_grid = (np.arange(num_grid) - num_grid // 2) * step
        phases = np.exp(2j * np.pi * np.outer(tau_grid, f))
        bvec_grid = phases @ corr.T
        solved = np.linalg.solve(gram, bvec_grid.T)
        energy = np.real(np.einsum("mk,km->m", np.conj(bvec_grid), solved))
        peak = int(np.argmax(energy))

        def profile(t: float) -> float:
            bvec = np.exp(2j * np.pi * f * t) @ corr.T
            return -float(np.real(np.vdot(bvec, np.linalg.solve(gram, bvec))))

        refined = minimize_scalar(
            profile,
            bounds=(tau_grid[peak] - step, tau_grid[peak] + step),
            method="bounded",
            options={"xatol": 1e-6 * step},
        )
        tau = _wrap_delay(float(refined.x), period)
        bvec = np.exp(2j * np.pi * f * tau) @ corr.T
        coef = np.linalg.solve(gram, bvec)
        if free_los_phase:
            phi = 0.0
            a_los = coef[0]
            g_los = float(abs(coef[0]))
            a_ground = coef[1] if num_atoms > 1 else 0j
        else:
            phi = float(_wrap_phases(np.angle(coef[0]))) if coef[0] != 0 else 0.0
            g_los = float(abs(coef[0]))
            a_los = complex(g_los)
            a_ground = coef[1] * np.exp(-1j * phi) if num_atoms > 1 else 0j
    else:
        phi = 0.0
        tau = 0.0
        if free_los_phase:
            coef = np.linalg.solve(gram, corr.sum(axis=1))
            a_los = coef[0]
            g_los = float(abs(coef[0]))
            a_ground = coef[1] if num_atoms > 1 else 0j
        else:
            los = A[0].ravel()
            y = Yf.ravel()
            if num_atoms > 1:
                ground = A[1].ravel()
                ground_norm = float(np.real(np.vdot(ground, ground)))
                los_proj = los - ground * np.vdot(ground, los) / ground_norm
                g_los = max(
                    0.0,
                    float(np.real(np.vdot(los_proj, y)))
                    / float(np.real(np.vdot(los_proj, los_proj))),
                )
                a_ground = np.vdot(ground, y - g_los * los) / ground_norm
            else:
                g_los = max(
                    0.0,
                    float(np.real(np.vdot(los, y))) / float(np.real(np.vdot(los, los))),
                )
                a_ground = 0j
            a_los = complex(g_los)

    factor = gauge_factor(np.float64(phi), np.float64(tau), f)
    summed = a_los * cols[0]
    if num_atoms > 1:
        summed = summed + a_ground * cols[1]
    model = factor * summed
    resid = _residual_stats(Y_c, model)
    residual_cfr = (Y_c - model)[None, None]
    volume = angle_delay_volume(residual_cfr, oversample=(1, POWER_DELAY_OVERSAMPLE))[0, 0]
    noise = _median_noise(np.abs(volume) ** 2)
    return {
        "g_los": g_los,
        "a_los": complex(a_los),
        "a_ground": complex(a_ground),
        "phi": phi,
        "tau": tau,
        "resid": resid,
        "noise": noise,
    }


def _fit_power(
    Y_c: np.ndarray,
    cols: list[np.ndarray],
    f: np.ndarray,
    period: float,
    *,
    with_gauge: bool,
    oversample: int,
) -> dict[str, Any]:
    """Nonnegative power fit of the LoS/ground nuisance atoms of one capture."""
    num_atoms = len(cols)
    num_bins = f.size
    obs = (
        np.abs(angle_delay_volume(Y_c[None, None], oversample=(1, POWER_DELAY_OVERSAMPLE))[0, 0])
        ** 2
    ).astype(np.float64)
    flat = obs.ravel()

    def design(t: float) -> np.ndarray:
        factor = gauge_factor(np.float64(0.0), np.float64(t), f)
        stacked = (np.stack(cols, axis=0) * factor)[:, None]
        volume = angle_delay_volume(stacked, oversample=(1, POWER_DELAY_OVERSAMPLE))
        profiles = np.abs(volume[:, 0]) ** 2
        return np.column_stack([profiles.reshape(num_atoms, -1).T, np.ones(flat.size)])

    def fit_at(t: float) -> tuple[np.ndarray, float]:
        weights, rnorm = nnls(design(t), flat)
        return weights, float(rnorm) ** 2

    if with_gauge:
        num_grid = oversample * num_bins
        step = period / num_grid
        tau_grid = (np.arange(num_grid) - num_grid // 2) * step
        costs = np.array([fit_at(t)[1] for t in tau_grid], dtype=np.float64)
        peak = int(np.argmin(costs))
        refined = minimize_scalar(
            lambda t: fit_at(t)[1],
            bounds=(tau_grid[peak] - step, tau_grid[peak] + step),
            method="bounded",
            options={"xatol": 1e-6 * step},
        )
        tau = _wrap_delay(float(refined.x), period)
        phi = float("nan")
    else:
        tau = 0.0
        phi = 0.0

    weights, cost = fit_at(tau)
    g_los = float(np.sqrt(max(weights[0], 0.0)))
    a_los = complex(g_los)
    if num_atoms > 1:
        a_ground = complex(np.sqrt(max(weights[1], 0.0)))
    else:
        a_ground = 0j
    reference = float(np.sum(obs**2))
    resid = float("inf") if reference == 0.0 else float(cost) / reference
    profiles = design(tau)[:, :num_atoms]
    noise = _median_noise(flat - profiles @ weights[:num_atoms])
    return {
        "g_los": g_los,
        "a_los": a_los,
        "a_ground": a_ground,
        "phi": phi,
        "tau": tau,
        "resid": resid,
        "noise": noise,
    }


def fit_los_ground(
    Y_c: np.ndarray,
    geom: CaptureGeometry,
    v: int,
    b: int,
    mode: str = "complex",
    with_gauge: bool = True,
    *,
    ground_height: float | None = 0.0,
    free_los_phase: bool = False,
    pattern: str = "tr38901",
    polarization: str = "none",
    oversample: int | None = None,
) -> dict[str, Any]:
    """Fit the known-geometry LoS/ground nuisance atoms to one capture.

    The model is ``exp(1j phi) exp(-2j pi f tau) (a_los col_0 + a_ground col_1)``
    in VS space.  In ``"complex"`` mode the LoS amplitude is real and nonnegative
    unless ``free_los_phase`` (the design C13 negative control, where the phase
    is absorbed into the amplitude); in ``"power"`` mode both amplitudes are
    nonnegative and the phase is not observable (``phi`` is NaN when
    ``with_gauge``).

    Returned keys:

    * ``g_los``: the LoS amplitude ``|a_los|`` (float ``>= 0``);
    * ``a_los`` / ``a_ground``: the LoS / ground coefficients (complex; real and
      nonnegative in power mode, ``a_ground = 0`` without a ground atom);
    * ``phi``: the capture phase in rad, wrapped to ``(-pi, pi]`` (complex mode
      with a gauge), ``NaN`` in power mode with a gauge, ``0.0`` without one;
    * ``tau``: the capture delay in s, wrapped to ``[-T/2, T/2)`` (``0.0``
      without a gauge);
    * ``resid``: the relative residual of the fit, ``||Y - model||^2 / ||Y||^2``
      in complex mode and the relative NNLS cost on the ``|c(u, t)|^2`` volume
      in power mode;
    * ``noise``: the same quantity in both modes, a robust estimate of the raw
      per-sample noise variance ``sigma^2`` (units of ``noise_var``): the median
      of the per-cell residual power of the capture's delay-oversampled
      angle-delay volume divided by ``ln 2`` (the median of an exponential
      cell power), clipped at 0. Complex mode uses ``|vol(Y - model)|^2``, power
      mode ``|vol(Y)|^2`` minus the fitted atom profiles (not the constant).
    """
    if mode not in FIT_MODES:
        raise ValueError(f"mode must be one of {FIT_MODES}, got {mode!r}")
    if not isinstance(geom, CaptureGeometry):
        raise ValueError("geom must be a CaptureGeometry")
    v = _capture_index(v, geom.num_views, "v")
    b = _capture_index(b, geom.num_bs, "b")
    rows, cols = geom.aperture_shape
    expected = (2, rows, cols, geom.num_bins)
    Y_arr = np.asarray(Y_c, dtype=np.complex128)
    if Y_arr.shape != expected:
        raise ValueError(f"Y_c must have shape {expected}, got {Y_arr.shape}")
    if not np.all(np.isfinite(Y_arr)):
        raise ValueError("Y_c must contain only finite values")
    if oversample is None:
        oversample = DEFAULT_OVERSAMPLE if mode == "complex" else POWER_OVERSAMPLE
    if isinstance(oversample, bool) or not isinstance(oversample, (int, np.integer)):
        raise ValueError("oversample must be an integer >= 1")
    if int(oversample) < 1:
        raise ValueError("oversample must be an integer >= 1")

    points = nuisance_points(geom, b, ground_height)
    capture = geom.select([v], [b])
    columns = [
        atom_cfr(
            points[k : k + 1],
            np.ones(1),
            capture,
            "vs",
            pattern=pattern,
            polarization=polarization,
        )[0, 0]
        for k in range(points.shape[0])
    ]
    frequencies = np.asarray(geom.freq_offsets, dtype=np.float64)
    period = float(geom.delay_period)
    if mode == "complex":
        return _fit_complex(
            Y_arr,
            columns,
            frequencies,
            period,
            with_gauge=bool(with_gauge),
            free_los_phase=bool(free_los_phase),
            oversample=int(oversample),
        )
    return _fit_power(
        Y_arr,
        columns,
        frequencies,
        period,
        with_gauge=bool(with_gauge),
        oversample=int(oversample),
    )


def power_xcorr_delay(
    p_obs: np.ndarray, p_ref: np.ndarray, period: float, *, upsample: int = 16
) -> float:
    """Circular delay maximising the continuous power cross-correlation.

    ``p_obs`` and ``p_ref`` are real, finite, identically shaped profiles whose
    last axis holds ``Nt >= 2`` uniform delay samples over ``[0, period)``;
    leading axes are summed.  The estimate satisfies ``p_obs(t) ~ p_ref(t -
    tau)`` (positive ``tau`` is later, the gauge convention) and is returned in
    ``[-period/2, period/2)``.  It is exact up to interpolation only when the
    profiles carry at least two samples per delay cell (delay oversampling
    ``>= 2``, e.g. ``|angle_delay_volume(Y, oversample=(1, 2))|**2``); on the
    native grid (``extract(Y, "ID")``) the power is aliased and the estimate is
    biased by up to about 0.3 cell.
    """
    obs = np.asarray(p_obs)
    ref = np.asarray(p_ref)
    if np.iscomplexobj(obs) or np.iscomplexobj(ref):
        raise ValueError("p_obs and p_ref must be real")
    obs = obs.astype(np.float64, copy=False)
    ref = ref.astype(np.float64, copy=False)
    if obs.shape != ref.shape:
        raise ValueError("p_obs and p_ref must have the same shape")
    if obs.ndim < 1 or obs.shape[-1] < 2:
        raise ValueError("the last axis must hold at least two delay samples")
    if not np.all(np.isfinite(obs)) or not np.all(np.isfinite(ref)):
        raise ValueError("p_obs and p_ref must contain only finite values")
    period_f = float(period)
    if not np.isfinite(period_f) or period_f <= 0.0:
        raise ValueError("period must be finite and > 0")
    if isinstance(upsample, bool) or not isinstance(upsample, (int, np.integer)):
        raise ValueError("upsample must be an integer >= 1")
    if int(upsample) < 1:
        raise ValueError("upsample must be an integer >= 1")

    num_samples = obs.shape[-1]
    length = int(upsample) * num_samples
    leading = tuple(range(obs.ndim - 1))
    spectrum = np.sum(np.fft.fft(obs, axis=-1) * np.conj(np.fft.fft(ref, axis=-1)), axis=leading)
    padded = np.zeros(length, dtype=np.complex128)
    if num_samples % 2 == 0:
        half = num_samples // 2
        padded[:half] = spectrum[:half]
        padded[half] = 0.5 * spectrum[half]
        padded[length - half] = 0.5 * spectrum[half]
        padded[length - half + 1 :] = spectrum[half + 1 :]
    else:
        half = num_samples // 2
        padded[: half + 1] = spectrum[: half + 1]
        padded[length - half :] = spectrum[half + 1 :]
    correlation = np.real(np.fft.ifft(padded))
    index = int(np.argmax(correlation))
    previous = correlation[(index - 1) % length]
    following = correlation[(index + 1) % length]
    denominator = previous - 2.0 * correlation[index] + following
    offset = 0.0 if denominator == 0.0 else 0.5 * (previous - following) / denominator
    tau = (index + offset) * period_f / length
    return _wrap_delay(tau, period_f)


def _validate_capture_shape(Y: np.ndarray, op: SeparableOperator) -> np.ndarray:
    """Return ``Y`` as finite complex128 with shape ``op.y_shape``."""
    if np.shape(Y) != op.y_shape:
        raise ValueError(f"Y must have shape {op.y_shape}, got {np.shape(Y)}")
    values = np.asarray(Y, dtype=np.complex128)
    if not np.all(np.isfinite(values)):
        raise ValueError("Y must contain only finite values")
    return values


def _profile_gauges(
    Y: np.ndarray, model: np.ndarray, f: np.ndarray, oversample: int
) -> tuple[np.ndarray, np.ndarray, float]:
    """Profile one gauge per capture with the shared phase/delay aligner."""
    num_views, num_bs = Y.shape[:2]
    phi = np.empty((num_views, num_bs), dtype=np.float64)
    tau = np.empty((num_views, num_bs), dtype=np.float64)
    cost = 0.0
    for v in range(num_views):
        for b in range(num_bs):
            alignment = align_common_phase_and_delay(
                Y[v, b], model[v, b], f, oversample=int(oversample)
            )
            phi[v, b] = alignment.phase_rad
            tau[v, b] = alignment.delay_s
            cost += float(np.sum(np.abs(Y[v, b] - alignment.aligned_ref) ** 2))
    return phi, tau, float(cost)


def varpro_cost_and_grad(
    x: np.ndarray,
    Y: np.ndarray,
    op: SeparableOperator,
    *,
    oversample: int = DEFAULT_OVERSAMPLE,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Profiled VarPro loss, gradient and per-capture gauges of design §3.6/§4.2.

    Any gauges carried by ``op`` are ignored.  The gradient satisfies
    ``d cost[x + t d]/dt = real(vdot(grad, d))`` at the optimal gauges
    (Danskin), and ``phi``/``tau`` are the raw per-capture gauges with no
    reference fixing.
    """
    if np.shape(x) != op.x_shape:
        raise ValueError(f"x must have shape {op.x_shape}, got {np.shape(x)}")
    values = np.asarray(x, dtype=np.complex128)
    if not np.all(np.isfinite(values)):
        raise ValueError("x must contain only finite values")
    data = _validate_capture_shape(Y, op)
    base = op.with_gauges(None)
    model = base.forward(values)
    frequencies = np.asarray(op.geom.freq_offsets, dtype=np.float64)
    phi, tau, cost = _profile_gauges(data, model, frequencies, int(oversample))
    residual = model - apply_gauge(data, -phi, -tau, frequencies)
    gradient = 2.0 * base.adjoint(residual)
    return cost, np.asarray(gradient, dtype=np.complex128), phi, tau


def self_calibrate(
    solve: Callable[[np.ndarray], np.ndarray],
    Y: np.ndarray,
    op_factory: Callable[[tuple[np.ndarray, np.ndarray] | None], SeparableOperator],
    n_iter: int = SELF_CAL_ITERATIONS,
    init: tuple[np.ndarray, np.ndarray] | None = None,
    ref: tuple[int, int] = (0, 0),
    *,
    rtol: float = SELF_CAL_RTOL,
    oversample: int = DEFAULT_OVERSAMPLE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Alternate between ``solve`` and per-capture gauge profiling.

    ``solve(Y_aligned)`` returns the S-mode amplitudes of de-gauged data;
    ``op_factory(gauges)`` builds the forward operator for a gauge pair (``None``
    is S mode).  Each iteration de-gauges ``Y`` with the current gauges, solves,
    re-profiles the gauges against ``op_factory(None).forward(x)`` and fixes the
    reference phase ``phi[ref] = 0`` by rotating ``x``.  Stops when the relative
    change of the profiled loss drops below ``rtol`` (or after ``n_iter``).
    """
    base = op_factory(None)
    if not isinstance(base, SeparableOperator):
        raise ValueError("op_factory(None) must return a SeparableOperator")
    frequencies = np.asarray(base.geom.freq_offsets, dtype=np.float64)
    data = _validate_capture_shape(Y, base)
    num_views, num_bs = base.y_shape[:2]
    if int(n_iter) < 1:
        raise ValueError("n_iter must be >= 1")
    if not np.isfinite(float(rtol)) or float(rtol) < 0.0:
        raise ValueError("rtol must be finite and >= 0")
    v0 = _capture_index(ref[0], num_views, "ref view")
    b0 = _capture_index(ref[1], num_bs, "ref BS")

    if init is None:
        phi = np.zeros((num_views, num_bs), dtype=np.float64)
        tau = np.zeros((num_views, num_bs), dtype=np.float64)
    else:
        phi = np.array(init[0], dtype=np.float64, copy=True)
        tau = np.array(init[1], dtype=np.float64, copy=True)
        if phi.shape != (num_views, num_bs) or tau.shape != (num_views, num_bs):
            raise ValueError(f"init gauges must both have shape {(num_views, num_bs)}")
        if not np.all(np.isfinite(phi)) or not np.all(np.isfinite(tau)):
            raise ValueError("init gauges must contain only finite values")

    history = np.empty(int(n_iter), dtype=np.float64)
    converged = False
    previous = 0.0
    done = 0
    x = np.zeros(base.x_shape, dtype=np.complex128)
    for iteration in range(1, int(n_iter) + 1):
        aligned = apply_gauge(data, -phi, -tau, frequencies)
        x = np.asarray(solve(aligned), dtype=np.complex128)
        if np.shape(x) != base.x_shape:
            raise ValueError(f"solve must return shape {base.x_shape}, got {np.shape(x)}")
        phi, tau, loss = _profile_gauges(data, base.forward(x), frequencies, int(oversample))
        rotation = phi[v0, b0]
        phi = _wrap_phases(phi - rotation)
        phi[v0, b0] = 0.0
        x = x * np.exp(1j * rotation)
        history[done] = loss
        done = iteration
        if iteration > 1 and abs(previous - loss) <= float(rtol) * previous:
            converged = True
            break
        previous = loss

    history_out = {
        "loss": history[:done].astype(np.float64, copy=True),
        "converged": bool(converged),
        "n_iter": int(done),
    }
    return x, phi, tau, history_out
