"""The benchmark runner (design §6, §8 T16): run every registered config, Sionna-free.

For every registered configuration the runner executes the T15 chain through the
executors, estimates N-mode gauges with the registered strategies, computes
``ill_posed`` with T15b at the estimate and scores detections with T10.
"""

from __future__ import annotations

import contextlib
import dataclasses
import multiprocessing
import shutil
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, TextIO

import numpy as np
from scipy.optimize import minimize

from plateau_rt.application import rf_tomography_io as tio
from plateau_rt.domain.rf_tomography import kernels as kernel_mod
from plateau_rt.domain.rf_tomography import metrics as metric_mod
from plateau_rt.domain.rf_tomography import sync as sync_mod
from plateau_rt.domain.rf_tomography.configs import (
    CONFIGS,
    Config,
    E2Output,
    get_config,
    run_e1,
    run_e2,
    run_roi,
    run_support,
)
from plateau_rt.domain.rf_tomography.forward_exact import atom_cfr
from plateau_rt.domain.rf_tomography.forward_sep import SeparableOperator
from plateau_rt.domain.rf_tomography.geometry import CaptureGeometry, VoxelGrid
from plateau_rt.domain.rf_tomography.identifiability import (
    gauge_reduced_fim,
    ill_posed,
    numeric_jacobian,
    return_model,
)
from plateau_rt.domain.rf_tomography.observables import extract
from plateau_rt.domain.rf_tomography.solvers import bp as bp_mod
from plateau_rt.domain.rf_tomography.solvers import coherent as coherent_mod
from plateau_rt.domain.rf_tomography.solvers import power as power_mod
from plateau_rt.domain.rf_tomography.views import nested_view_order

TRACKS: tuple[str, ...] = ("ideal-S", "ideal-N", "N-sep", "S_tau")
TRACK_SYNC: dict[str, str] = {
    "ideal-S": "S",
    "ideal-N": "N",
    "N-sep": "N_sep",
    "S_tau": "S_tau",
}
GATES_M: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
NMS_RADIUS_CELLS: float = 1.5
PEAK_REL_THRESHOLD: float = 0.05
RHO_MIN: float = 10.0
XCORR_ALTERNATIONS: int = 2
# The operators, E1 maps and gauge fits use their default BS pattern (TR 38.901, scalar
# polarisation); a dataset traced with another transmit pattern would be silently mismatched.
SUPPORTED_TX_PATTERNS: tuple[str, ...] = ("tr38901",)
LIST_COMPONENTS: dict[str, tuple[str, ...]] = {
    "D": ("u_y", "u_z", "t"),
    "ID": ("u_y", "u_z", "t"),
    "I": ("u_y", "u_z"),
    "I_n0": ("u_y", "u_z"),
    "P_W": ("u_y", "u_z"),
    "IP_W": ("u_y", "u_z"),
    "D-o": ("t",),
    "ID-o": ("t",),
    "DP-o": ("t",),
    "IDP-o": ("t",),
    "I-o": (),
    "IP-o": (),
    "P-o": (),
}
COMPLEX_NODES: tuple[str, ...] = ("P", "IP", "DP", "IDP", "PxK", "IPxK", "IDP-1el", "DP-1el")
_LIST_AXES: dict[str, int] = {"u_y": 0, "u_z": 1, "t": 2}


@dataclass(frozen=True)
class Suite:
    """One benchmark suite: spaces, realizations, iterations and grid defaults."""

    name: str
    spaces: tuple[str, ...]
    realizations: int
    e2_iterations: int
    grid_spacing: float
    grid_half_size: tuple[float, float, float]
    max_peaks: int
    roi_max: int
    ill_max_points: int
    snr_db: float = 30.0
    sigma_t: float = 10e-9


SUITES: Mapping[str, Suite] = MappingProxyType(
    {
        "unit": Suite("unit", ("bv",), 1, 10, 2.0, (4.0, 4.0, 2.0), 8, 2, 4),
        "smoke": Suite("smoke", ("bv", "vs"), 1, 10, 2.0, (10.0, 10.0, 6.0), 8, 4, 4),
        "full": Suite("full", ("bv", "vs"), 5, 200, 0.5, (10.0, 10.0, 10.0), 32, 8, 8),
    }
)


def config_track(cfg: Config, tracks: Sequence[str]) -> str | None:
    """Return the track of ``tracks`` selected for ``cfg`` (None when unsupported)."""
    wanted = list(tracks)
    if cfg.sync == "any":
        if "ideal-S" in wanted:
            return "ideal-S"
        if "ideal-N" in wanted:
            return "ideal-N"
        return None
    for track in wanted:
        if TRACK_SYNC.get(track) == cfg.sync:
            return track
    return None


def job_strategies(cfg: Config) -> tuple[str, ...]:
    """Return the strategy names of one ``(config, track, space)`` job."""
    if not cfg.gauge_unknowns:
        return ("none",)
    return ("none", *[strategy.name for strategy in cfg.n_strategies], "oracle")


def make_grid(
    center: Sequence[float] | np.ndarray,
    half_size: Sequence[float] | np.ndarray,
    spacing: float,
) -> VoxelGrid:
    """Build a voxel grid from ``center``, ``half_size`` and ``spacing``."""
    middle = np.asarray(center, dtype=np.float64).reshape(3)
    half = np.asarray(half_size, dtype=np.float64).reshape(3)
    return VoxelGrid.from_bounds(middle - half, middle + half, float(spacing))


def detect(density: np.ndarray, grid: VoxelGrid, max_peaks: int) -> tuple[np.ndarray, np.ndarray]:
    """Detect peaks of ``density`` sorted by score descending (stable)."""
    volume = np.asarray(density, dtype=np.float64)
    finite = np.isfinite(volume)
    if not bool(np.all(finite)):
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    low = float(np.min(volume))
    high = float(np.max(volume))
    if not high > low:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    peaks = metric_mod.nms_peaks(
        volume,
        grid,
        NMS_RADIUS_CELLS * grid.spacing,
        refine=True,
        min_value=low + PEAK_REL_THRESHOLD * (high - low),
        max_peaks=max_peaks,
    )
    order = np.argsort(-peaks.values, kind="stable")
    return np.asarray(peaks.positions[order], dtype=np.float64), np.asarray(
        peaks.values[order], dtype=np.float64
    )


@dataclass(frozen=True)
class GaugeEstimate:
    """One estimated gauge pair with the components actually estimated."""

    phi: np.ndarray
    tau: np.ndarray
    estimates: tuple[str, ...]
    info: Mapping[str, Any]


def _strategy_of(cfg: Config, strategy: str) -> Any:
    """Return the registered ``NStrategy`` of ``cfg`` called ``strategy``."""
    for item in cfg.n_strategies:
        if item.name == strategy:
            return item
    raise ValueError(f"unknown strategy {strategy!r} for config {cfg.name!r}")


def estimate_gauges(
    cfg: Config,
    strategy: str,
    Y: np.ndarray,
    geom: CaptureGeometry,
    grid: VoxelGrid,
    space: str,
    *,
    noise_var: float | None,
    ref: tuple[int, int],
    n_iter: int,
    sigma_t: float | str,
    points: np.ndarray | None = None,
) -> GaugeEstimate:
    """Estimate the N-mode gauges of ``Y`` with the registered ``strategy``.

    The ``xcorr`` estimate is biased by up to ~0.3 cell on the native delay grid
    (a known limitation of the power cross-correlation, not fixed here).
    """
    data = np.asarray(Y, dtype=np.complex128)
    period = float(geom.delay_period)
    num_views, num_bs = geom.num_views, geom.num_bs
    if strategy == "none":
        zeros = np.zeros((num_views, num_bs), dtype=np.float64)
        return GaugeEstimate(phi=zeros, tau=zeros.copy(), estimates=(), info={})
    if strategy == "oracle":
        raise ValueError("the oracle strategy is handled by the runner")
    strat = _strategy_of(cfg, strategy)
    if strategy == "los":
        phi = np.zeros((num_views, num_bs), dtype=np.float64)
        tau = np.zeros((num_views, num_bs), dtype=np.float64)
        resid = np.zeros((num_views, num_bs), dtype=np.float64)
        for view in range(num_views):
            for bs in range(num_bs):
                result = strat.fn(data[view, bs], geom, view, bs, **dict(strat.kwargs))
                phase = float(result["phi"])
                phi[view, bs] = phase if np.isfinite(phase) else 0.0
                tau[view, bs] = float(result["tau"])
                resid[view, bs] = float(result["resid"])
        return GaugeEstimate(
            phi=phi, tau=tau, estimates=tuple(strat.estimates), info={"resid": resid}
        )
    if strategy == "blind":
        factory, keywords = strat.aux["bp_fn"]
        width = 3.0 * float(sigma_t) if isinstance(sigma_t, float) else 0.5 * period
        width = min(float(width), 0.5 * period * (1.0 - 1e-9))
        # The registry factory kwargs omit noise_var, which the PHAT nodes (DP) need for
        # their mask; the envelope ignores it for the other nodes.
        bp_fn = factory(space, **dict(keywords), noise_var=noise_var)
        tau = np.asarray(strat.fn(bp_fn, data, geom, grid, (-width, width)), dtype=np.float64)
        phi = np.zeros((num_views, num_bs), dtype=np.float64)
        return GaugeEstimate(phi=phi, tau=tau, estimates=tuple(strat.estimates), info={})
    if strategy in ("self_cal", "varpro"):
        if points is not None:
            support = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        else:
            support = None
        if support is None or support.shape[0] == 0:
            if points is not None:
                raise ValueError("empty support")
            estimate = run_e1(cfg, data, geom, grid, space, noise_var=noise_var)
            support = run_support(estimate, grid)[1]
            if support.shape[0] == 0:
                raise ValueError("empty support")
        raw = bp_mod.node_data(data, cfg.node, noise_var=noise_var)
        scale = float(np.sqrt(np.mean(np.abs(raw) ** 2)))
        scaled = raw if scale == 0.0 else (raw / scale).astype(np.complex128)
        operator = SeparableOperator(support, geom, space)
        damp = 0.1 * float(np.sqrt(coherent_mod.lipschitz_constant(operator, safety=1.0)))
        if strategy == "self_cal":
            amplitudes, phi, tau, history = strat.fn(
                lambda aligned: (
                    coherent_mod.tikhonov_lsqr(operator, aligned, damp, iter_lim=int(n_iter)).x
                ),
                scaled,
                lambda gauges: SeparableOperator(support, geom, space, gauges=gauges),
                ref=ref,
            )
            del amplitudes
            return GaugeEstimate(
                phi=np.asarray(phi, dtype=np.float64),
                tau=np.asarray(tau, dtype=np.float64),
                estimates=tuple(strat.estimates),
                info={"n_iter": int(history["n_iter"]), "converged": bool(history["converged"])},
            )
        first = coherent_mod.tikhonov_lsqr(operator, scaled, damp, iter_lim=int(n_iter)).x
        start = np.concatenate([first.real.ravel(), first.imag.ravel()]).astype(np.float64)
        count = int(start.shape[0] // 2)

        def objective(vector: np.ndarray) -> tuple[float, np.ndarray]:
            """Unpack ``vector`` to complex amplitudes and return cost and gradient."""
            array = np.asarray(vector, dtype=np.float64)
            trial = array[:count] + 1j * array[count:]
            cost, gradient, _, _ = strat.fn(trial, scaled, operator)
            packed = np.concatenate(
                [
                    np.asarray(gradient.real, dtype=np.float64).ravel(),
                    np.asarray(gradient.imag, dtype=np.float64).ravel(),
                ]
            )
            return float(cost), packed

        result = minimize(
            objective, start, jac=True, method="L-BFGS-B", options={"maxiter": int(n_iter)}
        )
        optimum = np.asarray(result.x, dtype=np.float64)
        best = optimum[:count] + 1j * optimum[count:]
        _, _, phi_raw, tau_raw = strat.fn(best, scaled, operator)
        phi = np.asarray(phi_raw, dtype=np.float64)
        tau = np.asarray(tau_raw, dtype=np.float64)
        phi = np.angle(np.exp(1j * (phi - phi[ref])))
        phi[ref] = 0.0
        return GaugeEstimate(
            phi=phi, tau=tau, estimates=tuple(strat.estimates), info={"nit": int(result.nit)}
        )
    if strategy == "xcorr":
        product = cfg.e2[0].operator["product"]
        observed = extract(data, kernel_mod.PRODUCT_NODES[product]).data
        tau = np.zeros((num_views, num_bs), dtype=np.float64)
        for _ in range(XCORR_ALTERNATIONS):
            gauges = (np.zeros((num_views, num_bs), dtype=np.float64), tau)
            estimate = run_e1(cfg, data, geom, grid, space, gauges=gauges, noise_var=noise_var)
            _, support, _ = run_support(estimate, grid)
            if support.shape[0] == 0:
                raise ValueError("empty support")
            solved = run_e2(
                cfg,
                "kl_em",
                data,
                geom,
                support,
                space,
                gauges=gauges,
                noise_var=noise_var,
                n_iter=int(n_iter),
            )
            predicted = (
                kernel_mod.power_operator(support, geom, space, product)
                .matvec(solved.density)
                .reshape(observed.shape)
            )
            for view in range(num_views):
                for bs in range(num_bs):
                    if np.any(predicted[view, bs] > 0.0):
                        tau[view, bs] = float(
                            strat.fn(observed[view, bs], predicted[view, bs], period)
                        )
        phi = np.zeros((num_views, num_bs), dtype=np.float64)
        return GaugeEstimate(phi=phi, tau=tau, estimates=tuple(strat.estimates), info={})
    raise ValueError(f"unknown strategy {strategy!r} for config {cfg.name!r}")


def used_gauges(cfg: Config, estimate: GaugeEstimate) -> tuple[np.ndarray, np.ndarray]:
    """Mask ``estimate`` to the gauge unknowns of ``cfg`` (others become zeros)."""
    unknowns = cfg.gauge_unknowns
    phi = np.asarray(estimate.phi, dtype=np.float64)
    tau = np.asarray(estimate.tau, dtype=np.float64)
    used_phi = phi if "phi" in unknowns else np.zeros_like(phi)
    used_tau = tau if "tau" in unknowns else np.zeros_like(tau)
    return used_phi, used_tau


def gauge_error_summary(
    cfg: Config,
    used: tuple[np.ndarray, np.ndarray],
    truth: tuple[np.ndarray, np.ndarray],
    period: float,
    estimates: Sequence[str],
) -> dict[str, Any]:
    """Summarize wrapped gauge errors of ``used`` against ``truth``."""
    errors = metric_mod.gauge_errors(used, truth, period=float(period))
    summary: dict[str, Any] = {"estimated": list(estimates)}
    if "phi" in cfg.gauge_unknowns:
        phase = np.abs(np.asarray(errors.phase, dtype=np.float64))
        summary["phase_rms_deg"] = float(np.rad2deg(np.sqrt(np.mean(phase**2))))
        summary["phase_max_deg"] = float(np.rad2deg(np.max(phase)))
    if "tau" in cfg.gauge_unknowns:
        delay = np.abs(np.asarray(errors.delay, dtype=np.float64))
        summary["delay_rms_ns"] = float(1e9 * np.sqrt(np.mean(delay**2)))
        summary["delay_max_ns"] = float(1e9 * np.max(delay))
    return summary


@dataclass(frozen=True)
class IllPosedReport:
    """Identifiability of the estimate: flag, condition, CRB and provenance."""

    flag: bool
    cond: float
    crb_std: np.ndarray
    model: str
    components: tuple[str, ...]
    points: np.ndarray
    rho: np.ndarray
    nuisance: tuple[str, ...]


def capture_amplitudes(
    Y: np.ndarray, geom: CaptureGeometry, points: np.ndarray, space: str
) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares per-capture amplitudes and column norms of ``points``."""
    data = np.asarray(Y, dtype=np.complex128)
    locations = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    num_points = int(locations.shape[0])
    num_views, num_bs = geom.num_views, geom.num_bs
    amplitudes = np.zeros((num_views, num_bs, num_points), dtype=np.complex128)
    norms = np.zeros((num_views, num_bs, num_points), dtype=np.float64)
    if num_points == 0:
        return amplitudes, norms
    for view in range(num_views):
        for bs in range(num_bs):
            single = geom.select([view], [bs])
            columns = np.stack(
                [
                    np.asarray(
                        atom_cfr(locations[p : p + 1], np.ones(1), single, space).ravel(),
                        dtype=np.complex128,
                    )
                    for p in range(num_points)
                ],
                axis=1,
            )
            solution, _, _, _ = np.linalg.lstsq(columns, data[view, bs].ravel(), rcond=None)
            amplitudes[view, bs] = solution
            norms[view, bs] = np.sum(np.abs(columns) ** 2, axis=0)
    return amplitudes, norms


def list_sigmas(rho: np.ndarray, geom: CaptureGeometry) -> np.ndarray:
    """3-D harmonic CRB ``[V, B, P, 3]`` of an isolated path (σ_uy, σ_uz, σ_t in ns)."""
    weights = np.asarray(rho, dtype=np.float64)
    rows, cols = (int(value) for value in geom.aperture_shape)
    num_bins = int(geom.num_bins)
    wavelength = float(geom.wavelength)
    spacing = np.asarray(geom.elem_offsets, dtype=np.float64)
    if cols > 1:
        across = float(np.linalg.norm(spacing[1] - spacing[0])) / wavelength
    else:
        across = float("nan")
    if rows > 1:
        along = float(np.linalg.norm(spacing[cols] - spacing[0])) / wavelength
    else:
        along = float("nan")
    delta_f = float(geom.delta_f)
    positive = weights > 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_t = 1e9 * np.sqrt(6.0 / (weights * (num_bins**2 - 1))) / (2.0 * np.pi * delta_f)
        if cols > 1:
            sigma_uy = np.sqrt(6.0 / (weights * (cols**2 - 1))) / (2.0 * np.pi * across)
        else:
            sigma_uy = np.full(weights.shape, np.inf)
        if rows > 1:
            sigma_uz = np.sqrt(6.0 / (weights * (rows**2 - 1))) / (2.0 * np.pi * along)
        else:
            sigma_uz = np.full(weights.shape, np.inf)
    sigma_t = np.where(positive, sigma_t, np.inf)
    sigma_uy = np.where(positive, np.asarray(sigma_uy, dtype=np.float64), np.inf)
    sigma_uz = np.where(positive, np.asarray(sigma_uz, dtype=np.float64), np.inf)
    return np.stack([sigma_uy, sigma_uz, sigma_t], axis=-1).astype(np.float64)


def _node_kind(cfg: Config) -> str:
    """Return ``'list'`` or ``'complex'`` for the node of ``cfg``."""
    if cfg.node in LIST_COMPONENTS:
        return "list"
    if cfg.node in COMPLEX_NODES:
        return "complex"
    raise ValueError(f"unknown tomography node {cfg.node!r}")


def _expand_gauge(vector: np.ndarray, num_views: int, num_bs: int, separable: bool) -> np.ndarray:
    """Expand one gauge parameter vector to a full ``[V, B]`` array."""
    flat = np.asarray(vector, dtype=np.float64).ravel()
    if separable:
        return flat[:num_views, None] + flat[num_views:][None, :]
    return flat.reshape(num_views, num_bs)


def _complex_selection(
    cfg: Config, Y: np.ndarray
) -> tuple[str, tuple[int, ...] | tuple[int, int] | None]:
    """Return the row selection of a complex node (bins, element or all)."""
    if cfg.node in ("P", "IP"):
        return "bins", (int(np.asarray(Y).shape[-1] // 2),)
    if cfg.node in ("PxK", "IPxK"):
        return "bins", tuple(sorted(int(b) for b in extract(Y, "IPxK").meta["bins"]))
    if cfg.node in ("IDP-1el", "DP-1el"):
        element = extract(Y, "IDP-1el").meta["element"]
        return "element", (int(element[0]), int(element[1]))
    return "all", None


def _apply_selection(
    array: np.ndarray, kind: str, selection: tuple[int, ...] | tuple[int, int] | None
) -> np.ndarray:
    """Apply a row selection to a full-layout complex array."""
    if kind == "bins":
        assert selection is not None
        return np.asarray(array[..., list(selection)])
    if kind == "element":
        assert selection is not None
        row, col = selection
        return np.asarray(array[:, :, :, int(row), int(col), :])
    return np.asarray(array)


def ill_posed_at(
    cfg: Config,
    Y: np.ndarray,
    geom: CaptureGeometry,
    points: np.ndarray,
    space: str,
    *,
    noise_var: float | None,
    gauges: tuple[np.ndarray, np.ndarray] | None = None,
) -> IllPosedReport:
    """Compute ``ill_posed`` at the estimate with Gaussian weights (until T30)."""
    if noise_var is None:
        raise ValueError("ill_posed_at requires noise_var")
    variance = float(noise_var)
    if not np.isfinite(variance) or variance < 0.0:
        raise ValueError("noise_var must be finite and >= 0")
    data = np.asarray(Y, dtype=np.complex128)
    locations = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    num_views, num_bs = geom.num_views, geom.num_bs
    separable = cfg.sync == "N_sep"
    kind = _node_kind(cfg)
    if kind == "list":
        names = LIST_COMPONENTS[cfg.node]
        axes = tuple(_LIST_AXES[name] for name in names)
        nuisance: tuple[str, ...] = ("tau",) if "tau" in cfg.gauge_unknowns else ()
    else:
        names = ("re", "im")
        axes = ()
        nuisance = ("amplitude", *cfg.gauge_unknowns)
    if gauges is None:
        aligned = data
    else:
        phi_given, tau_given = gauges
        aligned = sync_mod.apply_gauge(data, -phi_given, -tau_given, geom.freq_offsets)
    amps, norms = capture_amplitudes(aligned, geom, locations, space)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = (
            np.abs(amps) ** 2 * norms / variance if variance > 0.0 else np.full(amps.shape, np.inf)
        )
    snr = np.asarray(snr, dtype=np.float64)
    keep = (
        np.max(snr, axis=(0, 1)) >= RHO_MIN
        if locations.shape[0] > 0
        else np.zeros((0,), dtype=bool)
    )
    kept_points = np.asarray(locations[keep], dtype=np.float64)
    kept_snr = np.asarray(snr[:, :, keep], dtype=np.float64)
    num_kept = int(kept_points.shape[0])
    if num_kept == 0:
        return IllPosedReport(
            flag=True,
            cond=float("inf"),
            crb_std=np.zeros((0,), dtype=np.float64),
            model=kind,
            components=tuple(names),
            points=np.zeros((0, 3), dtype=np.float64),
            rho=np.zeros((num_views, num_bs, 0), dtype=np.float64),
            nuisance=nuisance,
        )
    if kind == "list":
        count_tau = (
            (num_views + num_bs)
            if separable and nuisance
            else (num_views * num_bs if nuisance else 0)
        )
        theta = np.concatenate([kept_points.ravel(), np.zeros((count_tau,), dtype=np.float64)])

        def list_model(vector: np.ndarray) -> np.ndarray:
            """Return the ``[V, B, P, k]`` list mean for ``[pos, tau_ns]``."""
            array = np.asarray(vector, dtype=np.float64)
            pos = array[: 3 * num_kept].reshape(num_kept, 3)
            raw = array[3 * num_kept :]
            delays = (
                _expand_gauge(raw, num_views, num_bs, separable) * 1e-9
                if count_tau
                else np.zeros((num_views, num_bs), dtype=np.float64)
            )
            mean = return_model(pos, geom, delays, space=space).copy()
            mean[..., 2] *= 1e9
            return (
                mean[..., list(axes)]
                if axes
                else np.zeros((num_views, num_bs, num_kept, 0), dtype=np.float64)
            )

        jacobian = numeric_jacobian(list_model, theta)
        spread = list_sigmas(kept_snr, geom)
        selected = (
            spread[..., list(axes)]
            if axes
            else np.zeros((num_views, num_bs, num_kept, 0), dtype=np.float64)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            weights = (
                np.where((kept_snr >= RHO_MIN)[..., None], 1.0 / selected**2, 0.0).ravel(order="C")
                if axes
                else np.zeros((0,), dtype=np.float64)
            )
        weights = np.asarray(weights, dtype=np.float64)
        nuisance_idx = np.arange(3 * num_kept, 3 * num_kept + count_tau)
    else:
        sel_kind, selection = _complex_selection(cfg, data)
        design = np.stack(
            [
                _apply_selection(
                    atom_cfr(kept_points[p : p + 1], np.ones(1), geom, space),
                    sel_kind,
                    selection,
                ).ravel()
                for p in range(num_kept)
            ],
            axis=1,
        ).astype(np.complex128)
        target = _apply_selection(aligned, sel_kind, selection).ravel()
        shared, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
        shared = np.asarray(shared, dtype=np.complex128).reshape(-1)
        count_phi = (num_views + num_bs) if separable else num_views * num_bs
        count_phi = count_phi if "phi" in cfg.gauge_unknowns else 0
        count_tau = (num_views + num_bs) if separable else num_views * num_bs
        count_tau = count_tau if "tau" in cfg.gauge_unknowns else 0
        theta = np.concatenate(
            [
                kept_points.ravel(),
                shared.real,
                shared.imag,
                np.zeros((count_phi + count_tau,), dtype=np.float64),
            ]
        )
        num_complex = int(target.shape[0])

        def complex_model(vector: np.ndarray) -> np.ndarray:
            """Return the selected complex mean for ``[pos, re, im, phi, tau_ns]``."""
            array = np.asarray(vector, dtype=np.float64)
            pos = array[: 3 * num_kept].reshape(num_kept, 3)
            cursor = 3 * num_kept
            re = array[cursor : cursor + num_kept]
            im = array[cursor + num_kept : cursor + 2 * num_kept]
            cursor += 2 * num_kept
            phi_vec = array[cursor : cursor + count_phi]
            tau_vec = array[cursor + count_phi : cursor + count_phi + count_tau]
            phi_full = (
                _expand_gauge(phi_vec, num_views, num_bs, separable)
                if count_phi
                else np.zeros((num_views, num_bs), dtype=np.float64)
            )
            tau_full = (
                _expand_gauge(tau_vec, num_views, num_bs, separable) * 1e-9
                if count_tau
                else np.zeros((num_views, num_bs), dtype=np.float64)
            )
            mean = sync_mod.apply_gauge(
                atom_cfr(pos, re + 1j * im, geom, space),
                phi_full,
                tau_full,
                geom.freq_offsets,
            )
            return _apply_selection(mean, sel_kind, selection)

        jacobian = numeric_jacobian(complex_model, theta)
        weights = np.full(2 * num_complex, 2.0 / variance, dtype=np.float64)
        nuisance_idx = np.arange(3 * num_kept + 2 * num_kept, theta.shape[0])
    reduced = gauge_reduced_fim(jacobian, weights, nuisance_idx)
    flag, cond, std = ill_posed(reduced, np.arange(3 * num_kept))
    return IllPosedReport(
        flag=bool(flag),
        cond=float(cond),
        crb_std=np.asarray(std, dtype=np.float64),
        model=kind,
        components=tuple(names),
        points=kept_points,
        rho=kept_snr,
        nuisance=nuisance,
    )


def detection_metrics(
    det: np.ndarray,
    scores: np.ndarray,
    gt: np.ndarray,
    grid: VoxelGrid,
    ue_ref: np.ndarray,
) -> dict[str, Any]:
    """Score detections against ``gt`` at the design §6.4 gates."""
    detections = np.asarray(det, dtype=np.float64).reshape(-1, 3)
    values = np.asarray(scores, dtype=np.float64).reshape(detections.shape[0])
    truth = np.asarray(gt, dtype=np.float64).reshape(-1, 3)
    num_gt = int(truth.shape[0])
    num_det = int(detections.shape[0])
    gates: dict[str, dict[str, Any]] = {}
    for gate in GATES_M:
        matched = metric_mod.match(detections, truth, float(gate))
        gates[f"{gate}"] = {
            "tp": int(matched.tp),
            "fp": int(matched.fp),
            "fn": int(matched.fn),
            "precision": (float(matched.tp) / num_det) if num_det else None,
            "recall": float(matched.tp) / num_gt if num_gt else None,
        }
    if num_det:
        ap_1m = float(metric_mod.ap_at(detections, values, truth, 1.0))
        curve = metric_mod.froc(
            detections,
            values,
            truth,
            1.0,
            volume_m3=float(grid.size * grid.spacing**3),
        )
        recall_1fa = float(metric_mod.recall_at_fa(curve, 1.0))
    else:
        ap_1m = 0.0
        recall_1fa = 0.0
    matched_1m = metric_mod.match(detections, truth, 1.0)
    located = metric_mod.loc_error_decomposed(
        detections[matched_1m.det_idx], truth[matched_1m.gt_idx], np.asarray(ue_ref)
    ).summary()
    return {
        "num_gt": num_gt,
        "num_det": num_det,
        "gates": gates,
        "ap_1m": ap_1m,
        "recall_at_1fa": recall_1fa,
        "loc_1m": {key: float(item) for key, item in located.items()},
    }


@dataclass(frozen=True)
class BenchmarkRun:
    """One finished benchmark: output paths and every result row."""

    out_dir: Path
    results_path: Path
    run_manifest_path: Path
    rows: tuple[dict[str, Any], ...]


def _resolve_suite(suite: str | Suite) -> Suite:
    """Return the ``Suite`` for a name or instance."""
    if isinstance(suite, Suite):
        return suite
    try:
        return SUITES[suite]
    except KeyError as error:
        raise ValueError(f"unknown suite {suite!r}; expected one of {sorted(SUITES)}") from error


def _resolve_tracks(tracks: Sequence[str] | None) -> list[str]:
    """Return the validated track list (default all tracks)."""
    wanted = list(TRACKS) if tracks is None else list(tracks)
    unknown = [track for track in wanted if track not in TRACKS]
    if unknown:
        raise ValueError(f"unknown tracks {unknown}; expected a subset of {list(TRACKS)}")
    return wanted


def _resolve_configs(configs: Sequence[str] | None) -> list[Config]:
    """Return the selected configs in report order (aliases allowed)."""
    if configs is None:
        return list(CONFIGS.values())
    return [get_config(str(name)) for name in configs]


def _resolve_spaces(spaces: Sequence[str] | None, suite: Suite) -> list[str]:
    """Return the validated space list (default the suite's spaces)."""
    wanted = list(suite.spaces) if spaces is None else list(spaces)
    unknown = [space for space in wanted if space not in ("bv", "vs")]
    if unknown:
        raise ValueError(f"unknown spaces {unknown}; expected a subset of ['bv', 'vs']")
    return wanted


def _ill_payload(report: IllPosedReport) -> dict[str, Any]:
    """Convert an ``IllPosedReport`` to its JSON row payload."""
    std = [float(item) for item in np.asarray(report.crb_std, dtype=np.float64).ravel()]
    return {
        "flag": bool(report.flag),
        "cond": float(report.cond),
        "crb_std_m": std,
        "crb_std_max_m": (float(np.max(report.crb_std)) if std else None),
        "model": str(report.model),
        "components": list(report.components),
        "n_points": int(np.asarray(report.points).reshape(-1, 3).shape[0]),
        "points": np.asarray(report.points, dtype=np.float64).reshape(-1, 3).tolist(),
        "rho": np.asarray(report.rho, dtype=np.float64).tolist(),
        "nuisance": list(report.nuisance),
    }


def run_benchmark(
    dataset: tio.TomographyDataset | Path | str,
    out_dir: Path | str,
    suite: str | Suite = "unit",
    *,
    tracks: Sequence[str] | None = None,
    configs: Sequence[str] | None = None,
    strategies: Sequence[str] | None = None,
    spaces: Sequence[str] | None = None,
    gt_path: Path | str | None = None,
    grid_center: Sequence[float] | np.ndarray | None = None,
    grid_half_size: Sequence[float] | np.ndarray | None = None,
    grid_spacing: float | None = None,
    dataset_seed: int = 0,
    overwrite: bool = False,
    workers: int = 1,
) -> BenchmarkRun:
    """Run the tomography benchmark and stream rows, recon arrays and the manifest.

    ``workers == 1`` (the default) runs every strategy chain in this process, so
    the per-stage ``runtime_s`` is an uncontended wall time (design §6.4 M9).
    ``workers > 1`` runs the chains in a spawn-context process pool; rows are
    still written in job order and are identical apart from ``runtime_s``.
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be an integer >= 1")
    suite_obj = _resolve_suite(suite)
    track_list = _resolve_tracks(tracks)
    config_list = _resolve_configs(configs)
    space_list = _resolve_spaces(spaces, suite_obj)
    strategy_filter = None if strategies is None else list(strategies)
    data = dataset if isinstance(dataset, tio.TomographyDataset) else tio.load_dataset(dataset)
    if data.tx_pattern is not None and data.tx_pattern not in SUPPORTED_TX_PATTERNS:
        raise ValueError(
            f"dataset tx_pattern {data.tx_pattern!r} is not modelled; the tomography operators "
            f"assume one of {SUPPORTED_TX_PATTERNS}"
        )
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / tio.RESULTS_FILE
    manifest_path = out / tio.RUN_MANIFEST_FILE
    recon_path = out / tio.RECON_DIR
    existing = [entry for entry in (results_path, manifest_path, recon_path) if entry.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"output exists in {out}: {[e.name for e in existing]}")
    if overwrite:
        for entry in existing:
            if entry.is_dir() and not entry.is_symlink():
                shutil.rmtree(entry)
            else:
                entry.unlink()
    grid = make_grid(
        np.asarray(grid_center, dtype=np.float64) if grid_center is not None else data.target,
        np.asarray(grid_half_size, dtype=np.float64)
        if grid_half_size is not None
        else suite_obj.grid_half_size,
        float(grid_spacing) if grid_spacing is not None else suite_obj.grid_spacing,
    )
    ground_truth = tio.find_ground_truth(data, gt_path)
    geom = data.geom
    num_views, num_bs = geom.num_views, geom.num_bs
    period = float(geom.delay_period)
    ref = (int(nested_view_order(num_views, int(dataset_seed))[0]), 0)
    los_fallback = False
    try:
        p_ref, c_ref = sync_mod.reference_power(data.y_clean, data.los_visible)
    except ValueError:
        p_ref, c_ref = sync_mod.reference_power(
            data.y_clean, np.ones((num_views, num_bs), dtype=bool)
        )
        los_fallback = True
    collected: list[dict[str, Any]] = []
    noise_entries: list[dict[str, Any]] = []
    started = time.perf_counter()
    handle: TextIO = open(results_path, "w", encoding="utf-8")  # noqa: PTH123
    try:
        with contextlib.ExitStack() as stack:
            pool = (
                stack.enter_context(
                    ProcessPoolExecutor(
                        max_workers=workers, mp_context=multiprocessing.get_context("spawn")
                    )
                )
                if workers > 1
                else None
            )
            for realization in range(suite_obj.realizations):
                seeds = sync_mod.TrackSeeds(int(dataset_seed), int(realization))
                bundle, truth_gt = sync_mod.make_tracks(
                    data.y_clean,
                    suite_obj.snr_db,
                    float(p_ref),
                    seeds,
                    freq_offsets=geom.freq_offsets,
                    sigma_t=suite_obj.sigma_t,
                    ref=ref,
                    c_ref=c_ref,
                )
                noise_var = float(truth_gt["sigma2"])
                noise_entries.append(
                    {
                        "realization": int(realization),
                        "sigma2": float(truth_gt["sigma2"]),
                        "p_ref": float(truth_gt["p_ref"]),
                        "c_ref": list(truth_gt["c_ref"]) if truth_gt["c_ref"] is not None else None,
                        "snr_db": float(suite_obj.snr_db),
                        "sigma_t": suite_obj.sigma_t,
                        "los_fallback": bool(los_fallback),
                    }
                )
                jobs: list[tuple[Config, str, str, list[Any]]] = []
                for cfg in config_list:
                    track = config_track(cfg, track_list)
                    if track is None:
                        continue
                    signal = bundle[track]
                    truth = truth_gt["gauges"][track]
                    for space in space_list:
                        wanted = job_strategies(cfg)
                        if strategy_filter is not None:
                            wanted = tuple(s for s in wanted if s in strategy_filter)
                        if not wanted:
                            continue
                        futures = [
                            _submit(
                                pool,
                                cfg.name,
                                track,
                                space,
                                strategy,
                                signal,
                                truth,
                                data,
                                grid,
                                ground_truth,
                                geom,
                                suite_obj,
                                ref,
                                noise_var,
                                period,
                                realization,
                                out,
                            )
                            for strategy in wanted
                        ]
                        jobs.append((cfg, track, space, futures))
                        if pool is None:
                            _drain_job(jobs.pop(), handle, collected, data, realization)
                for job in jobs:
                    _drain_job(job, handle, collected, data, realization)
    finally:
        handle.close()
    runtime_s = time.perf_counter() - started
    counts = {"ok": 0, "n/a": 0, "error": 0}
    for row in collected:
        counts[str(row["status"])] += 1
    manifest = _run_manifest(
        data,
        suite_obj,
        grid,
        track_list,
        config_list,
        strategy_filter,
        tracks,
        configs,
        spaces,
        gt_path,
        grid_center,
        grid_half_size,
        grid_spacing,
        int(dataset_seed),
        ref,
        noise_entries,
        ground_truth,
        results_path,
        collected,
        counts,
        runtime_s,
    )
    tio.write_run_manifest(manifest_path, manifest)
    return BenchmarkRun(
        out_dir=out,
        results_path=results_path,
        run_manifest_path=manifest_path,
        rows=tuple(collected),
    )


def _drain_job(
    job: tuple[Config, str, str, list[Any]],
    handle: TextIO,
    collected: list[dict[str, Any]],
    data: tio.TomographyDataset,
    realization: int,
) -> None:
    """Write the chain rows of one job (in strategy order), then its planned rows."""
    cfg, track, space, futures = job
    for future in futures:
        chain_rows = future if isinstance(future, list) else future.result()
        for row in chain_rows:
            tio.write_result_row(handle, row)
            collected.append(tio.to_jsonable(row))
    for entry in cfg.planned:
        row = _planned_row(cfg, entry, track, space, data, realization)
        tio.write_result_row(handle, row)
        collected.append(tio.to_jsonable(row))


def _submit(pool: ProcessPoolExecutor | None, *args: Any) -> Any:
    """Run one chain now (no pool, returns its rows) or submit it (returns a future)."""
    if pool is None:
        return _run_chain(*args)
    return pool.submit(_run_chain, *args)


def _common_recon(
    cfg: Config,
    track: str,
    space: str,
    strategy: str,
    geom: CaptureGeometry,
    grid: VoxelGrid,
    gauges_used: tuple[np.ndarray, np.ndarray] | None,
) -> dict[str, Any]:
    """Return the recon arrays shared by every ok row."""
    if gauges_used is None:
        paired = np.full((geom.num_views, geom.num_bs, 2), np.nan, dtype=np.float64)
    else:
        paired = np.stack(
            [
                np.asarray(gauges_used[0], dtype=np.float64),
                np.asarray(gauges_used[1], dtype=np.float64),
            ],
            axis=-1,
        )
    return {
        "grid_origin": np.asarray(grid.origin, dtype=np.float64),
        "grid_spacing": float(grid.spacing),
        "grid_shape": np.asarray(grid.shape, dtype=np.int64),
        "gauges": paired,
        "space": str(space),
        "bs_index": np.arange(geom.num_bs, dtype=np.int64),
        "config": str(cfg.name),
        "track": str(track),
        "strategy": str(strategy),
    }


def _base_row(
    cfg: Config,
    track: str,
    space: str,
    strategy: str,
    data: tio.TomographyDataset,
    realization: int,
) -> dict[str, Any]:
    """Return the config/scene fields shared by every result row."""
    return {
        "schema": tio.RESULT_SCHEMA,
        "scene": data.name,
        "realization": int(realization),
        "config": cfg.name,
        "node": cfg.node,
        "subset": cfg.subset,
        "sync": cfg.sync,
        "column": cfg.column,
        "lattices": sorted(cfg.lattices),
        "budget": cfg.budget,
        "track": track,
        "space": space,
        "strategy": strategy,
    }


def _metrics_for(
    detections: np.ndarray,
    scores: np.ndarray,
    ground_truth: tio.GroundTruth | None,
    space: str,
    grid: VoxelGrid,
    geom: CaptureGeometry,
) -> dict[str, Any] | None:
    """Return detection metrics of one ok row (None without GT in ``space``)."""
    if ground_truth is None:
        return None
    gt_points = ground_truth.positions(space)
    if gt_points is None:
        return None
    return detection_metrics(detections, scores, gt_points, grid, geom.ue_pos.mean(axis=0))


def _emit_row(
    sink: list[dict[str, Any]],
    partial: dict[str, Any],
    stage: str,
    solver: str,
    status: str,
    reason: str | None,
    n_iter: int | None,
    hyper: Mapping[str, float],
    n_detections: int,
    metrics_payload: dict[str, Any] | None,
    gauge_payload: dict[str, Any] | None,
    ill_payload: dict[str, Any] | None,
    runtime_s: float,
    recon: str | None,
) -> None:
    """Assemble, validate and append one result row in ``RESULT_KEYS`` order."""
    row = dict(partial)
    row.update(
        {
            "stage": stage,
            "solver": solver,
            "status": status,
            "reason": reason,
            "n_iter": n_iter,
            "hyper": dict(hyper),
            "n_detections": int(n_detections),
            "metrics": metrics_payload,
            "gauge_errors": gauge_payload,
            "ill_posed": ill_payload,
            "runtime_s": float(runtime_s),
            "recon": recon,
        }
    )
    ordered = {key: row[key] for key in tio.RESULT_KEYS}
    tio.validate_result_row(ordered)
    sink.append(tio.to_jsonable(ordered))


def _planned_row(
    cfg: Config,
    entry: str,
    track: str,
    space: str,
    data: tio.TomographyDataset,
    realization: int,
) -> dict[str, Any]:
    """Return one ``planned`` row (strategy ``none``, status ``n/a``)."""
    row = _base_row(cfg, track, space, "none", data, realization)
    ordered = {key: row[key] for key in tio.RESULT_KEYS[:13]}
    ordered.update(
        {
            "stage": "planned",
            "solver": str(entry),
            "status": "n/a",
            "reason": str(entry),
            "n_iter": None,
            "hyper": {},
            "n_detections": 0,
            "metrics": None,
            "gauge_errors": None,
            "ill_posed": None,
            "runtime_s": 0.0,
            "recon": None,
        }
    )
    return ordered


def _run_chain(  # noqa: PLR0913
    cfg: Config | str,
    track: str,
    space: str,
    strategy: str,
    signal: np.ndarray,
    truth: tuple[np.ndarray, np.ndarray],
    data: tio.TomographyDataset,
    grid: VoxelGrid,
    ground_truth: tio.GroundTruth | None,
    geom: CaptureGeometry,
    suite_obj: Suite,
    ref: tuple[int, int],
    noise_var: float,
    period: float,
    realization: int,
    out: Path,
) -> list[dict[str, Any]]:
    """Run one strategy chain and return its E1/ROI/E2 rows in order.

    ``cfg`` may be a config name (resolved via :func:`get_config`, which keeps
    the process-pool payload picklable).
    """
    resolved = get_config(cfg) if isinstance(cfg, str) else cfg
    cfg = resolved
    sink: list[dict[str, Any]] = []
    num_views, num_bs = geom.num_views, geom.num_bs
    partial = _base_row(cfg, track, space, strategy, data, realization)
    zeros = (
        np.zeros((num_views, num_bs), dtype=np.float64),
        np.zeros((num_views, num_bs), dtype=np.float64),
    )
    gauge_start = time.perf_counter()
    try:
        if strategy == "none":
            gauges_used: tuple[np.ndarray, np.ndarray] | None = None
            estimates: tuple[str, ...] = ()
        elif strategy == "oracle":
            phi_true = np.asarray(truth[0], dtype=np.float64)
            tau_true = np.asarray(truth[1], dtype=np.float64)
            gauges_used = (
                phi_true if "phi" in cfg.gauge_unknowns else zeros[0].copy(),
                tau_true if "tau" in cfg.gauge_unknowns else zeros[1].copy(),
            )
            estimates = ("phi", "tau")
        else:
            estimated = estimate_gauges(
                cfg,
                strategy,
                signal,
                geom,
                grid,
                space,
                noise_var=noise_var,
                ref=ref,
                n_iter=suite_obj.e2_iterations,
                sigma_t=suite_obj.sigma_t,
            )
            gauges_used = used_gauges(cfg, estimated)
            estimates = tuple(estimated.estimates)
    except Exception as error:  # noqa: BLE001
        reason = f"gauges: {type(error).__name__}: {error}"
        _chain_gauge_failure(cfg, track, space, strategy, partial, out, sink, reason)
        return sink
    gauge_runtime = time.perf_counter() - gauge_start
    gauge_payload: dict[str, Any] | None = None
    if cfg.gauge_unknowns:
        gauge_payload = gauge_error_summary(cfg, gauges_used or zeros, truth, period, estimates)
        gauge_payload["runtime_s"] = gauge_runtime
    if cfg.e1 is None:
        reason = "; ".join(cfg.planned) if cfg.planned else "no E1 step"
        _emit_row(
            sink,
            partial,
            "E1",
            "-",
            "n/a",
            reason,
            None,
            {},
            0,
            None,
            gauge_payload,
            None,
            0.0,
            None,
        )
        return sink
    if space not in cfg.e1.spaces:
        _emit_row(
            sink,
            partial,
            "E1",
            cfg.e1.name,
            "n/a",
            f"space {space} not supported",
            None,
            {},
            0,
            None,
            gauge_payload,
            None,
            0.0,
            None,
        )
        return sink
    e1_start = time.perf_counter()
    try:
        estimate = run_e1(cfg, signal, geom, grid, space, gauges=gauges_used, noise_var=noise_var)
        detections, scores = detect(estimate, grid, suite_obj.max_peaks)
    except Exception as error:  # noqa: BLE001
        _emit_row(
            sink,
            partial,
            "E1",
            cfg.e1.name,
            "error",
            f"{type(error).__name__}: {error}",
            None,
            {},
            0,
            None,
            gauge_payload,
            None,
            time.perf_counter() - e1_start,
            None,
        )
        _chain_e1_failure(cfg, partial, sink, gauge_payload)
        return sink
    e1_runtime = time.perf_counter() - e1_start
    ill_start = time.perf_counter()
    try:
        report = ill_posed_at(
            cfg,
            signal,
            geom,
            detections[: suite_obj.ill_max_points],
            space,
            noise_var=noise_var,
            gauges=gauges_used,
        )
        ill_payload: dict[str, Any] | None = _ill_payload(report)
        ill_payload["runtime_s"] = time.perf_counter() - ill_start
    except Exception as error:  # noqa: BLE001
        ill_payload = {"error": f"{type(error).__name__}: {error}"}
    metrics_payload = _metrics_for(detections, scores, ground_truth, space, grid, geom)
    recon_arrays: dict[str, Any] = {
        "map": np.asarray(estimate, dtype=np.float64),
        "detections": detections,
        "scores": scores,
    }
    recon_arrays.update(_common_recon(cfg, track, space, strategy, geom, grid, gauges_used))
    relpath = tio.recon_relpath(
        data.name, cfg.name, "E1", cfg.e1.name, track, space, strategy, realization
    )
    tio.write_recon(out, relpath, recon_arrays)
    _emit_row(
        sink,
        partial,
        "E1",
        cfg.e1.name,
        "ok",
        None,
        None,
        cfg.e1.defaults(),
        int(detections.shape[0]),
        metrics_payload,
        gauge_payload,
        ill_payload,
        e1_runtime,
        relpath,
    )
    if cfg.roi is not None:
        _run_roi_row(
            cfg,
            track,
            space,
            strategy,
            signal,
            partial,
            data,
            grid,
            ground_truth,
            geom,
            suite_obj,
            noise_var,
            gauges_used,
            detections,
            scores,
            gauge_payload,
            ill_payload,
            realization,
            out,
            sink,
        )
    if cfg.e2:
        _run_e2_rows(
            cfg,
            track,
            space,
            strategy,
            signal,
            estimate,
            partial,
            data,
            grid,
            ground_truth,
            geom,
            suite_obj,
            noise_var,
            gauges_used,
            gauge_payload,
            ill_payload,
            realization,
            out,
            sink,
        )
    return sink


def _chain_gauge_failure(
    cfg: Config,
    track: str,
    space: str,
    strategy: str,
    partial: dict[str, Any],
    out: Path,
    sink: list[dict[str, Any]],
    reason: str,
) -> None:
    """Emit every stage row of a chain whose gauge estimation raised."""
    del out
    if cfg.e1 is not None:
        _emit_row(
            sink,
            partial,
            "E1",
            cfg.e1.name,
            "error",
            reason,
            None,
            {},
            0,
            None,
            None,
            None,
            0.0,
            None,
        )
    else:
        fail_reason = "; ".join(cfg.planned) if cfg.planned else "no E1 step"
        _emit_row(
            sink,
            partial,
            "E1",
            "-",
            "n/a",
            fail_reason,
            None,
            {},
            0,
            None,
            None,
            None,
            0.0,
            None,
        )
        return
    if cfg.roi is not None:
        _emit_row(
            sink,
            partial,
            "ROI",
            cfg.roi.name,
            "error",
            reason,
            None,
            {},
            0,
            None,
            None,
            None,
            0.0,
            None,
        )
    for step in cfg.e2:
        _emit_row(
            sink,
            partial,
            "E2",
            step.name,
            "error",
            reason,
            None,
            {},
            0,
            None,
            None,
            None,
            0.0,
            None,
        )


def _chain_e1_failure(
    cfg: Config,
    partial: dict[str, Any],
    sink: list[dict[str, Any]],
    gauge_payload: dict[str, Any] | None,
) -> None:
    """Emit ROI/E2 rows as errors after the E1 stage of the chain failed."""
    if cfg.roi is not None:
        _emit_row(
            sink,
            partial,
            "ROI",
            cfg.roi.name,
            "error",
            "E1 failed",
            None,
            {},
            0,
            None,
            gauge_payload,
            None,
            0.0,
            None,
        )
    for step in cfg.e2:
        _emit_row(
            sink,
            partial,
            "E2",
            step.name,
            "error",
            "E1 failed",
            None,
            {},
            0,
            None,
            gauge_payload,
            None,
            0.0,
            None,
        )


def _run_roi_row(  # noqa: PLR0913
    cfg: Config,
    track: str,
    space: str,
    strategy: str,
    signal: np.ndarray,
    partial: dict[str, Any],
    data: tio.TomographyDataset,
    grid: VoxelGrid,
    ground_truth: tio.GroundTruth | None,
    geom: CaptureGeometry,
    suite_obj: Suite,
    noise_var: float,
    gauges_used: tuple[np.ndarray, np.ndarray] | None,
    detections: np.ndarray,
    scores: np.ndarray,
    gauge_payload: dict[str, Any] | None,
    ill_payload: dict[str, Any] | None,
    realization: int,
    out: Path,
    sink: list[dict[str, Any]],
) -> None:
    """Run the ROI stage of one chain and append its row."""
    del scores
    assert cfg.roi is not None
    if space not in cfg.roi.spaces:
        _emit_row(
            sink,
            partial,
            "ROI",
            cfg.roi.name,
            "n/a",
            f"space {space} not supported",
            None,
            {},
            0,
            None,
            gauge_payload,
            ill_payload,
            0.0,
            None,
        )
        return
    if detections.shape[0] == 0:
        _emit_row(
            sink,
            partial,
            "ROI",
            cfg.roi.name,
            "n/a",
            "no detections",
            None,
            {},
            0,
            None,
            gauge_payload,
            ill_payload,
            0.0,
            None,
        )
        return
    started = time.perf_counter()
    try:
        result = run_roi(
            cfg,
            signal,
            geom,
            detections[: suite_obj.roi_max],
            space,
            gauges=gauges_used,
            noise_var=noise_var,
        )
        order = np.argsort(-np.asarray(result.values, dtype=np.float64), kind="stable")
        refined = np.asarray(result.positions, dtype=np.float64)[order]
        refined_scores = np.asarray(result.values, dtype=np.float64)[order]
    except Exception as error:  # noqa: BLE001
        _emit_row(
            sink,
            partial,
            "ROI",
            cfg.roi.name,
            "error",
            f"{type(error).__name__}: {error}",
            None,
            {},
            0,
            None,
            gauge_payload,
            ill_payload,
            time.perf_counter() - started,
            None,
        )
        return
    runtime = time.perf_counter() - started
    metrics_payload = _metrics_for(refined, refined_scores, ground_truth, space, grid, geom)
    recon_arrays: dict[str, Any] = {
        "detections": refined,
        "scores": refined_scores,
        "centers": np.asarray(result.centers, dtype=np.float64)[order],
    }
    recon_arrays.update(_common_recon(cfg, track, space, strategy, geom, grid, gauges_used))
    relpath = tio.recon_relpath(
        data.name, cfg.name, "ROI", cfg.roi.name, track, space, strategy, realization
    )
    tio.write_recon(out, relpath, recon_arrays)
    _emit_row(
        sink,
        partial,
        "ROI",
        cfg.roi.name,
        "ok",
        None,
        None,
        {},
        int(refined.shape[0]),
        metrics_payload,
        gauge_payload,
        ill_payload,
        runtime,
        relpath,
    )


def _run_e2_rows(  # noqa: PLR0913
    cfg: Config,
    track: str,
    space: str,
    strategy: str,
    signal: np.ndarray,
    estimate: np.ndarray,
    partial: dict[str, Any],
    data: tio.TomographyDataset,
    grid: VoxelGrid,
    ground_truth: tio.GroundTruth | None,
    geom: CaptureGeometry,
    suite_obj: Suite,
    noise_var: float,
    gauges_used: tuple[np.ndarray, np.ndarray] | None,
    gauge_payload: dict[str, Any] | None,
    ill_payload: dict[str, Any] | None,
    realization: int,
    out: Path,
    sink: list[dict[str, Any]],
) -> None:
    """Run the support and every E2 step of one chain and append their rows."""
    try:
        indices, support, edges = run_support(estimate, grid)
        support_error: str | None = None
    except Exception as error:  # noqa: BLE001
        indices, support, edges = (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 2), dtype=np.int64),
        )
        support_error = f"{type(error).__name__}: {error}"
    for step in cfg.e2:
        if space not in step.spaces:
            _emit_row(
                sink,
                partial,
                "E2",
                step.name,
                "n/a",
                f"space {space} not supported",
                None,
                {},
                0,
                None,
                gauge_payload,
                ill_payload,
                0.0,
                None,
            )
            continue
        if support_error is not None:
            _emit_row(
                sink,
                partial,
                "E2",
                step.name,
                "error",
                support_error,
                None,
                {},
                0,
                None,
                gauge_payload,
                ill_payload,
                0.0,
                None,
            )
            continue
        if support.shape[0] == 0:
            _emit_row(
                sink,
                partial,
                "E2",
                step.name,
                "n/a",
                "empty support",
                None,
                {},
                0,
                None,
                gauge_payload,
                ill_payload,
                0.0,
                None,
            )
            continue
        started = time.perf_counter()
        try:
            solved: E2Output = run_e2(
                cfg,
                step.name,
                signal,
                geom,
                support,
                space,
                gauges=gauges_used,
                noise_var=noise_var,
                n_iter=suite_obj.e2_iterations,
                edges=edges,
            )
            density_map = power_mod.support_to_map(solved.density, grid, indices)
            detections, scores = detect(density_map, grid, suite_obj.max_peaks)
        except Exception as error:  # noqa: BLE001
            _emit_row(
                sink,
                partial,
                "E2",
                step.name,
                "error",
                f"{type(error).__name__}: {error}",
                None,
                {},
                0,
                None,
                gauge_payload,
                ill_payload,
                time.perf_counter() - started,
                None,
            )
            continue
        runtime = time.perf_counter() - started
        metrics_payload = _metrics_for(detections, scores, ground_truth, space, grid, geom)
        recon_arrays: dict[str, Any] = {
            "map": np.asarray(density_map, dtype=np.float64),
            "density": np.asarray(solved.density, dtype=np.float64),
            "support_indices": np.asarray(indices, dtype=np.int64),
            "detections": detections,
            "scores": scores,
        }
        recon_arrays.update(_common_recon(cfg, track, space, strategy, geom, grid, gauges_used))
        relpath = tio.recon_relpath(
            data.name, cfg.name, "E2", step.name, track, space, strategy, realization
        )
        tio.write_recon(out, relpath, recon_arrays)
        _emit_row(
            sink,
            partial,
            "E2",
            step.name,
            "ok",
            None,
            int(suite_obj.e2_iterations),
            step.defaults(),
            int(detections.shape[0]),
            metrics_payload,
            gauge_payload,
            ill_payload,
            runtime,
            relpath,
        )


def _run_manifest(  # noqa: PLR0913
    data: tio.TomographyDataset,
    suite_obj: Suite,
    grid: VoxelGrid,
    track_list: list[str],
    config_list: list[Config],
    strategy_filter: list[str] | None,
    tracks_arg: Sequence[str] | None,
    configs_arg: Sequence[str] | None,
    spaces_arg: Sequence[str] | None,
    gt_path: Path | str | None,
    grid_center: Sequence[float] | np.ndarray | None,
    grid_half_size: Sequence[float] | np.ndarray | None,
    grid_spacing: float | None,
    dataset_seed: int,
    ref: tuple[int, int],
    noise_entries: list[dict[str, Any]],
    ground_truth: tio.GroundTruth | None,
    results_path: Path,
    collected: list[dict[str, Any]],
    counts: dict[str, int],
    runtime_s: float,
) -> dict[str, Any]:
    """Assemble the run-manifest payload of a finished benchmark."""
    manifest = data.manifest
    config = manifest.config
    if ground_truth is None:
        gt_entry = None
    else:
        gt_entry = {
            "path": None if ground_truth.path is None else str(ground_truth.path),
            "sha256": ground_truth.sha256,
            "points_space": ground_truth.points_space,
            "num_points": 0
            if ground_truth.points_pos is None
            else int(ground_truth.points_pos.shape[0]),
            "num_vs": 0 if ground_truth.vs_pos is None else int(ground_truth.vs_pos.shape[0]),
        }
    payload = {
        "schema": tio.RUN_MANIFEST_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "options": {
            "suite": suite_obj.name,
            "tracks": None if tracks_arg is None else list(tracks_arg),
            "configs": None if configs_arg is None else list(configs_arg),
            "strategies": None if strategy_filter is None else list(strategy_filter),
            "spaces": None if spaces_arg is None else list(spaces_arg),
            "gt_path": None if gt_path is None else str(gt_path),
            "grid_center": None
            if grid_center is None
            else np.asarray(grid_center, dtype=np.float64).tolist(),
            "grid_half_size": None
            if grid_half_size is None
            else np.asarray(grid_half_size, dtype=np.float64).tolist(),
            "grid_spacing": None if grid_spacing is None else float(grid_spacing),
            "dataset_seed": int(dataset_seed),
        },
        "versions": tio.software_versions(),
        "dataset": {
            "path": str(manifest.manifest_path),
            "name": data.name,
            "schema_version": int(manifest.schema_version),
            "num_views": int(manifest.num_views),
            "num_bs": int(manifest.num_bs),
            "num_bins": int(manifest.num_frequency_bins),
            "aperture_shape": [int(manifest.rx_rows), int(manifest.rx_cols)],
            "carrier_frequency_hz": float(manifest.carrier_frequency_hz),
            "bandwidth_hz": float(config["bandwidth_hz"]),
            "tx_pattern": data.tx_pattern,
            "los_visible_source": data.los_visible_source,
            "los_visible": np.asarray(data.los_visible, dtype=bool).tolist(),
            "hashes": dict(data.hashes),
        },
        "suite": dataclasses.asdict(suite_obj),
        "grid": {
            "origin": np.asarray(grid.origin, dtype=np.float64).tolist(),
            "spacing": float(grid.spacing),
            "shape": [int(value) for value in grid.shape],
        },
        "tracks": list(track_list),
        "configs": [cfg.name for cfg in config_list],
        "strategies": None if strategy_filter is None else list(strategy_filter),
        "seeds": {
            "dataset_seed": int(dataset_seed),
            "realizations": list(range(suite_obj.realizations)),
            "ref": [int(ref[0]), int(ref[1])],
        },
        "noise": [dict(entry) for entry in noise_entries],
        "gt": gt_entry,
        "results": {
            "path": tio.RESULTS_FILE,
            "sha256": tio.sha256_file(results_path),
            "rows": int(len(collected)),
            "status_counts": dict(counts),
        },
        "runtime_s": float(runtime_s),
    }
    return payload
