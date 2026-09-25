"""E1 back-projection solvers for RF tomography (NumPy/SciPy only).

This module is the E1 tier of design §4.1: one map per information subset, all
built from the Phase-0 E1 fast back-projection (:mod:`~plateau_rt.domain.
rf_tomography.backproject`) and the incoherent kernels
(:mod:`~plateau_rt.domain.rf_tomography.kernels`), never from a statistic
outside the node itself. All lengths are SI (metres, seconds, hertz); arrays are
float64/complex128. Maps are relative (arbitrary but consistent units) and are
exactly 0 at singular points (``forward_exact._singular_mask``): a point within
``LOS_VS_TOLERANCE_M`` of a UE, or of a BS in ``"bv"``.

Node to function
----------------
=================  ==========================================================
node               function
=================  ==========================================================
``I``, ``I_n0``    :func:`intensity_map` (cone BP + log-mean fusion)
``ID``             :func:`power_map` (angle-delay BP)
``D``              :func:`splat_returns` (return-to-point occupancy log-odds)
``P``, ``IP``      :func:`envelope_map` / :func:`coherent_map` (DC bin)
``DP``, ``IDP``    :func:`envelope_map` / :func:`coherent_map`
``P_W``, ``IP_W``  :func:`envelope_map` (per-bin incoherent)
=================  ==========================================================

GLRT normalisations
-------------------
The raw adjoint ``K^T y`` of a native-sampled power observable is biased toward
the native-grid cells (picket fence): on the micro scene it peaks 0.7-1.7 m away
for a noiseless point, while the normalised statistic peaks within 0.33 m. Every
map here is therefore GLRT-normalised.

* power: ``k(x)^T y / ||k(x)||``, with ``k`` the column of the per-capture power
  kernel and ``||k(x)||`` in closed form (:func:`power_column_norm`);
* envelope: ``|bp_c(x)|^2 / |gamma_c(x)|^2`` per capture (the GLRT for one
  unknown unit-modulus phase), summed over ``(v, b)`` (:func:`log_mean_fusion`
  is used only by :func:`intensity_map` and the blind-search consensus);
* coherent: ``|sum_c bp_c(x)|^2 / sum_c |gamma_c(x)|^2``, the cross-view GLRT.

Gauge and ``tau_hat``
---------------------
The Phase-0 gauge is ``Y_obs[c] = exp(1j phi_c) exp(-2j pi df_n tau_c) Y[c]``,
so a point of capture ``c`` at delay ``tau(x)`` appears at observed delay
``tau(x) + tau_c``. Passing ``tau_hat = tau_c`` to the back-projection samples
``c(u, tau(x) + tau_hat_c)`` and compensates it. ``tau_hat=None`` means zeros.
The envelope and coherent maps are invariant to ``phi_c`` by construction; all
maps are invariant to ``tau_c`` once ``tau_hat = tau_c`` is passed.

Sync invariance
---------------
``I``, ``I_n0``, ``P``, ``IP`` and the ``P_W``/``IP_W`` envelopes are
sync-invariant: identical in S and N. ``P``/``IP`` live at the DC bin, where the
delay gauge has no effect; ``I``/``I_n0`` are angle-only; ``P_W``/``IP_W`` remove
one phase per ``(capture, bin)`` before their per-bin incoherent sum.

Coherent maps (:func:`coherent_map`, and the coherent branch of
:func:`roi_refine`) are only meaningful on grids finer than ``lambda/4``
(design §3.5): their PSF has many near-equal grating lobes, so they are used in
ROI windows only and polished by phase extrapolation (:func:`roi_refine`).

NumPy/SciPy only: nothing here may import Sionna, Mitsuba or Dr.Jit.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from plateau_rt.domain.rf_tomography.backproject import (
    _aperture_spacing,
    _validate_y,
    apply_lookup,
    backproject,
    capture_lookup,
    capture_volume,
)
from plateau_rt.domain.rf_tomography.forward_exact import (
    SPACES,
    _prepare_points,
    _singular_mask,
    capture_factors,
)
from plateau_rt.domain.rf_tomography.geometry import CaptureGeometry, VoxelGrid
from plateau_rt.domain.rf_tomography.kernels import (
    _PRODUCT_AXES,
    _capture_tau,
    _product_const,
    dirichlet_power,
    noise_floor,
    power_backproject_grid,
)
from plateau_rt.domain.rf_tomography.metrics import nms_peaks
from plateau_rt.domain.rf_tomography.observables import DEFAULT_MASK_K, extract

COHERENT_NODES: tuple[str, ...] = ("P", "DP", "IP", "IDP")
PER_BIN_NODES: tuple[str, ...] = ("P_W", "IP_W")
POWER_NODES: tuple[str, ...] = ("I", "I_n0", "ID")
E1_NODES: tuple[str, ...] = (
    "I",
    "I_n0",
    "D",
    "P",
    "P_W",
    "DP",
    "ID",
    "IP",
    "IP_W",
    "IDP",
)
DEFAULT_FLOOR: float = 1e-3
DEFAULT_ROI_HALF_WIDTH: float = 0.25
DEFAULT_CANDIDATE_FRACTION: float = 0.5
DEFAULT_POLISH_STEPS: int = 3

_DATA_NODES: tuple[str, ...] = COHERENT_NODES + PER_BIN_NODES
_SPLAT_CHUNK: int = 4096
_POLISH_CHUNK: int = 256


def _check_space(space: str) -> None:
    """Raise ``ValueError`` unless ``space`` is one of :data:`SPACES`."""
    if space not in SPACES:
        raise ValueError(f"space must be one of {SPACES}, got {space!r}")


def _check_grid(grid: VoxelGrid) -> None:
    """Raise ``ValueError`` unless ``grid`` is a :class:`VoxelGrid`."""
    if not isinstance(grid, VoxelGrid):
        raise ValueError("grid must be a VoxelGrid")


def _check_node(node: str, allowed: tuple[str, ...], name: str) -> None:
    """Raise ``ValueError`` unless ``node`` is one of ``allowed``."""
    if node not in allowed:
        raise ValueError(f"unsupported node {node!r} for {name}; expected one of {allowed}")


def _check_noise_var(noise_var: float | None) -> None:
    """Raise ``ValueError`` unless ``noise_var`` is None or finite and >= 0."""
    if noise_var is None:
        return
    value = float(noise_var)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("noise_var must be finite and >= 0")


def _check_floor(floor: float) -> float:
    """Return ``floor`` as a finite positive float or raise ``ValueError``."""
    value = float(floor)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("floor must be finite and > 0")
    return value


def _as_raw_y(Y: np.ndarray) -> np.ndarray:
    """Return rank-6 complex128 ``Y`` (the raw capture layout)."""
    arr = np.asarray(Y, dtype=np.complex128)
    if arr.ndim != 6:
        raise ValueError("Y must have shape [V, B, H, R, C, N]")
    return arr


def capture_weights(
    geom: CaptureGeometry,
    points: np.ndarray,
    space: str,
    *,
    pattern: str = "tr38901",
    polarization: str = "none",
) -> np.ndarray:
    """Return ``|gamma_c(x)|^2`` as float64 ``[V, B, P]`` (0 at singular points)."""
    _check_space(space)
    pts = _prepare_points(points)
    singular = _singular_mask(pts, geom, space)
    ok = ~singular
    num_points = pts.shape[0]
    out = np.zeros((geom.num_views, geom.num_bs, num_points), dtype=np.float64)
    if not np.any(ok):
        return out
    for v in range(geom.num_views):
        for b in range(geom.num_bs):
            factors = capture_factors(
                pts[ok], geom, space, v, b, pattern=pattern, polarization=polarization
            )
            out[v, b, ok] = np.abs(factors.gamma) ** 2
    return out


def node_data(
    Y: np.ndarray,
    node: str,
    *,
    noise_var: float | None = None,
    mask_k: float = DEFAULT_MASK_K,
) -> np.ndarray:
    """Return the raw-layout complex128 ``[V, B, H, R, C, N]`` data of ``node``.

    Built only from :func:`observables.extract`, so no statistic outside the node
    leaks in. ``"IDP"`` copies ``Y``; ``"IP"``/``"P"`` are zero outside the DC
    bin ``n0``; PHAT nodes propagate ``extract``'s ``ValueError`` when
    ``noise_var`` is required and missing.
    """
    _check_node(node, _DATA_NODES, "node_data")
    arr = _as_raw_y(Y)
    if node == "IDP":
        return arr.copy()
    if node in ("DP", "P_W"):
        params: dict[str, float] = {"mask_k": float(mask_k)}
        if noise_var is not None:
            params["noise_var"] = float(noise_var)
        return extract(arr, node, params).data
    if node == "IP_W":
        return extract(arr, "IP_W").data
    n0 = arr.shape[-1] // 2
    out = np.zeros_like(arr, dtype=np.complex128)
    if node == "IP":
        out[..., n0] = extract(arr, "IP").data
    else:  # node == "P"
        out[..., n0] = extract(arr, "P", {"noise_var": noise_var, "mask_k": float(mask_k)}).data
    return out


def _dirichlet_axis_energy(xi: np.ndarray, length: int) -> np.ndarray:
    """Return ``sum_i dirichlet_power(xi - i / length, length)**2`` in closed form."""
    ell = float(length)
    values = np.asarray(xi, dtype=np.float64)
    return ((2.0 * ell * ell + 1.0) + (ell * ell - 1.0) * np.cos(2.0 * np.pi * ell * values)) / 3.0


def power_column_norm(
    geom: CaptureGeometry,
    points: np.ndarray,
    space: str,
    node: str = "ID",
    *,
    tau_hat: np.ndarray | None = None,
    pattern: str = "tr38901",
    polarization: str = "none",
) -> np.ndarray:
    """Return the float64 ``[V, B, P]`` norm of each ``PowerOperator`` column.

    The column norm factorises over the axes that carry a Dirichlet kernel, so it
    has the closed form ``const * |gamma|**2 * sqrt(prod_axis E_L)`` with
    ``E_L`` from :func:`_dirichlet_axis_energy`. 0 at singular points.
    """
    _check_space(space)
    _check_node(node, POWER_NODES, "power_column_norm")
    pts = _prepare_points(points)
    delays = _capture_tau(tau_hat, geom.num_views, geom.num_bs)
    singular = _singular_mask(pts, geom, space)
    ok = ~singular
    rows, cols = geom.aperture_shape
    num_bins = geom.num_bins
    spacing = _aperture_spacing(geom)
    const = _product_const(node, geom)
    use_y, use_z, use_t = _PRODUCT_AXES[node]

    out = np.zeros((geom.num_views, geom.num_bs, pts.shape[0]), dtype=np.float64)
    if not np.any(ok):
        return out
    for v in range(geom.num_views):
        for b in range(geom.num_bs):
            factors = capture_factors(
                pts[ok], geom, space, v, b, pattern=pattern, polarization=polarization
            )
            combined = np.ones(factors.gamma.shape[0], dtype=np.float64)
            if use_y:
                combined = combined * _dirichlet_axis_energy(spacing * factors.u_local[:, 1], cols)
            if use_z:
                combined = combined * _dirichlet_axis_energy(spacing * factors.u_local[:, 2], rows)
            if use_t:
                combined = combined * _dirichlet_axis_energy(
                    geom.delta_f * (factors.tau + delays[v, b]), num_bins
                )
            out[v, b, ok] = const * np.abs(factors.gamma) ** 2 * np.sqrt(combined)
    return out


def power_map(
    Y: np.ndarray,
    geom: CaptureGeometry,
    grid: VoxelGrid,
    space: str,
    node: str = "ID",
    *,
    tau_hat: np.ndarray | None = None,
    noise_var: float | None = None,
    kind: str = "trilinear",
    oversample: int = 8,
    pattern: str = "tr38901",
    polarization: str = "none",
    per_capture: bool = False,
) -> np.ndarray:
    """Return the GLRT-normalised power back-projection of ``node``.

    ``y`` is the node data minus :func:`kernels.noise_floor` when ``noise_var``
    is given, and ``KTy`` is its adjoint via
    :func:`kernels.power_backproject_grid`. ``per_capture=False`` divides by the
    norm of the summed per-capture kernel (the single-point GLRT
    ``k(x)^T y / ||k(x)||``); ``per_capture=True`` divides each capture by its own
    norm. 0 where the denominator is 0.

    The raw adjoint ``K^T y`` of the native-sampled power observable is biased
    toward the native-grid cells (picket fence): for one noiseless point it peaks
    0.7-1.7 m away on the micro scene, while this normalised statistic peaks
    within 0.33 m.
    """
    _check_space(space)
    _check_node(node, POWER_NODES, "power_map")
    _check_grid(grid)
    _validate_y(Y, geom)
    _check_noise_var(noise_var)
    delays = _capture_tau(tau_hat, geom.num_views, geom.num_bs)
    data = extract(Y, node).data
    if noise_var is not None:
        data = data - noise_floor(node, geom, noise_var)
    centers = grid.centers()

    if not per_capture:
        adjoint = power_backproject_grid(
            data,
            geom,
            grid,
            delays,
            space=space,
            product=node,
            kind=kind,
            oversample=oversample,
            pattern=pattern,
            polarization=polarization,
        )
        norms = power_column_norm(
            geom, centers, space, node, tau_hat=delays, pattern=pattern, polarization=polarization
        )
        denominator = np.sqrt(np.sum(norms**2, axis=(0, 1))).reshape(grid.shape)
        return np.where(
            denominator > 0.0, adjoint / np.where(denominator > 0.0, denominator, 1.0), 0.0
        )

    norms = power_column_norm(
        geom, centers, space, node, tau_hat=delays, pattern=pattern, polarization=polarization
    )
    out = np.zeros((geom.num_views, geom.num_bs) + tuple(grid.shape), dtype=np.float64)
    for v in range(geom.num_views):
        for b in range(geom.num_bs):
            selected = geom.select([v], [b])
            adjoint = power_backproject_grid(
                data[v : v + 1, b : b + 1],
                selected,
                grid,
                delays[v : v + 1, b : b + 1],
                space=space,
                product=node,
                kind=kind,
                oversample=oversample,
                pattern=pattern,
                polarization=polarization,
            )
            norm = norms[v, b].reshape(grid.shape)
            out[v, b] = np.where(norm > 0.0, adjoint / np.where(norm > 0.0, norm, 1.0), 0.0)
    return out


def log_mean_fusion(maps: np.ndarray, *, floor: float = DEFAULT_FLOOR) -> np.ndarray:
    """Return the log-domain (geometric-mean) fusion of ``maps`` float ``[V, B, *S]``.

    Negative values are clipped to 0, each capture is divided by its own maximum
    over ``S``, captures whose maximum is not finite and > 0 are dropped, and the
    result is ``exp(mean_c log(m_hat_c + floor))``. All captures dropped -> zeros.
    """
    value = _check_floor(floor)
    arr = np.asarray(maps, dtype=np.float64)
    if arr.ndim < 3:
        raise ValueError("maps must have shape [V, B, *S] with at least one spatial axis")
    spatial = arr.shape[2:]
    flat = np.maximum(arr, 0.0).reshape(arr.shape[0], arr.shape[1], -1)
    with np.errstate(invalid="ignore"):
        maxima = flat.max(axis=-1)
    keep = np.isfinite(maxima) & (maxima > 0.0)
    if not np.any(keep):
        return np.zeros(spatial, dtype=np.float64)
    normalised = flat[keep] / maxima[keep][..., None]
    fused = np.exp(np.mean(np.log(normalised + value), axis=0))
    return np.asarray(fused.reshape(spatial), dtype=np.float64)


def intensity_map(
    Y: np.ndarray,
    geom: CaptureGeometry,
    grid: VoxelGrid,
    space: str,
    node: str = "I",
    *,
    noise_var: float | None = None,
    floor: float = DEFAULT_FLOOR,
    kind: str = "trilinear",
    oversample: int = 8,
    pattern: str = "tr38901",
    polarization: str = "none",
) -> np.ndarray:
    """Return cone BP + log-mean fusion for the sync-invariant ``I`` / ``I_n0``."""
    _check_space(space)
    _check_node(node, ("I", "I_n0"), "intensity_map")
    _check_grid(grid)
    _check_floor(floor)
    per_capture = power_map(
        Y,
        geom,
        grid,
        space,
        node,
        noise_var=noise_var,
        kind=kind,
        oversample=oversample,
        pattern=pattern,
        polarization=polarization,
        per_capture=True,
    )
    return log_mean_fusion(per_capture, floor=floor)


def splat_returns(
    Y: np.ndarray,
    geom: CaptureGeometry,
    grid: VoxelGrid,
    space: str,
    *,
    tau_hat: np.ndarray | None = None,
    params: dict[str, object] | None = None,
    sigma_delay: float | None = None,
    p_hit: float = 0.7,
    p_false: float = 0.05,
    pattern: str = "tr38901",
    polarization: str = "none",
) -> np.ndarray:
    """Return-to-point splatting of the ``D`` returns into an occupancy log-odds grid.

    Voxel-driven: each voxel gathers the returns whose angle-delay footprint
    contains its predicted ``(u, t)``, which handles delay wrap-around and both
    spaces without unwrapping ranges. ``tau_hat`` is the N-mode per-capture delay
    (observed returns are at ``tau(x) + tau_c``). Singular voxels are exactly 0.
    """
    _check_space(space)
    _check_grid(grid)
    _validate_y(Y, geom)
    delays = _capture_tau(tau_hat, geom.num_views, geom.num_bs)
    hit_rate = float(p_hit)
    false_rate = float(p_false)
    if not (0.0 < false_rate < hit_rate < 1.0):
        raise ValueError("require 0 < p_false < p_hit < 1")
    if sigma_delay is None:
        sigma = 0.5 / geom.bandwidth
    else:
        sigma = float(sigma_delay)
        if not np.isfinite(sigma) or sigma <= 0.0:
            raise ValueError("sigma_delay must be finite and > 0")

    options = dict(params or {})
    options["delta_f"] = geom.delta_f
    obs = extract(Y, "D", options)
    returns = np.asarray(obs.data, dtype=np.float64)
    mask = np.asarray(obs.mask, dtype=bool)

    rows, cols = geom.aperture_shape
    spacing = _aperture_spacing(geom)
    fy = np.fft.fftshift(np.fft.fftfreq(cols))
    fz = np.fft.fftshift(np.fft.fftfreq(rows))
    period = geom.delay_period
    log_hit = float(np.log(hit_rate / false_rate))
    log_miss = float(np.log((1.0 - hit_rate) / (1.0 - false_rate)))

    centers = grid.centers()
    singular = _singular_mask(centers, geom, space)
    valid = np.flatnonzero(~singular)
    output = np.zeros(centers.shape[0], dtype=np.float64)

    for v in range(geom.num_views):
        for b in range(geom.num_bs):
            capture_returns = returns[v, b]
            capture_mask = mask[v, b] & np.isfinite(capture_returns)
            for start in range(0, valid.size, _SPLAT_CHUNK):
                chunk = valid[start : start + _SPLAT_CHUNK]
                if chunk.size == 0:
                    continue
                points = centers[chunk]
                factors = capture_factors(
                    points, geom, space, v, b, pattern=pattern, polarization=polarization
                )
                uy = factors.u_local[:, 1]
                uz = factors.u_local[:, 2]
                tau = factors.tau + delays[v, b]
                hemisphere = factors.hemisphere
                footprint = (
                    dirichlet_power(spacing * uy[:, None, None] - fy[None, :, None], cols)
                    / cols
                    * dirichlet_power(spacing * uz[:, None, None] - fz[None, None, :], rows)
                    / rows
                )
                selected_returns = capture_returns[hemisphere]
                selected_mask = capture_mask[hemisphere]
                difference = tau[:, None, None, None] - selected_returns
                wrapped = np.mod(difference + period / 2.0, period) - period / 2.0
                gaussian = np.exp(-0.5 * (wrapped / sigma) ** 2)
                combined = footprint[:, :, :, None] * gaussian
                combined = np.where(selected_mask, combined, 0.0)
                evidence = np.max(combined, axis=(1, 2, 3))
                output[chunk] += evidence * log_hit + (1.0 - evidence) * log_miss
    return output.reshape(grid.shape)


def _backproject_node(
    Y: np.ndarray,
    geom: CaptureGeometry,
    points: np.ndarray,
    space: str,
    node: str,
    tau_hat: np.ndarray | None,
    noise_var: float | None,
    mask_k: float,
    *,
    kind: str,
    window: str,
    oversample: tuple[int, int],
    pattern: str,
    polarization: str,
) -> np.ndarray:
    """Return the complex per-capture back-projection ``[V, B, P]`` of a coherent node.

    The ``[V, B, H, R, P]`` back-projection is summed over the hemisphere axis.
    """
    data = node_data(Y, node, noise_var=noise_var, mask_k=mask_k)
    projected = backproject(
        data,
        geom,
        points,
        space,
        tau_hat,
        True,
        kind=kind,
        window=window,
        oversample=oversample,
        pattern=pattern,
        polarization=polarization,
    )
    return projected.sum(axis=2).astype(np.complex128, copy=False)


def _envelope_per_bin(
    Y: np.ndarray,
    geom: CaptureGeometry,
    points: np.ndarray,
    space: str,
    node: str,
    weights: np.ndarray,
    noise_var: float | None,
    mask_k: float,
    *,
    kind: str,
    window: str,
    oversample: tuple[int, int],
    pattern: str,
    polarization: str,
) -> np.ndarray:
    """Return the float64 ``[V, B, P]`` per-bin incoherent envelope of ``node``."""
    data = node_data(Y, node, noise_var=noise_var, mask_k=mask_k)
    num_points = points.shape[0]
    out = np.zeros((geom.num_views, geom.num_bs, num_points), dtype=np.float64)
    for v in range(geom.num_views):
        for b in range(geom.num_bs):
            if not np.any(weights[v, b] > 0.0):
                continue
            selected = geom.select([v], [b])
            capture = data[v : v + 1, b : b + 1]
            lookups = [
                capture_lookup(
                    selected,
                    points,
                    space,
                    0,
                    0,
                    h,
                    kind=kind,
                    oversample=oversample,
                    pattern=pattern,
                    polarization=polarization,
                )
                for h in range(2)
            ]
            accumulated = np.zeros(num_points, dtype=np.float64)
            for n in range(geom.num_bins):
                if not np.any(capture[..., n] != 0.0):
                    continue
                single = np.zeros_like(capture)
                single[..., n] = capture[..., n]
                volume = capture_volume(
                    single, selected, 0, 0, window=window, oversample=oversample
                )
                for h in range(2):
                    projected = apply_lookup(volume[h], lookups[h])
                    accumulated += np.abs(projected) ** 2
            weight = weights[v, b]
            out[v, b] = np.where(
                weight > 0.0, accumulated / np.where(weight > 0.0, weight, 1.0), 0.0
            )
    return out


def envelope_map(
    Y: np.ndarray,
    geom: CaptureGeometry,
    points: np.ndarray,
    space: str,
    node: str = "IDP",
    *,
    tau_hat: np.ndarray | None = None,
    noise_var: float | None = None,
    mask_k: float = DEFAULT_MASK_K,
    per_capture: bool = False,
    kind: str = "trilinear",
    window: str = "taylor",
    oversample: tuple[int, int] = (8, 8),
    pattern: str = "tr38901",
    polarization: str = "none",
) -> np.ndarray:
    """Return the per-capture GLRT-normalised envelope of a coherent or per-bin node.

    Coherent nodes use ``|bp_c|**2 / |gamma_c|**2``; per-bin nodes sum ``|bp|**2``
    over bins incoherently (one unknown phase per ``(capture, bin)``) before
    dividing by the weight. ``per_capture=True`` returns float64 ``[V, B, P]``,
    otherwise the sum over ``(v, b)`` as float64 ``[P]``.
    """
    _check_space(space)
    _check_node(node, _DATA_NODES, "envelope_map")
    _validate_y(Y, geom)
    _check_noise_var(noise_var)
    pts = _prepare_points(points)
    delays = _capture_tau(tau_hat, geom.num_views, geom.num_bs)
    weights = capture_weights(geom, pts, space, pattern=pattern, polarization=polarization)

    if node in COHERENT_NODES:
        projected = _backproject_node(
            Y,
            geom,
            pts,
            space,
            node,
            delays,
            noise_var,
            mask_k,
            kind=kind,
            window=window,
            oversample=oversample,
            pattern=pattern,
            polarization=polarization,
        )
        values = np.where(
            weights > 0.0,
            np.abs(projected) ** 2 / np.where(weights > 0.0, weights, 1.0),
            0.0,
        )
    else:
        values = _envelope_per_bin(
            Y,
            geom,
            pts,
            space,
            node,
            weights,
            noise_var,
            mask_k,
            kind=kind,
            window=window,
            oversample=oversample,
            pattern=pattern,
            polarization=polarization,
        )

    result = values.astype(np.float64, copy=False)
    if per_capture:
        return result
    return result.sum(axis=(0, 1)).astype(np.float64)


def coherent_map(
    Y: np.ndarray,
    geom: CaptureGeometry,
    points: np.ndarray,
    space: str,
    node: str = "IDP",
    *,
    tau_hat: np.ndarray | None = None,
    noise_var: float | None = None,
    mask_k: float = DEFAULT_MASK_K,
    kind: str = "trilinear",
    window: str = "taylor",
    oversample: tuple[int, int] = (8, 8),
    pattern: str = "tr38901",
    polarization: str = "none",
) -> np.ndarray:
    """Return ``|sum_c bp_c|**2 / sum_c |gamma_c|**2`` for a coherent node.

    ``|bp_c|**2 / |gamma_c|**2`` is a per-capture GLRT; the coherent map is the
    cross-view GLRT. 0 where the denominator is 0. Only meaningful on grids finer
    than ``lambda/4`` (design §3.5): use the ROI windows of :func:`roi_refine`.
    """
    _check_space(space)
    _check_node(node, COHERENT_NODES, "coherent_map")
    _validate_y(Y, geom)
    _check_noise_var(noise_var)
    pts = _prepare_points(points)
    delays = _capture_tau(tau_hat, geom.num_views, geom.num_bs)
    projected = _backproject_node(
        Y,
        geom,
        pts,
        space,
        node,
        delays,
        noise_var,
        mask_k,
        kind=kind,
        window=window,
        oversample=oversample,
        pattern=pattern,
        polarization=polarization,
    )
    weights = capture_weights(geom, pts, space, pattern=pattern, polarization=polarization)
    numerator = np.abs(projected.sum(axis=(0, 1))) ** 2
    denominator = weights.sum(axis=(0, 1))
    return np.where(
        denominator > 0.0, numerator / np.where(denominator > 0.0, denominator, 1.0), 0.0
    ).astype(np.float64)


def roi_grid(
    center: np.ndarray,
    wavelength: float,
    *,
    half_width: float = DEFAULT_ROI_HALF_WIDTH,
    spacing: float | None = None,
) -> VoxelGrid:
    """Return a cube ROI grid centred at ``center`` (default ``lambda/4`` spacing)."""
    centre = np.asarray(center, dtype=np.float64)
    if centre.shape != (3,) or not np.all(np.isfinite(centre)):
        raise ValueError("center must be a finite [3] point")
    wave = float(wavelength)
    if not np.isfinite(wave) or wave <= 0.0:
        raise ValueError("wavelength must be finite and > 0")
    width = float(half_width)
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("half_width must be finite and > 0")
    step = wave / 4.0 if spacing is None else float(spacing)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("spacing must be finite and > 0")
    count = int(np.floor(2.0 * width / step + 1e-9)) + 1
    origin = centre - step * (count - 1) / 2.0
    return VoxelGrid(origin=origin, spacing=step, shape=(count, count, count))


@dataclass(frozen=True)
class RoiResult:
    """Refined ROI positions, their statistic and the input detections."""

    positions: np.ndarray  # float64 [K, 3] refined positions
    values: np.ndarray  # float64 [K] ROI statistic at the refined positions
    centers: np.ndarray  # float64 [K, 3] the input detections (ROI centres)


def _roi_offsets(spacing: float, polish_steps: int) -> np.ndarray:
    """Return the ``[(2 * polish_steps + 1)**3, 3]`` half-cell offset combinations."""
    step = spacing / (2.0 * polish_steps)
    axis = np.arange(-polish_steps, polish_steps + 1, dtype=np.float64) * step
    grids = np.meshgrid(axis, axis, axis, indexing="ij")
    return np.stack([grid.ravel() for grid in grids], axis=-1).astype(np.float64)


def _polish_coherent(
    geom: CaptureGeometry,
    grid: VoxelGrid,
    space: str,
    points: np.ndarray,
    projected: np.ndarray,
    statistic: np.ndarray,
    denominator: np.ndarray,
    candidate_fraction: float,
    polish_steps: int,
) -> tuple[np.ndarray, float]:
    """Return the polished position and value of the strongest ROI lobe."""
    finite = np.isfinite(statistic)
    if not np.any(finite):
        return points[0], 0.0
    maximum = float(np.max(statistic[finite]))
    threshold = candidate_fraction * maximum
    candidates = np.flatnonzero(finite & (statistic >= threshold))
    if candidates.size == 0:
        index = int(np.argmax(np.where(finite, statistic, -np.inf)))
        return points[index], float(statistic[index])

    offsets = _roi_offsets(grid.spacing, polish_steps)
    num_offsets = offsets.shape[0]
    num_captures = geom.num_views * geom.num_bs
    flat = projected.reshape(num_captures, points.shape[0])

    best_value = -np.inf
    best_position = points[int(candidates[0])]
    for start in range(0, candidates.size, _POLISH_CHUNK):
        chunk = candidates[start : start + _POLISH_CHUNK]
        centres = points[chunk]
        accumulate = np.zeros((chunk.size, num_offsets), dtype=np.complex128)
        for capture in range(num_captures):
            v, b = divmod(capture, geom.num_bs)
            to_ue = centres - geom.ue_pos[v]
            gradient = to_ue / np.linalg.norm(to_ue, axis=1)[:, None]
            if space == "bv":
                to_bs = centres - geom.bs_pos[b]
                gradient = gradient + to_bs / np.linalg.norm(to_bs, axis=1)[:, None]
            phase = geom.wavenumber * (gradient @ offsets.T)
            accumulate += flat[capture, chunk][:, None] * np.exp(1j * phase)
        weight = denominator[chunk]
        scores = np.abs(accumulate) ** 2 / np.where(weight > 0.0, weight, 1.0)[:, None]
        row, column = divmod(int(np.argmax(scores)), num_offsets)
        value = float(scores[row, column])
        if value > best_value:
            best_value = value
            best_position = centres[row] + offsets[column]
    return best_position, best_value


def roi_refine(
    Y: np.ndarray,
    geom: CaptureGeometry,
    detections: np.ndarray,
    space: str,
    node: str = "IDP",
    *,
    half_width: float = DEFAULT_ROI_HALF_WIDTH,
    spacing: float | None = None,
    candidate_fraction: float = DEFAULT_CANDIDATE_FRACTION,
    polish_steps: int = DEFAULT_POLISH_STEPS,
    tau_hat: np.ndarray | None = None,
    noise_var: float | None = None,
    mask_k: float = DEFAULT_MASK_K,
    kind: str = "trilinear",
    window: str = "taylor",
    oversample: tuple[int, int] = (8, 8),
    pattern: str = "tr38901",
    polarization: str = "none",
) -> RoiResult:
    """Refine detections inside one ROI window each (ROI coherent BP).

    Per-bin nodes take the ROI grid argmax of :func:`envelope_map`; coherent
    nodes polish the coherent PSF by first-order phase extrapolation
    (:func:`_polish_coherent`), because its grating lobes have nearly equal
    height and the ``lambda/4`` grid argmax is unreliable.
    """
    _check_space(space)
    _check_node(node, _DATA_NODES, "roi_refine")
    _validate_y(Y, geom)
    _check_noise_var(noise_var)
    if not 0.0 <= float(candidate_fraction) <= 1.0:
        raise ValueError("candidate_fraction must lie in [0, 1]")
    if isinstance(polish_steps, bool) or not isinstance(polish_steps, (int, np.integer)):
        raise ValueError("polish_steps must be an integer >= 1")
    if int(polish_steps) < 1:
        raise ValueError("polish_steps must be an integer >= 1")
    steps = int(polish_steps)
    centres_in = _prepare_points(detections)
    delays = _capture_tau(tau_hat, geom.num_views, geom.num_bs)

    positions = np.zeros((centres_in.shape[0], 3), dtype=np.float64)
    values = np.zeros(centres_in.shape[0], dtype=np.float64)
    for index, detection in enumerate(centres_in):
        grid = roi_grid(detection, geom.wavelength, half_width=half_width, spacing=spacing)
        points = grid.centers()
        if node in COHERENT_NODES:
            projected = _backproject_node(
                Y,
                geom,
                points,
                space,
                node,
                delays,
                noise_var,
                mask_k,
                kind=kind,
                window=window,
                oversample=oversample,
                pattern=pattern,
                polarization=polarization,
            )
            weights = capture_weights(
                geom, points, space, pattern=pattern, polarization=polarization
            )
            denominator = weights.sum(axis=(0, 1))
            statistic = np.where(
                denominator > 0.0,
                np.abs(projected.sum(axis=(0, 1))) ** 2
                / np.where(denominator > 0.0, denominator, 1.0),
                0.0,
            )
            position, value = _polish_coherent(
                geom,
                grid,
                space,
                points,
                projected,
                statistic,
                denominator,
                float(candidate_fraction),
                steps,
            )
        else:
            statistic = envelope_map(
                Y,
                geom,
                points,
                space,
                node,
                tau_hat=delays,
                noise_var=noise_var,
                mask_k=mask_k,
                per_capture=False,
                kind=kind,
                window=window,
                oversample=oversample,
                pattern=pattern,
                polarization=polarization,
            )
            best = int(np.argmax(statistic))
            position = points[best]
            value = float(statistic[best])
        positions[index] = position
        values[index] = value
    return RoiResult(positions=positions, values=values, centers=np.array(centres_in, copy=True))


def envelope_bp_fn(
    space: str, node: str = "IDP", **kwargs: Any
) -> Callable[[np.ndarray, CaptureGeometry, np.ndarray, np.ndarray], np.ndarray]:
    """Return the ``bp_fn(Y, geom, points, tau) -> [V, B, P]`` contract of the search."""
    _check_space(space)
    _check_node(node, _DATA_NODES, "envelope_bp_fn")

    def bp_fn(
        Y: np.ndarray, geom: CaptureGeometry, points: np.ndarray, tau: np.ndarray
    ) -> np.ndarray:
        return envelope_map(
            Y,
            geom,
            points,
            space,
            node,
            tau_hat=tau,
            per_capture=True,
            **kwargs,
        )

    return bp_fn


def _parabola_offset(values: np.ndarray, index: int) -> float:
    """Return the clipped 3-point parabolic vertex offset at ``index`` (0 at the ends)."""
    if index <= 0 or index >= values.shape[0] - 1:
        return 0.0
    previous, centre, following = values[index - 1], values[index], values[index + 1]
    denominator = previous - 2.0 * centre + following
    if denominator >= 0.0:
        return 0.0
    offset = 0.5 * (previous - following) / denominator
    return float(np.clip(offset, -0.5, 0.5))


def blind_tau_search(
    bp_fn: Callable[[np.ndarray, CaptureGeometry, np.ndarray, np.ndarray], np.ndarray],
    Y: np.ndarray,
    geom: CaptureGeometry,
    grid: VoxelGrid,
    tau_range: tuple[float, float],
    *,
    step: float | None = None,
    max_points: int = 4,
    rel: float = 0.3,
    fine: int = 10,
    levels: int = 3,
    floor: float = DEFAULT_FLOOR,
) -> np.ndarray:
    """Blind per-capture delay search maximising cross-view envelope consistency.

    Needs point-like, angularly separated scatterers and a ``bp_fn`` with
    tricubic ``kind``: trilinear interpolation biases the angle-only consensus by
    about 0.2 m (~1 ns). Returns float64 ``[V, B]`` (not wrapped; delays are
    identifiable modulo ``geom.delay_period``). ``bp_fn`` must evaluate every
    capture from its own ``tau[c]`` in one call.
    """
    _check_floor(floor)
    if not callable(bp_fn):
        raise ValueError("bp_fn must be callable")
    _check_grid(grid)
    try:
        low, high = (float(value) for value in tau_range)
    except (TypeError, ValueError) as error:
        raise ValueError("tau_range must be two finite floats lo < hi") from error
    if not np.isfinite(low) or not np.isfinite(high) or not low < high:
        raise ValueError("tau_range must be two finite floats lo < hi")
    period = geom.delay_period
    if high - low > period * (1.0 + 1e-9):
        raise ValueError("tau_range must not be wider than geom.delay_period")
    if step is None:
        step_value = 0.5 / geom.bandwidth
    else:
        step_value = float(step)
    if not np.isfinite(step_value) or step_value <= 0.0:
        raise ValueError("step must be finite and > 0")
    if isinstance(max_points, bool) or not isinstance(max_points, (int, np.integer)):
        raise ValueError("max_points must be an integer >= 1")
    if int(max_points) < 1:
        raise ValueError("max_points must be an integer >= 1")
    rel_value = float(rel)
    if not np.isfinite(rel_value) or not 0.0 <= rel_value < 1.0:
        raise ValueError("rel must lie in [0, 1)")
    if isinstance(fine, bool) or not isinstance(fine, (int, np.integer)) or int(fine) < 1:
        raise ValueError("fine must be an integer >= 1")
    if isinstance(levels, bool) or not isinstance(levels, (int, np.integer)) or int(levels) < 0:
        raise ValueError("levels must be an integer >= 0")

    num_views, num_bs = geom.num_views, geom.num_bs
    num_captures = num_views * num_bs
    num_bins = geom.num_bins
    floor_value = _check_floor(floor)

    def table(points: np.ndarray, taus: list[np.ndarray]) -> np.ndarray:
        rows = []
        for tau in taus:
            mapped = np.asarray(bp_fn(Y, geom, points, tau))
            expected = (num_views, num_bs, points.shape[0])
            if mapped.shape != expected:
                raise ValueError(f"bp_fn output must have shape {expected}, got {mapped.shape}")
            rows.append(mapped.reshape(num_captures, -1))
        return np.stack(rows, axis=0)

    marginal = [
        np.full((num_views, num_bs), -period / 2.0 + m * period / num_bins) for m in range(num_bins)
    ]
    consensus = table(grid.centers(), marginal).mean(axis=0)
    scale = consensus.max(axis=-1)
    fused = log_mean_fusion(
        consensus.reshape((num_views, num_bs) + tuple(grid.shape)), floor=floor_value
    )
    peaks = nms_peaks(
        fused,
        grid,
        2.0 * grid.spacing,
        min_value=rel_value * float(fused.max()),
        max_peaks=int(max_points),
    )
    if peaks.positions.shape[0] == 0:
        raise ValueError("blind_tau_search found no consensus peak")

    points_x = np.array(peaks.positions, dtype=np.float64)
    refine_scale = grid.spacing / 2.0
    for _ in range(int(levels)):
        for index in range(points_x.shape[0]):
            local = VoxelGrid(
                origin=points_x[index] - 2.0 * refine_scale,
                spacing=refine_scale,
                shape=(5, 5, 5),
            )
            local_centres = local.centers()
            local_map = table(local_centres, marginal).mean(axis=0)
            keep = scale > 0.0
            if not np.any(keep):
                continue
            ratio = np.maximum(local_map[keep], 0.0) / scale[keep][:, None] + floor_value
            score = np.mean(np.log(ratio), axis=0)
            points_x[index] = local_centres[int(np.argmax(score))]
        refine_scale /= 2.5

    candidates = np.arange(low, high + 0.5 * step_value, step_value)
    hypothesis = [np.full((num_views, num_bs), float(value)) for value in candidates]
    consistency = table(points_x, hypothesis)
    peak = consistency.max(axis=(0, 2))
    flat = np.zeros(num_captures, dtype=np.float64)
    for capture in range(num_captures):
        if peak[capture] <= 0.0:
            continue
        ratios = np.maximum(consistency[:, capture, :], 0.0) / peak[capture] + floor_value
        scores = np.sum(np.log(ratios), axis=1)
        flat[capture] = candidates[int(np.argmax(scores))]

    tau_estimate = flat.reshape(num_views, num_bs)
    offsets = np.arange(-int(fine), int(fine) + 1, dtype=np.float64) * (step_value / int(fine))
    fine_hypothesis = [tau_estimate + offset for offset in offsets]
    refined = table(points_x, fine_hypothesis)
    for capture in range(num_captures):
        if peak[capture] <= 0.0:
            continue
        ratios = np.maximum(refined[:, capture, :], 0.0) / peak[capture] + floor_value
        scores = np.sum(np.log(ratios), axis=1)
        best = int(np.argmax(scores))
        correction = _parabola_offset(scores, best)
        flat[capture] += offsets[best] + (step_value / int(fine)) * correction

    return flat.reshape(num_views, num_bs).astype(np.float64)
