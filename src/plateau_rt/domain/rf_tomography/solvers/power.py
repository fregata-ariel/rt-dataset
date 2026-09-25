"""E2 power inversion on the pruned E1 support (docs/tomography_baselines.md §4.1).

Model
-----
One nonnegative power ``x_p`` per support point ``p`` predicts every power bin as

    mu = K x + background,

where ``K`` is a real, nonnegative operator (:func:`kernels.power_operator`) and
``background`` is the noise-only mean (:func:`kernels.noise_floor`) or zero.

Solvers and objectives
----------------------
``kl_em`` carries out the EM / Richardson-Lucy update for the generalised KL
divergence ``D(y || mu) = sum(mu - y + y log(y / mu))`` with a known background.
``is_mlem`` is the Fevotte-Idier (2011) majorise-minimise update for the
exponential / Itakura-Saito negative log-likelihood ``sum(y / mu + log mu)``; the
``sqrt`` update with exponent ``1/2`` is the monotone one. ``nn_fista_l1``
minimises the nonnegative, smooth-plus-l1 objective

    F(x) = 0.5 ||K x + background - y||^2 + l1 * sum(x) + tv(x),   x >= 0,

with a monotone FISTA (MFISTA, Beck-Teboulle 2009) and an optional Huber-smoothed
anisotropic TV term on the support graph. All three track their objective per
iteration and can stop early on a relative decrease ``tol``.

Operator contract
------------------
The ``op`` argument is typed as :class:`scipy.sparse.linalg.LinearOperator` but is
used duck-typed: only ``op.shape``, ``op.matvec`` and ``op.rmatvec`` are touched.
The operator is never densified, and every forward / adjoint call is counted in
:class:`PowerSolution`. Matrices are real float64.

Support helpers
---------------
:func:`prune_support` turns an E1 density into a flat support: NMS seeds
(:func:`metrics.nms_peaks`), a ball dilation of ``radius`` metres and an optional
per-density cap. :func:`support_points`, :func:`support_edges` and
:func:`support_to_map` map that support to point coordinates, the 6-neighbour
support graph (for the TV term) and a dense grid map. NumPy/SciPy only.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.sparse.linalg import LinearOperator

from plateau_rt.domain.rf_tomography.geometry import VoxelGrid
from plateau_rt.domain.rf_tomography.metrics import nms_peaks
from plateau_rt.domain.rf_tomography.solvers._core import (
    CallCounter,
    IterativeResult,
    mfista,
    power_iteration,
    relative_decrease_stop,
)

__all__ = [
    "DEFAULT_CAP",
    "DEFAULT_RADIUS_M",
    "DEFAULT_REL_THRESHOLD",
    "SOLVERS",
    "PowerSolution",
    "is_mlem",
    "is_objective",
    "kl_em",
    "kl_objective",
    "nn_fista_l1",
    "prune_support",
    "support_edges",
    "support_points",
    "support_to_map",
]

SOLVERS: tuple[str, ...] = ("kl_em", "is_mlem", "nn_fista_l1")
DEFAULT_CAP: int = 50_000
DEFAULT_RADIUS_M: float = 2.0
DEFAULT_REL_THRESHOLD: float = 1e-2


@dataclass(frozen=True)
class PowerSolution(IterativeResult):
    """Result of one E2 power solve (fields of :class:`IterativeResult`).

    ``x`` is the ``[P]`` float64 power (``>= 0``), ``n_iter == len(objective) - 1``,
    ``n_forward`` / ``n_adjoint`` count ``op.matvec`` / ``op.rmatvec`` calls and
    ``converged`` means stopped early by ``tol`` or trivially solved.
    """


# --- validation helpers -------------------------------------------------------


def _flat_real(value: np.ndarray, name: str, *, nonnegative: bool) -> np.ndarray:
    """Return ``value`` as a flat float64 array, rejecting complex/invalid input."""
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    try:
        array = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a real array") from error
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    if nonnegative and np.any(array < 0.0):
        raise ValueError(f"{name} must be >= 0")
    return array


def _float_at_least(value: float, name: str, bound: float) -> float:
    """Return ``value`` as a finite float ``>= bound``."""
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite and >= {bound}") from error
    if not np.isfinite(number) or number < bound:
        raise ValueError(f"{name} must be finite and >= {bound}")
    return number


def _float_positive(value: float, name: str) -> float:
    """Return ``value`` as a finite float ``> 0``."""
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite and > 0") from error
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return number


def _int_at_least(value: int, name: str, bound: int) -> int:
    """Return ``value`` as an int ``>= bound`` (bool and floats are rejected)."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer >= {bound}")
    number = int(value)
    if number < bound:
        raise ValueError(f"{name} must be an integer >= {bound}")
    return number


def _background(value: np.ndarray | float, size: int, *, positive: bool) -> np.ndarray:
    """Return ``value`` broadcast / flattened to a float64 ``[size]`` vector."""
    if np.iscomplexobj(value):
        raise ValueError("background must be real")
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("background must be a real scalar or array") from error
    if array.ndim == 0:
        array = np.full((size,), float(array), dtype=np.float64)
    else:
        array = array.reshape(-1)
        if array.size != size:
            raise ValueError("background must have the same size as y")
    if not np.all(np.isfinite(array)):
        raise ValueError("background must contain only finite values")
    if positive:
        if np.any(array <= 0.0):
            raise ValueError("background must be > 0")
    elif np.any(array < 0.0):
        raise ValueError("background must be >= 0")
    return array


def _validate_inputs(
    op: LinearOperator,
    y: np.ndarray,
    background: np.ndarray | float,
    x0: np.ndarray | None,
    n_iter: int,
    tol: float,
    *,
    positive_background: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, int, float]:
    """Validate the shared data of the three solvers and return flat arrays."""
    rows, cols = op.shape
    y_flat = _flat_real(y, "y", nonnegative=True)
    if y_flat.size != rows:
        raise ValueError(f"y must have {rows} entries, got {y_flat.size}")
    b = _background(background, rows, positive=positive_background)
    x0_flat: np.ndarray | None = None
    if x0 is not None:
        x0_flat = _flat_real(x0, "x0", nonnegative=True)
        if x0_flat.shape != (cols,):
            raise ValueError(f"x0 must have shape ({cols},), got {x0_flat.shape}")
    return y_flat, b, x0_flat, _int_at_least(n_iter, "n_iter", 0), _float_at_least(tol, "tol", 0.0)


def _counted(
    op: LinearOperator,
) -> tuple[CallCounter, Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    """Return a float64 :class:`CallCounter` and its forward / adjoint bindings."""
    counter = CallCounter(
        lambda v: np.asarray(op.matvec(v), dtype=np.float64),
        lambda v: np.asarray(op.rmatvec(v), dtype=np.float64),
    )
    return counter, counter.forward, counter.adjoint


def _solution(
    x: np.ndarray,
    objective: list[float] | np.ndarray,
    counted: CallCounter,
    converged: bool,
) -> PowerSolution:
    """Package an objective list and the call counts into a :class:`PowerSolution`."""
    values = np.asarray(objective, dtype=np.float64)
    return PowerSolution(
        x=np.asarray(x, dtype=np.float64),
        objective=values,
        n_iter=int(values.shape[0]) - 1,
        n_forward=counted.n_forward,
        n_adjoint=counted.n_adjoint,
        converged=bool(converged),
    )


# --- objectives ---------------------------------------------------------------


def kl_objective(y: np.ndarray, mu: np.ndarray) -> float:
    """Return ``sum(mu - y + y log(y / mu))`` with ``0 log 0 = 0``."""
    y_arr = np.asarray(y, dtype=np.float64)
    mu_arr = np.asarray(mu, dtype=np.float64)
    terms = mu_arr - y_arr
    positive = y_arr > 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = terms + np.where(positive, y_arr * np.log(y_arr / mu_arr), 0.0)
    return float(np.sum(terms))


def is_objective(y: np.ndarray, mu: np.ndarray) -> float:
    """Return ``sum(y / mu + log(mu))`` (finite for ``y == 0``)."""
    y_arr = np.asarray(y, dtype=np.float64)
    mu_arr = np.asarray(mu, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        values = y_arr / mu_arr + np.log(mu_arr)
    return float(np.sum(values))


# --- support helpers ----------------------------------------------------------


def _as_indices(grid: VoxelGrid, indices: np.ndarray) -> np.ndarray:
    """Return a validated int64 ``[K]`` vector of flat voxel indices."""
    if not isinstance(grid, VoxelGrid):
        raise ValueError("grid must be a VoxelGrid")
    if np.iscomplexobj(indices):
        raise ValueError("indices must be integers")
    array = np.asarray(indices)
    if array.ndim != 1:
        raise ValueError("indices must be one-dimensional")
    if array.dtype.kind not in "iu":
        raise ValueError("indices must be integers")
    flat = array.astype(np.int64, copy=False)
    if flat.size and (np.any(flat < 0) or np.any(flat >= grid.size)):
        raise ValueError("indices out of range")
    return flat


def _as_edges(edges: np.ndarray, num_points: int) -> np.ndarray:
    """Return a validated int64 ``[E, 2]`` edge list with entries in ``[0, P)``."""
    if np.iscomplexobj(edges):
        raise ValueError("edges must be integers")
    array = np.asarray(edges)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError("edges must have shape [E, 2]")
    if array.dtype.kind not in "iu":
        raise ValueError("edges must be integers")
    flat = array.astype(np.int64, copy=False)
    if flat.size and (np.any(flat < 0) or np.any(flat >= num_points)):
        raise ValueError("edges entries must lie in [0, P)")
    return flat


def _ball_structure(spacing: float, radius: float) -> np.ndarray:
    """Return the boolean ball dilation structure of ``radius`` metres."""
    max_step = int(np.ceil(radius / spacing))
    steps = np.arange(-max_step, max_step + 1, dtype=np.int64)
    grid_x, grid_y, grid_z = np.meshgrid(steps, steps, steps, indexing="ij")
    offsets = np.stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()], axis=1)
    reach = radius * (1.0 + 1e-9) + 1e-12
    distance = spacing * np.sqrt(np.sum(offsets.astype(np.float64) ** 2, axis=1))
    return (distance <= reach).reshape((2 * max_step + 1,) * 3)


def prune_support(
    density: np.ndarray,
    grid: VoxelGrid,
    radius: float = DEFAULT_RADIUS_M,
    cap: int = DEFAULT_CAP,
    *,
    nms_radius: float | None = None,
    rel_threshold: float = DEFAULT_REL_THRESHOLD,
) -> np.ndarray:
    """Return the flat E2 support from an E1 density map (design §3.5).

    NMS seeds (``>= rel_threshold * max(density)``) are dilated by a ball of
    ``radius`` metres; at most ``cap`` voxels survive, chosen by decreasing
    density (ties keep the smaller flat index). Empty when ``density <= 0``.
    """
    if not isinstance(grid, VoxelGrid):
        raise ValueError("grid must be a VoxelGrid")
    if np.iscomplexobj(density):
        raise ValueError("density must be real")
    try:
        values = np.asarray(density, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("density must be a real array") from error
    if values.shape != tuple(grid.shape):
        raise ValueError(f"density shape must match grid.shape, got {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError("density must contain only finite values")

    radius_f = _float_at_least(radius, "radius", 0.0)
    nms_f = radius_f if nms_radius is None else _float_at_least(nms_radius, "nms_radius", 0.0)
    rel_f = _float_at_least(rel_threshold, "rel_threshold", 0.0)
    if rel_f >= 1.0:
        raise ValueError("rel_threshold must lie in [0, 1)")
    cap_i = _int_at_least(cap, "cap", 1)

    peak_value = float(np.max(values))
    if peak_value <= 0.0:
        return np.zeros(0, dtype=np.int64)

    peaks = nms_peaks(values, grid, nms_f, refine=False, min_value=rel_f * peak_value)
    mask = np.zeros(tuple(grid.shape), dtype=bool)
    if peaks.indices.shape[0] > 0:
        mask.ravel()[peaks.indices] = True
        mask = binary_dilation(mask, structure=_ball_structure(grid.spacing, radius_f))
    flat = np.flatnonzero(mask.ravel()).astype(np.int64)
    if flat.size > cap_i:
        order = np.lexsort((flat, -values.ravel()[flat]))
        flat = np.sort(flat[order[:cap_i]])
    return flat


def support_points(grid: VoxelGrid, indices: np.ndarray) -> np.ndarray:
    """Return the voxel centres ``grid.origin + spacing * multi`` of ``indices``."""
    flat = _as_indices(grid, indices)
    multi = np.stack(np.unravel_index(flat, grid.shape), axis=1).astype(np.float64)
    return np.asarray(grid.origin, dtype=np.float64) + float(grid.spacing) * multi


def support_edges(grid: VoxelGrid, indices: np.ndarray) -> np.ndarray:
    """Return the 6-neighbour support-graph edges as sorted positions ``[E, 2]``.

    ``indices`` must be sorted ascending and unique. Each row ``(i, j)`` holds
    positions into ``indices`` with ``i < j``; rows are sorted lexicographically.
    """
    flat = _as_indices(grid, indices)
    if flat.shape[0] > 1 and not np.all(np.diff(flat) > 0):
        raise ValueError("indices must be sorted ascending and unique")
    if flat.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.int64)
    multi = np.column_stack(np.unravel_index(flat, grid.shape)).astype(np.int64)
    pieces: list[np.ndarray] = []
    for axis in range(3):
        neighbour = multi.copy()
        neighbour[:, axis] += 1
        inside = neighbour[:, axis] < grid.shape[axis]
        if not np.any(inside):
            continue
        source = np.flatnonzero(inside)
        moved = neighbour[inside]
        target = np.ravel_multi_index((moved[:, 0], moved[:, 1], moved[:, 2]), grid.shape)
        position = np.searchsorted(flat, target)
        clipped = np.clip(position, 0, flat.shape[0] - 1)
        found = flat[clipped] == target
        if not np.any(found):
            continue
        pairs = np.column_stack([source[found], clipped[found]])
        pieces.append(np.sort(pairs, axis=1))
    if not pieces:
        return np.zeros((0, 2), dtype=np.int64)
    stacked = np.concatenate(pieces, axis=0)
    order = np.lexsort((stacked[:, 1], stacked[:, 0]))
    return stacked[order].astype(np.int64)


def support_to_map(x: np.ndarray, grid: VoxelGrid, indices: np.ndarray) -> np.ndarray:
    """Return a ``grid.shape`` float64 map with ``x`` written at ``indices``."""
    flat = _as_indices(grid, indices)
    if np.iscomplexobj(x):
        raise ValueError("x must be real")
    try:
        values = np.asarray(x, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("x must be a real array") from error
    if values.shape != (flat.shape[0],):
        raise ValueError(f"x must have shape ({flat.shape[0]},), got {values.shape}")
    output = np.zeros(tuple(grid.shape), dtype=np.float64)
    output.ravel()[flat] = values
    return output


# --- EM / MM state ------------------------------------------------------------


def _em_initial(
    y_flat: np.ndarray,
    b: np.ndarray,
    sens: np.ndarray,
    x0: np.ndarray | None,
) -> tuple[np.ndarray, bool]:
    """Return the EM/MM start point and whether the problem is trivially zero."""
    active = sens > 0.0
    if x0 is not None:
        return np.where(active, x0, 0.0).astype(np.float64), False
    total_y = float(np.sum(y_flat))
    total_sens = float(np.sum(sens))
    if total_y == 0.0 or total_sens == 0.0:
        return np.zeros(sens.shape[0], dtype=np.float64), True
    total_b = float(np.sum(b))
    scale = (total_y - total_b) / total_sens if total_y > total_b else total_y / total_sens
    return np.where(active, scale, 0.0).astype(np.float64), False


# --- solvers ------------------------------------------------------------------


def kl_em(
    op: LinearOperator,
    y: np.ndarray,
    background: np.ndarray | float,
    *,
    x0: np.ndarray | None = None,
    n_iter: int = 200,
    tol: float = 0.0,
) -> PowerSolution:
    """EM / Richardson-Lucy for ``min_x >= 0 sum(mu - y + y log(y / mu))``."""
    y_flat, b, x0_flat, iterations, tolerance = _validate_inputs(
        op, y, background, x0, n_iter, tol, positive_background=True
    )
    rows = op.shape[0]
    counted, matvec, rmatvec = _counted(op)

    sens = rmatvec(np.ones(rows, dtype=np.float64))
    active = sens > 0.0
    x, trivial = _em_initial(y_flat, b, sens, x0_flat)
    if trivial:
        return _solution(x, [kl_objective(y_flat, b)], counted, True)

    mu = matvec(x) + b
    objective = [kl_objective(y_flat, mu)]
    converged = False
    for _ in range(iterations):
        ratio = np.zeros_like(mu)
        np.divide(y_flat, mu, out=ratio, where=mu > 0.0)
        gradient = rmatvec(ratio)
        update = np.zeros_like(x)
        np.divide(x * gradient, sens, out=update, where=active)
        x = update
        mu = matvec(x) + b
        objective.append(kl_objective(y_flat, mu))
        if tolerance > 0.0 and relative_decrease_stop(objective, tolerance):
            converged = True
            break
    return _solution(x, objective, counted, converged)


def is_mlem(
    op: LinearOperator,
    y: np.ndarray,
    background: np.ndarray | float,
    *,
    x0: np.ndarray | None = None,
    n_iter: int = 200,
    tol: float = 0.0,
) -> PowerSolution:
    """Majorise-minimise (Fevotte-Idier) for ``sum(y / mu + log mu)``."""
    y_flat, b, x0_flat, iterations, tolerance = _validate_inputs(
        op, y, background, x0, n_iter, tol, positive_background=True
    )
    rows = op.shape[0]
    counted, matvec, rmatvec = _counted(op)

    sens = rmatvec(np.ones(rows, dtype=np.float64))
    active = sens > 0.0
    x, trivial = _em_initial(y_flat, b, sens, x0_flat)
    if trivial:
        return _solution(x, [is_objective(y_flat, b)], counted, True)

    mu = matvec(x) + b
    objective = [is_objective(y_flat, mu)]
    converged = False
    for _ in range(iterations):
        inverse_mu = np.zeros_like(mu)
        np.divide(1.0, mu, out=inverse_mu, where=mu > 0.0)
        numerator = rmatvec(y_flat * inverse_mu * inverse_mu)
        denominator = rmatvec(inverse_mu)
        factor = np.zeros_like(x)
        np.divide(numerator, denominator, out=factor, where=denominator > 0.0)
        x = np.where(active, x * np.sqrt(np.maximum(factor, 0.0)), 0.0)
        mu = matvec(x) + b
        objective.append(is_objective(y_flat, mu))
        if tolerance > 0.0 and relative_decrease_stop(objective, tolerance):
            converged = True
            break
    return _solution(x, objective, counted, converged)


def nn_fista_l1(
    op: LinearOperator,
    y: np.ndarray,
    background: np.ndarray | float = 0.0,
    *,
    lam: float = 0.03,
    tv: bool = False,
    edges: np.ndarray | None = None,
    tv_weight: float = 1e-3,
    tv_eps: float = 0.1,
    x0: np.ndarray | None = None,
    n_iter: int = 200,
    tol: float = 0.0,
    lipschitz: float | None = None,
    power_iters: int = 30,
) -> PowerSolution:
    """Monotone FISTA for ``0.5 ||Kx + b - y||^2 + l1 sum(x) + tv(x)``, ``x >= 0``."""
    if not isinstance(tv, (bool, np.bool_)):
        raise ValueError("tv must be a bool")
    use_tv = bool(tv)
    y_flat, b, x0_flat, iterations, tolerance = _validate_inputs(
        op, y, background, x0, n_iter, tol, positive_background=False
    )
    rows, cols = op.shape
    lam_f = _float_at_least(lam, "lam", 0.0)
    power_iters_i = _int_at_least(power_iters, "power_iters", 1)

    edge_arr: np.ndarray | None = None
    if edges is not None:
        if not use_tv:
            raise ValueError("edges may only be given when tv=True")
        edge_arr = _as_edges(edges, cols)
    if use_tv and edge_arr is None:
        raise ValueError("tv=True requires edges")
    if use_tv:
        tv_weight_f = _float_at_least(tv_weight, "tv_weight", 0.0)
        tv_eps_f = _float_positive(tv_eps, "tv_eps")
    else:
        tv_weight_f = 0.0
        tv_eps_f = 1.0
    if lipschitz is not None:
        lipschitz_f = _float_positive(lipschitz, "lipschitz")
    else:
        lipschitz_f = 0.0

    counted, matvec, rmatvec = _counted(op)

    residual = y_flat - b
    lam_max = float(np.max(rmatvec(residual)))
    start_value = 0.5 * float(np.dot(residual, residual))
    if lam_max <= 0.0:
        return _solution(np.zeros(cols, dtype=np.float64), [start_value], counted, True)

    if lipschitz is None:
        v0 = np.full(cols, 1.0 / np.sqrt(cols), dtype=np.float64)
        lip, _ = power_iteration(lambda v: rmatvec(matvec(v)), v0, power_iters_i, estimate="norm")
        lipschitz_value = 1.05 * lip
    else:
        lipschitz_value = lipschitz_f

    l1 = lam_f * lam_max
    if use_tv and edge_arr is not None:
        e0 = edge_arr[:, 0]
        e1 = edge_arr[:, 1]
        eps = tv_eps_f * (lam_max / lipschitz_value)
        lam_tv = tv_weight_f * lam_max
        if edge_arr.shape[0] > 0:
            degree = np.bincount(edge_arr.ravel(), minlength=cols)
            max_degree = int(np.max(degree))
        else:
            max_degree = 0
        l_tv = lam_tv * 2.0 * max_degree / eps
    else:
        e0 = np.zeros(0, dtype=np.int64)
        e1 = np.zeros(0, dtype=np.int64)
        lam_tv = 0.0
        eps = 1.0
        l_tv = 0.0
    lipschitz_total = lipschitz_value + l_tv

    def tv_term(vector: np.ndarray) -> float:
        if not use_tv or e0.shape[0] == 0:
            return 0.0
        difference = vector[e0] - vector[e1]
        magnitude = np.abs(difference)
        quadratic = difference * difference / (2.0 * eps)
        linear = magnitude - eps / 2.0
        return float(lam_tv * np.sum(np.where(magnitude <= eps, quadratic, linear)))

    def tv_gradient(vector: np.ndarray, result: np.ndarray) -> np.ndarray:
        if not use_tv or e0.shape[0] == 0:
            return result
        difference = vector[e0] - vector[e1]
        contribution = lam_tv * np.clip(difference / eps, -1.0, 1.0)
        np.add.at(result, e0, contribution)
        np.add.at(result, e1, -contribution)
        return result

    def objective_at(mapped: np.ndarray, vector: np.ndarray) -> float:
        difference = mapped - residual
        return (
            0.5 * float(np.dot(difference, difference))
            + l1 * float(np.sum(vector))
            + tv_term(vector)
        )

    if x0_flat is None:
        x = np.zeros(cols, dtype=np.float64)
        mapped = np.zeros(rows, dtype=np.float64)
    else:
        x = x0_flat
        mapped = matvec(x)
    f0 = objective_at(mapped, x)

    def prox_grad(w: np.ndarray, aw: np.ndarray) -> np.ndarray:
        gradient = tv_gradient(w, rmatvec(aw - residual))
        return np.maximum(w - (gradient + l1) / lipschitz_total, 0.0)

    def objective(z: np.ndarray, az: np.ndarray) -> float:
        return objective_at(az, z)

    x, history, _, converged = mfista(
        x,
        mapped,
        f0,
        forward=matvec,
        prox_grad=prox_grad,
        objective=objective,
        n_iter=iterations,
        stop="objective",
        tol=tolerance,
    )
    return _solution(x, history, counted, converged)
