"""E2 coherent inversion on point sets (docs/tomography_baselines.md §4.1).

Implements the coherent E2 tier over :class:`forward_sep.SeparableOperator`:
Tikhonov least squares (LSQR), complex-ℓ1 FISTA and ℓ2,1 group lasso (MMV over
the per-view amplitude maps), plus λ-scale helpers, per-point densities and the
λ/4 ROI windows of §3.5. NumPy/SciPy only: nothing here may import Sionna,
Mitsuba or Dr.Jit. All lengths are metres, phases radians.

The operator carries any gauge ``(phi, tau)`` itself; the solvers here never
touch gauges and invert the unwindowed data direct from amplitudes to CFR.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import LinearOperator, lsqr

from plateau_rt.domain.rf_tomography.forward_sep import (
    BETA_MODELS,
    SeparableOperator,
    project_shared_phase,
)
from plateau_rt.domain.rf_tomography.geometry import VoxelGrid
from plateau_rt.domain.rf_tomography.metrics import Peaks

DEFAULT_ITERATIONS: int = 200
DEFAULT_POWER_ITERATIONS: int = 30
LIPSCHITZ_SAFETY: float = 1.05
ROI_HALF_WIDTH_M: float = 0.25
LSQR_BAD_STOP: tuple[int, ...] = (3, 6, 7)

Prox = Callable[[np.ndarray, float], np.ndarray]
Penalty = Callable[[np.ndarray], float]


@dataclass(frozen=True)
class CoherentResult:
    """Output of one coherent E2 solve."""

    x: np.ndarray
    objective: np.ndarray
    n_iter: int
    n_forward: int
    n_adjoint: int
    converged: bool
    step: float


def _check_data(op: SeparableOperator, y: np.ndarray) -> np.ndarray:
    """Validate ``y`` against ``op.y_shape`` and return a finite complex128 array."""
    if np.shape(y) != op.y_shape:
        raise ValueError(f"y must have shape {op.y_shape}, got {np.shape(y)}")
    values = np.asarray(y, dtype=np.complex128)
    if not np.all(np.isfinite(values)):
        raise ValueError("y must contain only finite values")
    return values


def _check_nonneg(value: float, name: str) -> float:
    """Return ``value`` as a finite float ``>= 0``."""
    number = float(value)
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and >= 0")
    return number


def tikhonov_lsqr(
    op: SeparableOperator,
    y: np.ndarray,
    damp: float,
    *,
    iter_lim: int = DEFAULT_ITERATIONS,
    atol: float = 1e-12,
    btol: float = 1e-12,
) -> CoherentResult:
    """Minimise ``||A x - y||^2 + damp^2 ||x||^2`` with LSQR.

    The objective is ``0.5 ||A x - y||^2 + 0.5 damp^2 ||x||^2``; ``damp`` is
    absolute, in the same units as ``A``. LSQR is started from ``x = 0`` so the
    damping applies to ``x`` and not to ``x - x0``.
    """
    values = _check_data(op, y)
    damping = _check_nonneg(damp, "damp")
    linear = op.as_linear_operator()
    counts = {"fwd": 0, "adj": 0}

    def matvec(vector: np.ndarray) -> np.ndarray:
        counts["fwd"] += 1
        return linear.matvec(vector)

    def rmatvec(vector: np.ndarray) -> np.ndarray:
        counts["adj"] += 1
        return linear.rmatvec(vector)

    wrapped = LinearOperator(
        shape=linear.shape, matvec=matvec, rmatvec=rmatvec, dtype=np.complex128
    )
    out = lsqr(
        wrapped,
        values.reshape(-1),
        damp=damping,
        atol=float(atol),
        btol=float(btol),
        iter_lim=int(iter_lim),
    )
    x, istop, itn, r2norm = out[0], out[1], out[2], out[4]
    objective = np.array([0.5 * float(np.vdot(values, values).real), 0.5 * float(r2norm) ** 2])
    return CoherentResult(
        x=np.asarray(x, dtype=np.complex128).reshape(op.x_shape),
        objective=objective,
        n_iter=int(itn),
        n_forward=counts["fwd"],
        n_adjoint=counts["adj"],
        converged=int(istop) not in LSQR_BAD_STOP,
        step=float("nan"),
    )


def _power_iteration(op: SeparableOperator, n_iter: int, seed: int) -> tuple[float, int]:
    """Estimate ``||A||_2^2`` by power iteration on ``A^H A``.

    Returns ``(rho, n_calls)`` where ``n_calls`` is the number of forward (and
    adjoint) applications actually performed.
    """
    if int(n_iter) < 1:
        raise ValueError("n_iter must be >= 1")
    rng = np.random.default_rng(np.random.SeedSequence([int(seed)]))
    v = rng.standard_normal(op.x_shape) + 1j * rng.standard_normal(op.x_shape)
    norm = float(np.linalg.norm(v.ravel()))
    if norm == 0.0:
        return 0.0, 0
    v = v / norm
    rho = 0.0
    calls = 0
    for _ in range(int(n_iter)):
        w = op.adjoint(op.forward(v))
        calls += 1
        rho = float(np.vdot(v, w).real)
        wnorm = float(np.linalg.norm(w.ravel()))
        if wnorm == 0.0:
            break
        v = w / wnorm
    return rho, calls


def lipschitz_constant(
    op: SeparableOperator,
    *,
    n_iter: int = DEFAULT_POWER_ITERATIONS,
    seed: int = 0,
    safety: float = LIPSCHITZ_SAFETY,
) -> float:
    """Return ``safety * ||A||_2^2`` from a power iteration on ``A^H A``."""
    if int(n_iter) < 1:
        raise ValueError("n_iter must be >= 1")
    if not np.isfinite(float(safety)) or float(safety) < 1.0:
        raise ValueError("safety must be finite and >= 1")
    rho, _ = _power_iteration(op, int(n_iter), int(seed))
    if not np.isfinite(rho) or rho <= 0.0:
        raise ValueError("operator is zero")
    return float(safety) * rho


def _soft_threshold(z: np.ndarray, s: float) -> np.ndarray:
    """Elementwise complex soft threshold ``z * max(0, 1 - s / |z|)``."""
    magnitude = np.abs(z)
    factor = np.zeros_like(magnitude)
    np.divide(s, magnitude, out=factor, where=magnitude > 0.0)
    factor = np.maximum(0.0, 1.0 - factor)
    return (z * factor).astype(np.complex128, copy=False)


def _group_shrink(z: np.ndarray, s: float) -> np.ndarray:
    """Radial shrink of each point row: one group per axis-0 point.

    For a 1-D ``z`` this is exactly :func:`_soft_threshold`.
    """
    arr = np.asarray(z, dtype=np.complex128)
    if arr.ndim == 1:
        return _soft_threshold(arr, s)
    norms = np.sqrt(np.sum(np.abs(arr) ** 2, axis=tuple(range(1, arr.ndim))))
    factor = np.zeros_like(norms)
    np.divide(s, norms, out=factor, where=norms > 0.0)
    factor = np.maximum(0.0, 1.0 - factor)
    shape = (arr.shape[0],) + (1,) * (arr.ndim - 1)
    return (arr * factor.reshape(shape)).astype(np.complex128, copy=False)


def _constrained_prox(z: np.ndarray, s: float) -> np.ndarray:
    """Exact prox for constrained coefficients (one shared phase per point).

    The group norm is radial, so the best admissible subspace is the one that
    maximises ``||P z_p||``, i.e. the projection onto ``e^{j psi_p} R^3``; the
    shrink is then applied inside that subspace.
    """
    return _group_shrink(project_shared_phase(z), s)


def _l1_penalty(x: np.ndarray) -> float:
    """Elementwise complex ℓ1 penalty ``sum |x_i|``."""
    return float(np.sum(np.abs(x)))


def _group_penalty(x: np.ndarray) -> float:
    """ℓ2,1 penalty ``sum_p ||x_p||_2`` with one group per axis-0 point."""
    arr = np.asarray(x)
    if arr.ndim == 1:
        return float(np.sum(np.abs(arr)))
    norms = np.sqrt(np.sum(np.abs(arr) ** 2, axis=tuple(range(1, arr.ndim))))
    return float(np.sum(norms))


def _mfista(
    op: SeparableOperator,
    y: np.ndarray,
    lam: float,
    prox: Prox,
    penalty: Penalty,
    *,
    n_iter: int,
    tol: float,
    lipschitz: float | None,
    power_iterations: int,
    seed: int,
    x0: np.ndarray | None,
) -> CoherentResult:
    """Monotone FISTA (Beck & Teboulle 2009) for ``0.5 ||A x - y||^2 + lam R(x)``.

    Keeps ``A x`` for the iterates so each iteration costs exactly one forward
    and one adjoint application. The objective history is non-increasing.
    """
    if int(n_iter) < 1:
        raise ValueError("n_iter must be >= 1")
    if not np.isfinite(float(tol)) or float(tol) < 0.0:
        raise ValueError("tol must be finite and >= 0")
    n_forward = 0
    n_adjoint = 0
    if lipschitz is None:
        rho, calls = _power_iteration(op, int(power_iterations), int(seed))
        n_forward += calls
        n_adjoint += calls
        if not np.isfinite(rho) or rho <= 0.0:
            raise ValueError("operator is zero")
        big_l = LIPSCHITZ_SAFETY * rho
    else:
        big_l = float(lipschitz)
        if not np.isfinite(big_l) or big_l <= 0.0:
            raise ValueError("lipschitz must be finite and > 0")
    step = 1.0 / big_l

    if x0 is None:
        x = np.zeros(op.x_shape, dtype=np.complex128)
        ax = np.zeros(op.y_shape, dtype=np.complex128)
    else:
        x = np.array(x0, dtype=np.complex128, copy=True)
        if x.shape != op.x_shape:
            raise ValueError(f"x0 must have shape {op.x_shape}, got {x.shape}")
        ax = op.forward(x)
        n_forward += 1

    f = 0.5 * float(np.sum(np.abs(ax - y) ** 2)) + lam * penalty(x)
    history = [f]
    w, aw = x, ax
    theta = 1.0
    converged = False
    n_done = 0
    for k in range(int(n_iter)):
        grad = op.adjoint(aw - y)
        n_adjoint += 1
        z = prox(w - step * grad, step * lam)
        az = op.forward(z)
        n_forward += 1
        fz = 0.5 * float(np.sum(np.abs(az - y) ** 2)) + lam * penalty(z)
        theta_next = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * theta**2))
        if fz <= f:
            x_new, ax_new, f = z, az, fz
        else:
            x_new, ax_new = x, ax
        denom = max(float(np.linalg.norm(z.ravel())), float(np.linalg.norm(w.ravel())))
        residual = 0.0 if denom == 0.0 else float(np.linalg.norm((z - w).ravel())) / denom
        w = x_new + (theta / theta_next) * (z - x_new) + ((theta - 1.0) / theta_next) * (x_new - x)
        aw = (
            ax_new
            + (theta / theta_next) * (az - ax_new)
            + ((theta - 1.0) / theta_next) * (ax_new - ax)
        )
        x, ax, theta = x_new, ax_new, theta_next
        history.append(f)
        n_done = k + 1
        if residual <= float(tol):
            converged = True
            break
    return CoherentResult(
        x=np.asarray(x, dtype=np.complex128),
        objective=np.asarray(history, dtype=np.float64),
        n_iter=int(n_done),
        n_forward=int(n_forward),
        n_adjoint=int(n_adjoint),
        converged=bool(converged),
        step=float(step),
    )


def complex_l1_fista(
    op: SeparableOperator,
    y: np.ndarray,
    lam: float,
    *,
    n_iter: int = DEFAULT_ITERATIONS,
    tol: float = 1e-7,
    lipschitz: float | None = None,
    power_iterations: int = DEFAULT_POWER_ITERATIONS,
    seed: int = 0,
    x0: np.ndarray | None = None,
) -> CoherentResult:
    """Minimise ``0.5 ||A x - y||^2 + lam sum_i |x_i|`` with MFISTA.

    The step is ``1 / L`` with ``L`` from a power iteration unless ``lipschitz``
    is supplied. Allowed for the ``"shared"`` and ``"per_view"`` β models.
    """
    if op.beta_model == "constrained":
        raise ValueError("complex_l1_fista does not support 'constrained'; use mmv_group_lasso")
    values = _check_data(op, y)
    penalty_weight = _check_nonneg(lam, "lam")
    return _mfista(
        op,
        values,
        penalty_weight,
        _soft_threshold,
        _l1_penalty,
        n_iter=n_iter,
        tol=tol,
        lipschitz=lipschitz,
        power_iterations=power_iterations,
        seed=seed,
        x0=x0,
    )


def mmv_group_lasso(
    op: SeparableOperator,
    y: np.ndarray,
    lam: float,
    *,
    n_iter: int = DEFAULT_ITERATIONS,
    tol: float = 1e-7,
    lipschitz: float | None = None,
    power_iterations: int = DEFAULT_POWER_ITERATIONS,
    seed: int = 0,
    x0: np.ndarray | None = None,
) -> CoherentResult:
    """Minimise ``0.5 ||A x - y||^2 + lam sum_p ||x_p||_2`` with one group per point.

    ``"shared"`` groups are singletons (identical to :func:`complex_l1_fista`),
    ``"per_view"`` couples the per-view maps through a shared support and
    ``"constrained"`` keeps every iterate on the one-shared-phase set exactly.
    """
    values = _check_data(op, y)
    penalty_weight = _check_nonneg(lam, "lam")
    if op.beta_model == "constrained":
        prox: Prox = _constrained_prox
    else:
        prox = _group_shrink
    return _mfista(
        op,
        values,
        penalty_weight,
        prox,
        _group_penalty,
        n_iter=n_iter,
        tol=tol,
        lipschitz=lipschitz,
        power_iterations=power_iterations,
        seed=seed,
        x0=x0,
    )


def lambda_max(op: SeparableOperator, y: np.ndarray, *, group: bool = True) -> float:
    """Smallest ``lam`` for which ``x = 0`` solves the (group) lasso subproblem.

    ``group=False`` uses ``max_i |g_i|`` (for :func:`complex_l1_fista`);
    ``group=True`` uses ``max_p ||g_p||_2`` over point rows, projecting the
    constrained gradient onto the shared-phase set first.
    """
    values = _check_data(op, y)
    gradient = op.adjoint(values)
    if not group:
        return float(np.max(np.abs(gradient)))
    if op.beta_model == "constrained":
        gradient = project_shared_phase(gradient)
    if gradient.ndim == 1:
        return float(np.max(np.abs(gradient)))
    norms = np.sqrt(np.sum(np.abs(gradient) ** 2, axis=tuple(range(1, gradient.ndim))))
    return float(np.max(norms))


def point_density(x: np.ndarray, beta_model: str) -> np.ndarray:
    """Per-point density ``[P]`` float64 of coherent amplitudes ``x``.

    Shared uses ``|x|^2``, per-view the mean over ``(V, B)`` of ``|x|^2`` and
    constrained the sum of the three coefficient powers (a support score, not a
    physical power).
    """
    if beta_model not in BETA_MODELS:
        raise ValueError(f"beta_model must be one of {BETA_MODELS}, got {beta_model!r}")
    values = np.asarray(x, dtype=np.complex128)
    if beta_model == "shared":
        if values.ndim != 1:
            raise ValueError("shared amplitudes must be one-dimensional [P]")
        return (np.abs(values) ** 2).astype(np.float64)
    if beta_model == "per_view":
        if values.ndim != 3:
            raise ValueError("per_view amplitudes must have shape [P, V, B]")
        return np.mean(np.abs(values) ** 2, axis=(1, 2)).astype(np.float64)
    if values.ndim != 2:
        raise ValueError("constrained amplitudes must have shape [P, K]")
    return np.sum(np.abs(values) ** 2, axis=1).astype(np.float64)


def roi_grids_from_detections(
    det: np.ndarray | Peaks,
    *,
    half_width: float = ROI_HALF_WIDTH_M,
    spacing: float | None = None,
    wavelength: float | None = None,
) -> list[VoxelGrid]:
    """One cubic λ/4 :class:`VoxelGrid` per detection, covering ``±half_width``.

    ``spacing`` defaults to ``wavelength / 4``; the per-axis count is odd so the
    centre voxel sits exactly on the detection. Empty ``det`` returns ``[]``.
    """
    positions = det.positions if isinstance(det, Peaks) else det
    array = np.asarray(positions, dtype=np.float64)
    if array.size == 0:
        return []
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("det must have shape [K, 3]")
    if not np.all(np.isfinite(array)):
        raise ValueError("det must contain only finite positions")
    width = float(half_width)
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("half_width must be finite and > 0")
    if wavelength is not None:
        wave = float(wavelength)
        if not np.isfinite(wave) or wave <= 0.0:
            raise ValueError("wavelength must be finite and > 0")
    if spacing is None:
        if wavelength is None:
            raise ValueError("either spacing or wavelength must be given")
        step = float(wavelength) / 4.0
    else:
        step = float(spacing)
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("spacing must be finite and > 0")
    count = 2 * int(np.ceil(width / step - 1e-9)) + 1
    shift = (count - 1) / 2.0 * step
    shape = (count, count, count)
    return [VoxelGrid(origin=row - shift, spacing=step, shape=shape) for row in array]


def roi_points(grids: Sequence[VoxelGrid]) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate the centres of ``grids`` and return ``(points, owner)``.

    ``owner[p]`` is the index of the grid that owns point ``p``. Empty input
    returns ``(zeros((0, 3)), zeros((0,)))`` with ``int64`` owners.
    """
    if len(grids) == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=np.int64)
    blocks = [np.asarray(grid.centers(), dtype=np.float64) for grid in grids]
    owners = [np.full(block.shape[0], index, dtype=np.int64) for index, block in enumerate(blocks)]
    return np.concatenate(blocks, axis=0), np.concatenate(owners, axis=0)
