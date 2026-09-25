"""Unit tests for plateau_rt.domain.rf_tomography.solvers.power (E2, task T12)."""

from __future__ import annotations

import functools
import os
import time

import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.sparse.linalg import aslinearoperator

from plateau_rt.domain.rf_tomography import synthetic
from plateau_rt.domain.rf_tomography.forward_exact import atom_cfr
from plateau_rt.domain.rf_tomography.geometry import VoxelGrid
from plateau_rt.domain.rf_tomography.kernels import (
    noise_floor,
    power_backproject_grid,
    power_operator,
)
from plateau_rt.domain.rf_tomography.metrics import Matching, match, nms_peaks
from plateau_rt.domain.rf_tomography.observables import extract
from plateau_rt.domain.rf_tomography.solvers.power import (
    DEFAULT_CAP,
    DEFAULT_RADIUS_M,
    DEFAULT_REL_THRESHOLD,
    SOLVERS,
    PowerSolution,
    is_mlem,
    is_objective,
    kl_em,
    kl_objective,
    nn_fista_l1,
    prune_support,
    support_edges,
    support_points,
    support_to_map,
)
from plateau_rt.domain.rf_tomography.sync import capture_power

# --- A. Support helpers -------------------------------------------------------


def _two_ball_density() -> tuple[VoxelGrid, np.ndarray, np.ndarray]:
    grid = VoxelGrid(origin=np.zeros(3), spacing=1.0, shape=(15, 15, 15))
    centres = grid.centers()
    c1 = np.array([4.0, 4.0, 4.0])
    c2 = np.array([10.0, 9.0, 8.0])
    quad1 = np.sum((centres - c1) ** 2, axis=1) / (2.0 * 1.5**2)
    quad2 = np.sum((centres - c2) ** 2, axis=1) / (2.0 * 1.5**2)
    density = (np.exp(-quad1) + 0.5 * np.exp(-quad2)).reshape(grid.shape)
    return grid, density, np.stack([c1, c2])


def _brute_ball_voxels(grid: VoxelGrid, centres: np.ndarray, radius: float) -> np.ndarray:
    points = grid.centers()
    distance = np.linalg.norm(points[:, None, :] - centres[None, :, :], axis=2)
    return np.flatnonzero(np.min(distance, axis=1) <= radius).astype(np.int64)


def test_prune_support_union_of_balls() -> None:
    grid, density, centres = _two_ball_density()
    result = prune_support(density, grid, radius=2.0)
    expected = _brute_ball_voxels(grid, centres, 2.0)
    assert result.dtype == np.int64
    assert result.shape == (66,)
    np.testing.assert_array_equal(result, expected)


def test_prune_support_relative_threshold_keeps_one_ball() -> None:
    grid, density, centres = _two_ball_density()
    result = prune_support(density, grid, radius=2.0, rel_threshold=0.6)
    expected = _brute_ball_voxels(grid, centres[:1], 2.0)
    assert result.shape == (33,)
    np.testing.assert_array_equal(result, expected)


def test_prune_support_cap_keeps_top_density() -> None:
    grid, density, centres = _two_ball_density()
    union = _brute_ball_voxels(grid, centres, 2.0)
    order = np.lexsort((union, -density.ravel()[union]))
    expected = np.sort(union[order[:40]])
    result = prune_support(density, grid, radius=2.0, cap=40)
    assert result.shape == (40,)
    np.testing.assert_array_equal(result, expected)


def test_prune_support_validation() -> None:
    grid, density, _ = _two_ball_density()
    empty = prune_support(np.zeros(grid.shape), grid)
    assert empty.shape == (0,)
    assert empty.dtype == np.int64

    with pytest.raises(ValueError, match="shape"):
        prune_support(np.zeros((4, 4, 4)), grid)
    bad = density.copy()
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        prune_support(bad, grid)
    with pytest.raises(ValueError, match="cap"):
        prune_support(density, grid, cap=0)
    with pytest.raises(ValueError, match="cap"):
        prune_support(density, grid, cap=True)
    with pytest.raises(ValueError, match="radius"):
        prune_support(density, grid, radius=-1.0)
    with pytest.raises(ValueError, match="rel_threshold"):
        prune_support(density, grid, rel_threshold=1.0)
    with pytest.raises(ValueError, match="VoxelGrid"):
        prune_support(density, np.zeros(grid.shape))


def test_support_edges_matches_brute_force() -> None:
    grid = VoxelGrid(origin=np.zeros(3), spacing=1.0, shape=(6, 5, 4))
    rng = np.random.default_rng(0)
    indices = np.sort(rng.choice(grid.size, size=60, replace=False)).astype(np.int64)
    result = support_edges(grid, indices)

    multi = np.column_stack(np.unravel_index(indices, grid.shape))
    pairs: list[tuple[int, int]] = []
    for i in range(indices.size):
        for j in range(i + 1, indices.size):
            if int(np.abs(multi[i] - multi[j]).sum()) == 1:
                pairs.append((i, j))
    expected = np.asarray(sorted(pairs), dtype=np.int64).reshape(-1, 2)
    np.testing.assert_array_equal(result, expected)
    assert result.dtype == np.int64

    with pytest.raises(ValueError, match="sorted"):
        support_edges(grid, indices[::-1])
    empty = support_edges(grid, np.zeros(0, dtype=np.int64))
    assert empty.shape == (0, 2)


def test_support_points_and_map_round_trip() -> None:
    grid = VoxelGrid(origin=(-1.0, 2.0, 0.5), spacing=0.5, shape=(5, 6, 7))
    rng = np.random.default_rng(1)
    indices = np.sort(rng.choice(grid.size, size=40, replace=False)).astype(np.int64)
    points = support_points(grid, indices)
    np.testing.assert_allclose(points, grid.centers()[indices], rtol=0, atol=1e-12)

    values = np.arange(indices.size, dtype=np.float64)
    dense = support_to_map(values, grid, indices)
    assert dense.shape == grid.shape
    np.testing.assert_array_equal(dense.ravel()[indices], values)
    assert float(dense.sum()) == float(values.sum())

    with pytest.raises(ValueError, match="shape"):
        support_to_map(np.zeros(indices.size + 1), grid, indices)
    with pytest.raises(ValueError, match="range"):
        support_points(grid, np.array([grid.size], dtype=np.int64))


# --- B. Correctness on a small dense problem ----------------------------------


@functools.lru_cache(maxsize=None)
def _dense_problem() -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(0)
    K = rng.random((60, 8)) ** 3
    x_true = np.zeros(8)
    x_true[[1, 4, 6]] = [2.0, 0.5, 1.0]
    b = 0.05
    y = (K @ x_true + b) * rng.exponential(size=60)
    return K, y, b


def _minimise(fun, jac) -> tuple[np.ndarray, float]:
    result = minimize(
        fun,
        np.full(8, 0.5),
        jac=jac,
        method="L-BFGS-B",
        bounds=[(0.0, None)] * 8,
        options=dict(ftol=1e-15, gtol=1e-12, maxiter=20000),
    )
    return result.x, float(result.fun)


@functools.lru_cache(maxsize=None)
def _kl_reference() -> tuple[np.ndarray, float]:
    K, y, b = _dense_problem()

    def fun(x: np.ndarray) -> float:
        mu = K @ x + b
        return float(np.sum(mu - y + y * np.log(y / mu)))

    def jac(x: np.ndarray) -> np.ndarray:
        mu = K @ x + b
        return K.T @ (1.0 - y / mu)

    return _minimise(fun, jac)


@functools.lru_cache(maxsize=None)
def _is_reference() -> tuple[np.ndarray, float]:
    K, y, b = _dense_problem()

    def fun(x: np.ndarray) -> float:
        mu = K @ x + b
        return float(np.sum(y / mu + np.log(mu)))

    def jac(x: np.ndarray) -> np.ndarray:
        mu = K @ x + b
        return K.T @ (1.0 / mu - y / mu**2)

    return _minimise(fun, jac)


@functools.lru_cache(maxsize=None)
def _fista_reference() -> tuple[np.ndarray, float]:
    K, y, b = _dense_problem()
    lam_max = float(np.max(K.T @ (y - b)))
    lam = 0.05

    def fun(x: np.ndarray) -> float:
        return 0.5 * float(np.sum((K @ x + b - y) ** 2)) + lam * lam_max * float(np.sum(x))

    def jac(x: np.ndarray) -> np.ndarray:
        return K.T @ (K @ x + b - y) + lam * lam_max

    return _minimise(fun, jac)


@functools.lru_cache(maxsize=None)
def _fista_tv_reference() -> tuple[np.ndarray, float]:
    K, y, b = _dense_problem()
    lipschitz = float(np.linalg.norm(K, 2) ** 2)
    lam_max = float(np.max(K.T @ (y - b)))
    lam = 0.03
    eps = 0.1 * lam_max / lipschitz
    lam_tv = 0.05 * lam_max

    def fun(x: np.ndarray) -> float:
        value = 0.5 * float(np.sum((K @ x + b - y) ** 2)) + lam * lam_max * float(np.sum(x))
        difference = x[:-1] - x[1:]
        magnitude = np.abs(difference)
        huber = np.where(magnitude <= eps, difference**2 / (2.0 * eps), magnitude - eps / 2.0)
        return value + lam_tv * float(np.sum(huber))

    def jac(x: np.ndarray) -> np.ndarray:
        gradient = K.T @ (K @ x + b - y) + lam * lam_max
        difference = x[:-1] - x[1:]
        contribution = lam_tv * np.clip(difference / eps, -1.0, 1.0)
        np.add.at(gradient, np.arange(7), contribution)
        np.add.at(gradient, np.arange(1, 8), -contribution)
        return gradient

    return _minimise(fun, jac)


def _assert_monotone(objective: np.ndarray, n_iter: int) -> None:
    assert objective.shape == (n_iter + 1,)
    scale = float(np.max(np.abs(objective)))
    assert np.all(np.diff(objective) <= 1e-12 * scale)


def test_kl_em_matches_reference() -> None:
    K, y, b = _dense_problem()
    x_ref, f_ref = _kl_reference()
    solution = kl_em(aslinearoperator(K), y, b, n_iter=3000)
    assert isinstance(solution, PowerSolution)
    assert solution.objective[-1] <= f_ref + 1e-9 * abs(f_ref)
    assert float(np.max(np.abs(solution.x - x_ref))) <= 1e-6
    _assert_monotone(solution.objective, 3000)
    assert solution.x.dtype == np.float64
    assert np.all(solution.x >= 0.0)


def test_is_mlem_matches_reference() -> None:
    K, y, b = _dense_problem()
    x_ref, f_ref = _is_reference()
    solution = is_mlem(aslinearoperator(K), y, b, n_iter=3000)
    assert solution.objective[-1] <= f_ref + 1e-9 * abs(f_ref)
    assert float(np.max(np.abs(solution.x - x_ref))) <= 1e-6
    _assert_monotone(solution.objective, 3000)
    assert np.all(solution.x >= 0.0)


def test_nn_fista_l1_matches_reference() -> None:
    K, y, b = _dense_problem()
    x_ref, f_ref = _fista_reference()
    solution = nn_fista_l1(aslinearoperator(K), y, b, lam=0.05, n_iter=500)
    assert solution.objective[-1] <= f_ref + 1e-9 * abs(f_ref)
    assert float(np.max(np.abs(solution.x - x_ref))) <= 1e-6
    _assert_monotone(solution.objective, 500)


def test_nn_fista_l1_tv_matches_reference() -> None:
    K, y, b = _dense_problem()
    x_ref, f_ref = _fista_tv_reference()
    edges = np.asarray([(i, i + 1) for i in range(7)], dtype=np.int64)
    solution = nn_fista_l1(
        aslinearoperator(K),
        y,
        b,
        tv=True,
        edges=edges,
        tv_weight=0.05,
        tv_eps=0.1,
        lipschitz=float(np.linalg.norm(K, 2) ** 2),
        n_iter=3000,
    )
    assert solution.objective[-1] <= f_ref + 1e-9 * abs(f_ref)
    assert float(np.max(np.abs(solution.x - x_ref))) <= 1e-6
    _assert_monotone(solution.objective, 3000)


def test_objectives_match_direct_formulas() -> None:
    y = np.array([0.0, 2.0, 5.0])
    mu = np.array([1.0, 4.0, 2.0])
    expected_kl = float(np.sum([1.0, 4.0 - 2.0 + 2.0 * np.log(0.5), 2.0 - 5.0 + 5.0 * np.log(2.5)]))
    assert kl_objective(y, mu) == pytest.approx(expected_kl, rel=1e-12)
    expected_is = float(np.sum([0.0 / 1.0 + np.log(1.0), 0.5 + np.log(4.0), 2.5 + np.log(2.0)]))
    assert is_objective(y, mu) == pytest.approx(expected_is, rel=1e-12)


def test_single_iterations_match_the_pinned_updates() -> None:
    K, y, b = _dense_problem()
    op = aslinearoperator(K)
    x0 = np.linspace(0.2, 1.6, 8)
    mu0 = K @ x0 + b
    sens = K.T @ np.ones(60)

    kl_step = kl_em(op, y, b, x0=x0, n_iter=1)
    np.testing.assert_allclose(kl_step.x, x0 * (K.T @ (y / mu0)) / sens, rtol=1e-12)
    assert kl_step.objective[0] == pytest.approx(kl_objective(y, mu0), rel=1e-12)

    # Fevotte-Idier: the exponent 1/2 is what makes the IS update monotone.
    is_step = is_mlem(op, y, b, x0=x0, n_iter=1)
    factor = (K.T @ (y / mu0**2)) / (K.T @ (1.0 / mu0))
    np.testing.assert_allclose(is_step.x, x0 * np.sqrt(factor), rtol=1e-12)
    assert is_step.objective[1] == pytest.approx(is_objective(y, K @ is_step.x + b), rel=1e-12)

    lipschitz = 2.0 * float(np.linalg.norm(K, 2) ** 2)
    lam_max = float(np.max(K.T @ (y - b)))
    fista_step = nn_fista_l1(op, y, b, lam=0.05, x0=x0, lipschitz=lipschitz, n_iter=1)
    z = np.maximum(x0 - (K.T @ (mu0 - y) + 0.05 * lam_max) / lipschitz, 0.0)
    np.testing.assert_allclose(fista_step.x, z, rtol=1e-12, atol=1e-15)


# --- C. Operator contract and call counts -------------------------------------


class _CountingOperator:
    """Minimal duck-typed operator exposing only ``shape``, ``matvec``, ``rmatvec``."""

    def __init__(self, matrix: np.ndarray) -> None:
        self._matrix = matrix
        self.shape = matrix.shape
        self.dtype = matrix.dtype
        self.forward = 0
        self.adjoint = 0

    def matvec(self, vector: np.ndarray) -> np.ndarray:
        self.forward += 1
        return self._matrix @ np.asarray(vector, dtype=np.float64)

    def rmatvec(self, vector: np.ndarray) -> np.ndarray:
        self.adjoint += 1
        return self._matrix.T @ np.asarray(vector, dtype=np.float64)

    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"unexpected attribute access {name!r}")


def _counted() -> tuple[_CountingOperator, np.ndarray, float]:
    K, y, b = _dense_problem()
    return _CountingOperator(K), y, b


def test_call_counts() -> None:
    op, y, b = _counted()
    solution = kl_em(op, y, b, n_iter=5)
    assert (solution.n_forward, solution.n_adjoint) == (6, 6)
    assert (op.forward, op.adjoint) == (6, 6)

    op, y, b = _counted()
    solution = is_mlem(op, y, b, n_iter=5)
    assert (solution.n_forward, solution.n_adjoint) == (6, 11)
    assert (op.forward, op.adjoint) == (6, 11)

    op, y, b = _counted()
    solution = nn_fista_l1(op, y, b, n_iter=5)
    assert (solution.n_forward, solution.n_adjoint) == (35, 36)
    assert (op.forward, op.adjoint) == (35, 36)

    K, _, _ = _dense_problem()
    lipschitz = float(np.linalg.norm(K, 2) ** 2)
    op, y, b = _counted()
    solution = nn_fista_l1(op, y, b, n_iter=5, lipschitz=lipschitz)
    assert (solution.n_forward, solution.n_adjoint) == (5, 6)
    assert (op.forward, op.adjoint) == (5, 6)

    op, y, b = _counted()
    solution = nn_fista_l1(op, y, b, n_iter=5, lipschitz=lipschitz, x0=np.full(8, 0.1))
    assert (solution.n_forward, solution.n_adjoint) == (6, 6)
    assert (op.forward, op.adjoint) == (6, 6)


def test_zero_iterations_returns_start_point() -> None:
    op, y, b = _counted()
    solution = kl_em(op, y, b, n_iter=0)
    assert solution.n_iter == 0
    assert solution.objective.shape == (1,)
    K, _, _ = _dense_problem()
    sens = K.T @ np.ones(60)
    scale = (np.sum(y) - 60 * b) / np.sum(sens)
    np.testing.assert_allclose(solution.x, np.full(8, scale), rtol=1e-12)
    np.testing.assert_allclose(is_mlem(op, y, b, n_iter=0).x, np.full(8, scale), rtol=1e-12)

    op, y, b = _counted()
    start = np.full(8, 0.3)
    solution = nn_fista_l1(op, y, b, n_iter=0, lipschitz=1.0, x0=start)
    assert solution.objective.shape == (1,)
    np.testing.assert_array_equal(solution.x, start)


def test_operators_are_duck_typed() -> None:
    op, y, b = _counted()
    solution = kl_em(op, y, b, n_iter=2)
    assert np.all(np.isfinite(solution.x))
    assert SOLVERS == ("kl_em", "is_mlem", "nn_fista_l1")
    assert DEFAULT_CAP == 50_000
    assert DEFAULT_RADIUS_M == 2.0
    assert DEFAULT_REL_THRESHOLD == 1e-2


# --- D. Validation ------------------------------------------------------------


def test_validation_rejects_bad_data() -> None:
    K, y, b = _dense_problem()
    op = aslinearoperator(K)

    with pytest.raises(ValueError, match="y"):
        kl_em(op, -y, b)
    with pytest.raises(ValueError, match="finite"):
        kl_em(op, np.where(np.arange(60) == 0, np.nan, y), b)
    with pytest.raises(ValueError, match="real"):
        kl_em(op, y.astype(np.complex128), b)
    with pytest.raises(ValueError, match="entries"):
        kl_em(op, y[:-1], b)
    with pytest.raises(ValueError, match="background"):
        kl_em(op, y, 0.0)
    with pytest.raises(ValueError, match="background"):
        is_mlem(op, y, -0.05)
    with pytest.raises(ValueError, match="background"):
        nn_fista_l1(op, y, -0.05)
    with pytest.raises(ValueError, match="x0"):
        kl_em(op, y, b, x0=np.full(8, -1.0))
    with pytest.raises(ValueError, match="x0"):
        kl_em(op, y, b, x0=np.zeros(7))
    with pytest.raises(ValueError, match="n_iter"):
        kl_em(op, y, b, n_iter=-1)
    with pytest.raises(ValueError, match="n_iter"):
        kl_em(op, y, b, n_iter=True)
    with pytest.raises(ValueError, match="tol"):
        kl_em(op, y, b, tol=-1.0)
    with pytest.raises(ValueError, match="edges"):
        nn_fista_l1(op, y, b, tv=True)
    with pytest.raises(ValueError, match="edges"):
        nn_fista_l1(op, y, b, edges=np.array([[0, 1]], dtype=np.int64))
    with pytest.raises(ValueError, match="edges"):
        nn_fista_l1(op, y, b, tv=True, edges=np.array([[0, 8]], dtype=np.int64))
    with pytest.raises(ValueError, match="lipschitz"):
        nn_fista_l1(op, y, b, lipschitz=-1.0)
    with pytest.raises(ValueError, match="power_iters"):
        nn_fista_l1(op, y, b, power_iters=0)


# --- E. L0 two-point recovery (no inverse crime) ------------------------------


def _resolved(density: np.ndarray, grid: VoxelGrid, gt: np.ndarray) -> bool:
    return _peak_match(density, grid, gt).tp == 2


def _peak_match(density: np.ndarray, grid: VoxelGrid, gt: np.ndarray) -> Matching:
    peaks = nms_peaks(
        density,
        grid,
        1.5,
        refine=True,
        min_value=0.1 * float(density.max()),
        max_peaks=2,
    )
    return match(peaks.positions, gt, 1.0)


@functools.lru_cache(maxsize=None)
def _l0_problem(seed: int, axis: str, looks: int):
    geom = synthetic.ring_geometry(num_views=8, num_bins=32)
    grid = synthetic.default_grid(size=12.0, spacing=0.5)
    phantom = synthetic.l0b_pair(seed, 3.0, axis, geom=geom, grid=grid, box_size=4.0)
    gt = phantom.gt.points_pos
    noise_var = float(np.median(capture_power(phantom.y_clean))) / 100.0
    rng = np.random.default_rng(np.random.SeedSequence([seed, 7]))

    def noisy(Y: np.ndarray) -> np.ndarray:
        return Y + np.sqrt(noise_var / 2.0) * (
            rng.standard_normal(Y.shape) + 1j * rng.standard_normal(Y.shape)
        )

    if looks == 0:
        obs = extract(noisy(phantom.y_clean), "ID").data
    else:
        obs = 0.0
        for _ in range(looks):
            Y = atom_cfr(gt, np.exp(2j * np.pi * rng.random(2)), geom, "bv")
            obs = obs + extract(noisy(Y), "ID").data / looks
    e1 = power_backproject_grid(obs, geom, grid, space="bv", product="ID")
    support = prune_support(e1, grid, radius=3.0, cap=50_000)
    op = power_operator(support_points(grid, support), geom, "bv", "ID")
    background = noise_floor("ID", geom, noise_var)
    return geom, grid, gt, obs, e1, support, op, background


def _check_run(
    name: str,
    density: np.ndarray,
    grid: VoxelGrid,
    gt: np.ndarray,
    solution: PowerSolution,
) -> None:
    _assert_monotone(solution.objective, solution.n_iter)
    assert solution.objective[-1] < solution.objective[0]
    matched = _peak_match(density, grid, gt)
    assert matched.tp == 2, f"{name}: only {matched.tp} of 2 points recovered"
    print(f"\nE. {name}: matched distances {matched.distance} (max {matched.distance.max():.3f} m)")
    assert np.all(solution.x >= 0.0)


@pytest.mark.parametrize("solver", ["kl_em", "is_mlem"])
@pytest.mark.parametrize(("seed", "axis"), [(4, "range"), (3, "cross_range")])
def test_l0_em_recovery(solver: str, seed: int, axis: str) -> None:
    _, grid, gt, obs, e1, support, op, background = _l0_problem(seed, axis, 0)
    assert not _resolved(e1, grid, gt)
    run = kl_em if solver == "kl_em" else is_mlem
    solution = run(op, obs, background)
    density = support_to_map(solution.x, grid, support)
    _check_run(f"{solver} seed={seed} axis={axis}", density, grid, gt, solution)


@pytest.mark.parametrize("tv", [False, True])
@pytest.mark.parametrize(("seed", "axis"), [(4, "cross_range"), (0, "cross_range")])
def test_l0_fista_recovery(tv: bool, seed: int, axis: str) -> None:
    _, grid, gt, obs, e1, support, op, background = _l0_problem(seed, axis, 8)
    assert not _resolved(e1, grid, gt)
    edges = support_edges(grid, support) if tv else None
    solution = nn_fista_l1(op, obs, background, tv=tv, edges=edges)
    density = support_to_map(solution.x, grid, support)
    _check_run(f"nn_fista_l1 tv={tv} seed={seed} axis={axis}", density, grid, gt, solution)


# --- F. Benchmark -------------------------------------------------------------


@pytest.mark.skipif(os.environ.get("RF_TOMO_BENCH") != "1", reason="opt-in benchmark")
def test_benchmark_power_solvers() -> None:
    geom = synthetic.ring_geometry(num_views=16, num_bins=128)
    rng = np.random.default_rng(0)
    count = 50_000
    points = np.column_stack(
        [
            rng.uniform(-50.0, 50.0, count),
            rng.uniform(-50.0, 50.0, count),
            rng.uniform(-2.0, 40.0, count),
        ]
    )
    start = time.perf_counter()
    op = power_operator(points, geom, "bv", "ID")
    print(f"\npower_operator P={count} V=16 B=1 N=128: init {time.perf_counter() - start:.1f} s")
    y = op.matvec(rng.random(count)) + 1.0

    for name, run in (
        ("kl_em", lambda: kl_em(op, y, 1e-3, n_iter=10)),
        ("is_mlem", lambda: is_mlem(op, y, 1e-3, n_iter=10)),
        ("nn_fista_l1", lambda: nn_fista_l1(op, y, 1e-3, n_iter=10)),
    ):
        start = time.perf_counter()
        solution = run()
        elapsed = time.perf_counter() - start
        pairs = 0.5 * (solution.n_forward + solution.n_adjoint)
        per_pair = elapsed / pairs
        print(
            f"{name}: {elapsed:.1f} s for {solution.n_forward} K + {solution.n_adjoint} K^T, "
            f"{per_pair:.2f} s per pair, 200 pairs ~ {200.0 * per_pair:.0f} s (target 120 s)"
        )
        assert np.all(np.isfinite(solution.x))
        assert np.all(solution.x >= 0.0)
