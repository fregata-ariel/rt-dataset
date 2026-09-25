"""Tests for the E2 coherent solvers (T13)."""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

from plateau_rt.domain.rf_tomography.forward_exact import atom_cfr, dense_matrix
from plateau_rt.domain.rf_tomography.forward_sep import (
    SeparableOperator,
    incidence_cosine,
    project_shared_phase,
)
from plateau_rt.domain.rf_tomography.geometry import VoxelGrid, mirror_point
from plateau_rt.domain.rf_tomography.metrics import Peaks, match, nms_peaks
from plateau_rt.domain.rf_tomography.solvers.coherent import (
    complex_l1_fista,
    lambda_max,
    lipschitz_constant,
    mmv_group_lasso,
    point_density,
    roi_grids_from_detections,
    roi_points,
    tikhonov_lsqr,
)
from plateau_rt.domain.rf_tomography.sync import apply_gauge
from plateau_rt.domain.rf_tomography.synthetic import l0c_random, ring_geometry


@pytest.fixture(scope="module")
def small_geom():
    """Small 3-view, 8-bin, 4x4-aperture geometry."""
    return ring_geometry(num_views=3, num_bins=8, aperture_shape=(4, 4))


@pytest.fixture(scope="module")
def small_data(small_geom):
    """30 random BV points, 3 source atoms and their CFR."""
    rng = np.random.default_rng(np.random.SeedSequence([13, 0]))
    points = np.array([0.0, 0.0, 5.0]) + rng.uniform(-3.0, 3.0, size=(30, 3))
    sources = np.array([0.0, 0.0, 5.0]) + rng.uniform(-3.0, 3.0, size=(3, 3))
    y = atom_cfr(sources, np.array([1.0, 0.5j, -0.3]), small_geom, "bv")
    return points, sources, y


@pytest.fixture(scope="module")
def shared_op(small_geom, small_data):
    """Shared-β operator on the 30-point fixture."""
    points, _, _ = small_data
    return SeparableOperator(points, small_geom, "bv", beta_model="shared")


@pytest.fixture(scope="module")
def per_view_op(small_geom, small_data):
    """Per-view-β operator on the 30-point fixture."""
    points, _, _ = small_data
    return SeparableOperator(points, small_geom, "bv", beta_model="per_view")


def _noisy(y):
    """Add complex noise at 5 % of the rms sample amplitude to ``y``."""
    rng = np.random.default_rng(np.random.SeedSequence([13, 1]))
    y = np.asarray(y, dtype=np.complex128)
    scale = 0.05 * np.linalg.norm(y) / np.sqrt(y.size)
    return y + scale * (rng.standard_normal(y.shape) + 1j * rng.standard_normal(y.shape))


@pytest.mark.parametrize(
    ("beta_model", "damp_factor"),
    [("shared", 0.0), ("shared", 1e-2), ("shared", 1e-1), ("per_view", 1e-2)],
)
def test_tikhonov_lsqr_matches_dense_solve(small_geom, small_data, beta_model, damp_factor):
    points, _, y = small_data
    points20 = points[:20]
    op = SeparableOperator(points20, small_geom, "bv", beta_model=beta_model)
    noisy = _noisy(y)
    if beta_model == "shared":
        dense = dense_matrix(points20, small_geom, "bv")
    else:
        dense = np.column_stack([op.matvec(e) for e in np.eye(op.shape[1], dtype=np.complex128)])
    s_max = np.linalg.norm(dense, 2)
    damp = damp_factor * s_max
    b = noisy.reshape(-1)
    normal = dense.conj().T @ dense + damp**2 * np.eye(dense.shape[1])
    x_ref = np.linalg.solve(normal, dense.conj().T @ b)

    result = tikhonov_lsqr(op, noisy, damp)

    rel = np.linalg.norm(result.x.ravel() - x_ref) / np.linalg.norm(x_ref)
    assert rel <= 1e-6
    residual = dense @ result.x.ravel() - b
    objective = 0.5 * (
        float(np.sum(np.abs(residual) ** 2)) + damp**2 * float(np.sum(np.abs(result.x) ** 2))
    )
    assert abs(result.objective[-1] - objective) <= 1e-8 * abs(objective)
    assert result.converged
    assert result.x.shape == op.x_shape


def test_lipschitz_constant_bounds(small_geom, small_data, shared_op):
    points, _, _ = small_data
    dense = dense_matrix(points, small_geom, "bv")
    l_true = np.linalg.norm(dense, 2) ** 2
    assert l_true <= lipschitz_constant(shared_op)
    assert lipschitz_constant(shared_op) <= 1.05 * l_true * (1.0 + 1e-9)
    assert abs(lipschitz_constant(shared_op, safety=1.0) - l_true) <= 1e-6 * l_true


def test_complex_l1_fista_satisfies_kkt(shared_op, small_data):
    _, _, y = small_data
    lam = 0.2 * lambda_max(shared_op, y, group=False)
    result = complex_l1_fista(shared_op, y, lam, n_iter=3000, tol=0.0)
    x = result.x
    gradient = shared_op.adjoint(y - shared_op.forward(x))
    support = np.abs(x) > 1e-12 * np.max(np.abs(x))
    assert np.any(support)
    on_support = gradient[support] - lam * x[support] / np.abs(x[support])
    assert np.max(np.abs(on_support)) <= 1e-6 * lam
    assert np.max(np.abs(gradient[~support])) <= lam * (1.0 + 1e-6)
    assert np.all(np.diff(result.objective) <= 0.0)


def test_mmv_group_lasso_satisfies_kkt_per_view(per_view_op, small_data):
    _, _, y = small_data
    lam = 0.2 * lambda_max(per_view_op, y)
    result = mmv_group_lasso(per_view_op, y, lam, n_iter=3000, tol=0.0)
    x = result.x
    gradient = per_view_op.adjoint(y - per_view_op.forward(x))
    row_norms = np.sqrt(np.sum(np.abs(x) ** 2, axis=(1, 2)))
    gradient_norms = np.sqrt(np.sum(np.abs(gradient) ** 2, axis=(1, 2)))
    active = row_norms > 1e-12 * np.max(row_norms)
    assert np.any(active)
    assert np.any(~active)
    direction = x[active] / row_norms[active][:, None, None]
    assert np.max(np.abs(gradient[active] - lam * direction)) <= 1e-6 * lam
    assert np.max(gradient_norms[~active]) <= lam * (1.0 + 1e-6)
    assert np.all(np.diff(result.objective) <= 0.0)


def test_lambda_max_gives_zero_solution(shared_op, per_view_op, small_data):
    _, _, y = small_data
    l1_scale = lambda_max(shared_op, y, group=False)
    zero_l1 = complex_l1_fista(shared_op, y, 1.0001 * l1_scale, n_iter=50)
    assert np.all(zero_l1.x == 0)
    assert zero_l1.converged
    assert np.any(complex_l1_fista(shared_op, y, 0.9 * l1_scale, n_iter=50).x != 0)

    group_scale = lambda_max(per_view_op, y)
    zero_mmv = mmv_group_lasso(per_view_op, y, 1.0001 * group_scale, n_iter=50)
    assert np.all(zero_mmv.x == 0)
    assert zero_mmv.converged
    assert np.any(mmv_group_lasso(per_view_op, y, 0.9 * group_scale, n_iter=50).x != 0)


def test_mmv_shared_equals_complex_l1(shared_op, small_data):
    _, _, y = small_data
    lam = 0.05 * lambda_max(shared_op, y)
    l1 = complex_l1_fista(shared_op, y, lam, n_iter=100, tol=0.0)
    mmv = mmv_group_lasso(shared_op, y, lam, n_iter=100, tol=0.0)
    assert np.linalg.norm(l1.x - mmv.x) <= 1e-12 * np.linalg.norm(l1.x)
    assert np.max(np.abs(l1.objective - mmv.objective)) <= 1e-12 * np.max(np.abs(l1.objective))


def test_call_counts(small_geom, small_data, shared_op):
    points, _, y = small_data
    lam = 0.05 * lambda_max(shared_op, y, group=False)
    counted = complex_l1_fista(shared_op, y, lam, n_iter=10, tol=0.0, power_iterations=5)
    assert counted.n_forward == 15
    assert counted.n_adjoint == 15
    assert counted.n_iter == 10
    assert len(counted.objective) == 11

    fixed = complex_l1_fista(shared_op, y, lam, n_iter=10, tol=0.0, lipschitz=1.0)
    assert fixed.n_forward == 10
    assert fixed.n_adjoint == 10

    x0 = np.zeros(shared_op.x_shape, dtype=np.complex128)
    warm = complex_l1_fista(shared_op, y, lam, n_iter=10, tol=0.0, lipschitz=1.0, x0=x0)
    assert warm.n_forward == 11

    constrained = SeparableOperator(points, small_geom, "vs", beta_model="constrained")
    with pytest.raises(ValueError):
        complex_l1_fista(constrained, y, lam)
    with pytest.raises(ValueError):
        complex_l1_fista(shared_op, np.zeros((2, 2)), lam)
    with pytest.raises(ValueError):
        complex_l1_fista(shared_op, y, -1.0)
    with pytest.raises(ValueError):
        tikhonov_lsqr(shared_op, y, -1.0)


@pytest.fixture(scope="module")
def l0c_geometry():
    """8-view, 16-bin geometry for the L0c support test."""
    return ring_geometry(num_views=8, num_bins=16)


@pytest.fixture(scope="module")
def l0c_grid():
    """11^3 × 1 m support grid around the scene centre (not the phantom grid)."""
    centre = np.array([0.0, 0.0, 5.0])
    return VoxelGrid.from_bounds(centre - 5.0, centre + 5.0, 1.0)


@pytest.fixture(scope="module")
def l0c_per_view_op(l0c_geometry, l0c_grid):
    """Per-view operator on the L0c support grid."""
    return SeparableOperator(l0c_grid.centers(), l0c_geometry, "bv", beta_model="per_view")


@pytest.fixture(scope="module")
def l0c_shared_op(l0c_geometry, l0c_grid):
    """Shared operator on the L0c support grid (negative control)."""
    return SeparableOperator(l0c_grid.centers(), l0c_geometry, "bv", beta_model="shared")


def _l0c_data(seed, geom):
    """Render an L0c K=4 phantom with random per-view phases and noise."""
    phantom = l0c_random(seed, 4, geom=geom)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 99]))
    phi = rng.uniform(0.0, 2.0 * np.pi, size=(geom.num_views, 1))
    y = apply_gauge(phantom.y_clean, phi, np.zeros((geom.num_views, 1)), geom.freq_offsets)
    sigma2 = 1e-3 * np.mean(np.abs(phantom.y_clean) ** 2)
    y = y + np.sqrt(sigma2 / 2.0) * (
        rng.standard_normal(y.shape) + 1j * rng.standard_normal(y.shape)
    )
    return phantom, y


@pytest.mark.parametrize("seed", [9, 12, 17])
def test_mmv_recovers_l0c_support_with_random_view_phases(
    seed, l0c_geometry, l0c_grid, l0c_per_view_op
):
    phantom, y = _l0c_data(seed, l0c_geometry)
    res = mmv_group_lasso(l0c_per_view_op, y, 0.01 * lambda_max(l0c_per_view_op, y))
    density = point_density(res.x, "per_view").reshape(l0c_grid.shape)
    peaks = nms_peaks(density, l0c_grid, radius=1.5, min_value=0.0, max_peaks=4)
    assert match(peaks.positions, phantom.gt.points_pos, 1.0).tp == 4


def test_shared_model_fails_with_random_view_phases(l0c_geometry, l0c_grid, l0c_shared_op):
    phantom, y = _l0c_data(12, l0c_geometry)
    res = mmv_group_lasso(l0c_shared_op, y, 0.01 * lambda_max(l0c_shared_op, y))
    density = point_density(res.x, "shared").reshape(l0c_grid.shape)
    peaks = nms_peaks(density, l0c_grid, radius=1.5, min_value=0.0, max_peaks=4)
    assert match(peaks.positions, phantom.gt.points_pos, 1.0).tp <= 2


def test_constrained_mmv_keeps_shared_phase():
    geom = ring_geometry(num_views=6, num_bins=16, aperture_shape=(4, 4))
    bs = geom.bs_pos[0]
    planes = [((0, 0, 0), (0, 0, 1)), ((0, 12, 0), (0, -1, 0)), ((0, -12, 0), (0, 1, 0))]
    vs = np.stack([mirror_point(bs, np.array(p, float), np.array(n, float)) for p, n in planes])
    rng = np.random.default_rng(np.random.SeedSequence([7]))
    coef = rng.standard_normal((3, 3))
    psi = rng.uniform(0.0, 2.0 * np.pi, 3)
    amps = np.zeros((3, geom.num_views, geom.num_bs), dtype=np.complex128)
    for v in range(geom.num_views):
        c = incidence_cosine(vs, geom, v, 0)
        amps[:, v, 0] = np.exp(1j * psi) * (coef[:, 0] + coef[:, 1] * c + coef[:, 2] * c**2)
    y = atom_cfr(vs, amps, geom, "vs")
    offsets = 1.5 * np.eye(3)
    support = np.vstack([vs, (vs[:, None, :] + offsets[None]).reshape(-1, 3)])

    constrained = SeparableOperator(support, geom, "vs", beta_model="constrained")
    per_view = SeparableOperator(support, geom, "vs", beta_model="per_view")
    for op in (constrained, per_view):
        res = mmv_group_lasso(op, y, 1e-4 * lambda_max(op, y), n_iter=500)
        residual = np.linalg.norm(op.forward(res.x) - y) / np.linalg.norm(y)
        assert residual <= 2e-2
        assert np.all(np.diff(res.objective) <= 0.0)
        if op.beta_model == "constrained":
            projected = project_shared_phase(res.x)
            assert np.max(np.abs(projected - res.x)) <= 1e-12 * np.max(np.abs(res.x))

    phi = rng.uniform(0.0, 2.0 * np.pi, (geom.num_views, geom.num_bs))
    y_r = apply_gauge(y, phi, np.zeros((geom.num_views, geom.num_bs)), geom.freq_offsets)
    res_c = mmv_group_lasso(constrained, y_r, 1e-4 * lambda_max(constrained, y_r), n_iter=500)
    r_c = np.linalg.norm(constrained.forward(res_c.x) - y_r) / np.linalg.norm(y_r)
    assert r_c >= 0.1
    res_p = mmv_group_lasso(per_view, y_r, 1e-4 * lambda_max(per_view, y_r), n_iter=500)
    r_p = np.linalg.norm(per_view.forward(res_p.x) - y_r) / np.linalg.norm(y_r)
    assert r_p <= 1e-2


def test_roi_grids_from_detections():
    wl = ring_geometry().wavelength
    det = np.array([[1.0, 2.0, 3.0], [-4.0, 0.0, 7.5]])
    grids = roi_grids_from_detections(det, wavelength=wl)
    assert len(grids) == 2
    assert grids[0].spacing == pytest.approx(wl / 4.0, abs=1e-15)
    assert grids[0].shape == (25, 25, 25)
    for grid, position in zip(grids, det, strict=True):
        index = np.ravel_multi_index((12, 12, 12), grid.shape)
        assert np.allclose(grid.centers()[index], position, rtol=0.0, atol=1e-12)
        assert (grid.shape[0] - 1) / 2 * grid.spacing >= 0.25

    peaks = Peaks(
        positions=det,
        values=np.array([1.0, 2.0]),
        indices=np.array([0, 1]),
        offsets=np.zeros((2, 3)),
    )
    peak_grids = roi_grids_from_detections(peaks, wavelength=wl)
    for grid, peak_grid in zip(grids, peak_grids, strict=True):
        assert np.array_equal(grid.centers(), peak_grid.centers())

    assert roi_grids_from_detections(det, half_width=0.5, wavelength=wl)[0].shape == (49, 49, 49)
    assert roi_grids_from_detections(det, spacing=0.1)[0].shape == (7, 7, 7)
    assert roi_grids_from_detections(np.zeros((0, 3))) == []

    with pytest.raises(ValueError):
        roi_grids_from_detections(det)
    with pytest.raises(ValueError):
        roi_grids_from_detections(det, spacing=0.0)
    with pytest.raises(ValueError):
        roi_grids_from_detections(det, half_width=0.0, wavelength=wl)
    with pytest.raises(ValueError):
        roi_grids_from_detections(det, wavelength=-1.0)
    with pytest.raises(ValueError):
        roi_grids_from_detections(np.array([[np.nan, 0.0, 0.0]]), wavelength=wl)
    with pytest.raises(ValueError):
        roi_grids_from_detections(np.zeros((3, 2)), wavelength=wl)

    points, owner = roi_points(grids)
    block = 25**3
    assert points.shape == (2 * block, 3)
    assert owner.shape == (2 * block,)
    assert np.all(owner[:block] == 0)
    assert np.all(owner[block:] == 1)
    empty_points, empty_owner = roi_points([])
    assert empty_points.shape == (0, 3)
    assert empty_owner.shape == (0,)
    assert empty_owner.dtype == np.int64


def test_point_density():
    shared = np.array([1.0 + 1.0j, 2.0])
    assert np.allclose(point_density(shared, "shared"), [2.0, 4.0])

    per_view = np.arange(1, 13, dtype=np.complex128).reshape(2, 3, 2)
    assert np.allclose(
        point_density(per_view, "per_view"), np.mean(np.abs(per_view) ** 2, axis=(1, 2))
    )

    constrained = np.arange(1, 7, dtype=np.complex128).reshape(2, 3)
    assert np.allclose(
        point_density(constrained, "constrained"), np.sum(np.abs(constrained) ** 2, axis=1)
    )

    with pytest.raises(ValueError):
        point_density(shared, "per_view")
    with pytest.raises(ValueError):
        point_density(np.zeros((2, 2, 2)), "shared")
    with pytest.raises(ValueError):
        point_density(shared, "unknown")


@pytest.mark.skipif(os.environ.get("RF_TOMO_BENCH") != "1", reason="benchmark opt-in")
def test_coherent_benchmark():
    geom = ring_geometry(num_views=16, radius=40.0, num_bins=128)
    rng = np.random.default_rng(np.random.SeedSequence([0]))
    points = np.column_stack(
        [
            rng.uniform(-50.0, 50.0, 50_000),
            rng.uniform(-50.0, 50.0, 50_000),
            rng.uniform(-2.0, 40.0, 50_000),
        ]
    )
    op = SeparableOperator(points, geom, "bv", beta_model="shared")
    y = rng.standard_normal(op.y_shape) + 1j * rng.standard_normal(op.y_shape)
    lam = 0.01 * lambda_max(op, y, group=False)
    start = time.perf_counter()
    result = complex_l1_fista(op, y, lam, n_iter=3, tol=0.0, lipschitz=1.0)
    elapsed = time.perf_counter() - start
    per_pair = elapsed / 3.0
    print(
        f"coherent benchmark: {per_pair:.3f} s per forward+adjoint pair; "
        f"200 pairs ~ {per_pair * 200.0 / 60.0:.1f} min"
    )
    assert np.all(np.isfinite(result.x))
