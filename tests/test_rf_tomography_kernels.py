"""Tests for the incoherent power kernels (T08)."""

from __future__ import annotations

import math
import os
import time

import numpy as np
import pytest
from scipy.constants import c as SPEED_OF_LIGHT

from plateau_rt.domain.rf_tomography import kernels
from plateau_rt.domain.rf_tomography.antenna import bs_orientation
from plateau_rt.domain.rf_tomography.forward_exact import atom_cfr, dense_matrix
from plateau_rt.domain.rf_tomography.geometry import (
    CaptureGeometry,
    VoxelGrid,
    planar_element_offsets,
    rotations_from_orientations,
)
from plateau_rt.domain.rf_tomography.kernels import (
    PRODUCT_NODES,
    PRODUCTS,
    PowerOperator,
    dirichlet_power,
    noise_floor,
    power_backproject_grid,
    power_operator,
    product_shape,
)
from plateau_rt.domain.rf_tomography.observables import extract

F_C = 3.5e9
BANDWIDTH = 100e6


def _make_geometry(
    ue_pos: np.ndarray,
    orientations: np.ndarray,
    bs_pos: np.ndarray,
    *,
    rows: int = 8,
    cols: int = 8,
    num_bins: int = 16,
    bs_look_at: np.ndarray | None = None,
) -> CaptureGeometry:
    """Build an exact-float64-grid capture geometry for the synthetic tests."""
    wavelength = SPEED_OF_LIGHT / F_C
    elem_offsets = planar_element_offsets(wavelength, rows=rows, cols=cols)
    df = BANDWIDTH / num_bins
    freq_offsets = (np.arange(num_bins) - num_bins // 2) * df
    bs_rot = None
    if bs_look_at is not None:
        targets = np.asarray(bs_look_at, dtype=np.float64)
        if targets.shape == (3,):
            targets = np.broadcast_to(targets, (bs_pos.shape[0], 3))
        bs_rot = np.stack([bs_orientation(bs_pos[b], targets[b]) for b in range(bs_pos.shape[0])])
    return CaptureGeometry(
        ue_pos=ue_pos,
        ue_rot=rotations_from_orientations(orientations),
        bs_pos=bs_pos,
        elem_offsets=elem_offsets,
        freq_offsets=freq_offsets,
        f_c=F_C,
        aperture_shape=(rows, cols),
        bs_rot=bs_rot,
    )


def _synthetic_geometry(
    *,
    rows: int = 8,
    cols: int = 8,
    num_bins: int = 16,
    look_at: bool = True,
    num_views: int = 4,
    num_bs: int = 2,
) -> CaptureGeometry:
    """Views with distinct yaw/pitch/roll and BSs (the last view faces away)."""
    ue_pos = np.array([[0.0, 0.0, 1.5], [5.0, -3.0, 1.6], [-4.0, 2.0, 1.4], [0.0, 0.0, 1.5]])[
        :num_views
    ]
    orientations = np.array(
        [[0.0, 0.0, 0.0], [0.6, 0.1, 0.2], [-1.0, -0.2, 0.3], [math.pi, 0.0, 0.0]]
    )[:num_views]
    bs_pos = np.array([[10.0, 0.0, 8.0], [-12.0, 4.0, 6.0]])[:num_bs]
    targets = np.array([[0.0, 0.0, 1.5], [2.0, 1.0, 1.0]])[:num_bs] if look_at else None
    return _make_geometry(
        ue_pos,
        orientations,
        bs_pos,
        rows=rows,
        cols=cols,
        num_bins=num_bins,
        bs_look_at=targets,
    )


def _quantised_geometry(num_views: int = 4, num_bs: int = 2, num_bins: int = 16) -> CaptureGeometry:
    """Same pose bank but with the float32-quantised frequency grid of Sionna."""
    ue_pos = np.array([[0.0, 0.0, 1.5], [5.0, -3.0, 1.6], [-4.0, 2.0, 1.4], [0.0, 0.0, 1.5]])[
        :num_views
    ]
    orientations = np.array(
        [[0.0, 0.0, 0.0], [0.6, 0.1, 0.2], [-1.0, -0.2, 0.3], [math.pi, 0.0, 0.0]]
    )[:num_views]
    bs_pos = np.array([[10.0, 0.0, 8.0], [-12.0, 4.0, 6.0]])[:num_bs]
    targets = np.array([[0.0, 0.0, 1.5], [2.0, 1.0, 1.0]])[:num_bs]
    return CaptureGeometry.from_orientations(
        ue_pos,
        orientations,
        bs_pos,
        f_c=F_C,
        bandwidth=BANDWIDTH,
        num_bins=num_bins,
        bs_look_at=targets,
    )


def _far_from_special(point: np.ndarray, geom: CaptureGeometry) -> bool:
    """True when ``point`` is at least 1 m from every UE and BS."""
    ue = np.min(np.linalg.norm(geom.ue_pos - point, axis=1))
    bs = np.min(np.linalg.norm(geom.bs_pos - point, axis=1))
    return ue > 1.0 and bs > 1.0


def _off_grid_points(geom: CaptureGeometry, rng: np.random.Generator, count: int) -> np.ndarray:
    """Return ``count`` seeded points in the scene box, at least 1 m from special points."""
    rows: list[np.ndarray] = []
    while len(rows) < count:
        candidate = rng.uniform([-20.0, -20.0, -5.0], [20.0, 20.0, 10.0])
        if _far_from_special(candidate, geom):
            rows.append(candidate)
    return np.asarray(rows, dtype=np.float64)


def _gauge_apply(
    Y: np.ndarray, geom: CaptureGeometry, tau: np.ndarray, phi: np.ndarray
) -> np.ndarray:
    """Return ``exp(1j phi) exp(-2j pi df_n tau) Y`` (the docs §2.4 gauge)."""
    phase = np.exp(1j * phi)[:, :, None, None, None, None]
    delay = np.exp(-2j * np.pi * tau[:, :, None] * geom.freq_offsets[None, None, :])[
        :, :, None, None, None, :
    ]
    return phase * delay * Y


def _max_relative_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def _dirichlet_direct(x: np.ndarray, length: int) -> np.ndarray:
    """Return the direct sum ``|sum_l exp(2j pi l x)|^2 / length`` with reduced phases."""
    phases = np.mod(np.outer(np.arange(length, dtype=np.float64), x), 1.0)
    return np.abs(np.exp(2j * np.pi * phases).sum(axis=0)) ** 2 / length


@pytest.mark.parametrize("length", [1, 4, 8, 16, 128])
def test_dirichlet_power_matches_direct_sum(length: int) -> None:
    rng = np.random.default_rng(np.random.SeedSequence(11))
    x = np.concatenate(
        [
            rng.uniform(-3.0, 3.0, size=64),
            np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]),
            np.array([-1.0, 0.0, 1.0]) + 1e-9,
            np.array([-1.0, 0.0, 1.0]) - 1e-9,
            np.array([-0.5, 0.5, 2.5, -3.5]),
            np.array([1e3 + 0.3, -1e3 - 0.7]),
        ]
    )
    got = dirichlet_power(x, length)
    assert got.shape == x.shape
    assert got.dtype == np.float64
    np.testing.assert_allclose(got, _dirichlet_direct(x, length), rtol=0.0, atol=1e-12 * length)
    assert np.all(got >= 0.0)

    integers = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(dirichlet_power(integers, length), np.full(7, length))

    # Far from the origin, on the kernel slopes, only the period reduction keeps full
    # precision (the unreduced sinc ratio is off by up to ~1e-8 * length here).
    far = np.array([1e4 + 1e-7, 2e4 + 0.25 / length, -1e6 - 0.3 / length, 3e6 + 1.7 / length])
    reduced = far - np.round(far)
    np.testing.assert_allclose(
        dirichlet_power(far, length),
        _dirichlet_direct(reduced, length),
        rtol=0.0,
        atol=1e-12 * length,
    )

    with pytest.raises(ValueError):
        dirichlet_power(x, 0)
    with pytest.raises(ValueError):
        dirichlet_power(x, True)


@pytest.mark.parametrize("space", ["vs", "bv"])
@pytest.mark.parametrize("product", PRODUCTS)
def test_single_point_matches_atom_cfr(space: str, product: str) -> None:
    geom = _synthetic_geometry()
    rng = np.random.default_rng(np.random.SeedSequence(21))
    points = _off_grid_points(geom, rng, 6)
    if space == "vs":
        points = np.vstack([points, geom.bs_pos[0]])
    amp = np.array([1.7])
    sigma = 1.7**2
    for point in points:
        operator = PowerOperator(point, geom, space, product)
        reference = extract(atom_cfr(point, amp, geom, space), PRODUCT_NODES[product]).data
        got = operator.forward(np.array([sigma]))
        assert got.shape == product_shape(product, geom)
        assert _max_relative_error(got, reference) <= 1e-10

    operator = PowerOperator(points[0], geom, space, "ID", pattern="tr38901", polarization="vv")
    reference = extract(
        atom_cfr(points[0], amp, geom, space, pattern="tr38901", polarization="vv"), "ID"
    ).data
    assert _max_relative_error(operator.forward(np.array([sigma])), reference) <= 1e-10


@pytest.mark.parametrize("space", ["vs", "bv"])
@pytest.mark.parametrize("product", PRODUCTS)
def test_gauge_tau_matches_and_invariance(space: str, product: str) -> None:
    geom = _synthetic_geometry()
    rng = np.random.default_rng(np.random.SeedSequence(31))
    views, bss = geom.num_views, geom.num_bs
    point = _off_grid_points(geom, rng, 1)[0]
    sigma = 1.7**2
    amp = np.array([1.7])

    tau = rng.normal(scale=1e-8, size=(views, bss))
    tau[0, 0] = 0.0
    phi = rng.uniform(0.0, 2.0 * np.pi, size=(views, bss))
    base = atom_cfr(point, amp, geom, space)
    gauged = _gauge_apply(base, geom, tau, phi)

    operator = PowerOperator(point, geom, space, product, tau=tau)
    reference = extract(gauged, PRODUCT_NODES[product]).data
    assert _max_relative_error(operator.forward(np.array([sigma])), reference) <= 1e-10

    ungauged = PowerOperator(point, geom, space, product)
    with_tau = operator.forward(np.array([sigma]))
    without_tau = ungauged.forward(np.array([sigma]))
    difference = _max_relative_error(with_tau, without_tau)
    if product in ("I", "I_n0", "I_omni"):
        assert difference <= 1e-12
    else:
        assert difference > 0.1


@pytest.mark.parametrize("space", ["vs", "bv"])
@pytest.mark.parametrize("product", ["ID", "ID_omni"])
def test_quantised_geometry(space: str, product: str) -> None:
    geom = _quantised_geometry(num_bins=128)
    rng = np.random.default_rng(np.random.SeedSequence(41))
    point = _off_grid_points(geom, rng, 1)[0]
    sigma = 1.7**2
    reference = extract(atom_cfr(point, np.array([1.7]), geom, space), PRODUCT_NODES[product]).data
    got = PowerOperator(point, geom, space, product).forward(np.array([sigma]))
    assert _max_relative_error(got, reference) <= 1e-4


@pytest.mark.parametrize("space", ["vs", "bv"])
@pytest.mark.parametrize("product", PRODUCTS)
def test_adjoint_and_linear_operator(space: str, product: str) -> None:
    geom = _synthetic_geometry()
    rng = np.random.default_rng(np.random.SeedSequence(51))
    points = _off_grid_points(geom, rng, 12)
    views, bss = geom.num_views, geom.num_bs
    tau = rng.normal(scale=1e-8, size=(views, bss))
    x = rng.random(points.shape[0])
    y = rng.standard_normal(product_shape(product, geom))

    for active_tau in (None, tau):
        for cache in (True, False):
            operator = PowerOperator(points, geom, space, product, tau=active_tau, cache=cache)
            forward = operator.forward(x)
            adjoint = operator.adjoint(y)
            left = float(np.sum(forward * y))
            right = float(np.sum(x * adjoint))
            assert abs(left - right) <= 1e-12 * np.linalg.norm(forward) * np.linalg.norm(y)
            assert operator.cached is cache

            linear = operator.as_linear_operator()
            assert linear.dtype == np.float64
            assert linear.shape == operator.shape
            y_flat = y.reshape(-1)
            np.testing.assert_allclose(linear.matvec(x), forward.reshape(-1), rtol=0.0, atol=0.0)
            np.testing.assert_allclose(linear.rmatvec(y_flat), adjoint, rtol=0.0, atol=0.0)
            np.testing.assert_allclose(linear.H.matvec(y_flat), adjoint, rtol=0.0, atol=0.0)

    cached = PowerOperator(points, geom, space, product, tau=tau, cache=True)
    uncached = PowerOperator(points, geom, space, product, tau=tau, cache=False)
    assert _max_relative_error(cached.forward(x), uncached.forward(x)) <= 1e-13

    single = PowerOperator(points[0], geom, space, product, tau=tau)
    basis = np.eye(points.shape[0])[0]
    column = PowerOperator(points, geom, space, product, tau=tau).forward(basis)
    assert _max_relative_error(column, single.forward(np.array([1.0]))) <= 1e-13

    helper = power_operator(points, geom, space, product, tau=tau)
    direct = PowerOperator(points, geom, space, product, tau=tau).as_linear_operator()
    np.testing.assert_allclose(helper.matvec(x), direct.matvec(x), rtol=0.0, atol=0.0)


def test_cache_auto_choice(monkeypatch: pytest.MonkeyPatch) -> None:
    geom = _synthetic_geometry()
    points = _off_grid_points(geom, np.random.default_rng(np.random.SeedSequence(61)), 4)
    assert PowerOperator(points, geom, "vs", cache=None).cached is True
    monkeypatch.setattr(kernels, "CACHE_MAX_BYTES", 0)
    assert PowerOperator(points, geom, "vs", cache=None).cached is False


def test_monte_carlo_power_matches_model() -> None:
    geom = _synthetic_geometry(rows=4, cols=4, num_bins=16, num_views=4, num_bs=2).select(
        views=[0, 1], bss=[0]
    )
    space = "vs"
    atoms = np.array([[8.3, -6.1, 3.3], [6.2, 5.9, -2.1], [-7.4, 4.2, 6.0]])
    sigma = np.array([1.0, 0.5, 2.0])
    dense = dense_matrix(atoms, geom, space)
    operator = PowerOperator(atoms, geom, space, "ID")
    noise_var = 1e-2 * float(np.max(operator.forward(sigma)))

    views, bss = geom.num_views, geom.num_bs
    rows, cols = geom.aperture_shape
    bins = geom.num_bins
    total = 20000
    chunk = 4000
    rng = np.random.default_rng(np.random.SeedSequence(8))

    shapes = {product: int(np.prod(product_shape(product, geom))) for product in PRODUCTS}
    sums = {product: np.zeros(shapes[product]) for product in PRODUCTS}
    squares = {product: np.zeros(shapes[product]) for product in PRODUCTS}

    for _ in range(total // chunk):
        theta = rng.uniform(0.0, 2.0 * np.pi, size=(chunk, atoms.shape[0]))
        amps = np.sqrt(sigma)[None, :] * np.exp(1j * theta)
        data = (dense @ amps.T).T
        noise = np.sqrt(noise_var / 2.0) * (
            rng.standard_normal(data.shape) + 1j * rng.standard_normal(data.shape)
        )
        stacked = (data + noise).reshape(chunk, views, bss, 2, rows, cols, bins)
        stacked = stacked.reshape(chunk * views, bss, 2, rows, cols, bins)
        for product in PRODUCTS:
            values = extract(stacked, PRODUCT_NODES[product]).data.reshape(chunk, -1)
            sums[product] += values.sum(axis=0)
            squares[product] += np.sum(values**2, axis=0)

    for product in PRODUCTS:
        mean = sums[product] / total
        model = PowerOperator(atoms, geom, space, product).forward(sigma).reshape(-1) + noise_floor(
            product, geom, noise_var
        )
        squared_deviations = np.maximum(squares[product] - total * mean**2, 0.0)
        standard_error = np.sqrt(squared_deviations / (total * (total - 1)))
        assert np.all(standard_error <= 0.01 * model)
        assert np.all(np.abs(mean - model) <= 0.05 * model)


@pytest.mark.parametrize("space", ["vs", "bv"])
@pytest.mark.parametrize("product", PRODUCTS)
def test_nonnegativity(space: str, product: str) -> None:
    geom = _synthetic_geometry()
    rng = np.random.default_rng(np.random.SeedSequence(71))
    points = _off_grid_points(geom, rng, 10)
    x = rng.random(points.shape[0]) + 0.01
    y = rng.random(product_shape(product, geom)) + 0.01
    operator = PowerOperator(points, geom, space, product)
    assert np.min(operator.forward(x)) >= 0.0
    assert np.min(operator.adjoint(y)) >= 0.0

    grid = VoxelGrid(origin=(-2.0, -2.0, 0.0), spacing=1.0, shape=(4, 4, 4))
    volume = rng.random(product_shape(product, geom)) + 0.01
    backprojected = power_backproject_grid(volume, geom, grid, space=space, product=product)
    assert np.min(backprojected) >= 0.0
    assert np.min(dirichlet_power(rng.uniform(-3.0, 3.0, size=32), 8)) >= 0.0


def _backprojection_metrics(
    space: str,
    product: str,
    kind: str,
    geom: CaptureGeometry,
    grid: VoxelGrid,
    atoms: np.ndarray,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    y = PowerOperator(atoms, geom, space, product).forward(np.array([1.0, 0.5, 2.0]))
    y = y + rng.exponential(1e-3 * float(np.max(y)), y.shape)
    reference = PowerOperator(grid.centers(), geom, space, product).adjoint(y).reshape(grid.shape)
    backprojected = power_backproject_grid(
        y, geom, grid, space=space, product=product, kind=kind, oversample=8
    )
    emax = float(np.max(np.abs(backprojected - reference)) / np.max(reference))
    e2 = float(np.linalg.norm(backprojected - reference) / np.linalg.norm(reference))
    return emax, e2


@pytest.mark.parametrize("space", ["vs", "bv"])
@pytest.mark.parametrize("product", PRODUCTS)
def test_backprojection_accuracy(space: str, product: str) -> None:
    geom = _synthetic_geometry()
    grid = VoxelGrid(origin=(-15.3, -15.1, -4.2), spacing=0.9, shape=(34, 34, 16))
    atoms = np.array([[8.3, -6.1, 3.3], [-9.7, 7.2, -2.1], [3.1, 11.5, 5.2]])
    emax, e2 = _backprojection_metrics(space, product, "trilinear", geom, grid, atoms, 81)
    if product == "I_omni":
        assert emax <= 1e-12
    else:
        assert emax <= 6e-2
        assert e2 <= 3e-2

    if product in ("ID", "I"):
        emax_cubic, e2_cubic = _backprojection_metrics(
            space, product, "tricubic", geom, grid, atoms, 91
        )
        assert emax_cubic <= 3e-3
        assert e2_cubic <= 2e-3


def test_backprojection_n_mode_shift() -> None:
    geom = _synthetic_geometry()
    grid = VoxelGrid(origin=(-15.3, -15.1, -4.2), spacing=0.9, shape=(20, 20, 12))
    atoms = np.array([[8.3, -6.1, 3.3], [-9.7, 7.2, -2.1], [3.1, 11.5, 5.2]])
    amps = np.array([1.0, 0.5, 2.0])
    rng = np.random.default_rng(np.random.SeedSequence(101))
    views, bss = geom.num_views, geom.num_bs
    tau = rng.integers(-5, 6, size=(views, bss)).astype(np.float64) / geom.bandwidth
    tau[0, 0] = 0.0
    assert np.any(tau != 0.0)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=(views, bss))

    base = atom_cfr(atoms, amps, geom, "vs")
    gauged = _gauge_apply(base, geom, tau, phi)
    y_s = extract(base, "ID").data
    y_n = extract(gauged, "ID").data

    reference = power_backproject_grid(y_s, geom, grid, space="vs", product="ID")
    shifted = power_backproject_grid(y_n, geom, grid, tau, space="vs", product="ID")
    assert _max_relative_error(shifted, reference) <= 1e-10
    unshifted = power_backproject_grid(y_n, geom, grid, space="vs", product="ID")
    assert _max_relative_error(unshifted, reference) > 0.1

    y_s_i = extract(base, "I").data
    y_n_i = extract(gauged, "I").data
    assert _max_relative_error(y_n_i, y_s_i) <= 1e-12
    map_s = power_backproject_grid(y_s_i, geom, grid, space="vs", product="I")
    map_n = power_backproject_grid(y_n_i, geom, grid, tau, space="vs", product="I")
    assert _max_relative_error(map_n, map_s) <= 1e-12


def test_singular_voxels_and_chunking(monkeypatch: pytest.MonkeyPatch) -> None:
    geom = _synthetic_geometry()
    rng = np.random.default_rng(np.random.SeedSequence(111))
    volume = rng.random(product_shape("ID", geom)) + 0.1

    for space in ("vs", "bv"):
        grid = VoxelGrid(origin=geom.ue_pos[0], spacing=1.0, shape=(5, 5, 5))
        mapped = power_backproject_grid(volume, geom, grid, space=space)
        assert mapped[0, 0, 0] == 0.0
        rest = mapped.copy()
        rest[0, 0, 0] = 0.0
        assert np.all(np.isfinite(rest))

    bs_grid = VoxelGrid(origin=geom.bs_pos[0], spacing=1.0, shape=(3, 3, 3))
    assert power_backproject_grid(volume, geom, bs_grid, space="bv")[0, 0, 0] == 0.0
    assert power_backproject_grid(volume, geom, bs_grid, space="vs")[0, 0, 0] > 0.0

    with pytest.raises(ValueError):
        PowerOperator(geom.ue_pos[0], geom, "vs")

    grid = VoxelGrid(origin=(-3.0, -3.0, 0.0), spacing=1.0, shape=(6, 6, 6))
    reference = power_backproject_grid(volume, geom, grid, space="vs")
    monkeypatch.setattr(kernels, "POINT_CHUNK", 7)
    chunked = power_backproject_grid(volume, geom, grid, space="vs")
    assert _max_relative_error(chunked, reference) <= 1e-12


def test_validation() -> None:
    geom = _synthetic_geometry()
    rng = np.random.default_rng(np.random.SeedSequence(121))
    points = _off_grid_points(geom, rng, 4)
    views, bss = geom.num_views, geom.num_bs
    y_shape = product_shape("ID", geom)
    volume = rng.random(y_shape)
    grid = VoxelGrid(origin=(-2.0, -2.0, 0.0), spacing=1.0, shape=(4, 4, 4))

    with pytest.raises(ValueError):
        PowerOperator(points, geom, "vs", "unknown")
    with pytest.raises(ValueError):
        product_shape("unknown", geom)
    with pytest.raises(ValueError):
        noise_floor("unknown", geom, 1.0)
    with pytest.raises(ValueError):
        power_backproject_grid(volume, geom, grid, space="vs", product="unknown")

    with pytest.raises(ValueError):
        PowerOperator(points, geom, "vs", tau=np.zeros((views, bss - 1)))
    with pytest.raises(ValueError):
        PowerOperator(points, geom, "vs", tau=np.full((views, bss), np.nan))

    operator = PowerOperator(points, geom, "vs")
    with pytest.raises(ValueError):
        operator.forward(np.ones(points.shape[0] + 1))
    with pytest.raises(ValueError):
        operator.forward(np.ones(points.shape[0], dtype=np.complex128))
    with pytest.raises(ValueError):
        operator.adjoint(np.zeros(y_shape[:-1] + (y_shape[-1] + 1,)))
    with pytest.raises(ValueError):
        operator.adjoint(np.zeros(y_shape, dtype=np.complex128))
    with pytest.raises(ValueError):
        operator.matvec(np.ones(operator.shape[1] + 1))
    with pytest.raises(ValueError):
        operator.rmatvec(np.ones(operator.shape[0] + 1))

    with pytest.raises(ValueError):
        noise_floor("ID", geom, -1.0)
    with pytest.raises(ValueError):
        noise_floor("ID", geom, np.nan)

    moved = planar_element_offsets(geom.wavelength, rows=8, cols=8)
    moved = moved.copy()
    moved[3, 1] += 1e-3
    broken = CaptureGeometry(
        ue_pos=geom.ue_pos,
        ue_rot=geom.ue_rot,
        bs_pos=geom.bs_pos,
        elem_offsets=moved,
        freq_offsets=geom.freq_offsets,
        f_c=geom.f_c,
        aperture_shape=geom.aperture_shape,
        bs_rot=geom.bs_rot,
    )
    with pytest.raises(ValueError):
        PowerOperator(points, broken, "vs")
    with pytest.raises(ValueError):
        power_backproject_grid(volume, broken, grid, space="vs")

    with pytest.raises(ValueError):
        power_backproject_grid(volume[:, :, :, 0], geom, grid, space="vs")
    with pytest.raises(ValueError):
        power_backproject_grid(volume.astype(np.complex128), geom, grid, space="vs")
    with pytest.raises(ValueError):
        power_backproject_grid(volume, geom, grid, space="vs", kind="nearest")
    with pytest.raises(ValueError):
        power_backproject_grid(volume, geom, grid, space="vs", oversample=0)
    with pytest.raises(ValueError):
        power_backproject_grid(volume, geom, grid, space="vs", oversample=2.5)
    with pytest.raises(ValueError):
        power_backproject_grid(volume, geom, grid, space="vs", oversample=True)
    with pytest.raises(ValueError):
        power_backproject_grid(volume, geom, grid, space="unknown")


def test_benchmark_kernels() -> None:
    if os.environ.get("RF_TOMO_BENCH") != "1":
        pytest.skip("set RF_TOMO_BENCH=1 to run the power-kernel benchmark")

    rng = np.random.default_rng(0)
    views, bins = 16, 128
    angle = 2.0 * np.pi * np.arange(views) / views
    ue_pos = np.stack([40.0 * np.cos(angle), 40.0 * np.sin(angle), np.full(views, 1.5)], axis=1)
    orientations = np.stack([angle + np.pi, np.zeros(views), np.zeros(views)], axis=1)
    bs_pos = np.array([[-70.0, 5.0, 25.0]])
    geom = _make_geometry(ue_pos, orientations, bs_pos, num_bins=bins, bs_look_at=np.zeros(3))
    count = 50_000
    points = np.column_stack(
        [
            rng.uniform(-50.0, 50.0, size=count),
            rng.uniform(-50.0, 50.0, size=count),
            rng.uniform(-2.0, 40.0, size=count),
        ]
    )

    for cache in (True, False):
        start = time.perf_counter()
        operator = PowerOperator(points, geom, "bv", "ID", cache=cache)
        init_time = time.perf_counter() - start
        x = rng.random(count)
        y = rng.random(operator.shape[0])

        start = time.perf_counter()
        image = operator.matvec(x)
        forward_time = time.perf_counter() - start

        start = time.perf_counter()
        back = operator.rmatvec(y)
        adjoint_time = time.perf_counter() - start

        print(
            f"kernels P={count} V={views} B=1 N={bins} cache={cache}: "
            f"init {init_time:.1f} s, K {forward_time:.2f} s, KT {adjoint_time:.2f} s"
        )
        assert image.shape == (operator.shape[0],)
        assert back.shape == (count,)
        assert np.all(np.isfinite(image))
        assert np.all(np.isfinite(back))

    volume = rng.random(product_shape("ID", geom))
    grid = VoxelGrid.from_bounds((-50.0, -50.0, -2.0), (50.0, 50.0, 40.0), 2.0)
    start = time.perf_counter()
    mapped = power_backproject_grid(volume, geom, grid, space="bv", product="ID", oversample=8)
    print(f"kernels backprojection grid={grid.shape}: {time.perf_counter() - start:.2f} s")
    assert mapped.shape == grid.shape
    assert np.all(np.isfinite(mapped))
