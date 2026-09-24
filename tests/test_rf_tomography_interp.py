import os
import time

import numpy as np
import pytest

from plateau_rt.domain.rf_tomography.interp import (
    INTERP_KINDS,
    gather,
    periodic_weights,
    scatter_add,
)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(seed))


@pytest.mark.parametrize("kind", INTERP_KINDS)
@pytest.mark.parametrize("shape", [(5, 6, 7), (3, 2, 9)])
def test_scatter_add_is_exact_adjoint(kind, shape):
    periods = (2.0, 3.5, 1.6e-7)
    origins = (-1.0, 0.25, 0.0)
    num_points = 500
    rng = _rng(101)
    span = 3.0 * np.asarray(periods)
    coords = rng.uniform(-span, span, size=(num_points, 3))
    idx, w = periodic_weights(coords, shape, periods, kind, origins=origins)

    num_taps = 8 if kind == "trilinear" else 64
    assert idx.dtype == np.int64
    assert w.dtype == np.float64
    assert idx.shape == (num_points, num_taps)
    assert w.shape == (num_points, num_taps)

    vol = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    a = rng.standard_normal(num_points) + 1j * rng.standard_normal(num_points)
    lhs = np.vdot(a, gather(vol, idx, w))
    rhs = np.vdot(scatter_add(a, idx, w, shape), vol)
    assert abs(lhs - rhs) <= 1e-12 * max(1.0, abs(lhs))


@pytest.mark.parametrize("kind", INTERP_KINDS)
def test_partition_of_unity_and_exact_nodes(kind):
    shape = (5, 6, 7)
    periods = (2.0, 3.5, 1.6e-7)
    origins = (-1.0, 0.25, 0.0)
    rng = _rng(202)

    coords = np.asarray(origins) + rng.random((300, 3)) * np.asarray(periods)
    _, w = periodic_weights(coords, shape, periods, kind, origins=origins)
    np.testing.assert_allclose(w.sum(axis=1), np.ones(coords.shape[0]), atol=1e-12)

    # k outside 0..n-1 exercises the periodic wrap on every axis.
    k = rng.integers(-2 * np.asarray(shape), 2 * np.asarray(shape), size=(200, 3))
    node_coords = np.asarray(origins) + k * (np.asarray(periods) / np.asarray(shape))
    idx, w_nodes = periodic_weights(node_coords, shape, periods, kind, origins=origins)
    vol = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    values = gather(vol, idx, w_nodes)
    km = np.mod(k, np.asarray(shape))
    expected = vol[km[:, 0], km[:, 1], km[:, 2]]
    np.testing.assert_allclose(values, expected, atol=1e-12)


def test_trilinear_wraps_at_period_boundary():
    shape = (4, 3, 5)
    periods = (1.0, 1.0, 1.0)
    rng = _rng(303)
    vol = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

    coords = np.array([[3.5 / 4.0, 0.0, 0.0]])
    idx, w = periodic_weights(coords, shape, periods, "trilinear")

    np.testing.assert_allclose(gather(vol, idx, w), 0.5 * (vol[3, 0, 0] + vol[0, 0, 0]), atol=1e-12)
    nonzero = set(idx[w != 0.0].tolist())
    expected = {
        np.ravel_multi_index((3, 0, 0), shape),
        np.ravel_multi_index((0, 0, 0), shape),
    }
    assert nonzero == expected


def test_tricubic_wraps_below_lower_edge():
    shape = (4, 3, 5)
    periods = (1.0, 1.0, 1.0)
    coords = np.array([[0.2 / 4.0, 0.0, 0.0]])
    idx, w = periodic_weights(coords, shape, periods, "tricubic")

    # Axis-0 taps sample {3, 0, 1, 2}: index -1 wraps to 3.
    nonzero = set(idx[w != 0.0].tolist())
    expected = {np.ravel_multi_index((j, 0, 0), shape) for j in (3, 0, 1, 2)}
    assert nonzero == expected


@pytest.mark.parametrize("kind", INTERP_KINDS)
def test_period_shift_invariance(kind):
    shape = (8, 5, 6)
    periods = (2.0, 3.5, 1.6e-7)
    origins = (-1.0, 0.25, 0.0)
    rng = _rng(404)
    coords = np.asarray(origins) + rng.random((200, 3)) * np.asarray(periods)
    shift = np.array([1.0, -2.0, 3.0]) * np.asarray(periods)
    vol = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

    idx0, w0 = periodic_weights(coords, shape, periods, kind, origins=origins)
    idx1, w1 = periodic_weights(coords + shift, shape, periods, kind, origins=origins)
    np.testing.assert_allclose(gather(vol, idx0, w0), gather(vol, idx1, w1), atol=1e-12)


_N = 16
_BANDWIDTH = 100e6
_SHAPE = (64, 64, 8 * _N)
_PERIODS = (2.0, 2.0, _N / _BANDWIDTH)
_ORIGINS = (-1.0, -1.0, 0.0)
_KERNEL_PEAK = 8 * 8 * _N


def _array_factor(u: np.ndarray) -> np.ndarray:
    m = np.arange(-4, 4, dtype=np.float64)
    return np.exp(1j * np.pi * m[None, :] * u[:, None]).sum(axis=1)


def _delay_kernel(t: np.ndarray) -> np.ndarray:
    n = np.arange(_N, dtype=np.float64)
    exponent = 1j * 2.0 * np.pi * (n - _N / 2.0) * (_BANDWIDTH / _N)
    return np.exp(exponent[None, :] * t[:, None]).sum(axis=1)


def _sampled_kernel_volume() -> np.ndarray:
    u = _ORIGINS[0] + np.arange(_SHAPE[0]) * _PERIODS[0] / _SHAPE[0]
    t = _ORIGINS[2] + np.arange(_SHAPE[2]) * _PERIODS[2] / _SHAPE[2]
    af = _array_factor(u)
    return af[:, None, None] * af[None, :, None] * _delay_kernel(t)[None, None, :]


@pytest.mark.parametrize("kind,threshold", [("trilinear", 3e-2), ("tricubic", 1e-3)])
def test_accuracy_on_oversampled_dirichlet_kernel(kind, threshold):
    rng = _rng(505)
    vol = _sampled_kernel_volume()
    assert abs(np.abs(vol[32, 32, 0]) - _KERNEL_PEAK) < 1e-6

    points = 20000
    coords_full = np.asarray(_ORIGINS) + np.asarray(_PERIODS) * rng.random((points, 3))
    coords_lobe = np.column_stack(
        (
            rng.uniform(-0.25, 0.25, points),
            rng.uniform(-0.25, 0.25, points),
            rng.uniform(-1.0 / _BANDWIDTH, 1.0 / _BANDWIDTH, points),
        )
    )

    errors = {}
    for name, coords in (("full", coords_full), ("lobe", coords_lobe)):
        idx, w = periodic_weights(coords, _SHAPE, _PERIODS, kind, origins=_ORIGINS)
        values = gather(vol, idx, w)
        expected = (
            _array_factor(coords[:, 0]) * _array_factor(coords[:, 1]) * _delay_kernel(coords[:, 2])
        )
        errors[name] = np.max(np.abs(values - expected)) / _KERNEL_PEAK
        assert errors[name] < threshold

    # A kind mix-up would be caught: trilinear is far worse in the main lobe.
    if kind == "trilinear":
        assert errors["lobe"] > 1e-3


def test_periodic_weights_validation():
    coords = np.zeros((5, 3))
    with pytest.raises(ValueError):
        periodic_weights(np.zeros((5, 2)), (4, 4, 4), (1.0, 1.0, 1.0))
    bad = coords.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        periodic_weights(bad, (4, 4, 4), (1.0, 1.0, 1.0))
    with pytest.raises(ValueError):
        periodic_weights(coords, (4, 4, 4), (0.0, 1.0, 1.0))
    with pytest.raises(ValueError):
        periodic_weights(coords, (0, 4, 4), (1.0, 1.0, 1.0))
    with pytest.raises(ValueError):
        periodic_weights(coords, (4, 4, 4), (1.0, 1.0, 1.0), "nearest")


def test_gather_validation():
    with pytest.raises(ValueError):
        gather(np.zeros((2, 2)), np.zeros((1, 8), dtype=np.int64), np.zeros((1, 8)))


def test_scatter_add_validation():
    with pytest.raises(ValueError):
        scatter_add(np.zeros(3), np.zeros((4, 8), dtype=np.int64), np.zeros((4, 8)), (4, 4, 4))


def test_empty_points():
    idx, w = periodic_weights(np.zeros((0, 3)), (4, 4, 4), (1.0, 1.0, 1.0))
    assert idx.shape == (0, 8)
    assert w.shape == (0, 8)


@pytest.mark.skipif(os.environ.get("RF_TOMO_BENCH") != "1", reason="set RF_TOMO_BENCH=1")
def test_benchmark_throughput():
    rng = _rng(606)
    shape = (64, 64, 1024)
    periods = (1.0, 1.0, 1.0)
    vol = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    for kind, num_points in (("trilinear", 1_000_000), ("tricubic", 200_000)):
        coords = rng.random((num_points, 3))
        start = time.perf_counter()
        idx, w = periodic_weights(coords, shape, periods, kind)
        values = gather(vol, idx, w)
        elapsed = time.perf_counter() - start
        print(f"{kind}: P={num_points} {elapsed:.3f}s")
        assert np.all(np.isfinite(values))
