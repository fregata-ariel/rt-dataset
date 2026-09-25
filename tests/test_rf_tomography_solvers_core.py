"""Unit tests for the shared E2 solver machinery (T16c, Part A)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from plateau_rt.domain.rf_tomography.forward_sep import SeparableOperator
from plateau_rt.domain.rf_tomography.solvers import coherent, power
from plateau_rt.domain.rf_tomography.solvers._core import (
    CallCounter,
    IterativeResult,
    mfista,
    power_iteration,
)
from plateau_rt.domain.rf_tomography.synthetic import ring_geometry


def test_power_iteration_norm_and_rayleigh() -> None:
    diagonal = np.diag([4.0, 2.0, 1.0, 0.5])
    v0 = np.ones(4) / 2.0
    normal = lambda v: diagonal @ v  # noqa: E731
    norm, norm_calls = power_iteration(normal, v0, 60, estimate="norm")
    rayleigh, rayleigh_calls = power_iteration(normal, v0, 60, estimate="rayleigh")
    assert norm == pytest.approx(4.0, rel=1e-9)
    assert rayleigh == pytest.approx(4.0, rel=1e-9)
    assert norm_calls == 60
    assert rayleigh_calls == 60

    zero = lambda v: 0.0 * v  # noqa: E731
    assert power_iteration(zero, v0, 60, estimate="norm") == (0.0, 1)
    assert power_iteration(zero, v0, 60, estimate="rayleigh") == (0.0, 1)

    with pytest.raises(ValueError, match="estimate"):
        power_iteration(normal, v0, 3, estimate="unknown")


def test_mfista_identity_lasso() -> None:
    y = np.array([3.0, -0.5, 1.2])
    lam = 1.0
    forward = lambda x: x.copy()  # noqa: E731
    objective = lambda z, az: 0.5 * float(np.sum((az - y) ** 2)) + lam * float(np.sum(np.abs(z)))  # noqa: E731
    x0 = np.zeros(3)
    ax0 = np.zeros(3)
    f0 = 0.5 * float(np.sum((ax0 - y) ** 2)) + lam * float(np.sum(np.abs(x0)))

    def prox_grad(w: np.ndarray, aw: np.ndarray) -> np.ndarray:
        step = w - (aw - y)
        return np.sign(step) * np.maximum(np.abs(step) - lam, 0.0)

    x, history, n_done, converged = mfista(
        x0,
        ax0,
        f0,
        forward=forward,
        prox_grad=prox_grad,
        objective=objective,
        n_iter=10,
        stop="objective",
        tol=0.0,
    )
    del n_done, converged
    assert np.allclose(x, [2.0, 0.0, 0.2], rtol=0.0, atol=1e-15)
    assert np.all(np.diff(history) <= 0.0)

    _, _, done, stopped = mfista(
        x0,
        ax0,
        f0,
        forward=forward,
        prox_grad=prox_grad,
        objective=objective,
        n_iter=10,
        stop="step",
        tol=0.0,
    )
    assert stopped
    assert done <= 3

    with pytest.raises(ValueError, match="stop"):
        mfista(
            x0,
            ax0,
            f0,
            forward=forward,
            prox_grad=prox_grad,
            objective=objective,
            n_iter=3,
            stop="unknown",
            tol=0.0,
        )


def test_call_counter_counts_separately() -> None:
    counter = CallCounter(lambda v: 2.0 * v, lambda v: 3.0 * v)
    np.testing.assert_array_equal(counter.forward(np.ones(2)), np.full(2, 2.0))
    np.testing.assert_array_equal(counter.forward(np.ones(2)), np.full(2, 2.0))
    np.testing.assert_array_equal(counter.adjoint(np.ones(2)), np.full(2, 3.0))
    assert counter.n_forward == 2
    assert counter.n_adjoint == 1


def test_result_dataclasses() -> None:
    assert issubclass(power.PowerSolution, IterativeResult)
    assert issubclass(coherent.CoherentResult, IterativeResult)
    names = [field.name for field in dataclasses.fields(coherent.CoherentResult)]
    assert names == ["x", "objective", "n_iter", "n_forward", "n_adjoint", "converged", "step"]
    base = ["x", "objective", "n_iter", "n_forward", "n_adjoint", "converged"]
    assert [field.name for field in dataclasses.fields(power.PowerSolution)] == base


def test_normal_operator_norm_counts_and_matches_lipschitz() -> None:
    geom = ring_geometry(num_views=3, num_bins=8, aperture_shape=(4, 4))
    rng = np.random.default_rng(np.random.SeedSequence([13, 0]))
    points = np.array([0.0, 0.0, 5.0]) + rng.uniform(-3.0, 3.0, size=(30, 3))
    op = SeparableOperator(points, geom, "bv", beta_model="shared")
    rho, calls = coherent.normal_operator_norm(op, n_iter=7, seed=3)
    assert calls == 7
    assert coherent.lipschitz_constant(op, n_iter=7, seed=3, safety=1.0) == rho
