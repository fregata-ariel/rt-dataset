"""Unit tests for the tomography identifiability module (T15b)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import c as SPEED_OF_LIGHT

from plateau_rt.domain.rf_tomography import forward_exact, synthetic
from plateau_rt.domain.rf_tomography.forward_exact import atom_cfr
from plateau_rt.domain.rf_tomography.geometry import CaptureGeometry
from plateau_rt.domain.rf_tomography.identifiability import (
    gauge_reduced_fim,
    ill_posed,
    numeric_jacobian,
    return_model,
)
from plateau_rt.domain.rf_tomography.sync import apply_gauge

PH = synthetic.l0e_image_method(0)
VS = np.asarray(PH.gt.points_pos[1:4], dtype=np.float64)
BETA = np.asarray(PH.gt.points_rho[1:4, 0, 0], dtype=np.complex128)
GS = synthetic.ring_geometry(num_views=8, num_bins=16, aperture_shape=(4, 4))
SIG_U = 1e-3
SIG_T_NS = 0.03
N_POS = 9


def _dn_output(theta: np.ndarray, geom: CaptureGeometry) -> np.ndarray:
    """Return the D-N delay-list observable ``[V, K, 3]`` in ``(u_y, u_z, t_ns)``."""
    n_views = int(geom.num_views)
    n_pts = 3
    pos = np.asarray(theta[: 3 * n_pts], dtype=np.float64).reshape(n_pts, 3)
    tau_ns = np.asarray(theta[3 * n_pts :], dtype=np.float64)
    ret = return_model(pos, geom, tau=(1e-9 * tau_ns)[:, None])
    out = np.empty((n_views, n_pts, 3), dtype=np.float64)
    out[:, :, :2] = ret[:, 0, :, :2]
    out[:, :, 2] = 1e9 * ret[:, 0, :, 2]
    return out


def _dn_sync_output(pos_flat: np.ndarray, geom: CaptureGeometry) -> np.ndarray:
    """Return the D-S observable (positions only, gauge known)."""
    full = np.concatenate([np.asarray(pos_flat, dtype=np.float64), [0.0]])
    return _dn_output(full, geom)


def _dn_weights(geom: CaptureGeometry, sig_u: float = SIG_U, sig_t: float = SIG_T_NS) -> np.ndarray:
    """Return the D-N per-output weights tiled in C order."""
    count = int(geom.num_views) * 3
    return np.tile([1.0 / sig_u**2, 1.0 / sig_u**2, 1.0 / sig_t**2], count)


def _idp_output(theta: np.ndarray, geom: CaptureGeometry, fix_ref: bool = False) -> np.ndarray:
    """Return the gauged IDP-N CFR ``[V, 1, 2, R, C, N]``."""
    n_views = int(geom.num_views)
    n_pts = 3
    pos = np.asarray(theta[: 3 * n_pts], dtype=np.float64).reshape(n_pts, 3)
    beta = np.asarray(theta[3 * n_pts : 4 * n_pts], dtype=np.float64) + 1j * np.asarray(
        theta[4 * n_pts : 5 * n_pts], dtype=np.float64
    )
    if fix_ref:
        phi = np.zeros(n_views, dtype=np.float64)
        phi[1:] = np.asarray(theta[5 * n_pts : 5 * n_pts + n_views - 1], dtype=np.float64)
        tau_ns = np.asarray(theta[5 * n_pts + n_views - 1 :], dtype=np.float64)
    else:
        phi = np.asarray(theta[5 * n_pts : 5 * n_pts + n_views], dtype=np.float64)
        tau_ns = np.asarray(theta[5 * n_pts + n_views :], dtype=np.float64)
    clean = atom_cfr(pos, beta, geom, "vs")
    return apply_gauge(clean, phi[:, None], 1e-9 * tau_ns[:, None], geom.freq_offsets)


def _idp_theta0(geom: CaptureGeometry, fix_ref: bool = False) -> np.ndarray:
    """Return the IDP-N expansion point (shared VS, view-0 amplitudes, zero gauges)."""
    n_views = int(geom.num_views)
    parts = [np.asarray(VS, dtype=np.float64).ravel(), BETA.real, BETA.imag]
    if fix_ref:
        parts += [np.zeros(n_views - 1), np.zeros(n_views)]
    else:
        parts += [np.zeros(n_views), np.zeros(n_views)]
    return np.concatenate(parts)


def _idp_indices(
    geom: CaptureGeometry, fix_ref: bool = False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the ``(beta, phi, tau)`` index blocks of the IDP-N parameter vector."""
    n_views = int(geom.num_views)
    n_phi = n_views - 1 if fix_ref else n_views
    beta = np.arange(N_POS, N_POS + 6)
    phi = np.arange(N_POS + 6, N_POS + 6 + n_phi)
    tau = np.arange(N_POS + 6 + n_phi, N_POS + 6 + n_phi + n_views)
    return beta, phi, tau


def test_numeric_jacobian_closed_form() -> None:
    """Central differences match the analytic Jacobian of a closed-form function."""

    def model(t: np.ndarray) -> np.ndarray:
        """Evaluate the four-output test function."""
        return np.array(
            [t[0] ** 2 * t[1], np.sin(t[2]), np.exp(0.5 * t[0]) * t[2], t[1]],
            dtype=np.float64,
        )

    theta = np.array([1.5, -0.7, 0.4], dtype=np.float64)
    jac = numeric_jacobian(model, theta)
    assert jac.shape == (4, 3)
    assert jac.dtype == np.float64
    expected = np.array(
        [
            [2.0 * theta[0] * theta[1], theta[0] ** 2, 0.0],
            [0.0, 0.0, np.cos(theta[2])],
            [0.5 * np.exp(0.5 * theta[0]) * theta[2], 0.0, np.exp(0.5 * theta[0])],
            [0.0, 1.0, 0.0],
        ]
    )
    assert float(np.max(np.abs(jac - expected))) < 1e-8


def test_numeric_jacobian_physical_model() -> None:
    """Jacobian of the gauged VS atom matches the analytic derivative."""
    geom = GS.select(views=[0, 3])
    s0 = np.array([-70.0, 5.0, -25.0], dtype=np.float64)
    beta0 = complex(0.4 - 0.2j)

    def model(theta: np.ndarray) -> np.ndarray:
        """Evaluate the two-view gauged single-atom CFR."""
        pos = np.asarray(theta[:3], dtype=np.float64).reshape(1, 3)
        phi = np.array([float(theta[3]), 0.0], dtype=np.float64)[:, None]
        tau = (1e-9 * np.array([float(theta[4]), 0.0], dtype=np.float64))[:, None]
        clean = atom_cfr(pos, np.array([beta0]), geom, "vs", pattern="iso")
        return apply_gauge(clean, phi, tau, geom.freq_offsets)

    theta0 = np.array([s0[0], s0[1], s0[2], 0.0, 0.0], dtype=np.float64)
    jac_num = numeric_jacobian(model, theta0)
    y_obs = model(theta0)
    assert jac_num.shape == (2 * y_obs.size, 5)
    rows, cols = geom.aperture_shape
    freq = np.asarray(geom.freq_offsets, dtype=np.float64)
    wavenumber = float(geom.wavenumber)
    cols_full: list[np.ndarray] = []
    for col in range(5):
        deriv = np.zeros_like(y_obs)
        for v in range(geom.num_views):
            vec = s0 - np.asarray(geom.ue_pos[v], dtype=np.float64)
            dist = float(np.linalg.norm(vec))
            unit = vec / dist
            if col < 3:
                q_world = np.asarray(geom.elem_offsets, dtype=np.float64) @ geom.ue_rot[v].T
                proj = np.eye(3) - np.outer(unit, unit)
                factor = (
                    -unit[col] / dist
                    - 1j * wavenumber * unit[col]
                    - 2j * np.pi * freq[None, :] * unit[col] / SPEED_OF_LIGHT
                    + 1j * wavenumber * (q_world @ proj[:, col])[:, None] / dist
                )
                y_v = y_obs[v, 0]
                deriv[v, 0] = y_v * factor.reshape(1, rows, cols, geom.num_bins)
            elif v == 0 and col == 3:
                deriv[v] = 1j * y_obs[v]
            elif v == 0 and col == 4:
                deriv[v] = -2j * np.pi * freq[None, None, None, :] * 1e-9 * y_obs[v]
        flat = np.asarray(deriv).ravel(order="C")
        cols_full.append(np.concatenate([flat.real, flat.imag]))
    jac_an = np.stack(cols_full, axis=1)
    scale = float(np.max(np.abs(jac_an)))
    assert scale > 0.0
    assert float(np.max(np.abs(jac_num - jac_an))) / scale < 1e-6


def test_numeric_jacobian_validation() -> None:
    """Invalid inputs and inconsistent model outputs raise ValueError."""

    def real_model(t: np.ndarray) -> np.ndarray:
        """Square the input elementwise."""
        return np.asarray(t, dtype=np.float64) ** 2

    theta = np.array([1.0, 2.0], dtype=np.float64)
    with pytest.raises(ValueError):
        numeric_jacobian(real_model, theta, eps=0.0)
    with pytest.raises(ValueError):
        numeric_jacobian(real_model, theta, eps=-1e-6)
    with pytest.raises(ValueError):
        numeric_jacobian(real_model, theta, eps=np.array([1e-6, 1e-6, 1e-6]))
    with pytest.raises(ValueError):
        numeric_jacobian(real_model, np.array([[1.0, 2.0]]))
    with pytest.raises(ValueError):
        numeric_jacobian(real_model, np.array([1.0, np.inf]))
    with pytest.raises(ValueError):
        numeric_jacobian(
            lambda t: np.array([np.inf, 0.0]),
            np.array([1.0, 2.0]),
        )

    def growing(t: np.ndarray) -> np.ndarray:
        """Change the output size with the input."""
        return np.zeros(2 if float(t[0]) == 0.0 else 3, dtype=np.float64)

    with pytest.raises(ValueError):
        numeric_jacobian(growing, np.array([0.0]))

    def to_complex(t: np.ndarray) -> np.ndarray:
        """Turn complex away from the expansion point."""
        if float(t[0]) == 0.0:
            return np.zeros(2, dtype=np.float64)
        return np.zeros(2, dtype=np.complex128)

    with pytest.raises(ValueError):
        numeric_jacobian(to_complex, np.array([0.0]))

    before = np.array([1.0, -2.0, 3.0], dtype=np.float64)
    snapshot = before.copy()
    numeric_jacobian(real_model, before)
    np.testing.assert_array_equal(before, snapshot)


def test_gauge_reduced_fim_random() -> None:
    """The projection equals the Schur complement and is scale invariant."""
    rng = np.random.default_rng(np.random.SeedSequence(7))
    jac = rng.standard_normal((40, 6))
    weights = rng.uniform(0.5, 2.0, size=40)
    nuisance = np.array([1, 4])
    keep = np.setdiff1d(np.arange(6), nuisance)
    reduced = gauge_reduced_fim(jac, weights, nuisance)
    assert reduced.shape == (4, 4)
    np.testing.assert_allclose(reduced, reduced.T, rtol=0, atol=0)
    fim = jac.T @ (weights[:, None] * jac)
    np.testing.assert_allclose(
        np.linalg.inv(reduced), np.linalg.inv(fim)[np.ix_(keep, keep)], rtol=1e-10, atol=0
    )
    full = gauge_reduced_fim(jac, weights, np.empty(0, dtype=np.int64))
    np.testing.assert_allclose(full, fim, rtol=1e-12, atol=0)
    scaled = jac.copy()
    scaled[:, 1] *= 1e9
    scaled[:, 4] *= 1e-3
    np.testing.assert_allclose(
        gauge_reduced_fim(scaled, weights, nuisance), reduced, rtol=1e-9, atol=0
    )
    zeroed = weights.copy()
    zeroed[:5] = 0.0
    np.testing.assert_allclose(
        gauge_reduced_fim(jac, zeroed, nuisance),
        gauge_reduced_fim(jac[5:], zeroed[5:], nuisance),
        rtol=1e-12,
        atol=0,
    )
    with pytest.raises(ValueError):
        gauge_reduced_fim(jac, weights, np.array([1, 1]))
    with pytest.raises(ValueError):
        gauge_reduced_fim(jac, weights, np.array([6]))
    with pytest.raises(ValueError):
        gauge_reduced_fim(jac, weights, np.arange(6))
    with pytest.raises(ValueError):
        gauge_reduced_fim(jac, -weights, nuisance)
    with pytest.raises(ValueError):
        gauge_reduced_fim(jac, weights[:10], nuisance)
    with pytest.raises(ValueError):
        gauge_reduced_fim(np.full_like(jac, np.nan), weights, nuisance)


def test_global_phase_null() -> None:
    """The all-phi IDP-N FIM has only the global-phase null after elimination."""
    geom = GS
    theta0 = _idp_theta0(geom)
    jac = numeric_jacobian(lambda t: _idp_output(t, geom), theta0)
    y0 = _idp_output(theta0, geom)
    sigma2 = float(np.mean(np.abs(y0) ** 2)) / 1e3
    weights = np.full(2 * y0.size, 2.0 / sigma2)
    beta_idx, phi_idx, tau_idx = _idp_indices(geom)
    fim = jac.T @ (weights[:, None] * jac)
    direction = np.zeros(theta0.shape[0], dtype=np.float64)
    direction[N_POS : N_POS + 3] = BETA.imag
    direction[N_POS + 3 : N_POS + 6] = -BETA.real
    direction[phi_idx] = 1.0
    residual = float(np.linalg.norm(fim @ direction))
    denom = float(np.linalg.norm(fim, ord=2) * np.linalg.norm(direction))
    assert residual / denom < 1e-10
    spectrum = np.linalg.eigvalsh(fim)
    assert spectrum[0] / spectrum[-1] < 1e-15
    j_all = gauge_reduced_fim(jac, weights, np.concatenate([beta_idx, phi_idx, tau_idx]))
    assert j_all.shape == (9, 9)
    flag, cond, std = ill_posed(j_all, range(9))
    assert not flag
    assert cond < 1e5
    assert float(np.max(std)) < 0.05
    theta_ref = _idp_theta0(geom, fix_ref=True)
    jac_ref = numeric_jacobian(lambda t: _idp_output(t, geom, fix_ref=True), theta_ref)
    y_ref = _idp_output(theta_ref, geom, fix_ref=True)
    weights_ref = np.full(2 * y_ref.size, 2.0 / sigma2)
    beta_r, phi_r, tau_r = _idp_indices(geom, fix_ref=True)
    j_ref = gauge_reduced_fim(jac_ref, weights_ref, np.concatenate([beta_r, phi_r, tau_r]))
    assert float(np.max(np.abs(j_ref - j_all))) / float(np.max(np.abs(j_all))) < 1e-8
    j_gauge = gauge_reduced_fim(jac, weights, np.concatenate([phi_idx, tau_idx]))
    assert j_gauge.shape == (15, 15)
    eig = np.linalg.eigvalsh(j_gauge)
    assert eig[0] / eig[-1] < 1e-15
    flag_g, cond_g, std_g = ill_posed(j_gauge, range(9))
    assert not flag_g
    assert abs(cond_g - cond) / cond < 1e-6
    np.testing.assert_allclose(std_g, std, rtol=1e-6, atol=0)


def test_dn_single_view_flagged() -> None:
    """One D-N view cannot fix the delay gauge; the D-S control is well posed."""
    geom = PH.geom.select(views=[0])
    theta0 = np.concatenate([VS.ravel(), np.zeros(geom.num_views)])
    jac = numeric_jacobian(lambda t: _dn_output(t, geom), theta0)
    weights = _dn_weights(geom)
    reduced = gauge_reduced_fim(jac, weights, np.array([N_POS]))
    flag, cond, _ = ill_posed(reduced, range(N_POS))
    assert flag
    assert cond > 1e12
    jac_sync = numeric_jacobian(lambda t: _dn_sync_output(t, geom), VS.ravel())
    fim_sync = gauge_reduced_fim(jac_sync, weights, np.empty(0, dtype=np.int64))
    flag_s, cond_s, std_s = ill_posed(fim_sync, range(N_POS))
    assert not flag_s
    assert cond_s < 1e4
    assert float(np.max(std_s)) < 1.0


def test_dn_eight_views_shared() -> None:
    """Eight D-N views with shared VS identify positions up to the gauges."""
    geom = PH.geom
    theta0 = np.concatenate([VS.ravel(), np.zeros(geom.num_views)])
    jac = numeric_jacobian(lambda t: _dn_output(t, geom), theta0)
    weights = _dn_weights(geom)
    nuisance = np.arange(N_POS, N_POS + geom.num_views)
    reduced = gauge_reduced_fim(jac, weights, nuisance)
    flag, cond, std = ill_posed(reduced, range(N_POS))
    assert not flag
    assert cond < 1e5
    assert float(np.max(std)) < 0.5
    assert ill_posed(reduced, range(N_POS), cond_max=1e3)[0]


def test_dn_std_branch() -> None:
    """Coarse delay noise trips only the position-std criterion."""
    geom = PH.geom
    theta0 = np.concatenate([VS.ravel(), np.zeros(geom.num_views)])
    jac = numeric_jacobian(lambda t: _dn_output(t, geom), theta0)
    weights = _dn_weights(geom, sig_u=0.3, sig_t=30.0)
    nuisance = np.arange(N_POS, N_POS + geom.num_views)
    reduced = gauge_reduced_fim(jac, weights, nuisance)
    flag, cond, std = ill_posed(reduced, range(N_POS))
    assert flag
    assert cond < 1e4
    assert float(np.max(std)) > 10.0
    assert not ill_posed(reduced, range(N_POS), std_max_m=100.0)[0]


def test_ill_posed_edges() -> None:
    """Zero, diagonal and invalid FIM inputs behave as specified."""
    flag, cond, std = ill_posed(np.zeros((3, 3)), [0, 1, 2])
    assert flag
    assert cond == np.inf
    assert bool(np.all(~np.isfinite(std)))
    diag = np.diag([4.0, 1.0, 100.0])
    flag_d, cond_d, std_d = ill_posed(diag, [0, 1])
    assert not flag_d
    assert cond_d == pytest.approx(4.0)
    np.testing.assert_allclose(std_d, [0.5, 1.0], rtol=1e-12, atol=0)
    with pytest.raises(ValueError):
        ill_posed(diag, [])
    with pytest.raises(ValueError):
        ill_posed(diag, [0, 0])
    with pytest.raises(ValueError):
        ill_posed(diag, [0, 3])
    with pytest.raises(ValueError):
        ill_posed(np.zeros((2, 3)), [0, 1])


def test_return_model() -> None:
    """The D-list mean matches capture factors and shifts with the delay gauge."""
    geom = PH.geom
    points = np.asarray(PH.gt.points_pos, dtype=np.float64)
    out = return_model(points, geom)
    assert out.shape == (geom.num_views, geom.num_bs, points.shape[0], 3)
    assert out.dtype == np.float64
    for v in range(geom.num_views):
        for b in range(geom.num_bs):
            factors = forward_exact.capture_factors(points, geom, "vs", v, b, pattern="iso")
            np.testing.assert_allclose(out[v, b, :, 0], factors.u_local[:, 1], rtol=0, atol=0)
            np.testing.assert_allclose(out[v, b, :, 1], factors.u_local[:, 2], rtol=0, atol=0)
            np.testing.assert_allclose(
                out[v, b, :, 2], geom.vs_delay(points, v), rtol=1e-15, atol=0
            )
    rng = np.random.default_rng(np.random.SeedSequence(3))
    tau = rng.standard_normal((geom.num_views, geom.num_bs))
    shifted = return_model(points, geom, tau=tau)
    expect_shift = np.broadcast_to(tau[:, :, None], shifted[..., 2].shape)
    np.testing.assert_allclose(shifted[..., 2] - out[..., 2], expect_shift, rtol=0, atol=0)
    np.testing.assert_allclose(shifted[..., :2], out[..., :2], rtol=0, atol=0)
    with pytest.raises(ValueError):
        return_model(points, geom, tau=np.zeros((geom.num_views,)))
    with pytest.raises(ValueError):
        return_model(points, geom, tau=np.full((geom.num_views, geom.num_bs), np.inf))
