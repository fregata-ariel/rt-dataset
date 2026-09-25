"""Tests for the T14 nuisance atoms and gauge solvers."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from plateau_rt.domain.rf_tomography.forward_exact import atom_cfr
from plateau_rt.domain.rf_tomography.forward_sep import SeparableOperator
from plateau_rt.domain.rf_tomography.gauges import (
    fit_los_ground,
    nuisance_points,
    power_xcorr_delay,
    self_calibrate,
    varpro_cost_and_grad,
)
from plateau_rt.domain.rf_tomography.geometry import CaptureGeometry
from plateau_rt.domain.rf_tomography.metrics import gauge_errors, nmse_global_phase
from plateau_rt.domain.rf_tomography.observables import angle_delay_volume
from plateau_rt.domain.rf_tomography.solvers.coherent import (
    lipschitz_constant,
    tikhonov_lsqr,
)
from plateau_rt.domain.rf_tomography.sync import (
    TrackSeeds,
    apply_gauge,
    make_tracks,
    reference_power,
)
from plateau_rt.domain.rf_tomography.synthetic import (
    l0c_random,
    l0e_image_method,
    ring_geometry,
)

FIXTURES = Path(__file__).parent / "fixtures" / "rf_tomography"


def _wrap(value: float) -> float:
    """Wrap an angle into ``(-pi, pi]``."""
    return float(np.angle(np.exp(1j * value)))


def _delay_error(estimate: float, truth: float, period: float) -> float:
    """Circular delay error in seconds."""
    return float(abs((estimate - truth + 0.5 * period) % period - 0.5 * period))


@pytest.fixture(scope="module")
def los_scene():
    """L0e LoS/ground scene rendered with an injected LoS phase-model error."""
    geom = ring_geometry(num_views=8, num_bins=32)
    phantom = l0e_image_method(geom=geom)
    scenes: dict[int, dict[str, np.ndarray]] = {}
    for seed in (0, 1):
        rng = np.random.default_rng(np.random.SeedSequence([seed, 5]))
        eps = np.deg2rad(rng.uniform(-5.0, 5.0, (8, 1)))
        rho = phantom.gt.points_rho.copy()
        rho[0] *= np.exp(1j * eps)
        Y = atom_cfr(phantom.gt.points_pos, rho, geom, "vs")
        p_ref, _ = reference_power(Y, np.ones((8, 1), bool))
        tracks, gt = make_tracks(Y, 30.0, p_ref, TrackSeeds(seed), freq_offsets=geom.freq_offsets)
        phi_t, tau_t = gt["gauges"]["ideal-N"]
        scenes[seed] = {
            "eps": eps,
            "rho": rho,
            "Y": Y,
            "tracks": tracks,
            "gt": gt,
            "phi_t": phi_t,
            "tau_t": tau_t,
        }
    return geom, phantom, scenes


@pytest.fixture(scope="module")
def l0c_scene():
    """L0c point phantoms and their 30 dB paired tracks."""
    geom = ring_geometry(num_views=8, num_bins=16)
    scenes: dict[int, tuple] = {}
    for seed in (0, 7, 11):
        phantom = l0c_random(seed, 4, geom=geom)
        p_ref, _ = reference_power(phantom.y_clean, np.ones((8, 1), bool))
        tracks, gt = make_tracks(
            phantom.y_clean, 30.0, p_ref, TrackSeeds(seed), freq_offsets=geom.freq_offsets
        )
        scenes[seed] = (phantom, tracks, gt)
    return geom, scenes


@pytest.mark.parametrize("seed", [0, 1])
def test_fit_los_ground_recovers_injected_gauge(los_scene, seed: int) -> None:
    geom, _, scenes = los_scene
    scene = scenes[seed]
    tracks = scene["tracks"]
    period = geom.delay_period
    for v in range(geom.num_views):
        result = fit_los_ground(tracks["ideal-N"][v, 0], geom, v, 0)
        assert _delay_error(result["tau"], scene["tau_t"][v, 0], period) <= 0.05e-9
        assert abs(_wrap(result["phi"] - scene["phi_t"][v, 0] - scene["eps"][v, 0])) <= np.deg2rad(
            1.0
        )
        assert abs(_wrap(result["phi"] - scene["phi_t"][v, 0])) <= (
            np.deg2rad(1.0) + abs(scene["eps"][v, 0])
        )
        assert abs(result["g_los"] - 1.0) <= 0.02
        truth_ground = scene["rho"][1, v, 0] * np.exp(-1j * scene["eps"][v, 0])
        assert abs(result["a_ground"] - truth_ground) <= 0.25 * abs(scene["rho"][1, v, 0])
        assert 0.0 <= result["resid"] < 1.0

        # S track without a gauge: the real LoS amplitude sees only Re(exp(j eps)).
        synced = fit_los_ground(tracks["ideal-S"][v, 0], geom, v, 0, with_gauge=False)
        assert synced["phi"] == 0.0 and synced["tau"] == 0.0
        assert abs(synced["g_los"] - np.cos(scene["eps"][v, 0])) <= 0.02
        ground = scene["rho"][1, v, 0]
        assert abs(synced["a_ground"] - ground) <= 0.25 * abs(ground)

    # Optimality of the no-gauge fit (real LoS, complex ground): the residual is orthogonal
    # to the ground column and has zero real correlation with the LoS column.
    data = tracks["ideal-S"][2, 0]
    fit = fit_los_ground(data, geom, 2, 0, with_gauge=False)
    capture = geom.select([2], [0])
    los_col, ground_col = (
        atom_cfr(point[None], np.ones(1), capture, "vs")[0, 0]
        for point in nuisance_points(geom, 0, 0.0)
    )
    residual = data - fit["g_los"] * los_col - fit["a_ground"] * ground_col
    scale = np.linalg.norm(data)
    assert abs(np.vdot(ground_col, residual)) <= 1e-10 * np.linalg.norm(ground_col) * scale
    assert abs(np.real(np.vdot(los_col, residual))) <= 1e-10 * np.linalg.norm(los_col) * scale


def test_fit_los_ground_power_mode(los_scene) -> None:
    geom, _, scenes = los_scene
    scene = scenes[0]
    period = geom.delay_period
    for v in range(geom.num_views):
        result = fit_los_ground(scene["tracks"]["ideal-N"][v, 0], geom, v, 0, mode="power")
        assert _delay_error(result["tau"], scene["tau_t"][v, 0], period) <= 0.2e-9
        assert np.isnan(result["phi"])
        assert abs(20.0 * np.log10(result["g_los"])) <= 0.5
    no_gauge = fit_los_ground(
        scene["tracks"]["ideal-S"][0, 0], geom, 0, 0, mode="power", with_gauge=False
    )
    assert no_gauge["tau"] == 0.0
    assert no_gauge["phi"] == 0.0


def test_free_los_amplitude_cannot_recover_phase(los_scene) -> None:
    geom, _, scenes = los_scene
    data = scenes[0]["tracks"]["ideal-N"][3, 0]
    anchor = fit_los_ground(data, geom, 3, 0)
    free = fit_los_ground(data, geom, 3, 0, free_los_phase=True)
    assert np.isfinite(anchor["resid"])
    for theta in (0.0, 1.0, 2.5, -2.0):
        shifted = data * np.exp(1j * theta)
        model_phase = fit_los_ground(shifted, geom, 3, 0)
        assert abs(_wrap(model_phase["phi"] - anchor["phi"] - theta)) <= 1e-6
        ambiguous = fit_los_ground(shifted, geom, 3, 0, free_los_phase=True)
        assert ambiguous["phi"] == 0.0
        assert abs(ambiguous["resid"] - free["resid"]) <= 1e-9 * free["resid"]
        assert abs(ambiguous["a_los"] - free["a_los"] * np.exp(1j * theta)) <= 1e-7 * abs(
            free["a_los"]
        )
        assert abs(ambiguous["tau"] - model_phase["tau"]) <= 1e-15


def test_fit_los_ground_on_sionna_los() -> None:
    fixture = np.load(FIXTURES / "sionna_los_aperture.npz")
    geom = CaptureGeometry.from_orientations(
        fixture["ue_pos"],
        fixture["ue_orientation"],
        fixture["bs_pos"][None],
        f_c=float(fixture["f_c"]),
        bandwidth=100e6,
        num_bins=8,
    )
    geom = dataclasses.replace(geom, bs_rot=np.eye(3)[None])
    Y = fixture["aperture_cfr"].astype(np.complex128)[:, None]
    period = geom.delay_period

    for v in range(2):
        raw = fit_los_ground(
            Y[v, 0],
            geom,
            v,
            0,
            with_gauge=False,
            free_los_phase=True,
            ground_height=None,
            pattern="iso",
            polarization="vv",
        )
        assert abs(np.angle(raw["a_los"])) <= np.deg2rad(0.1)
        assert abs(abs(raw["a_los"]) - 1.0) <= 1e-3

    phi = np.array([[0.0], [2.0]])
    tau = np.array([[3.7e-9], [-41.3e-9]])
    gauged = apply_gauge(Y, phi, tau, geom.freq_offsets)
    for v in range(2):
        fit = fit_los_ground(
            gauged[v, 0],
            geom,
            v,
            0,
            ground_height=None,
            pattern="iso",
            polarization="vv",
        )
        assert abs(_wrap(fit["phi"] - phi[v, 0])) <= np.deg2rad(0.1)
        assert _delay_error(fit["tau"], tau[v, 0], period) <= 1e-12


def test_power_xcorr_delay() -> None:
    geom = ring_geometry(num_views=1, num_bins=32)
    rng = np.random.default_rng(0)
    points = np.array([0.0, 0.0, 5.0]) + rng.uniform(-5.0, 5.0, (3, 3))
    amplitudes = np.array([1.0, 0.5j, -0.3])
    Y = atom_cfr(points, amplitudes, geom, "vs")
    period = geom.delay_period
    for tau in (3.3e-9, -7.77e-9, 41.2e-9, 0.6 * period):
        gauged = apply_gauge(Y, [[1.0]], [[tau]], geom.freq_offsets)
        for oversample in ((1, 4), (1, 2)):
            p_ref = np.abs(angle_delay_volume(Y, oversample=oversample)) ** 2
            p_obs = np.abs(angle_delay_volume(gauged, oversample=oversample)) ** 2
            estimate = power_xcorr_delay(p_obs, p_ref, period)
            assert _delay_error(estimate, tau, period) <= 5e-12

    p_ref = np.abs(angle_delay_volume(Y, oversample=(1, 4))) ** 2
    with pytest.raises(ValueError):
        power_xcorr_delay(p_ref, p_ref[..., :-1], period)
    with pytest.raises(ValueError):
        power_xcorr_delay(p_ref.astype(np.complex128), p_ref, period)
    with pytest.raises(ValueError):
        power_xcorr_delay(p_ref, p_ref, 0.0)
    with pytest.raises(ValueError):
        power_xcorr_delay(p_ref, p_ref, period, upsample=0)


def test_varpro_cost_and_grad() -> None:
    geom = ring_geometry(num_views=3, num_bins=8, aperture_shape=(4, 4))
    rng = np.random.default_rng(np.random.SeedSequence([14, 0]))
    points = np.array([0.0, 0.0, 5.0]) + rng.uniform(-3.0, 3.0, (6, 3))
    x_true = rng.standard_normal(6) + 1j * rng.standard_normal(6)
    op = SeparableOperator(points, geom, "bv")
    phi = rng.uniform(0.0, 2.0 * np.pi, (3, 1))
    tau = rng.normal(0.0, 5e-9, (3, 1))
    phi[0, 0] = 0.0
    tau[0, 0] = 0.0
    clean = atom_cfr(points, x_true, geom, "bv")
    Y = apply_gauge(clean, phi, tau, geom.freq_offsets)

    cost, grad, phi_est, tau_est = varpro_cost_and_grad(x_true, Y, op)
    assert cost <= 1e-20 * float(np.sum(np.abs(Y) ** 2))
    assert np.max(np.abs(np.angle(np.exp(1j * (phi_est - phi))))) <= 1e-9
    half_period = geom.delay_period / 2.0
    delay_errors = (tau_est - tau + half_period) % geom.delay_period - half_period
    assert np.max(np.abs(delay_errors)) <= 1e-15

    rms = float(np.sqrt(np.mean(np.abs(Y) ** 2)))
    noise = 0.05 * rms * (rng.standard_normal(Y.shape) + 1j * rng.standard_normal(Y.shape))
    noisy = Y + noise
    x0 = x_true + 0.3 * (rng.standard_normal(6) + 1j * rng.standard_normal(6))
    base_cost, base_grad, _, _ = varpro_cost_and_grad(x0, noisy, op)
    step = 1e-5
    for _ in range(3):
        direction = rng.standard_normal(6) + 1j * rng.standard_normal(6)
        plus = varpro_cost_and_grad(x0 + step * direction, noisy, op)[0]
        minus = varpro_cost_and_grad(x0 - step * direction, noisy, op)[0]
        finite_difference = (plus - minus) / (2.0 * step)
        analytic = float(np.real(np.vdot(base_grad, direction)))
        assert abs(finite_difference - analytic) <= 1e-6 * abs(analytic)

    carried = varpro_cost_and_grad(x0, noisy, op.with_gauges((phi, tau)))
    assert abs(carried[0] - base_cost) <= 1e-12 * abs(base_cost)
    assert np.max(np.abs(carried[1] - base_grad)) <= 1e-12 * np.max(np.abs(base_grad))


@pytest.mark.parametrize("seed", [0, 7, 11])
def test_self_calibrate_l0c(l0c_scene, seed: int) -> None:
    geom, scenes = l0c_scene
    phantom, tracks, gt = scenes[seed]
    rho = phantom.gt.points_rho
    op = SeparableOperator(phantom.gt.points_pos, geom, "bv")
    damping = 0.1 * np.sqrt(lipschitz_constant(op, safety=1.0))

    def solve(y: np.ndarray) -> np.ndarray:
        return tikhonov_lsqr(op, y, damping).x

    x_s = solve(tracks["ideal-S"])
    x_n, phi, tau, history = self_calibrate(solve, tracks["ideal-N"], op.with_gauges, n_iter=30)
    assert history["converged"]
    assert phi[0, 0] == 0.0
    errors = gauge_errors((phi, tau), gt["gauges"]["ideal-N"], period=geom.delay_period)
    assert np.max(np.abs(errors.phase)) <= np.deg2rad(2.0)
    assert np.max(errors.delay) <= 0.05e-9
    assert 10.0 * np.log10(nmse_global_phase(x_n, rho)) <= (
        10.0 * np.log10(nmse_global_phase(x_s, rho)) + 1.0
    )
    assert nmse_global_phase(solve(tracks["ideal-N"]), rho) > 0.1

    # The returned (x, phi, tau) is one consistent model of the N data.
    model = apply_gauge(op.forward(x_n), phi, tau, geom.freq_offsets)
    loss = float(np.sum(np.abs(tracks["ideal-N"] - model) ** 2))
    assert abs(loss - history["loss"][-1]) <= 1e-9 * history["loss"][-1]


def test_self_calibrate_reference_and_init(l0c_scene) -> None:
    geom, scenes = l0c_scene
    phantom, tracks, gt = scenes[0]
    op = SeparableOperator(phantom.gt.points_pos, geom, "bv")
    damping = 0.1 * np.sqrt(lipschitz_constant(op, safety=1.0))

    def solve(y: np.ndarray) -> np.ndarray:
        return tikhonov_lsqr(op, y, damping).x

    x_ref, phi_ref, tau_ref, history_ref = self_calibrate(
        solve, tracks["ideal-N"], op.with_gauges, n_iter=30, ref=(3, 0)
    )
    assert phi_ref[3, 0] == 0.0
    errors = gauge_errors((phi_ref, tau_ref), gt["gauges"]["ideal-N"], period=geom.delay_period)
    assert np.max(np.abs(errors.phase)) <= np.deg2rad(2.0)
    assert np.max(errors.delay) <= 0.05e-9
    assert history_ref["loss"].dtype == np.float64
    assert history_ref["loss"].shape == (history_ref["n_iter"],)

    x_init, phi_init, tau_init, history_init = self_calibrate(
        solve, tracks["ideal-N"], op.with_gauges, n_iter=30, init=gt["gauges"]["ideal-N"]
    )
    assert history_init["n_iter"] <= 3
    errors_init = gauge_errors(
        (phi_init, tau_init), gt["gauges"]["ideal-N"], period=geom.delay_period
    )
    assert np.max(np.abs(errors_init.phase)) <= np.deg2rad(2.0)
    assert np.max(errors_init.delay) <= 0.05e-9

    with pytest.raises(ValueError):
        self_calibrate(solve, tracks["ideal-N"][:, :0], op.with_gauges)
    with pytest.raises(ValueError):
        self_calibrate(
            lambda y: np.zeros(5, dtype=np.complex128), tracks["ideal-N"], op.with_gauges
        )
    with pytest.raises(ValueError):
        self_calibrate(solve, tracks["ideal-N"], op.with_gauges, ref=(8, 0))


def test_input_validation(los_scene) -> None:
    geom, _, scenes = los_scene
    data = scenes[0]["tracks"]["ideal-N"][0, 0]

    assert nuisance_points(geom, 0, None).shape == (1, 3)
    expected_image = np.array([geom.bs_pos[0, 0], geom.bs_pos[0, 1], -geom.bs_pos[0, 2]])
    np.testing.assert_allclose(nuisance_points(geom, 0, 0.0)[1], expected_image)
    assert nuisance_points(geom, 0, 5.0)[1, 2] == -geom.bs_pos[0, 2] + 2.0 * 5.0

    with pytest.raises(ValueError):
        fit_los_ground(data[:, :, :-1], geom, 0, 0)
    with pytest.raises(ValueError):
        fit_los_ground(data, geom, 0, 0, mode="bogus")
    with pytest.raises(ValueError):
        fit_los_ground(data, geom, geom.num_views, 0)
    with pytest.raises(ValueError):
        fit_los_ground(data, geom, 0, 0, oversample=0)
    broken = data.copy()
    broken[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        fit_los_ground(broken, geom, 0, 0)
    with pytest.raises(ValueError):
        nuisance_points(geom, geom.num_bs, 0.0)
    with pytest.raises(ValueError):
        nuisance_points(geom, 0, np.inf)
