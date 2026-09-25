"""Unit tests for the array-level hardware-impairment helpers (Part A)."""

from __future__ import annotations

import cmath
import json
import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from plateau_rt.domain.rf_camera import impairments
from plateau_rt.domain.rf_camera.gauge import align_common_phase_and_delay
from plateau_rt.domain.rf_camera.imaging import frequency_offsets
from plateau_rt.domain.rf_camera.impairments import (
    ImpairmentConfig,
    apply_gauge,
    apply_hardware_impairments,
    apply_impairments,
    calibration_capture,
    capture_gt_record,
    draw_element_errors,
    gauge_factor,
    perturb_poses,
    rotation_from_vector,
)
from plateau_rt.domain.rf_tomography import sync
from plateau_rt.domain.rf_tomography.sync import draw_gauges
from plateau_rt.domain.rf_tomography.views import nested_view_order

BANDWIDTH_HZ = 100e6
N = 64
FREQ = frequency_offsets(BANDWIDTH_HZ, N).astype(np.float64)
PERIOD = N / BANDWIDTH_HZ


def _multipath(
    v: int, b: int, r: int, c: int, freq: np.ndarray, seed: int, k: int = 5
) -> np.ndarray:
    """Return random multipath aperture data ``[v, b, 2, r, c, N]``."""
    rng = np.random.default_rng(seed)
    tau_p = rng.uniform(50e-9, 400e-9, (v, b, k))
    a = rng.standard_normal((v, b, 2, r, c, k)) + 1j * rng.standard_normal((v, b, 2, r, c, k))
    ramp = np.exp(-2j * np.pi * freq[None, None, None, :] * tau_p[..., None])
    return np.einsum("vbhrck,vbkn->vbhrcn", a, ramp)


def _stacked_gains(num_views: int, rows: int, cols: int, seed: int) -> np.ndarray:
    """Return one 0.5 dB / 5 deg element-error gain per view, stacked."""
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    config = ImpairmentConfig(element_gain_std_db=0.5, element_phase_std_deg=5.0)
    return np.stack(
        [draw_element_errors(config, rows, cols, rng).complex_gain for _ in range(num_views)]
    )


def test_noise_var_abs_is_independent_of_signal_power() -> None:
    """The absolute noise variance never scales with the signal power."""
    freq = frequency_offsets(BANDWIDTH_HZ, 32).astype(np.float64)
    Y = _multipath(3, 2, 8, 8, freq, seed=1)
    G = _stacked_gains(3, 8, 8, seed=7)
    sigma2 = 0.01

    obs1, gt1 = apply_hardware_impairments(
        Y, element_gain=G, front_to_back_db=20.0, noise_var_abs=sigma2, rng=np.random.default_rng(3)
    )
    obs2, gt2 = apply_hardware_impairments(
        1000.0 * Y,
        element_gain=G,
        front_to_back_db=20.0,
        noise_var_abs=sigma2,
        rng=np.random.default_rng(3),
    )

    assert gt1["noise_var"] == sigma2
    assert gt2["noise_var"] == sigma2

    g = 10.0 ** (-1.0)
    collapsed = Y[:, :, 0] + g * Y[:, :, 1]
    gain6 = G[:, None, :, :, None]
    w1 = (obs1[:, :, 0] - gain6 * collapsed) / gain6
    w2 = (obs2[:, :, 0] - gain6 * (1000.0 * collapsed)) / gain6
    np.testing.assert_allclose(w1, w2, rtol=0.0, atol=1e-9)

    for v in range(3):
        for b in range(2):
            power = float(np.mean(np.abs(w1[v, b]) ** 2))
            assert 0.9 * sigma2 <= power <= 1.1 * sigma2

    delta = gt2["achieved_snr_db"] - gt1["achieved_snr_db"]
    np.testing.assert_allclose(delta, 60.0, rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(gt2["noise_power"], gt1["noise_power"], rtol=1e-12)


def test_explicit_formula_and_defaults() -> None:
    """The pinned arithmetic matches a pure-Python loop and the trivial limits."""
    rng = np.random.default_rng(np.random.SeedSequence(2))
    V, B, R, C, n = 2, 3, 2, 3, 8
    Y = rng.standard_normal((V, B, 2, R, C, n)) + 1j * rng.standard_normal((V, B, 2, R, C, n))
    w = rng.standard_normal((V, B, R, C, n)) + 1j * rng.standard_normal((V, B, R, C, n))
    gain = rng.standard_normal((V, R, C)) + 1j * rng.standard_normal((V, R, C))
    fb = 10.0
    g = 10.0 ** (-fb / 20.0)

    obs, _ = apply_hardware_impairments(
        Y, element_gain=gain, front_to_back_db=fb, noise_var_abs=0.5, noise=w
    )
    assert obs.shape == (V, B, 1, R, C, n)
    assert obs.dtype == np.complex128

    expected = np.zeros_like(obs)
    for v in range(V):
        for b in range(B):
            for r in range(R):
                for c in range(C):
                    for i in range(n):
                        expected[v, b, 0, r, c, i] = gain[v, r, c] * (
                            Y[v, b, 0, r, c, i] + g * Y[v, b, 1, r, c, i] + w[v, b, r, c, i]
                        )
    rel = np.abs(obs - expected) / np.maximum(np.abs(expected), 1e-300)
    assert np.max(rel) < 1e-13

    obs_none, _ = apply_hardware_impairments(Y, front_to_back_db=None, noise_var_abs=0.0)
    np.testing.assert_array_equal(obs_none[:, :, 0], Y[:, :, 0])

    obs_g, _ = apply_hardware_impairments(Y, front_to_back_db=fb, noise_var_abs=0.0)
    np.testing.assert_array_equal(obs_g[:, :, 0], Y[:, :, 0] + g * Y[:, :, 1])

    gain_rc = rng.standard_normal((R, C)) + 1j * rng.standard_normal((R, C))
    obs_rc, _ = apply_hardware_impairments(
        Y, element_gain=gain_rc, front_to_back_db=fb, noise_var_abs=0.0
    )
    obs_st, _ = apply_hardware_impairments(
        Y,
        element_gain=np.broadcast_to(gain_rc, (V, R, C)),
        front_to_back_db=fb,
        noise_var_abs=0.0,
    )
    np.testing.assert_array_equal(obs_rc, obs_st)


def test_noise_stream_is_pinned() -> None:
    """The realised noise matches the pinned stream; idle rngs are not advanced."""
    V, B, R, C, n = 2, 2, 3, 4, 8
    freq = frequency_offsets(BANDWIDTH_HZ, n).astype(np.float64)
    Y = _multipath(V, B, R, C, freq, seed=9)
    sigma2 = 0.3

    obs, gt = apply_hardware_impairments(
        Y,
        front_to_back_db=None,
        noise_var_abs=sigma2,
        rng=np.random.default_rng(np.random.SeedSequence(11)),
    )
    realized = obs[:, :, 0] - Y[:, :, 0]
    fresh = np.random.default_rng(np.random.SeedSequence(11))
    z = fresh.standard_normal((2, V, B, R, C, n))
    want = math.sqrt(sigma2 / 2.0) * (z[0] + 1j * z[1])
    np.testing.assert_allclose(realized, want, rtol=1e-12, atol=1e-15)
    assert gt["noise_var"] == sigma2

    idle = np.random.default_rng(np.random.SeedSequence(5))
    apply_hardware_impairments(Y, noise_var_abs=0.0, rng=idle)
    fresh_idle = np.random.default_rng(np.random.SeedSequence(5))
    np.testing.assert_array_equal(idle.standard_normal(4), fresh_idle.standard_normal(4))

    idle2 = np.random.default_rng(np.random.SeedSequence(6))
    explicit = np.zeros((V, B, R, C, n), dtype=np.complex128)
    apply_hardware_impairments(Y, noise_var_abs=sigma2, noise=explicit, rng=idle2)
    fresh2 = np.random.default_rng(np.random.SeedSequence(6))
    np.testing.assert_array_equal(idle2.standard_normal(4), fresh2.standard_normal(4))


def test_gt_records_recomputed() -> None:
    """Signal/noise powers and SNRs match independent (v, b) loops."""
    freq = frequency_offsets(BANDWIDTH_HZ, 32).astype(np.float64)
    V, B, R, C, n = 3, 2, 8, 8, 32
    Y = _multipath(V, B, R, C, freq, seed=4)
    G = _stacked_gains(V, R, C, seed=12)
    sigma2 = 0.02
    obs, gt = apply_hardware_impairments(
        Y, element_gain=G, front_to_back_db=20.0, noise_var_abs=sigma2, rng=np.random.default_rng(8)
    )
    g = 10.0 ** (-1.0)
    collapsed = Y[:, :, 0] + g * Y[:, :, 1]
    gain_power = np.mean(np.abs(G) ** 2, axis=(1, 2))

    for v in range(V):
        for b in range(B):
            sig = np.zeros((R, C, n), dtype=np.float64)
            noi = np.zeros((R, C, n), dtype=np.float64)
            for r in range(R):
                for c in range(C):
                    for i in range(n):
                        s = G[v, r, c] * collapsed[v, b, r, c, i]
                        sig[r, c, i] = (s * s.conjugate()).real
                        nn = obs[v, b, 0, r, c, i] - s
                        noi[r, c, i] = (nn * nn.conjugate()).real
            sig_p = float(np.mean(sig))
            noi_p = float(np.mean(noi))
            np.testing.assert_allclose(gt["signal_power"][v, b], sig_p, rtol=1e-10)
            np.testing.assert_allclose(gt["noise_power"][v, b], noi_p, rtol=1e-10)
            exp = 10.0 * math.log10(sig_p / (sigma2 * gain_power[v]))
            ach = 10.0 * math.log10(sig_p / noi_p)
            np.testing.assert_allclose(gt["expected_snr_db"][v, b], exp, rtol=1e-10)
            np.testing.assert_allclose(gt["achieved_snr_db"][v, b], ach, rtol=1e-10)
            assert abs(ach - exp) < 0.5


def test_apply_gauge_moved_same_object_and_unit_modulus() -> None:
    """The gauge helpers live in impairments and are re-exported by sync."""
    assert impairments.apply_gauge is sync.apply_gauge
    assert impairments.gauge_factor is sync.gauge_factor

    rng = np.random.default_rng(np.random.SeedSequence(21))
    phi = rng.uniform(0.0, 2.0 * np.pi, size=(3, 2))
    tau = rng.uniform(-1e-6, 1e-6, size=(3, 2))
    gauge = gauge_factor(phi, tau, FREQ)
    assert np.max(np.abs(np.abs(gauge) - 1.0)) < 1e-15

    v, b, n = 1, 1, 5
    want = cmath.exp(1j * phi[v, b]) * cmath.exp(-2j * math.pi * FREQ[n] * tau[v, b])
    assert abs(gauge[v, b, n] - want) / abs(want) < 1e-14


def test_gauge_alignment_recovers_common_phase_and_timing() -> None:
    """#5 acceptance: alignment recovers the N gauge; sign is discriminated."""
    V, B, R, C = 6, 2, 8, 8
    for seed in (0, 1, 2):
        Y = _multipath(V, B, R, C, FREQ, seed)
        ref = (int(nested_view_order(V, seed)[0]), 0)
        rng = np.random.default_rng(seed)
        theta0 = rng.uniform(0.0, 2.0 * np.pi)
        phi, tau = draw_gauges(
            V,
            B,
            "N",
            10e-9,
            np.random.default_rng(np.random.SeedSequence([seed, 99])),
            ref=ref,
        )
        Y_ref = np.exp(1j * theta0) * Y

        phi_hat = np.zeros((V, B))
        tau_hat = np.zeros((V, B))
        Y_obs = apply_gauge(Y, phi, tau, FREQ)
        for v in range(V):
            for b in range(B):
                alignment = align_common_phase_and_delay(Y_obs[v, b], Y_ref[v, b], FREQ)
                phi_hat[v, b] = alignment.phase_rad
                tau_hat[v, b] = alignment.delay_s
        max_phase, max_delay = _gauge_errors(phi_hat, tau_hat, phi, tau, ref)
        assert max_phase <= 1e-9
        assert max_delay <= 1e-18
        offset = phi_hat[ref] + theta0
        assert abs(math.atan2(math.sin(offset), math.cos(offset))) <= 1e-9

        Y_obs_neg = apply_gauge(Y, -phi, -tau, FREQ)
        phi_neg = np.zeros((V, B))
        tau_neg = np.zeros((V, B))
        for v in range(V):
            for b in range(B):
                alignment = align_common_phase_and_delay(Y_obs_neg[v, b], Y_ref[v, b], FREQ)
                phi_neg[v, b] = alignment.phase_rad
                tau_neg[v, b] = alignment.delay_s
        neg_phase, neg_delay = _gauge_errors(phi_neg, tau_neg, phi, tau, ref)
        assert neg_phase > 1.0 or neg_delay > 1e-9

        sigma2 = float(np.mean(np.abs(Y) ** 2)) / 1000.0
        noise = math.sqrt(sigma2 / 2.0) * (
            rng.standard_normal(Y.shape) + 1j * rng.standard_normal(Y.shape)
        )
        Y_obs_n = apply_gauge(Y + noise, phi, tau, FREQ)
        phi_n = np.zeros((V, B))
        tau_n = np.zeros((V, B))
        for v in range(V):
            for b in range(B):
                alignment = align_common_phase_and_delay(Y_obs_n[v, b], Y_ref[v, b], FREQ)
                phi_n[v, b] = alignment.phase_rad
                tau_n[v, b] = alignment.delay_s
        noisy_phase, noisy_delay = _gauge_errors(phi_n, tau_n, phi, tau, ref)
        assert noisy_phase <= math.radians(0.2)
        assert noisy_delay <= 0.02e-9


def _gauge_errors(
    phi_hat: np.ndarray,
    tau_hat: np.ndarray,
    phi: np.ndarray,
    tau: np.ndarray,
    ref: tuple[int, int],
) -> tuple[float, float]:
    """Return the max wrapped phase/delay error after fixing the reference gauge."""
    phi_rel = phi_hat - phi_hat[ref]
    max_phase = 0.0
    max_delay = 0.0
    for v in range(phi.shape[0]):
        for b in range(phi.shape[1]):
            dphase = phi_rel[v, b] - phi[v, b]
            max_phase = max(max_phase, abs(math.atan2(math.sin(dphase), math.cos(dphase))))
            max_delay = max(
                max_delay,
                abs((tau_hat[v, b] - tau[v, b] + PERIOD / 2) % PERIOD - PERIOD / 2),
            )
    return max_phase, max_delay


def test_m1_chain_gauge_gt_recovered() -> None:
    """The M1 chain common phase and timing are recovered by alignment."""
    for seed in range(5):
        cfr = _multipath(1, 1, 8, 8, FREQ, seed)[0, 0]
        config = ImpairmentConfig(random_common_phase=True, timing_offset_std_ns=10.0)
        noise_var = float(np.mean(np.abs(cfr[0]) ** 2)) / 1000.0
        observed, gt = apply_impairments(
            cfr, FREQ, config, np.random.default_rng(seed), noise_variance=noise_var
        )
        alignment = align_common_phase_and_delay(observed, cfr[0], FREQ)
        dphase = alignment.phase_rad - gt["common_phase_rad"]
        phase_error = abs(math.atan2(math.sin(dphase), math.cos(dphase)))
        delay_error = abs(
            (alignment.delay_s - gt["timing_offset_s"] + PERIOD / 2) % PERIOD - PERIOD / 2
        )
        assert phase_error <= math.radians(0.2)
        assert delay_error <= 0.02e-9


def test_calibration_capture_40db() -> None:
    """The 40 dB calibration estimate meets the design tolerances."""
    config = ImpairmentConfig(element_gain_std_db=0.5, element_phase_std_deg=5.0)
    for seed in (0, 1, 2):
        rng = np.random.default_rng(np.random.SeedSequence(seed))
        G = np.stack([draw_element_errors(config, 8, 8, rng).complex_gain for _ in range(4)])
        cal = calibration_capture(G, 40.0, rng, num_bins=64)
        ratio = cal.element_gain_est / cal.element_gain
        assert np.max(np.abs(20.0 * np.log10(np.abs(ratio)))) <= 0.1
        assert np.max(np.abs(np.angle(ratio))) <= math.radians(1.0)
        assert cal.capture.shape == (4, 8, 8, 64)
        np.testing.assert_allclose(cal.residual, G / cal.element_gain_est, rtol=1e-14)
        np.testing.assert_allclose(cal.noise_var, 1e-4, rtol=1e-14)

    rng = np.random.default_rng(np.random.SeedSequence(123))
    G = np.stack([draw_element_errors(config, 8, 8, rng).complex_gain for _ in range(4)])
    cal = calibration_capture(G, 20.0, rng, num_bins=64)
    rms = math.sqrt(float(np.mean(np.abs(cal.element_gain_est / cal.element_gain - 1.0) ** 2)))
    target = math.sqrt(10.0 ** (-2.0) / 64.0)
    assert 0.85 * target <= rms <= 1.15 * target


def test_calibration_stream_and_infinite_snr() -> None:
    """At infinite SNR the capture is noise-free and one stream draw is consumed."""
    config = ImpairmentConfig(element_gain_std_db=0.5, element_phase_std_deg=5.0)
    rng = np.random.default_rng(np.random.SeedSequence(22))
    G = np.stack([draw_element_errors(config, 8, 8, rng).complex_gain for _ in range(4)])

    stream = np.random.default_rng(np.random.SeedSequence(33))
    cal = calibration_capture(G, math.inf, stream, num_bins=64)
    np.testing.assert_array_equal(cal.capture, np.broadcast_to(G[..., None], cal.capture.shape))
    np.testing.assert_allclose(cal.element_gain_est, G, rtol=1e-13, atol=1e-15)
    assert cal.noise_var == 0.0

    fresh = np.random.default_rng(np.random.SeedSequence(33))
    fresh.standard_normal((2, 4, 8, 8, 64))
    np.testing.assert_array_equal(stream.standard_normal(4), fresh.standard_normal(4))

    bad = np.random.default_rng(0)
    for snr in (math.nan, -math.inf):
        with pytest.raises(ValueError):
            calibration_capture(np.ones((2, 2)), snr, bad, num_bins=4)
    for bins in (0, True, -1):
        with pytest.raises(ValueError):
            calibration_capture(np.ones((2, 2)), 10.0, bad, num_bins=bins)
    with pytest.raises(ValueError):
        calibration_capture(np.ones(4), 10.0, bad, num_bins=4)
    with pytest.raises(ValueError):
        calibration_capture(np.full((2, 2), np.nan, dtype=np.complex128), 10.0, bad, num_bins=4)


def test_pose_perturbation() -> None:
    """Pose perturbation draws, Rodrigues rotations and statistics are consistent."""
    V = 3
    rng = np.random.default_rng(np.random.SeedSequence(7))
    positions = rng.standard_normal((V, 3))
    rotations = np.tile(np.eye(3), (V, 1, 1))
    for v in range(V):
        q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
        if np.linalg.det(q) < 0.0:
            q[:, 0] *= -1.0
        rotations[v] = q

    stream = np.random.default_rng(np.random.SeedSequence(8))
    zero = perturb_poses(positions, rotations, stream, position_std_m=0.0, rotation_std_deg=0.0)
    np.testing.assert_array_equal(zero.positions, positions)
    np.testing.assert_array_equal(zero.rotations, rotations)
    np.testing.assert_array_equal(zero.delta_position, np.zeros((V, 3)))
    np.testing.assert_array_equal(zero.rotation_vector, np.zeros((V, 3)))
    fresh = np.random.default_rng(np.random.SeedSequence(8))
    fresh.normal(0.0, 0.0, size=(V, 3))
    fresh.normal(0.0, 0.0, size=(V, 3))
    np.testing.assert_array_equal(stream.standard_normal(4), fresh.standard_normal(4))

    vectors = rng.standard_normal((50, 3))
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors * rng.uniform(0.0, np.pi, size=(50, 1))
    np.testing.assert_allclose(
        rotation_from_vector(vectors), Rotation.from_rotvec(vectors).as_matrix(), atol=1e-12
    )
    np.testing.assert_array_equal(rotation_from_vector(np.zeros(3)), np.eye(3))

    big = 4000
    pos_big = rng.standard_normal((big, 3))
    # Random true rotations, so a right (body-frame) perturbation would be caught.
    rot_big = Rotation.random(big, random_state=rng).as_matrix()
    perturbed = perturb_poses(pos_big, rot_big, rng, position_std_m=0.01, rotation_std_deg=2.0)
    for axis in range(3):
        assert 0.95 * 0.01 <= np.std(perturbed.delta_position[:, axis]) <= 1.05 * 0.01
        assert (
            0.95 * math.radians(2.0)
            <= np.std(perturbed.rotation_vector[:, axis])
            <= 1.05 * math.radians(2.0)
        )
    np.testing.assert_allclose(
        perturbed.positions - pos_big, perturbed.delta_position, rtol=1e-15, atol=1e-15
    )
    stacked = perturbed.rotations @ np.transpose(rot_big, (0, 2, 1))
    np.testing.assert_allclose(stacked, rotation_from_vector(perturbed.rotation_vector), atol=1e-12)
    gram = perturbed.rotations @ np.transpose(perturbed.rotations, (0, 2, 1))
    np.testing.assert_allclose(gram, np.tile(np.eye(3), (big, 1, 1)), atol=1e-12)
    np.testing.assert_allclose(np.linalg.det(perturbed.rotations), 1.0, atol=1e-12)

    tilted = perturb_poses(pos_big, rot_big, rng, position_std_m=0.0, rotation_std_deg=5.0)
    relative = tilted.rotations @ np.transpose(rot_big, (0, 2, 1))
    trace = np.trace(relative, axis1=1, axis2=2)
    angle = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    np.testing.assert_allclose(angle, np.linalg.norm(tilted.rotation_vector, axis=1), atol=1e-6)

    first = perturb_poses(
        positions,
        rotations,
        np.random.default_rng(np.random.SeedSequence(55)),
        rotation_std_deg=0.0,
    )
    second = perturb_poses(
        positions,
        rotations,
        np.random.default_rng(np.random.SeedSequence(55)),
        rotation_std_deg=3.0,
    )
    np.testing.assert_array_equal(first.delta_position, second.delta_position)


def test_capture_gt_record() -> None:
    """capture_gt_record is JSON-safe, ordered and faithful to its sources."""
    V, B, R, C, n = 3, 2, 8, 8, 32
    freq = frequency_offsets(BANDWIDTH_HZ, n).astype(np.float64)
    Y = _multipath(V, B, R, C, freq, seed=6)
    G = _stacked_gains(V, R, C, seed=12)
    _, gt = apply_hardware_impairments(
        Y, element_gain=G, front_to_back_db=20.0, noise_var_abs=0.02, rng=np.random.default_rng(8)
    )
    phi, tau = draw_gauges(
        V, B, "N", 10e-9, np.random.default_rng(np.random.SeedSequence(77)), ref=(0, 0)
    )
    pose = perturb_poses(
        np.zeros((V, 3)),
        np.tile(np.eye(3), (V, 1, 1)),
        np.random.default_rng(np.random.SeedSequence(5)),
        position_std_m=0.01,
        rotation_std_deg=1.0,
    )
    cal = calibration_capture(
        G, 30.0, np.random.default_rng(np.random.SeedSequence(9)), num_bins=16
    )
    record = capture_gt_record(gt, 2, 1, gauge=(phi, tau), pose=pose, calibration=cal)
    json.dumps(record, allow_nan=False)

    expected_keys = [
        "view",
        "bs",
        "front_to_back_db",
        "front_to_back_gain",
        "element_gain_db",
        "element_phase_rad",
        "noise_variance",
        "signal_power",
        "noise_power",
        "expected_snr_db",
        "achieved_snr_db",
        "phase_rad",
        "delay_s",
        "pose_delta_position_m",
        "pose_rotation_vector_rad",
        "calibration_snr_db",
        "element_gain_est_db",
        "element_phase_est_rad",
    ]
    assert list(record) == expected_keys
    assert record["view"] == 2 and record["bs"] == 1
    assert record["front_to_back_db"] == 20.0
    assert record["front_to_back_gain"] == pytest.approx(float(gt["front_to_back_gain"]))
    assert record["noise_variance"] == pytest.approx(float(gt["noise_var"]))
    assert record["signal_power"] == pytest.approx(float(gt["signal_power"][2, 1]))
    assert record["noise_power"] == pytest.approx(float(gt["noise_power"][2, 1]))
    assert record["expected_snr_db"] == pytest.approx(float(gt["expected_snr_db"][2, 1]))
    assert record["achieved_snr_db"] == pytest.approx(float(gt["achieved_snr_db"][2, 1]))
    assert record["phase_rad"] == pytest.approx(float(phi[2, 1]))
    assert record["delay_s"] == pytest.approx(float(tau[2, 1]))
    assert record["pose_delta_position_m"] == [float(x) for x in pose.delta_position[2]]
    assert record["pose_rotation_vector_rad"] == [float(x) for x in pose.rotation_vector[2]]
    assert record["calibration_snr_db"] == 30.0
    np.testing.assert_allclose(
        np.array(record["element_gain_db"]), 20.0 * np.log10(np.abs(G[2])), rtol=1e-12
    )
    np.testing.assert_allclose(np.array(record["element_phase_rad"]), np.angle(G[2]), atol=1e-15)
    est = cal.element_gain_est[2]
    np.testing.assert_allclose(
        np.array(record["element_gain_est_db"]), 20.0 * np.log10(np.abs(est)), rtol=1e-12
    )
    np.testing.assert_allclose(np.array(record["element_phase_est_rad"]), np.angle(est), atol=1e-15)

    plain = capture_gt_record(gt, 0, 0)
    for key in (
        "phase_rad",
        "delay_s",
        "pose_delta_position_m",
        "pose_rotation_vector_rad",
        "calibration_snr_db",
        "element_gain_est_db",
        "element_phase_est_rad",
    ):
        assert plain[key] is None
    json.dumps(plain, allow_nan=False)

    _, gt0 = apply_hardware_impairments(Y, element_gain=G, front_to_back_db=20.0, noise_var_abs=0.0)
    silent = capture_gt_record(gt0, 1, 1)
    assert silent["expected_snr_db"] is None
    assert silent["achieved_snr_db"] is None

    with pytest.raises(ValueError):
        capture_gt_record(gt, 3, 0)
    with pytest.raises(ValueError):
        capture_gt_record(gt, 0, 2)


def test_validation() -> None:
    """Invalid inputs raise ValueError."""
    Y = _multipath(2, 2, 2, 2, FREQ, seed=3)
    with pytest.raises(ValueError):
        apply_hardware_impairments(Y[:, :, 0])
    with pytest.raises(ValueError):
        apply_hardware_impairments(Y[:, :, :1])
    non_finite = Y.copy()
    non_finite[0, 0, 0, 0, 0, 0] = np.inf
    with pytest.raises(ValueError):
        apply_hardware_impairments(non_finite)
    with pytest.raises(ValueError):
        apply_hardware_impairments(Y, element_gain=np.ones((3, 3), dtype=np.complex128))
    with pytest.raises(ValueError):
        apply_hardware_impairments(Y, element_gain=np.full((2, 2), np.inf, dtype=np.complex128))
    with pytest.raises(ValueError):
        apply_hardware_impairments(Y, noise_var_abs=-1.0)
    with pytest.raises(ValueError):
        apply_hardware_impairments(Y, noise_var_abs=math.nan)
    with pytest.raises(ValueError):
        apply_hardware_impairments(
            Y, noise_var_abs=1.0, noise=np.zeros((2, 2, 2, 2, 3), dtype=np.complex128)
        )
    with pytest.raises(ValueError):
        apply_hardware_impairments(Y, noise_var_abs=1.0)
    with pytest.raises(ValueError):
        apply_hardware_impairments(Y, front_to_back_db=math.inf)

    positions = np.zeros((2, 3))
    rotations = np.tile(np.eye(3), (2, 1, 1))
    with pytest.raises(ValueError):
        perturb_poses(positions, rotations, np.random.default_rng(0), position_std_m=-0.5)
    with pytest.raises(ValueError):
        perturb_poses(positions, rotations, np.random.default_rng(0), rotation_std_deg=math.inf)
    with pytest.raises(ValueError):
        perturb_poses(positions, rotations[:1], np.random.default_rng(0))
    with pytest.raises(ValueError):
        rotation_from_vector(np.zeros((4, 2)))
