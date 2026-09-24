"""Unit tests for the synchronisation gauges and paired data tracks."""

from __future__ import annotations

import cmath
import math

import numpy as np
import pytest

from plateau_rt.domain.rf_tomography import observables
from plateau_rt.domain.rf_tomography.sync import (
    DEFAULT_SIGMA_T,
    GAUGE_MODES,
    GAUGE_STREAM_TAG,
    HardwareConfig,
    TrackSeeds,
    apply_gauge,
    capture_power,
    draw_gauges,
    gauge_factor,
    make_tracks,
    reference_power,
)

N = 64
DELTA_F = 100e6 / 64
FREQ = (np.arange(N) - N // 2) * DELTA_F
PERIOD = 640e-9


def _grid(v: int = 4, b: int = 2, r: int = 8, c: int = 8) -> tuple[np.ndarray, int, int]:
    """Return random clean data and its (R, C) sizes."""
    rng = np.random.default_rng(np.random.SeedSequence(12345))
    Y = (
        rng.standard_normal((v, b, 2, r, c, N)) + 1j * rng.standard_normal((v, b, 2, r, c, N))
    ).astype(np.complex128)
    return Y, r, c


def test_reference_gauge_is_exactly_zero() -> None:
    """The reference capture gauge is exactly (0.0, 0.0) in every mode."""
    for mode in GAUGE_MODES:
        for ref in [(0, 0), (2, 1)]:
            for sigma_t in (1e-9, "uniform"):
                rng = np.random.default_rng(np.random.SeedSequence(7))
                phi, tau = draw_gauges(5, 3, mode, sigma_t, rng, ref=ref, period=PERIOD)
                assert phi[ref] == 0.0
                assert tau[ref] == 0.0
                assert np.all(phi >= 0.0) and np.all(phi < 2 * np.pi)


def test_s_and_s_tau_modes() -> None:
    """S is all zeros; S_tau has zero tau and non-trivial phi."""
    rng = np.random.default_rng(np.random.SeedSequence(1))
    phi, tau = draw_gauges(4, 2, "S", None, rng)
    np.testing.assert_array_equal(phi, np.zeros((4, 2)))
    np.testing.assert_array_equal(tau, np.zeros((4, 2)))
    rng = np.random.default_rng(np.random.SeedSequence(1))
    phi, tau = draw_gauges(4, 2, "S_tau", None, rng)
    np.testing.assert_array_equal(tau, np.zeros((4, 2)))
    assert np.any(phi != 0.0)


def test_n_mode_statistics() -> None:
    """N-mode gauges are uniform in phase and Gaussian/uniform in delay."""
    rng = np.random.default_rng(np.random.SeedSequence(11))
    phi, tau = draw_gauges(64, 4, "N", 10e-9, rng, period=None)
    assert abs(np.mean(np.exp(1j * phi))) < 0.25
    assert abs(np.std(tau) - 10e-9) / 10e-9 < 0.15
    assert abs(np.mean(tau)) < 0.25 * 10e-9
    rng = np.random.default_rng(np.random.SeedSequence(11))
    phi_u, tau_u = draw_gauges(64, 4, "N", "uniform", rng, period=PERIOD)
    assert np.all(tau_u >= -PERIOD / 2) and np.all(tau_u < PERIOD / 2)
    assert abs(np.std(tau_u) - PERIOD / math.sqrt(12)) / (PERIOD / math.sqrt(12)) < 0.15
    worst = 0.0
    for v in range(64):
        for vp in range(64):
            for b in range(4):
                for bp in range(4):
                    dd = phi[v, b] - phi[v, bp] - phi[vp, b] + phi[vp, bp]
                    worst = max(worst, abs(math.atan2(math.sin(dd), math.cos(dd))))
    assert worst > 0.1


def test_n_sep_structure() -> None:
    """N_sep gauges factorise into per-view plus per-BS clocks."""
    for ref in [(0, 0), (4, 2)]:
        rng = np.random.default_rng(np.random.SeedSequence(5))
        phi, tau = draw_gauges(6, 3, "N_sep", 10e-9, rng, ref=ref)
        for v in range(6):
            for vp in range(6):
                for b in range(3):
                    for bp in range(3):
                        dd = phi[v, b] - phi[v, bp] - phi[vp, b] + phi[vp, bp]
                        assert abs(math.atan2(math.sin(dd), math.cos(dd))) < 1e-12
                        assert abs(tau[v, b] - tau[v, bp] - tau[vp, b] + tau[vp, bp]) < 1e-20
        col_gap = tau[:, 1] - tau[:, 0]
        assert np.all(np.abs(col_gap - col_gap[0]) < 1e-20) and col_gap[0] != 0.0
        row_gap = tau[1, :] - tau[0, :]
        assert np.all(np.abs(row_gap - row_gap[0]) < 1e-20) and row_gap[0] != 0.0
        p_gap = np.array([math.atan2(math.sin(a), math.cos(a)) for a in (phi[:, 1] - phi[:, 0])])
        assert np.all(np.abs(p_gap - p_gap[0]) < 1e-12) and p_gap[0] != 0.0


def test_draw_gauges_deterministic_and_validated() -> None:
    """Gauge draws are reproducible, aliased and validated."""
    rng_a = np.random.default_rng(np.random.SeedSequence(3))
    rng_b = np.random.default_rng(np.random.SeedSequence(3))
    phi_a, tau_a = draw_gauges(4, 2, "N", 1e-9, rng_a, period=PERIOD)
    phi_b, tau_b = draw_gauges(4, 2, "N", 1e-9, rng_b, period=PERIOD)
    np.testing.assert_array_equal(phi_a, phi_b)
    np.testing.assert_array_equal(tau_a, tau_b)
    rng_c = np.random.default_rng(np.random.SeedSequence(4))
    phi_c, _ = draw_gauges(4, 2, "N", 1e-9, rng_c, period=PERIOD)
    assert not np.array_equal(phi_a, phi_c)
    for alias, canon in [("N-sep", "N_sep"), ("S_τ", "S_tau")]:
        r1 = np.random.default_rng(np.random.SeedSequence(9))
        r2 = np.random.default_rng(np.random.SeedSequence(9))
        sig = None if canon == "S_tau" else 1e-9
        p1, t1 = draw_gauges(4, 2, alias, sig, r1, period=PERIOD)
        p2, t2 = draw_gauges(4, 2, canon, sig, r2, period=PERIOD)
        np.testing.assert_array_equal(p1, p2)
        np.testing.assert_array_equal(t1, t2)
    rng = np.random.default_rng(np.random.SeedSequence(0))
    with pytest.raises(ValueError):
        draw_gauges(4, 2, "bogus", 1e-9, rng)
    with pytest.raises(ValueError):
        draw_gauges(4, 2, "N", -1e-9, rng)
    with pytest.raises(ValueError):
        draw_gauges(4, 2, "N", None, rng)
    with pytest.raises(ValueError):
        draw_gauges(4, 2, "N", "uniform", rng, period=None)
    with pytest.raises(ValueError):
        draw_gauges(4, 2, "N", 1e-9, rng, ref=(4, 0))
    with pytest.raises(ValueError):
        draw_gauges(4, 2, "N", 1e-9, np.random.RandomState(0))  # type: ignore[arg-type]


def test_gauge_factor_is_unit_modulus() -> None:
    """The gauge has unit modulus and preserves |Y|."""
    rng = np.random.default_rng(np.random.SeedSequence(21))
    phi = rng.uniform(0, 2 * np.pi, size=(3, 2))
    tau = rng.uniform(-1e-6, 1e-6, size=(3, 2))
    gauge = gauge_factor(phi, tau, FREQ)
    assert gauge.dtype == np.complex128
    assert np.max(np.abs(np.abs(gauge) - 1.0)) < 1e-15
    Y, _, _ = _grid(v=3, b=2)
    out = apply_gauge(Y, phi, tau, FREQ)
    assert out.dtype == np.complex128
    rel = np.abs(np.abs(out) - np.abs(Y)) / np.maximum(np.abs(Y), 1e-300)
    assert np.max(rel) < 1e-12


def test_apply_gauge_matches_explicit_formula() -> None:
    """apply_gauge matches an explicit cmath loop and inverts with conj."""
    rng = np.random.default_rng(np.random.SeedSequence(31))
    phi = rng.uniform(0, 2 * np.pi, size=(2, 2))
    tau = rng.uniform(-5e-9, 5e-9, size=(2, 2))
    Y, _, _ = _grid(v=2, b=2, r=3, c=3)
    out = apply_gauge(Y, phi, tau, FREQ)
    expected = np.zeros_like(Y)
    for v in range(2):
        for b in range(2):
            for n in range(N):
                g = cmath.exp(1j * phi[v, b]) * cmath.exp(-2j * math.pi * FREQ[n] * tau[v, b])
                for h in range(2):
                    for r in range(3):
                        for c in range(3):
                            expected[v, b, h, r, c, n] = Y[v, b, h, r, c, n] * g
    rel = np.abs(out - expected) / np.maximum(np.abs(expected), 1e-300)
    assert np.max(rel) < 1e-12
    gauge = np.zeros((2, 2, N), dtype=np.complex128)
    for v in range(2):
        for b in range(2):
            for n in range(N):
                gauge[v, b, n] = cmath.exp(1j * phi[v, b]) * cmath.exp(
                    -2j * math.pi * FREQ[n] * tau[v, b]
                )
    back = out * np.conj(gauge)[:, :, None, None, None, :]
    rel = np.abs(back - Y) / np.maximum(np.abs(Y), 1e-300)
    assert np.max(rel) < 1e-12


def test_gauge_is_a_delay_shift() -> None:
    """A positive tau shifts the path to a later arrival."""
    a = 1.5 + 0.5j
    tau0 = 3.2e-9
    phi1 = 0.7
    Y = np.zeros((1, 1, 1, 2, 2, N), dtype=np.complex128)
    for n in range(N):
        Y[..., n] = a * cmath.exp(-2j * math.pi * FREQ[n] * tau0)
    tau1 = 4.1e-9
    out = apply_gauge(Y, np.array([[phi1]]), np.array([[tau1]]), FREQ)
    for n in range(N):
        want = a * cmath.exp(1j * phi1) * cmath.exp(-2j * math.pi * FREQ[n] * (tau0 + tau1))
        assert abs(out[0, 0, 0, 0, 0, n] - want) / abs(want) < 1e-12
    out_dc = apply_gauge(Y, np.zeros((1, 1)), np.array([[tau1]]), FREQ)
    np.testing.assert_array_equal(out_dc[..., N // 2], Y[..., N // 2])
    # Integration with T04 observables: integer delay-bin shift.
    rng = np.random.default_rng(np.random.SeedSequence(41))
    Yr = (
        rng.standard_normal((2, 2, 1, 4, 4, N)) + 1j * rng.standard_normal((2, 2, 1, 4, 4, N))
    ).astype(np.complex128)
    tau_bin = 5.0 / (N * DELTA_F)
    Yn = apply_gauge(Yr, np.zeros((2, 2)), np.full((2, 2), tau_bin), FREQ)
    id_s = observables.extract(Yr, "ID").data
    id_n = observables.extract(Yn, "ID").data
    rel = np.abs(id_n - np.roll(id_s, 5, axis=-1)) / np.maximum(np.abs(id_s), 1e-300)
    assert np.max(rel) < 1e-10
    i_s = observables.extract(Yr, "I").data
    i_n = observables.extract(Yn, "I").data
    rel = np.abs(i_n - i_s) / np.maximum(np.abs(i_s), 1e-300)
    assert np.max(rel) < 1e-12


def test_capture_power_sums_hemisphere_powers() -> None:
    """capture_power sums |front|^2 + |back|^2 rather than |front + back|^2."""
    rng = np.random.default_rng(np.random.SeedSequence(51))
    wave = (
        rng.standard_normal((2, 2, 4, 4, N)) + 1j * rng.standard_normal((2, 2, 4, 4, N))
    ).astype(np.complex128)
    Y = np.stack([wave, wave], axis=2)
    got = capture_power(Y)
    expected = np.zeros((2, 2))
    for v in range(2):
        for b in range(2):
            total = 0.0
            for h in range(2):
                for r in range(4):
                    for c in range(4):
                        for n in range(N):
                            total += abs(Y[v, b, h, r, c, n]) ** 2
            expected[v, b] = total / (4 * 4 * N)
    np.testing.assert_allclose(got, expected, rtol=1e-12)
    assert got.dtype == np.float64


def test_reference_power_lower_median_of_los_visible() -> None:
    """reference_power picks the lower median of the LoS-visible captures."""
    targets = np.array([[5.0, 1.0], [3.0, 9.0], [7.0, 2.0]])
    rng = np.random.default_rng(np.random.SeedSequence(61))
    Y = np.zeros((3, 2, 2, 2, 2, N), dtype=np.complex128)
    for v in range(3):
        for b in range(2):
            mag_f = math.sqrt(targets[v, b] / 4.0)
            mag_b = math.sqrt(3.0 * targets[v, b] / 4.0)
            ph = rng.uniform(0, 2 * math.pi, size=(2, 2, 2, N))
            Y[v, b, 0] = mag_f * np.exp(1j * ph[0])
            Y[v, b, 1] = mag_b * np.exp(1j * ph[1])
    los = np.array([[True, False], [True, True], [False, True]])
    p_ref, c_ref = reference_power(Y, los)
    assert c_ref == (1, 0)
    assert abs(p_ref - 3.0) / 3.0 < 1e-12
    los2 = np.array([[True, False], [True, True], [True, True]])
    _, c_ref2 = reference_power(Y, los2)
    assert c_ref2 == (0, 0)
    with pytest.raises(ValueError):
        reference_power(Y, np.zeros((3, 2), dtype=bool))
    with pytest.raises(ValueError):
        reference_power(Y, np.zeros((2, 2), dtype=bool))


def test_one_sigma2_for_every_capture_and_hemisphere() -> None:
    """One noise variance serves every capture and hemisphere."""
    scales = 10.0 ** (np.arange(8, dtype=np.float64).reshape(4, 2) - 3.0)
    Y, R, C = _grid()
    for v in range(4):
        for b in range(2):
            Y[v, b, 0] *= scales[v, b]
            Y[v, b, 1] *= scales[v, b] * 0.1
    los = np.ones((4, 2), dtype=bool)
    p_ref, c_ref = reference_power(Y, los)
    tracks, gt = make_tracks(Y, 30.0, p_ref, TrackSeeds(17, 0), freq_offsets=FREQ, c_ref=c_ref)
    assert gt["sigma2"] == pytest.approx(p_ref / 1000.0, rel=1e-14)
    assert gt["sigma2"] == pytest.approx(p_ref / (10.0 ** (30.0 / 10.0)), rel=1e-14)
    realised = tracks["ideal-S"] - Y
    for v in range(4):
        for b in range(2):
            for h in range(2):
                cell = realised[v, b, h]
                ratio = float(np.mean(np.abs(cell) ** 2)) / gt["sigma2"]
                assert 0.92 <= ratio <= 1.08
                assert abs(complex(np.mean(cell**2))) / gt["sigma2"] < 0.08
    eps = gt["calibration_residual"]
    collapsed = Y[:, :, 0] + gt["front_to_back_gain"] * Y[:, :, 1]
    obs_noise = (tracks["observed-S"][:, :, 0] - eps[:, None, :, :, None] * collapsed) / eps[
        :, None, :, :, None
    ]
    for v in range(4):
        for b in range(2):
            ratio = float(np.mean(np.abs(obs_noise[v, b]) ** 2)) / gt["sigma2"]
            assert 0.92 <= ratio <= 1.08


def test_capture_noise_stream_is_pinned() -> None:
    """The realised noise of a capture matches its SeedSequence stream."""
    Y, R, C = _grid()
    ds, real = 13, 2
    los = np.ones((4, 2), dtype=bool)
    p_ref, _ = reference_power(Y, los)
    tracks, gt = make_tracks(Y, 30.0, p_ref, TrackSeeds(ds, real), freq_offsets=FREQ)
    rng = np.random.default_rng(np.random.SeedSequence([ds, 2, 1, real]))
    z = rng.standard_normal((2, 2, R, C, N))
    want = math.sqrt(gt["sigma2"] / 2.0) * (z[0] + 1j * z[1])
    np.testing.assert_allclose(tracks["ideal-S"][2, 1] - Y[2, 1], want, rtol=1e-12, atol=0)


def test_hardware_stream_is_pinned() -> None:
    """The hardware draws of view 2 match its SeedSequence stream."""
    Y, R, C = _grid()
    ds, real = 43, 7
    _, gt = make_tracks(Y, 30.0, 2.5, TrackSeeds(ds, real), freq_offsets=FREQ)
    rng = np.random.default_rng(np.random.SeedSequence([ds, 2, 0, real]))
    rng.standard_normal((2, 2, R, C, N))
    gain_db = rng.normal(0.0, 0.5, size=(R, C))
    phase = np.deg2rad(rng.normal(0.0, 5.0, size=(R, C)))
    zc = rng.standard_normal((2, R, C, N))
    w_cal = math.sqrt(1e-3 / 2.0) * (zc[0] + 1j * zc[1])
    g = 10.0 ** (gain_db / 20.0) * np.exp(1j * phase)
    np.testing.assert_allclose(gt["element_gain"][2], g, rtol=1e-12, atol=0)
    g_est = (g[..., None] * (1.0 + w_cal)).mean(axis=-1)
    np.testing.assert_allclose(gt["element_gain_est"][2], g_est, rtol=1e-12, atol=0)
    np.testing.assert_allclose(gt["calibration_residual"][2], g / g_est, rtol=1e-12, atol=0)


def test_gauge_streams_are_pinned() -> None:
    """The gauge draws match their dedicated SeedSequence streams."""
    Y, _, _ = _grid()
    ds, real = 29, 4
    ref = (1, 1)
    sigma_t = 3e-9
    _, gt = make_tracks(
        Y, 30.0, 2.5, TrackSeeds(ds, real), freq_offsets=FREQ, ref=ref, sigma_t=sigma_t
    )
    rng_0 = np.random.default_rng(np.random.SeedSequence([ds, real, GAUGE_STREAM_TAG, 0]))
    rng_1 = np.random.default_rng(np.random.SeedSequence([ds, real, GAUGE_STREAM_TAG, 1]))
    rng_2 = np.random.default_rng(np.random.SeedSequence([ds, real, GAUGE_STREAM_TAG, 2]))
    phi_n, tau_n = draw_gauges(4, 2, "N", sigma_t, rng_0, ref=ref)
    phi_sep, tau_sep = draw_gauges(4, 2, "N_sep", sigma_t, rng_1, ref=ref)
    phi_st, tau_st = draw_gauges(4, 2, "S_tau", None, rng_2, ref=ref)
    np.testing.assert_array_equal(gt["gauges"]["ideal-N"][0], phi_n)
    np.testing.assert_array_equal(gt["gauges"]["ideal-N"][1], tau_n)
    np.testing.assert_array_equal(gt["gauges"]["N-sep"][0], phi_sep)
    np.testing.assert_array_equal(gt["gauges"]["N-sep"][1], tau_sep)
    np.testing.assert_array_equal(gt["gauges"]["S_tau"][0], phi_st)
    np.testing.assert_array_equal(gt["gauges"]["S_tau"][1], tau_st)
    assert not np.array_equal(gt["gauges"]["ideal-N"][0], gt["gauges"]["N-sep"][0])


def test_s_and_n_tracks_share_the_noise_realisation() -> None:
    """Every N-type track is its S base times the unit-modulus gauge."""
    Y, _, _ = _grid()
    tracks, gt = make_tracks(Y, 30.0, 2.5, TrackSeeds(23, 1), freq_offsets=FREQ, ref=(1, 0))
    assert tuple(tracks) == ("ideal-S", "ideal-N", "observed-S", "observed-N", "N-sep", "S_tau")
    for name in tracks:
        assert tracks[name].dtype == np.complex128
    for name, base_key in [
        ("ideal-N", "ideal-S"),
        ("observed-N", "observed-S"),
        ("N-sep", "ideal-S"),
        ("S_tau", "ideal-S"),
    ]:
        phi, tau = gt["gauges"][name]
        gauge = np.zeros((4, 2, N), dtype=np.complex128)
        for v in range(4):
            for b in range(2):
                for n in range(N):
                    gauge[v, b, n] = cmath.exp(1j * phi[v, b]) * cmath.exp(
                        -2j * math.pi * FREQ[n] * tau[v, b]
                    )
        base = tracks[base_key]
        rel = np.abs(tracks[name] * np.conj(gauge)[:, :, None, None, None, :] - base) / np.maximum(
            np.abs(base), 1e-300
        )
        assert np.max(rel) < 1e-12
    np.testing.assert_array_equal(gt["gauges"]["ideal-N"][0], gt["gauges"]["observed-N"][0])
    np.testing.assert_array_equal(gt["gauges"]["ideal-N"][1], gt["gauges"]["observed-N"][1])
    for name in tracks:
        phi, tau = gt["gauges"][name]
        assert phi[1, 0] == 0.0 and tau[1, 0] == 0.0
    for name in ("ideal-S", "observed-S"):
        np.testing.assert_array_equal(gt["gauges"][name][0], np.zeros((4, 2)))
        np.testing.assert_array_equal(gt["gauges"][name][1], np.zeros((4, 2)))
    phi_sep, tau_sep = gt["gauges"]["N-sep"]
    for v in (0, 2):
        for vp in (1, 3):
            for b in (0, 1):
                for bp in (0, 1):
                    dd = phi_sep[v, b] - phi_sep[v, bp] - phi_sep[vp, b] + phi_sep[vp, bp]
                    assert abs(math.atan2(math.sin(dd), math.cos(dd))) < 1e-12
                    assert (
                        abs(tau_sep[v, b] - tau_sep[v, bp] - tau_sep[vp, b] + tau_sep[vp, bp])
                        < 1e-20
                    )
    np.testing.assert_array_equal(gt["gauges"]["S_tau"][1], np.zeros((4, 2)))


def test_tracks_are_paired_and_reproducible() -> None:
    """Seeds pin the tracks; hardware/sigma_t leave ideal-S untouched."""
    Y, _, _ = _grid()
    kwargs = {"freq_offsets": FREQ}
    t1, _ = make_tracks(Y, 30.0, 2.5, TrackSeeds(23, 1), **kwargs)
    t2, _ = make_tracks(Y, 30.0, 2.5, TrackSeeds(23, 1), **kwargs)
    for name in t1:
        np.testing.assert_array_equal(t1[name], t2[name])
    t3, _ = make_tracks(Y, 30.0, 2.5, TrackSeeds(23, 2), **kwargs)
    assert not np.array_equal(t1["ideal-S"], t3["ideal-S"])
    alt_hw, _ = make_tracks(
        Y, 30.0, 2.5, TrackSeeds(23, 1), hardware=HardwareConfig(element_gain_std_db=2.0), **kwargs
    )
    np.testing.assert_array_equal(t1["ideal-S"], alt_hw["ideal-S"])
    alt_sig, _ = make_tracks(Y, 30.0, 2.5, TrackSeeds(23, 1), sigma_t=50e-9, **kwargs)
    np.testing.assert_array_equal(t1["ideal-S"], alt_sig["ideal-S"])
    np.testing.assert_array_equal(t1["observed-S"], alt_sig["observed-S"])


def test_observed_track() -> None:
    """The observed track collapses hemispheres through the calibrated chain."""
    Y, R, C = _grid()
    hw = HardwareConfig(
        element_gain_std_db=0.0,
        element_phase_std_deg=0.0,
        front_to_back_db=10.0,
        calibration_snr_db=math.inf,
    )
    tracks, gt = make_tracks(Y, 30.0, 2.5, TrackSeeds(29, 0), freq_offsets=FREQ, hardware=hw)
    assert tracks["observed-S"].shape == (4, 2, 1, R, C, N)
    w_front = tracks["ideal-S"][:, :, 0] - Y[:, :, 0]
    want = (Y[:, :, 0] + 10.0 ** (-0.5) * Y[:, :, 1] + w_front)[:, :, None]
    np.testing.assert_allclose(tracks["observed-S"], want, rtol=1e-12, atol=0)
    tracks_d, gt_d = make_tracks(Y, 30.0, 2.5, TrackSeeds(29, 0), freq_offsets=FREQ)
    gain = gt_d["element_gain"]
    gain_db = 20.0 * np.log10(np.abs(gain))
    assert abs(np.std(gain_db) - 0.5) / 0.5 < 0.2
    assert abs(np.std(np.angle(gain)) - math.radians(5.0)) / math.radians(5.0) < 0.2
    eps = gt_d["calibration_residual"]
    collapsed = Y[:, :, 0] + gt_d["front_to_back_gain"] * Y[:, :, 1]
    front = tracks_d["ideal-S"][:, :, 0] - Y[:, :, 0]
    for b in range(2):
        np.testing.assert_allclose(
            tracks_d["observed-S"][:, b, 0],
            eps[..., None] * (collapsed[:, b] + front[:, b]),
            rtol=1e-12,
            atol=0,
        )
    resid = math.sqrt(float(np.mean(np.abs(eps - 1.0) ** 2)))
    target = math.sqrt(10.0 ** (-3.0) / N)
    assert 0.8 * target <= resid <= 1.25 * target


def test_snr_records() -> None:
    """Expected/achieved SNR records match independent power ratios."""
    Y, _, _ = _grid()
    los = np.ones((4, 2), dtype=bool)
    p_ref, c_ref = reference_power(Y, los)
    tracks, gt = make_tracks(Y, 30.0, p_ref, TrackSeeds(37, 0), freq_offsets=FREQ, c_ref=c_ref)
    pc = np.zeros((4, 2))
    for v in range(4):
        for b in range(2):
            total = 0.0
            for h in range(2):
                total += float(np.mean(np.abs(Y[v, b, h]) ** 2))
            pc[v, b] = total
    np.testing.assert_allclose(
        gt["expected_snr_db"], 10.0 * np.log10(pc / gt["sigma2"]), rtol=1e-10
    )
    assert abs(gt["expected_snr_db"][c_ref] - 30.0) < 1e-12
    realised = np.zeros((4, 2))
    for v in range(4):
        for b in range(2):
            realised[v, b] = float(np.mean(np.abs(tracks["ideal-S"][v, b] - Y[v, b]) ** 2))
    np.testing.assert_allclose(gt["achieved_snr_db"], 10.0 * np.log10(pc / realised), rtol=1e-10)
    assert np.all(np.abs(gt["achieved_snr_db"] - gt["expected_snr_db"]) < 0.3)
    rng = np.random.default_rng(np.random.SeedSequence(77))
    Y_los = (rng.standard_normal(Y.shape) + 1j * rng.standard_normal(Y.shape)).astype(np.complex128)
    shift = rng.uniform(0, 2 * np.pi, size=Y.shape)
    Y_scat = (0.1 * Y_los * np.exp(1j * shift)).astype(np.complex128)
    Y_mix = (Y_los + Y_scat).astype(np.complex128)
    tracks2, gt2 = make_tracks(
        Y_mix, 30.0, p_ref, TrackSeeds(37, 0), freq_offsets=FREQ, Y_los=Y_los
    )
    ps2 = np.zeros((4, 2))
    for v in range(4):
        for b in range(2):
            total = 0.0
            for h in range(2):
                for r in range(8):
                    for c in range(8):
                        for n in range(N):
                            total += abs(Y_scat[v, b, h, r, c, n]) ** 2
            ps2[v, b] = total / (8 * 8 * N)
    np.testing.assert_allclose(
        gt2["expected_scatter_snr_db"], 10.0 * np.log10(ps2 / gt2["sigma2"]), rtol=1e-10
    )
    np.testing.assert_allclose(
        gt2["achieved_scatter_snr_db"], 10.0 * np.log10(ps2 / gt2["noise_power"]), rtol=1e-10
    )
    assert np.all(np.isnan(gt["expected_scatter_snr_db"]))
    assert np.all(np.isnan(gt["achieved_scatter_snr_db"]))
    tracks_inf, gt_inf = make_tracks(Y, math.inf, p_ref, TrackSeeds(37, 0), freq_offsets=FREQ)
    assert gt_inf["sigma2"] == 0.0
    np.testing.assert_array_equal(tracks_inf["ideal-S"], Y)


def test_make_tracks_validation() -> None:
    """Invalid track inputs raise ValueError."""
    Y, _, _ = _grid()
    with pytest.raises(ValueError):
        make_tracks(Y[:, :, :1], 30.0, 1.0, TrackSeeds(0, 0), freq_offsets=FREQ)
    with pytest.raises(ValueError):
        make_tracks(Y, 30.0, 1.0, TrackSeeds(0, 0), freq_offsets=FREQ[:-1])
    with pytest.raises(ValueError):
        make_tracks(Y, 30.0, 0.0, TrackSeeds(0, 0), freq_offsets=FREQ)
    with pytest.raises(ValueError):
        make_tracks(Y, math.nan, 1.0, TrackSeeds(0, 0), freq_offsets=FREQ)
    with pytest.raises(ValueError):
        make_tracks(Y, 30.0, 1.0, TrackSeeds(0, 0), freq_offsets=FREQ, Y_los=Y[:, :, :1])
    with pytest.raises(ValueError):
        make_tracks(
            Y, 30.0, 1.0, TrackSeeds(0, 0), freq_offsets=FREQ, Y_los=np.full_like(Y, np.inf)
        )


def test_module_constants_and_stream_tag() -> None:
    """Module constants expose the gauge modes, tracks and defaults."""
    assert GAUGE_STREAM_TAG == 2**32 - 1
    assert DEFAULT_SIGMA_T == 10e-9
    with pytest.raises(ValueError):
        TrackSeeds(True, 0)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        TrackSeeds(-1, 0)
    with pytest.raises(ValueError):
        HardwareConfig(element_gain_std_db=-0.1)
    with pytest.raises(ValueError):
        HardwareConfig(calibration_snr_db=math.nan)
    with pytest.raises(ValueError):
        HardwareConfig(calibration_snr_db=-math.inf)
    assert math.isinf(HardwareConfig(calibration_snr_db=math.inf).calibration_snr_db)
