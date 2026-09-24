"""Unit tests for common phase/delay gauge alignment of CFR pairs."""

import numpy as np
import pytest

from plateau_rt.domain.rf_camera.gauge import (
    align_common_phase_and_delay,
    gauge_aligned_nmse,
    nmse,
)
from plateau_rt.domain.rf_camera.imaging import frequency_offsets


def _frequency_step(frequencies: np.ndarray) -> float:
    return float(np.min(np.diff(np.sort(frequencies))))


def _multipath_cfr(
    frequencies: np.ndarray,
    element_shape: tuple[int, ...],
    num_taps: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Random per-element multipath CFR ``sum_k a_k exp(-j 2 pi f tau_k)``."""
    frequencies = np.asarray(frequencies, dtype=np.float64)
    period = 1.0 / _frequency_step(frequencies)
    cfr = np.zeros(element_shape + (frequencies.size,), dtype=np.complex128)
    for _ in range(num_taps):
        amplitude = rng.normal(size=element_shape) + 1j * rng.normal(size=element_shape)
        delay = rng.uniform(0.0, period)
        cfr += amplitude[..., None] * np.exp(-1j * 2.0 * np.pi * frequencies * delay)
    return cfr


def _apply_gauge(h_ref: np.ndarray, frequencies: np.ndarray, phi: float, tau: float) -> np.ndarray:
    frequencies = np.asarray(frequencies, dtype=np.float64)
    return np.exp(1j * (phi - 2.0 * np.pi * frequencies * tau)) * h_ref


def _wrapped_phase_error(estimated: float, truth: float) -> float:
    return float(np.angle(np.exp(1j * (estimated - truth))))


def _wrapped_delay_error(estimated: float, truth: float, period: float) -> float:
    difference = (estimated - truth + 0.5 * period) % period - 0.5 * period
    return float(difference)


def _reference_fixture(
    num_bins: int = 32,
) -> tuple[np.ndarray, np.ndarray, float]:
    frequencies = frequency_offsets(100e6, num_bins)
    rng = np.random.default_rng(20240925)
    h_ref = _multipath_cfr(frequencies, (2, 4, 4), num_taps=4, rng=rng)
    delta_f = _frequency_step(frequencies)
    return h_ref, frequencies, 1.0 / delta_f


def test_identity_alignment_is_zero_gauge():
    h_ref, frequencies, _ = _reference_fixture()

    result = align_common_phase_and_delay(h_ref, h_ref, frequencies)

    assert abs(result.phase_rad) < 1e-10
    assert abs(result.delay_s) < 1e-12
    assert gauge_aligned_nmse(h_ref, h_ref, frequencies) < 1e-20


@pytest.mark.parametrize("num_bins", [32, 33])
def test_exact_recovery_of_arbitrary_phase_and_offgrid_delay(num_bins):
    h_ref, frequencies, period = _reference_fixture(num_bins)
    delay_bin = 1.0 / (frequencies.size * _frequency_step(frequencies))
    phi_true = 2.3
    delays = [5.37 * delay_bin, -3.37 * delay_bin]

    for tau_true in delays:
        h_obs = _apply_gauge(h_ref, frequencies, phi_true, tau_true)
        result = align_common_phase_and_delay(h_obs, h_ref, frequencies)

        assert abs(_wrapped_phase_error(result.phase_rad, phi_true)) < 1e-9
        assert abs(_wrapped_delay_error(result.delay_s, tau_true, period)) < 1e-13
        assert gauge_aligned_nmse(h_obs, h_ref, frequencies) < 1e-12


@pytest.mark.parametrize("phi_true", [3.1, -3.1, 3.1 + 2.0 * np.pi, -3.1 - 4.0 * np.pi])
def test_phase_wraparound(phi_true):
    h_ref, frequencies, _ = _reference_fixture()
    delay_bin = 1.0 / (frequencies.size * _frequency_step(frequencies))
    h_obs = _apply_gauge(h_ref, frequencies, phi_true, 0.37 * delay_bin)

    result = align_common_phase_and_delay(h_obs, h_ref, frequencies)

    assert abs(_wrapped_phase_error(result.phase_rad, phi_true)) < 1e-9
    assert -np.pi < result.phase_rad <= np.pi


def test_delay_aliasing_wraps_with_non_integer_grid_origin():
    df = 100e6 / 32
    frequencies = frequency_offsets(100e6, 32).astype(np.float64) + 0.3 * df
    df = float(frequencies[1] - frequencies[0])
    period = 1.0 / df
    phi_true = 1.1
    tau_true = 2e-9

    rng = np.random.default_rng(20240925)
    h_ref = _multipath_cfr(frequencies, (2, 4, 4), num_taps=4, rng=rng)
    h_obs = _apply_gauge(h_ref, frequencies, phi_true, tau_true + period)

    result = align_common_phase_and_delay(h_obs, h_ref, frequencies)

    assert abs(_wrapped_delay_error(result.delay_s, tau_true, period)) < 1e-13
    expected_phase = float(
        np.angle(np.exp(1j * (phi_true - 2.0 * np.pi * frequencies[0] * period)))
    )
    assert abs(_wrapped_phase_error(result.phase_rad, expected_phase)) < 1e-9


def test_noisy_recovery_and_nmse_scale():
    h_ref, frequencies, _ = _reference_fixture()
    delay_bin = 1.0 / (frequencies.size * _frequency_step(frequencies))
    phi_true = 1.7
    tau_true = 4.37 * delay_bin

    clean = _apply_gauge(h_ref, frequencies, phi_true, tau_true)
    signal_power = float(np.mean(np.abs(clean) ** 2))
    noise_power = signal_power / 10.0 ** (20.0 / 10.0)
    rng = np.random.default_rng(7)
    noise = np.sqrt(noise_power / 2.0) * (
        rng.normal(size=clean.shape) + 1j * rng.normal(size=clean.shape)
    )
    h_obs = clean + noise

    result = align_common_phase_and_delay(h_obs, h_ref, frequencies)

    period = 1.0 / _frequency_step(frequencies)
    assert abs(_wrapped_delay_error(result.delay_s, tau_true, period)) < 0.05 * delay_bin
    assert abs(_wrapped_phase_error(result.phase_rad, phi_true)) < 0.05
    assert gauge_aligned_nmse(h_obs, h_ref, frequencies) < 0.03
    assert nmse(h_obs, h_ref) > 0.5


def test_alignment_is_locally_optimal():
    frequencies = frequency_offsets(100e6, 32)
    delay_bin = 1.0 / (frequencies.size * _frequency_step(frequencies))
    rng = np.random.default_rng(5)
    h_ref = _multipath_cfr(frequencies, (3, 3), num_taps=3, rng=rng)
    phi_true = 2.0
    tau_true = 7.0 * delay_bin
    clean = _apply_gauge(h_ref, frequencies, phi_true, tau_true)
    h_obs = clean + 0.3 * (rng.normal(size=clean.shape) + 1j * rng.normal(size=clean.shape))

    aligned = align_common_phase_and_delay(h_obs, h_ref, frequencies)
    aligned_nmse = nmse(h_obs, aligned.aligned_ref)

    assert aligned_nmse * 10.0 < nmse(h_obs, h_ref)

    for phi_shift in (-1e-3, 1e-3):
        for tau_shift in (-1e-3 * delay_bin, 1e-3 * delay_bin):
            perturbed = _apply_gauge(
                h_ref, frequencies, aligned.phase_rad + phi_shift, aligned.delay_s + tau_shift
            )
            assert aligned_nmse <= nmse(h_obs, perturbed) + 1e-12


def test_unsorted_frequencies_recover_gauge_and_aligned_ref():
    h_ref, frequencies, period = _reference_fixture()
    phi_true = 0.7
    tau_true = 3.3e-9
    h_obs = _apply_gauge(h_ref, frequencies, phi_true, tau_true)

    permutation = np.random.default_rng(1234).permutation(frequencies.size)
    h_obs_permuted = h_obs[..., permutation]
    h_ref_permuted = h_ref[..., permutation]
    frequencies_permuted = frequencies[permutation]

    sorted_result = align_common_phase_and_delay(h_obs, h_ref, frequencies)
    result = align_common_phase_and_delay(h_obs_permuted, h_ref_permuted, frequencies_permuted)

    assert abs(_wrapped_phase_error(result.phase_rad, phi_true)) < 1e-9
    assert abs(_wrapped_delay_error(result.delay_s, tau_true, period)) < 1e-13
    assert gauge_aligned_nmse(h_obs_permuted, h_ref_permuted, frequencies_permuted) < 1e-12
    assert np.allclose(result.aligned_ref, h_obs_permuted)
    assert np.allclose(result.aligned_ref, sorted_result.aligned_ref[..., permutation])


def test_zero_cross_spectrum_returns_caller_order_reference():
    h_ref, frequencies, _ = _reference_fixture()
    permutation = np.random.default_rng(99).permutation(frequencies.size)
    h_ref_permuted = h_ref[..., permutation]
    frequencies_permuted = frequencies[permutation]

    zeros = np.zeros_like(h_ref_permuted)
    result = align_common_phase_and_delay(zeros, h_ref_permuted, frequencies_permuted)

    assert result.phase_rad == 0.0
    assert result.delay_s == 0.0
    assert np.array_equal(result.aligned_ref, h_ref_permuted.astype(np.complex128))

    zero_ref = align_common_phase_and_delay(h_ref_permuted, zeros, frequencies_permuted)

    assert zero_ref.aligned_ref.shape == zeros.shape
    assert np.array_equal(zero_ref.aligned_ref, zeros.astype(np.complex128))


@pytest.mark.parametrize("num_bins", [63, 128, 256, 1024])
def test_float32_uniform_grids_are_accepted(num_bins):
    frequencies = frequency_offsets(100e6, num_bins)
    assert frequencies.dtype == np.float32
    delta_f = float(
        (np.float64(frequencies[-1]) - np.float64(frequencies[0])) / (frequencies.size - 1)
    )
    delay_bin = 1.0 / (num_bins * delta_f)
    rng = np.random.default_rng(11)
    h_ref = _multipath_cfr(frequencies, (2, 2), num_taps=3, rng=rng)
    h_obs = _apply_gauge(h_ref, frequencies, -1.3, 2.37 * delay_bin)

    assert gauge_aligned_nmse(h_obs, h_ref, frequencies) < 1e-10


def test_non_finite_cfr_raises():
    _, frequencies, _ = _reference_fixture()
    h_obs = np.ones((2, 2, frequencies.size), dtype=np.complex128)
    h_ref = np.ones((2, 2, frequencies.size), dtype=np.complex128)

    h_ref_nan = h_ref.copy()
    h_ref_nan[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        align_common_phase_and_delay(h_obs, h_ref_nan, frequencies)

    h_obs_inf = h_obs.copy()
    h_obs_inf[1, 1, 3] = np.inf
    with pytest.raises(ValueError, match="finite"):
        align_common_phase_and_delay(h_obs_inf, h_ref, frequencies)


def test_non_positive_oversample_raises():
    h_ref, frequencies, _ = _reference_fixture()

    with pytest.raises(ValueError, match="oversample"):
        align_common_phase_and_delay(h_ref, h_ref, frequencies, oversample=0)


def test_duplicate_frequencies_raise():
    frequencies = np.array([-2.0, -1.0, -1.0, 1.0])
    h_ref = np.ones((2, 2, 4), dtype=np.complex128)

    with pytest.raises(ValueError, match="distinct"):
        align_common_phase_and_delay(h_ref, h_ref, frequencies)


def test_shape_mismatch_raises():
    _, frequencies, _ = _reference_fixture()
    h_obs = np.ones((2, 3, frequencies.size), dtype=np.complex128)
    h_ref = np.ones((2, 4, frequencies.size), dtype=np.complex128)

    with pytest.raises(ValueError, match="same shape"):
        align_common_phase_and_delay(h_obs, h_ref, frequencies)


def test_nonuniform_frequencies_raise():
    frequencies = np.array([-2.0, -1.0, 0.0, 1.2])
    h_ref = np.ones((2, 2, 4), dtype=np.complex128)

    with pytest.raises(ValueError, match="uniformly spaced"):
        align_common_phase_and_delay(h_ref, h_ref, frequencies)


def test_nmse_returns_infinity_for_zero_observed():
    h_obs = np.zeros((2, 2, 4), dtype=np.complex128)
    h_ref = np.ones((2, 2, 4), dtype=np.complex128)

    assert nmse(h_obs, h_ref) == float("inf")
