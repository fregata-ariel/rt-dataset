import numpy as np
import pytest

from plateau_rt.domain.rf_camera.delay import angular_cfr_to_delay
from plateau_rt.domain.rf_camera.imaging import frequency_offsets
from plateau_rt.domain.rf_camera.impairments import (
    ImpairmentConfig,
    NoiseSpec,
    apply_impairments,
    draw_element_errors,
    isotropic_mean_power,
    resolve_noise_variance,
)

BANDWIDTH_HZ = 100e6
NUM_BINS = 64


def _random_cfr(shape, seed):
    rng = np.random.default_rng(seed)
    real = rng.standard_normal(shape)
    imag = rng.standard_normal(shape)
    return (real + 1j * imag).astype(np.complex64)


def test_zero_config_reproduces_front_hemisphere_exactly():
    cfr = _random_cfr((2, 4, 5, NUM_BINS), seed=1)
    observed, gt = apply_impairments(
        cfr,
        frequency_offsets(BANDWIDTH_HZ, NUM_BINS),
        ImpairmentConfig(),
        np.random.default_rng(0),
    )

    assert observed.dtype == np.complex64
    assert np.array_equal(observed, cfr[0])
    assert gt["g"] == 0.0
    assert gt["common_phase_rad"] == 0.0
    assert gt["timing_offset_s"] == 0.0
    assert gt["expected_snr_db"] is None
    assert gt["achieved_snr_db"] is None
    assert gt["noise_variance"] == 0.0


def test_front_to_back_scaling_and_none_ignores_back():
    cfr = np.zeros((2, 2, 3, 8), dtype=np.complex64)
    cfr[1] = 1.0
    frequencies = frequency_offsets(BANDWIDTH_HZ, 8)

    observed, gt = apply_impairments(
        cfr,
        frequencies,
        ImpairmentConfig(front_to_back_db=20.0),
        np.random.default_rng(0),
    )
    assert gt["g"] == pytest.approx(0.1)
    np.testing.assert_allclose(np.abs(observed), 0.1, rtol=0.0, atol=1e-6)

    front_only, gt_none = apply_impairments(
        cfr,
        frequencies,
        ImpairmentConfig(front_to_back_db=None),
        np.random.default_rng(0),
    )
    assert gt_none["g"] == 0.0
    np.testing.assert_array_equal(front_only, np.zeros_like(front_only))


def test_timing_offset_shifts_delay_peak_by_exact_bins():
    frequencies = frequency_offsets(BANDWIDTH_HZ, NUM_BINS)
    base_bin = 5
    tau0_s = base_bin / BANDWIDTH_HZ
    cfr_1d = np.exp(-1j * 2.0 * np.pi * frequencies.astype(np.float64) * tau0_s)
    cfr = np.zeros((2, 2, 2, NUM_BINS), dtype=np.complex64)
    cfr[0] = cfr_1d

    base_volume = angular_cfr_to_delay(cfr[0], frequencies)
    base_peak = int(np.argmax(np.abs(base_volume.cir), axis=-1)[0, 0])
    assert base_peak == base_bin

    offset_bins = 3
    observed, _ = apply_impairments(
        cfr,
        frequencies,
        ImpairmentConfig(timing_offset_ns=offset_bins / BANDWIDTH_HZ * 1e9),
        np.random.default_rng(0),
    )
    volume = angular_cfr_to_delay(observed, frequencies)
    peak = int(np.argmax(np.abs(volume.cir), axis=-1)[0, 0])
    assert (peak - base_peak) % NUM_BINS == offset_bins


def test_fixed_common_phase_multiplies_by_j():
    cfr = _random_cfr((2, 2, 2, 8), seed=9)
    observed, gt = apply_impairments(
        cfr,
        frequency_offsets(BANDWIDTH_HZ, 8),
        ImpairmentConfig(common_phase_deg=90.0),
        np.random.default_rng(0),
    )

    assert gt["common_phase_rad"] == pytest.approx(np.pi / 2.0, rel=1e-6)
    np.testing.assert_allclose(observed, cfr[0] * 1j, rtol=1e-6, atol=1e-6)


def test_gt_rebuilds_the_observed_cfr_exactly():
    cfr = _random_cfr((2, 6, 7, 32), seed=21)
    frequencies = frequency_offsets(BANDWIDTH_HZ, 32)
    config = ImpairmentConfig(
        front_to_back_db=10.0,
        timing_offset_ns=7.0,
        timing_offset_std_ns=2.0,
        element_gain_std_db=0.8,
        element_phase_std_deg=9.0,
        random_common_phase=True,
    )

    observed, gt = apply_impairments(cfr, frequencies, config, np.random.default_rng(5))
    assert gt["common_phase_rad"] != 0.0

    _, other_gt = apply_impairments(cfr, frequencies, config, np.random.default_rng(6))
    assert gt["common_phase_rad"] != other_gt["common_phase_rad"]

    front = cfr[0].astype(np.complex128)
    back = cfr[1].astype(np.complex128)
    g = gt["g"]
    element = 10.0 ** (np.asarray(gt["element_gain_db"]) / 20.0) * np.exp(
        1j * np.asarray(gt["element_phase_rad"])
    )
    ramp = np.exp(-1j * 2.0 * np.pi * frequencies.astype(np.float64) * gt["timing_offset_s"])
    rebuilt = (
        (front + g * back)
        * element[:, :, None]
        * ramp[None, None, :]
        * np.exp(1j * gt["common_phase_rad"])
    )
    np.testing.assert_allclose(observed.astype(np.complex128), rebuilt, rtol=1e-5, atol=1e-6)


def test_draw_element_errors_shapes_and_complex_gain():
    config = ImpairmentConfig(element_gain_std_db=0.0, element_phase_std_deg=0.0)
    errors = draw_element_errors(config, rows=3, cols=5, rng=np.random.default_rng(0))
    assert errors.gain_db.shape == (3, 5)
    assert errors.phase_rad.shape == (3, 5)
    np.testing.assert_array_equal(errors.complex_gain, np.ones((3, 5), dtype=np.complex128))


def test_apply_impairments_uses_given_element_errors():
    cfr = _random_cfr((2, 2, 2, 8), seed=2)
    frequencies = frequency_offsets(BANDWIDTH_HZ, 8)
    config = ImpairmentConfig(element_gain_std_db=0.0, element_phase_std_deg=0.0)
    errors = draw_element_errors(
        ImpairmentConfig(element_gain_std_db=1.0), 2, 2, np.random.default_rng(7)
    )
    observed, _ = apply_impairments(
        cfr, frequencies, config, np.random.default_rng(0), element_errors=errors
    )
    np.testing.assert_allclose(observed, cfr[0] * errors.complex_gain[:, :, None], rtol=1e-6)

    with pytest.raises(ValueError):
        apply_impairments(
            cfr,
            frequencies,
            config,
            np.random.default_rng(0),
            element_errors=errors,
            noise_variance=-1.0,
        )


def test_isotropic_mean_power():
    cfr = np.zeros((2, 1, 1, 2), dtype=np.complex64)
    cfr[0] = 1.0
    cfr[1] = 1.0
    assert isotropic_mean_power(cfr) == pytest.approx(4.0)
    cfr[1] = 0.0
    assert isotropic_mean_power(cfr) == pytest.approx(1.0)


def test_resolve_noise_variance_modes():
    assert resolve_noise_variance(NoiseSpec(), 3.0) == 0.0
    assert resolve_noise_variance(NoiseSpec(noise_variance=2.5), 3.0) == pytest.approx(2.5)
    assert resolve_noise_variance(NoiseSpec(snr_db=10.0), 10.0) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        resolve_noise_variance(NoiseSpec(snr_db=0.0), 0.0)


def test_config_and_noise_validation():
    with pytest.raises(ValueError):
        ImpairmentConfig(timing_offset_std_ns=-1.0)
    with pytest.raises(ValueError):
        ImpairmentConfig(element_gain_std_db=-0.5)
    with pytest.raises(ValueError):
        ImpairmentConfig(element_phase_std_deg=float("nan"))
    with pytest.raises(ValueError):
        ImpairmentConfig(common_phase_deg=30.0, random_common_phase=True)
    with pytest.raises(ValueError):
        NoiseSpec(snr_db=10.0, noise_variance=1e-3)
    with pytest.raises(ValueError):
        NoiseSpec(noise_variance=-1.0)


def test_same_seed_is_deterministic_and_different_seed_differs():
    cfr = _random_cfr((2, 4, 4, 16), seed=5)
    frequencies = frequency_offsets(BANDWIDTH_HZ, 16)
    config = ImpairmentConfig(
        front_to_back_db=20.0,
        element_gain_std_db=0.5,
        element_phase_std_deg=5.0,
        timing_offset_ns=10.0,
        random_common_phase=True,
    )

    first, first_gt = apply_impairments(
        cfr, frequencies, config, np.random.default_rng(123), noise_variance=0.5
    )
    second, second_gt = apply_impairments(
        cfr, frequencies, config, np.random.default_rng(123), noise_variance=0.5
    )
    other, _ = apply_impairments(
        cfr, frequencies, config, np.random.default_rng(124), noise_variance=0.5
    )

    np.testing.assert_array_equal(first, second)
    assert first_gt == second_gt
    assert not np.array_equal(first, other)
    assert first_gt["expected_snr_db"] is not None
    assert first_gt["achieved_snr_db"] is not None
