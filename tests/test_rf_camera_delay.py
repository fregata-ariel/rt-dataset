import numpy as np
import pytest

from plateau_rt.domain.rf_camera.delay import (
    angular_cfr_to_delay,
    circular_delay_error_s,
    dominant_delay,
    propagating_direction_mask,
)
from plateau_rt.domain.rf_camera.imaging import frequency_offsets


def test_angle_delay_ifft_recovers_exact_delay_bin():
    num_bins = 64
    bandwidth_hz = 100e6
    delta_f = bandwidth_hz / num_bins
    frequencies = (np.arange(num_bins) - num_bins // 2) * delta_f

    expected_bin = 25
    expected_delay = expected_bin / bandwidth_hz
    cfr_1d = np.exp(-1j * 2.0 * np.pi * frequencies * expected_delay)
    cfr = np.broadcast_to(cfr_1d, (3, 4, num_bins)).copy()

    volume = angular_cfr_to_delay(cfr, frequencies)

    peak_bins = np.argmax(np.abs(volume.cir), axis=-1)
    np.testing.assert_array_equal(peak_bins, np.full((3, 4), expected_bin))
    assert volume.delay_s[expected_bin] == pytest.approx(expected_delay)
    assert volume.frequency_spacing_hz == pytest.approx(delta_f)
    assert volume.unambiguous_delay_s == pytest.approx(1.0 / delta_f)
    np.testing.assert_allclose(np.abs(volume.cir[:, :, expected_bin]), 1.0, atol=1e-6)


@pytest.mark.parametrize(
    ("bandwidth_hz", "num_bins"),
    [(100e6, 128), (400e6, 512)],
)
def test_angle_delay_recovers_exact_bin_on_float32_grid(bandwidth_hz, num_bins):
    frequencies = frequency_offsets(bandwidth_hz, num_bins)
    delta_f = bandwidth_hz / num_bins

    expected_bin = 25
    expected_delay = expected_bin / bandwidth_hz
    cfr_1d = np.exp(-1j * 2.0 * np.pi * frequencies * expected_delay)
    cfr = np.broadcast_to(cfr_1d, (3, 4, num_bins)).copy()

    volume = angular_cfr_to_delay(cfr, frequencies)

    peak_bins = np.argmax(np.abs(volume.cir), axis=-1)
    np.testing.assert_array_equal(peak_bins, np.full((3, 4), expected_bin))
    assert volume.delay_s[expected_bin] == pytest.approx(expected_delay)
    assert volume.frequency_spacing_hz == pytest.approx(delta_f)
    assert volume.unambiguous_delay_s == pytest.approx(1.0 / delta_f)
    np.testing.assert_allclose(np.abs(volume.cir[:, :, expected_bin]), 1.0, atol=1e-6)


def test_angle_delay_rejects_nonuniform_frequency_grid():
    frequencies = np.array([-2.0, -1.0, 0.0, 1.2])
    cfr = np.ones((2, 2, 4), dtype=np.complex64)

    with pytest.raises(ValueError, match="uniformly spaced"):
        angular_cfr_to_delay(cfr, frequencies)


def test_propagating_direction_mask_is_unit_disk_projection():
    ky = np.array([-1.0, 0.0, 1.0])
    kz = np.array([-1.0, 0.0, 1.0])

    mask = propagating_direction_mask(ky, kz)

    assert mask[1, 1]
    assert mask[1, 0]
    assert mask[0, 1]
    assert not mask[0, 0]
    assert not mask[2, 2]


def test_circular_delay_error_handles_wraparound():
    period = 640e-9
    assert circular_delay_error_s(630e-9, 10e-9, period) == pytest.approx(20e-9)
    assert circular_delay_error_s(250e-9, 254e-9, period) == pytest.approx(4e-9)


def test_dominant_delay_returns_strongest_bin_per_direction():
    power = np.zeros((2, 3, 4))
    power[0, 1, 2] = 5.0
    power[1, 2, 3] = 7.0
    power[..., 0] += 1.0
    delay_s = np.array([0.0, 10e-9, 20e-9, 30e-9])

    bins, delays, peak = dominant_delay(power, delay_s)

    assert bins[0, 1] == 2 and bins[1, 2] == 3 and bins[0, 0] == 0
    assert delays[0, 1] == pytest.approx(20e-9)
    np.testing.assert_array_equal(peak, power.max(axis=-1))
