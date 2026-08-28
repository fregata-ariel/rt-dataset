import numpy as np
import pytest

from plateau_rt.adapters.sionna.rf_camera_delay import (
    angular_cfr_to_delay,
    circular_delay_error_s,
    propagating_direction_mask,
)


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
