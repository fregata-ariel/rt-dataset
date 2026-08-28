import numpy as np

from plateau_rt.adapters.sionna.rf_camera import aperture_to_angular_fft
from plateau_rt.adapters.sionna.rf_camera_calibration import (
    angular_peak_projection,
    calibrate_angular_cfr,
    geometric_los_source_direction_local,
)


def test_calibration_recovers_physical_direction_and_phase_origin():
    rows = 8
    cols = 8
    fft_rows = 128
    fft_cols = 128
    d_v = 0.5
    d_h = 0.5

    # Choose direction cosines that land exactly on FFT bins.
    expected_ky = -0.5
    expected_kz = 0.25
    expected_phase = 0.7

    row = np.arange(rows)[:, None]
    col = np.arange(cols)[None, :]
    y = (col - (cols - 1) / 2.0) * d_h
    z = ((rows - 1) / 2.0 - row) * d_v

    aperture = np.exp(
        1j * (expected_phase + 2.0 * np.pi * (expected_ky * y + expected_kz * z))
    )[:, :, None]
    raw = aperture_to_angular_fft(aperture, fft_rows=fft_rows, fft_cols=fft_cols)

    calibrated = calibrate_angular_cfr(
        raw,
        aperture_rows=rows,
        aperture_cols=cols,
        horizontal_spacing_lambda=d_h,
        vertical_spacing_lambda=d_v,
    )
    peak_ky, peak_kz, peak_index = angular_peak_projection(
        calibrated.cfr[:, :, 0],
        ky_over_k=calibrated.ky_over_k,
        kz_over_k=calibrated.kz_over_k,
    )

    assert peak_ky == expected_ky
    assert peak_kz == expected_kz
    peak_value = calibrated.cfr[peak_index[0], peak_index[1], 0]
    np.testing.assert_allclose(np.angle(peak_value), expected_phase, atol=1e-6)
    np.testing.assert_allclose(np.abs(peak_value), rows * cols, atol=1e-6)


def test_default_geometry_los_direction_matches_expected_projection():
    direction = geometric_los_source_direction_local(
        tx_position=(-50.0, -50.0, 30.0),
        ue_position=(0.0, 0.0, 1.5),
        ue_orientation=(0.0, 0.0, 0.0),
    )

    np.testing.assert_allclose(
        direction,
        [-0.65583994, -0.65583994, 0.37382877],
        atol=1e-7,
    )
