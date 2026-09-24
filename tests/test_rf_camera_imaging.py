import numpy as np

from plateau_rt.domain.rf_camera.imaging import (
    aperture_to_angular_fft,
    frequency_offsets,
    reshape_planar_column_first,
    split_pattern_axis,
)


def test_frequency_offsets_are_centered_on_dc():
    offsets = frequency_offsets(100e6, 4)
    np.testing.assert_allclose(offsets, [-50e6, -25e6, 0.0, 25e6])


def test_planar_array_column_first_numbering_is_restored():
    # Sionna PlanarArray numbers all rows in column 0 first, then column 1.
    flat = np.arange(6, dtype=np.float32)[:, None]
    aperture = reshape_planar_column_first(flat, rows=3, cols=2)

    expected = np.array(
        [
            [0.0, 3.0],
            [1.0, 4.0],
            [2.0, 5.0],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(aperture[:, :, 0], expected)


def test_spatial_fft_finds_known_phase_ramp_bin():
    rows = 8
    cols = 8
    row_bin = 1
    col_bin = -2

    row = np.arange(rows)[:, None]
    col = np.arange(cols)[None, :]
    phase = 2.0 * np.pi * (row_bin * row / rows + col_bin * col / cols)
    aperture = np.exp(1j * phase)[:, :, None]

    spectrum = aperture_to_angular_fft(aperture, fft_rows=rows, fft_cols=cols)
    peak = np.unravel_index(np.abs(spectrum[:, :, 0]).argmax(), (rows, cols))

    # fftshift puts DC at rows//2, cols//2.
    assert peak == (rows // 2 + row_bin, cols // 2 + col_bin)


def test_pattern_axis_is_split_pattern_major_then_column_first():
    # Sionna fuses [pattern, antenna]: channel p * rows * cols + a.
    rows, cols = 3, 2
    flat = np.arange(2 * rows * cols, dtype=np.float32)[:, None]

    split = split_pattern_axis(flat, num_patterns=2, rows=rows, cols=cols)

    assert split.shape == (2, rows, cols, 1)
    np.testing.assert_array_equal(split[0, :, :, 0], [[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]])
    np.testing.assert_array_equal(split[1, :, :, 0], [[6.0, 9.0], [7.0, 10.0], [8.0, 11.0]])
