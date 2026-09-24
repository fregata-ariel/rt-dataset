import numpy as np
import pytest

from plateau_rt.domain.rf_camera.imaging import split_pattern_axis, split_tx_pattern_axes


def test_split_tx_pattern_axes_matches_per_tx_split():
    rows, cols, num_patterns, num_tx, num_freqs = 3, 2, 2, 2, 5
    flat = num_patterns * rows * cols
    # Known values: channel c of tx t at freq f is 100 * t + c + f / 100.
    channels = np.arange(flat, dtype=np.float64)
    freqs = np.arange(num_freqs, dtype=np.float64) / 100.0
    rx_cfr = np.stack(
        [100.0 * t + channels[:, None] + freqs[None, :] for t in range(num_tx)], axis=1
    )

    out = split_tx_pattern_axes(
        rx_cfr, num_tx=num_tx, num_patterns=num_patterns, rows=rows, cols=cols
    )

    assert out.shape == (num_tx, num_patterns, rows, cols, num_freqs)
    for tx in range(num_tx):
        expected = split_pattern_axis(
            rx_cfr[:, tx, :], num_patterns=num_patterns, rows=rows, cols=cols
        )
        np.testing.assert_array_equal(out[tx], expected)


def test_split_tx_pattern_axes_single_tx_matches_split_pattern_axis():
    rows, cols, num_freqs = 2, 2, 4
    rx_cfr = np.arange(2 * rows * cols * num_freqs, dtype=np.complex128).reshape(
        2 * rows * cols, 1, num_freqs
    )

    out = split_tx_pattern_axes(rx_cfr, num_tx=1, num_patterns=2, rows=rows, cols=cols)

    assert out.shape == (1, 2, rows, cols, num_freqs)
    np.testing.assert_array_equal(
        out[0], split_pattern_axis(rx_cfr[:, 0, :], num_patterns=2, rows=rows, cols=cols)
    )


@pytest.mark.parametrize(
    "shape",
    [
        (2 * 6, 3, 5),  # wrong num_tx axis
        (2 * 6 + 1, 2, 5),  # wrong fused channel count
        (2 * 6, 2),  # missing frequency axis
    ],
)
def test_split_tx_pattern_axes_rejects_shape_mismatch(shape):
    rx_cfr = np.zeros(shape, dtype=np.complex128)

    with pytest.raises(ValueError):
        split_tx_pattern_axes(rx_cfr, num_tx=2, num_patterns=2, rows=3, cols=2)
