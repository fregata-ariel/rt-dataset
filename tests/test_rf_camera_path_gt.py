import numpy as np

from plateau_rt.domain.rf_camera.imaging import split_pattern_axis
from plateau_rt.domain.rf_camera.paths import (
    apply_path_order,
    canonical_path_order,
    synthesize_cfr,
)


def test_ordering_puts_invalid_paths_last():
    tau = np.array([[30e-9, -1.0, 10e-9, -1.0]])
    power = np.array([[1.0, 0.0, 2.0, 0.0]])
    valid = np.array([[True, False, True, False]])

    order = canonical_path_order(tau, power, valid)

    assert order.dtype == np.int64
    np.testing.assert_array_equal(order, [[2, 0, 1, 3]])
    sorted_valid = apply_path_order(valid, order, path_axis=-1)
    np.testing.assert_array_equal(sorted_valid, [[True, True, False, False]])


def test_ties_within_tau_quantum_are_broken_by_power():
    tau = np.array([[10e-9, 10e-9 + 1e-13, 10e-9 + 5e-12]])
    power = np.array([[1.0, 5.0, 3.0]])
    valid = np.ones_like(tau, dtype=bool)

    order = canonical_path_order(tau, power, valid)

    # First two delays fall in the same 1 ps bin: stronger power wins.
    np.testing.assert_array_equal(order, [[1, 0, 2]])


def test_permutation_gives_same_sorted_output():
    tau = np.array([[40e-9, 10e-9, -1.0, 20e-9, 10e-9 + 2e-13]])
    power = np.array([[1.0, 2.0, 0.0, 4.0, 9.0]])
    valid = np.array([[True, True, False, True, True]])

    order = canonical_path_order(tau, power, valid)
    perm = np.array([3, 0, 4, 1, 2])
    order_perm = canonical_path_order(tau[:, perm], power[:, perm], valid[:, perm])

    np.testing.assert_array_equal(
        apply_path_order(tau, order, path_axis=-1),
        apply_path_order(tau[:, perm], order_perm, path_axis=-1),
    )
    np.testing.assert_array_equal(
        apply_path_order(power, order, path_axis=-1),
        apply_path_order(power[:, perm], order_perm, path_axis=-1),
    )


def test_synthesize_cfr_matches_hand_built_two_path_cfr():
    a = np.array([[[0.5 + 0.25j, 0.0, 2.0 - 1.0j]]])
    tau = np.array([[[100e-9, -1.0, 250e-9]]])
    offsets = np.array([-50e6, 0.0, 25e6])

    synthesized = synthesize_cfr(a, tau, offsets)

    expected = (0.5 + 0.25j) * np.exp(-1j * 2 * np.pi * offsets * 100e-9) + (2.0 - 1.0j) * np.exp(
        -1j * 2 * np.pi * offsets * 250e-9
    )
    np.testing.assert_allclose(synthesized[0, 0], expected, rtol=1e-12, atol=1e-12)


def test_invalid_paths_are_ignored():
    a = np.array([1.0 + 0.0j, 999.0 + 0.0j])
    tau = np.array([50e-9, -1.0])
    offsets = np.array([0.0, 10e6])

    synthesized = synthesize_cfr(a, tau, offsets)

    expected = 1.0 * np.exp(-1j * 2 * np.pi * offsets * 50e-9)
    np.testing.assert_allclose(synthesized, expected, rtol=1e-12, atol=1e-12)


def test_split_pattern_axis_round_trip_matches_direct_sum():
    rows, cols, num_patterns = 2, 3, 2
    rng = np.random.default_rng(1)
    fused = rng.standard_normal((num_patterns * rows * cols, 4)) + 1j * rng.standard_normal(
        (num_patterns * rows * cols, 4)
    )
    tau = np.array([80e-9, 120e-9, -1.0, 200e-9])
    offsets = np.array([-20e6, 0.0, 20e6])

    split = split_pattern_axis(fused, num_patterns=num_patterns, rows=rows, cols=cols)
    via_split = synthesize_cfr(split, tau, offsets)
    direct = synthesize_cfr(fused, tau, offsets)

    # Fused channels are pattern-major with column-first antenna numbering:
    # channel p * rows * cols + c * rows + r holds element (row=r, col=c).
    size = rows * cols
    for pattern in range(num_patterns):
        for row in range(rows):
            for col in range(cols):
                channel = pattern * size + col * rows + row
                np.testing.assert_allclose(
                    via_split[pattern, row, col], direct[channel], rtol=1e-12, atol=1e-12
                )
