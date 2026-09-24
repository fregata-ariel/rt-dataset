"""Unit tests for plateau_rt.domain.rf_camera.solver_metrics (NumPy only)."""

import warnings

import numpy as np
import pytest

from plateau_rt.domain.rf_camera.solver_metrics import (
    count_paths_by_type,
    hemisphere_energy,
    per_view_relative_difference,
    relative_cfr_difference,
)


def test_relative_difference_identical_is_zero():
    rng = np.random.default_rng(0)
    ref = rng.standard_normal((2, 2, 3)) + 1j * rng.standard_normal((2, 2, 3))
    assert relative_cfr_difference(ref, ref) == pytest.approx(0.0)


def test_relative_difference_scaled_input():
    rng = np.random.default_rng(1)
    ref = rng.standard_normal((4, 5)) + 1j * rng.standard_normal((4, 5))
    for scale in (1.5, 0.25):
        got = relative_cfr_difference(scale * ref, ref)
        assert got == pytest.approx(abs(1.0 - scale))


def test_relative_difference_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        relative_cfr_difference(np.zeros((2, 2)), np.zeros((2, 3)))


def test_relative_difference_zero_reference():
    assert relative_cfr_difference(np.zeros((2, 2)), np.zeros((2, 2))) == 0.0
    # Nonzero test against an all-zero reference is infinite.
    got = relative_cfr_difference(np.ones((2, 2)), np.zeros((2, 2)))
    assert np.isinf(got)


def test_hemisphere_energy_hand_built():
    # [hemisphere, row, col, freq]: front has ones, back has twos.
    cfr = np.stack(
        [np.ones((1, 1, 2), dtype=np.complex64), 2.0 * np.ones((1, 1, 2), dtype=np.complex64)]
    )
    energy = hemisphere_energy(cfr)
    assert set(energy) == {"front", "back"}
    assert energy["front"] == pytest.approx(2.0)
    assert energy["back"] == pytest.approx(8.0)


def test_hemisphere_energy_multi_view_layout():
    single = np.stack(
        [np.ones((1, 1, 2), dtype=np.complex64), 2.0 * np.ones((1, 1, 2), dtype=np.complex64)]
    )
    multi = np.stack([single, 3.0 * single])  # [rx=2, hemi, row, col, freq]
    energy = hemisphere_energy(multi)
    assert energy["front"] == pytest.approx(2.0 + 18.0)
    assert energy["back"] == pytest.approx(8.0 + 72.0)


def test_per_view_relative_difference():
    ref = np.ones((3, 2, 2), dtype=np.complex128)
    test = np.stack([ref[0], 2.0 * ref[1], ref[2]])
    got = per_view_relative_difference(test, ref)
    assert got.shape == (3,)
    assert got[0] == pytest.approx(0.0)
    assert got[1] == pytest.approx(1.0)
    assert got[2] == pytest.approx(0.0)
    with pytest.raises(ValueError, match="shape mismatch"):
        per_view_relative_difference(np.zeros((2, 2)), np.zeros((3, 2)))


def _mixed_flags_fixture():
    # 4 path slots: LoS, specular, diffuse+refraction, diffraction(invalid).
    valid = np.array([[[True, True, True, False]]])  # [rx=1, tx=1, paths=4]
    interactions = np.zeros((2, 1, 1, 4), dtype=np.uint8)
    interactions[0, 0, 0, 1] = 1  # specular at depth 0
    interactions[0, 0, 0, 2] = 2  # diffuse at depth 0
    interactions[1, 0, 0, 2] = 4  # refraction at depth 1
    interactions[0, 0, 0, 3] = 8  # diffraction but invalid -> ignored
    return valid, interactions


def test_per_view_relative_difference_zero_reference():
    ref = np.zeros((3, 2, 2), dtype=np.complex128)
    ref[0] = 1.0
    ref[2] = 1.0
    test = ref.copy()
    test[1] = 2.0  # nonzero test against an all-zero reference view
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = per_view_relative_difference(test, ref)
    assert got.shape == (3,)
    assert got[0] == pytest.approx(0.0)
    assert np.isinf(got[1])
    assert got[2] == pytest.approx(0.0)
    # All-zero view on both sides gives exactly 0.0.
    both_zero = per_view_relative_difference(np.zeros((1, 2, 2)), np.zeros((1, 2, 2)))
    assert both_zero[0] == 0.0


def test_count_paths_synthetic_layout():
    valid, interactions = _mixed_flags_fixture()
    counts = count_paths_by_type(valid, interactions)
    assert counts == {
        "los": 1,
        "specular": 1,
        "diffuse": 1,
        "refraction": 1,
        "diffraction": 0,
        "total": 3,
        "num_links": 1,
        "mean_paths_per_link": pytest.approx(3.0),
        "max_paths_per_link": 3,
    }


def test_count_paths_non_synthetic_layout():
    # [rx=1, rx_ant=2, tx=1, tx_ant=1, paths=2]
    valid = np.zeros((1, 2, 1, 1, 2), dtype=bool)
    valid[0, 0, 0, 0, 0] = True  # LoS on antenna 0
    valid[0, 0, 0, 0, 1] = True  # specular on antenna 0
    valid[0, 1, 0, 0, 1] = True  # specular on antenna 1
    interactions = np.zeros((1, 1, 2, 1, 1, 2), dtype=np.uint8)
    interactions[0, 0, 0, 0, 0, 1] = 1
    interactions[0, 0, 1, 0, 0, 1] = 1
    counts = count_paths_by_type(valid, interactions)
    assert counts["los"] == 1
    assert counts["specular"] == 2
    assert counts["total"] == 3
    # num_links is rx * rx_ant * tx * tx_ant after pattern de-duplication.
    assert counts["num_links"] == 2
    assert counts["mean_paths_per_link"] == pytest.approx(3.0 / 2.0)
    assert counts["max_paths_per_link"] == 2


def _pattern_major_base_fixture():
    # Base per-antenna fixture [rx=1, ant=2, tx=1, tx_ant=1, paths=2].
    valid = np.zeros((1, 2, 1, 1, 2), dtype=bool)
    valid[0, 0, 0, 0, 0] = True  # LoS on antenna 0, path 0
    valid[0, 0, 0, 0, 1] = True  # specular on antenna 0, path 1
    valid[0, 1, 0, 0, 1] = True  # specular on antenna 1, path 1
    interactions = np.zeros((1, 1, 2, 1, 1, 2), dtype=np.uint8)
    interactions[0, 0, 0, 0, 0, 1] = 1
    interactions[0, 0, 1, 0, 0, 1] = 1
    return valid, interactions


def test_count_paths_pattern_deduplication():
    base_valid, base_interactions = _pattern_major_base_fixture()
    # Tile pattern-major over 2 rx patterns: fused [a0, a1, a0, a1].
    tiled_valid = np.tile(base_valid, (1, 2, 1, 1, 1))
    assert tiled_valid.shape == (1, 4, 1, 1, 2)
    tiled_interactions = np.tile(base_interactions, (1, 1, 2, 1, 1, 1))
    assert tiled_interactions.shape == (1, 1, 4, 1, 1, 2)
    base_counts = count_paths_by_type(base_valid, base_interactions, num_rx_patterns=1)
    tiled_counts = count_paths_by_type(tiled_valid, tiled_interactions, num_rx_patterns=2)
    for key in (
        "los",
        "specular",
        "diffuse",
        "refraction",
        "diffraction",
        "total",
        "num_links",
        "max_paths_per_link",
    ):
        assert tiled_counts[key] == base_counts[key], key
    assert tiled_counts["mean_paths_per_link"] == pytest.approx(base_counts["mean_paths_per_link"])
    assert tiled_counts["num_links"] == 2
    assert tiled_counts["mean_paths_per_link"] == pytest.approx(3.0 / 2.0)
    assert tiled_counts["max_paths_per_link"] == 2


def test_count_paths_pattern_axis_not_divisible_raises():
    valid = np.zeros((1, 3, 1, 1, 2), dtype=bool)
    interactions = np.zeros((1, 1, 3, 1, 1, 2), dtype=np.uint8)
    with pytest.raises(ValueError, match="not divisible"):
        count_paths_by_type(valid, interactions, num_rx_patterns=2)


def test_count_paths_shape_mismatch_raises():
    with pytest.raises(ValueError, match="must equal valid.shape"):
        count_paths_by_type(np.zeros((1, 1, 2), dtype=bool), np.zeros((2, 1, 1, 3)))
