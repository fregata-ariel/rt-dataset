"""Unit tests for the tomography pose bank, splits and noise reference."""

from __future__ import annotations

import numpy as np
import pytest

from plateau_rt.domain.rf_camera.calibration import rotation_matrix
from plateau_rt.domain.rf_camera.camera import generate_ring_views, look_at_orientation
from plateau_rt.domain.rf_tomography.bank import (
    DEFAULT_VIEW_SUBSET_SIZES,
    JITTER_STREAM_TAG,
    MECHANISM_VARIANTS,
    SPLIT_STREAM_TAG,
    bs_split,
    bs_subsets,
    jitter_look_at,
    nested_training_orders,
    noise_reference,
    ring_poses,
    split_bank,
    view_subset_sizes,
)
from plateau_rt.domain.rf_tomography.views import nested_view_order

TARGET = (0.0, 0.0, 8.0)


def test_ring_poses_order_positions_and_angles() -> None:
    poses = ring_poses(TARGET, ((20.0, 1.5), (30.0, 10.0)), 8, 22.5)
    assert len(poses) == 16
    for ring, (radius, height) in enumerate(((20.0, 1.5), (30.0, 10.0))):
        expected = generate_ring_views(
            target=TARGET,
            radius_m=radius,
            ue_height_m=height,
            num_views=8,
            start_azimuth_deg=22.5,
        )
        for k in range(8):
            pose = poses[ring * 8 + k]
            assert pose.position == expected[k].position
            horizontal = float(np.hypot(pose.position[0] - TARGET[0], pose.position[1] - TARGET[1]))
            assert horizontal == pytest.approx(radius, abs=1e-12)
            assert pose.position[2] == height
            assert pose.azimuth_deg == pytest.approx(22.5 + 45.0 * k)
            assert pose.radius_m == radius
            assert pose.height_m == height
            assert pose.ring_index == k


def test_ring_poses_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        ring_poses(TARGET, (), 8)
    with pytest.raises(ValueError):
        ring_poses(TARGET, ((20.0, 1.5),), 0)
    with pytest.raises(ValueError):
        ring_poses(TARGET, ((0.0, 1.5),), 8)


def test_jitter_look_at_zero_max() -> None:
    positions = np.array([[40.0, 0.0, 1.5], [0.0, -30.0, 10.0]])
    look_ats, orientations, jitter = jitter_look_at(positions, TARGET, max_deg=0.0, rng=None)
    assert look_ats == [TARGET, TARGET]
    assert (jitter == 0.0).all() and jitter.shape == (2, 2)
    for position, orientation in zip(positions, orientations):
        assert orientation == look_at_orientation(tuple(position), TARGET)


def test_jitter_look_at_matches_forward_direction() -> None:
    rng = np.random.default_rng(7)
    positions = np.empty((50, 3), dtype=np.float64)
    positions[:, 0] = rng.uniform(-50.0, 50.0, size=50)
    positions[:, 1] = rng.uniform(-50.0, 50.0, size=50)
    positions[:, 2] = rng.uniform(1.0, 20.0, size=50)
    positions[np.hypot(positions[:, 0], positions[:, 1]) < 10.0, 0] += 20.0
    look_ats, orientations, jitter = jitter_look_at(
        positions, TARGET, max_deg=15.0, rng=np.random.default_rng(7)
    )
    assert bool(np.all(np.abs(jitter) <= 15.0))
    assert bool(np.any(np.abs(jitter) > 10.0))
    target = np.asarray(TARGET, dtype=np.float64)
    for i in range(50):
        delta = target - positions[i]
        dist = float(np.linalg.norm(delta))
        az0 = float(np.arctan2(delta[1], delta[0]))
        el0 = float(np.arctan2(delta[2], np.hypot(delta[0], delta[1])))
        forward = rotation_matrix(orientations[i])[:, 0]
        az_err = np.degrees(np.arctan2(forward[1], forward[0])) - np.degrees(az0) - jitter[i, 0]
        az_err = (az_err + 180.0) % 360.0 - 180.0
        assert az_err == pytest.approx(0.0, abs=1e-9)
        assert np.degrees(np.arcsin(forward[2])) - np.degrees(el0) == pytest.approx(
            jitter[i, 1], abs=1e-9
        )
        look_at = np.asarray(look_ats[i], dtype=np.float64)
        assert float(np.linalg.norm(look_at - positions[i])) == pytest.approx(dist, rel=1e-9)


def test_jitter_look_at_rng_contract() -> None:
    positions = np.tile(np.array([[40.0, 0.0, 1.5]]), (50, 1))
    _, _, first = jitter_look_at(positions, TARGET, max_deg=15.0, rng=np.random.default_rng(3))
    _, _, second = jitter_look_at(positions, TARGET, max_deg=15.0, rng=np.random.default_rng(3))
    assert np.array_equal(first, second)
    used = np.random.default_rng(3)
    jitter_look_at(positions, TARGET, max_deg=15.0, rng=used)
    fresh = np.random.default_rng(3)
    fresh.uniform(-15.0, 15.0, size=(50, 2))
    assert used.random() == fresh.random()
    with pytest.raises(ValueError):
        jitter_look_at(positions, TARGET, max_deg=-1.0, rng=np.random.default_rng(0))
    with pytest.raises(ValueError):
        jitter_look_at(positions, TARGET, max_deg=90.0, rng=np.random.default_rng(0))
    with pytest.raises(ValueError):
        jitter_look_at(positions, TARGET, max_deg=float("nan"), rng=np.random.default_rng(0))
    with pytest.raises(TypeError):
        jitter_look_at(positions, TARGET, max_deg=5.0, rng=None)


def test_split_bank_counts_and_stream() -> None:
    train, held = split_bank(64, holdout_fraction=0.25, min_holdout=4, seed=0)
    assert (len(held), len(train)) == (16, 48)
    assert split_bank(8, holdout_fraction=0.25, min_holdout=2, seed=0)[1].size == 2
    assert split_bank(8, holdout_fraction=0.25, min_holdout=4, seed=0)[1].size == 4
    assert split_bank(8, holdout_fraction=0.0, min_holdout=0, seed=0)[1].size == 0
    assert train.dtype == held.dtype == np.int64
    assert np.array_equal(np.sort(np.concatenate([train, held])), np.arange(64))
    held_sets = {
        tuple(split_bank(64, holdout_fraction=0.25, min_holdout=4, seed=s)[1]) for s in range(10)
    }
    assert len(held_sets) >= 2
    stream = np.random.default_rng(np.random.SeedSequence([0, SPLIT_STREAM_TAG]))
    assert np.array_equal(held, np.sort(stream.permutation(64)[:16]))
    with pytest.raises(ValueError):
        split_bank(4, holdout_fraction=0.25, min_holdout=4, seed=0)
    with pytest.raises(ValueError):
        split_bank(8, holdout_fraction=1.0, min_holdout=0, seed=0)


def test_bs_split_and_subsets() -> None:
    train, held = bs_split(5, 1)
    assert train.tolist() == [0, 1, 2, 3] and held.tolist() == [4]
    subsets = bs_subsets(train, (1, 2, 4))
    assert {k: v.tolist() for k, v in subsets.items()} == {1: [0], 2: [0, 1], 4: [0, 1, 2, 3]}
    train2, _ = bs_split(2, 0)
    assert list(bs_subsets(train2, (1, 2, 4))) == [1, 2]
    with pytest.raises(ValueError):
        bs_split(2, 2)
    with pytest.raises(ValueError):
        bs_subsets(train, (2, 1))


def test_nested_training_orders_and_subset_sizes() -> None:
    train, _ = split_bank(48, holdout_fraction=0.25, min_holdout=4, seed=1)
    orders = nested_training_orders(train, (0, 1, 2))
    for seed, order in orders.items():
        assert np.array_equal(np.sort(order), train)
        assert np.array_equal(order, train[nested_view_order(len(train), seed)])
    assert not np.array_equal(orders[0], orders[1])
    assert view_subset_sizes(48) == (1, 2, 4, 8, 16, 32)
    assert view_subset_sizes(6) == (1, 2, 4)
    assert DEFAULT_VIEW_SUBSET_SIZES == (1, 2, 4, 8, 16, 32)
    assert JITTER_STREAM_TAG == 0x4A495454


def test_noise_reference_lower_median() -> None:
    power = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    amp = np.sqrt(power)
    y = np.zeros((4, 2, 2, 2, 2, 3), dtype=np.complex128)
    y[:, :, 0] = amp[:, :, None, None, None]
    los_visible = np.ones((4, 2), dtype=bool)
    los_visible[1, 0] = False
    los_visible[3, 1] = False
    capture_mask = np.ones((4, 2), dtype=bool)
    capture_mask[2] = False
    ref = noise_reference(y, los_visible, capture_mask, 30.0)
    assert ref.p_ref == pytest.approx(2.0, rel=1e-15)
    assert ref.c_ref == (0, 1)
    assert ref.sigma2 == pytest.approx(2.0 / 1000.0, rel=1e-15)
    assert ref.los_fallback is False
    assert ref.expected_snr_db[3, 0] == pytest.approx(10.0 * np.log10(7.0 / 0.002), rel=1e-12)
    assert ref.scatter_power is None and ref.expected_scatter_snr_db is None

    zeroed = y.copy()
    zeroed[0, 0] = 0.0
    ref_zero = noise_reference(zeroed, los_visible, capture_mask, 30.0)
    assert ref_zero.expected_snr_db[0, 0] == float("-inf")

    fallback = noise_reference(y, np.zeros((4, 2), dtype=bool), capture_mask, 30.0)
    assert fallback.los_fallback is True
    assert fallback.p_ref == pytest.approx(3.0, rel=1e-15)
    assert fallback.c_ref == (1, 0)

    with pytest.raises(ValueError):
        noise_reference(y, los_visible, np.zeros((4, 2), dtype=bool), 30.0)
    with pytest.raises(ValueError):
        noise_reference(y, los_visible, capture_mask, 30.0, Y_los_free=np.zeros((2, 2)))

    scattered = noise_reference(y, los_visible, capture_mask, 30.0, Y_los_free=0.1 * y)
    finite = np.isfinite(scattered.expected_snr_db)
    assert scattered.scatter_power is not None and scattered.expected_scatter_snr_db is not None
    diff = scattered.expected_scatter_snr_db[finite] - scattered.expected_snr_db[finite]
    assert np.allclose(diff, -20.0, atol=1e-9)


def test_mechanism_variants_are_cumulative() -> None:
    assert list(MECHANISM_VARIANTS) == ["specular", "refraction", "diffraction"]
    flags = [
        (v.specular_reflection, v.refraction, v.diffraction) for v in MECHANISM_VARIANTS.values()
    ]
    assert flags == [(True, False, False), (True, True, False), (True, True, True)]
    for variant in MECHANISM_VARIANTS.values():
        payload = variant.to_dict()
        assert list(payload) == [
            "name",
            "los",
            "specular_reflection",
            "refraction",
            "diffraction",
            "diffuse_reflection",
        ]
        assert payload["los"] is True and payload["diffuse_reflection"] is False
    assert SPLIT_STREAM_TAG == 0x53504C54
