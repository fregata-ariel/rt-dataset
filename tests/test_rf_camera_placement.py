"""Tests for coverage-map UE placement (CPU only, no Sionna)."""

from __future__ import annotations

import json
import math
import warnings

import numpy as np
import pytest

from plateau_rt.domain.rf_camera.calibration import rotation_matrix
from plateau_rt.domain.rf_camera.camera import generate_ring_views
from plateau_rt.domain.rf_camera.placement import (
    AGGREGATIONS,
    ORIENTATION_POLICIES,
    THRESHOLD_MODES,
    CandidateCells,
    CoveragePlacementSettings,
    CoverageThreshold,
    RadioMapGrid,
    box_footprint,
    candidate_cells,
    dilate_mask,
    footprint_mask,
    orient_views,
    path_gain_db,
    plan_coverage_placement,
    sample_placements,
    views_from_placement_record,
)

LAMBDA_M = 0.1
BS_POSITIONS = [(-12.0, 0.0, 10.0), (12.0, 0.0, 10.0)]
UE_HEIGHT = 1.5


def make_grid(size: float = 40.0, cell: float = 1.0, height: float = UE_HEIGHT) -> RadioMapGrid:
    """Return a square test grid centred on the origin."""
    return RadioMapGrid(
        center_m=(0.0, 0.0, height),
        size_m=(size, size),
        cell_size_m=(cell, cell),
    )


def free_space_gain(
    grid: RadioMapGrid,
    bs_positions: list[tuple[float, float, float]] = BS_POSITIONS,
    wavelength: float = LAMBDA_M,
) -> np.ndarray:
    """Return linear free-space-like gain ``[B, ny, nx]`` for a grid."""
    centers = grid.cell_centers()
    layers = []
    for bs in bs_positions:
        dist = np.linalg.norm(centers - np.asarray(bs, dtype=np.float64), axis=-1)
        layers.append((wavelength / (4.0 * np.pi * dist)) ** 2)
    return np.stack(layers, axis=0)


def make_settings(**overrides) -> CoveragePlacementSettings:
    """Return default end-to-end settings with optional overrides."""
    params: dict = {
        "num_views": 6,
        "placement_seed": 0,
        "threshold": CoverageThreshold(mode="relative_to_max_db", value=30.0),
        "aggregation": "max",
        "orientation_policy": "face_bs",
    }
    params.update(overrides)
    return CoveragePlacementSettings(**params)


# --- RadioMapGrid ------------------------------------------------------------


def test_grid_shape_uses_ceil() -> None:
    grid = RadioMapGrid(center_m=(0.0, 0.0, 0.0), size_m=(10.0, 10.0), cell_size_m=(3.0, 3.0))
    assert grid.shape == (4, 4)
    assert all(isinstance(v, int) for v in grid.shape)


def test_grid_cell_centers_match_formulas() -> None:
    grid = make_grid()
    centers = grid.cell_centers()
    assert centers.shape == (40, 40, 3)
    assert centers.dtype == np.float64
    np.testing.assert_allclose(centers[0, 0], [-19.5, -19.5, UE_HEIGHT])
    np.testing.assert_allclose(centers[-1, -1], [19.5, 19.5, UE_HEIGHT])
    np.testing.assert_allclose(centers[0, -1], [19.5, -19.5, UE_HEIGHT])


def test_grid_cell_centers_non_integer_ceil_case() -> None:
    grid = RadioMapGrid(center_m=(1.0, 2.0, 3.0), size_m=(10.0, 10.0), cell_size_m=(3.0, 3.0))
    centers = grid.cell_centers()
    assert centers.shape == (4, 4, 3)
    # x = 1 + (ix + 0.5) * 3 - 5 ; y = 2 + (iy + 0.5) * 3 - 5
    np.testing.assert_allclose(centers[0, 0], [-2.5, -1.5, 3.0])
    np.testing.assert_allclose(centers[3, 3], [6.5, 7.5, 3.0])


def test_grid_dict_round_trip() -> None:
    grid = RadioMapGrid(center_m=(1.0, 2.0, 3.0), size_m=(10.0, 7.0), cell_size_m=(3.0, 2.0))
    record = grid.to_dict()
    assert record["orientation_rad"] == [0.0, 0.0, 0.0]
    assert record["shape"] == [4, 4]
    assert record["axis_order"] == ["y", "x"]
    assert RadioMapGrid.from_dict(json.loads(json.dumps(record))) == grid


def test_grid_from_dict_rejects_tilted_orientation() -> None:
    grid = make_grid()
    record = grid.to_dict()
    record["orientation_rad"] = [0.0, 0.0, 0.5]
    with pytest.raises(ValueError):
        RadioMapGrid.from_dict(record)


def test_grid_invalid_values_raise() -> None:
    with pytest.raises(ValueError):
        RadioMapGrid(center_m=(0.0, 0.0, 0.0), size_m=(0.0, 5.0), cell_size_m=(1.0, 1.0)).validate()
    with pytest.raises(ValueError):
        RadioMapGrid(
            center_m=(0.0, 0.0, 0.0), size_m=(5.0, 5.0), cell_size_m=(1.0, -1.0)
        ).validate()
    with pytest.raises(ValueError):
        RadioMapGrid(
            center_m=(0.0, math.nan, 0.0),
            size_m=(5.0, 5.0),
            cell_size_m=(1.0, 1.0),
        ).validate()
    with pytest.raises(ValueError):
        RadioMapGrid(
            center_m=(0.0, 0.0, 0.0),
            size_m=(5.0, math.inf),
            cell_size_m=(1.0, 1.0),
        ).validate()


def test_threshold_validation() -> None:
    CoverageThreshold(mode="absolute_db", value=-120.0).validate()
    with pytest.raises(ValueError):
        CoverageThreshold(mode="nope", value=1.0).validate()
    with pytest.raises(ValueError):
        CoverageThreshold(mode="relative_to_max_db", value=-1.0).validate()
    with pytest.raises(ValueError):
        CoverageThreshold(mode="percentile", value=101.0).validate()
    with pytest.raises(ValueError):
        CoverageThreshold(mode="percentile", value=math.nan).validate()


# --- path_gain_db ------------------------------------------------------------


def test_path_gain_db_promotes_2d_and_handles_nonpositive() -> None:
    gain = np.array([[10.0, 0.0], [-2.0, 0.1]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gain_db = path_gain_db(gain)
    assert gain_db.shape == (1, 2, 2)
    assert gain_db.dtype == np.float64
    assert gain_db[0, 0, 0] == pytest.approx(10.0)
    assert gain_db[0, 1, 1] == pytest.approx(-10.0)
    assert gain_db[0, 0, 1] == -math.inf
    assert gain_db[0, 1, 0] == -math.inf


def test_path_gain_db_maps_nan_and_inf_to_minus_inf() -> None:
    gain = np.array([[[1.0, math.nan], [math.inf, 0.0]]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        gain_db = path_gain_db(gain)
    assert gain_db[0, 0, 0] == pytest.approx(0.0)
    assert bool(np.all(gain_db[0, :, :][:, 1:] == -math.inf))
    assert gain_db[0, 0, 1] == -math.inf


def test_path_gain_db_rejects_other_ndim() -> None:
    with pytest.raises(ValueError):
        path_gain_db(np.ones((4,)))
    with pytest.raises(ValueError):
        path_gain_db(np.ones((2, 2, 2, 2)))


# --- footprint_mask / dilate_mask --------------------------------------------


def test_footprint_mask_box() -> None:
    grid = make_grid()
    mask = footprint_mask(grid, [box_footprint(-5.0, -5.0, 5.0, 5.0)])
    assert mask.shape == grid.shape
    assert mask.dtype == bool
    centers = grid.cell_centers()
    inside = (np.abs(centers[:, :, 0]) <= 4.5) & (np.abs(centers[:, :, 1]) <= 4.5)
    np.testing.assert_array_equal(mask, inside)
    assert int(np.sum(mask)) == 100


def test_footprint_mask_clearance() -> None:
    grid = make_grid()
    box = box_footprint(-5.0, -5.0, 5.0, 5.0)
    tight = footprint_mask(grid, [box])
    wide = footprint_mask(grid, [box], clearance_m=1.0)
    centers = grid.cell_centers()
    # Cell centre (5.5, 0.5) is outside the box but 0.5 m from its edge.
    pick = (np.abs(centers[:, :, 0] - 5.5) < 1e-12) & (np.abs(centers[:, :, 1] - 0.5) < 1e-12)
    assert int(np.sum(pick)) == 1
    assert not bool(tight[pick][0])
    assert bool(wide[pick][0])
    assert bool(np.all(wide[tight]))
    with pytest.raises(ValueError):
        footprint_mask(grid, [box], clearance_m=-1.0)
    empty = footprint_mask(grid, [])
    assert empty.shape == grid.shape
    assert not bool(np.any(empty))


def test_footprint_mask_l_shaped_polygon() -> None:
    grid = RadioMapGrid(center_m=(2.0, 2.0, 1.5), size_m=(6.0, 6.0), cell_size_m=(1.0, 1.0))
    ell = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [2.0, 2.0], [2.0, 4.0], [0.0, 4.0]])
    mask = footprint_mask(grid, [ell])
    centers = grid.cell_centers()

    def at(x: float, y: float) -> bool:
        pick = (np.abs(centers[:, :, 0] - x) < 1e-12) & (np.abs(centers[:, :, 1] - y) < 1e-12)
        assert int(np.sum(pick)) == 1
        return bool(mask[pick][0])

    assert at(0.5, 0.5)
    assert at(3.5, 0.5)
    assert at(0.5, 3.5)
    assert not at(3.5, 3.5)


def test_dilate_mask_identity_and_reference() -> None:
    grid = RadioMapGrid(center_m=(0.0, 0.0, 1.5), size_m=(7.0, 5.0), cell_size_m=(1.0, 1.0))
    rng = np.random.default_rng(3)
    mask = rng.uniform(size=grid.shape) < 0.3
    identity = dilate_mask(mask, grid, 0.0)
    np.testing.assert_array_equal(identity, mask)
    identity[0, 0] = not identity[0, 0]
    assert bool(mask[0, 0]) != bool(identity[0, 0])

    single = np.zeros(grid.shape, dtype=bool)
    single[2, 3] = True
    radius = 2.5
    got = dilate_mask(single, grid, radius)
    centers = grid.cell_centers()
    dist = np.linalg.norm(centers - centers[2, 3][None, None, :], axis=-1)
    np.testing.assert_array_equal(got, dist <= radius)


# --- candidate_cells ---------------------------------------------------------


def test_candidate_cells_threshold_modes() -> None:
    grid = make_grid(size=8.0)
    gain = free_space_gain(grid)
    ny, nx = grid.shape
    assert (ny, nx) == (8, 8)
    for mode, value in (
        ("absolute_db", -75.0),
        ("relative_to_max_db", 10.0),
        ("percentile", 50.0),
    ):
        threshold = CoverageThreshold(mode=mode, value=value)
        threshold.validate()
        candidates = candidate_cells(gain, grid, threshold=threshold, aggregation="max")
        assert isinstance(candidates, CandidateCells)
        assert len(candidates.threshold_db) == 1
        resolved = candidates.threshold_db[0]
        if mode == "absolute_db":
            assert resolved == pytest.approx(value)
        eligible_db = path_gain_db(gain).max(axis=0)[np.isfinite(path_gain_db(gain).max(axis=0))]
        if mode == "relative_to_max_db":
            assert resolved == pytest.approx(float(np.max(eligible_db)) - value)
        if mode == "percentile":
            assert resolved == pytest.approx(float(np.percentile(eligible_db, value)))
        assert bool(np.all(candidates.gain_db >= resolved))
        assert candidates.count > 0


def test_candidate_cells_aggregations_differ() -> None:
    grid = make_grid(size=4.0)
    ny, nx = grid.shape
    gain = np.full((2, ny, nx), 1e-6)
    gain[1, 0, 1] = 0.0  # no ray from BS1 into this cell
    threshold = CoverageThreshold(mode="absolute_db", value=-70.0)
    max_cand = candidate_cells(gain, grid, threshold=threshold, aggregation="max")
    all_cand = candidate_cells(gain, grid, threshold=threshold, aggregation="all")
    assert (0, 1) in [tuple(idx) for idx in max_cand.indices.tolist()]
    assert (0, 1) not in [tuple(idx) for idx in all_cand.indices.tolist()]
    assert len(all_cand.threshold_db) == 2
    assert len(max_cand.threshold_db) == 1
    sum_cand = candidate_cells(gain, grid, threshold=threshold, aggregation="sum")
    assert sum_cand.count == max_cand.count
    with pytest.raises(ValueError):
        candidate_cells(gain, grid, threshold=threshold, aggregation="median")


def test_candidate_cells_invalid_and_excluded_never_selected() -> None:
    grid = make_grid()
    gain = free_space_gain(grid)
    box = box_footprint(-5.0, -5.0, 5.0, 5.0)
    building = footprint_mask(grid, [box])
    zeroed = gain.copy()
    zeroed[:, building] = 0.0
    threshold = CoverageThreshold(mode="relative_to_max_db", value=30.0)
    candidates = candidate_cells(zeroed, grid, threshold=threshold, aggregation="max")
    assert candidates.counts["invalid"] == int(np.sum(building))
    assert not bool(np.any(candidates.mask[building]))

    kept = candidate_cells(
        gain, grid, threshold=threshold, aggregation="max", exclusion_mask=building
    )
    assert kept.counts["excluded"] == int(np.sum(building))
    assert not bool(np.any(kept.mask[building]))
    assert kept.count > 0


def test_candidate_cells_min_bs_distance() -> None:
    grid = make_grid(size=8.0)
    gain = free_space_gain(grid)
    centers = grid.cell_centers()
    near_bs = [
        (float(centers[0, 0, 0]), float(centers[0, 0, 1]), UE_HEIGHT),
        BS_POSITIONS[1],
    ]
    threshold = CoverageThreshold(mode="relative_to_max_db", value=30.0)
    free = candidate_cells(gain, grid, threshold=threshold, aggregation="max", bs_positions=near_bs)
    assert bool(free.mask[0, 0])
    fenced = candidate_cells(
        gain,
        grid,
        threshold=threshold,
        aggregation="max",
        bs_positions=near_bs,
        min_bs_distance_m=2.0,
    )
    assert not bool(fenced.mask[0, 0])
    assert fenced.counts["too_close_to_bs"] > 0
    with pytest.raises(ValueError):
        candidate_cells(gain, grid, threshold=threshold, min_bs_distance_m=2.0)


def test_candidate_cells_counts_precedence_and_sum() -> None:
    grid = make_grid(size=8.0)
    gain = free_space_gain(grid)
    box = box_footprint(-2.0, -2.0, 2.0, 2.0)
    building = footprint_mask(grid, [box])
    gain[:, building] = 0.0  # invalid AND excluded: must count as invalid
    threshold = CoverageThreshold(mode="absolute_db", value=-40.0)
    candidates = candidate_cells(
        gain, grid, threshold=threshold, aggregation="max", exclusion_mask=building
    )
    counts = candidates.counts
    assert list(counts.keys()) == [
        "cells",
        "invalid",
        "excluded",
        "too_close_to_bs",
        "below_threshold",
        "candidates",
    ]
    assert counts["invalid"] == int(np.sum(building))
    assert counts["excluded"] == 0
    assert (
        counts["invalid"]
        + counts["excluded"]
        + counts["too_close_to_bs"]
        + counts["below_threshold"]
        + counts["candidates"]
        == counts["cells"]
    )
    assert all(isinstance(v, int) for v in counts.values())


def test_candidate_cells_rejects_shape_mismatch_and_empty() -> None:
    grid = make_grid(size=8.0)
    threshold = CoverageThreshold(mode="absolute_db", value=-200.0)
    with pytest.raises(ValueError):
        candidate_cells(np.ones((3, 3)), grid, threshold=threshold)
    with pytest.raises(ValueError):
        candidate_cells(np.zeros((2, 8, 8)), grid, threshold=threshold)


# --- sample_placements -------------------------------------------------------


def _roomy_candidates() -> tuple[CandidateCells, RadioMapGrid]:
    grid = make_grid(size=8.0)
    gain = free_space_gain(grid)
    candidates = candidate_cells(
        gain,
        grid,
        threshold=CoverageThreshold(mode="relative_to_max_db", value=30.0),
    )
    assert candidates.count == 64
    return candidates, grid


def test_sample_placements_deterministic_and_unique() -> None:
    candidates, grid = _roomy_candidates()
    first = sample_placements(
        candidates, 8, rng=np.random.default_rng(0), grid=grid, min_spacing_m=2.0
    )
    second = sample_placements(
        candidates, 8, rng=np.random.default_rng(0), grid=grid, min_spacing_m=2.0
    )
    np.testing.assert_array_equal(first.candidate_index, second.candidate_index)
    np.testing.assert_array_equal(first.positions_m, second.positions_m)
    assert len(set(first.candidate_index.tolist())) == 8
    other = sample_placements(
        candidates, 8, rng=np.random.default_rng(1), grid=grid, min_spacing_m=2.0
    )
    assert not bool(np.array_equal(first.candidate_index, other.candidate_index))
    for i in range(8):
        for j in range(i + 1, 8):
            dist = float(
                np.hypot(
                    first.positions_m[i, 0] - first.positions_m[j, 0],
                    first.positions_m[i, 1] - first.positions_m[j, 1],
                )
            )
            assert dist >= 2.0


def test_sample_placements_spacing_failure_and_jitter() -> None:
    candidates, grid = _roomy_candidates()
    with pytest.raises(ValueError):
        sample_placements(
            candidates, 64, rng=np.random.default_rng(0), grid=grid, min_spacing_m=2.0
        )
    plain = sample_placements(
        candidates, 5, rng=np.random.default_rng(0), grid=grid, jitter_fraction=0.0
    )
    np.testing.assert_array_equal(plain.positions_m, candidates.positions_m[plain.candidate_index])
    jittered = sample_placements(
        candidates,
        5,
        rng=np.random.default_rng(0),
        grid=grid,
        jitter_fraction=0.5,
    )
    assert not bool(np.array_equal(jittered.positions_m, plain.positions_m))
    for row, chosen in enumerate(jittered.candidate_index.tolist()):
        iy, ix = (int(v) for v in candidates.indices[int(chosen)])
        center = grid.cell_centers()[iy, ix]
        offset = jittered.positions_m[row] - center
        assert abs(float(offset[0])) <= 0.25 + 1e-12
        assert abs(float(offset[1])) <= 0.25 + 1e-12
        assert float(offset[2]) == 0.0


def test_sample_placements_rejects_bad_rng_and_args() -> None:
    candidates, grid = _roomy_candidates()
    with pytest.raises(TypeError):
        sample_placements(candidates, 2, rng="nope", grid=grid)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        sample_placements(candidates, 0, rng=np.random.default_rng(0), grid=grid)
    with pytest.raises(ValueError):
        sample_placements(
            candidates,
            2,
            rng=np.random.default_rng(0),
            grid=None,
            jitter_fraction=0.5,
        )


# --- orient_views ------------------------------------------------------------


def test_orient_views_look_at_target() -> None:
    positions = np.array([[0.0, 0.0, 1.5], [10.0, 0.0, 1.5]])
    target = (0.0, 10.0, 1.5)
    views, facing = orient_views(positions, policy="look_at_target", target=target)
    assert [v.view_id for v in views] == ["ue_000000", "ue_000001"]
    assert facing == [None, None]
    for view, pos in zip(views, positions, strict=True):
        forward = rotation_matrix(view.orientation)[:, 0]
        want = np.asarray(target) - pos
        want /= np.linalg.norm(want)
        np.testing.assert_allclose(forward, want, atol=1e-12)
    with pytest.raises(ValueError):
        orient_views(positions, policy="look_at_target")


def test_orient_views_face_bs_strongest() -> None:
    positions = np.array([[0.0, 0.0, 1.5]])
    per_bs = np.array([[-70.0, -60.0]])
    views, facing = orient_views(
        positions,
        policy="face_bs",
        bs_positions=BS_POSITIONS,
        face_bs="strongest",
        per_bs_gain_db=per_bs,
    )
    assert facing == [1]
    assert list(views[0].look_at) == list(BS_POSITIONS[1])
    local = rotation_matrix(views[0].orientation).T @ (np.asarray(BS_POSITIONS[1]) - positions[0])
    local /= np.linalg.norm(local)
    assert local[0] == pytest.approx(1.0, abs=1e-12)
    fixed, facing_fixed = orient_views(
        positions, policy="face_bs", bs_positions=BS_POSITIONS, face_bs=0
    )
    assert facing_fixed == [0]
    assert list(fixed[0].look_at) == list(BS_POSITIONS[0])


def test_orient_views_random_yaw_pitch() -> None:
    positions = np.array([[1.0, 2.0, 1.5], [3.0, 4.0, 1.5]])
    flat, _ = orient_views(
        positions, policy="random_yaw", rng=np.random.default_rng(0), pitch_deg=0.0
    )
    for view, pos in zip(flat, positions, strict=True):
        forward = rotation_matrix(view.orientation)[:, 0]
        assert forward[2] == pytest.approx(0.0, abs=1e-12)
        np.testing.assert_allclose(np.linalg.norm(forward), 1.0, atol=1e-12)
        assert view.look_at[2] == pytest.approx(pos[2])
    tilted, _ = orient_views(
        positions, policy="random_yaw", rng=np.random.default_rng(0), pitch_deg=20.0
    )
    for view in tilted:
        forward = rotation_matrix(view.orientation)[:, 0]
        assert forward[2] == pytest.approx(math.sin(math.radians(20.0)), abs=1e-12)
    with pytest.raises(ValueError):
        orient_views(positions, policy="spin")
    with pytest.raises(ValueError):
        orient_views(positions, policy="random_yaw", rng=np.random.default_rng(0), pitch_deg=90.0)


# --- plan_coverage_placement end to end --------------------------------------


def _e2e_inputs(seed: int = 0, **settings_overrides):
    grid = make_grid()
    gain = free_space_gain(grid)
    box = box_footprint(-5.0, -5.0, 5.0, 5.0)
    exclusion = footprint_mask(grid, [box], clearance_m=1.0)
    settings = make_settings(
        placement_seed=seed,
        min_spacing_m=3.0,
        jitter_fraction=0.5,
        **settings_overrides,
    )
    return grid, gain, exclusion, settings


def test_plan_determinism_and_seed_sensitivity() -> None:
    grid, gain, exclusion, settings = _e2e_inputs()
    before = gain.copy()
    first = plan_coverage_placement(
        gain, grid, settings, exclusion_mask=exclusion, bs_positions=BS_POSITIONS
    )
    second = plan_coverage_placement(
        gain, grid, settings, exclusion_mask=exclusion, bs_positions=BS_POSITIONS
    )
    assert first.views == second.views
    assert json.dumps(first.to_record(), sort_keys=True) == json.dumps(
        second.to_record(), sort_keys=True
    )
    np.testing.assert_array_equal(before, gain)
    other = plan_coverage_placement(
        gain,
        grid,
        make_settings(placement_seed=1, min_spacing_m=3.0, jitter_fraction=0.5),
        exclusion_mask=exclusion,
        bs_positions=BS_POSITIONS,
    )
    assert [v.position for v in other.views] != [v.position for v in first.views]


def test_plan_validity_for_several_seeds() -> None:
    for seed in (0, 1, 2):
        grid, gain, exclusion, settings = _e2e_inputs(seed)
        placement = plan_coverage_placement(
            gain, grid, settings, exclusion_mask=exclusion, bs_positions=BS_POSITIONS
        )
        resolved = placement.candidates.threshold_db[0]
        picked = placement.sampled.positions_m
        for row, chosen in enumerate(placement.sampled.candidate_index.tolist()):
            iy, ix = (int(v) for v in placement.candidates.indices[int(chosen)])
            assert bool(placement.candidates.mask[iy, ix])
            assert float(placement.candidates.gain_db[int(chosen)]) >= resolved
            assert not bool(exclusion[iy, ix])
            center = grid.cell_centers()[iy, ix]
            assert abs(float(picked[row, 0]) - float(center[0])) <= 0.25 + 1e-12
            assert abs(float(picked[row, 1]) - float(center[1])) <= 0.25 + 1e-12
        for i in range(len(picked)):
            for j in range(i + 1, len(picked)):
                dist = float(np.hypot(picked[i, 0] - picked[j, 0], picked[i, 1] - picked[j, 1]))
                assert dist >= 3.0


def test_plan_orientation_change_keeps_positions() -> None:
    grid, gain, exclusion, settings = _e2e_inputs()
    base = plan_coverage_placement(
        gain, grid, settings, exclusion_mask=exclusion, bs_positions=BS_POSITIONS
    )
    turned = plan_coverage_placement(
        gain,
        grid,
        make_settings(
            placement_seed=0,
            min_spacing_m=3.0,
            jitter_fraction=0.5,
            orientation_policy="random_yaw",
            pitch_deg=10.0,
        ),
        exclusion_mask=exclusion,
        bs_positions=BS_POSITIONS,
    )
    assert [v.position for v in turned.views] == [v.position for v in base.views]
    assert [v.orientation for v in turned.views] != [v.orientation for v in base.views]


def test_plan_leaves_global_rng_untouched() -> None:
    np.random.seed(12345)
    before = np.random.get_state()
    grid, gain, exclusion, _ = _e2e_inputs()
    plan_coverage_placement(
        gain,
        grid,
        make_settings(
            placement_seed=7,
            min_spacing_m=2.0,
            jitter_fraction=0.5,
            orientation_policy="random_yaw",
        ),
        exclusion_mask=exclusion,
        bs_positions=BS_POSITIONS,
    )
    after = np.random.get_state()
    assert before[0] == after[0]
    assert np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_plan_record_round_trip_with_minus_inf_gain() -> None:
    grid = make_grid(size=8.0)
    gain = free_space_gain(grid)
    gain[1, 0, 0] = 0.0  # one BS blind here, still a "max" candidate
    settings = make_settings(
        placement_seed=0,
        num_views=8,
        threshold=CoverageThreshold(mode="absolute_db", value=-90.0),
    )
    placement = plan_coverage_placement(gain, grid, settings, bs_positions=BS_POSITIONS)
    record = placement.to_record()
    assert record["method"] == "coverage"
    assert record["rng"]["streams"] == ["positions", "orientation"]
    assert record["candidate_count"] == placement.candidates.count
    text = json.dumps(record, allow_nan=False)
    rebuilt = views_from_placement_record(json.loads(text))
    assert rebuilt == placement.views
    with pytest.raises(ValueError):
        views_from_placement_record({"method": "ring", "views": []})
    with pytest.raises(ValueError):
        views_from_placement_record({"method": "coverage", "views": [{"view_id": "ue_000000"}]})


def test_settings_dict_round_trip_and_validation() -> None:
    settings = make_settings(orientation_policy="look_at_target", target=(0.0, 0.0, 5.0), face_bs=1)
    assert (
        CoveragePlacementSettings.from_dict(json.loads(json.dumps(settings.to_dict()))) == settings
    )
    with pytest.raises(ValueError):
        make_settings(num_views=0).validate()
    with pytest.raises(ValueError):
        make_settings(placement_seed=True).validate()  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        make_settings(orientation_policy="look_at_target", target=None).validate()
    with pytest.raises(ValueError):
        make_settings(aggregation="median").validate()


def test_placement_module_avoids_global_rng() -> None:
    from pathlib import Path

    source = (
        Path(__file__)
        .resolve()
        .parent.parent.joinpath("src/plateau_rt/domain/rf_camera/placement.py")
        .read_text()
    )
    for forbidden in (
        "np.random.seed",
        "np.random.rand",
        "np.random.choice",
        "np.random.uniform",
        "np.random.permutation",
        "np.random.shuffle",
        "import random",
        "from random",
    ):
        assert forbidden not in source


# --- ring path unchanged -----------------------------------------------------


def test_ring_views_regression_pin() -> None:
    views = generate_ring_views(target=(5.0, 5.0, 5.0), radius_m=30.0, ue_height_m=1.5, num_views=8)
    assert [v.view_id for v in views] == [f"ue_{i:06d}" for i in range(8)]
    assert views[0].position == (35.0, 5.0, 1.5)


def test_public_mode_tuples() -> None:
    assert THRESHOLD_MODES == ("absolute_db", "relative_to_max_db", "percentile")
    assert AGGREGATIONS == ("max", "sum", "all")
    assert ORIENTATION_POLICIES == ("look_at_target", "face_bs", "random_yaw")
