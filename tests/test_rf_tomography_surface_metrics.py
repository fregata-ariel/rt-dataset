"""Unit tests for surface/plane metrics and stratified recall (M2, M3, M6)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from plateau_rt.domain.rf_tomography.geometry import VoxelGrid
from plateau_rt.domain.rf_tomography.metrics import (
    MAP_REL_THRESHOLD,
    PLANE_MAX_ANGLE_DEG,
    PLANE_MAX_OFFSET_M,
    SURFACE_THRESHOLDS_M,
    ChamferScore,
    PlaneMatching,
    SurfaceScore,
    energy_within,
    map_point_cloud,
    match,
    match_planes,
    nearest_distance,
    plane_errors,
    stratified_recall,
    surface_prf,
    surface_report,
    weighted_chamfer,
)

REF = np.array([[float(x), 0.0, 0.0] for x in range(10)])


def test_constants() -> None:
    """Threshold and gate constants hold their design values."""
    assert SURFACE_THRESHOLDS_M == (0.5, 1.0, 2.0)
    assert MAP_REL_THRESHOLD == 0.1
    assert PLANE_MAX_ANGLE_DEG == 10.0
    assert PLANE_MAX_OFFSET_M == 1.0


def test_nearest_distance() -> None:
    """Nearest distances follow the hand-computed values, incl. edge cases."""
    src = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
    dst = np.array([[0.0, 0.0, 1.0], [3.0, 0.0, 0.0]])
    # d([0,0,0] -> [0,0,1]) = 1; d([3,4,0] -> [3,0,0]) = 4
    np.testing.assert_allclose(nearest_distance(src, dst), [1.0, 4.0], atol=1e-12)
    np.testing.assert_allclose(nearest_distance(src, np.zeros((0, 3))), [np.inf, np.inf], atol=0.0)
    assert nearest_distance(np.zeros((0, 3)), dst).shape == (0,)
    with pytest.raises(ValueError):
        nearest_distance(src, np.zeros((2, 2)))
    with pytest.raises(ValueError):
        nearest_distance(np.full((1, 3), np.nan), dst)


def test_surface_prf() -> None:
    """Precision/recall/F-score match the hand-computed gate table."""
    pred = np.array([[0.0, 0.0, 0.4], [5.0, 0.0, 0.9], [20.0, 0.0, 0.0]])
    # d_pred = [0.4, 0.9, 11]; scoped recall hits at t=0.5: {x=0}; at t=1: {x=0, x=5}
    got = surface_prf(pred, REF, 0.5)
    assert isinstance(got, SurfaceScore)
    assert got.precision == pytest.approx(1.0 / 3.0, abs=1e-12)
    assert got.recall == pytest.approx(1.0 / 10.0, abs=1e-12)
    assert got.f_score == pytest.approx(2.0 / 13.0, abs=1e-12)
    assert got.num_pred == 3 and got.num_ref == 10
    got = surface_prf(pred, REF, 1.0)
    assert got.precision == pytest.approx(2.0 / 3.0, abs=1e-12)
    assert got.recall == pytest.approx(1.0 / 5.0, abs=1e-12)
    assert got.f_score == pytest.approx(4.0 / 13.0, abs=1e-12)

    mask = REF[:, 0] <= 3.0  # scoped refs x = 0..3, only x=0 within t=1
    got = surface_prf(pred, REF, 1.0, recall_mask=mask)
    assert got.precision == pytest.approx(2.0 / 3.0, abs=1e-12)
    assert got.recall == pytest.approx(1.0 / 4.0, abs=1e-12)
    assert got.f_score == pytest.approx(4.0 / 11.0, abs=1e-12)
    assert got.num_ref == 4

    # pred weights [1,2,1]: hits carry 1 + 2 of 4 -> P = 3/4, F = 6/19
    got = surface_prf(pred, REF, 1.0, pred_weight=np.array([1.0, 2.0, 1.0]))
    assert got.precision == pytest.approx(3.0 / 4.0, abs=1e-12)
    assert got.recall == pytest.approx(1.0 / 5.0, abs=1e-12)
    assert got.f_score == pytest.approx(6.0 / 19.0, abs=1e-12)

    # ref weights 1..10: hits x=0 (w=1) and x=5 (w=6) of 55 -> R = 7/55
    got = surface_prf(pred, REF, 1.0, ref_weight=REF[:, 0] + 1.0)
    assert got.recall == pytest.approx(7.0 / 55.0, abs=1e-12)

    # inclusive boundary: d = 1.0 <= t = 1.0 counts
    got = surface_prf(np.array([[2.0, 0.0, 1.0]]), REF, 1.0)
    assert got.precision == pytest.approx(1.0, abs=1e-12)
    assert got.recall == pytest.approx(1.0 / 10.0, abs=1e-12)

    empty = surface_prf(np.zeros((0, 3)), REF, 1.0)
    assert math.isnan(empty.precision)
    assert empty.recall == pytest.approx(0.0, abs=1e-12)
    assert empty.f_score == pytest.approx(0.0, abs=1e-12)
    scoped_out = surface_prf(pred, REF, 1.0, recall_mask=np.zeros((10,), dtype=bool))
    assert math.isnan(scoped_out.recall) and math.isnan(scoped_out.f_score)
    with pytest.raises(ValueError):
        surface_prf(pred, REF, 1.0, pred_weight=np.array([-1.0, 0.0, 0.0]))
    with pytest.raises(ValueError):
        surface_prf(pred, REF, 0.0)
    with pytest.raises(ValueError):
        surface_prf(pred, REF, -1.0)


def test_weighted_chamfer() -> None:
    """Chamfer terms match the hand-listed nearest-distance table."""
    pred = np.array([[0.0, 0.0, 0.4], [5.0, 0.0, 0.9]])
    weights = np.array([1.0, 3.0])
    # accuracy = (0.4 * 1 + 0.9 * 3) / 4 = 0.775
    listed = [
        0.4,
        math.hypot(1, 0.4),
        math.hypot(2, 0.4),
        math.hypot(2, 0.9),
        math.hypot(1, 0.9),
        0.9,
        math.hypot(1, 0.9),
        math.hypot(2, 0.9),
        math.hypot(3, 0.9),
        math.hypot(4, 0.9),
    ]
    comp = float(np.mean(listed))
    got = weighted_chamfer(pred, REF, pred_weight=weights)
    assert isinstance(got, ChamferScore)
    assert got.accuracy == pytest.approx(0.775, abs=1e-12)
    assert got.completeness == pytest.approx(comp, abs=1e-12)
    assert got.chamfer == pytest.approx(0.5 * (0.775 + comp), abs=1e-12)
    # unweighted accuracy = (0.4 + 0.9) / 2 = 0.65
    assert weighted_chamfer(pred, REF).accuracy == pytest.approx(0.65, abs=1e-12)
    scoped = weighted_chamfer(pred, REF, recall_mask=REF[:, 0] <= 3.0)
    assert scoped.completeness == pytest.approx(float(np.mean(listed[:4])), abs=1e-12)
    empty = weighted_chamfer(np.zeros((0, 3)), REF)
    assert math.isnan(empty.accuracy) and math.isnan(empty.completeness)
    assert math.isnan(empty.chamfer)


def test_energy_within() -> None:
    """Captured energy fractions follow the hand-computed cumulative table."""
    pos = np.array([[0.2, 0.0, 0.0], [0.0, 0.6, 0.0], [0.0, 0.0, 1.5], [3.0, 0.0, 0.0]])
    weights = np.array([1.0, 2.0, 3.0, 4.0])
    ref = np.array([[0.0, 0.0, 0.0]])
    # distances [0.2, 0.6, 1.5, 3.0]; cumulative weights [1, 3, 6, 10] of 10
    assert energy_within(pos, weights, ref, 0.5) == pytest.approx(0.1, abs=1e-12)
    assert energy_within(pos, weights, ref, 1.0) == pytest.approx(0.3, abs=1e-12)
    assert energy_within(pos, weights, ref, 2.0) == pytest.approx(0.6, abs=1e-12)
    assert energy_within(pos, weights, ref, 3.0) == pytest.approx(1.0, abs=1e-12)
    assert math.isnan(energy_within(pos, np.zeros(4), ref, 1.0))
    assert energy_within(pos, weights, np.zeros((0, 3)), 5.0) == pytest.approx(0.0, abs=0.0)
    with pytest.raises(ValueError):
        energy_within(pos, np.array([-1.0, 0.0, 0.0, 0.0]), ref, 1.0)
    with pytest.raises(ValueError):
        energy_within(pos, weights, ref, -1.0)


def test_map_point_cloud() -> None:
    """Cloud extraction keeps voxels above the min-max level in C order."""
    grid = VoxelGrid(origin=(1.0, 2.0, 3.0), spacing=0.5, shape=(3, 3, 3))
    density = np.zeros((3, 3, 3))
    density[1, 1, 1] = 10.0
    density[2, 2, 2] = 2.0
    density[0, 0, 0] = 0.5
    density[0, 1, 2] = -1.0
    # low = -1, high = 10, level = -1 + 0.1 * 11 = 0.1
    pos, weights = map_point_cloud(density, grid)
    np.testing.assert_allclose(pos, [[1.0, 2.0, 3.0], [1.5, 2.5, 3.5], [2.0, 3.0, 4.0]], atol=1e-12)
    np.testing.assert_allclose(weights, [1.5, 11.0, 3.0], atol=1e-12)
    # rel 0.5 -> level 4.5 keeps only the peak voxel
    pos, weights = map_point_cloud(density, grid, 0.5)
    np.testing.assert_allclose(pos, [[1.5, 2.5, 3.5]], atol=1e-12)
    np.testing.assert_allclose(weights, [11.0], atol=1e-12)
    # a NaN voxel is ignored and leaves the result unchanged
    density[2, 0, 0] = np.nan
    pos, weights = map_point_cloud(density, grid)
    np.testing.assert_allclose(pos, [[1.0, 2.0, 3.0], [1.5, 2.5, 3.5], [2.0, 3.0, 4.0]], atol=1e-12)
    np.testing.assert_allclose(weights, [1.5, 11.0, 3.0], atol=1e-12)
    pos, weights = map_point_cloud(np.full((3, 3, 3), 2.0), grid)
    assert pos.shape == (0, 3) and weights.shape == (0,)
    with pytest.raises(ValueError):
        map_point_cloud(np.zeros((2, 2, 2)), grid)
    with pytest.raises(ValueError):
        map_point_cloud(density, grid, 1.5)


def test_plane_errors() -> None:
    """Plane errors are sign/scale invariant with anchored offsets."""
    deg5 = math.radians(5.0)
    normal = np.array([math.cos(deg5), 0.0, math.sin(deg5)])
    offset = float(normal @ np.array([20.3, 0.0, 10.0]))
    gt_n = np.array([[1.0, 0.0, 0.0]])
    gt_d = np.array([20.0])
    # sign flip of (n, d) describes the same plane
    angle, off = plane_errors(
        -normal[None, :], np.array([-offset]), gt_n, gt_d, gt_anchor=np.array([[20.0, 0.0, 10.0]])
    )
    assert angle[0] == pytest.approx(deg5, abs=1e-12)
    assert off[0] == pytest.approx(0.3 * math.cos(deg5), abs=1e-12)
    # an off-plane anchor projects back onto the GT plane
    _, off2 = plane_errors(
        -normal[None, :], np.array([-offset]), gt_n, gt_d, gt_anchor=np.array([[25.0, 0.0, 10.0]])
    )
    assert off2[0] == pytest.approx(0.3 * math.cos(deg5), abs=1e-12)
    # default anchor (origin) adds the 10 * sin5 lever arm
    _, off3 = plane_errors(normal[None, :], np.array([offset]), gt_n, gt_d)
    assert off3[0] == pytest.approx(0.3 * math.cos(deg5) + 10.0 * math.sin(deg5), abs=1e-9)
    # scaling (n, d) describes the same plane
    angle4, off4 = plane_errors(
        2.0 * normal[None, :],
        np.array([2.0 * offset]),
        gt_n,
        gt_d,
        gt_anchor=np.array([[20.0, 0.0, 10.0]]),
    )
    assert angle4[0] == pytest.approx(deg5, abs=1e-12)
    assert off4[0] == pytest.approx(0.3 * math.cos(deg5), abs=1e-12)
    with pytest.raises(ValueError):
        plane_errors(np.array([[0.0, 0.0, 0.0]]), np.array([1.0]), gt_n, gt_d)


def test_match_planes() -> None:
    """Plane matching gates, scores and prefers maximum cardinality."""
    deg2 = math.radians(2.0)
    gt_n = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    gt_d = np.array([20.0, 25.0, 0.0])
    est_n = np.array(
        [
            [math.sin(deg2), math.cos(deg2), 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [math.cos(math.radians(20.0)), math.sin(math.radians(20.0)), 0.0],
        ]
    )
    est_d = np.array([25.0 * math.cos(deg2) + 0.2, -20.5, 3.0, 20.0 * math.cos(math.radians(20.0))])
    got = match_planes(est_n, est_d, gt_n, gt_d)
    assert isinstance(got, PlaneMatching)
    np.testing.assert_array_equal(got.est_idx, [0, 1])
    np.testing.assert_array_equal(got.gt_idx, [1, 0])
    np.testing.assert_allclose(got.angle, np.radians([2.0, 0.0]), atol=1e-9)
    np.testing.assert_allclose(got.offset, [0.2, 0.5], atol=1e-9)
    assert (got.tp, got.fp, got.fn) == (2, 2, 1)
    summary = got.summary()
    assert summary["angle_deg_median"] == pytest.approx(1.0, abs=1e-12)
    assert summary["offset_m_p90"] == pytest.approx(np.percentile([0.2, 0.5], 90))

    # cardinality first: two admissible pairs beat one greedy nearest pair
    near = match_planes(
        np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        np.array([20.45, 19.4]),
        np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        np.array([20.0, 21.0]),
    )
    np.testing.assert_array_equal(near.est_idx, [0, 1])
    np.testing.assert_array_equal(near.gt_idx, [1, 0])
    assert near.tp == 2

    # cardinality first even when one zero-cost pair is available: est0 fits A exactly
    # (cost 0) and B at 9 deg / 0.95 m (0.9 + 0.95); est1 fits only A at 9 deg / 0.95 m.
    # Both matched costs 3.7; a solver that merely charges 2 per missed pair keeps est0-A.
    deg9 = math.radians(9.0)
    gt_b = np.array([math.cos(deg9), math.sin(deg9), 0.0])
    tight = match_planes(
        np.array([[1.0, 0.0, 0.0], [math.cos(deg9), -math.sin(deg9), 0.0]]),
        np.array([20.0, 20.0 * math.cos(deg9) - 0.95]),
        np.array([[1.0, 0.0, 0.0], gt_b]),
        np.array([20.0, 20.95 / math.cos(deg9)]),
    )
    np.testing.assert_array_equal(tight.est_idx, [0, 1])
    np.testing.assert_array_equal(tight.gt_idx, [1, 0])
    np.testing.assert_allclose(tight.angle, [deg9, deg9], atol=1e-12)
    np.testing.assert_allclose(tight.offset, [0.95, 0.95], atol=1e-9)

    # the offset gate is inclusive: offset exactly 1.0 still matches
    edge = match_planes(
        np.array([[1.0, 0.0, 0.0]]),
        np.array([21.0]),
        np.array([[1.0, 0.0, 0.0]]),
        np.array([20.0]),
    )
    assert edge.tp == 1
    empty = match_planes(np.zeros((0, 3)), np.zeros((0,)), gt_n, gt_d)
    assert empty.tp == 0
    assert all(math.isnan(value) for value in empty.summary().values())


def test_stratified_recall() -> None:
    """Per-label recall counts matched GT points with sorted label order."""
    gt = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [30.0, 0.0, 0.0]])
    det = np.array([[0.1, 0.0, 0.0], [10.2, 0.0, 0.0], [50.0, 0.0, 0.0]])
    labels = ["a", "b", "a", "c"]
    got = stratified_recall(match(det, gt, 1.0), labels)
    assert list(got) == ["a", "b", "c"]
    assert got["a"]["num_gt"] == 2 and got["a"]["tp"] == 1
    assert got["a"]["recall"] == pytest.approx(0.5, abs=1e-12)
    assert got["b"] == {"num_gt": 1, "tp": 1, "recall": 1.0, "weighted_recall": 1.0}
    assert got["c"]["recall"] == pytest.approx(0.0, abs=1e-12)
    assert got["c"]["tp"] == 0
    weighted = stratified_recall(
        match(det, gt, 1.0), labels, gt_weight=np.array([1.0, 2.0, 3.0, 4.0])
    )
    # label a: matched weight 1 of (1 + 3) -> 0.25
    assert weighted["a"]["weighted_recall"] == pytest.approx(0.25, abs=1e-12)
    assert weighted["b"]["weighted_recall"] == pytest.approx(1.0, abs=1e-12)
    assert weighted["c"]["weighted_recall"] == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError):
        stratified_recall(match(det, gt, 1.0), ["a", "b"])


def test_surface_report_matches_single_metrics() -> None:
    """The one-pass report equals the per-gate functions on the hand case."""
    pred = np.array([[0.0, 0.0, 0.4], [5.0, 0.0, 0.9], [20.0, 0.0, 0.0]])
    weights = np.array([1.0, 3.0, 2.0])
    mask = REF[:, 0] <= 6.0
    energy_pos = np.array([[0.2, 0.0, 0.0], [0.0, 0.6, 3.0], [9.0, 0.0, 1.5]])
    energy_w = np.array([1.0, 2.0, 3.0])
    report = surface_report(
        pred,
        REF,
        SURFACE_THRESHOLDS_M,
        pred_weight=weights,
        recall_mask=mask,
        energy_pos=energy_pos,
        energy_weight=energy_w,
    )
    for gate, score in zip(SURFACE_THRESHOLDS_M, report.scores, strict=True):
        # P/R/F are unweighted: the prediction weights only enter the Chamfer accuracy
        assert score == surface_prf(pred, REF, gate, recall_mask=mask)
    assert report.chamfer == weighted_chamfer(pred, REF, pred_weight=weights, recall_mask=mask)
    assert report.energy == tuple(
        energy_within(energy_pos, energy_w, REF, gate) for gate in SURFACE_THRESHOLDS_M
    )
    # energy distances [0.2, hypot(0.6, 3), 1.5]: fractions 1/6, 1/6, 4/6
    np.testing.assert_allclose(report.energy, [1 / 6, 1 / 6, 4 / 6], atol=1e-12)
    assert surface_report(pred, REF, (1.0,)).energy == ()
