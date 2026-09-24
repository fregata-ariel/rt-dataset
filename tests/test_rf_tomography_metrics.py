"""Unit tests for plateau_rt.domain.rf_tomography.metrics (M1, M5, M6)."""

from __future__ import annotations

import os

import numpy as np
import pytest

from plateau_rt.domain.rf_tomography.geometry import VoxelGrid
from plateau_rt.domain.rf_tomography.metrics import (
    GaugeErrors,
    ap_at,
    average_precision,
    detectable_mask,
    froc,
    gauge_errors,
    loc_error_decomposed,
    match,
    nms_peaks,
    nmse_global_phase,
    nmse_power_scale,
    recall_at_fa,
)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence(seed))


def _gaussian_density(
    grid: VoxelGrid, centre: np.ndarray, sigma: float, amplitude: float = 1.0
) -> np.ndarray:
    diff = grid.centers() - centre
    quad = np.sum(diff**2, axis=1) / (2.0 * sigma**2)
    return (amplitude * np.exp(-quad)).reshape(grid.shape)


# --- NMS and refinement -----------------------------------------------------


@pytest.mark.parametrize(
    "delta_true",
    [
        (0.3, 0.0, 0.0),
        (0.0, 0.3, 0.0),
        (0.0, 0.0, 0.3),
        (0.3, -0.3, 0.3),
        (-0.3, 0.3, -0.3),
    ],
)
def test_nms_refines_gaussian_blob(delta_true: tuple[float, float, float]) -> None:
    grid = VoxelGrid(origin=(-3.0, 2.0, 1.0), spacing=0.5, shape=(20, 18, 16))
    sigma = 1.5 * grid.spacing
    voxel_centre = grid.origin + grid.spacing * np.array([9.0, 8.0, 7.0])
    true_centre = voxel_centre + grid.spacing * np.asarray(delta_true)
    density = _gaussian_density(grid, true_centre, sigma)

    peaks = nms_peaks(density, grid, radius=1.0, min_value=1e-6)

    assert peaks.positions.shape == (1, 3)
    assert int(peaks.indices[0]) == np.ravel_multi_index((9, 8, 7), grid.shape)
    np.testing.assert_allclose(peaks.values, [density[9, 8, 7]], rtol=0, atol=0)
    assert float(np.max(np.abs(peaks.positions[0] - true_centre))) < 0.05 * grid.spacing
    # The test discriminates: the raw voxel centre is much farther away.
    assert float(np.max(np.abs(voxel_centre - true_centre))) >= 0.25 * grid.spacing


@pytest.mark.parametrize("delta_true", [(0.3, -0.2, 0.1), (0.2, 0.2, -0.3), (-0.3, 0.1, 0.25)])
def test_nms_refines_rotated_anisotropic_blob(delta_true: tuple[float, float, float]) -> None:
    # A blob tilted 45 deg in x-y needs the cross terms of the full 3x3x3 quadratic fit:
    # per-axis parabolas (or a wrong Hessian) are off by 0.15-0.22 voxel here.
    grid = VoxelGrid(origin=(-3.0, 2.0, 1.0), spacing=0.5, shape=(20, 18, 16))
    angle = np.pi / 4.0
    rot = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0, 0, 1]]
    )
    sigmas = grid.spacing * np.array([4.0, 1.5, 2.0])
    precision = np.linalg.inv(rot @ np.diag(sigmas**2) @ rot.T)
    true_centre = grid.origin + grid.spacing * (np.array([9.0, 8.0, 7.0]) + delta_true)
    diff = grid.centers() - true_centre
    density = np.exp(-0.5 * np.einsum("pi,ij,pj->p", diff, precision, diff)).reshape(grid.shape)

    peaks = nms_peaks(density, grid, radius=1.0, min_value=1e-6)

    assert int(peaks.indices[0]) == np.ravel_multi_index((9, 8, 7), grid.shape)
    assert peaks.positions.shape == (1, 3)
    assert float(np.max(np.abs(peaks.positions[0] - true_centre))) < 0.05 * grid.spacing


def test_nms_two_blobs_ordering_and_refine_flag() -> None:
    grid = VoxelGrid(origin=(-3.0, 2.0, 1.0), spacing=0.5, shape=(20, 18, 16))
    sigma = 1.5 * grid.spacing
    centre_a = grid.origin + grid.spacing * (np.array([5.0, 5.0, 5.0]) + (0.2, -0.3, 0.25))
    centre_b = grid.origin + grid.spacing * (np.array([14.0, 12.0, 10.0]) + (-0.25, 0.3, -0.2))
    density = _gaussian_density(grid, centre_a, sigma, 1.0)
    density += _gaussian_density(grid, centre_b, sigma, 0.5)

    peaks = nms_peaks(density, grid, radius=1.0, min_value=1e-6)

    assert peaks.positions.shape == (2, 3)
    assert int(peaks.indices[0]) == np.ravel_multi_index((5, 5, 5), grid.shape)
    assert int(peaks.indices[1]) == np.ravel_multi_index((14, 12, 10), grid.shape)
    assert float(peaks.values[0]) > float(peaks.values[1])
    np.testing.assert_allclose(peaks.positions[0], centre_a, rtol=0, atol=0.05 * 0.5)
    np.testing.assert_allclose(peaks.positions[1], centre_b, rtol=0, atol=0.05 * 0.5)

    raw = nms_peaks(density, grid, radius=1.0, refine=False, min_value=1e-6)
    np.testing.assert_array_equal(raw.indices, peaks.indices)
    np.testing.assert_array_equal(raw.positions, grid.centers()[raw.indices])
    np.testing.assert_array_equal(raw.offsets, np.zeros((2, 3)))


def _spike_grid() -> tuple[VoxelGrid, np.ndarray]:
    grid = VoxelGrid(origin=(0.0, 0.0, 0.0), spacing=1.0, shape=(12, 5, 5))
    density = np.zeros(grid.shape)
    density[2, 2, 2] = 5.0
    density[4, 2, 2] = 3.0
    density[9, 2, 2] = 1.0
    return grid, density


def test_nms_greedy_suppression() -> None:
    grid, density = _spike_grid()
    flat = lambda ix: int(np.ravel_multi_index(ix, grid.shape))  # noqa: E731

    peaks = nms_peaks(density, grid, radius=1.5, refine=False, min_value=0.0)
    np.testing.assert_array_equal(
        peaks.indices, [flat((2, 2, 2)), flat((4, 2, 2)), flat((9, 2, 2))]
    )
    np.testing.assert_array_equal(peaks.values, [5.0, 3.0, 1.0])

    # The gate is inclusive: distance 2.0 m <= radius suppresses (4, 2, 2).
    peaks = nms_peaks(density, grid, radius=2.0, refine=False, min_value=0.0)
    np.testing.assert_array_equal(peaks.indices, [flat((2, 2, 2)), flat((9, 2, 2))])
    np.testing.assert_array_equal(peaks.values, [5.0, 1.0])

    peaks = nms_peaks(density, grid, radius=2.0, refine=False, min_value=0.0, max_peaks=1)
    np.testing.assert_array_equal(peaks.indices, [flat((2, 2, 2))])

    peaks = nms_peaks(density, grid, radius=1.5, refine=False, min_value=2.0)
    np.testing.assert_array_equal(peaks.indices, [flat((2, 2, 2)), flat((4, 2, 2))])


@pytest.mark.parametrize("radius", [0.0, 0.5, 0.7, 1.0, 1.3])
def test_nms_stencil_matches_brute_force(radius: float) -> None:
    grid = VoxelGrid(origin=(0.0, 0.0, 0.0), spacing=0.5, shape=(14, 12, 10))
    rng = _rng(41)
    from scipy.ndimage import gaussian_filter

    density = gaussian_filter(rng.random(grid.shape), 1.0)
    min_value = float(np.median(density))
    peaks = nms_peaks(density, grid, radius=radius, refine=False, min_value=min_value)

    # Independent candidates with an explicit 26-neighbour comparison.
    nx, ny, nz = grid.shape
    flats: list[int] = []
    vals: list[float] = []
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                value = float(density[ix, iy, iz])
                if not np.isfinite(value) or not value > min_value:
                    continue
                is_peak = True
                for ax in (-1, 0, 1):
                    for ay in (-1, 0, 1):
                        for az in (-1, 0, 1):
                            if (ax, ay, az) == (0, 0, 0):
                                continue
                            jx, jy, jz = ix + ax, iy + ay, iz + az
                            if 0 <= jx < nx and 0 <= jy < ny and 0 <= jz < nz:
                                other = float(density[jx, jy, jz])
                                if np.isfinite(other) and other > value:
                                    is_peak = False
                                    break
                        if not is_peak:
                            break
                    if not is_peak:
                        break
                if is_peak:
                    flats.append(int(np.ravel_multi_index((ix, iy, iz), grid.shape)))
                    vals.append(value)
    order = np.lexsort((np.asarray(flats), -np.asarray(vals)))
    ordered = [flats[k] for k in order]
    centres = {
        flat: grid.origin + grid.spacing * np.asarray(np.unravel_index(flat, grid.shape))
        for flat in ordered
    }
    expected: list[int] = []
    for flat in ordered:
        if any(float(np.linalg.norm(centres[flat] - centres[keep])) <= radius for keep in expected):
            continue
        expected.append(flat)
    np.testing.assert_array_equal(peaks.indices, np.asarray(expected, dtype=np.int64))


def test_nms_zero_plateaus_need_min_value() -> None:
    grid, density = _spike_grid()
    spikes = {(2, 2, 2), (4, 2, 2), (9, 2, 2)}

    # Independent count: zero voxels with no spike among existing neighbours.
    expected_zeros = 0
    for ix in range(grid.shape[0]):
        for iy in range(grid.shape[1]):
            for iz in range(grid.shape[2]):
                if (ix, iy, iz) in spikes:
                    continue
                has_spike = False
                for ax in (-1, 0, 1):
                    for ay in (-1, 0, 1):
                        for az in (-1, 0, 1):
                            if (ax, ay, az) == (0, 0, 0):
                                continue
                            jx, jy, jz = ix + ax, iy + ay, iz + az
                            if (
                                0 <= jx < grid.shape[0]
                                and 0 <= jy < grid.shape[1]
                                and 0 <= jz < grid.shape[2]
                                and density[jx, jy, jz] != 0.0
                            ):
                                has_spike = True
                if not has_spike:
                    expected_zeros += 1

    peaks = nms_peaks(density, grid, radius=0.0, refine=False)
    assert peaks.positions.shape[0] == expected_zeros + 3
    assert peaks.positions.shape[0] > 100  # documents the need for min_value


def test_nms_boundary_and_nonfinite_neighbourhood() -> None:
    grid = VoxelGrid(origin=(0.0, 0.0, 0.0), spacing=0.5, shape=(10, 12, 11))
    sigma = 1.5 * grid.spacing

    edge_centre = grid.origin + grid.spacing * (np.array([0.0, 6.0, 5.0]) + (0.0, 0.3, -0.3))
    density = _gaussian_density(grid, edge_centre, sigma)
    peaks = nms_peaks(density, grid, radius=1.0, min_value=1e-6)
    assert peaks.positions.shape == (1, 3)
    assert int(peaks.indices[0]) == np.ravel_multi_index((0, 6, 5), grid.shape)
    assert peaks.positions[0, 0] == pytest.approx(grid.origin[0])
    assert abs(peaks.positions[0, 1] - edge_centre[1]) < 0.05 * grid.spacing
    assert abs(peaks.positions[0, 2] - edge_centre[2]) < 0.05 * grid.spacing

    inner_centre = grid.origin + grid.spacing * (np.array([5.0, 6.0, 5.0]) + (0.0, 0.3, 0.0))
    density = _gaussian_density(grid, inner_centre, sigma)
    density[4, 6, 5] = np.nan
    peaks = nms_peaks(density, grid, radius=1.0, min_value=1e-6)
    assert peaks.positions.shape == (1, 3)
    assert int(peaks.indices[0]) == np.ravel_multi_index((5, 6, 5), grid.shape)
    assert np.all(np.isfinite(peaks.positions))
    assert np.all(np.isfinite(peaks.offsets))
    assert np.all(np.isfinite(peaks.values))
    assert peaks.offsets[0, 0] == 0.0
    assert abs(peaks.positions[0, 1] - inner_centre[1]) < 0.05 * grid.spacing
    assert abs(peaks.offsets[0, 2]) < 0.05
    # NaN voxels are never peaks.
    assert int(peaks.indices[0]) != np.ravel_multi_index((4, 6, 5), grid.shape)


def test_nms_validation() -> None:
    grid = VoxelGrid(origin=(0.0, 0.0, 0.0), spacing=1.0, shape=(4, 4, 4))
    good = np.zeros(grid.shape)
    with pytest.raises(ValueError, match="shape"):
        nms_peaks(np.zeros((4, 4, 5)), grid, radius=1.0)
    with pytest.raises(ValueError, match="real"):
        nms_peaks(np.zeros(grid.shape, dtype=np.complex128), grid, radius=1.0)
    with pytest.raises(ValueError, match="radius"):
        nms_peaks(good, grid, radius=-1.0)
    with pytest.raises(ValueError, match="radius"):
        nms_peaks(good, grid, radius=np.nan)
    with pytest.raises(ValueError, match="max_peaks"):
        nms_peaks(good, grid, radius=1.0, max_peaks=0)

    empty = nms_peaks(np.full(grid.shape, np.nan), grid, radius=1.0)
    assert empty.positions.shape == (0, 3)
    assert empty.values.shape == (0,)
    assert empty.indices.shape == (0,)
    assert empty.offsets.shape == (0, 3)


# --- Matching ---------------------------------------------------------------


def test_match_hungarian_beats_greedy() -> None:
    det = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    gt = np.array([[0.45, 0.0, 0.0], [-0.6, 0.0, 0.0]])
    matched = match(det, gt, gate=0.7)
    assert matched.tp == 2
    assert matched.fp == 0
    assert matched.fn == 0
    np.testing.assert_array_equal(matched.det_idx, [0, 1])
    np.testing.assert_array_equal(matched.gt_idx, [1, 0])
    np.testing.assert_allclose(matched.distance, [0.6, 0.55], rtol=0, atol=1e-12)


def test_match_gate_inclusive() -> None:
    det = np.array([[0.0, 0.0, 0.0]])
    assert match(det, np.array([[0.5, 0.0, 0.0]]), gate=0.5).tp == 1
    missed = match(det, np.array([[0.5000001, 0.0, 0.0]]), gate=0.5)
    assert (missed.tp, missed.fp, missed.fn) == (0, 1, 1)


def test_match_empty() -> None:
    gt = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    det = np.array([[0.0, 0.0, 0.0]])
    first = match(np.zeros((0, 3)), gt, gate=1.0)
    assert (first.tp, first.fp, first.fn) == (0, 0, 2)
    assert (first.num_det, first.num_gt) == (0, 2)
    second = match(det, np.zeros((0, 3)), gate=1.0)
    assert (second.tp, second.fp, second.fn) == (0, 1, 0)
    both = match(np.zeros((0, 3)), np.zeros((0, 3)), gate=1.0)
    assert (both.tp, both.fp, both.fn) == (0, 0, 0)
    for result in (first, second, both):
        assert result.det_idx.shape == (0,)
        assert result.gt_idx.shape == (0,)
        assert result.distance.shape == (0,)
        assert result.det_idx.dtype == np.int64
        assert result.gt_idx.dtype == np.int64
        assert result.distance.dtype == np.float64
    with pytest.raises(ValueError, match="gate"):
        match(det, gt, gate=0.0)


def _brute_force_match(dist: np.ndarray, gate: float) -> tuple[int, float, list[tuple[int, int]]]:
    best = (0, 0.0, [])
    num_det, num_gt = dist.shape

    def rec(i: int, used: set[int], pairs: list[tuple[int, int]], total: float) -> None:
        nonlocal best
        if i == num_det:
            if len(pairs) > best[0] or (len(pairs) == best[0] and total < best[1] - 1e-15):
                best = (len(pairs), total, list(pairs))
            return
        rec(i + 1, used, pairs, total)
        for j in range(num_gt):
            if j not in used and dist[i, j] <= gate:
                used.add(j)
                pairs.append((i, j))
                rec(i + 1, used, pairs, total + float(dist[i, j]))
                pairs.pop()
                used.remove(j)

    rec(0, set(), [], 0.0)
    return best


@pytest.mark.parametrize("seed", range(30))
def test_match_bruteforce(seed: int) -> None:
    rng = _rng(1000 + seed)
    num_det = int(rng.integers(0, 6))
    num_gt = int(rng.integers(0, 6))
    det = rng.random((num_det, 3)) * 3.0
    gt = rng.random((num_gt, 3)) * 3.0
    gate = 1.0

    matched = match(det, gt, gate=gate)
    if num_det == 0 or num_gt == 0:
        dist = np.zeros((num_det, num_gt))
    else:
        dist = np.linalg.norm(det[:, None, :] - gt[None, :, :], axis=-1)
    card, total, _ = _brute_force_match(dist, gate)

    assert matched.tp == card
    assert float(np.sum(matched.distance)) == pytest.approx(total, abs=1e-9)
    assert np.all(matched.distance <= gate)
    assert matched.det_idx.dtype == np.int64


# --- FROC / AP / recall at FA -------------------------------------------------


def _handmade_froc() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gt = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
    # d1..d5 in shuffled order [d3, d5, d1, d4, d2].
    det = np.array(
        [
            [10.0, 0.2, 0.0],
            [0.0, 10.0, 0.4],
            [0.1, 0.0, 0.0],
            [0.0, 0.3, 0.0],
            [50.0, 50.0, 50.0],
        ]
    )
    scores = np.array([0.7, 0.5, 0.9, 0.6, 0.8])
    det_w = np.array([3.0, 1.0, 4.0, 1.0, 1.0])
    return det, scores, gt, det_w


def test_froc_handmade() -> None:
    det, scores, gt, det_w = _handmade_froc()
    gt_w = np.array([1.0, 2.0, 7.0])
    curve = froc(det, scores, gt, gate=1.0, gt_weight=gt_w, det_weight=det_w, volume_m3=2000.0)

    np.testing.assert_allclose(curve.thresholds, [0.9, 0.8, 0.7, 0.6, 0.5], rtol=0, atol=0)
    np.testing.assert_array_equal(curve.num_det, [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(curve.tp, [1, 1, 2, 2, 3])
    np.testing.assert_array_equal(curve.fp, [0, 1, 1, 2, 2])
    np.testing.assert_allclose(curve.recall, [1 / 3, 1 / 3, 2 / 3, 2 / 3, 1.0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(curve.precision, [1.0, 0.5, 2 / 3, 0.5, 0.6], rtol=0, atol=1e-12)
    np.testing.assert_allclose(curve.weighted_recall, [0.1, 0.1, 0.3, 0.3, 1.0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        curve.weighted_precision, [1.0, 4 / 5, 7 / 8, 7 / 9, 8 / 10], rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(curve.fa_per_1000m3, [0.0, 0.5, 0.5, 1.0, 1.0], rtol=0, atol=1e-12)
    assert curve.num_gt == 3

    assert average_precision(curve) == pytest.approx(34 / 45, abs=1e-12)
    assert ap_at(det, scores, gt, gate=1.0) == pytest.approx(34 / 45, abs=1e-12)
    assert recall_at_fa(curve, 1.0) == pytest.approx(1.0)
    assert recall_at_fa(curve, 0.5) == pytest.approx(2 / 3)
    assert recall_at_fa(curve, 0.0) == pytest.approx(1 / 3)


def test_froc_reassignment_across_thresholds() -> None:
    gt = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]])
    det = np.array([[0.7, 0.0, 0.0], [-0.5, 0.0, 0.0]])
    scores = np.array([0.9, 0.8])
    curve = froc(det, scores, gt, gate=1.0, volume_m3=1000.0)
    np.testing.assert_array_equal(curve.tp, [1, 2])
    np.testing.assert_allclose(curve.recall, [0.5, 1.0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(curve.precision, [1.0, 1.0], rtol=0, atol=1e-12)
    assert average_precision(curve) == pytest.approx(1.0)


def test_average_precision_interpolated() -> None:
    gt = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    det = np.array([[50.0, 50.0, 50.0], [0.1, 0.0, 0.0], [10.0, 0.1, 0.0]])
    scores = np.array([0.9, 0.8, 0.7])
    curve = froc(det, scores, gt, gate=1.0, volume_m3=1000.0)
    np.testing.assert_allclose(curve.recall, [0.0, 0.5, 1.0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(curve.precision, [0.0, 0.5, 2 / 3], rtol=0, atol=1e-12)
    ap = average_precision(curve)
    assert ap == pytest.approx(2 / 3, abs=1e-12)
    assert abs(ap - 7 / 12) > 0.05


def test_froc_weighted_nan_rules() -> None:
    gt = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    det = np.array([[0.1, 0.0, 0.0], [5.1, 0.0, 0.0]])
    scores = np.array([0.9, 0.4])
    zero_gt = np.zeros(2)
    curve = froc(det, scores, gt, gate=1.0, gt_weight=zero_gt, volume_m3=1000.0)
    assert np.all(np.isnan(curve.weighted_recall))
    np.testing.assert_allclose(curve.recall, [0.5, 1.0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(curve.precision, [1.0, 1.0], rtol=0, atol=1e-12)

    zero_det = np.zeros(2)
    curve = froc(det, scores, gt, gate=1.0, det_weight=zero_det, volume_m3=1000.0)
    assert np.all(np.isnan(curve.weighted_precision))
    np.testing.assert_allclose(curve.recall, [0.5, 1.0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(curve.precision, [1.0, 1.0], rtol=0, atol=1e-12)


def test_froc_ties_and_edge_cases() -> None:
    gt = np.array([[0.0, 0.0, 0.0]])
    det = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
    curve = froc(det, np.array([0.5, 0.5]), gt, gate=1.0, volume_m3=1000.0)
    assert curve.thresholds.shape == (1,)
    np.testing.assert_array_equal(curve.num_det, [2])

    empty = froc(np.zeros((0, 3)), np.zeros((0,)), gt, gate=1.0, volume_m3=1000.0)
    assert empty.thresholds.shape == (0,)
    assert average_precision(empty) == 0.0
    assert recall_at_fa(empty, 1.0) == 0.0

    no_gt = froc(det, np.array([0.9, 0.4]), np.zeros((0, 3)), gate=1.0, volume_m3=1000.0)
    assert np.all(np.isnan(no_gt.recall))
    assert np.isnan(average_precision(no_gt))

    no_volume = froc(det, np.array([0.9, 0.4]), gt, gate=1.0)
    assert np.all(np.isnan(no_volume.fa_per_1000m3))
    with pytest.raises(ValueError, match="volume"):
        recall_at_fa(no_volume, 1.0)

    with pytest.raises(ValueError, match="det_score"):
        froc(det, np.array([0.9, np.nan]), gt, gate=1.0)


def test_detectable_mask() -> None:
    power = np.array([[1.0, 0.0], [10.0**-3.5, 0.0], [10.0**-3.5, 5e-4], [0.0, 0.0]])
    np.testing.assert_array_equal(detectable_mask(power), [True, False, True, False])
    np.testing.assert_array_equal(
        detectable_mask(power, reference=np.array([1.0, 1.0])),
        [True, False, False, False],
    )
    np.testing.assert_array_equal(
        detectable_mask(power, dynamic_range_db=40.0), [True, True, True, False]
    )
    with pytest.raises(ValueError, match="power"):
        detectable_mask(np.array([[1.0, np.nan]]))
    with pytest.raises(ValueError, match="power"):
        detectable_mask(np.array([[-1.0, 0.0]]))


# --- Localisation error -------------------------------------------------------


def test_loc_error_frames() -> None:
    ref = np.array([0.0, 0.0, 0.0])
    cases = [
        (
            np.array([[10.3, -0.4, 0.2]]),
            np.array([[10.0, 0.0, 0.0]]),
            (0.3, -0.4, 0.2, np.sqrt(0.29)),
        ),
        (
            np.array([[0.5, 10.0, 0.0]]),
            np.array([[0.0, 10.0, 0.0]]),
            (0.0, -0.5, 0.0, 0.5),
        ),
        (
            np.array([[6.0, 0.0, 9.0]]),
            np.array([[6.0, 0.0, 8.0]]),
            (0.8, 0.0, 0.6, 1.0),
        ),
        (
            np.array([[0.2, 0.1, 5.3]]),
            np.array([[0.0, 0.0, 5.0]]),
            (0.3, 0.2, 0.1, np.sqrt(0.14)),
        ),
    ]
    for est, gt, (rng, hor, ver, tot) in cases:
        err = loc_error_decomposed(est, gt, ref)
        np.testing.assert_allclose(err.range, [rng], rtol=0, atol=1e-12)
        np.testing.assert_allclose(err.horizontal, [hor], rtol=0, atol=1e-12)
        np.testing.assert_allclose(err.vertical, [ver], rtol=0, atol=1e-12)
        np.testing.assert_allclose(err.total, [tot], rtol=0, atol=1e-12)
        # Per-pair references match the broadcast reference.
        stacked = np.broadcast_to(ref, est.shape).copy()
        same = loc_error_decomposed(est, gt, stacked)
        np.testing.assert_allclose(same.range, err.range, rtol=0, atol=0)
        np.testing.assert_allclose(same.horizontal, err.horizontal, rtol=0, atol=0)
        np.testing.assert_allclose(same.vertical, err.vertical, rtol=0, atol=0)
    with pytest.raises(ValueError, match="ref"):
        loc_error_decomposed(np.array([[1.0, 0.0, 0.0]]), np.array([[0.0, 0.0, 0.0]]), ref)


def test_loc_error_orthogonality() -> None:
    rng = _rng(7)
    length = 200
    gt = rng.random((length, 3)) * 20.0 - 10.0
    est = gt + rng.standard_normal((length, 3))
    ref = rng.random((length, 3)) * 20.0 - 10.0
    err = loc_error_decomposed(est, gt, ref)
    lhs = err.range**2 + err.horizontal**2 + err.vertical**2
    rhs = err.total**2
    np.testing.assert_allclose(lhs, rhs, rtol=1e-12, atol=1e-24)


def test_loc_error_summary() -> None:
    gt = np.zeros((10, 3))
    gt[:, 0] = 100.0
    est = gt.copy()
    est[:, 0] += np.arange(1.0, 11.0)
    summary = loc_error_decomposed(est, gt, np.zeros(3)).summary()
    assert summary["range_median"] == pytest.approx(5.5)
    assert summary["range_p90"] == pytest.approx(9.1)
    assert summary["horizontal_median"] == pytest.approx(0.0, abs=1e-12)
    assert summary["horizontal_p90"] == pytest.approx(0.0, abs=1e-12)
    assert summary["vertical_median"] == pytest.approx(0.0, abs=1e-12)
    assert summary["vertical_p90"] == pytest.approx(0.0, abs=1e-12)
    assert summary["total_median"] == pytest.approx(5.5)
    assert summary["total_p90"] == pytest.approx(9.1)

    empty = loc_error_decomposed(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(3))
    assert all(np.isnan(v) for v in empty.summary().values())


# --- NMSE ---------------------------------------------------------------------


def test_nmse_global_phase() -> None:
    ref = np.array([1.0, 1j, -2.0, 0.5 - 0.5j])
    assert nmse_global_phase(np.exp(0.7j) * ref, ref) < 1e-28
    assert nmse_global_phase(2.0 * np.exp(-1.1j) * ref, ref) == pytest.approx(1.0, abs=1e-12)
    assert nmse_global_phase(np.array([0.0, 1.0]), np.array([1.0, 0.0])) == pytest.approx(
        2.0, abs=1e-12
    )

    rng = _rng(11)
    est = rng.standard_normal(50) + 1j * rng.standard_normal(50)
    ref = rng.standard_normal(50) + 1j * rng.standard_normal(50)
    value = nmse_global_phase(est, ref)
    phases = np.linspace(-np.pi, np.pi, 20001)
    residuals = np.array([np.sum(np.abs(np.exp(1j * p) * est - ref) ** 2) for p in phases]) / float(
        np.sum(np.abs(ref) ** 2)
    )
    grid_best = float(np.min(residuals))
    assert value <= grid_best + 1e-12
    assert abs(value - grid_best) <= 1e-6 * max(grid_best, 1e-12)
    closed = (
        float(np.sum(np.abs(est) ** 2))
        + float(np.sum(np.abs(ref) ** 2))
        - 2.0 * abs(complex(np.vdot(est, ref)))
    ) / float(np.sum(np.abs(ref) ** 2))
    assert abs(value - closed) <= 1e-10
    for phi in (0.3, -2.1, np.pi):
        assert abs(nmse_global_phase(np.exp(1j * phi) * est, ref) - value) <= 1e-12

    with pytest.raises(ValueError, match="nonzero"):
        nmse_global_phase(np.array([1.0]), np.array([0.0]))
    with pytest.raises(ValueError, match="same shape"):
        nmse_global_phase(np.array([1.0, 2.0]), np.array([1.0]))


def test_nmse_power_scale() -> None:
    rng = _rng(13)
    ref = rng.random(30) + 0.1
    assert nmse_power_scale(3.0 * ref, ref) < 1e-28

    est = np.zeros(6)
    est[0] = 1.0
    disjoint = np.zeros(6)
    disjoint[1] = 1.0
    assert nmse_power_scale(est, disjoint) == pytest.approx(1.0)

    pert = ref + 0.1 * ref * rng.random(ref.shape)
    value = nmse_power_scale(pert, ref)
    grid = np.arange(0.0, 3.0 + 5e-9, 1e-4)
    residuals = np.array([np.sum((a * pert - ref) ** 2) for a in grid]) / float(np.sum(ref**2))
    grid_best = float(np.min(residuals))
    assert value <= grid_best + 1e-12
    assert abs(value - grid_best) <= 1e-6 * max(grid_best, 1e-12)
    assert nmse_power_scale(-ref, ref) == pytest.approx(1.0)

    with pytest.raises(ValueError, match="real"):
        nmse_power_scale(np.array([1.0 + 1j]), np.array([1.0]))
    with pytest.raises(ValueError, match="same shape"):
        nmse_power_scale(np.array([1.0, 2.0]), np.array([1.0]))
    with pytest.raises(ValueError, match="nonzero"):
        nmse_power_scale(np.array([1.0]), np.array([0.0]))


# --- Gauge errors ---------------------------------------------------------------


def test_gauge_errors() -> None:
    rng = _rng(17)
    period = 1.28e-6
    phi_gt = rng.random((2, 2)) * 2.0 * np.pi
    tau_gt = rng.random((2, 2)) * 100e-9
    err = np.array([[0.1, -0.1], [0.0, 0.0]])
    shifts = np.array([[1, -2], [0, 3]])
    phi_est = phi_gt + 1.0 + err + 2.0 * np.pi * shifts
    tau_est = tau_gt + np.array([[1e-9, -2e-9], [period - 0.5e-9, 0.0]])

    out = gauge_errors((phi_est, tau_est), (phi_gt, tau_gt), period=period)
    assert isinstance(out, GaugeErrors)
    assert out.global_phase == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(out.phase, err, rtol=0, atol=1e-12)
    np.testing.assert_allclose(out.delay, [[1e-9, 2e-9], [0.5e-9, 0.0]], rtol=0, atol=1e-18)

    plain = gauge_errors((phi_est, tau_est), (phi_gt, tau_gt))
    assert plain.delay[1, 0] == pytest.approx(period - 0.5e-9, abs=1e-18)

    # Delay is absolute while phase drops the common offset.
    shifted = gauge_errors((phi_gt + 0.3, tau_gt + 5e-9), (phi_gt, tau_gt), period=period)
    np.testing.assert_allclose(shifted.delay, np.full((2, 2), 5e-9), rtol=0, atol=1e-18)
    np.testing.assert_allclose(shifted.phase, np.zeros((2, 2)), rtol=0, atol=1e-12)
    assert shifted.global_phase == pytest.approx(0.3, abs=1e-12)

    with pytest.raises(ValueError, match="same shape"):
        gauge_errors((phi_est, tau_est), (phi_gt[:, :1], tau_gt))
    with pytest.raises(ValueError, match="finite"):
        gauge_errors((np.full((2, 2), np.nan), tau_est), (phi_gt, tau_gt), period=period)
    with pytest.raises(ValueError, match="period"):
        gauge_errors((phi_est, tau_est), (phi_gt, tau_gt), period=0.0)


# --- Benchmark ------------------------------------------------------------------


@pytest.mark.skipif(os.environ.get("RF_TOMO_BENCH") != "1", reason="opt-in benchmark")
def test_benchmark() -> None:
    import time

    from scipy.ndimage import gaussian_filter

    grid = VoxelGrid.from_bounds((-50.0, -50.0, -2.0), (50.0, 50.0, 40.0), 0.5)
    rng = _rng(23)
    density = gaussian_filter(rng.random(grid.shape), 2.0)
    start = time.perf_counter()
    peaks = nms_peaks(density, grid, radius=2.0, min_value=1e-3 * float(density.max()))
    nms_time = time.perf_counter() - start

    det = rng.random((500, 3)) * 100.0 - 50.0
    det[:, 2] = rng.random(500) * 42.0 - 2.0
    scores = rng.random(500)
    gt_points = rng.random((60, 3)) * 100.0 - 50.0
    gt_points[:, 2] = rng.random(60) * 42.0 - 2.0
    start = time.perf_counter()
    curve = froc(det, scores, gt_points, gate=1.0, volume_m3=100.0 * 100.0 * 42.0)
    froc_time = time.perf_counter() - start
    print(f"\nnms_peaks: {nms_time:.3f} s, froc: {froc_time:.3f} s")

    assert np.all(np.isfinite(peaks.positions))
    assert np.all(np.isfinite(curve.recall)) or curve.thresholds.shape[0] == 0

    raw_rng = _rng(29)
    raw_density = raw_rng.random(grid.shape)
    start = time.perf_counter()
    raw_peaks = nms_peaks(
        raw_density, grid, radius=2.0, min_value=float(np.median(raw_density)), refine=False
    )
    raw_time = time.perf_counter() - start
    print(f"nms_peaks raw noise: {raw_time:.3f} s ({raw_peaks.indices.shape[0]} peaks)")

    assert np.all(np.isfinite(raw_peaks.values))
