"""Tests for the E1 back-projection solvers (T11)."""

from __future__ import annotations

import functools
import os
import time
from typing import Any

import numpy as np
import pytest

from plateau_rt.domain.rf_camera.camera import look_at_orientation
from plateau_rt.domain.rf_tomography import metrics
from plateau_rt.domain.rf_tomography.backproject import backproject
from plateau_rt.domain.rf_tomography.forward_exact import atom_cfr, capture_factors
from plateau_rt.domain.rf_tomography.geometry import CaptureGeometry, VoxelGrid
from plateau_rt.domain.rf_tomography.kernels import (
    PowerOperator,
    noise_floor,
    power_backproject_grid,
)
from plateau_rt.domain.rf_tomography.observables import extract
from plateau_rt.domain.rf_tomography.solvers.bp import (
    COHERENT_NODES,
    DEFAULT_FLOOR,
    E1_NODES,
    PER_BIN_NODES,
    POWER_NODES,
    blind_tau_search,
    capture_weights,
    coherent_map,
    envelope_bp_fn,
    envelope_map,
    intensity_map,
    log_mean_fusion,
    node_data,
    power_column_norm,
    power_map,
    roi_grid,
    roi_refine,
    splat_returns,
)
from plateau_rt.domain.rf_tomography.sync import (
    TrackSeeds,
    apply_gauge,
    make_tracks,
    reference_power,
)
from plateau_rt.domain.rf_tomography.synthetic import l0a_point, offgrid_points

CENTER = np.array([0.0, 0.0, 5.0])
MICRO_VOXELS = ((0, 0, 1), (4, 4, 1))
SNR_DB = 30.0


def _micro_geometry(
    aperture: tuple[int, int] = (4, 4),
    num_bins: int = 16,
    bs_pos: tuple[tuple[float, float, float], ...] = ((3.0, -2.0, 40.0),),
) -> CaptureGeometry:
    """L0 micro capture (design §6.7): 4 ring views at r = 15 m, 4x4 aperture, 16 bins, 100 MHz."""
    pos, ori = [], []
    for i, height in enumerate((1.5, 12.0, 1.5, 12.0)):
        az = np.deg2rad(20.0 + 90.0 * i)
        p = (15.0 * np.cos(az), 15.0 * np.sin(az), height)
        pos.append(p)
        ori.append(look_at_orientation(p, tuple(CENTER)))
    return CaptureGeometry.from_orientations(
        np.array(pos),
        np.array(ori),
        np.array(bs_pos),
        f_c=3.5e9,
        bandwidth=100e6,
        num_bins=num_bins,
        aperture_shape=aperture,
        bs_look_at=CENTER,
    )


def _micro_grid() -> VoxelGrid:
    """The 5x5x3 CI micro grid at 2 m."""
    return VoxelGrid.from_bounds(CENTER - [4.0, 4.0, 2.0], CENTER + [4.0, 4.0, 2.0], 2.0)


def _tracks(y_clean, geom, seed):
    p_ref, _ = reference_power(y_clean, np.ones(y_clean.shape[:2], dtype=bool))
    return make_tracks(y_clean, SNR_DB, p_ref, TrackSeeds(seed), freq_offsets=geom.freq_offsets)


def _micro_scene(seed):
    """Two BV points <= 0.2 m off the centres of voxels MICRO_VOXELS, 30 dB, S and N tracks."""
    geom, grid = _micro_geometry(), _micro_grid()
    rng = np.random.default_rng(np.random.SeedSequence([seed, 7]))
    pts = np.stack(
        [
            offgrid_points(
                grid, 1, rng, offset_max=0.2, voxels=np.array([np.ravel_multi_index(v, grid.shape)])
            )[0][0]
            for v in MICRO_VOXELS
        ]
    )
    amps = np.array([1.0, 0.7]) * np.exp(2j * np.pi * rng.uniform(size=2))
    tracks, gt = _tracks(atom_cfr(pts, amps, geom, "bv"), geom, seed)
    return geom, grid, pts, tracks, gt


@functools.lru_cache(maxsize=None)
def _scene(seed):
    """Module-scoped cache of the micro scene (dicts with arrays are fine as return values)."""
    return _micro_scene(seed)


def _reldiff(a: np.ndarray, b: np.ndarray) -> float:
    """Return ``max|a - b| / max|b|``."""
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


# ---------------------------------------------------------------------------
# 1. Acceptance: ID-S and IDP-S localise both micro points within 1.2 m.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_id_and_idp_localise(seed: int) -> None:
    geom, grid, pts, tracks, gt = _scene(seed)
    y_s = tracks["ideal-S"]

    id_map = power_map(y_s, geom, grid, "bv", "ID", noise_var=gt["sigma2"])
    id_peaks = metrics.nms_peaks(id_map, grid, 3.0, min_value=0.0, max_peaks=2)
    assert metrics.match(id_peaks.positions, pts, 1.2).tp == 2

    idp = envelope_map(y_s, geom, grid.centers(), "bv", "IDP").reshape(grid.shape)
    idp_peaks = metrics.nms_peaks(idp, grid, 3.0, min_value=0.0, max_peaks=2)
    assert metrics.match(idp_peaks.positions, pts, 1.2).tp == 2


# ---------------------------------------------------------------------------
# 2. Acceptance: IDP-S ROI within 0.25 m.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_idp_roi(seed: int) -> None:
    geom, grid, pts, tracks, _ = _scene(seed)
    y_s = tracks["ideal-S"]
    idp = envelope_map(y_s, geom, grid.centers(), "bv", "IDP").reshape(grid.shape)
    peaks = metrics.nms_peaks(idp, grid, 3.0, min_value=0.0, max_peaks=2)

    result = roi_refine(y_s, geom, peaks.positions, "bv", "IDP")
    assert metrics.match(result.positions, pts, 0.25).tp == 2

    reference = coherent_map(y_s, geom, result.positions, "bv", "IDP")
    np.testing.assert_allclose(result.values, reference, rtol=1e-2)


# ---------------------------------------------------------------------------
# 3. Acceptance: blind tau within 1 ns at 30 dB.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1])
def test_blind_tau(seed: int) -> None:
    geom = _micro_geometry()
    phantom = l0a_point(seed, grid=_micro_grid(), geom=geom, offset_max=0.2)
    tracks, gt = _tracks(phantom.y_clean, geom, seed)
    y_n = tracks["ideal-N"]
    tau = gt["gauges"]["ideal-N"][1]
    assert np.abs(tau).max() < 25e-9

    search_grid = VoxelGrid.from_bounds(CENTER - [4, 4, 2], CENTER + [4, 4, 2], 1.0)
    estimate = blind_tau_search(
        envelope_bp_fn("bv", "IDP", kind="tricubic"),
        y_n,
        geom,
        search_grid,
        (-30e-9, 30e-9),
    )
    assert np.abs(estimate - tau).max() < 1e-9


# ---------------------------------------------------------------------------
# 4. N-mode sums with tau_hat.
# ---------------------------------------------------------------------------


def _gauge_arrays(seed: int, geom: CaptureGeometry) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(np.random.SeedSequence([seed, 12345]))
    phi = rng.uniform(0.0, 2.0 * np.pi, size=(geom.num_views, geom.num_bs))
    tau = rng.integers(-5, 6, size=(geom.num_views, geom.num_bs)) / geom.bandwidth
    if not np.any(tau != 0.0):
        tau[0, 0] = 1.0 / geom.bandwidth
    return phi, tau


def test_n_mode_tau_hat() -> None:
    geom, grid, _, tracks, gt = _scene(3)
    y_s = tracks["ideal-S"]
    sigma2 = gt["sigma2"]
    points = grid.centers()
    phi, tau = _gauge_arrays(3, geom)
    y_n = apply_gauge(y_s, phi, tau, geom.freq_offsets)

    for node in ("IDP", "DP"):
        kwargs = {"noise_var": sigma2} if node == "DP" else {}
        fixed = envelope_map(y_n, geom, points, "bv", node, tau_hat=tau, **kwargs)
        clean = envelope_map(y_s, geom, points, "bv", node, **kwargs)
        assert _reldiff(fixed, clean) <= 1e-10
        broken = envelope_map(y_n, geom, points, "bv", node, **kwargs)
        assert _reldiff(broken, clean) > 1e-2

    fixed_power = power_map(y_n, geom, grid, "bv", "ID", noise_var=sigma2, tau_hat=tau)
    clean_power = power_map(y_s, geom, grid, "bv", "ID", noise_var=sigma2)
    assert _reldiff(fixed_power, clean_power) <= 1e-10
    broken_power = power_map(y_n, geom, grid, "bv", "ID", noise_var=sigma2)
    assert _reldiff(broken_power, clean_power) > 1e-2

    fixed_splat = splat_returns(y_n, geom, grid, "bv", tau_hat=tau)
    clean_splat = splat_returns(y_s, geom, grid, "bv")
    assert _reldiff(fixed_splat, clean_splat) <= 1e-10
    broken_splat = splat_returns(y_n, geom, grid, "bv")
    assert _reldiff(broken_splat, clean_splat) > 1e-2

    y_t = apply_gauge(y_s, np.zeros_like(phi), tau, geom.freq_offsets)
    fixed_coherent = coherent_map(y_t, geom, points, "bv", "IDP", tau_hat=tau)
    clean_coherent = coherent_map(y_s, geom, points, "bv", "IDP")
    assert _reldiff(fixed_coherent, clean_coherent) <= 1e-10


# ---------------------------------------------------------------------------
# 5. Sync-invariant nodes (S == N by construction).
# ---------------------------------------------------------------------------


def test_sync_invariant_nodes() -> None:
    geom, grid, _, tracks, gt = _scene(3)
    y_s = tracks["ideal-S"]
    sigma2 = gt["sigma2"]
    rng = np.random.default_rng(np.random.SeedSequence([3, 999]))
    phi = rng.uniform(0.0, 2.0 * np.pi, size=(geom.num_views, geom.num_bs))
    tau = rng.normal(0.0, 10e-9, size=(geom.num_views, geom.num_bs))
    y_n = apply_gauge(y_s, phi, tau, geom.freq_offsets)

    for node in ("I", "I_n0"):
        diff = intensity_map(y_n, geom, grid, "bv", node, noise_var=sigma2) - intensity_map(
            y_s, geom, grid, "bv", node, noise_var=sigma2
        )
        assert np.max(np.abs(diff)) <= 1e-10 * np.max(
            np.abs(intensity_map(y_s, geom, grid, "bv", node, noise_var=sigma2))
        )

    points = grid.centers()
    for node in ("P", "IP", "P_W", "IP_W"):
        clean = envelope_map(y_s, geom, points, "bv", node, noise_var=sigma2)
        gauged = envelope_map(y_n, geom, points, "bv", node, noise_var=sigma2)
        assert _reldiff(gauged, clean) <= 1e-10


# ---------------------------------------------------------------------------
# 6. Per-bin envelope equals the per-bin back-projection.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("node", ["IP_W", "P_W"])
def test_per_bin_envelope(node: str) -> None:
    geom, grid, _, tracks, gt = _scene(0)
    y_s = tracks["ideal-S"]
    points = grid.centers()
    weight = capture_weights(geom, points, "bv")

    got = envelope_map(y_s, geom, points, "bv", node, noise_var=gt["sigma2"], per_capture=True)
    data = node_data(y_s, node, noise_var=gt["sigma2"])
    expected = np.zeros_like(got)
    for n in range(geom.num_bins):
        masked = np.zeros_like(data)
        masked[..., n] = data[..., n]
        projected = backproject(masked, geom, points, "bv")
        expected += (np.abs(projected) ** 2).sum(axis=2)
    expected = np.where(weight > 0.0, expected / np.where(weight > 0.0, weight, 1.0), 0.0)
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# 7. Closed-form column norms.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("node", ["ID", "I", "I_n0", "ID_omni", "I_omni"])
def test_power_column_norm(node: str) -> None:
    geom, grid, _, _, _ = _scene(0)
    points = grid.centers()[:10]
    seeds = {"ID": 1, "I": 2, "I_n0": 3, "ID_omni": 4, "I_omni": 5}
    rng = np.random.default_rng(np.random.SeedSequence([0, seeds[node]]))
    tau_hat = rng.normal(0.0, 5e-9, size=(geom.num_views, geom.num_bs))
    got = power_column_norm(geom, points, "bv", node, tau_hat=tau_hat)

    operator = PowerOperator(points, geom, "bv", node, tau=tau_hat)
    expected = np.zeros(got.shape, dtype=np.float64)
    for j in range(points.shape[0]):
        column = operator.forward(np.eye(points.shape[0])[j]).reshape(
            geom.num_views, geom.num_bs, -1
        )
        expected[:, :, j] = np.linalg.norm(column, axis=-1)
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-13)


# ---------------------------------------------------------------------------
# 7b. Map definitions: GLRT normalisations match the direct expressions.
# ---------------------------------------------------------------------------


def test_map_definitions() -> None:
    geom, grid, _, tracks, gt = _scene(0)
    y_s = tracks["ideal-S"]
    sigma2 = gt["sigma2"]
    points = grid.centers()
    w = capture_weights(geom, points, "bv")

    for node in ("IDP", "DP", "IP", "P"):
        bp = backproject(node_data(y_s, node, noise_var=sigma2), geom, points, "bv").sum(axis=2)
        got_env = envelope_map(y_s, geom, points, "bv", node, noise_var=sigma2, per_capture=True)
        expected_env = np.where(w > 0.0, np.abs(bp) ** 2 / np.where(w > 0.0, w, 1.0), 0.0)
        np.testing.assert_allclose(got_env, expected_env, rtol=1e-12, atol=0)

        got_coh = coherent_map(y_s, geom, points, "bv", node, noise_var=sigma2)
        expected_coh = np.abs(bp.sum(axis=(0, 1))) ** 2 / w.sum(axis=(0, 1))
        np.testing.assert_allclose(got_coh, expected_coh, rtol=1e-12)

    expected_power = power_backproject_grid(
        extract(y_s, "ID").data - noise_floor("ID", geom, sigma2),
        geom,
        grid,
        space="bv",
        product="ID",
    ) / np.sqrt((power_column_norm(geom, points, "bv", "ID") ** 2).sum((0, 1))).reshape(grid.shape)
    got_power = power_map(y_s, geom, grid, "bv", "ID", noise_var=sigma2)
    np.testing.assert_allclose(got_power, expected_power.reshape(grid.shape), rtol=1e-12)

    expected_intensity = log_mean_fusion(
        power_map(y_s, geom, grid, "bv", "I", noise_var=sigma2, per_capture=True)
    )
    got_intensity = intensity_map(y_s, geom, grid, "bv", "I", noise_var=sigma2)
    np.testing.assert_array_equal(got_intensity, expected_intensity)


@pytest.mark.parametrize(("node", "name"), [("ID_omni", "ID-o"), ("I_omni", "I-o")])
def test_omni_power_map_definition(node: str, name: str) -> None:
    geom, grid, _, tracks, gt = _scene(0)
    y = tracks["ideal-S"]
    sigma2 = gt["sigma2"]
    centers = grid.centers()
    got = power_map(y, geom, grid, "bv", node, noise_var=sigma2)
    adjoint = power_backproject_grid(
        extract(y, name).data - noise_floor(node, geom, sigma2),
        geom,
        grid,
        space="bv",
        product=node,
    )
    norm = power_column_norm(geom, centers, "bv", node)
    denominator = np.sqrt(np.sum(norm**2, axis=(0, 1))).reshape(grid.shape)
    expected = adjoint / denominator
    np.testing.assert_allclose(got, expected, rtol=1e-12)


def test_omni_glrt_removes_bias() -> None:
    geom = _micro_geometry()
    rng = np.random.default_rng(np.random.SeedSequence([16, 5]))
    points = CENTER + rng.uniform(-3.0, 3.0, (8, 3)) * np.array([1.0, 1.0, 0.5])
    for node, name in (("ID_omni", "ID-o"), ("I_omni", "I-o")):
        errors = []
        raw_errors = []
        for point in points:
            y = atom_cfr(point[None], np.ones(1), geom, "bv")
            grid = VoxelGrid.from_bounds(np.round(point) - 2.0, np.round(point) + 2.0, 0.25)
            normalised = power_map(y, geom, grid, "bv", node)
            best = grid.centers()[int(np.argmax(normalised))]
            errors.append(float(np.linalg.norm(best - point)))
            raw = power_backproject_grid(
                extract(y, name).data, geom, grid, space="bv", product=node
            )
            raw_best = grid.centers()[int(np.argmax(raw))]
            raw_errors.append(float(np.linalg.norm(raw_best - point)))
        assert max(errors) <= (0.6 if node == "ID_omni" else 0.4)
        assert max(raw_errors) > 1.0


# ---------------------------------------------------------------------------
# 7c. VS path-length gradient in the ROI polishing.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 2, 5])
def test_vs_space(seed: int) -> None:
    geom = _micro_geometry()
    grid = _micro_grid()
    rng = np.random.default_rng(seed)
    x0 = grid.centers()[rng.integers(grid.size)] + rng.uniform(-0.2, 0.2, 3)
    y = atom_cfr(x0[None], np.ones(1), geom, "vs")
    ys = _tracks(y, geom, seed)[0]["ideal-S"]

    envelope = envelope_map(ys, geom, grid.centers(), "vs", "IDP").reshape(grid.shape)
    peaks = metrics.nms_peaks(envelope, grid, 3.0, min_value=0.0, max_peaks=1)
    assert peaks.positions.shape[0] == 1
    assert np.linalg.norm(peaks.positions[0] - x0) <= 1.2

    result = roi_refine(ys, geom, peaks.positions, "vs", "IDP")
    assert np.linalg.norm(result.positions[0] - x0) <= 0.4
    reference = coherent_map(ys, geom, result.positions, "vs", "IDP")
    np.testing.assert_allclose(result.values, reference, rtol=1e-2)


# ---------------------------------------------------------------------------
# 7d. Two-BS phase extrapolation reaches lambda scale.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_roi_two_bs_lambda_scale(seed: int) -> None:
    # One BS hides the common BS term in |sum_c ...|; two BSs are needed to expose it.
    geom = _micro_geometry(bs_pos=((3.0, -2.0, 40.0), (-30.0, 20.0, 25.0)))
    grid = _micro_grid()
    rng = np.random.default_rng(seed)
    x0 = grid.centers()[rng.integers(grid.size)] + rng.uniform(-0.2, 0.2, 3)
    y = atom_cfr(x0[None], np.ones(1), geom, "bv")
    result = roi_refine(y, geom, (x0 + np.array([0.10, -0.08, 0.06]))[None], "bv", "IDP")
    assert np.linalg.norm(result.positions[0] - x0) <= 0.02
    reference = coherent_map(y, geom, result.positions, "bv", "IDP")
    np.testing.assert_allclose(result.values, reference, rtol=2e-3)


# ---------------------------------------------------------------------------
# 8. GLRT normalisation removes the picket-fence bias.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_glrt_picket_fence(seed: int) -> None:
    geom = _micro_geometry()
    rng = np.random.default_rng(seed)
    x0 = CENTER + rng.uniform(-3.0, 3.0, 3) * np.array([1.0, 1.0, 0.5])
    y = atom_cfr(x0[None], np.ones(1), geom, "bv")
    fine = VoxelGrid(origin=x0 - 1.5 + rng.uniform(0.0, 0.25, 3), spacing=0.25, shape=(13, 13, 13))

    normalised = power_map(y, geom, fine, "bv", "ID", kind="tricubic")
    best = fine.centers()[int(np.argmax(normalised))]
    assert np.linalg.norm(best - x0) <= 0.4

    power_volume = extract(y, "ID").data
    raw = power_backproject_grid(
        power_volume, geom, fine, space="bv", product="ID", kind="tricubic"
    )
    raw_best = fine.centers()[int(np.argmax(raw))]
    assert np.linalg.norm(raw_best - x0) > 0.4


# ---------------------------------------------------------------------------
# 9. Intensity cone BP + log-mean fusion.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1])
def test_intensity(seed: int) -> None:
    geom = _micro_geometry()
    grid = _micro_grid()
    phantom = l0a_point(seed, grid=grid, geom=geom, offset_max=0.2)
    tracks, gt = _tracks(phantom.y_clean, geom, seed)
    density = intensity_map(tracks["ideal-S"], geom, grid, "bv", "I", noise_var=gt["sigma2"])
    peaks = metrics.nms_peaks(density, grid, 3.0, min_value=0.0, max_peaks=1)
    assert peaks.positions.shape[0] == 1
    error = np.linalg.norm(peaks.positions[0] - phantom.gt.points_pos[0])
    assert error <= 1.2


# ---------------------------------------------------------------------------
# 10. D splatting.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_splatting(seed: int) -> None:
    geom, grid, pts, tracks, _ = _scene(seed)
    density = splat_returns(tracks["ideal-S"], geom, grid, "bv")
    peaks = metrics.nms_peaks(density, grid, 3.0, min_value=density.min(), max_peaks=2)
    assert metrics.match(peaks.positions, pts, 1.2).tp == 2


# ---------------------------------------------------------------------------
# 11. Coherent GLRT is maximal at the true point.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1])
def test_coherent_max_at_point(seed: int) -> None:
    geom = _micro_geometry()
    phantom = l0a_point(seed, grid=_micro_grid(), geom=geom, offset_max=0.2)
    x0 = phantom.gt.points_pos[0]
    grid = roi_grid(x0 + np.array([0.12, -0.10, 0.08]), geom.wavelength)
    values = coherent_map(phantom.y_clean, geom, grid.centers(), "bv", "IDP", kind="tricubic")
    at_point = coherent_map(phantom.y_clean, geom, x0[None], "bv", "IDP", kind="tricubic")[0]
    assert at_point >= values.max() * (1.0 - 1e-9)


# ---------------------------------------------------------------------------
# 12. log_mean_fusion by hand.
# ---------------------------------------------------------------------------


def test_log_mean_fusion_hand() -> None:
    floor = DEFAULT_FLOOR
    maps = np.array([[[1.0, 0.5, 0.0]], [[2.0, 4.0, -1.0]]]).reshape(2, 1, 3)
    out = log_mean_fusion(maps)
    expected = np.array(
        [
            np.exp(0.5 * (np.log(1.0 + floor) + np.log(0.5 + floor))),
            np.exp(0.5 * (np.log(0.5 + floor) + np.log(1.0 + floor))),
            floor,
        ]
    )
    np.testing.assert_allclose(out, expected, rtol=1e-12)

    dropped = log_mean_fusion(np.array([[[1.0]], [[-1.0]]]).reshape(2, 1, 1))
    np.testing.assert_allclose(dropped, np.array([1.0 + floor]), rtol=1e-12)

    empty = log_mean_fusion(np.zeros((2, 1, 4)))
    np.testing.assert_array_equal(empty, np.zeros(4))


# ---------------------------------------------------------------------------
# 13. Small semantics.
# ---------------------------------------------------------------------------


def test_semantics() -> None:
    geom, grid, _, tracks, gt = _scene(0)
    y_s = tracks["ideal-S"]
    sigma2 = gt["sigma2"]

    points = np.vstack([grid.centers()[:5], geom.ue_pos[0][None]])
    weight = capture_weights(geom, points, "bv")
    factors = capture_factors(points[:-1], geom, "bv", 0, 0)
    np.testing.assert_allclose(weight[0, 0, :-1], np.abs(factors.gamma) ** 2, rtol=1e-13)
    assert np.all(weight[:, :, -1] == 0.0)

    n0 = geom.num_bins // 2
    for node in ("IP", "P"):
        data = node_data(y_s, node, noise_var=sigma2)
        assert data.shape == y_s.shape
        assert np.all(data[..., :n0] == 0.0)
        assert np.all(data[..., n0 + 1 :] == 0.0)
        assert np.any(data[..., n0] != 0.0)

    roi = roi_grid(CENTER, geom.wavelength)
    assert roi.spacing == geom.wavelength / 4.0
    assert roi.shape == (24, 24, 24)
    np.testing.assert_allclose(roi.centers().mean(axis=0), CENTER, rtol=1e-12, atol=1e-12)

    singular_grid = VoxelGrid(origin=geom.ue_pos[0], spacing=0.5, shape=(3, 3, 3))
    power = power_map(y_s, geom, singular_grid, "bv", "ID", noise_var=sigma2)
    assert power[0, 0, 0] == 0.0
    assert np.all(np.isfinite(power))
    splat = splat_returns(y_s, geom, singular_grid, "bv")
    assert splat[0, 0, 0] == 0.0
    assert np.all(np.isfinite(splat))
    envelope = envelope_map(y_s, geom, singular_grid.centers(), "bv", "IDP")
    assert envelope[0] == 0.0
    assert np.all(np.isfinite(envelope[1:]))

    bp_fn = envelope_bp_fn("bv", "IDP")
    output = bp_fn(y_s, geom, grid.centers(), np.zeros((geom.num_views, geom.num_bs)))
    assert output.shape == (geom.num_views, geom.num_bs, grid.size)
    assert np.all(np.isfinite(output))


# ---------------------------------------------------------------------------
# 14. Validation.
# ---------------------------------------------------------------------------


def test_validation() -> None:
    geom, grid, _, tracks, _ = _scene(0)
    y_s = tracks["ideal-S"]
    points = grid.centers()

    with pytest.raises(ValueError):
        power_map(y_s, geom, grid, "xx", "ID")
    with pytest.raises(ValueError):
        envelope_map(y_s, geom, points, "xx", "IDP")
    with pytest.raises(ValueError):
        coherent_map(y_s, geom, points, "bv", "IP_W")
    with pytest.raises(ValueError):
        power_map(y_s, geom, grid, "bv", "IDP")
    with pytest.raises(ValueError):
        intensity_map(y_s, geom, grid, "bv", "ID")
    with pytest.raises(ValueError):
        node_data(y_s, "I")
    with pytest.raises(ValueError):
        node_data(y_s, "D")
    with pytest.raises(ValueError):
        power_column_norm(geom, points, "bv", "IDP")
    with pytest.raises(ValueError):
        roi_refine(y_s, geom, points[:1], "bv", "I")
    with pytest.raises(ValueError):
        envelope_bp_fn("xx")
    with pytest.raises(ValueError):
        envelope_bp_fn("bv", "I")

    with pytest.raises(ValueError):
        power_map(y_s, geom, grid, "bv", "ID", noise_var=-1.0)
    with pytest.raises(ValueError):
        power_map(y_s, geom, grid, "bv", "ID", tau_hat=np.zeros((2, 2)))
    with pytest.raises(ValueError):
        power_map(y_s, geom, grid, "bv", "ID", tau_hat=np.full((4, 1), np.nan))
    with pytest.raises(ValueError):
        power_map(y_s, geom, np.zeros((2, 2, 2)), "bv", "ID")
    with pytest.raises(ValueError):
        power_map(y_s[:, :, :, :, :, :-1], geom, grid, "bv", "ID")

    with pytest.raises(ValueError):
        log_mean_fusion(np.zeros((2, 1, 3)), floor=0.0)
    with pytest.raises(ValueError):
        log_mean_fusion(np.zeros(5))
    with pytest.raises(ValueError):
        intensity_map(y_s, geom, grid, "bv", "I", floor=-1.0)

    with pytest.raises(ValueError):
        splat_returns(y_s, geom, grid, "bv", p_hit=0.4, p_false=0.5)
    with pytest.raises(ValueError):
        splat_returns(y_s, geom, grid, "bv", p_hit=0.5, p_false=0.5)
    with pytest.raises(ValueError):
        splat_returns(y_s, geom, grid, "bv", sigma_delay=0.0)

    with pytest.raises(ValueError):
        roi_refine(y_s, geom, points[:1], "bv", "IDP", candidate_fraction=1.5)
    with pytest.raises(ValueError):
        roi_refine(y_s, geom, points[:1], "bv", "IDP", polish_steps=0)

    with pytest.raises(ValueError):
        roi_grid(CENTER, geom.wavelength, half_width=0.0)
    with pytest.raises(ValueError):
        roi_grid(CENTER, -1.0)
    with pytest.raises(ValueError):
        roi_grid(np.zeros(2), geom.wavelength)

    bp_fn = envelope_bp_fn("bv", "IDP")
    with pytest.raises(ValueError):
        blind_tau_search(bp_fn, y_s, geom, grid, (0.0, -1e-9))
    with pytest.raises(ValueError):
        blind_tau_search(bp_fn, y_s, geom, grid, (-1e-6, 1e-6))
    with pytest.raises(ValueError):
        blind_tau_search(bp_fn, y_s, geom, grid, (-30e-9, 30e-9), step=-1.0)
    with pytest.raises(ValueError):
        blind_tau_search(bp_fn, y_s, geom, grid, (-30e-9, 30e-9), max_points=0)
    with pytest.raises(ValueError):
        blind_tau_search(bp_fn, y_s, geom, grid, (-30e-9, 30e-9), rel=1.0)
    with pytest.raises(ValueError):
        blind_tau_search(bp_fn, y_s, geom, grid, (-30e-9, 30e-9), fine=0)
    with pytest.raises(ValueError):
        blind_tau_search(bp_fn, y_s, geom, grid, (-30e-9, 30e-9), levels=-1)

    def wrong_shape(*_args: Any) -> np.ndarray:
        return np.zeros((1, 1, 1))

    with pytest.raises(ValueError):
        blind_tau_search(wrong_shape, y_s, geom, grid, (-30e-9, 30e-9))
    not_callable: Any = "not callable"
    with pytest.raises(ValueError):
        blind_tau_search(not_callable, y_s, geom, grid, (-30e-9, 30e-9))

    assert COHERENT_NODES == ("P", "DP", "IP", "IDP")
    assert PER_BIN_NODES == ("P_W", "IP_W")
    assert POWER_NODES == ("I", "I_n0", "ID")
    assert set(E1_NODES) == set(COHERENT_NODES) | set(PER_BIN_NODES) | {"I", "I_n0", "ID", "D"}


# ---------------------------------------------------------------------------
# 15. Benchmark (skipped unless RF_TOMO_BENCH=1).
# ---------------------------------------------------------------------------


def _bench_geometry() -> CaptureGeometry:
    return _micro_geometry((8, 8), num_bins=128)


@pytest.mark.skipif(os.environ.get("RF_TOMO_BENCH") != "1", reason="set RF_TOMO_BENCH=1")
def test_benchmark() -> None:
    geom = _bench_geometry()
    grid = VoxelGrid.from_bounds(CENTER - [10.0, 10.0, 5.0], CENTER + [10.0, 10.0, 5.0], 1.0)
    y = atom_cfr(CENTER[None], np.ones(1), geom, "bv")

    start = time.perf_counter()
    power = power_map(y, geom, grid, "bv", "ID", noise_var=1e-9)
    t_power = time.perf_counter() - start

    start = time.perf_counter()
    envelope = envelope_map(y, geom, grid.centers(), "bv", "IDP")
    t_envelope = time.perf_counter() - start

    start = time.perf_counter()
    splat = splat_returns(y, geom, grid, "bv")
    t_splat = time.perf_counter() - start

    peaks = metrics.nms_peaks(power, grid, 2.0, min_value=0.0, max_peaks=1)
    start = time.perf_counter()
    result = roi_refine(y, geom, peaks.positions, "bv", "IDP")
    t_roi = time.perf_counter() - start

    print(
        f"power_map {t_power:.3f}s envelope_map {t_envelope:.3f}s "
        f"splat_returns {t_splat:.3f}s roi_refine {t_roi:.3f}s"
    )
    del envelope, splat, result
    assert np.all(np.isfinite(power))
    assert np.isfinite(t_power + t_envelope + t_splat + t_roi)
