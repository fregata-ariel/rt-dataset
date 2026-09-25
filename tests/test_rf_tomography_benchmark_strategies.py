"""Strategy-level tests for the tomography benchmark (T16, §6.3)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from test_rf_tomography_benchmark import BS, CENTER, write_micro_dataset

from plateau_rt.application import rf_tomography_benchmark as bench
from plateau_rt.application import rf_tomography_io as tio
from plateau_rt.domain.rf_camera.camera import look_at_orientation
from plateau_rt.domain.rf_tomography import metrics as metric_mod
from plateau_rt.domain.rf_tomography import sync as sync_mod
from plateau_rt.domain.rf_tomography import synthetic
from plateau_rt.domain.rf_tomography.configs import get_config
from plateau_rt.domain.rf_tomography.forward_exact import atom_cfr
from plateau_rt.domain.rf_tomography.geometry import CaptureGeometry
from plateau_rt.domain.rf_tomography.synthetic import offgrid_points
from plateau_rt.domain.rf_tomography.views import nested_view_order

HEIGHTS = (1.5, 12.0, 1.5, 12.0)
REF = (2, 0)


def micro_geometry(num_views: int = 4) -> CaptureGeometry:
    """Build the §6 micro geometry directly (no dataset)."""
    positions = []
    orientations = []
    for index in range(num_views):
        azimuth = np.deg2rad(20 + 90 * index)
        position = (
            float(15.0 * np.cos(azimuth)),
            float(15.0 * np.sin(azimuth)),
            float(HEIGHTS[index % 4]),
        )
        positions.append(position)
        orientations.append(look_at_orientation(position, CENTER))
    return CaptureGeometry.from_orientations(
        ue_pos=positions,
        ue_orientations=orientations,
        bs_pos=[BS],
        f_c=3.5e9,
        bandwidth=100e6,
        num_bins=16,
        aperture_shape=(4, 4),
        bs_look_at=np.asarray(CENTER, dtype=np.float64),
    )


def two_point_scene(
    geom: CaptureGeometry, grid: Any, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Render the two-point micro scene and return ``(y, points, amplitudes)``."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, 7]))
    found = []
    for voxel in ((0, 0, 1), (4, 4, 1)):
        flat = np.array([np.ravel_multi_index(voxel, grid.shape)])
        points, _ = offgrid_points(grid, 1, rng, offset_max=0.2, voxels=flat)
        found.append(points[0])
    pts = np.stack(found).astype(np.float64)
    amps = (np.array([1.0, 0.7]) * np.exp(2j * np.pi * rng.uniform(size=2))).astype(np.complex128)
    return (
        np.asarray(atom_cfr(pts, amps, geom, "bv"), dtype=np.complex128),
        pts,
        amps,
    )


def n_track(
    y: np.ndarray, geom: CaptureGeometry, seed: int
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray], float]:
    """Return the ideal-N track, its true gauges and the noise variance."""
    num_views, num_bs = geom.num_views, geom.num_bs
    p_ref, _ = sync_mod.reference_power(y, np.ones((num_views, num_bs), dtype=bool))
    tracks, gt = sync_mod.make_tracks(
        y,
        30.0,
        float(p_ref),
        sync_mod.TrackSeeds(0, seed),
        freq_offsets=geom.freq_offsets,
        ref=REF,
    )
    return tracks["ideal-N"], gt["gauges"]["ideal-N"], float(gt["sigma2"])


def gauge_deg_ns(
    estimate: bench.GaugeEstimate, truth: tuple[np.ndarray, np.ndarray], period: float
) -> tuple[float, float]:
    """Return ``(max phase deg, max delay ns)`` of ``estimate`` against ``truth``."""
    errors = metric_mod.gauge_errors((estimate.phi, estimate.tau), truth, period=period)
    phase = float(np.max(np.abs(errors.phase))) * 180.0 / np.pi
    delay = float(np.max(np.abs(errors.delay))) * 1e9
    return phase, delay


def test_los_anchor() -> None:
    """The LoS anchor recovers N-track gauges to ≤1° and ≤0.05 ns."""
    geom = micro_geometry()
    grid = bench.make_grid(CENTER, (4.0, 4.0, 2.0), 2.0)
    assert int(nested_view_order(4, 0)[0]) == REF[0]
    for seed in (0, 1):
        phantom = synthetic.l0e_image_method(seed, geom=geom, walls=())
        y_n, truth, sigma2 = n_track(np.asarray(phantom.y_clean), geom, seed)
        estimate = bench.estimate_gauges(
            get_config("IDP-N"),
            "los",
            y_n,
            geom,
            grid,
            "bv",
            noise_var=sigma2,
            ref=REF,
            n_iter=10,
            sigma_t=10e-9,
        )
        assert estimate.estimates == ("phi", "tau")
        phase, delay = gauge_deg_ns(estimate, truth, geom.delay_period)
        assert phase <= 1.0
        assert delay <= 0.05
        tau_only = bench.estimate_gauges(
            get_config("ID-N"),
            "los",
            y_n,
            geom,
            grid,
            "bv",
            noise_var=sigma2,
            ref=REF,
            n_iter=10,
            sigma_t=10e-9,
        )
        assert tau_only.estimates == ("tau",)
        assert bool(np.all(tau_only.phi == 0.0))


def test_varpro_and_self_cal() -> None:
    """VarPro and self-calibration recover gauges on the true support."""
    geom = micro_geometry()
    grid = bench.make_grid(CENTER, (4.0, 4.0, 2.0), 2.0)
    for seed in (0, 1):
        y, pts, _ = two_point_scene(geom, grid, seed)
        y_n, truth, sigma2 = n_track(y, geom, seed)
        for strategy in ("self_cal", "varpro"):
            estimate = bench.estimate_gauges(
                get_config("IDP-N"),
                strategy,
                y_n,
                geom,
                grid,
                "bv",
                noise_var=sigma2,
                ref=REF,
                n_iter=10,
                sigma_t=10e-9,
                points=pts,
            )
            assert estimate.estimates == ("phi", "tau")
            phase, delay = gauge_deg_ns(estimate, truth, geom.delay_period)
            assert phase <= 1.0, (strategy, seed)
            assert delay <= 0.05, (strategy, seed)
            assert estimate.phi[REF] == 0.0


def test_blind() -> None:
    """The blind delay search recovers sub-nanosecond delays."""
    geom = micro_geometry()
    phantom = synthetic.l0a_point(
        0, grid=bench.make_grid(CENTER, (4, 4, 2), 2.0), geom=geom, offset_max=0.2
    )
    y_n, truth, sigma2 = n_track(np.asarray(phantom.y_clean), geom, 0)
    search = bench.make_grid(CENTER, (4.0, 4.0, 2.0), 1.0)
    estimate = bench.estimate_gauges(
        get_config("IDP-N"),
        "blind",
        y_n,
        geom,
        search,
        "bv",
        noise_var=sigma2,
        ref=REF,
        n_iter=10,
        sigma_t=10e-9,
    )
    assert estimate.estimates == ("tau",)
    assert bool(np.all(estimate.phi == 0.0))
    _, delay = gauge_deg_ns(estimate, truth, geom.delay_period)
    assert delay < 1.0


def test_xcorr_contract() -> None:
    """The xcorr strategy is deterministic with finite delays and zero phases."""
    geom = micro_geometry()
    grid = bench.make_grid(CENTER, (4.0, 4.0, 2.0), 2.0)
    y, _, _ = two_point_scene(geom, grid, 0)
    y_n, _, sigma2 = n_track(y, geom, 0)
    first = bench.estimate_gauges(
        get_config("ID-N"),
        "xcorr",
        y_n,
        geom,
        grid,
        "bv",
        noise_var=sigma2,
        ref=REF,
        n_iter=10,
        sigma_t=10e-9,
    )
    second = bench.estimate_gauges(
        get_config("ID-N"),
        "xcorr",
        y_n,
        geom,
        grid,
        "bv",
        noise_var=sigma2,
        ref=REF,
        n_iter=10,
        sigma_t=10e-9,
    )
    assert first.estimates == ("tau",)
    assert first.tau.shape == (geom.num_views, geom.num_bs)
    assert bool(np.all(np.isfinite(first.tau)))
    assert bool(np.all(first.phi == 0.0))
    np.testing.assert_array_equal(first.tau, second.tau)


def test_estimate_errors() -> None:
    """Unknown strategies raise and ``used_gauges`` masks unestimated unknowns."""
    geom = micro_geometry()
    grid = bench.make_grid(CENTER, (4.0, 4.0, 2.0), 2.0)
    y, _, _ = two_point_scene(geom, grid, 0)
    y_n, _, sigma2 = n_track(y, geom, 0)
    cfg = get_config("IDP-N")
    with pytest.raises(ValueError):
        bench.estimate_gauges(
            cfg,
            "oracle",
            y_n,
            geom,
            grid,
            "bv",
            noise_var=sigma2,
            ref=REF,
            n_iter=10,
            sigma_t=10e-9,
        )
    with pytest.raises(ValueError):
        bench.estimate_gauges(
            cfg, "nope", y_n, geom, grid, "bv", noise_var=sigma2, ref=REF, n_iter=10, sigma_t=10e-9
        )
    estimate = bench.GaugeEstimate(
        phi=np.ones((geom.num_views, geom.num_bs)),
        tau=np.ones((geom.num_views, geom.num_bs)),
        estimates=("phi", "tau"),
        info={},
    )
    used_phi, used_tau = bench.used_gauges(get_config("IDP-S_tau"), estimate)
    assert bool(np.all(used_tau == 0.0))
    assert bool(np.all(used_phi == 1.0))
    used_phi, used_tau = bench.used_gauges(get_config("ID-N"), estimate)
    assert bool(np.all(used_phi == 0.0))
    assert bool(np.all(used_tau == 1.0))


def test_ill_posed_at() -> None:
    """``ill_posed_at`` flags single-view N, omni intensity and empty supports."""
    geom = micro_geometry()
    grid = bench.make_grid(CENTER, (4.0, 4.0, 2.0), 2.0)
    y, pts, _ = two_point_scene(geom, grid, 0)
    p_ref, _ = sync_mod.reference_power(y, np.ones((4, 1), dtype=bool))
    tracks, gt = sync_mod.make_tracks(
        y,
        30.0,
        float(p_ref),
        sync_mod.TrackSeeds(0, 0),
        freq_offsets=geom.freq_offsets,
        ref=REF,
    )
    sigma2 = float(gt["sigma2"])
    report = bench.ill_posed_at(
        get_config("IDP-S"), tracks["ideal-S"], geom, pts, "bv", noise_var=sigma2
    )
    assert report.flag is False
    assert report.model == "complex"
    assert report.components == ("re", "im")
    one_view = geom.select([0])
    flagged = bench.ill_posed_at(
        get_config("IDP-N"),
        tracks["ideal-S"][:1],
        one_view,
        pts,
        "bv",
        noise_var=sigma2,
    )
    assert flagged.flag is True
    omni = bench.ill_posed_at(
        get_config("I-o"), tracks["ideal-S"], geom, pts, "bv", noise_var=sigma2
    )
    assert omni.flag is True
    assert omni.cond == float("inf")
    empty = bench.ill_posed_at(
        get_config("IDP-S"),
        tracks["ideal-S"],
        geom,
        np.zeros((0, 3)),
        "bv",
        noise_var=sigma2,
    )
    assert empty.nuisance == ("amplitude",)
    assert empty.points.shape == (0, 3)
    assert empty.rho.shape == (geom.num_views, geom.num_bs, 0)
    assert empty.crb_std.shape == (0,)
    assert empty.cond == float("inf")
    assert empty.flag is True
    rho = np.full((geom.num_views, geom.num_bs, 1), 100.0)
    spread = bench.list_sigmas(rho, geom)
    rows, cols = geom.aperture_shape
    lam = geom.wavelength
    offsets = np.asarray(geom.elem_offsets, dtype=np.float64)
    d_col = float(np.linalg.norm(offsets[1] - offsets[0])) / lam
    d_row = float(np.linalg.norm(offsets[cols] - offsets[0])) / lam
    delta_f = float(geom.delta_f)
    expected_t = 1e9 * np.sqrt(6.0 / (100.0 * (16**2 - 1))) / (2.0 * np.pi * delta_f)
    expected_y = np.sqrt(6.0 / (100.0 * (cols**2 - 1))) / (2.0 * np.pi * d_col)
    expected_z = np.sqrt(6.0 / (100.0 * (rows**2 - 1))) / (2.0 * np.pi * d_row)
    np.testing.assert_allclose(spread[..., 0], expected_y, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(spread[..., 1], expected_z, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(spread[..., 2], expected_t, rtol=1e-12, atol=0.0)


@pytest.fixture(scope="module")
def strategy_run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Run five N-type configs with every strategy (bv), and IDP-S in vs space."""
    root, pts, amps = write_micro_dataset(tmp_path_factory.mktemp("strat"), seed=0)
    run = bench.run_benchmark(
        root,
        tmp_path_factory.mktemp("strat-out"),
        "unit",
        configs=("ID-N", "IDP-N", "DP-N", "ID-o-N", "DP-S_tau", "IDP-N_sep"),
        spaces=("bv",),
        workers=4,
    )
    vs_run = bench.run_benchmark(
        root,
        tmp_path_factory.mktemp("strat-vs-out"),
        "unit",
        configs=("IDP-S",),
        tracks=("ideal-S",),
        spaces=("vs",),
    )
    return {"run": run, "vs_run": vs_run, "root": root, "pts": pts, "amps": amps}


def test_runner_all_strategies(strategy_run: dict[str, Any]) -> None:
    """Every registered strategy runs cleanly with finite gauge errors."""
    run = strategy_run["run"]
    assert [row for row in run.rows if row["status"] == "error"] == []
    names = ("ID-N", "IDP-N", "DP-N", "ID-o-N", "DP-S_tau", "IDP-N_sep")
    for name in names:
        cfg = get_config(name)
        for strategy in cfg.n_strategies:
            rows = [
                row
                for row in run.rows
                if row["config"] == name and row["strategy"] == strategy.name
            ]
            assert rows, (name, strategy.name)
        for row in run.rows:
            if row["config"] != name or row["status"] != "ok":
                continue
            if row["strategy"] in ("los", "blind", "self_cal", "varpro", "xcorr"):
                errors = row["gauge_errors"]
                assert errors is not None
                keys = {"runtime_s"}
                if "phi" in cfg.gauge_unknowns:
                    keys |= {"phase_rms_deg", "phase_max_deg"}
                if "tau" in cfg.gauge_unknowns:
                    keys |= {"delay_rms_ns", "delay_max_ns"}
                assert set(errors) == keys | {"estimated", "by_los"}, (name, sorted(errors))
                for key in keys:
                    assert np.isfinite(errors[key]), (name, key)
                assert set(errors["by_los"]) == {"los_visible", "los_blocked"}
    vs_rows = strategy_run["vs_run"].rows
    assert [row for row in vs_rows if row["status"] == "error"] == []
    mmv = [row for row in vs_rows if row["solver"] == "mmv_constrained"]
    assert len(mmv) == 1 and mmv[0]["status"] == "ok" and mmv[0]["space"] == "vs"


@pytest.mark.skipif(os.environ.get("RF_TOMO_BENCH") != "1", reason="needs RF_TOMO_BENCH=1")
def test_smoke_benchmark(tmp_path: Path) -> None:
    """Smoke-suite timing benchmark on an 8-view, 2-BS synthetic dataset (not gating)."""
    from plateau_rt.domain.rf_camera.camera import generate_ring_views

    target = (5.0, 5.0, 5.0)
    views = generate_ring_views(target=target, radius_m=30.0, ue_height_m=1.5, num_views=8)
    root = tmp_path / "smoke-bench"
    from rf_manifest_fixtures import write_v3_dataset as write_v3

    write_v3(root, rows=8, cols=8, bins=32, views=views)
    dataset = tio.load_dataset(root)
    phantom = synthetic.l0c_random(0, 8, geom=dataset.geom)
    assert phantom.gt.points_pos is not None and phantom.gt.points_rho is not None
    rendered = atom_cfr(
        np.asarray(phantom.gt.points_pos),
        np.asarray(phantom.gt.points_rho),
        dataset.geom,
        "bv",
    )
    for view_index, view in enumerate(dataset.manifest.views):
        np.save(view.aperture_cfr_path, rendered[view_index].astype(np.complex64))
    out = tmp_path / "smoke-out"
    run = bench.run_benchmark(root, out, "smoke")
    totals: dict[str, float] = {}
    for row in run.rows:
        totals[str(row["config"])] = totals.get(str(row["config"]), 0.0) + float(row["runtime_s"])
    counts = {"ok": 0, "n/a": 0, "error": 0}
    for row in run.rows:
        counts[str(row["status"])] += 1
    print(f"status counts: {counts}")
    for name in sorted(totals):
        print(f"{name}: {totals[name]:.2f}s")


def test_ill_posed_degauging_and_selection() -> None:
    """``ill_posed_at`` removes the given gauges and restricts complex rows to the node."""
    geom = micro_geometry()
    grid = bench.make_grid(CENTER, (4.0, 4.0, 2.0), 2.0)
    y, pts, _ = two_point_scene(geom, grid, 0)
    p_ref, _ = sync_mod.reference_power(y, np.ones((4, 1), dtype=bool))
    tracks, gt = sync_mod.make_tracks(
        y, 30.0, float(p_ref), sync_mod.TrackSeeds(0, 0), freq_offsets=geom.freq_offsets, ref=REF
    )
    sigma2 = float(gt["sigma2"])
    cfg = get_config("IDP-N")
    gauged = bench.ill_posed_at(
        cfg, tracks["ideal-N"], geom, pts, "bv", noise_var=sigma2, gauges=gt["gauges"]["ideal-N"]
    )
    plain = bench.ill_posed_at(cfg, tracks["ideal-S"], geom, pts, "bv", noise_var=sigma2)
    np.testing.assert_allclose(gauged.rho, plain.rho, rtol=1e-9)
    np.testing.assert_allclose(gauged.crb_std, plain.crb_std, rtol=1e-6)
    wideband = bench.ill_posed_at(
        get_config("IDP-S"), tracks["ideal-S"], geom, pts, "bv", noise_var=sigma2
    )
    partial = bench.ill_posed_at(
        get_config("IPxK-S"), tracks["ideal-S"], geom, pts, "bv", noise_var=sigma2
    )
    narrow = bench.ill_posed_at(
        get_config("IP-S"), tracks["ideal-S"], geom, pts, "bv", noise_var=sigma2
    )
    # Fewer bins carry less range information: 16 bins < 4 bins < the DC bin.
    assert np.max(wideband.crb_std) * 1.5 < np.max(partial.crb_std)
    assert np.max(partial.crb_std) * 1.5 < np.max(narrow.crb_std)
