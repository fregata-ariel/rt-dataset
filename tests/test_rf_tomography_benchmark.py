"""End-to-end tests for the tomography benchmark runner (T16, §3.5)."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from click.testing import CliRunner
from rf_manifest_fixtures import write_v3_dataset

from plateau_rt.application import rf_tomography_benchmark as bench
from plateau_rt.application import rf_tomography_io as tio
from plateau_rt.cli.main import cli
from plateau_rt.domain.rf_camera.camera import RFViewSpec, look_at_orientation
from plateau_rt.domain.rf_tomography import configs as cfg_mod
from plateau_rt.domain.rf_tomography import metrics as metric_mod
from plateau_rt.domain.rf_tomography.forward_exact import atom_cfr
from plateau_rt.domain.rf_tomography.synthetic import offgrid_points

CENTER = (0.0, 0.0, 5.0)
BS = (3.0, -2.0, 40.0)


def write_micro_dataset(
    root: Path | str, seed: int = 0, num_views: int = 4
) -> tuple[Path, np.ndarray, np.ndarray]:
    """Write the §6 micro scene and return ``(root, points, amplitudes)``."""
    root = Path(root)
    heights = (1.5, 12.0, 1.5, 12.0)
    views = []
    for index in range(num_views):
        azimuth = np.deg2rad(20 + 90 * index)
        position = (
            float(15.0 * np.cos(azimuth)),
            float(15.0 * np.sin(azimuth)),
            float(heights[index % 4]),
        )
        views.append(
            RFViewSpec(
                f"ue_{index:06d}",
                position,
                CENTER,
                look_at_orientation(position, CENTER),
            )
        )
    write_v3_dataset(
        root,
        rows=4,
        cols=4,
        bins=16,
        views=views,
        bs_positions=[BS],
        bs_look_at=CENTER,
    )
    dataset = tio.load_dataset(root)
    grid = bench.make_grid(CENTER, (4.0, 4.0, 2.0), 2.0)
    rng = np.random.default_rng(np.random.SeedSequence([seed, 7]))
    found = []
    for voxel in ((0, 0, 1), (4, 4, 1)):
        flat = np.array([np.ravel_multi_index(voxel, grid.shape)])
        points, _ = offgrid_points(grid, 1, rng, offset_max=0.2, voxels=flat)
        found.append(points[0])
    pts = np.stack(found).astype(np.float64)
    amps = (np.array([1.0, 0.7]) * np.exp(2j * np.pi * rng.uniform(size=2))).astype(np.complex128)
    rendered = atom_cfr(pts, amps, dataset.geom, "bv")
    for view_index, view in enumerate(dataset.manifest.views):
        np.save(view.aperture_cfr_path, rendered[view_index].astype(np.complex64))
    tio.write_ground_truth(
        root / tio.GT_FILE_NAME, points_pos=pts, points_rho=amps, points_space="bv"
    )
    manifest_path = root / "dataset_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["tomography_gt"] = {"artifact": tio.GT_FILE_NAME}
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    return root, pts, amps


@pytest.fixture(scope="module")
def micro_run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Run the unit suite once on the micro scene (none + oracle strategies)."""
    root, pts, amps = write_micro_dataset(tmp_path_factory.mktemp("micro"), seed=0)
    out = tmp_path_factory.mktemp("micro-out")
    run = bench.run_benchmark(root, out, "unit", strategies=("none", "oracle"))
    return {"run": run, "root": root, "pts": pts, "amps": amps}


@pytest.fixture(scope="module")
def single_view_run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Run D-S/D-N on the single-view micro scene."""
    root, pts, amps = write_micro_dataset(tmp_path_factory.mktemp("micro1"), seed=0, num_views=1)
    out = tmp_path_factory.mktemp("micro1-out")
    run = bench.run_benchmark(
        root,
        out,
        "unit",
        configs=("D-S", "D-N"),
        tracks=("ideal-S", "ideal-N"),
        strategies=("none",),
    )
    return {"run": run, "root": root, "pts": pts, "amps": amps}


def _rows_by(run: bench.BenchmarkRun, **fields: Any) -> list[dict[str, Any]]:
    """Return rows of ``run`` matching every ``fields`` entry."""
    return [row for row in run.rows if all(row[key] == value for key, value in fields.items())]


def _recon(run: bench.BenchmarkRun, row: dict[str, Any]) -> dict[str, np.ndarray]:
    """Load the recon npz of ``row``."""
    assert row["recon"] is not None
    with np.load(run.out_dir / str(row["recon"])) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def test_schema(micro_run: dict[str, Any]) -> None:
    """Rows validate, match the manifest, and every ok recon file is complete."""
    run = micro_run["run"]
    assert isinstance(run, bench.BenchmarkRun)
    stored = tio.read_results(run.results_path)
    assert len(stored) == len(run.rows)
    for line, row in zip(stored, run.rows, strict=True):
        tio.validate_result_row(line)
        assert line == tio.to_jsonable(row)
    manifest = json.loads(run.run_manifest_path.read_text(encoding="utf-8"))
    tio.validate_run_manifest(manifest)
    assert manifest["results"]["sha256"] == tio.sha256_file(run.results_path)
    assert manifest["results"]["rows"] == len(run.rows)
    dataset = tio.load_dataset(micro_run["root"])
    assert manifest["dataset"]["hashes"] == dict(dataset.hashes)
    grid_shape = tuple(manifest["grid"]["shape"])
    num_views = manifest["dataset"]["num_views"]
    num_bs = manifest["dataset"]["num_bs"]
    for row in run.rows:
        if row["status"] != "ok":
            assert row["recon"] is None
            continue
        recon = _recon(run, row)
        assert recon["gauges"].shape == (num_views, num_bs, 2)
        assert recon["detections"].shape == (recon["scores"].shape[0], 3)
        assert bool(np.all(np.diff(recon["scores"]) <= 0.0))
        assert recon["space"] == row["space"]
        if row["stage"] == "E1":
            assert recon["map"].shape == grid_shape
            assert bool(np.all(np.isfinite(recon["map"])))
        if row["stage"] == "E2":
            assert recon["map"].shape == grid_shape
            assert bool(np.all(np.isfinite(recon["density"])))
            assert bool(np.all(recon["density"] >= 0.0))
            assert "support_indices" in recon


def test_every_node_runs(micro_run: dict[str, Any]) -> None:
    """Every config runs without errors and stage coverage matches the registry."""
    run = micro_run["run"]
    geom = tio.load_dataset(micro_run["root"]).geom
    assert _rows_by(run, status="error") == []
    names = {row["config"] for row in run.rows}
    assert names == set(cfg_mod.CONFIGS)
    for name, cfg in cfg_mod.CONFIGS.items():
        track = bench.config_track(cfg, list(bench.TRACKS))
        assert track is not None
        job = [s for s in ("none", "oracle") if s in bench.job_strategies(cfg)]
        assert job
        if cfg.e1 is None:
            e1_rows = _rows_by(run, config=name, stage="E1")
            assert len(e1_rows) == len(job)
            for row in e1_rows:
                assert row["status"] == "n/a"
                assert row["solver"] == "-"
                assert cfg.planned[0] in str(row["reason"])
            continue
        for strategy in job:
            rows = _rows_by(
                run, config=name, track=track, space="bv", strategy=strategy, stage="E1"
            )
            assert len(rows) == 1
            assert rows[0]["status"] == "ok", (name, strategy)
            assert rows[0]["n_forward"] is None and rows[0]["n_adjoint"] is None
        for step in cfg.e2:
            rows = _rows_by(
                run,
                config=name,
                track=track,
                space="bv",
                strategy="none",
                stage="E2",
                solver=step.name,
            )
            assert len(rows) == 1
            if "bv" in step.spaces:
                assert rows[0]["status"] == "ok", (name, step.name)
                row = rows[0]
                factor = geom.num_bins if step.call == "coherent_per_bin" else 1
                for key in ("n_iter", "n_forward", "n_adjoint"):
                    assert isinstance(row[key], int) and not isinstance(row[key], bool), key
                assert 1 <= row["n_iter"] <= 10 * max(1, factor)
                assert row["n_forward"] >= row["n_iter"]
                assert row["n_adjoint"] >= row["n_iter"]
                if step.call == "power":
                    assert row["n_iter"] == 10
                if step.call == "coherent_per_bin" and step.operator["bins"] == "all":
                    # one solve per bin, each at least one iteration: never the budget of 10
                    assert row["n_iter"] >= geom.num_bins
            else:
                assert rows[0]["status"] == "n/a"
    kl = _rows_by(
        run, config="ID-S", track="ideal-S", space="bv", strategy="none", stage="E2", solver="kl_em"
    )
    assert len(kl) == 1
    assert (kl[0]["n_iter"], kl[0]["n_forward"], kl[0]["n_adjoint"]) == (10, 11, 11)
    mmv = _rows_by(run, config="DP-S", stage="E2", solver="mmv_constrained")
    assert mmv and all(row["status"] == "n/a" for row in mmv)
    for name, cfg in cfg_mod.CONFIGS.items():
        track = bench.config_track(cfg, list(bench.TRACKS))
        assert track is not None
        planned = _rows_by(run, config=name, track=track, space="bv", stage="planned")
        assert len(planned) == len(cfg.planned)
        for row in planned:
            assert row["status"] == "n/a"
            assert row["strategy"] == "none"
            assert row["solver"] == row["reason"]


def test_id_and_idp_within_1_2m(micro_run: dict[str, Any]) -> None:
    """ID-S and IDP-S E1 detections match both GT points within 1.2 m."""
    run = micro_run["run"]
    pts = micro_run["pts"]
    for name in ("ID-S", "IDP-S"):
        rows = _rows_by(run, config=name, track="ideal-S", space="bv", strategy="none", stage="E1")
        assert len(rows) == 1
        det = _recon(run, rows[0])["detections"]
        assert metric_mod.match(det[:2], pts, 1.2).tp == 2


def test_idp_roi_within_0_25m(micro_run: dict[str, Any]) -> None:
    """The IDP-S ROI row matches both GT points within 0.25 m."""
    run = micro_run["run"]
    pts = micro_run["pts"]
    rows = _rows_by(run, config="IDP-S", track="ideal-S", space="bv", strategy="none", stage="ROI")
    assert len(rows) == 1
    det = _recon(run, rows[0])["detections"]
    assert metric_mod.match(det, pts, 0.25).tp == 2


def test_metric_plumbing(micro_run: dict[str, Any]) -> None:
    """Row metrics equal the T10 metrics recomputed from the recon detections."""
    run = micro_run["run"]
    pts = micro_run["pts"]
    rows = _rows_by(run, config="IDP-S", track="ideal-S", space="bv", strategy="none", stage="E1")
    assert len(rows) == 1
    row = rows[0]
    recon = _recon(run, row)
    det, scores = recon["detections"], recon["scores"]
    assert row["metrics"]["gates"]["4.0"]["recall"] == 1.0
    for gate in ("0.5", "1.0", "2.0", "4.0"):
        matched = metric_mod.match(det, pts, float(gate))
        assert row["metrics"]["gates"][gate]["tp"] == matched.tp
        assert row["metrics"]["gates"][gate]["fp"] == matched.fp
        assert row["metrics"]["gates"][gate]["fn"] == matched.fn
    assert row["metrics"]["ap_1m"] == metric_mod.ap_at(det, scores, pts, 1.0)


def test_controls(micro_run: dict[str, Any]) -> None:
    """Sync-invariance, oracle and track-selection controls hold."""
    run = micro_run["run"]

    def e1_map(**fields: Any) -> np.ndarray:
        """Return the E1 recon map of the single matching row."""
        return _recon(run, _rows_by(run, stage="E1", **fields)[0])["map"]

    i_s = e1_map(config="I-S", track="ideal-S", strategy="none")
    i_n = e1_map(config="I-N", track="ideal-N", strategy="none")
    assert float(np.max(np.abs(i_n - i_s))) / float(np.max(np.abs(i_s))) <= 1e-12
    s_map = e1_map(config="IDP-S", track="ideal-S", strategy="none")
    oracle_map = e1_map(config="IDP-N", track="ideal-N", strategy="oracle")
    scale = float(np.max(np.abs(s_map)))
    assert float(np.max(np.abs(oracle_map - s_map))) / scale <= 1e-10
    roi_s = _rows_by(run, config="IDP-S", track="ideal-S", strategy="none", stage="ROI")[0]
    roi_o = _rows_by(run, config="IDP-N", track="ideal-N", strategy="oracle", stage="ROI")[0]
    np.testing.assert_allclose(
        _recon(run, roi_o)["detections"], _recon(run, roi_s)["detections"], rtol=0.0, atol=1e-9
    )
    blind_map = e1_map(config="IDP-N", track="ideal-N", strategy="none")
    assert float(np.max(np.abs(blind_map - s_map))) / scale > 1e-2
    for row in run.rows:
        if row["strategy"] != "oracle" or row["status"] != "ok":
            continue
        cfg = cfg_mod.get_config(str(row["config"]))
        errors = row["gauge_errors"]
        assert errors is not None
        if "phi" in cfg.gauge_unknowns:
            assert errors["phase_max_deg"] <= 1e-9
        if "tau" in cfg.gauge_unknowns:
            assert errors["delay_max_ns"] <= 1e-9
    in0_tracks = {row["track"] for row in run.rows if row["config"] == "I@n0"}
    assert in0_tracks == {"ideal-S"}


def test_d_n_ill_posed_rebuilt(micro_run: dict[str, Any]) -> None:
    """The D-N ill_posed payload equals the independently computed T15b test."""
    from plateau_rt.domain.rf_tomography.identifiability import (
        gauge_reduced_fim,
        ill_posed,
        numeric_jacobian,
        return_model,
    )

    run = micro_run["run"]
    rows = _rows_by(run, config="D-N", track="ideal-N", space="bv", strategy="none", stage="E1")
    assert len(rows) == 1
    payload = rows[0]["ill_posed"]
    assert payload is not None and "error" not in payload
    geom = tio.load_dataset(micro_run["root"]).geom
    points = np.asarray(payload["points"], dtype=np.float64)
    rho = np.asarray(payload["rho"], dtype=np.float64)
    num_points = int(points.shape[0])
    assert num_points > 0
    rows_g, cols_g, bins = geom.aperture_shape[0], geom.aperture_shape[1], geom.num_bins
    lam = geom.wavelength
    spacing = np.asarray(geom.elem_offsets, dtype=np.float64)
    d_col = float(np.linalg.norm(spacing[1] - spacing[0])) / lam
    d_row = float(np.linalg.norm(spacing[cols_g] - spacing[0])) / lam
    delta_f = float(geom.delta_f)
    with np.errstate(divide="ignore", invalid="ignore"):
        sig_t = 1e9 * np.sqrt(6.0 / (rho * (bins**2 - 1))) / (2.0 * np.pi * delta_f)
        sig_y = np.sqrt(6.0 / (rho * (cols_g**2 - 1))) / (2.0 * np.pi * d_col)
        sig_z = np.sqrt(6.0 / (rho * (rows_g**2 - 1))) / (2.0 * np.pi * d_row)
    sig = np.stack([sig_y, sig_z, sig_t], axis=-1)
    num_views, num_bs = geom.num_views, geom.num_bs

    def model(vector: np.ndarray) -> np.ndarray:
        """Return the ``[V, B, P, 3]`` list mean in (direction, ns) units."""
        array = np.asarray(vector, dtype=np.float64)
        pos = array[: 3 * num_points].reshape(num_points, 3)
        tau_ns = array[3 * num_points :].reshape(num_views, num_bs)
        mean = return_model(pos, geom, tau_ns * 1e-9, space="bv").copy()
        mean[..., 2] *= 1e9
        return mean

    theta = np.concatenate([points.ravel(), np.zeros(num_views * num_bs)])
    jac = numeric_jacobian(model, theta)
    weights = np.where((rho >= 10.0)[..., None], 1.0 / sig**2, 0.0).ravel(order="C")
    reduced = gauge_reduced_fim(jac, weights, np.arange(3 * num_points, theta.shape[0]))
    flag, cond, std = ill_posed(reduced, np.arange(3 * num_points))
    assert flag == payload["flag"]
    assert abs(cond - float(payload["cond"])) / float(payload["cond"]) <= 1e-9
    np.testing.assert_allclose(std, payload["crb_std_m"], rtol=1e-9, atol=0.0)


def test_single_view(single_view_run: dict[str, Any]) -> None:
    """With one view D-N is flagged ill-posed and D-S is not."""
    run = single_view_run["run"]
    assert _rows_by(run, status="error") == []
    d_n = _rows_by(run, config="D-N", track="ideal-N", strategy="none", stage="E1")[0]
    d_s = _rows_by(run, config="D-S", track="ideal-S", strategy="none", stage="E1")[0]
    assert d_n["ill_posed"]["flag"] is True
    assert d_s["ill_posed"]["flag"] is False


def test_cli(micro_run: dict[str, Any], tmp_path: Path) -> None:
    """The CLI runs the benchmark, refuses to overwrite, then overwrites."""
    root = micro_run["root"]
    out = tmp_path / "cli-out"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "rf-tomo-bench",
            "--dataset",
            str(root),
            "--suite",
            "unit",
            "--tracks",
            "ideal-S",
            "--configs",
            "I-S",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out / tio.RESULTS_FILE).exists()
    assert (out / tio.RUN_MANIFEST_FILE).exists()
    assert (out / tio.RECON_DIR).is_dir()
    result = runner.invoke(
        cli,
        [
            "rf-tomo-bench",
            "--dataset",
            str(root),
            "--suite",
            "unit",
            "--tracks",
            "ideal-S",
            "--configs",
            "I-S",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code != 0
    result = runner.invoke(
        cli,
        [
            "rf-tomo-bench",
            "--dataset",
            str(root),
            "--suite",
            "unit",
            "--tracks",
            "ideal-S",
            "--configs",
            "I-S",
            "--out",
            str(out),
            "--overwrite",
        ],
    )
    assert result.exit_code == 0, result.output
    command = cli.commands["rf-tomo-bench"]
    options = {param.name: param for param in command.params}
    suite_choices = getattr(options["suite"].type, "choices", ())
    track_choices = getattr(options["tracks"].type, "choices", ())
    assert list(suite_choices) == sorted(bench.SUITES)
    assert list(track_choices) == list(bench.TRACKS)


def test_registry_coverage() -> None:
    """List and complex nodes cover every config node exactly once."""
    nodes = {cfg.node for cfg in cfg_mod.CONFIGS.values()}
    assert set(bench.LIST_COMPONENTS) | set(bench.COMPLEX_NODES) == nodes
    assert set(bench.LIST_COMPONENTS) & set(bench.COMPLEX_NODES) == set()
    assert bench.config_track(cfg_mod.get_config("I@n0"), ("ideal-N",)) == "ideal-N"
    assert bench.config_track(cfg_mod.get_config("IDP-S_tau"), list(bench.TRACKS)) == "S_tau"
    assert bench.job_strategies(cfg_mod.get_config("ID-N")) == ("none", "los", "xcorr", "oracle")
    assert bench.job_strategies(cfg_mod.get_config("I-N")) == ("none",)


def _strip_runtimes(value: Any) -> Any:
    """Return ``value`` with every ``runtime_s`` entry removed (recursively)."""
    if isinstance(value, dict):
        return {key: _strip_runtimes(item) for key, item in value.items() if key != "runtime_s"}
    if isinstance(value, list):
        return [_strip_runtimes(item) for item in value]
    return value


def test_workers_match_sequential(micro_run: dict[str, Any], tmp_path: Path) -> None:
    """A process-pool run writes the same rows as the in-process run (runtimes aside)."""
    kwargs: dict[str, Any] = {
        "configs": ("I-S", "ID-N"),
        "tracks": ("ideal-S", "ideal-N"),
        "strategies": ("none", "los"),
    }
    root = micro_run["root"]
    serial = bench.run_benchmark(root, tmp_path / "serial", "unit", **kwargs)
    pooled = bench.run_benchmark(root, tmp_path / "pooled", "unit", workers=2, **kwargs)
    assert _strip_runtimes(list(pooled.rows)) == _strip_runtimes(list(serial.rows))
    assert [row["gauge_errors"] for row in serial.rows if row["config"] == "I-S"] == [None] * len(
        [row for row in serial.rows if row["config"] == "I-S"]
    )
    assert any(row["strategy"] == "los" for row in serial.rows)
    with pytest.raises(ValueError):
        bench.run_benchmark(root, tmp_path / "bad", "unit", workers=0, **kwargs)
    iso = dataclasses.replace(tio.load_dataset(root), tx_pattern="iso")
    with pytest.raises(ValueError, match="tx_pattern"):
        bench.run_benchmark(iso, tmp_path / "iso", "unit", **kwargs)
    assert not (tmp_path / "iso").exists()
