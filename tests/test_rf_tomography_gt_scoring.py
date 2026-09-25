"""Unit tests for tomography GT scoring: detectable VS, strata, surface, planes."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from test_rf_tomography_gt_app import _make_dataset

from plateau_rt.application import rf_tomography_benchmark as bench
from plateau_rt.application import rf_tomography_gt as gt_app
from plateau_rt.application import rf_tomography_io as tio
from plateau_rt.domain.rf_tomography import configs
from plateau_rt.domain.rf_tomography.geometry import VoxelGrid


def _hand_gt() -> dict[str, np.ndarray]:
    """Return the hand-made (V=2, B=2, P=3, M=3) GT dict of tests 1-3."""
    path_type = np.full((2, 2, 3), 1, dtype=np.int8)
    path_type[..., 2] = -1
    path_power = np.zeros((2, 2, 3))
    path_power[0, 0] = [1.0, 0.1, 1e6]
    path_power[0, 1] = [1e-2, 1e-3, 1e6]
    path_power[1, 0] = [1.0, 0.5, 1e6]
    path_power[1, 1] = [1.0, 0.2, 1e6]
    return {
        "path_type": path_type,
        "path_power": path_power,
        "vs_pos": np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]]),
        "vs_bs": np.array([0, 1, 1]),
        "vs_order": np.array([0, 1, 4]),
        "vs_visibility": np.array([[True, True], [True, True], [False, True]]),
        "vs_power": np.array([[1e-4, 1e-2], [1.2e-5, 1e-4], [0.0, 5e-4]]),
        "vs_path_type": np.array([[1, 2], [1, 1], [-1, 1]], dtype=np.int8),
        "los_visible": np.array([[True, True], [False, False]]),
    }


def test_vs_detectable() -> None:
    """Detectable VS follow the per-capture dynamic-range rule."""
    gt = _hand_gt()
    # vs0: -20 dB in (1, 0); vs1: -29.2 dB in (0, 1); vs2: -33 dB -> out
    np.testing.assert_array_equal(gt_app.vs_detectable(gt), [True, True, False])
    captures = np.array([[True, False], [True, True]])
    np.testing.assert_array_equal(gt_app.vs_detectable(gt, captures=captures), [True, False, False])
    np.testing.assert_array_equal(
        gt_app.vs_detectable(gt, dynamic_range_db=35.0), [True, True, True]
    )
    empty = dict(gt)
    for key, shape in (
        ("vs_pos", (0, 3)),
        ("vs_bs", (0,)),
        ("vs_order", (0,)),
        ("vs_visibility", (0, 2)),
        ("vs_power", (0, 2)),
        ("vs_path_type", (0, 2)),
    ):
        empty[key] = np.zeros(shape, dtype=gt[key].dtype)
    got = gt_app.vs_detectable(empty)
    assert got.shape == (0,) and got.dtype == bool


def test_vs_strata() -> None:
    """VS labels follow strongest-view mechanism, order bin and LoS visibility."""
    strata = gt_app.vs_strata(_hand_gt())
    # strongest visible views: vs0 -> v=1 (type 2); vs1 -> v=1 (type 1); vs2 -> v=1 (type 1)
    assert strata["mechanism"].tolist() == ["refraction", "specular", "specular"]
    assert strata["order"].tolist() == ["0", "1", "3+"]
    # vs2 (BS 1) is seen only in view 1, whose LoS is blocked, although view 0 has LoS
    assert strata["los"].tolist() == ["los_visible", "los_visible", "los_blocked"]


def test_vs_recall_strata() -> None:
    """Stratified recall matches the hand-computed gate table exactly."""
    det = np.array([[0.3, 0.0, 0.0], [20.2, 0.0, 0.0]])
    # detectable GT: vs0 (refraction / 0 / los_visible) hit; vs1 (specular / 1 / blocked) missed
    ones = {"0.5": 1.0, "1.0": 1.0, "2.0": 1.0, "4.0": 1.0}
    zeros = {"0.5": 0.0, "1.0": 0.0, "2.0": 0.0, "4.0": 0.0}
    halves = {"0.5": 0.5, "1.0": 0.5, "2.0": 0.5, "4.0": 0.5}
    assert gt_app.vs_recall_strata(det, _hand_gt()) == {
        "mechanism": {
            "refraction": {"num_gt": 1, "recall": dict(ones)},
            "specular": {"num_gt": 1, "recall": dict(zeros)},
        },
        "order": {
            "0": {"num_gt": 1, "recall": dict(ones)},
            "1": {"num_gt": 1, "recall": dict(zeros)},
        },
        "los": {
            "los_visible": {"num_gt": 2, "recall": dict(halves)},
        },
    }


def _surface_gt() -> dict[str, np.ndarray]:
    """Return the surface GT dict of test 4 (samples x = 0..9 on the x axis)."""
    samples = np.array([[float(x), 0.0, 0.0] for x in range(10)])
    return {
        "surface_samples": samples,
        "surface_observable": np.array([x not in (3, 4) for x in range(10)]),
        "surface_specular_support": np.array([x in (2, 3) for x in range(10)]),
    }


def test_score_surface_map() -> None:
    """Surface scores match the hand-computed per-stratum table."""
    grid = VoxelGrid(origin=(0.0, 0.0, 0.0), spacing=1.0, shape=(5, 1, 1))
    density = np.array([0.0, 10.0, 0.5, 5.0, 1.0]).reshape(5, 1, 1)
    # cloud (level 1.0): x = 1, 3, 4 with weights [10, 5, 1]; box x in [-0.5, 4.5]
    # energy uses the whole map: weights [0, 10, 0.5, 5, 1] (total 16.5)
    result = gt_app.score_surface_map(_surface_gt(), density, grid)
    assert result["box"] == [[-0.5, -0.5, -0.5], [4.5, 0.5, 0.5]]
    assert result["rel_threshold"] == pytest.approx(0.1, abs=1e-12)
    strata = result["strata"]
    assert set(strata) == {"all", "observable", "specular"}

    all_s = strata["all"]
    assert all_s["num_ref"] == 5 and all_s["num_pred"] == 3
    assert all_s["prf"]["0.5"]["precision"] == pytest.approx(1.0, abs=1e-12)
    assert all_s["prf"]["0.5"]["recall"] == pytest.approx(3.0 / 5.0, abs=1e-12)
    assert all_s["prf"]["0.5"]["f"] == pytest.approx(0.75, abs=1e-12)
    assert all_s["prf"]["1.0"]["recall"] == pytest.approx(1.0, abs=1e-12)
    assert all_s["prf"]["2.0"] == {"precision": 1.0, "recall": 1.0, "f": 1.0}
    assert all_s["chamfer"] == {"accuracy": 0.0, "completeness": 0.4, "chamfer": 0.2}
    assert all_s["energy_within"] == {"0.5": 1.0, "1.0": 1.0, "2.0": 1.0}

    obs = strata["observable"]
    assert obs["num_ref"] == 3
    assert obs["prf"]["0.5"]["precision"] == pytest.approx(1.0 / 3.0, abs=1e-12)
    assert obs["prf"]["0.5"]["recall"] == pytest.approx(1.0 / 3.0, abs=1e-12)
    assert obs["prf"]["0.5"]["f"] == pytest.approx(1.0 / 3.0, abs=1e-12)
    assert obs["prf"]["1.0"] == {"precision": 1.0, "recall": 1.0, "f": 1.0}
    # accuracy = (10 * 0 + 5 * 1 + 1 * 1) / 16 = 6/16; completeness = 2/3
    assert obs["chamfer"]["accuracy"] == pytest.approx(6.0 / 16.0, abs=1e-12)
    assert obs["chamfer"]["completeness"] == pytest.approx(2.0 / 3.0, abs=1e-12)
    assert obs["chamfer"]["chamfer"] == pytest.approx(0.5 * (6.0 / 16.0 + 2.0 / 3.0), abs=1e-12)
    # energy distances to observable x: [0, 0, 0, 1, 1] -> (10 + 0.5) / 16.5
    assert obs["energy_within"]["0.5"] == pytest.approx(10.5 / 16.5, abs=1e-12)
    assert obs["energy_within"]["1.0"] == pytest.approx(1.0, abs=1e-12)
    assert obs["energy_within"]["2.0"] == pytest.approx(1.0, abs=1e-12)

    spec = strata["specular"]
    assert spec["num_ref"] == 2
    assert spec["prf"]["0.5"]["precision"] == pytest.approx(1.0 / 3.0, abs=1e-12)
    assert spec["prf"]["0.5"]["recall"] == pytest.approx(0.5, abs=1e-12)
    assert spec["prf"]["0.5"]["f"] == pytest.approx(0.4, abs=1e-12)
    assert spec["prf"]["1.0"] == {"precision": 1.0, "recall": 1.0, "f": 1.0}
    # accuracy = (10 * 1 + 5 * 0 + 1 * 1) / 16 = 11/16; completeness = 0.5
    assert spec["chamfer"]["accuracy"] == pytest.approx(11.0 / 16.0, abs=1e-12)
    assert spec["chamfer"]["completeness"] == pytest.approx(0.5, abs=1e-12)
    assert spec["chamfer"]["chamfer"] == pytest.approx(0.5 * (11.0 / 16.0 + 0.5), abs=1e-12)
    # energy distances to specular x: [2, 1, 0, 0, 1] -> (0.5 + 5) / 16.5
    assert spec["energy_within"]["0.5"] == pytest.approx(5.5 / 16.5, abs=1e-12)
    assert spec["energy_within"]["1.0"] == pytest.approx(1.0, abs=1e-12)
    assert spec["energy_within"]["2.0"] == pytest.approx(1.0, abs=1e-12)


def test_score_planes() -> None:
    """Plane scoring matches the hand-computed anchored matches."""
    deg5 = math.radians(5.0)
    normal = np.array([math.cos(deg5), 0.0, math.sin(deg5)])
    gt = {
        "plane_normal": np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        "plane_offset": np.array([20.0, 25.0]),
        "interaction_points": np.array(
            [
                [20.0, 0.0, 10.0],
                [20.0, 2.0, 10.0],
                [3.0, 25.0, 4.0],
                [5.0, 25.0, 4.0],
                [0.0, 0.0, 0.0],
            ]
        ),
        "interaction_plane": np.array([0, 0, 1, 1, -1]),
    }
    # anchors: (20, 1, 10) and (4, 25, 4)
    est_n = np.array([normal, [0.0, -1.0, 0.0]])
    est_d = np.array([float(normal @ np.array([20.3, 1.0, 10.0])), -25.4])
    result = gt_app.score_planes(gt, est_n, est_d)
    assert result["num_est"] == 2 and result["num_gt"] == 2
    assert (result["tp"], result["fp"], result["fn"]) == (2, 0, 0)
    assert result["angle_deg_median"] == pytest.approx(2.5, abs=1e-9)
    assert result["angle_deg_p90"] == pytest.approx(4.5, abs=1e-9)
    assert len(result["matches"]) == 2
    first, second = result["matches"]
    assert first["est"] == 0 and first["gt"] == 0
    assert first["angle_deg"] == pytest.approx(5.0, abs=1e-9)
    assert first["offset_m"] == pytest.approx(0.3 * math.cos(deg5), abs=1e-9)
    assert second["est"] == 1 and second["gt"] == 1
    assert second["angle_deg"] == pytest.approx(0.0, abs=1e-9)
    assert second["offset_m"] == pytest.approx(0.4, abs=1e-9)


def test_gauge_error_summary_by_los() -> None:
    """Gauge errors stratify by LoS visibility with model-error maxima."""
    cfg = configs.get_config("IDP-N")
    used = (np.array([[0.2], [-0.2], [0.0], [0.0]]), np.array([[1e-9], [0.0], [3e-9], [0.0]]))
    truth = (np.zeros((4, 1)), np.zeros((4, 1)))
    los_visible = np.array([[True], [True], [False], [False]])
    los_model_error = np.array([[0.01], [-0.03], [np.nan], [np.nan]])
    summary = bench.gauge_error_summary(
        cfg,
        used,
        truth,
        1e-6,
        ("phi", "tau"),
        los_visible=los_visible,
        los_model_error=los_model_error,
    )
    assert set(summary) >= {
        "phase_rms_deg",
        "phase_max_deg",
        "delay_rms_ns",
        "delay_max_ns",
        "by_los",
    }
    visible = summary["by_los"]["los_visible"]
    assert visible["num"] == 2
    # global phase removed over all captures is 0, so |phase| = [0.2, 0.2] rad here
    assert visible["phase_rms_deg"] == pytest.approx(math.degrees(0.2), abs=1e-9)
    assert visible["phase_max_deg"] == pytest.approx(math.degrees(0.2), abs=1e-9)
    assert visible["delay_rms_ns"] == pytest.approx(math.sqrt(0.5), abs=1e-9)
    assert visible["delay_max_ns"] == pytest.approx(1.0, abs=1e-9)
    assert visible["los_model_error_deg_max"] == pytest.approx(math.degrees(0.03), abs=1e-9)
    blocked = summary["by_los"]["los_blocked"]
    assert blocked["num"] == 2
    assert blocked["phase_rms_deg"] == pytest.approx(0.0, abs=1e-9)
    assert blocked["phase_max_deg"] == pytest.approx(0.0, abs=1e-9)
    assert blocked["delay_rms_ns"] == pytest.approx(3.0 / math.sqrt(2.0), abs=1e-9)
    assert blocked["delay_max_ns"] == pytest.approx(3.0, abs=1e-9)

    all_visible = bench.gauge_error_summary(
        cfg, used, truth, 1e-6, ("phi", "tau"), los_visible=np.ones((4, 1), dtype=bool)
    )
    assert all_visible["by_los"]["los_blocked"]["num"] == 0
    assert all_visible["by_los"]["los_blocked"]["phase_max_deg"] is None
    assert all_visible["by_los"]["los_blocked"]["delay_max_ns"] is None

    plain = bench.gauge_error_summary(cfg, used, truth, 1e-6, ("phi", "tau"))
    assert "by_los" not in plain
    assert set(plain) == {
        "estimated",
        "phase_rms_deg",
        "phase_max_deg",
        "delay_rms_ns",
        "delay_max_ns",
    }


def _recon(run: bench.BenchmarkRun, row: dict[str, Any]) -> dict[str, np.ndarray]:
    """Load the recon npz of ``row``."""
    assert row["recon"] is not None
    with np.load(run.out_dir / str(row["recon"])) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def test_runner_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Runner rows carry detectable-VS, surface and by-LoS payloads."""
    original = gt_app.vs_detectable

    def narrow(arrays: Any, dynamic_range_db: float = 10.0, captures: Any = None) -> np.ndarray:
        """Detectability at 10 dB so that the mirror scene has an undetectable VS."""
        return original(arrays, dynamic_range_db, captures)

    monkeypatch.setattr(gt_app, "vs_detectable", narrow)
    root = tmp_path / "dataset"
    _make_dataset(root)
    gt_app.write_tomography_gt(root)
    out = tmp_path / "out"
    run = bench.run_benchmark(
        root,
        out,
        "unit",
        configs=["IDP-S", "ID-S", "IDP-N"],
        spaces=["bv", "vs"],
        strategies=["none", "oracle"],
    )
    gt = gt_app.load_tomography_gt(root / "tomography_gt.npz")
    gt_arrays = gt
    detectable = gt_app.vs_detectable(gt_arrays)
    num_detectable = int(detectable.sum())
    num_vs = int(gt_arrays["vs_pos"].shape[0])
    assert 0 < num_detectable < num_vs

    for row in run.rows:
        assert row["schema"] == "rf_tomo_result/3"
    checked = {"surface": 0, "roi": 0, "vs": 0, "gauge": 0}
    for row in run.rows:
        if row["status"] != "ok" or row["space"] != "bv" or row["stage"] not in ("E1", "E2"):
            continue
        checked["surface"] += 1
        recon = _recon(run, row)
        grid = VoxelGrid(
            recon["grid_origin"], float(recon["grid_spacing"]), tuple(recon["grid_shape"])
        )
        expected = tio.to_jsonable(gt_app.score_surface_map(gt, recon["map"], grid))
        assert row["metrics"] is not None
        assert tio.to_jsonable(row["metrics"]["surface"]) == expected
    for row in run.rows:
        if row["status"] == "ok" and row["space"] == "bv" and row["stage"] == "ROI":
            checked["roi"] += 1
            assert row["metrics"] is None
    for row in run.rows:
        if row["status"] != "ok" or row["space"] != "vs":
            continue
        checked["vs"] += 1
        payload = row["metrics"]
        assert payload is not None
        assert payload["num_gt"] == num_detectable
        assert payload["num_gt_total"] == num_vs
        assert payload["gt_subset"] == "vs_detectable"
        recon = _recon(run, row)
        assert tio.to_jsonable(payload["strata"]) == tio.to_jsonable(
            gt_app.vs_recall_strata(recon["detections"], gt)
        )
    num_captures = 0
    for row in run.rows:
        if row["config"] != "IDP-N" or row["gauge_errors"] is None:
            continue
        checked["gauge"] += 1
        by_los = row["gauge_errors"]["by_los"]
        assert set(by_los) == {"los_visible", "los_blocked"}
        num_captures = by_los["los_visible"]["num"] + by_los["los_blocked"]["num"]
        if row["strategy"] == "oracle":
            for stratum in ("los_visible", "los_blocked"):
                entry = by_los[stratum]
                if entry["num"] == 0:
                    continue
                assert entry["phase_max_deg"] is not None and entry["phase_max_deg"] <= 1e-9
                assert entry["delay_max_ns"] is not None and entry["delay_max_ns"] <= 1e-9
    assert min(checked.values()) > 0, checked
    dataset = tio.load_dataset(root)
    assert num_captures == dataset.geom.num_views * dataset.geom.num_bs
    manifest = json.loads((out / tio.RUN_MANIFEST_FILE).read_text(encoding="utf-8"))
    assert manifest["gt"]["num_vs_detectable"] == num_detectable
    assert manifest["gt"]["num_surface_samples"] == gt_arrays["surface_samples"].shape[0]
    assert gt_app.summarize_tomography_gt(gt)["num_vs_detectable"] == num_detectable


def test_load_tomography_gt_requires_schema(tmp_path: Path) -> None:
    """An npz without ``schema`` is rejected."""
    path = tmp_path / "no_schema.npz"
    np.savez(path, vs_pos=np.zeros((2, 3)))
    with pytest.raises(ValueError):
        gt_app.load_tomography_gt(path)
