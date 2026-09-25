"""Selection of BS captures and sub-bands for the tomography benchmark (T22)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from test_rf_tomography_gt_app import _make_dataset

from plateau_rt.application import rf_tomography_benchmark as bench
from plateau_rt.application import rf_tomography_gt as gt_app
from plateau_rt.application import rf_tomography_io as tio
from plateau_rt.domain.rf_tomography import gt as gt_domain
from plateau_rt.domain.rf_tomography.configs import get_config

MASK = [
    [True, False],
    [False, True],
    [True, True],
    [False, False],
    [True, True],
    [True, False],
]


@pytest.fixture(scope="module")
def dataset_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the mirror-scene dataset with tomography GT once."""
    root = tmp_path_factory.mktemp("selection") / "dataset"
    _make_dataset(root)
    gt_app.write_tomography_gt(root)
    return root


def test_select_captures(dataset_root: Path) -> None:
    """BS and sub-band selection slices data, geometry and provenance."""
    full = tio.load_dataset(dataset_root)
    sub = tio.select_captures(full, bs=[1], num_bins=32)
    np.testing.assert_array_equal(sub.y_clean, full.y_clean[:, [1]][..., 16:48])
    np.testing.assert_array_equal(sub.geom.freq_offsets, full.geom.freq_offsets[16:48])
    assert float(sub.geom.freq_offsets[16]) == 0.0
    np.testing.assert_array_equal(sub.geom.bs_pos, full.geom.bs_pos[[1]])
    np.testing.assert_array_equal(sub.geom.ue_pos, full.geom.ue_pos)
    np.testing.assert_array_equal(sub.los_visible, full.los_visible[:, [1]])
    assert dict(sub.selection) == {"bs": [1], "bins": [16, 48]}
    assert tio.select_captures(full, bs=None, num_bins=None) is full
    for bad_bins in (1, 65, True):
        with pytest.raises(ValueError):
            tio.select_captures(full, num_bins=bad_bins)
    for bad_bs in ([], [2], [0, 0], [True]):
        with pytest.raises(ValueError):
            tio.select_captures(full, bs=bad_bs)
    with pytest.raises(ValueError):
        tio.select_captures(sub, bs=[0])


def test_gt_select_bs(dataset_root: Path) -> None:
    """GT BS subsetting keeps the BS-1 rows and remaps every index."""
    full = gt_app.load_tomography_gt(dataset_root / "tomography_gt.npz")
    sub = gt_domain.select_bs(full, [1])
    assert set(sub) == set(full)
    for key in gt_domain.GT_CAPTURE_KEYS:
        if key == "path_vs":
            continue
        np.testing.assert_array_equal(sub[key], np.asarray(full[key])[:, [1]])
    keep = np.asarray(full["vs_bs"]).reshape(-1) == 1
    for key in gt_domain.GT_VS_KEYS:
        if key == "vs_bs":
            np.testing.assert_array_equal(sub[key], np.zeros(int(keep.sum()), dtype=np.int64))
        else:
            np.testing.assert_array_equal(sub[key], np.asarray(full[key])[keep])
    old_path_vs = np.asarray(full["path_vs"])
    new_path_vs = np.asarray(sub["path_vs"])
    old_pos = np.asarray(full["vs_pos"])
    new_pos = np.asarray(sub["vs_pos"])
    assert new_path_vs.shape == (old_path_vs.shape[0], 1, old_path_vs.shape[2])
    for v in range(old_path_vs.shape[0]):
        for p in range(old_path_vs.shape[2]):
            old = int(old_path_vs[v, 1, p])
            new = int(new_path_vs[v, 0, p])
            if old < 0:
                assert new == -1
            else:
                np.testing.assert_array_equal(new_pos[new], old_pos[old])
    interaction_bs = np.asarray(full["interaction_bs"]).reshape(-1)
    mask = interaction_bs == 1
    for key in gt_domain.GT_INTERACTION_KEYS:
        if key == "interaction_bs":
            np.testing.assert_array_equal(sub[key], np.zeros(int(mask.sum()), dtype=np.int64))
        else:
            np.testing.assert_array_equal(sub[key], np.asarray(full[key])[mask])
    assert np.asarray(sub["bs_ids"]).tolist() == [np.asarray(full["bs_ids"]).tolist()[1]]
    np.testing.assert_array_equal(
        gt_app.vs_detectable(sub), gt_app.vs_detectable(full)[np.asarray(full["vs_bs"]) == 1]
    )
    for label in gt_app.vs_strata(sub):
        np.testing.assert_array_equal(
            gt_app.vs_strata(sub)[label], gt_app.vs_strata(full)[label][keep]
        )
    swapped = gt_domain.select_bs(full, [1, 0])
    swapped_bs = np.asarray(swapped["vs_bs"])
    assert bool(np.all(swapped_bs[np.asarray(full["vs_bs"]) == 1] == 0))
    assert bool(np.all(swapped_bs[np.asarray(full["vs_bs"]) == 0] == 1))
    for bad in ([], [2], [0, 0], [True], [0, 1, 2]):
        with pytest.raises(ValueError):
            gt_domain.select_bs(full, bad)
    with pytest.raises(ValueError):
        gt_domain.select_bs({"vs_pos": np.zeros((2, 3))}, [0])


def test_select_ground_truth(dataset_root: Path, tmp_path: Path) -> None:
    """Ground-truth subsetting follows the capture arrays; points GT is untouched."""
    data = tio.load_dataset(dataset_root)
    gt = tio.find_ground_truth(data)
    assert gt is not None
    sub = tio.select_ground_truth(gt, [1])
    np.testing.assert_array_equal(sub.vs_pos, gt_domain.select_bs(gt.arrays, [1])["vs_pos"])
    assert sub.path == gt.path
    assert sub.sha256 == gt.sha256
    points_path = tmp_path / "points.npz"
    tio.write_ground_truth(points_path, points_pos=np.zeros((2, 3)))
    points_gt = tio.load_ground_truth(points_path)
    assert tio.select_ground_truth(points_gt, [1]) is points_gt


def test_run_benchmark_selection(dataset_root: Path, tmp_path: Path) -> None:
    """A selected benchmark run scores 6 captures and records the selection."""
    full = tio.load_dataset(dataset_root)
    out = tmp_path / "out"
    run = bench.run_benchmark(
        dataset_root,
        out,
        "unit",
        configs=["ID-N", "IDP-N"],
        spaces=["vs"],
        strategies=["none", "los", "oracle"],
        bs=[1],
        num_bins=32,
    )
    assert [row for row in run.rows if row["status"] == "error"] == []
    for row in run.rows:
        payload = row["gauge_errors"]
        if payload is None:
            continue
        by_los = payload["by_los"]
        assert by_los["los_visible"]["num"] + by_los["los_blocked"]["num"] == 6
        if row["strategy"] == "los":
            assert payload["los_fit"]["fitted"] == full.los_visible[:, [1]].tolist()
    manifest = json.loads((out / tio.RUN_MANIFEST_FILE).read_text(encoding="utf-8"))
    assert manifest["options"]["bs"] == [1]
    assert manifest["options"]["num_bins"] == 32
    assert manifest["dataset"]["selection"] == {"bs": [1], "bins": [16, 48]}


def test_estimate_gauges_los_visible(dataset_root: Path) -> None:
    """The LoS fit skips blocked captures and matches the unmasked fit."""
    data = tio.load_dataset(dataset_root)
    grid = bench.make_grid(data.target, (4.0, 4.0, 2.0), 2.0)
    mask = np.asarray(MASK, dtype=bool)
    cfg = get_config("IDP-N")
    masked = bench.estimate_gauges(
        cfg,
        "los",
        data.y_clean,
        data.geom,
        grid,
        "vs",
        noise_var=1.0,
        ref=(0, 0),
        n_iter=10,
        sigma_t=10e-9,
        los_visible=mask,
    )
    assert masked.phi.shape == (6, 2)
    assert bool(np.all(masked.phi[~mask] == 0.0))
    assert bool(np.all(masked.tau[~mask] == 0.0))
    np.testing.assert_array_equal(masked.info["fitted"], mask)
    assert bool(np.all(np.isnan(masked.info["g_los"][~mask])))
    assert bool(np.all(np.isnan(masked.info["resid"][~mask])))
    assert bool(np.all(np.isfinite(masked.info["g_los"][mask])))
    assert bool(np.all(np.isfinite(masked.info["resid"][mask])))
    plain = bench.estimate_gauges(
        cfg,
        "los",
        data.y_clean,
        data.geom,
        grid,
        "vs",
        noise_var=1.0,
        ref=(0, 0),
        n_iter=10,
        sigma_t=10e-9,
    )
    np.testing.assert_array_equal(masked.phi[mask], plain.phi[mask])
    np.testing.assert_array_equal(masked.tau[mask], plain.tau[mask])
    with pytest.raises(ValueError):
        bench.estimate_gauges(
            cfg,
            "los",
            data.y_clean,
            data.geom,
            grid,
            "vs",
            noise_var=1.0,
            ref=(0, 0),
            n_iter=10,
            sigma_t=10e-9,
            los_visible=np.ones((6, 3), dtype=bool),
        )


def test_los_fit_summary() -> None:
    """The LoS amplitude summary matches the hand-computed dB errors."""
    info = {"fitted": [[True], [False], [True]], "g_los": [[0.5], [float("nan")], [0.0]]}
    summary = bench.los_fit_summary(info, np.array([[-6.0], [float("nan")], [1.0]]))
    assert summary["g_los_db"][0][0] == pytest.approx(20.0 * math.log10(0.5))
    assert summary["g_los_db"][1] == [None]
    assert summary["g_los_db"][2] == [None]
    assert summary["amp_error_db"][0][0] == pytest.approx(20.0 * math.log10(0.5) + 6.0)
    assert summary["amp_error_db"][1] == [None]
    assert summary["amp_error_db"][2] == [None]
    assert summary["amp_error_db_max"] == pytest.approx(abs(20.0 * math.log10(0.5) + 6.0))
    assert summary["num_fitted"] == 2
    plain = bench.los_fit_summary(info, None)
    assert plain["amp_error_db"] is None
    assert plain["amp_error_db_max"] is None
    with pytest.raises(ValueError):
        bench.los_fit_summary(info, np.zeros((3, 2)))


def test_gauge_error_summary_per_stratum_phase() -> None:
    """Each LoS stratum removes its own global phase, not the joint one."""
    cfg = get_config("IDP-N")
    used = (
        np.array([[0.3], [0.3], [0.3], [0.0]]),
        np.array([[1e-9], [0.0], [2e-9], [0.0]]),
    )
    truth = (np.zeros((4, 1)), np.zeros((4, 1)))
    summary = bench.gauge_error_summary(
        cfg,
        used,
        truth,
        1e-6,
        ("phi", "tau"),
        los_visible=np.array([[True], [True], [True], [False]]),
        los_model_error=np.array([[0.01], [np.nan], [-0.02], [np.nan]]),
    )
    visible = summary["by_los"]["los_visible"]
    assert visible["phase_deg"] == pytest.approx([0, 0, 0], abs=1e-9)
    assert visible["phase_max_deg"] <= 1e-9
    assert visible["captures"] == [[0, 0], [1, 0], [2, 0]]
    assert visible["delay_ns"] == pytest.approx([1, 0, 2])
    assert visible["los_model_error_deg"][0] == pytest.approx(math.degrees(0.01))
    assert visible["los_model_error_deg"][1] is None
    assert visible["los_model_error_deg"][2] == pytest.approx(math.degrees(-0.02))
    blocked = summary["by_los"]["los_blocked"]
    assert blocked["captures"] == [[3, 0]]
    assert blocked["phase_deg"] == pytest.approx([0])
    assert summary["phase_max_deg"] > 5
