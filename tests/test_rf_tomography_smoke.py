"""Unit tests for the heavy CI tomography smoke (T22)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from test_rf_tomography_gt_app import _make_dataset

from plateau_rt.application import rf_tomography_gt as gt_app
from plateau_rt.application import rf_tomography_io as tio
from plateau_rt.application import rf_tomography_smoke as smoke
from plateau_rt.domain.rf_tomography.geometry import VoxelGrid


@pytest.fixture(scope="module")
def dataset_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the mirror-scene dataset with tomography GT once."""
    root = tmp_path_factory.mktemp("smoke") / "dataset"
    _make_dataset(root)
    gt_app.write_tomography_gt(root)
    return root


def test_classify_and_exit_code() -> None:
    """Classify maps outcomes with the expected-failure table."""
    expected = {"a": "reason"}
    assert smoke.classify("b", True, expected) == "pass"
    assert smoke.classify("b", False, expected) == "fail"
    assert smoke.classify("a", False, expected) == "xfail"
    assert smoke.classify("a", True, expected) == "xpass"
    assert smoke.exit_code([smoke.Check("a", "xfail", None, None)]) == 0
    assert smoke.exit_code([smoke.Check("a", "xpass", None, None)]) == 0
    assert smoke.exit_code([smoke.Check("a", "pass", None, None)]) == 0
    assert smoke.exit_code([smoke.Check("a", "fail", None, None)]) == 1


def test_min_distance_and_localisation() -> None:
    """Localisation gates are inclusive and honour expected failures."""
    assert smoke.min_distance(np.zeros((0, 3)), np.zeros(3)) == math.inf
    assert smoke.min_distance(np.array([[3.0, 4.0, 0.0]]), np.zeros(3)) == pytest.approx(5.0)
    gate = smoke.GATE_FACTOR * 2.0
    assert smoke.check_localisation("n", np.array([[2.0, 0.0, 0.0]]), np.zeros(3), gate).status == (
        "pass"
    )
    assert smoke.check_localisation(
        "n", np.array([[gate, 0.0, 0.0]]), np.zeros(3), gate
    ).status == ("pass")
    assert smoke.check_localisation("n", np.array([[2.2, 0.0, 0.0]]), np.zeros(3), gate).status == (
        "fail"
    )
    expected = {"n": "reason"}
    assert smoke.check_localisation(
        "n", np.array([[5.0, 0.0, 0.0]]), np.zeros(3), gate, expected
    ).status == ("xfail")
    assert smoke.check_localisation(
        "n", np.array([[1.0, 0.0, 0.0]]), np.zeros(3), gate, expected
    ).status == ("xpass")


def test_negative_control() -> None:
    """The negative control passes only when the calibrated maps hit."""
    assert smoke.check_negative_control("n", 5.0, 1.0, 1.0, 2.078).status == "pass"
    assert smoke.check_negative_control("n", 1.0, 1.0, 1.0, 2.078).status == "fail"
    assert smoke.check_negative_control("n", 5.0, 3.0, 1.0, 2.078).status == "fail"
    assert smoke.check_negative_control("n", 5.0, 1.0, 3.0, 2.078).status == "fail"


def _gauge_payload(
    phase: list[float] | None,
    delay: list[float],
    eps: list[float | None] | None,
    fitted: list[list[bool]],
) -> dict[str, object]:
    """Build a hand-made gauge payload with one LoS-visible stratum."""
    entry: dict[str, object] = {
        "num": 1,
        "captures": [[0, 0]],
        "delay_ns": list(delay),
        "los_model_error_deg": None if eps is None else list(eps),
    }
    if phase is not None:
        entry["phase_deg"] = list(phase)
    return {"by_los": {"los_visible": entry}, "los_fit": {"fitted": fitted}}


def test_check_gauge_los() -> None:
    """Per-capture gauge gates are strict with model-error slack."""
    visible = np.ones((1, 1), dtype=bool)
    assert smoke.check_gauge_los(
        "n", _gauge_payload([10.05], [0.5], [0.1], [[True]]), visible
    ).status == ("pass")
    assert smoke.check_gauge_los(
        "n", _gauge_payload([10.05], [0.5], [0.0], [[True]]), visible
    ).status == ("fail")
    assert smoke.check_gauge_los(
        "n", _gauge_payload([10.05], [0.5], [None], [[True]]), visible
    ).status == ("fail")
    assert smoke.check_gauge_los(
        "n", _gauge_payload([1.0], [0.99], [0.0], [[True]]), visible
    ).status == ("pass")
    assert smoke.check_gauge_los(
        "n", _gauge_payload([1.0], [1.0], [0.0], [[True]]), visible
    ).status == ("fail")
    assert smoke.check_gauge_los(
        "n", _gauge_payload([10.05], [0.5], [0.1], [[False]]), visible
    ).status == ("fail")
    assert smoke.check_gauge_los(
        "n", _gauge_payload(None, [0.5], [0.0], [[True]]), visible
    ).status == ("pass")


def test_check_los_amplitude() -> None:
    """The amplitude gate accepts sub-dB fits on the fitted captures."""
    visible = np.array([[True], [True]])
    good = {"los_fit": {"fitted": [[True], [True]], "amp_error_db_max": 0.9, "num_fitted": 2}}
    assert smoke.check_los_amplitude("n", good, visible).status == "pass"
    bad = {"los_fit": {"fitted": [[True], [True]], "amp_error_db_max": 1.1, "num_fitted": 2}}
    assert smoke.check_los_amplitude("n", bad, visible).status == "fail"
    assert smoke.check_los_amplitude("n", None, visible).status == "fail"
    assert smoke.check_los_amplitude(
        "n",
        {"los_fit": {"fitted": [[True], [False]], "amp_error_db_max": 0.1, "num_fitted": 1}},
        visible,
    ).status == ("fail")


def test_check_sync_pairs() -> None:
    """Sync invariance compares maps relatively and detections absolutely."""
    recon = {
        "map": np.array([1.0, 2.0, 4.0]),
        "detections": np.array([[1.0, 0.0, 0.0]]),
    }
    assert smoke.check_sync_pairs("n", [(recon, dict(recon))]).status == "pass"
    perturbed = dict(recon)
    perturbed["map"] = np.array([1.0 + 1e-6 * 4.0, 2.0, 4.0])
    assert smoke.check_sync_pairs("n", [(recon, perturbed)]).status == "fail"
    other = dict(recon)
    other["detections"] = np.zeros((2, 3))
    assert smoke.check_sync_pairs("n", [(recon, other)]).status == "fail"
    assert smoke.check_sync_pairs("n", []).status == "fail"


def test_smoke_plan() -> None:
    """The L2 plan adds the building box; other scenes have three runs."""
    runs = smoke.smoke_plan(np.array(smoke.L2_BS_POSITIONS))
    assert [run.name for run in runs] == ["bv", "vs_bs_000", "vs_bs_001", "vs_building"]
    assert [run.kind for run in runs] == ["bv", "bs", "bs", "building"]
    for run in runs:
        if run.kind != "bs":
            continue
        assert run.grid_center is not None and run.grid_half_size is not None
        grid = VoxelGrid.from_bounds(
            np.asarray(run.grid_center) - np.asarray(run.grid_half_size),
            np.asarray(run.grid_center) + np.asarray(run.grid_half_size),
            2.0,
        )
        assert run.bs is not None
        bs_pos = np.array(smoke.L2_BS_POSITIONS)[run.bs[0]]
        image = np.array([bs_pos[0], bs_pos[1], -bs_pos[2]])
        for point in (bs_pos, image):
            assert float(np.min(np.linalg.norm(grid.centers() - point, axis=1))) < 1e-9
    building = runs[-1]
    assert building.grid_center is not None and building.grid_half_size is not None
    grid = VoxelGrid.from_bounds(
        np.asarray(building.grid_center) - np.asarray(building.grid_half_size),
        np.asarray(building.grid_center) + np.asarray(building.grid_half_size),
        2.0,
    )
    inside = np.array([-12.0, -40.0, 20.0])
    assert float(np.min(np.linalg.norm(grid.centers() - inside, axis=1))) < 1e-9
    lower = np.asarray(building.grid_center) - np.asarray(building.grid_half_size)
    upper = np.asarray(building.grid_center) + np.asarray(building.grid_half_size)
    for outside in (np.array([60.0, -40.0, 20.0]), np.array([60.0, -40.0, -20.0])):
        assert not bool(np.all(outside >= lower - 1e-9) and np.all(outside <= upper + 1e-9))

    from rf_tomography_gt_fixtures import BS_POS

    mirror_runs = smoke.smoke_plan(np.array(BS_POS))
    assert [run.name for run in mirror_runs] == ["bv", "vs_bs_000", "vs_bs_001"]
    assert "vs_building" not in [run.name for run in mirror_runs]


def test_smoke_end_to_end(dataset_root: Path, tmp_path: Path) -> None:
    """Two small smoke runs evaluate, report and refuse a second run."""
    out = tmp_path / "smoke-out"
    plan = [
        smoke.SmokeRun(
            "bv",
            "bv",
            "bv",
            None,
            (0, 0, 5),
            (4, 4, 2),
            ("I-S", "I-N", "IDP-N", "ID-N"),
            ("none", "los", "oracle"),
        ),
        smoke.SmokeRun(
            "vs_bs_000",
            "bs",
            "vs",
            (0,),
            (-40, 5, 0),
            (4, 4, 29),
            ("I-S", "I-N", "ID-S", "IDP-S", "IDP-N"),
            ("none", "los", "oracle"),
        ),
    ]
    smoke.run_smoke(dataset_root, out, workers=1, num_bins=32, plan=plan)
    checks = smoke.evaluate_smoke(out)
    summary_path, report_path = smoke.write_smoke_report(out, checks)
    assert summary_path.exists()
    assert report_path.exists()
    names = [check.name for check in checks]
    for wanted in (
        "schema:bv",
        "schema:vs_bs_000",
        "sync_invariance:bv",
        "sync_invariance:vs_bs_000",
        "los_e1:I-S:bs_000",
        "mirror:ID-S",
        "negative_control:bs_000",
        "gauge_los:IDP-N:ideal-N",
        "los_amplitude:ID-N:ideal-N",
        "runtime",
    ):
        assert wanted in names, names
    by_name = {check.name: check for check in checks}
    assert "all_nodes" in by_name
    assert by_name["all_nodes"].status == "fail"
    for check in checks:
        assert check.status in ("pass", "fail", "xfail", "xpass")
    for wanted in (
        "schema:bv",
        "schema:vs_bs_000",
        "sync_invariance:bv",
        "sync_invariance:vs_bs_000",
    ):
        assert by_name[wanted].status == "pass", (wanted, by_name[wanted].measured)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert sum(summary["counts"].values()) == len(checks)
    report = report_path.read_text(encoding="utf-8")
    assert "# Tomography heavy smoke (T22)" in report
    for name in names:
        assert name in report
    with pytest.raises(FileExistsError):
        smoke.run_smoke(dataset_root, out, workers=1, num_bins=32, plan=plan)

    # the mirror check scores the E1 detections against the GT ground image of BS 0
    vs_dir = out / "vs_bs_000"
    rows = tio.read_results(vs_dir / tio.RESULTS_FILE)
    row = next(
        r for r in rows if r["config"] == "IDP-S" and r["stage"] == "E1" and r["strategy"] == "none"
    )
    with np.load(vs_dir / row["recon"]) as payload:
        detections = np.asarray(payload["detections"])
    image = smoke.ground_image_target(
        gt_app.load_tomography_gt(dataset_root / "tomography_gt.npz"), 0
    )
    assert image is not None
    assert by_name["mirror:IDP-S"].measured["distance_m"] == pytest.approx(
        smoke.min_distance(detections, image)
    )

    # a non-finite recon map and an error row fail the schema check
    rows = tio.read_results(out / "bv" / tio.RESULTS_FILE)
    row = next(r for r in rows if r["status"] == "ok" and r["stage"] == "E1")
    path = out / "bv" / row["recon"]
    with np.load(path) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    arrays["map"] = np.full_like(arrays["map"], np.nan)
    np.savez_compressed(path, **arrays)
    again = {check.name: check for check in smoke.evaluate_smoke(out)}
    assert again["schema:bv"].status == "fail"
    assert again["schema:vs_bs_000"].status == "pass"
    lines = (out / "vs_bs_000" / tio.RESULTS_FILE).read_text(encoding="utf-8").splitlines()
    broken = json.loads(lines[0])
    broken.update({"status": "error", "reason": "boom", "recon": None})
    lines[0] = json.dumps(broken)
    (out / "vs_bs_000" / tio.RESULTS_FILE).write_text("\n".join(lines), encoding="utf-8")
    again = {check.name: check for check in smoke.evaluate_smoke(out)}
    assert again["schema:vs_bs_000"].status == "fail"


def test_image_targets(dataset_root: Path) -> None:
    """Ground and building image targets follow the GT object labels."""
    gt = gt_app.load_tomography_gt(dataset_root / "tomography_gt.npz")
    ground = smoke.ground_image_target(gt, 0)
    assert ground is not None
    np.testing.assert_allclose(ground, [-40.0, 5.0, -25.0], atol=1e-4)
    everywhere = (np.full(3, -1e9), np.full(3, 1e9))
    targets = smoke.building_targets(gt, 0, *everywhere)
    assert targets.shape[0] >= 1
    assert float(np.min(np.linalg.norm(targets - ground, axis=1))) > 1.0
    detectable = gt_app.vs_detectable(gt)
    names = [str(name) for name in gt["path_object_names"]]
    for point in targets:
        m = int(np.argmin(np.linalg.norm(gt["vs_pos"] - point, axis=1)))
        assert int(gt["vs_bs"][m]) == 0 and int(gt["vs_order"][m]) >= 1 and detectable[m]
        assert any(names[o] != "ground_plane" for o in gt["vs_objects"][m] if o >= 0)
    first = targets[0]
    boxed = smoke.building_targets(gt, 0, first - 1e-3, first + 1e-3)
    np.testing.assert_allclose(boxed, first[None, :])
    assert smoke.building_targets(gt, 0, first + 1.0, first + 2.0).shape == (0, 3)
