import json

import numpy as np
import pytest
from click.testing import CliRunner
from rf_manifest_fixtures import (
    BANDWIDTH_HZ,
    CARRIER_HZ,
    write_v2_dataset,
    write_v3_dataset,
)

from plateau_rt.application.rf_dataset_manifest import ManifestError
from plateau_rt.domain.rf_camera.camera import generate_ring_views
from plateau_rt.experimental.compare_direct_path import (
    compare_dataset,
    direct_path_ratio_prediction,
    main,
    segment_intersects_aabb,
    tr38901_amplitude,
)
from plateau_rt.experimental.rf_scatterer_fit import ApertureSpec, direct_path_cfr

ROWS = 4
COLS = 4
NUM_FREQ = 8
TARGET = (5.0, 5.0, 5.0)


def _ring_views():
    return generate_ring_views(target=TARGET, radius_m=30.0, ue_height_m=1.5, num_views=3)


def _spec():
    return ApertureSpec(
        rows=ROWS,
        cols=COLS,
        carrier_hz=CARRIER_HZ,
        bandwidth_hz=BANDWIDTH_HZ,
        num_freq=NUM_FREQ,
        spacing_lambda=0.5,
    )


def _write_v3_model_dataset(root, scales=None):
    """Write a v3 fixture, then overwrite apertures with scaled model direct paths.

    ``scales`` maps ``(view_index, bs_index)`` to an amplitude scale (default 1).
    """
    scales = scales or {}
    views = _ring_views()
    manifest = write_v3_dataset(root, views=views, num_bs=2, rows=ROWS, cols=COLS, bins=NUM_FREQ)
    spec = _spec()
    for view_index, view in enumerate(views):
        per_bs = []
        for bs_index, bs in enumerate(manifest["base_stations"]):
            bs_position = tuple(bs["position_m"])
            _, _, predicted = direct_path_ratio_prediction(
                bs_position, tuple(bs["look_at_m"]), view.position, view.orientation
            )
            model = direct_path_cfr(
                spec, bs_position, view.position, view.orientation, plane_wave=True
            )
            front = scales.get((view_index, bs_index), 1.0) * predicted * model
            per_bs.append(np.stack([front, np.zeros_like(front)], axis=0))
        aperture = np.stack(per_bs, axis=0)
        assert aperture.shape == (2, 2, ROWS, COLS, NUM_FREQ)
        np.save(root / f"views/{view.view_id}/rf/aperture_cfr.npy", aperture.astype(np.complex64))
    return manifest, views


def _write_v2_model_dataset(root, scales=None):
    """Write a v2 fixture, then overwrite apertures with scaled model direct paths.

    ``scales`` maps ``view_index`` to an amplitude scale (default 1).
    """
    scales = scales or {}
    views = _ring_views()
    manifest = write_v2_dataset(root, views=views, rows=ROWS, cols=COLS, bins=NUM_FREQ)
    spec = _spec()
    bs_position = tuple(manifest["config"]["tx_position"])
    bs_look_at = tuple(manifest["config"]["tx_look_at"])
    for view_index, view in enumerate(views):
        _, _, predicted = direct_path_ratio_prediction(
            bs_position, bs_look_at, view.position, view.orientation
        )
        model = direct_path_cfr(spec, bs_position, view.position, view.orientation, plane_wave=True)
        front = scales.get(view_index, 1.0) * predicted * model
        aperture = np.stack([front, np.zeros_like(front)], axis=0)
        assert aperture.shape == (2, ROWS, COLS, NUM_FREQ)
        np.save(root / f"views/{view.view_id}/rf/aperture_cfr.npy", aperture.astype(np.complex64))
    return manifest, views


def _rewrite_manifest(root, mutate):
    path = root / "dataset_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mutate(manifest)
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def _write_v3_multipath_dataset(root, factor=0.4):
    """Write a v3 model dataset with a weak delayed copy on (view 0, bs_001).

    The second component is ``factor * predicted`` times the direct-path CFR
    from ``mirror_bs`` (bs_001 mirrored in the plane x = 10). With
    ``factor=0.4`` the pair keeps a near-predicted amplitude (db ~+0.20 dB,
    within 1 dB) while the correlation drops to ~0.977 (< 0.999).
    """
    manifest, views = _write_v3_model_dataset(root)
    spec = _spec()
    bs_position = tuple(manifest["base_stations"][1]["position_m"])
    bs_look_at = tuple(manifest["base_stations"][1]["look_at_m"])
    mirror_bs = (20.0 - bs_position[0], bs_position[1], bs_position[2])
    view = views[0]
    _, _, predicted = direct_path_ratio_prediction(
        bs_position, bs_look_at, view.position, view.orientation
    )
    mirror_model = direct_path_cfr(
        spec, mirror_bs, view.position, view.orientation, plane_wave=True
    )
    aperture_path = root / f"views/{view.view_id}/rf/aperture_cfr.npy"
    aperture = np.load(aperture_path)
    aperture[1, 0] = aperture[1, 0] + (factor * predicted * mirror_model).astype(np.complex64)
    np.save(aperture_path, aperture)
    return manifest, views


def test_compare_dataset_v3_per_bs_pairs(tmp_path):
    _, views = _write_v3_model_dataset(tmp_path, scales={(1, 1): 0.1})
    report = compare_dataset(tmp_path)

    assert report["schema_version"] == 3
    assert report["bs_ids"] == ["bs_000", "bs_001"]
    assert [(row["view_id"], row["bs_id"]) for row in report["pairs"]] == [
        (view.view_id, bs_id) for view in views for bs_id in ("bs_000", "bs_001")
    ]
    for row in report["pairs"]:
        assert row["correlation"] > 0.99999

    for row in report["pairs"]:
        if (row["view_id"], row["bs_id"]) == (views[1].view_id, "bs_001"):
            assert row["ratio_over_predicted"] == pytest.approx(0.1, abs=1e-5)
            assert row["outlier"] is True
        else:
            assert row["ratio_over_predicted"] == pytest.approx(1.0, abs=1e-5)
            assert row["outlier"] is False

    assert report["outliers"] == [{"view_id": views[1].view_id, "bs_id": "bs_001"}]
    assert report["per_bs"]["bs_001"]["outliers"] == [views[1].view_id]
    assert report["per_bs"]["bs_000"]["outliers"] == []


def test_compare_dataset_v2_through_reader(tmp_path):
    _, views = _write_v2_model_dataset(tmp_path, scales={2: 0.1})
    report = compare_dataset(tmp_path)

    assert report["schema_version"] == 2
    assert report["bs_ids"] == ["bs_000"]
    assert [row["view_id"] for row in report["pairs"]] == [view.view_id for view in views]
    for row in report["pairs"]:
        assert row["correlation"] > 0.99999

    assert report["pairs"][2]["ratio_over_predicted"] == pytest.approx(0.1, abs=1e-5)
    assert report["pairs"][2]["outlier"] is True
    assert report["pairs"][0]["outlier"] is False
    assert report["pairs"][1]["outlier"] is False
    assert report["outliers"] == [{"view_id": views[2].view_id, "bs_id": "bs_000"}]


def test_compare_dataset_blocked_segments_per_bs(tmp_path):
    manifest, views = _write_v3_model_dataset(tmp_path)

    unblocked = compare_dataset(tmp_path, blocker_aabb=((0.0, 0.0, 0.0), (10.0, 10.0, 10.0)))
    assert [row["segment_blocked"] for row in unblocked["pairs"]] == [False] * 6

    # A 2 m cube around the (view 0, bs_000) segment at t = 0.8; misses the rest.
    start = np.asarray(manifest["base_stations"][0]["position_m"], dtype=np.float64)
    end = np.asarray(views[0].position, dtype=np.float64)
    point = start + 0.8 * (end - start)
    box_min: tuple[float, float, float] = (
        float(point[0] - 1.0),
        float(point[1] - 1.0),
        float(point[2] - 1.0),
    )
    box_max: tuple[float, float, float] = (
        float(point[0] + 1.0),
        float(point[1] + 1.0),
        float(point[2] + 1.0),
    )
    blocked = compare_dataset(tmp_path, blocker_aabb=(box_min, box_max))
    assert [row["segment_blocked"] for row in blocked["pairs"]] == [
        True,
        False,
        False,
        False,
        False,
        False,
    ]

    assert [row["segment_blocked"] for row in compare_dataset(tmp_path)["pairs"]] == [None] * 6


def test_compare_dataset_zero_energy_pair(tmp_path):
    _, views = _write_v3_model_dataset(tmp_path)
    aperture_path = tmp_path / f"views/{views[2].view_id}/rf/aperture_cfr.npy"
    aperture = np.load(aperture_path)
    aperture[0] = 0
    np.save(aperture_path, aperture)

    report = compare_dataset(tmp_path)
    row = next(
        r for r in report["pairs"] if r["view_id"] == views[2].view_id and r["bs_id"] == "bs_000"
    )
    assert row["no_energy"] is True
    assert row["outlier"] is True
    assert row["ratio_over_predicted"] is None
    assert row["ratio_over_predicted_db"] is None
    json.dumps(report, allow_nan=False)


def test_compare_dataset_rejects_unsupported_schema(tmp_path):
    _write_v3_model_dataset(tmp_path)
    _rewrite_manifest(tmp_path, lambda manifest: manifest.update(schema_version=1))

    with pytest.raises(ManifestError):
        compare_dataset(tmp_path)


def test_compare_dataset_rejects_mismatched_spacing(tmp_path):
    _write_v3_model_dataset(tmp_path)

    def _widen(manifest):
        manifest["config"]["vertical_spacing_lambda"] = 0.7

    _rewrite_manifest(tmp_path, _widen)

    with pytest.raises(ValueError, match="vertical_spacing_lambda"):
        compare_dataset(tmp_path)


def test_compare_dataset_rejects_frequency_grid_mismatch(tmp_path):
    _write_v3_model_dataset(tmp_path)

    def _retune(manifest):
        manifest["config"]["bandwidth_hz"] = 80e6

    _rewrite_manifest(tmp_path, _retune)

    with pytest.raises(ValueError, match="frequency grid"):
        compare_dataset(tmp_path)


def test_cli_reports_pairs_and_clean_error(tmp_path):
    _write_v3_model_dataset(tmp_path)
    json_out = tmp_path / "report.json"
    runner = CliRunner()
    result = runner.invoke(main, ["--dataset", str(tmp_path), "--json-out", str(json_out)])
    assert result.exit_code == 0
    assert "ue_000000 bs_001:" in result.output
    assert "bs_001: correlation range" in result.output
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert len(payload["pairs"]) == 6

    bad = tmp_path / "bad"
    bad.mkdir()
    _write_v3_model_dataset(bad)
    _rewrite_manifest(bad, lambda manifest: manifest.update(schema_version=1))
    failed = runner.invoke(main, ["--dataset", str(bad)])
    assert failed.exit_code == 1
    assert "Error" in failed.output
    assert "Traceback" not in failed.output
    assert isinstance(failed.exception, SystemExit)

    empty = tmp_path / "empty"
    empty.mkdir()
    missing = runner.invoke(main, ["--dataset", str(empty)])
    assert missing.exit_code == 1
    assert "dataset_manifest.json" in missing.output
    assert isinstance(missing.exception, SystemExit)


def test_tr38901_pattern_amplitude():
    assert tr38901_amplitude(np.pi / 2.0, 0.0) == pytest.approx(np.sqrt(10.0**0.8))
    assert tr38901_amplitude(np.pi / 2.0, np.deg2rad(32.5)) == pytest.approx(np.sqrt(10.0**0.5))
    assert tr38901_amplitude(np.pi / 2.0, np.pi) == pytest.approx(np.sqrt(10.0 ** (-2.2)))


def test_segment_intersects_aabb():
    box_min = (0.0, 0.0, 0.0)
    box_max = (10.0, 10.0, 10.0)

    assert segment_intersects_aabb((-50.0, -50.0, 30.0), (26.21, 26.21, 1.5), box_min, box_max)
    assert not segment_intersects_aabb((-20.0, 20.0, 5.0), (20.0, 20.0, 5.0), box_min, box_max)
    assert not segment_intersects_aabb((-50.0, -50.0, 30.0), (-5.0, -5.0, 15.0), box_min, box_max)


def test_compare_dataset_flags_multipath_pair(tmp_path):
    _, views = _write_v3_multipath_dataset(tmp_path, factor=0.4)
    report = compare_dataset(tmp_path)

    target_id = views[0].view_id
    row = next(r for r in report["pairs"] if r["view_id"] == target_id and r["bs_id"] == "bs_001")
    # The old ratio-only test misses this pair: amplitude stays within 1 dB.
    assert row["ratio_over_predicted_db"] is not None
    assert abs(row["ratio_over_predicted_db"]) <= 1.0
    assert row["correlation"] < 0.999
    assert row["low_correlation"] is True
    assert row["outlier"] is True
    assert report["outliers"] == [{"view_id": target_id, "bs_id": "bs_001"}]
    assert report["min_correlation_threshold"] == pytest.approx(0.999)
    for other in report["pairs"]:
        if (other["view_id"], other["bs_id"]) == (target_id, "bs_001"):
            continue
        assert other["low_correlation"] is False
        assert other["outlier"] is False


def test_compare_dataset_reports_num_valid_paths(tmp_path):
    _, views = _write_v3_model_dataset(tmp_path)
    # The shared fixture writes a canonical path GT (plus path_schema.json)
    # with two valid paths per (view, BS) pair.
    default_report = compare_dataset(tmp_path)
    assert [row["num_valid_paths"] for row in default_report["pairs"]] == [2] * 6

    gt_path = tmp_path / "path_geometry_gt.npz"
    num_views = len(views)
    num_bs = 2
    valid = np.zeros((num_views, num_bs, 3), dtype=bool)
    valid[:, :, 0] = True
    valid[1, 0, 1:] = True
    tau = np.zeros((num_views, num_bs, 3), dtype=np.float32)
    np.savez_compressed(gt_path, valid=valid, tau=tau)

    report = compare_dataset(tmp_path)
    for row in report["pairs"]:
        if row["view_id"] == views[1].view_id and row["bs_id"] == "bs_000":
            assert row["num_valid_paths"] == 3
        else:
            assert row["num_valid_paths"] == 1

    gt_path.unlink()
    missing = compare_dataset(tmp_path)
    assert [row["num_valid_paths"] for row in missing["pairs"]] == [None] * 6


def test_num_valid_paths_needs_view_bs_path_axes(tmp_path):
    _write_v3_model_dataset(tmp_path)
    schema_path = tmp_path / "path_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema["arrays"]["valid"]["axes"] = ["view", "rx_ant", "bs", "tx_ant", "path"]
    schema_path.write_text(json.dumps(schema), encoding="utf-8")
    native = compare_dataset(tmp_path)
    assert [row["num_valid_paths"] for row in native["pairs"]] == [None] * 6

    schema_path.write_text("{not json", encoding="utf-8")
    unreadable = compare_dataset(tmp_path)
    assert [row["num_valid_paths"] for row in unreadable["pairs"]] == [None] * 6


def test_cli_min_correlation_option(tmp_path):
    _, views = _write_v3_multipath_dataset(tmp_path, factor=0.4)
    target_prefix = f"{views[0].view_id} bs_001:"
    runner = CliRunner()

    default = runner.invoke(main, ["--dataset", str(tmp_path)])
    assert default.exit_code == 0
    assert "LOW_CORR" in default.output
    assert any(target_prefix in line and "LOW_CORR" in line for line in default.output.splitlines())

    relaxed = runner.invoke(main, ["--dataset", str(tmp_path), "--min-correlation", "0.5"])
    assert relaxed.exit_code == 0
    assert "LOW_CORR" not in relaxed.output
