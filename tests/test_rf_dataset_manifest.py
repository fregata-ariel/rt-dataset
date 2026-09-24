"""Tests for the typed RF dataset manifest reader (schema v2 + v3)."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from rf_manifest_fixtures import (
    BANDWIDTH_HZ,
    CARRIER_HZ,
    centred_frequency_offsets,
    write_v2_dataset,
    write_v3_dataset,
)

from plateau_rt.application.rf_dataset_manifest import (
    APERTURE_CFR_AXIS_ORDER,
    MANIFEST_FILE_NAME,
    PER_BS_ARTIFACT_KEYS,
    V2_APERTURE_CFR_AXIS_ORDER,
    ManifestError,
    load_rf_dataset_manifest,
    parse_rf_dataset_manifest,
)

ROWS, COLS, BINS = 2, 3, 4


def test_manifest_error_is_value_error():
    assert issubclass(ManifestError, ValueError)


def test_v3_round_trip(tmp_path: Path):
    manifest_dict = write_v3_dataset(tmp_path, num_views=2, num_bs=2)
    manifest = load_rf_dataset_manifest(tmp_path)

    assert manifest.schema_version == 3
    assert manifest.mode == "multibs_multiue_rf_camera_dataset"
    assert manifest.source_scene == "mock_scene.xml"
    assert manifest.num_views == 2
    assert manifest.num_bs == 2
    assert manifest.view_ids == ("ue_000000", "ue_000001")
    assert manifest.bs_ids == ("bs_000", "bs_001")
    assert manifest.carrier_frequency_hz == pytest.approx(CARRIER_HZ)
    assert manifest.rx_rows == ROWS
    assert manifest.rx_cols == COLS
    assert manifest.num_frequency_bins == BINS
    assert manifest.hemispheres == ("front", "back")
    assert manifest.stored_aperture_axis_order == APERTURE_CFR_AXIS_ORDER
    assert manifest.aperture_cfr_axis_order == APERTURE_CFR_AXIS_ORDER
    assert manifest.aperture_cfr_shape == (2, 2, ROWS, COLS, BINS)
    assert manifest.camera_model_path == tmp_path / "camera_model.npz"
    assert manifest.delay_resolution_s == pytest.approx(1.0 / BANDWIDTH_HZ)
    assert manifest.unambiguous_delay_s == pytest.approx(BINS / BANDWIDTH_HZ)

    expected_offsets = centred_frequency_offsets(BINS)
    assert manifest.frequency_offsets_hz.dtype == np.float64
    assert np.array_equal(manifest.frequency_offsets_hz, expected_offsets)
    assert manifest.absolute_frequencies_hz.dtype == np.float64
    assert np.array_equal(manifest.absolute_frequencies_hz, CARRIER_HZ + expected_offsets)

    bs_000 = manifest.base_station("bs_000")
    assert bs_000.index == 0
    assert bs_000.position_m == (-50.0, -50.0, 30.0)
    assert bs_000.look_at_m == (5.0, 5.0, 5.0)
    bs_001 = manifest.base_station("bs_001")
    assert bs_001.index == 1
    assert bs_001.position_m == (60.0, 35.0, 25.0)

    assert manifest.path_geometry_gt is not None
    assert manifest.path_geometry_gt.path == tmp_path / "path_geometry_gt.npz"
    assert manifest.path_geometry_gt.axis_order == ("rx", "tx", "path")
    assert manifest.path_geometry_gt.note

    assert manifest.raw == manifest_dict
    parsed = parse_rf_dataset_manifest(manifest_dict, root=tmp_path)
    assert parsed.raw is manifest_dict

    for view in manifest.views:
        assert view.pose_path == tmp_path / f"views/{view.view_id}/pose.json"
        assert view.aperture_cfr_path == tmp_path / f"views/{view.view_id}/rf/aperture_cfr.npy"
        for path in view.artifacts.values():
            assert path.is_absolute()
            assert tmp_path in path.parents
        assert set(view.artifacts) == {"pose", "aperture_cfr"}
        assert len(view.bs) == 2
        stored = np.load(view.aperture_cfr_path)
        assert stored.shape == (2, 2, ROWS, COLS, BINS)
        for bs_index, entry in enumerate(view.bs):
            assert entry.view_id == view.view_id
            assert entry.bs_index == bs_index
            assert set(entry.artifacts) == set(PER_BS_ARTIFACT_KEYS)
            for path in entry.artifacts.values():
                assert path.is_absolute()
                assert tmp_path in path.parents
            expected_energy = {
                name: float(np.sum(np.abs(stored[bs_index, i]) ** 2))
                for i, name in enumerate(("front", "back"))
            }
            assert entry.hemisphere_energy == pytest.approx(expected_energy)
            assert entry.total_energy == pytest.approx(sum(expected_energy.values()))
            assert isinstance(entry.bs_in_front_hemisphere, bool)
            assert entry.bs_direction_local is not None
            assert len(entry.bs_direction_local) == 3

    pairs = list(manifest.pairs())
    assert [(view.view_id, entry.bs_id) for view, entry in pairs] == [
        ("ue_000000", "bs_000"),
        ("ue_000000", "bs_001"),
        ("ue_000001", "bs_000"),
        ("ue_000001", "bs_001"),
    ]

    view = manifest.view("ue_000001")
    assert view.index == 1
    assert view.bs_entry("bs_001").bs_id == "bs_001"
    with pytest.raises(ManifestError, match="view_id"):
        manifest.view("ue_999999")
    with pytest.raises(ManifestError, match="bs_id"):
        manifest.base_station("bs_999")
    with pytest.raises(ManifestError, match="bs_999"):
        view.bs_entry("bs_999")
    with pytest.raises(ManifestError, match="no view-level artifact"):
        view.artifact("nope")
    with pytest.raises(ManifestError, match="nope"):
        view.bs_entry("bs_000").artifact("nope")


def test_v3_load_aperture_cfr_exact(tmp_path: Path):
    write_v3_dataset(tmp_path)
    manifest = load_rf_dataset_manifest(tmp_path)
    for view in manifest.views:
        expected = np.load(view.aperture_cfr_path)
        assert np.array_equal(manifest.load_aperture_cfr(view), expected)
        assert np.array_equal(manifest.load_aperture_cfr(view.view_id), expected)


def test_v2_read_as_single_bs(tmp_path: Path):
    manifest_dict = write_v2_dataset(tmp_path)
    for view_dict in manifest_dict["views"]:
        view_dict["artifacts"]["optical_pinhole_rgba"] = (
            f"views/{view_dict['view_id']}/optical/pinhole_rgba.png"
        )
    (tmp_path / MANIFEST_FILE_NAME).write_text(json.dumps(manifest_dict, indent=2))
    manifest = load_rf_dataset_manifest(tmp_path)

    assert manifest.schema_version == 2
    assert manifest.num_bs == 1
    assert manifest.bs_ids == ("bs_000",)
    station = manifest.base_station("bs_000")
    assert station.index == 0
    assert station.position_m == tuple(manifest_dict["config"]["tx_position"])
    assert station.look_at_m == tuple(manifest_dict["config"]["tx_look_at"])
    assert manifest.stored_aperture_axis_order == V2_APERTURE_CFR_AXIS_ORDER
    assert manifest.aperture_cfr_axis_order == APERTURE_CFR_AXIS_ORDER
    assert manifest.aperture_cfr_shape == (1, 2, ROWS, COLS, BINS)
    assert manifest.path_geometry_gt is not None
    assert manifest.path_geometry_gt.path == tmp_path / "path_geometry_gt.npz"
    assert manifest.path_geometry_gt.axis_order == ("rx", "tx", "path")

    for view_dict, view in zip(manifest_dict["views"], manifest.views):
        assert len(view.bs) == 1
        entry = view.bs_entry("bs_000")
        assert entry.bs_index == 0
        assert entry.hemisphere_energy == pytest.approx(view_dict["hemisphere_energy"])
        assert entry.bs_in_front_hemisphere == view_dict["bs_in_front_hemisphere"]
        assert entry.bs_direction_local == pytest.approx(view_dict["bs_direction_local"])
        for key in PER_BS_ARTIFACT_KEYS:
            assert key in entry.artifacts
            assert key not in view.artifacts
        assert set(view.artifacts) == {"pose", "aperture_cfr", "optical_pinhole_rgba"}


def test_v2_load_aperture_cfr_adds_bs_axis(tmp_path: Path):
    write_v2_dataset(tmp_path)
    manifest = load_rf_dataset_manifest(tmp_path)
    view = manifest.view("ue_000000")
    stored = np.load(view.aperture_cfr_path)
    assert stored.shape == (2, ROWS, COLS, BINS)
    loaded = manifest.load_aperture_cfr("ue_000000")
    assert loaded.shape == (1, 2, ROWS, COLS, BINS)
    assert np.array_equal(loaded, stored[np.newaxis])


@pytest.mark.parametrize("as_str", [False, True])
@pytest.mark.parametrize("use_file", [False, True])
def test_load_accepts_dir_and_file(tmp_path: Path, use_file: bool, as_str: bool):
    write_v3_dataset(tmp_path)
    location: Path | str = tmp_path / MANIFEST_FILE_NAME if use_file else tmp_path
    if as_str:
        location = str(location)
    manifest = load_rf_dataset_manifest(location)
    assert manifest.num_views == 2
    assert manifest.num_bs == 2
    assert manifest.manifest_path == tmp_path / MANIFEST_FILE_NAME
    assert manifest.root == tmp_path


@pytest.mark.parametrize("version", [1, 4])
def test_unsupported_schema_version(tmp_path: Path, version: int):
    manifest_dict = write_v3_dataset(tmp_path)
    manifest_dict["schema_version"] = version
    with pytest.raises(ManifestError, match="schema_version"):
        parse_rf_dataset_manifest(manifest_dict, root=tmp_path)


def test_missing_and_empty_views(tmp_path: Path):
    manifest_dict = write_v3_dataset(tmp_path)
    missing = copy.deepcopy(manifest_dict)
    del missing["views"]
    with pytest.raises(ManifestError, match="views"):
        parse_rf_dataset_manifest(missing, root=tmp_path)
    empty = copy.deepcopy(manifest_dict)
    empty["views"] = []
    with pytest.raises(ManifestError, match="views"):
        parse_rf_dataset_manifest(empty, root=tmp_path)


def test_duplicate_view_id(tmp_path: Path):
    manifest_dict = write_v3_dataset(tmp_path)
    manifest_dict["views"][1]["view_id"] = manifest_dict["views"][0]["view_id"]
    with pytest.raises(ManifestError, match="view_id"):
        parse_rf_dataset_manifest(manifest_dict, root=tmp_path)


def test_v3_view_bs_reordered_and_too_short(tmp_path: Path):
    manifest_dict = write_v3_dataset(tmp_path)
    reordered = copy.deepcopy(manifest_dict)
    reordered["views"][0]["bs"] = reordered["views"][0]["bs"][::-1]
    with pytest.raises(ManifestError, match="bs"):
        parse_rf_dataset_manifest(reordered, root=tmp_path)
    short = copy.deepcopy(manifest_dict)
    short["views"][1]["bs"] = short["views"][1]["bs"][:1]
    with pytest.raises(ManifestError, match="bs_001"):
        parse_rf_dataset_manifest(short, root=tmp_path)


def test_base_stations_index_mismatch(tmp_path: Path):
    manifest_dict = write_v3_dataset(tmp_path)
    manifest_dict["base_stations"][1]["index"] = 0
    with pytest.raises(ManifestError, match="index"):
        parse_rf_dataset_manifest(manifest_dict, root=tmp_path)


def test_raw_observation_bs_ids_mismatch(tmp_path: Path):
    manifest_dict = write_v3_dataset(tmp_path)
    manifest_dict["raw_observation"]["bs_ids"] = ["bs_001", "bs_000"]
    with pytest.raises(ManifestError, match="bs_ids"):
        parse_rf_dataset_manifest(manifest_dict, root=tmp_path)


def test_wrong_axis_order(tmp_path: Path):
    manifest_dict = write_v3_dataset(tmp_path)
    manifest_dict["raw_observation"]["axis_order"] = list(V2_APERTURE_CFR_AXIS_ORDER)
    with pytest.raises(ManifestError, match="axis_order"):
        parse_rf_dataset_manifest(manifest_dict, root=tmp_path)


def test_frequency_length_mismatch_num_bins(tmp_path: Path):
    manifest_dict = write_v3_dataset(tmp_path)
    manifest_dict["config"]["num_frequency_bins"] = BINS + 1
    with pytest.raises(ManifestError, match="num_frequency_bins"):
        parse_rf_dataset_manifest(manifest_dict, root=tmp_path)


def test_absolute_frequencies_length_mismatch(tmp_path: Path):
    manifest_dict = write_v3_dataset(tmp_path)
    manifest_dict["absolute_frequencies_hz"] = manifest_dict["absolute_frequencies_hz"][:-1]
    with pytest.raises(ManifestError, match="absolute_frequencies_hz"):
        parse_rf_dataset_manifest(manifest_dict, root=tmp_path)


def test_non_finite_position(tmp_path: Path):
    manifest_dict = write_v3_dataset(tmp_path)
    manifest_dict["views"][0]["position_m"] = [float("inf"), 0.0, 0.0]
    with pytest.raises(ManifestError, match="position_m"):
        parse_rf_dataset_manifest(manifest_dict, root=tmp_path)


def test_v2_missing_tx_position(tmp_path: Path):
    manifest_dict = write_v2_dataset(tmp_path)
    del manifest_dict["config"]["tx_position"]
    with pytest.raises(ManifestError, match="tx_position"):
        parse_rf_dataset_manifest(manifest_dict, root=tmp_path)


def test_v2_missing_tx_look_at(tmp_path: Path):
    manifest_dict = write_v2_dataset(tmp_path)
    del manifest_dict["config"]["tx_look_at"]
    with pytest.raises(ManifestError, match="tx_look_at"):
        parse_rf_dataset_manifest(manifest_dict, root=tmp_path)


def test_load_aperture_cfr_wrong_shape(tmp_path: Path):
    write_v3_dataset(tmp_path)
    manifest = load_rf_dataset_manifest(tmp_path)
    view = manifest.view("ue_000000")
    np.save(view.aperture_cfr_path, np.zeros((2, 2, ROWS, COLS, BINS + 1), dtype=np.complex64))
    with pytest.raises(ManifestError, match="shape"):
        manifest.load_aperture_cfr(view)


def test_unknown_extra_keys_ignored(tmp_path: Path):
    manifest_dict = write_v3_dataset(tmp_path)
    manifest_dict["path_schema"] = {"version": 99}
    manifest_dict["observed"] = True
    manifest_dict["views"][0]["observed"] = {"by": "test"}
    manifest_dict["views"][0]["bs"][0]["observed"] = 1
    manifest = parse_rf_dataset_manifest(manifest_dict, root=tmp_path)
    assert manifest.num_views == 2
    assert manifest.view("ue_000000").bs_entry("bs_000").bs_id == "bs_000"


def test_optical_view_artifacts(tmp_path: Path):
    manifest_dict = write_v3_dataset(tmp_path)
    for view_dict in manifest_dict["views"]:
        view_dict["artifacts"]["optical_pinhole_rgba"] = (
            f"views/{view_dict['view_id']}/optical/pinhole_rgba.png"
        )
        view_dict["artifacts"]["optical_hemisphere_range_m"] = (
            f"views/{view_dict['view_id']}/optical/hemisphere_range_m.npy"
        )
    (tmp_path / MANIFEST_FILE_NAME).write_text(json.dumps(manifest_dict, indent=2))
    manifest = load_rf_dataset_manifest(tmp_path)
    for view in manifest.views:
        assert view.artifact("optical_pinhole_rgba") == (
            tmp_path / f"views/{view.view_id}/optical/pinhole_rgba.png"
        )
        assert view.artifact("optical_hemisphere_range_m") == (
            tmp_path / f"views/{view.view_id}/optical/hemisphere_range_m.npy"
        )


def test_missing_optionals_default_to_none(tmp_path: Path):
    manifest_dict = write_v3_dataset(tmp_path)
    for key in (
        "mode",
        "source_scene",
        "delay_resolution_s",
        "unambiguous_delay_s",
        "path_geometry_gt",
    ):
        manifest_dict.pop(key, None)
    manifest_dict["views"][0].pop("look_at_m", None)
    manifest_dict["views"][0].pop("orientation_rad", None)
    manifest_dict["views"][0]["bs"][0].pop("bs_direction_local", None)
    manifest_dict["views"][0]["bs"][0].pop("bs_in_front_hemisphere", None)
    manifest = parse_rf_dataset_manifest(manifest_dict, root=tmp_path)
    assert manifest.mode is None
    assert manifest.source_scene is None
    assert manifest.delay_resolution_s is None
    assert manifest.unambiguous_delay_s is None
    assert manifest.path_geometry_gt is None
    assert manifest.view("ue_000000").look_at_m is None
    assert manifest.view("ue_000000").orientation_rad is None
    entry = manifest.view("ue_000000").bs_entry("bs_000")
    assert entry.bs_direction_local is None
    assert entry.bs_in_front_hemisphere is None
