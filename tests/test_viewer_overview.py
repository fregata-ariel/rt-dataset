"""Tests for the eager overview deriver (plateau_rt.viewer.derive.overview)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from viewer_bundle_fixtures import BundleFixture, write_fixture_bundle
from viewer_fixtures import BANDWIDTH_HZ, BOX_CENTER_M, BS_POSITIONS_M, CARRIER_HZ

from plateau_rt.viewer.derive import DERIVER_MODULES, derive_eager, registered_derivers
from plateau_rt.viewer.derive.overview import DEFAULT_M1_IMAGE_AXES, OVERVIEW, build_overview
from plateau_rt.viewer.kinds import dataset_manifest_from_bytes
from plateau_rt.viewer.settings import ViewerSettings
from plateau_rt.viewer.store import Store
from plateau_rt.viewer.testing import assert_deterministic, bundle_members, derive_in_temp_store

BUILTIN_DERIVERS = registered_derivers()

OPTICAL_KEYS = [
    "optical_hemisphere_range_m",
    "optical_hemisphere_rgba",
    "optical_pinhole_depth_m",
    "optical_pinhole_range_m",
    "optical_pinhole_rgba",
]


@pytest.fixture(scope="module")
def v3_bundle(tmp_path_factory: pytest.TempPathFactory) -> BundleFixture:
    """Build the canonical v3 fixture bundle (with its truth) once per module."""
    root = tmp_path_factory.mktemp("overview-v3") / "bundle"
    return write_fixture_bundle(root, schema_version=3)


@pytest.fixture(scope="module")
def v3_root(v3_bundle: BundleFixture) -> Path:
    """Return the root of the canonical v3 fixture bundle."""
    return v3_bundle.root


@pytest.fixture(scope="module")
def v2_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the canonical v2 fixture bundle once per module."""
    root = tmp_path_factory.mktemp("overview-v2") / "bundle"
    write_fixture_bundle(root, schema_version=2)
    return root


def derive_overview(root: Path) -> dict[str, Any]:
    """Derive overview.json for the dataset member and return the payload."""
    _, data = derive_in_temp_store(OVERVIEW, root, {}, member="dataset")
    assert set(data) == {"overview.json"}
    return json.loads(data["overview.json"])


def manifest_of(root: Path, member_path: str = "dataset") -> Any:
    """Parse the dataset manifest of a fixture bundle from disk."""
    data = (Path(root) / member_path / "dataset_manifest.json").read_bytes()
    return dataset_manifest_from_bytes(data, member_path)


def test_overview_v3_identity(v3_root: Path) -> None:
    """The v3 overview reports the dataset identity, stations and views."""
    payload = derive_overview(v3_root)
    assert payload["member"] == "dataset"
    assert payload["schema_version"] == 3
    assert payload["num_views"] == 3
    assert payload["num_bs"] == 2
    assert [station["bs_id"] for station in payload["base_stations"]] == ["bs_000", "bs_001"]
    for index, station in enumerate(payload["base_stations"]):
        assert station["position_m"] == list(BS_POSITIONS_M[index])
        assert station["look_at_m"] == list(BOX_CENTER_M)
    assert [view["view_id"] for view in payload["views"]] == [
        "ue_000000",
        "ue_000001",
        "ue_000002",
    ]


def test_overview_v3_frequency(v3_root: Path) -> None:
    """The v3 overview reports carrier, bandwidth, spacing and delays."""
    payload = derive_overview(v3_root)
    frequency = payload["frequency"]
    assert frequency["carrier_frequency_hz"] == CARRIER_HZ
    assert frequency["num_bins"] == 16
    assert frequency["bandwidth_hz"] == BANDWIDTH_HZ
    assert frequency["bin_spacing_hz"] == pytest.approx(BANDWIDTH_HZ / 16)
    assert frequency["delay_resolution_s"] == pytest.approx(1 / BANDWIDTH_HZ)
    assert frequency["unambiguous_delay_s"] == pytest.approx(16 / BANDWIDTH_HZ)
    assert frequency["unambiguous_delay_s"] == pytest.approx(
        float(manifest_of(v3_root).unambiguous_delay_s)
    )


def test_overview_v3_camera_model(v3_root: Path) -> None:
    """The v3 overview reports the camera grid from valid_mask and config."""
    payload = derive_overview(v3_root)
    camera_model = payload["camera_model"]
    assert camera_model["fft_rows"] == 32
    assert camera_model["fft_cols"] == 32
    assert camera_model["rx_rows"] == 8
    assert camera_model["rx_cols"] == 8
    assert camera_model["horizontal_spacing_lambda"] == 0.5
    assert camera_model["vertical_spacing_lambda"] == 0.5
    assert camera_model["hemispheres"] == ["front", "back"]


def test_overview_v3_pairs(v3_bundle: BundleFixture) -> None:
    """The v3 pairs match the manifest order, energies and front/back flags."""
    v3_root = v3_bundle.root
    payload = derive_overview(v3_root)
    truth = {(pair.view_id, pair.bs_id): pair.bs_hemisphere for pair in v3_bundle.truth.pairs}
    assert {(p["view_id"], p["bs_id"]): p["bs_in_front_hemisphere"] for p in payload["pairs"]} == {
        key: hemisphere == "front" for key, hemisphere in truth.items()
    }
    # The fixture has BSs both in front of and behind the cameras.
    assert {p["bs_in_front_hemisphere"] for p in payload["pairs"]} == {True, False}
    manifest = manifest_of(v3_root)
    assert len(payload["pairs"]) == 6
    expected = [(view.view_id, entry.bs_id) for view, entry in manifest.pairs()]
    assert [(pair["view_id"], pair["bs_id"]) for pair in payload["pairs"]] == expected
    for (view, entry), pair in zip(manifest.pairs(), payload["pairs"]):
        assert pair["bs_in_front_hemisphere"] is (entry.bs_in_front_hemisphere is True)
        assert pair["hemisphere_energy"] == {
            key: float(value) for key, value in entry.hemisphere_energy.items()
        }
        front = float(entry.hemisphere_energy.get("front", 0.0))
        back = float(entry.hemisphere_energy.get("back", 0.0))
        assert pair["total_energy"] == pytest.approx(front + back)
        if entry.bs_in_front_hemisphere:
            assert back == 0.0 and front > 0.0
            assert pair["back_fraction"] == pytest.approx(0.0)
        else:
            assert front == 0.0 and back > 0.0
            assert pair["back_fraction"] == pytest.approx(1.0)


def test_overview_v3_contents(v3_root: Path) -> None:
    """The v3 contents report optical, GT, observations, links and image axes."""
    payload = derive_overview(v3_root)
    contents = payload["contents"]
    assert contents["optical"] is True
    assert contents["optical_artifacts"] == sorted(OPTICAL_KEYS)
    assert contents["transforms_json"] is True
    assert contents["path_gt"] is True
    assert contents["path_schema"] is True
    assert contents["observations"] == ["obs"]
    assert contents["partials"] == ["p0", "p1"]
    assert contents["scene"] == "scene"
    assert contents["placement"] is False
    assert contents["tomography_gt"] is False
    assert payload["image_axes"] == {"source": "default_m1", "axes": DEFAULT_M1_IMAGE_AXES}


def test_overview_v2(v2_root: Path) -> None:
    """The v2 overview has one base station, three pairs and no observations."""
    payload = derive_overview(v2_root)
    assert payload["schema_version"] == 2
    assert payload["num_bs"] == 1
    assert payload["base_stations"][0]["bs_id"] == "bs_000"
    assert len(payload["pairs"]) == 3
    assert payload["contents"]["observations"] == []
    assert payload["contents"]["partials"] == ["p0", "p1"]
    assert payload["frequency"]["bandwidth_hz"] == BANDWIDTH_HZ
    assert payload["frequency"]["num_bins"] == 16
    assert payload["frequency"]["delay_resolution_s"] == pytest.approx(1 / BANDWIDTH_HZ)
    assert payload["frequency"]["unambiguous_delay_s"] == pytest.approx(16 / BANDWIDTH_HZ)
    assert payload["base_stations"][0]["position_m"] == list(BS_POSITIONS_M[0])


def test_overview_delay_fallbacks(tmp_path: Path, v3_root: Path) -> None:
    """Missing delay and bandwidth fields fall back to 1/B and N/B derivations."""
    root = tmp_path / "bundle"
    shutil.copytree(v3_root, root)
    manifest_path = root / "dataset" / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["delay_resolution_s"]
    del manifest["unambiguous_delay_s"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    payload = derive_overview(root)
    assert payload["frequency"]["delay_resolution_s"] == pytest.approx(1 / BANDWIDTH_HZ)
    assert payload["frequency"]["unambiguous_delay_s"] == pytest.approx(16 / BANDWIDTH_HZ)
    del manifest["config"]["bandwidth_hz"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    payload = derive_overview(root)
    assert payload["frequency"]["bandwidth_hz"] == pytest.approx(BANDWIDTH_HZ)
    assert payload["frequency"]["delay_resolution_s"] == pytest.approx(1 / BANDWIDTH_HZ)
    assert payload["frequency"]["unambiguous_delay_s"] == pytest.approx(16 / BANDWIDTH_HZ)


def test_overview_manifest_image_axes(tmp_path: Path, v3_root: Path) -> None:
    """A manifest camera_model.image_axes mapping wins over the default."""
    root = tmp_path / "bundle"
    shutil.copytree(v3_root, root)
    manifest_path = root / "dataset" / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["camera_model"]["image_axes"] = {"col": "x", "row": "y"}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    payload = derive_overview(root)
    assert payload["image_axes"] == {"source": "manifest", "axes": {"col": "x", "row": "y"}}


def test_overview_placement_and_tomography_flags(tmp_path: Path, v3_root: Path) -> None:
    """A placement directory and tomography_gt.npz flip their flags on."""
    root = tmp_path / "bundle"
    shutil.copytree(v3_root, root)
    placement = root / "dataset" / "placement"
    placement.mkdir()
    (placement / "plan.json").write_text("{}", encoding="utf-8")
    np.savez(root / "dataset" / "tomography_gt.npz", recon=np.zeros(2, dtype=np.float32))
    payload = derive_overview(root)
    assert payload["contents"]["placement"] is True
    assert payload["contents"]["tomography_gt"] is True


def test_build_overview_without_links(v3_root: Path) -> None:
    """Members without links keys give no partials and no scene."""
    manifest = manifest_of(v3_root)
    members = [{"id": "dataset", "kind": "rf_dataset", "path": "dataset"}]
    payload = build_overview(
        manifest,
        member="dataset",
        members=members,
        camera_model=None,
        exists=lambda relpath: False,
        is_dir=lambda relpath: False,
    )
    assert payload["contents"]["partials"] == []
    assert payload["contents"]["scene"] is None


def test_overview_deterministic(v3_root: Path, v2_root: Path) -> None:
    """The overview deriver writes one deterministic overview.json file."""
    shas_v3 = assert_deterministic(OVERVIEW, v3_root, {}, member="dataset")
    assert set(shas_v3) == {"overview.json"}
    shas_v2 = assert_deterministic(OVERVIEW, v2_root, {}, member="dataset")
    assert set(shas_v2) == {"overview.json"}


def test_overview_is_registered_builtin() -> None:
    """Overview is registered globally and listed in DERIVER_MODULES."""
    assert "overview" in {deriver.spec.name for deriver in BUILTIN_DERIVERS}
    assert "overview" in {deriver.spec.name for deriver in registered_derivers()}
    assert "plateau_rt.viewer.derive.overview" in DERIVER_MODULES


def test_overview_eager(tmp_path: Path, v3_root: Path) -> None:
    """derive_eager over the stored fixture bundle yields one overview outcome."""
    store = Store(ViewerSettings(data_dir=tmp_path / "store"))
    root = store.stage_from_directory(v3_root, store.settings.extract_limits)
    digest, _ = store.commit(root, name="t", members=bundle_members(root))
    outcomes = derive_eager(store, digest)
    assert len(outcomes) == 1
    outcome = outcomes[0]
    assert (outcome.deriver, outcome.member) == ("overview", "dataset")
    assert outcome.error is None
    assert outcome.result is not None
    assert {record.name for record in outcome.result.files} == {"overview.json"}
