"""Tests for bundle kind detection and member validation (plateau_rt.viewer.kinds)."""

from __future__ import annotations

import builtins
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from viewer_bundle_fixtures import (
    BROKEN_CASE_MESSAGES,
    BROKEN_CASES,
    make_archive,
    write_broken_dataset,
    write_bundle_dir,
    write_fixture_bundle,
    write_scene,
)
from viewer_fixtures import write_rf_dataset

from plateau_rt.application.rf_dataset_manifest import (
    ManifestError,
    parse_rf_dataset_manifest,
)
from plateau_rt.application.scene_files import SceneFileError, scene_file_references
from plateau_rt.viewer import kinds
from plateau_rt.viewer.extract import safe_extract
from plateau_rt.viewer.kinds import (
    BundleValidationError,
    Member,
    detect_members,
    validate_bundle,
    validate_member,
)
from plateau_rt.viewer.settings import ViewerSettings

EXPECTED_V3_MEMBERS = [
    Member("dataset", "rf_dataset", "dataset", {}),
    Member("scene", "scene", "scene/scene.xml", {"for": "dataset"}),
    Member("p0", "rf_partial", "partials/p0", {"source": "dataset"}),
    Member("p1", "rf_partial", "partials/p1", {"source": "dataset"}),
]


@pytest.fixture(scope="module")
def v3_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the canonical v3 fixture bundle once per module."""
    root = tmp_path_factory.mktemp("kinds-v3") / "bundle"
    write_fixture_bundle(root, schema_version=3)
    return root


@pytest.fixture(scope="module")
def v2_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the canonical v2 fixture bundle once per module."""
    root = tmp_path_factory.mktemp("kinds-v2") / "bundle"
    write_fixture_bundle(root, schema_version=2)
    return root


def copy_bundle(src: Path, tmp_path: Path) -> Path:
    """Copy a fixture bundle into tmp_path and return the copy."""
    dest = tmp_path / "bundle"
    shutil.copytree(src, dest)
    return dest


def write_raw_bundle_json(root: Path, payload: Any) -> Path:
    """Write payload as raw JSON bundle.json (bypassing fixture validation)."""
    path = Path(root) / "bundle.json"
    if isinstance(payload, (dict, list)):
        path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    else:
        path.write_text(str(payload), encoding="utf-8")
    return path


def dataset_member(root: Path, member_id: str = "dataset") -> Member:
    """Return the rf_dataset member detected in root with the given id."""
    for member in detect_members(root):
        if member.id == member_id:
            assert member.kind == "rf_dataset"
            return member
    raise AssertionError(f"no member {member_id!r} in {root}")


def test_detect_fixture_bundle_members(v3_root: Path) -> None:
    """The v3 fixture bundle detects four members with the p1 manifest link."""
    assert detect_members(v3_root) == EXPECTED_V3_MEMBERS


def test_member_to_dict(v3_root: Path) -> None:
    """Member.to_dict returns the store record with a copied links mapping."""
    member = detect_members(v3_root)[1]
    assert member.to_dict() == {
        "id": "scene",
        "kind": "scene",
        "path": "scene/scene.xml",
        "links": {"for": "dataset"},
    }
    assert member.to_dict()["links"] is not member.links


def test_detect_single_member_bundle(tmp_path: Path) -> None:
    """A bundle.json with one rf_dataset member detects exactly it."""
    write_rf_dataset(tmp_path / "dataset")
    write_bundle_dir(tmp_path, members=[{"id": "d", "kind": "rf_dataset", "path": "dataset"}])
    assert detect_members(tmp_path) == [Member("d", "rf_dataset", "dataset", {})]


def test_detect_fallback_dataset(tmp_path: Path) -> None:
    """A lone dataset_manifest.json at the root detects the fallback member."""
    write_rf_dataset(tmp_path)
    assert detect_members(tmp_path) == [Member("dataset", "rf_dataset", ".", {})]


def test_detect_fallback_partial(tmp_path: Path, v3_root: Path) -> None:
    """A lone partial directory detects the fallback partial with no link."""
    shutil.copytree(v3_root / "partials" / "p1", tmp_path / "solo")
    assert detect_members(tmp_path / "solo") == [Member("partial", "rf_partial", ".", {})]


def test_detect_fallback_run(tmp_path: Path) -> None:
    """A lone run_manifest.json at the root detects the fallback tomo_run member."""
    (tmp_path / "run_manifest.json").write_text("{}", encoding="utf-8")
    assert detect_members(tmp_path) == [Member("run", "tomo_run", ".", {})]


def test_detect_archive_round_trip(tmp_path: Path, v3_root: Path) -> None:
    """A zipped bundle extracts to the same detected members."""
    archive = tmp_path / "b.zip"
    make_archive(v3_root, archive, "zip", root_name="mybundle")
    dest = tmp_path / "extracted"
    dest.mkdir()
    safe_extract(archive, dest, ViewerSettings(data_dir=tmp_path / "store").extract_limits)
    assert detect_members(dest / "mybundle") == detect_members(v3_root)


def test_partial_link_negative(tmp_path: Path, v3_root: Path) -> None:
    """A source_dataset that misses the bundle gives the partial no link."""
    root = copy_bundle(v3_root, tmp_path)
    manifest_path = root / "partials" / "p1" / "partial_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_dataset"] = "../../elsewhere"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    members = {member.id: member for member in detect_members(root)}
    assert members["p1"].links == {}
    manifest["source_dataset"] = "/abs/dataset"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    members = {member.id: member for member in detect_members(root)}
    assert members["p1"].links == {}
    # A path that names another (non-rf_dataset) member is not a source either.
    manifest["source_dataset"] = "../p0"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    members = {member.id: member for member in detect_members(root)}
    assert members["p1"].links == {}


def test_fallback_several_markers(tmp_path: Path) -> None:
    """Two root markers without bundle.json name both files in the error."""
    write_rf_dataset(tmp_path)
    (tmp_path / "partial_manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(BundleValidationError) as excinfo:
        detect_members(tmp_path)
    assert excinfo.value.member_id is None
    assert "several marker manifests" in excinfo.value.message
    assert "dataset_manifest.json" in excinfo.value.message
    assert "partial_manifest.json" in excinfo.value.message


def test_fallback_no_markers(tmp_path: Path) -> None:
    """An empty directory reports that no marker manifest was found."""
    with pytest.raises(BundleValidationError) as excinfo:
        detect_members(tmp_path)
    assert excinfo.value.member_id is None
    assert "no marker manifest" in excinfo.value.message


def _minimal_dataset_bundle(tmp_path: Path) -> Path:
    """Write a minimal valid single-dataset bundle root and return it."""
    write_rf_dataset(tmp_path / "dataset")
    return tmp_path


def _member_bundle(
    members: Any = None,
    version: Any = 1,
    **extra: Any,
) -> dict[str, Any]:
    """Build a bundle.json payload with one dataset member by default."""
    payload: dict[str, Any] = {
        "bundle_format_version": version,
        "members": (
            [{"id": "d", "kind": "rf_dataset", "path": "dataset"}] if members is None else members
        ),
    }
    payload.update(extra)
    return payload


@pytest.mark.parametrize(
    ("payload", "substring"),
    [
        (_member_bundle(version=2), "unsupported bundle_format_version"),
        (_member_bundle(version=True), "unsupported bundle_format_version"),
        (_member_bundle(version="1"), "unsupported bundle_format_version"),
        (
            {"members": [{"id": "d", "kind": "rf_dataset", "path": "dataset"}]},
            "unsupported bundle_format_version",
        ),
        (_member_bundle(members=[]), "'members' must be a non-empty list"),
        (_member_bundle(members="nope"), "'members' must be a non-empty list"),
        (_member_bundle(members=[{"id": "d", "kind": "bogus", "path": "dataset"}]), "invalid kind"),
        (_member_bundle(members=[{"id": "d", "path": "dataset"}]), "invalid kind"),
        (
            _member_bundle(members=[{"id": "a/b", "kind": "rf_dataset", "path": "dataset"}]),
            "missing or invalid id",
        ),
        (
            _member_bundle(members=[{"id": "..", "kind": "rf_dataset", "path": "dataset"}]),
            "missing or invalid id",
        ),
        (
            _member_bundle(members=[{"kind": "rf_dataset", "path": "dataset"}]),
            "missing or invalid id",
        ),
        (
            _member_bundle(members=[{"id": "d", "kind": "rf_dataset", "path": "/abs"}]),
            "invalid path",
        ),
        (
            _member_bundle(members=[{"id": "d", "kind": "rf_dataset", "path": "a/../b"}]),
            "invalid path",
        ),
        (
            _member_bundle(members=[{"id": "d", "kind": "rf_dataset", "path": "./dataset"}]),
            "invalid path",
        ),
        (
            _member_bundle(members=[{"id": "d", "kind": "rf_dataset", "path": "dataset/"}]),
            "invalid path",
        ),
        (_member_bundle(members=[{"id": "d", "kind": "rf_dataset", "path": "."}]), "invalid path"),
        (
            _member_bundle(members=[{"id": "d", "kind": "rf_dataset", "path": "a\\b"}]),
            "invalid path",
        ),
        (
            _member_bundle(members=[{"id": "d", "kind": "rf_dataset", "path": "missing"}]),
            "does not exist",
        ),
        (
            _member_bundle(
                members=[{"id": "d", "kind": "rf_dataset", "path": "dataset", "sorce": "d"}]
            ),
            "unknown key(s)",
        ),
        (_member_bundle(members=["nope"]), "must be an object"),
        (_member_bundle(bogus=1), "unknown key(s)"),
        (_member_bundle(created_by=[1]), "'created_by' must be an object"),
        (
            _member_bundle(
                members=[{"id": "d", "kind": "rf_dataset", "path": "dataset", "for": "d"}]
            ),
            "'for' is only allowed on scene members",
        ),
    ],
)
def test_bundle_json_rejections(tmp_path: Path, payload: dict[str, Any], substring: str) -> None:
    """Malformed bundle.json payloads fail with the documented message part."""
    _minimal_dataset_bundle(tmp_path)
    write_raw_bundle_json(tmp_path, payload)
    with pytest.raises(BundleValidationError) as excinfo:
        detect_members(tmp_path)
    assert substring in excinfo.value.message


def test_bundle_json_too_large(tmp_path: Path) -> None:
    """A bundle.json over 1 MiB is rejected with its size message."""
    _minimal_dataset_bundle(tmp_path)
    payload = {
        "bundle_format_version": 1,
        "members": [{"id": "d", "kind": "rf_dataset", "path": "dataset"}],
        "created_by": {"tool": "x" * (2 * 1024 * 1024), "tool_version": 1},
    }
    write_raw_bundle_json(tmp_path, payload)
    with pytest.raises(BundleValidationError) as excinfo:
        detect_members(tmp_path)
    assert excinfo.value.member_id is None
    assert "larger than 1048576 bytes" in excinfo.value.message


def test_bundle_json_invalid_json(tmp_path: Path) -> None:
    """A bundle.json that is not JSON is rejected."""
    _minimal_dataset_bundle(tmp_path)
    write_raw_bundle_json(tmp_path, "{not json")
    with pytest.raises(BundleValidationError) as excinfo:
        detect_members(tmp_path)
    assert excinfo.value.member_id is None
    assert "not valid UTF-8 JSON" in excinfo.value.message


def test_bundle_json_not_object(tmp_path: Path) -> None:
    """A bundle.json that is not an object is rejected."""
    _minimal_dataset_bundle(tmp_path)
    write_raw_bundle_json(tmp_path, [1, 2])
    with pytest.raises(BundleValidationError) as excinfo:
        detect_members(tmp_path)
    assert "must be a JSON object" in excinfo.value.message


def test_bundle_json_not_regular_file(tmp_path: Path) -> None:
    """A bundle.json directory is rejected as not a regular file."""
    (tmp_path / "bundle.json").mkdir()
    with pytest.raises(BundleValidationError) as excinfo:
        detect_members(tmp_path)
    assert excinfo.value.member_id is None
    assert "not a regular file" in excinfo.value.message


def test_bundle_json_too_many_members(tmp_path: Path) -> None:
    """1025 members exceed the bundle limit."""
    _minimal_dataset_bundle(tmp_path)
    members = [{"id": f"m{i:04d}", "kind": "rf_dataset", "path": "dataset"} for i in range(1025)]
    write_raw_bundle_json(tmp_path, {"bundle_format_version": 1, "members": members})
    with pytest.raises(BundleValidationError) as excinfo:
        detect_members(tmp_path)
    assert "at most 1024" in excinfo.value.message


def test_bundle_json_duplicate_id(tmp_path: Path) -> None:
    """Two members with the same id are rejected naming the duplicate."""
    _minimal_dataset_bundle(tmp_path)
    members = [
        {"id": "d", "kind": "rf_dataset", "path": "dataset"},
        {"id": "d", "kind": "rf_dataset", "path": "dataset"},
    ]
    write_bundle_dir(tmp_path, members=members, validate=False)  # type: ignore[arg-type]
    with pytest.raises(BundleValidationError) as excinfo:
        detect_members(tmp_path)
    assert excinfo.value.member_id == "d"
    assert "duplicate member id" in excinfo.value.message


def test_bundle_json_symlinked_member(tmp_path: Path) -> None:
    """A member directory reached through a symlink is rejected."""
    write_rf_dataset(tmp_path / "real")
    os.symlink(tmp_path / "real", tmp_path / "link")
    write_bundle_dir(
        tmp_path, members=[{"id": "d", "kind": "rf_dataset", "path": "link"}], validate=False
    )
    with pytest.raises(BundleValidationError) as excinfo:
        detect_members(tmp_path)
    assert excinfo.value.member_id == "d"
    assert "goes through a symlink" in excinfo.value.message


def test_bundle_json_file_for_directory_kind(tmp_path: Path) -> None:
    """A regular file used as an rf_dataset path is rejected."""
    (tmp_path / "afile.txt").write_text("x", encoding="utf-8")
    write_bundle_dir(
        tmp_path, members=[{"id": "d", "kind": "rf_dataset", "path": "afile.txt"}], validate=False
    )
    with pytest.raises(BundleValidationError) as excinfo:
        detect_members(tmp_path)
    assert excinfo.value.member_id == "d"
    assert "must be a directory for kind rf_dataset" in excinfo.value.message


def test_bundle_json_scene_not_xml(tmp_path: Path) -> None:
    """A scene member that is not an .xml file is rejected."""
    (tmp_path / "scene.txt").write_text("<scene/>", encoding="utf-8")
    write_bundle_dir(
        tmp_path, members=[{"id": "s", "kind": "scene", "path": "scene.txt"}], validate=False
    )
    with pytest.raises(BundleValidationError) as excinfo:
        detect_members(tmp_path)
    assert excinfo.value.member_id == "s"
    assert "must be an .xml file" in excinfo.value.message


def test_bundle_json_missing_marker(tmp_path: Path) -> None:
    """A directory member without its marker manifest is rejected."""
    (tmp_path / "empty").mkdir()
    write_bundle_dir(
        tmp_path, members=[{"id": "d", "kind": "rf_dataset", "path": "empty"}], validate=False
    )
    with pytest.raises(BundleValidationError) as excinfo:
        detect_members(tmp_path)
    assert excinfo.value.member_id == "d"
    assert "missing dataset_manifest.json" in excinfo.value.message


def test_bundle_json_scene_bad_root(tmp_path: Path) -> None:
    """A scene XML whose root is not <scene> is rejected."""
    (tmp_path / "bad.xml").write_text("<foo/>", encoding="utf-8")
    write_bundle_dir(
        tmp_path, members=[{"id": "s", "kind": "scene", "path": "bad.xml"}], validate=False
    )
    with pytest.raises(BundleValidationError) as excinfo:
        detect_members(tmp_path)
    assert excinfo.value.member_id == "s"
    assert "not <scene>" in excinfo.value.message


def test_bundle_json_duplicate_directory_paths(tmp_path: Path) -> None:
    """Two directory members sharing one path are rejected."""
    _minimal_dataset_bundle(tmp_path)
    members = [
        {"id": "d1", "kind": "rf_dataset", "path": "dataset"},
        {"id": "d2", "kind": "rf_dataset", "path": "dataset"},
    ]
    write_bundle_dir(tmp_path, members=members, validate=False)  # type: ignore[arg-type]
    with pytest.raises(BundleValidationError) as excinfo:
        detect_members(tmp_path)
    assert excinfo.value.member_id == "d2"
    assert "same directory path as member 'd1'" in excinfo.value.message


def test_bundle_json_nested_directories(tmp_path: Path) -> None:
    """A directory member nested inside another is rejected."""
    write_rf_dataset(tmp_path / "dataset")
    nested = tmp_path / "dataset" / "nested"
    nested.mkdir()
    (nested / "partial_manifest.json").write_text("{}", encoding="utf-8")
    members = [
        {"id": "d", "kind": "rf_dataset", "path": "dataset"},
        {"id": "p", "kind": "rf_partial", "path": "dataset/nested"},
    ]
    write_bundle_dir(tmp_path, members=members, validate=False)  # type: ignore[arg-type]
    with pytest.raises(BundleValidationError) as excinfo:
        detect_members(tmp_path)
    assert excinfo.value.member_id == "p"
    assert "is nested inside member 'd'" in excinfo.value.message


def test_bundle_json_source_on_scene(tmp_path: Path, v3_root: Path) -> None:
    """A scene member with source is rejected."""
    root = copy_bundle(v3_root, tmp_path)
    payload = json.loads((root / "bundle.json").read_text(encoding="utf-8"))
    payload["members"][1]["source"] = "dataset"
    write_raw_bundle_json(root, payload)
    with pytest.raises(BundleValidationError) as excinfo:
        detect_members(root)
    assert excinfo.value.member_id == "scene"
    assert "'source' is only allowed on rf_partial and tomo_run members" in excinfo.value.message


def test_bundle_json_for_unknown_id(tmp_path: Path, v3_root: Path) -> None:
    """A scene for naming an unknown id is rejected."""
    root = copy_bundle(v3_root, tmp_path)
    payload = json.loads((root / "bundle.json").read_text(encoding="utf-8"))
    payload["members"][1]["for"] = "nope"
    write_raw_bundle_json(root, payload)
    with pytest.raises(BundleValidationError) as excinfo:
        detect_members(root)
    assert excinfo.value.member_id == "scene"
    assert "'for' names 'nope'" in excinfo.value.message
    assert "not an rf_dataset member" in excinfo.value.message


def test_bundle_json_source_naming_scene(tmp_path: Path, v3_root: Path) -> None:
    """A partial source naming a scene member is rejected."""
    root = copy_bundle(v3_root, tmp_path)
    payload = json.loads((root / "bundle.json").read_text(encoding="utf-8"))
    payload["members"][2]["source"] = "scene"
    write_raw_bundle_json(root, payload)
    with pytest.raises(BundleValidationError) as excinfo:
        detect_members(root)
    assert excinfo.value.member_id == "p0"
    assert "'source' names 'scene'" in excinfo.value.message


def test_bundle_json_two_scenes_same_for(tmp_path: Path, v3_root: Path) -> None:
    """Two scenes linked to the same dataset are rejected."""
    root = copy_bundle(v3_root, tmp_path)
    shutil.copy(root / "scene" / "scene.xml", root / "scene" / "other.xml")
    payload = json.loads((root / "bundle.json").read_text(encoding="utf-8"))
    payload["members"].append(
        {"id": "scene2", "kind": "scene", "path": "scene/other.xml", "for": "dataset"}
    )
    write_raw_bundle_json(root, payload)
    with pytest.raises(BundleValidationError) as excinfo:
        detect_members(root)
    assert excinfo.value.member_id == "scene2"
    assert "two scenes have 'for' 'dataset'" in excinfo.value.message


def test_bundle_json_created_by_not_object(tmp_path: Path) -> None:
    """A non-object created_by is rejected."""
    _minimal_dataset_bundle(tmp_path)
    payload = {
        "bundle_format_version": 1,
        "members": [{"id": "d", "kind": "rf_dataset", "path": "dataset"}],
        "created_by": "tool",
    }
    write_raw_bundle_json(tmp_path, payload)
    with pytest.raises(BundleValidationError) as excinfo:
        detect_members(tmp_path)
    assert "'created_by' must be an object" in excinfo.value.message


@pytest.mark.parametrize("case", BROKEN_CASES)
def test_broken_datasets_fail_verbatim(tmp_path: Path, case: str) -> None:
    """Broken manifests fail validation with the ManifestError text verbatim."""
    target = tmp_path / case
    write_broken_dataset(target, case)
    members = detect_members(target)
    assert members == [Member("dataset", "rf_dataset", ".", {})]
    with pytest.raises(BundleValidationError) as excinfo:
        validate_member(members[0], target)
    assert excinfo.value.member_id == "dataset"
    assert BROKEN_CASE_MESSAGES[case] in excinfo.value.message
    manifest = json.loads((target / "dataset_manifest.json").read_text(encoding="utf-8"))
    try:
        parse_rf_dataset_manifest(manifest, root=Path("."))
    except ManifestError as exc:
        expected = str(exc)
    else:  # pragma: no cover - every broken case must raise
        raise AssertionError(f"broken case {case!r} did not raise ManifestError")
    assert excinfo.value.message == expected


def test_validate_bundle_v3(v3_root: Path) -> None:
    """The v3 fixture bundle validates with the expected summaries and files."""
    infos = validate_bundle(v3_root)
    assert [info.member.kind for info in infos] == [
        "rf_dataset",
        "scene",
        "rf_partial",
        "rf_partial",
    ]
    dataset = infos[0]
    assert dataset.schema_version == 3
    assert dataset.summary == {"num_views": 3, "num_bs": 2, "num_frequency_bins": 16}
    assert "dataset/views/ue_000000/rf/aperture_cfr.npy" in dataset.files
    assert "dataset/camera_model.npz" in dataset.files
    scene = infos[1]
    assert scene.schema_version is None
    assert "scene/box.ply" in scene.files
    assert scene.summary == {"num_shapes": 2}


def test_validate_bundle_v2(v2_root: Path) -> None:
    """The v2 fixture bundle validates with one base station."""
    infos = validate_bundle(v2_root)
    assert infos[0].summary == {"num_views": 3, "num_bs": 1, "num_frequency_bins": 16}
    assert infos[0].schema_version == 2


def test_validate_aperture_wrong_dtype(tmp_path: Path, v3_root: Path) -> None:
    """A real-valued aperture CFR fails the complex shape check."""
    root = copy_bundle(v3_root, tmp_path)
    aperture = root / "dataset" / "views" / "ue_000000" / "rf" / "aperture_cfr.npy"
    np.save(aperture, np.zeros((2, 2, 8, 8, 16), dtype=np.float32))
    with pytest.raises(BundleValidationError) as excinfo:
        validate_member(dataset_member(root), root)
    assert excinfo.value.member_id == "dataset"
    assert "aperture_cfr" in excinfo.value.message
    assert "does not match expected" in excinfo.value.message


def test_validate_aperture_wrong_shape(tmp_path: Path, v3_root: Path) -> None:
    """A wrongly shaped complex aperture CFR fails the shape check."""
    root = copy_bundle(v3_root, tmp_path)
    aperture = root / "dataset" / "views" / "ue_000000" / "rf" / "aperture_cfr.npy"
    np.save(aperture, np.zeros((2, 2, 8, 8, 15), dtype=np.complex64))
    with pytest.raises(BundleValidationError) as excinfo:
        validate_member(dataset_member(root), root)
    assert "aperture_cfr" in excinfo.value.message
    assert "does not match expected" in excinfo.value.message


def test_validate_missing_artifact(tmp_path: Path, v3_root: Path) -> None:
    """A deleted per-BS artifact file fails with does not exist naming it."""
    root = copy_bundle(v3_root, tmp_path)
    manifest = json.loads((root / "dataset" / "dataset_manifest.json").read_text(encoding="utf-8"))
    artifacts = manifest["views"][0]["bs"][0]["artifacts"]
    key = next(iter(artifacts))
    (root / "dataset" / artifacts[key]).unlink()
    with pytest.raises(BundleValidationError) as excinfo:
        validate_member(dataset_member(root), root)
    assert excinfo.value.member_id == "dataset"
    assert key in excinfo.value.message
    assert "does not exist" in excinfo.value.message


def test_validate_corrupt_npz(tmp_path: Path, v3_root: Path) -> None:
    """A corrupt camera_model.npz fails naming the camera_model artifact."""
    root = copy_bundle(v3_root, tmp_path)
    (root / "dataset" / "camera_model.npz").write_bytes(b"not a zip file at all")
    with pytest.raises(BundleValidationError) as excinfo:
        validate_member(dataset_member(root), root)
    assert excinfo.value.member_id == "dataset"
    assert "camera_model" in excinfo.value.message


def test_validate_tomo_run_rejected(tmp_path: Path) -> None:
    """A tomo_run member is rejected until V3-2."""
    (tmp_path / "run_manifest.json").write_text("{}", encoding="utf-8")
    member = detect_members(tmp_path)[0]
    with pytest.raises(BundleValidationError) as excinfo:
        validate_member(member, tmp_path)
    assert excinfo.value.member_id == "run"
    assert "unsupported member kind 'tomo_run'" in excinfo.value.message


def test_validate_partial_bad_schema(tmp_path: Path, v3_root: Path) -> None:
    """A partial with schema_version 2 is rejected."""
    root = copy_bundle(v3_root, tmp_path)
    manifest_path = root / "partials" / "p0" / "partial_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 2
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    member = next(member for member in detect_members(root) if member.id == "p0")
    with pytest.raises(BundleValidationError) as excinfo:
        validate_member(member, root)
    assert excinfo.value.member_id == "p0"
    assert "unsupported partial schema_version 2 (supported: 1)" in excinfo.value.message


def test_validate_partial_bad_mode(tmp_path: Path, v3_root: Path) -> None:
    """A partial with a wrong mode is rejected."""
    root = copy_bundle(v3_root, tmp_path)
    manifest_path = root / "partials" / "p0" / "partial_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["mode"] = "bogus"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    member = next(member for member in detect_members(root) if member.id == "p0")
    with pytest.raises(BundleValidationError) as excinfo:
        validate_member(member, root)
    assert "unsupported partial mode 'bogus'" in excinfo.value.message


def test_validate_partial_invalid_json(tmp_path: Path, v3_root: Path) -> None:
    """A partial with an unparsable manifest is rejected."""
    root = copy_bundle(v3_root, tmp_path)
    (root / "partials" / "p0" / "partial_manifest.json").write_text("{bad", encoding="utf-8")
    member = next(member for member in detect_members(root) if member.id == "p0")
    with pytest.raises(BundleValidationError) as excinfo:
        validate_member(member, root)
    assert "partial_manifest.json is not a valid JSON object" in excinfo.value.message


def _write_outside(tmp_path: Path, root: Path) -> tuple[Path, Path]:
    """Write a valid complex aperture array and a secret PLY outside the bundle."""
    outside = tmp_path / "outside"
    outside.mkdir()
    array_path = outside / "x.npy"
    np.save(array_path, np.zeros((2, 2, 8, 8, 16), dtype=np.complex64))
    (outside / "secret.ply").write_bytes(b"ply-secret")
    return outside, array_path


def _install_open_spy(
    monkeypatch: pytest.MonkeyPatch, outside: Path
) -> tuple[list[str], dict[str, Any]]:
    """Record every file open; return the records and the original loaders."""
    opened: list[str] = []
    real_open = builtins.open

    def spy_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(file, (str, os.PathLike)):
            opened.append(os.fspath(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", spy_open)
    originals = {
        "load_npy": kinds.safeio.load_npy,
        "read_npy_header": kinds.safeio.read_npy_header,
        "check_npz": kinds.safeio.check_npz,
        "read_bytes": kinds.safeio.read_bytes,
    }

    def make_spy(name: str) -> Any:
        def spy(path: Any, *args: Any, **kwargs: Any) -> Any:
            opened.append(os.fspath(path))
            return originals[name](path, *args, **kwargs)

        return spy

    for name in originals:
        monkeypatch.setattr(kinds.safeio, name, make_spy(name))

    def assert_no_outside_opens() -> None:
        """Fail when any recorded path lies inside the outside directory."""
        real_outside = os.path.realpath(outside)
        for record in opened:
            try:
                real_record = os.path.realpath(record)
            except (OSError, ValueError):
                continue
            assert real_record != real_outside and not real_record.startswith(
                real_outside + os.sep
            ), f"opened outside path {record!r}"

    return opened, {"assert_no_outside_opens": assert_no_outside_opens}


def _rewrite_dataset_manifest(root: Path, mutate: Any) -> None:
    """Load the v3 dataset manifest under root, mutate it and write it back."""
    manifest_path = root / "dataset" / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_escape_absolute_aperture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, v3_root: Path
) -> None:
    """An absolute aperture_cfr path is rejected and never opened."""
    root = copy_bundle(v3_root, tmp_path)
    outside, array_path = _write_outside(tmp_path, root)
    _rewrite_dataset_manifest(
        root,
        lambda manifest: manifest["views"][0]["artifacts"].__setitem__(
            "aperture_cfr", str(array_path)
        ),
    )
    _, guard = _install_open_spy(monkeypatch, outside)
    with pytest.raises(BundleValidationError) as excinfo:
        validate_member(dataset_member(root), root)
    assert "aperture_cfr" in excinfo.value.message
    assert "absolute" in excinfo.value.message
    guard["assert_no_outside_opens"]()


def test_escape_relative_aperture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, v3_root: Path
) -> None:
    """A ../ aperture_cfr escape is rejected and never opened."""
    root = copy_bundle(v3_root, tmp_path)
    outside, _ = _write_outside(tmp_path, root)
    _rewrite_dataset_manifest(
        root,
        lambda manifest: manifest["views"][0]["artifacts"].__setitem__(
            "aperture_cfr", "../outside/x.npy"
        ),
    )
    _, guard = _install_open_spy(monkeypatch, outside)
    with pytest.raises(BundleValidationError) as excinfo:
        validate_member(dataset_member(root), root)
    assert "aperture_cfr" in excinfo.value.message
    assert "leaves the root" in excinfo.value.message
    guard["assert_no_outside_opens"]()


def test_escape_camera_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, v3_root: Path
) -> None:
    """An absolute camera_model path is rejected and never opened."""
    root = copy_bundle(v3_root, tmp_path)
    outside, array_path = _write_outside(tmp_path, root)
    _rewrite_dataset_manifest(
        root,
        lambda manifest: manifest["camera_model"].__setitem__("ray_directions", str(array_path)),
    )
    _, guard = _install_open_spy(monkeypatch, outside)
    with pytest.raises(BundleValidationError) as excinfo:
        validate_member(dataset_member(root), root)
    assert "camera_model" in excinfo.value.message
    assert "absolute" in excinfo.value.message
    guard["assert_no_outside_opens"]()


def test_escape_observed_aperture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, v3_root: Path
) -> None:
    """A per-BS observed aperture escape is rejected and never opened."""
    root = copy_bundle(v3_root, tmp_path)
    outside, _ = _write_outside(tmp_path, root)
    _rewrite_dataset_manifest(
        root,
        lambda manifest: manifest["views"][0]["bs"][0]["artifacts"].__setitem__(
            "observed.obs.aperture_cfr", "../../../outside/x.npy"
        ),
    )
    _, guard = _install_open_spy(monkeypatch, outside)
    with pytest.raises(BundleValidationError) as excinfo:
        validate_member(dataset_member(root), root)
    assert "observed.obs.aperture_cfr" in excinfo.value.message
    assert "leaves the root" in excinfo.value.message
    guard["assert_no_outside_opens"]()


def test_escape_path_geometry_gt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, v3_root: Path
) -> None:
    """An absolute path_geometry_gt is rejected and never opened."""
    root = copy_bundle(v3_root, tmp_path)
    outside, array_path = _write_outside(tmp_path, root)
    _rewrite_dataset_manifest(
        root, lambda manifest: manifest.__setitem__("path_geometry_gt", str(array_path))
    )
    _, guard = _install_open_spy(monkeypatch, outside)
    with pytest.raises(BundleValidationError) as excinfo:
        validate_member(dataset_member(root), root)
    assert "path_geometry_gt" in excinfo.value.message
    assert "absolute" in excinfo.value.message
    guard["assert_no_outside_opens"]()


def test_escape_symlinked_aperture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, v3_root: Path
) -> None:
    """A symlink aperture pointing outside the root is rejected unopened."""
    root = copy_bundle(v3_root, tmp_path)
    outside, array_path = _write_outside(tmp_path, root)
    aperture = root / "dataset" / "views" / "ue_000000" / "rf" / "aperture_cfr.npy"
    aperture.unlink()
    os.symlink(array_path, aperture)
    _, guard = _install_open_spy(monkeypatch, outside)
    with pytest.raises(BundleValidationError) as excinfo:
        validate_member(dataset_member(root), root)
    assert "leaves the root" in excinfo.value.message
    guard["assert_no_outside_opens"]()


@pytest.mark.parametrize("filename", ["../../etc/passwd", "/etc/passwd", "..\\..\\x.ply"])
def test_scene_unsafe_filenames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, filename: str
) -> None:
    """Unsafe scene filenames are errors and /etc/passwd is never opened."""
    scene_dir = tmp_path / "scene"
    xml_path = write_scene(scene_dir)
    text = xml_path.read_text(encoding="utf-8").replace('"box.ply"', f'"{filename}"', 1)
    xml_path.write_text(text, encoding="utf-8")
    payload = {
        "bundle_format_version": 1,
        "members": [{"id": "scene", "kind": "scene", "path": "scene/scene.xml"}],
    }
    write_raw_bundle_json(tmp_path, payload)
    opened: list[str] = []
    real_open = builtins.open

    def spy_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(file, (str, os.PathLike)):
            opened.append(os.fspath(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", spy_open)
    member = detect_members(tmp_path)[0]
    with pytest.raises(BundleValidationError) as excinfo:
        validate_member(member, tmp_path)
    assert excinfo.value.member_id == "scene"
    assert "is unsafe" in excinfo.value.message
    assert "/etc/passwd" not in opened


def test_scene_shape_without_filename(tmp_path: Path) -> None:
    """A ply shape without a filename is rejected."""
    scene_dir = tmp_path / "scene"
    xml_path = write_scene(scene_dir)
    text = xml_path.read_text(encoding="utf-8").replace(
        '<string name="filename" value="box.ply"/>', "", 1
    )
    xml_path.write_text(text, encoding="utf-8")
    write_raw_bundle_json(
        tmp_path,
        {
            "bundle_format_version": 1,
            "members": [{"id": "scene", "kind": "scene", "path": "scene/scene.xml"}],
        },
    )
    member = detect_members(tmp_path)[0]
    with pytest.raises(BundleValidationError) as excinfo:
        validate_member(member, tmp_path)
    assert "has no filename" in excinfo.value.message


def test_scene_missing_ply(tmp_path: Path) -> None:
    """A missing referenced PLY file is rejected naming it."""
    scene_dir = tmp_path / "scene"
    write_scene(scene_dir)
    (scene_dir / "box.ply").unlink()
    write_raw_bundle_json(
        tmp_path,
        {
            "bundle_format_version": 1,
            "members": [{"id": "scene", "kind": "scene", "path": "scene/scene.xml"}],
        },
    )
    member = detect_members(tmp_path)[0]
    with pytest.raises(BundleValidationError) as excinfo:
        validate_member(member, tmp_path)
    assert "does not exist" in excinfo.value.message


def test_scene_symlink_escape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A scene PLY symlink pointing outside the root is rejected unopened."""
    root = tmp_path / "bundle"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.ply").write_bytes(b"ply-secret")
    scene_dir = root / "scene"
    xml_path = write_scene(scene_dir)
    text = xml_path.read_text(encoding="utf-8").replace('"box.ply"', '"link.ply"', 1)
    xml_path.write_text(text, encoding="utf-8")
    (scene_dir / "box.ply").unlink()
    os.symlink(outside / "secret.ply", scene_dir / "link.ply")
    write_raw_bundle_json(
        root,
        {
            "bundle_format_version": 1,
            "members": [{"id": "scene", "kind": "scene", "path": "scene/scene.xml"}],
        },
    )
    opened: list[str] = []
    real_open = builtins.open

    def spy_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(file, (str, os.PathLike)):
            opened.append(os.fspath(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", spy_open)
    member = detect_members(root)[0]
    with pytest.raises(BundleValidationError):
        validate_member(member, root)
    real_outside = os.path.realpath(outside)
    for record in opened:
        real_record = os.path.realpath(record)
        assert not real_record.startswith(real_outside + os.sep)


def test_scene_file_references_fixture(v3_root: Path) -> None:
    """The fixture scene has two ply refs with ids, bsdfs and safe paths."""
    xml_path = v3_root / "scene" / "scene.xml"
    refs = scene_file_references(xml_path)
    assert [(ref.shape_id, ref.shape_type) for ref in refs] == [("box", "ply"), ("ground", "ply")]
    assert [ref.bsdf_id for ref in refs] == ["mat-itu_concrete", "mat-itu_medium_dry_ground"]
    assert [ref.filename for ref in refs] == ["box.ply", "ground.ply"]
    assert all(not ref.unsafe for ref in refs)
    assert [ref.path for ref in refs] == [
        xml_path.parent / "box.ply",
        xml_path.parent / "ground.ply",
    ]


def test_scene_file_references_unsafe(tmp_path: Path) -> None:
    """Unsafe filenames return refs with unsafe True and path None."""
    xml_path = write_scene(tmp_path / "scene")
    text = xml_path.read_text(encoding="utf-8").replace('"box.ply"', '"/abs.ply"', 1)
    xml_path.write_text(text, encoding="utf-8")
    refs = scene_file_references(xml_path)
    assert refs[0].unsafe is True
    assert refs[0].path is None


def test_scene_file_references_unsafe_xml(tmp_path: Path) -> None:
    """A scene XML with a DTD entity is rejected as unsafe XML."""
    xml_path = tmp_path / "evil.xml"
    xml_path.write_text(
        '<?xml version="1.0"?>\n<!DOCTYPE scene [<!ENTITY a "x">]>\n<scene>&a;</scene>\n',
        encoding="utf-8",
    )
    with pytest.raises(SceneFileError) as excinfo:
        scene_file_references(xml_path)
    assert "unsafe XML" in str(excinfo.value)


def test_scene_file_references_malformed(tmp_path: Path) -> None:
    """Truncated XML is rejected as malformed."""
    xml_path = tmp_path / "bad.xml"
    xml_path.write_text("<scene><shape>", encoding="utf-8")
    with pytest.raises(SceneFileError) as excinfo:
        scene_file_references(xml_path)
    assert "malformed XML" in str(excinfo.value)


def test_scene_file_references_bad_root(tmp_path: Path) -> None:
    """A non-scene root is rejected naming the tag."""
    xml_path = tmp_path / "foo.xml"
    xml_path.write_text("<foo/>", encoding="utf-8")
    with pytest.raises(SceneFileError) as excinfo:
        scene_file_references(xml_path)
    assert "not <scene>" in str(excinfo.value)


def test_scene_file_references_inline_bsdf(tmp_path: Path) -> None:
    """A nested <bsdf> child supplies bsdf_id when no <ref> matches."""
    xml_path = tmp_path / "inline.xml"
    xml_path.write_text(
        '<scene><shape type="ply" id="s"><string name="filename" value="a.ply"/>'
        '<bsdf id="inline"/></shape></scene>',
        encoding="utf-8",
    )
    refs = scene_file_references(xml_path)
    assert refs[0].bsdf_id == "inline"
