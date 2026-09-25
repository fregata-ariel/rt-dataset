"""Tests for SceneTransform, the CityJSON CRS/origin plumbing and build manifests."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import numpy as np
import pytest

from plateau_rt.adapters.plateau.cityjson_parser import CityJSONAdapter
from plateau_rt.application.build_scene import SceneBuilder
from plateau_rt.application.rf_dataset_manifest import ManifestError, load_scene_transform
from plateau_rt.domain.scene_transform import SCENE_TRANSFORM_DEFINITION, SceneTransform

REPO_ROOT = Path(__file__).resolve().parents[1]
MOCK_CITY_JSON = REPO_ROOT / "data/raw/mock_city.city.json"


def _decode_raw_vertices(path: Path) -> np.ndarray:
    """Decode CityJSON vertices with the file's scale/translate (independent copy)."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    city_transform = raw.get("transform", {})
    scale = np.asarray(city_transform.get("scale", [1.0, 1.0, 1.0]), dtype=np.float64)
    translate = np.asarray(city_transform.get("translate", [0.0, 0.0, 0.0]), dtype=np.float64)
    return np.asarray(raw.get("vertices", []), dtype=np.float64) * scale + translate


def test_round_trip_inverse() -> None:
    """to_projected inverts to_local on random points."""
    transform = SceneTransform((1000.25, -2000.5, 5.0), "EPSG:6677")
    rng = np.random.default_rng(0)
    points = rng.standard_normal((50, 3)) * 100.0
    round_tripped = transform.to_projected(transform.to_local(points))
    np.testing.assert_allclose(round_tripped, points, atol=1e-9)
    np.testing.assert_allclose(
        transform.to_local(np.array([1000.25, -2000.5, 5.0])), np.zeros(3), atol=0.0, rtol=0.0
    )
    single = transform.to_projected(np.array([1.0, 2.0, 3.0]))
    assert single.shape == (3,)
    assert transform.to_local(np.array([1.0, 2.0, 3.0])).shape == (3,)
    with pytest.raises(ValueError):
        transform.to_projected(np.zeros((2, 4)))
    with pytest.raises(ValueError):
        transform.to_local(np.zeros((2, 4)))


def test_translation_and_payload_round_trip() -> None:
    """translation_xyz is -origin and the payload round-trips through JSON."""
    transform = SceneTransform((1000.25, -2000.5, 5.0), "EPSG:6677")
    assert transform.translation_xyz == (-1000.25, 2000.5, -5.0)
    payload = transform.to_payload()
    assert json.dumps(payload)
    assert SceneTransform.from_payload(payload) == transform
    assert payload["definition"] == SCENE_TRANSFORM_DEFINITION


def test_from_payload_rejects_bad_inputs() -> None:
    """Malformed payloads raise ValueError."""
    good = SceneTransform((1.0, 2.0, 3.0), "EPSG:6677").to_payload()
    cases = [
        {k: v for k, v in good.items() if k != "origin_projected_xyz"},
        {**good, "origin_projected_xyz": [1.0, 2.0]},
        {**good, "origin_projected_xyz": [1.0, 2.0, float("nan")]},
        {**good, "origin_projected_xyz": [1.0, True, 3.0]},
        {**good, "translation_xyz": [0.0, -2.0, -3.0]},
        {**good, "source_crs": ""},
    ]
    for payload in cases:
        with pytest.raises(ValueError):
            SceneTransform.from_payload(payload)


def test_legacy_center() -> None:
    """Old center_lat_lon maps to a legacy transform with unknown z."""
    legacy = SceneTransform.from_legacy_center([10.0, 20.0])
    assert legacy.legacy is True
    assert legacy.origin_projected_xyz[0] == 10.0
    assert legacy.origin_projected_xyz[1] == 20.0
    assert math.isnan(legacy.origin_projected_xyz[2])
    projected = legacy.to_projected(np.array([0.0, 0.0, 1.0]))
    assert projected[0] == 10.0 and projected[1] == 20.0 and math.isnan(projected[2])
    with pytest.raises(ValueError):
        legacy.to_payload()
    with pytest.raises(ValueError):
        SceneTransform.from_legacy_center([1.0])


def test_parser_origin_and_missing_crs() -> None:
    """The parser recovers the mock-city origin independently; CRS is None."""
    scene = CityJSONAdapter(MOCK_CITY_JSON).parse()
    decoded = _decode_raw_vertices(MOCK_CITY_JSON)
    expected = (
        float((decoded[:, 0].min() + decoded[:, 0].max()) / 2.0),
        float((decoded[:, 1].min() + decoded[:, 1].max()) / 2.0),
        float(decoded[:, 2].min()),
    )
    assert scene.transform.origin_projected_xyz == expected
    assert scene.transform.origin_projected_xyz == (1000.0, 2000.0, 5.0)
    assert scene.transform.source_crs is None


def test_build_round_trip_against_raw_vertices(tmp_path: Path) -> None:
    """Acceptance: the reader's transform maps scene vertices back onto raw ones."""
    out_dir = tmp_path / "out"
    SceneBuilder(MOCK_CITY_JSON, out_dir).run()
    transform = load_scene_transform(out_dir)
    assert transform is not None and not transform.legacy

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["center_lat_lon"] == [1000.0, 2000.0]
    assert isinstance(manifest.get("provenance"), dict)
    assert isinstance(manifest["provenance"].get("packages"), dict)

    decoded = _decode_raw_vertices(MOCK_CITY_JSON)
    scene = CityJSONAdapter(MOCK_CITY_JSON).parse()
    local = np.array(
        [
            vertex
            for building in scene.buildings
            for surface in building.surfaces
            for vertex in surface.vertices
        ],
        dtype=np.float64,
    )
    projected = transform.to_projected(local)
    distances = np.linalg.norm(projected[:, None, :] - decoded[None, :, :], axis=-1)
    assert bool((distances.min(axis=1) <= 1e-9).all())

    identity = np.linalg.norm(local[:, None, :] - decoded[None, :, :], axis=-1)
    assert bool((identity.min(axis=1) > 1e-9).all())

    relocated = transform.to_local(decoded)
    assert float(relocated[:, 2].min()) == 0.0
    assert abs(float((relocated[:, 0].min() + relocated[:, 0].max()) / 2.0)) <= 1e-12
    assert abs(float((relocated[:, 1].min() + relocated[:, 1].max()) / 2.0)) <= 1e-12


def test_build_records_crs(tmp_path: Path) -> None:
    """A referenceSystem in the CityJSON metadata becomes source_crs."""
    crs = "https://www.opengis.net/def/crs/EPSG/0/6677"
    copied = tmp_path / "city_crs.city.json"
    shutil.copy(MOCK_CITY_JSON, copied)
    raw = json.loads(copied.read_text(encoding="utf-8"))
    raw["metadata"] = {"referenceSystem": crs}
    copied.write_text(json.dumps(raw), encoding="utf-8")

    out_dir = tmp_path / "out"
    SceneBuilder(copied, out_dir).run()
    transform = load_scene_transform(out_dir)
    assert transform is not None
    assert transform.source_crs == crs


def test_legacy_build_manifest(tmp_path: Path) -> None:
    """Old manifests map center_lat_lon to a legacy transform; bad files raise."""
    legacy_path = tmp_path / "legacy.json"
    legacy_path.write_text(json.dumps({"center_lat_lon": [1.5, 2.5]}), encoding="utf-8")
    legacy = load_scene_transform(legacy_path)
    assert legacy is not None and legacy.legacy is True
    assert legacy.origin_projected_xyz[0] == 1.5
    assert legacy.origin_projected_xyz[1] == 2.5
    assert math.isnan(legacy.origin_projected_xyz[2])

    empty_path = tmp_path / "empty.json"
    empty_path.write_text(json.dumps({}), encoding="utf-8")
    assert load_scene_transform(empty_path) is None

    bad_path = tmp_path / "bad.json"
    bad_path.write_text(
        json.dumps({"scene_transform": {"origin_projected_xyz": [1, 2]}}), encoding="utf-8"
    )
    with pytest.raises(ManifestError):
        load_scene_transform(bad_path)
    with pytest.raises(ManifestError):
        load_scene_transform(tmp_path / "missing.json")

    both_path = tmp_path / "both.json"
    fresh = SceneTransform((7.0, 8.0, 9.0), "EPSG:6677").to_payload()
    both_path.write_text(
        json.dumps({"scene_transform": fresh, "center_lat_lon": [1.5, 2.5]}), encoding="utf-8"
    )
    both = load_scene_transform(both_path)
    assert both is not None and not both.legacy
    assert both.origin_projected_xyz == (7.0, 8.0, 9.0)
