"""Tests for the richer mock scene (ground plane + mock city)."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import trimesh
from click.testing import CliRunner

from plateau_rt.adapters.plateau.cityjson_parser import CityJSONAdapter
from plateau_rt.application.build_scene import SceneBuilder
from plateau_rt.application.scene_checks import check_scene_carrier_frequency
from plateau_rt.cli.main import cli
from plateau_rt.domain.ground import (
    GROUND_ITU_MATERIAL,
    GROUND_PLANE_ID,
    GROUND_PLANE_Z_M,
    GROUND_VALID_CARRIER_RANGE_HZ,
    check_ground_carrier_frequency,
    ground_plane_mesh,
)
from plateau_rt.domain.models import SurfaceType

REPO_ROOT = Path(__file__).resolve().parents[1]
MOCK_CITY_JSON = REPO_ROOT / "data/raw/mock_city.city.json"
MOCK_BUILDING_JSON = REPO_ROOT / "data/raw/mock_building.city.json"

EXPECTED_ROOF_HEIGHTS = {
    "bldg_city_nw": 18.0,
    "bldg_city_ne": 30.0,
    "bldg_city_se": 12.0,
    "bldg_city_sw": 24.0,
}


def test_ground_plane_mesh_geometry() -> None:
    """The 200 m ground plane is a +z-normal square of area 40000."""
    vertices, faces = ground_plane_mesh(200)

    assert vertices.shape == (4, 3)
    assert faces.shape == (2, 3)
    assert vertices.dtype == np.float64
    np.testing.assert_allclose(vertices[:, 2], -0.01)
    assert float(np.min(vertices[:, 0])) == pytest.approx(-100.0)
    assert float(np.max(vertices[:, 0])) == pytest.approx(100.0)
    assert float(np.min(vertices[:, 1])) == pytest.approx(-100.0)
    assert float(np.max(vertices[:, 1])) == pytest.approx(100.0)

    total_area = 0.0
    for face in faces:
        v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]
        normal = np.cross(v1 - v0, v2 - v0)
        assert normal[2] > 0.0
        total_area += 0.5 * float(np.linalg.norm(normal))
    assert total_area == pytest.approx(40000.0)


def test_ground_plane_mesh_rejects_non_positive_size() -> None:
    """Non-positive sizes raise ValueError."""
    with pytest.raises(ValueError):
        ground_plane_mesh(0)
    with pytest.raises(ValueError):
        ground_plane_mesh(-1.0)


def test_ground_plane_mesh_rejects_non_finite_size() -> None:
    """NaN and inf sizes raise ValueError."""
    with pytest.raises(ValueError):
        ground_plane_mesh(math.nan)
    with pytest.raises(ValueError):
        ground_plane_mesh(math.inf)


def test_scene_builder_rejects_bad_ground_size(tmp_path: Path) -> None:
    """Negative, NaN and inf ground sizes raise ValueError."""
    for bad in (-5.0, math.nan, math.inf):
        with pytest.raises(ValueError):
            SceneBuilder(MOCK_CITY_JSON, tmp_path / "out", ground_plane_size_m=bad)


def test_build_cli_rejects_negative_ground_size(tmp_path: Path) -> None:
    """The CLI rejects --ground-plane-size-m -5 with exit code 2."""
    result = CliRunner().invoke(
        cli,
        ["build", str(MOCK_CITY_JSON), str(tmp_path / "out"), "--ground-plane-size-m", "-5"],
    )
    assert result.exit_code == 2


def test_mock_city_parses_to_four_buildings() -> None:
    """The mock city has 4 box buildings with expected surfaces and extents."""
    scene = CityJSONAdapter(MOCK_CITY_JSON).parse()

    assert len(scene.buildings) == 4
    assert {b.building_id for b in scene.buildings} == set(EXPECTED_ROOF_HEIGHTS)

    all_z: list[float] = []
    all_x: list[float] = []
    all_y: list[float] = []
    for building in scene.buildings:
        kinds = [s.surface_type for s in building.surfaces]
        assert kinds.count(SurfaceType.ROOF) == 1
        assert kinds.count(SurfaceType.GROUND) == 1
        assert kinds.count(SurfaceType.WALL) == 4

        roof_z = max(v[2] for s in building.surfaces for v in s.vertices)
        assert roof_z == pytest.approx(EXPECTED_ROOF_HEIGHTS[building.building_id])

        for surface in building.surfaces:
            for x, y, z in surface.vertices:
                all_x.append(x)
                all_y.append(y)
                all_z.append(z)

    assert min(all_z) == pytest.approx(0.0)
    assert min(all_x) == pytest.approx(-26.0)
    assert max(all_x) == pytest.approx(26.0)
    assert min(all_y) == pytest.approx(-24.0)
    assert max(all_y) == pytest.approx(24.0)


def test_scene_builder_with_ground_plane(tmp_path: Path) -> None:
    """Building the mock city with a ground plane adds PLY, XML and manifest entries."""
    out_dir = tmp_path / "mock_city"
    xml_path = SceneBuilder(MOCK_CITY_JSON, out_dir, ground_plane_size_m=200).run()

    assert (out_dir / "ground_plane.ply").exists()

    root = ET.parse(xml_path).getroot()
    shapes = {el.get("id"): el for el in root.findall("shape")}
    assert GROUND_PLANE_ID in shapes
    refs = [el.get("id") for el in shapes[GROUND_PLANE_ID].findall("ref")]
    assert f"mat-{GROUND_ITU_MATERIAL}" in refs

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "ground_plane" in manifest
    assert isinstance(manifest["ground_plane"]["size_m"], float)
    assert manifest["ground_plane"]["size_m"] == pytest.approx(200.0)
    assert manifest["ground_plane"]["z_m"] == pytest.approx(GROUND_PLANE_Z_M)
    assert manifest["ground_plane"]["material"] == GROUND_ITU_MATERIAL
    assert manifest["ground_plane"]["valid_carrier_range_hz"] == [1e9, 1e10]
    assert manifest["material_mapping"]["ground_plane"] == "itu_medium_dry_ground"


def test_ground_plane_ply_bounds_and_normals(tmp_path: Path) -> None:
    """The exported ground PLY spans [-100, 100]^2 at z=-0.01 with +z normals."""
    out_dir = tmp_path / "mock_city"
    SceneBuilder(MOCK_CITY_JSON, out_dir, ground_plane_size_m=200).run()

    mesh = trimesh.load(out_dir / "ground_plane.ply", process=False)
    np.testing.assert_allclose(
        mesh.bounds,
        [[-100, -100, -0.01], [100, 100, -0.01]],
        atol=1e-5,
    )
    assert bool(np.all(mesh.face_normals[:, 2] > 0))


def test_mock_city_mesh_counts_with_and_without_ground(tmp_path: Path) -> None:
    """Mock city has 13 meshes with ground and 12 without."""
    with_ground = tmp_path / "with_ground"
    xml_with = SceneBuilder(MOCK_CITY_JSON, with_ground, ground_plane_size_m=200).run()
    manifest_with = json.loads((with_ground / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_with["outputs"]["mesh_count"] == 13

    without_ground = tmp_path / "without_ground"
    xml_without = SceneBuilder(MOCK_CITY_JSON, without_ground).run()
    manifest_without = json.loads((without_ground / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_without["outputs"]["mesh_count"] == 12

    root_with = ET.parse(xml_with).getroot()
    bsdf_ids_with = [el.get("id") for el in root_with.findall("bsdf")]
    assert bsdf_ids_with == ["mat-itu_concrete", "mat-itu_medium_dry_ground"]
    shape_ids_with = [el.get("id") for el in root_with.findall("shape")]
    assert GROUND_PLANE_ID in shape_ids_with

    root = ET.parse(xml_without).getroot()
    bsdf_ids = [el.get("id") for el in root.findall("bsdf")]
    assert f"mat-{GROUND_ITU_MATERIAL}" not in bsdf_ids
    shape_ids = [el.get("id") for el in root.findall("shape")]
    assert GROUND_PLANE_ID not in shape_ids


def test_scene_builder_default_has_no_ground_plane(tmp_path: Path) -> None:
    """The default build leaves the old mock outputs unchanged (no ground key)."""
    out_dir = tmp_path / "mock_building"
    xml_path = SceneBuilder(MOCK_BUILDING_JSON, out_dir).run()

    assert not (out_dir / "ground_plane.ply").exists()
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "ground_plane" not in manifest
    ply_files = list(out_dir.glob("*.ply"))
    assert manifest["outputs"]["mesh_count"] == len(ply_files)

    root = ET.parse(xml_path).getroot()
    bsdf_ids = [el.get("id") for el in root.findall("bsdf")]
    assert bsdf_ids == ["mat-itu_concrete"]


def test_scene_xml_material_order_is_hash_seed_independent(tmp_path: Path) -> None:
    """Scene XML is byte-identical for PYTHONHASHSEED 1-4 with sorted bsdf ids."""
    code = (
        "import sys; sys.path.insert(0, %r); "
        "from pathlib import Path; "
        "from plateau_rt.application.build_scene import SceneBuilder; "
        "out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); "
        "SceneBuilder(Path(%r), out, ground_plane_size_m=200).run()"
        % (str(REPO_ROOT / "src"), str(MOCK_CITY_JSON))
    )
    xml_texts: list[bytes] = []
    for seed in (1, 2, 3, 4):
        out_dir = tmp_path / f"seed{seed}"
        env = {**os.environ, "PYTHONHASHSEED": str(seed), "PYTHONPATH": str(REPO_ROOT / "src")}
        subprocess.run([sys.executable, "-c", code, str(out_dir)], env=env, check=True)
        xml_files = list(out_dir.glob("*.xml"))
        assert len(xml_files) == 1
        xml_texts.append(xml_files[0].read_bytes())
    assert all(text == xml_texts[0] for text in xml_texts[1:])

    root = ET.fromstring(xml_texts[0])
    bsdf_ids = [str(el.get("id")) for el in root.findall("bsdf")]
    assert bsdf_ids == sorted(bsdf_ids)


def test_check_ground_carrier_frequency() -> None:
    """The ground material accepts 1-10 GHz and rejects anything else."""
    for ok in (1e9, 3.5e9, 10e9):
        check_ground_carrier_frequency(ok)
    for bad in (0.9e9, 10.0001e9, 28e9, math.nan):
        with pytest.raises(ValueError, match=r"1.*10 GHz"):
            check_ground_carrier_frequency(bad)


def test_check_scene_carrier_frequency(tmp_path: Path) -> None:
    """Scenes with ground fail fast at 28 GHz; scenes without pass."""
    city_dir = tmp_path / "city"
    city_xml = SceneBuilder(MOCK_CITY_JSON, city_dir, ground_plane_size_m=200).run()
    check_scene_carrier_frequency(city_xml, 3.5e9)
    with pytest.raises(ValueError, match=r"1.*10 GHz"):
        check_scene_carrier_frequency(city_xml, 28e9)

    plain_dir = tmp_path / "plain"
    plain_xml = SceneBuilder(MOCK_BUILDING_JSON, plain_dir).run()
    check_scene_carrier_frequency(plain_xml, 28e9)


def test_ground_valid_carrier_range_manifest(tmp_path: Path) -> None:
    """The manifest records the ground material's valid carrier range."""
    out_dir = tmp_path / "mock_city"
    SceneBuilder(MOCK_CITY_JSON, out_dir, ground_plane_size_m=200).run()
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["ground_plane"]["valid_carrier_range_hz"] == [1e9, 1e10]
    assert list(GROUND_VALID_CARRIER_RANGE_HZ) == [1e9, 1e10]


def test_multiview_cli_rejects_unsupported_carrier(tmp_path: Path) -> None:
    """rf-camera-multiview fails fast with exit code 2 at 28 GHz on ground scenes."""
    city_dir = tmp_path / "city"
    city_xml = SceneBuilder(MOCK_CITY_JSON, city_dir, ground_plane_size_m=200).run()
    result = CliRunner().invoke(
        cli,
        [
            "rf-camera-multiview",
            str(city_xml),
            str(tmp_path / "out"),
            "--carrier-ghz",
            "28",
        ],
    )
    assert result.exit_code == 2
    assert "10 GHz" in result.output


def test_single_view_cli_rejects_unsupported_carrier(tmp_path: Path) -> None:
    """rf-camera fails fast with exit code 2 at 28 GHz on ground scenes."""
    city_dir = tmp_path / "city"
    city_xml = SceneBuilder(MOCK_CITY_JSON, city_dir, ground_plane_size_m=200).run()
    result = CliRunner().invoke(
        cli,
        [
            "rf-camera",
            str(city_xml),
            str(tmp_path / "out"),
            "--carrier-ghz",
            "28",
        ],
    )
    assert result.exit_code == 2
    assert "10 GHz" in result.output


def test_sionna_ground_material_cross_check(tmp_path: Path) -> None:
    """Cross-check our constant against Sionna's ITU table and scene loading (CPU)."""
    city_dir = tmp_path / "city"
    city_xml = SceneBuilder(MOCK_CITY_JSON, city_dir, ground_plane_size_m=200).run()
    code = "\n".join(
        [
            "import sys",
            f"sys.path.insert(0, {str(REPO_ROOT / 'src')!r})",
            "from plateau_rt.domain.ground import GROUND_VALID_CARRIER_RANGE_HZ",
            "import sionna.rt",
            "from sionna.rt.radio_materials.itu import ITU_MATERIALS_PROPERTIES",
            "props = ITU_MATERIALS_PROPERTIES['medium_dry_ground']",
            "assert list(props.keys()) == [(1.0, 10.0)], props.keys()",
            "assert (GROUND_VALID_CARRIER_RANGE_HZ[0] / 1e9, "
            "GROUND_VALID_CARRIER_RANGE_HZ[1] / 1e9) == (1.0, 10.0)",
            f"scene = sionna.rt.load_scene({str(city_xml)!r})",
            "scene.frequency = 3.5e9",
            "scene.frequency = 1e9",
            "scene.frequency = 10e9",
            "assert 'ground_plane' in scene.objects",
            "try:",
            "    scene.frequency = 28e9",
            "except ValueError:",
            "    pass",
            "else:",
            "    raise AssertionError('28 GHz should raise ValueError')",
        ]
    )
    probe = None
    try:
        probe = subprocess.run(
            [sys.executable, "-c", "import sionna.rt"],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        pytest.skip("sionna import probe timed out")
    assert probe is not None
    if probe.returncode != 0:
        pytest.skip("sionna.rt not available")
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)
