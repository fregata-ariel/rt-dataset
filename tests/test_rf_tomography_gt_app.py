"""Application and CLI tests for ``tomography_gt.npz`` (T17)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest
import trimesh
from click.testing import CliRunner
from rf_manifest_fixtures import write_v3_dataset
from rf_tomography_gt_fixtures import BS_POS, OBJECT_NAMES, TARGET, build_mirror_scene

from plateau_rt.application import rf_tomography_gt as app
from plateau_rt.application import rf_tomography_io
from plateau_rt.application.rf_dataset_manifest import ManifestError, load_rf_dataset_manifest
from plateau_rt.cli.main import cli
from plateau_rt.domain.ground import ground_plane_mesh
from plateau_rt.domain.rf_camera.camera import generate_ring_views
from plateau_rt.domain.rf_camera.paths import PATH_GT_MODE_CANONICAL, build_path_schema
from plateau_rt.domain.rf_tomography.gt import PATH_TYPE_NAMES, PathGT, path_ground_truth

CARRIER_HZ = 3.5e9
VIEWS = generate_ring_views(target=TARGET, radius_m=15.0, ue_height_m=1.5, num_views=6)
PATH_KEYS = {
    "path_type",
    "path_power",
    "path_vs",
    "beyond_period",
    "los_visible",
    "ground_bounce_visible",
    "los_phase_model_error",
    "los_amp_model_error_db",
    "vs_pos",
    "vs_bs",
    "vs_order",
    "vs_num_paths",
    "vs_objects",
    "vs_plane_ids",
    "vs_spread",
    "vs_visibility",
    "vs_power",
    "vs_rho_eff",
    "vs_theta_inc",
    "vs_path_type",
    "plane_normal",
    "plane_offset",
    "plane_object",
    "plane_num_vertices",
    "interaction_points",
    "interaction_view",
    "interaction_bs",
    "interaction_path",
    "interaction_depth",
    "interaction_type",
    "interaction_object",
    "interaction_plane",
}
METADATA_KEYS = {
    "schema",
    "path_type_names",
    "path_object_names",
    "view_ids",
    "bs_ids",
    "source_path_gt_sha256",
    "pattern",
    "los_polarization",
    "cluster_tol_m",
    "delay_period_s",
}
SURFACE_KEYS = {
    "surface_samples",
    "surface_normals",
    "surface_object",
    "surface_observable",
    "surface_specular_support",
    "surface_roi",
    "surface_object_names",
    "source_mesh_sha256",
    "surface_spacing_m",
    "specular_support_radius_m",
}


def _write_scene(root: Path) -> None:
    scene_dir = root / "scene"
    scene_dir.mkdir(parents=True, exist_ok=True)
    box = trimesh.creation.box(extents=(2.0, 2.0, 3.0))
    box.apply_translation((0.0, 0.0, 1.5))
    box.export(scene_dir / "box.ply")
    vertices, faces = ground_plane_mesh(60.0, z_m=-0.01)
    ground = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    ground.export(scene_dir / "ground_plane.ply")
    (scene_dir / "scene.xml").write_text(
        '<scene version="2.1.0">\n'
        '  <shape type="ply" id="box">\n'
        '    <string name="filename" value="box.ply"/>\n'
        "  </shape>\n"
        '  <shape type="ply" id="ground_plane">\n'
        '    <string name="filename" value="ground_plane.ply"/>\n'
        "  </shape>\n"
        "</scene>\n",
        encoding="utf-8",
    )


def _make_dataset(root: Path) -> None:
    write_v3_dataset(
        root,
        views=VIEWS,
        rows=4,
        cols=4,
        bins=64,
        bs_positions=BS_POS,
        bs_look_at=TARGET,
        source_scene="scene/scene.xml",
    )
    scene = build_mirror_scene(float32=True)
    np.savez_compressed(root / "path_geometry_gt.npz", **scene.arrays)
    schema = build_path_schema(
        scene.arrays,
        mode=PATH_GT_MODE_CANONICAL,
        object_names=OBJECT_NAMES,
        carrier_frequency_hz=CARRIER_HZ,
        bs_ids=["bs_000", "bs_001"],
        view_ids=[view.view_id for view in VIEWS],
    )
    (root / "path_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    _write_scene(root)


@pytest.fixture()
def dataset_root(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    _make_dataset(root)
    return root


def test_write_tomography_gt(dataset_root: Path) -> None:
    out = app.write_tomography_gt(dataset_root)
    assert out == dataset_root / rf_tomography_io.GT_FILE_NAME
    with np.load(out, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    assert PATH_KEYS <= set(arrays)
    assert METADATA_KEYS <= set(arrays)
    assert SURFACE_KEYS <= set(arrays)
    assert arrays["schema"].item() == "rf_tomo_gt/1"
    np.testing.assert_array_equal(arrays["path_type_names"], list(PATH_TYPE_NAMES))

    scene = build_mirror_scene(float32=True)
    path = PathGT.from_arrays(scene.arrays, OBJECT_NAMES)
    data = rf_tomography_io.load_dataset(dataset_root)
    expected = path_ground_truth(path, data.geom, pattern="tr38901")
    np.testing.assert_array_equal(arrays["vs_pos"], expected["vs_pos"])
    np.testing.assert_array_equal(arrays["vs_rho_eff"], expected["vs_rho_eff"])
    assert arrays["vs_pos"].shape[0] == 7

    manifest = load_rf_dataset_manifest(dataset_root)
    entry = manifest.raw["tomography_gt"]
    assert entry["artifact"] == "tomography_gt.npz"
    assert entry["source_path_gt_sha256"] == rf_tomography_io.sha256_file(
        dataset_root / "path_geometry_gt.npz"
    )
    assert re.fullmatch(r"[0-9a-f]{64}", entry["source_mesh_sha256"])
    found = rf_tomography_io.find_ground_truth(rf_tomography_io.load_dataset(dataset_root))
    assert found is not None
    np.testing.assert_array_equal(found.vs_pos, arrays["vs_pos"])


def test_build_determinism(dataset_root: Path) -> None:
    first = app.build_tomography_gt(dataset_root)
    second = app.build_tomography_gt(dataset_root)
    assert set(first) == set(second)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])


def test_surfaces(dataset_root: Path) -> None:
    arrays = app.build_tomography_gt(dataset_root)
    assert tuple(arrays["surface_object_names"].tolist()) == ("box", "ground_plane")
    samples = arrays["surface_samples"]
    roi = arrays["surface_roi"]
    inside = np.all((samples >= roi[0]) & (samples <= roi[1]), axis=1)
    assert bool(np.all(inside))

    plus_x = (
        np.isclose(samples[:, 0], 1.0)
        & (np.abs(samples[:, 1]) <= 1.0 + 1e-9)
        & (samples[:, 2] >= 0.0)
        & (samples[:, 2] <= 3.0)
    )
    assert bool(np.all(arrays["surface_observable"][plus_x]))
    top = (
        np.isclose(samples[:, 2], 3.0)
        & (np.abs(samples[:, 0]) <= 1.0 + 1e-9)
        & (np.abs(samples[:, 1]) <= 1.0 + 1e-9)
    )
    assert not bool(np.any(arrays["surface_observable"][top]))
    footprint = (
        np.isclose(samples[:, 2], -0.01)
        & (np.abs(samples[:, 0]) < 1.0)
        & (np.abs(samples[:, 1]) < 1.0)
    )
    assert not bool(np.any(arrays["surface_observable"][footprint]))

    ground = np.isclose(samples[:, 2], -0.01)
    assert bool(np.any(arrays["surface_specular_support"][ground]))


def test_resolve_scene_and_no_surfaces(dataset_root: Path) -> None:
    manifest = load_rf_dataset_manifest(dataset_root)
    resolved = app.resolve_scene_xml(manifest)
    assert resolved == dataset_root / "scene" / "scene.xml"
    with pytest.raises(FileNotFoundError):
        app.resolve_scene_xml(manifest, dataset_root / "missing.xml")

    arrays = app.build_tomography_gt(dataset_root, surfaces=False)
    assert (SURFACE_KEYS - {"source_mesh_sha256"}).isdisjoint(arrays)
    assert str(arrays["source_mesh_sha256"].item()) == ""

    manifest_path = dataset_root / "dataset_manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw["source_scene"] = "nowhere/scene.xml"
    manifest_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        app.build_tomography_gt(dataset_root)


def test_pattern_resolution(dataset_root: Path) -> None:
    manifest_path = dataset_root / "dataset_manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw["config"]["tx_pattern"] = "iso"
    manifest_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    arrays = app.build_tomography_gt(dataset_root)
    assert str(arrays["pattern"].item()) == "iso"

    raw["config"]["tx_pattern"] = "dipole"
    manifest_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    with pytest.raises(ValueError):
        app.build_tomography_gt(dataset_root)


def test_cli(dataset_root: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["rf-tomo-gt", str(dataset_root)])
    assert result.exit_code == 0, result.output
    assert "tomography_gt:" in result.output
    assert '"num_vs": 7' in result.output

    other = tmp_path / "other"
    _make_dataset(other)
    out = tmp_path / "x.npz"
    result = runner.invoke(
        cli,
        [
            "rf-tomo-gt",
            str(other),
            "--no-surfaces",
            "--out",
            str(out),
            "--no-register",
        ],
    )
    assert result.exit_code == 0, result.output
    with np.load(out, allow_pickle=False) as payload:
        assert (SURFACE_KEYS - {"source_mesh_sha256"}).isdisjoint(set(payload.files))
    raw = json.loads((other / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert "tomography_gt" not in raw

    result = runner.invoke(cli, ["rf-tomo-gt", str(other), "--scene", str(tmp_path / "nope.xml")])
    assert result.exit_code != 0


def test_load_path_gt_requires_canonical(dataset_root: Path) -> None:
    schema_path = dataset_root / "path_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema["mode"] = "sionna_native"
    schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    manifest = load_rf_dataset_manifest(dataset_root)
    with pytest.raises(ManifestError):
        app.load_path_gt(manifest)
