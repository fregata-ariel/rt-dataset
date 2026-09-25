"""CLI tests for ``rf-camera-multiview --placement coverage`` (#16)."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

import plateau_rt.adapters.sionna.radio_map as radio_map_module
import plateau_rt.adapters.sionna.rf_camera_dataset as dataset_module
from plateau_rt.application.ue_placement import (
    building_exclusion_mask,
    load_radio_map,
)
from plateau_rt.cli.main import cli
from plateau_rt.domain.rf_camera.camera import generate_ring_views
from plateau_rt.domain.rf_camera.placement import (
    CoveragePlacementSettings,
    RadioMapGrid,
)

LAMBDA_M = 0.1
DUMMY_XML = '<scene version="2.1.0"/>'


class FakeDataset:
    """Record the dataset call and write nothing else."""

    instances: list[FakeDataset] = []

    def __init__(self, xml_path, *, views, config=None, placement=None):
        self.xml_path = xml_path
        self.views = list(views)
        self.config = config
        self.placement = placement
        FakeDataset.instances.append(self)

    def run(self, output_dir):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        return Path(output_dir) / "dataset_manifest.json"


def _synthetic_radio_map(xml_path, *, dataset_config, grid, solver):
    """Return a free-space-like radio map with a box of indoor cells."""
    centers = grid.cell_centers()
    layers = []
    for _bs_id, position, _look_at in dataset_config.resolve_base_stations():
        dist = np.linalg.norm(centers - np.asarray(position, dtype=np.float64), axis=-1)
        layers.append((LAMBDA_M / (4.0 * np.pi * dist)) ** 2)
    gain = np.stack(layers, axis=0).astype(np.float32)
    indoor = (np.abs(centers[:, :, 0]) <= 5.0) & (np.abs(centers[:, :, 1]) <= 5.0)
    return radio_map_module.RadioMapResult(
        path_gain=gain, indoor_mask=indoor.astype(bool), grid=grid
    )


@pytest.fixture()
def scene_xml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A dummy scene plus patched Sionna adapter and dataset writer."""
    xml_path = tmp_path / "scene.xml"
    xml_path.write_text(DUMMY_XML, encoding="utf-8")
    FakeDataset.instances.clear()
    monkeypatch.setattr(dataset_module, "RFMultiViewDataset", FakeDataset)
    monkeypatch.setattr(radio_map_module, "compute_radio_map", _synthetic_radio_map)
    return xml_path


def _coverage_args(xml_path: Path, out_dir: Path, *extra: str) -> list[str]:
    """Return CLI args for a small coverage run."""
    return [
        "rf-camera-multiview",
        str(xml_path),
        str(out_dir),
        "--placement",
        "coverage",
        "--num-views",
        "4",
        "--rm-size",
        "40",
        "40",
        "--rm-center",
        "0",
        "0",
        *extra,
    ]


def test_default_ring_uses_ring_views_and_no_placement(scene_xml: Path, tmp_path: Path) -> None:
    result = CliRunner().invoke(cli, ["rf-camera-multiview", str(scene_xml), str(tmp_path / "out")])
    assert result.exit_code == 0, result.output
    instance = FakeDataset.instances[-1]
    expected = generate_ring_views(
        target=(5.0, 5.0, 5.0), radius_m=30.0, ue_height_m=1.5, num_views=8
    )
    assert instance.views == expected
    assert instance.placement is None
    assert instance.config.carrier_frequency_hz == pytest.approx(3.5e9)
    assert not (tmp_path / "out" / "placement").exists()


def test_coverage_seed_determinism(scene_xml: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    first = runner.invoke(cli, _coverage_args(scene_xml, tmp_path / "first"))
    second = runner.invoke(cli, _coverage_args(scene_xml, tmp_path / "second"))
    other = runner.invoke(
        cli, _coverage_args(scene_xml, tmp_path / "other", "--placement-seed", "1")
    )
    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert other.exit_code == 0, other.output

    first_instance = FakeDataset.instances[-3]
    second_instance = FakeDataset.instances[-2]
    other_instance = FakeDataset.instances[-1]
    assert first_instance.views == second_instance.views
    assert [view.position for view in first_instance.views] != [
        view.position for view in other_instance.views
    ]
    assert asdict(first_instance.config) == asdict(second_instance.config)
    assert asdict(first_instance.config) == asdict(other_instance.config)
    assert first_instance.placement["method"] == "coverage"
    assert first_instance.placement["placement_seed"] == 0
    assert other_instance.placement["placement_seed"] == 1


def test_radio_map_reuse_reproduces_views_without_tracing(
    scene_xml: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = CliRunner()
    first = runner.invoke(cli, _coverage_args(scene_xml, tmp_path / "first"))
    assert first.exit_code == 0, first.output
    reference = FakeDataset.instances[-1].views

    def _forbidden(*args, **kwargs):
        raise AssertionError("compute_radio_map must not be called with --radio-map")

    monkeypatch.setattr(radio_map_module, "compute_radio_map", _forbidden)
    reused = runner.invoke(
        cli,
        _coverage_args(
            scene_xml,
            tmp_path / "reused",
            "--radio-map",
            str(tmp_path / "first" / "placement" / "radio_map.json"),
        ),
    )
    assert reused.exit_code == 0, reused.output
    assert FakeDataset.instances[-1].views == reference
    assert FakeDataset.instances[-1].placement["radio_map"]["source"] == "loaded"


def test_radio_map_reuse_rejects_different_bs(scene_xml: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    first = runner.invoke(cli, _coverage_args(scene_xml, tmp_path / "first"))
    assert first.exit_code == 0, first.output
    mismatch = runner.invoke(
        cli,
        _coverage_args(
            scene_xml,
            tmp_path / "mismatch",
            "--radio-map",
            str(tmp_path / "first" / "placement" / "radio_map.json"),
            "--bs-position",
            "1",
            "2",
            "3",
        ),
    )
    assert mismatch.exit_code != 0
    assert "base station" in mismatch.output or "base stations" in mismatch.output


def test_coverage_views_are_outside_buildings_and_on_candidates(
    scene_xml: Path, tmp_path: Path
) -> None:
    out_dir = tmp_path / "out"
    result = CliRunner().invoke(
        cli,
        _coverage_args(
            scene_xml,
            out_dir,
            "--building-clearance-m",
            "1",
            "--min-bs-distance-m",
            "5",
            "--min-ue-spacing-m",
            "2",
        ),
    )
    assert result.exit_code == 0, result.output
    instance = FakeDataset.instances[-1]
    section = instance.placement
    saved = load_radio_map(out_dir / "placement" / "radio_map.json")
    settings = CoveragePlacementSettings.from_dict(section["settings"])
    exclusion = building_exclusion_mask(
        saved.indoor_mask, saved.grid, clearance_m=section["exclusion"]["building_clearance_m"]
    )
    assert settings.orientation_policy == "face_bs"
    for entry in section["views"]:
        iy, ix = entry["cell_index"]
        assert not bool(saved.indoor_mask[iy, ix])
        assert not bool(exclusion[iy, ix])
        center = saved.grid.cell_centers()[iy, ix]
        assert abs(float(entry["position_m"][0]) - float(center[0])) <= 0.5 + 1e-9
        assert abs(float(entry["position_m"][1]) - float(center[1])) <= 0.5 + 1e-9


def test_coverage_accepts_explicit_radio_map_grid(scene_xml: Path, tmp_path: Path) -> None:
    result = CliRunner().invoke(
        cli,
        _coverage_args(
            scene_xml,
            tmp_path / "out",
            "--rm-cell-size",
            "2",
            "2",
            "--pl-threshold-mode",
            "absolute_db",
            "--pl-threshold",
            "-120",
            "--orientation-policy",
            "look_at_target",
        ),
    )
    assert result.exit_code == 0, result.output
    section = FakeDataset.instances[-1].placement
    assert section["settings"]["orientation_policy"] == "look_at_target"
    assert section["radio_map"]["grid"]["cell_size_m"] == [2.0, 2.0]


def test_coverage_rejects_bad_face_bs(scene_xml: Path, tmp_path: Path) -> None:
    result = CliRunner().invoke(
        cli, _coverage_args(scene_xml, tmp_path / "out", "--face-bs", "abc")
    )
    assert result.exit_code == 2


def test_coverage_grid_is_centered_on_target_by_default(scene_xml: Path, tmp_path: Path) -> None:
    result = CliRunner().invoke(
        cli,
        [
            "rf-camera-multiview",
            str(scene_xml),
            str(tmp_path / "out"),
            "--placement",
            "coverage",
            "--num-views",
            "4",
            "--target",
            "5",
            "5",
            "5",
            "--rm-size",
            "20",
            "20",
        ],
    )
    assert result.exit_code == 0, result.output
    grid = RadioMapGrid.from_dict(FakeDataset.instances[-1].placement["radio_map"]["grid"])
    assert grid.center_m == (5.0, 5.0, 1.5)


def test_radio_map_with_ring_placement_is_rejected(scene_xml: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    first = runner.invoke(cli, _coverage_args(scene_xml, tmp_path / "first"))
    assert first.exit_code == 0, first.output
    count = len(FakeDataset.instances)
    result = runner.invoke(
        cli,
        [
            "rf-camera-multiview",
            str(scene_xml),
            str(tmp_path / "ring"),
            "--radio-map",
            str(tmp_path / "first" / "placement" / "radio_map.json"),
        ],
    )
    assert result.exit_code == 2
    assert "--placement coverage" in result.output
    assert len(FakeDataset.instances) == count


def test_radio_map_reuse_warns_on_other_scene(scene_xml: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    first = runner.invoke(cli, _coverage_args(scene_xml, tmp_path / "first"))
    assert first.exit_code == 0, first.output
    saved_map = str(tmp_path / "first" / "placement" / "radio_map.json")
    same = runner.invoke(
        cli, _coverage_args(scene_xml, tmp_path / "same", "--radio-map", saved_map)
    )
    assert same.exit_code == 0, same.output
    assert "warning" not in same.output
    other_xml = tmp_path / "other_scene.xml"
    other_xml.write_text(DUMMY_XML, encoding="utf-8")
    other = runner.invoke(
        cli, _coverage_args(other_xml, tmp_path / "other", "--radio-map", saved_map)
    )
    assert other.exit_code == 0, other.output
    assert "warning: the saved radio map was computed on" in other.output
