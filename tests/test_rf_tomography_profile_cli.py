"""CLI tests for ``rf-tomo-dataset`` (#15 T20)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

import plateau_rt.adapters.sionna.radio_map as radio_map_module
import plateau_rt.adapters.sionna.rf_camera_dataset as dataset_module
from plateau_rt.application.rf_tomography_profile import (
    PROFILES,
    verify_tomography_dataset,
)
from plateau_rt.cli.main import cli

DUMMY_XML = '<scene version="2.1.0"/>'
MOCK_BOXES = (
    (-26.0, -12.0, 10.0, 24.0),
    (12.0, 24.0, 10.0, 24.0),
    (10.0, 26.0, -24.0, -12.0),
    (-24.0, -12.0, -24.0, -10.0),
)


class FakeDataset:
    """Record the dataset call and write a small real dataset instead of tracing."""

    instances: list[FakeDataset] = []

    def __init__(self, xml_path, *, views, config=None, placement=None, view_placements=None):
        assert config is not None
        self.xml_path = xml_path
        self.views = list(views)
        self.config = config
        self.placement = placement
        self.view_placements = (
            None if view_placements is None else [dict(entry) for entry in view_placements]
        )
        FakeDataset.instances.append(self)

    def run(self, output_dir):
        from tests.rf_manifest_fixtures import add_direct_paths_and_oracle, write_v3_dataset

        assert self.view_placements is not None
        output_dir = Path(output_dir)
        write_v3_dataset(
            output_dir,
            views=self.views,
            bs_positions=self.config.tx_positions,
            bs_look_at=self.config.tx_look_at,
            rows=2,
            cols=3,
            bins=4,
        )
        manifest_path = output_dir / "dataset_manifest.json"
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        raw["placement"] = self.placement
        for index, entry in enumerate(raw["views"]):
            rebuilt = {}
            for key, value in entry.items():
                if key == "artifacts":
                    rebuilt["placement"] = self.view_placements[index]
                rebuilt[key] = value
            raw["views"][index] = rebuilt
        manifest_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
        add_direct_paths_and_oracle(output_dir, oracle=self.config.los_free_trace)
        return manifest_path


def _synthetic_radio_map(xml_path, *, dataset_config, grid, solver):
    """Return a free-space-like radio map with mock-city indoor cells and a LoS mask."""
    centers = grid.cell_centers()
    layers = []
    los_layers = []
    for _bs_id, position, _look_at in dataset_config.resolve_base_stations():
        dist = np.linalg.norm(centers - np.asarray(position, dtype=np.float64), axis=-1)
        layers.append((0.0857 / (4.0 * np.pi * dist)) ** 2)
        los_layers.append(centers[:, :, 0] >= float(position[0]))
    gain = np.stack(layers, axis=0).astype(np.float32)
    indoor = np.zeros(centers.shape[:2], dtype=bool)
    for x_min, x_max, y_min, y_max in MOCK_BOXES:
        indoor |= (
            (centers[:, :, 0] >= x_min)
            & (centers[:, :, 0] <= x_max)
            & (centers[:, :, 1] >= y_min)
            & (centers[:, :, 1] <= y_max)
        )
    los_mask = np.stack(los_layers, axis=0).astype(bool)
    return radio_map_module.RadioMapResult(
        path_gain=gain, indoor_mask=indoor.astype(bool), grid=grid, los_mask=los_mask
    )


@pytest.fixture()
def scene_xml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A dummy scene plus a patched Sionna adapter and dataset writer."""
    xml_path = tmp_path / "scene.xml"
    xml_path.write_text(DUMMY_XML, encoding="utf-8")
    FakeDataset.instances.clear()
    monkeypatch.setattr(dataset_module, "RFMultiViewDataset", FakeDataset)
    monkeypatch.setattr(radio_map_module, "compute_radio_map", _synthetic_radio_map)
    return xml_path


def _load_check_module():
    """Load scripts/ci/check_tomography_profile.py as a module."""
    script = (
        Path(__file__).resolve().parent.parent / "scripts" / "ci" / "check_tomography_profile.py"
    )
    spec = importlib.util.spec_from_file_location("check_tomography_profile", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ci_profile_writes_bank_and_section(scene_xml: Path, tmp_path: Path, capsys) -> None:
    out_dir = tmp_path / "out"
    result = CliRunner().invoke(cli, ["rf-tomo-dataset", str(scene_xml), str(out_dir)])
    assert result.exit_code == 0, result.output
    instance = FakeDataset.instances[-1]
    assert len(instance.views) == 8
    assert instance.config.num_frequency_bins == 128
    assert instance.config.los_free_trace is True
    assert instance.config.refraction is True
    assert instance.config.diffraction is False
    manifest = json.loads((out_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["tomography"]["profile"] == "ci"
    assert manifest["placement"]["method"] == "tomography_bank"

    check = _load_check_module()
    assert check.main([str(out_dir), "--profile", "ci", "--num-views", "8", "--num-bs", "2"]) == 1
    assert "128" in capsys.readouterr().err
    verify_tomography_dataset(out_dir)


def test_full_profile_reuses_radio_map_across_variants(scene_xml: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    out_dir = tmp_path / "out"
    result = runner.invoke(
        cli,
        [
            "rf-tomo-dataset",
            str(scene_xml),
            str(out_dir),
            "--profile",
            "full",
            "--placement-seed",
            "0",
        ],
    )
    assert result.exit_code == 0, result.output
    first = FakeDataset.instances[-1]
    assert len(first.views) == 64
    assert (out_dir / "placement" / "radio_map.json").is_file()
    section = json.loads((out_dir / "dataset_manifest.json").read_text(encoding="utf-8"))[
        "tomography"
    ]
    assert section["splits"]["train_bs"] == [0, 1, 2, 3]
    assert section["splits"]["held_out_bs"] == [4]
    assert len(section["splits"]["held_out_views"]) == 16

    out2 = tmp_path / "out2"
    result = runner.invoke(
        cli,
        [
            "rf-tomo-dataset",
            str(scene_xml),
            str(out2),
            "--profile",
            "full",
            "--variant",
            "specular",
            "--radio-map",
            str(out_dir / "placement" / "radio_map.json"),
        ],
    )
    assert result.exit_code == 0, result.output
    second = FakeDataset.instances[-1]
    assert [v.position for v in second.views] == [v.position for v in first.views]
    assert [v.orientation for v in second.views] == [v.orientation for v in first.views]
    assert second.config.refraction is False
    assert second.config.specular_reflection is True
    verify_tomography_dataset(out2)


def test_tomo_dataset_options(scene_xml: Path, tmp_path: Path) -> None:
    runner = CliRunner()
    radio_map = tmp_path / "map.json"
    radio_map.write_text("{}", encoding="utf-8")
    result = runner.invoke(
        cli,
        ["rf-tomo-dataset", str(scene_xml), str(tmp_path / "ring"), "--radio-map", str(radio_map)],
    )
    assert result.exit_code == 2
    result = runner.invoke(
        cli,
        ["rf-tomo-dataset", str(scene_xml), str(tmp_path / "bad"), "--variant", "diffuse"],
    )
    assert result.exit_code == 2

    out_dir = tmp_path / "no_los_free"
    result = runner.invoke(cli, ["rf-tomo-dataset", str(scene_xml), str(out_dir), "--no-los-free"])
    assert result.exit_code == 0, result.output
    instance = FakeDataset.instances[-1]
    assert instance.config.los_free_trace is False
    manifest = json.loads((out_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    assert manifest["tomography"]["oracle_los_free"] is None

    elevated = tmp_path / "elevated"
    result = runner.invoke(
        cli,
        ["rf-tomo-dataset", str(scene_xml), str(elevated), "--elevated-height", "10"],
    )
    assert result.exit_code == 0, result.output
    views = FakeDataset.instances[-1].views
    assert len(views) == 16
    target = PROFILES["ci"].target
    for view in views[8:]:
        assert view.position[2] == 10.0
        horizontal = float(np.hypot(view.position[0] - target[0], view.position[1] - target[1]))
        assert horizontal == pytest.approx(30.0)
