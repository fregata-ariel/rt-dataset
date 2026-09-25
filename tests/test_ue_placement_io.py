"""Tests for Sionna-free radio-map persistence and placement manifest glue."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from rf_manifest_fixtures import BS_POSITIONS_M, TARGET_M, write_v3_dataset

from plateau_rt.application.rf_dataset_manifest import (
    MANIFEST_FILE_NAME,
    load_rf_dataset_manifest,
)
from plateau_rt.application.ue_placement import (
    PLACEMENT_DIR,
    RADIO_MAP_INDOOR_MASK_FILE,
    RADIO_MAP_LOS_MASK_FILE,
    RADIO_MAP_METADATA_FILE,
    RADIO_MAP_PATH_GAIN_FILE,
    building_exclusion_mask,
    check_radio_map_matches,
    copy_radio_map,
    file_sha256,
    load_radio_map,
    placement_manifest_section,
    replan_from_manifest,
    save_radio_map,
)
from plateau_rt.domain.rf_camera.placement import (
    CoveragePlacementSettings,
    CoverageThreshold,
    RadioMapGrid,
    box_footprint,
    footprint_mask,
    plan_coverage_placement,
)

LAMBDA_M = 0.1
UE_HEIGHT = 1.5
CARRIER_HZ = 3.5e9
SOLVER = {"solver": "sionna.rt.RadioMapSolver", "max_depth": 5, "samples_per_tx": 10000}


def make_grid(size: float = 40.0, cell: float = 1.0) -> RadioMapGrid:
    """Return a square test grid centred on the origin at UE height."""
    return RadioMapGrid(
        center_m=(0.0, 0.0, UE_HEIGHT),
        size_m=(size, size),
        cell_size_m=(cell, cell),
    )


def free_space_gain(
    grid: RadioMapGrid, bs_positions: list[tuple[float, float, float]]
) -> np.ndarray:
    """Return a linear free-space-like gain map ``[B, ny, nx]``."""
    centers = grid.cell_centers()
    layers = []
    for bs in bs_positions:
        dist = np.linalg.norm(centers - np.asarray(bs, dtype=np.float64), axis=-1)
        layers.append((LAMBDA_M / (4.0 * np.pi * dist)) ** 2)
    return np.stack(layers, axis=0).astype(np.float32)


def _base_stations() -> list[dict]:
    """Return the fixture base stations as radio-map metadata entries."""
    return [
        {"bs_id": f"bs_{index:03d}", "position_m": list(position), "look_at_m": list(TARGET_M)}
        for index, position in enumerate(BS_POSITIONS_M)
    ]


def _saved(tmp_path: Path) -> tuple:
    """Write and load a small radio map; return ``(saved, gain, mask, grid)``."""
    grid = make_grid()
    gain = free_space_gain(grid, list(BS_POSITIONS_M))
    mask = footprint_mask(grid, [box_footprint(-5.0, -5.0, 5.0, 5.0)])
    json_path = save_radio_map(
        tmp_path,
        path_gain=gain,
        indoor_mask=mask,
        grid=grid,
        solver=SOLVER,
        base_stations=_base_stations(),
        carrier_frequency_hz=CARRIER_HZ,
        source_scene="mock_scene.xml",
    )
    return load_radio_map(json_path), gain, mask, grid


def test_save_load_round_trip(tmp_path: Path) -> None:
    saved, gain, mask, grid = _saved(tmp_path)
    placement_dir = tmp_path / PLACEMENT_DIR
    assert (placement_dir / RADIO_MAP_PATH_GAIN_FILE).is_file()
    assert (placement_dir / RADIO_MAP_INDOOR_MASK_FILE).is_file()
    assert (placement_dir / RADIO_MAP_METADATA_FILE).is_file()

    assert saved.path_gain.dtype == np.float32
    assert saved.indoor_mask.dtype == bool
    assert np.array_equal(saved.path_gain, gain.astype(np.float32))
    assert np.array_equal(saved.indoor_mask, mask)
    assert saved.grid == grid
    assert saved.metadata["format_version"] == 2
    assert saved.los_mask is None
    assert "los_mask" not in saved.metadata["artifacts"]
    assert "los_mask" not in saved.metadata["sha256"]
    assert saved.metadata["path_gain_axis_order"] == ["bs", "y", "x"]
    assert saved.metadata["path_gain_scale"] == "linear"
    assert saved.metadata["source_scene"] == "mock_scene.xml"
    assert saved.metadata_path == placement_dir / RADIO_MAP_METADATA_FILE

    for key, file_name in (
        ("path_gain", RADIO_MAP_PATH_GAIN_FILE),
        ("indoor_mask", RADIO_MAP_INDOOR_MASK_FILE),
    ):
        assert saved.metadata["sha256"][key] == file_sha256(placement_dir / file_name)
        assert saved.metadata["artifacts"][key] == file_name


def test_load_via_json_placement_and_dataset_dir(tmp_path: Path) -> None:
    saved, gain, mask, _ = _saved(tmp_path)
    for path in (
        saved.metadata_path,
        tmp_path / PLACEMENT_DIR,
        tmp_path,
    ):
        loaded = load_radio_map(path)
        assert np.array_equal(loaded.path_gain, saved.path_gain)
        assert np.array_equal(loaded.indoor_mask, saved.indoor_mask)
        assert loaded.grid == saved.grid
    assert np.array_equal(saved.path_gain, gain.astype(np.float32))
    assert np.array_equal(saved.indoor_mask, mask)


def test_tampered_npy_raises_naming_the_file(tmp_path: Path) -> None:
    saved, _, _, _ = _saved(tmp_path)
    artifact = tmp_path / PLACEMENT_DIR / RADIO_MAP_PATH_GAIN_FILE
    np.save(artifact, np.ones((2, 40, 40), dtype=np.float32))
    with pytest.raises(ValueError, match=RADIO_MAP_PATH_GAIN_FILE):
        load_radio_map(saved.metadata_path)


def test_copy_radio_map_byte_identical(tmp_path: Path) -> None:
    source = tmp_path / "source"
    saved, _, _, _ = _saved(source)
    destination = tmp_path / "destination"
    copied = copy_radio_map(saved, destination)
    assert copied == destination / PLACEMENT_DIR / RADIO_MAP_METADATA_FILE
    for file_name in (
        RADIO_MAP_METADATA_FILE,
        RADIO_MAP_PATH_GAIN_FILE,
        RADIO_MAP_INDOOR_MASK_FILE,
    ):
        assert (source / PLACEMENT_DIR / file_name).read_bytes() == (
            destination / PLACEMENT_DIR / file_name
        ).read_bytes()
    # Copying into the source directory is a no-op that returns the same path.
    assert copy_radio_map(saved, source) == saved.metadata_path


def test_check_radio_map_matches(tmp_path: Path) -> None:
    saved, _, _, _ = _saved(tmp_path)
    base_stations = [
        (f"bs_{index:03d}", tuple(position), tuple(TARGET_M))
        for index, position in enumerate(BS_POSITIONS_M)
    ]
    check_radio_map_matches(
        saved,
        carrier_frequency_hz=CARRIER_HZ,
        base_stations=base_stations,
        ue_height_m=UE_HEIGHT,
    )
    with pytest.raises(ValueError, match="carrier"):
        check_radio_map_matches(
            saved,
            carrier_frequency_hz=CARRIER_HZ * 1.01,
            base_stations=base_stations,
            ue_height_m=UE_HEIGHT,
        )
    with pytest.raises(ValueError, match="base stations"):
        check_radio_map_matches(
            saved,
            carrier_frequency_hz=CARRIER_HZ,
            base_stations=base_stations[:1],
            ue_height_m=UE_HEIGHT,
        )
    with pytest.raises(ValueError, match="position_m"):
        check_radio_map_matches(
            saved,
            carrier_frequency_hz=CARRIER_HZ,
            base_stations=[("bs_000", (0.0, 0.0, 0.0), tuple(TARGET_M)), base_stations[1]],
            ue_height_m=UE_HEIGHT,
        )
    with pytest.raises(ValueError, match="height"):
        check_radio_map_matches(
            saved,
            carrier_frequency_hz=CARRIER_HZ,
            base_stations=base_stations,
            ue_height_m=UE_HEIGHT + 1.0,
        )


def _plan(tmp_path: Path, settings: CoveragePlacementSettings):
    """Save a radio map, plan a placement and build its manifest section."""
    grid = make_grid()
    gain = free_space_gain(grid, list(BS_POSITIONS_M))
    mask = footprint_mask(grid, [box_footprint(-5.0, -5.0, 5.0, 5.0)])
    json_path = save_radio_map(
        tmp_path,
        path_gain=gain,
        indoor_mask=mask,
        grid=grid,
        solver=SOLVER,
        base_stations=_base_stations(),
        carrier_frequency_hz=CARRIER_HZ,
        source_scene="mock_scene.xml",
    )
    saved = load_radio_map(json_path)
    exclusion = building_exclusion_mask(mask, grid, clearance_m=1.0)
    placement = plan_coverage_placement(
        saved.path_gain,
        saved.grid,
        settings,
        exclusion_mask=exclusion,
        bs_positions=[list(position) for position in BS_POSITIONS_M],
    )
    section = placement_manifest_section(
        placement,
        saved=saved,
        dataset_dir=tmp_path,
        radio_map_source="computed",
        radio_map_origin=None,
        building_clearance_m=1.0,
    )
    return placement, section


def test_placement_manifest_section_is_json_safe(tmp_path: Path) -> None:
    settings = CoveragePlacementSettings(
        num_views=4,
        placement_seed=0,
        threshold=CoverageThreshold(mode="relative_to_max_db", value=30.0),
        aggregation="max",
        min_bs_distance_m=2.0,
        min_spacing_m=3.0,
        orientation_policy="face_bs",
    )
    placement, section = _plan(tmp_path, settings)
    text = json.dumps(section, allow_nan=False)
    rebuilt = json.loads(text)
    assert rebuilt["method"] == "coverage"
    assert rebuilt["views"] == placement.to_record()["views"]
    assert rebuilt["exclusion"]["building_clearance_m"] == 1.0
    assert rebuilt["exclusion"]["min_bs_distance_m"] == 2.0
    assert rebuilt["radio_map"]["source"] == "computed"
    assert rebuilt["radio_map"]["metadata"] == f"{PLACEMENT_DIR}/{RADIO_MAP_METADATA_FILE}"
    assert rebuilt["radio_map"]["solver"] == SOLVER
    assert rebuilt["radio_map"]["grid"] == placement.grid.to_dict()


def test_replan_from_manifest_reproduces_views(tmp_path: Path) -> None:
    write_v3_dataset(tmp_path, num_views=2, num_bs=2)
    settings = CoveragePlacementSettings(
        num_views=4,
        placement_seed=3,
        threshold=CoverageThreshold(mode="relative_to_max_db", value=30.0),
        aggregation="max",
        min_bs_distance_m=2.0,
        min_spacing_m=3.0,
        orientation_policy="face_bs",
    )
    placement, section = _plan(tmp_path, settings)
    manifest = json.loads((tmp_path / MANIFEST_FILE_NAME).read_text(encoding="utf-8"))
    manifest["placement"] = section
    (tmp_path / MANIFEST_FILE_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    parsed = load_rf_dataset_manifest(tmp_path)
    assert parsed.placement == section
    replanned = replan_from_manifest(tmp_path)
    assert replanned.views == placement.views
    assert replanned.to_record()["views"] == section["views"]
    assert replanned.to_record()["views"] == parsed.placement["views"]


def test_replan_rejects_missing_placement(tmp_path: Path) -> None:
    write_v3_dataset(tmp_path, num_views=2, num_bs=2)
    with pytest.raises(ValueError, match="placement"):
        replan_from_manifest(tmp_path)


def _los_mask(grid: RadioMapGrid) -> np.ndarray:
    """Return a bool ``[2, ny, nx]`` LoS mask (BS0: x < 0, BS1: y < 0)."""
    centres = grid.cell_centers()
    return np.stack([centres[:, :, 0] < 0.0, centres[:, :, 1] < 0.0], axis=0)


def _save_with_los(tmp_path: Path) -> tuple:
    """Save a radio map with a LoS mask and antenna info; return (saved, mask)."""
    grid = make_grid()
    gain = free_space_gain(grid, list(BS_POSITIONS_M))
    mask = footprint_mask(grid, [box_footprint(-5.0, -5.0, 5.0, 5.0)])
    los = _los_mask(grid)
    json_path = save_radio_map(
        tmp_path,
        path_gain=gain,
        indoor_mask=mask,
        grid=grid,
        solver=SOLVER,
        base_stations=_base_stations(),
        carrier_frequency_hz=CARRIER_HZ,
        source_scene="mock_scene.xml",
        los_mask=los,
        tx_pattern="tr38901",
        polarization="V",
    )
    return load_radio_map(json_path), los


def test_save_load_round_trip_with_los_mask(tmp_path: Path) -> None:
    saved, los = _save_with_los(tmp_path)
    placement_dir = tmp_path / PLACEMENT_DIR
    assert saved.los_mask is not None
    assert saved.los_mask.dtype == bool
    assert np.array_equal(saved.los_mask, los)
    assert saved.metadata["format_version"] == 2
    assert saved.metadata["artifacts"]["los_mask"] == RADIO_MAP_LOS_MASK_FILE
    assert saved.metadata["sha256"]["los_mask"] == file_sha256(
        placement_dir / RADIO_MAP_LOS_MASK_FILE
    )
    assert saved.metadata["antenna"] == {"tx_pattern": "tr38901", "polarization": "V"}


def test_tampered_or_bad_los_mask(tmp_path: Path) -> None:
    saved, los = _save_with_los(tmp_path)
    artifact = tmp_path / PLACEMENT_DIR / RADIO_MAP_LOS_MASK_FILE
    np.save(artifact, ~los)
    with pytest.raises(ValueError, match=RADIO_MAP_LOS_MASK_FILE):
        load_radio_map(saved.metadata_path)

    grid = make_grid()
    gain = free_space_gain(grid, list(BS_POSITIONS_M))
    mask = footprint_mask(grid, [box_footprint(-5.0, -5.0, 5.0, 5.0)])
    with pytest.raises(ValueError):
        save_radio_map(
            tmp_path,
            path_gain=gain,
            indoor_mask=mask,
            grid=grid,
            solver=SOLVER,
            base_stations=_base_stations(),
            carrier_frequency_hz=CARRIER_HZ,
            source_scene="mock_scene.xml",
            los_mask=np.ones((3, *grid.shape), dtype=bool),
        )


def test_load_v1_and_reject_v3(tmp_path: Path) -> None:
    saved, _, _, _ = _saved(tmp_path)
    metadata_path = saved.metadata_path
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["format_version"] = 1
    metadata.pop("antenna", None)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    loaded = load_radio_map(metadata_path)
    assert loaded.los_mask is None

    metadata["format_version"] = 3
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="format_version"):
        load_radio_map(metadata_path)


def test_copy_radio_map_copies_los_file(tmp_path: Path) -> None:
    source = tmp_path / "source"
    saved, _los = _save_with_los(source)
    destination = tmp_path / "destination"
    copy_radio_map(saved, destination)
    assert (source / PLACEMENT_DIR / RADIO_MAP_LOS_MASK_FILE).read_bytes() == (
        destination / PLACEMENT_DIR / RADIO_MAP_LOS_MASK_FILE
    ).read_bytes()


def test_check_radio_map_matches_antenna(tmp_path: Path) -> None:
    saved, _los = _save_with_los(tmp_path)
    base_stations = [
        (f"bs_{index:03d}", tuple(position), tuple(TARGET_M))
        for index, position in enumerate(BS_POSITIONS_M)
    ]
    check_radio_map_matches(
        saved,
        carrier_frequency_hz=CARRIER_HZ,
        base_stations=base_stations,
        ue_height_m=UE_HEIGHT,
        tx_pattern="tr38901",
        polarization="V",
    )
    with pytest.raises(ValueError, match="tx_pattern"):
        check_radio_map_matches(
            saved,
            carrier_frequency_hz=CARRIER_HZ,
            base_stations=base_stations,
            ue_height_m=UE_HEIGHT,
            tx_pattern="iso",
        )
    with pytest.raises(ValueError, match="polarization"):
        check_radio_map_matches(
            saved,
            carrier_frequency_hz=CARRIER_HZ,
            base_stations=base_stations,
            ue_height_m=UE_HEIGHT,
            polarization="H",
        )
    plain, _, _, _ = _saved(tmp_path / "plain")
    with pytest.raises(ValueError, match="tx_pattern"):
        check_radio_map_matches(
            plain,
            carrier_frequency_hz=CARRIER_HZ,
            base_stations=base_stations,
            ue_height_m=UE_HEIGHT,
            tx_pattern="tr38901",
        )
    check_radio_map_matches(
        plain,
        carrier_frequency_hz=CARRIER_HZ,
        base_stations=base_stations,
        ue_height_m=UE_HEIGHT,
    )


def test_replan_with_los_fraction(tmp_path: Path) -> None:
    grid = make_grid()
    gain = free_space_gain(grid, list(BS_POSITIONS_M))
    mask = footprint_mask(grid, [box_footprint(-5.0, -5.0, 5.0, 5.0)])
    los = _los_mask(grid)
    json_path = save_radio_map(
        tmp_path,
        path_gain=gain,
        indoor_mask=mask,
        grid=grid,
        solver=SOLVER,
        base_stations=_base_stations(),
        carrier_frequency_hz=CARRIER_HZ,
        source_scene="mock_scene.xml",
        los_mask=los,
    )
    saved = load_radio_map(json_path)
    settings = CoveragePlacementSettings(
        num_views=6,
        placement_seed=4,
        threshold=CoverageThreshold(mode="relative_to_max_db", value=30.0),
        aggregation="any",
        min_spacing_m=2.0,
        los_fraction=0.5,
        los_reference="any",
        orientation_policy="face_bs",
    )
    exclusion = building_exclusion_mask(mask, grid, clearance_m=1.0)
    placement = plan_coverage_placement(
        saved.path_gain,
        saved.grid,
        settings,
        exclusion_mask=exclusion,
        bs_positions=[list(p) for p in BS_POSITIONS_M],
        los_mask=saved.los_mask,
    )
    section = placement_manifest_section(
        placement,
        saved=saved,
        dataset_dir=tmp_path,
        radio_map_source="computed",
        radio_map_origin=None,
        building_clearance_m=1.0,
    )
    assert section["radio_map"]["artifacts"]["los_mask"] == (
        f"{PLACEMENT_DIR}/{RADIO_MAP_LOS_MASK_FILE}"
    )
    assert "los_mask" in section["radio_map"]["sha256"]
    assert "antenna" in section["radio_map"]

    write_v3_dataset(tmp_path, num_views=2, num_bs=2)
    manifest = json.loads((tmp_path / MANIFEST_FILE_NAME).read_text(encoding="utf-8"))
    manifest["placement"] = section
    (tmp_path / MANIFEST_FILE_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    replanned = replan_from_manifest(tmp_path)
    assert replanned.views == placement.views
    assert replanned.to_record()["views"] == section["views"]


def test_replan_rejects_legacy_jitter_sampler(tmp_path: Path) -> None:
    settings = CoveragePlacementSettings(
        num_views=4,
        placement_seed=3,
        threshold=CoverageThreshold(mode="relative_to_max_db", value=30.0),
        min_spacing_m=3.0,
        jitter_fraction=0.5,
        orientation_policy="face_bs",
    )
    _, section = _plan(tmp_path, settings)
    write_v3_dataset(tmp_path, num_views=2, num_bs=2)
    manifest = json.loads((tmp_path / MANIFEST_FILE_NAME).read_text(encoding="utf-8"))
    section.pop("sampler_version", None)
    manifest["placement"] = section
    (tmp_path / MANIFEST_FILE_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="sampler_version"):
        replan_from_manifest(tmp_path)
