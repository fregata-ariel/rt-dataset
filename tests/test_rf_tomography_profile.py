"""Unit tests for the tomography dataset profile (Sionna-free post-processing)."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from plateau_rt.application.rf_dataset_manifest import load_rf_dataset_manifest
from plateau_rt.application.rf_tomography_profile import (
    ORACLE_RTOL,
    PROFILES,
    TomographyProfile,
    build_pose_bank,
    build_tomography_section,
    load_tomography_section,
    replan_pose_bank,
    verify_tomography_dataset,
    write_tomography_section,
)
from plateau_rt.application.ue_placement import (
    building_exclusion_mask,
    load_radio_map,
    save_radio_map,
)
from plateau_rt.domain.rf_camera.camera import (
    RFViewSpec,
    generate_ring_views,
    look_at_orientation,
)
from plateau_rt.domain.rf_camera.paths import synthesize_cfr
from plateau_rt.domain.rf_tomography import bank, sync
from tests.rf_manifest_fixtures import add_direct_paths_and_oracle, write_v3_dataset

MOCK_BOXES = (
    (-26.0, -12.0, 10.0, 24.0),
    (12.0, 24.0, 10.0, 24.0),
    (10.0, 26.0, -24.0, -12.0),
    (-24.0, -12.0, -24.0, -10.0),
)


def _mock_city_radio_map(tmp_dir: Path, profile: TomographyProfile):
    """Save a free-space-like 5-BS radio map with mock-city indoor cells."""
    assert profile.coverage is not None
    grid = profile.coverage.grid(profile.target)
    centers = grid.cell_centers()
    layers = []
    los_layers = []
    for bs_position in profile.bs_positions:
        dist = np.linalg.norm(centers - np.asarray(bs_position, dtype=np.float64), axis=-1)
        layers.append((0.0857 / (4.0 * np.pi * dist)) ** 2)
        los_layers.append(centers[:, :, 0] >= float(bs_position[0]))
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
    save_radio_map(
        tmp_dir,
        path_gain=gain,
        indoor_mask=indoor,
        grid=grid,
        solver={"max_depth": 5, "samples_per_tx": 1, "seed": 42},
        base_stations=[
            {
                "bs_id": f"bs_{i:03d}",
                "position_m": list(p),
                "look_at_m": list(profile.target),
            }
            for i, p in enumerate(profile.bs_positions)
        ],
        carrier_frequency_hz=3.5e9,
        source_scene="mock.xml",
        los_mask=los_mask,
        tx_pattern="tr38901",
        polarization="V",
    )
    return load_radio_map(tmp_dir)


def _dataset_from_bank(
    root: Path,
    profile: TomographyProfile,
    bank_out: Any,
    *,
    oracle: bool = False,
    nlos: tuple[tuple[int, int], ...] = (),
) -> dict[str, Any]:
    """Write a small v3 dataset from a pose bank, with placement and oracle files."""
    write_v3_dataset(
        root,
        views=bank_out.views,
        bs_positions=profile.bs_positions,
        bs_look_at=profile.target,
        rows=2,
        cols=3,
        bins=4,
    )
    manifest_path = root / "dataset_manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw["placement"] = bank_out.record
    for i, entry in enumerate(raw["views"]):
        rebuilt = {}
        for key, value in entry.items():
            if key == "artifacts":
                rebuilt["placement"] = bank_out.view_placements[i]
            rebuilt[key] = value
        raw["views"][i] = rebuilt
    manifest_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    add_direct_paths_and_oracle(root, nlos=nlos, oracle=oracle)
    return json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))


def test_presets_roundtrip_and_config_kwargs() -> None:
    from plateau_rt.adapters.sionna.rf_camera_dataset import RFMultiViewConfig
    from plateau_rt.domain.rf_tomography.bank import MECHANISM_VARIANTS

    for name in ("ci", "full"):
        profile = PROFILES[name]
        profile.validate()
        assert TomographyProfile.from_dict(json.loads(json.dumps(profile.to_dict()))) == profile
    full = PROFILES["full"]
    assert len(full.bs_positions) == 5 and full.num_poses == 64
    field_names = {f.name for f in fields(RFMultiViewConfig)}
    for variant in ("specular", "refraction", "diffraction"):
        kwargs = full.dataset_config_kwargs(variant)
        assert set(kwargs) <= field_names
        config = RFMultiViewConfig(**kwargs)
        config.validate()
        assert config.num_frequency_bins == 128
        expected = MECHANISM_VARIANTS[variant]
        assert config.specular_reflection is expected.specular_reflection
        assert config.refraction is expected.refraction
        assert config.diffraction is expected.diffraction
        assert config.los_free_trace is True
    assert full.dataset_config_kwargs("specular", los_free_trace=False)["los_free_trace"] is False
    with pytest.raises(ValueError):
        full.dataset_config_kwargs("bogus")
    elevated = full.with_elevated_heights((10.0, 25.0))
    assert elevated.rings[-2:] == ((30.0, 10.0), (30.0, 25.0))
    with pytest.raises(ValueError):
        dataclasses.replace(full, num_held_out_bs=5).validate()
    with pytest.raises(ValueError):
        dataclasses.replace(full, look_at_jitter_deg=90.0).validate()
    with pytest.raises(ValueError):
        dataclasses.replace(full, coverage=None).validate()
    with pytest.raises(ValueError):
        dataclasses.replace(full, order_seeds=(0, 0)).validate()


def test_ci_pose_bank_is_a_plain_ring() -> None:
    profile = PROFILES["ci"]
    pose_bank = build_pose_bank(profile, placement_seed=0)
    assert len(pose_bank.views) == 8
    expected = generate_ring_views(
        target=(0.0, 0.0, 8.0), radius_m=40.0, ue_height_m=1.5, num_views=8
    )
    for i, view in enumerate(pose_bank.views):
        assert isinstance(view, RFViewSpec)
        assert view.view_id == f"ue_{i:06d}"
        assert view.position == expected[i].position
        assert view.look_at == (0.0, 0.0, 8.0)
        assert view.orientation == look_at_orientation(view.position, (0.0, 0.0, 8.0))
    assert pose_bank.record["sources"] == {"ring": 8, "coverage": 0}
    assert pose_bank.record["coverage"] is None
    assert pose_bank.record["rings"]["dropped"] == []
    for placement in pose_bank.view_placements:
        assert placement["source"] == "ring"
        assert placement["look_at_jitter_deg"] == [0.0, 0.0]
    json.dumps(pose_bank.record, allow_nan=False)

    near_bs = dataclasses.replace(
        profile,
        rings=((40.0, 1.5),),
        bs_positions=((40.0, 0.0, 1.5), (60.0, -40.0, 20.0)),
    )
    filtered = build_pose_bank(near_bs, placement_seed=0)
    assert len(filtered.views) == 7
    assert len(filtered.record["rings"]["dropped"]) == 1
    assert filtered.record["rings"]["dropped"][0]["reason"] == "too_close_to_bs"


def test_full_pose_bank_coverage_fill(tmp_path: Path) -> None:
    profile = PROFILES["full"]
    saved = _mock_city_radio_map(tmp_path, profile)
    pose_bank = build_pose_bank(
        profile, placement_seed=0, saved_radio_map=saved, dataset_dir=tmp_path
    )
    assert len(pose_bank.views) == 64
    assert [v.view_id for v in pose_bank.views] == [f"ue_{i:06d}" for i in range(64)]
    rings = pose_bank.record["rings"]
    assert rings["requested"] == 24
    assert len(rings["dropped"]) == 1
    drop = rings["dropped"][0]
    assert drop["ring_radius_m"] == 30.0 and drop["ring_index"] == 3
    assert drop["reason"] == "excluded"
    assert pose_bank.record["sources"] == {"ring": 23, "coverage": 41}

    ring_positions = [v.position for v in pose_bank.views[:23]]
    assert len({tuple(p) for p in ring_positions}) == 23
    exclusion = building_exclusion_mask(
        saved.indoor_mask, saved.grid, clearance_m=profile.building_clearance_m
    )
    coverage_positions = [v.position for v in pose_bank.views[23:]]
    for i, view in enumerate(pose_bank.views[23:]):
        placement = pose_bank.view_placements[23 + i]
        iy, ix = placement["cell_index"]
        assert not bool(exclusion[iy, ix])
        for ring in ring_positions:
            horizontal = float(np.hypot(view.position[0] - ring[0], view.position[1] - ring[1]))
            assert horizontal >= 5.0
        for other in coverage_positions:
            if other == view.position:
                continue
            horizontal = float(np.hypot(view.position[0] - other[0], view.position[1] - other[1]))
            assert horizontal >= 5.0 - 1e-9
        for bs in profile.bs_positions:
            dist = float(np.linalg.norm(np.asarray(view.position) - np.asarray(bs)))
            assert dist >= 10.0
        assert view.position[2] == 1.5
        assert all(abs(v) <= 15.0 for v in placement["look_at_jitter_deg"])
        assert look_at_orientation(view.position, view.look_at) == view.orientation
    jitters = np.array([p["look_at_jitter_deg"] for p in pose_bank.view_placements])
    assert bool(np.any(np.abs(jitters) > 10.0))
    cov_views = pose_bank.record["coverage"]["views"]
    assert [entry["view_id"] for entry in cov_views] == [f"ue_{23 + j:06d}" for j in range(41)]
    assert pose_bank.record["coverage"]["exclusion"]["ring_exclusion_radius_m"] == 5.0

    again = build_pose_bank(profile, placement_seed=0, saved_radio_map=saved, dataset_dir=tmp_path)
    assert [v.position for v in again.views] == [v.position for v in pose_bank.views]
    other = build_pose_bank(profile, placement_seed=1, saved_radio_map=saved, dataset_dir=tmp_path)
    assert [v.position for v in other.views[:23]] == ring_positions
    assert [v.position for v in other.views[23:]] != coverage_positions
    assert [v.orientation for v in other.views] != [v.orientation for v in pose_bank.views]
    with pytest.raises(ValueError):
        build_pose_bank(profile, placement_seed=0)


def test_replan_pose_bank(tmp_path: Path) -> None:
    for name in ("ci", "full"):
        profile = PROFILES[name]
        root = tmp_path / f"ds_{name}"
        saved = None
        kwargs: dict[str, Any] = {}
        if profile.coverage is not None:
            saved = _mock_city_radio_map(root, profile)
            kwargs = {"saved_radio_map": saved, "dataset_dir": root}
        pose_bank = build_pose_bank(profile, placement_seed=0, **kwargs)
        _dataset_from_bank(root, profile, pose_bank)
        replanned = replan_pose_bank(root)
        assert [v.view_id for v in replanned.views] == [v.view_id for v in pose_bank.views]
        assert [v.position for v in replanned.views] == [v.position for v in pose_bank.views]
        assert [v.look_at for v in replanned.views] == [v.look_at for v in pose_bank.views]
        assert [v.orientation for v in replanned.views] == [v.orientation for v in pose_bank.views]
        assert replanned.view_placements == pose_bank.view_placements
        assert json.loads(json.dumps(replanned.record)) == json.loads(json.dumps(pose_bank.record))


def test_build_tomography_section(tmp_path: Path) -> None:
    profile = PROFILES["ci"]
    pose_bank = build_pose_bank(profile, placement_seed=0)
    root = tmp_path / "ds"
    _dataset_from_bank(root, profile, pose_bank, oracle=True, nlos=((3, 0), (5, 1)))
    section = build_tomography_section(root, profile=profile, variant="refraction", split_seed=0)
    assert list(section) == [
        "schema",
        "profile",
        "variant",
        "mechanism",
        "seeds",
        "bank",
        "splits",
        "los_visible",
        "los_visible_source",
        "noise",
        "antenna",
        "oracle_los_free",
        "hashes",
    ]
    json.dumps(section, allow_nan=False)

    train_v, held_v = bank.split_bank(8, holdout_fraction=0.25, min_holdout=2, seed=0)
    assert section["splits"]["train_views"] == train_v.tolist()
    assert section["splits"]["held_out_views"] == held_v.tolist()
    assert section["splits"]["train_bs"] == [0, 1]
    assert section["splits"]["held_out_bs"] == []
    assert section["splits"]["bs_subsets"] == {"1": [0], "2": [0, 1]}
    assert section["splits"]["view_subset_sizes"] == [1, 2, 4]
    nested = section["splits"]["nested_view_orders"]["0"]
    assert sorted(nested) == sorted(section["splits"]["train_views"])
    assert section["splits"]["gauge_references"]["0"] == [nested[0], 0]

    los_visible = np.array(section["los_visible"])
    assert los_visible.shape == (8, 2)
    assert not los_visible[3, 0] and not los_visible[5, 1]
    assert los_visible.sum() == 14
    assert section["los_visible_source"] == "path_gt"

    manifest = load_rf_dataset_manifest(root)
    y = np.stack([np.load(v.aperture_cfr_path) for v in manifest.views]).astype(np.complex128)
    powers = np.sum(np.abs(y) ** 2, axis=2).mean(axis=(2, 3, 4))
    train_mask = np.isin(np.arange(8), train_v)[:, None]
    candidates = los_visible & train_mask
    order = np.argsort(powers[candidates], kind="stable")
    flat = np.flatnonzero(candidates.ravel(order="C"))
    expected_idx = flat[order[(flat.size - 1) // 2]]
    v_ref, b_ref = int(expected_idx // 2), int(expected_idx % 2)
    noise = section["noise"]
    assert noise["p_ref"] == pytest.approx(float(powers[v_ref, b_ref]), rel=1e-12)
    assert noise["c_ref"] == [v_ref, b_ref]
    assert noise["sigma2"] == pytest.approx(noise["p_ref"] / 1000.0, rel=1e-12)
    assert noise["sigma2_by_snr_db"]["20"] == pytest.approx(noise["p_ref"] / 100.0, rel=1e-12)
    scatter = np.array(noise["expected_scatter_snr_db"], dtype=np.float64)
    expected = np.array(noise["expected_snr_db"], dtype=np.float64)
    y_free = np.stack(
        [np.load(view.artifacts["aperture_cfr_los_free"]) for view in manifest.views]
    ).astype(np.complex128)
    cp_free = np.sum(np.abs(y_free) ** 2, axis=2).mean(axis=(2, 3, 4))
    with np.errstate(divide="ignore"):
        recomputed = 10.0 * np.log10(cp_free / float(noise["sigma2"]))
    finite = np.isfinite(recomputed)
    assert np.all(np.isfinite(scatter[finite]))
    assert np.allclose(scatter[finite], recomputed[finite], atol=1e-9)
    assert scatter[3, 0] == pytest.approx(expected[3, 0], abs=1e-9)
    assert scatter[5, 1] == pytest.approx(expected[5, 1], abs=1e-9)

    hashes = section["hashes"]
    for view in manifest.views:
        for key in ("aperture_cfr", "aperture_cfr_los_free"):
            path = root / manifest.raw["views"][view.index]["artifacts"][key]
            assert hashes[key][view.view_id] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert (
        hashes["path_geometry_gt"]
        == hashlib.sha256((root / "path_geometry_gt.npz").read_bytes()).hexdigest()
    )
    assert (
        hashes["path_schema"]
        == hashlib.sha256((root / "path_schema.json").read_bytes()).hexdigest()
    )
    assert (
        hashes["camera_model"]
        == hashlib.sha256((root / "camera_model.npz").read_bytes()).hexdigest()
    )

    for entry in section["antenna"]["base_stations"]:
        rotation = np.asarray(entry["world_from_local_rotation"])
        forward = rotation[:, 0]
        direction = np.asarray(entry["look_at_m"]) - np.asarray(entry["position_m"])
        direction /= np.linalg.norm(direction)
        assert np.allclose(forward, direction, atol=1e-12)
    assert section["antenna"]["tx_pattern"] == "tr38901"

    manifest_path = write_tomography_section(root, section)
    assert manifest_path == root / "dataset_manifest.json"
    assert load_tomography_section(root) == section
    assert dict(load_rf_dataset_manifest(root).tomography) == section
    reloaded = load_rf_dataset_manifest(root)
    assert reloaded.num_views == 8
    before = json.loads(manifest_path.read_text(encoding="utf-8"))
    write_tomography_section(root, section)
    after = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert before == after and list(after).count("tomography") == 1

    plain = tmp_path / "plain"
    _dataset_from_bank(plain, profile, pose_bank)
    plain_section = build_tomography_section(
        plain, profile=profile, variant="refraction", split_seed=0
    )
    assert plain_section["oracle_los_free"] is None
    assert plain_section["noise"]["scatter_power"] is None
    assert plain_section["hashes"]["aperture_cfr_los_free"] is None

    partial = tmp_path / "partial"
    _dataset_from_bank(partial, profile, pose_bank, oracle=True, nlos=((3, 0), (5, 1)))
    partial_raw = json.loads((partial / "dataset_manifest.json").read_text(encoding="utf-8"))
    del partial_raw["views"][0]["artifacts"]["aperture_cfr_los_free"]
    (partial / "views" / "ue_000000" / "rf" / "aperture_cfr_los_free.npy").unlink()
    (partial / "dataset_manifest.json").write_text(json.dumps(partial_raw), encoding="utf-8")
    with pytest.raises(ValueError):
        build_tomography_section(partial, profile=profile, variant="refraction", split_seed=0)
    moved = dataclasses.replace(profile, bs_positions=((0.0, 0.0, 30.0), (1.0, 1.0, 30.0)))
    with pytest.raises(ValueError):
        build_tomography_section(root, profile=moved, variant="refraction", split_seed=0)
    with pytest.raises(ValueError):
        build_tomography_section(root, profile=profile, variant="bogus", split_seed=0)
    flagged = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    flagged["config"]["refraction"] = False
    (root / "dataset_manifest.json").write_text(json.dumps(flagged), encoding="utf-8")
    with pytest.raises(ValueError):
        build_tomography_section(root, profile=profile, variant="refraction", split_seed=0)
    flagged["config"]["refraction"] = True
    (root / "dataset_manifest.json").write_text(json.dumps(flagged), encoding="utf-8")
    no_bank = tmp_path / "nobank"
    write_v3_dataset(
        no_bank,
        views=pose_bank.views,
        bs_positions=profile.bs_positions,
        bs_look_at=profile.target,
        rows=2,
        cols=3,
        bins=4,
    )
    with pytest.raises(ValueError):
        build_tomography_section(no_bank, profile=profile, variant="refraction", split_seed=0)


def test_noise_reference_excludes_held_out_bs(tmp_path: Path) -> None:
    """P_ref comes from training views x training BSs only (held-out BS 1 is ignored)."""
    profile = dataclasses.replace(PROFILES["ci"], num_held_out_bs=1, bs_subset_sizes=(1,))
    pose_bank = build_pose_bank(profile, placement_seed=0)
    root = tmp_path / "ds"
    _dataset_from_bank(root, profile, pose_bank)
    # Make every BS-1 capture far stronger than any BS-0 capture, so including the
    # held-out BS would move the lower median onto BS 1.
    raw = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    for view in raw["views"]:
        path = root / view["artifacts"]["aperture_cfr"]
        aperture = np.load(path)
        aperture[1] *= 100.0
        np.save(path, aperture)
    section = build_tomography_section(root, profile=profile, variant="refraction", split_seed=0)
    assert section["splits"]["train_bs"] == [0]
    assert section["splits"]["held_out_bs"] == [1]
    manifest = load_rf_dataset_manifest(root)
    y = np.stack([np.load(v.aperture_cfr_path) for v in manifest.views]).astype(np.complex128)
    powers = np.sum(np.abs(y) ** 2, axis=2).mean(axis=(2, 3, 4))
    train_v = section["splits"]["train_views"]
    ordered = sorted(train_v, key=lambda v: (powers[v, 0], v))
    v_ref = ordered[(len(ordered) - 1) // 2]
    assert section["noise"]["c_ref"] == [v_ref, 0]
    assert section["noise"]["p_ref"] == pytest.approx(float(powers[v_ref, 0]), rel=1e-12)


def test_verify_tomography_dataset(tmp_path: Path) -> None:
    profile = PROFILES["ci"]
    pose_bank = build_pose_bank(profile, placement_seed=0)
    root = tmp_path / "ds"
    _dataset_from_bank(root, profile, pose_bank, oracle=True, nlos=((3, 0), (5, 1)))
    section = build_tomography_section(root, profile=profile, variant="refraction", split_seed=0)
    write_tomography_section(root, section)
    summary = verify_tomography_dataset(root)
    assert summary["oracle_rtol"] == pytest.approx(1e-4, rel=1e-12)
    assert summary["max_oracle_residual"] <= 1e-9
    assert summary["p_ref"] == section["noise"]["p_ref"]
    check_manifest = load_rf_dataset_manifest(root)
    y_check = np.stack(
        [check_manifest.load_aperture_cfr(view) for view in check_manifest.views]
    ).astype(np.complex128)
    with np.load(root / "path_geometry_gt.npz", allow_pickle=False) as payload:
        gt = {name: np.asarray(payload[name]) for name in payload.files}
    los_mask = np.asarray(gt["valid"], dtype=bool) & (np.asarray(gt["num_interactions"]) == 0)
    a_dir = np.where(los_mask[:, :, None, None, None, :], gt["a_baseband"], 0)
    tau_dir = np.where(los_mask, gt["tau"], -1.0)[:, :, None, None, None, :]
    y_dir = synthesize_cfr(a_dir, tau_dir, check_manifest.frequency_offsets_hz)
    denom = sync.capture_power(y_check)
    shares = sync.capture_power(y_dir) / denom
    los_vis = np.asarray(section["los_visible"], dtype=bool)
    expected_share = min(
        float(shares[v, b])
        for v in range(shares.shape[0])
        for b in range(shares.shape[1])
        if bool(los_vis[v, b]) and denom[v, b] > 0.0
    )
    assert summary["min_los_share"] == pytest.approx(expected_share, rel=1e-9)

    def _rewrite(mutate) -> None:
        path = root / "dataset_manifest.json"
        raw = json.loads(path.read_text(encoding="utf-8"))
        mutate(raw)
        path.write_text(json.dumps(raw, indent=2), encoding="utf-8")

    aperture = root / "views" / "ue_000000" / "rf" / "aperture_cfr.npy"
    aperture.write_bytes((np.load(aperture) + 1.0).tobytes())
    with pytest.raises(ValueError):
        verify_tomography_dataset(root)
    _dataset_from_bank(root, profile, pose_bank, oracle=True, nlos=((3, 0), (5, 1)))
    write_tomography_section(
        root, build_tomography_section(root, profile=profile, variant="refraction", split_seed=0)
    )

    _rewrite(lambda raw: raw["tomography"]["splits"].__setitem__("train_views", [0, 1, 2]))
    with pytest.raises(ValueError):
        verify_tomography_dataset(root)
    _dataset_from_bank(root, profile, pose_bank, oracle=True, nlos=((3, 0), (5, 1)))
    write_tomography_section(
        root, build_tomography_section(root, profile=profile, variant="refraction", split_seed=0)
    )

    free = root / "views" / "ue_000003" / "rf" / "aperture_cfr_los_free.npy"
    free_arr = np.load(free)
    free_arr[0] = 1.1 * free_arr[0]
    np.save(free, free_arr)
    write_tomography_section(
        root, build_tomography_section(root, profile=profile, variant="refraction", split_seed=0)
    )
    with pytest.raises(ValueError, match="oracle los-free") as excinfo:
        verify_tomography_dataset(root)
    assert "(3, 0)" in str(excinfo.value)
    verify_tomography_dataset(root, oracle_rtol=0.5)

    _dataset_from_bank(root, profile, pose_bank, oracle=True, nlos=((3, 0), (5, 1)))
    write_tomography_section(
        root, build_tomography_section(root, profile=profile, variant="refraction", split_seed=0)
    )
    gt_path = root / "path_geometry_gt.npz"
    with np.load(gt_path, allow_pickle=False) as payload:
        gt_arrays = {name: np.asarray(payload[name]) for name in payload.files}
    gt_arrays["a_baseband"][0, 1, :, :, :, 0] = 0
    np.savez_compressed(gt_path, **gt_arrays)
    write_tomography_section(
        root, build_tomography_section(root, profile=profile, variant="refraction", split_seed=0)
    )
    with pytest.raises(ValueError, match="oracle los-free") as excinfo_gt:
        verify_tomography_dataset(root)
    assert "(0, 1)" in str(excinfo_gt.value)
    _dataset_from_bank(root, profile, pose_bank, oracle=True, nlos=((3, 0), (5, 1)))
    write_tomography_section(
        root, build_tomography_section(root, profile=profile, variant="refraction", split_seed=0)
    )

    full_profile = PROFILES["full"]
    full_root = tmp_path / "full"
    saved = _mock_city_radio_map(full_root, full_profile)
    full_bank = build_pose_bank(
        full_profile, placement_seed=0, saved_radio_map=saved, dataset_dir=full_root
    )
    _dataset_from_bank(full_root, full_profile, full_bank, oracle=True)
    write_tomography_section(
        full_root,
        build_tomography_section(
            full_root, profile=full_profile, variant="refraction", split_seed=0
        ),
    )
    verify_tomography_dataset(full_root)
    _rewrite_full = full_root / "dataset_manifest.json"
    full_raw = json.loads(_rewrite_full.read_text(encoding="utf-8"))
    full_raw["placement"]["placement_seed"] = 1
    _rewrite_full.write_text(json.dumps(full_raw, indent=2), encoding="utf-8")
    with pytest.raises(ValueError):
        verify_tomography_dataset(full_root)


def test_oracle_rtol_defaults() -> None:
    """Per-variant oracle tolerances match the GPU-traced mock-city evidence."""
    assert ORACLE_RTOL == {"specular": 1e-4, "refraction": 1e-4, "diffraction": 3e-2}
