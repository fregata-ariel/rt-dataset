"""Unit tests for the tomography benchmark IO helpers (T16, §2)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from rf_manifest_fixtures import TARGET_M, write_v2_dataset, write_v3_dataset

from plateau_rt.application import rf_tomography_io as tio


def _valid_row() -> dict[str, Any]:
    """Return one valid result row in ``RESULT_KEYS`` order."""
    return {
        "schema": tio.RESULT_SCHEMA,
        "scene": "scene",
        "realization": 0,
        "config": "IDP-S",
        "node": "IDP",
        "subset": "IDP",
        "sync": "S",
        "column": "core",
        "lattices": ["WB"],
        "budget": "Co",
        "track": "ideal-S",
        "space": "bv",
        "strategy": "none",
        "stage": "E1",
        "solver": "envelope_map",
        "status": "ok",
        "reason": None,
        "n_iter": None,
        "n_forward": None,
        "n_adjoint": None,
        "hyper": {"floor": 0.001},
        "n_detections": 2,
        "metrics": {"num_gt": 2},
        "gauge_errors": None,
        "ill_posed": None,
        "runtime_s": 0.5,
        "recon": "recon/scene/IDP-S/E1-envelope_map/ideal-S.bv.none.r000.npz",
    }


def test_v3_load(tmp_path: Path) -> None:
    """v3 datasets load with exact aperture data, geometry, hashes and target."""
    root = tmp_path / "v3"
    write_v3_dataset(root, num_views=3, num_bs=2, rows=2, cols=3, bins=4)
    dataset = tio.load_dataset(root)
    assert dataset.y_clean.shape == (3, 2, 2, 2, 3, 4)
    assert dataset.y_clean.dtype == np.complex128
    stacked = np.stack([np.load(view.aperture_cfr_path) for view in dataset.manifest.views]).astype(
        np.complex128
    )
    np.testing.assert_array_equal(dataset.y_clean, stacked)
    manifest = dataset.manifest
    np.testing.assert_array_equal(dataset.geom.ue_pos, [v.position_m for v in manifest.views])
    np.testing.assert_array_equal(
        dataset.geom.bs_pos, [b.position_m for b in manifest.base_stations]
    )
    for index, view in enumerate(manifest.views):
        pose = json.loads((view.pose_path).read_text(encoding="utf-8"))
        np.testing.assert_allclose(
            dataset.geom.ue_rot[index], pose["world_from_local_rotation"], atol=1e-12, rtol=0.0
        )
    np.testing.assert_allclose(
        dataset.geom.freq_offsets,
        manifest.frequency_offsets_hz,
        atol=1e-6 * dataset.geom.delta_f,
        rtol=0.0,
    )
    assert dataset.geom.aperture_shape == (2, 3)
    assert (
        dataset.hashes["manifest"]
        == hashlib.sha256(manifest.manifest_path.read_bytes()).hexdigest()
    )
    for view in manifest.views:
        assert (
            dataset.hashes[f"aperture_cfr/{view.view_id}"]
            == hashlib.sha256(view.aperture_cfr_path.read_bytes()).hexdigest()
        )
    np.testing.assert_allclose(dataset.target, TARGET_M, atol=0.0, rtol=0.0)
    assert dataset.tx_pattern == "tr38901"
    assert dataset.name == root.name


def test_v2_load(tmp_path: Path) -> None:
    """v2 datasets load with a single base station."""
    root = tmp_path / "v2"
    write_v2_dataset(root, num_views=2, rows=2, cols=3, bins=4)
    dataset = tio.load_dataset(root)
    assert dataset.geom.num_bs == 1
    assert dataset.y_clean.shape[1] == 1
    assert dataset.y_clean.shape == (2, 1, 2, 2, 3, 4)


def test_bs_look_at(tmp_path: Path) -> None:
    """The BS boresight (local +x) points from the BS to its look-at target."""
    root = tmp_path / "look"
    write_v3_dataset(
        root,
        num_views=1,
        num_bs=1,
        bs_positions=[(3.0, -2.0, 40.0)],
        bs_look_at=(0.0, 0.0, 5.0),
    )
    dataset = tio.load_dataset(root)
    assert dataset.geom.bs_rot is not None
    boresight = dataset.geom.bs_rot[0][:, 0]
    direction = np.array([0.0 - 3.0, 0.0 + 2.0, 5.0 - 40.0])
    direction /= float(np.linalg.norm(direction))
    np.testing.assert_allclose(boresight, direction, atol=1e-12, rtol=0.0)


def test_los_visibility(tmp_path: Path) -> None:
    """LoS visibility follows the path GT and falls back to assumed."""
    root = tmp_path / "los"
    write_v3_dataset(root)
    manifest_path = root / "dataset_manifest.json"
    dataset = tio.load_dataset(root)
    assert dataset.los_visible_source == "path_gt"
    assert dataset.los_visible.shape == (dataset.geom.num_views, dataset.geom.num_bs)
    assert bool(np.all(dataset.los_visible))
    gt_path = root / "path_geometry_gt.npz"
    with np.load(gt_path) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    arrays["valid"] = arrays["valid"].copy()
    arrays["valid"][1, 0, :] = False
    np.savez_compressed(gt_path, **arrays)
    manifest = tio.load_dataset(root).manifest
    visible, source = tio.los_visibility(manifest)
    assert source == "path_gt"
    assert not bool(visible[1, 0])
    assert bool(np.all(np.delete(visible.ravel(), 1 * visible.shape[1])))
    del arrays["num_interactions"]
    np.savez_compressed(gt_path, **arrays)
    manifest = tio.load_dataset(root).manifest
    visible, source = tio.los_visibility(manifest)
    assert source == "assumed"
    assert bool(np.all(visible))
    assert manifest_path.exists()


def test_mismatched_spacings(tmp_path: Path) -> None:
    """Differing horizontal/vertical spacings raise ``ValueError``."""
    root = tmp_path / "spacing"
    manifest = write_v3_dataset(root)
    manifest["config"]["vertical_spacing_lambda"] = 0.6
    (root / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        tio.load_dataset(root)


def test_ground_truth_round_trip(tmp_path: Path) -> None:
    """Ground truth round-trips points, amplitudes, spaces and virtual sources."""
    rng = np.random.default_rng(0)
    points = rng.standard_normal((3, 3))
    rho = rng.standard_normal(3) + 1j * rng.standard_normal(3)
    vs = rng.standard_normal((2, 3))
    path = tmp_path / "gt.npz"
    tio.write_ground_truth(path, points_pos=points, points_rho=rho, points_space="bv")
    loaded = tio.load_ground_truth(path)
    np.testing.assert_array_equal(loaded.points_pos, points)
    np.testing.assert_array_equal(loaded.points_rho, rho)
    assert loaded.points_space == "bv"
    assert loaded.vs_pos is None
    np.testing.assert_array_equal(loaded.positions("bv"), points)
    assert loaded.positions("vs") is None
    assert loaded.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    tio.write_ground_truth(path, vs_pos=vs)
    loaded = tio.load_ground_truth(path)
    assert loaded.points_pos is None
    np.testing.assert_array_equal(loaded.vs_pos, vs)
    assert loaded.positions("bv") is None
    np.testing.assert_array_equal(loaded.positions("vs"), vs)
    tio.write_ground_truth(path, points_pos=points, points_rho=rho, points_space="vs", vs_pos=vs)
    loaded = tio.load_ground_truth(path)
    np.testing.assert_array_equal(loaded.positions("vs"), points)
    assert loaded.positions("bv") is None
    with pytest.raises(ValueError):
        tio.write_ground_truth(path, points_pos=np.zeros((2, 2)))
    with pytest.raises(ValueError):
        tio.write_ground_truth(path, points_pos=points, points_space="xx")
    bad = tmp_path / "bad.npz"
    np.savez_compressed(bad, points_pos=np.zeros((2, 2)))
    with pytest.raises(ValueError):
        tio.load_ground_truth(bad)


def test_find_ground_truth(tmp_path: Path) -> None:
    """Ground truth is found via override and via the manifest key."""
    root = tmp_path / "gt-find"
    write_v3_dataset(root)
    points = np.array([[0.0, 0.0, 5.0]])
    override = tmp_path / "override.npz"
    tio.write_ground_truth(override, points_pos=points)
    dataset = tio.load_dataset(root)
    assert tio.find_ground_truth(dataset) is None
    found = tio.find_ground_truth(dataset, override)
    assert found is not None
    np.testing.assert_array_equal(found.positions("bv"), points)
    artifact = root / tio.GT_FILE_NAME
    tio.write_ground_truth(artifact, points_pos=points)
    manifest_path = root / "dataset_manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["tomography_gt"] = {"artifact": tio.GT_FILE_NAME}
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    dataset = tio.load_dataset(root)
    found = tio.find_ground_truth(dataset)
    assert found is not None
    np.testing.assert_array_equal(found.positions("bv"), points)


def test_to_jsonable() -> None:
    """``to_jsonable`` converts numpy, paths, tuples and non-finite floats."""
    value = {
        "a": np.float64(np.nan),
        "b": np.float32(np.inf),
        "c": -np.float64(np.inf),
        "d": np.int64(3),
        "e": np.array([1.0, np.nan]),
        "f": (1, 2),
        "g": Path("x/y"),
        "h": {"nested": np.complex128(1j)},
        7: "int-key",
    }
    converted = tio.to_jsonable(value)
    assert converted["a"] is None
    assert converted["b"] is None
    assert converted["c"] is None
    assert converted["d"] == 3 and isinstance(converted["d"], int)
    assert converted["e"] == [1.0, None]
    assert converted["f"] == [1, 2]
    assert converted["g"] == "x/y"
    assert converted["h"] == {"nested": "1j"}
    assert converted["7"] == "int-key"


def test_recon_relpath() -> None:
    """Recon paths follow the documented layout with sanitized components."""
    assert (
        tio.recon_relpath("scene", "I@n0", "E2", "kl_em", "ideal-S", "bv", "none", 3)
        == "recon/scene/I@n0/E2-kl_em/ideal-S.bv.none.r003.npz"
    )
    assert tio.recon_relpath("a/b", "c d", "E1", "e\\f", "t", "bv", "s", 0) == (
        "recon/a_b/c_d/E1-e_f/t.bv.s.r000.npz"
    )


def test_validate_result_row() -> None:
    """Valid rows pass; missing/reordered keys and bad fields raise."""
    tio.validate_result_row(_valid_row())
    row = _valid_row()
    del row["metrics"]
    with pytest.raises(ValueError):
        tio.validate_result_row(row)
    row = _valid_row()
    reordered = {key: row[key] for key in reversed(list(row))}
    with pytest.raises(ValueError):
        tio.validate_result_row(reordered)
    for key, value in [
        ("status", "broken"),
        ("recon", None),
        ("runtime_s", float("nan")),
        ("runtime_s", -1.0),
        ("stage", "E3"),
        ("space", "xx"),
    ]:
        row = _valid_row()
        row[key] = value
        with pytest.raises(ValueError):
            tio.validate_result_row(row)
    row = _valid_row()
    row["status"] = "error"
    row["reason"] = "boom"
    row["recon"] = None
    tio.validate_result_row(row)
    row = _valid_row()
    row["status"] = "ok"
    row["reason"] = "must-be-none"
    with pytest.raises(ValueError):
        tio.validate_result_row(row)

    for value in (1.5, True):
        row = _valid_row()
        row["n_forward"] = value
        with pytest.raises(ValueError):
            tio.validate_result_row(row)

    e2 = _valid_row()
    e2["stage"] = "E2"
    e2["n_iter"] = 4
    e2["n_forward"] = 5
    e2["n_adjoint"] = 6
    tio.validate_result_row(e2)
    e2["n_forward"] = None
    with pytest.raises(ValueError):
        tio.validate_result_row(e2)

    assert tio.RESULT_SCHEMA == "rf_tomo_result/2"


def test_result_row_io(tmp_path: Path) -> None:
    """Result rows stream as JSON lines and read back exactly."""
    path = tmp_path / tio.RESULTS_FILE
    row = _valid_row()
    with open(path, "w", encoding="utf-8") as handle:
        tio.write_result_row(handle, row)
    loaded = tio.read_results(path)
    assert loaded == [tio.to_jsonable(row)]


def test_validate_run_manifest() -> None:
    """Run manifests validate their keys, schema and results section."""
    payload: dict[str, Any] = {key: None for key in tio.RUN_MANIFEST_KEYS}
    payload["schema"] = tio.RUN_MANIFEST_SCHEMA
    payload["results"] = {"path": "results.jsonl", "sha256": "x", "rows": 0, "status_counts": {}}
    tio.validate_run_manifest(payload)
    broken = dict(payload)
    del broken["grid"]
    with pytest.raises(ValueError):
        tio.validate_run_manifest(broken)
    broken = dict(payload)
    broken["results"] = {"path": "results.jsonl"}
    with pytest.raises(ValueError):
        tio.validate_run_manifest(broken)


def test_software_versions() -> None:
    """Software versions expose the five documented keys."""
    versions = tio.software_versions()
    assert set(versions) == {"python", "numpy", "scipy", "plateau_rt", "git_commit"}
