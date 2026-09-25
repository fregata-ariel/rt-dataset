"""Tests for dataset-manifest metadata: provenance, power, axes and transform."""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
from rf_manifest_fixtures import write_v2_dataset, write_v3_dataset

from plateau_rt.application.provenance import collect_provenance
from plateau_rt.application.rf_dataset_manifest import (
    ManifestError,
    load_rf_dataset_manifest,
)
from plateau_rt.domain.rf_camera.camera import (
    build_direction_cosine_camera_model,
    channel_gain_reference_payload,
    image_axes_payload,
)
from plateau_rt.domain.rf_camera.optical import (
    PinholeIntrinsics,
    pinhole_ray_directions_local,
)
from plateau_rt.domain.scene_transform import SceneTransform


def test_old_datasets_have_no_metadata(tmp_path: Path) -> None:
    """v2 and v3 fixtures parse exactly as before, with None metadata."""
    v3_root = tmp_path / "v3"
    write_v3_dataset(v3_root)
    v3 = load_rf_dataset_manifest(v3_root)
    v2_root = tmp_path / "v2"
    write_v2_dataset(v2_root)
    v2 = load_rf_dataset_manifest(v2_root)
    for dataset in (v3, v2):
        assert dataset.provenance is None
        assert dataset.tx_power_dbm is None
        assert dataset.channel_gain_reference is None
        assert dataset.image_axes is None
        assert dataset.scene_transform is None


def _new_metadata_manifest() -> dict:
    """Return the writer-style metadata sections for a v3 manifest dict."""
    return {
        "provenance": collect_provenance(
            argv=["prog", "rf-camera-multiview"],
            now=datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
            environ={},
        ),
        "tx_power_dbm": 30.0,
        "channel_gain_reference": channel_gain_reference_payload(),
        "image_axes": image_axes_payload(),
        "scene_transform": SceneTransform((1000.0, 2000.0, 5.0), "EPSG:6677").to_payload(),
    }


def test_new_dataset_metadata_round_trip(tmp_path: Path) -> None:
    """Added sections read back typed, with shapes and views unchanged."""
    old_dict = write_v3_dataset(tmp_path)
    old = load_rf_dataset_manifest(tmp_path)
    manifest = copy.deepcopy(old_dict)
    meta = _new_metadata_manifest()
    manifest["provenance"] = meta["provenance"]
    manifest["config"]["tx_power_dbm"] = meta["tx_power_dbm"]
    manifest["config"]["channel_gain_reference"] = meta["channel_gain_reference"]
    manifest["camera_model"]["image_axes"] = meta["image_axes"]
    manifest["scene_transform"] = meta["scene_transform"]
    (tmp_path / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    dataset = load_rf_dataset_manifest(tmp_path)
    assert dataset.provenance == meta["provenance"]
    assert dataset.tx_power_dbm == 30.0
    assert dataset.channel_gain_reference == meta["channel_gain_reference"]
    assert dataset.image_axes == meta["image_axes"]
    assert dataset.scene_transform == SceneTransform((1000.0, 2000.0, 5.0), "EPSG:6677")
    assert dataset.aperture_cfr_shape == old.aperture_cfr_shape
    assert dataset.view_ids == old.view_ids


def _write_manifest(root: Path, manifest: dict) -> None:
    (root / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=True), encoding="utf-8"
    )


def test_malformed_metadata_raises(tmp_path: Path) -> None:
    """Each malformed metadata section raises ManifestError."""
    base = write_v3_dataset(tmp_path)

    bad = copy.deepcopy(base)
    bad["provenance"] = []
    _write_manifest(tmp_path, bad)
    with pytest.raises(ManifestError):
        load_rf_dataset_manifest(tmp_path)

    for tx_power in ("hot", float("inf")):
        bad = copy.deepcopy(base)
        bad["config"]["tx_power_dbm"] = tx_power
        _write_manifest(tmp_path, bad)
        with pytest.raises(ManifestError):
            load_rf_dataset_manifest(tmp_path)

    bad = copy.deepcopy(base)
    bad["config"]["channel_gain_reference"] = "x"
    _write_manifest(tmp_path, bad)
    with pytest.raises(ManifestError):
        load_rf_dataset_manifest(tmp_path)

    bad = copy.deepcopy(base)
    bad["camera_model"]["image_axes"] = ["row"]
    _write_manifest(tmp_path, bad)
    with pytest.raises(ManifestError):
        load_rf_dataset_manifest(tmp_path)

    bad = copy.deepcopy(base)
    bad["scene_transform"] = {"origin_projected_xyz": "abc"}
    _write_manifest(tmp_path, bad)
    with pytest.raises(ManifestError):
        load_rf_dataset_manifest(tmp_path)


def test_image_axes_match_real_camera_model() -> None:
    """kz/ky increase along rows/cols and flip both axes for photo orientation."""
    model = build_direction_cosine_camera_model(
        fft_rows=16,
        fft_cols=16,
        horizontal_spacing_lambda=0.5,
        vertical_spacing_lambda=0.5,
    )
    kz_over_k = np.asarray(model["kz_over_k"]).ravel()
    ky_over_k = np.asarray(model["ky_over_k"]).ravel()
    assert bool(np.all(np.diff(kz_over_k) > 0))
    assert bool(np.all(np.diff(ky_over_k) > 0))

    rays = np.asarray(model["ray_directions_local"])
    valid_mask = np.asarray(model["valid_mask"], dtype=bool)
    for col in range(rays.shape[1]):
        column = rays[:, col, 2][valid_mask[:, col]]
        assert bool(np.all(np.diff(column) >= 0))
    for row in range(rays.shape[0]):
        line = rays[row, :, 1][valid_mask[row, :]]
        assert bool(np.all(np.diff(line) >= 0))

    photo = pinhole_ray_directions_local(PinholeIntrinsics.from_horizontal_fov(16, 16, 90.0))
    flipped = np.flip(rays, axis=(0, 1))
    valid = np.flip(valid_mask, axis=(0, 1))
    for component in (1, 2):
        assert bool(
            (np.sign(flipped[..., component]) * np.sign(photo[..., component]) >= 0)[valid].all()
        )
    corr_y = float(np.corrcoef(flipped[..., 1][valid].ravel(), photo[..., 1][valid].ravel())[0, 1])
    assert corr_y > 0.99

    displayed = np.flipud(rays)
    displayed_valid = np.flipud(valid_mask)
    mirror_corr_y = float(
        np.corrcoef(
            displayed[..., 1][displayed_valid].ravel(), photo[..., 1][displayed_valid].ravel()
        )[0, 1]
    )
    assert mirror_corr_y < -0.99

    payload = image_axes_payload()
    assert payload["row"] == "+kz (up)"
    assert payload["col"] == "+ky (camera left)"
    assert payload["mirrored_vs_pinhole_photo"] is True
    other = image_axes_payload()
    assert other == payload and other is not payload


def test_channel_gain_reference_payload() -> None:
    """Stored channels are unit-transmit-power coefficients."""
    payload = channel_gain_reference_payload()
    assert payload["reference"] == "unit_transmit_power"
    assert payload["tx_power_applied"] is False
