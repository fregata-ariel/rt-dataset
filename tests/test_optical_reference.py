"""CPU-only tests for :mod:`plateau_rt.application.optical_reference`.

A fake renderer stands in for :class:`RayRenderer`: an analytic infinite
ground plane at world ``z = 0`` (hit iff the world ray direction has
``dir_z < 0``, ``range = origin_z / -dir_z``, constant RGB on hits). This lets
every geometric claim (hit mask, z-depth, range, the hemisphere PNG row flip)
be checked against a closed-form expectation without Sionna/Mitsuba.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from plateau_rt.application.optical_reference import render_optical_references
from plateau_rt.domain.rf_camera.calibration import rotation_matrix
from plateau_rt.domain.rf_camera.camera import (
    build_direction_cosine_camera_model,
    generate_ring_views,
    view_pose_payload,
)
from plateau_rt.domain.rf_camera.optical import (
    PinholeIntrinsics,
    camera_to_world_opengl,
    local_to_world_rays,
    pinhole_ray_directions_local,
)

WIDTH, HEIGHT, FOV_X_DEG = 16, 12, 90.0
FFT_ROWS = FFT_COLS = 16
SPP, SEED = 4, 7
GROUND_RGB = (0.2, 0.4, 0.6)


@dataclass
class _FakeRenderResult:
    rgb: np.ndarray
    hit: np.ndarray
    range_m: np.ndarray


class GroundPlaneRenderer:
    """Analytic infinite ground plane at world z=0, standing in for RayRenderer."""

    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, np.ndarray, int, int]] = []

    def render(self, origins, directions, *, spp: int = 64, seed: int = 0):
        origins = np.asarray(origins, dtype=np.float64)
        directions = np.asarray(directions, dtype=np.float64)
        self.calls.append((origins.copy(), directions.copy(), spp, seed))

        hit = directions[:, 2] < -1e-12
        range_m = np.full(origins.shape[0], np.nan, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            range_m[hit] = origins[hit, 2] / -directions[hit, 2]
        rgb = np.zeros((origins.shape[0], 3), dtype=np.float64)
        rgb[hit] = GROUND_RGB
        return _FakeRenderResult(
            rgb=rgb.astype(np.float32),
            hit=hit,
            range_m=range_m.astype(np.float32),
        )


def _build_dataset(tmp_path: Path) -> tuple[Path, list]:
    """Build a tiny 2-view dataset_manifest.json + poses + camera_model.npz."""
    views = generate_ring_views(target=(0.0, 0.0, 0.0), radius_m=20.0, ue_height_m=8.0, num_views=2)
    dataset_dir = tmp_path / "dataset"
    for view in views:
        view_dir = dataset_dir / "views" / view.view_id
        view_dir.mkdir(parents=True)
        (view_dir / "pose.json").write_text(json.dumps(view_pose_payload(view)), encoding="utf-8")

    camera_model = build_direction_cosine_camera_model(
        fft_rows=FFT_ROWS,
        fft_cols=FFT_COLS,
        horizontal_spacing_lambda=0.5,
        vertical_spacing_lambda=0.5,
    )
    np.savez(dataset_dir / "camera_model.npz", **camera_model)

    manifest = {
        "schema_version": 2,
        "source_scene": "unused_scene.xml",
        "views": [
            {"view_id": view.view_id, "position_m": list(view.position), "artifacts": {}}
            for view in views
        ],
    }
    (dataset_dir / "dataset_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return dataset_dir, views


def test_render_optical_references_writes_files_with_expected_shapes(tmp_path):
    dataset_dir, views = _build_dataset(tmp_path)

    transforms_path = render_optical_references(
        dataset_dir,
        renderer=GroundPlaneRenderer(),
        width=WIDTH,
        height=HEIGHT,
        fov_x_deg=FOV_X_DEG,
        spp=SPP,
        seed=SEED,
    )

    assert transforms_path == dataset_dir / "transforms.json"
    assert transforms_path.is_file()

    for view in views:
        optical_dir = dataset_dir / "views" / view.view_id / "optical"
        pinhole_rgba = np.asarray(Image.open(optical_dir / "pinhole_rgba.png"))
        assert pinhole_rgba.shape == (HEIGHT, WIDTH, 4)
        assert pinhole_rgba.dtype == np.uint8

        depth = np.load(optical_dir / "pinhole_depth_m.npy")
        assert depth.shape == (HEIGHT, WIDTH)
        assert depth.dtype == np.float32

        range_m = np.load(optical_dir / "pinhole_range_m.npy")
        assert range_m.shape == (HEIGHT, WIDTH)
        assert range_m.dtype == np.float32

        hemisphere_rgba = np.asarray(Image.open(optical_dir / "hemisphere_rgba.png"))
        assert hemisphere_rgba.shape == (FFT_ROWS, FFT_COLS, 4)

        hemisphere_range = np.load(optical_dir / "hemisphere_range_m.npy")
        assert hemisphere_range.shape == (FFT_ROWS, FFT_COLS)
        assert hemisphere_range.dtype == np.float32


def _expected_pinhole_geometry(view):
    """Analytic ground-plane hit/range/depth for one view's pinhole frame."""
    intr = PinholeIntrinsics.from_horizontal_fov(WIDTH, HEIGHT, FOV_X_DEG)
    dirs_local = pinhole_ray_directions_local(intr)
    rotation = rotation_matrix(view.orientation)
    origins, dirs_world = local_to_world_rays(dirs_local, rotation, np.asarray(view.position))

    hit = dirs_world[..., 2] < -1e-12
    range_m = np.full((HEIGHT, WIDTH), np.nan)
    range_m[hit] = origins[hit][:, 2] / -dirs_world[hit][:, 2]
    depth = range_m * dirs_local[..., 0]
    return hit, range_m, depth


def test_pinhole_alpha_and_depth_match_analytic_ground_plane(tmp_path):
    dataset_dir, views = _build_dataset(tmp_path)
    render_optical_references(
        dataset_dir,
        renderer=GroundPlaneRenderer(),
        width=WIDTH,
        height=HEIGHT,
        fov_x_deg=FOV_X_DEG,
        spp=SPP,
        seed=SEED,
    )

    for view in views:
        expected_hit, expected_range, expected_depth = _expected_pinhole_geometry(view)
        # A real scene must produce a mix of hits and misses for this check to be meaningful.
        assert 0 < expected_hit.sum() < expected_hit.size

        optical_dir = dataset_dir / "views" / view.view_id / "optical"
        pinhole_rgba = np.asarray(Image.open(optical_dir / "pinhole_rgba.png"))
        alpha = pinhole_rgba[..., 3]
        np.testing.assert_array_equal(alpha > 0, expected_hit)
        np.testing.assert_array_equal(alpha[expected_hit], 255)
        np.testing.assert_array_equal(alpha[~expected_hit], 0)
        assert np.all(pinhole_rgba[..., :3][expected_hit] > 0)

        depth = np.load(optical_dir / "pinhole_depth_m.npy")
        np.testing.assert_allclose(depth[expected_hit], expected_depth[expected_hit], atol=1e-4)
        np.testing.assert_array_equal(depth[~expected_hit], 0.0)

        range_m = np.load(optical_dir / "pinhole_range_m.npy")
        np.testing.assert_allclose(range_m[expected_hit], expected_range[expected_hit], atol=1e-4)
        assert np.all(np.isnan(range_m[~expected_hit]))


def test_hemisphere_alpha_matches_analytic_ground_plane_and_valid_mask(tmp_path):
    dataset_dir, views = _build_dataset(tmp_path)
    render_optical_references(
        dataset_dir,
        renderer=GroundPlaneRenderer(),
        width=WIDTH,
        height=HEIGHT,
        fov_x_deg=FOV_X_DEG,
        spp=SPP,
        seed=SEED,
    )

    camera_model = np.load(dataset_dir / "camera_model.npz")
    valid_mask = camera_model["valid_mask"]
    hemisphere_dirs_local = camera_model["ray_directions_local"]

    for view in views:
        rotation = rotation_matrix(view.orientation)
        position = np.asarray(view.position)
        origins, dirs_world = local_to_world_rays(hemisphere_dirs_local, rotation, position)

        expected_hit = np.zeros(valid_mask.shape, dtype=bool)
        expected_range = np.full(valid_mask.shape, np.nan)
        valid_hit = dirs_world[..., 2] < -1e-12
        expected_hit[valid_mask] = valid_hit[valid_mask]
        with np.errstate(divide="ignore", invalid="ignore"):
            valid_range = origins[..., 2] / -dirs_world[..., 2]
        expected_range[expected_hit] = valid_range[expected_hit]
        # A real scene must produce a mix of hits and misses for this check to be meaningful.
        assert 0 < expected_hit[valid_mask].sum() < int(valid_mask.sum())

        optical_dir = dataset_dir / "views" / view.view_id / "optical"
        hemisphere_range = np.load(optical_dir / "hemisphere_range_m.npy")
        np.testing.assert_allclose(
            hemisphere_range[expected_hit], expected_range[expected_hit], atol=1e-4
        )
        assert np.all(np.isnan(hemisphere_range[~valid_mask]))
        assert np.all(np.isnan(hemisphere_range[valid_mask & ~expected_hit]))

        # PNG rows are flipped (np.flipud) relative to the array-grid indexing
        # of the .npy files, so +kz sits at the top of the image.
        hemisphere_rgba_png = np.asarray(Image.open(optical_dir / "hemisphere_rgba.png"))
        hemisphere_rgba_grid = np.flipud(hemisphere_rgba_png)
        alpha_grid = hemisphere_rgba_grid[..., 3]
        np.testing.assert_array_equal(alpha_grid > 0, expected_hit)
        np.testing.assert_array_equal(alpha_grid[expected_hit], 255)
        np.testing.assert_array_equal(alpha_grid[~valid_mask], 0)
        rgb_grid = hemisphere_rgba_grid[..., :3]
        assert np.all(rgb_grid[expected_hit] > 0)


def test_transforms_json_matches_domain_camera_to_world(tmp_path):
    dataset_dir, views = _build_dataset(tmp_path)
    render_optical_references(
        dataset_dir,
        renderer=GroundPlaneRenderer(),
        width=WIDTH,
        height=HEIGHT,
        fov_x_deg=FOV_X_DEG,
        spp=SPP,
        seed=SEED,
    )

    transforms = json.loads((dataset_dir / "transforms.json").read_text())
    assert transforms["w"] == WIDTH
    assert transforms["h"] == HEIGHT
    assert len(transforms["frames"]) == len(views)

    for view, frame in zip(views, transforms["frames"]):
        rotation = rotation_matrix(view.orientation)
        expected_c2w = camera_to_world_opengl(rotation, np.asarray(view.position))
        np.testing.assert_allclose(frame["transform_matrix"], expected_c2w, atol=1e-9)

        assert frame["file_path"] == f"views/{view.view_id}/optical/pinhole_rgba.png"
        assert frame["depth_file_path"] == f"views/{view.view_id}/optical/pinhole_depth_m.npy"
        assert (dataset_dir / frame["file_path"]).is_file()
        assert (dataset_dir / frame["depth_file_path"]).is_file()


def test_manifest_gains_optical_reference_and_per_view_artifacts(tmp_path):
    dataset_dir, views = _build_dataset(tmp_path)
    render_optical_references(
        dataset_dir,
        renderer=GroundPlaneRenderer(),
        width=WIDTH,
        height=HEIGHT,
        fov_x_deg=FOV_X_DEG,
        spp=SPP,
        seed=SEED,
    )

    manifest = json.loads((dataset_dir / "dataset_manifest.json").read_text())
    assert manifest["schema_version"] == 2
    assert manifest["source_scene"] == "unused_scene.xml"  # untouched

    optical = manifest["optical_reference"]
    assert "NOT an RF training target" in optical["purpose"]
    assert optical["spp"] == SPP
    assert optical["seed"] == SEED
    assert optical["pinhole"]["width"] == WIDTH
    assert optical["pinhole"]["height"] == HEIGHT
    assert optical["pinhole"]["fov_x_deg"] == pytest.approx(FOV_X_DEG)
    assert optical["pinhole"]["transforms"] == "transforms.json"
    assert "mirror" in optical["hemisphere"]["png_orientation"].lower()

    for view in views:
        artifacts = manifest["views"][views.index(view)]["artifacts"]
        prefix = f"views/{view.view_id}/optical/"
        assert artifacts["optical_pinhole_rgba"] == f"{prefix}pinhole_rgba.png"
        assert artifacts["optical_pinhole_depth_m"] == f"{prefix}pinhole_depth_m.npy"
        assert artifacts["optical_pinhole_range_m"] == f"{prefix}pinhole_range_m.npy"
        assert artifacts["optical_hemisphere_rgba"] == f"{prefix}hemisphere_rgba.png"
        assert artifacts["optical_hemisphere_range_m"] == f"{prefix}hemisphere_range_m.npy"
        for rel_path in artifacts.values():
            assert (dataset_dir / rel_path).is_file()


def test_render_optical_references_is_idempotent(tmp_path):
    dataset_dir, _views = _build_dataset(tmp_path)

    render_optical_references(
        dataset_dir,
        renderer=GroundPlaneRenderer(),
        width=WIDTH,
        height=HEIGHT,
        fov_x_deg=FOV_X_DEG,
        spp=SPP,
        seed=SEED,
    )
    transforms_1 = (dataset_dir / "transforms.json").read_text()
    manifest_1 = (dataset_dir / "dataset_manifest.json").read_text()
    files_1 = sorted(p.relative_to(dataset_dir).as_posix() for p in dataset_dir.rglob("*"))

    render_optical_references(
        dataset_dir,
        renderer=GroundPlaneRenderer(),
        width=WIDTH,
        height=HEIGHT,
        fov_x_deg=FOV_X_DEG,
        spp=SPP,
        seed=SEED,
    )
    transforms_2 = (dataset_dir / "transforms.json").read_text()
    manifest_2 = (dataset_dir / "dataset_manifest.json").read_text()
    files_2 = sorted(p.relative_to(dataset_dir).as_posix() for p in dataset_dir.rglob("*"))

    assert transforms_1 == transforms_2
    assert manifest_1 == manifest_2
    assert files_1 == files_2
