import json
import warnings
from itertools import product
from math import atan, ceil, cos, degrees, floor, radians, sin

import numpy as np
import pytest

from plateau_rt.domain.rf_camera.calibration import rotation_matrix
from plateau_rt.domain.rf_camera.camera import look_at_orientation
from plateau_rt.domain.rf_camera.optical import (
    PinholeIntrinsics,
    camera_to_world_opengl,
    linear_to_srgb,
    local_to_world_rays,
    nerf_transforms,
    pinhole_ray_directions_local,
    project_local_points,
    to_rgba8,
    z_depth_from_range,
)


def test_intrinsics_from_horizontal_fov():
    intrinsics = PinholeIntrinsics.from_horizontal_fov(200, 100, 90.0)

    assert intrinsics.fx == pytest.approx(100.0)
    assert intrinsics.fy == pytest.approx(100.0)
    assert intrinsics.cx == pytest.approx(100.0)
    assert intrinsics.cy == pytest.approx(50.0)
    assert intrinsics.fov_x_deg == pytest.approx(90.0)
    assert intrinsics.fov_y_deg == pytest.approx(degrees(2.0 * atan(0.5)))
    assert intrinsics.fov_y_deg == pytest.approx(53.13010235, abs=1e-6)

    np.testing.assert_allclose(
        intrinsics.matrix(),
        [[100.0, 0.0, 100.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
    )
    assert intrinsics.matrix().dtype == np.float64


def test_intrinsics_from_horizontal_fov_round_trip():
    intrinsics = PinholeIntrinsics.from_horizontal_fov(64, 48, 60.0)
    assert intrinsics.fov_x_deg == pytest.approx(60.0)
    assert intrinsics.fx == pytest.approx(intrinsics.fy)


@pytest.mark.parametrize("width,height", [(0, 100), (200, 0), (-1, 100)])
def test_intrinsics_rejects_bad_size(width, height):
    with pytest.raises(ValueError):
        PinholeIntrinsics.from_horizontal_fov(width, height, 90.0)


@pytest.mark.parametrize("fov", [0.0, 180.0, -10.0, 200.0])
def test_intrinsics_rejects_bad_fov(fov):
    with pytest.raises(ValueError):
        PinholeIntrinsics.from_horizontal_fov(200, 100, fov)


def test_pinhole_intrinsics_rejects_nonpositive_focal():
    with pytest.raises(ValueError):
        PinholeIntrinsics(width=10, height=10, fx=0.0, fy=1.0, cx=5.0, cy=5.0)
    with pytest.raises(ValueError):
        PinholeIntrinsics(width=10, height=10, fx=1.0, fy=-1.0, cx=5.0, cy=5.0)


def test_pinhole_intrinsics_rejects_nonfinite():
    with pytest.raises(ValueError):
        PinholeIntrinsics(10, 10, float("nan"), 1.0, 5.0, 5.0)
    with pytest.raises(ValueError):
        PinholeIntrinsics(10, 10, 1.0, float("inf"), 5.0, 5.0)
    with pytest.raises(ValueError):
        PinholeIntrinsics(10, 10, 1.0, 1.0, float("nan"), 5.0)


def test_pinhole_ray_directions_local_shape_and_orientation():
    intrinsics = PinholeIntrinsics.from_horizontal_fov(8, 6, 90.0)
    directions = pinhole_ray_directions_local(intrinsics)

    assert directions.shape == (6, 8, 3)
    assert directions.dtype == np.float64
    np.testing.assert_allclose(np.linalg.norm(directions, axis=-1), 1.0, atol=1e-12)
    assert np.all(directions[..., 0] > 0.0)
    assert np.all(directions[:, 0, 1] > 0.0)
    assert np.all(directions[:, -1, 1] < 0.0)
    assert np.all(directions[0, :, 2] > 0.0)
    assert np.all(directions[-1, :, 2] < 0.0)


def test_pinhole_ray_directions_local_mirror_symmetry():
    intrinsics = PinholeIntrinsics.from_horizontal_fov(8, 6, 90.0)
    directions = pinhole_ray_directions_local(intrinsics)

    np.testing.assert_allclose(directions[:, ::-1, 1], -directions[:, :, 1], atol=1e-12)
    np.testing.assert_allclose(directions[:, ::-1, 0], directions[:, :, 0], atol=1e-12)
    np.testing.assert_allclose(directions[:, ::-1, 2], directions[:, :, 2], atol=1e-12)
    np.testing.assert_allclose(directions[::-1, :, 2], -directions[:, :, 2], atol=1e-12)
    np.testing.assert_allclose(directions[::-1, :, 0], directions[:, :, 0], atol=1e-12)
    np.testing.assert_allclose(directions[::-1, :, 1], directions[:, :, 1], atol=1e-12)


def test_pinhole_ray_directions_local_central_pixels_symmetric():
    intrinsics = PinholeIntrinsics.from_horizontal_fov(4, 2, 90.0)
    directions = pinhole_ray_directions_local(intrinsics)

    top_left = directions[0, 1]
    top_right = directions[0, 2]
    bottom_left = directions[1, 1]

    np.testing.assert_allclose(top_left[0], top_right[0], atol=1e-12)
    np.testing.assert_allclose(top_left[1], -top_right[1], atol=1e-12)
    np.testing.assert_allclose(top_left[2], top_right[2], atol=1e-12)
    np.testing.assert_allclose(top_left[2], -bottom_left[2], atol=1e-12)
    np.testing.assert_allclose(top_left[:2], bottom_left[:2], atol=1e-12)
    np.testing.assert_allclose(bottom_left, top_right * np.array([1.0, -1.0, -1.0]), atol=1e-12)


def test_pinhole_ray_directions_local_concrete_pixel():
    intrinsics = PinholeIntrinsics.from_horizontal_fov(4, 2, 90.0)
    directions = pinhole_ray_directions_local(intrinsics)

    right = (0 + 0.5 - intrinsics.cx) / intrinsics.fx
    up = (intrinsics.cy - (0 + 0.5)) / intrinsics.fy
    expected = np.array([1.0, -right, up])
    expected /= np.linalg.norm(expected)

    np.testing.assert_allclose(directions[0, 0], expected, atol=1e-12)


def test_project_local_points_fov_edges():
    intrinsics = PinholeIntrinsics.from_horizontal_fov(200, 100, 90.0)
    angle = radians(intrinsics.fov_x_deg / 2.0)

    left = np.array([cos(angle), sin(angle), 0.0])
    col, row = project_local_points(left, intrinsics)
    assert col == pytest.approx(0.0)
    assert row == pytest.approx(intrinsics.cy)

    right = np.array([cos(angle), -sin(angle), 0.0])
    col, _ = project_local_points(right, intrinsics)
    assert col == pytest.approx(float(intrinsics.width))


def test_project_local_points_round_trip():
    intrinsics = PinholeIntrinsics.from_horizontal_fov(7, 5, 70.0)
    directions = pinhole_ray_directions_local(intrinsics)
    col, row = project_local_points(directions, intrinsics)

    cols, rows = np.meshgrid(np.arange(7), np.arange(5))
    np.testing.assert_allclose(col, cols + 0.5, atol=1e-12)
    np.testing.assert_allclose(row, rows + 0.5, atol=1e-12)


def test_project_local_points_scale_invariant():
    intrinsics = PinholeIntrinsics.from_horizontal_fov(7, 5, 70.0)
    directions = pinhole_ray_directions_local(intrinsics)
    scales = np.linspace(0.5, 12.0, directions.shape[0] * directions.shape[1]).reshape(
        directions.shape[:2]
    )

    col, row = project_local_points(directions * scales[..., None], intrinsics)
    cols, rows = np.meshgrid(np.arange(7), np.arange(5))
    np.testing.assert_allclose(col, cols + 0.5, atol=1e-12)
    np.testing.assert_allclose(row, rows + 0.5, atol=1e-12)


def test_project_local_points_nonpositive_depth_is_nan_without_warnings():
    intrinsics = PinholeIntrinsics.from_horizontal_fov(4, 4, 90.0)
    points = np.array([[0.0, 1.0, 1.0], [-1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        col, row = project_local_points(points, intrinsics)

    assert np.isnan(col[0]) and np.isnan(row[0])
    assert np.isnan(col[1]) and np.isnan(row[1])
    assert np.isfinite(col[2]) and np.isfinite(row[2])


def test_project_local_points_rejects_bad_last_axis():
    intrinsics = PinholeIntrinsics.from_horizontal_fov(4, 4, 90.0)
    with pytest.raises(ValueError):
        project_local_points(np.zeros((4, 4, 2)), intrinsics)


def test_camera_to_world_opengl_identity_axes():
    matrix = camera_to_world_opengl(np.eye(3), (1.0, 2.0, 3.0))

    assert matrix.shape == (4, 4)
    assert matrix.dtype == np.float64
    np.testing.assert_allclose(matrix @ np.array([0.0, 0.0, -1.0, 0.0]), [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(matrix @ np.array([0.0, 1.0, 0.0, 0.0]), [0.0, 0.0, 1.0, 0.0])
    np.testing.assert_allclose(matrix @ np.array([1.0, 0.0, 0.0, 0.0]), [0.0, -1.0, 0.0, 0.0])
    np.testing.assert_allclose(matrix[:, 3], [1.0, 2.0, 3.0, 1.0])
    np.testing.assert_allclose(matrix[3, :], [0.0, 0.0, 0.0, 1.0])

    assert np.linalg.det(matrix[:3, :3]) == pytest.approx(1.0)
    np.testing.assert_allclose(matrix[:3, :3] @ matrix[:3, :3].T, np.eye(3), atol=1e-12)


@pytest.mark.parametrize(
    "position,target",
    [((35.0, 5.0, 1.5), (5.0, 5.0, 5.0)), ((1.0, 2.0, 3.0), (-4.0, 0.5, 2.0))],
)
def test_camera_to_world_opengl_forward_axis_matches_look_at(position, target):
    rotation = rotation_matrix(look_at_orientation(position, target))
    matrix = camera_to_world_opengl(rotation, position)

    forward = -matrix[:3, 2]
    expected = np.asarray(target) - np.asarray(position)
    expected /= np.linalg.norm(expected)

    np.testing.assert_allclose(forward, expected, atol=1e-9)
    assert np.linalg.det(matrix[:3, :3]) == pytest.approx(1.0)


def test_local_to_world_rays_center_pixel_looks_at_target():
    position = (35.0, 5.0, 1.5)
    target = (5.0, 5.0, 5.0)
    rotation = rotation_matrix(look_at_orientation(position, target))
    intrinsics = PinholeIntrinsics.from_horizontal_fov(5, 3, 80.0)
    directions = pinhole_ray_directions_local(intrinsics)

    origins, world = local_to_world_rays(directions, rotation, np.asarray(position))

    assert origins.shape == (3, 5, 3)
    assert world.shape == (3, 5, 3)
    assert origins.flags.writeable
    np.testing.assert_allclose(
        origins, np.broadcast_to(np.asarray(position), (3, 5, 3)), atol=1e-12
    )
    np.testing.assert_allclose(np.linalg.norm(world, axis=-1), 1.0, atol=1e-12)

    expected = np.asarray(target) - np.asarray(position)
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(world[1, 2], expected, atol=1e-12)


def test_mock_box_silhouette_matches_mitsuba_measurement():
    # Bounds measured with Sionna/Mitsuba for the mock axis-aligned box
    # (x, y in [-5, 5], z in [0, 10]) seen from p = (35, 5, 1.5) looking at
    # t = (5, 5, 5) with W = 200, H = 100 and a 90 degree horizontal FOV.
    position = (35.0, 5.0, 1.5)
    target = (5.0, 5.0, 5.0)
    rotation = rotation_matrix(look_at_orientation(position, target))
    intrinsics = PinholeIntrinsics.from_horizontal_fov(200, 100, 90.0)

    corners = np.array(list(product([-5.0, 5.0], [-5.0, 5.0], [0.0, 10.0])))
    local = (corners - np.asarray(position)) @ rotation
    col, row = project_local_points(local, intrinsics)

    assert np.all(np.isfinite(col))
    assert np.all(np.isfinite(row))
    assert floor(col.min()) == 66
    assert ceil(col.max() - 1e-9) - 1 == 99
    assert floor(row.min()) == 33
    assert ceil(row.max() - 1e-9) - 1 == 66


def test_local_to_world_rays_rejects_bad_shapes():
    rotation = np.eye(3)
    position = (0.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        local_to_world_rays(np.zeros((2, 2)), rotation, position)
    with pytest.raises(ValueError):
        local_to_world_rays(np.zeros((2, 3)), np.eye(4), position)
    with pytest.raises(ValueError):
        local_to_world_rays(np.zeros((2, 3)), rotation, (0.0, 0.0))


def test_z_depth_from_range_optical_axis_projection():
    angle = radians(30.0)
    direction = np.array([cos(angle), sin(angle), 0.0])

    assert z_depth_from_range(2.0, direction) == pytest.approx(2.0 * cos(angle))
    assert np.isnan(z_depth_from_range(np.nan, direction))


def test_z_depth_from_range_broadcasts_over_image():
    intrinsics = PinholeIntrinsics.from_horizontal_fov(4, 3, 60.0)
    directions = pinhole_ray_directions_local(intrinsics)

    depth = z_depth_from_range(np.full((3, 4), 2.0), directions)

    assert depth.shape == (3, 4)
    np.testing.assert_allclose(depth, 2.0 * directions[..., 0], atol=1e-12)


def test_z_depth_from_range_rejects_bad_last_axis():
    with pytest.raises(ValueError):
        z_depth_from_range(np.ones((2, 2)), np.zeros((2, 2)))


def test_linear_to_srgb_transfer_function():
    assert linear_to_srgb(np.array(0.0)) == pytest.approx(0.0)
    assert linear_to_srgb(np.array(1.0)) == pytest.approx(1.055 - 0.055)
    assert linear_to_srgb(np.array(0.5)) == pytest.approx(0.735357, abs=1e-6)
    assert linear_to_srgb(np.array(0.002)) == pytest.approx(0.02584, abs=1e-9)


def test_linear_to_srgb_clips_and_is_monotonic():
    assert linear_to_srgb(np.array(-1.0)) == pytest.approx(0.0)
    assert linear_to_srgb(np.array(2.0)) == pytest.approx(1.055 - 0.055)

    values = np.linspace(0.0, 1.0, 256)
    encoded = linear_to_srgb(values)
    assert encoded.dtype == np.float64
    assert np.all(np.diff(encoded) >= 0.0)


def test_to_rgba8_full_and_zero():
    full = to_rgba8(np.ones((2, 3, 3)), np.ones((2, 3)))
    assert full.dtype == np.uint8
    assert full.shape == (2, 3, 4)
    assert np.all(full == 255)

    zero = to_rgba8(np.zeros((2, 2, 3)), np.zeros((2, 2)))
    assert np.all(zero == 0)


def test_to_rgba8_mid_values_and_nan():
    mid = to_rgba8(np.full((1, 1, 3), 0.5), np.full((1, 1), 0.5))
    assert mid[0, 0, 0] == 188
    assert mid[0, 0, 3] == 128

    nan = to_rgba8(np.full((1, 3), np.nan), np.array([np.nan]))
    assert np.all(nan == 0)


def test_to_rgba8_clips_out_of_range():
    clipped = to_rgba8(np.array([[-1.0, 2.0, 0.5]]), np.array([2.0]))
    assert clipped[0, 0] == 0
    assert clipped[0, 1] == 255
    assert clipped[0, 3] == 255


def test_to_rgba8_rejects_bad_shapes():
    with pytest.raises(ValueError):
        to_rgba8(np.zeros((2, 2)), np.zeros((2,)))
    with pytest.raises(ValueError):
        to_rgba8(np.zeros((2, 3, 3)), np.zeros((2, 2)))


def test_nerf_transforms_structure_and_frames():
    intrinsics = PinholeIntrinsics.from_horizontal_fov(64, 48, 60.0)
    first = camera_to_world_opengl(rotation_matrix((0.1, 0.2, 0.3)), (1.0, 2.0, 3.0))
    second = camera_to_world_opengl(rotation_matrix((-0.4, 0.1, 0.2)), (4.0, 5.0, 6.0))
    frames = [
        {"file_path": "a.png", "camera_to_world": first, "depth_file_path": "a_depth.png"},
        {"file_path": "b.png", "camera_to_world": second},
    ]

    result = nerf_transforms(intrinsics, frames)

    assert result["camera_model"] == "OPENCV"
    assert isinstance(result["w"], int) and result["w"] == 64
    assert isinstance(result["h"], int) and result["h"] == 48
    assert result["fl_x"] == pytest.approx(intrinsics.fx)
    assert result["fl_y"] == pytest.approx(intrinsics.fy)
    assert result["cx"] == pytest.approx(intrinsics.cx)
    assert result["cy"] == pytest.approx(intrinsics.cy)
    assert result["camera_angle_x"] == pytest.approx(radians(60.0))
    assert result["k1"] == 0.0
    assert result["k2"] == 0.0
    assert result["p1"] == 0.0
    assert result["p2"] == 0.0

    assert [frame["file_path"] for frame in result["frames"]] == ["a.png", "b.png"]
    np.testing.assert_allclose(result["frames"][0]["transform_matrix"], first)
    np.testing.assert_allclose(result["frames"][1]["transform_matrix"], second)
    assert isinstance(result["frames"][0]["transform_matrix"][0][0], float)
    assert result["frames"][0]["depth_file_path"] == "a_depth.png"
    assert "depth_file_path" not in result["frames"][1]

    assert json.loads(json.dumps(result)) == result


def test_nerf_transforms_rejects_bad_matrix():
    intrinsics = PinholeIntrinsics.from_horizontal_fov(32, 32, 90.0)
    frames = [{"file_path": "a.png", "camera_to_world": np.eye(3)}]
    with pytest.raises(ValueError):
        nerf_transforms(intrinsics, frames)
