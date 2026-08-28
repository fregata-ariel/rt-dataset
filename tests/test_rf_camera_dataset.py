import numpy as np

from plateau_rt.adapters.sionna.rf_camera_calibration import rotation_matrix_numpy
from plateau_rt.adapters.sionna.rf_camera_dataset import (
    build_direction_cosine_camera_model,
    generate_ring_views,
    look_at_orientation,
)


def test_look_at_orientation_points_local_x_at_target():
    position = (10.0, 0.0, 2.0)
    target = (0.0, 0.0, 2.0)
    orientation = look_at_orientation(position, target)
    rotation = rotation_matrix_numpy(orientation)

    expected = np.array([-1.0, 0.0, 0.0])
    np.testing.assert_allclose(rotation[:, 0], expected, atol=1e-12)


def test_ring_views_are_deterministic_and_look_at_target():
    target = (5.0, 5.0, 5.0)
    views = generate_ring_views(
        target=target,
        radius_m=30.0,
        ue_height_m=1.5,
        num_views=8,
    )

    assert [view.view_id for view in views] == [f"ue_{i:06d}" for i in range(8)]

    target_np = np.asarray(target)
    for view in views:
        position = np.asarray(view.position)
        horizontal_radius = np.linalg.norm(position[:2] - target_np[:2])
        np.testing.assert_allclose(horizontal_radius, 30.0, atol=1e-12)

        expected_forward = target_np - position
        expected_forward /= np.linalg.norm(expected_forward)
        rotation = rotation_matrix_numpy(view.orientation)
        np.testing.assert_allclose(rotation[:, 0], expected_forward, atol=1e-12)


def test_direction_cosine_camera_model_is_unit_front_hemisphere():
    model = build_direction_cosine_camera_model(
        fft_rows=128,
        fft_cols=128,
        horizontal_spacing_lambda=0.5,
        vertical_spacing_lambda=0.5,
    )

    rays = model["ray_directions_local"]
    valid = model["valid_mask"]
    ky = model["ky_over_k"]
    kz = model["kz_over_k"]

    assert rays.shape == (128, 128, 3)
    assert valid.shape == (128, 128)
    assert np.all(rays[valid, 0] >= 0.0)
    np.testing.assert_allclose(np.linalg.norm(rays[valid], axis=-1), 1.0, atol=1e-6)

    row = int(np.argmin(np.abs(kz)))
    col = int(np.argmin(np.abs(ky)))
    np.testing.assert_allclose(rays[row, col], [1.0, 0.0, 0.0], atol=1e-6)
