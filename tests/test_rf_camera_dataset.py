import numpy as np

from plateau_rt.domain.rf_camera.calibration import rotation_matrix
from plateau_rt.domain.rf_camera.camera import (
    build_direction_cosine_camera_model,
    generate_ring_views,
    look_at_orientation,
    solid_angle_weight,
    to_solid_angle_amplitude,
)


def test_look_at_orientation_points_local_x_at_target():
    position = (10.0, 0.0, 2.0)
    target = (0.0, 0.0, 2.0)
    orientation = look_at_orientation(position, target)
    rotation = rotation_matrix(orientation)

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
        rotation = rotation_matrix(view.orientation)
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


def test_solid_angle_weight_is_normal_component_on_the_disk():
    ky = np.array([-1.0, -0.6, 0.0, 0.6, 1.0])
    kz = np.array([-0.8, 0.0, 0.8])

    weight = solid_angle_weight(ky, kz)

    assert weight.shape == (3, 5)
    assert weight[1, 2] == 1.0  # boresight
    np.testing.assert_allclose(weight[1, 3], 0.8)  # sqrt(1 - 0.6^2)
    np.testing.assert_allclose(weight[0, 3], 0.0, atol=1e-12)  # on the rim
    assert weight[0, 0] == 0.0  # outside the propagating disk


def test_solid_angle_amplitude_scales_every_frequency_by_kx():
    ky = np.array([0.0, 0.6])
    kz = np.array([0.0])
    spectrum = np.ones((1, 2, 3), dtype=np.complex64) * (1.0 + 1.0j)

    amplitude = to_solid_angle_amplitude(spectrum, ky, kz)

    np.testing.assert_allclose(amplitude[0, 0], 1.0 + 1.0j)
    np.testing.assert_allclose(amplitude[0, 1], 0.8 * (1.0 + 1.0j))


def test_camera_model_solid_angle_weight_matches_ray_x_component():
    model = build_direction_cosine_camera_model(
        fft_rows=32, fft_cols=32, horizontal_spacing_lambda=0.5, vertical_spacing_lambda=0.5
    )

    np.testing.assert_allclose(
        model["solid_angle_weight"], model["ray_directions_local"][..., 0], atol=1e-7
    )
    assert np.all(model["solid_angle_weight"][~model["valid_mask"]] == 0.0)
