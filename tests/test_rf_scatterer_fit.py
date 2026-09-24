import numpy as np
import pytest

from plateau_rt.domain.rf_camera.calibration import rotation_matrix
from plateau_rt.domain.rf_camera.camera import generate_ring_views
from plateau_rt.domain.rf_camera.delay import SPEED_OF_LIGHT_M_S
from plateau_rt.experimental.rf_scatterer_fit import (
    ApertureSpec,
    direct_path_cfr,
    element_positions_world,
    frequencies_hz,
    reconstruct,
    relative_error,
    scatterer_cfr,
    scatterer_channels,
    system_matrix,
    tikhonov_path,
    voxel_grid,
)


def _phase_slope_delay(frequencies: np.ndarray, cfr: np.ndarray) -> float:
    phase = np.unwrap(np.angle(cfr))
    slope = float(np.polyfit(frequencies, phase, 1)[0])
    return -slope / (2.0 * np.pi)


def test_scatterer_channels_are_reciprocal():
    transmitter = np.array([-50.0, -50.0, 30.0])
    receivers = np.array([[1.0, 2.0, 3.0], [4.0, -5.0, 6.0]])
    points = np.array([[0.0, 0.0, 5.0], [2.0, -1.0, 4.0]])
    frequencies = frequencies_hz(ApertureSpec(rows=4, cols=4))
    wavelength = SPEED_OF_LIGHT_M_S / 3.5e9

    forward = scatterer_channels(points, transmitter, receivers, frequencies, wavelength)
    swapped = scatterer_channels(
        points, receivers[0], transmitter[None, :], frequencies, wavelength
    )

    np.testing.assert_allclose(forward[0], swapped[0], rtol=1e-12, atol=0.0)


def test_scatterer_delay_matches_geometry():
    # Many narrow bins keep the per-bin phase step below pi, so unwrapping is honest.
    spec = ApertureSpec(rows=1, cols=1, bandwidth_hz=100e6, num_freq=512)
    bs = np.array([-50.0, -50.0, 30.0])
    ue_position = (0.0, 0.0, 0.0)
    ue_orientation = (0.0, 0.0, 0.0)
    point = np.array([[1.0, 2.0, 3.0]])

    cfr = scatterer_cfr(spec, bs, ue_position, ue_orientation, point, np.array([1.0 + 0.0j]))
    frequencies = frequencies_hz(spec)
    expected = (
        np.linalg.norm(point[0] - bs) + np.linalg.norm(np.asarray(ue_position) - point[0])
    ) / SPEED_OF_LIGHT_M_S

    assert _phase_slope_delay(frequencies, cfr[0, 0]) == pytest.approx(expected, abs=1e-12)


def test_direct_path_delay_matches_range_over_c():
    spec = ApertureSpec(rows=1, cols=1, bandwidth_hz=100e6, num_freq=512)
    bs = np.array([-50.0, -50.0, 30.0])
    ue_position = (0.0, 0.0, 0.0)

    cfr = direct_path_cfr(spec, bs, ue_position, (0.0, 0.0, 0.0))
    frequencies = frequencies_hz(spec)
    expected = np.linalg.norm(bs - np.asarray(ue_position)) / SPEED_OF_LIGHT_M_S

    assert _phase_slope_delay(frequencies, cfr[0, 0]) == pytest.approx(expected, abs=1e-12)


def _plane_wave_phase_residual(bs: np.ndarray, orientation: tuple[float, float, float]) -> float:
    spec = ApertureSpec(rows=8, cols=8)
    ue_position = np.zeros(3)
    elements = element_positions_world(spec, ue_position, orientation)
    frequencies = frequencies_hz(spec)

    direction = bs - ue_position
    centre_distance = float(np.linalg.norm(direction))
    direction = direction / centre_distance
    exact = direct_path_cfr(spec, bs, ue_position, orientation)
    centre = (
        (SPEED_OF_LIGHT_M_S / spec.carrier_hz)
        / (4.0 * np.pi * centre_distance)
        * np.exp(-1j * 2.0 * np.pi * frequencies * centre_distance / SPEED_OF_LIGHT_M_S)
    )
    ratio = exact / centre[None, None, :]

    rotation = rotation_matrix(orientation)
    local_offsets = (elements - ue_position).reshape(-1, 3) @ rotation
    direction_local = rotation.T @ direction
    projection = local_offsets @ direction_local
    plane_wave = np.exp(1j * 2.0 * np.pi * np.outer(projection, frequencies) / SPEED_OF_LIGHT_M_S)
    residual = np.angle(ratio.reshape(-1, frequencies.size) / plane_wave)
    return float(np.max(np.abs(residual)))


def test_far_field_limit_and_sensitivity():
    direction = np.array([0.8, 0.4, 0.4472135955])
    direction = direction / np.linalg.norm(direction)
    orientation = (0.3, -0.2, 0.1)

    far_error = _plane_wave_phase_residual(1e5 * direction, orientation)
    near_error = _plane_wave_phase_residual(50.0 * direction, orientation)

    assert far_error < 1e-3
    assert near_error > 10.0 * far_error
    assert near_error > 1e-3


def test_aperture_layout_top_right_corner():
    spec = ApertureSpec(rows=4, cols=4)
    positions = element_positions_world(spec, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    spacing = spec.spacing_lambda * SPEED_OF_LIGHT_M_S / spec.carrier_hz

    assert positions[0, 0, 1] == pytest.approx(-(spec.cols - 1) * spacing / 2.0)
    assert positions[0, 0, 2] == pytest.approx(+(spec.rows - 1) * spacing / 2.0)


def test_plane_wave_element_phase_is_frequency_independent():
    spec = ApertureSpec(rows=3, cols=5, bandwidth_hz=400e6, num_freq=8)
    bs = (-40.0, 15.0, 25.0)
    ue_position = (2.0, -1.0, 1.0)
    ue_orientation = (0.7, -0.3, 0.2)

    cfr = direct_path_cfr(spec, bs, ue_position, ue_orientation, plane_wave=True)
    ratios = cfr / cfr[0, 0, :][None, None, :]
    np.testing.assert_allclose(
        ratios, np.broadcast_to(ratios[..., :1], ratios.shape), rtol=1e-12, atol=0.0
    )

    elements = element_positions_world(spec, ue_position, ue_orientation)
    rotation = rotation_matrix(ue_orientation)
    local_offsets = (elements - np.asarray(ue_position)).reshape(-1, 3) @ rotation
    direction_world = np.asarray(bs, dtype=np.float64) - np.asarray(ue_position)
    direction_local = rotation.T @ (direction_world / np.linalg.norm(direction_world))
    projection = (local_offsets @ direction_local).reshape(spec.rows, spec.cols)
    carrier_wavenumber = 2.0 * np.pi * spec.carrier_hz / SPEED_OF_LIGHT_M_S
    expected = np.broadcast_to(
        np.exp(1j * carrier_wavenumber * (projection - projection[0, 0]))[..., None],
        ratios.shape,
    )

    np.testing.assert_allclose(ratios, expected, rtol=1e-12, atol=0.0)


def test_direct_path_matches_plane_wave_at_large_range():
    spec = ApertureSpec(rows=8, cols=8)
    bs = (1e5, 2e4, 3e4)
    ue_position = (0.0, 0.0, 0.0)
    ue_orientation = (0.3, -0.2, 0.1)

    exact = direct_path_cfr(spec, bs, ue_position, ue_orientation)
    plane_wave = direct_path_cfr(spec, bs, ue_position, ue_orientation, plane_wave=True)
    frequencies = frequencies_hz(spec)
    carrier_index = frequencies.size // 2

    assert frequencies[carrier_index] == pytest.approx(spec.carrier_hz)
    np.testing.assert_allclose(
        exact[..., carrier_index], plane_wave[..., carrier_index], rtol=1e-3, atol=0.0
    )

    elements = element_positions_world(spec, ue_position, ue_orientation)
    max_offset = float(np.max(np.linalg.norm(elements - np.asarray(ue_position), axis=-1)))
    max_offset_hz = float(np.max(np.abs(frequencies - spec.carrier_hz)))
    bound = 2.0 * np.pi * max_offset_hz * max_offset / SPEED_OF_LIGHT_M_S
    phase = np.angle((exact / plane_wave).reshape(-1, frequencies.size))
    assert np.max(np.abs(phase)) <= bound + 1e-9


def test_exact_recovery_well_posed():
    spec = ApertureSpec(rows=8, cols=8, bandwidth_hz=100e6, num_freq=16)
    voxels = voxel_grid((-2.0, -2.0, 0.0), (2.0, 2.0, 2.0), 2.0)
    views = generate_ring_views(target=(0.0, 0.0, 1.0), radius_m=30.0, ue_height_m=1.5, num_views=8)
    A = system_matrix(spec, (-50.0, -50.0, 30.0), views, voxels)

    assert np.linalg.matrix_rank(A) == voxels.shape[0]

    rng = np.random.default_rng(0)
    truth = rng.standard_normal(voxels.shape[0]) + 1j * rng.standard_normal(voxels.shape[0])
    observations = A @ truth

    for method in ("lsqr", "normal"):
        estimate = reconstruct(A, observations, damp=1e-9, method=method)
        assert relative_error(estimate, truth) < 1e-6


def test_tikhonov_path_matches_reconstruct():
    rng = np.random.default_rng(11)
    A = rng.standard_normal((40, 12)) + 1j * rng.standard_normal((40, 12))
    y = rng.standard_normal(40) + 1j * rng.standard_normal(40)
    damps = np.array([1e-3, 1e-1, 1.0, 10.0])

    path = tikhonov_path(A, y, damps)
    assert path.shape == (damps.size, 12)
    for index, damp in enumerate(damps):
        expected = reconstruct(A, y, damp=damp, method="normal")
        np.testing.assert_allclose(path[index], expected, rtol=1e-9, atol=1e-12)


def test_system_matrix_matches_scatterer_cfr():
    spec = ApertureSpec(rows=3, cols=5, bandwidth_hz=100e6, num_freq=4)
    bs = (-50.0, -50.0, 30.0)
    views = generate_ring_views(target=(0.0, 0.0, 1.0), radius_m=30.0, ue_height_m=1.5, num_views=3)
    rng = np.random.default_rng(7)
    points = rng.uniform(-4.0, 4.0, size=(5, 3))
    rho = rng.standard_normal(5) + 1j * rng.standard_normal(5)

    matrix = system_matrix(spec, bs, views, points)
    assert matrix.shape == (3 * 3 * 5 * 4, 5)

    expected = np.concatenate(
        [
            scatterer_cfr(spec, bs, view.position, view.orientation, points, rho).reshape(-1)
            for view in views
        ]
    )
    np.testing.assert_allclose(matrix @ rho, expected, rtol=1e-12, atol=0.0)


def test_off_grid_scatterer_recovers_nearest_voxel():
    spec = ApertureSpec(rows=8, cols=8, bandwidth_hz=100e6, num_freq=16)
    bs = (-50.0, -50.0, 30.0)
    views = generate_ring_views(target=(0.0, 0.0, 5.0), radius_m=30.0, ue_height_m=1.5, num_views=8)
    voxels = voxel_grid((-5.0, -5.0, 0.0), (5.0, 5.0, 10.0), 2.0)
    offsets = np.array([0.0, 0.0, 0.4])
    points = np.array([[-1.0, 1.0, 4.0], [3.0, -3.0, 6.0]]) + offsets
    rho = np.array([1.0 + 0.0j, 0.9 * np.exp(0.6j)])

    observations = np.concatenate(
        [
            scatterer_cfr(spec, bs, view.position, view.orientation, points, rho).reshape(-1)
            for view in views
        ]
    )
    matrix = system_matrix(spec, bs, views, voxels)
    s_max = float(np.linalg.svd(np.asarray(matrix), compute_uv=False)[0])
    estimate = reconstruct(matrix, observations, damp=1e-2 * s_max, method="normal")

    for point in points:
        nearest = int(np.argmin(np.linalg.norm(voxels - point, axis=1)))
        top_two = set(np.argsort(-np.abs(estimate))[:2].tolist())
        assert nearest in top_two


def test_single_view_narrowband_is_worse_identified():
    voxels = voxel_grid((-4.0, -4.0, 0.0), (4.0, 4.0, 4.0), 2.0)
    bs = (-50.0, -50.0, 30.0)
    well_spec = ApertureSpec(rows=8, cols=8, bandwidth_hz=100e6, num_freq=16)
    ill_spec = ApertureSpec(rows=4, cols=4, bandwidth_hz=20e6, num_freq=16)
    well_views = generate_ring_views(
        target=(0.0, 0.0, 2.0), radius_m=30.0, ue_height_m=1.5, num_views=8
    )
    ill_views = generate_ring_views(
        target=(0.0, 0.0, 2.0), radius_m=30.0, ue_height_m=1.5, num_views=1
    )

    rng = np.random.default_rng(1)
    chosen = rng.choice(voxels.shape[0], size=4, replace=False)
    truth = np.zeros(voxels.shape[0], dtype=np.complex128)
    truth[chosen] = rng.standard_normal(chosen.size) + 1j * rng.standard_normal(chosen.size)
    points = voxels[chosen]
    rho = truth[chosen]

    well_error = _noisy_error(well_spec, bs, well_views, voxels, points, rho, truth, rng)
    ill_error = _noisy_error(ill_spec, bs, ill_views, voxels, points, rho, truth, rng)

    assert ill_error > 3.0 * well_error


def _noisy_error(spec, bs, views, voxels, points, rho, truth, rng) -> float:
    observations = np.concatenate(
        [
            scatterer_cfr(spec, bs, view.position, view.orientation, points, rho).reshape(-1)
            for view in views
        ]
    )
    power = float(np.mean(np.abs(observations) ** 2))
    sigma = np.sqrt(power / 100.0)
    noise = (
        sigma
        / np.sqrt(2.0)
        * (rng.standard_normal(observations.shape) + 1j * rng.standard_normal(observations.shape))
    )
    matrix = system_matrix(spec, bs, views, voxels)
    damp = 1e-2 * float(np.linalg.norm(matrix))
    estimate = reconstruct(matrix, observations + noise, damp=damp, method="normal")
    return relative_error(estimate, truth)
