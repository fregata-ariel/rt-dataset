from pathlib import Path

import numpy as np
import pytest
from scipy.constants import c as SPEED_OF_LIGHT

from plateau_rt.domain.rf_camera.camera import look_at_orientation
from plateau_rt.domain.rf_camera.imaging import reshape_planar_column_first
from plateau_rt.domain.rf_tomography.geometry import (
    CaptureGeometry,
    VoxelGrid,
    hemisphere_index,
    mirror_point,
    planar_element_offsets,
    rotations_from_orientations,
    single_bounce_point,
)
from plateau_rt.domain.rf_tomography.views import nested_view_order

FIXTURE = Path(__file__).parent / "fixtures" / "rf_tomography" / "sionna_los_aperture.npz"


def _minimal_geometry(**overrides: object) -> CaptureGeometry:
    kwargs: dict[str, object] = {
        "ue_pos": np.zeros((1, 3)),
        "ue_rot": np.eye(3)[None, :, :],
        "bs_pos": np.zeros((1, 3)),
        "elem_offsets": np.zeros((1, 3)),
        "freq_offsets": np.array([-1e6, 0.0, 1e6]),
        "f_c": 3.5e9,
        "aperture_shape": (1, 1),
    }
    kwargs.update(overrides)
    return CaptureGeometry(**kwargs)  # type: ignore[arg-type]


def test_voxel_grid_flat_index_is_c_order() -> None:
    grid = VoxelGrid(origin=(1.0, -2.0, 0.5), spacing=2.0, shape=(5, 5, 3))
    assert grid.size == 75
    assert grid.shape == (5, 5, 3)
    assert grid.origin.dtype == np.float64
    assert not grid.origin.flags.writeable
    np.testing.assert_array_equal(grid.centers()[21], (1.0 + 2.0, -2.0 + 4.0, 0.5 + 0.0))
    np.testing.assert_array_equal(grid.index([[3.0, 2.0, 0.5]]), [21])


def test_voxel_grid_index_roundtrip_and_outside() -> None:
    grid = VoxelGrid(origin=(1.0, -2.0, 0.5), spacing=2.0, shape=(5, 5, 3))
    centers = grid.centers()
    np.testing.assert_array_equal(grid.index(centers), np.arange(grid.size))

    rng = np.random.default_rng(0)
    offsets = rng.uniform(-0.49, 0.49, size=(grid.size, 3)) * grid.spacing
    np.testing.assert_array_equal(grid.index(centers + offsets), np.arange(grid.size))

    for axis in range(3):
        lower = centers[0].copy()
        lower[axis] -= 0.5 * grid.spacing + 1e-6
        assert grid.index(lower)[0] == -1
        upper = centers[-1].copy()
        upper[axis] += 0.5 * grid.spacing + 1e-6
        assert grid.index(upper)[0] == -1
    assert grid.index([[np.nan, 0.0, 0.0]])[0] == -1
    with pytest.raises(ValueError):
        grid.index([1.0, 2.0])


def test_voxel_grid_from_bounds_matches_design_grid() -> None:
    physical = VoxelGrid.from_bounds((-50.0, -50.0, -2.0), (50.0, 50.0, 40.0), 0.5)
    assert physical.shape == (201, 201, 85)
    assert physical.size == 3_434_085

    micro = VoxelGrid.from_bounds((0.0, 0.0, 0.0), (8.0, 8.0, 4.0), 2.0)
    assert micro.shape == (5, 5, 3)
    np.testing.assert_array_equal(micro.centers()[-1], (8.0, 8.0, 4.0))

    with pytest.raises(ValueError):
        VoxelGrid.from_bounds((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 0.0)
    with pytest.raises(ValueError):
        VoxelGrid(origin=(0.0, 0.0, 0.0), spacing=1.0, shape=(2, 0, 2))
    with pytest.raises(ValueError):
        VoxelGrid.from_bounds((0.0, 0.0, 0.0), (1.0, 1.0, -1.0), 1.0)


def test_element_offsets_match_sionna_planar_array() -> None:
    fixture = np.load(FIXTURE)
    wavelength = SPEED_OF_LIGHT / float(fixture["f_c"])
    spacing = 0.5 * wavelength

    sionna_offsets = reshape_planar_column_first(
        fixture["sionna_positions"], rows=8, cols=8
    ).reshape(64, 3)
    offsets = planar_element_offsets(wavelength)
    np.testing.assert_allclose(offsets, sionna_offsets, atol=1e-7)

    np.testing.assert_allclose(offsets[0], (0.0, -3.5 * spacing, 3.5 * spacing), atol=1e-15)
    np.testing.assert_allclose(offsets[1], (0.0, -2.5 * spacing, 3.5 * spacing), atol=1e-15)


def test_geometry_reproduces_sionna_los_aperture() -> None:
    fixture = np.load(FIXTURE)
    geom = CaptureGeometry.from_orientations(
        fixture["ue_pos"],
        fixture["ue_orientation"],
        fixture["bs_pos"][None, :],
        f_c=float(fixture["f_c"]),
        bandwidth=100e6,
        num_bins=8,
    )
    np.testing.assert_array_equal(geom.freq_offsets, fixture["freq_offsets"])

    aperture = fixture["aperture_cfr"]
    hemispheres = []
    for view in range(2):
        source = geom.bs_pos[0]
        direction = geom.local_direction(source, view)[0]
        hemisphere = int(hemisphere_index(direction))
        hemispheres.append(hemisphere)
        tau = geom.vs_delay(source, view)[0]
        model = np.exp(1j * geom.wavenumber * (geom.elem_offsets @ direction))[:, None] * np.exp(
            -2j * np.pi * geom.freq_offsets[None, :] * tau
        )
        response = aperture[view]
        np.testing.assert_array_equal(
            response[1 - hemisphere], np.zeros_like(response[1 - hemisphere])
        )
        ratio = response[hemisphere].reshape(64, 8) / model
        assert np.max(np.abs(ratio / ratio.mean() - 1.0)) < 1e-4
        carrier = ratio.mean() * np.exp(1j * geom.wavenumber * SPEED_OF_LIGHT * tau)
        assert abs(np.angle(carrier)) < 1e-3

    assert set(hemispheres) == {0, 1}


def test_local_direction_hand_cases() -> None:
    rotation = rotations_from_orientations([[0.5 * np.pi, 0.0, 0.0]])
    np.testing.assert_allclose(
        rotation[0], [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], atol=1e-15
    )
    geom = _minimal_geometry(ue_pos=[[1.0, 2.0, 3.0]], ue_rot=rotation)

    points = np.array([[1.0, 7.0, 3.0], [-4.0, 2.0, 3.0], [1.0, 2.0, 10.0], [1.0, -1.0, 3.0]])
    local = geom.local_direction(points, 0)
    expected = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])
    np.testing.assert_allclose(local, expected, atol=1e-12)
    np.testing.assert_array_equal(hemisphere_index(expected), [0, 0, 0, 1])

    target = (7.0, -4.0, 8.0)
    orientation = look_at_orientation((1.0, 2.0, 3.0), target)
    camera = _minimal_geometry(
        ue_pos=[[1.0, 2.0, 3.0]], ue_rot=rotations_from_orientations([orientation])
    )
    target_local = camera.local_direction([target], 0)
    np.testing.assert_allclose(target_local, [[1.0, 0.0, 0.0]], atol=1e-12)

    assert np.all(np.isnan(geom.local_direction([[1.0, 2.0, 3.0]], 0)))


def test_delay_symmetry() -> None:
    rng = np.random.default_rng(2024)
    tx = rng.uniform(-100.0, 100.0, 3)
    ue = rng.uniform(-100.0, 100.0, 3)
    points = rng.uniform(-100.0, 100.0, (50, 3))
    identity = np.eye(3)[None, :, :]
    geometry_a = _minimal_geometry(ue_pos=[ue], bs_pos=[tx], ue_rot=identity)
    geometry_b = _minimal_geometry(ue_pos=[tx], bs_pos=[ue], ue_rot=identity)
    np.testing.assert_allclose(
        geometry_a.bistatic_delay(points, 0, 0),
        geometry_b.bistatic_delay(points, 0, 0),
        rtol=1e-15,
        atol=1e-20,
    )

    hand = _minimal_geometry(ue_pos=[[3.0, 4.0, 0.0]], bs_pos=[[0.0, 0.0, 0.0]])
    r1, r2 = hand.bistatic_ranges([[3.0, 0.0, 0.0]], 0, 0)
    np.testing.assert_allclose([r1[0], r2[0]], [3.0, 4.0], rtol=1e-15)
    np.testing.assert_allclose(
        hand.bistatic_delay([[3.0, 0.0, 0.0]], 0, 0), [7.0 / SPEED_OF_LIGHT], rtol=1e-15
    )

    ue = np.array([3.0, 4.0, 12.0])
    bs = np.zeros(3)
    los_geometry = _minimal_geometry(ue_pos=[ue], bs_pos=[bs])
    on_segment = bs + 0.3 * (ue - bs)
    np.testing.assert_allclose(
        los_geometry.bistatic_delay([on_segment], 0, 0),
        los_geometry.vs_delay([bs], 0),
        rtol=1e-15,
    )
    np.testing.assert_allclose(los_geometry.vs_delay([bs], 0), [13.0 / SPEED_OF_LIGHT], rtol=1e-15)


def test_mirror_point_involution_and_hand_cases() -> None:
    rng = np.random.default_rng(99)
    plane_point = rng.uniform(-10.0, 10.0, 3)
    normal = rng.uniform(-2.0, 2.0, 3)
    while np.linalg.norm(normal) < 1e-3:
        normal = rng.uniform(-2.0, 2.0, 3)
    points = rng.uniform(-10.0, 10.0, (100, 3))

    mirrored = mirror_point(points, plane_point, normal)
    np.testing.assert_allclose(mirror_point(mirrored, plane_point, normal), points, atol=1e-12)

    unit = normal / np.linalg.norm(normal)
    signed = (points - plane_point) @ unit
    on_plane = points - signed[:, None] * unit
    np.testing.assert_allclose(mirror_point(on_plane, plane_point, normal), on_plane, atol=1e-12)
    np.testing.assert_allclose((mirrored - plane_point) @ unit, -signed, atol=1e-12)

    np.testing.assert_allclose(
        mirror_point([[1.0, 2.0, 3.0]], (7.0, -3.0, 0.0), (0.0, 0.0, 5.0)),
        [[1.0, 2.0, -3.0]],
        atol=1e-15,
    )
    base = mirror_point([[1.1, -2.3, 3.7]], (7.0, -3.0, 0.0), (0.0, 0.0, 5.0))
    np.testing.assert_allclose(
        mirror_point([[1.1, -2.3, 3.7]], (7.0, -3.0, 0.0), (0.0, 0.0, -2.0)),
        base,
        atol=1e-15,
    )
    with pytest.raises(ValueError):
        mirror_point([[1.0, 2.0, 3.0]], (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


def _random_mirror_ray(rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    normal = rng.normal(size=3)
    normal /= np.linalg.norm(normal)
    seed = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(normal, seed)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(normal, e1)

    plane_point = rng.uniform(-5.0, 5.0, 3)
    x0 = plane_point + rng.uniform(-3.0, 3.0) * e1 + rng.uniform(-3.0, 3.0) * e2
    while True:
        outgoing = rng.normal(size=3)
        length = np.linalg.norm(outgoing)
        if length == 0.0:
            continue
        outgoing /= length
        if outgoing @ normal > 0.1:
            break
    rx = x0 + rng.uniform(1.0, 100.0) * outgoing
    incoming = outgoing - 2.0 * (outgoing @ normal) * normal
    tx = x0 - rng.uniform(1.0, 100.0) * incoming
    other_plane = plane_point + rng.uniform(-2.0, 2.0) * e1 + rng.uniform(-2.0, 2.0) * e2
    return normal, plane_point, x0, tx, rx, incoming, other_plane


def test_single_bounce_round_trip() -> None:
    rng = np.random.default_rng(4242)
    for _ in range(200):
        normal, plane_point, x0, tx, rx, _, other_plane = _random_mirror_ray(rng)
        scale = rng.uniform(0.1, 3.0) * (1.0 if rng.random() < 0.5 else -1.0)

        point, valid = single_bounce_point(tx, rx, other_plane, scale * normal)
        assert bool(valid)
        np.testing.assert_allclose(point, x0, atol=1e-9)

        mirror_tx = mirror_point(tx, plane_point, normal)
        path_length = np.linalg.norm(tx - point) + np.linalg.norm(point - rx)
        np.testing.assert_allclose(path_length, np.linalg.norm(mirror_tx - rx), atol=1e-9)

        rx_opposite = mirror_point(rx, plane_point, normal)
        bad_point, bad_valid = single_bounce_point(tx, rx_opposite, plane_point, normal)
        assert not bool(bad_valid)
        assert np.all(np.isnan(bad_point))


def test_vs_departure_dir_matches_analytic_mirror_ray() -> None:
    rng = np.random.default_rng(31337)
    for _ in range(200):
        normal, plane_point, x0, tx, rx, incoming, _ = _random_mirror_ray(rng)
        geom = _minimal_geometry(ue_pos=[rx], bs_pos=[tx], ue_rot=np.eye(3)[None, :, :])

        source = mirror_point(tx, plane_point, normal)
        departure = geom.vs_departure_dir(np.stack([source, source]), 0, 0)
        expected = (x0 - tx) / np.linalg.norm(x0 - tx)
        np.testing.assert_allclose(departure, np.stack([expected, expected]), atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(departure, axis=-1), 1.0, atol=1e-12)
        np.testing.assert_allclose(expected, incoming, atol=1e-12)

        los = geom.vs_departure_dir(tx, 0, 0)
        assert not np.any(np.isnan(los))
        np.testing.assert_allclose(los[0], (rx - tx) / np.linalg.norm(rx - tx), atol=1e-12)


def test_capture_geometry_derived_quantities() -> None:
    geom = CaptureGeometry.from_orientations(
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        np.zeros((3, 3)),
        np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
        f_c=3.5e9,
        bandwidth=100e6,
        num_bins=64,
    )
    assert abs(geom.wavelength - 0.08565) < 1e-5
    assert abs(geom.delta_f - 1.5625e6) < 1e-3
    assert abs(geom.delay_period - 640e-9) < 1e-15
    assert geom.freq_offsets[32] == 0.0
    assert geom.num_views == 3
    assert geom.num_bs == 2
    assert geom.num_elements == 64
    assert geom.num_bins == 64

    # float32 offset quantisation makes the 128-bin unambiguous delay a few
    # femtoseconds short of N / B; keep a tight but achievable tolerance.
    geom128 = CaptureGeometry.from_orientations(
        np.zeros((1, 3)),
        np.zeros((1, 3)),
        np.zeros((1, 3)),
        f_c=3.5e9,
        bandwidth=100e6,
        num_bins=128,
    )
    assert abs(geom128.delay_period - 1280e-9) < 1e-13

    for array in (geom.ue_pos, geom.ue_rot, geom.bs_pos, geom.elem_offsets, geom.freq_offsets):
        assert array.dtype == np.float64
        assert not array.flags.writeable
    with pytest.raises(ValueError):
        geom.ue_pos[0, 0] = 1.0

    selected = geom.select(views=[2, 0], bss=[1])
    np.testing.assert_array_equal(selected.ue_pos, geom.ue_pos[[2, 0]])
    np.testing.assert_array_equal(selected.ue_rot, geom.ue_rot[[2, 0]])
    np.testing.assert_array_equal(selected.bs_pos, geom.bs_pos[[1]])


def test_capture_geometry_validation() -> None:
    base: dict[str, object] = {
        "ue_pos": np.zeros((1, 3)),
        "ue_rot": np.eye(3)[None, :, :],
        "bs_pos": np.zeros((1, 3)),
        "elem_offsets": np.zeros((64, 3)),
        "freq_offsets": np.array([-1e6, -0.5e6, 0.0, 0.5e6]),
        "f_c": 3.5e9,
        "aperture_shape": (8, 8),
    }

    with pytest.raises(ValueError):
        CaptureGeometry(**{**base, "ue_rot": (1.01 * np.eye(3))[None, :, :]})
    with pytest.raises(ValueError):
        CaptureGeometry(**{**base, "ue_rot": np.diag([1.0, 1.0, -1.0])[None, :, :]})
    with pytest.raises(ValueError):
        CaptureGeometry(**{**base, "elem_offsets": np.zeros((63, 3))})
    with pytest.raises(ValueError):
        CaptureGeometry(**{**base, "freq_offsets": np.array([-2e6, 0.0, 1e6, 3e6])})
    with pytest.raises(ValueError):
        CaptureGeometry(**{**base, "freq_offsets": np.arange(8) * 1e6})
    with pytest.raises(ValueError):
        CaptureGeometry(**{**base, "f_c": 0.0})
    with pytest.raises(ValueError):
        CaptureGeometry(**{**base, "ue_pos": np.array([[np.nan, 0.0, 0.0]])})
    with pytest.raises(ValueError):
        CaptureGeometry.from_orientations(
            np.zeros((1, 3)),
            np.zeros((1, 3)),
            np.zeros((1, 3)),
            f_c=-1.0,
            bandwidth=100e6,
            num_bins=8,
        )


def test_nested_view_order() -> None:
    orders = []
    for seed in range(5):
        order = nested_view_order(64, seed)
        assert order.dtype == np.int64
        np.testing.assert_array_equal(np.sort(order), np.arange(64))
        np.testing.assert_array_equal(order, nested_view_order(64, seed))
        for k in range(64):
            assert set(order[:k].tolist()) <= set(order[: k + 1].tolist())
        orders.append(order)
    for i in range(5):
        for j in range(i + 1, 5):
            assert not np.array_equal(orders[i], orders[j])

    np.testing.assert_array_equal(nested_view_order(8, 0), [2, 4, 3, 6, 5, 0, 1, 7])
    np.testing.assert_array_equal(nested_view_order(8, 1), [5, 0, 1, 4, 2, 6, 3, 7])

    with pytest.raises(ValueError):
        nested_view_order(0, 0)
    with pytest.raises(ValueError):
        nested_view_order(8, -1)
