import subprocess
import sys

import numpy as np
import pytest

from plateau_rt.domain.rf_camera.calibration import (
    calibrate_angular_cfr,
    geometric_los_source_direction_local,
)
from plateau_rt.domain.rf_camera.camera import look_at_orientation, to_solid_angle_amplitude
from plateau_rt.domain.rf_camera.delay import SPEED_OF_LIGHT_M_S, propagating_direction_mask
from plateau_rt.domain.rf_camera.image_sources import (
    arrival_unit_vectors,
    find_image_peaks,
    hann_taper,
    local_direction_to_image_coords,
    match_nearest,
    match_peaks_to_sources,
    mirror_point,
    pattern_summed_element_power,
    point_to_ray_distance,
    polyline_length,
    source_recall,
    unfold_specular_chain,
    virtual_source_positions,
    world_to_local_directions,
)
from plateau_rt.domain.rf_camera.imaging import aperture_to_angular_fft


def _angles_from_direction(direction: np.ndarray) -> tuple[float, float]:
    direction = direction / np.linalg.norm(direction)
    theta = float(np.arccos(direction[2]))
    phi = float(np.arctan2(direction[1], direction[0]))
    return theta, phi


def _reference_find_image_peaks(
    power: np.ndarray,
    *,
    mask: np.ndarray,
    threshold_db: float,
    size: int,
) -> list[tuple[int, int]]:
    """Old SciPy-based peak finder, kept as a test oracle for the NumPy rewrite."""
    from scipy.ndimage import maximum_filter

    values = np.asarray(power, dtype=np.float64)
    valid = np.asarray(mask, dtype=bool)
    work = np.where(valid, values, -np.inf)
    local_max = maximum_filter(work, size=size, mode="constant", cval=-np.inf)
    candidates = valid & np.isfinite(work) & (work >= local_max)
    if not np.any(candidates):
        return []
    peak_value = float(np.max(work[candidates]))
    if peak_value <= 0.0:
        return []
    floor = peak_value * 10.0 ** (threshold_db / 10.0)
    candidates &= work >= floor
    rows, cols = np.nonzero(candidates)
    order = np.argsort(work[rows, cols])[::-1]
    rows, cols = rows[order], cols[order]
    keep: list[tuple[int, int]] = []
    min_separation = max(size // 2, 1)
    for row, col in zip(rows.tolist(), cols.tolist()):
        if all(
            max(abs(row - kept_row), abs(col - kept_col)) > min_separation
            for kept_row, kept_col in keep
        ):
            keep.append((row, col))
    return keep


def test_ground_bounce_virtual_source_is_mirror_of_bs():
    bs = np.array([0.0, 0.0, 10.0])
    ue = np.array([20.0, 0.0, 2.0])
    plane_point = np.zeros(3)
    plane_normal = np.array([0.0, 0.0, 1.0])

    virtual_source = mirror_point(bs, plane_point, plane_normal)
    np.testing.assert_allclose(virtual_source, [0.0, 0.0, -10.0], atol=1e-12)

    # Specular point where the straight image-to-UE line crosses z = 0.
    t = -virtual_source[2] / (ue[2] - virtual_source[2])
    specular_point = virtual_source + t * (ue - virtual_source)
    np.testing.assert_allclose(specular_point, [20.0 * 10.0 / 12.0, 0.0, 0.0], atol=1e-12)

    distance = float(np.linalg.norm(virtual_source - ue))
    tau = distance / SPEED_OF_LIGHT_M_S
    theta, phi = _angles_from_direction(specular_point - ue)

    recovered = virtual_source_positions(ue, np.array(tau), np.array(theta), np.array(phi))
    np.testing.assert_allclose(recovered, virtual_source, atol=1e-9)
    assert point_to_ray_distance(virtual_source, ue, specular_point - ue) == pytest.approx(0.0)


def test_two_bounce_corner_image_and_polyline_length():
    # Constructed specular path: BS -> wall x=0 -> wall y=0 -> UE.
    bs = np.array([4.0, 8.0, 2.0])
    vertex_x = np.array([0.0, 4.0, 2.0])
    vertex_y = np.array([4.0, 0.0, 2.0])
    ue = np.array([6.0, 2.0, 2.0])

    # Unfolding across the walls x = 0 and y = 0 yields (-x, -y, z).
    virtual_source = mirror_point(
        mirror_point(bs, np.zeros(3), np.array([1.0, 0.0, 0.0])),
        np.zeros(3),
        np.array([0.0, 1.0, 0.0]),
    )
    np.testing.assert_allclose(virtual_source, [-4.0, -8.0, 2.0], atol=1e-12)

    polyline = np.vstack([bs, vertex_x, vertex_y, ue])
    c_tau = float(np.linalg.norm(ue - virtual_source))
    assert polyline_length(polyline) == pytest.approx(c_tau, rel=1e-12)
    assert c_tau == pytest.approx(10.0 * np.sqrt(2.0), rel=1e-12)

    tau = c_tau / SPEED_OF_LIGHT_M_S
    theta, phi = _angles_from_direction(vertex_y - ue)
    recovered = virtual_source_positions(ue, np.array(tau), np.array(theta), np.array(phi))
    np.testing.assert_allclose(recovered, virtual_source, atol=1e-9)


def test_arrival_unit_vectors_point_from_ue_toward_source():
    theta = np.array([[0.0, np.pi / 2.0], [np.pi, np.pi / 3.0]])
    phi = np.array([[0.0, 0.0], [0.0, np.pi / 2.0]])

    vectors = arrival_unit_vectors(theta, phi)

    assert vectors.shape == (2, 2, 3)
    np.testing.assert_allclose(vectors[0, 0], [0.0, 0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(vectors[0, 1], [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(vectors[1, 0], [0.0, 0.0, -1.0], atol=1e-12)


def test_world_to_local_direction_cosine_projection():
    orientation = look_at_orientation((0.0, 0.0, 0.0), (10.0, 0.0, 0.0))

    front = world_to_local_directions(np.array([1.0, 0.0, 0.0]), orientation)
    ky, kz, in_front = local_direction_to_image_coords(front)
    np.testing.assert_allclose(front, [1.0, 0.0, 0.0], atol=1e-12)
    assert ky == pytest.approx(0.0) and kz == pytest.approx(0.0)
    assert bool(in_front)

    offset = np.array([5.0, 2.0, 1.0])
    offset /= np.linalg.norm(offset)
    local = world_to_local_directions(offset, orientation)
    ky, kz, in_front = local_direction_to_image_coords(local)
    assert ky == pytest.approx(offset[1]) and kz == pytest.approx(offset[2])
    assert bool(in_front)

    behind = world_to_local_directions(np.array([-1.0, 0.0, 0.0]), orientation)
    _, _, in_front = local_direction_to_image_coords(behind)
    assert not bool(in_front)


def test_synthetic_two_wave_image_recovers_direction_cosines():
    rows = cols = 16
    spacing = 0.5
    fft_size = 128
    expected_ky = np.array([0.25, -0.375])
    expected_kz = np.array([-0.125, 0.5])
    expected = np.stack([expected_ky, expected_kz], axis=-1)

    row = np.arange(rows)[:, None]
    col = np.arange(cols)[None, :]
    y = (col - (cols - 1) / 2.0) * spacing
    z = ((rows - 1) / 2.0 - row) * spacing

    aperture = np.zeros((rows, cols, 1), dtype=np.complex128)
    for ky, kz in zip(expected_ky, expected_kz):
        aperture += np.exp(1j * 2.0 * np.pi * (ky * y + kz * z))[:, :, None]

    raw = aperture_to_angular_fft(aperture, fft_rows=fft_size, fft_cols=fft_size)
    calibration = calibrate_angular_cfr(
        raw,
        aperture_rows=rows,
        aperture_cols=cols,
        horizontal_spacing_lambda=spacing,
        vertical_spacing_lambda=spacing,
    )
    image = to_solid_angle_amplitude(calibration.cfr, calibration.ky_over_k, calibration.kz_over_k)
    power = np.abs(image[:, :, 0]) ** 2
    mask = propagating_direction_mask(calibration.ky_over_k, calibration.kz_over_k)

    peaks = find_image_peaks(power, mask=mask, threshold_db=-20.0, size=5)
    assert len(peaks) >= 2
    detected = np.array(
        [[calibration.ky_over_k[col], calibration.kz_over_k[row]] for row, col in peaks]
    )

    pitch = 1.0 / (fft_size * spacing)
    _, distances = match_nearest(expected, detected)
    assert np.all(distances <= pitch)


def test_pattern_summed_element_power_combines_hemispheres():
    rows, cols, num_patterns = 2, 3, 2
    size = rows * cols
    fused = np.zeros((num_patterns * size, 3), dtype=np.complex128)
    fused[0, 0] = 0.5 + 0.5j
    fused[size, 1] = 1j

    power = pattern_summed_element_power(fused, rows=rows, cols=cols, axis=0)

    assert power.shape == (3,)
    np.testing.assert_allclose(power, [0.5, 1.0, 0.0], atol=1e-12)

    # The old "channel 0 only" formula loses the back-hemisphere path entirely.
    old = np.abs(fused[0, :]) ** 2
    assert old[1] == pytest.approx(0.0)
    assert power[1] != old[1]

    # Leading receiver axis and axis=1.
    leading = np.zeros((4, num_patterns * size, 3), dtype=np.complex128)
    leading[0, 0, 0] = 1.0
    leading[1, size, 0] = 2j
    out = pattern_summed_element_power(leading, rows=rows, cols=cols, axis=1)
    assert out.shape == (4, 3)
    assert out[0, 0] == pytest.approx(1.0)
    assert out[1, 0] == pytest.approx(4.0)

    with pytest.raises(ValueError):
        pattern_summed_element_power(np.zeros((5, 3)), rows=rows, cols=cols, axis=0)
    with pytest.raises(ValueError):
        pattern_summed_element_power(fused, rows=rows, cols=cols, element=size, axis=0)


def test_match_peaks_to_sources_ignores_back_hemisphere():
    sources = np.array([[0.10, 0.0], [0.20, 0.0]])
    source_ids = np.array([5, 9])
    candidate_mask = np.array([False, True])
    peaks = np.array([[0.11, 0.0]])

    match_ids, distances = match_peaks_to_sources(
        peaks, sources, source_ids=source_ids, candidate_mask=candidate_mask
    )
    assert match_ids[0] == 9
    assert distances[0] == pytest.approx(0.09)

    # Old plain nearest-neighbour would pick the back-hemisphere source in front.
    old_indices, _ = match_nearest(peaks, sources)
    assert old_indices[0] == 0
    assert source_ids[old_indices[0]] == 5


def test_match_peaks_to_sources_returns_path_ids():
    sources = np.array([[0.0, 0.0], [0.1, 0.0], [0.5, 0.5]])
    source_ids = np.array([7, 3, 12])
    peak = np.array([[0.11, 0.0]])

    match_ids, _ = match_peaks_to_sources(
        peak, sources, source_ids=source_ids, candidate_mask=np.ones(3, dtype=bool)
    )
    assert match_ids[0] == 3


def test_match_peaks_to_sources_gate_and_empty_cases():
    sources = np.array([[0.0, 0.0]])
    peak = np.array([[0.5, 0.0]])

    match_ids, distances = match_peaks_to_sources(
        peak, sources, source_ids=np.array([1]), candidate_mask=np.array([True]), max_distance=0.1
    )
    assert match_ids[0] == -1
    assert np.isfinite(distances[0]) and distances[0] == pytest.approx(0.5)

    match_ids, distances = match_peaks_to_sources(
        peak,
        sources,
        source_ids=np.array([1]),
        candidate_mask=np.array([False]),
        max_distance=1.0,
    )
    assert match_ids[0] == -1 and np.isinf(distances[0])

    empty_ids, empty_distances = match_peaks_to_sources(
        np.empty((0, 2)),
        sources,
        source_ids=np.array([1]),
        candidate_mask=np.array([True]),
    )
    assert empty_ids.shape == (0,) and empty_distances.shape == (0,)


def test_hann_taper_shape_and_symmetry():
    taper = hann_taper(5, 7)
    assert taper.shape == (5, 7)
    assert np.all(taper > 0.0)
    np.testing.assert_allclose(taper, np.flipud(taper), atol=1e-12)
    np.testing.assert_allclose(taper, np.fliplr(taper), atol=1e-12)
    assert int(np.argmax(taper)) == int(np.ravel_multi_index((2, 3), taper.shape))
    assert taper[2, 3] == pytest.approx(float(taper.max()))

    with pytest.raises(ValueError):
        hann_taper(0, 3)
    with pytest.raises(ValueError):
        hann_taper(3, 0)


def _single_plane_wave_image(ky: float, kz: float, *, tapered: bool):
    rows = cols = 16
    spacing = 0.5
    fft_size = 128
    row = np.arange(rows)[:, None]
    col = np.arange(cols)[None, :]
    y = (col - (cols - 1) / 2.0) * spacing
    z = ((rows - 1) / 2.0 - row) * spacing
    aperture = np.exp(1j * 2.0 * np.pi * (ky * y + kz * z))[:, :, None]
    if tapered:
        aperture = aperture * hann_taper(rows, cols)[:, :, None]
    raw = aperture_to_angular_fft(aperture, fft_rows=fft_size, fft_cols=fft_size)
    calibration = calibrate_angular_cfr(
        raw,
        aperture_rows=rows,
        aperture_cols=cols,
        horizontal_spacing_lambda=spacing,
        vertical_spacing_lambda=spacing,
    )
    image = to_solid_angle_amplitude(calibration.cfr, calibration.ky_over_k, calibration.kz_over_k)
    power = np.abs(image[:, :, 0]) ** 2
    mask = propagating_direction_mask(calibration.ky_over_k, calibration.kz_over_k)
    return calibration, power, mask


def test_hann_taper_suppresses_sidelobes_and_source_recall():
    ky, kz = 0.3, -0.2
    pitch = 1.0 / (128 * 0.5)

    calibration, power, mask = _single_plane_wave_image(ky, kz, tapered=False)
    untapered = find_image_peaks(power, mask=mask, threshold_db=-20.0, size=5)
    assert len(untapered) > 1

    calibration, power, mask = _single_plane_wave_image(ky, kz, tapered=True)
    tapered = find_image_peaks(power, mask=mask, threshold_db=-20.0, size=5)
    assert len(tapered) == 1
    row, col = tapered[0]
    detected = np.array([calibration.ky_over_k[col], calibration.kz_over_k[row]])
    assert np.linalg.norm(detected - np.array([ky, kz])) <= pitch

    recalled = source_recall(
        np.array([[0.01, 0.0]]), np.array([[0.0, 0.0], [1.0, 1.0]]), max_distance=0.1
    )
    np.testing.assert_array_equal(recalled, [True, False])
    assert not np.any(source_recall(np.empty((0, 2)), np.array([[0.0, 0.0]]), max_distance=0.1))


def test_world_to_local_non_symmetric_rotation():
    orientation = look_at_orientation((0.0, 0.0, 0.0), (0.0, 10.0, 0.0))

    forward = world_to_local_directions(np.array([0.0, 1.0, 0.0]), orientation)
    np.testing.assert_allclose(forward, [1.0, 0.0, 0.0], atol=1e-12)

    left_world = world_to_local_directions(np.array([-1.0, 0.0, 0.0]), orientation)
    assert left_world[1] == pytest.approx(1.0)
    right_world = world_to_local_directions(np.array([1.0, 0.0, 0.0]), orientation)
    assert right_world[1] == pytest.approx(-1.0)

    pitched = look_at_orientation((0.0, 0.0, 0.0), (-10.0, 5.0, 4.0))
    forward_world = np.asarray([-10.0, 5.0, 4.0])
    forward_world /= np.linalg.norm(forward_world)
    left = np.cross(np.array([0.0, 0.0, 1.0]), forward_world)
    left /= np.linalg.norm(left)
    up = np.cross(forward_world, left)

    rng = np.random.default_rng(0)
    directions = rng.normal(size=(5, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    for direction in directions:
        expected = np.array([direction @ forward_world, direction @ left, direction @ up])
        np.testing.assert_allclose(
            world_to_local_directions(direction, pitched), expected, atol=1e-12
        )

    tx = (3.0, -7.0, 9.0)
    ue = (1.0, 2.0, 1.5)
    world = np.asarray(tx) - np.asarray(ue)
    world /= np.linalg.norm(world)
    np.testing.assert_allclose(
        world_to_local_directions(world, pitched),
        geometric_los_source_direction_local(
            tx_position=tx, ue_position=ue, ue_orientation=pitched
        ),
        atol=1e-12,
    )


def test_find_image_peaks_matches_scipy_reference():
    for seed in range(5):
        local = np.random.default_rng(seed)
        power = local.random((40, 50))
        mask = local.random((40, 50)) > 0.3
        for size in (3, 5):
            expected = _reference_find_image_peaks(power, mask=mask, threshold_db=-20.0, size=size)
            assert find_image_peaks(power, mask=mask, threshold_db=-20.0, size=size) == expected


def test_find_image_peaks_plateau_mask_and_threshold():
    power = np.zeros((7, 7))
    power[1:3, 1:3] = 5.0
    mask = np.ones((7, 7), dtype=bool)
    peaks = find_image_peaks(power, mask=mask, threshold_db=-20.0, size=3)
    assert peaks[0] in {(1, 1), (1, 2), (2, 1), (2, 2)}
    assert len(peaks) == 1

    outside = np.zeros((7, 7))
    outside[0, 0] = 100.0
    outside[3, 3] = 1.0
    mask = np.ones((7, 7), dtype=bool)
    mask[0, 0] = False
    assert find_image_peaks(outside, mask=mask, threshold_db=-20.0, size=3) == [(3, 3)]

    assert find_image_peaks(np.ones((5, 5)), mask=np.zeros((5, 5), bool)) == []
    assert find_image_peaks(np.zeros((5, 5)), mask=np.ones((5, 5), bool)) == []

    weaker = np.zeros((7, 7))
    weaker[1, 1] = 1.0
    weaker[5, 5] = 10.0 ** (-25.0 / 10.0)
    full_mask = np.ones((7, 7), dtype=bool)
    assert len(find_image_peaks(weaker, mask=full_mask, threshold_db=-20.0, size=3)) == 1
    assert len(find_image_peaks(weaker, mask=full_mask, threshold_db=-30.0, size=3)) == 2

    with pytest.raises(ValueError):
        find_image_peaks(weaker, mask=np.ones((7, 7), bool), size=4)


def test_image_sources_module_drops_scipy_and_nearest_pixel():
    code = (
        "import sys\n"
        "import plateau_rt.domain.rf_camera.image_sources as module\n"
        "print('scipy' in sys.modules, hasattr(module, 'nearest_pixel'))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False False"


def test_unfold_specular_chain_ground_bounce():
    bs = np.array([0.0, 0.0, 10.0])
    vertex = np.array([20.0 * 10.0 / 12.0, 0.0, 0.0])
    ue = np.array([20.0, 0.0, 2.0])

    image = unfold_specular_chain(bs, vertex[None, :], np.array([True]), ue)
    np.testing.assert_allclose(image, [0.0, 0.0, -10.0], atol=1e-9)


def test_unfold_specular_chain_two_bounce_corner():
    bs = np.array([4.0, 8.0, 2.0])
    vertices = np.array([[0.0, 4.0, 2.0], [4.0, 0.0, 2.0]])
    ue = np.array([6.0, 2.0, 2.0])

    image = unfold_specular_chain(bs, vertices, np.array([True, True]), ue)
    np.testing.assert_allclose(image, [-4.0, -8.0, 2.0], atol=1e-9)


def test_unfold_specular_chain_refraction_slab_is_straight_through():
    bs = np.array([0.0, 0.0, 5.0])
    ue = np.array([10.0, 0.0, 5.0])
    specular = np.array([5.0, 10.0, 5.0])
    vertices = np.array(
        [
            0.5 * (bs + specular),
            specular,
            0.5 * (specular + ue),
        ]
    )
    image = unfold_specular_chain(bs, vertices, np.array([False, True, False]), ue)
    np.testing.assert_allclose(image, [0.0, 20.0, 5.0], atol=1e-9)


def test_unfold_specular_chain_ignores_jittered_refraction_vertices():
    bs = np.array([0.0, 0.0, 5.0])
    ue = np.array([10.0, 0.0, 5.0])
    specular = np.array([5.0, 10.0, 5.0])
    # Straight-through refraction vertices displaced by 1 mm perpendicular to
    # their segments; the exact mirror across y = 10 is unaffected.
    jitter = np.array([0.0, 0.0, 1e-3])
    vertices = np.array(
        [
            0.5 * (bs + specular) + jitter,
            specular,
            0.5 * (specular + ue) + jitter,
        ]
    )
    image = unfold_specular_chain(bs, vertices, np.array([False, True, False]), ue)
    np.testing.assert_allclose(image, [0.0, 20.0, 5.0], atol=1e-9)


def test_unfold_specular_chain_without_vertices_returns_tx():
    bs = np.array([1.0, 2.0, 3.0])
    ue = np.array([4.0, 5.0, 6.0])
    image = unfold_specular_chain(bs, np.empty((0, 3)), np.empty(0, dtype=bool), ue)
    np.testing.assert_allclose(image, bs, atol=1e-12)


def test_point_to_ray_distance_vertex_on_ray_is_zero():
    vertex = np.array([2.0, -1.0, 3.0])
    ue = np.array([0.0, 0.0, 0.0])
    assert point_to_ray_distance(vertex, ue, vertex - ue) == pytest.approx(0.0)
