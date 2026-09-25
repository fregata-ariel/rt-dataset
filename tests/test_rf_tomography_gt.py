"""Unit tests for the tomography ground-truth domain module (T17, design §6.1)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh
from rf_tomography_gt_fixtures import (
    build_mirror_scene,
    expected_planes,
)
from scipy.constants import c as SPEED_OF_LIGHT

from plateau_rt.domain.rf_camera.image_sources import (
    arrival_unit_vectors,
    virtual_source_positions,
)
from plateau_rt.domain.rf_tomography.forward_exact import capture_factors
from plateau_rt.domain.rf_tomography.geometry import CaptureGeometry
from plateau_rt.domain.rf_tomography.gt import (
    PATH_TYPE_NAMES,
    PathGT,
    aperture_coefficient,
    beyond_period,
    ground_bounce_visibility,
    interaction_table,
    los_model_error,
    los_visibility,
    path_ground_truth,
    path_power,
    path_types,
    reflection_planes,
    sample_triangles,
    segments_blocked,
    specular_support,
    surface_ground_truth,
    surface_observability,
    triangle_normals,
    virtual_sources,
)

FIXTURES = Path(__file__).parent / "fixtures" / "rf_tomography"


def _path(scene) -> PathGT:
    """Return the scene's :class:`PathGT`."""
    return PathGT.from_arrays(scene.arrays, scene.object_names)


def _match(vs, bs: int, pos: np.ndarray) -> int:
    """Return the cluster index on ``bs`` nearest to ``pos``."""
    candidates = np.nonzero(vs.bs == bs)[0]
    distances = np.linalg.norm(vs.pos[candidates] - pos, axis=1)
    return int(candidates[int(np.argmin(distances))])


def test_path_types() -> None:
    scene = build_mirror_scene()
    path = _path(scene)
    np.testing.assert_array_equal(path_types(path), scene.path_type)

    num_views, num_bs, num_paths, depth = 1, 1, 6, 2
    interactions = np.zeros((num_views, num_bs, num_paths, depth), dtype=np.uint32)
    interactions[0, 0, 0] = [1, 2]
    interactions[0, 0, 1] = [8, 4]
    interactions[0, 0, 2] = [4, 1]
    interactions[0, 0, 3] = [1, 1]
    valid = np.ones((num_views, num_bs, num_paths), dtype=bool)
    valid[0, 0, 5] = False
    arrays = {
        "valid": valid,
        "tau": np.zeros((num_views, num_bs, num_paths)),
        "theta_t": np.zeros((num_views, num_bs, num_paths)),
        "phi_t": np.zeros((num_views, num_bs, num_paths)),
        "theta_r": np.zeros((num_views, num_bs, num_paths)),
        "phi_r": np.zeros((num_views, num_bs, num_paths)),
        "a_baseband": np.zeros((num_views, num_bs, 2, 1, 1, num_paths), dtype=np.complex128),
        "interactions": interactions,
        "object_index": np.full(interactions.shape, -1, dtype=np.int32),
        "vertices": np.zeros((num_views, num_bs, num_paths, depth, 3)),
    }
    handmade = PathGT.from_arrays(arrays, ("ground_plane",))
    np.testing.assert_array_equal(
        path_types(handmade), np.array([[[4, 3, 2, 1, 0, -1]]], dtype=np.int8)
    )
    assert PATH_TYPE_NAMES == ("los", "specular", "refraction", "diffraction", "diffuse")


def test_path_labels_and_power() -> None:
    scene = build_mirror_scene()
    path = _path(scene)
    wrap = beyond_period(path, scene.geom.delay_period)
    np.testing.assert_array_equal(wrap, scene.beyond_period)
    assert int(np.count_nonzero(wrap)) == 1
    np.testing.assert_array_equal(los_visibility(path), scene.los_visible)
    np.testing.assert_array_equal(ground_bounce_visibility(path), scene.ground_bounce_visible)
    power = path_power(path)
    expected = np.sum(np.abs(path.a_baseband) ** 2, axis=(2, 3, 4))
    np.testing.assert_allclose(power, expected, rtol=0.0, atol=0.0)
    assert bool(np.all(power[~path.valid] == 0.0))


def test_virtual_sources_float64() -> None:
    scene = build_mirror_scene()
    path = _path(scene)
    geom = scene.geom
    vs = virtual_sources(path, geom, pattern="tr38901")
    assert vs.pos.shape[0] == len(scene.vs) == 7

    own = virtual_source_positions(
        geom.ue_pos[:, None, None, :], path.tau, path.theta_r, path.phi_r
    )
    candidate = path.valid & np.isin(path_types(path), (0, 1, 2))
    for v in range(path.num_views):
        for b in range(path.num_bs):
            for p in range(path.num_paths):
                if candidate[v, b, p]:
                    index = int(vs.path_vs[v, b, p])
                    assert index >= 0
                    assert np.linalg.norm(vs.pos[index] - own[v, b, p]) < 1e-6
                else:
                    assert vs.path_vs[v, b, p] == -1

    total_power = vs.power.sum(axis=1)
    for i in range(1, vs.pos.shape[0]):
        assert (vs.bs[i], vs.order[i]) >= (vs.bs[i - 1], vs.order[i - 1])
        if vs.bs[i] == vs.bs[i - 1] and vs.order[i] == vs.order[i - 1]:
            assert total_power[i] <= total_power[i - 1] + 1e-12

    for expected in scene.vs:
        m = _match(vs, expected.bs, expected.pos)
        assert np.linalg.norm(vs.pos[m] - expected.pos) < 1e-9
        assert vs.spread[m] < 1e-6
        assert vs.order[m] == expected.order
        if expected.order:
            np.testing.assert_array_equal(
                vs.objects[m, : expected.order], np.asarray(expected.objects)
            )
        assert bool(np.all(vs.objects[m, expected.order :] == -1))
        assert set(np.nonzero(vs.visibility[m])[0].tolist()) == set(expected.beta)
        for v, beta in expected.beta.items():
            assert abs(vs.rho_eff[m, v] - beta) <= 1e-9 * abs(beta)
            assert vs.path_type[m, v] == expected.path_type[v]
        for v, cos_inc in expected.cos_inc.items():
            assert abs(vs.theta_inc[m, v] - np.arccos(cos_inc)) <= 1e-9
        invisible = [v for v in range(scene.geom.num_views) if v not in expected.beta]
        for v in invisible:
            assert np.isnan(vs.rho_eff[m, v])
            assert np.isnan(vs.theta_inc[m, v])
        if expected.order >= 2:
            assert bool(np.all(np.isnan(vs.theta_inc[m])))

    los_bs0 = _match(vs, 0, np.array([-40.0, 5.0, 25.0]))
    assert vs.path_type[los_bs0, 1] == 2
    assert abs(vs.rho_eff[los_bs0, 1] - 0.01 * np.exp(0.3j)) <= 1e-9


def test_virtual_sources_float32() -> None:
    scene = build_mirror_scene(float32=True)
    path = _path(scene)
    vs = virtual_sources(path, scene.geom, pattern="tr38901")
    assert vs.pos.shape[0] == 7
    assert float(np.max(vs.spread)) < 1e-4
    for expected in scene.vs:
        m = _match(vs, expected.bs, expected.pos)
        assert np.linalg.norm(vs.pos[m] - expected.pos) < 1e-4
        for v, beta in expected.beta.items():
            assert abs(vs.rho_eff[m, v] - beta) <= 2e-3 * abs(beta)


def test_cluster_tolerance() -> None:
    scene = build_mirror_scene()
    arrays = {name: value.copy() for name, value in scene.arrays.items()}
    arrays["tau"][0, 0, 2] += 0.05 / SPEED_OF_LIGHT
    path = PathGT.from_arrays(arrays, scene.object_names)
    default = virtual_sources(path, scene.geom, pattern="tr38901")
    assert default.pos.shape[0] == 8
    loose = virtual_sources(path, scene.geom, pattern="tr38901", cluster_tol_m=0.2)
    assert loose.pos.shape[0] == 7
    spread_mask = loose.spread > 0.03
    assert int(np.count_nonzero(spread_mask)) == 1


def test_second_order_departure() -> None:
    scene = build_mirror_scene()
    path = _path(scene)
    geom = scene.geom
    vs = virtual_sources(path, geom, pattern="tr38901")
    order_two = next(expected for expected in scene.vs if expected.order == 2)
    m = _match(vs, order_two.bs, order_two.pos)
    errors = []
    for v in order_two.beta:
        p = 4
        u_local = arrival_unit_vectors(path.theta_r[v, 0, p], path.phi_r[v, 0, p]) @ geom.ue_rot[v]
        alpha = aperture_coefficient(path.a_baseband[v, 0, :, :, :, p], u_local, geom)
        wrong = alpha / capture_factors(vs.pos[m][None, :], geom, "vs", v, 0).gamma[0]
        beta = order_two.beta[v]
        errors.append(abs(wrong - beta) / abs(beta))
    assert max(errors) > 1e-2


def test_los_model_error_iso_and_tr38901() -> None:
    iso = build_mirror_scene(pattern="iso")
    phase, amp_db = los_model_error(_path(iso), iso.geom, pattern="iso")
    finite = np.isfinite(phase)
    np.testing.assert_array_equal(finite, iso.los_visible)
    assert float(np.max(np.abs(phase[finite]))) < 1e-12
    assert float(np.max(np.abs(amp_db[finite]))) < 1e-10
    assert bool(np.all(np.isnan(phase[~iso.los_visible])))

    default = build_mirror_scene()
    phase, amp_db = los_model_error(_path(default), default.geom, pattern="tr38901")
    finite = np.isfinite(phase)
    np.testing.assert_array_equal(finite, default.los_visible)
    assert float(np.max(np.abs(phase[finite]))) < 1e-12
    assert float(np.max(np.abs(amp_db[finite]))) < 1e-10

    wrong = los_model_error(_path(iso), iso.geom, pattern="tr38901")[1]
    assert bool(np.all(np.abs(wrong[iso.los_visible]) > 1.0))


def test_los_model_error_sionna_fixture() -> None:
    fixture = np.load(FIXTURES / "sionna_mock_los.npz")
    geom = CaptureGeometry.from_orientations(
        ue_pos=fixture["ue_pos"],
        ue_orientations=fixture["ue_orientation"],
        bs_pos=fixture["bs_pos"],
        f_c=float(fixture["f_c"]),
        bandwidth=100e6,
        num_bins=64,
        bs_look_at=fixture["bs_look_at"],
    )
    departure = fixture["ue_pos"] - fixture["bs_pos"]
    departure = departure / np.linalg.norm(departure, axis=1, keepdims=True)
    arrival = -departure
    theta_t = np.arccos(departure[:, 2])
    phi_t = np.arctan2(departure[:, 1], departure[:, 0])
    theta_r = np.arccos(arrival[:, 2])
    phi_r = np.arctan2(arrival[:, 1], arrival[:, 0])
    num_views = fixture["ue_pos"].shape[0]
    valid = np.ones((num_views, 1, 1), dtype=bool)
    tau = fixture["path_tau"].reshape(num_views, 1, 1)
    angles = {
        "theta_t": theta_t.reshape(num_views, 1, 1),
        "phi_t": phi_t.reshape(num_views, 1, 1),
        "theta_r": theta_r.reshape(num_views, 1, 1),
        "phi_r": phi_r.reshape(num_views, 1, 1),
    }
    a_baseband = np.zeros((num_views, 1, 2, 8, 8, 1), dtype=np.complex128)
    for v in range(num_views):
        a_baseband[v, 0, :, :, :, 0] = fixture["aperture_cfr"][v, :, :, :, 8]
    arrays = {
        "valid": valid,
        "tau": tau,
        "a_baseband": a_baseband,
        "interactions": np.zeros((num_views, 1, 1, 1), dtype=np.uint32),
        "object_index": np.full((num_views, 1, 1, 1), -1, dtype=np.int32),
        "vertices": np.zeros((num_views, 1, 1, 1, 3)),
        **angles,
    }
    path = PathGT.from_arrays(arrays, ("ground_plane",))
    phase, amp_db = los_model_error(path, geom, pattern="tr38901", polarization="vv")
    assert float(np.max(np.abs(np.degrees(phase)))) < 0.1
    assert float(np.max(np.abs(amp_db))) < 0.01

    phase_none, amp_none = los_model_error(path, geom, pattern="tr38901", polarization="none")
    assert float(np.max(np.abs(np.degrees(phase_none)))) < 0.1
    assert bool(np.all(amp_none < 0.0)) and bool(np.all(amp_none > -0.5))


def test_reflection_planes() -> None:
    scene = build_mirror_scene()
    path = _path(scene)
    geom = scene.geom
    planes = reflection_planes(path, geom)
    assert planes.normal.shape[0] == 3
    expected = expected_planes()
    name_to_index: dict[str, int] = {}
    for name, (normal, offset, obj) in expected.items():
        distance = np.linalg.norm(planes.normal - normal, axis=1)
        matches = np.nonzero((distance < 1e-9) & (np.abs(planes.offset - offset) < 1e-9))[0]
        matches = [index for index in matches if planes.object[index] == obj]
        assert len(matches) == 1
        name_to_index[name] = int(matches[0])

    vs = virtual_sources(path, geom, pattern="tr38901", planes=planes)
    for expected_vs in scene.vs:
        m = _match(vs, expected_vs.bs, expected_vs.pos)
        if expected_vs.order == 0:
            assert bool(np.all(vs.plane_ids[m] == -1))
        else:
            ids = [name_to_index[name] for name in expected_vs.planes]
            np.testing.assert_array_equal(vs.plane_ids[m, : expected_vs.order], ids)
            assert bool(np.all(vs.plane_ids[m, expected_vs.order :] == -1))

    table = interaction_table(path)
    for q in range(table.points.shape[0]):
        plane = planes.vertex_plane[table.view[q], table.bs[q], table.path[q], table.depth[q]]
        if table.type[q] & 1:
            assert plane >= 0
        else:
            assert plane == -1

    float_scene = build_mirror_scene(float32=True)
    float_planes = reflection_planes(_path(float_scene), float_scene.geom)
    for name, (normal, offset, obj) in expected.items():
        distance = np.linalg.norm(float_planes.normal - normal, axis=1)
        matches = np.nonzero((distance < 1e-4) & (np.abs(float_planes.offset - offset) < 1e-4))[0]
        assert any(float_planes.object[index] == obj for index in matches)


def test_interaction_table() -> None:
    scene = build_mirror_scene()
    path = _path(scene)
    table = interaction_table(path)
    total = int(path.num_interactions[path.valid].sum())
    assert table.points.shape[0] == total
    index = 0
    for v in range(path.num_views):
        for b in range(path.num_bs):
            for p in range(path.num_paths):
                if not path.valid[v, b, p]:
                    continue
                for d in range(int(path.num_interactions[v, b, p])):
                    assert (table.view[index], table.bs[index], table.path[index]) == (v, b, p)
                    assert table.depth[index] == d
                    assert table.type[index] == path.interactions[v, b, p, d]
                    assert table.object[index] == path.object_index[v, b, p, d]
                    np.testing.assert_array_equal(table.points[index], path.vertices[v, b, p, d])
                    index += 1
    assert index == total


def test_triangle_normals_and_sampling() -> None:
    normals, valid = triangle_normals(
        np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=np.float64)
    )
    np.testing.assert_allclose(normals[0], [0, 0, 1], atol=1e-12)
    assert bool(valid[0])

    right = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=np.float64)
    points, sample_normals, index = sample_triangles(right, 0.25)
    i, j = np.meshgrid(np.arange(4), np.arange(4), indexing="ij")
    keep = (i + j <= 3).ravel()
    expected = np.stack([((i.ravel() + 0.5) / 4)[keep], ((j.ravel() + 0.5) / 4)[keep]], axis=1)
    expected_points = np.concatenate([expected, np.zeros((expected.shape[0], 1))], axis=1)
    sorted_points = points[np.lexsort((points[:, 2], points[:, 1], points[:, 0]))]
    sorted_expected = expected_points[
        np.lexsort((expected_points[:, 2], expected_points[:, 1], expected_points[:, 0]))
    ]
    np.testing.assert_allclose(sorted_points, sorted_expected, atol=1e-12)
    np.testing.assert_array_equal(sample_normals, np.broadcast_to([0, 0, 1], points.shape))
    assert bool(np.all(index == 0))

    square = np.array(
        [
            [[0, 0, 0], [1, 0, 0], [1, 1, 0]],
            [[0, 0, 0], [1, 1, 0], [0, 1, 0]],
        ],
        dtype=np.float64,
    )
    points, _, _ = sample_triangles(square, 0.25)
    assert points.shape[0] == 16
    flipped = square.copy()
    flipped[1] = flipped[1][::-1]
    flipped_points, _, _ = sample_triangles(flipped, 0.25)
    assert flipped_points.shape[0] == 16
    order = np.lexsort((points[:, 2], points[:, 1], points[:, 0]))
    flipped_order = np.lexsort((flipped_points[:, 2], flipped_points[:, 1], flipped_points[:, 0]))
    np.testing.assert_allclose(points[order], flipped_points[flipped_order], atol=1e-12)

    tiny = np.array([[[0.3, 0.3, 0], [0.4, 0.3, 0], [0.3, 0.4, 0]]], dtype=np.float64)
    points, _, _ = sample_triangles(tiny, 0.25)
    np.testing.assert_allclose(points, [[1.0 / 3.0, 1.0 / 3.0, 0.0]], atol=1e-12)

    degenerate = np.array([[[0, 0, 0], [1, 0, 0], [2, 0, 0]]], dtype=np.float64)
    points, _, _ = sample_triangles(degenerate, 0.25)
    assert points.shape[0] == 0

    wall = np.array(
        [
            [[2, 0, 0], [2, 2, 0], [2, 2, 1]],
            [[2, 0, 0], [2, 2, 1], [2, 0, 1]],
        ],
        dtype=np.float64,
    )
    points, _, _ = sample_triangles(wall, 0.25)
    assert points.shape[0] == 32
    np.testing.assert_allclose(points[:, 0], 2.0, atol=1e-12)


def test_segments_blocked() -> None:
    triangle = np.array([[[0, 0, 0], [1, 0, 0], [0, 1, 0]]], dtype=np.float64)

    def blocked(origin, target):
        return bool(
            segments_blocked(
                np.array([origin], dtype=np.float64),
                np.array([target], dtype=np.float64),
                triangle,
            )[0]
        )

    assert blocked((0.2, 0.2, 1.0), (0.2, 0.2, -1.0))
    assert not blocked((0.8, 0.8, 1.0), (0.8, 0.8, -1.0))
    assert not blocked((0.2, 0.2, 1.0), (0.2, 0.2, 0.5))
    assert not blocked((0.2, 0.2, -0.5), (0.2, 0.2, -1.0))
    assert not blocked((0.2, 0.2, 0.5), (0.9, 0.9, 0.5))


def _box() -> trimesh.Trimesh:
    box = trimesh.creation.box(extents=(2.0, 2.0, 3.0))
    box.apply_translation((0.0, 0.0, 1.5))
    return box


def _ground() -> np.ndarray:
    return np.array(
        [
            [[-10, -10, -0.01], [10, -10, -0.01], [10, 10, -0.01]],
            [[-10, -10, -0.01], [10, 10, -0.01], [-10, 10, -0.01]],
        ],
        dtype=np.float64,
    )


def _slab_hit(origin: np.ndarray, target: np.ndarray, low: np.ndarray, high: np.ndarray) -> bool:
    direction = target - origin
    enter, exit_ = -np.inf, np.inf
    for axis in range(3):
        if abs(direction[axis]) < 1e-15:
            if origin[axis] < low[axis] or origin[axis] > high[axis]:
                return False
            continue
        t1 = (low[axis] - origin[axis]) / direction[axis]
        t2 = (high[axis] - origin[axis]) / direction[axis]
        if t1 > t2:
            t1, t2 = t2, t1
        enter = max(enter, t1)
        exit_ = min(exit_, t2)
    return max(enter, 0.0) < min(exit_, 1.0)


def test_surface_observability_box() -> None:
    box = _box()
    box_triangles = np.asarray(box.vertices, dtype=np.float64)[np.asarray(box.faces)]
    ground = _ground()
    triangles = np.concatenate([box_triangles, ground], axis=0)
    bs_pos = np.array([[15.0, 4.0, 12.0]])
    ue_pos = np.array([[12.0, -3.0, 1.5], [-12.0, 6.0, 1.5]])
    points, normals, _ = sample_triangles(triangles, 0.25)
    assert points.shape[0] == 6912
    observable = surface_observability(points, normals, triangles, bs_pos, ue_pos)
    assert int(np.count_nonzero(observable)) == 6266

    low = np.array([-1.0, -1.0, 0.0])
    high = np.array([1.0, 1.0, 3.0])
    expected = np.zeros(points.shape[0], dtype=bool)
    for side in (1.0, -1.0):
        origin = points + side * 1e-3 * normals
        bs_ok = np.zeros(points.shape[0], dtype=bool)
        for b in range(bs_pos.shape[0]):
            facing = side * np.einsum("si,si->s", normals, bs_pos[b] - points) > 0.0
            for s in np.nonzero(facing)[0]:
                if not _slab_hit(origin[s], bs_pos[b], low, high):
                    bs_ok[s] = True
        ue_ok = np.zeros(points.shape[0], dtype=bool)
        for p in range(ue_pos.shape[0]):
            facing = side * np.einsum("si,si->s", normals, ue_pos[p] - points) > 0.0
            for s in np.nonzero(facing)[0]:
                if not _slab_hit(origin[s], ue_pos[p], low, high):
                    ue_ok[s] = True
        expected |= bs_ok & ue_ok
    np.testing.assert_array_equal(observable, expected)

    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    plus_x = np.isclose(x, 1.0) & (np.abs(y) <= 1.0 + 1e-9) & (z >= 0.0) & (z <= 3.0)
    plus_y = np.isclose(y, 1.0) & (np.abs(x) <= 1.0 + 1e-9) & (z >= 0.0) & (z <= 3.0)
    minus_x = np.isclose(x, -1.0) & (np.abs(y) <= 1.0 + 1e-9) & (z >= 0.0) & (z <= 3.0)
    minus_y = np.isclose(y, -1.0) & (np.abs(x) <= 1.0 + 1e-9) & (z >= 0.0) & (z <= 3.0)
    top = np.isclose(z, 3.0) & (np.abs(x) <= 1.0 + 1e-9) & (np.abs(y) <= 1.0 + 1e-9)
    assert bool(np.all(observable[plus_x]))
    assert bool(np.all(observable[plus_y]))
    assert not bool(np.any(observable[minus_x]))
    assert not bool(np.any(observable[minus_y]))
    assert not bool(np.any(observable[top]))
    ground_mask = np.isclose(z, -0.01)
    footprint = ground_mask & (np.abs(x) < 1.0) & (np.abs(y) < 1.0)
    assert not bool(np.any(observable[footprint]))
    outside = ground_mask & (~((np.abs(x) <= 1.0) & (np.abs(y) <= 1.0)))
    assert int(np.count_nonzero(~observable[outside])) >= 50

    # Two-sided: reversing every triangle's winding flips the normals, not the answer.
    flipped = triangles[:, ::-1, :]
    flipped_points, flipped_normals, _ = sample_triangles(flipped, 0.25)
    order = np.lexsort(np.round(points, 6).T)
    flipped_order = np.lexsort(np.round(flipped_points, 6).T)
    np.testing.assert_allclose(flipped_points[flipped_order], points[order], atol=1e-9)
    np.testing.assert_allclose(flipped_normals[flipped_order], -normals[order], atol=1e-12)
    flipped_observable = surface_observability(
        flipped_points, flipped_normals, flipped, bs_pos, ue_pos
    )
    np.testing.assert_array_equal(flipped_observable[flipped_order], observable[order])


def test_specular_support() -> None:
    interaction = np.array([[0.0, 0.0, 0.0]])
    near = np.array([[0.49, 0.0, 0.0]])
    far = np.array([[0.51, 0.0, 0.0]])
    assert bool(specular_support(near, interaction, 0.5)[0])
    assert not bool(specular_support(far, interaction, 0.5)[0])
    assert not bool(np.any(specular_support(near, np.empty((0, 3)), 0.5)))


def test_surface_ground_truth_roi() -> None:
    box = _box()
    triangles = np.asarray(box.vertices, dtype=np.float64)[np.asarray(box.faces)]
    triangle_object = np.zeros(triangles.shape[0], dtype=np.int64)
    bs_pos = np.array([[15.0, 4.0, 12.0]])
    ue_pos = np.array([[12.0, -3.0, 1.5]])
    roi = (np.array([0.0, -1.0, 0.0]), np.array([1.0, 1.0, 3.0]))
    result = surface_ground_truth(
        triangles,
        triangle_object,
        bs_pos,
        ue_pos,
        np.empty((0, 3)),
        spacing=0.25,
        roi=roi,
    )
    full_count = sample_triangles(triangles, 0.25)[0].shape[0]
    assert result["surface_samples"].shape[0] < full_count
    inside = np.all(
        (result["surface_samples"] >= roi[0]) & (result["surface_samples"] <= roi[1]), axis=1
    )
    assert bool(np.all(inside))
    assert result["surface_object"].shape == (result["surface_samples"].shape[0],)
    assert bool(np.all(result["surface_object"] == 0))
    np.testing.assert_array_equal(result["surface_roi"], np.stack(roi))


def test_path_ground_truth_keys_dtypes() -> None:
    scene = build_mirror_scene()
    path = _path(scene)
    geom = scene.geom
    arrays = path_ground_truth(path, geom, pattern="tr38901")
    expected_keys = {
        "path_type",
        "path_power",
        "path_vs",
        "beyond_period",
        "los_visible",
        "ground_bounce_visible",
        "los_phase_model_error",
        "los_amp_model_error_db",
        "vs_pos",
        "vs_bs",
        "vs_order",
        "vs_num_paths",
        "vs_objects",
        "vs_plane_ids",
        "vs_spread",
        "vs_visibility",
        "vs_power",
        "vs_rho_eff",
        "vs_theta_inc",
        "vs_path_type",
        "plane_normal",
        "plane_offset",
        "plane_object",
        "plane_num_vertices",
        "interaction_points",
        "interaction_view",
        "interaction_bs",
        "interaction_path",
        "interaction_depth",
        "interaction_type",
        "interaction_object",
        "interaction_plane",
    }
    assert set(arrays) == expected_keys
    views, bss, paths = geom.num_views, geom.num_bs, path.num_paths
    assert arrays["path_type"].shape == (views, bss, paths)
    assert arrays["path_type"].dtype == np.int8
    assert arrays["path_power"].shape == (views, bss, paths)
    assert arrays["beyond_period"].dtype == bool
    assert arrays["los_visible"].shape == (views, bss)
    assert arrays["vs_pos"].shape[1] == 3
    assert arrays["vs_rho_eff"].dtype == np.complex128
    assert arrays["vs_plane_ids"].shape[1] == path.max_depth
    assert arrays["interaction_points"].shape[1] == 3

    with pytest.raises(ValueError):
        path_ground_truth(path, geom, pattern="unknown")
    with pytest.raises(ValueError):
        path_ground_truth(path, geom, pattern="tr38901", los_polarization="unknown")
