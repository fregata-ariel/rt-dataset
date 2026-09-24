"""Tests for the L0 analytic phantoms (T09)."""

from __future__ import annotations

import dataclasses
import os
import time

import numpy as np
import pytest
from scipy.constants import c as SPEED_OF_LIGHT
from scipy.constants import epsilon_0 as EPSILON_0

from plateau_rt.domain.rf_camera.camera import look_at_orientation
from plateau_rt.domain.rf_tomography import synthetic
from plateau_rt.domain.rf_tomography.forward_exact import atom_cfr
from plateau_rt.domain.rf_tomography.geometry import (
    CaptureGeometry,
    VoxelGrid,
    mirror_point,
    planar_element_offsets,
    rotations_from_orientations,
)
from plateau_rt.domain.rf_tomography.synthetic import (
    default_grid,
    l0_mm,
    l0a_point,
    l0b_pair,
    l0c_random,
    l0d_plate,
    l0e_image_method,
    plane_wave_floor,
    ring_geometry,
)


def _small_geom(
    num_views: int = 4,
    num_bins: int = 16,
    radius: float = 30.0,
    bs_pos: tuple = ((-70.0, 5.0, 25.0),),
) -> CaptureGeometry:
    """Build a small ring geometry for fast tests."""
    return ring_geometry(num_views=num_views, num_bins=num_bins, radius=radius, bs_pos=bs_pos)


def _assert_meta_equal(a: dict, b: dict) -> None:
    """Assert two meta dicts are bitwise identical."""
    assert set(a) == set(b)
    for key in a:
        va, vb = a[key], b[key]
        if isinstance(va, np.ndarray):
            assert isinstance(vb, np.ndarray)
            assert va.shape == vb.shape
            assert np.array_equal(va, vb, equal_nan=True)
        elif isinstance(va, list):
            assert va == vb
        elif isinstance(va, float):
            assert va == vb
        else:
            assert va == vb


def _assert_phantoms_identical(x: synthetic.Phantom, y: synthetic.Phantom) -> None:
    """Assert two phantoms are bitwise identical."""
    assert np.array_equal(x.y_clean, y.y_clean)
    assert np.array_equal(x.gt.points_pos, y.gt.points_pos)
    assert np.array_equal(x.gt.points_rho, y.gt.points_rho)
    _assert_meta_equal(x.gt.meta, y.gt.meta)


def _offgrid_check(points: np.ndarray, grid: VoxelGrid, offset_max: float, spacing: float) -> None:
    """Independently check the off-grid rule for points."""
    origin = np.asarray(grid.origin, dtype=np.float64)
    for p in np.asarray(points, dtype=np.float64).reshape(-1, 3):
        nearest = origin + spacing * np.round((p - origin) / spacing)
        off = np.abs(p - nearest)
        assert np.all(off >= 0.05 * spacing - 1e-12)
        assert np.all(off <= offset_max + 1e-12)
        assert int(grid.index(p.reshape(1, 3))[0]) != -1


def test_reproducible_per_seed() -> None:
    """Same seed gives identical outputs; different seeds move the scene."""
    geom = _small_geom()
    cases = [
        l0a_point(0, geom=geom),
        l0a_point(0, geom=geom),
    ]
    _assert_phantoms_identical(cases[0], cases[1])
    assert not np.array_equal(
        l0a_point(0, geom=geom).gt.points_pos, l0a_point(1, geom=geom).gt.points_pos
    )
    _assert_phantoms_identical(
        l0b_pair(0, 1.0, "cross_range", geom=geom), l0b_pair(0, 1.0, "cross_range", geom=geom)
    )
    assert not np.array_equal(
        l0b_pair(0, 1.0, "cross_range", geom=geom).gt.points_pos,
        l0b_pair(1, 1.0, "cross_range", geom=geom).gt.points_pos,
    )
    _assert_phantoms_identical(l0c_random(0, 16, geom=geom), l0c_random(0, 16, geom=geom))
    assert not np.array_equal(
        l0c_random(0, 16, geom=geom).gt.points_pos, l0c_random(1, 16, geom=geom).gt.points_pos
    )
    _assert_phantoms_identical(l0_mm(0, base="L0a", geom=geom), l0_mm(0, base="L0a", geom=geom))
    _assert_phantoms_identical(
        l0_mm(0, base="L0c", num_points=16, geom=geom),
        l0_mm(0, base="L0c", num_points=16, geom=geom),
    )
    assert not np.array_equal(
        l0_mm(0, base="L0a", geom=geom).gt.points_pos,
        l0_mm(1, base="L0a", geom=geom).gt.points_pos,
    )
    _assert_phantoms_identical(l0d_plate(0, size=1.0, geom=geom), l0d_plate(0, size=1.0, geom=geom))
    _assert_phantoms_identical(l0e_image_method(0, geom=geom), l0e_image_method(0, geom=geom))
    j0 = l0_mm(0, base="L0a", mismatch=("element_jitter",), geom=geom)
    j1 = l0_mm(1, base="L0a", mismatch=("element_jitter",), geom=geom)
    assert not np.array_equal(j0.gt.meta["elem_offsets_true"], j1.gt.meta["elem_offsets_true"])


def test_offgrid_placement() -> None:
    """Every point is off-grid within bounds on both grids."""
    geom = _small_geom()
    grid = default_grid()
    spacing = float(grid.spacing)
    bound = 0.45 * spacing
    ci_grid = VoxelGrid.from_bounds((-4, -4, 1), (4, 4, 5), 2.0)
    assert ci_grid.shape == (5, 5, 3)
    ci_spacing = 2.0
    ci_bound = 0.2
    for seed in range(10):
        for ph in (
            l0a_point(seed, geom=geom, grid=grid),
            l0a_point(seed, voxel=0, geom=geom, grid=grid),
            l0c_random(seed, 64, geom=geom, grid=grid),
        ):
            _offgrid_check(ph.gt.points_pos, grid, bound, spacing)
        for axis in ("range", "cross_range", "vertical"):
            for sep in (0.1, 1.0, 10.0):
                ph = l0b_pair(seed, sep, axis, geom=geom, grid=grid)
                _offgrid_check(ph.gt.points_pos, grid, bound, spacing)
        ph_ci_a = l0a_point(seed, geom=geom, grid=ci_grid, offset_max=ci_bound)
        _offgrid_check(ph_ci_a.gt.points_pos, ci_grid, ci_bound, ci_spacing)
        ph_ci_c = l0c_random(seed, 10, geom=geom, grid=ci_grid, offset_max=ci_bound)
        _offgrid_check(ph_ci_c.gt.points_pos, ci_grid, ci_bound, ci_spacing)
        ph_ci_b = l0b_pair(seed, 2.0, "cross_range", geom=geom, grid=ci_grid, offset_max=ci_bound)
        _offgrid_check(ph_ci_b.gt.points_pos, ci_grid, ci_bound, ci_spacing)
        with pytest.raises(ValueError):
            l0b_pair(seed, 1.0, "cross_range", geom=geom, grid=ci_grid, offset_max=ci_bound)


def test_offgrid_points_validation() -> None:
    """Invalid placement arguments raise ValueError."""
    grid = default_grid()
    spacing = float(grid.spacing)
    rng = np.random.default_rng(np.random.SeedSequence([0, 0]))
    with pytest.raises(ValueError):
        synthetic.offgrid_points(grid, 1, rng, offset_max=0.05 * spacing)
    with pytest.raises(ValueError):
        synthetic.offgrid_points(grid, 1, rng, offset_max=0.6 * spacing)
    with pytest.raises(ValueError):
        synthetic.offgrid_points(grid, grid.size + 1, rng)
    with pytest.raises(ValueError):
        synthetic.offgrid_points(grid, 0, rng)
    with pytest.raises(ValueError):
        synthetic.offgrid_points(grid, 1, rng, voxels=np.array([grid.size]))
    with pytest.raises(ValueError):
        synthetic.offgrid_points(grid, 1, rng, voxels=np.array([-1]))
    points, drawn = synthetic.offgrid_points(grid, 5, rng)
    assert len(set(int(v) for v in drawn)) == 5
    assert np.array_equal(grid.index(points), drawn)


def test_l0a_matches_reference() -> None:
    """L0a render equals the exact operator and fills the front hemisphere."""
    geom = _small_geom()
    ph = l0a_point(0, geom=geom)
    assert ph.y_clean.shape == (4, 1, 2, 8, 8, 16)
    assert ph.y_clean.dtype == np.complex128
    ref = atom_cfr(ph.gt.points_pos, ph.gt.points_rho, geom, "bv", pattern="tr38901")
    denom = float(np.max(np.abs(ref)))
    assert denom > 0.0
    assert float(np.max(np.abs(ph.y_clean - ref))) / denom <= 1e-12
    assert np.all(ph.y_clean[:, :, 1] == 0.0)
    assert ph.gt.level == "L0a"
    assert ph.gt.space == "bv"
    assert ph.gt.mismatch == ()
    assert ph.gt.points_rho.dtype == np.complex128


def _independent_dirs(
    grid: VoxelGrid, geom: CaptureGeometry, ref_view: int = 0
) -> dict[str, np.ndarray]:
    """Compute L0b directions independently from UE position and grid centre."""
    centre = (
        np.asarray(grid.origin, float)
        + float(grid.spacing) * (np.asarray(grid.shape, float) - 1.0) / 2.0
    )
    rng = (centre - np.asarray(geom.ue_pos[ref_view], float)) / np.linalg.norm(
        centre - np.asarray(geom.ue_pos[ref_view], float)
    )
    cr_raw = np.cross(np.array([0.0, 0.0, 1.0]), rng)
    cr = cr_raw / np.linalg.norm(cr_raw)
    vert = np.cross(rng, cr)
    vert = vert / np.linalg.norm(vert)
    return {"range": rng, "cross_range": cr, "vertical": vert}


def test_l0b_geometry() -> None:
    """Pair separation and direction match the independent frame."""
    geom = _small_geom()
    grid = default_grid()
    dirs = _independent_dirs(grid, geom)
    assert abs(float(np.dot(dirs["range"], dirs["cross_range"]))) < 1e-12
    assert abs(float(np.dot(dirs["range"], dirs["vertical"]))) < 1e-12
    assert abs(float(np.dot(dirs["cross_range"], dirs["vertical"]))) < 1e-12
    assert abs(float(dirs["cross_range"][2])) < 1e-12
    assert float(dirs["vertical"][2]) > 0.0
    for axis in ("range", "cross_range", "vertical"):
        for sep in (0.1, 1.0, 10.0):
            ph = l0b_pair(3, sep, axis, geom=geom, grid=grid)
            p1, p2 = ph.gt.points_pos[0], ph.gt.points_pos[1]
            assert abs(float(np.linalg.norm(p2 - p1)) - sep) <= 1e-12
            assert np.allclose((p2 - p1) / sep, dirs[axis], atol=1e-12)
            assert np.allclose(ph.gt.meta["axis_dir"], dirs[axis], atol=1e-12)
    with pytest.raises(ValueError):
        l0b_pair(0, 1.0, "diagonal", geom=geom, grid=grid)
    with pytest.raises(ValueError):
        l0b_pair(0, 0.0, "range", geom=geom, grid=grid)
    with pytest.raises(ValueError):
        l0b_pair(0, -1.0, "range", geom=geom, grid=grid)


def test_l0c_amplitudes_and_box() -> None:
    """Amplitudes span the dynamic range and points stay in the box."""
    geom = _small_geom()
    grid = default_grid()
    centre = (
        np.asarray(grid.origin, float)
        + float(grid.spacing) * (np.asarray(grid.shape, float) - 1.0) / 2.0
    )
    for count in (4, 16, 64):
        ph = l0c_random(0, count, geom=geom, grid=grid)
        assert ph.gt.points_rho.shape == (count,)
        db = 20.0 * np.log10(np.abs(ph.gt.points_rho))
        assert np.all(db <= 0.0 + 1e-9)
        assert np.all(db >= -30.0 - 1e-9)
        assert np.all(np.max(np.abs(ph.gt.points_pos - centre), axis=1) <= 10.0 / 2.0 + 1e-9)
        assert len(set(int(v) for v in ph.gt.meta["voxels"])) == count
        ref = atom_cfr(ph.gt.points_pos, ph.gt.points_rho, geom, "bv", pattern="tr38901")
        denom = float(np.max(np.abs(ref)))
        assert float(np.max(np.abs(ph.y_clean - ref))) / denom <= 1e-12
    ph64 = l0c_random(0, 64, geom=geom, grid=grid)
    spread = 20.0 * np.log10(
        float(np.max(np.abs(ph64.gt.points_rho))) / float(np.min(np.abs(ph64.gt.points_rho)))
    )
    assert spread > 20.0


def test_l0d_lambda_quarter_sampling() -> None:
    """Plate sampling is lambda/4 on the plane with the BS-facing normal."""
    geom = ring_geometry(num_views=2, num_bins=16)
    lam = SPEED_OF_LIGHT / float(geom.f_c)
    step = lam / 4.0
    ph = l0d_plate(0, geom=geom)
    assert ph.gt.meta["n_side"] == 187
    assert ph.gt.points_pos.shape == (187**2, 3)
    centre = np.asarray(ph.gt.meta["center"], float)
    normal = np.asarray(ph.gt.meta["normal"], float)
    axes = np.asarray(ph.gt.meta["axes"], float)
    assert abs(float(normal[2])) < 1e-12
    assert abs(float(np.linalg.norm(normal)) - 1.0) < 1e-12
    to_bs = np.asarray(geom.bs_pos[0], float) - centre
    to_bs[2] = 0.0
    to_bs = to_bs / np.linalg.norm(to_bs)
    assert float(normal @ to_bs) > 0.999
    dist = np.abs((ph.gt.points_pos - centre) @ normal)
    assert float(np.max(dist)) <= 1e-12
    proj1 = (ph.gt.points_pos - centre) @ axes[0]
    proj2 = (ph.gt.points_pos - centre) @ axes[1]
    u1 = np.unique(np.round(proj1 / step) * step)
    u2 = np.unique(np.round(proj2 / step) * step)
    assert u1.size == 187
    assert u2.size == 187
    assert np.allclose(np.diff(np.sort(u1)), step, atol=1e-12)
    assert np.allclose(np.diff(np.sort(u2)), step, atol=1e-12)
    extent = (187 - 1) * step
    assert extent <= 4.0 + 1e-12
    assert extent > 4.0 - step
    expected_vs = mirror_point(np.asarray(geom.bs_pos, float), centre, normal)
    assert np.allclose(ph.gt.meta["vs_pos"], expected_vs, atol=0.0)


def _hand_geometry(ue_pos: np.ndarray, bs_pos: np.ndarray, num_bins: int = 16) -> CaptureGeometry:
    """Build a one-view one-BS hand geometry with exact float grids."""
    f_c = 3.5e9
    lam = SPEED_OF_LIGHT / f_c
    orientations = np.asarray(
        [look_at_orientation(tuple(ue_pos), (0.0, 0.0, 0.0))], dtype=np.float64
    )
    freq = (np.arange(num_bins, dtype=np.float64) - num_bins // 2) * (100e6 / num_bins)
    return CaptureGeometry(
        ue_pos=np.asarray(ue_pos, float).reshape(1, 3),
        ue_rot=rotations_from_orientations(orientations),
        bs_pos=np.asarray(bs_pos, float).reshape(1, 3),
        elem_offsets=planar_element_offsets(lam),
        freq_offsets=freq,
        f_c=f_c,
        aperture_shape=(8, 8),
        bs_rot=None,
    )


def test_l0d_born_plate_matches_image_source() -> None:
    """Born density reproduces the image source up to 1/cos(theta)."""
    normal = np.array([1.0, 0.0, 0.0])
    centre = np.array([0.0, 0.0, 0.0])
    cases = [
        (np.array([10.0, 0.0, 0.3]), np.array([10.0, 0.0, -0.3])),
        (
            np.array([10.0 * np.cos(np.deg2rad(20.0)), -10.0 * np.sin(np.deg2rad(20.0)), 0.0]),
            np.array([10.0 * np.cos(np.deg2rad(20.0)), 10.0 * np.sin(np.deg2rad(20.0)), 0.0]),
        ),
    ]
    for bs, ue in cases:
        geom = _hand_geometry(ue, bs)
        ph = l0d_plate(0, center=centre, normal=normal, size=4.0, geom=geom, pattern="iso")
        image = mirror_point(np.asarray(bs, float).reshape(1, 3), centre, normal)[0]
        y_vs = atom_cfr(
            image.reshape(1, 3), np.array([1.0], dtype=np.complex128), geom, "vs", pattern="iso"
        )
        ratio = complex(np.vdot(y_vs, ph.y_clean) / np.vdot(y_vs, y_vs))
        diff = np.asarray(ue, float) - np.asarray(image, float)
        cos_inc = abs(float(diff @ np.array([1.0, 0.0, 0.0]))) / float(np.linalg.norm(diff))
        assert abs(abs(ratio) * cos_inc - 1.0) < 0.1
        assert abs(float(np.angle(ratio))) < 0.1


def test_fresnel_coefficients() -> None:
    """Fresnel values match the lossless reference and the ITU table."""
    r_te, r_tm = synthetic.fresnel_coefficients(1.0, 4.0 + 0.0j)
    assert abs(complex(np.asarray(r_te).reshape(-1)[0]) + 1.0 / 3.0) < 1e-12
    assert abs(complex(np.asarray(r_tm).reshape(-1)[0]) - 1.0 / 3.0) < 1e-12
    r_te_g, r_tm_g = synthetic.fresnel_coefficients(0.0, 4.0 + 0.0j)
    assert abs(complex(np.asarray(r_te_g).reshape(-1)[0]) + 1.0) < 1e-12
    assert abs(complex(np.asarray(r_tm_g).reshape(-1)[0]) + 1.0) < 1e-12
    cos_b = float(np.cos(np.arctan(2.0)))
    _, r_tm_b = synthetic.fresnel_coefficients(cos_b, 4.0 + 0.0j)
    assert abs(complex(np.asarray(r_tm_b).reshape(-1)[0])) < 1e-12
    _, r_tm_below = synthetic.fresnel_coefficients(float(np.cos(np.deg2rad(50.0))), 4.0 + 0.0j)
    _, r_tm_above = synthetic.fresnel_coefficients(float(np.cos(np.deg2rad(75.0))), 4.0 + 0.0j)
    assert float(np.asarray(r_tm_below).real.reshape(-1)[0]) > 0.0
    assert float(np.asarray(r_tm_above).real.reshape(-1)[0]) < 0.0
    f = 3.5e9
    f_ghz = f / 1e9
    eta_conc = 5.24 * f_ghz**0.0 - 1j * (0.0462 * f_ghz**0.7822) / (2 * np.pi * f * EPSILON_0)
    eta_med = 15.0 * f_ghz**-0.1 - 1j * (0.035 * f_ghz**1.63) / (2 * np.pi * f * EPSILON_0)
    got_conc = synthetic.itu_permittivity("concrete", f)
    got_med = synthetic.itu_permittivity("medium_dry_ground", f)
    assert abs(got_conc - eta_conc) / abs(eta_conc) < 1e-12
    assert abs(got_med - eta_med) / abs(eta_med) < 1e-12
    with pytest.raises(ValueError):
        synthetic.itu_permittivity("unobtainium", f)
    with pytest.raises(ValueError):
        synthetic.itu_permittivity("medium_dry_ground", 20e9)
    with pytest.raises(ValueError):
        synthetic.fresnel_coefficients(1.5, 4.0 + 0.0j)


def test_l0e_two_ray_reference() -> None:
    """Two-ray ground phantom matches an explicit element loop."""
    f_c = 3.5e9
    lam = SPEED_OF_LIGHT / f_c
    num_bins = 16
    ue = np.array([30.0, 0.0, 1.5])
    bs = np.array([-70.0, 5.0, 25.0])
    orientations = np.asarray([look_at_orientation(tuple(ue), (0.0, 0.0, 5.0))], float)
    freq = (np.arange(num_bins, dtype=float) - num_bins // 2) * (100e6 / num_bins)
    geom = CaptureGeometry(
        ue_pos=ue.reshape(1, 3),
        ue_rot=rotations_from_orientations(orientations),
        bs_pos=bs.reshape(1, 3),
        elem_offsets=planar_element_offsets(lam),
        freq_offsets=freq,
        f_c=f_c,
        aperture_shape=(8, 8),
        bs_rot=None,
    )
    ph = l0e_image_method(
        0,
        ground_material="medium_dry_ground",
        ground_height=0.0,
        walls=(),
        geom=geom,
        pattern="iso",
    )
    assert ph.gt.points_pos.shape[0] == 2
    f_ghz = f_c / 1e9
    eta = 15.0 * f_ghz**-0.1 - 1j * (0.035 * f_ghz**1.63) / (2 * np.pi * f_c * EPSILON_0)
    image = np.array([bs[0], bs[1], -bs[2]])
    dist = float(np.linalg.norm(ue - image))
    cos_t = float((ue[2] + bs[2]) / dist)
    root = np.sqrt(eta - (1.0 - cos_t**2))
    beta_img = (eta * cos_t - root) / (eta * cos_t + root)
    sources = [(np.asarray(bs, float), 1.0 + 0.0j), (image, complex(beta_img))]
    ref = np.zeros((1, 1, 2, 8, 8, num_bins), dtype=np.complex128)
    rot = np.asarray(geom.ue_rot[0], float)
    offsets = np.asarray(geom.elem_offsets, float)
    for src, beta in sources:
        r = float(np.linalg.norm(ue - src))
        u_local = (np.asarray(src, float) - ue) / r @ rot
        h = 0 if u_local[0] >= 0.0 else 1
        for m in range(64):
            row, col = divmod(m, 8)
            el_phase = np.exp(1j * (2 * np.pi / lam) * float(u_local @ offsets[m]))
            for n in range(num_bins):
                ref[0, 0, h, row, col, n] += (
                    beta
                    * lam
                    / (4 * np.pi * r)
                    * np.exp(-1j * (2 * np.pi / lam) * r)
                    * el_phase
                    * np.exp(-2j * np.pi * float(freq[n]) * r / SPEED_OF_LIGHT)
                )
    denom = float(np.max(np.abs(ref)))
    assert float(np.max(np.abs(ph.y_clean - ref))) <= 1e-10 * denom


def test_l0e_structure() -> None:
    """VS list ordering, visibility and Fresnel amplitudes are exact."""
    geom = ring_geometry(num_views=4, num_bins=16, bs_pos=((-70.0, 5.0, 25.0), (-20.0, 60.0, 20.0)))
    ph = l0e_image_method(0, geom=geom)
    assert ph.gt.points_pos.shape[0] == 8
    assert np.array_equal(ph.gt.meta["vs_bs"], np.array([0, 0, 0, 0, 1, 1, 1, 1]))
    assert ph.gt.meta["vs_plane"][0] == -1
    assert ph.gt.meta["vs_plane"][4] == -1
    for k in range(8):
        b = int(ph.gt.meta["vs_bs"][k])
        for bb in range(2):
            if bb != b:
                assert np.all(ph.gt.points_rho[k, :, bb] == 0.0)
    assert np.all(ph.gt.points_rho[0, :, 0] == 1.0)
    assert np.all(ph.gt.points_rho[4, :, 1] == 1.0)
    plane_points = np.asarray(ph.gt.meta["plane_points"], float)
    plane_normals = np.asarray(ph.gt.meta["plane_normals"], float)
    materials = list(ph.gt.meta["plane_materials"])
    assert plane_points.shape[0] == 3
    for k in range(8):
        b = int(ph.gt.meta["vs_bs"][k])
        j = int(ph.gt.meta["vs_plane"][k])
        if j == -1:
            assert np.allclose(ph.gt.points_pos[k], geom.bs_pos[b])
        else:
            expected = mirror_point(
                geom.bs_pos[b].reshape(1, 3), plane_points[j], plane_normals[j]
            )[0]
            assert np.allclose(ph.gt.points_pos[k], expected, atol=0.0)
    assert bool(np.all(ph.gt.meta["vs_visibility"]))
    f = float(geom.f_c)
    f_ghz = f / 1e9
    table = {
        "concrete": (5.24, 0.0, 0.0462, 0.7822),
        "medium_dry_ground": (15.0, -0.1, 0.035, 1.63),
    }
    for k in range(8):
        j = int(ph.gt.meta["vs_plane"][k])
        if j == -1:
            continue
        b = int(ph.gt.meta["vs_bs"][k])
        a, bb, cc, dd = table[materials[j]]
        eta = a * f_ghz**bb - 1j * (cc * f_ghz**dd) / (2 * np.pi * f * EPSILON_0)
        for v in range(geom.num_views):
            src = ph.gt.points_pos[k]
            ue = np.asarray(geom.ue_pos[v], float)
            diff = ue - np.asarray(src, float)
            cos_t = abs(float(diff @ plane_normals[j])) / float(np.linalg.norm(diff))
            root = np.sqrt(eta - (1.0 - cos_t**2))
            if j == 0:
                beta = (eta * cos_t - root) / (eta * cos_t + root)
            else:
                beta = (cos_t - root) / (cos_t + root)
            assert abs(complex(ph.gt.points_rho[k, v, b]) - complex(beta)) < 1e-12
    with pytest.raises(ValueError):
        l0e_image_method(0, walls=[((0, 0, 0), (0, 0, 1), "concrete")], geom=geom)
    ref = atom_cfr(ph.gt.points_pos, ph.gt.points_rho, geom, "vs", pattern=ph.gt.pattern)
    denom = float(np.max(np.abs(ref)))
    assert float(np.max(np.abs(ph.y_clean - ref))) / denom <= 1e-12


def test_l0mm_floor_matches_prediction() -> None:
    """Measured plane-wave floor matches the analytic prediction within 3 dB."""
    for radius in (20, 40):
        geom = ring_geometry(num_views=4, num_bins=32, radius=radius)
        for mismatch in (("spherical",), ("element_jitter",), synthetic.MISMATCHES):
            ph = l0_mm(1, base="L0a", mismatch=mismatch, geom=geom)
            measured = plane_wave_floor(ph)
            pred_db = float(ph.gt.meta["nmse_floor_pred_db"])
            assert abs(10.0 * np.log10(measured) - pred_db) <= 3.0
    geom = ring_geometry(num_views=4, num_bins=32, radius=30.0)
    ph_sq = l0_mm(2, base="L0a", mismatch=("squint",), voxel=0, geom=geom)
    assert (
        abs(10.0 * np.log10(plane_wave_floor(ph_sq)) - float(ph_sq.gt.meta["nmse_floor_pred_db"]))
        <= 3.0
    )
    ph_c = l0_mm(0, base="L0c", num_points=16, mismatch=synthetic.MISMATCHES, geom=geom)
    assert (
        abs(10.0 * np.log10(plane_wave_floor(ph_c)) - float(ph_c.gt.meta["nmse_floor_pred_db"]))
        <= 3.0
    )
    geom_b = ring_geometry(num_views=4, num_bins=32, radius=30.0)
    ph_b = l0_mm(0, base="L0a", mismatch=("spherical",), geom=geom_b)
    measured_b = plane_wave_floor(ph_b)
    offsets = np.asarray(geom_b.elem_offsets, float)
    k = 2 * np.pi / float(geom_b.wavelength)
    r = float(np.linalg.norm(ph_b.gt.points_pos[0] - np.asarray(geom_b.ue_pos[0], float)))
    u = np.array([1.0, 0.0, 0.0])
    eps_m = -k * (np.sum(offsets**2, axis=1) - (offsets @ u) ** 2) / (2.0 * r)
    closed = float(np.var(eps_m))
    assert abs(10.0 * np.log10(measured_b) - 10.0 * np.log10(closed)) <= 3.0


def test_l0mm_expected_range() -> None:
    """Mismatch floors sit at -36..-24 dB while plane-wave data fits exactly."""
    for radius in (20, 40):
        geom = ring_geometry(num_views=4, num_bins=32, radius=radius)
        ph = l0_mm(0, base="L0a", mismatch=synthetic.MISMATCHES, geom=geom)
        floor_db = 10.0 * np.log10(plane_wave_floor(ph))
        assert -36.0 <= floor_db <= -24.0
        plain = l0a_point(0, geom=geom)
        assert plane_wave_floor(plain) < 1e-20
        rel = float(
            np.linalg.norm((ph.y_clean - plain.y_clean).ravel())
            / np.linalg.norm(plain.y_clean.ravel())
        )
        assert rel > 1e-3


def test_l0mm_shares_scene_and_nominal_geometry() -> None:
    """L0-mm shares the scene but renders with the jittered array."""
    geom = _small_geom(num_views=4, num_bins=32)
    lam = float(geom.wavelength)
    a_mm = l0_mm(5, base="L0a", geom=geom)
    a_plain = l0a_point(5, geom=geom)
    assert np.array_equal(a_mm.gt.points_pos, a_plain.gt.points_pos)
    assert np.array_equal(a_mm.gt.points_rho, a_plain.gt.points_rho)
    c_mm = l0_mm(5, base="L0c", num_points=16, geom=geom)
    c_plain = l0c_random(5, 16, geom=geom)
    assert np.array_equal(c_mm.gt.points_pos, c_plain.gt.points_pos)
    assert np.array_equal(c_mm.gt.points_rho, c_plain.gt.points_rho)
    expected_nominal = planar_element_offsets(lam)
    assert np.array_equal(np.asarray(a_mm.geom.elem_offsets, float), expected_nominal)
    diff = np.asarray(a_mm.gt.meta["elem_offsets_true"], float) - expected_nominal
    est = float(np.std(diff))
    assert abs(est - lam / 200.0) / (lam / 200.0) <= 0.3
    geom_true = dataclasses.replace(
        geom, elem_offsets=np.asarray(a_mm.gt.meta["elem_offsets_true"], float)
    )
    ref = atom_cfr(
        a_mm.gt.points_pos,
        a_mm.gt.points_rho,
        geom_true,
        "bv",
        wavefront="spherical",
        squint=True,
        pattern=a_mm.gt.pattern,
    )
    denom = float(np.max(np.abs(ref)))
    assert float(np.max(np.abs(a_mm.y_clean - ref))) / denom <= 1e-12
    assert a_mm.gt.level == "L0-mm"
    assert a_mm.gt.meta["base"] == "L0a"
    with pytest.raises(ValueError):
        l0_mm(0, base="L0a", mismatch=("unknown",), geom=geom)
    with pytest.raises(ValueError):
        l0_mm(0, base="L0a", mismatch=("spherical", "spherical"), geom=geom)
    with pytest.raises(ValueError):
        l0_mm(0, base="L0a", mismatch=(), geom=geom)
    with pytest.raises(ValueError):
        l0_mm(0, base="L0b", geom=geom)
    with pytest.raises(ValueError):
        l0_mm(0, base="L0a", mismatch=("element_jitter",), jitter_std=-1.0, geom=geom)


def test_generate_dispatch() -> None:
    """Level dispatcher maps to the generators."""
    geom = _small_geom()
    assert np.array_equal(
        synthetic.generate("L0a", 3, geom=geom).y_clean, l0a_point(3, geom=geom).y_clean
    )
    assert np.array_equal(
        synthetic.generate("L0-mm", 3, geom=geom).y_clean, l0_mm(3, geom=geom).y_clean
    )
    with pytest.raises(ValueError):
        synthetic.generate("L0x", 0, geom=geom)
    with pytest.raises(ValueError):
        synthetic.generate("L0a", -1, geom=geom)


def test_benchmark_generators() -> None:
    """Benchmark timings (only with RF_TOMO_BENCH=1)."""
    if os.environ.get("RF_TOMO_BENCH") != "1":
        pytest.skip("benchmark only with RF_TOMO_BENCH=1")
    geom = ring_geometry()
    start = time.perf_counter()
    l0c_random(0, 64, geom=geom)
    t_c = time.perf_counter() - start
    start = time.perf_counter()
    ph = l0_mm(0, base="L0c", num_points=64, geom=geom)
    plane_wave_floor(ph)
    t_mm = time.perf_counter() - start
    start = time.perf_counter()
    l0d_plate(geom=geom)
    t_d = time.perf_counter() - start
    print(f"l0c64={t_c:.3f}s l0mm64+floor={t_mm:.3f}s l0d={t_d:.3f}s")
