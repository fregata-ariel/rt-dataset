"""Tests for the exact element-level reference forward operator (T03)."""

from __future__ import annotations

import dataclasses
import math
import os
import time
from pathlib import Path

import numpy as np
import pytest
from scipy.constants import c as SPEED_OF_LIGHT

from plateau_rt.domain.rf_tomography import forward_exact
from plateau_rt.domain.rf_tomography.antenna import bs_orientation, tr38901_gain
from plateau_rt.domain.rf_tomography.forward_exact import (
    atom_cfr,
    capture_factors,
    dense_matrix,
)
from plateau_rt.domain.rf_tomography.geometry import (
    CaptureGeometry,
    hemisphere_index,
    mirror_point,
    planar_element_offsets,
    rotations_from_orientations,
)

FIXTURES = Path(__file__).parent / "fixtures" / "rf_tomography"
F_C = 3.5e9
BANDWIDTH = 100e6


def _make_geometry(
    ue_pos: np.ndarray,
    orientations: np.ndarray,
    bs_pos: np.ndarray,
    *,
    rows: int = 8,
    cols: int = 8,
    num_bins: int = 16,
    bs_look_at: np.ndarray | None = None,
) -> CaptureGeometry:
    """Build an exact-float64-grid capture geometry for the synthetic tests."""
    wavelength = SPEED_OF_LIGHT / F_C
    elem_offsets = planar_element_offsets(wavelength, rows=rows, cols=cols)
    df = BANDWIDTH / num_bins
    freq_offsets = (np.arange(num_bins) - num_bins // 2) * df
    bs_rot = None
    if bs_look_at is not None:
        targets = np.asarray(bs_look_at, dtype=np.float64)
        if targets.shape == (3,):
            targets = np.broadcast_to(targets, (bs_pos.shape[0], 3))
        bs_rot = np.stack([bs_orientation(bs_pos[b], targets[b]) for b in range(bs_pos.shape[0])])
    return CaptureGeometry(
        ue_pos=ue_pos,
        ue_rot=rotations_from_orientations(orientations),
        bs_pos=bs_pos,
        elem_offsets=elem_offsets,
        freq_offsets=freq_offsets,
        f_c=F_C,
        aperture_shape=(rows, cols),
        bs_rot=bs_rot,
    )


def _synthetic_geometry(
    *,
    rows: int = 8,
    cols: int = 8,
    num_bins: int = 16,
    look_at: bool = True,
    num_views: int = 4,
    num_bs: int = 2,
) -> CaptureGeometry:
    """Views with distinct yaw/pitch/roll and BSs (the last view faces away)."""
    ue_pos = np.array([[0.0, 0.0, 1.5], [5.0, -3.0, 1.6], [-4.0, 2.0, 1.4], [0.0, 0.0, 1.5]])[
        :num_views
    ]
    orientations = np.array(
        [[0.0, 0.0, 0.0], [0.6, 0.1, 0.2], [-1.0, -0.2, 0.3], [math.pi, 0.0, 0.0]]
    )[:num_views]
    bs_pos = np.array([[10.0, 0.0, 8.0], [-12.0, 4.0, 6.0]])[:num_bs]
    targets = np.array([[0.0, 0.0, 1.5], [2.0, 1.0, 1.0]])[:num_bs] if look_at else None
    return _make_geometry(
        ue_pos,
        orientations,
        bs_pos,
        rows=rows,
        cols=cols,
        num_bins=num_bins,
        bs_look_at=targets,
    )


def _los_reference(beta: np.ndarray, geom: CaptureGeometry, pattern: str) -> np.ndarray:
    """Independent scalar VS-at-BS reference (explicit loops over element and bin)."""
    views, bss = geom.num_views, geom.num_bs
    rows, cols = geom.aperture_shape
    bins = geom.num_bins
    lam, k = geom.wavelength, geom.wavenumber
    out = np.zeros((views, bss, 2, rows, cols, bins), dtype=np.complex128)
    for v in range(views):
        position = geom.ue_pos[v]
        rotation = geom.ue_rot[v]
        for b in range(bss):
            target = geom.bs_pos[b]
            offset = target - position
            distance = float(np.linalg.norm(offset))
            u_loc = rotation.T @ (offset / distance)
            h = 0 if u_loc[0] >= 0.0 else 1
            if pattern == "tr38901":
                assert geom.bs_rot is not None
                local = geom.bs_rot[b].T @ ((position - target) / distance)
                theta = math.acos(min(max(float(local[2]), -1.0), 1.0))
                phi = math.atan2(float(local[1]), float(local[0]))
                field = math.sqrt(float(tr38901_gain(theta, phi)))
            else:
                field = 1.0
            carrier = field * lam / (4.0 * math.pi * distance) * np.exp(-1j * k * distance)
            for row in range(rows):
                for col in range(cols):
                    q = geom.elem_offsets[row * cols + col]
                    phase = np.exp(1j * k * float(np.dot(u_loc, q)))
                    for n in range(bins):
                        delay = np.exp(
                            -1j
                            * 2.0
                            * math.pi
                            * float(geom.freq_offsets[n])
                            * distance
                            / SPEED_OF_LIGHT
                        )
                        out[v, b, h, row, col, n] = beta[b] * carrier * phase * delay
    return out


def _bv_reference(
    beta: complex, point: np.ndarray, geom: CaptureGeometry, pattern: str
) -> np.ndarray:
    """Independent scalar BV reference (explicit loops over element and bin)."""
    views, bss = geom.num_views, geom.num_bs
    rows, cols = geom.aperture_shape
    bins = geom.num_bins
    lam, k = geom.wavelength, geom.wavenumber
    x = np.asarray(point, dtype=np.float64)
    out = np.zeros((views, bss, 2, rows, cols, bins), dtype=np.complex128)
    for v in range(views):
        position = geom.ue_pos[v]
        rotation = geom.ue_rot[v]
        for b in range(bss):
            target = geom.bs_pos[b]
            r1 = float(np.linalg.norm(x - target))
            r2 = float(np.linalg.norm(x - position))
            u_loc = rotation.T @ ((x - position) / r2)
            h = 0 if u_loc[0] >= 0.0 else 1
            if pattern == "tr38901":
                assert geom.bs_rot is not None
                local = geom.bs_rot[b].T @ ((x - target) / r1)
                theta = math.acos(min(max(float(local[2]), -1.0), 1.0))
                phi = math.atan2(float(local[1]), float(local[0]))
                field = math.sqrt(float(tr38901_gain(theta, phi)))
            else:
                field = 1.0
            carrier = field * lam / ((4.0 * math.pi) ** 1.5 * r1 * r2) * np.exp(-1j * k * (r1 + r2))
            for row in range(rows):
                for col in range(cols):
                    q = geom.elem_offsets[row * cols + col]
                    phase = np.exp(1j * k * float(np.dot(u_loc, q)))
                    for n in range(bins):
                        delay = np.exp(
                            -1j
                            * 2.0
                            * math.pi
                            * float(geom.freq_offsets[n])
                            * (r1 + r2)
                            / SPEED_OF_LIGHT
                        )
                        out[v, b, h, row, col, n] = beta * carrier * phase * delay
    return out


def _max_relative_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def test_matches_sionna_mock_los() -> None:
    fixture = np.load(FIXTURES / "sionna_mock_los.npz")
    geom = CaptureGeometry(
        fixture["ue_pos"],
        rotations_from_orientations(fixture["ue_orientation"]),
        fixture["bs_pos"],
        planar_element_offsets(SPEED_OF_LIGHT / float(fixture["f_c"])),
        fixture["freq_offsets"].astype(np.float64),
        float(fixture["f_c"]),
        bs_rot=bs_orientation(fixture["bs_pos"][0], fixture["bs_look_at"])[None],
    )

    for v in range(2):
        reference = fixture["aperture_cfr"][v].astype(np.complex128)
        model = atom_cfr(
            fixture["bs_pos"], [1.0], geom, "vs", pattern="tr38901", polarization="vv"
        )[v, 0]
        assert _max_relative_error(model, reference) <= 1e-3

        received = int(np.argmax(np.sum(np.abs(reference) ** 2, axis=(1, 2, 3))))
        assert np.all(model[1 - received] == 0.0)
        assert np.all(reference[1 - received] == 0.0)

        beta = np.vdot(model, reference) / np.vdot(model, model)
        assert abs(beta - 1.0) <= 1e-3
        assert _max_relative_error(beta * model, reference) <= 1e-4

        raw_none = atom_cfr(
            fixture["bs_pos"], [1.0], geom, "vs", pattern="tr38901", polarization="none"
        )[v, 0]
        assert _max_relative_error(raw_none, reference) >= 5e-3

        np.testing.assert_allclose(
            geom.vs_delay(fixture["bs_pos"], v)[0], fixture["path_tau"][v], atol=1e-12
        )


def test_matches_sionna_t01_fixture() -> None:
    fixture = np.load(FIXTURES / "sionna_los_aperture.npz")
    geom = CaptureGeometry.from_orientations(
        fixture["ue_pos"],
        fixture["ue_orientation"],
        fixture["bs_pos"][None],
        f_c=float(fixture["f_c"]),
        bandwidth=100e6,
        num_bins=8,
    )
    np.testing.assert_array_equal(geom.freq_offsets, fixture["freq_offsets"])
    geom = dataclasses.replace(geom, bs_rot=np.eye(3)[None])

    for v in range(2):
        reference = fixture["aperture_cfr"][v].astype(np.complex128)
        model = atom_cfr(
            fixture["bs_pos"][None], [1.0], geom, "vs", pattern="iso", polarization="vv"
        )[v, 0]
        assert _max_relative_error(model, reference) <= 1e-3

        raw_none = atom_cfr(
            fixture["bs_pos"][None], [1.0], geom, "vs", pattern="iso", polarization="none"
        )[v, 0]
        assert _max_relative_error(raw_none, reference) >= 0.1


def test_vs_at_bs_equals_analytic_los() -> None:
    geom = _synthetic_geometry(num_bs=1)
    rng = np.random.default_rng(7)
    beta = rng.normal(size=geom.num_bs) + 1j * rng.normal(size=geom.num_bs)
    for pattern in ("tr38901", "iso"):
        model = atom_cfr(geom.bs_pos, beta, geom, "vs", pattern=pattern, polarization="none")
        reference = _los_reference(beta, geom, pattern)
        assert _max_relative_error(model, reference) <= 1e-12

    local_x = np.array(
        [
            geom.local_direction(geom.bs_pos[b], v)[0, 0]
            for b in range(geom.num_bs)
            for v in range(geom.num_views)
        ]
    )
    assert np.any(local_x < 0.0)


def test_bv_point_equals_analytic_bistatic() -> None:
    geom = _synthetic_geometry()
    point = np.array([3.0, -2.0, 4.0])
    beta = 0.7 - 0.3j
    for pattern in ("tr38901", "iso"):
        model = atom_cfr(point, [beta], geom, "bv", pattern=pattern, polarization="none")
        reference = _bv_reference(beta, point, geom, pattern)
        assert _max_relative_error(model, reference) <= 1e-12


def test_hemisphere_mask() -> None:
    geom = _synthetic_geometry(look_at=False, num_views=1, num_bs=1)
    front = np.array([[6.0, 1.0, 1.0], [8.0, -2.0, 2.0]])
    behind = np.array([[-7.0, 0.5, 1.0]])
    in_plane = np.array([[0.0, 5.0, 0.0]])
    points = np.vstack([front, behind, in_plane])
    amps = np.ones(points.shape[0])

    total = atom_cfr(points, amps, geom, "vs", pattern="iso", polarization="none")
    assert np.max(np.abs(total[..., 1, :, :, :])) > 0.0
    # The single view has identity rotation, so local x == world x.
    for p_front in front:
        single = atom_cfr(p_front, [1.0], geom, "vs", pattern="iso", polarization="none")
        assert np.all(single[..., 1, :, :, :] == 0.0)
    for p_behind in behind:
        single = atom_cfr(p_behind, [1.0], geom, "vs", pattern="iso", polarization="none")
        assert np.all(single[..., 0, :, :, :] == 0.0)
    plane_single = atom_cfr(in_plane, [1.0], geom, "vs", pattern="iso", polarization="none")
    assert np.all(plane_single[..., 1, :, :, :] == 0.0)

    front_sum = np.zeros_like(total)
    for p in (*front, in_plane[0]):
        front_sum += atom_cfr(p, [1.0], geom, "vs", pattern="iso", polarization="none")
    back_sum = np.zeros_like(total)
    for p in behind:
        back_sum += atom_cfr(p, [1.0], geom, "vs", pattern="iso", polarization="none")
    assert _max_relative_error(total[..., 0, :, :, :], front_sum[..., 0, :, :, :]) <= 1e-12
    assert _max_relative_error(total[..., 1, :, :, :], back_sum[..., 1, :, :, :]) <= 1e-12


def _local_direction(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    azimuth = math.radians(azimuth_deg)
    elevation = math.radians(elevation_deg)
    return np.array(
        [
            math.cos(elevation) * math.cos(azimuth),
            math.cos(elevation) * math.sin(azimuth),
            math.sin(elevation),
        ]
    )


def test_plane_vs_spherical_phase_bound() -> None:
    geom = _synthetic_geometry()
    position = geom.ue_pos[0]
    rotation = geom.ue_rot[0]
    max_q2 = float(np.max(np.sum(geom.elem_offsets**2, axis=1)))
    k = geom.wavenumber
    directions = ((0.0, 0.0), (30.0, 20.0), (-60.0, -35.0), (150.0, 10.0))

    for distance in (10.0, 40.0, 200.0):
        bound = k * max_q2 / (2.0 * distance)
        maxima = []
        for azimuth, elevation in directions:
            point = position + rotation @ (_local_direction(azimuth, elevation) * distance)
            plane = atom_cfr(
                point, [1.0], geom, "vs", wavefront="plane", pattern="iso", polarization="none"
            )
            spherical = atom_cfr(
                point, [1.0], geom, "vs", wavefront="spherical", pattern="iso", polarization="none"
            )
            assert _max_relative_error(np.abs(spherical), np.abs(plane)) <= 1e-12

            h = int(hemisphere_index(geom.local_direction(point, 0))[0])
            ratio = spherical[0, 0, h] / plane[0, 0, h]
            phase = np.abs(np.angle(ratio))
            assert float(np.max(phase)) <= bound + 1e-12
            maxima.append(float(np.max(phase)))

            # The spherical/plane ratio does not depend on the BS: for iso the
            # geometry factor gamma cancels exactly.
            other_bs = 1
            spherical_other = atom_cfr(
                point,
                [1.0],
                geom,
                "vs",
                wavefront="spherical",
                pattern="iso",
                polarization="none",
            )[0, other_bs]
            plane_other = atom_cfr(
                point, [1.0], geom, "vs", wavefront="plane", pattern="iso", polarization="none"
            )[0, other_bs]
            assert _max_relative_error(spherical_other[h] / plane_other[h], ratio) <= 1e-12
        assert max(maxima) >= 0.5 * bound

    # Far-field check at r = 200 m.
    point = position + rotation @ (_local_direction(0.0, 0.0) * 200.0)
    plane = atom_cfr(point, [1.0], geom, "vs", pattern="iso", polarization="none")
    spherical = atom_cfr(
        point, [1.0], geom, "vs", wavefront="spherical", pattern="iso", polarization="none"
    )
    assert float(np.linalg.norm(spherical - plane) / np.linalg.norm(plane)) < 1e-2

    # BV spherical reduces to plane for a far point.
    far = np.array([1.0e5, 0.0, 0.0])
    bv_plane = atom_cfr(far, [1.0], geom, "bv", pattern="iso", polarization="none")
    bv_spherical = atom_cfr(
        far, [1.0], geom, "bv", wavefront="spherical", pattern="iso", polarization="none"
    )
    assert _max_relative_error(bv_spherical, bv_plane) < 1e-4


def test_delay_periodicity() -> None:
    geom = _synthetic_geometry(look_at=False, num_views=1, num_bs=1)
    position = geom.ue_pos[0]
    rotation = geom.ue_rot[0]
    direction = _local_direction(0.0, 0.0)
    distance = 30.0
    source = position + rotation @ (direction * distance)
    period = geom.delay_period
    shift = SPEED_OF_LIGHT * period
    beta = 1.0 + 0.5j

    base = atom_cfr(source, [beta], geom, "vs", pattern="iso", polarization="none")

    distance2 = distance + shift
    source2 = position + rotation @ (direction * distance2)
    beta2 = beta * (distance2 / distance) * np.exp(1j * geom.wavenumber * shift)
    shifted = atom_cfr(source2, [beta2], geom, "vs", pattern="iso", polarization="none")
    assert _max_relative_error(shifted, base) <= 1e-9

    half = 0.5 * shift
    distance3 = distance + half
    source3 = position + rotation @ (direction * distance3)
    beta3 = beta * (distance3 / distance) * np.exp(1j * geom.wavenumber * half)
    off = atom_cfr(source3, [beta3], geom, "vs", pattern="iso", polarization="none")
    assert _max_relative_error(off, base) > 0.1


def test_squint() -> None:
    geom = _synthetic_geometry(look_at=False)
    point = np.array([20.0, 5.0, 3.0])
    beta = 0.4 + 0.9j
    squinted = atom_cfr(point, [beta], geom, "vs", pattern="iso", polarization="none", squint=True)
    plain = atom_cfr(point, [beta], geom, "vs", pattern="iso", polarization="none")

    # Independent loop reference (iso pattern, scalar model).
    reference = np.zeros_like(plain)
    center = geom.num_bins // 2
    lam, k_wave = geom.wavelength, geom.wavenumber
    for v in range(geom.num_views):
        for b in range(geom.num_bs):
            position = geom.ue_pos[v]
            offset = point - position
            distance = float(np.linalg.norm(offset))
            u_loc = geom.ue_rot[v].T @ (offset / distance)
            h = 0 if u_loc[0] >= 0.0 else 1
            carrier = lam / (4.0 * math.pi * distance) * np.exp(-1j * k_wave * distance)
            for row in range(geom.aperture_shape[0]):
                for col in range(geom.aperture_shape[1]):
                    q = geom.elem_offsets[row * geom.aperture_shape[1] + col]
                    dot = float(np.dot(u_loc, q))
                    for n in range(geom.num_bins):
                        frequency = geom.f_c + float(geom.freq_offsets[n])
                        k_n = 2.0 * math.pi * frequency / SPEED_OF_LIGHT
                        reference[v, b, h, row, col, n] = (
                            beta
                            * carrier
                            * np.exp(1j * k_n * dot)
                            * np.exp(
                                -1j
                                * 2.0
                                * math.pi
                                * float(geom.freq_offsets[n])
                                * distance
                                / SPEED_OF_LIGHT
                            )
                        )
    assert _max_relative_error(squinted, reference) <= 1e-12

    np.testing.assert_array_equal(squinted[..., center], plain[..., center])
    other = [n for n in range(geom.num_bins) if n != center]
    assert np.max(np.abs(squinted[..., other] - plain[..., other])) > 0.0
    assert _max_relative_error(squinted[..., other], plain[..., other]) > 1e-3


def test_polarization_vv_analytic() -> None:
    identity = np.eye(3)[None]
    bs_pos = np.zeros((1, 3))
    ue_pos = np.array([[20.0, 0.0, 0.0]])
    for roll in (0.0, 0.3, 1.0):
        geometry = _make_geometry(
            ue_pos, np.array([[math.pi, 0.0, roll]]), bs_pos, num_bins=8, bs_look_at=None
        )
        geometry = dataclasses.replace(geometry, bs_rot=identity)
        none = atom_cfr(bs_pos, [1.0], geometry, "vs", pattern="iso", polarization="none")
        vv = atom_cfr(bs_pos, [1.0], geometry, "vs", pattern="iso", polarization="vv")
        if roll == 0.0:
            np.testing.assert_allclose(vv, none, rtol=1e-12, atol=0.0)
        else:
            np.testing.assert_allclose(vv, math.cos(roll) * none, rtol=1e-12, atol=0.0)

    roll = math.pi / 2.0
    geometry = _make_geometry(
        ue_pos, np.array([[math.pi, 0.0, roll]]), bs_pos, num_bins=8, bs_look_at=None
    )
    geometry = dataclasses.replace(geometry, bs_rot=identity)
    none = atom_cfr(bs_pos, [1.0], geometry, "vs", pattern="iso", polarization="none")
    vv = atom_cfr(bs_pos, [1.0], geometry, "vs", pattern="iso", polarization="vv")
    assert np.max(np.abs(vv)) <= 1e-12 * np.max(np.abs(none))

    bs_high = np.array([[0.0, 0.0, 10.0]])
    geometry = _make_geometry(
        np.array([[20.0, 0.0, 1.5]]),
        np.array([[math.pi, 0.0, 0.0]]),
        bs_high,
        num_bins=8,
        bs_look_at=None,
    )
    geometry = dataclasses.replace(geometry, bs_rot=identity)
    source = mirror_point(bs_high[0], (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    none = atom_cfr(source, [1.0], geometry, "vs", pattern="iso", polarization="none")
    vv = atom_cfr(source, [1.0], geometry, "vs", pattern="iso", polarization="vv")
    np.testing.assert_allclose(vv, -none, rtol=1e-12, atol=1e-15)


def test_amplitude_shapes_and_linearity() -> None:
    geom = _synthetic_geometry()
    rng = np.random.default_rng(11)
    points = rng.uniform(-25.0, 25.0, size=(6, 3))
    amps = rng.normal(size=6) + 1j * rng.normal(size=6)
    views, bss = geom.num_views, geom.num_bs

    shared = atom_cfr(points, amps, geom, "vs", pattern="iso", polarization="none")
    per_capture = atom_cfr(
        points,
        np.broadcast_to(amps[:, None, None], (6, views, bss)).copy(),
        geom,
        "vs",
        pattern="iso",
        polarization="none",
    )
    np.testing.assert_array_equal(shared, per_capture)
    assert shared.dtype == np.complex128
    assert shared.shape == (views, bss, 2, 8, 8, geom.num_bins)

    zeroed = np.broadcast_to(amps[:, None, None], (6, views, bss)).copy()
    zeroed[:, 1, 0] = 0.0
    zero_model = atom_cfr(points, zeroed, geom, "vs", pattern="iso", polarization="none")
    assert np.all(zero_model[1, 0] == 0.0)
    for v in range(views):
        for b in range(bss):
            if (v, b) != (1, 0):
                np.testing.assert_array_equal(zero_model[v, b], shared[v, b])

    amps_b = rng.normal(size=6) + 1j * rng.normal(size=6)
    combo = atom_cfr(points, amps + 2j * amps_b, geom, "vs", pattern="iso", polarization="none")
    separate = atom_cfr(points, amps, geom, "vs", pattern="iso", polarization="none") + 2j * (
        atom_cfr(points, amps_b, geom, "vs", pattern="iso", polarization="none")
    )
    assert _max_relative_error(combo, separate) <= 1e-12

    small = _synthetic_geometry(rows=4, cols=4)
    small_model = atom_cfr(points, amps, small, "vs", pattern="iso", polarization="none")
    assert small_model.shape == (views, bss, 2, 4, 4, small.num_bins)
    assert small_model.dtype == np.complex128


def test_dense_matrix_matches_atom_cfr(monkeypatch: pytest.MonkeyPatch) -> None:
    geom = _synthetic_geometry()
    rng = np.random.default_rng(13)
    points = rng.uniform(-25.0, 25.0, size=(5, 3))
    amps = rng.normal(size=5) + 1j * rng.normal(size=5)

    cases = [
        ("vs", "plane", False, "tr38901", "vv"),
        ("bv", "plane", False, "tr38901", "vv"),
        ("vs", "spherical", True, "iso", "none"),
        ("bv", "spherical", True, "iso", "none"),
    ]
    for space, wavefront, squint, pattern, polarization in cases:
        matrix = dense_matrix(
            points,
            geom,
            space,
            wavefront=wavefront,
            squint=squint,
            pattern=pattern,
            polarization=polarization,
        )
        expected = atom_cfr(
            points,
            amps,
            geom,
            space,
            wavefront=wavefront,
            squint=squint,
            pattern=pattern,
            polarization=polarization,
        ).ravel()
        rows = geom.num_views * geom.num_bs * 2 * geom.num_elements * geom.num_bins
        assert matrix.shape == (rows, 5)
        assert _max_relative_error(matrix @ amps, expected) <= 1e-12

    monkeypatch.setattr(forward_exact, "DENSE_MAX_ENTRIES", 1)
    with pytest.raises(ValueError):
        dense_matrix(points, geom, "vs", pattern="iso", polarization="none")


def test_capture_factors() -> None:
    geom = _synthetic_geometry()
    rng = np.random.default_rng(17)
    points = rng.uniform(-25.0, 25.0, size=(7, 3))
    lam = geom.wavelength

    for v in range(geom.num_views):
        for b in range(geom.num_bs):
            vs = capture_factors(points, geom, "vs", v, b, pattern="iso")
            np.testing.assert_allclose(vs.tau, geom.vs_delay(points, v), rtol=1e-13)
            np.testing.assert_array_equal(vs.hemisphere, hemisphere_index(vs.u_local))
            r = np.linalg.norm(points - geom.ue_pos[v], axis=-1)
            np.testing.assert_allclose(vs.rx_range, r, rtol=1e-13)
            np.testing.assert_allclose(np.abs(vs.gamma), lam / (4.0 * np.pi * r), rtol=1e-13)

            bv = capture_factors(points, geom, "bv", v, b, pattern="iso")
            np.testing.assert_allclose(bv.tau, geom.bistatic_delay(points, v, b), rtol=1e-13)
            np.testing.assert_array_equal(bv.hemisphere, hemisphere_index(bv.u_local))
            r1, r2 = geom.bistatic_ranges(points, v, b)
            np.testing.assert_allclose(bv.rx_range, r2, rtol=1e-13)
            np.testing.assert_allclose(
                np.abs(bv.gamma), lam / ((4.0 * np.pi) ** 1.5 * r1 * r2), rtol=1e-13
            )


def test_validation_errors() -> None:
    geom = _synthetic_geometry()
    plain = _synthetic_geometry(look_at=False)
    points = np.array([[1.0, 2.0, 3.0]])
    amps = np.array([1.0 + 0.0j])

    with pytest.raises(ValueError):
        atom_cfr(points, amps, geom, "xx")
    with pytest.raises(ValueError):
        atom_cfr(points, amps, geom, "vs", wavefront="xx")
    with pytest.raises(ValueError):
        atom_cfr(points, amps, geom, "vs", pattern="dipole")
    with pytest.raises(ValueError):
        atom_cfr(points, amps, geom, "vs", polarization="hh")
    with pytest.raises(ValueError):
        atom_cfr(geom.bs_pos, amps, plain, "vs", pattern="tr38901")
    with pytest.raises(ValueError):
        atom_cfr(geom.bs_pos, amps, plain, "vs", pattern="iso", polarization="vv")
    with pytest.raises(ValueError):
        atom_cfr(geom.ue_pos[0], amps, geom, "vs", pattern="iso")
    with pytest.raises(ValueError):
        atom_cfr(geom.bs_pos[0], amps, geom, "bv", pattern="iso")
    with pytest.raises(ValueError):
        atom_cfr(points, np.ones(3), geom, "vs", pattern="iso")
    with pytest.raises(ValueError):
        atom_cfr(np.array([[np.nan, 0.0, 0.0]]), amps, geom, "vs", pattern="iso")
    # Negative or out-of-range capture indices must not wrap around silently.
    for v, b in ((-1, 0), (geom.num_views, 0), (0, -1), (0, geom.num_bs)):
        with pytest.raises(ValueError):
            capture_factors(points, geom, "vs", v, b, pattern="iso")

    accepted = atom_cfr(plain.bs_pos[0], [1.0], plain, "vs", pattern="iso", polarization="none")
    assert accepted.shape == (plain.num_views, plain.num_bs, 2, 8, 8, plain.num_bins)


def test_benchmark_atom_cfr() -> None:
    if os.environ.get("RF_TOMO_BENCH") != "1":
        pytest.skip("set RF_TOMO_BENCH=1 to run the atom_cfr benchmark")

    rng = np.random.default_rng(0)
    ue_pos = rng.uniform(-20.0, 20.0, size=(8, 3))
    ue_pos[:, 2] = 1.5
    orientations = rng.uniform(-math.pi, math.pi, size=(8, 3))
    bs_pos = np.array([[10.0, 0.0, 8.0], [-12.0, 4.0, 6.0]])
    geom = _make_geometry(
        ue_pos, orientations, bs_pos, num_bins=128, bs_look_at=np.array([0.0, 0.0, 1.5])
    )
    points = rng.uniform(-50.0, 50.0, size=(2000, 3))
    points[:, 2] = rng.uniform(0.5, 20.0, size=2000)
    amps = rng.normal(size=2000) + 1j * rng.normal(size=2000)
    start = time.perf_counter()
    result = atom_cfr(points, amps, geom, "vs", pattern="tr38901", polarization="vv")
    elapsed = time.perf_counter() - start
    print(f"atom_cfr P=2000 V=8 B=2 N=128 8x8: {elapsed:.3f} s ({result.shape})")
    assert result.shape == (8, 2, 2, 8, 8, 128)
