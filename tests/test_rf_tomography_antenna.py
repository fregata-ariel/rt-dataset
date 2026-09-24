"""Tests for the NumPy BS antenna pattern (Sionna TR 38.901 element)."""

from __future__ import annotations

import numpy as np
import pytest

from plateau_rt.domain.rf_tomography.antenna import (
    bs_orientation,
    bs_pattern,
    tr38901_gain,
)

BS_A = np.array([-70.0, 5.0, 25.0])
TARGET_A = np.array([0.0, 0.0, 1.5])
BS_B = np.array([10.0, -20.0, 30.0])
TARGET_B = np.array([40.0, 5.0, 0.0])


def _unit(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def _independent_frame(
    bs: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = _unit(target - bs)
    y = _unit(np.cross(np.array([0.0, 0.0, 1.0]), x))
    z = np.cross(x, y)
    return x, y, z


def test_boresight_gain_is_8_dbi() -> None:
    gain = tr38901_gain(np.pi / 2.0, 0.0)
    assert 10.0 * np.log10(float(gain)) == pytest.approx(8.0, abs=1e-12)


def test_minus_3db_at_half_beamwidth() -> None:
    half = np.deg2rad(32.5)
    boresight_db = 10.0 * np.log10(float(tr38901_gain(np.pi / 2.0, 0.0)))
    for phi in (half, -half):
        gain_db = 10.0 * np.log10(float(tr38901_gain(np.pi / 2.0, phi)))
        assert gain_db == pytest.approx(5.0, abs=1e-12)
        assert gain_db - boresight_db == pytest.approx(-3.0, abs=1e-12)
    for theta in (np.pi / 2.0 - half, np.pi / 2.0 + half):
        gain_db = 10.0 * np.log10(float(tr38901_gain(theta, 0.0)))
        assert gain_db == pytest.approx(5.0, abs=1e-12)
        assert gain_db - boresight_db == pytest.approx(-3.0, abs=1e-12)


def test_30db_floor() -> None:
    theta = np.linspace(0.0, np.pi, 181)
    phi = np.linspace(-np.pi, np.pi, 361)
    grid = tr38901_gain(theta[:, None], phi[None, :])
    boresight = float(tr38901_gain(np.pi / 2.0, 0.0))
    rel_db = 10.0 * np.log10(grid / boresight)
    assert bool(np.all(rel_db >= -30.0 - 1e-12))
    for theta_deg, phi_deg in ((90.0, 180.0), (0.0, 90.0), (180.0, 120.0)):
        gain = float(tr38901_gain(np.deg2rad(theta_deg), np.deg2rad(phi_deg)))
        assert 10.0 * np.log10(gain / boresight) == pytest.approx(-30.0, abs=1e-12)
    gain = float(tr38901_gain(np.deg2rad(90.0), np.deg2rad(100.0)))
    expected_db = 8.0 - 12.0 * (100.0 / 65.0) ** 2
    assert 10.0 * np.log10(gain) == pytest.approx(expected_db, abs=1e-12)
    assert expected_db > -22.0


def test_gain_matches_table_formula_in_degrees() -> None:
    def expected_db(theta_deg: float, phi_deg: float) -> float:
        phi_w = (phi_deg + 180.0) % 360.0 - 180.0
        a_v = min(12.0 * ((theta_deg - 90.0) / 65.0) ** 2, 30.0)
        a_h = min(12.0 * (phi_w / 65.0) ** 2, 30.0)
        return 8.0 - min(a_v + a_h, 30.0)

    for theta_deg, phi_deg in ((60.0, 20.0), (120.0, -45.0), (95.0, 170.0), (30.0, 400.0)):
        gain = float(tr38901_gain(np.deg2rad(theta_deg), np.deg2rad(phi_deg)))
        assert 10.0 * np.log10(gain) == pytest.approx(expected_db(theta_deg, phi_deg), abs=1e-12)
    wrapped = float(tr38901_gain(np.deg2rad(30.0), np.deg2rad(400.0)))
    direct = float(tr38901_gain(np.deg2rad(30.0), np.deg2rad(40.0)))
    assert 10.0 * np.log10(wrapped) == pytest.approx(10.0 * np.log10(direct), abs=1e-12)
    theta = np.deg2rad(np.array([[60.0], [120.0], [30.0]]))
    phi = np.deg2rad(np.array([[20.0, -45.0, 170.0, 400.0]]))
    assert tr38901_gain(theta, phi).shape == (3, 4)
    assert tr38901_gain(theta, phi).dtype == np.float64


def test_iso_is_one() -> None:
    rng = np.random.default_rng(np.random.SeedSequence(0))
    dirs = rng.normal(size=(9, 3))
    rotation = bs_orientation(BS_A, TARGET_A)
    out = bs_pattern(dirs, rotation, kind="iso")
    assert out.dtype == np.complex128
    assert out.shape == (9,)
    assert np.all(out.real == 1.0) and np.all(out.imag == 0.0)


def test_bs_orientation_matches_look_at_frame() -> None:
    for bs, target in ((BS_A, TARGET_A), (BS_B, TARGET_B)):
        rotation = bs_orientation(bs, target)
        x, y, z = _independent_frame(bs, target)
        expected = np.stack([x, y, z], axis=1)
        assert np.abs(rotation - expected).max() < 1e-12
        assert rotation.dtype == np.float64
        assert float(np.linalg.det(rotation)) == pytest.approx(1.0, abs=1e-12)
        assert np.abs(rotation.T @ rotation - np.eye(3)).max() < 1e-12


def test_bs_pattern_boresight_and_half_power() -> None:
    rotation = bs_orientation(BS_A, TARGET_A)
    x, y, z = _independent_frame(BS_A, TARGET_A)
    look = (TARGET_A - BS_A).reshape(1, 3)
    gain = bs_pattern(look, rotation)
    assert gain.dtype == np.complex128
    assert gain.shape == (1,)
    assert np.all(gain.imag == 0.0)
    assert (np.abs(gain) ** 2)[0] == pytest.approx(10.0**0.8, rel=1e-12)
    half = np.deg2rad(32.5)
    tilted = np.stack(
        [
            np.cos(half) * x + np.sin(half) * y,
            np.cos(half) * x - np.sin(half) * y,
            np.cos(half) * x + np.sin(half) * z,
            np.cos(half) * x - np.sin(half) * z,
        ]
    )
    tilted_gain = bs_pattern(tilted, rotation)
    assert tilted_gain.shape == (4,)
    for value in np.abs(tilted_gain) ** 2:
        assert float(value) == pytest.approx(10.0**0.5, rel=1e-12)
    back = bs_pattern(-look, rotation)
    assert float(np.abs(back[0]) ** 2) == pytest.approx(10.0 ** (-2.2), rel=1e-12)


def test_bs_pattern_scale_invariance_and_shapes() -> None:
    rng = np.random.default_rng(np.random.SeedSequence(0))
    dirs = rng.normal(size=(2, 5, 3))
    rotation = bs_orientation(BS_A, TARGET_A)
    out = bs_pattern(dirs, rotation)
    assert out.shape == (2, 5)
    assert out.dtype == np.complex128
    scaled = bs_pattern(7.3 * dirs, rotation)
    rel = np.abs(scaled - out) / np.abs(out)
    assert rel.max() < 1e-15


def test_bs_pattern_rejects_bad_input() -> None:
    rotation = bs_orientation(BS_A, TARGET_A)
    with pytest.raises(ValueError):
        bs_pattern(np.zeros((1, 3)), rotation)
    with pytest.raises(ValueError):
        bs_pattern(np.zeros((4, 2)), rotation)
    with pytest.raises(ValueError):
        bs_pattern(np.eye(3), np.zeros((2, 3)))
    with pytest.raises(ValueError, match="tr38901.*iso|iso.*tr38901"):
        bs_pattern(np.eye(3), rotation, kind="dipole")
    with pytest.raises(ValueError):
        bs_orientation(BS_A, BS_A)


def test_matches_sionna_tr38901_element() -> None:
    rt = pytest.importorskip("sionna.rt")
    assert rt is not None
    import drjit as dr
    import mitsuba as mi
    from sionna.rt.antenna_pattern import v_tr38901_pattern

    theta = np.linspace(0.0, np.pi, 37)
    phi = np.linspace(-3.0 * np.pi, 3.0 * np.pi, 145)
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    theta_flat = theta_grid.ravel()
    phi_flat = phi_grid.ravel()
    pattern = v_tr38901_pattern(mi.Float(theta_flat), mi.Float(phi_flat))
    sionna = np.array(dr.real(pattern)) + 1j * np.array(dr.imag(pattern))
    expected = np.sqrt(tr38901_gain(theta_flat, phi_flat))
    assert np.abs(sionna - expected).max() < 1e-5


def test_matches_sionna_transmitter_look_at() -> None:
    rt = pytest.importorskip("sionna.rt")
    from sionna.rt.utils import rotation_matrix as sionna_rotation

    tx = rt.Transmitter(name="tx", position=[-70.0, 5.0, 25.0], look_at=[0.0, 0.0, 1.5])
    matrix = sionna_rotation(tx.orientation)
    sionna = np.array([[matrix[i][j][0] for j in range(3)] for i in range(3)], dtype=np.float64)
    ours = bs_orientation(BS_A, TARGET_A)
    assert np.abs(ours - sionna).max() < 1e-6


def test_matches_sionna_path_solver_los() -> None:
    rt = pytest.importorskip("sionna.rt")
    rng = np.random.default_rng(np.random.SeedSequence(1))
    directions = rng.normal(size=(40, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    rx_positions = BS_A + 30.0 * directions

    gains: dict[str, np.ndarray] = {}
    for kind in ("tr38901", "iso"):
        scene = rt.load_scene()
        scene.frequency = 3.5e9
        scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern=kind, polarization="V")
        scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
        scene.add(rt.Transmitter(name="tx", position=[-70.0, 5.0, 25.0], look_at=[0.0, 0.0, 1.5]))
        for i, position in enumerate(rx_positions):
            scene.add(rt.Receiver(name=f"rx{i}", position=[float(v) for v in position]))
        paths = rt.PathSolver()(
            scene=scene,
            max_depth=0,
            los=True,
            specular_reflection=False,
            diffuse_reflection=False,
            refraction=False,
            synthetic_array=True,
        )
        a_real, a_imag = paths.a
        gains[kind] = (np.array(a_real) + 1j * np.array(a_imag)).reshape(len(rx_positions), -1)[
            :, 0
        ]
    ratio = gains["tr38901"] / gains["iso"]
    expected = bs_pattern(directions, bs_orientation(BS_A, TARGET_A))
    assert np.abs(ratio - expected).max() < 1e-5
