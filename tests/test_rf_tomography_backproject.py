"""Tests for the E1 fast back-projection operator (T07b)."""

from __future__ import annotations

import dataclasses
import os
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pytest
from scipy.constants import c as SPEED_OF_LIGHT

from plateau_rt.domain.rf_tomography import backproject as backproject_module
from plateau_rt.domain.rf_tomography import forward_exact, observables
from plateau_rt.domain.rf_tomography.antenna import bs_orientation
from plateau_rt.domain.rf_tomography.backproject import (
    apply_lookup,
    backproject,
    capture_lookup,
    capture_volume,
    envelope_sum,
)
from plateau_rt.domain.rf_tomography.geometry import (
    CaptureGeometry,
    planar_element_offsets,
    rotations_from_orientations,
)
from plateau_rt.domain.rf_tomography.sync import apply_gauge

FIXTURES = Path(__file__).parent / "fixtures" / "rf_tomography"
F_C = 3.5e9
BANDWIDTH = 100e6
NUM_BINS = 16
SPACING = 0.5


def _make_geometry() -> CaptureGeometry:
    """Build the shared exact-float64-grid geometry of the brief (8x8, N=16)."""
    wavelength = SPEED_OF_LIGHT / F_C
    elem_offsets = planar_element_offsets(wavelength, rows=8, cols=8)
    df = BANDWIDTH / NUM_BINS
    freq_offsets = (np.arange(NUM_BINS) - NUM_BINS // 2) * df
    ue_pos = np.array([[0.0, 0.0, 1.5], [20.0, -10.0, 1.6], [-15.0, 8.0, 1.4]])
    orientations = np.array([[0.3, 0.05, 0.0], [2.5, -0.1, 0.1], [-1.2, 0.2, -0.05]])
    bs_pos = np.array([[-30.0, 5.0, 15.0], [25.0, 30.0, 10.0]])
    bs_rot = np.stack([bs_orientation(bs_pos[b], np.zeros(3)) for b in range(bs_pos.shape[0])])
    return CaptureGeometry(
        ue_pos=ue_pos,
        ue_rot=rotations_from_orientations(orientations),
        bs_pos=bs_pos,
        elem_offsets=elem_offsets,
        freq_offsets=freq_offsets,
        f_c=F_C,
        aperture_shape=(8, 8),
        bs_rot=bs_rot,
    )


def _sample_far(rng: np.random.Generator, geom: CaptureGeometry, count: int) -> np.ndarray:
    """Draw ``count`` points in the box, each at least 5 m from every UE and BS."""
    low = np.array([-25.0, -25.0, -3.0])
    high = np.array([25.0, 25.0, 12.0])
    obstacles = np.vstack([geom.ue_pos, geom.bs_pos])
    kept: list[np.ndarray] = []
    while len(kept) < count:
        candidates = rng.uniform(low, high, size=(count, 3))
        for candidate in candidates:
            if float(np.min(np.linalg.norm(obstacles - candidate, axis=1))) >= 5.0:
                kept.append(candidate)
                if len(kept) >= count:
                    break
    return np.asarray(kept, dtype=np.float64)


def _evaluation_points(
    rng: np.random.Generator, geom: CaptureGeometry, atoms: np.ndarray
) -> np.ndarray:
    """Return atoms, their random neighbours and uniform points, away from UE/BS."""
    parts = [atoms]
    for atom in atoms:
        parts.append(atom + rng.uniform(-1.5, 1.5, size=(10, 3)))
    parts.append(rng.uniform([-25.0, -25.0, -3.0], [25.0, 25.0, 12.0], size=(100, 3)))
    all_points = np.vstack(parts)
    obstacles = np.vstack([geom.ue_pos, geom.bs_pos])
    distance = np.min(
        np.linalg.norm(all_points[:, None, :] - obstacles[None, :, :], axis=-1), axis=1
    )
    return all_points[distance >= 1.0]


def _window(geom: CaptureGeometry, window: str | None) -> np.ndarray:
    """Return the separable window ``W[R, C, N]`` (ones for ``None``)."""
    rows, cols, bins = geom.aperture_shape[0], geom.aperture_shape[1], geom.num_bins
    if window is None:
        return np.ones((rows, cols, bins), dtype=np.float64)
    return (
        observables.taylor_window(rows)[:, None, None]
        * observables.taylor_window(cols)[None, :, None]
        * observables.taylor_window(bins)[None, None, :]
    )


def _exact(
    points: np.ndarray,
    Y: np.ndarray,
    geom: CaptureGeometry,
    space: str,
    window: str | None,
    per_capture: bool,
    *,
    pattern: str = "tr38901",
    polarization: str = "none",
    matrix: np.ndarray | None = None,
) -> np.ndarray:
    """Return the dense adjoint reference ``A^H (W * Y)``."""
    if matrix is None:
        matrix = forward_exact.dense_matrix(
            points, geom, space, pattern=pattern, polarization=polarization
        )
    num_rows, num_cols = geom.aperture_shape
    mn = num_rows * num_cols * geom.num_bins
    weighted = (_window(geom, window) * Y).reshape(geom.num_views, geom.num_bs, 2, mn)
    if per_capture:
        operator = matrix.reshape(geom.num_views, geom.num_bs, 2, mn, points.shape[0])
        return np.einsum("vbhrp,vbhr->vbhp", operator.conj(), weighted)
    return matrix.conj().T @ weighted.ravel()


def _err_db(fast: np.ndarray, exact: np.ndarray) -> float:
    """Return ``20 log10(||fast - exact|| / ||exact||)`` over all entries."""
    return float(20.0 * np.log10(np.linalg.norm(fast - exact) / np.linalg.norm(exact)))


def _max_rel(a: np.ndarray, b: np.ndarray) -> float:
    """Return the max absolute difference over the max absolute value of ``b``."""
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


@pytest.fixture(scope="module")
def scene() -> dict[str, object]:
    """Atoms, amplitudes, evaluation points, atom-model data and dense references."""
    geom = _make_geometry()
    rng = np.random.default_rng(np.random.SeedSequence(6))
    atoms = _sample_far(rng, geom, 10)
    amps = rng.standard_normal(10) + 1j * rng.standard_normal(10)
    evaluation = _evaluation_points(rng, geom, atoms)
    return {
        "geom": geom,
        "atoms": atoms,
        "amps": amps,
        "evaluation": evaluation,
        "Y": {
            "vs": forward_exact.atom_cfr(atoms, amps, geom, "vs"),
            "bv": forward_exact.atom_cfr(atoms, amps, geom, "bv"),
        },
        "dense": {
            "vs": forward_exact.dense_matrix(evaluation, geom, "vs"),
            "bv": forward_exact.dense_matrix(evaluation, geom, "bv"),
        },
    }


@pytest.mark.parametrize("space", ["vs", "bv"])
@pytest.mark.parametrize("window", ["taylor", None])
@pytest.mark.parametrize("per_capture", [True, False])
@pytest.mark.parametrize(
    "kind,doc_gate,pin",
    [("trilinear", -30.0, -36.0), ("tricubic", -40.0, -60.0)],
)
def test_doc_acceptance_gate(
    scene: dict[str, object],
    space: str,
    window: str | None,
    per_capture: bool,
    kind: str,
    doc_gate: float,
    pin: float,
) -> None:
    geom = scene["geom"]
    evaluation = scene["evaluation"]
    Y = scene["Y"][space]
    exact = _exact(evaluation, Y, geom, space, window, per_capture, matrix=scene["dense"][space])
    fast = backproject(
        Y, geom, evaluation, space, per_capture=per_capture, kind=kind, window=window
    )
    error = _err_db(fast, exact)
    print(f"space={space} window={window} per_capture={per_capture} kind={kind}: {error:.2f} dB")
    assert error < doc_gate
    assert error < pin


@pytest.mark.parametrize("window", [None, "taylor"])
@pytest.mark.parametrize("kind", ["trilinear", "tricubic"])
def test_exact_at_grid_nodes(window: str | None, kind: str) -> None:
    geom = _make_geometry()
    rng = np.random.default_rng(np.random.SeedSequence(7))
    shape = (
        geom.num_views,
        geom.num_bs,
        2,
        geom.aperture_shape[0],
        geom.aperture_shape[1],
        geom.num_bins,
    )
    Y = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex128)

    v = 0
    rows, cols = geom.aperture_shape
    oversample = 8
    u_y, u_z, _ = observables.volume_axes(
        oversample * cols,
        oversample * rows,
        oversample * geom.num_bins,
        delta_f=geom.delta_f,
        spacing_lambda=SPACING,
    )
    iy, iz = np.nonzero(u_y[:, None] ** 2 + u_z[None, :] ** 2 <= 0.8)
    angular = list(zip(iy.tolist(), iz.tolist()))
    delay_index = np.arange(oversample * geom.num_bins)
    ranges = SPEED_OF_LIGHT * delay_index / (oversample * geom.num_bins * geom.delta_f)
    valid_delays = delay_index[(ranges >= 6.0) & (ranges <= 40.0)]

    angular_pick = [angular[i] for i in np.linspace(0, len(angular) - 1, 12).astype(int)]
    delay_pick = valid_delays[np.linspace(0, valid_delays.size - 1, 6).astype(int)]

    points = []
    radii = []
    for row_index, col_index in angular_pick:
        u_x = np.sqrt(max(0.0, 1.0 - u_y[row_index] ** 2 - u_z[col_index] ** 2))
        for sign in (1.0, -1.0):
            for delay in delay_pick:
                radius = ranges[delay]
                local = np.array([sign * u_x, u_y[row_index], u_z[col_index]])
                points.append(geom.ue_pos[v] + radius * (geom.ue_rot[v] @ local))
                radii.append(radius)
    points = np.asarray(points, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    assert points.shape[0] <= 150

    assert np.std(np.angle(np.exp(-1j * geom.wavenumber * radii))) > 0.5

    exact = _exact(points, Y, geom, "vs", window, True)
    fast = backproject(Y, geom, points, "vs", kind=kind, window=window, oversample=(8, 8))
    assert _max_rel(fast[v], exact[v]) <= 1e-10


@pytest.mark.parametrize("space", ["vs", "bv"])
def test_hemisphere_mask(scene: dict[str, object], space: str) -> None:
    geom = scene["geom"]
    evaluation = scene["evaluation"]
    Y = scene["Y"][space]
    fast = backproject(Y, geom, evaluation, space, kind="trilinear", window="taylor")
    rows = np.arange(evaluation.shape[0])
    for v in range(geom.num_views):
        for b in range(geom.num_bs):
            hemisphere = forward_exact.capture_factors(evaluation, geom, space, v, b).hemisphere
            assert np.all(fast[v, b][1 - hemisphere, rows] == 0.0)
            assert np.all(np.abs(fast[v, b][hemisphere, rows]) > 0.0)


def test_n_mode_random_tau(scene: dict[str, object]) -> None:
    geom = scene["geom"]
    evaluation = scene["evaluation"]
    Y_S = scene["Y"]["vs"]
    num_views, num_bs = geom.num_views, geom.num_bs

    rng = np.random.default_rng(np.random.SeedSequence(99))
    phi = rng.uniform(0.0, 2.0 * np.pi, size=(num_views, num_bs))
    tau = rng.normal(0.0, 10e-9, size=(num_views, num_bs))
    phi[0, 0] = 0.0
    tau[0, 0] = 0.0

    offsets = geom.freq_offsets
    Y_N = (
        np.exp(1j * phi)[..., None, None, None, None]
        * np.exp(-2j * np.pi * tau[..., None, None, None, None] * offsets)
        * Y_S
    )
    # Same gauge convention as the T06 data generator (docs section 2.4).
    np.testing.assert_allclose(
        Y_N, apply_gauge(Y_S, phi, tau, offsets), rtol=0.0, atol=1e-15 * np.max(np.abs(Y_S))
    )

    env_S = envelope_sum(backproject(Y_S, geom, evaluation, "vs"))
    env_N = envelope_sum(backproject(Y_N, geom, evaluation, "vs"))
    assert _max_rel(env_N, env_S) > 0.1

    wrong = tau + 3e-9
    wrong[0, 0] = tau[0, 0]
    env_wrong = envelope_sum(backproject(Y_N, geom, evaluation, "vs", tau=wrong))
    assert _max_rel(env_wrong, env_S) > 0.1

    env_fixed = envelope_sum(backproject(Y_N, geom, evaluation, "vs", tau=tau))
    assert _max_rel(env_fixed, env_S) <= 1e-10

    per_S = backproject(Y_S, geom, evaluation, "vs")
    per_N = backproject(Y_N, geom, evaluation, "vs", tau=tau)
    assert _max_rel(per_N, np.exp(1j * phi)[:, :, None, None] * per_S) <= 1e-10

    coherent_S = backproject(Y_S, geom, evaluation, "vs", per_capture=False)
    coherent_N = backproject(Y_N, geom, evaluation, "vs", tau=tau, per_capture=False)
    assert _max_rel(coherent_N, coherent_S) > 0.1


def test_real_sionna_data() -> None:
    fixture = np.load(FIXTURES / "sionna_mock_los.npz")
    geom = CaptureGeometry(
        ue_pos=fixture["ue_pos"],
        ue_rot=rotations_from_orientations(fixture["ue_orientation"]),
        bs_pos=fixture["bs_pos"],
        elem_offsets=planar_element_offsets(SPEED_OF_LIGHT / float(fixture["f_c"])),
        freq_offsets=fixture["freq_offsets"].astype(np.float64),
        f_c=float(fixture["f_c"]),
        aperture_shape=(8, 8),
        bs_rot=bs_orientation(fixture["bs_pos"][0], fixture["bs_look_at"])[None],
    )
    Y = fixture["aperture_cfr"].astype(np.complex128)[:, None]

    bs = fixture["bs_pos"][0]
    offsets = (np.arange(5) - 2) * 1.0 + 0.37
    cube = bs[None] + np.stack(
        np.meshgrid(offsets, offsets, offsets, indexing="ij"), axis=-1
    ).reshape(-1, 3)
    points = np.vstack([bs[None], cube])

    for per_capture in (True, False):
        exact = _exact(
            points, Y, geom, "vs", "taylor", per_capture, pattern="tr38901", polarization="vv"
        )
        fast = backproject(
            Y,
            geom,
            points,
            "vs",
            per_capture=per_capture,
            kind="trilinear",
            window="taylor",
            pattern="tr38901",
            polarization="vv",
        )
        error = _err_db(fast, exact)
        assert error < -30.0


def test_lookup_api(scene: dict[str, object]) -> None:
    geom = scene["geom"]
    evaluation = scene["evaluation"]
    Y = scene["Y"]["vs"]
    fast = backproject(Y, geom, evaluation, "vs", per_capture=True, oversample=(8, 8))

    for v in range(geom.num_views):
        for b in range(geom.num_bs):
            volume = capture_volume(Y, geom, v, b, oversample=(8, 8))
            assert volume.shape == (2, 64, 64, 128)
            for h in range(2):
                lookup = capture_lookup(geom, evaluation, "vs", v, b, h, oversample=(8, 8))
                assert lookup.shape == (64, 64, 128)
                assert lookup.idx.dtype == np.int64
                assert lookup.idx.shape == (evaluation.shape[0], 8)
                assert lookup.w.dtype == np.float64
                assert lookup.carrier.dtype == np.complex128
                assert lookup.valid.dtype == np.bool_
                assert _max_rel(apply_lookup(volume[h], lookup), fast[v, b, h]) <= 1e-13

    lookup_cubic = capture_lookup(geom, evaluation, "vs", 0, 0, 0, kind="tricubic")
    assert lookup_cubic.idx.shape == (evaluation.shape[0], 64)

    tau = 3.7e-8
    volume_tau = capture_volume(Y, geom, 0, 0, tau=tau, oversample=(8, 8))
    compensated = Y.copy()
    compensated[0, 0] = compensated[0, 0] * np.exp(
        2j * np.pi * geom.freq_offsets[None, None, :] * tau
    )
    volume_compensated = capture_volume(compensated, geom, 0, 0, oversample=(8, 8))
    np.testing.assert_allclose(volume_tau, volume_compensated, rtol=1e-13, atol=1e-13)


def test_chunking(scene: dict[str, object], monkeypatch: pytest.MonkeyPatch) -> None:
    geom = scene["geom"]
    evaluation = scene["evaluation"]
    Y = scene["Y"]["vs"]
    default = backproject(Y, geom, evaluation, "vs")
    monkeypatch.setattr(backproject_module, "POINT_CHUNK", 7)
    chunked = backproject(Y, geom, evaluation, "vs")
    assert chunked.shape == default.shape
    assert _max_rel(chunked, default) <= 1e-13


def test_singular_points(scene: dict[str, object]) -> None:
    geom = scene["geom"]
    evaluation = scene["evaluation"]
    for space in ("vs", "bv"):
        Y = scene["Y"][space]
        base = backproject(Y, geom, evaluation, space)

        with_ue = np.vstack([evaluation, geom.ue_pos[0][None]])
        got_ue = backproject(Y, geom, with_ue, space)
        assert np.all(got_ue[:, :, :, -1] == 0.0)
        np.testing.assert_array_equal(got_ue[:, :, :, :-1], base)

        with_bs = np.vstack([evaluation, geom.bs_pos[0][None]])
        got_bs = backproject(Y, geom, with_bs, space)
        if space == "bv":
            assert np.all(got_bs[:, :, :, -1] == 0.0)
        else:
            assert np.any(got_bs[:, :, :, -1] != 0.0)


def test_envelope_sum() -> None:
    rng = np.random.default_rng(np.random.SeedSequence(3))
    bp = rng.standard_normal((3, 2, 2, 11)) + 1j * rng.standard_normal((3, 2, 2, 11))
    env = envelope_sum(bp)
    np.testing.assert_array_equal(env, np.sum(np.abs(bp) ** 2, axis=(0, 1, 2)))
    assert env.dtype == np.float64

    one_d = rng.standard_normal(7) + 1j * rng.standard_normal(7)
    assert envelope_sum(one_d).dtype == np.float64
    np.testing.assert_array_equal(envelope_sum(one_d), np.abs(one_d) ** 2)


def test_value_errors(scene: dict[str, object]) -> None:
    geom = scene["geom"]
    evaluation = scene["evaluation"]
    Y = scene["Y"]["vs"]

    with pytest.raises(ValueError):
        backproject(Y, geom, evaluation, "xx")
    with pytest.raises(ValueError):
        backproject(Y, geom, evaluation, "vs", kind="nearest")
    with pytest.raises(ValueError):
        backproject(Y, geom, evaluation, "vs", window="hann")
    with pytest.raises(ValueError):
        backproject(Y, geom, evaluation, "vs", oversample=(0, 8))
    with pytest.raises(ValueError):
        capture_lookup(geom, evaluation, "vs", 0, 0, 2)
    with pytest.raises(ValueError):
        capture_lookup(geom, evaluation, "vs", geom.num_views, 0, 0)
    with pytest.raises(ValueError):
        capture_lookup(geom, evaluation, "vs", 0, geom.num_bs, 0)
    with pytest.raises(ValueError):
        backproject(Y[:, :, :, :, :, :-1], geom, evaluation, "vs")
    with pytest.raises(ValueError):
        backproject(Y, geom, evaluation, "vs", tau=np.zeros((2, 2)))
    with pytest.raises(ValueError):
        backproject(Y, geom, evaluation, "vs", tau=np.full((3, 2), np.nan))
    with pytest.raises(ValueError):
        backproject(Y, geom, evaluation, "vs", tau=np.full((3, 2), np.inf))
    with pytest.raises(ValueError):
        backproject(Y, geom, np.zeros((0, 3)), "vs")
    with pytest.raises(ValueError):
        backproject(Y, geom, np.array([[np.nan, 0.0, 0.0]]), "vs")
    with pytest.raises(ValueError):
        backproject(Y, geom, np.zeros((4, 2)), "vs")

    jittered = np.array(geom.elem_offsets, copy=True)
    jittered[0, 0] += 1e-3
    broken = dataclasses.replace(geom, elem_offsets=jittered)
    with pytest.raises(ValueError):
        backproject(Y, broken, evaluation, "vs")

    lookup = capture_lookup(geom, evaluation, "vs", 0, 0, 0)
    with pytest.raises(ValueError):
        apply_lookup(np.zeros((8, 8, 8), dtype=np.complex128), lookup)
    with pytest.raises(ValueError):
        envelope_sum(np.array(1.0 + 1.0j))


def _bench_geometry() -> CaptureGeometry:
    """Build the N=128, 4-view, 1-BS geometry of the benchmark test."""
    wavelength = SPEED_OF_LIGHT / F_C
    elem_offsets = planar_element_offsets(wavelength, rows=8, cols=8)
    df = BANDWIDTH / 128
    freq_offsets = (np.arange(128) - 64) * df
    ue_pos = np.array([[0.0, 0.0, 1.5], [20.0, -10.0, 1.6], [-15.0, 8.0, 1.4], [10.0, 15.0, 1.3]])
    orientations = np.array(
        [[0.3, 0.05, 0.0], [2.5, -0.1, 0.1], [-1.2, 0.2, -0.05], [0.8, 0.1, 0.2]]
    )
    bs_pos = np.array([[-30.0, 5.0, 15.0]])
    bs_rot = bs_orientation(bs_pos[0], np.zeros(3))[None]
    return CaptureGeometry(
        ue_pos=ue_pos,
        ue_rot=rotations_from_orientations(orientations),
        bs_pos=bs_pos,
        elem_offsets=elem_offsets,
        freq_offsets=freq_offsets,
        f_c=F_C,
        aperture_shape=(8, 8),
        bs_rot=bs_rot,
    )


@pytest.mark.skipif(os.environ.get("RF_TOMO_BENCH") != "1", reason="set RF_TOMO_BENCH=1")
def test_benchmark_backproject() -> None:
    geom = _bench_geometry()
    rng = np.random.default_rng(np.random.SeedSequence(2026))
    points = rng.uniform(-100.0, 100.0, size=(100_000, 3))
    shape = (
        geom.num_views,
        geom.num_bs,
        2,
        geom.aperture_shape[0],
        geom.aperture_shape[1],
        geom.num_bins,
    )
    Y = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex128)

    captures = geom.num_views * geom.num_bs
    start = time.perf_counter()
    for v in range(geom.num_views):
        for b in range(geom.num_bs):
            capture_volume(Y, geom, v, b, window="taylor")
    t_vol = (time.perf_counter() - start) / captures

    tracemalloc.start()
    start = time.perf_counter()
    result = backproject(Y, geom, points, "vs", per_capture=True, kind="trilinear", window="taylor")
    t_backproject = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    t_pt = (t_backproject - captures * t_vol) / (captures * points.shape[0])
    target = 16 * t_vol + 3.4e6 * 16 * t_pt
    print(f"backproject P=1e5 N=128 V=4 B=1: {t_backproject:.3f} s")
    print(f"t_vol (capture_volume per capture): {t_vol * 1e3:.2f} ms")
    print(f"t_pt (per point-capture): {t_pt * 1e9:.1f} ns")
    print(f"extrapolated docs 4.1 target (3.4e6 pts, 16 captures): {target:.1f} s (target 60 s)")
    print(f"peak traced memory: {peak / 1e6:.1f} MB")
    assert np.all(np.isfinite(result))
    assert np.isfinite(t_vol) and np.isfinite(t_pt)
