"""Tests for the separable exact forward operator (T07c)."""

from __future__ import annotations

import math
import os
import time
from pathlib import Path

import numpy as np
import pytest
from scipy.constants import c as SPEED_OF_LIGHT

from plateau_rt.domain.rf_tomography import forward_sep
from plateau_rt.domain.rf_tomography.antenna import bs_orientation
from plateau_rt.domain.rf_tomography.forward_exact import (
    atom_cfr,
    capture_factors,
    dense_matrix,
)
from plateau_rt.domain.rf_tomography.forward_sep import (
    CONSTRAINED_DEGREE,
    SeparableOperator,
    incidence_cosine,
    project_shared_phase,
)
from plateau_rt.domain.rf_tomography.geometry import (
    LOS_VS_TOLERANCE_M,
    CaptureGeometry,
    mirror_point,
    planar_element_offsets,
    rotations_from_orientations,
)

FIXTURES = Path(__file__).parent / "fixtures" / "rf_tomography"
F_C = 3.5e9
BANDWIDTH = 100e6
PATTERN_POLARIZATION = (
    ("tr38901", "none"),
    ("tr38901", "vv"),
    ("iso", "none"),
)


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


def _quantised_geometry(num_views: int = 4, num_bs: int = 2) -> CaptureGeometry:
    """Same pose bank but with the float32-quantised frequency grid of Sionna."""
    ue_pos = np.array([[0.0, 0.0, 1.5], [5.0, -3.0, 1.6], [-4.0, 2.0, 1.4], [0.0, 0.0, 1.5]])[
        :num_views
    ]
    orientations = np.array(
        [[0.0, 0.0, 0.0], [0.6, 0.1, 0.2], [-1.0, -0.2, 0.3], [math.pi, 0.0, 0.0]]
    )[:num_views]
    bs_pos = np.array([[10.0, 0.0, 8.0], [-12.0, 4.0, 6.0]])[:num_bs]
    targets = np.array([[0.0, 0.0, 1.5], [2.0, 1.0, 1.0]])[:num_bs]
    return CaptureGeometry.from_orientations(
        ue_pos,
        orientations,
        bs_pos,
        f_c=F_C,
        bandwidth=BANDWIDTH,
        num_bins=16,
        bs_look_at=targets,
    )


def _far_from_special(point: np.ndarray, geom: CaptureGeometry) -> bool:
    """True when ``point`` is not (numerically) on a UE or a BS."""
    ue = np.min(np.linalg.norm(geom.ue_pos - point, axis=1))
    bs = np.min(np.linalg.norm(geom.bs_pos - point, axis=1))
    return ue > 1e-6 and bs > 1e-6


def _point_cloud(
    geom: CaptureGeometry, rng: np.random.Generator, extra: int = 14, *, include_los: bool = False
) -> np.ndarray:
    """Off-grid points with, for every view, one point in front and one behind."""
    rows: list[np.ndarray] = []
    for v in range(geom.num_views):
        for sign in (1.0, -1.0):
            point = geom.ue_pos[v]
            for _ in range(50):
                local = np.array([sign, rng.uniform(-0.5, 0.5), rng.uniform(-0.4, 0.4)])
                local = local / np.linalg.norm(local)
                candidate = geom.ue_pos[v] + geom.ue_rot[v] @ (local * rng.uniform(12.0, 45.0))
                if _far_from_special(candidate, geom):
                    point = candidate
                    break
            rows.append(point)
    for _ in range(extra):
        point = np.array([0.0, 0.0, 0.0])
        for _ in range(50):
            candidate = rng.uniform([-35.0, -35.0, 0.5], [35.0, 35.0, 25.0])
            if _far_from_special(candidate, geom):
                point = candidate
                break
        rows.append(point)
    points = np.asarray(rows, dtype=np.float64)
    if include_los:
        points = np.vstack([points, geom.bs_pos[0]])
    return points


def _assert_both_hemispheres(
    points: np.ndarray, geom: CaptureGeometry, space: str, pattern: str, polarization: str
) -> None:
    """Assert every capture of ``space`` has at least one point per hemisphere."""
    for v in range(geom.num_views):
        for b in range(geom.num_bs):
            factors = capture_factors(
                points, geom, space, v, b, pattern=pattern, polarization=polarization
            )
            assert set(np.unique(factors.hemisphere).tolist()) == {0, 1}


def _complex(rng: np.random.Generator, shape: tuple[int, ...] | int) -> np.ndarray:
    return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def _max_relative_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def _beta_values(x: np.ndarray, cos_inc: np.ndarray, model: str) -> np.ndarray:
    """Return the explicit per-capture amplitudes ``[P, V, B]`` of ``x``."""
    if model == "shared":
        return np.broadcast_to(x[:, None, None], cos_inc.shape).copy()
    if model == "per_view":
        return x
    basis = np.stack([cos_inc**degree for degree in range(CONSTRAINED_DEGREE + 1)], axis=-1)
    return np.einsum("pvbk,pvbk->pvb", basis, np.broadcast_to(x[:, None, None, :], basis.shape))


def test_forward_matches_atom_cfr_shared() -> None:
    rng = np.random.default_rng(101)
    geoms = [_synthetic_geometry(), _quantised_geometry()]
    for geom in geoms:
        base = _point_cloud(geom, rng)
        for space in ("vs", "bv"):
            points = np.vstack([base, geom.bs_pos[0]]) if space == "vs" else base
            for pattern, polarization in PATTERN_POLARIZATION:
                _assert_both_hemispheres(points, geom, space, pattern, polarization)
            x = _complex(rng, points.shape[0])
            for pattern, polarization in PATTERN_POLARIZATION:
                operator = SeparableOperator(
                    points,
                    geom,
                    space,
                    beta_model="shared",
                    pattern=pattern,
                    polarization=polarization,
                )
                expected = atom_cfr(
                    points, x, geom, space, pattern=pattern, polarization=polarization
                )
                assert _max_relative_error(operator.forward(x), expected) <= 1e-12


def test_forward_matches_atom_cfr_per_view() -> None:
    rng = np.random.default_rng(202)
    geom = _synthetic_geometry()
    points = _point_cloud(geom, rng)
    x = _complex(rng, (points.shape[0], geom.num_views, geom.num_bs))
    for space in ("vs", "bv"):
        _assert_both_hemispheres(points, geom, space, "tr38901", "vv")
        operator = SeparableOperator(
            points, geom, space, beta_model="per_view", pattern="tr38901", polarization="vv"
        )
        expected = atom_cfr(points, x, geom, space, pattern="tr38901", polarization="vv")
        assert operator.x_shape == x.shape
        assert _max_relative_error(operator.forward(x), expected) <= 1e-12


def test_forward_matches_atom_cfr_constrained() -> None:
    geom = _synthetic_geometry()
    t0 = geom.bs_pos[0]
    planes = [
        (np.zeros(3), np.array([0.0, 0.0, 1.0])),
        (np.array([30.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])),
        (np.array([0.0, -25.0, 0.0]), np.array([0.0, 1.0, 0.0])),
    ]
    mirror_points = [mirror_point(t0, plane_point, normal) for plane_point, normal in planes]
    points = np.vstack([*mirror_points, t0])
    rng = np.random.default_rng(303)
    x = _complex(rng, (points.shape[0], CONSTRAINED_DEGREE + 1))

    views, bss = geom.num_views, geom.num_bs
    cos_inc = np.zeros((points.shape[0], views, bss))
    for index, source in enumerate(points):
        for v in range(views):
            arrival = geom.ue_pos[v] - source
            arrival = arrival / np.linalg.norm(arrival)
            for b in range(bss):
                to_bs = source - geom.bs_pos[b]
                distance = float(np.linalg.norm(to_bs))
                if distance <= LOS_VS_TOLERANCE_M:
                    cos_inc[index, v, b] = 1.0
                else:
                    normal = to_bs / distance
                    cos_inc[index, v, b] = abs(float(np.dot(arrival, normal)))

    operator = SeparableOperator(points, geom, "vs", beta_model="constrained")
    for v in range(views):
        for b in range(bss):
            np.testing.assert_allclose(
                incidence_cosine(points, geom, v, b), cos_inc[:, v, b], rtol=0.0, atol=1e-12
            )
    # The mirror points are exact images of t0 across the known planes for b = 0.
    for index, (_, normal) in enumerate(planes):
        arrival = geom.ue_pos[0] - points[index]
        arrival = arrival / np.linalg.norm(arrival)
        unit_normal = normal / np.linalg.norm(normal)
        np.testing.assert_allclose(
            incidence_cosine(points[index], geom, 0, 0),
            abs(float(np.dot(arrival, unit_normal))),
            rtol=0.0,
            atol=1e-12,
        )
    # The LoS point is exact in its own capture.
    np.testing.assert_allclose(incidence_cosine(t0, geom, 0, 0), 1.0, rtol=0.0, atol=1e-15)

    amplitudes = _beta_values(x, cos_inc, "constrained")
    expected = atom_cfr(points, amplitudes, geom, "vs", pattern="tr38901", polarization="none")
    assert _max_relative_error(operator.forward(x), expected) <= 1e-12

    with pytest.raises(ValueError):
        SeparableOperator(points, geom, "bv", beta_model="constrained")


def test_gauges_match_explicit_diagonal() -> None:
    rng = np.random.default_rng(404)
    geom = _synthetic_geometry()
    points = _point_cloud(geom, rng)
    views, bss = geom.num_views, geom.num_bs
    phi = rng.uniform(0.0, 2.0 * np.pi, size=(views, bss))
    tau = rng.normal(scale=1e-8, size=(views, bss))
    df = geom.freq_offsets
    gauge_phase = np.exp(1j * phi)[:, :, None, None, None, None]
    gauge_delay = np.exp(-2j * np.pi * tau[:, :, None] * df[None, None, :])[
        :, :, None, None, None, :
    ]

    cos_inc = np.zeros((points.shape[0], views, bss))
    for v in range(views):
        for b in range(bss):
            cos_inc[:, v, b] = incidence_cosine(points, geom, v, b)
    for space, model in (
        ("vs", "shared"),
        ("bv", "shared"),
        ("vs", "per_view"),
        ("bv", "per_view"),
        ("vs", "constrained"),
    ):
        operator = SeparableOperator(points, geom, space, beta_model=model)
        x = _complex(rng, operator.x_shape)
        base = atom_cfr(
            points,
            _beta_values(x, cos_inc, model),
            geom,
            space,
            pattern="tr38901",
            polarization="none",
        )
        gauged = SeparableOperator(points, geom, space, beta_model=model, gauges=(phi, tau))
        assert _max_relative_error(gauged.forward(x), gauge_phase * gauge_delay * base) <= 1e-12

        zeros = np.zeros((views, bss))
        zeroed = SeparableOperator(points, geom, space, beta_model=model, gauges=(zeros, zeros))
        assert _max_relative_error(zeroed.forward(x), operator.forward(x)) <= 1e-15


def test_adjoint_random_vectors() -> None:
    rng = np.random.default_rng(505)
    geom = _synthetic_geometry()
    points = _point_cloud(geom, rng)
    views, bss = geom.num_views, geom.num_bs
    gauges = (
        rng.uniform(0.0, 2.0 * np.pi, size=(views, bss)),
        rng.normal(scale=1e-8, size=(views, bss)),
    )
    for space, model in (
        ("vs", "shared"),
        ("bv", "shared"),
        ("vs", "per_view"),
        ("bv", "per_view"),
        ("vs", "constrained"),
    ):
        for active_gauges in (None, gauges):
            for cache in (True, False):
                operator = SeparableOperator(
                    points,
                    geom,
                    space,
                    beta_model=model,
                    gauges=active_gauges,
                    cache=cache,
                )
                x = _complex(rng, operator.x_shape)
                y = _complex(rng, operator.y_shape)
                forward = operator.forward(x)
                left = np.vdot(y, forward)
                right = np.vdot(operator.adjoint(y), x)
                assert abs(left - right) <= 1e-10 * np.linalg.norm(forward) * np.linalg.norm(y)

                linear = operator.as_linear_operator()
                xf, yf = x.reshape(-1), y.reshape(-1)
                image = linear.matvec(xf)
                assert np.allclose(image, forward.reshape(-1))
                assert np.allclose(linear.rmatvec(yf), operator.adjoint(y).reshape(-1))
                assert np.allclose(linear.H.matvec(yf), operator.adjoint(y).reshape(-1))
                assert abs(np.vdot(yf, image) - np.vdot(linear.rmatvec(yf), xf)) <= (
                    1e-10 * np.linalg.norm(image) * np.linalg.norm(yf)
                )


def test_matvec_matches_dense_matrix() -> None:
    rng = np.random.default_rng(606)
    geom = _synthetic_geometry()
    base = _point_cloud(geom, rng)
    for space in ("vs", "bv"):
        points = np.vstack([base, geom.bs_pos[0]]) if space == "vs" else base
        operator = SeparableOperator(
            points, geom, space, beta_model="shared", pattern="tr38901", polarization="vv"
        )
        dense = dense_matrix(points, geom, space, pattern="tr38901", polarization="vv")
        assert operator.shape == dense.shape
        x = _complex(rng, points.shape[0])
        y = _complex(rng, dense.shape[0])
        assert _max_relative_error(operator.matvec(x), dense @ x) <= 1e-12
        assert _max_relative_error(operator.rmatvec(y), dense.conj().T @ y) <= 1e-12


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
    operator = SeparableOperator(
        fixture["bs_pos"], geom, "vs", pattern="tr38901", polarization="vv"
    )
    model = operator.forward(np.array([1.0]))
    assert model.shape == (2, 1, 2, 8, 8, 16)

    for v in range(2):
        reference = fixture["aperture_cfr"][v].astype(np.complex128)
        assert _max_relative_error(model[v, 0], reference) <= 1e-3

        received = int(np.argmax(np.sum(np.abs(reference) ** 2, axis=(1, 2, 3))))
        assert np.all(model[v, 0, received] != 0.0)
        assert np.all(model[v, 0, 1 - received] == 0.0)
        assert np.all(reference[1 - received] == 0.0)


def test_cache_modes(monkeypatch: pytest.MonkeyPatch) -> None:
    rng = np.random.default_rng(707)
    geom = _synthetic_geometry()
    points = _point_cloud(geom, rng)
    x = _complex(rng, points.shape[0])
    y = _complex(rng, (geom.num_views, geom.num_bs, 2, 8, 8, geom.num_bins))

    cached = SeparableOperator(points, geom, "vs", cache=True)
    uncached = SeparableOperator(points, geom, "vs", cache=False)
    assert cached.cached is True
    assert uncached.cached is False
    assert _max_relative_error(cached.forward(x), uncached.forward(x)) <= 1e-15
    assert _max_relative_error(cached.adjoint(y), uncached.adjoint(y)) <= 1e-15

    assert SeparableOperator(points, geom, "vs", cache=None).cached is True
    monkeypatch.setattr(forward_sep, "CACHE_MAX_BYTES", 0)
    assert SeparableOperator(points, geom, "vs", cache=None).cached is False


def test_with_gauges() -> None:
    rng = np.random.default_rng(808)
    geom = _synthetic_geometry()
    points = _point_cloud(geom, rng)
    views, bss = geom.num_views, geom.num_bs
    operator = SeparableOperator(points, geom, "vs", beta_model="per_view")
    x = _complex(rng, operator.x_shape)
    original = operator.forward(x).copy()

    phi = rng.uniform(0.0, 2.0 * np.pi, size=(views, bss))
    tau = rng.normal(scale=1e-8, size=(views, bss))
    gauged = operator.with_gauges((phi, tau))
    reference = SeparableOperator(
        points, geom, "vs", beta_model="per_view", gauges=(phi, tau)
    ).forward(x)
    assert _max_relative_error(gauged.forward(x), reference) <= 1e-15
    np.testing.assert_array_equal(operator.forward(x), original)

    ungauged = operator.with_gauges(None)
    assert _max_relative_error(ungauged.forward(x), original) <= 1e-15

    phase_only = operator.with_gauges((phi, np.zeros((views, bss))))
    phased = phase_only.forward(x)
    for v in range(views):
        for b in range(bss):
            scale = np.linalg.norm(original[v, b])
            assert abs(np.linalg.norm(phased[v, b]) - scale) <= 1e-12 * scale


def test_project_shared_phase() -> None:
    rng = np.random.default_rng(909)
    x = _complex(rng, (6, 3))
    x[0] = 0.0
    projected = project_shared_phase(x)

    magnitude = max(1.0, float(np.max(np.abs(projected)) ** 2))
    rank_one = np.imag(projected[:, :, None] * np.conj(projected[:, None, :]))
    assert np.max(np.abs(rank_one)) <= 1e-12 * magnitude
    assert _max_relative_error(project_shared_phase(projected), projected) <= 1e-12

    grid = np.linspace(0.0, np.pi, 2001)
    for row in range(x.shape[0]):
        best = min(
            np.linalg.norm(x[row] - np.exp(1j * psi) * np.real(np.exp(-1j * psi) * x[row]))
            for psi in grid
        )
        assert np.linalg.norm(x[row] - projected[row]) <= best + 1e-12

    phase = rng.uniform(-np.pi, np.pi, size=5)
    real_part = rng.standard_normal((5, 3))
    shared = np.exp(1j * phase)[:, None] * real_part
    np.testing.assert_allclose(project_shared_phase(shared), shared, rtol=0.0, atol=1e-12)
    assert np.all(project_shared_phase(np.zeros((4, 3))) == 0.0)

    with pytest.raises(ValueError):
        project_shared_phase(np.zeros(4))


def test_validation() -> None:
    geom = _synthetic_geometry()
    points = np.array([[3.0, -2.0, 4.0]])
    views, bss = geom.num_views, geom.num_bs
    y_shape = (views, bss, 2, 8, 8, geom.num_bins)

    with pytest.raises(ValueError):
        SeparableOperator(points, geom, "vs", beta_model="unknown")
    with pytest.raises(ValueError):
        SeparableOperator(points, geom, "bv", beta_model="constrained")
    with pytest.raises(ValueError):
        SeparableOperator(points, geom, "vs", pattern="dipole")
    with pytest.raises(ValueError):
        SeparableOperator(geom.ue_pos[0], geom, "vs")
    with pytest.raises(ValueError):
        SeparableOperator(
            points, geom, "vs", gauges=(np.zeros((views, bss - 1)), np.zeros((views, bss)))
        )
    with pytest.raises(ValueError):
        SeparableOperator(
            points,
            geom,
            "vs",
            gauges=(np.full((views, bss), np.nan), np.zeros((views, bss))),
        )

    shared = SeparableOperator(points, geom, "vs")
    with pytest.raises(ValueError):
        shared.forward(np.ones(points.shape[0] + 1))
    with pytest.raises(ValueError):
        shared.adjoint(np.zeros((views, bss, 2, 8, 8, geom.num_bins + 1)))
    with pytest.raises(ValueError):
        shared.matvec(np.ones(shared.shape[1] + 1))
    with pytest.raises(ValueError):
        shared.rmatvec(np.ones(shared.shape[0] + 1))

    per_view = SeparableOperator(points, geom, "vs", beta_model="per_view")
    with pytest.raises(ValueError):
        per_view.forward(np.ones(points.shape[0]))
    with pytest.raises(ValueError):
        per_view.matvec(np.ones(shared.shape[1]))

    accepted = shared.forward(np.ones(points.shape[0]))
    assert accepted.shape == y_shape
    assert accepted.dtype == np.complex128


def test_benchmark_separable_operator() -> None:
    if os.environ.get("RF_TOMO_BENCH") != "1":
        pytest.skip("set RF_TOMO_BENCH=1 to run the separable operator benchmark")

    rng = np.random.default_rng(0)
    views, bins = 16, 128
    angle = 2.0 * np.pi * np.arange(views) / views
    ue_pos = np.stack([40.0 * np.cos(angle), 40.0 * np.sin(angle), np.full(views, 1.5)], axis=1)
    orientations = np.stack([angle + np.pi, np.zeros(views), np.zeros(views)], axis=1)
    bs_pos = np.array([[-70.0, 5.0, 25.0]])
    geom = _make_geometry(ue_pos, orientations, bs_pos, num_bins=bins, bs_look_at=np.zeros(3))
    count = 50_000
    points = np.column_stack(
        [
            rng.uniform(-50.0, 50.0, size=count),
            rng.uniform(-50.0, 50.0, size=count),
            rng.uniform(-2.0, 40.0, size=count),
        ]
    )

    for cache in (True, False):
        start = time.perf_counter()
        operator = SeparableOperator(points, geom, "bv", beta_model="shared", cache=cache)
        init_time = time.perf_counter() - start
        x = _complex(rng, count)
        y = _complex(rng, operator.shape[0])

        start = time.perf_counter()
        image = operator.matvec(x)
        forward_time = time.perf_counter() - start

        start = time.perf_counter()
        back = operator.rmatvec(y)
        adjoint_time = time.perf_counter() - start

        print(
            f"separable P={count} V={views} B=1 N={bins} cache={cache}: "
            f"init {init_time:.1f} s, A {forward_time:.2f} s, AH {adjoint_time:.2f} s "
            f"(target <= 3 s)"
        )
        assert image.shape == (operator.shape[0],)
        assert back.shape == (count,)
        assert np.all(np.isfinite(image))
        assert np.all(np.isfinite(back))


def _bins_cases(num_bins: int) -> list[tuple[int, ...]]:
    """Return the bin subsets of the brief: DC, strided and first bin."""
    return [(num_bins // 2,), (1, 5, 6), (0,)]


def test_bins_forward_matches_full_band_subset() -> None:
    rng = np.random.default_rng(1001)
    geom = _synthetic_geometry()
    points = _point_cloud(geom, rng)
    views, bss = geom.num_views, geom.num_bs
    gauges = (
        rng.uniform(0.0, 2.0 * np.pi, size=(views, bss)),
        rng.normal(scale=1e-8, size=(views, bss)),
    )
    for model, space in (
        ("shared", "vs"),
        ("shared", "bv"),
        ("per_view", "bv"),
        ("constrained", "vs"),
    ):
        for active_gauges in (None, gauges):
            full = SeparableOperator(points, geom, space, beta_model=model, gauges=active_gauges)
            x = _complex(rng, full.x_shape)
            reference = full.forward(x)
            for bins in _bins_cases(geom.num_bins):
                sub = SeparableOperator(
                    points, geom, space, beta_model=model, gauges=active_gauges, bins=bins
                )
                assert sub.bins == tuple(bins)
                assert sub.y_shape == reference.shape[:-1] + (len(bins),)
                np.testing.assert_array_equal(sub.freq_offsets, geom.freq_offsets[list(bins)])
                assert _max_relative_error(sub.forward(x), reference[..., list(bins)]) <= 1e-12


def test_bins_adjoint() -> None:
    rng = np.random.default_rng(1002)
    geom = _synthetic_geometry()
    points = _point_cloud(geom, rng)
    views, bss = geom.num_views, geom.num_bs
    gauges = (
        rng.uniform(0.0, 2.0 * np.pi, size=(views, bss)),
        rng.normal(scale=1e-8, size=(views, bss)),
    )
    for model, space in (
        ("shared", "vs"),
        ("shared", "bv"),
        ("per_view", "bv"),
        ("constrained", "vs"),
    ):
        for active_gauges in (None, gauges):
            for bins in _bins_cases(geom.num_bins):
                operator = SeparableOperator(
                    points, geom, space, beta_model=model, gauges=active_gauges, bins=bins
                )
                x = _complex(rng, operator.x_shape)
                y = _complex(rng, operator.y_shape)
                forward = operator.forward(x)
                left = np.vdot(y, forward)
                right = np.vdot(operator.adjoint(y), x)
                assert abs(left - right) <= 1e-10 * abs(left)


def test_bins_with_gauges_keeps_bins() -> None:
    rng = np.random.default_rng(1003)
    geom = _synthetic_geometry()
    points = _point_cloud(geom, rng)
    views, bss = geom.num_views, geom.num_bs
    gauges = (
        rng.uniform(0.0, 2.0 * np.pi, size=(views, bss)),
        rng.normal(scale=1e-8, size=(views, bss)),
    )
    for model, space in (("shared", "bv"), ("per_view", "bv"), ("constrained", "vs")):
        for bins in _bins_cases(geom.num_bins):
            operator = SeparableOperator(points, geom, space, beta_model=model, bins=bins)
            x = _complex(rng, operator.x_shape)
            gauged = operator.with_gauges(gauges)
            assert gauged.bins == operator.bins == tuple(bins)
            fresh = SeparableOperator(
                points, geom, space, beta_model=model, gauges=gauges, bins=bins
            )
            assert _max_relative_error(gauged.forward(x), fresh.forward(x)) <= 1e-15


def test_bins_invalid() -> None:
    geom = _synthetic_geometry()
    points = np.array([[3.0, -2.0, 4.0]])
    num_bins = geom.num_bins
    for bad in ([], [num_bins], [-1], [2, 2], [[0, 1]], [0.5], [True]):
        with pytest.raises(ValueError):
            SeparableOperator(points, geom, "vs", bins=bad)


def test_bins_none_is_default() -> None:
    rng = np.random.default_rng(1004)
    geom = _synthetic_geometry()
    points = _point_cloud(geom, rng)
    operator = SeparableOperator(points, geom, "bv", beta_model="per_view")
    explicit = SeparableOperator(points, geom, "bv", beta_model="per_view", bins=None)
    assert operator.bins is None
    assert explicit.bins is None
    x = _complex(rng, operator.x_shape)
    np.testing.assert_array_equal(explicit.forward(x), operator.forward(x))
