"""Tests for the tomography configuration registry (T15, section A/C)."""

from __future__ import annotations

import dataclasses
import functools
import importlib
import inspect
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from plateau_rt.domain.rf_camera.camera import look_at_orientation
from plateau_rt.domain.rf_tomography import configs as cfg_mod
from plateau_rt.domain.rf_tomography import kernels, sync
from plateau_rt.domain.rf_tomography.configs import (
    BUDGETS,
    COLUMNS,
    COMPLETION_CONFIGS,
    CONFIG_ALIASES,
    CONFIGS,
    CORE_CONFIGS,
    LATTICE_EDGES,
    LATTICE_NAMES,
    LATTICE_NODES,
    NODE_GAUGE_SENSITIVITY,
    ONE_ELEMENT_CONFIGS,
    PARTIAL_D_CONFIGS,
    RELATIVE_SCALES,
    RESTRICTIONS,
    SPACES,
    STRATEGY_NAMES,
    SUBSETS,
    SYNC_MODES,
    Config,
    HyperRange,
    get_config,
    lattice_configs,
    run_e1,
    run_e2,
    run_roi,
    run_support,
)
from plateau_rt.domain.rf_tomography.forward_exact import atom_cfr
from plateau_rt.domain.rf_tomography.forward_sep import SeparableOperator
from plateau_rt.domain.rf_tomography.geometry import CaptureGeometry, VoxelGrid
from plateau_rt.domain.rf_tomography.observables import NODE_NAMES, extract
from plateau_rt.domain.rf_tomography.solvers import bp, coherent, power
from plateau_rt.domain.rf_tomography.synthetic import offgrid_points

CENTER = np.array([0.0, 0.0, 5.0])
EXPECTED_NAMES = (
    "I-S",
    "I-N",
    "D-S",
    "D-N",
    "P-S",
    "P-N",
    "ID-S",
    "ID-N",
    "IP-S",
    "IP-N",
    "DP-S",
    "DP-N",
    "IDP-S",
    "IDP-N",
    "I@n0",
    "P_W",
    "IP_W",
    "I-o",
    "D-o-S",
    "D-o-N",
    "P-o",
    "ID-o-S",
    "ID-o-N",
    "IP-o",
    "DP-o-S",
    "DP-o-N",
    "IDP-o-S",
    "IDP-o-N",
    "IDP-1el",
    "DP-1el",
    "PxK-S",
    "IPxK-S",
    "DP-S_tau",
    "IDP-S_tau",
    "D-N_sep",
    "P-N_sep",
    "ID-N_sep",
    "IP-N_sep",
    "DP-N_sep",
    "IDP-N_sep",
)
TRACK_OF_SYNC = {
    "S": "ideal-S",
    "N": "ideal-N",
    "S_tau": "S_tau",
    "N_sep": "N-sep",
    "any": "ideal-S",
}


def _micro_geometry() -> CaptureGeometry:
    """Return the 4-view micro capture geometry of the brief."""
    pos, ori = [], []
    for i, height in enumerate((1.5, 12.0, 1.5, 12.0)):
        az = np.deg2rad(20.0 + 90.0 * i)
        p = (15.0 * np.cos(az), 15.0 * np.sin(az), height)
        pos.append(p)
        ori.append(look_at_orientation(p, tuple(CENTER)))
    return CaptureGeometry.from_orientations(
        np.array(pos),
        np.array(ori),
        np.array([[3.0, -2.0, 40.0]]),
        f_c=3.5e9,
        bandwidth=100e6,
        num_bins=16,
        aperture_shape=(4, 4),
        bs_look_at=CENTER,
    )


@functools.lru_cache(maxsize=1)
def _scene() -> SimpleNamespace:
    """Build the micro scene once per module."""
    geom = _micro_geometry()
    grid = VoxelGrid.from_bounds(CENTER - [4.0, 4.0, 2.0], CENTER + [4.0, 4.0, 2.0], 2.0)
    rng = np.random.default_rng(np.random.SeedSequence([0, 7]))
    pts = np.stack(
        [
            offgrid_points(
                grid,
                1,
                rng,
                offset_max=0.2,
                voxels=np.array([np.ravel_multi_index(v, grid.shape)]),
            )[0][0]
            for v in ((0, 0, 1), (4, 4, 1))
        ]
    )
    amps = np.array([1.0, 0.7]) * np.exp(2j * np.pi * rng.uniform(size=2))
    y_clean = atom_cfr(pts, amps, geom, "bv")
    p_ref, _ = sync.reference_power(y_clean, np.ones(y_clean.shape[:2], dtype=bool))
    tracks, gt = sync.make_tracks(
        y_clean, 30.0, p_ref, sync.TrackSeeds(0), freq_offsets=geom.freq_offsets
    )
    return SimpleNamespace(
        geom=geom, grid=grid, pts=pts, tracks=tracks, gt=gt, noise_var=gt["sigma2"]
    )


def _track_data(cfg: Config) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray] | None]:
    """Return ``(Y, gauges)`` for ``cfg``: true gauges for non-S/any configs."""
    scene = _scene()
    track = TRACK_OF_SYNC[cfg.sync]
    y = scene.tracks[track]
    if cfg.sync in ("S", "any"):
        return y, None
    return y, scene.gt["gauges"][track]


def _relative(a: np.ndarray, b: np.ndarray) -> float:
    """Return ``max|a - b| / max|a|`` (inf when ``a`` is all zero and ``b`` is not)."""
    x = np.asarray(a, dtype=np.complex128)
    y = np.asarray(b, dtype=np.complex128)
    denom = float(np.max(np.abs(x))) if x.size else 0.0
    numer = float(np.max(np.abs(x - y))) if x.size else 0.0
    if denom == 0.0:
        return 0.0 if numer == 0.0 else float("inf")
    return numer / denom


def _all_hyper() -> list[HyperRange]:
    """Return every HyperRange of the registry, deduplicated by identity."""
    seen: set[int] = set()
    out: list[HyperRange] = []
    for cfg in CONFIGS.values():
        for step in cfg.chain:
            for h in step.hyper:
                if id(h) not in seen:
                    seen.add(id(h))
                    out.append(h)
    return out


def test_all_names_present() -> None:
    assert tuple(CONFIGS) == EXPECTED_NAMES
    groups = {
        "core": CORE_CONFIGS,
        "completion": COMPLETION_CONFIGS,
        "omni": cfg_mod.OMNI_CONFIGS,
        "1el": ONE_ELEMENT_CONFIGS,
        "partial_d": PARTIAL_D_CONFIGS,
        "sync": cfg_mod.SYNC_VARIANT_CONFIGS,
    }
    flat = [name for names in groups.values() for name in names]
    assert sorted(flat) == sorted(CONFIGS)
    assert len(set(flat)) == len(flat) == 40
    for column, names in groups.items():
        for name in names:
            assert CONFIGS[name].column == column
    for alias, target in CONFIG_ALIASES.items():
        assert get_config(alias).name == target
    assert get_config("P×K-S").name == "PxK-S"
    assert get_config("IDP-S_τ").name == "IDP-S_tau"
    assert get_config("IDP-N-sep").name == "IDP-N_sep"
    assert get_config("I_n0").name == "I@n0"
    with pytest.raises(ValueError):
        get_config("no-such-config")


def test_config_fields() -> None:
    for cfg in CONFIGS.values():
        assert cfg.node in NODE_NAMES
        assert cfg.subset in SUBSETS
        assert cfg.sync in SYNC_MODES
        assert cfg.column in COLUMNS
        assert cfg.budget in BUDGETS
        assert cfg.restrict in RESTRICTIONS
        assert set(cfg.lattices) <= set(LATTICE_NAMES)
        assert tuple(cfg.spaces) == SPACES
        assert isinstance(cfg.planned, tuple)
        assert all(isinstance(p, str) and p for p in cfg.planned)
        if cfg.e1 is not None:
            assert cfg.e1.stage == "E1"
        if cfg.roi is not None:
            assert cfg.roi.stage == "ROI"
        for step in cfg.e2:
            assert step.stage == "E2"
        names = [step.name for step in cfg.chain]
        assert len(set(names)) == len(names)
    for cls in (Config, cfg_mod.Step, cfg_mod.NStrategy):
        for f in dataclasses.fields(cls):
            assert "ill" not in f.name
    core_completion = set(CORE_CONFIGS) | set(COMPLETION_CONFIGS)
    for name in core_completion:
        cfg = CONFIGS[name]
        assert cfg.e1 is not None
        if name in ("D-S", "D-N"):
            assert cfg.e2 == ()
        else:
            assert len(cfg.e2) >= 1


def test_chain_callables_are_real_public_functions() -> None:
    prefix = "plateau_rt.domain.rf_tomography"
    for cfg in CONFIGS.values():
        for step in cfg.chain:
            fn = step.fn
            assert callable(fn)
            assert fn.__module__.startswith(prefix)
            assert getattr(importlib.import_module(fn.__module__), fn.__qualname__) is fn
            assert not fn.__qualname__.split(".")[-1].startswith("_")
            assert not fn.__module__.split(".")[-1].startswith("_")
            inspect.signature(fn).bind_partial(**dict(step.kwargs), **step.defaults())
            params = inspect.signature(fn).parameters
            for h in step.hyper:
                assert h.name in params
        for strategy in cfg.n_strategies:
            assert callable(strategy.fn)
            assert strategy.fn.__module__.startswith(prefix)
            module = importlib.import_module(strategy.fn.__module__)
            assert getattr(module, strategy.fn.__qualname__) is strategy.fn
            inspect.signature(strategy.fn).bind_partial(**dict(strategy.kwargs))
            for key, (factory, aux_kw) in strategy.aux.items():
                assert callable(factory)
                assert factory.__module__.startswith(prefix)
                fmod = importlib.import_module(factory.__module__)
                assert getattr(fmod, factory.__qualname__) is factory
                inspect.signature(factory).bind_partial("bv", **dict(aux_kw))
                assert isinstance(key, str) and key


def test_hyper_ranges() -> None:
    ranges = _all_hyper()
    # Exactly the eight canonical module constants (floor/p_hit/p_false/power-lam/
    # tv-weight/damp/l1-lam/mmv-lam), shared by identity across all steps.
    assert len(ranges) == 8
    for h in ranges:
        assert np.isfinite(h.low) and np.isfinite(h.high) and np.isfinite(h.default)
        assert h.low < h.high
        assert h.low <= h.default <= h.high
        if h.log:
            assert h.low > 0.0
        assert h.relative_to in RELATIVE_SCALES
        trials = h.trials()
        assert trials.shape == (20,) and trials.dtype == np.float64
        assert trials[0] == pytest.approx(h.low, rel=1e-12)
        assert trials[-1] == pytest.approx(h.high, rel=1e-12)
        assert bool(np.all(np.diff(trials) > 0.0))
    with pytest.raises(ValueError):
        HyperRange("x", 1.0, 0.5, 0.7)
    with pytest.raises(ValueError):
        HyperRange("x", 0.1, 1.0, 2.0)
    with pytest.raises(ValueError):
        HyperRange("x", -1.0, 1.0, 0.5)
    with pytest.raises(ValueError):
        HyperRange("x", 0.1, 1.0, 0.5, relative_to="nope")


def test_lattice_edges_match_design() -> None:
    nb = {("empty", "I_n0"), ("empty", "P"), ("I_n0", "IP"), ("P", "IP")}
    wb = {
        ("empty", "I"),
        ("empty", "D"),
        ("empty", "P_W"),
        ("I", "ID"),
        ("I", "IP_W"),
        ("D", "ID"),
        ("D", "DP"),
        ("P_W", "IP_W"),
        ("P_W", "DP"),
        ("ID", "IDP"),
        ("IP_W", "IDP"),
        ("DP", "IDP"),
    }
    omni = {
        ("empty", "I-o"),
        ("empty", "D-o"),
        ("empty", "P-o"),
        ("I-o", "ID-o"),
        ("I-o", "IP-o"),
        ("D-o", "ID-o"),
        ("D-o", "DP-o"),
        ("P-o", "IP-o"),
        ("P-o", "DP-o"),
        ("ID-o", "IDP-o"),
        ("IP-o", "IDP-o"),
        ("DP-o", "IDP-o"),
    }
    assert set(LATTICE_EDGES["NB"]) == nb
    assert set(LATTICE_EDGES["WB"]) == wb
    assert set(LATTICE_EDGES["omni"]) == omni
    assert len(LATTICE_EDGES["NB"]) == 4
    assert len(LATTICE_EDGES["WB"]) == 12
    assert len(LATTICE_EDGES["omni"]) == 12
    for lattice in LATTICE_NAMES:
        for sync_mode in ("S", "N"):
            mapping = lattice_configs(lattice, sync_mode)
            letters = {"empty": set()}
            for node, name in mapping.items():
                letters[node] = set(get_config(name).subset)
            expected = {
                (a, b)
                for a in letters
                for b in letters
                if letters[a] < letters[b] and len(letters[b]) == len(letters[a]) + 1
            }
            assert set(LATTICE_EDGES[lattice]) == expected
    assert cfg_mod.INVARIANCE_EDGES == frozenset({("D", "DP"), ("D-o", "DP-o")})
    assert cfg_mod.INVARIANCE_EDGES <= set(LATTICE_EDGES["WB"]) | set(LATTICE_EDGES["omni"])


def test_lattice_configs() -> None:
    seen: set[str] = set()
    for lattice in LATTICE_NAMES:
        for sync_mode in ("S", "N"):
            mapping = lattice_configs(lattice, sync_mode)
            assert set(mapping) == set(LATTICE_NODES[lattice]) - {"empty"}
            for node, name in mapping.items():
                cfg = get_config(name)
                assert cfg.node == node
                assert cfg.sync in (sync_mode, "any")
                assert lattice in cfg.lattices
                seen.add(name)
    for cfg in CONFIGS.values():
        if cfg.lattices:
            assert cfg.name in seen
    assert lattice_configs("WB", "N")["I"] == "I-N"
    assert lattice_configs("NB", "S")["I_n0"] == "I@n0"
    assert lattice_configs("omni", "N")["ID-o"] == "ID-o-N"
    with pytest.raises(ValueError):
        lattice_configs("nope", "S")
    with pytest.raises(ValueError):
        lattice_configs("WB", "N_sep")


def _extract_invariant(data_a: np.ndarray, data_b: np.ndarray) -> bool:
    """Return True when two extract outputs agree to 1e-9 relative (NaN-aware)."""
    a = np.asarray(data_a)
    b = np.asarray(data_b)
    if a.shape != b.shape:
        return False
    if a.size == 0:
        return True
    nan_a = np.isnan(a)
    nan_b = np.isnan(b)
    if not np.array_equal(nan_a, nan_b):
        return False
    valid = ~nan_a
    if not np.any(valid):
        return True
    scale = float(np.max(np.abs(a[valid])))
    diff = float(np.max(np.abs(a[valid] - b[valid])))
    return bool(diff <= 1e-9 * scale)


def test_gauge_sensitivity_matches_observables() -> None:
    scene = _scene()
    geom = scene.geom
    y = scene.tracks["ideal-S"]
    nv = scene.noise_var
    rng = np.random.default_rng(7)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=y.shape[:2])
    tau = rng.uniform(1e-9, 9e-9, size=y.shape[:2])
    yp = sync.apply_gauge(y, phi, np.zeros_like(tau), geom.freq_offsets)
    yt = sync.apply_gauge(y, np.zeros_like(phi), tau, geom.freq_offsets)
    params: dict[str, Any] = {"noise_var": nv, "delta_f": geom.delta_f}
    for node, sens in NODE_GAUGE_SENSITIVITY.items():
        base = extract(y, node, params).data
        inv_p = _extract_invariant(base, extract(yp, node, params).data)
        inv_t = _extract_invariant(base, extract(yt, node, params).data)
        assert ("phi" in sens) == (not inv_p), node
        assert ("tau" in sens) == (not inv_t), node
    for cfg in CONFIGS.values():
        assert cfg.node in NODE_GAUGE_SENSITIVITY
    empty_sens: dict[str, tuple[str, ...]] = {"empty": ()}
    for lattice in LATTICE_NAMES:
        sens_map = dict(empty_sens)
        for node in LATTICE_NODES[lattice]:
            if node != "empty":
                sens_map[node] = NODE_GAUGE_SENSITIVITY[node]
        for sub, sup in LATTICE_EDGES[lattice]:
            assert set(sens_map[sub]) <= set(sens_map[sup]), (lattice, sub, sup)


def test_gauge_unknowns() -> None:
    assert get_config("I-N").gauge_unknowns == ()
    assert get_config("ID-N").gauge_unknowns == ("tau",)
    assert get_config("P-N").gauge_unknowns == ("phi",)
    assert get_config("IDP-N").gauge_unknowns == ("phi", "tau")
    assert get_config("IDP-S_tau").gauge_unknowns == ("phi",)
    assert get_config("IDP-o-N").gauge_unknowns == ("tau",)
    assert get_config("IDP-S").gauge_unknowns == ()
    assert get_config("IP_W").gauge_unknowns == ()
    for cfg in CONFIGS.values():
        if cfg.sync == "any":
            assert cfg.gauge_unknowns == ()


def test_n_strategies() -> None:
    for cfg in CONFIGS.values():
        if cfg.sync in ("S", "any"):
            assert cfg.n_strategies == ()
        names = [s.name for s in cfg.n_strategies]
        assert len(set(names)) == len(names)
        for strategy in cfg.n_strategies:
            assert strategy.name in STRATEGY_NAMES
            assert strategy.estimates
            if cfg.sync == "S_tau":
                # S_tau strategies reuse the complex estimators, which also return the
                # (known-zero) tau alongside phi; they deliver at least the unknowns.
                assert set(cfg.gauge_unknowns) <= set(strategy.estimates)
            else:
                assert set(strategy.estimates) <= set(cfg.gauge_unknowns)
    expected = {
        "ID-N": {"los", "xcorr"},
        "DP-N": {"los", "blind", "self_cal"},
        "IDP-N": {"los", "blind", "self_cal", "varpro"},
        "IDP-S_tau": {"los", "self_cal", "varpro"},
        "ID-o-N": {"xcorr"},
        "D-N": set(),
        "P-N": set(),
        "IP-N": set(),
        "I-N": set(),
        "ID-N_sep": {"los", "xcorr"},
        "DP-N_sep": {"los", "blind", "self_cal"},
        "IDP-N_sep": {"los", "blind", "self_cal", "varpro"},
        "DP-S_tau": {"los", "self_cal"},
        "D-N_sep": set(),
        "P-N_sep": set(),
        "IP-N_sep": set(),
    }
    for name, names in expected.items():
        assert {s.name for s in get_config(name).n_strategies} == names, name


def test_every_chain_runs_on_micro_scene() -> None:
    scene = _scene()
    geom, grid, nv = scene.geom, scene.grid, scene.noise_var
    for cfg in CONFIGS.values():
        if cfg.e1 is None:
            continue
        y, gauges = _track_data(cfg)
        for space in SPACES:
            out = run_e1(cfg, y, geom, grid, space, gauges=gauges, noise_var=nv)
            assert out.shape == grid.shape and out.dtype == np.float64
            assert bool(np.all(np.isfinite(out))) and bool(np.any(out != 0.0))
    roi_cfgs = [cfg for cfg in CONFIGS.values() if cfg.roi is not None]
    assert len(roi_cfgs) >= 10
    for cfg in roi_cfgs:
        y, gauges = _track_data(cfg)
        result = run_roi(
            cfg,
            y,
            geom,
            scene.pts + [0.05, -0.05, 0.05],
            "bv",
            gauges=gauges,
            noise_var=nv,
        )
        assert result.positions.shape == (2, 3)
        assert bool(np.all(np.isfinite(result.positions)))
    support_in = run_e1(get_config("ID-S"), scene.tracks["ideal-S"], geom, grid, "bv", noise_var=nv)
    indices, points, edges = run_support(support_in, grid)
    assert points.shape[0] >= 2
    for cfg in CONFIGS.values():
        y, gauges = _track_data(cfg)
        for step in cfg.e2:
            for space in step.spaces:
                out = run_e2(
                    cfg,
                    step.name,
                    y,
                    geom,
                    points,
                    space,
                    gauges=gauges,
                    noise_var=nv,
                    n_iter=5,
                    edges=edges,
                )
                assert out.density.shape == (points.shape[0],)
                assert out.density.dtype == np.float64
                assert bool(np.all(np.isfinite(out.density)))
                assert bool(np.all(out.density >= 0.0))
                assert float(np.max(out.density)) > 0.0
                if step.call == "coherent_per_bin":
                    if step.operator["bins"] == "all":
                        assert len(out.results) == geom.num_bins
                    else:
                        assert len(out.results) == 4
                else:
                    assert len(out.results) == 1


def test_dispatch_matches_direct_calls() -> None:
    scene = _scene()
    geom, grid, nv = scene.geom, scene.grid, scene.noise_var
    y = scene.tracks["ideal-S"]
    assert np.array_equal(
        run_e1(get_config("ID-S"), y, geom, grid, "bv", noise_var=nv),
        bp.power_map(y, geom, grid, "bv", "ID", noise_var=nv),
    )
    assert np.array_equal(
        run_e1(get_config("IDP-S"), y, geom, grid, "bv", noise_var=nv),
        bp.envelope_map(y, geom, grid.centers(), "bv", "IDP", noise_var=nv).reshape(grid.shape),
    )
    support_in = run_e1(get_config("ID-S"), y, geom, grid, "bv", noise_var=nv)
    _, points, edges = run_support(support_in, grid)
    out = run_e2(get_config("ID-S"), "kl_em", y, geom, points, "bv", noise_var=nv, n_iter=5)
    assert np.array_equal(
        out.density,
        power.kl_em(
            kernels.power_operator(points, geom, "bv", "ID"),
            extract(y, "ID").data.ravel(),
            kernels.noise_floor("ID", geom, nv),
            n_iter=5,
        ).x,
    )
    out = run_e2(
        get_config("IDP-S"),
        "tikhonov_lsqr",
        y,
        geom,
        points,
        "bv",
        noise_var=nv,
        hyper={"damp": 0.2},
        n_iter=5,
    )
    op = SeparableOperator(points, geom, "bv")
    yd = bp.node_data(y, "IDP", noise_var=nv)
    assert np.array_equal(
        out.density,
        np.abs(
            coherent.tikhonov_lsqr(
                op,
                yd,
                0.2 * np.sqrt(coherent.lipschitz_constant(op, safety=1.0)),
                iter_lim=5,
            ).x
        )
        ** 2,
    )
    n0 = geom.num_bins // 2
    out = run_e2(
        get_config("IP-S"), "complex_l1_fista", y, geom, points, "bv", noise_var=nv, n_iter=5
    )
    op_n0 = SeparableOperator(points, geom, "bv", bins=(n0,))
    yd_ip = bp.node_data(y, "IP", noise_var=nv)
    assert np.array_equal(
        out.density,
        coherent.point_density(
            coherent.complex_l1_fista(
                op_n0,
                yd_ip[..., [n0]],
                0.05 * coherent.lambda_max(op_n0, yd_ip[..., [n0]], group=False),
                n_iter=5,
            ).x,
            "shared",
        ),
    )
    with pytest.raises(ValueError):
        run_e2(get_config("ID-S"), "nope", y, geom, points, "bv", noise_var=nv, n_iter=5)
    with pytest.raises(ValueError):
        run_e1(get_config("ID-S"), y, geom, grid, "bv", noise_var=nv, hyper={"nope": 1.0})
    with pytest.raises(ValueError):
        run_e2(
            get_config("ID-S"),
            "kl_em",
            y,
            geom,
            points,
            "bv",
            noise_var=nv,
            hyper={"nope": 1.0},
            n_iter=5,
        )
    with pytest.raises(ValueError):
        run_e2(
            get_config("IDP-S"),
            "mmv_constrained",
            y,
            geom,
            points,
            "bv",
            noise_var=nv,
            n_iter=5,
            edges=edges,
        )
    vs_out = run_e2(
        get_config("IDP-S"),
        "mmv_constrained",
        y,
        geom,
        points,
        "vs",
        noise_var=nv,
        n_iter=5,
        edges=edges,
    )
    assert vs_out.density.shape == (points.shape[0],)
    with pytest.raises(ValueError):
        run_e1(get_config("IDP-o-S"), y, geom, grid, "bv", noise_var=nv)
    with pytest.raises(ValueError):
        run_e2(get_config("ID-S"), "kl_em", y, geom, points, "bv", n_iter=5)
    with pytest.raises(ValueError):
        run_e2(get_config("ID-S"), "nn_fista_tv", y, geom, points, "bv", noise_var=nv, n_iter=5)


def test_no_statistic_outside_the_node() -> None:
    scene = _scene()
    geom, grid, nv = scene.geom, scene.grid, scene.noise_var
    n0 = geom.num_bins // 2
    sigma = float(np.std(scene.tracks["ideal-S"]))
    rng = np.random.default_rng(11)

    def perturbed(y: np.ndarray, mask: np.ndarray) -> np.ndarray:
        noise = rng.standard_normal(y.shape) + 1j * rng.standard_normal(y.shape)
        noise = noise * (sigma / np.sqrt(2.0))
        out = y.copy()
        out[mask] += noise[mask]
        return out

    for name in ("P-S", "IP-S", "P-N", "IP-N", "I@n0"):
        cfg = get_config(name)
        y, gauges = _track_data(cfg)
        keep = np.zeros(y.shape[-1], dtype=bool)
        keep[n0] = True
        y2 = perturbed(y, np.broadcast_to(~keep, y.shape))
        assert np.array_equal(
            run_e1(cfg, y2, geom, grid, "bv", gauges=gauges, noise_var=nv),
            run_e1(cfg, y, geom, grid, "bv", gauges=gauges, noise_var=nv),
        )
        first = cfg.e2[0].name
        assert np.array_equal(
            run_e2(
                cfg, first, y2, geom, scene.pts, "bv", gauges=gauges, noise_var=nv, n_iter=5
            ).density,
            run_e2(
                cfg, first, y, geom, scene.pts, "bv", gauges=gauges, noise_var=nv, n_iter=5
            ).density,
        )
        if cfg.roi is not None:
            det = scene.pts + [0.05, -0.05, 0.05]
            assert np.array_equal(
                run_roi(cfg, y2, geom, det, "bv", gauges=gauges, noise_var=nv).positions,
                run_roi(cfg, y, geom, det, "bv", gauges=gauges, noise_var=nv).positions,
            )
    for name in ("PxK-S", "IPxK-S"):
        cfg = get_config(name)
        y, gauges = _track_data(cfg)
        bins = list(extract(y, "IPxK").meta["bins"])
        keep = np.zeros(y.shape[-1], dtype=bool)
        keep[bins] = True
        y2 = perturbed(y, np.broadcast_to(~keep, y.shape))
        assert np.array_equal(
            run_e1(cfg, y2, geom, grid, "bv", gauges=gauges, noise_var=nv),
            run_e1(cfg, y, geom, grid, "bv", gauges=gauges, noise_var=nv),
        )
        for step in cfg.e2:
            assert np.array_equal(
                run_e2(
                    cfg, step.name, y2, geom, scene.pts, "bv", gauges=gauges, noise_var=nv, n_iter=5
                ).density,
                run_e2(
                    cfg, step.name, y, geom, scene.pts, "bv", gauges=gauges, noise_var=nv, n_iter=5
                ).density,
            )
    for name in ("IDP-1el", "DP-1el"):
        cfg = get_config(name)
        y, gauges = _track_data(cfg)
        element = extract(y, "IDP-1el").meta["element"]
        keep = np.zeros(y.shape[3:5], dtype=bool)
        keep[element] = True
        full_keep = np.broadcast_to(keep[None, None, None, :, :, None], y.shape)
        y2 = perturbed(y, ~full_keep)
        assert np.array_equal(
            run_e1(cfg, y2, geom, grid, "bv", gauges=gauges, noise_var=nv),
            run_e1(cfg, y, geom, grid, "bv", gauges=gauges, noise_var=nv),
        )
        det = scene.pts + [0.05, -0.05, 0.05]
        assert np.array_equal(
            run_roi(cfg, y2, geom, det, "bv", gauges=gauges, noise_var=nv).positions,
            run_roi(cfg, y, geom, det, "bv", gauges=gauges, noise_var=nv).positions,
        )
    cfg = get_config("P-S")
    y, _ = _track_data(cfg)
    n0_mask = np.broadcast_to((np.arange(y.shape[-1]) == n0).reshape(1, 1, 1, 1, 1, -1), y.shape)
    changed = run_e1(cfg, perturbed(y, n0_mask), geom, grid, "bv", noise_var=nv)
    assert _relative(run_e1(cfg, y, geom, grid, "bv", noise_var=nv), changed) > 1e-3
    cfg_s = get_config("IDP-S")
    rows, cols = y.shape[3:5]
    other = (0, 0) if extract(y, "IDP-1el").meta["element"] != (0, 0) else (0, 1)
    assert 0 <= other[0] < rows and 0 <= other[1] < cols
    el_mask = np.zeros(y.shape[3:5], dtype=bool)
    el_mask[other] = True
    el_full = np.broadcast_to(el_mask[None, None, None, :, :, None], y.shape)
    changed = run_e1(cfg_s, perturbed(y, el_full), geom, grid, "bv", noise_var=nv)
    assert _relative(run_e1(cfg_s, y, geom, grid, "bv", noise_var=nv), changed) > 1e-3


def test_sync_invariant_configs_equal_in_s_and_n() -> None:
    scene = _scene()
    geom, grid, nv = scene.geom, scene.grid, scene.noise_var
    ys, yn = scene.tracks["ideal-S"], scene.tracks["ideal-N"]
    support_in = run_e1(get_config("ID-S"), ys, geom, grid, "bv", noise_var=nv)
    _, points, _ = run_support(support_in, grid)
    invariant = [
        cfg
        for cfg in CONFIGS.values()
        if cfg.e1 is not None and NODE_GAUGE_SENSITIVITY[cfg.node] == ()
    ]
    assert {cfg.name for cfg in invariant} >= {"I-S", "I@n0", "P_W", "IP_W", "I-o"}
    for cfg in invariant:
        assert (
            _relative(
                run_e1(cfg, ys, geom, grid, "bv", noise_var=nv),
                run_e1(cfg, yn, geom, grid, "bv", noise_var=nv),
            )
            <= 1e-9
        )
        first = cfg.e2[0].name
        assert (
            _relative(
                run_e2(cfg, first, ys, geom, points, "bv", noise_var=nv, n_iter=5).density,
                run_e2(cfg, first, yn, geom, points, "bv", noise_var=nv, n_iter=5).density,
            )
            <= 1e-8
        )
    assert (
        _relative(
            run_e1(get_config("IDP-N"), yn, geom, grid, "bv", noise_var=nv),
            run_e1(get_config("IDP-S"), ys, geom, grid, "bv", noise_var=nv),
        )
        > 1e-2
    )


def test_n_mode_true_gauges_reproduce_s() -> None:
    scene = _scene()
    geom, grid, nv = scene.geom, scene.grid, scene.noise_var
    ys, yn = scene.tracks["ideal-S"], scene.tracks["ideal-N"]
    gauges = scene.gt["gauges"]["ideal-N"]
    support_in = run_e1(get_config("ID-S"), ys, geom, grid, "bv", noise_var=nv)
    _, points, _ = run_support(support_in, grid)
    det = scene.pts + [0.05, -0.05, 0.05]
    for node in ("DP", "IDP"):
        cfg_s, cfg_n = get_config(f"{node}-S"), get_config(f"{node}-N")
        assert (
            _relative(
                run_e1(cfg_n, yn, geom, grid, "bv", gauges=gauges, noise_var=nv),
                run_e1(cfg_s, ys, geom, grid, "bv", noise_var=nv),
            )
            <= 1e-9
        )
        assert (
            _relative(
                run_e2(
                    cfg_n,
                    "tikhonov_lsqr",
                    yn,
                    geom,
                    points,
                    "bv",
                    gauges=gauges,
                    noise_var=nv,
                    n_iter=5,
                ).density,
                run_e2(
                    cfg_s, "tikhonov_lsqr", ys, geom, points, "bv", noise_var=nv, n_iter=5
                ).density,
            )
            <= 1e-8
        )
        pos_n = run_roi(cfg_n, yn, geom, det, "bv", gauges=gauges, noise_var=nv).positions
        pos_s = run_roi(cfg_s, ys, geom, det, "bv", noise_var=nv).positions
        assert float(np.max(np.abs(pos_n - pos_s))) <= 1e-9
