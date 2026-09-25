"""Declarative registry of every tomography baseline configuration.

This module lists the 40 configurations of design §1.2 (core, completion,
omni, 1el, partial-D and sync-variant columns over the §2.3 nodes and
lattices), each as a chain of :class:`Step` rows: an E1 map, an optional ROI
refinement, the shared support pruning and the E2 solver alternatives of §4.2.
The N-mode gauge strategies of §2.4 and the tuning ranges of §6.5 live here as
well.

Call conventions (section A.4)
------------------------------
Every :class:`Step` is executed by exactly one of :func:`run_e1`,
:func:`run_roi`, :func:`run_support` or :func:`run_e2`:

====================  ==========================================================================
call                  exact call
====================  ==========================================================================
``grid``              ``fn(Yr, geom, grid, space, **kwargs, **values, tau_hat=tau,
                      noise_var=noise_var)``; ``tau_hat`` only when gauges are given
                      and ``fn`` takes it, ``noise_var`` only when ``fn`` takes it
``points``            as ``grid`` but with ``grid.centers()``; result reshaped to
                      ``grid.shape``
``power_grid``        ``fn(extract(Yr, PRODUCT_NODES[product]).data, geom, grid, tau,
                      ``space=space, **kwargs)`` with ``tau = gauges[1]`` or None
``roi``               ``Yd`` de-gauged when gauges are given, else ``Yr``;
                      ``fn(Yd, geom, detections, space, **kwargs, noise_var=noise_var)``
``power``             ``op = power_operator(points, geom, space, product, tau=tau)``;
                      ``y`` the product extract ravelled; ``background =
                      noise_floor(product, geom, noise_var)``;
                      ``fn(op, y, background, **kwargs, **values, n_iter)``
                      (plus ``edges=edges`` when ``kwargs["tv"]``)
``coherent``          ``bins`` from the operator entry (``None``/``(n0,)``/partial);
                      ``y`` the ``data_node`` slice; ``op = SeparableOperator(points,
                      geom, space, beta_model=..., gauges=gauges, bins=bins)``;
                      relative hyper values resolved per operator and data;
                      ``fn(op, y, **kwargs, **values, n_iter/iter_lim)``;
                      ``density = point_density(res.x, beta_model)``
``coherent_per_bin``  one ``coherent`` call per bin in ascending order; the density
                      is the mean over bins, ``results`` holds every per-bin result
====================  ==========================================================================

``Yr`` is the restricted data (:func:`_restricted`); relative scales are
``sigma_max`` (× ``sqrt(lipschitz_constant(op, safety=1.0))``), ``lambda_max``
(× ``lambda_max(op, y, group=True)``) and ``lambda_max_l1`` (×
``lambda_max(op, y, group=False)``). Whether a configuration is ill-posed is
computed by ``identifiability.py`` at run time and is deliberately not a field
here. NumPy/SciPy only: nothing here may import Sionna, Mitsuba or Dr.Jit.
"""

from __future__ import annotations

import inspect
import types
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from plateau_rt.domain.rf_tomography import gauges as _gauges
from plateau_rt.domain.rf_tomography import kernels as _kernels
from plateau_rt.domain.rf_tomography import sync as _sync
from plateau_rt.domain.rf_tomography.forward_sep import SeparableOperator
from plateau_rt.domain.rf_tomography.geometry import CaptureGeometry, VoxelGrid
from plateau_rt.domain.rf_tomography.observables import extract
from plateau_rt.domain.rf_tomography.solvers import bp as _bp
from plateau_rt.domain.rf_tomography.solvers import coherent as _coherent
from plateau_rt.domain.rf_tomography.solvers import power as _power

SYNC_MODES: tuple[str, ...] = ("S", "N", "S_tau", "N_sep", "any")
SYNC_UNKNOWNS: dict[str, tuple[str, ...]] = {
    "S": (),
    "N": ("phi", "tau"),
    "S_tau": ("phi",),
    "N_sep": ("phi", "tau"),
    "any": (),
}
COLUMNS: tuple[str, ...] = ("core", "completion", "omni", "1el", "partial_d", "sync")
SUBSETS: tuple[str, ...] = ("I", "D", "P", "ID", "IP", "DP", "IDP")
LATTICE_NAMES: tuple[str, ...] = ("NB", "WB", "omni")
EMPTY_NODE: str = "empty"
SPACES: tuple[str, ...] = ("bv", "vs")
BUDGETS: tuple[str, ...] = ("Pw", "Co")
STAGES: tuple[str, ...] = ("E1", "ROI", "support", "E2")
CALLS: tuple[str, ...] = (
    "grid",
    "points",
    "power_grid",
    "roi",
    "support",
    "power",
    "coherent",
    "coherent_per_bin",
)
RESTRICTIONS: tuple[str | None, ...] = (None, "element", "partial")
BIN_SELECTIONS: tuple[str, ...] = ("all", "n0", "partial")
RELATIVE_SCALES: tuple[str | None, ...] = (None, "sigma_max", "lambda_max", "lambda_max_l1")
STRATEGY_NAMES: tuple[str, ...] = ("los", "blind", "xcorr", "self_cal", "varpro")
TUNING_TRIALS: int = 20
E2_ITERATIONS: int = 200

NODE_GAUGE_SENSITIVITY: dict[str, tuple[str, ...]] = {
    "I": (),
    "I_n0": (),
    "D": ("tau",),
    "P": ("phi",),
    "P_W": (),
    "ID": ("tau",),
    "IP": ("phi",),
    "IP_W": (),
    "DP": ("phi", "tau"),
    "IDP": ("phi", "tau"),
    "I-o": (),
    "D-o": ("tau",),
    "P-o": (),
    "ID-o": ("tau",),
    "IP-o": (),
    "DP-o": ("tau",),
    "IDP-o": ("tau",),
    "IDP-1el": ("phi", "tau"),
    "DP-1el": ("phi", "tau"),
    "PxK": ("phi", "tau"),
    "IPxK": ("phi", "tau"),
}

LATTICE_NODES: dict[str, tuple[str, ...]] = {
    "NB": ("empty", "I_n0", "P", "IP"),
    "WB": ("empty", "I", "D", "P_W", "ID", "IP_W", "DP", "IDP"),
    "omni": ("empty", "I-o", "D-o", "P-o", "ID-o", "IP-o", "DP-o", "IDP-o"),
}
LATTICE_EDGES: dict[str, tuple[tuple[str, str], ...]] = {
    "NB": (("empty", "I_n0"), ("empty", "P"), ("I_n0", "IP"), ("P", "IP")),
    "WB": (
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
    ),
    "omni": (
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
    ),
}
INVARIANCE_EDGES: frozenset[tuple[str, str]] = frozenset({("D", "DP"), ("D-o", "DP-o")})

CORE_CONFIGS = (
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
)
COMPLETION_CONFIGS = ("I@n0", "P_W", "IP_W")
OMNI_CONFIGS = (
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
)
ONE_ELEMENT_CONFIGS = ("IDP-1el", "DP-1el")
PARTIAL_D_CONFIGS = ("PxK-S", "IPxK-S")
SYNC_VARIANT_CONFIGS = (
    "DP-S_tau",
    "IDP-S_tau",
    "D-N_sep",
    "P-N_sep",
    "ID-N_sep",
    "IP-N_sep",
    "DP-N_sep",
    "IDP-N_sep",
)
CONFIG_ALIASES: dict[str, str] = {
    "I_n0": "I@n0",
    "P×K-S": "PxK-S",
    "IP×K-S": "IPxK-S",
    "DP-S_τ": "DP-S_tau",
    "IDP-S_τ": "IDP-S_tau",
    "D-N-sep": "D-N_sep",
    "P-N-sep": "P-N_sep",
    "ID-N-sep": "ID-N_sep",
    "IP-N-sep": "IP-N_sep",
    "DP-N-sep": "DP-N_sep",
    "IDP-N-sep": "IDP-N_sep",
}


@dataclass(frozen=True)
class HyperRange:
    """Log/linear tuning range of one step keyword (design §6.5)."""

    name: str
    low: float
    high: float
    default: float
    log: bool = True
    relative_to: str | None = None

    def __post_init__(self) -> None:
        """Validate finiteness, ordering, log-positivity and the scale name."""
        low = float(self.low)
        high = float(self.high)
        default = float(self.default)
        if not np.isfinite(low) or not np.isfinite(high) or not np.isfinite(default):
            raise ValueError("low, high and default must be finite")
        if not low < high:
            raise ValueError("require low < high")
        if not low <= default <= high:
            raise ValueError("require low <= default <= high")
        if bool(self.log) and low <= 0.0:
            raise ValueError("log ranges require low > 0")
        if self.relative_to not in RELATIVE_SCALES:
            raise ValueError(f"relative_to must be one of {RELATIVE_SCALES}")
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)
        object.__setattr__(self, "default", default)
        object.__setattr__(self, "log", bool(self.log))

    def trials(self, n: int = TUNING_TRIALS) -> np.ndarray:
        """Return ``n`` float64 trials spanning ``[low, high]``."""
        if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or int(n) < 2:
            raise ValueError("n must be an integer >= 2")
        count = int(n)
        if self.log:
            return np.asarray(np.geomspace(self.low, self.high, count), dtype=np.float64)
        return np.asarray(np.linspace(self.low, self.high, count), dtype=np.float64)


@dataclass(frozen=True)
class Step:
    """One callable row of a configuration chain."""

    name: str
    stage: str
    call: str
    fn: Callable[..., Any]
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    operator: Mapping[str, Any] = field(default_factory=dict)
    hyper: tuple[HyperRange, ...] = ()
    spaces: tuple[str, ...] = SPACES

    def __post_init__(self) -> None:
        """Freeze mappings and validate stage, call, spaces and the callable."""
        object.__setattr__(self, "kwargs", types.MappingProxyType(dict(self.kwargs)))
        object.__setattr__(self, "operator", types.MappingProxyType(dict(self.operator)))
        object.__setattr__(self, "hyper", tuple(self.hyper))
        object.__setattr__(self, "spaces", tuple(self.spaces))
        if self.stage not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}, got {self.stage!r}")
        if self.call not in CALLS:
            raise ValueError(f"call must be one of {CALLS}, got {self.call!r}")
        if not self.spaces or any(space not in SPACES for space in self.spaces):
            raise ValueError(f"spaces must be a non-empty subset of {SPACES}")
        if not callable(self.fn):
            raise ValueError("fn must be callable")

    @property
    def qualname(self) -> str:
        """Return ``f"{fn.__module__}.{fn.__qualname__}"``."""
        return f"{self.fn.__module__}.{self.fn.__qualname__}"

    def defaults(self) -> dict[str, float]:
        """Return ``{h.name: h.default for h in hyper}``."""
        return {h.name: h.default for h in self.hyper}


@dataclass(frozen=True)
class NStrategy:
    """One N-mode gauge estimation strategy of a configuration."""

    name: str
    fn: Callable[..., Any]
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    estimates: tuple[str, ...] = ()
    aux: Mapping[str, tuple[Callable[..., Any], Mapping[str, Any]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Freeze the kwargs and aux mappings."""
        object.__setattr__(self, "kwargs", types.MappingProxyType(dict(self.kwargs)))
        object.__setattr__(self, "aux", types.MappingProxyType(dict(self.aux)))
        object.__setattr__(self, "estimates", tuple(self.estimates))
        if self.name not in STRATEGY_NAMES:
            raise ValueError(f"name must be one of {STRATEGY_NAMES}, got {self.name!r}")
        if not callable(self.fn):
            raise ValueError("fn must be callable")
        if not self.estimates or any(e not in ("phi", "tau") for e in self.estimates):
            raise ValueError("estimates must be a non-empty subset of ('phi', 'tau')")


@dataclass(frozen=True)
class Config:
    """One tomography configuration: an observable, a sync mode and a solver chain."""

    name: str
    node: str
    subset: str
    sync: str
    column: str
    lattices: frozenset[str]
    spaces: tuple[str, ...]
    budget: str
    e1: Step | None
    roi: Step | None = None
    e2: tuple[Step, ...] = ()
    n_strategies: tuple[NStrategy, ...] = ()
    restrict: str | None = None
    planned: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Freeze the set/tuple fields and validate the categorical fields."""
        object.__setattr__(self, "lattices", frozenset(self.lattices))
        object.__setattr__(self, "spaces", tuple(self.spaces))
        object.__setattr__(self, "e2", tuple(self.e2))
        object.__setattr__(self, "n_strategies", tuple(self.n_strategies))
        object.__setattr__(self, "planned", tuple(self.planned))
        if self.subset not in SUBSETS:
            raise ValueError(f"subset must be one of {SUBSETS}, got {self.subset!r}")
        if self.sync not in SYNC_MODES:
            raise ValueError(f"sync must be one of {SYNC_MODES}, got {self.sync!r}")
        if self.column not in COLUMNS:
            raise ValueError(f"column must be one of {COLUMNS}, got {self.column!r}")
        if any(lattice not in LATTICE_NAMES for lattice in self.lattices):
            raise ValueError(f"lattices must be a subset of {LATTICE_NAMES}")
        if self.budget not in BUDGETS:
            raise ValueError(f"budget must be one of {BUDGETS}, got {self.budget!r}")
        if self.restrict not in RESTRICTIONS:
            raise ValueError(f"restrict must be one of {RESTRICTIONS}")

    @property
    def gauge_unknowns(self) -> tuple[str, ...]:
        """Ordered (phi, tau) unknowns: sync unknowns that the node is sensitive to."""
        sync_unknowns = SYNC_UNKNOWNS[self.sync]
        sensitivity = NODE_GAUGE_SENSITIVITY[self.node]
        return tuple(u for u in ("phi", "tau") if u in sync_unknowns and u in sensitivity)

    @property
    def chain(self) -> tuple[Step, ...]:
        """Return ``(e1, roi, SUPPORT, *e2)`` skipping None (SUPPORT only with e2)."""
        parts: list[Step] = []
        if self.e1 is not None:
            parts.append(self.e1)
        if self.roi is not None:
            parts.append(self.roi)
        if self.e2:
            parts.append(SUPPORT)
            parts.extend(self.e2)
        return tuple(parts)


SUPPORT = Step(
    "prune_support",
    "support",
    "support",
    _power.prune_support,
    kwargs={"radius": _power.DEFAULT_RADIUS_M, "cap": _power.DEFAULT_CAP},
)


_FLOOR = HyperRange("floor", 1e-4, 1e-1, 1e-3)
_HIT = HyperRange("p_hit", 0.5, 0.95, 0.7, log=False)
_FALSE = HyperRange("p_false", 0.01, 0.2, 0.05)
_POWER_LAM = HyperRange("lam", 1e-3, 0.3, 0.03)
_TV_WEIGHT = HyperRange("tv_weight", 1e-5, 1e-1, 1e-3)
_DAMP = HyperRange("damp", 1e-3, 1.0, 0.1, relative_to="sigma_max")
_L1_LAM = HyperRange("lam", 1e-3, 0.5, 0.05, relative_to="lambda_max_l1")
_MMV_LAM = HyperRange("lam", 1e-3, 0.5, 0.05, relative_to="lambda_max")


def _e1_grid_intensity(node: str) -> Step:
    """Return the E1 ``grid`` intensity-map step for ``node``."""
    return Step(
        "intensity_map", "E1", "grid", _bp.intensity_map, kwargs={"node": node}, hyper=(_FLOOR,)
    )


def _e1_grid_splat() -> Step:
    """Return the E1 ``grid`` return-splatting step."""
    return Step("splat_returns", "E1", "grid", _bp.splat_returns, hyper=(_HIT, _FALSE))


def _e1_grid_power(node: str) -> Step:
    """Return the E1 ``grid`` power-map step for ``node``."""
    return Step("power_map", "E1", "grid", _bp.power_map, kwargs={"node": node})


def _e1_points_envelope(node: str) -> Step:
    """Return the E1 ``points`` envelope-map step for ``node``."""
    return Step("envelope_map", "E1", "points", _bp.envelope_map, kwargs={"node": node})


def _e1_power_grid(product: str) -> Step:
    """Return the E1 ``power_grid`` back-projection step for ``product``."""
    return Step(
        "power_backproject_grid",
        "E1",
        "power_grid",
        _kernels.power_backproject_grid,
        kwargs={"product": product},
    )


def _roi_step(node: str) -> Step:
    """Return the ROI refinement step for ``node``."""
    return Step("roi_refine", "ROI", "roi", _bp.roi_refine, kwargs={"node": node})


def _coh(
    name: str,
    fn: Callable[..., Any],
    node: str,
    beta: str,
    bins: str,
    hyper: tuple[HyperRange, ...],
    spaces: tuple[str, ...] = SPACES,
) -> Step:
    """Build one full/DC-bin coherent-E2 step."""
    return Step(
        name,
        "E2",
        "coherent",
        fn,
        operator={"data_node": node, "beta_model": beta, "bins": bins},
        hyper=hyper,
        spaces=spaces,
    )


def _coh_pb(
    name: str,
    fn: Callable[..., Any],
    node: str,
    beta: str,
    bins: str,
    hyper: tuple[HyperRange, ...],
) -> Step:
    """Build one per-bin coherent-E2 step."""
    return Step(
        name,
        "E2",
        "coherent_per_bin",
        fn,
        operator={"data_node": node, "beta_model": beta, "bins": bins},
        hyper=hyper,
    )


def _power_e2(product: str) -> tuple[Step, ...]:
    """Return the four power-E2 solver steps for ``product``."""
    op = {"product": product}
    return (
        Step("kl_em", "E2", "power", _power.kl_em, operator=op),
        Step("is_mlem", "E2", "power", _power.is_mlem, operator=op),
        Step(
            "nn_fista_l1",
            "E2",
            "power",
            _power.nn_fista_l1,
            kwargs={"tv": False},
            operator=op,
            hyper=(_POWER_LAM,),
        ),
        Step(
            "nn_fista_tv",
            "E2",
            "power",
            _power.nn_fista_l1,
            kwargs={"tv": True},
            operator=op,
            hyper=(_POWER_LAM, _TV_WEIGHT),
        ),
    )


def _coh_full(data_node: str) -> tuple[Step, ...]:
    """Return the four full-band coherent-E2 solver steps."""
    return (
        _coh("tikhonov_lsqr", _coherent.tikhonov_lsqr, data_node, "shared", "all", (_DAMP,)),
        _coh(
            "complex_l1_fista", _coherent.complex_l1_fista, data_node, "shared", "all", (_L1_LAM,)
        ),
        _coh("mmv_per_view", _coherent.mmv_group_lasso, data_node, "per_view", "all", (_MMV_LAM,)),
        _coh(
            "mmv_constrained",
            _coherent.mmv_group_lasso,
            data_node,
            "constrained",
            "all",
            (_MMV_LAM,),
            spaces=("vs",),
        ),
    )


def _coh_n0_s(data_node: str) -> tuple[Step, ...]:
    """Return the three DC-bin coherent-E2 solver steps (S mode)."""
    return (
        _coh("tikhonov_lsqr", _coherent.tikhonov_lsqr, data_node, "shared", "n0", (_DAMP,)),
        _coh("complex_l1_fista", _coherent.complex_l1_fista, data_node, "shared", "n0", (_L1_LAM,)),
        _coh("mmv_per_view", _coherent.mmv_group_lasso, data_node, "per_view", "n0", (_MMV_LAM,)),
    )


def _coh_n0_n(data_node: str) -> tuple[Step, ...]:
    """Return the DC-bin per-view coherent-E2 step (N mode)."""
    return (
        _coh("mmv_per_view", _coherent.mmv_group_lasso, data_node, "per_view", "n0", (_MMV_LAM,)),
    )


def _per_bin_w(data_node: str) -> tuple[Step, ...]:
    """Return the per-bin coherent-E2 steps over all bins (``_W`` nodes)."""
    return (
        _coh_pb("tikhonov_lsqr", _coherent.tikhonov_lsqr, data_node, "per_view", "all", (_DAMP,)),
        _coh_pb(
            "mmv_per_view", _coherent.mmv_group_lasso, data_node, "per_view", "all", (_MMV_LAM,)
        ),
    )


def _per_bin_k(data_node: str) -> tuple[Step, ...]:
    """Return the per-bin coherent-E2 steps over the partial-D bins."""
    return (
        _coh_pb("tikhonov_lsqr", _coherent.tikhonov_lsqr, data_node, "shared", "partial", (_DAMP,)),
        _coh_pb(
            "complex_l1_fista",
            _coherent.complex_l1_fista,
            data_node,
            "shared",
            "partial",
            (_L1_LAM,),
        ),
    )


LOS_POWER = NStrategy("los", _gauges.fit_los_ground, {"mode": "power"}, ("tau",))
LOS_COMPLEX = NStrategy("los", _gauges.fit_los_ground, {"mode": "complex"}, ("phi", "tau"))
XCORR = NStrategy("xcorr", _gauges.power_xcorr_delay, {}, ("tau",))
SELF_CAL = NStrategy("self_cal", _gauges.self_calibrate, {}, ("phi", "tau"))
VARPRO = NStrategy("varpro", _gauges.varpro_cost_and_grad, {}, ("phi", "tau"))


def _blind(node: str) -> NStrategy:
    """Return the blind delay-search strategy anchored on ``node``."""
    return NStrategy(
        "blind",
        _bp.blind_tau_search,
        {},
        ("tau",),
        aux={"bp_fn": (_bp.envelope_bp_fn, {"node": node, "kind": "tricubic"})},
    )


_E3 = "E3 CLEAN/OMP + NOMP (T23)"
_E1_OMNI = "E1 ellipsoid back-projection (omni)"
_E2_OCC = "E2 occupancy fit"
_N_ANCHOR = "N: LoS anchor on D returns"
_N_ENTROPY = "N: min-entropy cross-view search"
_N_PSEUDO = "N: pseudoranges (T24c)"
_N_NARROW = "N: narrowband LoS anchor"
_E3_ESPRIT = "E3 2D ESPRIT (T24a)"
_E2_SUBSET = "E2 element-subset operator"
_E2_IRLS = "E2 projected-normal IRLS"


def _cfg(
    name: str,
    node: str,
    subset: str,
    sync: str,
    column: str,
    lattices: str,
    budget: str,
    e1: Step | None,
    *,
    roi: Step | None = None,
    e2: tuple[Step, ...] = (),
    strategies: tuple[NStrategy, ...] = (),
    restrict: str | None = None,
    planned: tuple[str, ...] = (),
) -> Config:
    """Build one :class:`Config` from short keyword arguments."""
    return Config(
        name,
        node,
        subset,
        sync,
        column,
        frozenset({lattices}) if lattices else frozenset(),
        SPACES,
        budget,
        e1,
        roi,
        e2,
        strategies,
        restrict,
        planned,
    )


def _make_configs() -> dict[str, Config]:
    """Build the 40 configurations in report order."""
    cfgs: dict[str, Config] = {}
    # fmt: off
    # Tabular registry: one short _cfg call per config, kept packed for readability
    # (ruff check still enforces the 100-column limit on these lines).
    cfgs["I-S"] = _cfg("I-S", "I", "I", "S", "core", "WB", "Pw", _e1_grid_intensity("I"),
                        e2=_power_e2("I"), planned=(_E3,))
    cfgs["I-N"] = _cfg("I-N", "I", "I", "N", "core", "WB", "Pw", _e1_grid_intensity("I"),
                        e2=_power_e2("I"), planned=(_E3,))
    cfgs["D-S"] = _cfg("D-S", "D", "D", "S", "core", "WB", "Pw", _e1_grid_splat(),
                        planned=(_E3, _E2_OCC))
    cfgs["D-N"] = _cfg("D-N", "D", "D", "N", "core", "WB", "Pw", _e1_grid_splat(),
                        planned=(_E3, _E2_OCC, _N_ANCHOR, _N_ENTROPY, _N_PSEUDO))
    cfgs["P-S"] = _cfg("P-S", "P", "P", "S", "core", "NB", "Co", _e1_points_envelope("P"),
                       roi=_roi_step("P"), e2=_coh_n0_s("P"), planned=(_E3, _E2_IRLS))
    cfgs["P-N"] = _cfg("P-N", "P", "P", "N", "core", "NB", "Pw", _e1_points_envelope("P"),
                        e2=_coh_n0_n("P"), planned=(_E3, _N_NARROW, _E3_ESPRIT))
    cfgs["ID-S"] = _cfg("ID-S", "ID", "ID", "S", "core", "WB", "Pw", _e1_grid_power("ID"),
                         e2=_power_e2("ID"), planned=(_E3,))
    cfgs["ID-N"] = _cfg("ID-N", "ID", "ID", "N", "core", "WB", "Pw", _e1_grid_power("ID"),
                         e2=_power_e2("ID"), strategies=(LOS_POWER, XCORR),
                         planned=(_E3, _N_PSEUDO))
    cfgs["IP-S"] = _cfg("IP-S", "IP", "IP", "S", "core", "NB", "Co",
                        _e1_points_envelope("IP"), roi=_roi_step("IP"), e2=_coh_n0_s("IP"),
                        planned=(_E3,))
    cfgs["IP-N"] = _cfg("IP-N", "IP", "IP", "N", "core", "NB", "Co",
                         _e1_points_envelope("IP"), e2=_coh_n0_n("IP"),
                         planned=(_E3, _N_NARROW, _E3_ESPRIT))
    cfgs["DP-S"] = _cfg("DP-S", "DP", "DP", "S", "core", "WB", "Co",
                         _e1_points_envelope("DP"), roi=_roi_step("DP"), e2=_coh_full("DP"),
                         planned=(_E3, _E2_IRLS))
    cfgs["DP-N"] = _cfg("DP-N", "DP", "DP", "N", "core", "WB", "Co",
                         _e1_points_envelope("DP"), roi=_roi_step("DP"), e2=_coh_full("DP"),
                         strategies=(LOS_COMPLEX, _blind("DP"), SELF_CAL),
                         planned=(_E3, _N_PSEUDO))
    cfgs["IDP-S"] = _cfg("IDP-S", "IDP", "IDP", "S", "core", "WB", "Co",
                          _e1_points_envelope("IDP"), roi=_roi_step("IDP"),
                          e2=_coh_full("IDP"), planned=(_E3,))
    cfgs["IDP-N"] = _cfg("IDP-N", "IDP", "IDP", "N", "core", "WB", "Co",
                          _e1_points_envelope("IDP"), roi=_roi_step("IDP"),
                          e2=_coh_full("IDP"),
                          strategies=(LOS_COMPLEX, _blind("IDP"), SELF_CAL, VARPRO),
                          planned=(_E3, _N_PSEUDO))
    cfgs["I@n0"] = _cfg("I@n0", "I_n0", "I", "any", "completion", "NB", "Pw",
                         _e1_grid_intensity("I_n0"), e2=_power_e2("I_n0"), planned=(_E3,))
    cfgs["P_W"] = _cfg("P_W", "P_W", "P", "any", "completion", "WB", "Pw",
                        _e1_points_envelope("P_W"), e2=_per_bin_w("P_W"), planned=(_E3,))
    cfgs["IP_W"] = _cfg("IP_W", "IP_W", "IP", "any", "completion", "WB", "Co",
                         _e1_points_envelope("IP_W"), e2=_per_bin_w("IP_W"), planned=(_E3,))
    cfgs["I-o"] = _cfg("I-o", "I-o", "I", "any", "omni", "omni", "Pw",
                       _e1_power_grid("I_omni"), e2=_power_e2("I_omni"))
    cfgs["D-o-S"] = _cfg("D-o-S", "D-o", "D", "S", "omni", "omni", "Pw", None, planned=(_E1_OMNI,))
    cfgs["D-o-N"] = _cfg("D-o-N", "D-o", "D", "N", "omni", "omni", "Pw", None, planned=(_E1_OMNI,))
    cfgs["P-o"] = _cfg("P-o", "P-o", "P", "any", "omni", "omni", "Pw", None, planned=(_E1_OMNI,))
    cfgs["ID-o-S"] = _cfg("ID-o-S", "ID-o", "ID", "S", "omni", "omni", "Pw",
                           _e1_power_grid("ID_omni"), e2=_power_e2("ID_omni"))
    cfgs["ID-o-N"] = _cfg("ID-o-N", "ID-o", "ID", "N", "omni", "omni", "Pw",
                           _e1_power_grid("ID_omni"), e2=_power_e2("ID_omni"),
                           strategies=(XCORR,))
    cfgs["IP-o"] = _cfg(
        "IP-o", "IP-o", "IP", "any", "omni", "omni", "Pw", None, planned=(_E1_OMNI,)
    )
    cfgs["DP-o-S"] = _cfg(
        "DP-o-S", "DP-o", "DP", "S", "omni", "omni", "Pw", None, planned=(_E1_OMNI,)
    )
    cfgs["DP-o-N"] = _cfg(
        "DP-o-N", "DP-o", "DP", "N", "omni", "omni", "Pw", None, planned=(_E1_OMNI,)
    )
    cfgs["IDP-o-S"] = _cfg(
        "IDP-o-S", "IDP-o", "IDP", "S", "omni", "omni", "Pw", None, planned=(_E1_OMNI,)
    )
    cfgs["IDP-o-N"] = _cfg(
        "IDP-o-N", "IDP-o", "IDP", "N", "omni", "omni", "Pw", None, planned=(_E1_OMNI,)
    )
    cfgs["IDP-1el"] = _cfg("IDP-1el", "IDP-1el", "IDP", "S", "1el", "", "Pw",
                            _e1_points_envelope("IDP"), roi=_roi_step("IDP"),
                            restrict="element", planned=(_E2_SUBSET,))
    cfgs["DP-1el"] = _cfg("DP-1el", "DP-1el", "DP", "S", "1el", "", "Pw",
                           _e1_points_envelope("DP"), roi=_roi_step("DP"),
                           restrict="element", planned=(_E2_SUBSET,))
    cfgs["PxK-S"] = _cfg("PxK-S", "PxK", "P", "S", "partial_d", "", "Co",
                          _e1_points_envelope("P_W"), e2=_per_bin_k("DP"),
                          restrict="partial")
    cfgs["IPxK-S"] = _cfg("IPxK-S", "IPxK", "IP", "S", "partial_d", "", "Co",
                           _e1_points_envelope("IP_W"), e2=_per_bin_k("IDP"),
                           restrict="partial")
    for base_name in ("DP-N", "IDP-N"):
        base = cfgs[base_name]
        tau_name = base_name[:-2] + "-S_tau"
        phi_only = tuple(s for s in base.n_strategies if "phi" in s.estimates)
        cfgs[tau_name] = replace(base, name=tau_name, sync="S_tau", column="sync",
                                 lattices=frozenset(), n_strategies=phi_only,
                                 planned=(_E3,))
    for base_name in ("D-N", "P-N", "ID-N", "IP-N", "DP-N", "IDP-N"):
        base = cfgs[base_name]
        sep_name = f"{base_name}_sep"
        cfgs[sep_name] = replace(base, name=sep_name, sync="N_sep", column="sync",
                                 lattices=frozenset())
    # fmt: on
    return cfgs


CONFIGS: Mapping[str, Config] = types.MappingProxyType(_make_configs())


def get_config(name: str) -> Config:
    """Return the config ``name`` after resolving :data:`CONFIG_ALIASES`."""
    resolved = CONFIG_ALIASES.get(name, name)
    try:
        return CONFIGS[resolved]
    except KeyError as error:
        raise ValueError(f"unknown configuration {name!r}") from error


def lattice_configs(lattice: str, sync: str) -> dict[str, str]:
    """Map every non-empty node of ``lattice`` to its config for ``sync``."""
    if lattice not in LATTICE_NAMES:
        raise ValueError(f"unknown lattice {lattice!r}")
    if sync not in ("S", "N"):
        raise ValueError(f"sync must be 'S' or 'N', got {sync!r}")
    mapping: dict[str, str] = {}
    for node in LATTICE_NODES[lattice]:
        if node == EMPTY_NODE:
            continue
        matches = [
            name
            for name, cfg in CONFIGS.items()
            if cfg.node == node and lattice in cfg.lattices and cfg.sync in (sync, "any")
        ]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one config for node {node!r}, got {matches}")
        mapping[node] = matches[0]
    return mapping


def _restricted(cfg: Config, Y: np.ndarray) -> np.ndarray:
    """Return the restricted data for ``cfg`` (``Y`` itself is not modified)."""
    arr = np.asarray(Y, dtype=np.complex128)
    if cfg.restrict is None:
        return arr
    out = np.zeros_like(arr)
    if cfg.restrict == "element":
        element = extract(arr, "IDP-1el").meta["element"]
        out[..., element[0], element[1], :] = arr[..., element[0], element[1], :]
        return out
    if cfg.restrict == "partial":
        bins = extract(arr, "IPxK").meta["bins"]
        out[..., list(bins)] = arr[..., list(bins)]
        return out
    raise ValueError(f"unknown restriction {cfg.restrict!r}")


def _hyper_values(step: Step, hyper: Mapping[str, float] | None) -> dict[str, float]:
    """Return ``step.defaults()`` updated with ``hyper`` (unknown names rejected)."""
    values = step.defaults()
    if hyper:
        unknown = set(hyper) - set(values)
        if unknown:
            raise ValueError(f"unknown hyperparameter(s): {sorted(unknown)}")
        values.update({key: float(value) for key, value in hyper.items()})
    return values


def _check_space(cfg: Config, step: Step | None, space: str) -> None:
    """Raise ``ValueError`` unless ``space`` is supported by ``cfg`` and ``step``."""
    if space not in cfg.spaces:
        raise ValueError(f"space {space!r} not in config spaces {cfg.spaces}")
    if step is not None and space not in step.spaces:
        raise ValueError(f"space {space!r} not in step spaces {step.spaces}")


def _iteration_kwarg(fn: Callable[..., Any], n_iter: int) -> dict[str, int]:
    """Return ``{"n_iter": n}`` or ``{"iter_lim": n}`` depending on ``fn``."""
    params = inspect.signature(fn).parameters
    if "n_iter" in params:
        return {"n_iter": int(n_iter)}
    if "iter_lim" in params:
        return {"iter_lim": int(n_iter)}
    raise ValueError(f"solver {getattr(fn, '__name__', fn)!r} takes neither n_iter nor iter_lim")


def run_e1(
    cfg: Config,
    Y: np.ndarray,
    geom: CaptureGeometry,
    grid: VoxelGrid,
    space: str,
    *,
    gauges: tuple[np.ndarray, np.ndarray] | None = None,
    noise_var: float | None = None,
    hyper: Mapping[str, float] | None = None,
) -> np.ndarray:
    """Run the E1 step of ``cfg`` and return the float64 map of ``grid.shape``."""
    if cfg.e1 is None:
        raise ValueError(f"config {cfg.name!r} has no E1 step")
    _check_space(cfg, cfg.e1, space)
    step = cfg.e1
    values = _hyper_values(step, hyper)
    yr = _restricted(cfg, Y)
    tau = gauges[1] if gauges is not None else None
    params = dict(step.kwargs)
    params.update(values)
    signature = inspect.signature(step.fn).parameters
    if step.call in ("grid", "points"):
        if gauges is not None and "tau_hat" in signature:
            params["tau_hat"] = tau
        if "noise_var" in signature:
            params["noise_var"] = noise_var
        if step.call == "grid":
            out = step.fn(yr, geom, grid, space, **params)
        else:
            out = step.fn(yr, geom, grid.centers(), space, **params)
        return np.asarray(out, dtype=np.float64).reshape(grid.shape)
    if step.call == "power_grid":
        data = extract(yr, _kernels.PRODUCT_NODES[step.kwargs["product"]]).data
        out = step.fn(data, geom, grid, tau, space=space, **dict(step.kwargs))
        return np.asarray(out, dtype=np.float64).reshape(grid.shape)
    raise ValueError(f"unknown E1 call {step.call!r}")


def run_roi(
    cfg: Config,
    Y: np.ndarray,
    geom: CaptureGeometry,
    detections: np.ndarray,
    space: str,
    *,
    gauges: tuple[np.ndarray, np.ndarray] | None = None,
    noise_var: float | None = None,
) -> _bp.RoiResult:
    """Run the ROI step of ``cfg`` on (de-gauged) data."""
    if cfg.roi is None:
        raise ValueError(f"config {cfg.name!r} has no ROI step")
    _check_space(cfg, cfg.roi, space)
    yr = _restricted(cfg, Y)
    if gauges is not None:
        phi, tau = gauges
        yd = _sync.apply_gauge(yr, -phi, -tau, geom.freq_offsets)
    else:
        yd = yr
    step = cfg.roi
    return step.fn(yd, geom, detections, space, **dict(step.kwargs), noise_var=noise_var)


def run_support(density: np.ndarray, grid: VoxelGrid) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prune ``density`` to a support and return ``(indices, points, edges)``."""
    indices = SUPPORT.fn(density, grid, **SUPPORT.kwargs)
    return (
        np.asarray(indices),
        np.asarray(_power.support_points(grid, indices), dtype=np.float64),
        np.asarray(_power.support_edges(grid, indices)),
    )


@dataclass(frozen=True)
class E2Output:
    """Output of one E2 solve: the nonnegative density and the raw result(s)."""

    density: np.ndarray
    results: tuple[Any, ...]


def _resolve_relative(
    step: Step, values: dict[str, float], op: SeparableOperator, y: np.ndarray
) -> dict[str, float]:
    """Return ``values`` with relative scales resolved against ``op`` and ``y``."""
    scales = {h.name: h.relative_to for h in step.hyper}
    absolute = dict(values)
    for key, scale in scales.items():
        if scale is None:
            continue
        if scale == "sigma_max":
            factor = float(np.sqrt(_coherent.lipschitz_constant(op, safety=1.0)))
        elif scale == "lambda_max":
            factor = float(_coherent.lambda_max(op, y, group=True))
        elif scale == "lambda_max_l1":
            factor = float(_coherent.lambda_max(op, y, group=False))
        else:  # pragma: no cover - validated by HyperRange
            raise ValueError(f"unknown relative scale {scale!r}")
        absolute[key] = float(values[key]) * factor
    return absolute


def _selected_bins(step: Step, yr: np.ndarray) -> tuple[int, ...] | None:
    """Return the bin tuple of a coherent step (None means all bins, else ascending)."""
    selection = step.operator["bins"]
    if selection == "all":
        return None
    if selection == "n0":
        return (int(np.asarray(yr).shape[-1] // 2),)
    if selection == "partial":
        return tuple(sorted(int(b) for b in extract(np.asarray(yr), "IPxK").meta["bins"]))
    raise ValueError(f"unknown bin selection {selection!r}")


def _solve_coherent_bin(
    step: Step,
    op: SeparableOperator,
    y: np.ndarray,
    values: dict[str, float],
    n_iter: int,
) -> Any:
    """Run one coherent solver call with resolved relative values."""
    absolute = _resolve_relative(step, values, op, y)
    call_kwargs = dict(step.kwargs)
    call_kwargs.update(absolute)
    call_kwargs.update(_iteration_kwarg(step.fn, n_iter))
    return step.fn(op, y, **call_kwargs)


def run_e2(
    cfg: Config,
    solver: str,
    Y: np.ndarray,
    geom: CaptureGeometry,
    points: np.ndarray,
    space: str,
    *,
    gauges: tuple[np.ndarray, np.ndarray] | None = None,
    noise_var: float | None = None,
    hyper: Mapping[str, float] | None = None,
    n_iter: int | None = None,
    edges: np.ndarray | None = None,
) -> E2Output:
    """Run the E2 solver ``solver`` of ``cfg`` and return the density and results."""
    matches = [step for step in cfg.e2 if step.name == solver]
    if not matches:
        raise ValueError(f"unknown solver {solver!r} for config {cfg.name!r}")
    step = matches[0]
    _check_space(cfg, step, space)
    values = _hyper_values(step, hyper)
    budget = E2_ITERATIONS if n_iter is None else int(n_iter)
    yr = _restricted(cfg, Y)
    tau = gauges[1] if gauges is not None else None
    pts = np.asarray(points, dtype=np.float64)
    if step.call == "power":
        product = step.operator["product"]
        op = _kernels.power_operator(pts, geom, space, product, tau=tau)
        y = extract(yr, _kernels.PRODUCT_NODES[product]).data.ravel()
        if noise_var is None:
            raise ValueError("power E2 requires noise_var")
        background = _kernels.noise_floor(product, geom, noise_var)
        call_kwargs: dict[str, Any] = dict(step.kwargs)
        call_kwargs.update(values)
        if call_kwargs.get("tv"):
            if edges is None:
                raise ValueError("tv solvers require edges")
            call_kwargs["edges"] = np.asarray(edges)
        call_kwargs.update(_iteration_kwarg(step.fn, budget))
        result = step.fn(op, y, background, **call_kwargs)
        density = np.asarray(result.x, dtype=np.float64).reshape(-1)
        return E2Output(density=density, results=(result,))
    if step.call == "coherent":
        data_node = step.operator["data_node"]
        beta_model = step.operator["beta_model"]
        bins = _selected_bins(step, yr)
        data = _bp.node_data(yr, data_node, noise_var=noise_var)
        y = data if bins is None else data[..., list(bins)]
        op = SeparableOperator(pts, geom, space, beta_model=beta_model, gauges=gauges, bins=bins)
        result = _solve_coherent_bin(step, op, y, values, budget)
        density = np.asarray(
            _coherent.point_density(result.x, beta_model), dtype=np.float64
        ).reshape(-1)
        return E2Output(density=density, results=(result,))
    if step.call == "coherent_per_bin":
        data_node = step.operator["data_node"]
        beta_model = step.operator["beta_model"]
        data = _bp.node_data(yr, data_node, noise_var=noise_var)
        selected = _selected_bins(step, yr)
        if selected is None:
            bin_list = list(range(int(np.asarray(yr).shape[-1])))
        else:
            bin_list = list(selected)
        densities: list[np.ndarray] = []
        results: list[Any] = []
        for single in bin_list:
            y_bin = data[..., [single]]
            op_bin = SeparableOperator(
                pts, geom, space, beta_model=beta_model, gauges=gauges, bins=(single,)
            )
            res = _solve_coherent_bin(step, op_bin, y_bin, values, budget)
            results.append(res)
            densities.append(
                np.asarray(_coherent.point_density(res.x, beta_model), dtype=np.float64).reshape(-1)
            )
        density = np.asarray(np.mean(densities, axis=0), dtype=np.float64).reshape(-1)
        return E2Output(density=density, results=tuple(results))
    raise ValueError(f"unknown E2 call {step.call!r}")
