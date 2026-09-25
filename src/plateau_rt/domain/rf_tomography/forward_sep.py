"""Separable exact forward operator for RF tomography point sets.

The operator maps complex amplitudes (one value per point for the ``"shared"``
model, per point and capture for ``"per_view"``, or a quadratic incidence-cosine
basis per point for ``"constrained"``) to the multi-view aperture CFR
``Y[V, B, 2, R, C, N]``. Per capture ``c = (v, b)`` and hemisphere ``h`` it is the
exact factorisation

    Y_{c,h} = A_ang,c,h * diag(gamma_c,h * beta_c,h) * D_c,h^T,

with the element steering ``A[m, p] = exp(+1j k q_m . u_p)``, the delay
``D[n, p] = exp(-2j pi df_n tau_p)`` and the per-point ``gamma`` from
:func:`forward_exact.capture_factors`; only points of hemisphere ``h`` enter
``Y_{c,h}``. The gauge ``(phi, tau)`` enters as a unit-modulus diagonal on the
bin axis, ``g_c[n] = exp(1j phi_[v, b]) exp(-2j pi df_n tau[v, b])``. The
operator uses BLAS GEMMs only (no dense matrix), has an exact adjoint with
respect to ``numpy.vdot`` and is exposed as a
:class:`scipy.sparse.linalg.LinearOperator`.

The operator may be restricted to a subset of frequency bins via the
keyword-only ``bins`` parameter of :class:`SeparableOperator` (``None`` keeps
today's full-band behaviour exactly). Note that
``gauges.varpro_cost_and_grad`` / ``gauges.self_calibrate`` expect a
full-band operator.

This is the ``wavefront="plane"`` carrier-only model of
``forward_exact.atom_cfr`` (``squint=False``); the spherical-wavefront and
squint mismatch sweeps use ``forward_exact``. NumPy/SciPy only: nothing here may
import Sionna, Mitsuba or Dr.Jit. All lengths are metres, angles radians,
frequencies hertz.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import LinearOperator

from plateau_rt.domain.rf_tomography.forward_exact import _prepare_points, capture_factors
from plateau_rt.domain.rf_tomography.geometry import (
    LOS_VS_TOLERANCE_M,
    CaptureGeometry,
)
from plateau_rt.domain.rf_tomography.sync import gauge_factor

BETA_MODELS: tuple[str, ...] = ("shared", "per_view", "constrained")
CONSTRAINED_DEGREE: int = 2
CACHE_MAX_BYTES: int = 4 * 1024**3


def incidence_cosine(points: np.ndarray, geom: CaptureGeometry, v: int, b: int) -> np.ndarray:
    """Return ``|cos theta_inc|`` [P] of first-order virtual sources.

    For a virtual source ``s`` the arrival direction is ``d = (p_v - s) / |p_v - s|``
    and the mirror-plane normal of a first-order image of ``t_b`` is
    ``n = (s - t_b) / |s - t_b|``, so ``cos_inc = |d . n|`` in ``[0, 1]``. For the
    LoS (``|s - t_b| <= LOS_VS_TOLERANCE_M``) the value is exactly ``1.0``.
    """
    pts = _prepare_points(points)
    to_ue = geom.ue_pos[v] - pts
    range_ue = np.linalg.norm(to_ue, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        direction = to_ue / range_ue[:, None]

    to_bs = pts - geom.bs_pos[b]
    range_bs = np.linalg.norm(to_bs, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        normal = to_bs / range_bs[:, None]

    cos_inc = np.abs(np.einsum("pi,pi->p", direction, normal))
    los = range_bs <= LOS_VS_TOLERANCE_M
    cos_inc = np.where(los, 1.0, cos_inc)
    return np.clip(cos_inc, 0.0, 1.0)


def project_shared_phase(x: np.ndarray) -> np.ndarray:
    """Project constrained coefficients ``[P, K]`` onto ``e^{j psi_p} (real vector)``.

    This is the exact minimiser of ``||x - e^{j psi} b||`` over a shared phase
    ``psi`` and a real vector ``b``: ``psi = 0.5 * angle(sum_j x_j**2)`` and
    ``b = real(exp(-1j psi) x)``.
    """
    values = np.asarray(x)
    if values.ndim != 2:
        raise ValueError("x must be two-dimensional [P, K]")
    values = values.astype(np.complex128)
    phase = 0.5 * np.angle(np.sum(values**2, axis=1))
    projected = np.real(np.exp(-1j * phase)[:, None] * values)
    return np.exp(1j * phase)[:, None] * projected


@dataclass
class _Capture:
    """Precomputed per-capture factors and, optionally, the cached A and D."""

    v: int
    b: int
    perm: np.ndarray  # [P] permutation sorting points by hemisphere
    n_front: int  # number of points with hemisphere 0
    u: np.ndarray  # [P, 3] permuted UE-local arrival directions
    tau: np.ndarray  # [P] permuted delays in s
    gamma: np.ndarray  # [P] permuted complex128 factors
    basis: np.ndarray | None  # [P, K] permuted constrained basis, or None
    a: np.ndarray | None = None  # [M, P] cached element steering
    d: np.ndarray | None = None  # [N, P] cached delay matrix


def _validate_gauges(
    gauges: tuple[np.ndarray, np.ndarray] | None, num_views: int, num_bs: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Validate a gauge pair ``(phi[V, B], tau[V, B])``; return float64 copies."""
    if gauges is None:
        return None
    if not isinstance(gauges, tuple) or len(gauges) != 2:
        raise ValueError("gauges must be None or a pair (phi, tau)")
    phi = np.array(gauges[0], dtype=np.float64, copy=True)
    tau = np.array(gauges[1], dtype=np.float64, copy=True)
    if phi.shape != (num_views, num_bs) or tau.shape != (num_views, num_bs):
        raise ValueError(f"gauges must both have shape {(num_views, num_bs)}")
    if not np.all(np.isfinite(phi)) or not np.all(np.isfinite(tau)):
        raise ValueError("gauges must contain only finite values")
    return phi, tau


def _validate_bins(
    bins: Sequence[int] | np.ndarray | None, num_bins: int
) -> tuple[int, ...] | None:
    """Return ``bins`` as a tuple of distinct bin indices, or None."""
    if bins is None:
        return None
    arr = np.asarray(bins)
    if arr.ndim != 1 or arr.size == 0 or arr.dtype.kind not in "iu":
        raise ValueError("bins must be distinct integers in [0, num_bins)")
    if np.any(arr < 0) or np.any(arr >= int(num_bins)):
        raise ValueError("bins must be distinct integers in [0, num_bins)")
    if np.unique(arr).size != arr.size:
        raise ValueError("bins must be distinct integers in [0, num_bins)")
    return tuple(int(b) for b in arr)


class SeparableOperator:
    """Exact separable plane-wave operator ``A_ang diag(gamma * beta) D^T``.

    With ``bins`` given, only those frequency bins are modelled: ``y_shape``
    becomes ``[V, B, 2, R, C, len(bins)]`` and ``freq_offsets`` the selected
    offsets. ``bins=None`` keeps the full-band behaviour exactly.
    """

    def __init__(
        self,
        points: np.ndarray,
        geom: CaptureGeometry,
        space: str,
        *,
        beta_model: str = "shared",
        gauges: tuple[np.ndarray, np.ndarray] | None = None,
        pattern: str = "tr38901",
        polarization: str = "none",
        cache: bool | None = None,
        bins: Sequence[int] | np.ndarray | None = None,
    ) -> None:
        if beta_model not in BETA_MODELS:
            raise ValueError(f"beta_model must be one of {BETA_MODELS}, got {beta_model!r}")
        if beta_model == "constrained" and space == "bv":
            raise ValueError("beta_model='constrained' is only defined in 'vs' space")

        validated_bins = _validate_bins(bins, geom.num_bins)
        self._geom = geom
        self._space = space
        self._beta_model = beta_model
        self._pattern = pattern
        self._polarization = polarization
        self._num_views = geom.num_views
        self._num_bs = geom.num_bs
        self._num_elements = geom.num_elements
        self._bins = validated_bins
        if validated_bins is None:
            self._num_bins = geom.num_bins
            self._df = geom.freq_offsets
        else:
            self._num_bins = len(validated_bins)
            selected = np.asarray(geom.freq_offsets, dtype=np.float64)[list(validated_bins)]
            selected.setflags(write=False)
            self._df = selected
        self._rows, self._cols = geom.aperture_shape
        self._q = geom.elem_offsets
        self._k = geom.wavenumber
        self._num_points = _prepare_points(points).shape[0]

        validated = _validate_gauges(gauges, self._num_views, self._num_bs)
        self._gauge: tuple[np.ndarray, np.ndarray] | None = validated

        self._captures: list[_Capture] = []
        for v in range(self._num_views):
            for b in range(self._num_bs):
                factors = capture_factors(
                    points,
                    geom,
                    space,
                    v,
                    b,
                    pattern=pattern,
                    polarization=polarization,
                )
                perm = np.argsort(factors.hemisphere, kind="stable")
                n_front = int(np.count_nonzero(factors.hemisphere == 0))
                basis = None
                if beta_model == "constrained":
                    cos_inc = incidence_cosine(points, geom, v, b)
                    basis = np.stack(
                        [cos_inc**degree for degree in range(CONSTRAINED_DEGREE + 1)], axis=1
                    )[perm]
                self._captures.append(
                    _Capture(
                        v=v,
                        b=b,
                        perm=perm,
                        n_front=n_front,
                        u=factors.u_local[perm],
                        tau=factors.tau[perm],
                        gamma=factors.gamma[perm],
                        basis=basis,
                    )
                )

        if beta_model == "shared":
            self._x_shape: tuple[int, ...] = (self._num_points,)
        elif beta_model == "per_view":
            self._x_shape = (self._num_points, self._num_views, self._num_bs)
        else:
            self._x_shape = (self._num_points, CONSTRAINED_DEGREE + 1)
        self._y_shape = (
            self._num_views,
            self._num_bs,
            2,
            self._rows,
            self._cols,
            self._num_bins,
        )
        self._shape = (int(np.prod(self._y_shape)), int(np.prod(self._x_shape)))

        if cache is None:
            estimate = (
                self._num_views
                * self._num_bs
                * self._num_points
                * (self._num_elements + self._num_bins)
                * 16
            )
            cache = estimate <= CACHE_MAX_BYTES
        self._cached = bool(cache)
        if self._cached:
            for capture in self._captures:
                capture.a = self._steering(capture)
                capture.d = self._delay(capture)

    @property
    def num_points(self) -> int:
        """Number of points ``P``."""
        return self._num_points

    @property
    def x_shape(self) -> tuple[int, ...]:
        """Shape of the amplitude vector ``x``."""
        return self._x_shape

    @property
    def beta_model(self) -> str:
        """Amplitude model: ``"shared"``, ``"per_view"`` or ``"constrained"``."""
        return self._beta_model

    @property
    def geom(self) -> CaptureGeometry:
        """Capture geometry the operator was built on."""
        return self._geom

    @property
    def y_shape(self) -> tuple[int, ...]:
        """Shape of the CFR ``Y``."""
        return self._y_shape

    @property
    def shape(self) -> tuple[int, int]:
        """Flat operator shape ``(prod(y_shape), prod(x_shape))``."""
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        """Operator dtype, ``complex128``."""
        return np.dtype(np.complex128)

    @property
    def cached(self) -> bool:
        """True when the per-capture ``A`` and ``D`` are stored."""
        return self._cached

    @property
    def bins(self) -> tuple[int, ...] | None:
        """Selected frequency bins in the given order, or None for full band."""
        return self._bins

    @property
    def freq_offsets(self) -> np.ndarray:
        """Selected frequency offsets (read-only float64 array)."""
        return self._df

    def _steering(self, capture: _Capture) -> np.ndarray:
        """Return ``exp(+1j k q_m . u_p)`` as ``[M, P]`` (permuted order)."""
        return np.exp(1j * self._k * (self._q @ capture.u.T))

    def _delay(self, capture: _Capture) -> np.ndarray:
        """Return ``exp(-2j pi df_n tau_p)`` as ``[N, P]`` (permuted order)."""
        return np.exp(-2j * np.pi * self._df[:, None] * capture.tau[None, :])

    def _capture_steering(self, capture: _Capture) -> np.ndarray:
        """Return the cached or freshly computed element steering ``[M, P]``."""
        if capture.a is not None:
            return capture.a
        return self._steering(capture)

    def _capture_delay(self, capture: _Capture) -> np.ndarray:
        """Return the cached or freshly computed delay matrix ``[N, P]``."""
        if capture.d is not None:
            return capture.d
        return self._delay(capture)

    def _gauge_vector(self, v: int, b: int) -> np.ndarray | None:
        """Return ``g_c[n]`` [N] or ``None`` when no gauge is set."""
        gauge = self._gauge
        if gauge is None:
            return None
        phi, tau = gauge
        return gauge_factor(phi[v, b], tau[v, b], self._df)

    def _beta(self, x: np.ndarray, capture: _Capture) -> np.ndarray:
        """Return the per-point amplitude ``beta_c[p]`` [P] (permuted order)."""
        if self._beta_model == "shared":
            return x[capture.perm]
        if self._beta_model == "per_view":
            return x[capture.perm, capture.v, capture.b]
        assert capture.basis is not None
        return np.einsum("pj,pj->p", capture.basis, x[capture.perm])

    def _check_forward_shape(self, x: np.ndarray) -> np.ndarray:
        """Return ``x`` as complex128 after checking its shape."""
        if np.shape(x) != self._x_shape:
            raise ValueError(f"x must have shape {self._x_shape}, got {np.shape(x)}")
        return np.asarray(x, dtype=np.complex128)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Return ``Y[V, B, 2, R, C, N]`` for amplitudes ``x`` of shape ``x_shape``."""
        values = self._check_forward_shape(x)
        result = np.zeros(self._y_shape, dtype=np.complex128)
        for capture in self._captures:
            steering = self._capture_steering(capture)
            delay = self._capture_delay(capture)
            weight = capture.gamma * self._beta(values, capture)
            gauge = self._gauge_vector(capture.v, capture.b)
            for h, (start, stop) in enumerate(
                ((0, capture.n_front), (capture.n_front, self._num_points))
            ):
                weighted = steering[:, start:stop] * weight[None, start:stop]
                block = (delay[:, start:stop] @ weighted.T).T
                if gauge is not None:
                    block = block * gauge[None, :]
                result[capture.v, capture.b, h] = block.reshape(
                    self._rows, self._cols, self._num_bins
                )
        return result

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        """Return the exact adjoint of :meth:`forward` for ``y`` of shape ``y_shape``."""
        if np.shape(y) != self._y_shape:
            raise ValueError(f"y must have shape {self._y_shape}, got {np.shape(y)}")
        values = np.asarray(y, dtype=np.complex128)
        result = np.zeros(self._x_shape, dtype=np.complex128)
        for capture in self._captures:
            steering = self._capture_steering(capture)
            delay = self._capture_delay(capture)
            gauge = self._gauge_vector(capture.v, capture.b)
            update = np.zeros(self._num_points, dtype=np.complex128)
            for h, (start, stop) in enumerate(
                ((0, capture.n_front), (capture.n_front, self._num_points))
            ):
                block = values[capture.v, capture.b, h].reshape(self._num_elements, self._num_bins)
                projected = np.conj(block)
                if gauge is not None:
                    projected = projected * gauge[None, :]
                transpose = projected @ delay[:, start:stop]
                summed = np.sum(steering[:, start:stop] * transpose, axis=0)
                update[start:stop] = np.conj(capture.gamma[start:stop]) * np.conj(summed)
            if self._beta_model == "shared":
                result[capture.perm] += update
            elif self._beta_model == "per_view":
                result[capture.perm, capture.v, capture.b] += update
            else:
                assert capture.basis is not None
                result[capture.perm] += capture.basis * update[:, None]
        return result

    def matvec(self, x: np.ndarray) -> np.ndarray:
        """Apply the flattening of :meth:`forward` to any ``x`` with ``x.size == shape[1]``."""
        values = np.asarray(x)
        if values.size != self._shape[1]:
            raise ValueError(f"x.size must be {self._shape[1]}, got {values.size}")
        return self.forward(values.reshape(self._x_shape)).reshape(-1)

    def rmatvec(self, y: np.ndarray) -> np.ndarray:
        """Apply the flattening of :meth:`adjoint` to any ``y`` with ``y.size == shape[0]``."""
        values = np.asarray(y)
        if values.size != self._shape[0]:
            raise ValueError(f"y.size must be {self._shape[0]}, got {values.size}")
        return self.adjoint(values.reshape(self._y_shape)).reshape(-1)

    def as_linear_operator(self) -> LinearOperator:
        """Return the operator as a :class:`scipy.sparse.linalg.LinearOperator`."""
        return LinearOperator(
            shape=self._shape,
            matvec=self.matvec,
            rmatvec=self.rmatvec,
            dtype=np.complex128,
        )

    def with_gauges(self, gauges: tuple[np.ndarray, np.ndarray] | None) -> SeparableOperator:
        """Return a copy sharing the cached factors but using ``gauges`` (``None`` clears)."""
        validated = _validate_gauges(gauges, self._num_views, self._num_bs)
        clone = copy.copy(self)
        clone._gauge = validated
        return clone
