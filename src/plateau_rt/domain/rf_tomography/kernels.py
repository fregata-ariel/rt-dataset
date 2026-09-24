"""Incoherent (power-domain) kernels for RF tomography.

Incoherent model (docs/tomography_baselines.md §3.3)
----------------------------------------------------
A point ``p`` carries a nonnegative power ``sigma_p = E|rho_p|^2`` (``"bv"``) or
``E|beta_p|^2`` (``"vs"``, view-independent); the point phases are independent
and uniform and the raw noise is ``CN(0, noise_var)``. For every power observable
``obs`` of :func:`observables.extract` the mean is

    E[obs] = K @ sigma + noise_floor(product, geom, noise_var),

with the real, nonnegative and separable operator ``K`` below. With deterministic
phases the model misses the cross terms of paths that share one angle-delay cell:
this is the known limitation of the incoherent model (docs §3.3).

Products
--------
``K`` is the 3-axis separable kernel over ``(u_y, u_z, t)``. For capture
``c = (v, b)``, hemisphere ``h`` and point ``p`` with
``f = capture_factors(points, geom, space, v, b, ...)``, ``w_p = |f.gamma_p|^2``,
``u_y = f.u_local[:, 1]``, ``u_z = f.u_local[:, 2]``, ``tau_tot = f.tau + tau_c``
and ``R, C = geom.aperture_shape``, ``M = R*C``, ``N = geom.num_bins``,
``df = geom.delta_f``:

    Ky[i, p] = dirichlet_power(s * u_y[p] - fy[i], C),  fy = fftshift(fftfreq(C))
    Kz[i, p] = dirichlet_power(s * u_z[p] - fz[i], R),  fz = fftshift(fftfreq(R))
    Kt[i, p] = dirichlet_power(i / N - df * tau_tot[p], N)

singleton axes contribute the factor ``1`` and the kernel entry is
``1[h_p == h] * const * w_p * Ky[iy, p] * Kz[iz, p] * Kt[it, p]``:

===========  =================  ==================================
product      extract node       observable shape (V, B, 2, ...)
===========  =================  ==================================
``ID``       ``ID``             ``(C, R, N)``, const 1
``I``        ``I``              ``(C, R)``, const ``N``
``I_n0``     ``I_n0``           ``(C, R)``, const 1
``ID_omni``  ``ID-o``           ``(N,)``, const ``M``
``I_omni``   ``I-o``            ``()``, const ``M * N``
===========  =================  ==================================

``dirichlet_power(x, L) = |sum_{l<L} exp(2j pi l x)|^2 / L`` is 1-periodic and
even; it is ``L`` at integer ``x`` and ``0`` at non-integer multiples of ``1/L``.
The element spacing ``s`` (in wavelengths) is derived from ``geom``; the kernels
only describe an ideal planar aperture.

Gauge (docs §2.4)
-----------------
``Y_obs[c] = exp(1j phi_c) exp(-2j pi df_n tau_c) Y[c]``. Power products do not
depend on ``phi``; ``tau_c`` delays every point of capture ``c``, so the delay
kernel uses ``tau_tot = tau + tau_c``. ``I``, ``I_n0`` and ``I_omni`` are exactly
tau-invariant; ``ID`` and ``ID_omni`` are not. ``tau=None`` equals
``tau=np.zeros((V, B))``.

Back-projection
---------------
:func:`power_backproject_grid` approximates ``K^T y`` on a
:class:`geometry.VoxelGrid` without building ``K``: a circular FFT correlation
onto an oversampled grid, then the T07a periodic interpolation
(:func:`interp.periodic_weights` / :func:`interp.gather`; trilinear default).
A voxel within ``LOS_VS_TOLERANCE_M`` of any UE (and of any BS in ``"bv"``) is
singular: its value is exactly 0 and it is never passed to ``capture_factors``.
In ``"vs"`` a voxel at a BS is the LoS source and is valid.

NumPy/SciPy only: nothing here may import Sionna, Mitsuba or Dr.Jit. All lengths
are metres, angles radians, frequencies hertz.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse.linalg import LinearOperator

from plateau_rt.domain.rf_tomography.backproject import _aperture_spacing, _singular_mask
from plateau_rt.domain.rf_tomography.forward_exact import (
    SPACES,
    _prepare_points,
    capture_factors,
)
from plateau_rt.domain.rf_tomography.geometry import CaptureGeometry, VoxelGrid
from plateau_rt.domain.rf_tomography.interp import (
    INTERP_KINDS,
    gather,
    periodic_weights,
)
from plateau_rt.domain.rf_tomography.observables import volume_axes

PRODUCTS: tuple[str, ...] = ("ID", "I", "I_n0", "ID_omni", "I_omni")
PRODUCT_NODES: dict[str, str] = {
    "ID": "ID",
    "I": "I",
    "I_n0": "I_n0",
    "ID_omni": "ID-o",
    "I_omni": "I-o",
}
CACHE_MAX_BYTES: int = 4 * 1024**3
POINT_CHUNK: int = 65_536
DEFAULT_OVERSAMPLE: int = 8

# Per product: whether the ``(y, z, t)`` axes carry the Dirichlet kernel.
_PRODUCT_AXES: dict[str, tuple[bool, bool, bool]] = {
    "ID": (True, True, True),
    "I": (True, True, False),
    "I_n0": (True, True, False),
    "ID_omni": (False, False, True),
    "I_omni": (False, False, False),
}


def _validate_product(product: str) -> None:
    """Raise ``ValueError`` unless ``product`` is one of :data:`PRODUCTS`."""
    if product not in PRODUCTS:
        raise ValueError(f"product must be one of {PRODUCTS}, got {product!r}")


def dirichlet_power(x: np.ndarray, length: int) -> np.ndarray:
    """Return ``|sum_{l < length} exp(2j pi l x)|^2 / length`` elementwise.

    Evaluated stably as ``length * (sinc(length * xr) / sinc(xr)) ** 2`` with
    ``xr = x - round(x)``. The result is float64 with the shape of ``x``,
    exactly ``length`` at integers and ``0`` at non-integer multiples of
    ``1 / length``.
    """
    if isinstance(length, (bool, np.bool_)) or not isinstance(length, (int, np.integer)):
        raise ValueError(f"length must be an integer >= 1, got {length!r}")
    count = int(length)
    if count < 1:
        raise ValueError(f"length must be an integer >= 1, got {length!r}")
    values = np.asarray(x, dtype=np.float64)
    xr = values - np.round(values)
    return count * (np.sinc(count * xr) / np.sinc(xr)) ** 2


def product_shape(product: str, geom: CaptureGeometry) -> tuple[int, ...]:
    """Return the full observable shape ``(V, B, 2, ...)`` of ``product``."""
    _validate_product(product)
    rows, cols = geom.aperture_shape
    base = (geom.num_views, geom.num_bs, 2)
    if product == "ID":
        return base + (cols, rows, geom.num_bins)
    if product in ("I", "I_n0"):
        return base + (cols, rows)
    if product == "ID_omni":
        return base + (geom.num_bins,)
    return base


def noise_floor(product: str, geom: CaptureGeometry, noise_var: float) -> float:
    """Return ``noise_dof / 2 * noise_var`` for ``product`` (``noise_dof`` as in extract)."""
    _validate_product(product)
    var = float(noise_var)
    if not np.isfinite(var) or var < 0.0:
        raise ValueError("noise_var must be finite and >= 0")
    num_elements = geom.num_elements
    num_bins = geom.num_bins
    dof = {
        "ID": 2,
        "I": 2 * num_bins,
        "I_n0": 2,
        "ID_omni": 2 * num_elements,
        "I_omni": 2 * num_elements * num_bins,
    }[product]
    return 0.5 * dof * var


def _product_const(product: str, geom: CaptureGeometry) -> float:
    """Return the scalar constant ``const`` of ``product``."""
    num_elements = geom.num_elements
    num_bins = geom.num_bins
    return {
        "ID": 1.0,
        "I": float(num_bins),
        "I_n0": 1.0,
        "ID_omni": float(num_elements),
        "I_omni": float(num_elements * num_bins),
    }[product]


def _capture_tau(tau: np.ndarray | None, num_views: int, num_bs: int) -> np.ndarray:
    """Return the validated capture gauge delays ``[V, B]`` float64 (zeros for ``None``)."""
    if tau is None:
        return np.zeros((num_views, num_bs), dtype=np.float64)
    delays = np.array(tau, dtype=np.float64, copy=True)
    if delays.shape != (num_views, num_bs):
        raise ValueError(f"tau must have shape {(num_views, num_bs)}, got {delays.shape}")
    if not np.all(np.isfinite(delays)):
        raise ValueError("tau must contain only finite values")
    return delays


@dataclass
class _CaptureData:
    """Per-capture kernel factors of :class:`PowerOperator`."""

    v: int
    b: int
    w: np.ndarray  # [P] |gamma|^2 * const
    front: np.ndarray  # [P0] indices of hemisphere 0 points
    back: np.ndarray  # [P1] indices of hemisphere 1 points
    uy: np.ndarray  # [P] local y direction cosine
    uz: np.ndarray  # [P] local z direction cosine
    tau_tot: np.ndarray  # [P] point delay plus capture gauge
    ky: np.ndarray | None = None  # [ny, P] cached
    kz: np.ndarray | None = None  # [nz, P] cached
    kt: np.ndarray | None = None  # [nt, P] cached


class PowerOperator:
    """Incoherent power operator ``E[obs] = K sigma + noise_floor``.

    The operator has shape ``(prod(product_shape), P)`` and is real float64. It
    is built from the per-capture :func:`forward_exact.capture_factors` and the
    separable Dirichlet kernels of the module docstring.
    """

    def __init__(
        self,
        points: np.ndarray,
        geom: CaptureGeometry,
        space: str,
        product: str = "ID",
        *,
        tau: np.ndarray | None = None,
        pattern: str = "tr38901",
        polarization: str = "none",
        cache: bool | None = None,
    ) -> None:
        _validate_product(product)
        pts = _prepare_points(points)
        self._geom = geom
        self._space = space
        self._product = product
        self._pattern = pattern
        self._polarization = polarization
        self._num_points = int(pts.shape[0])
        self._num_views = geom.num_views
        self._num_bs = geom.num_bs
        self._rows, self._cols = geom.aperture_shape
        self._num_bins = geom.num_bins
        self._delta_f = geom.delta_f
        self._spacing = _aperture_spacing(geom)
        self._tau = _capture_tau(tau, geom.num_views, geom.num_bs)

        self._y_dirichlet, self._z_dirichlet, self._t_dirichlet = _PRODUCT_AXES[product]
        self._ny = self._cols if self._y_dirichlet else 1
        self._nz = self._rows if self._z_dirichlet else 1
        self._nt = self._num_bins if self._t_dirichlet else 1
        self._const = _product_const(product, geom)
        self._fy = np.fft.fftshift(np.fft.fftfreq(self._cols))
        self._fz = np.fft.fftshift(np.fft.fftfreq(self._rows))
        self._bin_index = np.arange(self._num_bins, dtype=np.float64) / self._num_bins

        self._captures: list[_CaptureData] = []
        for v in range(self._num_views):
            for b in range(self._num_bs):
                factors = capture_factors(
                    pts,
                    geom,
                    space,
                    v,
                    b,
                    pattern=pattern,
                    polarization=polarization,
                )
                hemisphere = factors.hemisphere
                self._captures.append(
                    _CaptureData(
                        v=v,
                        b=b,
                        w=np.abs(factors.gamma) ** 2 * self._const,
                        front=np.flatnonzero(hemisphere == 0),
                        back=np.flatnonzero(hemisphere == 1),
                        uy=factors.u_local[:, 1],
                        uz=factors.u_local[:, 2],
                        tau_tot=factors.tau + self._tau[v, b],
                    )
                )

        self._x_shape: tuple[int] = (self._num_points,)
        self._y_shape = product_shape(product, geom)
        self._shape = (int(np.prod(self._y_shape)), self._num_points)

        if cache is None:
            estimate = (
                self._num_views
                * self._num_bs
                * self._num_points
                * (self._ny + self._nz + self._nt)
                * 8
            )
            cache = estimate <= CACHE_MAX_BYTES
        self._cached = bool(cache)
        if self._cached:
            for capture in self._captures:
                capture.ky, capture.kz, capture.kt = self._kernels(capture)

    @property
    def num_points(self) -> int:
        """Number of points ``P``."""
        return self._num_points

    @property
    def product(self) -> str:
        """Product name."""
        return self._product

    @property
    def x_shape(self) -> tuple[int]:
        """Shape ``(P,)`` of the power vector ``sigma``."""
        return self._x_shape

    @property
    def y_shape(self) -> tuple[int, ...]:
        """Shape of the observable."""
        return self._y_shape

    @property
    def shape(self) -> tuple[int, int]:
        """Flat operator shape ``(prod(y_shape), P)``."""
        return self._shape

    @property
    def dtype(self) -> np.dtype:
        """Operator dtype, ``float64``."""
        return np.dtype(np.float64)

    @property
    def cached(self) -> bool:
        """True when the per-capture kernels are stored."""
        return self._cached

    def _kernels(self, capture: _CaptureData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(Ky, Kz, Kt)`` of ``capture`` (singleton axes are all ones)."""
        if self._y_dirichlet:
            ky = dirichlet_power(
                self._spacing * capture.uy[None, :] - self._fy[:, None], self._cols
            )
        else:
            ky = np.ones((1, self._num_points), dtype=np.float64)
        if self._z_dirichlet:
            kz = dirichlet_power(
                self._spacing * capture.uz[None, :] - self._fz[:, None], self._rows
            )
        else:
            kz = np.ones((1, self._num_points), dtype=np.float64)
        if self._t_dirichlet:
            kt = dirichlet_power(
                self._bin_index[:, None] - self._delta_f * capture.tau_tot[None, :],
                self._num_bins,
            )
        else:
            kt = np.ones((1, self._num_points), dtype=np.float64)
        return ky, kz, kt

    def _capture_kernels(self, capture: _CaptureData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the cached kernels of ``capture`` or rebuild them."""
        if self._cached:
            assert capture.ky is not None and capture.kz is not None and capture.kt is not None
            return capture.ky, capture.kz, capture.kt
        return self._kernels(capture)

    def _check_x(self, x: np.ndarray) -> np.ndarray:
        """Return real float64 ``x`` of shape ``(P,)`` after validation."""
        if np.shape(x) != self._x_shape:
            raise ValueError(f"x must have shape {self._x_shape}, got {np.shape(x)}")
        if np.iscomplexobj(x):
            raise ValueError("x must be real")
        return np.asarray(x, dtype=np.float64).reshape(self._num_points)

    def _check_y(self, y: np.ndarray) -> np.ndarray:
        """Return real float64 ``y`` of shape ``y_shape`` after validation."""
        if np.shape(y) != self._y_shape:
            raise ValueError(f"y must have shape {self._y_shape}, got {np.shape(y)}")
        if np.iscomplexobj(y):
            raise ValueError("y must be real")
        return np.asarray(y, dtype=np.float64).reshape(self._y_shape)

    def _ground(self, capture: _CaptureData, index: np.ndarray) -> np.ndarray:
        """Return ``(Ky[:, None, idx] * Kz[None, :, idx]).reshape(ny * nz, len(idx))``."""
        ky, kz, _ = self._capture_kernels(capture)
        return (ky[:, None, index] * kz[None, :, index]).reshape(self._ny * self._nz, index.size)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Return ``E[obs]`` for powers ``x`` of shape ``(P,)``."""
        values = self._check_x(x)
        out = np.zeros((self._num_views, self._num_bs, 2, self._ny, self._nz, self._nt))
        for capture in self._captures:
            _, _, kt = self._capture_kernels(capture)
            for h, index in enumerate((capture.front, capture.back)):
                if index.size == 0:
                    continue
                kyz = self._ground(capture, index)
                weighted = capture.w[index] * values[index]
                block = (kyz * weighted[None, :]) @ kt[:, index].T
                out[capture.v, capture.b, h] = block.reshape(self._ny, self._nz, self._nt)
        return out.reshape(self._y_shape)

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        """Return the exact transpose of :meth:`forward` applied to ``y``."""
        values = self._check_y(y)
        result = np.zeros(self._num_points, dtype=np.float64)
        for capture in self._captures:
            _, _, kt = self._capture_kernels(capture)
            for h, index in enumerate((capture.front, capture.back)):
                if index.size == 0:
                    continue
                kyz = self._ground(capture, index)
                block = values[capture.v, capture.b, h].reshape(self._ny * self._nz, self._nt)
                tmp = kyz.T @ block
                result[index] += capture.w[index] * np.sum(tmp * kt[:, index].T, axis=1)
        return result

    def matvec(self, x: np.ndarray) -> np.ndarray:
        """Apply the flattening of :meth:`forward` to any ``x`` with ``x.size == P``."""
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
        """Return the operator as a float64 :class:`scipy.sparse.linalg.LinearOperator`."""
        return LinearOperator(
            shape=self._shape,
            matvec=self.matvec,
            rmatvec=self.rmatvec,
            dtype=np.float64,
        )


def power_operator(
    points: np.ndarray,
    geom: CaptureGeometry,
    space: str,
    product: str = "ID",
    **kwargs: Any,
) -> LinearOperator:
    """Return :class:`PowerOperator` as a :class:`scipy.sparse.linalg.LinearOperator`."""
    return PowerOperator(points, geom, space, product, **kwargs).as_linear_operator()


def _validate_oversample(oversample: int) -> int:
    """Return a validated integer oversampling factor ``>= 1`` (bool rejected)."""
    if isinstance(oversample, (bool, np.bool_)) or not isinstance(oversample, (int, np.integer)):
        raise ValueError(f"oversample must be an integer >= 1, got {oversample!r}")
    factor = int(oversample)
    if factor < 1:
        raise ValueError(f"oversample must be an integer >= 1, got {oversample!r}")
    return factor


def _axis_sampling(
    native: int, length: int, dirichlet: bool, factor: int, *, delay: bool = False
) -> tuple[int, np.ndarray, np.ndarray]:
    """Return ``(Q, kernel samples k[Q], native positions j[native])`` of one axis.

    A Dirichlet axis is sampled as ``length * (sinc(length * xr) / sinc(xr))**2``
    at ``xr = l / Q``; a singleton axis contributes ``k = [1.0]`` at ``j = [0]``.
    Angle positions follow ``j[i] = oversample * (i - n // 2) + Q // 2`` (both
    angle grids are fftshifted), delay positions ``j[i] = oversample * i``.
    """
    if not dirichlet:
        return 1, np.ones(1, dtype=np.float64), np.zeros(1, dtype=np.int64)
    q = factor * native
    samples = dirichlet_power(np.arange(q, dtype=np.float64) / q, length)
    if delay:
        positions = factor * np.arange(native, dtype=np.int64)
    else:
        positions = factor * (np.arange(native, dtype=np.int64) - native // 2) + q // 2
    return q, samples, positions


def _correlate(
    native_volume: np.ndarray,
    qshape: tuple[int, int, int],
    kernel_hat: np.ndarray,
    positions: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    """Return the nonnegative oversampled circular correlation of one native volume."""
    jy, jz, jt = positions
    stuffed = np.zeros(qshape, dtype=np.float64)
    stuffed[np.ix_(jy, jz, jt)] = np.asarray(native_volume, dtype=np.float64).reshape(
        jy.size, jz.size, jt.size
    )
    spectrum = np.fft.rfftn(stuffed, axes=(0, 1, 2)) * kernel_hat
    correlated = np.fft.irfftn(spectrum, s=qshape, axes=(0, 1, 2))
    return np.maximum(correlated, 0.0)


def power_backproject_grid(
    power_volume: np.ndarray,
    geom: CaptureGeometry,
    grid: VoxelGrid,
    tau: np.ndarray | None = None,
    *,
    space: str,
    product: str = "ID",
    kind: str = "trilinear",
    oversample: int = DEFAULT_OVERSAMPLE,
    pattern: str = "tr38901",
    polarization: str = "none",
) -> np.ndarray:
    """Return an approximation of ``K^T power_volume`` on ``grid``.

    The exact adjoint is evaluated by an FFT correlation onto an ``oversample``x
    grid and the T07a periodic interpolation (``kind``). Voxels within
    ``LOS_VS_TOLERANCE_M`` of a UE (and of a BS in ``"bv"``) are singular: their
    value is exactly 0.
    """
    _validate_product(product)
    if space not in SPACES:
        raise ValueError(f"space must be one of {SPACES}, got {space!r}")
    if kind not in INTERP_KINDS:
        raise ValueError(f"kind must be one of {INTERP_KINDS}, got {kind!r}")
    factor = _validate_oversample(oversample)
    if np.iscomplexobj(power_volume):
        raise ValueError("power_volume must be real")
    volume = np.asarray(power_volume, dtype=np.float64)
    expected = product_shape(product, geom)
    if volume.shape != expected:
        raise ValueError(f"power_volume must have shape {expected}, got {volume.shape}")
    spacing = _aperture_spacing(geom)
    delays = _capture_tau(tau, geom.num_views, geom.num_bs)

    rows, cols = geom.aperture_shape
    y_dirichlet, z_dirichlet, t_dirichlet = _PRODUCT_AXES[product]
    qy, ky, jy = _axis_sampling(cols, cols, y_dirichlet, factor)
    qz, kz, jz = _axis_sampling(rows, rows, z_dirichlet, factor)
    qt, kt, jt = _axis_sampling(geom.num_bins, geom.num_bins, t_dirichlet, factor, delay=True)
    qshape = (qy, qz, qt)
    kernel_hat = np.fft.rfftn(
        ky[:, None, None] * kz[None, :, None] * kt[None, None, :], axes=(0, 1, 2)
    )

    uy_ax, uz_ax, _ = volume_axes(qy, qz, qt, delta_f=geom.delta_f, spacing_lambda=spacing)
    periods = (1.0 / spacing, 1.0 / spacing, geom.delay_period)
    origins = (float(uy_ax[0]), float(uz_ax[0]), 0.0)
    const = _product_const(product, geom)

    centers = grid.centers()
    valid = np.flatnonzero(~_singular_mask(centers, geom, space))
    output = np.zeros(centers.shape[0], dtype=np.float64)
    chunk_size = int(POINT_CHUNK)

    for v in range(geom.num_views):
        for b in range(geom.num_bs):
            correlated = [
                _correlate(volume[v, b, h], qshape, kernel_hat, (jy, jz, jt)) for h in range(2)
            ]
            for start in range(0, valid.size, chunk_size):
                chunk = valid[start : start + chunk_size]
                if chunk.size == 0:
                    continue
                points = centers[chunk]
                factors = capture_factors(
                    points,
                    geom,
                    space,
                    v,
                    b,
                    pattern=pattern,
                    polarization=polarization,
                )
                for h in range(2):
                    select = factors.hemisphere == h
                    if not np.any(select):
                        continue
                    coords = np.stack(
                        [
                            factors.u_local[select, 1],
                            factors.u_local[select, 2],
                            factors.tau[select] + delays[v, b],
                        ],
                        axis=1,
                    )
                    idx, weights = periodic_weights(
                        coords,
                        qshape,
                        periods,
                        kind,
                        origins=origins,
                    )
                    values = (
                        const
                        * np.abs(factors.gamma[select]) ** 2
                        * gather(correlated[h], idx, weights)
                    )
                    output[chunk[select]] += values
    return output.reshape(grid.shape)
