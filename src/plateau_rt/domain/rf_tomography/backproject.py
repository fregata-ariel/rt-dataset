"""Fast E1 back-projection for RF tomography (NumPy only).

Approximates the exact adjoint ``A^H (W * Y)`` of the T03 element-level operator
without materialising ``A``: one Taylor-windowed, 8x-oversampled complex
angle-delay volume is built per capture-hemisphere by FFT, and every point is
back-projected through a sparse periodic gather on that volume.

Conventions
-----------
* The approximated quantity is
  ``(A^H (W * Y))[c, h, p] = sum_{m, n} conj(A[(c, h, m, n), p]) W[r, col, n]
  Y[c, h, r, col, n]`` with ``W[r, col, n] = w_r[r] w_c[col] w_n[n]``. With the
  same ``W`` on both sides the reference is exactly
  ``dense_matrix(...)^H @ (W * Y).ravel()`` up to the interpolation error; no
  window-gain division is applied.
* Gauge (docs section 2.4): ``Y_obs[c] = exp(j phi_c)
  exp(-j 2 pi df_n tau_c) Y[c]``. The delay is compensated **on ``Y`` before the
  FFT** by ``Y_c <- Y_c * exp(+j 2 pi df_n tau_c)``, so
  ``capture_volume(..., tau=tau_c)`` equals ``exp(j phi_c)`` times the
  synchronous volume and the point lookups stay independent of ``tau``.
* Integer-element shift: the sampled volume is the index-origin array factor,
  whose phase reference is element ``(r=0, col=0)``. Multiplying it by the
  exactly periodic phase ``pre`` moves the reference to element ``(K_r, K_c)``
  next to the aperture centre; the residual half-element phase is applied
  analytically to the continuous direction as part of the per-point ``carrier``.
  Their product is exactly :func:`observables.aperture_centre_phase`, and the
  shift removes most of the linear phase ramp, which makes the periodic
  interpolation accurate.
* Singular points: :func:`forward_exact.capture_factors` rejects points within
  ``LOS_VS_TOLERANCE_M`` of a UE (both spaces) or of a BS (``"bv"`` only).
  Back-projection must not raise, so those points are marked ``valid=False``
  with ``carrier = 0`` and zero taps, and ``capture_factors`` is only called on
  the remaining points. In ``"vs"`` a point at a BS is the LoS source and stays
  valid.
* Memory: the full ``[V, B, 2, Qy, Qz, Nt]`` volume is never built; one
  capture-hemisphere is processed at a time and points are handled in chunks of
  :data:`POINT_CHUNK` (read at call time), keeping peak memory bounded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from plateau_rt.domain.rf_tomography.forward_exact import (
    SPACES,
    _capture_index,
    _prepare_points,
    _singular_mask,
    capture_factors,
)
from plateau_rt.domain.rf_tomography.geometry import CaptureGeometry, planar_element_offsets
from plateau_rt.domain.rf_tomography.interp import INTERP_KINDS, gather, periodic_weights
from plateau_rt.domain.rf_tomography.observables import (
    _validate_oversample,
    angle_delay_volume,
    volume_axes,
)

POINT_CHUNK: int = 65_536
WINDOWS: tuple[str | None, ...] = (None, "taylor")
_TAPS: dict[str, int] = {"trilinear": 8, "tricubic": 64}


@dataclass(frozen=True)
class Lookup:
    """Sparse gather taps of one capture-hemisphere (v, b, h) for a point set."""

    idx: np.ndarray  # int64 [P, T] flat C-order indices into the [Qy, Qz, Nt] volume
    w: np.ndarray  # float64 [P, T] interpolation weights (T = 8 trilinear, 64 tricubic)
    carrier: np.ndarray  # complex128 [P] per-point factor of step 4 (0 where not valid)
    valid: np.ndarray  # bool [P]: in hemisphere h and not singular
    shape: tuple[int, int, int]  # (Qy, Qz, Nt) of the volume these taps address


@dataclass(frozen=True)
class _Axes:
    """Prepared aperture and volume conventions shared by volume and lookup."""

    s: float
    rows: int
    cols: int
    K_c: int
    K_r: int
    shape: tuple[int, int, int]
    periods: tuple[float, float, float]
    origins: tuple[float, float, float]
    u_y: np.ndarray
    u_z: np.ndarray


def _validate_space(space: str) -> None:
    """Raise ``ValueError`` unless ``space`` is ``"bv"`` or ``"vs"``."""
    if space not in SPACES:
        raise ValueError(f"space must be one of {SPACES}, got {space!r}")


def _validate_kind(kind: str) -> None:
    """Raise ``ValueError`` unless ``kind`` is a known interpolation kind."""
    if kind not in INTERP_KINDS:
        raise ValueError(f"kind must be one of {INTERP_KINDS}, got {kind!r}")


def _validate_window(window: str | None) -> None:
    """Raise ``ValueError`` unless ``window`` is ``None`` or ``"taylor"``."""
    if window not in WINDOWS:
        raise ValueError(f"window must be None or 'taylor', got {window!r}")


def _validate_h(h: int) -> int:
    """Return ``h`` as an int, requiring ``0`` (front) or ``1`` (back)."""
    try:
        index = int(h)
    except (TypeError, ValueError) as error:
        raise ValueError("h must be 0 or 1") from error
    if index not in (0, 1):
        raise ValueError("h must be 0 or 1")
    return index


def _validate_y(Y: np.ndarray, geom: CaptureGeometry) -> None:
    """Raise ``ValueError`` unless ``Y`` matches the geometry's capture layout."""
    expected = (
        geom.num_views,
        geom.num_bs,
        2,
        geom.aperture_shape[0],
        geom.aperture_shape[1],
        geom.num_bins,
    )
    if np.asarray(Y).shape != expected:
        raise ValueError(f"Y must have shape {expected}")


def _aperture_spacing(geom: CaptureGeometry) -> float:
    """Return the ideal planar aperture spacing in wavelengths (step 1).

    Raises ``ValueError`` when ``geom.elem_offsets`` is not the ideal planar
    aperture the FFT volume represents.
    """
    rows, cols = geom.aperture_shape
    wavelength = geom.wavelength
    if cols > 1:
        s = float(geom.elem_offsets[1, 1] - geom.elem_offsets[0, 1]) / wavelength
    elif rows > 1:
        s = float(geom.elem_offsets[0, 2] - geom.elem_offsets[cols, 2]) / wavelength
    else:
        s = 0.5
    expected = planar_element_offsets(wavelength, rows=rows, cols=cols, spacing_lambda=s)
    if not np.allclose(geom.elem_offsets, expected, rtol=0.0, atol=1e-9 * wavelength):
        raise ValueError("elem_offsets must be the ideal planar aperture represented by the FFT")
    return s


def _capture_axes(geom: CaptureGeometry, oversample: tuple[int, int]) -> _Axes:
    """Return the volume axes, spacing and integer-shift indices (steps 1 and 3)."""
    a, d_t = _validate_oversample(oversample)
    rows, cols = geom.aperture_shape
    s = _aperture_spacing(geom)
    qy, qz, nt = a * cols, a * rows, d_t * geom.num_bins
    u_y, u_z, _ = volume_axes(qy, qz, nt, delta_f=geom.delta_f, spacing_lambda=s)
    return _Axes(
        s=s,
        rows=rows,
        cols=cols,
        K_c=(cols - 1) // 2,
        K_r=(rows - 1) // 2,
        shape=(qy, qz, nt),
        periods=(1.0 / s, 1.0 / s, geom.delay_period),
        origins=(float(u_y[0]), float(u_z[0]), 0.0),
        u_y=u_y,
        u_z=u_z,
    )


def _carrier_factor(
    geom: CaptureGeometry, axes: _Axes, u_local: np.ndarray, gamma: np.ndarray
) -> np.ndarray:
    """Return the per-point factor of step 4 (steps 3-4 residual centre phase)."""
    exponent = axes.s * (
        (axes.K_c - (axes.cols - 1) / 2.0) * u_local[:, 1]
        + ((axes.rows - 1) / 2.0 - axes.K_r) * u_local[:, 2]
    )
    scale = np.sqrt(axes.rows * axes.cols * geom.num_bins)
    return (scale * np.conj(gamma) * np.exp(-2j * np.pi * exponent)).astype(np.complex128)


def _taps(
    geom: CaptureGeometry,
    points: np.ndarray,
    space: str,
    v: int,
    b: int,
    *,
    kind: str,
    pattern: str,
    polarization: str,
    singular: np.ndarray,
    axes: _Axes,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the hemisphere-independent taps of one point chunk (steps 4 and 6).

    ``capture_factors`` and ``periodic_weights`` are called once for the whole
    chunk; the caller selects each hemisphere from ``ok & (hemisphere == h)``.
    Singular points get zero taps, ``carrier = 0`` and ``hemisphere = 0``.
    """
    num_points = points.shape[0]
    num_taps = _TAPS[kind]
    idx = np.zeros((num_points, num_taps), dtype=np.int64)
    weights = np.zeros((num_points, num_taps), dtype=np.float64)
    carrier = np.zeros(num_points, dtype=np.complex128)
    hemisphere = np.zeros(num_points, dtype=np.int64)
    ok = ~singular

    if np.any(ok):
        factors = capture_factors(
            points[ok], geom, space, v, b, pattern=pattern, polarization=polarization
        )
        coords = np.stack((factors.u_local[:, 1], factors.u_local[:, 2], factors.tau), axis=1)
        idx_ok, w_ok = periodic_weights(
            coords, axes.shape, axes.periods, kind, origins=axes.origins
        )
        carrier[ok] = _carrier_factor(geom, axes, factors.u_local, factors.gamma)
        hemisphere[ok] = factors.hemisphere
        idx[ok] = idx_ok
        weights[ok] = w_ok
    return idx, weights, carrier, hemisphere, ok


def capture_volume(
    Y: np.ndarray,
    geom: CaptureGeometry,
    v: int,
    b: int,
    *,
    tau: float | None = None,
    window: str | None = "taylor",
    oversample: tuple[int, int] = (8, 8),
) -> np.ndarray:
    """Return the integer-shift-corrected capture volume ``[2, Qy, Qz, Nt]``.

    The gauge delay ``tau`` is compensated on ``Y`` before the FFT (step 2),
    ``angle_delay_volume`` builds the windowed volume (step 3) and the periodic
    integer-element shift ``pre`` is applied to its angular axes.
    """
    _validate_window(window)
    v = _capture_index(v, geom.num_views, "v")
    b = _capture_index(b, geom.num_bs, "b")
    _validate_y(Y, geom)
    axes = _capture_axes(geom, oversample)

    capture = np.asarray(Y)[v, b]
    if tau is not None:
        delay = float(tau)
        if not np.isfinite(delay):
            raise ValueError("tau must be finite")
        capture = capture * np.exp(2j * np.pi * geom.freq_offsets[None, None, :] * delay)
    capture = np.ascontiguousarray(capture, dtype=np.complex128)

    volume = angle_delay_volume(capture[None, None], window, oversample)[0, 0]
    shift = -axes.K_c * axes.u_y[:, None] + axes.K_r * axes.u_z[None, :]
    pre = np.exp(-2j * np.pi * axes.s * shift)
    return (volume * pre[None, :, :, None]).astype(np.complex128, copy=False)


def capture_lookup(
    geom: CaptureGeometry,
    points: np.ndarray,
    space: str,
    v: int,
    b: int,
    h: int,
    *,
    kind: str = "trilinear",
    oversample: tuple[int, int] = (8, 8),
    pattern: str = "tr38901",
    polarization: str = "none",
) -> Lookup:
    """Return the sparse gather taps of capture-hemisphere ``(v, b, h)`` (steps 4 and 6)."""
    _validate_space(space)
    _validate_kind(kind)
    h = _validate_h(h)
    v = _capture_index(v, geom.num_views, "v")
    b = _capture_index(b, geom.num_bs, "b")
    pts = _prepare_points(points)
    axes = _capture_axes(geom, oversample)
    singular = _singular_mask(pts, geom, space)

    num_points = pts.shape[0]
    num_taps = _TAPS[kind]
    idx = np.empty((num_points, num_taps), dtype=np.int64)
    weights = np.empty((num_points, num_taps), dtype=np.float64)
    carrier = np.empty(num_points, dtype=np.complex128)
    valid = np.empty(num_points, dtype=bool)
    for start in range(0, num_points, POINT_CHUNK):
        stop = min(start + POINT_CHUNK, num_points)
        idx_block, w_block, carrier_block, hemisphere, ok = _taps(
            geom,
            pts[start:stop],
            space,
            v,
            b,
            kind=kind,
            pattern=pattern,
            polarization=polarization,
            singular=singular[start:stop],
            axes=axes,
        )
        valid_block = ok & (hemisphere == h)
        idx[start:stop] = idx_block
        weights[start:stop] = w_block
        carrier[start:stop] = np.where(valid_block, carrier_block, 0.0)
        valid[start:stop] = valid_block
    return Lookup(idx=idx, w=weights, carrier=carrier, valid=valid, shape=axes.shape)


def apply_lookup(volume: np.ndarray, lookup: Lookup) -> np.ndarray:
    """Return ``carrier * gather(volume, idx, w)``, exactly 0 where not valid.

    ``volume`` is one hemisphere ``capture_volume(...)[h]`` and its shape must
    equal ``lookup.shape``.
    """
    vol = np.asarray(volume)
    if vol.shape != lookup.shape:
        raise ValueError(f"volume shape {vol.shape} does not match lookup shape {lookup.shape}")
    gathered = gather(vol, lookup.idx, lookup.w)
    out = lookup.carrier * gathered
    return np.where(lookup.valid, out, 0.0).astype(np.complex128)


def backproject(
    Y: np.ndarray,
    geom: CaptureGeometry,
    points: np.ndarray,
    space: str,
    tau: np.ndarray | None = None,
    per_capture: bool = True,
    *,
    kind: str = "trilinear",
    window: str | None = "taylor",
    oversample: tuple[int, int] = (8, 8),
    pattern: str = "tr38901",
    polarization: str = "none",
) -> np.ndarray:
    """Return the E1 fast back-projection of the windowed data ``Y``.

    ``per_capture=True`` returns ``complex128 [V, B, 2, P]`` with one value per
    capture-hemisphere; ``per_capture=False`` returns ``complex128 [P]``, the
    coherent sum over ``(v, b, h)``, i.e. the approximation of
    ``dense_matrix(...)^H @ (W * Y).ravel()``.
    """
    _validate_space(space)
    _validate_kind(kind)
    _validate_window(window)
    pts = _prepare_points(points)
    _validate_y(Y, geom)
    axes = _capture_axes(geom, oversample)

    if tau is None:
        delays = np.zeros((geom.num_views, geom.num_bs), dtype=np.float64)
    else:
        delays = np.asarray(tau, dtype=np.float64)
        if delays.shape != (geom.num_views, geom.num_bs):
            raise ValueError(f"tau must have shape {(geom.num_views, geom.num_bs)}")
        if not np.all(np.isfinite(delays)):
            raise ValueError("tau must contain only finite values")

    num_points = pts.shape[0]
    singular = _singular_mask(pts, geom, space)
    result = np.zeros((geom.num_views, geom.num_bs, 2, num_points), dtype=np.complex128)
    for v in range(geom.num_views):
        for b in range(geom.num_bs):
            volume = capture_volume(
                Y, geom, v, b, tau=delays[v, b], window=window, oversample=oversample
            )
            for start in range(0, num_points, POINT_CHUNK):
                stop = min(start + POINT_CHUNK, num_points)
                idx, weights, carrier, hemisphere, ok = _taps(
                    geom,
                    pts[start:stop],
                    space,
                    v,
                    b,
                    kind=kind,
                    pattern=pattern,
                    polarization=polarization,
                    singular=singular[start:stop],
                    axes=axes,
                )
                for h in range(2):
                    sel = ok & (hemisphere == h)
                    result[v, b, h, start + np.flatnonzero(sel)] = carrier[sel] * gather(
                        volume[h], idx[sel], weights[sel]
                    )

    if per_capture:
        return result
    return result.sum(axis=(0, 1, 2)).astype(np.complex128)


def envelope_sum(bp: np.ndarray) -> np.ndarray:
    """Return ``float64 [P]``, the sum of ``|bp|**2`` over every leading axis.

    A one-dimensional input returns ``|bp|**2`` (the per-capture envelope map
    ``sum_c |A_c^H y_c|^2`` for a ``[V, B, 2, P]`` input).
    """
    arr = np.asarray(bp)
    if arr.ndim == 0:
        raise ValueError("bp must have at least one dimension")
    power = np.abs(arr) ** 2
    return np.sum(power, axis=tuple(range(arr.ndim - 1))).astype(np.float64)
