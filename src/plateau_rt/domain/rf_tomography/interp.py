"""Sparse periodic interpolation over a sampled 3-D volume (NumPy only).

The E1 fast back-projection evaluates a sampled periodic volume (two angular
axes and a delay axis) at arbitrary continuous coordinates. Every axis is
periodic: sample ``k`` sits at ``origins[a] + k * periods[a] / shape[a]`` and
the sampled function repeats with ``periods[a]``.

:func:`periodic_weights` builds the sparse interpolation weights (8 taps for
trilinear, 64 for Keys tricubic), :func:`gather` applies them to a volume and
:func:`scatter_add` is its exact adjoint. This package must not depend on
Sionna, so it can be tested and reused CPU-only.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

INTERP_KINDS: tuple[str, ...] = ("trilinear", "tricubic")
CUBIC_A: float = -0.5  # Keys cubic-convolution parameter (Catmull-Rom)

_TRILINEAR_OFFSETS = np.array([0, 1], dtype=np.int64)
_TRICUBIC_OFFSETS = np.array([-1, 0, 1, 2], dtype=np.int64)


def _keys_kernel(s: np.ndarray, a: float) -> np.ndarray:
    """Return the Keys cubic-convolution kernel ``K(s)`` with parameter ``a``."""
    abs_s = np.abs(s)
    out = np.zeros_like(abs_s)
    within_one = abs_s <= 1.0
    out[within_one] = (a + 2.0) * abs_s[within_one] ** 3 - (a + 3.0) * abs_s[within_one] ** 2 + 1.0
    between = (abs_s > 1.0) & (abs_s < 2.0)
    out[between] = (
        a * abs_s[between] ** 3 - 5.0 * a * abs_s[between] ** 2 + 8.0 * a * abs_s[between] - 4.0 * a
    )
    return out


def _axis_offsets(frac: np.ndarray, kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Return per-axis tap offsets and weights for a fractional sample index."""
    if kind == "trilinear":
        weights = np.stack((1.0 - frac, frac), axis=1)
        return _TRILINEAR_OFFSETS, weights
    offsets = _TRICUBIC_OFFSETS
    s = frac[:, None] - offsets[None, :].astype(np.float64)
    return offsets, _keys_kernel(s, CUBIC_A)


def _as_coords(coords: np.ndarray) -> np.ndarray:
    arr = np.asarray(coords, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("coords must have shape [P, 3]")
    if not np.all(np.isfinite(arr)):
        raise ValueError("coords must be finite")
    return arr


def _as_shape(shape: Sequence[int]) -> tuple[int, int, int]:
    try:
        values = tuple(shape)
    except TypeError as exc:
        raise ValueError("shape must be three positive integers") from exc
    if len(values) != 3:
        raise ValueError("shape must be three positive integers")
    out = []
    for value in values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError("shape must be three positive integers")
        if value < 1:
            raise ValueError("shape must be three positive integers")
        out.append(int(value))
    return out[0], out[1], out[2]


def _as_periods(periods: Sequence[float]) -> np.ndarray:
    try:
        arr = np.asarray(tuple(periods), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("periods must be three finite positive values") from exc
    if arr.shape != (3,) or not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
        raise ValueError("periods must be three finite positive values")
    return arr


def _as_origins(origins: Sequence[float] | None) -> np.ndarray:
    if origins is None:
        return np.zeros(3, dtype=np.float64)
    try:
        arr = np.asarray(tuple(origins), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("origins must be three finite values") from exc
    if arr.shape != (3,) or not np.all(np.isfinite(arr)):
        raise ValueError("origins must be three finite values")
    return arr


def periodic_weights(
    coords: np.ndarray,
    shape: Sequence[int],
    periods: Sequence[float],
    kind: str = "trilinear",
    *,
    origins: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sparse periodic interpolation taps ``(idx, w)`` for ``coords``.

    ``coords`` has shape ``[P, 3]`` in physical units. Sample ``k`` of axis ``a``
    sits at ``origins[a] + k * periods[a] / shape[a]`` and repeats with
    ``periods[a]``. ``idx`` is ``int64 [P, T]`` of flat C-order volume indices
    and ``w`` is ``float64 [P, T]``; ``T`` is 8 for ``"trilinear"`` and 64 for
    ``"tricubic"``. Duplicate indices are allowed and summed by consumers.
    """
    if kind not in INTERP_KINDS:
        raise ValueError(f"unknown interpolation kind: {kind!r}")
    coords_arr = _as_coords(coords)
    n0, n1, n2 = _as_shape(shape)
    periods_arr = _as_periods(periods)
    origins_arr = _as_origins(origins)
    shape_arr = (n0, n1, n2)

    num_points = coords_arr.shape[0]
    taps_per_axis: list[np.ndarray] = []
    weights_per_axis: list[np.ndarray] = []
    for axis in range(3):
        x = (coords_arr[:, axis] - origins_arr[axis]) * shape_arr[axis] / periods_arr[axis]
        x = np.mod(x, float(shape_arr[axis]))
        base = np.floor(x).astype(np.int64)
        frac = x - base
        offsets, weights = _axis_offsets(frac, kind)
        taps = np.mod(base[:, None] + offsets[None, :], shape_arr[axis])
        taps_per_axis.append(taps)
        weights_per_axis.append(weights)

    j0, j1, j2 = taps_per_axis
    w0, w1, w2 = weights_per_axis
    num_taps = j0.shape[1] * j1.shape[1] * j2.shape[1]
    idx = ((j0[:, :, None, None] * n1 + j1[:, None, :, None]) * n2 + j2[:, None, None, :]).reshape(
        num_points, num_taps
    )
    weights = (w0[:, :, None, None] * w1[:, None, :, None] * w2[:, None, None, :]).reshape(
        num_points, num_taps
    )
    return idx.astype(np.int64, copy=False), weights.astype(np.float64, copy=False)


def gather(vol: np.ndarray, idx: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Return ``out[p] = sum_t vol.ravel()[idx[p, t]] * w[p, t]`` with shape ``[P]``."""
    vol_arr = np.asarray(vol)
    if vol_arr.ndim != 3:
        raise ValueError("vol must have exactly three dimensions")
    idx_arr = np.asarray(idx)
    w_arr = np.asarray(w)
    if idx_arr.ndim != 2 or w_arr.ndim != 2 or idx_arr.shape != w_arr.shape:
        raise ValueError("idx and w must be 2-D arrays with identical shapes")
    out = (vol_arr.reshape(-1)[idx_arr] * w_arr).sum(axis=1)
    dtype = np.complex128 if np.iscomplexobj(vol_arr) else np.float64
    return np.asarray(out, dtype=dtype)


def scatter_add(
    vals: np.ndarray, idx: np.ndarray, w: np.ndarray, shape: Sequence[int]
) -> np.ndarray:
    """Adjointly scatter ``vals`` back onto a volume of shape ``shape``.

    Returns ``vol.ravel()[i] = sum_{p, t : idx[p, t] == i} vals[p] * w[p, t]``.
    The output is real ``float64`` for real ``vals`` and ``complex128`` otherwise.
    """
    vals_arr = np.asarray(vals)
    if vals_arr.ndim != 1:
        raise ValueError("vals must be one-dimensional")
    idx_arr = np.asarray(idx)
    w_arr = np.asarray(w)
    if idx_arr.ndim != 2 or w_arr.ndim != 2 or idx_arr.shape != w_arr.shape:
        raise ValueError("idx and w must be 2-D arrays with identical shapes")
    if vals_arr.shape[0] != idx_arr.shape[0]:
        raise ValueError("vals length must match idx rows")
    n0, n1, n2 = _as_shape(shape)
    size = n0 * n1 * n2
    flat_idx = idx_arr.reshape(-1)
    if np.iscomplexobj(vals_arr):
        real = np.bincount(
            flat_idx, weights=(vals_arr.real[:, None] * w_arr).reshape(-1), minlength=size
        )
        imag = np.bincount(
            flat_idx, weights=(vals_arr.imag[:, None] * w_arr).reshape(-1), minlength=size
        )
        out = real + 1j * imag
        return out.reshape(n0, n1, n2).astype(np.complex128)
    out = np.bincount(flat_idx, weights=(vals_arr[:, None] * w_arr).reshape(-1), minlength=size)
    return out.reshape(n0, n1, n2).astype(np.float64)
