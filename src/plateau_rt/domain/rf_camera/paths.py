"""Canonical path ordering, CFR resynthesis and path-GT schema (NumPy only).

GPU path tracing is not bit-reproducible: path order changes between runs and
values jitter at the ulp level. This module provides a canonical ordering
(valid paths first, sorted by quantised delay, ties broken by power), a
resynthesis helper that rebuilds the baseband CFR from stored coefficients and
the single schema that describes the stored ``path_geometry_gt.npz`` arrays.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from plateau_rt.domain.rf_camera.camera import HEMISPHERES

PATH_GEOMETRY_GT_FILE_NAME = "path_geometry_gt.npz"
PATH_SCHEMA_FILE_NAME = "path_schema.json"

PATH_GT_MODE_CANONICAL = "canonical"
PATH_GT_MODE_SIONNA_NATIVE = "sionna_native"

TAU_QUANTUM_S = 1e-12


def quantise_tau(
    tau: np.ndarray,
    tau_quantum_s: float = TAU_QUANTUM_S,
) -> np.ndarray:
    """Quantise delays to ``tau_quantum_s`` bins as int64 (float64 division).

    Everything is computed in float64 so a float32 ``tau`` does not lose
    precision before the floor. The result may be negative for invalid
    (negative) delays; callers are expected to mask those out.
    """
    if tau_quantum_s <= 0.0:
        raise ValueError(f"tau_quantum_s must be > 0, got {tau_quantum_s!r}")
    tau_array = np.asarray(tau, dtype=np.float64)
    return np.floor(tau_array / tau_quantum_s).astype(np.int64)


def canonical_path_order(
    tau: np.ndarray,
    power: np.ndarray,
    valid: np.ndarray,
    *,
    tau_quantum_s: float = TAU_QUANTUM_S,
) -> np.ndarray:
    """Return int64 sort indices over the last axis (``[..., num_paths]``).

    Valid paths come first, sorted by delay quantised to ``tau_quantum_s``,
    then by power in descending order. Invalid paths go last. Uses
    :func:`numpy.lexsort` along the last axis. Quantisation *reduces* but does
    not prevent run-to-run reordering: paths straddling a bin edge can still
    swap, and exact ties in bin and power keep Sionna's input order.
    """
    tau = np.asarray(tau)
    power = np.asarray(power)
    valid = np.asarray(valid, dtype=bool)
    if not (tau.shape == power.shape == valid.shape):
        raise ValueError(
            f"tau, power and valid must share a shape, got {tau.shape}, "
            f"{power.shape}, {valid.shape}"
        )
    if tau.shape[-1] == 0:
        return np.empty_like(valid, dtype=np.int64)

    prefix = tau.shape[:-1]
    num_paths = tau.shape[-1]
    flat_tau = tau.reshape(-1, num_paths)
    flat_power = power.reshape(-1, num_paths)
    flat_valid = valid.reshape(-1, num_paths)
    order = np.empty_like(flat_valid, dtype=np.int64)
    invalid_tau = np.iinfo(np.int64).max
    for row in range(flat_tau.shape[0]):
        invalid_flag = (~flat_valid[row]).astype(np.int64)
        tau_row = np.where(
            flat_valid[row],
            quantise_tau(flat_tau[row], tau_quantum_s),
            invalid_tau,
        )
        neg_power = np.where(flat_valid[row], -flat_power[row], 0.0)
        order[row] = np.lexsort((neg_power, tau_row, invalid_flag))
    return order.reshape(prefix + (num_paths,)).astype(np.int64, copy=False)


def apply_path_order(array: np.ndarray, order: np.ndarray, path_axis: int) -> np.ndarray:
    """Gather ``array`` along ``path_axis`` according to ``order``.

    ``order`` has shape ``[..., num_paths]`` with the path dimension last. Its
    leading dimensions must **equal** the leading dimensions of ``array``
    (after ``path_axis`` is moved last), otherwise ``ValueError`` is raised
    naming both shapes. Extra middle dimensions between those leading
    dimensions and the path axis (e.g. hemisphere/row/col or depth) are
    broadcast.
    """
    array = np.asarray(array)
    order = np.asarray(order, dtype=np.int64)
    if order.ndim < 1:
        raise ValueError(f"order must have at least one dimension, got {order.shape}")
    num_dims = array.ndim
    path_axis_norm = path_axis % num_dims
    moved = np.moveaxis(array, path_axis_norm, -1)
    if order.shape[-1] != moved.shape[-1]:
        raise ValueError(
            f"Path dimension mismatch: array has {moved.shape[-1]}, order has {order.shape[-1]}"
        )
    order_prefix = order.shape[:-1]
    array_prefix = moved.shape[:-1]
    if order_prefix != array_prefix[: order.ndim - 1]:
        raise ValueError(
            f"order leading dimensions {order_prefix} do not match array leading "
            f"dimensions {array_prefix}"
        )
    extra = len(array_prefix) - (order.ndim - 1)
    expanded = np.broadcast_to(
        order.reshape(order_prefix + (1,) * extra + (order.shape[-1],)),
        moved.shape,
    )
    gathered = np.take_along_axis(moved, expanded, axis=-1)
    return np.moveaxis(gathered, -1, path_axis_norm)


def synthesize_cfr(
    a_baseband: np.ndarray,
    tau: np.ndarray,
    frequency_offsets_hz: np.ndarray,
) -> np.ndarray:
    """Resynthesise the CFR as ``sum_p a_b * exp(-j*2*pi*df*tau_p)``.

    ``a_baseband`` has shape ``[..., P]`` and ``tau`` has shape ``[P]`` or a
    shape that broadcasts against it. Returns ``[..., F]`` computed in
    complex128. Invalid paths (``tau < 0``) contribute 0.
    """
    a_baseband = np.asarray(a_baseband, dtype=np.complex128)
    offsets = np.asarray(frequency_offsets_hz, dtype=np.float64).ravel()
    a_broadcast, tau_broadcast = np.broadcast_arrays(a_baseband, np.asarray(tau, dtype=np.float64))
    a_clean = np.where(tau_broadcast < 0.0, 0.0, a_broadcast).astype(np.complex128)
    tau_safe = np.where(tau_broadcast < 0.0, 0.0, tau_broadcast)
    phase = np.exp(-1j * 2.0 * np.pi * tau_safe[..., None] * offsets[None, ...])
    return np.sum(a_clean[..., None] * phase, axis=-2)


_CANONICAL_AXES: dict[str, tuple[str, ...]] = {
    "valid": ("view", "bs", "path"),
    "tau": ("view", "bs", "path"),
    "theta_t": ("view", "bs", "path"),
    "phi_t": ("view", "bs", "path"),
    "theta_r": ("view", "bs", "path"),
    "phi_r": ("view", "bs", "path"),
    "a": ("view", "bs", "hemisphere", "row", "col", "path"),
    "a_baseband": ("view", "bs", "hemisphere", "row", "col", "path"),
    "interactions": ("view", "bs", "path", "depth"),
    "object_index": ("view", "bs", "path", "depth"),
    "primitives": ("view", "bs", "path", "depth"),
    "vertices": ("view", "bs", "path", "depth", "xyz"),
    "num_interactions": ("view", "bs", "path"),
}

_SIONNA_NATIVE_AXES: dict[str, tuple[str, ...]] = {
    "valid": ("view", "rx_ant", "bs", "tx_ant", "path"),
    "tau": ("view", "rx_ant", "bs", "tx_ant", "path"),
    "theta_t": ("view", "rx_ant", "bs", "tx_ant", "path"),
    "phi_t": ("view", "rx_ant", "bs", "tx_ant", "path"),
    "theta_r": ("view", "rx_ant", "bs", "tx_ant", "path"),
    "phi_r": ("view", "rx_ant", "bs", "tx_ant", "path"),
}

_ARRAY_UNITS: dict[str, str] = {
    "valid": "flag",
    "tau": "s",
    "theta_t": "rad",
    "phi_t": "rad",
    "theta_r": "rad",
    "phi_r": "rad",
    "a": "linear",
    "a_baseband": "linear",
    "interactions": "flag",
    "object_index": "index",
    "primitives": "index",
    "vertices": "m",
    "num_interactions": "count",
}

_INTERACTION_TYPE_FLAGS: dict[str, int] = {
    "NONE": 0,
    "SPECULAR": 1,
    "DIFFUSE": 2,
    "REFRACTION": 4,
    "DIFFRACTION": 8,
}

_INVALID_MARKERS: dict[str, Any] = {
    "primitive": 4294967295,
    "shape": 4294967295,
    "object_index": -1,
    "tau_invalid": "tau < 0",
}

_ORDERING_NOTE = (
    "canonical: valid paths first, sorted by delay quantised to tau_quantum_s; "
    "ties in the quantised delay are broken by descending power (sum |a_baseband|^2 "
    "over hemisphere, row, col); invalid paths last. Quantisation reduces but does "
    "not prevent jitter-induced reordering: two paths straddling a bin edge can "
    "still swap, and exact ties in quantised delay and power keep Sionna's "
    "run-dependent input order."
)

_NATIVE_ORDERING_NOTE = (
    "sionna_native: explicit arrays are stored unchanged in Sionna's native "
    "[view, rx_ant, bs, tx_ant, path] order; no path reordering is applied."
)

_RESYNTHESIS_NOTE = (
    "a_baseband = a * exp(-j*2*pi*carrier_frequency_hz*tau), so "
    "a = a_baseband * exp(+j*2*pi*carrier_frequency_hz*tau) for valid paths; "
    "invalid paths (tau < 0) contribute 0."
)


def _array_spec(
    name: str, value: np.ndarray | tuple[tuple[int, ...], Any]
) -> tuple[tuple[int, ...], np.dtype]:
    """Return ``(shape, dtype)`` for a stored array or a ``(shape, dtype)`` pair."""
    if isinstance(value, np.ndarray):
        return tuple(value.shape), np.dtype(value.dtype)
    try:
        shape, dtype = value
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"array {name!r} must be a NumPy array or a (shape, dtype) pair, got {value!r}"
        ) from exc
    return tuple(int(dim) for dim in shape), np.dtype(dtype)


def build_path_schema(
    arrays: Mapping[str, np.ndarray] | Mapping[str, tuple[tuple[int, ...], Any]],
    *,
    mode: str,
    object_names: Sequence[str],
    carrier_frequency_hz: float,
    bs_ids: Sequence[str],
    view_ids: Sequence[str],
) -> dict[str, Any]:
    """Describe a stored ``path_geometry_gt.npz`` as a JSON-serialisable schema.

    ``mode`` is :data:`PATH_GT_MODE_CANONICAL` for synthetic arrays (canonical
    ordering, coefficients and depth arrays) or
    :data:`PATH_GT_MODE_SIONNA_NATIVE` for explicit arrays (Sionna's native
    ``[view, rx_ant, bs, tx_ant, path]`` geometry only). Every stored array is
    described with its ``dtype``, ``shape``, ``axes`` and ``unit``.
    """
    if mode not in (PATH_GT_MODE_CANONICAL, PATH_GT_MODE_SIONNA_NATIVE):
        raise ValueError(f"unknown path-GT mode {mode!r}")
    axis_lookup = _CANONICAL_AXES if mode == PATH_GT_MODE_CANONICAL else _SIONNA_NATIVE_AXES

    described: dict[str, Any] = {}
    for name, value in arrays.items():
        shape, dtype = _array_spec(name, value)
        described[name] = {
            "dtype": str(dtype),
            "shape": list(shape),
            "axes": list(axis_lookup.get(name, ())),
            "unit": _ARRAY_UNITS.get(name, "unknown"),
        }

    schema: dict[str, Any] = {
        "file": PATH_GEOMETRY_GT_FILE_NAME,
        "mode": mode,
        "synthetic_array": mode == PATH_GT_MODE_CANONICAL,
        "arrays": described,
        "bs_ids": list(bs_ids),
        "view_ids": list(view_ids),
        "hemispheres": list(HEMISPHERES),
        "tau_quantum_s": TAU_QUANTUM_S,
        "interaction_type_flags": dict(_INTERACTION_TYPE_FLAGS),
        "invalid_markers": dict(_INVALID_MARKERS),
        "object_names": list(object_names),
        "carrier_frequency_hz": carrier_frequency_hz,
    }
    if mode == PATH_GT_MODE_CANONICAL:
        schema["ordering"] = _ORDERING_NOTE
        schema["resynthesis"] = {
            "formula": (
                "H[..., f] = sum_p a_baseband[..., p] * exp(-j*2*pi*frequency_offsets_hz[f]*tau[p])"
            ),
            "note": _RESYNTHESIS_NOTE,
        }
    else:
        schema["ordering"] = _NATIVE_ORDERING_NOTE
    return schema
