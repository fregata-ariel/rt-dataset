"""Sionna RT tracing shared by the RF-camera generators.

The RF camera is a planar UE receive aperture illuminated by one active BS
antenna/port. These helpers configure that geometry, trace paths and return
per-receiver aperture CFRs in the ``[row, col, frequency]`` layout used by
:mod:`plateau_rt.domain.rf_camera`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from sionna.rt import PathSolver, PlanarArray

from plateau_rt.domain.rf_camera.imaging import split_tx_pattern_axes
from plateau_rt.domain.rf_camera.paths import (
    PATH_GT_MODE_CANONICAL,
    PATH_GT_MODE_SIONNA_NATIVE,
    apply_path_order,
    canonical_path_order,
)

PATH_ANGLE_FIELDS = ("theta_t", "phi_t", "theta_r", "phi_r")


@dataclass(frozen=True)
class PathGroundTruthResult:
    """Path-level GT arrays with the schema mode that describes them.

    ``arrays`` maps the ``path_geometry_gt.npz`` names to NumPy arrays,
    ``object_names`` is the sorted list of scene object names (the canonical
    ``object_index`` values index into it) and ``mode`` is one of
    :data:`path_gt_mode` strings (``canonical`` or ``sionna_native``).
    """

    arrays: dict[str, np.ndarray]
    object_names: list[str]
    mode: str


def configure_rf_camera_arrays(
    scene: Any,
    *,
    rx_rows: int,
    rx_cols: int,
    vertical_spacing_lambda: float,
    horizontal_spacing_lambda: float,
    tx_pattern: str,
    rx_pattern: str,
    polarization: str = "V",
) -> None:
    """Set a single-port BS array and the UE receive aperture on ``scene``.

    Exciting one Tx antenna/port keeps transmit beamforming out of the
    RF-camera image-formation problem.
    """
    scene.tx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern=tx_pattern,
        polarization=polarization,
    )
    scene.rx_array = PlanarArray(
        num_rows=rx_rows,
        num_cols=rx_cols,
        vertical_spacing=vertical_spacing_lambda,
        horizontal_spacing=horizontal_spacing_lambda,
        pattern=rx_pattern,
        polarization=polarization,
    )


def trace_paths(
    scene: Any,
    *,
    max_depth: int,
    synthetic_array: bool,
    seed: int,
    los: bool = True,
    specular_reflection: bool = True,
    diffuse_reflection: bool = False,
    refraction: bool = True,
    diffraction: bool = False,
    edge_diffraction: bool = False,
    samples_per_src: int | None = None,
) -> Any:
    """Trace paths for all Tx/Rx with configurable PathSolver flags."""
    kwargs: dict[str, Any] = {}
    if samples_per_src is not None:
        kwargs["samples_per_src"] = samples_per_src
    return PathSolver()(
        scene=scene,
        max_depth=max_depth,
        los=los,
        specular_reflection=specular_reflection,
        diffuse_reflection=diffuse_reflection,
        refraction=refraction,
        diffraction=diffraction,
        edge_diffraction=edge_diffraction,
        synthetic_array=synthetic_array,
        seed=seed,
        **kwargs,
    )


def multi_tx_aperture_cfrs(
    paths: Any,
    frequency_offsets_hz: np.ndarray,
    *,
    num_rx: int,
    num_tx: int,
    rx_rows: int,
    rx_cols: int,
) -> np.ndarray:
    """Return the complex aperture CFR of every receiver and BS.

    Output is ``[rx, tx, pattern, row, col, freq]``. The pattern axis has one
    entry per receive antenna pattern, e.g. front and back for
    :data:`~plateau_rt.adapters.sionna.rf_patterns.HEMISPHERE_SPLIT_PATTERN`.

    ``Paths.cfr()`` operates on baseband frequency offsets around the scene's
    carrier. Keeping the carrier in ``scene.frequency`` avoids applying the
    carrier propagation phase a second time. Absolute path delays are kept
    (``normalize_delays=False``).
    """
    cfr = np.asarray(
        paths.cfr(
            frequencies=frequency_offsets_hz,
            normalize_delays=False,
            normalize=False,
            out_type="numpy",
        )
    )
    print(f"Paths.cfr shape={cfr.shape}, dtype={cfr.dtype}")

    # [num_rx, num_rx_patterns * num_rx_ant, num_tx, num_tx_ant, time, frequency]
    num_patterns = len(paths.rx_array.antenna_pattern.patterns)
    expected = (num_rx, num_patterns * rx_rows * rx_cols, num_tx, 1, 1, len(frequency_offsets_hz))
    if cfr.shape != expected:
        raise RuntimeError(
            f"Unexpected Sionna Paths.cfr shape: expected={expected}, actual={cfr.shape}"
        )

    return np.stack(
        [
            split_tx_pattern_axes(
                cfr[rx, :, :, 0, 0, :],
                num_tx=num_tx,
                num_patterns=num_patterns,
                rows=rx_rows,
                cols=rx_cols,
            )
            for rx in range(num_rx)
        ]
    )


def aperture_cfrs(
    paths: Any,
    frequency_offsets_hz: np.ndarray,
    *,
    num_rx: int,
    rx_rows: int,
    rx_cols: int,
) -> np.ndarray:
    """Return the complex aperture CFR of every receiver, ``[rx, pattern, row, col, freq]``.

    The pattern axis has one entry per receive antenna pattern, e.g. front and
    back for :data:`~plateau_rt.adapters.sionna.rf_patterns.HEMISPHERE_SPLIT_PATTERN`.

    ``Paths.cfr()`` operates on baseband frequency offsets around the scene's
    carrier. Keeping the carrier in ``scene.frequency`` avoids applying the
    carrier propagation phase a second time. Absolute path delays are kept
    (``normalize_delays=False``).

    Single-BS shorthand for :func:`multi_tx_aperture_cfrs` with ``num_tx=1``.
    """
    return multi_tx_aperture_cfrs(
        paths,
        frequency_offsets_hz,
        num_rx=num_rx,
        num_tx=1,
        rx_rows=rx_rows,
        rx_cols=rx_cols,
    )[:, 0]


def path_attributes(paths: Any, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Export ``Paths`` attributes as NumPy arrays, skipping unavailable ones."""
    payload: dict[str, np.ndarray] = {}
    for name in names:
        try:
            payload[name] = np.asarray(getattr(paths, name))
        except Exception as exc:  # pragma: no cover - depends on Sionna backend
            print(f"Warning: could not export paths.{name}: {exc}")
    return payload


def _split_fused_coefficient(
    fused: np.ndarray,
    *,
    num_patterns: int,
    rx_rows: int,
    rx_cols: int,
) -> np.ndarray:
    """Split ``[rx, tx, fused, path]`` into ``[rx, tx, pattern, row, col, path]``.

    The fused receive axis is pattern-major with column-first antenna
    numbering: channel ``p * rows * cols + c * rows + r`` is element
    ``(row=r, col=c)`` of pattern ``p``, matching
    :func:`~plateau_rt.domain.rf_camera.imaging.split_pattern_axis`.
    """
    fused = np.asarray(fused)
    if fused.ndim != 4:
        raise ValueError(f"Expected fused coefficients [rx, tx, fused, path], got {fused.shape}")
    num_rx, num_tx = fused.shape[0], fused.shape[1]
    num_paths = fused.shape[-1]
    size = rx_rows * rx_cols
    if fused.shape[2] != num_patterns * size:
        raise ValueError(f"Expected {num_patterns} x {size} fused channels, got {fused.shape[2]}")
    reshaped = fused.reshape(num_rx, num_tx, num_patterns, rx_cols, rx_rows, num_paths)
    return np.swapaxes(reshaped, 3, 4)


def _scene_object_index(scene: Any) -> tuple[list[str], dict[int, int]]:
    """Return sorted scene object names and the object-id -> index lookup."""
    try:
        id_to_name = {int(obj.object_id): name for name, obj in scene.objects.items()}
    except Exception as exc:  # pragma: no cover - depends on Sionna backend
        warnings.warn(f"could not map scene object ids: {exc}", stacklevel=2)
        return [], {}
    object_names = sorted(id_to_name.values())
    name_to_index = {name: index for index, name in enumerate(object_names)}
    id_to_index = {object_id: name_to_index[name] for object_id, name in id_to_name.items()}
    return object_names, id_to_index


def _map_object_ids(raw_objects: np.ndarray, id_to_index: dict[int, int]) -> np.ndarray:
    """Map raw uint32 object ids to sorted-object indices, -1 for unknown/none."""
    object_index = np.full(raw_objects.shape, -1, dtype=np.int32)
    if not id_to_index:
        return object_index
    known_ids = np.fromiter(sorted(id_to_index), dtype=np.int64, count=len(id_to_index))
    known_indices = np.fromiter(
        (id_to_index[int(object_id)] for object_id in known_ids),
        dtype=np.int32,
        count=known_ids.size,
    )
    flat = raw_objects.astype(np.int64, copy=False).ravel()
    positions = np.searchsorted(known_ids, flat)
    positions = np.clip(positions, 0, known_ids.size - 1)
    matched = known_ids[positions] == flat
    mapped = np.where(matched, known_indices[positions], -1).astype(np.int32, copy=False)
    return mapped.reshape(raw_objects.shape)


def path_ground_truth(
    paths: Any, scene: Any, *, rx_rows: int, rx_cols: int
) -> PathGroundTruthResult:
    """Export path-level ground truth from a traced ``Paths`` object.

    With Sionna synthetic arrays (``paths.synthetic_array`` true) the
    coefficients are split pattern-major with column-first antenna numbering,
    exactly as :func:`aperture_cfrs` does, and all path axes are reordered
    with :func:`~plateau_rt.domain.rf_camera.paths.canonical_path_order`.
    ``a_baseband`` is the baseband coefficient (carrier phase removed).
    With explicit arrays the six geometry attributes are exported unchanged in
    Sionna's native ``[view, rx_ant, bs, tx_ant, path]`` order; coefficients
    and depth arrays are not stored.

    Returns a :class:`PathGroundTruthResult` with the arrays plus the sorted
    scene object names (``object_index`` indexes into them, -1 for none or
    unknown) and the schema mode.
    """
    object_names, id_to_index = _scene_object_index(scene)

    if not bool(paths.synthetic_array):
        native: dict[str, np.ndarray] = {"valid": np.asarray(paths.valid).astype(bool)}
        native["tau"] = np.asarray(paths.tau, dtype=np.float32)
        for name in PATH_ANGLE_FIELDS:
            native[name] = np.asarray(getattr(paths, name), dtype=np.float32)
        return PathGroundTruthResult(native, object_names, PATH_GT_MODE_SIONNA_NATIVE)

    valid = np.asarray(paths.valid).astype(bool)
    tau = np.asarray(paths.tau, dtype=np.float32)
    angles: dict[str, np.ndarray] = {}
    for name in PATH_ANGLE_FIELDS:
        angles[name] = np.asarray(getattr(paths, name), dtype=np.float32)

    a_real, a_imag = paths.a
    a_real = np.asarray(a_real)
    a_imag = np.asarray(a_imag)
    if a_real.ndim != 5 or a_imag.ndim != 5:
        raise RuntimeError(f"Unexpected paths.a shape: {a_real.shape}")
    num_rx, fused, num_tx, num_tx_ant, num_paths = a_real.shape
    if num_tx_ant != 1:
        raise RuntimeError(f"path_ground_truth needs num_tx_ant == 1, got {num_tx_ant}")
    num_patterns = len(paths.rx_array.antenna_pattern.patterns)
    if fused != num_patterns * rx_rows * rx_cols:
        raise RuntimeError(
            f"Unexpected fused Rx size: fused={fused}, "
            f"patterns={num_patterns}, aperture={rx_rows}x{rx_cols}"
        )
    # [rx, fused, tx, path] -> tx next to rx so path ordering aligns on (rx, tx).
    a_fused = (a_real + 1j * a_imag).astype(np.complex64)[:, :, :, 0, :]
    a_fused = np.moveaxis(a_fused, 2, 1)

    try:
        cir_result, _ = paths.cir(normalize_delays=False, out_type="numpy")
        if isinstance(cir_result, (tuple, list)):
            # Dr.Jit-style (real, imag) pair (e.g. out_type="drjit").
            cir_real = np.asarray(cir_result[0])
            cir_imag = np.asarray(cir_result[1])
            a_baseband_full = (cir_real + 1j * cir_imag).astype(np.complex64)
        else:
            # NumPy output is a single complex array.
            a_baseband_full = np.asarray(cir_result).astype(np.complex64)
        # Drop the trailing time axis (num_time_steps=1).
        if a_baseband_full.ndim == 6 and a_baseband_full.shape[-1] == 1:
            a_baseband_full = a_baseband_full[..., 0]
        if a_baseband_full.ndim == 5:
            a_baseband_full = a_baseband_full[:, :, :, 0, :]
        if a_baseband_full.shape != (num_rx, fused, num_tx, num_paths):
            raise RuntimeError(f"Unexpected baseband coefficient shape: {a_baseband_full.shape}")
        a_baseband_fused = np.moveaxis(a_baseband_full, 2, 1)
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"Could not export baseband coefficients: {exc}") from exc

    power = np.sum(np.abs(a_fused.astype(np.complex128)) ** 2, axis=2)
    order = canonical_path_order(tau, power, valid)

    a_baseband = _split_fused_coefficient(
        apply_path_order(a_baseband_fused, order, path_axis=-1),
        num_patterns=num_patterns,
        rx_rows=rx_rows,
        rx_cols=rx_cols,
    ).astype(np.complex64, copy=False)

    payload: dict[str, np.ndarray] = {
        "valid": apply_path_order(valid, order, path_axis=-1).astype(bool, copy=False),
        "tau": apply_path_order(tau, order, path_axis=-1).astype(np.float32, copy=False),
        "a_baseband": a_baseband,
    }
    for name in PATH_ANGLE_FIELDS:
        payload[name] = apply_path_order(angles[name], order, path_axis=-1).astype(
            np.float32, copy=False
        )

    # Optional per-interaction components: [depth, rx, tx, path] -> [rx, tx, path, depth].
    max_depth = 0
    interactions: np.ndarray | None = None
    raw_objects: np.ndarray | None = None
    primitives: np.ndarray | None = None
    vertices: np.ndarray | None = None
    try:
        interactions = np.asarray(paths.interactions, dtype=np.uint32)
    except Exception as exc:  # pragma: no cover - depends on Sionna backend
        warnings.warn(f"could not export paths.interactions: {exc}", stacklevel=2)
    try:
        raw_objects = np.asarray(paths.objects, dtype=np.uint32)
    except Exception as exc:  # pragma: no cover - depends on Sionna backend
        warnings.warn(f"could not export paths.objects: {exc}", stacklevel=2)
    try:
        primitives = np.asarray(paths.primitives, dtype=np.uint32)
    except Exception as exc:  # pragma: no cover - depends on Sionna backend
        warnings.warn(f"could not export paths.primitives: {exc}", stacklevel=2)
    try:
        vertices = np.asarray(paths.vertices, dtype=np.float32)
    except Exception as exc:  # pragma: no cover - depends on Sionna backend
        warnings.warn(f"could not export paths.vertices: {exc}", stacklevel=2)

    for component in (interactions, raw_objects, primitives, vertices):
        if component is not None:
            max_depth = component.shape[0]
            break

    def _sorted_depth(component: np.ndarray | None, dtype: np.dtype) -> np.ndarray:
        if component is None:
            shape = (num_rx, num_tx, num_paths, max_depth)
            if dtype == np.float32:
                return np.zeros(shape + (3,), dtype=dtype)
            fill = np.uint32(0xFFFFFFFF) if dtype == np.uint32 else np.int32(-1)
            return np.full(shape, fill, dtype=dtype)
        # Vertices carry a trailing xyz axis: [depth, rx, tx, path, 3].
        moved = (
            np.moveaxis(component, 0, -2) if component.ndim == 5 else np.moveaxis(component, 0, -1)
        )
        return apply_path_order(moved, order, path_axis=2).astype(dtype, copy=False)

    payload["interactions"] = _sorted_depth(interactions, np.dtype(np.uint32))
    payload["primitives"] = _sorted_depth(primitives, np.dtype(np.uint32))
    payload["vertices"] = _sorted_depth(vertices, np.dtype(np.float32))

    if raw_objects is None:
        payload["object_index"] = np.full(
            (num_rx, num_tx, num_paths, max_depth), -1, dtype=np.int32
        )
    else:
        moved_objects = apply_path_order(np.moveaxis(raw_objects, 0, -1), order, path_axis=2)
        payload["object_index"] = _map_object_ids(moved_objects, id_to_index)

    payload["num_interactions"] = np.sum(payload["interactions"] != 0, axis=-1).astype(
        np.int32, copy=False
    )
    return PathGroundTruthResult(payload, object_names, PATH_GT_MODE_CANONICAL)
