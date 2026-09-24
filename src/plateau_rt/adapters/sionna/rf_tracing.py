"""Sionna RT tracing shared by the RF-camera generators.

The RF camera is a planar UE receive aperture illuminated by one active BS
antenna/port. These helpers configure that geometry, trace paths and return
per-receiver aperture CFRs in the ``[row, col, frequency]`` layout used by
:mod:`plateau_rt.domain.rf_camera`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sionna.rt import PathSolver, PlanarArray

from plateau_rt.domain.rf_camera.imaging import split_pattern_axis

PATH_ANGLE_FIELDS = ("theta_t", "phi_t", "theta_r", "phi_r")


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


def trace_paths(scene: Any, *, max_depth: int, synthetic_array: bool, seed: int) -> Any:
    """Trace LoS, specular reflection and refraction paths for all Tx/Rx."""
    return PathSolver()(
        scene=scene,
        max_depth=max_depth,
        los=True,
        specular_reflection=True,
        diffuse_reflection=False,
        refraction=True,
        synthetic_array=synthetic_array,
        seed=seed,
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
    expected = (num_rx, num_patterns * rx_rows * rx_cols, 1, 1, 1, len(frequency_offsets_hz))
    if cfr.shape != expected:
        raise RuntimeError(
            f"Unexpected Sionna Paths.cfr shape: expected={expected}, actual={cfr.shape}"
        )

    return np.stack(
        [
            split_pattern_axis(
                cfr[rx, :, 0, 0, 0, :], num_patterns=num_patterns, rows=rx_rows, cols=rx_cols
            )
            for rx in range(num_rx)
        ]
    )


def path_attributes(paths: Any, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Export ``Paths`` attributes as NumPy arrays, skipping unavailable ones."""
    payload: dict[str, np.ndarray] = {}
    for name in names:
        try:
            payload[name] = np.asarray(getattr(paths, name))
        except Exception as exc:  # pragma: no cover - depends on Sionna backend
            print(f"Warning: could not export paths.{name}: {exc}")
    return payload
