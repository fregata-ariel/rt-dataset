"""Rebuild RF-camera development parameters from a dataset manifest (NumPy only).

The writer stores its :class:`~plateau_rt.adapters.sionna.rf_camera_dataset.RFMultiViewConfig`
in the manifest's ``config`` mapping. This module turns the subset needed by
:mod:`plateau_rt.domain.rf_camera.develop` back into :class:`DevelopParams`, so
a viewer can redevelop a dataset without importing Sionna.
"""

from __future__ import annotations

import numpy as np

from plateau_rt.application.rf_dataset_manifest import ManifestError, RFDatasetManifest
from plateau_rt.domain.rf_camera.develop import DevelopParams

_INT_KEYS = ("fft_rows", "fft_cols", "rx_rows", "rx_cols")
_FLOAT_KEYS = ("horizontal_spacing_lambda", "vertical_spacing_lambda", "phase_floor_db")


def develop_params_from_manifest(dataset: RFDatasetManifest) -> DevelopParams:
    """Return the development parameters stored in ``dataset.config``.

    Both schema v2 and v3 carry the same config keys. Raises
    :class:`ManifestError` naming the offending key when it is missing, has the
    wrong type or yields invalid :class:`DevelopParams`.
    """
    config = dataset.config
    int_values: dict[str, int] = {}
    for key in _INT_KEYS:
        if key not in config:
            raise ManifestError(f"'config' is missing required key {key!r}")
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ManifestError(f"'config' key {key!r} must be an integer, got {value!r}")
        int_values[key] = int(value)

    float_values: dict[str, float] = {}
    for key in _FLOAT_KEYS:
        if key not in config:
            raise ManifestError(f"'config' is missing required key {key!r}")
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ManifestError(f"'config' key {key!r} must be a finite number, got {value!r}")
        number = float(value)
        if not np.isfinite(number):
            raise ManifestError(f"'config' key {key!r} must be a finite number, got {value!r}")
        float_values[key] = number

    try:
        return DevelopParams(
            fft_rows=int_values["fft_rows"],
            fft_cols=int_values["fft_cols"],
            rx_rows=int_values["rx_rows"],
            rx_cols=int_values["rx_cols"],
            horizontal_spacing_lambda=float_values["horizontal_spacing_lambda"],
            vertical_spacing_lambda=float_values["vertical_spacing_lambda"],
            phase_floor_db=float_values["phase_floor_db"],
        )
    except ValueError as exc:
        raise ManifestError(str(exc)) from exc
