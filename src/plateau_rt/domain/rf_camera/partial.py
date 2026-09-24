"""Partial and summary observations of a multi-view RF-camera dataset (NumPy only).

A full RF-camera observation is a complex aperture CFR with shape
``[bs, hemisphere, row, col, frequency_offset]`` (the normalised 5-D layout
returned by
:func:`plateau_rt.application.rf_dataset_manifest.RFDatasetManifest.load_aperture_cfr`).
This module provides the small, deterministic building blocks used to derive a
*partial* observation (a subset of views, a masked aperture, a contiguous
frequency subband) or a *summary* of it (per-element power, or one dominant
delay per (view, BS)).

Conventions:
    * An element mask is a bool array ``[rows, cols]`` where ``True`` means the
      element is kept. Masked-out elements are zeroed but the array shape is
      preserved, so a partial sample stays aligned with its full sample. The
      mask broadcasts over the bs, hemisphere and frequency axes.
    * A subband is a contiguous ``START:STOP`` slice (Python semantics, ``STOP``
      exclusive) so the kept frequency offsets stay uniformly spaced.
    * The dominant delay of one (view, BS) pair is a scalar: the IFFT of each
      kept front-hemisphere element CFR, power summed incoherently over
      elements, then the argmax. The delay is reported modulo the unambiguous
      period. :func:`view_dominant_delay` takes ONE front CFR
      ``[rows, cols, freq]`` and is called once per BS.
"""

from __future__ import annotations

import numpy as np

from plateau_rt.domain.rf_camera.delay import angular_cfr_to_delay, dominant_delay

ELEMENT_MASK_KINDS = ("none", "random", "every_other_row", "every_other_col", "checkerboard")

SUMMARY_KINDS = ("none", "power", "delay")

# Independent random streams so the view draw and the mask draw never share a
# generator state for the same seed.
VIEW_STREAM = 0
MASK_STREAM = 1


def _half_up_count(fraction: float, n: int) -> int:
    """Return ``max(1, floor(fraction * n + 0.5))`` clamped to ``n`` (round half up)."""
    count = max(1, int(np.floor(float(fraction) * int(n) + 0.5)))
    return min(count, int(n))


def select_views(num_views: int, fraction: float, seed: int) -> np.ndarray:
    """Return sorted unique view indices keeping ``round-half-up(fraction * num_views)`` views.

    At least one view is kept. The draw uses
    ``np.random.default_rng([seed, VIEW_STREAM])`` without replacement, so the
    same seed always gives the same indices, independently of the element-mask
    stream (``MASK_STREAM``). Counts use round-half-up
    (``floor(fraction * n + 0.5)``), not banker's rounding.
    """
    if not isinstance(num_views, (int, np.integer)) or int(num_views) < 1:
        raise ValueError("num_views must be a positive integer")
    num_views = int(num_views)
    fraction = float(fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    count = _half_up_count(fraction, num_views)
    rng = np.random.default_rng([seed, VIEW_STREAM])
    if count == num_views:
        return np.arange(num_views, dtype=np.int64)
    return np.sort(rng.choice(num_views, size=count, replace=False)).astype(np.int64)


def element_mask(
    rows: int, cols: int, kind: str, *, fraction: float = 0.5, seed: int = 0
) -> np.ndarray:
    """Return a bool ``[rows, cols]`` element mask (``True`` = kept).

    Kinds: ``"none"`` keeps everything; ``"every_other_row"`` keeps even rows;
    ``"every_other_col"`` keeps even columns; ``"checkerboard"`` keeps elements
    with ``(row + col) % 2 == 0``; ``"random"`` keeps
    ``round-half-up(fraction * rows * cols)`` elements (at least one) drawn with
    ``np.random.default_rng([seed, MASK_STREAM])``. Counts use round-half-up
    (``floor(fraction * n + 0.5)``), not banker's rounding.
    """
    if not isinstance(rows, (int, np.integer)) or int(rows) < 1:
        raise ValueError("rows must be a positive integer")
    if not isinstance(cols, (int, np.integer)) or int(cols) < 1:
        raise ValueError("cols must be a positive integer")
    rows, cols = int(rows), int(cols)
    if kind not in ELEMENT_MASK_KINDS:
        raise ValueError(f"unknown element mask kind: {kind!r}")
    fraction = float(fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")

    if kind == "none":
        return np.ones((rows, cols), dtype=bool)
    if kind == "every_other_row":
        return (np.arange(rows) % 2 == 0)[:, None].repeat(cols, axis=1)
    if kind == "every_other_col":
        return (np.arange(cols) % 2 == 0)[None, :].repeat(rows, axis=0)
    if kind == "checkerboard":
        rr, cc = np.indices((rows, cols))
        return ((rr + cc) % 2 == 0).astype(bool)
    total = rows * cols
    count = _half_up_count(fraction, total)
    rng = np.random.default_rng([seed, MASK_STREAM])
    flat = np.zeros(total, dtype=bool)
    flat[rng.choice(total, size=count, replace=False)] = True
    return flat.reshape((rows, cols))


def apply_element_mask(aperture_cfr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Zero masked-out aperture elements, keeping the array shape.

    ``aperture_cfr`` has shape ``[bs, hemisphere, row, col, freq]`` and ``mask``
    has shape ``[rows, cols]`` with ``True`` = kept. The mask broadcasts over
    the bs, hemisphere and frequency axes.
    """
    cfr = np.asarray(aperture_cfr)
    mask_arr = np.asarray(mask, dtype=bool)
    if cfr.ndim != 5:
        raise ValueError("aperture_cfr must have shape [bs, hemisphere, row, col, freq]")
    if mask_arr.shape != cfr.shape[2:4]:
        raise ValueError(
            f"mask shape {mask_arr.shape} does not match aperture rows/cols {cfr.shape[2:4]}"
        )
    return (cfr * mask_arr[None, None, :, :, None]).astype(cfr.dtype, copy=False)


def parse_subband(spec: str | None, num_bins: int) -> tuple[int, int]:
    """Parse a ``"START:STOP"`` subband spec (Python slice semantics).

    ``None`` means the full band ``(0, num_bins)``. Raises ``ValueError`` on a
    malformed spec or when ``0 <= START`` and ``STOP - START >= 1`` and
    ``STOP <= num_bins`` is violated.
    """
    if not isinstance(num_bins, (int, np.integer)) or int(num_bins) < 1:
        raise ValueError("num_bins must be a positive integer")
    num_bins = int(num_bins)
    if spec is None:
        return (0, num_bins)
    if not isinstance(spec, str):
        raise ValueError(f"bad subband spec: {spec!r}")
    parts = spec.split(":")
    if len(parts) != 2:
        raise ValueError(f"bad subband spec: {spec!r}; expected 'START:STOP'")
    try:
        start, stop = int(parts[0].strip()), int(parts[1].strip())
    except ValueError:
        raise ValueError(f"bad subband spec: {spec!r}; expected 'START:STOP'") from None
    if not 0 <= start <= num_bins:
        raise ValueError(f"bad subband {spec!r}: START must satisfy 0 <= START <= {num_bins}")
    if not start < stop <= num_bins:
        raise ValueError(
            f"bad subband {spec!r}: must satisfy START < STOP <= {num_bins} "
            "(at least one bin, STOP exclusive)"
        )
    return (start, stop)


def select_subband(
    aperture_cfr: np.ndarray,
    frequency_offsets_hz: np.ndarray,
    start: int,
    stop: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice a contiguous frequency subband from an aperture CFR and its offsets.

    ``aperture_cfr`` has shape ``[bs, hemisphere, row, col, freq]``; the slice
    is taken on the last axis.
    """
    cfr = np.asarray(aperture_cfr)
    offsets = np.asarray(frequency_offsets_hz)
    if cfr.ndim != 5:
        raise ValueError("aperture_cfr must have shape [bs, hemisphere, row, col, freq]")
    num_bins = cfr.shape[-1]
    if offsets.ndim != 1 or offsets.shape[0] != num_bins:
        raise ValueError("frequency_offsets_hz must match the CFR frequency axis")
    if (
        not isinstance(start, (int, np.integer))
        or not isinstance(stop, (int, np.integer))
        or not 0 <= int(start) < int(stop) <= num_bins
    ):
        raise ValueError(
            f"bad subband [{start}:{stop}] for {num_bins} bins: need 0 <= START < STOP <= num_bins"
        )
    return cfr[..., start:stop], offsets[start:stop]


def element_power(aperture_cfr: np.ndarray) -> np.ndarray:
    """Return per-element power ``[bs, hemisphere, row, col]`` (sum of |CFR|^2 over freq)."""
    cfr = np.asarray(aperture_cfr)
    if cfr.ndim != 5:
        raise ValueError("aperture_cfr must have shape [bs, hemisphere, row, col, freq]")
    return np.sum(np.abs(cfr) ** 2, axis=-1).astype(np.float32)


def hemisphere_total_power(aperture_cfr: np.ndarray) -> np.ndarray:
    """Return total power per (BS, hemisphere) ``[bs, hemisphere]`` (sum over elements and freq)."""
    cfr = np.asarray(aperture_cfr)
    if cfr.ndim != 5:
        raise ValueError("aperture_cfr must have shape [bs, hemisphere, row, col, freq]")
    return np.sum(np.abs(cfr, dtype=np.float64) ** 2, axis=(2, 3, 4)).astype(np.float64)


def view_dominant_delay(
    front_cfr: np.ndarray,
    frequency_offsets_hz: np.ndarray,
    element_mask: np.ndarray | None = None,
) -> dict[str, float | bool]:
    """Return the scalar dominant delay of one (view, BS) front-hemisphere CFR.

    Each kept element's CFR is IFFTed to a delay profile, profiles are summed
    incoherently over elements, and the argmax gives the delay (modulo the
    unambiguous period ``1 / delta_f``). Needs at least 2 frequency bins and at
    least one kept element, else ``ValueError``.

    ``front_cfr`` has shape ``[rows, cols, freq]`` (ONE BS's front hemisphere;
    call once per BS). Returns ``delay_s``, ``power``, ``delay_resolution_s``,
    ``unambiguous_delay_s`` and ``valid``. ``valid`` only means "non-zero
    front-hemisphere signal": it is True iff ``power > 0`` and finite; when not
    valid (e.g. the source lies entirely in the back hemisphere so the front
    CFR is all zeros), ``delay_s`` is ``float("nan")`` and ``power == 0.0``.
    Check ``valid`` before trusting ``delay_s``.
    """
    cfr = np.asarray(front_cfr)
    offsets = np.asarray(frequency_offsets_hz, dtype=np.float64)
    if cfr.ndim != 3:
        raise ValueError("front_cfr must have shape [rows, cols, freq]")
    if offsets.ndim != 1 or offsets.shape[0] != cfr.shape[-1]:
        raise ValueError("frequency_offsets_hz must match the CFR frequency axis")
    if cfr.shape[-1] < 2:
        raise ValueError("at least two frequency bins are required for delay imaging")
    rows, cols = cfr.shape[0], cfr.shape[1]
    if element_mask is None:
        kept = cfr.reshape(rows * cols, cfr.shape[-1])
    else:
        mask_arr = np.asarray(element_mask, dtype=bool)
        if mask_arr.shape != (rows, cols):
            raise ValueError(
                f"element mask shape {mask_arr.shape} does not match aperture {(rows, cols)}"
            )
        kept = cfr[mask_arr]
        if kept.shape[0] == 0:
            raise ValueError("element mask keeps no elements")

    volume = angular_cfr_to_delay(kept[:, None, :], offsets)
    profile = np.sum(np.abs(volume.cir) ** 2, axis=(0, 1))
    _, delay, power = dominant_delay(profile[None, None, :], volume.delay_s)
    power_f = float(power[0, 0])
    valid = bool(np.isfinite(power_f) and power_f > 0.0)
    delay_f = float(delay[0, 0]) if valid else float("nan")
    return {
        "delay_s": delay_f,
        "power": power_f,
        "delay_resolution_s": float(volume.delay_resolution_s),
        "unambiguous_delay_s": float(volume.unambiguous_delay_s),
        "valid": valid,
    }
