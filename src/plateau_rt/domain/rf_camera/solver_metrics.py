"""Solver-profile metrics for the RF-camera multi-view generator (NumPy only).

This module must never import sionna, mitsuba or drjit so it stays usable in
CPU-only unit tests and post-processing.
"""

from __future__ import annotations

import numpy as np

from plateau_rt.domain.rf_camera.camera import HEMISPHERES

# Local copies of the sionna.rt.constants.InteractionType bit flags.
# Kept as plain NumPy ints so this module stays Sionna-free.
INTERACTION_SPECULAR = np.int64(1)
INTERACTION_DIFFUSE = np.int64(2)
INTERACTION_REFRACTION = np.int64(4)
INTERACTION_DIFFRACTION = np.int64(8)


def hemisphere_energy(aperture_cfr: np.ndarray) -> dict[str, float]:
    """Return the sum of |x|**2 per hemisphere.

    Accepts ``[hemisphere, row, col, freq]`` (single view) or
    ``[rx, hemisphere, ...]`` (multi-view) layouts. All axes except the
    hemisphere axis are summed over.
    """
    cfr = np.asarray(aperture_cfr)
    if cfr.ndim == 4:
        hemi_axis = 0
    elif cfr.ndim == 5:
        hemi_axis = 1
    else:
        raise ValueError(f"aperture_cfr must be 4D or 5D, got ndim={cfr.ndim}")
    if cfr.shape[hemi_axis] != len(HEMISPHERES):
        raise ValueError(
            f"hemisphere axis has size {cfr.shape[hemi_axis]}, "
            f"expected {len(HEMISPHERES)} for {HEMISPHERES}"
        )
    axes = tuple(i for i in range(cfr.ndim) if i != hemi_axis)
    energy = np.sum(np.abs(cfr) ** 2, axis=axes)
    return {name: float(energy[i]) for i, name in enumerate(HEMISPHERES)}


def relative_cfr_difference(test: np.ndarray, reference: np.ndarray, eps: float = 1e-30) -> float:
    """Return ``||test - ref||_2 / ||ref||_2`` with exact-zero semantics.

    When the reference norm is exactly 0 and the difference norm is > 0,
    return ``inf``; when both norms are 0, return ``0.0``. Otherwise return
    ``diff / ref``. Raises on a shape mismatch.

    ``eps`` is kept for backwards compatibility but is unused.
    """
    _ = eps
    test_arr = np.asarray(test)
    ref_arr = np.asarray(reference)
    if test_arr.shape != ref_arr.shape:
        raise ValueError(f"shape mismatch: test={test_arr.shape}, reference={ref_arr.shape}")
    ref_norm = float(np.linalg.norm(ref_arr.ravel()))
    diff_norm = float(np.linalg.norm((test_arr - ref_arr).ravel()))
    if ref_norm == 0.0:
        return 0.0 if diff_norm == 0.0 else float("inf")
    return diff_norm / ref_norm


def per_view_relative_difference(test: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return :func:`relative_cfr_difference` computed per view over axis 0 (rx).

    Uses exact-zero semantics element-wise: 0/0 -> 0.0, nonzero/0 -> inf.
    No divide warnings are emitted.
    """
    test_arr = np.asarray(test)
    ref_arr = np.asarray(reference)
    if test_arr.shape != ref_arr.shape:
        raise ValueError(f"shape mismatch: test={test_arr.shape}, reference={ref_arr.shape}")
    if test_arr.ndim < 1:
        raise ValueError("inputs must have at least one axis (rx)")
    flat_test = test_arr.reshape(test_arr.shape[0], -1)
    flat_ref = ref_arr.reshape(ref_arr.shape[0], -1)
    ref_norm = np.linalg.norm(flat_ref, axis=1)
    diff_norm = np.linalg.norm(flat_test - flat_ref, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = diff_norm / ref_norm
    out = np.where(
        ref_norm == 0.0,
        np.where(diff_norm == 0.0, 0.0, np.inf),
        ratio,
    )
    return np.asarray(out, dtype=np.float64)


def count_paths_by_type(
    valid: np.ndarray,
    interactions: np.ndarray,
    *,
    num_rx_patterns: int = 1,
    num_tx_patterns: int = 1,
) -> dict[str, int | float]:
    """Count valid paths by interaction type.

    Keys are ``los, specular, diffuse, refraction, diffraction, total`` plus
    ``num_links``, ``mean_paths_per_link`` and ``max_paths_per_link``. A valid
    path counts toward every type bit that appears at any depth; LoS means
    valid with all interactions equal to zero.

    Layout convention: ``interactions.shape[1:] == valid.shape`` is required.
    The synthetic-array layout is ``valid: [num_rx, num_tx, num_paths]`` and
    the non-synthetic (explicit-array) layout is
    ``valid: [num_rx, num_rx_pat * num_rx_ant, num_tx, num_tx_pat * num_tx_ant,
    num_paths]``. In Sionna 2.0.1 the pattern and antenna axes are fused
    pattern-major (``_fuse_pattern_array_dims``): fused index
    ``pattern * num_ant + ant``. The geometry is pattern-independent, so this
    function reshapes the fused axes to ``(pattern, ant)`` in pattern-major
    order and keeps pattern index 0 on both sides (the same slicing is applied
    to ``interactions``, whose axes are shifted by one because of the leading
    depth axis). Synthetic 3D inputs ignore the pattern arguments.

    ``total`` and the per-type counts tally every valid antenna-pair entry
    after pattern de-duplication, so with an explicit array one physical path
    contributes once per antenna pair. ``num_links`` is ``rx * tx`` for the
    synthetic layout and ``rx * rx_ant * tx * tx_ant`` for the explicit layout
    after de-duplication; ``mean_paths_per_link`` is ``total / num_links`` and
    ``max_paths_per_link`` is the maximum over links of the per-link path
    count. These keys mean the same thing in both layouts.
    """
    valid_arr = np.asarray(valid, dtype=bool)
    inter_arr = np.asarray(interactions)
    if inter_arr.shape[1:] != valid_arr.shape:
        raise ValueError(
            f"interactions.shape[1:]={inter_arr.shape[1:]} must equal valid.shape={valid_arr.shape}"
        )
    if num_rx_patterns < 1 or num_tx_patterns < 1:
        raise ValueError("num_rx_patterns and num_tx_patterns must be >= 1")
    if valid_arr.ndim == 3:
        dedup_valid = valid_arr
        dedup_inter = inter_arr
        num_rx, num_tx, _num_paths = valid_arr.shape
        num_links = int(num_rx * num_tx)
        per_link = np.sum(dedup_valid, axis=2)
    elif valid_arr.ndim == 5:
        num_rx, fused_rx, num_tx, fused_tx, num_paths = valid_arr.shape
        if fused_rx % num_rx_patterns != 0:
            raise ValueError(
                f"axis 1 has size {fused_rx}, not divisible by num_rx_patterns={num_rx_patterns}"
            )
        if fused_tx % num_tx_patterns != 0:
            raise ValueError(
                f"axis 3 has size {fused_tx}, not divisible by num_tx_patterns={num_tx_patterns}"
            )
        num_rx_ant = fused_rx // num_rx_patterns
        num_tx_ant = fused_tx // num_tx_patterns
        # Pattern-major fusion: index p * num_ant + a reshapes to (pattern, ant).
        dedup_valid = valid_arr.reshape(
            num_rx,
            num_rx_patterns,
            num_rx_ant,
            num_tx,
            num_tx_patterns,
            num_tx_ant,
            num_paths,
        )[:, 0, :, :, 0, :, :]
        depth = inter_arr.shape[0]
        dedup_inter = inter_arr.reshape(
            depth,
            num_rx,
            num_rx_patterns,
            num_rx_ant,
            num_tx,
            num_tx_patterns,
            num_tx_ant,
            num_paths,
        )[:, :, 0, :, :, 0, :, :]
        num_links = int(num_rx * num_rx_ant * num_tx * num_tx_ant)
        per_link = np.sum(dedup_valid, axis=4)
    else:
        raise ValueError(
            "valid must be 3D [num_rx, num_tx, num_paths] (synthetic) or "
            "5D [num_rx, num_rx_pat * num_rx_ant, num_tx, num_tx_pat * num_tx_ant, "
            f"num_paths], got ndim={valid_arr.ndim}"
        )

    if dedup_inter.shape[0] == 0:
        any_nonzero = np.zeros_like(dedup_valid, dtype=bool)
        has_spec = has_diff = has_refr = has_diffra = np.zeros_like(dedup_valid, dtype=bool)
    else:
        any_nonzero = np.any(dedup_inter != 0, axis=0)
        has_spec = np.any((dedup_inter & INTERACTION_SPECULAR) != 0, axis=0)
        has_diff = np.any((dedup_inter & INTERACTION_DIFFUSE) != 0, axis=0)
        has_refr = np.any((dedup_inter & INTERACTION_REFRACTION) != 0, axis=0)
        has_diffra = np.any((dedup_inter & INTERACTION_DIFFRACTION) != 0, axis=0)

    is_los = dedup_valid & ~any_nonzero
    total = int(np.sum(dedup_valid))
    counts: dict[str, int | float] = {
        "los": int(np.sum(is_los)),
        "specular": int(np.sum(dedup_valid & has_spec)),
        "diffuse": int(np.sum(dedup_valid & has_diff)),
        "refraction": int(np.sum(dedup_valid & has_refr)),
        "diffraction": int(np.sum(dedup_valid & has_diffra)),
        "total": total,
        "num_links": num_links,
        "mean_paths_per_link": float(total / num_links) if num_links else 0.0,
        "max_paths_per_link": int(np.max(per_link)) if per_link.size else 0,
    }
    return counts
