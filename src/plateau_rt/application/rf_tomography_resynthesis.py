"""Validate the tomography operators against traced data (design §8 T21): L0f VS
resynthesis, BS-pattern direct-path check, eps_LoS and polarisation statistics. Sionna-free.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from plateau_rt.application import rf_tomography_gt, rf_tomography_io
from plateau_rt.domain.rf_camera import gauge
from plateau_rt.domain.rf_camera.image_sources import arrival_unit_vectors
from plateau_rt.domain.rf_camera.paths import synthesize_cfr
from plateau_rt.domain.rf_tomography import gt
from plateau_rt.domain.rf_tomography.antenna import PATTERN_KINDS, bs_pattern
from plateau_rt.domain.rf_tomography.forward_exact import atom_cfr, capture_factors
from plateau_rt.domain.rf_tomography.geometry import CaptureGeometry

__all__ = [
    "DIRECT_PATH_DB_MAX",
    "GAIN_RATIO_DB_MAX",
    "NMSE_MAX",
    "RESYNTHESIS_SCHEMA",
    "ROW_FLIP_NMSE_MIN",
    "UNMODELLED_PATH_TYPES",
    "CaptureErrors",
    "OperatorAmplitudes",
    "capture_errors",
    "direct_path_summary",
    "operator_amplitudes",
    "polarization_factor",
    "polarization_summary",
    "resynthesis_report",
    "stats",
    "unmodelled_cfr",
    "vs_resynthesis",
]

RESYNTHESIS_SCHEMA = "rf_tomo_resynthesis/1"
NMSE_MAX = 1e-3  # design §3.3 / §8 T21: L0f NMSE gate (raw and gauge-aligned)
DIRECT_PATH_DB_MAX = 0.5  # design §8 T21: tr38901 direct-path ratio gate (V-pol model)
ROW_FLIP_NMSE_MIN = 0.1  # control: flipping the element rows must break the resynthesis
# Orders 0 and 1: the operator departure direction must equal the traced one (G ratio 0 dB).
GAIN_RATIO_DB_MAX = 0.01
UNMODELLED_PATH_TYPES = ("diffraction", "diffuse")


@dataclass(frozen=True)
class OperatorAmplitudes:
    """Per-view VS atom amplitudes referenced to the operator's departure model."""

    amps: np.ndarray  # [M, V, B] complex128, sum over the member paths of each (m, v)
    physical: np.ndarray  # [M, V, B] complex128, the same sum without the gain ratio
    gain_ratio_db: np.ndarray  # [V, B, P] float64, 20 log10 |G(d_true)/G(d_op)|, NaN if no member
    num_members: np.ndarray  # [M, V] int64


@dataclass(frozen=True)
class CaptureErrors:
    """Per-capture resynthesis errors; NaN for captures without target energy."""

    nmse: np.ndarray  # [V, B]
    nmse_aligned: np.ndarray  # [V, B] after the common (phase, delay) gauge fit
    phase: np.ndarray  # [V, B] fitted gauge phase, rad
    delay: np.ndarray  # [V, B] fitted gauge delay, s
    energy: np.ndarray  # [V, B] sum |target|^2


def operator_amplitudes(
    path: gt.PathGT,
    geom: CaptureGeometry,
    path_vs: np.ndarray,
    vs_pos: np.ndarray,
    vs_bs: np.ndarray,
    *,
    pattern: str,
) -> OperatorAmplitudes:
    """Return the per-view VS atom amplitudes re-referenced to ``d_op`` (design §3.2)."""
    num_views, num_bs, num_paths = path.num_views, path.num_bs, path.num_paths
    positions = np.asarray(vs_pos, dtype=np.float64).reshape(-1, 3)
    bs_of_vs = np.asarray(vs_bs).reshape(-1).astype(np.int64)
    num_vs = positions.shape[0]
    amps = np.zeros((num_vs, num_views, num_bs), dtype=np.complex128)
    physical = np.zeros((num_vs, num_views, num_bs), dtype=np.complex128)
    gain_ratio_db = np.full((num_views, num_bs, num_paths), np.nan, dtype=np.float64)
    num_members = np.zeros((num_vs, num_views), dtype=np.int64)
    slots = np.asarray(path_vs, dtype=np.int64)
    for v in range(num_views):
        for b in range(num_bs):
            bs_rot = np.eye(3) if geom.bs_rot is None else geom.bs_rot[b]
            for p in range(num_paths):
                if not path.valid[v, b, p]:
                    continue
                m = int(slots[v, b, p])
                if m < 0:
                    continue
                if int(bs_of_vs[m]) != b:
                    raise ValueError(
                        f"path slot {(v, b, p)} maps to VS {m} of BS {int(bs_of_vs[m])}, "
                        f"expected BS {b}"
                    )
                rho = gt.effective_rho(path, geom, (v, b, p), pattern)
                d_true = arrival_unit_vectors(path.theta_t[v, b, p], path.phi_t[v, b, p])
                d_op = geom.vs_departure_dir(positions[m][None, :], v, b)[0]
                d_op = d_op / np.linalg.norm(d_op)
                ratio = (
                    bs_pattern(d_true[None], bs_rot, kind=pattern)[0]
                    / bs_pattern(d_op[None], bs_rot, kind=pattern)[0]
                )
                amps[m, v, b] += rho * ratio
                physical[m, v, b] += rho
                gain_ratio_db[v, b, p] = 20.0 * np.log10(np.abs(ratio))
                num_members[m, v] += 1
    return OperatorAmplitudes(
        amps=amps,
        physical=physical,
        gain_ratio_db=gain_ratio_db,
        num_members=num_members,
    )


def unmodelled_cfr(path: gt.PathGT, freq_offsets: np.ndarray) -> np.ndarray:
    """Return the CFR ``[V, B, 2, R, C, N]`` of the diffraction/diffuse (non-VS) paths."""
    offsets = np.asarray(freq_offsets, dtype=np.float64).reshape(-1)
    codes = [gt.PATH_TYPE_NAMES.index(name) for name in UNMODELLED_PATH_TYPES]
    selected = np.isin(gt.path_types(path), codes)
    rows, cols = path.a_baseband.shape[3], path.a_baseband.shape[4]
    out = np.zeros((path.num_views, path.num_bs, 2, rows, cols, offsets.size), dtype=np.complex128)
    baseband = np.where(selected[..., None, None, None, :], path.a_baseband, 0.0)
    tau = np.where(selected, path.tau, -1.0)
    # A single call on the full array materialises tens of GB on the 64-view profile.
    for v in range(path.num_views):
        out[v] = synthesize_cfr(
            np.asarray(baseband[v], dtype=np.complex128), tau[v][:, None, None, None, :], offsets
        )
    return out


def vs_resynthesis(
    path: gt.PathGT,
    geom: CaptureGeometry,
    gt_arrays: Mapping[str, np.ndarray],
    *,
    pattern: str,
) -> tuple[np.ndarray, np.ndarray, OperatorAmplitudes]:
    """Return ``(operator CFR, physical CFR, amplitudes)`` of the VS atoms."""
    amps = operator_amplitudes(
        path,
        geom,
        gt_arrays["path_vs"],
        gt_arrays["vs_pos"],
        gt_arrays["vs_bs"],
        pattern=pattern,
    )
    positions = np.asarray(gt_arrays["vs_pos"], dtype=np.float64).reshape(-1, 3)
    if positions.shape[0] == 0:
        rows, cols = geom.aperture_shape
        zeros = np.zeros(
            (geom.num_views, geom.num_bs, 2, rows, cols, geom.num_bins), dtype=np.complex128
        )
        return zeros, zeros.copy(), amps
    y_operator = atom_cfr(positions, amps.amps, geom, "vs", pattern=pattern)
    y_physical = atom_cfr(positions, amps.physical, geom, "vs", pattern=pattern)
    return y_operator, y_physical, amps


def capture_errors(y: np.ndarray, y_model: np.ndarray, freq_offsets: np.ndarray) -> CaptureErrors:
    """Return per-capture NMSE (raw and gauge-aligned) of ``y_model`` against ``y``."""
    observed = np.asarray(y, dtype=np.complex128)
    model = np.asarray(y_model, dtype=np.complex128)
    num_views, num_bs = observed.shape[0], observed.shape[1]
    nmse = np.full((num_views, num_bs), np.nan, dtype=np.float64)
    nmse_aligned = np.full((num_views, num_bs), np.nan, dtype=np.float64)
    phase = np.full((num_views, num_bs), np.nan, dtype=np.float64)
    delay = np.full((num_views, num_bs), np.nan, dtype=np.float64)
    energy = np.zeros((num_views, num_bs), dtype=np.float64)
    for v in range(num_views):
        for b in range(num_bs):
            obs = observed[v, b]
            energy[v, b] = float(np.sum(np.abs(obs) ** 2))
            if energy[v, b] == 0.0:
                continue
            ref = model[v, b]
            nmse[v, b] = gauge.nmse(obs, ref)
            alignment = gauge.align_common_phase_and_delay(obs, ref, freq_offsets)
            nmse_aligned[v, b] = gauge.nmse(obs, alignment.aligned_ref)
            phase[v, b] = alignment.phase_rad
            delay[v, b] = alignment.delay_s
    return CaptureErrors(
        nmse=nmse,
        nmse_aligned=nmse_aligned,
        phase=phase,
        delay=delay,
        energy=energy,
    )


def polarization_factor(
    points: np.ndarray, geom: CaptureGeometry, v: int, b: int, *, pattern: str
) -> np.ndarray:
    """Return the real V-pol co-polar factor ``[P]`` of ``points`` in capture ``(v, b)``."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    copolar = capture_factors(pts, geom, "vs", v, b, pattern=pattern, polarization="vv").gamma
    scalar = capture_factors(pts, geom, "vs", v, b, pattern=pattern, polarization="none").gamma
    return np.asarray((copolar / scalar).real, dtype=np.float64)


def stats(values: np.ndarray) -> dict[str, float | int | None]:
    """Return ``n/p50/p90/max`` over the finite entries of ``values``, ``None`` when empty."""
    finite = np.asarray(values, dtype=np.float64).ravel()
    finite = finite[np.isfinite(finite)]
    count = int(finite.size)
    if count == 0:
        return {"n": 0, "p50": None, "p90": None, "max": None}
    return {
        "n": count,
        "p50": float(np.percentile(finite, 50.0)),
        "p90": float(np.percentile(finite, 90.0)),
        "max": float(np.max(finite)),
    }


def direct_path_summary(path: gt.PathGT, geom: CaptureGeometry, *, pattern: str) -> dict[str, Any]:
    """Return the direct-path (LoS) ratio and phase statistics against the scalar model."""
    phase_vv, amp_vv = gt.los_model_error(path, geom, pattern=pattern, polarization="vv")
    phase_none, amp_none = gt.los_model_error(path, geom, pattern=pattern, polarization="none")
    los_visible = np.isfinite(phase_vv)
    num_los = int(np.count_nonzero(los_visible))
    with np.errstate(invalid="ignore", over="ignore"):
        r_vv = 10.0 ** (amp_vv / 20.0) * np.exp(1j * phase_vv)
        r_none = 10.0 ** (amp_none / 20.0) * np.exp(1j * phase_none)
    worst: list[int] | None = None
    if num_los > 0:
        masked = np.where(los_visible, np.abs(amp_vv), -np.inf)
        index = np.unravel_index(int(np.argmax(masked)), masked.shape)
        worst = [int(index[0]), int(index[1])]
    iso_control: dict[str, float | int | None] | None = None
    if pattern == "tr38901":
        _, amp_iso = gt.los_model_error(path, geom, pattern="iso", polarization="vv")
        iso_control = stats(np.abs(amp_iso))
    return {
        "num_los": num_los,
        "pattern_db": stats(np.abs(amp_vv)),
        "worst_pattern_capture": worst,
        "scalar_db": stats(np.abs(amp_none)),
        "rel_error_vv": stats(np.abs(1.0 - r_vv)),
        "rel_error_none": stats(np.abs(1.0 - r_none)),
        "eps_los_deg_none": stats(np.degrees(np.abs(phase_none))),
        "eps_los_deg_vv": stats(np.degrees(np.abs(phase_vv))),
        "iso_control_db": iso_control,
    }


def polarization_summary(
    path: gt.PathGT,
    geom: CaptureGeometry,
    gt_arrays: Mapping[str, np.ndarray],
    *,
    pattern: str,
) -> dict[str, Any]:
    """Return the LoS and first-order V-pol co-polar factor statistics."""
    los_visible = np.asarray(gt_arrays["los_visible"], dtype=bool)
    los_factors: list[float] = []
    for v in range(los_visible.shape[0]):
        for b in range(los_visible.shape[1]):
            if not los_visible[v, b]:
                continue
            los_factors.append(
                float(polarization_factor(geom.bs_pos[b][None], geom, v, b, pattern=pattern)[0])
            )

    specular_code = gt.PATH_TYPE_NAMES.index("specular")
    vs_order = np.asarray(gt_arrays["vs_order"]).reshape(-1)
    vs_visibility = np.asarray(gt_arrays["vs_visibility"], dtype=bool)
    vs_path_type = np.asarray(gt_arrays["vs_path_type"])
    vs_pos = np.asarray(gt_arrays["vs_pos"], dtype=np.float64).reshape(-1, 3)
    vs_bs = np.asarray(gt_arrays["vs_bs"]).reshape(-1).astype(np.int64)
    first_order_factors: list[float] = []
    per_vs: dict[int, list[float]] = {}
    for m in range(vs_order.shape[0]):
        if int(vs_order[m]) != 1:
            continue
        for v in range(vs_visibility.shape[1]):
            if not vs_visibility[m, v]:
                continue
            if int(vs_path_type[m, v]) != specular_code:
                continue
            value = float(
                polarization_factor(vs_pos[m][None], geom, v, int(vs_bs[m]), pattern=pattern)[0]
            )
            first_order_factors.append(value)
            per_vs.setdefault(m, []).append(value)

    mixed = sum(
        1
        for values in per_vs.values()
        if any(value < 0.0 for value in values) and any(value > 0.0 for value in values)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        los_db = stats(np.abs(20.0 * np.log10(np.abs(los_factors))))
        first_db = stats(np.abs(20.0 * np.log10(np.abs(first_order_factors))))
    return {
        "los": {
            "n": len(los_factors),
            "min": min(los_factors) if los_factors else None,
            "db": los_db,
            "negative": int(sum(1 for value in los_factors if value < 0.0)),
        },
        "first_order": {
            "n": len(first_order_factors),
            "db": first_db,
            "negative": int(sum(1 for value in first_order_factors if value < 0.0)),
            "num_vs": len(per_vs),
            "vs_mixed_sign": mixed,
        },
    }


def _label(view_ids: Any, bs_ids: Any, v: int, b: int) -> str:
    """Return the ``"<view_id>/<bs_id>"`` label of a capture."""
    return f"{view_ids[v]}/{bs_ids[b]}"


def _worst_label(values: np.ndarray, view_ids: Any, bs_ids: Any) -> tuple[str | None, float | None]:
    """Return the label and value of the finite maximum of ``values`` (or ``(None, None)``)."""
    finite = np.isfinite(values)
    if not bool(np.any(finite)):
        return None, None
    masked = np.where(finite, values, -np.inf)
    v, b = np.unravel_index(int(np.argmax(masked)), masked.shape)
    return _label(view_ids, bs_ids, int(v), int(b)), float(masked[v, b])


def _load_gt(
    dataset: Path | str,
    data: rf_tomography_io.TomographyDataset,
    gt_file: Path | str | None,
) -> tuple[dict[str, np.ndarray], str, list[str]]:
    """Return ``(gt_arrays, gt_source, failures)`` for the requested ground truth."""
    failures: list[str] = []
    if gt_file is not None:
        gt_arrays = rf_tomography_gt.load_tomography_gt(gt_file)
        gt_source = "file"
    else:
        raw = data.manifest.raw
        entry = raw.get("tomography_gt") if isinstance(raw, Mapping) else None
        if isinstance(entry, Mapping) and isinstance(entry.get("artifact"), str):
            artifact = Path(str(entry["artifact"]))
            if not artifact.is_absolute():
                artifact = data.manifest.root / artifact
            gt_arrays = rf_tomography_gt.load_tomography_gt(artifact)
            gt_source = "registered"
        else:
            gt_arrays = rf_tomography_gt.build_tomography_gt(dataset, surfaces=False)
            gt_source = "built"
    if gt_source in ("file", "registered"):
        stored = str(np.asarray(gt_arrays["source_path_gt_sha256"]).reshape(()))
        path_gt = data.manifest.path_geometry_gt
        assert path_gt is not None  # load_path_gt already required it
        actual = rf_tomography_io.sha256_file(path_gt.path)
        if stored != actual:
            failures.append("stale tomography GT: source_path_gt_sha256 does not match the path GT")
    return gt_arrays, gt_source, failures


def _gain_ratio_by_order(
    amps: OperatorAmplitudes,
    path: gt.PathGT,
    gt_arrays: Mapping[str, np.ndarray],
) -> dict[str, dict[str, float | int | None]]:
    """Return ``|gain_ratio_db|`` percentiles grouped by VS order."""
    vs_order = np.asarray(gt_arrays["vs_order"]).reshape(-1)
    path_vs = np.asarray(gt_arrays["path_vs"])
    groups: dict[str, list[float]] = {str(int(o)): [] for o in np.unique(vs_order)}
    for v in range(path.num_views):
        for b in range(path.num_bs):
            for p in range(path.num_paths):
                if not path.valid[v, b, p]:
                    continue
                m = int(path_vs[v, b, p])
                if m < 0:
                    continue
                value = float(amps.gain_ratio_db[v, b, p])
                if not np.isfinite(value):
                    continue
                groups.setdefault(str(int(vs_order[m])), []).append(abs(value))
    ordered = sorted(groups.items(), key=lambda item: int(item[0]))
    return {key: stats(np.asarray(values)) for key, values in ordered}


def resynthesis_report(
    dataset: Path | str,
    *,
    gt_file: Path | str | None = None,
    nmse_max: float = NMSE_MAX,
    direct_db_max: float = DIRECT_PATH_DB_MAX,
) -> dict[str, Any]:
    """Return the JSON-serialisable T21 resynthesis, pattern and polarisation report."""
    data = rf_tomography_io.load_dataset(dataset)
    pattern = data.tx_pattern
    if pattern is None or pattern not in PATTERN_KINDS:
        raise ValueError(f"dataset tx_pattern must be one of {PATTERN_KINDS}, got {pattern!r}")
    path = rf_tomography_gt.load_path_gt(data.manifest)
    gt_arrays, gt_source, failures = _load_gt(dataset, data, gt_file)

    num_views = data.geom.num_views
    num_bs = data.geom.num_bs
    freq = np.asarray(data.geom.freq_offsets, dtype=np.float64)
    unmodelled = unmodelled_cfr(path, freq)
    y_clean = np.asarray(data.y_clean, dtype=np.complex128)
    target = y_clean - unmodelled
    y_operator, y_physical, amps = vs_resynthesis(path, data.geom, gt_arrays, pattern=pattern)
    errors = capture_errors(target, y_operator, freq)
    physical = capture_errors(target, y_physical, freq)
    flip = capture_errors(target, y_operator[:, :, :, ::-1, :, :], freq)

    clean_energy = np.sum(np.abs(y_clean) ** 2, axis=(2, 3, 4, 5))
    unmodelled_energy = np.sum(np.abs(unmodelled) ** 2, axis=(2, 3, 4, 5))
    with np.errstate(divide="ignore", invalid="ignore"):
        unmodelled_fraction = np.where(clean_energy > 0.0, unmodelled_energy / clean_energy, np.nan)

    view_ids = data.manifest.view_ids
    bs_ids = data.manifest.bs_ids
    direct_path = direct_path_summary(path, data.geom, pattern=pattern)
    polarization = polarization_summary(path, data.geom, gt_arrays, pattern=pattern)
    worst_aligned, _ = _worst_label(errors.nmse_aligned, view_ids, bs_ids)

    l0f = {
        "num_captures": int(np.count_nonzero(np.isfinite(errors.nmse))),
        "empty_captures": int(np.count_nonzero(errors.energy == 0.0)),
        "nmse": stats(errors.nmse),
        "nmse_aligned": stats(errors.nmse_aligned),
        "worst_capture": worst_aligned,
        "gauge_phase_deg": stats(np.degrees(np.abs(errors.phase))),
        "gauge_delay_ps": stats(np.abs(errors.delay) * 1e12),
        "unmodelled_fraction": stats(unmodelled_fraction),
        "physical_nmse_aligned": stats(physical.nmse_aligned),
        "gain_ratio_db_by_order": _gain_ratio_by_order(amps, path, gt_arrays),
        "multi_member_views": int(np.count_nonzero(amps.num_members >= 2)),
    }
    iso_control = direct_path["iso_control_db"]
    iso_direct_db_max = None if iso_control is None else iso_control["max"]
    controls = {
        "row_flip_nmse_aligned": stats(flip.nmse_aligned),
        "iso_direct_db_max": iso_direct_db_max,
    }

    if l0f["num_captures"] == 0:
        failures.append("l0f: no capture with energy")
    for key, values in (("nmse", errors.nmse), ("nmse_aligned", errors.nmse_aligned)):
        label, value = _worst_label(values, view_ids, bs_ids)
        if value is not None and value >= nmse_max:
            failures.append(f"l0f: max {key} {value:.3e} >= {nmse_max:.1e} at {label}")

    num_los = int(direct_path["num_los"])
    pattern_max = direct_path["pattern_db"]["max"]
    if num_los == 0:
        failures.append("direct path: no LoS-visible capture")
    elif pattern_max is not None and pattern_max > direct_db_max:
        worst_capture = direct_path["worst_pattern_capture"]
        label = _label(view_ids, bs_ids, int(worst_capture[0]), int(worst_capture[1]))
        failures.append(
            f"direct path: max |ratio| {pattern_max:.3f} dB > {direct_db_max} dB at {label}"
        )

    for order in ("0", "1"):
        ratio_max = l0f["gain_ratio_db_by_order"].get(order, {}).get("max")
        if ratio_max is not None and ratio_max > GAIN_RATIO_DB_MAX:
            failures.append(
                f"l0f: order-{order} departure gain ratio {ratio_max:.3e} dB > "
                f"{GAIN_RATIO_DB_MAX} dB (operator departure model)"
            )

    flip_p50 = controls["row_flip_nmse_aligned"]["p50"]
    if flip_p50 is not None and flip_p50 < ROW_FLIP_NMSE_MIN:
        failures.append(f"control: row-flip NMSE p50 {flip_p50:.3e} < {ROW_FLIP_NMSE_MIN}")
    if pattern == "tr38901" and num_los > 0 and iso_direct_db_max is not None:
        if iso_direct_db_max <= direct_db_max:
            failures.append(
                "control: iso pattern indistinguishable from tr38901 "
                f"(max {iso_direct_db_max:.3f} dB <= {direct_db_max} dB)"
            )

    return {
        "schema": RESYNTHESIS_SCHEMA,
        "dataset": str(data.manifest.root),
        "pattern": pattern,
        "gt_source": gt_source,
        "num_views": num_views,
        "num_bs": num_bs,
        "num_vs": int(np.asarray(gt_arrays["vs_pos"]).reshape(-1, 3).shape[0]),
        "thresholds": {
            "nmse_max": nmse_max,
            "direct_db_max": direct_db_max,
            "row_flip_nmse_min": ROW_FLIP_NMSE_MIN,
            "gain_ratio_db_max": GAIN_RATIO_DB_MAX,
        },
        "l0f": l0f,
        "direct_path": direct_path,
        "polarization": polarization,
        "controls": controls,
        "failures": failures,
        "ok": not failures,
    }
