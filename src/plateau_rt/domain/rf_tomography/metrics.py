"""Core metrics for tomography reconstructions (M1, M2, M3, M5 and M6).

Non-maximum-suppression peak finding with sub-voxel refinement, Hungarian
detection matching, FROC / average-precision scoring, decomposed localisation
errors, global-phase / global-scale NMSE, wrapped gauge errors, surface
reconstruction scores (M2), reflection-plane matching (M3) and stratified
recall.

NumPy/SciPy only: nothing in this module may import Sionna, Mitsuba or Dr.Jit.
Arrays are float64 / complex128 / int64 with SI units (metres, seconds, radians).
"""

from __future__ import annotations

import math
import operator
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree

from plateau_rt.domain.rf_camera.delay import circular_delay_error_s
from plateau_rt.domain.rf_tomography.geometry import VoxelGrid

_AX = np.array([-1.0, 0.0, 1.0])
_GX, _GY, _GZ = np.meshgrid(_AX, _AX, _AX, indexing="ij")
_OX = _GX.ravel().astype(np.int64)
_OY = _GY.ravel().astype(np.int64)
_OZ = _GZ.ravel().astype(np.int64)
_FX = _GX.ravel()
_FY = _GY.ravel()
_FZ = _GZ.ravel()
_DESIGN = np.stack(
    [
        np.ones(27),
        _FX,
        _FY,
        _FZ,
        _FX**2,
        _FY**2,
        _FZ**2,
        _FX * _FY,
        _FX * _FZ,
        _FY * _FZ,
    ],
    axis=1,
)
_DESIGN_PINV = np.linalg.pinv(_DESIGN)


@dataclass(frozen=True)
class Peaks:
    """Refined NMS peaks of a density map."""

    positions: np.ndarray
    values: np.ndarray
    indices: np.ndarray
    offsets: np.ndarray


@dataclass(frozen=True)
class Matching:
    """One-to-one detection to ground-truth assignment inside a gate."""

    det_idx: np.ndarray
    gt_idx: np.ndarray
    distance: np.ndarray
    num_det: int
    num_gt: int

    @property
    def tp(self) -> int:
        """Number of matched pairs."""
        return int(self.det_idx.shape[0])

    @property
    def fp(self) -> int:
        """Number of unmatched detections."""
        return int(self.num_det) - int(self.det_idx.shape[0])

    @property
    def fn(self) -> int:
        """Number of unmatched ground-truth points."""
        return int(self.num_gt) - int(self.det_idx.shape[0])


@dataclass(frozen=True)
class FrocCurve:
    """Free-response ROC curve over detection-score thresholds."""

    thresholds: np.ndarray
    num_det: np.ndarray
    tp: np.ndarray
    fp: np.ndarray
    recall: np.ndarray
    precision: np.ndarray
    weighted_recall: np.ndarray
    weighted_precision: np.ndarray
    fa_per_1000m3: np.ndarray
    num_gt: int


@dataclass(frozen=True)
class LocationError:
    """Signed localisation error split into range/horizontal/vertical."""

    range: np.ndarray
    horizontal: np.ndarray
    vertical: np.ndarray
    total: np.ndarray

    def summary(self) -> dict[str, float]:
        """Median and P90 of the absolute components and the total error."""
        keys = (
            "range_median",
            "range_p90",
            "horizontal_median",
            "horizontal_p90",
            "vertical_median",
            "vertical_p90",
            "total_median",
            "total_p90",
        )
        if self.total.shape[0] == 0:
            return {key: float("nan") for key in keys}
        parts = (
            np.abs(self.range),
            np.abs(self.horizontal),
            np.abs(self.vertical),
            self.total,
        )
        out: dict[str, float] = {}
        for key, values in zip(("range", "horizontal", "vertical", "total"), parts, strict=True):
            out[f"{key}_median"] = float(np.median(values))
            out[f"{key}_p90"] = float(np.percentile(values, 90))
        return out


@dataclass(frozen=True)
class GaugeErrors:
    """Wrapped phase and absolute delay errors after global-phase removal."""

    phase: np.ndarray
    delay: np.ndarray
    global_phase: float


def _as_point_cloud(value: np.ndarray, name: str) -> np.ndarray:
    """Return ``value`` as float64 ``[N, 3]`` (empty input becomes ``[0, 3]``)."""
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a real array") from error
    if arr.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must have shape [N, 3]")
    return arr


def nms_peaks(
    density: np.ndarray,
    grid: VoxelGrid,
    radius: float,
    refine: bool = True,
    *,
    min_value: float | None = None,
    max_peaks: int | None = None,
) -> Peaks:
    """Detect local maxima with greedy suppression and quadratic refinement."""
    if not isinstance(grid, VoxelGrid):
        raise ValueError("grid must be a VoxelGrid")
    if np.iscomplexobj(density):
        raise ValueError("density must be real")
    try:
        raw = np.asarray(density, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("density must be a real array") from error
    if raw.shape != tuple(grid.shape):
        raise ValueError("density shape must match grid.shape")
    try:
        radius_f = float(radius)
    except (TypeError, ValueError) as error:
        raise ValueError("radius must be finite and >= 0") from error
    if not np.isfinite(radius_f) or radius_f < 0.0:
        raise ValueError("radius must be finite and >= 0")
    if min_value is not None:
        try:
            min_value = float(min_value)
        except (TypeError, ValueError) as error:
            raise ValueError("min_value must be a real scalar") from error
    limit: int | None = None
    if max_peaks is not None:
        if isinstance(max_peaks, bool):
            raise ValueError("max_peaks must be None or >= 1")
        try:
            limit = int(operator.index(max_peaks))
        except TypeError as error:
            raise ValueError("max_peaks must be None or >= 1") from error
        if limit < 1:
            raise ValueError("max_peaks must be None or >= 1")

    work = np.where(np.isfinite(raw), raw, -np.inf)
    local_max = work == maximum_filter(work, size=3, mode="constant", cval=-np.inf)
    candidates = np.isfinite(work) & local_max
    if min_value is not None:
        candidates &= work > min_value
    flats = np.flatnonzero(candidates)
    if flats.shape[0] == 0:
        return Peaks(
            positions=np.zeros((0, 3), dtype=np.float64),
            values=np.zeros((0,), dtype=np.float64),
            indices=np.zeros((0,), dtype=np.int64),
            offsets=np.zeros((0, 3), dtype=np.float64),
        )
    order = np.lexsort((flats, -work.ravel()[flats]))
    ordered = flats[order]

    if radius_f == 0.0:
        accepted = list(ordered if limit is None else ordered[:limit])
    else:
        nx, ny, nz = (int(v) for v in grid.shape)
        spacing = float(grid.spacing)
        max_step = int(math.floor(radius_f / spacing + 1e-9))
        reach = radius_f * (1.0 + 1e-9) + 1e-12
        cube = (2 * max_step + 1) ** 3
        cand_multi = np.column_stack(np.unravel_index(ordered, tuple(grid.shape)))
        cand_multi = np.asarray(cand_multi, dtype=np.int64)
        accepted = []
        if cube > 1_000_000:
            bound2 = reach * reach / (spacing * spacing)
            kept = np.empty((ordered.shape[0], 3), dtype=np.int64)
            count = 0
            for k in range(ordered.shape[0]):
                cand = cand_multi[k]
                if count > 0:
                    diff = (kept[:count] - cand).astype(np.float64)
                    if np.any(np.einsum("ij,ij->i", diff, diff) <= bound2):
                        continue
                kept[count] = cand
                accepted.append(int(ordered[k]))
                count += 1
                if limit is not None and count >= limit:
                    break
        else:
            steps = np.arange(-max_step, max_step + 1, dtype=np.int64)
            ax, ay, az = np.meshgrid(steps, steps, steps, indexing="ij")
            offs = np.stack([ax.ravel(), ay.ravel(), az.ravel()], axis=1)
            dist = spacing * np.sqrt(np.sum(offs.astype(np.float64) ** 2, axis=1))
            stencil = offs[dist <= reach]
            sxx = stencil[:, 0]
            syy = stencil[:, 1]
            szz = stencil[:, 2]
            suppressed = np.zeros(tuple(grid.shape), dtype=bool)
            for k in range(ordered.shape[0]):
                ix, iy, iz = (int(cand_multi[k, 0]), int(cand_multi[k, 1]), int(cand_multi[k, 2]))
                if suppressed[ix, iy, iz]:
                    continue
                accepted.append(int(ordered[k]))
                xs = ix + sxx
                ys = iy + syy
                zs = iz + szz
                ok = (xs >= 0) & (xs < nx) & (ys >= 0) & (ys < ny) & (zs >= 0) & (zs < nz)
                suppressed[xs[ok], ys[ok], zs[ok]] = True
                if limit is not None and len(accepted) >= limit:
                    break
    acc = np.asarray(accepted, dtype=np.int64)
    num = acc.shape[0]
    multi = np.column_stack(np.unravel_index(acc, grid.shape)).astype(np.int64)
    values = raw.ravel()[acc]
    voxel_pos = grid.origin + grid.spacing * multi.astype(np.float64)

    if not refine:
        return Peaks(
            positions=np.array(voxel_pos, dtype=np.float64, copy=True),
            values=np.array(values, dtype=np.float64, copy=True),
            indices=np.array(acc, dtype=np.int64, copy=True),
            offsets=np.zeros((num, 3), dtype=np.float64),
        )

    nx, ny, nz = (int(v) for v in grid.shape)
    finite = np.isfinite(raw)
    delta = np.zeros((num, 3), dtype=np.float64)
    full_ok = np.zeros((num,), dtype=bool)
    inside = (
        (multi[:, 0] > 0)
        & (multi[:, 0] < nx - 1)
        & (multi[:, 1] > 0)
        & (multi[:, 1] < ny - 1)
        & (multi[:, 2] > 0)
        & (multi[:, 2] < nz - 1)
    )
    inner = np.flatnonzero(inside)
    if inner.shape[0] > 0:
        sub = multi[inner]
        patch = np.empty((inner.shape[0], 27), dtype=np.float64)
        patch_finite = np.ones((inner.shape[0],), dtype=bool)
        for j in range(27):
            ix = sub[:, 0] + _OX[j]
            iy = sub[:, 1] + _OY[j]
            iz = sub[:, 2] + _OZ[j]
            patch[:, j] = raw[ix, iy, iz]
            patch_finite &= finite[ix, iy, iz]
        ready = inner[patch_finite]
        if ready.shape[0] > 0:
            coeffs = patch[patch_finite] @ _DESIGN_PINV.T
            grad = coeffs[:, 1:4]
            hess = np.empty((ready.shape[0], 3, 3), dtype=np.float64)
            hess[:, 0, 0] = 2.0 * coeffs[:, 4]
            hess[:, 1, 1] = 2.0 * coeffs[:, 5]
            hess[:, 2, 2] = 2.0 * coeffs[:, 6]
            hess[:, 0, 1] = hess[:, 1, 0] = coeffs[:, 7]
            hess[:, 0, 2] = hess[:, 2, 0] = coeffs[:, 8]
            hess[:, 1, 2] = hess[:, 2, 1] = coeffs[:, 9]
            eig = np.linalg.eigvalsh(hess)
            neg_def = eig[:, -1] < -1e-9 * np.max(np.abs(eig), axis=1)
            shift = np.zeros((ready.shape[0], 3), dtype=np.float64)
            if np.any(neg_def):
                shift[neg_def] = np.linalg.solve(hess[neg_def], -grad[neg_def][..., None])[..., 0]
            take = neg_def & (np.max(np.abs(shift), axis=1) <= 1.0)
            delta[ready[take]] = shift[take]
            full_ok[ready[take]] = True

    need = ~full_ok
    if np.any(need):
        f0 = raw[multi[:, 0], multi[:, 1], multi[:, 2]]
        for axis in range(3):
            extent = grid.shape[axis]
            lo = multi[:, axis] - 1
            hi = multi[:, axis] + 1
            ok = need & (lo >= 0) & (hi < extent)
            idx_lo = multi.copy()
            idx_lo[:, axis] = np.clip(lo, 0, extent - 1)
            idx_hi = multi.copy()
            idx_hi[:, axis] = np.clip(hi, 0, extent - 1)
            f_minus = raw[idx_lo[:, 0], idx_lo[:, 1], idx_lo[:, 2]]
            f_plus = raw[idx_hi[:, 0], idx_hi[:, 1], idx_hi[:, 2]]
            ok &= finite[idx_lo[:, 0], idx_lo[:, 1], idx_lo[:, 2]]
            ok &= finite[idx_hi[:, 0], idx_hi[:, 1], idx_hi[:, 2]]
            den = f_minus - 2.0 * f0 + f_plus
            use = ok & (den < 0.0)
            delta[use, axis] = 0.5 * (f_minus[use] - f_plus[use]) / den[use]
    np.clip(delta, -0.5, 0.5, out=delta)
    positions = grid.origin + grid.spacing * (multi.astype(np.float64) + delta)
    return Peaks(
        positions=np.asarray(positions, dtype=np.float64),
        values=np.asarray(values, dtype=np.float64),
        indices=np.asarray(acc, dtype=np.int64),
        offsets=delta,
    )


def match(det_pos: np.ndarray, gt_pos: np.ndarray, gate: float) -> Matching:
    """Match detections to ground truth with maximum cardinality first."""
    det = _as_point_cloud(det_pos, "det_pos")
    gt = _as_point_cloud(gt_pos, "gt_pos")
    try:
        gate_f = float(gate)
    except (TypeError, ValueError) as error:
        raise ValueError("gate must be finite and > 0") from error
    if not np.isfinite(gate_f) or gate_f <= 0.0:
        raise ValueError("gate must be finite and > 0")
    num_det, num_gt = det.shape[0], gt.shape[0]
    if num_det == 0 or num_gt == 0:
        return Matching(
            det_idx=np.zeros((0,), dtype=np.int64),
            gt_idx=np.zeros((0,), dtype=np.int64),
            distance=np.zeros((0,), dtype=np.float64),
            num_det=int(num_det),
            num_gt=int(num_gt),
        )
    dist = np.linalg.norm(det[:, None, :] - gt[None, :, :], axis=-1)
    big = 2.0 * gate_f * (min(num_det, num_gt) + 1)
    rows, cols = linear_sum_assignment(np.where(dist <= gate_f, dist, big))
    keep = dist[rows, cols] <= gate_f
    rows, cols = np.asarray(rows[keep], dtype=np.int64), np.asarray(cols[keep], dtype=np.int64)
    return Matching(
        det_idx=rows,
        gt_idx=cols,
        distance=np.array(dist[rows, cols], dtype=np.float64),
        num_det=int(num_det),
        num_gt=int(num_gt),
    )


def _as_scores(value: np.ndarray, length: int, name: str) -> np.ndarray:
    """Return ``value`` as a finite float64 ``[length]`` vector."""
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a real array") from error
    if arr.shape != (length,):
        raise ValueError(f"{name} must have shape [{length}]")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _as_weights(value: np.ndarray | None, length: int, name: str) -> np.ndarray:
    """Return ``value`` as a finite non-negative float64 ``[length]`` vector."""
    if value is None:
        return np.ones((length,), dtype=np.float64)
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a real array") from error
    if arr.shape != (length,):
        raise ValueError(f"{name} must have shape [{length}]")
    if not np.all(np.isfinite(arr)) or np.any(arr < 0.0):
        raise ValueError(f"{name} must be finite and >= 0")
    return arr


def froc(
    det_pos: np.ndarray,
    det_score: np.ndarray,
    gt_pos: np.ndarray,
    gate: float,
    *,
    gt_weight: np.ndarray | None = None,
    det_weight: np.ndarray | None = None,
    volume_m3: float | None = None,
) -> FrocCurve:
    """Build the FROC curve by re-matching every score threshold subset."""
    det = _as_point_cloud(det_pos, "det_pos")
    gt = _as_point_cloud(gt_pos, "gt_pos")
    scores = _as_scores(det_score, det.shape[0], "det_score")
    gt_w = _as_weights(gt_weight, gt.shape[0], "gt_weight")
    det_w = _as_weights(det_weight, det.shape[0], "det_weight")
    volume: float | None = None
    if volume_m3 is not None:
        try:
            volume = float(volume_m3)
        except (TypeError, ValueError) as error:
            raise ValueError("volume_m3 must be finite and > 0") from error
        if not np.isfinite(volume) or volume <= 0.0:
            raise ValueError("volume_m3 must be finite and > 0")
    num_det, num_gt = det.shape[0], gt.shape[0]
    if num_det == 0:
        nan = np.zeros((0,), dtype=np.float64)
        return FrocCurve(
            thresholds=np.zeros((0,), dtype=np.float64),
            num_det=np.zeros((0,), dtype=np.int64),
            tp=np.zeros((0,), dtype=np.int64),
            fp=np.zeros((0,), dtype=np.int64),
            recall=nan.copy(),
            precision=nan.copy(),
            weighted_recall=nan.copy(),
            weighted_precision=nan.copy(),
            fa_per_1000m3=nan.copy(),
            num_gt=int(num_gt),
        )
    thresholds = np.unique(scores)[::-1].copy()
    count = thresholds.shape[0]
    kept_det = np.zeros((count,), dtype=np.int64)
    tp = np.zeros((count,), dtype=np.int64)
    fp = np.zeros((count,), dtype=np.int64)
    recall = np.zeros((count,), dtype=np.float64)
    precision = np.zeros((count,), dtype=np.float64)
    weighted_recall = np.zeros((count,), dtype=np.float64)
    weighted_precision = np.zeros((count,), dtype=np.float64)
    fa_rate = np.zeros((count,), dtype=np.float64)
    gt_total = float(np.sum(gt_w))
    for t, level in enumerate(thresholds):
        kept = scores >= level
        kept_idx = np.flatnonzero(kept)
        matched = match(det[kept], gt, gate)
        n_kept = int(np.sum(kept))
        n_tp = matched.tp
        kept_det[t] = n_kept
        tp[t] = n_tp
        fp[t] = n_kept - n_tp
        recall[t] = (n_tp / num_gt) if num_gt > 0 else float("nan")
        precision[t] = n_tp / n_kept
        if gt_total == 0.0:
            weighted_recall[t] = float("nan")
        else:
            weighted_recall[t] = float(np.sum(gt_w[matched.gt_idx]) / gt_total)
        det_total = float(np.sum(det_w[kept]))
        if det_total == 0.0:
            weighted_precision[t] = float("nan")
        else:
            global_det = kept_idx[matched.det_idx]
            weighted_precision[t] = float(np.sum(det_w[global_det]) / det_total)
        fa_rate[t] = (fp[t] / volume * 1000.0) if volume is not None else float("nan")
    return FrocCurve(
        thresholds=np.array(thresholds, dtype=np.float64),
        num_det=kept_det,
        tp=tp,
        fp=fp,
        recall=recall,
        precision=precision,
        weighted_recall=weighted_recall,
        weighted_precision=weighted_precision,
        fa_per_1000m3=fa_rate,
        num_gt=int(num_gt),
    )


def average_precision(curve: FrocCurve) -> float:
    """All-point interpolated area under the precision-recall curve."""
    if curve.num_gt == 0:
        return float("nan")
    if curve.thresholds.shape[0] == 0:
        return 0.0
    best = np.maximum.accumulate(curve.precision[::-1])[::-1]
    area = 0.0
    prev = 0.0
    for recall, prec in zip(curve.recall, best, strict=True):
        area += (float(recall) - prev) * float(prec)
        prev = float(recall)
    return float(area)


def ap_at(
    det_pos: np.ndarray, det_score: np.ndarray, gt_pos: np.ndarray, gate: float = 1.0
) -> float:
    """Average precision of the FROC curve at the given gate (default 1 m)."""
    return average_precision(froc(det_pos, det_score, gt_pos, gate))


def recall_at_fa(curve: FrocCurve, fa_per_1000m3: float = 1.0) -> float:
    """Maximum recall over thresholds with false alarms at most ``fa``."""
    try:
        rate = float(fa_per_1000m3)
    except (TypeError, ValueError) as error:
        raise ValueError("fa_per_1000m3 must be finite and >= 0") from error
    if not np.isfinite(rate) or rate < 0.0:
        raise ValueError("fa_per_1000m3 must be finite and >= 0")
    if curve.thresholds.shape[0] == 0:
        return 0.0
    if np.any(np.isnan(curve.fa_per_1000m3)):
        raise ValueError("curve has no scored volume")
    ok = curve.fa_per_1000m3 <= rate
    if not np.any(ok):
        return 0.0
    return float(np.max(curve.recall[ok]))


def detectable_mask(
    power: np.ndarray,
    dynamic_range_db: float = 30.0,
    *,
    reference: np.ndarray | None = None,
) -> np.ndarray:
    """Flag GT points within the dynamic range of the strongest path."""
    if np.iscomplexobj(power):
        raise ValueError("power must be real")
    try:
        table = np.asarray(power, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("power must be a real array") from error
    if table.ndim != 2:
        raise ValueError("power must have shape [M, C]")
    if not np.all(np.isfinite(table)) or np.any(table < 0.0):
        raise ValueError("power must be finite and >= 0")
    try:
        width = float(dynamic_range_db)
    except (TypeError, ValueError) as error:
        raise ValueError("dynamic_range_db must be finite") from error
    if not np.isfinite(width):
        raise ValueError("dynamic_range_db must be finite")
    num_pts, num_cap = table.shape
    if num_pts == 0:
        return np.zeros((0,), dtype=bool)
    if reference is None:
        ref = np.max(table, axis=0)
    else:
        if np.iscomplexobj(reference):
            raise ValueError("reference must be real")
        try:
            ref = np.asarray(reference, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("reference must be a real array") from error
        if ref.shape != (num_cap,):
            raise ValueError("reference must have shape [C]")
        if not np.all(np.isfinite(ref)) or np.any(ref < 0.0):
            raise ValueError("reference must be finite and >= 0")
    level = ref * 10.0 ** (-width / 10.0)
    present = (table > 0.0) & (ref[None, :] > 0.0) & (table >= level[None, :])
    return np.array(np.any(present, axis=1), dtype=bool)


def loc_error_decomposed(est: np.ndarray, gt: np.ndarray, ref: np.ndarray) -> LocationError:
    """Split matched errors into range, horizontal and vertical components."""
    est_a = _as_point_cloud(est, "est")
    gt_a = _as_point_cloud(gt, "gt")
    if est_a.shape != gt_a.shape:
        raise ValueError("est and gt must have the same shape")
    if not np.all(np.isfinite(est_a)) or not np.all(np.isfinite(gt_a)):
        raise ValueError("est and gt must contain only finite values")
    count = est_a.shape[0]
    if np.iscomplexobj(ref):
        raise ValueError("ref must be real")
    try:
        ref_a = np.asarray(ref, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("ref must be a real array") from error
    if ref_a.shape == (3,):
        ref_b = np.broadcast_to(ref_a, (count, 3))
    elif ref_a.shape == (count, 3):
        ref_b = ref_a
    else:
        raise ValueError("ref must have shape [3] or [L, 3]")
    if not np.all(np.isfinite(ref_b)):
        raise ValueError("ref must contain only finite values")
    span = gt_a - ref_b
    length = np.linalg.norm(span, axis=1)
    if np.any(length == 0.0):
        raise ValueError("gt must differ from ref")
    radial = span / length[:, None]
    z_hat = np.array([0.0, 0.0, 1.0])
    cross = np.cross(np.broadcast_to(z_hat, (count, 3)), radial)
    cross_len = np.linalg.norm(cross, axis=1)
    fallback = cross_len < 1e-12
    horizontal = np.empty((count, 3), dtype=np.float64)
    horizontal[~fallback] = cross[~fallback] / cross_len[~fallback, None]
    horizontal[fallback] = np.array([1.0, 0.0, 0.0])
    vertical = np.cross(radial, horizontal)
    err = est_a - gt_a
    return LocationError(
        range=np.einsum("ij,ij->i", err, radial),
        horizontal=np.einsum("ij,ij->i", err, horizontal),
        vertical=np.einsum("ij,ij->i", err, vertical),
        total=np.linalg.norm(err, axis=1),
    )


def nmse_global_phase(est: np.ndarray, ref: np.ndarray) -> float:
    """Complex NMSE after removing the best global phase."""
    try:
        est_a = np.asarray(est)
        ref_a = np.asarray(ref)
    except (TypeError, ValueError) as error:
        raise ValueError("est and ref must be arrays") from error
    if est_a.shape != ref_a.shape:
        raise ValueError("est and ref must have the same shape")
    if not np.all(np.isfinite(est_a)) or not np.all(np.isfinite(ref_a)):
        raise ValueError("est and ref must contain only finite values")
    denom = float(np.vdot(ref_a, ref_a).real)
    if denom == 0.0:
        raise ValueError("ref must have nonzero norm")
    overlap = np.vdot(est_a, ref_a)
    phasor = 1.0 if overlap == 0 else overlap / abs(overlap)
    resid = float(np.sum(np.abs(phasor * est_a - ref_a) ** 2).real)
    return resid / denom


def nmse_power_scale(est: np.ndarray, ref: np.ndarray) -> float:
    """Real power NMSE after removing the best non-negative global scale."""
    if np.iscomplexobj(est) or np.iscomplexobj(ref):
        raise ValueError("est and ref must be real")
    try:
        est_a = np.asarray(est, dtype=np.float64)
        ref_a = np.asarray(ref, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("est and ref must be real arrays") from error
    if est_a.shape != ref_a.shape:
        raise ValueError("est and ref must have the same shape")
    if not np.all(np.isfinite(est_a)) or not np.all(np.isfinite(ref_a)):
        raise ValueError("est and ref must contain only finite values")
    denom = float(np.sum(ref_a**2))
    if denom == 0.0:
        raise ValueError("ref must have nonzero norm")
    scale_denom = float(np.sum(est_a**2))
    if scale_denom == 0.0:
        scale = 0.0
    else:
        scale = max(0.0, float(np.sum(est_a * ref_a)) / scale_denom)
    return float(np.sum((scale * est_a - ref_a) ** 2) / denom)


def gauge_errors(
    est: tuple[np.ndarray, np.ndarray],
    gt: tuple[np.ndarray, np.ndarray],
    *,
    period: float | None = None,
) -> GaugeErrors:
    """Wrapped phase and absolute delay errors after global-phase removal."""
    try:
        (phi_est_raw, tau_est_raw), (phi_gt_raw, tau_gt_raw) = est, gt
    except (TypeError, ValueError) as error:
        raise ValueError("est and gt must each hold (phase, delay)") from error
    parts = []
    for name, raw in (
        ("phi_est", phi_est_raw),
        ("tau_est", tau_est_raw),
        ("phi_gt", phi_gt_raw),
        ("tau_gt", tau_gt_raw),
    ):
        if np.iscomplexobj(raw):
            raise ValueError(f"{name} must be real")
        try:
            parts.append(np.asarray(raw, dtype=np.float64))
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be a real array") from error
    phi_est, tau_est, phi_gt, tau_gt = parts
    if not (phi_est.shape == tau_est.shape == phi_gt.shape == tau_gt.shape):
        raise ValueError("all gauge arrays must have the same shape")
    if not all(np.all(np.isfinite(part)) for part in parts):
        raise ValueError("gauge arrays must contain only finite values")
    period_f: float | None = None
    if period is not None:
        try:
            period_f = float(period)
        except (TypeError, ValueError) as error:
            raise ValueError("period must be finite and > 0") from error
        if not np.isfinite(period_f) or period_f <= 0.0:
            raise ValueError("period must be finite and > 0")
    diff = phi_est - phi_gt
    total = np.sum(np.exp(1j * diff))
    global_phase = 0.0 if total == 0 else float(np.angle(total))
    phase = np.angle(np.exp(1j * (diff - global_phase)))
    if period_f is None:
        delay = np.abs(tau_est - tau_gt)
    else:
        delay = np.vectorize(circular_delay_error_s, otypes=[float])(tau_est, tau_gt, period_f)
    return GaugeErrors(
        phase=np.asarray(phase, dtype=np.float64),
        delay=np.asarray(delay, dtype=np.float64),
        global_phase=float(global_phase),
    )


SURFACE_THRESHOLDS_M: tuple[float, ...] = (0.5, 1.0, 2.0)
MAP_REL_THRESHOLD: float = 0.1
PLANE_MAX_ANGLE_DEG: float = 10.0
PLANE_MAX_OFFSET_M: float = 1.0


@dataclass(frozen=True)
class SurfaceScore:
    """Precision/recall/F-score of a point cloud against a reference at one gate."""

    threshold: float
    precision: float
    recall: float
    f_score: float
    num_pred: int
    num_ref: int


@dataclass(frozen=True)
class ChamferScore:
    """Weighted mean nearest-neighbour distances between two point clouds."""

    accuracy: float
    completeness: float
    chamfer: float


@dataclass(frozen=True)
class PlaneMatching:
    """One-to-one estimated-to-GT plane assignment inside the angle/offset gates."""

    est_idx: np.ndarray
    gt_idx: np.ndarray
    angle: np.ndarray
    offset: np.ndarray
    num_est: int
    num_gt: int

    @property
    def tp(self) -> int:
        """Number of matched pairs."""
        return int(self.est_idx.shape[0])

    @property
    def fp(self) -> int:
        """Number of unmatched estimated planes."""
        return int(self.num_est) - int(self.est_idx.shape[0])

    @property
    def fn(self) -> int:
        """Number of unmatched ground-truth planes."""
        return int(self.num_gt) - int(self.est_idx.shape[0])

    def summary(self) -> dict[str, float]:
        """Median and P90 of the matched angles (deg) and offsets (m)."""
        keys = ("angle_deg_median", "angle_deg_p90", "offset_m_median", "offset_m_p90")
        if self.est_idx.shape[0] == 0:
            return {key: float("nan") for key in keys}
        angle_deg = np.degrees(np.asarray(self.angle, dtype=np.float64))
        offset_m = np.asarray(self.offset, dtype=np.float64)
        return {
            "angle_deg_median": float(np.median(angle_deg)),
            "angle_deg_p90": float(np.percentile(angle_deg, 90)),
            "offset_m_median": float(np.median(offset_m)),
            "offset_m_p90": float(np.percentile(offset_m, 90)),
        }


def nearest_distance(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Distance from each row of ``src`` to its nearest row of ``dst``."""
    src_a = _as_point_cloud(src, "src")
    dst_a = _as_point_cloud(dst, "dst")
    if src_a.shape[0] == 0:
        return np.zeros((0,), dtype=np.float64)
    if not np.all(np.isfinite(src_a)):
        raise ValueError("src must contain only finite values")
    if not np.all(np.isfinite(dst_a)):
        raise ValueError("dst must contain only finite values")
    if dst_a.shape[0] == 0:
        return np.full((src_a.shape[0],), np.inf, dtype=np.float64)
    dist, _ = cKDTree(dst_a).query(src_a, k=1)
    return np.asarray(dist, dtype=np.float64)


def _as_threshold(value: float, name: str) -> float:
    """Return ``value`` as a finite positive float."""
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite and > 0") from error
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and > 0")
    return result


def _as_mask(value: np.ndarray | None, length: int, name: str) -> np.ndarray:
    """Return ``value`` as a bool ``[length]`` mask (None becomes all True)."""
    if value is None:
        return np.ones((length,), dtype=bool)
    try:
        arr = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must have shape [{length}] and bool dtype") from error
    if arr.shape != (length,) or arr.dtype != np.dtype(bool):
        raise ValueError(f"{name} must have shape [{length}] and bool dtype")
    return np.array(arr, dtype=bool, copy=True)


def _as_finite_cloud(value: np.ndarray, name: str) -> np.ndarray:
    """Return ``value`` as a finite float64 ``[N, 3]`` point cloud."""
    arr = _as_point_cloud(value, name)
    if arr.shape[0] > 0 and not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _as_radius(value: float, name: str) -> float:
    """Return ``value`` as a finite float >= 0."""
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite and >= 0") from error
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and >= 0")
    return result


def _prf_from(
    dist_pred: np.ndarray,
    pred_w: np.ndarray,
    dist_ref: np.ndarray,
    ref_w: np.ndarray,
    gate: float,
) -> SurfaceScore:
    """Precision/recall/F-score from precomputed nearest distances."""
    pred_total = float(np.sum(pred_w))
    precision = (
        float(np.sum(pred_w[dist_pred <= gate]) / pred_total) if pred_total > 0.0 else float("nan")
    )
    ref_total = float(np.sum(ref_w))
    recall = float(np.sum(ref_w[dist_ref <= gate]) / ref_total) if ref_total > 0.0 else float("nan")
    if not np.isfinite(recall):
        f_score = float("nan")
    elif not np.isfinite(precision):
        f_score = 0.0
    elif precision + recall > 0.0:
        f_score = float(2.0 * precision * recall / (precision + recall))
    else:
        f_score = 0.0
    return SurfaceScore(
        threshold=float(gate),
        precision=precision,
        recall=recall,
        f_score=f_score,
        num_pred=int(dist_pred.shape[0]),
        num_ref=int(dist_ref.shape[0]),
    )


def _chamfer_from(
    dist_pred: np.ndarray, pred_w: np.ndarray, dist_ref: np.ndarray, ref_w: np.ndarray
) -> ChamferScore:
    """Weighted Chamfer terms from precomputed nearest distances."""
    pred_total = float(np.sum(pred_w))
    ref_total = float(np.sum(ref_w))
    if pred_total == 0.0 or ref_total == 0.0:
        nan = float("nan")
        return ChamferScore(accuracy=nan, completeness=nan, chamfer=nan)
    accuracy = float(np.sum(pred_w * dist_pred) / pred_total)
    completeness = float(np.sum(ref_w * dist_ref) / ref_total)
    return ChamferScore(
        accuracy=accuracy, completeness=completeness, chamfer=0.5 * (accuracy + completeness)
    )


def _energy_from(dist: np.ndarray, weight: np.ndarray, radius: float) -> float:
    """Fraction of ``weight`` at distance <= ``radius`` (NaN for zero total weight)."""
    total = float(np.sum(weight))
    if total == 0.0:
        return float("nan")
    return float(np.sum(weight[dist <= radius]) / total)


def _surface_inputs(
    pred: np.ndarray,
    ref: np.ndarray,
    pred_weight: np.ndarray | None,
    ref_weight: np.ndarray | None,
    recall_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate and return ``(dist_pred, pred_w, dist_ref, scoped_ref_w)``."""
    pred_a = _as_finite_cloud(pred, "pred")
    ref_a = _as_finite_cloud(ref, "ref")
    pred_w = _as_weights(pred_weight, pred_a.shape[0], "pred_weight")
    ref_w = _as_weights(ref_weight, ref_a.shape[0], "ref_weight")
    mask = _as_mask(recall_mask, ref_a.shape[0], "recall_mask")
    dist_pred = nearest_distance(pred_a, ref_a)
    dist_ref = nearest_distance(ref_a[mask], pred_a)
    return dist_pred, pred_w, dist_ref, ref_w[mask]


def surface_prf(
    pred: np.ndarray,
    ref: np.ndarray,
    threshold: float,
    *,
    pred_weight: np.ndarray | None = None,
    ref_weight: np.ndarray | None = None,
    recall_mask: np.ndarray | None = None,
) -> SurfaceScore:
    """Precision/recall/F-score of ``pred`` against ``ref`` inside ``threshold``."""
    gate = _as_threshold(threshold, "threshold")
    dist_pred, pred_w, dist_ref, ref_w = _surface_inputs(
        pred, ref, pred_weight, ref_weight, recall_mask
    )
    return _prf_from(dist_pred, pred_w, dist_ref, ref_w, gate)


def weighted_chamfer(
    pred: np.ndarray,
    ref: np.ndarray,
    *,
    pred_weight: np.ndarray | None = None,
    ref_weight: np.ndarray | None = None,
    recall_mask: np.ndarray | None = None,
) -> ChamferScore:
    """Weighted mean nearest-neighbour distances between ``pred`` and ``ref``."""
    dist_pred, pred_w, dist_ref, ref_w = _surface_inputs(
        pred, ref, pred_weight, ref_weight, recall_mask
    )
    return _chamfer_from(dist_pred, pred_w, dist_ref, ref_w)


def energy_within(pos: np.ndarray, weight: np.ndarray, ref: np.ndarray, radius: float) -> float:
    """Fraction of ``weight`` within ``radius`` of the reference cloud ``ref``."""
    pos_a = _as_finite_cloud(pos, "pos")
    ref_a = _as_finite_cloud(ref, "ref")
    if weight is None:
        raise ValueError("weight must be finite and >= 0 with shape [N]")
    point_w = _as_weights(weight, pos_a.shape[0], "weight")
    return _energy_from(nearest_distance(pos_a, ref_a), point_w, _as_radius(radius, "radius"))


@dataclass(frozen=True)
class SurfaceReport:
    """P/R/F at several gates, weighted Chamfer and energy fractions of one prediction."""

    scores: tuple[SurfaceScore, ...]
    chamfer: ChamferScore
    energy: tuple[float, ...]


def surface_report(
    pred: np.ndarray,
    ref: np.ndarray,
    thresholds: Sequence[float],
    *,
    pred_weight: np.ndarray | None = None,
    recall_mask: np.ndarray | None = None,
    energy_pos: np.ndarray | None = None,
    energy_weight: np.ndarray | None = None,
) -> SurfaceReport:
    """Score ``pred`` against ``ref`` at every gate with one nearest-distance pass.

    Equals ``surface_prf`` (unweighted) per gate, ``weighted_chamfer`` with
    ``pred_weight`` and ``energy_within(energy_pos, energy_weight, ref, gate)``
    per gate (``energy`` is empty when ``energy_pos`` is None).
    """
    gates = [_as_threshold(gate, "thresholds") for gate in thresholds]
    dist_pred, pred_w, dist_ref, ref_w = _surface_inputs(pred, ref, pred_weight, None, recall_mask)
    ones = np.ones_like(pred_w)
    scores = tuple(_prf_from(dist_pred, ones, dist_ref, ref_w, gate) for gate in gates)
    energy: tuple[float, ...] = ()
    if energy_pos is not None:
        pos_a = _as_finite_cloud(energy_pos, "energy_pos")
        if energy_weight is None:
            raise ValueError("energy_weight is required with energy_pos")
        energy_w = _as_weights(energy_weight, pos_a.shape[0], "energy_weight")
        dist_energy = nearest_distance(pos_a, _as_finite_cloud(ref, "ref"))
        energy = tuple(_energy_from(dist_energy, energy_w, gate) for gate in gates)
    return SurfaceReport(
        scores=scores,
        chamfer=_chamfer_from(dist_pred, pred_w, dist_ref, ref_w),
        energy=energy,
    )


def map_point_cloud(
    density: np.ndarray, grid: VoxelGrid, rel_threshold: float = MAP_REL_THRESHOLD
) -> tuple[np.ndarray, np.ndarray]:
    """Extract a weighted point cloud from ``density`` above a relative level."""
    if not isinstance(grid, VoxelGrid):
        raise ValueError("grid must be a VoxelGrid")
    if np.iscomplexobj(density):
        raise ValueError("density must be real")
    try:
        raw = np.asarray(density, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("density must be a real array") from error
    if raw.shape != tuple(grid.shape):
        raise ValueError("density shape must match grid.shape")
    try:
        level_rel = float(rel_threshold)
    except (TypeError, ValueError) as error:
        raise ValueError("rel_threshold must be finite and in [0, 1]") from error
    if not np.isfinite(level_rel) or level_rel < 0.0 or level_rel > 1.0:
        raise ValueError("rel_threshold must be finite and in [0, 1]")
    finite = np.isfinite(raw)
    if not bool(np.any(finite)):
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    low = float(np.min(raw[finite]))
    high = float(np.max(raw[finite]))
    if not high > low:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    level = low + level_rel * (high - low)
    flats = np.flatnonzero(finite & (raw >= level))
    positions = np.asarray(grid.centers()[flats], dtype=np.float64)
    weights = np.asarray(raw.ravel(order="C")[flats] - low, dtype=np.float64)
    return positions, weights


def _as_planes(
    normal: np.ndarray, offset: np.ndarray, normal_name: str, offset_name: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(normal, offset)`` as float64 ``[K, 3]`` / ``[K]`` plane pairs."""
    if np.iscomplexobj(normal) or np.iscomplexobj(offset):
        raise ValueError(f"{normal_name} and {offset_name} must be real")
    try:
        vec = np.asarray(normal, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{normal_name} must be a real array") from error
    try:
        dist = np.asarray(offset, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{offset_name} must be a real array") from error
    if vec.size == 0 and dist.size == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    if vec.ndim == 1:
        if vec.shape != (3,):
            raise ValueError(f"{normal_name} must have shape [K, 3]")
        vec = vec[None, :]
    if vec.ndim != 2 or vec.shape[1] != 3:
        raise ValueError(f"{normal_name} must have shape [K, 3]")
    count = vec.shape[0]
    if dist.ndim == 0:
        dist = np.full((count,), float(dist), dtype=np.float64)
    if dist.shape != (count,):
        raise ValueError(f"{offset_name} must have shape [{count}]")
    if not np.all(np.isfinite(vec)) or not np.all(np.isfinite(dist)):
        raise ValueError(f"{normal_name} and {offset_name} must contain only finite values")
    norms = np.linalg.norm(vec, axis=1)
    if np.any(norms == 0.0):
        raise ValueError(f"{normal_name} must have no zero row")
    return np.asarray(vec, dtype=np.float64), np.asarray(dist, dtype=np.float64)


def _as_anchor(value: np.ndarray | None, count: int, name: str) -> np.ndarray:
    """Return ``value`` as a finite float64 ``[count, 3]`` anchor set (origin default)."""
    if value is None:
        return np.zeros((count, 3), dtype=np.float64)
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a real array") from error
    if arr.size == 0 and count == 0:
        return np.zeros((0, 3), dtype=np.float64)
    if arr.shape != (count, 3):
        raise ValueError(f"{name} must have shape [{count}, 3]")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def plane_errors(
    est_normal: np.ndarray,
    est_offset: np.ndarray,
    gt_normal: np.ndarray,
    gt_offset: np.ndarray,
    *,
    gt_anchor: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Angular (rad) and anchored offset (m) errors of estimated planes vs GT."""
    est_n, est_d = _as_planes(est_normal, est_offset, "est_normal", "est_offset")
    gt_n, gt_d = _as_planes(gt_normal, gt_offset, "gt_normal", "gt_offset")
    if est_n.shape[0] != gt_n.shape[0]:
        raise ValueError("est and gt plane sets must have the same length")
    count = est_n.shape[0]
    anchors = _as_anchor(gt_anchor, count, "gt_anchor")
    est_hat = est_n / np.linalg.norm(est_n, axis=1)[:, None]
    est_off = est_d / np.linalg.norm(est_n, axis=1)
    gt_hat = gt_n / np.linalg.norm(gt_n, axis=1)[:, None]
    gt_off = gt_d / np.linalg.norm(gt_n, axis=1)
    cos_angle = np.clip(np.abs(np.einsum("ij,ij->i", est_hat, gt_hat)), 0.0, 1.0)
    angle = np.arccos(cos_angle)
    anchored = anchors - ((np.einsum("ij,ij->i", gt_hat, anchors) - gt_off)[:, None] * gt_hat)
    offset = np.abs(np.einsum("ij,ij->i", est_hat, anchored) - est_off)
    return np.asarray(angle, dtype=np.float64), np.asarray(offset, dtype=np.float64)


def match_planes(
    est_normal: np.ndarray,
    est_offset: np.ndarray,
    gt_normal: np.ndarray,
    gt_offset: np.ndarray,
    *,
    max_angle_deg: float = PLANE_MAX_ANGLE_DEG,
    max_offset_m: float = PLANE_MAX_OFFSET_M,
    gt_anchor: np.ndarray | None = None,
) -> PlaneMatching:
    """Match estimated planes to GT planes with maximum cardinality first."""
    est_n, est_d = _as_planes(est_normal, est_offset, "est_normal", "est_offset")
    gt_n, gt_d = _as_planes(gt_normal, gt_offset, "gt_normal", "gt_offset")
    anchors = _as_anchor(gt_anchor, gt_n.shape[0], "gt_anchor")
    try:
        max_angle = math.radians(float(max_angle_deg))
    except (TypeError, ValueError) as error:
        raise ValueError("max_angle_deg must be finite and > 0") from error
    if not np.isfinite(max_angle) or max_angle <= 0.0:
        raise ValueError("max_angle_deg must be finite and > 0")
    try:
        max_offset = float(max_offset_m)
    except (TypeError, ValueError) as error:
        raise ValueError("max_offset_m must be finite and > 0") from error
    if not np.isfinite(max_offset) or max_offset <= 0.0:
        raise ValueError("max_offset_m must be finite and > 0")
    num_est, num_gt = est_n.shape[0], gt_n.shape[0]
    empty = (
        np.zeros((0,), dtype=np.int64),
        np.zeros((0,), dtype=np.int64),
        np.zeros((0,), dtype=np.float64),
        np.zeros((0,), dtype=np.float64),
    )
    if num_est == 0 or num_gt == 0:
        est_idx, gt_idx, angle, offset = empty
        return PlaneMatching(
            est_idx=est_idx,
            gt_idx=gt_idx,
            angle=angle,
            offset=offset,
            num_est=int(num_est),
            num_gt=int(num_gt),
        )
    est_rows = np.repeat(np.arange(num_est), num_gt)
    gt_rows = np.tile(np.arange(num_gt), num_est)
    angle_flat, offset_flat = plane_errors(
        est_n[est_rows], est_d[est_rows], gt_n[gt_rows], gt_d[gt_rows], gt_anchor=anchors[gt_rows]
    )
    angle_all = angle_flat.reshape(num_est, num_gt)
    offset_all = offset_flat.reshape(num_est, num_gt)
    admissible = (angle_all <= max_angle) & (offset_all <= max_offset)
    big = 4.0 * (min(num_est, num_gt) + 1)
    cost = np.where(admissible, angle_all / max_angle + offset_all / max_offset, big).astype(
        np.float64
    )
    rows, cols = linear_sum_assignment(cost)
    keep = admissible[rows, cols]
    rows = np.asarray(rows[keep], dtype=np.int64)
    cols = np.asarray(cols[keep], dtype=np.int64)
    order = np.argsort(rows, kind="stable")
    rows, cols = rows[order], cols[order]
    return PlaneMatching(
        est_idx=rows,
        gt_idx=cols,
        angle=np.array(angle_all[rows, cols], dtype=np.float64),
        offset=np.array(offset_all[rows, cols], dtype=np.float64),
        num_est=int(num_est),
        num_gt=int(num_gt),
    )


def stratified_recall(
    matching: Matching,
    labels: np.ndarray | list[str] | tuple[str, ...],
    *,
    gt_weight: np.ndarray | None = None,
) -> dict[str, dict[str, float | int]]:
    """Per-label recall of a ``Matching`` over the GT ``labels``."""
    if not isinstance(matching, Matching):
        raise ValueError("matching must be a Matching")
    num_gt = int(matching.num_gt)
    try:
        items = list(labels)
    except TypeError as error:
        raise ValueError("labels must have one entry per GT point") from error
    if len(items) != num_gt:
        raise ValueError(f"labels must have length {num_gt}, got {len(items)}")
    for item in items:
        if not isinstance(item, str):
            raise ValueError("labels must all be str")
    weights = _as_weights(gt_weight, num_gt, "gt_weight")
    matched = np.zeros((num_gt,), dtype=bool)
    matched[np.asarray(matching.gt_idx, dtype=np.int64)] = True
    out: dict[str, dict[str, float | int]] = {}
    for label in sorted(set(items)):
        selected = np.array([item == label for item in items], dtype=bool)
        count = int(np.count_nonzero(selected))
        hits = int(np.count_nonzero(selected & matched))
        total = float(np.sum(weights[selected]))
        out[label] = {
            "num_gt": count,
            "tp": hits,
            "recall": float(hits / count),
            "weighted_recall": (
                float(np.sum(weights[selected & matched]) / total) if total > 0.0 else float("nan")
            ),
        }
    return out
