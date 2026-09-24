"""Core metrics for tomography reconstructions (M1, M5 and M6).

Non-maximum-suppression peak finding with sub-voxel refinement, Hungarian
detection matching, FROC / average-precision scoring, decomposed localisation
errors, global-phase / global-scale NMSE and wrapped gauge errors.

NumPy/SciPy only: nothing in this module may import Sionna, Mitsuba or Dr.Jit.
Arrays are float64 / complex128 / int64 with SI units (metres, seconds, radians).
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter
from scipy.optimize import linear_sum_assignment

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
