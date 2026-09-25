"""Identifiability test of docs/tomography_baselines.md T15b (section 6.5 item 7).

The Fisher information of a model with independent Gaussian real observations is
``F = jac.T @ diag(weights) @ jac`` (weights are inverse variances until T30 provides the exact
likelihoods; complex CN(0, sigma^2) data is stacked real-then-imaginary with weight 2/sigma^2).
The per-capture gauges are eliminated as nuisance parameters (Schur complement), which also
removes the global-phase null, and ``ill_posed`` flags a configuration whose position-marginal
FIM has a condition number above 1e8 or a position CRB std above 10 m. NumPy only.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from plateau_rt.domain.rf_tomography.forward_exact import capture_factors
from plateau_rt.domain.rf_tomography.geometry import CaptureGeometry

COND_MAX: float = 1e8
STD_MAX_M: float = 10.0
NUISANCE_RTOL: float = 1e-8
MARGINAL_RTOL: float = 1e-12


def _validate_indices(
    idx: Sequence[int] | np.ndarray, n_par: int, *, allow_empty: bool, name: str
) -> np.ndarray:
    """Return ``idx`` as an int64 ``[K]`` array of unique entries in ``[0, n_par)``."""
    try:
        items = list(idx)  # type: ignore[arg-type]
    except TypeError as error:
        raise ValueError(f"{name} must be a sequence of integers") from error
    if len(items) == 0:
        if allow_empty:
            return np.empty(0, dtype=np.int64)
        raise ValueError(f"{name} must be non-empty")
    values: list[int] = []
    for item in items:
        if isinstance(item, bool):
            raise ValueError(f"{name} must contain integers in [0, {n_par})")
        if isinstance(item, (int, np.integer)):
            values.append(int(item))
        elif isinstance(item, (float, np.floating)):
            if not np.isfinite(item) or float(item) != float(int(item)):
                raise ValueError(f"{name} must contain integers in [0, {n_par})")
            values.append(int(item))
        else:
            raise ValueError(f"{name} must contain integers in [0, {n_par})")
    arr = np.asarray(values, dtype=np.int64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must contain unique entries")
    if bool(np.any(arr < 0)) or bool(np.any(arr >= n_par)):
        raise ValueError(f"{name} entries must lie in [0, {n_par})")
    return arr


def _scale_invariant_pinv(mat: np.ndarray) -> np.ndarray:
    """Return the pseudo-inverse of a PSD ``mat`` after diagonal equilibration.

    Eigenvalues of the equilibrated matrix at or below ``MARGINAL_RTOL`` times the largest are
    treated as null, so the result does not depend on the parameter units.
    """
    m = np.asarray(mat, dtype=np.float64)
    n = m.shape[0]
    diag = np.diag(m)
    scale = np.sqrt(np.where(diag > 0.0, diag, 0.0))
    nz = scale > 0.0
    out = np.zeros((n, n), dtype=np.float64)
    if not bool(np.any(nz)):
        return out
    sub_idx = np.flatnonzero(nz)
    sub = m[np.ix_(sub_idx, sub_idx)] / np.outer(scale[nz], scale[nz])
    sub = 0.5 * (sub + sub.T)
    lam, vec = np.linalg.eigh(sub)
    lam_max = float(lam.max())
    if not np.isfinite(lam_max) or lam_max <= 0.0:
        return out
    keep = (lam > MARGINAL_RTOL * lam_max) & (lam > 0.0)
    if not bool(np.any(keep)):
        return out
    vec_r = vec[:, keep]
    inv = vec_r / lam[keep][None, :]
    sub_pinv = inv @ vec_r.T
    out[np.ix_(sub_idx, sub_idx)] = sub_pinv
    safe = np.where(scale == 0.0, 1.0, scale)
    return out / np.outer(safe, safe)


def numeric_jacobian(
    model_fn: Callable[[np.ndarray], np.ndarray],
    theta: np.ndarray,
    eps: float | np.ndarray = 1e-6,
) -> np.ndarray:
    """Return the central-difference Jacobian of ``model_fn`` at ``theta``.

    ``eps`` is an absolute step (scalar or ``[n_par]``) in each parameter's own units. The model
    output is flattened in C order; a complex output gives ``[2 * n_out, n_par]`` with the real
    parts of all outputs first and then the imaginary parts, a real one ``[n_out, n_par]``.
    """
    theta_arr = np.asarray(theta, dtype=np.float64)
    if theta_arr.ndim != 1 or theta_arr.shape[0] < 1:
        raise ValueError("theta must be a 1-D real array with n_par >= 1")
    if not bool(np.all(np.isfinite(theta_arr))):
        raise ValueError("theta must contain only finite values")
    n_par = int(theta_arr.shape[0])
    if np.isscalar(eps):
        step = np.full(n_par, float(eps), dtype=np.float64)
    else:
        step = np.asarray(eps, dtype=np.float64)
        if step.ndim == 0:
            step = np.full(n_par, float(step), dtype=np.float64)
        elif step.ndim != 1 or step.shape[0] != n_par:
            raise ValueError("eps must be a scalar or have shape [n_par]")
    if not bool(np.all(np.isfinite(step))) or not bool(np.all(step > 0.0)):
        raise ValueError("eps entries must be finite and > 0")
    base_raw = np.asarray(model_fn(theta_arr.copy()))
    base_is_complex = bool(np.iscomplexobj(base_raw))
    base = base_raw.ravel(order="C")
    if not bool(np.all(np.isfinite(base))):
        raise ValueError("model output must be finite")
    n_out = int(base.size)
    if base_is_complex:
        jac = np.zeros((2 * n_out, n_par), dtype=np.float64)
    else:
        jac = np.zeros((n_out, n_par), dtype=np.float64)
    for i in range(n_par):
        h = float(step[i])
        plus = theta_arr.copy()
        plus[i] += h
        minus = theta_arr.copy()
        minus[i] -= h
        fwd_raw = np.asarray(model_fn(plus))
        bwd_raw = np.asarray(model_fn(minus))
        fwd = fwd_raw.ravel(order="C")
        bwd = bwd_raw.ravel(order="C")
        if fwd.size != n_out or bwd.size != n_out:
            raise ValueError("model output size must not depend on theta")
        if not base_is_complex and (
            bool(np.iscomplexobj(fwd_raw)) or bool(np.iscomplexobj(bwd_raw))
        ):
            raise ValueError("model output must stay real")
        if not bool(np.all(np.isfinite(fwd))) or not bool(np.all(np.isfinite(bwd))):
            raise ValueError("model output must be finite")
        if base_is_complex:
            col = (fwd - bwd) / (2.0 * h)
            jac[:, i] = np.concatenate(
                [np.asarray(col.real, dtype=np.float64), np.asarray(col.imag, dtype=np.float64)]
            )
        else:
            jac[:, i] = np.asarray((fwd - bwd) / (2.0 * h), dtype=np.float64)
    return jac


def gauge_reduced_fim(
    jac: np.ndarray,
    weights: np.ndarray,
    nuisance_idx: Sequence[int] | np.ndarray,
    *,
    rtol: float = NUISANCE_RTOL,
) -> np.ndarray:
    """Return the gauge-reduced FIM of the parameters not in ``nuisance_idx``.

    The kept parameters stay in ascending original order. The weighted kept columns are projected
    onto the orthogonal complement of the range of the (unit-normalised) nuisance columns, with
    nuisance singular values at or below ``rtol`` times the largest treated as null. This is the
    Schur complement with a pseudo-inverse of F_nn: invariant to the nuisance units, and free of
    the global-phase null when every phase gauge and complex amplitude is nuisance.
    """
    j = np.asarray(jac, dtype=np.float64)
    if j.ndim != 2:
        raise ValueError("jac must be a real 2-D array [n_obs, n_par]")
    if not bool(np.all(np.isfinite(j))):
        raise ValueError("jac must contain only finite values")
    n_obs, n_par = int(j.shape[0]), int(j.shape[1])
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 1 or w.shape[0] != n_obs:
        raise ValueError("weights must have shape [n_obs]")
    if not bool(np.all(np.isfinite(w))) or not bool(np.all(w >= 0.0)):
        raise ValueError("weights must be finite and >= 0")
    rtol_f = float(rtol)
    if not np.isfinite(rtol_f) or rtol_f < 0.0:
        raise ValueError("rtol must be finite and >= 0")
    nuisance = _validate_indices(nuisance_idx, n_par, allow_empty=True, name="nuisance_idx")
    if nuisance.shape[0] >= n_par:
        raise ValueError("at least one parameter must be kept")
    keep = np.setdiff1d(np.arange(n_par), nuisance)
    a_mat = np.sqrt(w)[:, None] * j
    a_keep = a_mat[:, keep]
    if nuisance.shape[0] == 0:
        fim = a_keep.T @ a_keep
        return (0.5 * (fim + fim.T)).astype(np.float64, copy=False)
    a_nuis = a_mat[:, nuisance]
    norms = np.linalg.norm(a_nuis, axis=0)
    nonzero = norms > 0.0
    if bool(np.any(nonzero)):
        scaled = a_nuis[:, nonzero] / norms[nonzero][None, :]
        vec_u, sing, _ = np.linalg.svd(scaled, full_matrices=False)
        if bool(np.all(np.isfinite(sing))) and float(sing.max()) > 0.0:
            basis = vec_u[:, sing > rtol_f * float(sing.max())]
            if basis.shape[1] > 0:
                a_keep = a_keep - basis @ (basis.T @ a_keep)
    fim = a_keep.T @ a_keep
    return (0.5 * (fim + fim.T)).astype(np.float64, copy=False)


def ill_posed(
    J: np.ndarray,
    pos_idx: Sequence[int] | np.ndarray,
    cond_max: float = COND_MAX,
    std_max_m: float = STD_MAX_M,
) -> tuple[bool, float, np.ndarray]:
    """Return ``(flag, cond, crb_std)`` for the positions ``pos_idx`` of the FIM ``J``.

    Parameters of ``J`` outside ``pos_idx`` are eliminated first (scale-invariant Schur
    complement), so ``cond`` is that of the position-marginal FIM (consistent units) and
    ``crb_std`` is its per-coordinate CRB std in the order of ``pos_idx``. A non-positive
    eigenvalue gives ``cond = inf`` and infinite stds. ``flag`` is ``cond > cond_max`` or any
    std above ``std_max_m``.
    """
    mat = np.asarray(J, dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("J must be a square 2-D array")
    n = int(mat.shape[0])
    if n < 1 or not bool(np.all(np.isfinite(mat))):
        raise ValueError("J must be finite with n >= 1")
    pos = _validate_indices(pos_idx, n, allow_empty=False, name="pos_idx")
    cond_f = float(cond_max)
    std_f = float(std_max_m)
    if not np.isfinite(cond_f) or cond_f <= 0.0:
        raise ValueError("cond_max must be finite and > 0")
    if not np.isfinite(std_f) or std_f <= 0.0:
        raise ValueError("std_max_m must be finite and > 0")
    sym = 0.5 * (mat + mat.T)
    other = np.setdiff1d(np.arange(n), pos)
    j_pp = sym[np.ix_(pos, pos)]
    if other.shape[0] == 0:
        j_pos = 0.5 * (j_pp + j_pp.T)
    else:
        j_po = sym[np.ix_(pos, other)]
        j_oo = sym[np.ix_(other, other)]
        j_oo_pinv = _scale_invariant_pinv(j_oo)
        j_pos = j_pp - j_po @ j_oo_pinv @ j_po.T
        j_pos = 0.5 * (j_pos + j_pos.T)
    lam, vec = np.linalg.eigh(j_pos)
    if lam.size == 0 or float(lam.max()) <= 0.0 or float(lam.min()) <= 0.0:
        cond = float("inf")
        crb = np.full(pos.shape[0], np.inf, dtype=np.float64)
        return True, cond, crb
    cond = float(float(lam.max()) / float(lam.min()))
    inv_lam = 1.0 / lam
    crb = np.sqrt(np.sum((vec**2) * inv_lam[None, :], axis=1)).astype(np.float64)
    flag = bool(cond > cond_f or np.any(crb > std_f))
    return flag, float(cond), crb


def return_model(
    points: np.ndarray,
    geom: CaptureGeometry,
    tau: np.ndarray | None = None,
    *,
    space: str = "vs",
) -> np.ndarray:
    """Return the noiseless D-list mean ``(u_y, u_z, t)`` as ``[V, B, P, 3]``.

    ``u_y, u_z`` are the UE-local arrival direction components and ``t`` the delay in s plus the
    capture delay gauge ``tau[v, b]`` (unwrapped), from :func:`capture_factors` with the iso
    pattern (geometry only).
    """
    n_views, n_bs = int(geom.num_views), int(geom.num_bs)
    if tau is None:
        delays = np.zeros((n_views, n_bs), dtype=np.float64)
    else:
        delays = np.asarray(tau, dtype=np.float64)
        if delays.shape != (n_views, n_bs):
            raise ValueError("tau must have shape [V, B]")
        if not bool(np.all(np.isfinite(delays))):
            raise ValueError("tau must contain only finite values")
    rows = []
    for v in range(n_views):
        for b in range(n_bs):
            factors = capture_factors(points, geom, space, v, b, pattern="iso")
            rows.append(np.column_stack([factors.u_local[:, 1:3], factors.tau + delays[v, b]]))
    return np.stack(rows).reshape(n_views, n_bs, -1, 3).astype(np.float64, copy=False)
