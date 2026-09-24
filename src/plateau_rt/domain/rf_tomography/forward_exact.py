"""Exact element-level reference forward operator for RF tomography.

Maps point atoms (bistatic Born scatterers in ``"bv"`` space or virtual sources
in ``"vs"`` space) to Sionna RT's multi-view aperture CFR
``Y[V, B, 2, R, C, N]``. This is the reference every later tomography operator
is tested against; the conventions (element order, hemispheres, carrier-only
synthetic phase, delay grid) follow ``docs/tomography_baselines.md``
sections 3.1-3.3. NumPy only: nothing here may import Sionna, Mitsuba or
Dr.Jit. All lengths are metres, angles radians, frequencies hertz.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.constants import c as SPEED_OF_LIGHT

from plateau_rt.domain.rf_tomography.antenna import PATTERN_KINDS, bs_pattern
from plateau_rt.domain.rf_tomography.geometry import (
    LOS_VS_TOLERANCE_M,
    CaptureGeometry,
    _as_points,
    hemisphere_index,
)

SPACES: tuple[str, ...] = ("bv", "vs")
WAVEFRONTS: tuple[str, ...] = ("plane", "spherical")
POLARIZATIONS: tuple[str, ...] = ("none", "vv")
DENSE_MAX_ENTRIES: int = 200_000_000

# Keep every ``[M, N, chunk]`` temporary below this many elements, avoiding a
# ``[M, N, P]`` tensor for the squint and spherical wavefront sweeps.
_ELEMENT_CHUNK_ENTRIES = 1 << 22


@dataclass(frozen=True)
class CaptureFactors:
    """Per-point factors of one capture (v, b); the element and delay phases are not included."""

    u_local: np.ndarray  # [P, 3] UE-local unit arrival vector toward the point
    tau: np.ndarray  # [P] delay in s (VS: r / c; BV: (r1 + r2) / c)
    gamma: np.ndarray  # [P] complex128: G_b * pol * spreading * carrier phase
    hemisphere: np.ndarray  # [P] int64, hemisphere_index(u_local)
    rx_range: np.ndarray  # [P] distance from the UE aperture centre (VS: r; BV: r2)


def _prepare_points(points: np.ndarray) -> np.ndarray:
    """Return finite ``[P, 3]`` float64 points (P >= 1), promoting a single ``[3]`` point."""
    pts = _as_points(points)
    if pts.shape[0] < 1:
        raise ValueError("points must contain at least one point")
    if not np.all(np.isfinite(pts)):
        raise ValueError("points must contain only finite values")
    return pts


def _prepare_amps(amps: np.ndarray, num_points: int, num_views: int, num_bs: int) -> np.ndarray:
    """Return complex128 amplitudes of shape ``[P, V, B]`` from ``[P]`` or ``[P, V, B]``."""
    values = np.asarray(amps).astype(np.complex128)
    if values.ndim == 1:
        if values.shape[0] != num_points:
            raise ValueError("amps must have shape [P] or [P, V, B]")
        return np.broadcast_to(values[:, None, None], (num_points, num_views, num_bs))
    if values.ndim == 3 and values.shape == (num_points, num_views, num_bs):
        return values
    raise ValueError("amps must have shape [P] or [P, V, B]")


def _validate_choice(value: str, allowed: tuple[str, ...], name: str) -> None:
    """Raise ``ValueError`` unless ``value`` is one of ``allowed``."""
    if value not in allowed:
        raise ValueError(f"{name} must be one of {allowed}, got {value!r}")


def _singular_mask(points: np.ndarray, geom: CaptureGeometry, space: str) -> np.ndarray:
    """Return the ``[P]`` mask of points :func:`capture_factors` rejects.

    A point is singular within ``LOS_VS_TOLERANCE_M`` of any UE (both spaces) or
    of any BS (``"bv"`` only; in ``"vs"`` a point at a BS is the LoS source).
    """
    mask = np.zeros(points.shape[0], dtype=bool)
    for ue in geom.ue_pos:
        mask |= np.linalg.norm(points - ue, axis=-1) <= LOS_VS_TOLERANCE_M
    if space == "bv":
        for bs in geom.bs_pos:
            mask |= np.linalg.norm(points - bs, axis=-1) <= LOS_VS_TOLERANCE_M
    return mask


def _reject_points_on_ue(points: np.ndarray, geom: CaptureGeometry) -> None:
    """Raise ``ValueError`` if a point coincides with any UE position."""
    if np.any(_singular_mask(points, geom, "vs")):
        raise ValueError("a point coincides with a UE position")


def _reject_points_on_bs(points: np.ndarray, geom: CaptureGeometry) -> None:
    """Raise ``ValueError`` if a point coincides with any BS position."""
    for bs in geom.bs_pos:
        if np.any(np.linalg.norm(points - bs, axis=-1) <= LOS_VS_TOLERANCE_M):
            raise ValueError("a BV point coincides with a BS position")


def _capture_index(value: int, size: int, name: str) -> int:
    """Return ``value`` as an int in ``[0, size)`` (negative indices are rejected)."""
    try:
        index = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer in [0, {size})") from error
    if index != value or not 0 <= index < size:
        raise ValueError(f"{name} must be an integer in [0, {size})")
    return index


def _v_pol_unit(local_dir: np.ndarray) -> np.ndarray:
    """Return the local V-pol field unit vector of directions ``[..., 3]``.

    ``theta_hat(w) = (cos t cos f, cos t sin f, -sin t)`` with
    ``t = arccos(w_z)`` and ``f = arctan2(w_y, w_x)`` (boresight local +x).
    """
    local = local_dir / np.linalg.norm(local_dir, axis=-1, keepdims=True)
    theta = np.arccos(np.clip(local[..., 2], -1.0, 1.0))
    phi = np.arctan2(local[..., 1], local[..., 0])
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    cos_f, sin_f = np.cos(phi), np.sin(phi)
    return np.stack([cos_t * cos_f, cos_t * sin_f, -sin_t], axis=-1)


def _polarization_factor(
    points: np.ndarray,
    geom: CaptureGeometry,
    space: str,
    v: int,
    b: int,
    u_local: np.ndarray,
    departure: np.ndarray,
    polarization: str,
) -> np.ndarray:
    """Return the real co-polar factor ``[P]`` of the V-pol BS field and UE element."""
    if polarization == "none":
        return np.ones(points.shape[0], dtype=np.float64)
    if geom.bs_rot is None:  # guarded by capture_factors
        raise ValueError("vv polarization requires geom.bs_rot")
    bs_rot = geom.bs_rot[b]
    ue_rot = geom.ue_rot[v]
    e_t = _v_pol_unit(departure @ bs_rot) @ bs_rot.T
    e_r = _v_pol_unit(u_local) @ ue_rot.T
    if space == "bv":
        return np.einsum("pi,pi->p", e_t, e_r)

    to_source = points - geom.bs_pos[b]
    distance = np.linalg.norm(to_source, axis=-1)
    los = distance <= LOS_VS_TOLERANCE_M
    with np.errstate(divide="ignore", invalid="ignore"):
        normal = to_source / distance[:, None]
    mirror = np.eye(3)[None, :, :] - 2.0 * normal[:, :, None] * normal[:, None, :]
    mirror = np.where(los[:, None, None], np.eye(3)[None, :, :], mirror)
    mirrored = np.einsum("pij,pj->pi", mirror, e_t)
    return np.einsum("pi,pi->p", mirrored, e_r)


def capture_factors(
    points: np.ndarray,
    geom: CaptureGeometry,
    space: str,
    v: int,
    b: int,
    *,
    pattern: str = "tr38901",
    polarization: str = "none",
) -> CaptureFactors:
    """Return the per-point factors of the capture ``(v, b)`` (no element/delay phase)."""
    pts = _prepare_points(points)
    _validate_choice(space, SPACES, "space")
    _validate_choice(polarization, POLARIZATIONS, "polarization")
    _validate_choice(pattern, PATTERN_KINDS, "pattern")
    v = _capture_index(v, geom.num_views, "v")
    b = _capture_index(b, geom.num_bs, "b")
    if (pattern == "tr38901" or polarization == "vv") and geom.bs_rot is None:
        raise ValueError("tr38901 pattern and vv polarization require geom.bs_rot")
    _reject_points_on_ue(pts, geom)
    if space == "bv":
        _reject_points_on_bs(pts, geom)

    if space == "vs":
        r = np.linalg.norm(pts - geom.ue_pos[v], axis=-1)
        u_local = geom.local_direction(pts, v)
        tau = r / SPEED_OF_LIGHT
        departure = geom.vs_departure_dir(pts, v, b)
        rx_range = r
        carrier = np.exp(-1j * geom.wavenumber * r)
        spreading = geom.wavelength / (4.0 * np.pi * r)
    else:
        r1, r2 = geom.bistatic_ranges(pts, v, b)
        u_local = geom.local_direction(pts, v)
        tau = (r1 + r2) / SPEED_OF_LIGHT
        departure = (pts - geom.bs_pos[b]) / r1[:, None]
        rx_range = r2
        carrier = np.exp(-1j * geom.wavenumber * (r1 + r2))
        spreading = geom.wavelength / ((4.0 * np.pi) ** 1.5 * r1 * r2)

    bs_rot_b = np.eye(3) if geom.bs_rot is None else geom.bs_rot[b]
    field = bs_pattern(departure, bs_rot_b, kind=pattern)
    pol = _polarization_factor(pts, geom, space, v, b, u_local, departure, polarization)
    gamma = field * pol * spreading * carrier
    return CaptureFactors(
        u_local=u_local,
        tau=tau,
        gamma=gamma,
        hemisphere=hemisphere_index(u_local),
        rx_range=rx_range,
    )


def _plane_phase(geom: CaptureGeometry, u_local: np.ndarray) -> np.ndarray:
    """Return ``exp(+j k u_p . q_m)`` as ``[M, P]``."""
    return np.exp(1j * geom.wavenumber * (geom.elem_offsets @ u_local.T))


def _delay_matrix(geom: CaptureGeometry, tau: np.ndarray) -> np.ndarray:
    """Return ``exp(-j 2 pi df_n tau_p)`` as ``[N, P]``."""
    return np.exp(-2j * np.pi * geom.freq_offsets[:, None] * tau[None, :])


def _element_wavenumbers(geom: CaptureGeometry, squint: bool) -> np.ndarray:
    """Return the per-bin element wavenumbers ``[N]`` (carrier only unless squint)."""
    if not squint:
        return np.full(geom.num_bins, geom.wavenumber, dtype=np.float64)
    # Written as a DC bin plus an offset so the DC entry is exactly ``wavenumber``,
    # which makes the squinted and plain operators agree bit-for-bit at n = N // 2.
    return geom.wavenumber + 2.0 * np.pi * geom.freq_offsets / SPEED_OF_LIGHT


def _element_path(
    points: np.ndarray,
    geom: CaptureGeometry,
    v: int,
    factors: CaptureFactors,
    wavefront: str,
) -> np.ndarray:
    """Return ``phi[m, p]`` with element phase ``exp(+j k_e[n] phi[m, p])``.

    Plane: ``phi = u_p . q_m``. Spherical: ``phi = r_rx - r_m`` (the sign of the
    exact excess path), so ``|Y|`` is unchanged.
    """
    if wavefront == "plane":
        return geom.elem_offsets @ factors.u_local.T
    centers = geom.ue_pos[v] + geom.elem_offsets @ geom.ue_rot[v].T
    r_m = np.linalg.norm(points[None, :, :] - centers[:, None, :], axis=-1)
    return factors.rx_range[None, :] - r_m


def _capture_atom_block(
    points: np.ndarray,
    geom: CaptureGeometry,
    v: int,
    b: int,
    factors: CaptureFactors,
    weight: np.ndarray,
    wavefront: str,
    squint: bool,
) -> np.ndarray:
    """Return the summed ``[2, M, N]`` block of one capture for weights ``[P]``."""
    num_elements, num_bins = geom.num_elements, geom.num_bins
    delay = _delay_matrix(geom, factors.tau)
    block = np.zeros((2, num_elements, num_bins), dtype=np.complex128)
    if wavefront == "plane" and not squint:
        phase = _plane_phase(geom, factors.u_local)
        for h in range(2):
            masked = np.where(factors.hemisphere == h, weight, 0.0)
            block[h] = (phase * masked[None, :]) @ delay.T
        return block

    phi = _element_path(points, geom, v, factors, wavefront)
    k_e = _element_wavenumbers(geom, squint)
    chunk = max(1, _ELEMENT_CHUNK_ENTRIES // (num_elements * num_bins))
    for start in range(0, points.shape[0], chunk):
        sl = slice(start, start + chunk)
        element_phase = np.exp(1j * k_e[None, :, None] * phi[:, sl][:, None, :])
        for h in range(2):
            masked = np.where(factors.hemisphere[sl] == h, weight[sl], 0.0)
            weighted = (element_phase * masked[None, None, :]) * delay[:, sl][None, :, :]
            block[h] += weighted.sum(axis=2)
    return block


def atom_cfr(
    points: np.ndarray,
    amps: np.ndarray,
    geom: CaptureGeometry,
    space: str,
    *,
    wavefront: str = "plane",
    squint: bool = False,
    pattern: str = "tr38901",
    polarization: str = "none",
) -> np.ndarray:
    """Return the exact aperture CFR ``[V, B, 2, R, C, N]`` of point atoms.

    ``Y[v, b, h, m, n] = sum_p amp_p gamma_p 1[h = h_p] exp(+j k_e u_p . q_m)
    exp(-j 2 pi df_n tau_p)`` with ``gamma`` from :func:`capture_factors`. VS:
    ``gamma = G_b(d_dep) pol lam / (4 pi r) exp(-j k r)``, ``tau = r / c``; BV:
    ``gamma = G_b((x - t_b) / r1) pol lam / ((4 pi)^1.5 r1 r2) exp(-j k (r1 + r2))``,
    ``tau = (r1 + r2) / c``. ``k_e`` is the carrier wavenumber (Sionna's synthetic
    array) unless ``squint``; ``wavefront="spherical"`` replaces the plane-wave
    element phase by the exact element distance. ``polarization="vv"`` adds the
    V-pol co-polar factor, which Sionna applies (exact for the LoS). ``amps`` is
    ``[P]`` or per capture ``[P, V, B]``.
    """
    pts = _prepare_points(points)
    _validate_choice(space, SPACES, "space")
    _validate_choice(wavefront, WAVEFRONTS, "wavefront")
    _validate_choice(polarization, POLARIZATIONS, "polarization")
    _validate_choice(pattern, PATTERN_KINDS, "pattern")
    amplitudes = _prepare_amps(amps, pts.shape[0], geom.num_views, geom.num_bs)

    rows, cols = geom.aperture_shape
    result = np.zeros(
        (geom.num_views, geom.num_bs, 2, rows, cols, geom.num_bins), dtype=np.complex128
    )
    for v in range(geom.num_views):
        for b in range(geom.num_bs):
            factors = capture_factors(
                pts, geom, space, v, b, pattern=pattern, polarization=polarization
            )
            weight = amplitudes[:, v, b] * factors.gamma
            block = _capture_atom_block(pts, geom, v, b, factors, weight, wavefront, squint)
            result[v, b] = block.reshape(2, rows, cols, geom.num_bins)
    return result


def dense_matrix(
    points: np.ndarray,
    geom: CaptureGeometry,
    space: str,
    *,
    wavefront: str = "plane",
    squint: bool = False,
    pattern: str = "tr38901",
    polarization: str = "none",
) -> np.ndarray:
    """Return the dense operator ``[V * B * 2 * M * N, P]`` (column p is one atom)."""
    pts = _prepare_points(points)
    _validate_choice(space, SPACES, "space")
    _validate_choice(wavefront, WAVEFRONTS, "wavefront")
    _validate_choice(polarization, POLARIZATIONS, "polarization")
    _validate_choice(pattern, PATTERN_KINDS, "pattern")

    num_points = pts.shape[0]
    num_views, num_bs = geom.num_views, geom.num_bs
    rows, cols = geom.aperture_shape
    num_elements, num_bins = geom.num_elements, geom.num_bins
    row_count = num_views * num_bs * 2 * num_elements * num_bins
    if num_points * row_count > DENSE_MAX_ENTRIES:
        raise ValueError(
            f"dense_matrix needs {num_points * row_count} entries, "
            f"exceeding DENSE_MAX_ENTRIES={DENSE_MAX_ENTRIES}"
        )

    out = np.zeros((num_views, num_bs, 2, num_elements, num_bins, num_points), dtype=np.complex128)
    chunk = max(1, _ELEMENT_CHUNK_ENTRIES // (num_elements * num_bins))
    for v in range(num_views):
        for b in range(num_bs):
            factors = capture_factors(
                pts, geom, space, v, b, pattern=pattern, polarization=polarization
            )
            target = out[v, b]
            delay = _delay_matrix(geom, factors.tau)
            if wavefront == "plane" and not squint:
                phase = _plane_phase(geom, factors.u_local)
                for h in range(2):
                    masked = np.where(factors.hemisphere == h, factors.gamma, 0.0)
                    target[h] = phase[:, None, :] * masked[None, None, :] * delay[None, :, :]
                continue
            phi = _element_path(pts, geom, v, factors, wavefront)
            k_e = _element_wavenumbers(geom, squint)
            for start in range(0, num_points, chunk):
                sl = slice(start, start + chunk)
                element_phase = np.exp(1j * k_e[None, :, None] * phi[:, sl][:, None, :])
                for h in range(2):
                    masked = np.where(factors.hemisphere[sl] == h, factors.gamma[sl], 0.0)
                    target[h][:, :, sl] = (element_phase * masked[None, None, :]) * delay[:, sl][
                        None, :, :
                    ]
    return out.reshape(row_count, num_points)
