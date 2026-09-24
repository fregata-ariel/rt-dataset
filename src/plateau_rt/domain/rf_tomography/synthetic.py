"""L0 analytic phantoms for RF tomography (design section 6.2).

``Y_clean`` is rendered by ``forward_exact.atom_cfr`` with the scalar model
(``polarization="none"``); point phantoms are placed off-grid to avoid the
rf-gs-toy inverse crime. NumPy only.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
from scipy.constants import c as SPEED_OF_LIGHT
from scipy.constants import epsilon_0 as EPSILON_0

from plateau_rt.domain.rf_camera.camera import generate_ring_views
from plateau_rt.domain.rf_tomography.forward_exact import (
    atom_cfr,
    capture_factors,
    dense_matrix,
)
from plateau_rt.domain.rf_tomography.geometry import (
    CaptureGeometry,
    VoxelGrid,
    mirror_point,
    single_bounce_point,
)

LEVELS: tuple[str, ...] = ("L0a", "L0b", "L0c", "L0d", "L0e", "L0-mm")
MISMATCHES: tuple[str, ...] = ("spherical", "squint", "element_jitter")
L0B_AXES: tuple[str, ...] = ("range", "cross_range", "vertical")
MIN_OFFSET_FRACTION: float = 0.05
DEFAULT_OFFSET_FRACTION: float = 0.45
DEFAULT_CENTER: tuple[float, float, float] = (0.0, 0.0, 5.0)
DEFAULT_BS_POS: tuple[tuple[float, float, float], ...] = ((-70.0, 5.0, 25.0),)
DEFAULT_GRID_SIZE: float = 20.0
DEFAULT_SPACING: float = 0.5
DEFAULT_BOX_SIZE: float = 10.0
DEFAULT_JITTER_FRACTION: float = 1.0 / 200.0
DEFAULT_WALLS: tuple[tuple[tuple[float, float, float], tuple[float, float, float], str], ...] = (
    ((45.0, 0.0, 0.0), (-1.0, 0.0, 0.0), "concrete"),
    ((0.0, -40.0, 0.0), (0.0, 1.0, 0.0), "concrete"),
)
ITU_MATERIALS: dict[str, tuple[float, float, float, float, float, float]] = {
    "concrete": (1.0, 100.0, 5.24, 0.0, 0.0462, 0.7822),
    "brick": (1.0, 40.0, 3.91, 0.0, 0.0238, 0.16),
    "wood": (0.001, 100.0, 1.99, 0.0, 0.0047, 1.0718),
    "glass": (0.1, 100.0, 6.31, 0.0, 0.0036, 1.3394),
    "metal": (1.0, 100.0, 1.0, 0.0, 1e7, 0.0),
    "asphalt_concrete": (1.0, 40.0, 4.83, 0.0, 0.0108, 1.3969),
    "very_dry_ground": (1.0, 10.0, 3.0, 0.0, 0.00015, 2.52),
    "medium_dry_ground": (1.0, 10.0, 15.0, -0.1, 0.035, 1.63),
    "wet_ground": (1.0, 10.0, 30.0, -0.4, 0.15, 1.30),
}


@dataclass(frozen=True)
class PhantomGT:
    """Ground truth of one analytic phantom."""

    level: str
    space: str
    pattern: str
    points_pos: np.ndarray
    points_rho: np.ndarray
    grid: VoxelGrid | None
    seed: int
    mismatch: tuple[str, ...]
    meta: dict[str, Any]


class Phantom(NamedTuple):
    """A rendered phantom: unpacks as ``(y_clean, geom, gt)``."""

    y_clean: np.ndarray
    geom: CaptureGeometry
    gt: PhantomGT


def _check_seed(seed: int) -> int:
    """Return ``seed`` as int, raising unless it is an int >= 0."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ValueError("seed must be an int >= 0")
    value = int(seed)
    if value < 0:
        raise ValueError("seed must be an int >= 0")
    return value


def _normalise_mismatch(mismatch: Sequence[str]) -> tuple[str, ...]:
    """Validate and order a mismatch sequence in ``MISMATCHES`` order."""
    names = list(mismatch)
    for name in names:
        if name not in MISMATCHES:
            raise ValueError(f"unknown mismatch {name!r}")
    if len(set(names)) != len(names):
        raise ValueError("duplicate mismatch names")
    return tuple(name for name in MISMATCHES if name in names)


def _resolve_offset_max(grid: VoxelGrid, offset_max: float | None) -> float:
    """Return the validated offset bound for ``grid``."""
    spacing = float(grid.spacing)
    value = DEFAULT_OFFSET_FRACTION * spacing if offset_max is None else float(offset_max)
    if not np.isfinite(value) or not (MIN_OFFSET_FRACTION * spacing < value <= 0.5 * spacing):
        raise ValueError("offset_max must satisfy MIN_OFFSET*spacing < offset_max <= 0.5*spacing")
    return value


def ring_geometry(
    center: Sequence[float] | np.ndarray = DEFAULT_CENTER,
    *,
    num_views: int = 8,
    radius: float = 30.0,
    ue_height: float = 1.5,
    bs_pos: Sequence[Sequence[float]] | np.ndarray = DEFAULT_BS_POS,
    f_c: float = 3.5e9,
    bandwidth: float = 100e6,
    num_bins: int = 128,
    aperture_shape: tuple[int, int] = (8, 8),
    start_azimuth_deg: float = 0.0,
) -> CaptureGeometry:
    """Build the default multi-view ring capture geometry looking at ``center``."""
    target = tuple(float(v) for v in np.asarray(center, dtype=np.float64).ravel())
    if len(target) != 3:
        raise ValueError("center must have 3 entries")
    views = generate_ring_views(
        target=target,
        radius_m=float(radius),
        ue_height_m=float(ue_height),
        num_views=int(num_views),
        start_azimuth_deg=float(start_azimuth_deg),
    )
    ue_pos = np.asarray([v.position for v in views], dtype=np.float64)
    orientations = np.asarray([v.orientation for v in views], dtype=np.float64)
    bs_array = np.asarray(bs_pos, dtype=np.float64).reshape(-1, 3)
    return CaptureGeometry.from_orientations(
        ue_pos,
        orientations,
        bs_array,
        f_c=float(f_c),
        bandwidth=float(bandwidth),
        num_bins=int(num_bins),
        aperture_shape=(int(aperture_shape[0]), int(aperture_shape[1])),
        bs_look_at=np.asarray(target, dtype=np.float64),
    )


def default_grid(
    center: Sequence[float] | np.ndarray = DEFAULT_CENTER,
    *,
    size: float = DEFAULT_GRID_SIZE,
    spacing: float = DEFAULT_SPACING,
) -> VoxelGrid:
    """Build the reconstruction grid tiling the cube of edge ``size`` around ``center``."""
    centre = np.asarray(center, dtype=np.float64).reshape(3)
    size_f = float(size)
    spacing_f = float(spacing)
    lower = centre - size_f / 2.0 + spacing_f / 2.0
    upper = centre + size_f / 2.0 - spacing_f / 2.0
    return VoxelGrid.from_bounds(lower, upper, spacing_f)


def offgrid_points(
    grid: VoxelGrid,
    num_points: int,
    rng: np.random.Generator,
    *,
    offset_max: float | None = None,
    voxels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw distinct voxels and place one off-grid point in each.

    Each point is its voxel centre plus a per-axis offset with random sign and
    magnitude uniform in ``[0.05 * spacing, offset_max]`` (``offset_max``
    defaults to ``0.45 * spacing``), so ``0.05 * spacing <= |offset| <=
    offset_max`` holds on every axis. Returns ``(points,
    drawn)`` with ``points`` [K, 3] float64 and ``drawn`` [K] int64 flat voxel
    indices. The RNG draw order is ``choice`` (voxels), ``uniform``
    (magnitudes), ``choice`` (signs).
    """
    bound = _resolve_offset_max(grid, offset_max)
    low = MIN_OFFSET_FRACTION * float(grid.spacing)
    try:
        count = int(num_points)
    except (TypeError, ValueError) as error:
        raise ValueError("num_points must be an integer >= 1") from error
    if count != num_points or count < 1:
        raise ValueError("num_points must be an integer >= 1")
    if voxels is None:
        candidates = np.arange(grid.size, dtype=np.int64)
    else:
        candidates = np.asarray(voxels).reshape(-1)
        if candidates.size == 0:
            raise ValueError("voxels must be non-empty")
        if candidates.dtype.kind not in "iu":
            try:
                candidates = candidates.astype(np.int64)
            except (TypeError, ValueError) as error:
                raise ValueError("voxel indices must be integers") from error
        else:
            candidates = candidates.astype(np.int64, copy=True)
        if np.any(candidates < 0) or np.any(candidates >= grid.size):
            raise ValueError("voxel index out of range")
    if count > candidates.size:
        raise ValueError("num_points exceeds the number of candidate voxels")
    drawn = rng.choice(candidates, size=count, replace=False).astype(np.int64)
    magnitude = rng.uniform(low, bound, size=(count, 3))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(count, 3))
    multi = np.stack(np.unravel_index(drawn, grid.shape), axis=1).astype(np.float64)
    centres = np.asarray(grid.origin, dtype=np.float64) + float(grid.spacing) * multi
    points = centres + signs * magnitude
    return points.astype(np.float64), drawn.astype(np.int64)


def _grid_centre(grid: VoxelGrid) -> np.ndarray:
    """Return the geometric centre of ``grid``."""
    return (
        np.asarray(grid.origin, dtype=np.float64)
        + float(grid.spacing) * (np.asarray(grid.shape, dtype=np.float64) - 1.0) / 2.0
    )


def _axis_direction(axis: str, grid: VoxelGrid, geom: CaptureGeometry, ref_view: int) -> np.ndarray:
    """Return the unit separation direction for ``axis``."""
    if axis not in L0B_AXES:
        raise ValueError(f"unknown axis {axis!r}")
    if not isinstance(ref_view, (int, np.integer)) or int(ref_view) < 0:
        raise ValueError("ref_view out of range")
    ref = int(ref_view)
    if ref >= geom.num_views:
        raise ValueError("ref_view out of range")
    centre = _grid_centre(grid)
    to_centre = centre - np.asarray(geom.ue_pos[ref], dtype=np.float64)
    norm = float(np.linalg.norm(to_centre))
    if norm == 0.0:
        raise ValueError("UE position coincides with the grid centre")
    range_dir = to_centre / norm
    if axis == "range":
        return range_dir
    cross = np.cross(np.array([0.0, 0.0, 1.0]), range_dir)
    cross_norm = float(np.linalg.norm(cross))
    if cross_norm <= 1e-12:
        raise ValueError("range direction is vertical")
    cross_dir = cross / cross_norm
    if axis == "cross_range":
        return cross_dir
    vertical = np.cross(range_dir, cross_dir)
    vnorm = float(np.linalg.norm(vertical))
    if vnorm == 0.0:
        raise ValueError("degenerate axis frame")
    return vertical / vnorm


def itu_permittivity(material: str, frequency: float) -> complex:
    """Return the ITU-R P.2040 complex relative permittivity at ``frequency``."""
    if material not in ITU_MATERIALS:
        raise ValueError(f"unknown material {material!r}")
    f_min, f_max, a, b, c_coef, d = ITU_MATERIALS[material]
    freq = float(frequency)
    if not np.isfinite(freq) or freq <= 0.0:
        raise ValueError("frequency must be finite and > 0")
    f_ghz = freq / 1e9
    if not (f_min <= f_ghz <= f_max):
        raise ValueError("frequency outside the tabulated range")
    sigma = c_coef * f_ghz**d
    eta = a * f_ghz**b - 1j * sigma / (2.0 * np.pi * freq * EPSILON_0)
    return complex(eta)


def fresnel_coefficients(
    cos_theta: np.ndarray | float, eta: complex
) -> tuple[np.ndarray, np.ndarray]:
    """Return the (TE, TM) Fresnel coefficients of ITU-R P.2040 eq. (37)."""
    cos_arr = np.asarray(cos_theta, dtype=np.float64)
    if cos_arr.size == 0:
        raise ValueError("cos_theta must be non-empty")
    if not bool(np.all(np.isfinite(cos_arr))):
        raise ValueError("cos_theta must be finite")
    if bool(np.any(cos_arr < 0.0)) or bool(np.any(cos_arr > 1.0)):
        raise ValueError("cos_theta must lie in [0, 1]")
    eta_c = complex(eta)
    if not np.isfinite(eta_c.real) or not np.isfinite(eta_c.imag):
        raise ValueError("eta must be finite")
    cos_c = cos_arr.astype(np.complex128)
    root = np.sqrt(eta_c - (1.0 - cos_c**2))
    r_te = (cos_c - root) / (cos_c + root)
    r_tm = (eta_c * cos_c - root) / (eta_c * cos_c + root)
    return r_te.astype(np.complex128), r_tm.astype(np.complex128)


def mismatch_floor_prediction(
    points: np.ndarray,
    amps: np.ndarray,
    geom: CaptureGeometry,
    space: str,
    *,
    mismatch: Sequence[str],
    elem_offsets_true: np.ndarray | None = None,
    pattern: str = "tr38901",
) -> float:
    """Predict the plane-wave operator NMSE floor under ``mismatch``.

    Model (one pass over captures ``c = (v, b)``, per atom ``p``): atom ``p``
    sits at ``points[p]`` with amplitude ``a_p = amps[p]``; for capture ``c``,
    ``u`` (UE-local arrival unit vector), ``r`` (receive range) and ``gamma``
    come from ``capture_factors``; nominal element offsets ``q_m``
    (``geom.elem_offsets``), true offsets ``q'_m`` (``elem_offsets_true``,
    default ``q_m``); ``k_e[n] = k_c + 2 pi df_n / c`` with ``"squint"``, else
    ``k_e[n] = k_c``; ``phi_true[m, n] = k_e[n] (u.q'_m - s (|q'_m|^2 -
    (u.q'_m)^2) / (2 r))`` with ``s = 1`` under ``"spherical"`` (second-order
    Taylor of ``-k_e (r_m - r)``), else ``s = 0``; ``phi_model[m] = k_c
    u.q_m``; ``eps = phi_true - phi_model`` (radians); ``w_{p,c} = |a_p|^2
    |gamma_{p,c}|^2``. Per atom, ``S0[p] += w[p] M N``, ``S1[p] += w[p] sum
    eps[p]``, ``S2[p] += w[p] sum eps[p]^2``; the refitted complex amplitude of
    atom ``p`` absorbs the weighted mean ``eps_bar_p = S1[p] / S0[p]``. For
    ``|eps| << 1``, ``1 - |E e^{j eps}|^2 ~= var(eps)``, and for
    near-orthogonal atom columns ``NMSE_pred = sum_p (S2 - S1^2 / S0) / sum_p
    S0`` (0.0 when ``sum_p S0 == 0``). Closed-form orders: spherical ``~
    k_c^2 var_m(|q|^2 - (u.q)^2) / (4 r^2)`` (falls as ``1/r^2``), squint ``~
    (2 pi / c)^2 (B^2 / 12) E_m[(u.q)^2]`` (zero at boresight), jitter with 3-D
    isotropic std sigma ``~ k_c^2 sigma^2 (1 - 1/M)``.
    """
    names = _normalise_mismatch(tuple(mismatch))
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 1:
        raise ValueError("points must have shape [K, 3] with K >= 1")
    amp = np.asarray(amps)
    if amp.ndim != 1 or amp.shape[0] != pts.shape[0]:
        raise ValueError("amps must have shape [K]")
    amp = amp.astype(np.complex128)
    if space not in ("bv", "vs"):
        raise ValueError("space must be 'bv' or 'vs'")
    num_elements = geom.num_elements
    num_bins = geom.num_bins
    if elem_offsets_true is None:
        true_offsets = np.asarray(geom.elem_offsets, dtype=np.float64)
    else:
        true_offsets = np.asarray(elem_offsets_true, dtype=np.float64)
        if true_offsets.shape != (num_elements, 3):
            raise ValueError("elem_offsets_true must have shape [M, 3]")
        if not np.all(np.isfinite(true_offsets)):
            raise ValueError("elem_offsets_true must be finite")
    nominal = np.asarray(geom.elem_offsets, dtype=np.float64)
    use_spherical = "spherical" in names
    use_squint = "squint" in names
    k_c = float(geom.wavenumber)
    if use_squint:
        k_e = k_c + 2.0 * np.pi * np.asarray(geom.freq_offsets, dtype=np.float64) / SPEED_OF_LIGHT
    else:
        k_e = np.full(num_bins, k_c, dtype=np.float64)
    true_norm2 = np.sum(true_offsets**2, axis=1)
    amp_power = np.abs(amp) ** 2
    order = num_elements * num_bins
    s_flag = 1.0 if use_spherical else 0.0
    s0 = np.zeros(pts.shape[0], dtype=np.float64)
    s1 = np.zeros(pts.shape[0], dtype=np.float64)
    s2 = np.zeros(pts.shape[0], dtype=np.float64)
    for v in range(geom.num_views):
        for b in range(geom.num_bs):
            factors = capture_factors(pts, geom, space, v, b, pattern=pattern)
            u_local = np.asarray(factors.u_local, dtype=np.float64)
            rx_range = np.asarray(factors.rx_range, dtype=np.float64)
            gamma = np.asarray(factors.gamma, dtype=np.complex128)
            weight = amp_power * np.abs(gamma) ** 2
            uq_nom = u_local @ nominal.T
            uq_true = u_local @ true_offsets.T
            if use_spherical:
                curve = (true_norm2[None, :] - uq_true**2) / (2.0 * rx_range[:, None])
            else:
                curve = 0.0
            phase = uq_true - s_flag * curve
            eps = phase[:, :, None] * k_e[None, None, :] - (k_c * uq_nom)[:, :, None]
            s0 += weight * order
            s1 += weight * np.sum(eps, axis=(1, 2))
            s2 += weight * np.sum(eps**2, axis=(1, 2))
    denom = float(np.sum(s0))
    if denom == 0.0:
        return 0.0
    valid = s0 > 0.0
    numerator = float(np.sum(s2[valid] - s1[valid] ** 2 / s0[valid]))
    return float(numerator / denom)


def plane_wave_floor(phantom: Phantom) -> float:
    """Measure the NMSE floor of the nominal plane-wave operator on ``phantom``."""
    rho = np.asarray(phantom.gt.points_rho)
    if rho.ndim != 1:
        raise ValueError("plane_wave_floor needs shared amplitudes [K]")
    mat = dense_matrix(
        np.asarray(phantom.gt.points_pos, dtype=np.float64),
        phantom.geom,
        phantom.gt.space,
        pattern=phantom.gt.pattern,
    )
    vec = np.asarray(phantom.y_clean, dtype=np.complex128).ravel()
    denom = float(np.vdot(vec, vec).real)
    if denom == 0.0:
        return 0.0
    coef, _, _, _ = np.linalg.lstsq(mat, vec, rcond=None)
    resid = vec - mat @ coef
    return float(np.vdot(resid, resid).real / denom)


def _render_point_phantom(
    points: np.ndarray,
    rho: np.ndarray,
    geom: CaptureGeometry,
    space: str,
    pattern: str,
    plain_level: str,
    seed: int,
    mismatch: Sequence[str],
    jitter_std: float | None,
) -> tuple[np.ndarray, str, tuple[str, ...], dict[str, Any]]:
    """Render BV points with optional mismatch; return (y, level, mismatch, extra meta)."""
    names = _normalise_mismatch(tuple(mismatch))
    pts = np.asarray(points, dtype=np.float64)
    amp = np.asarray(rho, dtype=np.complex128)
    if len(names) == 0:
        y_clean = atom_cfr(pts, amp, geom, space, pattern=pattern)
        return y_clean, plain_level, (), {}
    jitter_value: float | None = None if jitter_std is None else float(jitter_std)
    if jitter_value is not None and (not np.isfinite(jitter_value) or jitter_value < 0.0):
        raise ValueError("jitter_std must be finite and >= 0")
    default_jitter = DEFAULT_JITTER_FRACTION * float(geom.wavelength)
    if "element_jitter" in names:
        js = default_jitter if jitter_value is None else jitter_value
        rng_jitter = np.random.default_rng(np.random.SeedSequence([int(seed), 2]))
        delta = rng_jitter.normal(0.0, js, size=(geom.num_elements, 3))
        elem_true = np.asarray(geom.elem_offsets, dtype=np.float64) + delta
        stored = float(js)
    else:
        elem_true = np.asarray(geom.elem_offsets, dtype=np.float64).copy()
        stored = 0.0
    geom_true = dataclasses.replace(geom, elem_offsets=elem_true)
    wavefront = "spherical" if "spherical" in names else "plane"
    squint = "squint" in names
    y_clean = atom_cfr(
        pts, amp, geom_true, space, wavefront=wavefront, squint=squint, pattern=pattern
    )
    pred = mismatch_floor_prediction(
        pts, amp, geom, space, mismatch=names, elem_offsets_true=elem_true, pattern=pattern
    )
    with np.errstate(divide="ignore"):
        pred_db = float(10.0 * np.log10(pred)) if pred > 0.0 else float("-inf")
    extra = {
        "base": plain_level,
        "elem_offsets_true": np.asarray(elem_true, dtype=np.float64),
        "jitter_std": float(stored),
        "nmse_floor_pred": float(pred),
        "nmse_floor_pred_db": float(pred_db),
    }
    return y_clean, "L0-mm", names, extra


def l0a_point(
    seed: int,
    *,
    voxel: int | None = None,
    amplitude: complex = 1.0,
    grid: VoxelGrid | None = None,
    offset_max: float | None = None,
    geom: CaptureGeometry | None = None,
    pattern: str = "tr38901",
    mismatch: Sequence[str] = (),
    jitter_std: float | None = None,
) -> Phantom:
    """Generate the L0a single off-grid point phantom.

    Scene: one BV atom with complex ``amplitude``, placed off-grid in ``voxel``
    (default: the central voxel) via the ``SeedSequence([seed, 0])`` placement
    stream. Rendered with the nominal plane-wave array unless ``mismatch`` is
    given (see ``l0_mm``). Meta keys: ``voxel`` (plus ``base``,
    ``elem_offsets_true``, ``jitter_std``, ``nmse_floor_pred`` and
    ``nmse_floor_pred_db`` when ``mismatch`` is non-empty).
    """
    seed_i = _check_seed(seed)
    grid_r = default_grid() if grid is None else grid
    geom_r = ring_geometry() if geom is None else geom
    if voxel is None:
        voxel_i = int(np.ravel_multi_index(tuple(s // 2 for s in grid_r.shape), grid_r.shape))
    else:
        if isinstance(voxel, bool) or not isinstance(voxel, (int, np.integer)):
            raise ValueError("voxel must be an integer flat index")
        voxel_i = int(voxel)
    rng_place = np.random.default_rng(np.random.SeedSequence([seed_i, 0]))
    points, _ = offgrid_points(
        grid_r, 1, rng_place, offset_max=offset_max, voxels=np.array([voxel_i])
    )
    rho = np.asarray([amplitude], dtype=np.complex128).reshape(1)
    y_clean, level, names, extra = _render_point_phantom(
        points, rho, geom_r, "bv", pattern, "L0a", seed_i, mismatch, jitter_std
    )
    meta: dict[str, Any] = {"voxel": int(voxel_i)}
    meta.update(extra)
    gt = PhantomGT(
        level=level,
        space="bv",
        pattern=pattern,
        points_pos=np.asarray(points, dtype=np.float64),
        points_rho=np.asarray(rho, dtype=np.complex128),
        grid=grid_r,
        seed=seed_i,
        mismatch=names,
        meta=meta,
    )
    return Phantom(y_clean=np.asarray(y_clean, dtype=np.complex128), geom=geom_r, gt=gt)


def l0b_pair(
    seed: int,
    separation: float,
    axis: str,
    *,
    amps: tuple[complex, complex] = (1.0, 1.0),
    ref_view: int = 0,
    box_size: float = DEFAULT_BOX_SIZE,
    grid: VoxelGrid | None = None,
    offset_max: float | None = None,
    geom: CaptureGeometry | None = None,
    pattern: str = "tr38901",
    mismatch: Sequence[str] = (),
    jitter_std: float | None = None,
) -> Phantom:
    """Generate the L0b two-point resolution phantom.

    Scene: two BV atoms with ``amps``, separated by ``separation`` metres along
    ``axis`` (``range`` / ``cross_range`` / ``vertical``). The frame is built
    from ``ref_view`` and the grid centre: ``range`` points from the UE to the
    centre, ``cross_range`` is horizontal and orthogonal to it, ``vertical``
    completes the triad. Both points are off-grid (rejection sampling on the
    ``SeedSequence([seed, 0])`` stream, 64 rounds of 1024 proposals; the pair
    midpoint must stay in the centred cube of edge ``box_size``). Meta keys:
    ``separation``, ``axis``, ``axis_dir`` (plus the L0-mm keys when
    ``mismatch`` is non-empty).
    """
    seed_i = _check_seed(seed)
    sep = float(separation)
    if not np.isfinite(sep) or sep <= 0.0:
        raise ValueError("separation must be finite and > 0")
    box = float(box_size)
    if not np.isfinite(box) or box <= 0.0:
        raise ValueError("box_size must be finite and > 0")
    grid_r = default_grid() if grid is None else grid
    geom_r = ring_geometry() if geom is None else geom
    axis_dir = _axis_direction(axis, grid_r, geom_r, ref_view)
    bound = _resolve_offset_max(grid_r, offset_max)
    low = MIN_OFFSET_FRACTION * float(grid_r.spacing)
    centre = _grid_centre(grid_r)
    centres = np.asarray(grid_r.centers(), dtype=np.float64)
    shifted = centres + sep * axis_dir
    shifted_index = grid_r.index(shifted)
    midpoints = centres + sep * axis_dir / 2.0
    in_box = np.max(np.abs(midpoints - centre), axis=1) <= box / 2.0
    candidates = np.where((shifted_index != -1) & in_box)[0].astype(np.int64)
    if candidates.size == 0:
        raise ValueError("no candidate voxel for the requested pair")
    rng_place = np.random.default_rng(np.random.SeedSequence([seed_i, 0]))
    signs_set = np.array([-1.0, 1.0])
    found: tuple[np.ndarray, np.ndarray] | None = None
    origin = np.asarray(grid_r.origin, dtype=np.float64)
    spacing = float(grid_r.spacing)
    for _ in range(64):
        vox_batch = rng_place.choice(candidates, size=1024, replace=True)
        magnitude = rng_place.uniform(low, bound, size=(1024, 3))
        signs = rng_place.choice(signs_set, size=(1024, 3))
        p1_batch = centres[vox_batch] + signs * magnitude
        p2_batch = p1_batch + sep * axis_dir
        inside = grid_r.index(p2_batch) != -1
        idx = np.where(inside)[0]
        if idx.size == 0:
            continue
        cand_p2 = p2_batch[idx]
        nearest = origin + spacing * np.round((cand_p2 - origin) / spacing)
        off = np.abs(cand_p2 - nearest)
        ok = np.all((off >= low - 1e-12) & (off <= bound + 1e-12), axis=1)
        ok_idx = np.where(ok)[0]
        if ok_idx.size == 0:
            continue
        first = int(idx[int(ok_idx[0])])
        found = (p1_batch[first], p2_batch[first])
        break
    if found is None:
        raise ValueError("no feasible pair found within the attempt budget")
    points = np.stack([found[0], found[1]], axis=0).astype(np.float64)
    rho = np.asarray(amps, dtype=np.complex128).reshape(2)
    if rho.shape != (2,):
        raise ValueError("amps must have two entries")
    y_clean, level, names, extra = _render_point_phantom(
        points, rho, geom_r, "bv", pattern, "L0b", seed_i, mismatch, jitter_std
    )
    meta: dict[str, Any] = {
        "separation": float(sep),
        "axis": str(axis),
        "axis_dir": np.asarray(axis_dir, dtype=np.float64),
    }
    meta.update(extra)
    gt = PhantomGT(
        level=level,
        space="bv",
        pattern=pattern,
        points_pos=np.asarray(points, dtype=np.float64),
        points_rho=np.asarray(rho, dtype=np.complex128),
        grid=grid_r,
        seed=seed_i,
        mismatch=names,
        meta=meta,
    )
    return Phantom(y_clean=np.asarray(y_clean, dtype=np.complex128), geom=geom_r, gt=gt)


def l0c_random(
    seed: int,
    num_points: int = 16,
    *,
    dynamic_range_db: float = 30.0,
    box_size: float = DEFAULT_BOX_SIZE,
    grid: VoxelGrid | None = None,
    offset_max: float | None = None,
    geom: CaptureGeometry | None = None,
    pattern: str = "tr38901",
    mismatch: Sequence[str] = (),
    jitter_std: float | None = None,
) -> Phantom:
    """Generate the L0c random multi-point phantom in a centred box.

    Scene: ``num_points`` BV atoms in distinct voxels whose centres lie within
    the centred cube of edge ``box_size``. ``20 log10 |rho|`` is uniform in
    ``[-dynamic_range_db, 0]`` with uniform phase (``SeedSequence([seed, 1])``
    amplitude stream; placement uses ``SeedSequence([seed, 0])``). Rendered
    with the nominal array unless ``mismatch`` is given (see ``l0_mm``). Meta
    keys: ``box_size``, ``dynamic_range_db``, ``voxels`` (plus the L0-mm keys
    when ``mismatch`` is non-empty).
    """
    seed_i = _check_seed(seed)
    try:
        count = int(num_points)
    except (TypeError, ValueError) as error:
        raise ValueError("num_points must be an integer >= 1") from error
    if count != num_points or count < 1:
        raise ValueError("num_points must be an integer >= 1")
    dr_db = float(dynamic_range_db)
    if not np.isfinite(dr_db) or dr_db < 0.0:
        raise ValueError("dynamic_range_db must be finite and >= 0")
    box = float(box_size)
    if not np.isfinite(box) or box <= 0.0:
        raise ValueError("box_size must be finite and > 0")
    grid_r = default_grid() if grid is None else grid
    geom_r = ring_geometry() if geom is None else geom
    bound = _resolve_offset_max(grid_r, offset_max)
    centre = _grid_centre(grid_r)
    centres = np.asarray(grid_r.centers(), dtype=np.float64)
    allowed = box / 2.0 - bound
    mask = np.max(np.abs(centres - centre), axis=1) <= allowed
    candidates = np.where(mask)[0].astype(np.int64)
    if candidates.size == 0:
        raise ValueError("no candidate voxel inside the box")
    rng_place = np.random.default_rng(np.random.SeedSequence([seed_i, 0]))
    rng_amp = np.random.default_rng(np.random.SeedSequence([seed_i, 1]))
    points, drawn = offgrid_points(grid_r, count, rng_place, offset_max=bound, voxels=candidates)
    mag_db = rng_amp.uniform(0.0, dr_db, size=count)
    magnitude = 10.0 ** (-mag_db / 20.0)
    phase = 2.0 * np.pi * rng_amp.uniform(0.0, 1.0, size=count)
    rho = (magnitude * np.exp(1j * phase)).astype(np.complex128)
    y_clean, level, names, extra = _render_point_phantom(
        points, rho, geom_r, "bv", pattern, "L0c", seed_i, mismatch, jitter_std
    )
    meta: dict[str, Any] = {
        "box_size": float(box),
        "dynamic_range_db": float(dr_db),
        "voxels": np.asarray(drawn, dtype=np.int64),
    }
    meta.update(extra)
    gt = PhantomGT(
        level=level,
        space="bv",
        pattern=pattern,
        points_pos=np.asarray(points, dtype=np.float64),
        points_rho=np.asarray(rho, dtype=np.complex128),
        grid=grid_r,
        seed=seed_i,
        mismatch=names,
        meta=meta,
    )
    return Phantom(y_clean=np.asarray(y_clean, dtype=np.complex128), geom=geom_r, gt=gt)


def l0d_plate(
    seed: int = 0,
    *,
    center: Sequence[float] | None = None,
    normal: Sequence[float] | None = None,
    size: float = 4.0,
    sample_spacing: float | None = None,
    reflectivity: complex = 1.0,
    geom: CaptureGeometry | None = None,
    pattern: str = "tr38901",
) -> Phantom:
    """Generate the L0d square Born plate phantom.

    The plate is a uniform grid of BV atoms at ``center`` with ``lambda / 4``
    sampling (``sample_spacing`` override), in-plane axes ``e1``/``e2``,
    outward ``normal`` (default: horizontal, facing the first BS), and per-atom
    amplitude ``rho_s * step^2`` with ``rho_s = j sqrt(4 pi) reflectivity /
    lam``. Stationary-phase derivation: near the specular point, ``int e^{-j k
    (r1 + r2)} dA ~= -j lambda R1 R2 / ((R1 + R2) cos theta) e^{-j k (R1 +
    R2)}``, so a density ``rho_s`` of BV atoms (``lam / ((4 pi)^1.5 r1 r2)``
    each) sums to ``rho_s (-j lam^2) / ((4 pi)^1.5 (R1 + R2) cos theta)``;
    equating with the image source ``Gamma lam / (4 pi (R1 + R2))`` at normal
    incidence gives ``rho_s = j sqrt(4 pi) Gamma / lam``; at incidence
    ``theta`` the plate then returns ``Gamma / cos theta`` (the
    Born-on-specular error the level exists to expose). Meta keys: ``center``,
    ``normal``, ``axes``, ``size``, ``sample_spacing``, ``n_side``,
    ``reflectivity``, ``rho_surface``, ``vs_pos``.
    """
    seed_i = _check_seed(seed)
    geom_r = ring_geometry() if geom is None else geom
    centre = (
        np.asarray(DEFAULT_CENTER, dtype=np.float64)
        if center is None
        else np.asarray(center, dtype=np.float64).reshape(3)
    )
    if centre.shape != (3,) or not np.all(np.isfinite(centre)):
        raise ValueError("center must have 3 finite entries")
    size_f = float(size)
    if not np.isfinite(size_f) or size_f <= 0.0:
        raise ValueError("size must be finite and > 0")
    step = float(geom_r.wavelength / 4.0) if sample_spacing is None else float(sample_spacing)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("sample_spacing must be finite and > 0")
    if normal is None:
        to_bs = np.asarray(geom_r.bs_pos[0], dtype=np.float64) - centre
        to_bs[2] = 0.0
        proj = float(np.linalg.norm(to_bs))
        if proj == 0.0:
            normal_v = np.array([1.0, 0.0, 0.0])
        else:
            normal_v = to_bs / proj
    else:
        normal_v = np.asarray(normal, dtype=np.float64).reshape(3)
        if normal_v.shape != (3,) or not np.all(np.isfinite(normal_v)):
            raise ValueError("normal must have 3 finite entries")
        norm_n = float(np.linalg.norm(normal_v))
        if norm_n == 0.0:
            raise ValueError("normal must be non-zero")
        normal_v = normal_v / norm_n
    if abs(float(normal_v[2])) > 1.0 - 1e-12:
        e1 = np.array([1.0, 0.0, 0.0])
    else:
        e1_raw = np.cross(np.array([0.0, 0.0, 1.0]), normal_v)
        e1_norm = float(np.linalg.norm(e1_raw))
        if e1_norm == 0.0:
            raise ValueError("degenerate plate frame")
        e1 = e1_raw / e1_norm
    e2 = np.cross(normal_v, e1)
    e2_norm = float(np.linalg.norm(e2))
    if e2_norm == 0.0:
        raise ValueError("degenerate plate frame")
    e2 = e2 / e2_norm
    n_side = int(np.floor(size_f / step + 1e-9)) + 1
    offsets = (np.arange(n_side, dtype=np.float64) - (n_side - 1) / 2.0) * step
    off_i, off_j = np.meshgrid(offsets, offsets, indexing="ij")
    points = (
        centre[None, :] + off_i.reshape(-1, 1) * e1[None, :] + off_j.reshape(-1, 1) * e2[None, :]
    ).astype(np.float64)
    refl = complex(reflectivity)
    lam = float(geom_r.wavelength)
    rho_surface = 1j * np.sqrt(4.0 * np.pi) * refl / lam
    rho = np.full(points.shape[0], rho_surface * step**2, dtype=np.complex128)
    y_clean = atom_cfr(points, rho, geom_r, "bv", pattern=pattern)
    vs_pos = np.asarray(
        mirror_point(np.asarray(geom_r.bs_pos, dtype=np.float64), centre, normal_v),
        dtype=np.float64,
    )
    meta: dict[str, Any] = {
        "center": np.asarray(centre, dtype=np.float64),
        "normal": np.asarray(normal_v, dtype=np.float64),
        "axes": np.stack([e1, e2], axis=0).astype(np.float64),
        "size": float(size_f),
        "sample_spacing": float(step),
        "n_side": int(n_side),
        "reflectivity": complex(refl),
        "rho_surface": complex(rho_surface),
        "vs_pos": np.asarray(vs_pos, dtype=np.float64),
    }
    gt = PhantomGT(
        level="L0d",
        space="bv",
        pattern=pattern,
        points_pos=np.asarray(points, dtype=np.float64),
        points_rho=np.asarray(rho, dtype=np.complex128),
        grid=None,
        seed=seed_i,
        mismatch=(),
        meta=meta,
    )
    return Phantom(y_clean=np.asarray(y_clean, dtype=np.complex128), geom=geom_r, gt=gt)


def l0e_image_method(
    seed: int = 0,
    *,
    ground_material: str | None = "medium_dry_ground",
    ground_height: float = 0.0,
    walls: Sequence[tuple[Sequence[float], Sequence[float], str]] = DEFAULT_WALLS,
    geom: CaptureGeometry | None = None,
    pattern: str = "tr38901",
) -> Phantom:
    """Generate the L0e analytic image-method phantom.

    Scene: LoS plus first-order images of every BS (the ``ground_material``
    plane at ``ground_height`` plus the vertical ``walls``), with ITU-R P.2040
    Fresnel coefficients evaluated per view at the specular incidence angle.
    This is the scalar co-polar model (exact at grazing incidence): V-pol over
    horizontal ground is TM and V-pol on a vertical wall is TE. Images without
    a valid specular path keep zero amplitude but stay in the VS list. Meta
    keys: ``vs_bs``, ``vs_order``, ``vs_plane``, ``vs_visibility``,
    ``vs_theta_inc``, ``plane_points``, ``plane_normals``, ``plane_materials``
    and ``plane_eta``.
    """
    seed_i = _check_seed(seed)
    geom_r = ring_geometry() if geom is None else geom
    height = float(ground_height)
    if not np.isfinite(height):
        raise ValueError("ground_height must be finite")
    plane_points: list[np.ndarray] = []
    plane_normals: list[np.ndarray] = []
    plane_materials: list[str] = []
    is_ground: list[bool] = []
    if ground_material is not None:
        if ground_material not in ITU_MATERIALS:
            raise ValueError(f"unknown material {ground_material!r}")
        plane_points.append(np.array([0.0, 0.0, height], dtype=np.float64))
        plane_normals.append(np.array([0.0, 0.0, 1.0], dtype=np.float64))
        plane_materials.append(str(ground_material))
        is_ground.append(True)
    for entry in list(walls):
        point_s, normal_s, material = entry
        point = np.asarray(point_s, dtype=np.float64).reshape(3)
        if point.shape != (3,) or not np.all(np.isfinite(point)):
            raise ValueError("wall point must have 3 finite entries")
        normal = np.asarray(normal_s, dtype=np.float64).reshape(3)
        if normal.shape != (3,) or not np.all(np.isfinite(normal)):
            raise ValueError("wall normal must have 3 finite entries")
        norm_n = float(np.linalg.norm(normal))
        if norm_n == 0.0:
            raise ValueError("wall normal must be non-zero")
        unit = normal / norm_n
        if abs(float(unit[2])) > 1e-12:
            raise ValueError("wall normals must be horizontal")
        if material not in ITU_MATERIALS:
            raise ValueError(f"unknown material {material!r}")
        plane_points.append(point.astype(np.float64))
        plane_normals.append(unit.astype(np.float64))
        plane_materials.append(str(material))
        is_ground.append(False)
    num_planes = len(plane_points)
    plane_eta = np.asarray(
        [itu_permittivity(m, float(geom_r.f_c)) for m in plane_materials],
        dtype=np.complex128,
    )
    num_views = geom_r.num_views
    num_bs = geom_r.num_bs
    per_bs = 1 + num_planes
    num_points = num_bs * per_bs
    bs_array = np.asarray(geom_r.bs_pos, dtype=np.float64)
    ue_array = np.asarray(geom_r.ue_pos, dtype=np.float64)
    points_list: list[np.ndarray] = []
    vs_bs = np.zeros(num_points, dtype=np.int64)
    vs_order = np.zeros(num_points, dtype=np.int64)
    vs_plane = np.full(num_points, -1, dtype=np.int64)
    visibility = np.zeros((num_points, num_views), dtype=bool)
    theta_inc = np.full((num_points, num_views), np.nan, dtype=np.float64)
    rho = np.zeros((num_points, num_views, num_bs), dtype=np.complex128)
    for b in range(num_bs):
        tx = bs_array[b]
        base = b * per_bs
        points_list.append(tx.copy())
        vs_bs[base] = b
        vs_order[base] = 0
        vs_plane[base] = -1
        visibility[base, :] = True
        rho[base, :, b] = 1.0
        for j in range(num_planes):
            key = base + 1 + j
            image = np.asarray(
                mirror_point(tx, plane_points[j], plane_normals[j]), dtype=np.float64
            ).reshape(3)
            points_list.append(image)
            vs_bs[key] = b
            vs_order[key] = 1
            vs_plane[key] = j
            normal_j = plane_normals[j]
            for v in range(num_views):
                _, valid = single_bounce_point(tx, ue_array[v], plane_points[j], normal_j)
                valid_b = bool(np.asarray(valid).reshape(-1)[0])
                if not valid_b:
                    continue
                diff = ue_array[v] - image
                dist = float(np.linalg.norm(diff))
                if dist == 0.0:
                    continue
                cos_t = abs(float(diff @ normal_j)) / dist
                cos_t = min(max(cos_t, 0.0), 1.0)
                theta_inc[key, v] = float(np.arccos(cos_t))
                visibility[key, v] = True
                r_te, r_tm = fresnel_coefficients(cos_t, complex(plane_eta[j]))
                beta = (
                    complex(np.asarray(r_tm).reshape(-1)[0])
                    if is_ground[j]
                    else complex(np.asarray(r_te).reshape(-1)[0])
                )
                rho[key, v, b] = beta
    points_pos = np.stack(points_list, axis=0).astype(np.float64)
    y_clean = atom_cfr(points_pos, rho, geom_r, "vs", pattern=pattern)
    meta: dict[str, Any] = {
        "vs_bs": np.asarray(vs_bs, dtype=np.int64),
        "vs_order": np.asarray(vs_order, dtype=np.int64),
        "vs_plane": np.asarray(vs_plane, dtype=np.int64),
        "vs_visibility": np.asarray(visibility, dtype=bool),
        "vs_theta_inc": np.asarray(theta_inc, dtype=np.float64),
        "plane_points": np.stack(plane_points, axis=0).astype(np.float64)
        if num_planes
        else np.zeros((0, 3), dtype=np.float64),
        "plane_normals": np.stack(plane_normals, axis=0).astype(np.float64)
        if num_planes
        else np.zeros((0, 3), dtype=np.float64),
        "plane_materials": list(plane_materials),
        "plane_eta": np.asarray(plane_eta, dtype=np.complex128),
    }
    gt = PhantomGT(
        level="L0e",
        space="vs",
        pattern=pattern,
        points_pos=np.asarray(points_pos, dtype=np.float64),
        points_rho=np.asarray(rho, dtype=np.complex128),
        grid=None,
        seed=seed_i,
        mismatch=(),
        meta=meta,
    )
    return Phantom(y_clean=np.asarray(y_clean, dtype=np.complex128), geom=geom_r, gt=gt)


def l0_mm(
    seed: int,
    *,
    base: str = "L0a",
    mismatch: Sequence[str] = MISMATCHES,
    jitter_std: float | None = None,
    **kwargs: Any,
) -> Phantom:
    """Render an L0a/L0c scene with model mismatch (the L0-mm entry point).

    Scene: the ``base`` (``L0a``/``L0c``) point layout and amplitudes, bitwise
    identical to the plain generator; only the rendering uses the mismatched
    array (``spherical`` wavefront, ``squint`` and/or ``element_jitter``). The
    jittered array is shared by all captures while the returned geometry stays
    nominal. RNG streams are ``SeedSequence([seed, 0])`` (placement),
    ``SeedSequence([seed, 1])`` (amplitudes) and ``SeedSequence([seed, 2])``
    (jitter). Meta keys: those of the base plus ``base``,
    ``elem_offsets_true``, ``jitter_std``, ``nmse_floor_pred`` and
    ``nmse_floor_pred_db``.
    """
    seed_i = _check_seed(seed)
    names = _normalise_mismatch(tuple(mismatch))
    if len(names) == 0:
        raise ValueError("mismatch must be non-empty")
    if base == "L0a":
        return l0a_point(seed_i, mismatch=names, jitter_std=jitter_std, **kwargs)
    if base == "L0c":
        return l0c_random(seed_i, mismatch=names, jitter_std=jitter_std, **kwargs)
    raise ValueError(f"unknown base {base!r}")


def generate(level: str, seed: int, **kwargs: Any) -> Phantom:
    """Dispatch a level name to its generator."""
    if level == "L0a":
        return l0a_point(seed, **kwargs)
    if level == "L0b":
        return l0b_pair(seed, **kwargs)
    if level == "L0c":
        return l0c_random(seed, **kwargs)
    if level == "L0d":
        return l0d_plate(seed, **kwargs)
    if level == "L0e":
        return l0e_image_method(seed, **kwargs)
    if level == "L0-mm":
        return l0_mm(seed, **kwargs)
    raise ValueError(f"unknown level {level!r}")
