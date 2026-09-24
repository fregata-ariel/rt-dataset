"""Geometry shared by the tomography operators.

Voxel grids, capture poses, bistatic and virtual-source delays and directions,
and planar mirrors. NumPy only: nothing in this module may import Sionna,
Mitsuba or Dr.Jit. All lengths are metres, angles radians, frequencies hertz.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.constants import c as SPEED_OF_LIGHT

from plateau_rt.domain.rf_camera.calibration import rotation_matrix
from plateau_rt.domain.rf_camera.imaging import frequency_offsets, uniform_frequency_spacing
from plateau_rt.domain.rf_tomography.antenna import bs_orientation

LOS_VS_TOLERANCE_M: float = 1e-9


def _readonly_float64(value: np.ndarray) -> np.ndarray:
    """Return a read-only float64 copy of ``value``."""
    array = np.array(value, dtype=np.float64, copy=True)
    array.setflags(write=False)
    return array


def _as_points(value: np.ndarray) -> np.ndarray:
    """Return ``[P, 3]`` float64 points, promoting a single ``[3]`` point."""
    points = np.asarray(value, dtype=np.float64)
    if points.ndim == 1:
        if points.shape[0] != 3:
            raise ValueError("points must have shape [P, 3] or [3]")
        points = points[None, :]
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape [P, 3] or [3]")
    return points


@dataclass(frozen=True)
class VoxelGrid:
    """Uniform axis-aligned voxel grid.

    The flat index is ``np.ravel_multi_index((ix, iy, iz), shape)`` (C order, z
    fastest); ``origin`` is the centre of voxel ``(0, 0, 0)``.
    """

    origin: np.ndarray
    spacing: float
    shape: tuple[int, int, int]

    def __post_init__(self) -> None:
        origin = np.array(self.origin, dtype=np.float64, copy=True)
        if origin.shape != (3,):
            raise ValueError("origin must have shape (3,)")
        if not np.all(np.isfinite(origin)):
            raise ValueError("origin must contain only finite values")
        origin.setflags(write=False)

        spacing = float(self.spacing)
        if not np.isfinite(spacing) or spacing <= 0.0:
            raise ValueError("spacing must be finite and > 0")

        try:
            shape = tuple(int(value) for value in self.shape)
        except TypeError as error:
            raise ValueError("shape must be an iterable of three integers") from error
        if len(shape) != 3 or any(value < 1 for value in shape):
            raise ValueError("shape must have three entries, each >= 1")

        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "spacing", spacing)
        object.__setattr__(self, "shape", shape)

    @property
    def size(self) -> int:
        """Total number of voxels."""
        nx, ny, nz = self.shape
        return int(nx * ny * nz)

    def axes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return the centre coordinate array of each axis."""
        return (
            self.origin[0] + self.spacing * np.arange(self.shape[0]),
            self.origin[1] + self.spacing * np.arange(self.shape[1]),
            self.origin[2] + self.spacing * np.arange(self.shape[2]),
        )

    def centers(self) -> np.ndarray:
        """Return all voxel centres as ``[P, 3]`` row-major in ``(ix, iy, iz)``."""
        grids = np.meshgrid(*self.axes(), indexing="ij")
        return np.stack(grids, axis=-1).reshape(-1, 3)

    def index(self, points: np.ndarray) -> np.ndarray:
        """Return the nearest-voxel flat index ``[P]`` (``-1`` outside the grid).

        Coordinates are rounded to the nearest centre; ties go to the higher
        index. Non-finite points are reported as outside (``-1``).
        """
        pts = _as_points(points).reshape(-1, 3)
        nx, ny, nz = self.shape
        scaled = (pts - self.origin) / self.spacing + 0.5
        ix, iy, iz = np.floor(scaled[:, 0]), np.floor(scaled[:, 1]), np.floor(scaled[:, 2])
        inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (iz >= 0) & (iz < nz)
        result = np.full(pts.shape[0], -1, dtype=np.int64)
        if np.any(inside):
            result[inside] = np.ravel_multi_index(
                (
                    ix[inside].astype(np.int64),
                    iy[inside].astype(np.int64),
                    iz[inside].astype(np.int64),
                ),
                self.shape,
            )
        return result

    @classmethod
    def from_bounds(cls, lower: np.ndarray, upper: np.ndarray, spacing: float) -> VoxelGrid:
        """Build a grid whose centres start at ``lower`` and reach ``upper``.

        The number of voxels per axis is ``floor((upper - lower) / spacing + eps)
        + 1`` so ``upper`` is included when the extent is a multiple of
        ``spacing``.
        """
        lower = np.asarray(lower, dtype=np.float64)
        upper = np.asarray(upper, dtype=np.float64)
        if lower.shape != (3,) or upper.shape != (3,):
            raise ValueError("lower and upper must have shape (3,)")
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise ValueError("lower and upper must contain only finite values")
        spacing = float(spacing)
        if not np.isfinite(spacing) or spacing <= 0.0:
            raise ValueError("spacing must be finite and > 0")
        if np.any(upper < lower):
            raise ValueError("upper must be >= lower on every axis")
        shape = (
            int(np.floor((upper[0] - lower[0]) / spacing + 1e-9)) + 1,
            int(np.floor((upper[1] - lower[1]) / spacing + 1e-9)) + 1,
            int(np.floor((upper[2] - lower[2]) / spacing + 1e-9)) + 1,
        )
        return cls(origin=lower, spacing=spacing, shape=shape)


def planar_element_offsets(
    wavelength: float, *, rows: int = 8, cols: int = 8, spacing_lambda: float = 0.5
) -> np.ndarray:
    """Return the local receive-aperture element offsets ``[rows * cols, 3]``.

    Element ``m = r * cols + col`` sits at
    ``(0, d * (col - (cols - 1) / 2), d * ((rows - 1) / 2 - r))`` with
    ``d = spacing_lambda * wavelength``: the row axis runs top to bottom, so z
    decreases with the row, and ``m`` is the C-order flattening of
    ``Y[..., r, col, :]``.

    Derivation: Sionna's ``PlanarArray`` places antenna ``a`` at
    ``y = d * (jj - (C - 1) / 2)``, ``z = d * ((R - 1) / 2 - ii)`` with column-first
    numbering ``ii = a % R``, ``jj = a // R``; ``imaging.reshape_planar_column_first``
    stores antenna ``a`` at ``[row = a % R, col = a // R]``. Row 0 is therefore the
    top (+z), which is why ``calibration.calibrate_angular_cfr`` flips the row axis.
    The synthetic array applies ``exp(+j k_c u . q_m)`` at the carrier, with ``u``
    the local unit vector toward the source. Verified against Sionna 2.0.1
    (``tests/fixtures/rf_tomography/sionna_los_aperture.npz``).
    """
    if rows < 1 or cols < 1:
        raise ValueError("rows and cols must be >= 1")
    if not np.isfinite(wavelength) or wavelength <= 0.0:
        raise ValueError("wavelength must be finite and > 0")
    if not np.isfinite(spacing_lambda) or spacing_lambda <= 0.0:
        raise ValueError("spacing_lambda must be finite and > 0")

    spacing = spacing_lambda * wavelength
    row = np.arange(rows, dtype=np.float64)[:, None]
    col = np.arange(cols, dtype=np.float64)[None, :]
    offsets = np.zeros((rows, cols, 3), dtype=np.float64)
    offsets[..., 1] = spacing * (col - (cols - 1) / 2.0)
    offsets[..., 2] = spacing * ((rows - 1) / 2.0 - row)
    return offsets.reshape(rows * cols, 3)


def rotations_from_orientations(orientations: np.ndarray) -> np.ndarray:
    """Return ``world_from_local`` rotations ``[V, 3, 3]`` for Sionna Euler triples."""
    orientations = np.asarray(orientations, dtype=np.float64)
    if orientations.ndim != 2 or orientations.shape[1] != 3:
        raise ValueError("orientations must have shape [V, 3]")
    return np.stack([rotation_matrix(tuple(row)) for row in orientations], axis=0)


def _bs_rotations(bs_pos: np.ndarray, bs_look_at: np.ndarray | None) -> np.ndarray | None:
    """Return look-at BS rotations ``[B, 3, 3]`` or ``None`` when no target is given."""
    if bs_look_at is None:
        return None
    positions = np.asarray(bs_pos, dtype=np.float64)
    targets = np.asarray(bs_look_at, dtype=np.float64)
    num_bs = positions.shape[0] if positions.ndim == 2 else 0
    if targets.shape == (3,):
        targets = np.broadcast_to(targets, (num_bs, 3))
    if targets.shape != (num_bs, 3):
        raise ValueError("bs_look_at must have shape [3] or [B, 3]")
    return np.stack([bs_orientation(positions[b], targets[b]) for b in range(num_bs)], axis=0)


def hemisphere_index(u_local: np.ndarray) -> np.ndarray:
    """Return 0 for the front hemisphere (``u_x >= 0``) and 1 for the back."""
    u_local = np.asarray(u_local, dtype=np.float64)
    if u_local.ndim < 1 or u_local.shape[-1] != 3:
        raise ValueError("u_local must have shape [..., 3]")
    return (u_local[..., 0] < 0.0).astype(np.int64)


def mirror_point(
    points: np.ndarray, plane_point: np.ndarray, plane_normal: np.ndarray
) -> np.ndarray:
    """Mirror ``[..., 3]`` points across the plane ``n . (x - p0) = 0``.

    The normal is normalised and may have either sign; a zero normal is invalid.
    """
    pts = np.asarray(points, dtype=np.float64)
    plane_point = np.asarray(plane_point, dtype=np.float64)
    plane_normal = np.asarray(plane_normal, dtype=np.float64)
    if pts.ndim < 1 or pts.shape[-1] != 3:
        raise ValueError("points must have shape [..., 3]")
    if plane_point.shape != (3,) or plane_normal.shape != (3,):
        raise ValueError("plane_point and plane_normal must have shape (3,)")
    normal_length = float(np.linalg.norm(plane_normal))
    if normal_length == 0.0:
        raise ValueError("plane_normal must be non-zero")
    normal = plane_normal / normal_length
    signed_distance = np.einsum("...i,i->...", pts - plane_point, normal)
    return pts - 2.0 * signed_distance[..., None] * normal


def single_bounce_point(
    tx: np.ndarray, rx: np.ndarray, plane_point: np.ndarray, plane_normal: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the specular point of the path ``tx -> plane -> rx``.

    The point is where the segment from ``rx`` to the mirror image of ``tx``
    crosses the plane. ``valid`` is True only when ``tx`` and ``rx`` lie strictly
    on the same side; invalid entries are NaN.
    """
    tx = np.asarray(tx, dtype=np.float64)
    rx = np.asarray(rx, dtype=np.float64)
    plane_point = np.asarray(plane_point, dtype=np.float64)
    plane_normal = np.asarray(plane_normal, dtype=np.float64)
    if tx.ndim < 1 or tx.shape[-1] != 3 or rx.ndim < 1 or rx.shape[-1] != 3:
        raise ValueError("tx and rx must have shape [..., 3]")
    if plane_point.shape != (3,) or plane_normal.shape != (3,):
        raise ValueError("plane_point and plane_normal must have shape (3,)")
    normal_length = float(np.linalg.norm(plane_normal))
    if normal_length == 0.0:
        raise ValueError("plane_normal must be non-zero")
    normal = plane_normal / normal_length

    image = mirror_point(tx, plane_point, normal)
    segment = image - rx
    numerator = np.einsum("...i,i->...", plane_point - rx, normal)
    denominator = np.einsum("...i,i->...", segment, normal)
    with np.errstate(divide="ignore", invalid="ignore"):
        fraction = numerator / denominator
    points = rx + fraction[..., None] * segment

    tx_side = np.einsum("...i,i->...", tx - plane_point, normal)
    rx_side = np.einsum("...i,i->...", rx - plane_point, normal)
    valid = (tx_side * rx_side) > 0.0
    points = np.where(valid[..., None], points, np.nan)
    return points, valid


@dataclass(frozen=True)
class CaptureGeometry:
    """A pose bank of UE apertures and base stations.

    ``ue_rot[v]`` is ``world_from_local`` (columns are the local x, y, z axes).
    ``freq_offsets`` is the baseband grid with DC at index ``N // 2`` and
    ``elem_offsets[m]`` uses the row-major element order ``m = r * cols + col``.
    ``bs_rot`` holds the optional ``world_from_local`` orientation of each BS.
    """

    ue_pos: np.ndarray
    ue_rot: np.ndarray
    bs_pos: np.ndarray
    elem_offsets: np.ndarray
    freq_offsets: np.ndarray
    f_c: float
    aperture_shape: tuple[int, int] = (8, 8)
    bs_rot: np.ndarray | None = None

    def __post_init__(self) -> None:
        ue_pos = _readonly_float64(self.ue_pos)
        ue_rot = _readonly_float64(self.ue_rot)
        bs_pos = _readonly_float64(self.bs_pos)
        elem_offsets = _readonly_float64(self.elem_offsets)
        freq_offsets = _readonly_float64(self.freq_offsets)

        if ue_pos.ndim != 2 or ue_pos.shape[1] != 3 or ue_pos.shape[0] < 1:
            raise ValueError("ue_pos must have shape [V, 3] with V >= 1")
        if not np.all(np.isfinite(ue_pos)):
            raise ValueError("ue_pos must contain only finite values")
        if ue_rot.shape != (ue_pos.shape[0], 3, 3):
            raise ValueError("ue_rot must have shape [V, 3, 3]")
        if not np.all(np.isfinite(ue_rot)):
            raise ValueError("ue_rot must contain only finite values")
        gram = np.einsum("vji,vjk->vik", ue_rot, ue_rot)
        if np.max(np.abs(gram - np.eye(3))) > 1e-9 or np.any(np.linalg.det(ue_rot) <= 0.0):
            raise ValueError("each ue_rot[v] must be a proper rotation")
        if bs_pos.ndim != 2 or bs_pos.shape[1] != 3 or bs_pos.shape[0] < 1:
            raise ValueError("bs_pos must have shape [B, 3] with B >= 1")
        if not np.all(np.isfinite(bs_pos)):
            raise ValueError("bs_pos must contain only finite values")
        if elem_offsets.ndim != 2 or elem_offsets.shape[1] != 3 or elem_offsets.shape[0] < 1:
            raise ValueError("elem_offsets must have shape [M, 3] with M >= 1")
        if not np.all(np.isfinite(elem_offsets)):
            raise ValueError("elem_offsets must contain only finite values")
        if freq_offsets.ndim != 1 or freq_offsets.size < 2:
            raise ValueError("freq_offsets must have shape [N] with N >= 2")
        if not np.all(np.isfinite(freq_offsets)):
            raise ValueError("freq_offsets must contain only finite values")
        delta_f = uniform_frequency_spacing(freq_offsets)
        if abs(freq_offsets[freq_offsets.size // 2]) > 1e-6 * delta_f:
            raise ValueError("the zero frequency bin must be at index N // 2")

        try:
            aperture_shape = tuple(int(value) for value in self.aperture_shape)
        except TypeError as error:
            raise ValueError("aperture_shape must be a pair of integers") from error
        if len(aperture_shape) != 2 or any(value < 1 for value in aperture_shape):
            raise ValueError("aperture_shape must be a pair of integers >= 1")
        if aperture_shape[0] * aperture_shape[1] != elem_offsets.shape[0]:
            raise ValueError("aperture_shape must match the number of element offsets")

        f_c = float(self.f_c)
        if not np.isfinite(f_c) or f_c <= 0.0:
            raise ValueError("f_c must be finite and > 0")

        bs_rot: np.ndarray | None = None
        if self.bs_rot is not None:
            bs_rot_array = _readonly_float64(self.bs_rot)
            if bs_rot_array.shape != (bs_pos.shape[0], 3, 3):
                raise ValueError("bs_rot must have shape [B, 3, 3]")
            if not np.all(np.isfinite(bs_rot_array)):
                raise ValueError("bs_rot must contain only finite values")
            gram_bs = np.einsum("bji,bjk->bik", bs_rot_array, bs_rot_array)
            if np.max(np.abs(gram_bs - np.eye(3))) > 1e-9 or np.any(
                np.linalg.det(bs_rot_array) <= 0.0
            ):
                raise ValueError("each bs_rot[b] must be a proper rotation")
            bs_rot = bs_rot_array

        object.__setattr__(self, "ue_pos", ue_pos)
        object.__setattr__(self, "ue_rot", ue_rot)
        object.__setattr__(self, "bs_pos", bs_pos)
        object.__setattr__(self, "elem_offsets", elem_offsets)
        object.__setattr__(self, "freq_offsets", freq_offsets)
        object.__setattr__(self, "f_c", f_c)
        object.__setattr__(self, "aperture_shape", aperture_shape)
        object.__setattr__(self, "bs_rot", bs_rot)

    @classmethod
    def from_orientations(
        cls,
        ue_pos: np.ndarray,
        ue_orientations: np.ndarray,
        bs_pos: np.ndarray,
        *,
        f_c: float,
        bandwidth: float,
        num_bins: int,
        aperture_shape: tuple[int, int] = (8, 8),
        spacing_lambda: float = 0.5,
        bs_look_at: np.ndarray | None = None,
    ) -> CaptureGeometry:
        """Build a geometry from Sionna Euler orientations and a frequency grid.

        The frequency offsets reuse the float32 grid traced by Sionna, promoted
        to float64. When ``bs_look_at`` is given (``[3]`` shared by all BSs or
        ``[B, 3]`` per BS), each BS orientation is the look-at rotation.
        """
        try:
            rows, cols = (int(value) for value in aperture_shape)
        except (TypeError, ValueError) as error:
            raise ValueError("aperture_shape must be a pair of integers") from error
        f_c = float(f_c)
        if not np.isfinite(f_c) or f_c <= 0.0:
            raise ValueError("f_c must be finite and > 0")
        wavelength = SPEED_OF_LIGHT / f_c
        ue_rot = rotations_from_orientations(ue_orientations)
        elem_offsets = planar_element_offsets(
            wavelength, rows=rows, cols=cols, spacing_lambda=spacing_lambda
        )
        freq = frequency_offsets(bandwidth, num_bins).astype(np.float64)
        bs_rot = _bs_rotations(bs_pos, bs_look_at)
        return cls(
            ue_pos=ue_pos,
            ue_rot=ue_rot,
            bs_pos=bs_pos,
            elem_offsets=elem_offsets,
            freq_offsets=freq,
            f_c=f_c,
            aperture_shape=(rows, cols),
            bs_rot=bs_rot,
        )

    def select(
        self,
        views: Sequence[int] | np.ndarray | None = None,
        bss: Sequence[int] | np.ndarray | None = None,
    ) -> CaptureGeometry:
        """Return a geometry keeping only the given view and BS indices in order."""
        view_indices = np.arange(self.num_views) if views is None else np.asarray(views)
        bs_indices = np.arange(self.num_bs) if bss is None else np.asarray(bss)
        return CaptureGeometry(
            ue_pos=self.ue_pos[view_indices],
            ue_rot=self.ue_rot[view_indices],
            bs_pos=self.bs_pos[bs_indices],
            elem_offsets=self.elem_offsets,
            freq_offsets=self.freq_offsets,
            f_c=self.f_c,
            aperture_shape=self.aperture_shape,
            bs_rot=None if self.bs_rot is None else self.bs_rot[bs_indices],
        )

    @property
    def num_views(self) -> int:
        """Number of UE views ``V``."""
        return int(self.ue_pos.shape[0])

    @property
    def num_bs(self) -> int:
        """Number of base stations ``B``."""
        return int(self.bs_pos.shape[0])

    @property
    def num_elements(self) -> int:
        """Number of receive-aperture elements ``M``."""
        return int(self.elem_offsets.shape[0])

    @property
    def num_bins(self) -> int:
        """Number of frequency bins ``N``."""
        return int(self.freq_offsets.shape[0])

    @property
    def wavelength(self) -> float:
        """Carrier wavelength ``c / f_c``."""
        return SPEED_OF_LIGHT / self.f_c

    @property
    def wavenumber(self) -> float:
        """Carrier wavenumber ``2 pi / wavelength``."""
        return 2.0 * np.pi / self.wavelength

    @property
    def delta_f(self) -> float:
        """Uniform frequency-grid spacing."""
        return uniform_frequency_spacing(self.freq_offsets)

    @property
    def bandwidth(self) -> float:
        """Sampled bandwidth ``N * delta_f``."""
        return self.num_bins * self.delta_f

    @property
    def delay_period(self) -> float:
        """Unambiguous delay ``1 / delta_f`` (equivalently ``N / B``)."""
        return 1.0 / self.delta_f

    def local_direction(self, x: np.ndarray, v: int) -> np.ndarray:
        """Return the UE-local unit vector from view ``v`` toward ``x`` [P, 3]."""
        points = _as_points(x)
        relative = points - self.ue_pos[v]
        distance = np.linalg.norm(relative, axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            unit_world = relative / distance[..., None]
        return unit_world @ self.ue_rot[v]

    def bistatic_ranges(self, x: np.ndarray, v: int, b: int) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(|x - t_b|, |x - p_v|)`` [P]."""
        points = _as_points(x)
        r1 = np.linalg.norm(points - self.bs_pos[b], axis=-1)
        r2 = np.linalg.norm(points - self.ue_pos[v], axis=-1)
        return r1, r2

    def bistatic_delay(self, x: np.ndarray, v: int, b: int) -> np.ndarray:
        """Return the bistatic delay ``(|x - t_b| + |x - p_v|) / c`` [P]."""
        r1, r2 = self.bistatic_ranges(x, v, b)
        return (r1 + r2) / SPEED_OF_LIGHT

    def vs_delay(self, s: np.ndarray, v: int) -> np.ndarray:
        """Return the virtual-source delay ``|s - p_v| / c`` [P]."""
        points = _as_points(s)
        return np.linalg.norm(points - self.ue_pos[v], axis=-1) / SPEED_OF_LIGHT

    def vs_departure_dir(self, s: np.ndarray, v: int, b: int) -> np.ndarray:
        """Return the world-frame departure direction at BS ``b`` [P, 3].

        For a first-order image the direction is the Householder reflection of
        the propagation direction at the UE about the BS-to-source normal; when
        ``|s - t_b| <= LOS_VS_TOLERANCE_M`` the virtual source is the BS itself
        (LoS) and the departure direction is that same direction.
        """
        points = _as_points(s)
        to_ue = self.ue_pos[v] - points
        range_ue = np.linalg.norm(to_ue, axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            direction = to_ue / range_ue[..., None]

        to_bs = points - self.bs_pos[b]
        range_bs = np.linalg.norm(to_bs, axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            normal = to_bs / range_bs[..., None]

        alignment = np.einsum("...i,...i->...", direction, normal)
        reflected = direction - 2.0 * alignment[..., None] * normal
        los = range_bs <= LOS_VS_TOLERANCE_M
        return np.where(los[..., None], direction, reflected)
