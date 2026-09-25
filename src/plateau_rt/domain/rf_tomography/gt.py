"""Tomography ground truth from path-level Sionna ground truth (design §6.1, T17).

NumPy/SciPy only: virtual-source clustering with effective amplitudes, LoS
model errors, mechanism labels, delay-wrap flags, reflection planes and mesh
surface sampling with raycast observability.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.constants import c as SPEED_OF_LIGHT
from scipy.spatial import cKDTree

from plateau_rt.domain.ground import GROUND_PLANE_ID
from plateau_rt.domain.rf_camera.image_sources import (
    arrival_unit_vectors,
    virtual_source_positions,
)
from plateau_rt.domain.rf_tomography.antenna import PATTERN_KINDS, bs_pattern
from plateau_rt.domain.rf_tomography.forward_exact import POLARIZATIONS, capture_factors
from plateau_rt.domain.rf_tomography.forward_sep import incidence_cosine
from plateau_rt.domain.rf_tomography.geometry import CaptureGeometry, hemisphere_index

GT_SCHEMA = "rf_tomo_gt/1"
INTERACTION_SPECULAR, INTERACTION_DIFFUSE, INTERACTION_REFRACTION, INTERACTION_DIFFRACTION = (
    1,
    2,
    4,
    8,
)
PATH_TYPE_NAMES: tuple[str, ...] = ("los", "specular", "refraction", "diffraction", "diffuse")
PATH_TYPE_INVALID = -1
DEFAULT_CLUSTER_TOL_M = 1e-2
DEFAULT_PLANE_ANGLE_TOL = 1e-3  # |n_i - n_k| (unit normals)
DEFAULT_PLANE_OFFSET_TOL_M = 5e-3
DEFAULT_SURFACE_SPACING_M = 0.25
DEFAULT_SUPPORT_RADIUS_M = 0.5
SURFACE_OFFSET_M = 1e-3
DEDUP_DECIMALS = 6

_LOS, _SPECULAR, _REFRACTION, _DIFFRACTION, _DIFFUSE = 0, 1, 2, 3, 4
_VS_CANDIDATE_TYPES = (_LOS, _SPECULAR, _REFRACTION)

# GT arrays by the axis a BS subset slices (``select_bs``).
GT_CAPTURE_KEYS: tuple[str, ...] = (
    "path_type",
    "path_power",
    "path_vs",
    "beyond_period",
    "los_visible",
    "ground_bounce_visible",
    "los_phase_model_error",
    "los_amp_model_error_db",
)
GT_VS_KEYS: tuple[str, ...] = (
    "vs_pos",
    "vs_bs",
    "vs_order",
    "vs_num_paths",
    "vs_objects",
    "vs_plane_ids",
    "vs_spread",
    "vs_visibility",
    "vs_power",
    "vs_rho_eff",
    "vs_theta_inc",
    "vs_path_type",
)
GT_INTERACTION_KEYS: tuple[str, ...] = (
    "interaction_points",
    "interaction_view",
    "interaction_bs",
    "interaction_path",
    "interaction_depth",
    "interaction_type",
    "interaction_object",
    "interaction_plane",
)


def _validate_pattern(pattern: str) -> None:
    """Raise ``ValueError`` unless ``pattern`` is a known antenna pattern."""
    if pattern not in PATTERN_KINDS:
        raise ValueError(f"pattern must be one of {PATTERN_KINDS}, got {pattern!r}")


def _validate_polarization(polarization: str) -> None:
    """Raise ``ValueError`` unless ``polarization`` is a known forward model."""
    if polarization not in POLARIZATIONS:
        raise ValueError(f"polarization must be one of {POLARIZATIONS}, got {polarization!r}")


@dataclass(frozen=True)
class PathGT:
    """Path-level ground truth arrays in canonical ``[V, B, P, ...]`` order."""

    valid: np.ndarray  # [V,B,P] bool
    tau: np.ndarray  # [V,B,P] float64
    theta_t: np.ndarray  # [V,B,P] float64
    phi_t: np.ndarray  # [V,B,P] float64
    theta_r: np.ndarray  # [V,B,P] float64
    phi_r: np.ndarray  # [V,B,P] float64
    a_baseband: np.ndarray  # [V,B,2,R,C,P] complex128
    interactions: np.ndarray  # [V,B,P,D] int64
    object_index: np.ndarray  # [V,B,P,D] int64
    vertices: np.ndarray  # [V,B,P,D,3] float64
    object_names: tuple[str, ...]

    @classmethod
    def from_arrays(cls, arrays: Mapping[str, np.ndarray], object_names: Sequence[str]) -> PathGT:
        """Build a :class:`PathGT` from stored arrays, casting every field.

        Missing ``interactions`` / ``object_index`` / ``vertices`` default to
        ``D = 0``. Raises ``ValueError`` on inconsistent shapes.
        """
        if "valid" not in arrays:
            raise ValueError("arrays must contain 'valid'")
        valid = np.asarray(arrays["valid"], dtype=bool)
        if valid.ndim != 3:
            raise ValueError(f"valid must have shape [V, B, P], got {valid.shape}")
        num_views, num_bs, num_paths = valid.shape

        def _flat(name: str) -> np.ndarray:
            value = np.array(arrays[name], dtype=np.float64, copy=True)
            if value.shape != (num_views, num_bs, num_paths):
                raise ValueError(
                    f"{name} must have shape {(num_views, num_bs, num_paths)}, got {value.shape}"
                )
            return value

        tau = _flat("tau")
        theta_t = _flat("theta_t")
        phi_t = _flat("phi_t")
        theta_r = _flat("theta_r")
        phi_r = _flat("phi_r")
        a_baseband = np.array(arrays["a_baseband"], dtype=np.complex128, copy=True)
        if (
            a_baseband.ndim != 6
            or a_baseband.shape[0] != num_views
            or a_baseband.shape[1] != num_bs
            or a_baseband.shape[2] != 2
            or a_baseband.shape[5] != num_paths
        ):
            raise ValueError(
                "a_baseband must have shape "
                f"[{num_views}, {num_bs}, 2, R, C, {num_paths}], got {a_baseband.shape}"
            )

        has_depth = "interactions" in arrays
        if has_depth:
            interactions = np.array(arrays["interactions"], dtype=np.int64, copy=True)
            if interactions.ndim != 4 or interactions.shape[:3] != valid.shape:
                raise ValueError(
                    "interactions must have shape "
                    f"[{num_views}, {num_bs}, {num_paths}, D], got {interactions.shape}"
                )
            depth = interactions.shape[3]
            object_index = np.array(arrays["object_index"], dtype=np.int64, copy=True)
            if object_index.shape != (num_views, num_bs, num_paths, depth):
                raise ValueError(
                    f"object_index must have shape {interactions.shape}, got {object_index.shape}"
                )
            vertices = np.array(arrays["vertices"], dtype=np.float64, copy=True)
            if vertices.shape != (num_views, num_bs, num_paths, depth, 3):
                raise ValueError(
                    "vertices must have shape "
                    f"{(num_views, num_bs, num_paths, depth, 3)}, got {vertices.shape}"
                )
        else:
            interactions = np.zeros((num_views, num_bs, num_paths, 0), dtype=np.int64)
            object_index = np.zeros((num_views, num_bs, num_paths, 0), dtype=np.int64)
            vertices = np.zeros((num_views, num_bs, num_paths, 0, 3), dtype=np.float64)

        return cls(
            valid=valid,
            tau=tau,
            theta_t=theta_t,
            phi_t=phi_t,
            theta_r=theta_r,
            phi_r=phi_r,
            a_baseband=a_baseband,
            interactions=interactions,
            object_index=object_index,
            vertices=vertices,
            object_names=tuple(str(name) for name in object_names),
        )

    @property
    def num_views(self) -> int:
        """Number of UE views ``V``."""
        return int(self.valid.shape[0])

    @property
    def num_bs(self) -> int:
        """Number of base stations ``B``."""
        return int(self.valid.shape[1])

    @property
    def num_paths(self) -> int:
        """Number of path slots ``P``."""
        return int(self.valid.shape[2])

    @property
    def max_depth(self) -> int:
        """Maximum stored interaction depth ``D``."""
        return int(self.interactions.shape[3])

    @property
    def num_interactions(self) -> np.ndarray:
        """Number of non-zero interactions per path, shape ``[V,B,P]`` int64."""
        return np.count_nonzero(self.interactions, axis=-1).astype(np.int64)


def path_types(path: PathGT) -> np.ndarray:
    """Return the mechanism code ``[V,B,P]`` int8 (``-1`` invalid, codes index names)."""
    valid = path.valid
    interactions = path.interactions
    diffuse = np.any((interactions & INTERACTION_DIFFUSE) != 0, axis=-1)
    diffraction = np.any((interactions & INTERACTION_DIFFRACTION) != 0, axis=-1)
    refraction = np.any((interactions & INTERACTION_REFRACTION) != 0, axis=-1)
    num_interactions = path.num_interactions
    code = np.full(valid.shape, _SPECULAR, dtype=np.int8)
    code[num_interactions == 0] = _LOS
    code[refraction] = _REFRACTION
    code[diffraction] = _DIFFRACTION
    code[diffuse] = _DIFFUSE
    code[~valid] = PATH_TYPE_INVALID
    return code


def path_power(path: PathGT) -> np.ndarray:
    """Return ``sum |a_baseband|^2`` over hemisphere, row and col, shape ``[V,B,P]``."""
    power = np.sum(np.abs(path.a_baseband) ** 2, axis=(2, 3, 4))
    return np.where(path.valid, power, 0.0)


def beyond_period(path: PathGT, period: float) -> np.ndarray:
    """Return the delay-wrap flag ``valid & (tau >= period)``, shape ``[V,B,P]``."""
    return path.valid & (path.tau >= float(period))


def los_visibility(path: PathGT) -> np.ndarray:
    """Return ``[V,B]`` True where a valid zero-interaction path exists."""
    return np.any(path.valid & (path.num_interactions == 0), axis=-1)


def ground_bounce_visibility(path: PathGT, ground_object: str = GROUND_PLANE_ID) -> np.ndarray:
    """Return ``[V,B]`` True where a single specular ground bounce exists."""
    if ground_object not in path.object_names:
        return np.zeros((path.num_views, path.num_bs), dtype=bool)
    ground_index = path.object_names.index(ground_object)
    single = path.valid & (path_types(path) == _SPECULAR) & (path.num_interactions == 1)
    if path.max_depth == 0:
        return np.zeros((path.num_views, path.num_bs), dtype=bool)
    on_ground = path.object_index[:, :, :, 0] == ground_index
    return np.any(single & on_ground, axis=-1)


def aperture_coefficient(a_elem: np.ndarray, u_local: np.ndarray, geom: CaptureGeometry) -> complex:
    """Return the aperture-centre coefficient of ``a_elem`` ``[2,R,C]``.

    Inverts the carrier-only element phase ``exp(+j k q_m . u)`` on the path's
    own hemisphere and averages over the elements.
    """
    a = np.asarray(a_elem, dtype=np.complex128)
    if a.ndim != 3 or a.shape[0] != 2:
        raise ValueError(f"a_elem must have shape [2, R, C], got {a.shape}")
    u = np.asarray(u_local, dtype=np.float64).reshape(3)
    h = int(hemisphere_index(u))
    coefficients = a[h].reshape(-1)
    phase = np.exp(-1j * geom.wavenumber * (geom.elem_offsets @ u))
    return complex(np.mean(coefficients * phase))


def _los_path(path: PathGT, power: np.ndarray, v: int, b: int) -> int | None:
    """Return the largest-power valid zero-interaction path index for ``(v, b)``."""
    candidates = np.nonzero(path.valid[v, b] & (path.num_interactions[v, b] == 0))[0]
    if candidates.size == 0:
        return None
    return int(candidates[int(np.argmax(power[v, b, candidates]))])


def los_model_error(
    path: PathGT,
    geom: CaptureGeometry,
    *,
    pattern: str,
    polarization: str = "none",
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(phase [V,B] rad, amp_db [V,B])`` of the path GT against the C13 LoS atom."""
    _validate_pattern(pattern)
    _validate_polarization(polarization)
    phase = np.full((path.num_views, path.num_bs), np.nan)
    amp_db = np.full((path.num_views, path.num_bs), np.nan)
    power = path_power(path)
    for v in range(path.num_views):
        for b in range(path.num_bs):
            p = _los_path(path, power, v, b)
            if p is None:
                continue
            factors = capture_factors(
                geom.bs_pos[b][None, :],
                geom,
                "vs",
                v,
                b,
                pattern=pattern,
                polarization=polarization,
            )
            model = np.zeros((2, geom.num_elements), dtype=np.complex128)
            hemisphere = int(factors.hemisphere[0])
            model[hemisphere] = factors.gamma[0] * np.exp(
                1j * geom.wavenumber * (geom.elem_offsets @ factors.u_local[0])
            )
            measured = path.a_baseband[v, b, :, :, :, p].reshape(2, -1)
            ratio = np.vdot(model, measured) / np.vdot(model, model)
            phase[v, b] = np.angle(ratio)
            amp_db[v, b] = 20.0 * np.log10(np.abs(ratio))
    return phase, amp_db


@dataclass(frozen=True)
class ReflectionPlanes:
    """Specular reflection planes fitted from the path vertices."""

    normal: np.ndarray  # [Np,3] unit, pointing to the side the rays are on
    offset: np.ndarray  # [Np] plane is normal . x = offset
    object: np.ndarray  # [Np] int64 object index
    num_vertices: np.ndarray  # [Np] int64 number of specular vertices assigned
    vertex_plane: np.ndarray  # [V,B,P,D] int64 plane index of each specular vertex, else -1


def reflection_planes(
    path: PathGT,
    geom: CaptureGeometry,
    *,
    angle_tol: float = DEFAULT_PLANE_ANGLE_TOL,
    offset_tol: float = DEFAULT_PLANE_OFFSET_TOL_M,
) -> ReflectionPlanes:
    """Fit specular reflection planes from the VS-candidate path vertices."""
    num_views, num_bs, num_paths = path.num_views, path.num_bs, path.num_paths
    depth = path.max_depth
    vertex_plane = np.full((num_views, num_bs, num_paths, depth), -1, dtype=np.int64)
    types = path_types(path)
    candidate = path.valid & np.isin(types, _VS_CANDIDATE_TYPES)

    first_normal: list[np.ndarray] = []
    first_offset: list[float] = []
    plane_object: list[int] = []
    member_normals: list[list[np.ndarray]] = []
    member_offsets: list[list[float]] = []
    member_counts: list[int] = []

    for v in range(num_views):
        for b in range(num_bs):
            for p in range(num_paths):
                if not candidate[v, b, p]:
                    continue
                num_int = int(path.num_interactions[v, b, p])
                if num_int == 0 or depth == 0:
                    continue
                flags = path.interactions[v, b, p]
                for d in range(num_int):
                    if not (flags[d] & INTERACTION_SPECULAR):
                        continue
                    q = path.vertices[v, b, p, d]
                    prev = geom.bs_pos[b]
                    for dd in range(d - 1, -1, -1):
                        if flags[dd] & INTERACTION_SPECULAR:
                            prev = path.vertices[v, b, p, dd]
                            break
                    nxt = geom.ue_pos[v]
                    for dd in range(d + 1, num_int):
                        if flags[dd] & INTERACTION_SPECULAR:
                            nxt = path.vertices[v, b, p, dd]
                            break
                    incoming = q - prev
                    outgoing = nxt - q
                    norm_in = float(np.linalg.norm(incoming))
                    norm_out = float(np.linalg.norm(outgoing))
                    if norm_in == 0.0 or norm_out == 0.0:
                        continue
                    delta = outgoing / norm_out - incoming / norm_in
                    norm_delta = float(np.linalg.norm(delta))
                    if norm_delta < 1e-12:
                        continue
                    normal = delta / norm_delta
                    offset = float(np.dot(normal, q))
                    object_index = int(path.object_index[v, b, p, d])
                    assigned = -1
                    for plane_index in range(len(first_normal)):
                        if plane_object[plane_index] != object_index:
                            continue
                        if (
                            np.linalg.norm(normal - first_normal[plane_index]) <= angle_tol
                            and abs(offset - first_offset[plane_index]) <= offset_tol
                        ):
                            assigned = plane_index
                            break
                    if assigned < 0:
                        assigned = len(first_normal)
                        first_normal.append(normal)
                        first_offset.append(offset)
                        plane_object.append(object_index)
                        member_normals.append([])
                        member_offsets.append([])
                        member_counts.append(0)
                    member_normals[assigned].append(normal)
                    member_offsets[assigned].append(offset)
                    member_counts[assigned] += 1
                    vertex_plane[v, b, p, d] = assigned

    num_planes = len(first_normal)
    if num_planes == 0:
        return ReflectionPlanes(
            normal=np.empty((0, 3), dtype=np.float64),
            offset=np.empty(0, dtype=np.float64),
            object=np.empty(0, dtype=np.int64),
            num_vertices=np.empty(0, dtype=np.int64),
            vertex_plane=vertex_plane,
        )
    normals = np.stack([np.mean(member_normals[i], axis=0) for i in range(num_planes)])
    lengths = np.linalg.norm(normals, axis=1)
    normals = normals / lengths[:, None]
    offsets = np.array(
        [float(np.mean(member_offsets[i])) for i in range(num_planes)], dtype=np.float64
    )
    return ReflectionPlanes(
        normal=normals,
        offset=offsets,
        object=np.asarray(plane_object, dtype=np.int64),
        num_vertices=np.asarray(member_counts, dtype=np.int64),
        vertex_plane=vertex_plane,
    )


@dataclass(frozen=True)
class VirtualSources:
    """Clustered virtual sources with per-view amplitudes and attributes."""

    pos: np.ndarray  # [M,3]
    bs: np.ndarray  # [M]
    order: np.ndarray  # [M]
    objects: np.ndarray  # [M,D]
    plane_ids: np.ndarray  # [M,D]
    spread: np.ndarray  # [M]
    num_paths: np.ndarray  # [M]
    visibility: np.ndarray  # [M,V] bool
    power: np.ndarray  # [M,V]
    rho_eff: np.ndarray  # [M,V] complex128
    theta_inc: np.ndarray  # [M,V]
    path_type: np.ndarray  # [M,V] int8
    path_vs: np.ndarray  # [V,B,P]


def _specular_depths(flags: np.ndarray, num_int: int) -> list[int]:
    """Return the depths of the specular interactions in ``flags[:num_int]``."""
    return [d for d in range(num_int) if flags[d] & INTERACTION_SPECULAR]


@dataclass
class _Cluster:
    """Mutable accumulator of one virtual-source cluster before sorting."""

    bs: int
    order: int
    pos: np.ndarray
    objects: np.ndarray
    plane_ids: np.ndarray
    spread: float
    num_paths: int
    visibility: np.ndarray
    power: np.ndarray
    rho_eff: np.ndarray
    theta_inc: np.ndarray
    path_type: np.ndarray
    total_power: float
    members: list[tuple[int, int, int]]


def _empty_virtual_sources(
    num_views: int, num_bs: int, num_paths: int, depth: int
) -> VirtualSources:
    """Return correctly shaped empty virtual-source arrays."""
    return VirtualSources(
        pos=np.empty((0, 3), dtype=np.float64),
        bs=np.empty(0, dtype=np.int64),
        order=np.empty(0, dtype=np.int64),
        objects=np.empty((0, depth), dtype=np.int64),
        plane_ids=np.empty((0, depth), dtype=np.int64),
        spread=np.empty(0, dtype=np.float64),
        num_paths=np.empty(0, dtype=np.int64),
        visibility=np.empty((0, num_views), dtype=bool),
        power=np.empty((0, num_views), dtype=np.float64),
        rho_eff=np.empty((0, num_views), dtype=np.complex128),
        theta_inc=np.empty((0, num_views), dtype=np.float64),
        path_type=np.empty((0, num_views), dtype=np.int8),
        path_vs=np.full((num_views, num_bs, num_paths), -1, dtype=np.int64),
    )


def _cluster_members(
    members: list[tuple[int, int, int]], positions: np.ndarray, cluster_tol_m: float
) -> list[list[int]]:
    """Single-linkage cluster ``positions`` and return member-index groups."""
    if len(members) == 1:
        return [[0]]
    if cluster_tol_m <= 0.0:
        return [[index] for index in range(len(members))]
    tree = linkage(positions, "single")
    labels = fcluster(tree, cluster_tol_m, "distance")
    groups: list[list[int]] = []
    for label in np.unique(labels):
        groups.append([int(index) for index in np.nonzero(labels == label)[0]])
    return groups


def virtual_sources(
    path: PathGT,
    geom: CaptureGeometry,
    *,
    pattern: str,
    cluster_tol_m: float = DEFAULT_CLUSTER_TOL_M,
    planes: ReflectionPlanes | None = None,
) -> VirtualSources:
    """Cluster the VS-candidate paths into virtual sources and derive their truth."""
    _validate_pattern(pattern)
    num_views, num_bs, num_paths = path.num_views, path.num_bs, path.num_paths
    depth = path.max_depth
    types = path_types(path)
    candidate = path.valid & np.isin(types, _VS_CANDIDATE_TYPES)
    power = path_power(path)
    vs_all = virtual_source_positions(
        geom.ue_pos[:, None, None, :], path.tau, path.theta_r, path.phi_r
    )
    if planes is None:
        planes = reflection_planes(path, geom)

    groups: dict[tuple[int, tuple[int, ...]], list[tuple[int, int, int]]] = {}
    for v in range(num_views):
        for b in range(num_bs):
            for p in range(num_paths):
                if not candidate[v, b, p]:
                    continue
                num_int = int(path.num_interactions[v, b, p])
                flags = path.interactions[v, b, p]
                key_objects = tuple(
                    int(path.object_index[v, b, p, d]) for d in _specular_depths(flags, num_int)
                )
                groups.setdefault((b, key_objects), []).append((v, b, p))

    clusters: list[_Cluster] = []
    path_to_cluster = np.full((num_views, num_bs, num_paths), -1, dtype=np.int64)

    for (b, key_objects), members in groups.items():
        positions = np.array([vs_all[v, b, p] for v, b, p in members], dtype=np.float64)
        order = len(key_objects)
        if order == 0:
            # The order-0 virtual source is the BS itself by definition, so no
            # clustering: every member (LoS or the refraction stand-in) joins
            # one cluster centred exactly on the BS position.
            groups_idx = [list(range(len(members)))]
        else:
            groups_idx = _cluster_members(members, positions, cluster_tol_m)
        for member_index in groups_idx:
            member_slots = [members[index] for index in member_index]
            member_positions = positions[member_index]
            if order == 0:
                bs_position = np.array(geom.bs_pos[b], dtype=np.float64, copy=True)
                centre = bs_position
                spread = float(np.max(np.linalg.norm(member_positions - bs_position, axis=1)))
            else:
                centre = np.mean(member_positions, axis=0)
                spread = float(np.max(np.linalg.norm(member_positions - centre, axis=1)))
            member_powers = np.array([power[slot] for slot in member_slots])
            representative = member_slots[int(np.argmax(member_powers))]

            rep_flags = path.interactions[representative[0], representative[1], representative[2]]
            rep_num = int(
                path.num_interactions[representative[0], representative[1], representative[2]]
            )
            rep_depths = _specular_depths(rep_flags, rep_num)
            plane_ids = np.full(depth, -1, dtype=np.int64)
            for j, d in enumerate(rep_depths):
                if j < depth:
                    plane_ids[j] = planes.vertex_plane[
                        representative[0], representative[1], representative[2], d
                    ]

            objects = np.full(depth, -1, dtype=np.int64)
            for j, object_index in enumerate(key_objects):
                if j < depth:
                    objects[j] = object_index

            visibility = np.zeros(num_views, dtype=bool)
            view_power = np.zeros(num_views, dtype=np.float64)
            rho_eff = np.full(num_views, np.nan + 1j * np.nan, dtype=np.complex128)
            theta_inc = np.full(num_views, np.nan, dtype=np.float64)
            view_type = np.full(num_views, PATH_TYPE_INVALID, dtype=np.int8)

            for v in range(num_views):
                view_members = [slot for slot in member_slots if slot[0] == v]
                if not view_members:
                    continue
                visibility[v] = True
                view_power[v] = float(sum(power[slot] for slot in view_members))
                view_powers = np.array([power[slot] for slot in view_members])
                best = view_members[int(np.argmax(view_powers))]
                rho_eff[v] = effective_rho(path, geom, best, pattern)
                view_type[v] = types[best]
                if order == 0:
                    theta_inc[v] = 0.0
                elif order == 1:
                    cosine = incidence_cosine(centre[None, :], geom, v, b)[0]
                    theta_inc[v] = float(np.arccos(np.clip(cosine, -1.0, 1.0)))

            clusters.append(
                _Cluster(
                    bs=b,
                    order=order,
                    pos=centre,
                    objects=objects,
                    plane_ids=plane_ids,
                    spread=spread,
                    num_paths=len(member_slots),
                    visibility=visibility,
                    power=view_power,
                    rho_eff=rho_eff,
                    theta_inc=theta_inc,
                    path_type=view_type,
                    total_power=float(np.sum(view_power)),
                    members=member_slots,
                )
            )

    if not clusters:
        return _empty_virtual_sources(num_views, num_bs, num_paths, depth)

    bs_key = np.array([cluster.bs for cluster in clusters], dtype=np.int64)
    order_key = np.array([cluster.order for cluster in clusters], dtype=np.int64)
    power_key = -np.array([cluster.total_power for cluster in clusters], dtype=np.float64)
    positions = np.stack([cluster.pos for cluster in clusters])
    sort_index = np.lexsort(
        (
            positions[:, 2],
            positions[:, 1],
            positions[:, 0],
            power_key,
            order_key,
            bs_key,
        )
    )
    for new_index, old_index in enumerate(sort_index):
        for slot in clusters[int(old_index)].members:
            path_to_cluster[slot] = new_index

    return VirtualSources(
        pos=positions[sort_index],
        bs=bs_key[sort_index],
        order=order_key[sort_index],
        objects=np.stack([clusters[index].objects for index in sort_index]),
        plane_ids=np.stack([clusters[index].plane_ids for index in sort_index]),
        spread=np.array([clusters[index].spread for index in sort_index], dtype=np.float64),
        num_paths=np.array([clusters[index].num_paths for index in sort_index], dtype=np.int64),
        visibility=np.stack([clusters[index].visibility for index in sort_index]),
        power=np.stack([clusters[index].power for index in sort_index]),
        rho_eff=np.stack([clusters[index].rho_eff for index in sort_index]),
        theta_inc=np.stack([clusters[index].theta_inc for index in sort_index]),
        path_type=np.stack([clusters[index].path_type for index in sort_index]),
        path_vs=path_to_cluster,
    )


def effective_rho(
    path: PathGT,
    geom: CaptureGeometry,
    slot: tuple[int, int, int],
    pattern: str,
) -> complex:
    """Return the effective VS amplitude of one path slot ``(v, b, p)`` (design §6.1)."""
    v, b, p = slot
    u_local = arrival_unit_vectors(path.theta_r[v, b, p], path.phi_r[v, b, p]) @ geom.ue_rot[v]
    alpha = aperture_coefficient(path.a_baseband[v, b, :, :, :, p], u_local, geom)
    departure = arrival_unit_vectors(path.theta_t[v, b, p], path.phi_t[v, b, p])
    bs_rot = np.eye(3) if geom.bs_rot is None else geom.bs_rot[b]
    gain = bs_pattern(departure[None, :], bs_rot, kind=pattern)[0]
    length = SPEED_OF_LIGHT * float(path.tau[v, b, p])
    denominator = (
        gain * geom.wavelength / (4.0 * np.pi * length) * np.exp(-1j * geom.wavenumber * length)
    )
    return complex(alpha / denominator)


@dataclass(frozen=True)
class Interactions:
    """Flattened interaction table in C order of ``(v, b, p, d)``."""

    points: np.ndarray  # [Q,3] float64
    view: np.ndarray  # [Q] int64
    bs: np.ndarray  # [Q] int64
    path: np.ndarray  # [Q] int64
    depth: np.ndarray  # [Q] int64
    type: np.ndarray  # [Q] int64
    object: np.ndarray  # [Q] int64


def interaction_table(path: PathGT) -> Interactions:
    """Return all valid path interactions in C order of ``(v, b, p, d)``."""
    views: list[int] = []
    bss: list[int] = []
    paths: list[int] = []
    depths: list[int] = []
    types: list[int] = []
    objects: list[int] = []
    points: list[np.ndarray] = []
    num_int = path.num_interactions
    for v in range(path.num_views):
        for b in range(path.num_bs):
            for p in range(path.num_paths):
                if not path.valid[v, b, p]:
                    continue
                for d in range(int(num_int[v, b, p])):
                    views.append(v)
                    bss.append(b)
                    paths.append(p)
                    depths.append(d)
                    types.append(int(path.interactions[v, b, p, d]))
                    objects.append(int(path.object_index[v, b, p, d]))
                    points.append(path.vertices[v, b, p, d])
    return Interactions(
        points=np.array(points, dtype=np.float64).reshape(-1, 3),
        view=np.asarray(views, dtype=np.int64),
        bs=np.asarray(bss, dtype=np.int64),
        path=np.asarray(paths, dtype=np.int64),
        depth=np.asarray(depths, dtype=np.int64),
        type=np.asarray(types, dtype=np.int64),
        object=np.asarray(objects, dtype=np.int64),
    )


def path_ground_truth(
    path: PathGT,
    geom: CaptureGeometry,
    *,
    pattern: str,
    los_polarization: str = "none",
    cluster_tol_m: float = DEFAULT_CLUSTER_TOL_M,
    ground_object: str = GROUND_PLANE_ID,
) -> dict[str, np.ndarray]:
    """Return the full path-derived tomography ground truth dict (design §6.1)."""
    _validate_pattern(pattern)
    _validate_polarization(los_polarization)
    if path.a_baseband.shape[0] != geom.num_views or path.a_baseband.shape[1] != geom.num_bs:
        raise ValueError(
            "path and geom disagree on (V, B): "
            f"{path.a_baseband.shape[:2]} vs {(geom.num_views, geom.num_bs)}"
        )
    rows, cols = geom.aperture_shape
    if path.a_baseband.shape[3:5] != (rows, cols):
        raise ValueError(
            f"a_baseband aperture {(path.a_baseband.shape[3], path.a_baseband.shape[4])} "
            f"does not match geom.aperture_shape {(rows, cols)}"
        )

    types = path_types(path)
    power = path_power(path)
    planes = reflection_planes(path, geom)
    sources = virtual_sources(
        path, geom, pattern=pattern, cluster_tol_m=cluster_tol_m, planes=planes
    )
    table = interaction_table(path)
    los_phase, los_amp_db = los_model_error(
        path, geom, pattern=pattern, polarization=los_polarization
    )
    interaction_plane = planes.vertex_plane[table.view, table.bs, table.path, table.depth]
    return {
        "path_type": types,
        "path_power": power,
        "path_vs": sources.path_vs,
        "beyond_period": beyond_period(path, geom.delay_period),
        "los_visible": los_visibility(path),
        "ground_bounce_visible": ground_bounce_visibility(path, ground_object),
        "los_phase_model_error": los_phase,
        "los_amp_model_error_db": los_amp_db,
        "vs_pos": sources.pos,
        "vs_bs": sources.bs,
        "vs_order": sources.order,
        "vs_num_paths": sources.num_paths,
        "vs_objects": sources.objects,
        "vs_plane_ids": sources.plane_ids,
        "vs_spread": sources.spread,
        "vs_visibility": sources.visibility,
        "vs_power": sources.power,
        "vs_rho_eff": sources.rho_eff,
        "vs_theta_inc": sources.theta_inc,
        "vs_path_type": sources.path_type,
        "plane_normal": planes.normal,
        "plane_offset": planes.offset,
        "plane_object": planes.object,
        "plane_num_vertices": planes.num_vertices,
        "interaction_points": table.points,
        "interaction_view": table.view,
        "interaction_bs": table.bs,
        "interaction_path": table.path,
        "interaction_depth": table.depth,
        "interaction_type": table.type,
        "interaction_object": table.object,
        "interaction_plane": interaction_plane,
    }


def triangle_normals(triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(normals [T,3], valid [T])`` of the triangle array ``[T,3,3]``."""
    tri = np.asarray(triangles, dtype=np.float64)
    if tri.ndim != 3 or tri.shape[1:] != (3, 3):
        raise ValueError(f"triangles must have shape [T, 3, 3], got {tri.shape}")
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(cross, axis=1)
    valid = norm > 1e-12
    normals = np.zeros_like(cross)
    np.divide(cross, norm[:, None], out=normals, where=valid[:, None])
    return normals, valid


def _points_in_triangle_2d(
    points: np.ndarray, triangle: np.ndarray, tolerance: float = 1e-9
) -> np.ndarray:
    """Return which ``[K,2]`` points lie inside the ``[3,2]`` triangle (inclusive)."""
    p0, p1, p2 = triangle[0], triangle[1], triangle[2]
    edge0 = p1 - p0
    edge1 = p2 - p0
    denominator = edge0[0] * edge1[1] - edge0[1] * edge1[0]
    if denominator == 0.0:
        return np.zeros(points.shape[0], dtype=bool)
    offset = points - p0
    bary1 = (offset[:, 0] * edge1[1] - offset[:, 1] * edge1[0]) / denominator
    bary2 = (edge0[0] * offset[:, 1] - edge0[1] * offset[:, 0]) / denominator
    bary0 = 1.0 - bary1 - bary2
    return (bary0 >= -tolerance) & (bary1 >= -tolerance) & (bary2 >= -tolerance)


def sample_triangles(
    triangles: np.ndarray, spacing: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return deterministic plane-anchored lattice samples of ``triangles``.

    Returns ``(points [S,3], normals [S,3], triangle_index [S])``. A valid
    triangle with no lattice point inside contributes its centroid; degenerate
    triangles contribute nothing; duplicate points keep their first occurrence.
    """
    spacing = float(spacing)
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError(f"spacing must be finite and > 0, got {spacing!r}")
    tri = np.asarray(triangles, dtype=np.float64)
    normals, valid = triangle_normals(tri)

    point_blocks: list[np.ndarray] = []
    normal_blocks: list[np.ndarray] = []
    index_blocks: list[np.ndarray] = []
    for t in np.nonzero(valid)[0]:
        normal = normals[t]
        anchor = np.array([0.0, 0.0, 1.0]) if abs(normal[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        u = np.cross(anchor, normal)
        u = u / np.linalg.norm(u)
        w = np.cross(normal, u)
        verts2d = np.stack([tri[t] @ u, tri[t] @ w], axis=1)
        low = verts2d.min(axis=0)
        high = verts2d.max(axis=0)
        i_lo = int(np.ceil(low[0] / spacing - 0.5 - 1e-12))
        i_hi = int(np.floor(high[0] / spacing - 0.5 + 1e-12))
        j_lo = int(np.ceil(low[1] / spacing - 0.5 - 1e-12))
        j_hi = int(np.floor(high[1] / spacing - 0.5 + 1e-12))
        if i_hi >= i_lo and j_hi >= j_lo:
            grid_u = (np.arange(i_lo, i_hi + 1) + 0.5) * spacing
            grid_w = (np.arange(j_lo, j_hi + 1) + 0.5) * spacing
            mesh_u, mesh_w = np.meshgrid(grid_u, grid_w, indexing="ij")
            candidates = np.stack([mesh_u.ravel(), mesh_w.ravel()], axis=1)
            inside = _points_in_triangle_2d(candidates, verts2d)
            samples2d = candidates[inside]
        else:
            samples2d = np.empty((0, 2), dtype=np.float64)
        if samples2d.shape[0] == 0:
            samples2d = verts2d.mean(axis=0, keepdims=True)
        offset = float(np.dot(tri[t, 0], normal))
        samples3d = (
            samples2d[:, 0:1] * u[None, :]
            + samples2d[:, 1:2] * w[None, :]
            + offset * normal[None, :]
        )
        point_blocks.append(samples3d)
        normal_blocks.append(np.broadcast_to(normal, samples3d.shape))
        index_blocks.append(np.full(samples3d.shape[0], t, dtype=np.int64))

    if not point_blocks:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
            np.empty(0, dtype=np.int64),
        )
    points = np.concatenate(point_blocks, axis=0)
    sample_normals = np.concatenate(normal_blocks, axis=0)
    triangle_index = np.concatenate(index_blocks, axis=0)
    if points.shape[0] > 0:
        _, first = np.unique(np.round(points, DEDUP_DECIMALS), axis=0, return_index=True)
        keep = np.sort(first)
        points = points[keep]
        sample_normals = sample_normals[keep]
        triangle_index = triangle_index[keep]
    return points, sample_normals, triangle_index


def segments_blocked(origins: np.ndarray, targets: np.ndarray, triangles: np.ndarray) -> np.ndarray:
    """Return ``[S]`` bool: does any segment hit any triangle (Moller-Trumbore)."""
    origins = np.asarray(origins, dtype=np.float64).reshape(-1, 3)
    targets = np.asarray(targets, dtype=np.float64).reshape(-1, 3)
    tri = np.asarray(triangles, dtype=np.float64)
    if origins.shape != targets.shape:
        raise ValueError("origins and targets must share a shape [S, 3]")
    if tri.ndim != 3 or tri.shape[1:] != (3, 3):
        raise ValueError(f"triangles must have shape [T, 3, 3], got {tri.shape}")
    num_segments = origins.shape[0]
    blocked = np.zeros(num_segments, dtype=bool)
    if num_segments == 0 or tri.shape[0] == 0:
        return blocked

    directions = targets - origins
    edge1 = tri[:, 1] - tri[:, 0]
    edge2 = tri[:, 2] - tri[:, 0]
    vertex0 = tri[:, 0]
    segment_chunk = 32768
    triangle_chunk = 64
    for start in range(0, num_segments, segment_chunk):
        stop = min(start + segment_chunk, num_segments)
        origin = origins[start:stop]
        direction = directions[start:stop]
        local = np.zeros(stop - start, dtype=bool)
        for t_start in range(0, tri.shape[0], triangle_chunk):
            t_stop = min(t_start + triangle_chunk, tri.shape[0])
            e1 = edge1[t_start:t_stop][None, :, :]
            e2 = edge2[t_start:t_stop][None, :, :]
            v0 = vertex0[t_start:t_stop][None, :, :]
            pvec = np.cross(direction[:, None, :], e2)
            det = np.einsum("cti,cti->ct", e1, pvec)
            ok = np.abs(det) > 1e-15
            inverse = np.zeros_like(det)
            np.divide(1.0, det, out=inverse, where=ok)
            tvec = origin[:, None, :] - v0
            bary_u = np.einsum("cti,cti->ct", tvec, pvec) * inverse
            qvec = np.cross(tvec, e1)
            bary_v = np.einsum("cti,cti->ct", direction[:, None, :], qvec) * inverse
            param = np.einsum("cti,cti->ct", e2, qvec) * inverse
            hit = (
                ok
                & (bary_u >= 0.0)
                & (bary_v >= 0.0)
                & (bary_u + bary_v <= 1.0)
                & (param > 1e-9)
                & (param < 1.0 - 1e-9)
            )
            local |= np.any(hit, axis=1)
        blocked[start:stop] = local
    return blocked


def surface_observability(
    points: np.ndarray,
    normals: np.ndarray,
    triangles: np.ndarray,
    bs_pos: np.ndarray,
    ue_pos: np.ndarray,
    *,
    offset: float = SURFACE_OFFSET_M,
) -> np.ndarray:
    """Return ``[S]`` bool: sample visible from at least one BS and one UE on one side."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    normals = np.asarray(normals, dtype=np.float64).reshape(-1, 3)
    bs = np.asarray(bs_pos, dtype=np.float64).reshape(-1, 3)
    ue = np.asarray(ue_pos, dtype=np.float64).reshape(-1, 3)
    tri = np.asarray(triangles, dtype=np.float64)
    num_samples = points.shape[0]
    observable = np.zeros(num_samples, dtype=bool)
    if num_samples == 0:
        return observable

    for side in (1.0, -1.0):
        origin = points + side * offset * normals
        bs_ok = np.zeros(num_samples, dtype=bool)
        for b in range(bs.shape[0]):
            facing = side * np.einsum("si,si->s", normals, bs[b] - points) > 0.0
            active = (~observable) & facing & (~bs_ok)
            if not np.any(active):
                continue
            indices = np.nonzero(active)[0]
            targets = np.broadcast_to(bs[b], (indices.size, 3))
            blocked = segments_blocked(origin[indices], targets, tri)
            bs_ok[indices[~blocked]] = True
        if not np.any(bs_ok):
            continue
        ue_ok = np.zeros(num_samples, dtype=bool)
        for p in range(ue.shape[0]):
            facing = side * np.einsum("si,si->s", normals, ue[p] - points) > 0.0
            active = (~observable) & bs_ok & facing & (~ue_ok)
            if not np.any(active):
                continue
            indices = np.nonzero(active)[0]
            targets = np.broadcast_to(ue[p], (indices.size, 3))
            blocked = segments_blocked(origin[indices], targets, tri)
            ue_ok[indices[~blocked]] = True
        observable |= (~observable) & bs_ok & ue_ok
    return observable


def specular_support(
    points: np.ndarray,
    interaction_points: np.ndarray,
    radius: float = DEFAULT_SUPPORT_RADIUS_M,
) -> np.ndarray:
    """Return ``[S]`` bool: an interaction point lies within ``radius`` (inclusive)."""
    samples = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    interactions = np.asarray(interaction_points, dtype=np.float64).reshape(-1, 3)
    if samples.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    if interactions.shape[0] == 0:
        return np.zeros(samples.shape[0], dtype=bool)
    tree = cKDTree(interactions)
    distance, _ = tree.query(samples, k=1)
    return np.asarray(distance <= float(radius), dtype=bool)


def surface_ground_truth(
    triangles: np.ndarray,
    triangle_object: np.ndarray,
    bs_pos: np.ndarray,
    ue_pos: np.ndarray,
    specular_points: np.ndarray,
    *,
    spacing: float = DEFAULT_SURFACE_SPACING_M,
    roi: tuple[Sequence[float], Sequence[float]] | None = None,
    support_radius: float = DEFAULT_SUPPORT_RADIUS_M,
) -> dict[str, np.ndarray]:
    """Sample the mesh and score observability and specular support of each sample."""
    tri = np.asarray(triangles, dtype=np.float64)
    object_of_triangle = np.asarray(triangle_object, dtype=np.int64)
    samples, normals, triangle_index = sample_triangles(tri, spacing)
    if roi is None:
        roi_array = np.full((2, 3), np.nan, dtype=np.float64)
    else:
        low = np.asarray(roi[0], dtype=np.float64).reshape(3)
        high = np.asarray(roi[1], dtype=np.float64).reshape(3)
        roi_array = np.stack([low, high])
        inside = np.all((samples >= low) & (samples <= high), axis=1)
        samples = samples[inside]
        normals = normals[inside]
        triangle_index = triangle_index[inside]
    observable = surface_observability(samples, normals, tri, bs_pos, ue_pos)
    support = specular_support(samples, specular_points, support_radius)
    return {
        "surface_samples": samples,
        "surface_normals": normals,
        "surface_object": object_of_triangle[triangle_index],
        "surface_observable": observable,
        "surface_specular_support": support,
        "surface_roi": roi_array,
    }


def bs_indices(bs: Sequence[int], num_bs: int) -> list[int]:
    """Validate a BS subset (non-empty, unique ints in ``[0, num_bs)``, kept in order)."""
    if isinstance(bs, (str, bytes)):
        raise ValueError("bs must be a non-empty sequence of BS indices")
    try:
        values = list(bs)
    except TypeError as error:
        raise ValueError("bs must be a non-empty sequence of BS indices") from error
    if not values or any(
        isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in values
    ):
        raise ValueError("bs must be a non-empty sequence of integer BS indices")
    indices = [int(value) for value in values]
    if any(not 0 <= index < num_bs for index in indices):
        raise ValueError(f"bs indices {indices} out of range [0, {num_bs})")
    if len(set(indices)) != len(indices):
        raise ValueError("bs must hold unique indices")
    return indices


def select_bs(arrays: Mapping[str, np.ndarray], bs: Sequence[int]) -> dict[str, np.ndarray]:
    """Return a copy of the GT ``arrays`` restricted to the BS indices ``bs`` (in order).

    Capture arrays are sliced on their BS axis, VS and interaction rows of other
    BSs are dropped, and ``vs_bs``, ``interaction_bs`` and ``path_vs`` are
    remapped to the new indices (``path_vs`` of a dropped VS becomes -1).
    """
    present = [key for key in GT_CAPTURE_KEYS if key in arrays]
    if not present:
        raise ValueError("arrays hold no GT capture key with a BS axis")
    num_bs = int(np.asarray(arrays[present[0]]).shape[1])
    if any(np.ndim(arrays[key]) < 2 or np.shape(arrays[key])[1] != num_bs for key in present):
        raise ValueError(f"GT capture arrays must have shape [V, B, ...] with B = {num_bs}")
    index = np.asarray(bs_indices(bs, num_bs), dtype=np.int64)
    new_bs = np.full(num_bs, -1, dtype=np.int64)
    new_bs[index] = np.arange(index.size)
    out = {key: np.asarray(value) for key, value in arrays.items()}
    for key in present:
        out[key] = out[key][:, index]
    if "bs_ids" in out:
        out["bs_ids"] = out["bs_ids"][index]
    if "vs_bs" in out:
        keep = new_bs[out["vs_bs"].astype(np.int64)] >= 0
        new_vs = np.full(keep.size, -1, dtype=np.int64)
        new_vs[keep] = np.arange(int(keep.sum()))
        for key in GT_VS_KEYS:
            if key in out:
                out[key] = out[key][keep]
        out["vs_bs"] = new_bs[out["vs_bs"].astype(np.int64)]
        if "path_vs" in out:
            path_vs = out["path_vs"].astype(np.int64)
            out["path_vs"] = np.where(path_vs >= 0, new_vs[np.maximum(path_vs, 0)], -1)
    if "interaction_bs" in out:
        keep = new_bs[out["interaction_bs"].astype(np.int64)] >= 0
        for key in GT_INTERACTION_KEYS:
            if key in out:
                out[key] = out[key][keep]
        out["interaction_bs"] = new_bs[out["interaction_bs"].astype(np.int64)]
    return out
