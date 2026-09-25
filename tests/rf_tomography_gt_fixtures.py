"""Synthetic path ground truth with analytic virtual sources for the T17 tests.

Builds a canonical ``path_geometry_gt``-style array dict (float64/complex128,
optionally Sionna's float32/complex64) for a small scene: a flat ground at
z = 0 and two walls (x = 20 and y = 25) of one building object, one or two
BSs and a ring of UEs. Every specular path is generated with the scalar VS
model of design §3.2 (``beta * G_b(d_dep) * lam / (4 pi L) * exp(-j k L)``
times the carrier-only element phase), so the expected ``vs_rho_eff`` is the
injected ``beta``. This is a plain helper module, not a test file.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.constants import c as SPEED_OF_LIGHT

from plateau_rt.domain.rf_camera.camera import generate_ring_views
from plateau_rt.domain.rf_tomography.antenna import bs_pattern
from plateau_rt.domain.rf_tomography.geometry import CaptureGeometry, hemisphere_index

F_C = 3.5e9
BANDWIDTH = 100e6
NUM_BINS = 64
APERTURE = (4, 4)
TARGET = (0.0, 0.0, 5.0)
BS_POS = ((-40.0, 5.0, 25.0), (10.0, -40.0, 20.0))
OBJECT_NAMES = ("ground_plane", "building")
GROUND, BUILDING = 0, 1
# (point on plane, unit normal toward the scene side, object index)
PLANES = {
    "ground": (np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]), GROUND),
    "wall_a": (np.array([20.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0]), BUILDING),
    "wall_b": (np.array([0.0, 25.0, 0.0]), np.array([0.0, -1.0, 0.0]), BUILDING),
}
SPECULAR, DIFFUSE, REFRACTION, DIFFRACTION = 1, 2, 4, 8
NUM_SLOTS = 8
MAX_DEPTH = 3


@dataclass
class ExpectedVS:
    """Analytic virtual source: position, BS, order and per-view truth."""

    bs: int
    order: int
    pos: np.ndarray
    objects: tuple[int, ...]
    planes: tuple[str, ...]
    beta: dict[int, complex] = field(default_factory=dict)
    path_type: dict[int, int] = field(default_factory=dict)
    cos_inc: dict[int, float] = field(default_factory=dict)


@dataclass
class MirrorScene:
    """The synthetic scene: geometry, path-GT arrays and the analytic truth."""

    geom: CaptureGeometry
    arrays: dict[str, np.ndarray]
    object_names: tuple[str, ...]
    vs: list[ExpectedVS]
    path_type: np.ndarray  # [V, B, P] int8 expected mechanism codes
    los_visible: np.ndarray  # [V, B]
    ground_bounce_visible: np.ndarray  # [V, B]
    beyond_period: np.ndarray  # [V, B, P]


def mirror(point: np.ndarray, plane: str) -> np.ndarray:
    """Mirror ``point`` across one of :data:`PLANES`."""
    p0, n, _ = PLANES[plane]
    return point - 2.0 * np.dot(point - p0, n) * n


def _hit(start: np.ndarray, end: np.ndarray, plane: str) -> np.ndarray:
    """Return the intersection of segment start->end with ``plane`` (asserted inside)."""
    p0, n, _ = PLANES[plane]
    t = np.dot(p0 - start, n) / np.dot(end - start, n)
    assert 0.0 < t < 1.0, (plane, t)
    return start + t * (end - start)


def _angles(direction: np.ndarray) -> tuple[float, float]:
    """World zenith and azimuth of a unit vector (Sionna convention)."""
    d = direction / np.linalg.norm(direction)
    return float(np.arccos(np.clip(d[2], -1.0, 1.0))), float(np.arctan2(d[1], d[0]))


def _geometry(num_views: int) -> CaptureGeometry:
    views = generate_ring_views(target=TARGET, radius_m=15.0, ue_height_m=1.5, num_views=num_views)
    return CaptureGeometry.from_orientations(
        ue_pos=np.array([view.position for view in views]),
        ue_orientations=np.array([view.orientation for view in views]),
        bs_pos=np.array(BS_POS),
        f_c=F_C,
        bandwidth=BANDWIDTH,
        num_bins=NUM_BINS,
        aperture_shape=APERTURE,
        bs_look_at=np.array(TARGET),
    )


def _element_coefficients(
    geom: CaptureGeometry,
    v: int,
    b: int,
    beta: complex,
    length: float,
    d_arr: np.ndarray,
    d_dep: np.ndarray,
    pattern: str,
) -> np.ndarray:
    """Per-element baseband coefficient ``[2, R, C]`` of one path (scalar VS model)."""
    rows, cols = geom.aperture_shape
    u = d_arr @ geom.ue_rot[v]
    gain = bs_pattern(d_dep[None], geom.bs_rot[b], kind=pattern)[0]
    alpha = beta * gain * geom.wavelength / (4.0 * np.pi * length)
    alpha *= np.exp(-1j * geom.wavenumber * length)
    out = np.zeros((2, rows * cols), dtype=np.complex128)
    out[int(hemisphere_index(u))] = alpha * np.exp(1j * geom.wavenumber * (geom.elem_offsets @ u))
    return out.reshape(2, rows, cols)


def build_mirror_scene(
    num_views: int = 6, *, pattern: str = "tr38901", float32: bool = False
) -> MirrorScene:
    """Return the synthetic mirror-plane scene (see the module docstring)."""
    geom = _geometry(num_views)
    V, B, P, D = num_views, len(BS_POS), NUM_SLOTS, MAX_DEPTH
    rows, cols = geom.aperture_shape
    arrays: dict[str, np.ndarray] = {
        "valid": np.zeros((V, B, P), dtype=bool),
        "tau": np.full((V, B, P), -1.0),
        "theta_t": np.zeros((V, B, P)),
        "phi_t": np.zeros((V, B, P)),
        "theta_r": np.zeros((V, B, P)),
        "phi_r": np.zeros((V, B, P)),
        "a_baseband": np.zeros((V, B, 2, rows, cols, P), dtype=np.complex128),
        "interactions": np.zeros((V, B, P, D), dtype=np.uint32),
        "object_index": np.full((V, B, P, D), -1, dtype=np.int32),
        "primitives": np.full((V, B, P, D), 4294967295, dtype=np.uint32),
        "vertices": np.zeros((V, B, P, D, 3)),
    }
    path_type = np.full((V, B, P), -1, dtype=np.int8)
    beyond = np.zeros((V, B, P), dtype=bool)
    expected: dict[tuple[int, str], ExpectedVS] = {}

    def add(
        v: int,
        b: int,
        slot: int,
        *,
        beta: complex,
        chain: list[tuple[np.ndarray, int, int]],
        code: int,
        length: float | None = None,
        key: str | None = None,
        order: int = 0,
        planes: tuple[str, ...] = (),
        vs_pos: np.ndarray | None = None,
        cos_inc: float | None = None,
    ) -> None:
        ue, bs = geom.ue_pos[v], geom.bs_pos[b]
        points = [bs] + [point for point, _, _ in chain] + [ue]
        if length is None:
            length = float(sum(np.linalg.norm(np.diff(np.array(points), axis=0), axis=1)))
        d_dep = (points[1] - bs) / np.linalg.norm(points[1] - bs)
        d_arr = (points[-2] - ue) / np.linalg.norm(points[-2] - ue)
        arrays["valid"][v, b, slot] = True
        arrays["tau"][v, b, slot] = length / SPEED_OF_LIGHT
        arrays["theta_t"][v, b, slot], arrays["phi_t"][v, b, slot] = _angles(d_dep)
        arrays["theta_r"][v, b, slot], arrays["phi_r"][v, b, slot] = _angles(d_arr)
        arrays["a_baseband"][v, b, ..., slot] = _element_coefficients(
            geom, v, b, beta, length, d_arr, d_dep, pattern
        )
        for depth, (point, flag, obj) in enumerate(chain):
            arrays["interactions"][v, b, slot, depth] = flag
            arrays["object_index"][v, b, slot, depth] = obj
            arrays["primitives"][v, b, slot, depth] = 7 + obj
            arrays["vertices"][v, b, slot, depth] = point
        path_type[v, b, slot] = code
        if key is not None:
            entry = expected.setdefault(
                (b, key),
                ExpectedVS(
                    bs=b,
                    order=order,
                    pos=vs_pos if vs_pos is not None else bs.copy(),
                    objects=tuple(PLANES[name][2] for name in planes),
                    planes=planes,
                ),
            )
            entry.beta[v] = complex(beta)
            entry.path_type[v] = code
            if cos_inc is not None:
                entry.cos_inc[v] = cos_inc

    los_visible = np.ones((V, B), dtype=bool)
    ground_visible = np.ones((V, B), dtype=bool)
    for v in range(V):
        ue = geom.ue_pos[v]
        for b in range(B):
            bs = geom.bs_pos[b]
            slot = 0
            if b == 0 and v == 1:
                # LoS blocked: a weak refraction-only path through the building.
                q1, q2 = bs + 0.40 * (ue - bs), bs + 0.45 * (ue - bs)
                add(
                    v,
                    b,
                    slot,
                    beta=0.01 * np.exp(0.3j),
                    chain=[(q1, REFRACTION, BUILDING), (q2, REFRACTION, BUILDING)],
                    code=2,
                    key="los",
                    cos_inc=1.0,
                )
                los_visible[v, b] = False
            else:
                add(v, b, slot, beta=1.0, chain=[], code=0, key="los", cos_inc=1.0)
            slot += 1
            planes = ("ground", "wall_a", "wall_b") if b == 0 else ("ground",)
            for name in planes:
                if b == 1 and v == 3:
                    ground_visible[v, b] = False
                    continue
                image = mirror(bs, name)
                q = _hit(ue, image, name)
                n = PLANES[name][1]
                cos_inc = abs(float(np.dot((ue - q) / np.linalg.norm(ue - q), n)))
                beta = (0.3 + 0.5 * cos_inc) * np.exp(1j * (0.7 + v + 2 * b + slot))
                add(
                    v,
                    b,
                    slot,
                    beta=beta,
                    chain=[(q, SPECULAR, PLANES[name][2])],
                    code=1,
                    key=name,
                    order=1,
                    planes=(name,),
                    vs_pos=image,
                    cos_inc=cos_inc,
                )
                slot += 1
            if b == 0:
                s1 = mirror(bs, "wall_a")
                s2 = mirror(s1, "ground")
                q2 = _hit(ue, s2, "ground")
                q1 = _hit(q2, s1, "wall_a")
                add(
                    v,
                    b,
                    slot,
                    beta=0.2 * np.exp(-1.1j - 0.2j * v),
                    chain=[(q1, SPECULAR, BUILDING), (q2, SPECULAR, GROUND)],
                    code=1,
                    key="wall_a+ground",
                    order=2,
                    planes=("wall_a", "ground"),
                    vs_pos=s2,
                )
                slot += 1
                if v == 0:
                    q = np.array([20.0, 3.0, 6.0])
                    add(v, b, slot, beta=0.05, chain=[(q, DIFFUSE, BUILDING)], code=4)
                    slot += 1
                if v == 2:
                    q = np.array([20.0, 25.0, 10.0])
                    length = geom.delay_period * SPEED_OF_LIGHT + 1.5
                    add(
                        v,
                        b,
                        slot,
                        beta=0.02,
                        chain=[(q, DIFFRACTION, BUILDING)],
                        code=3,
                        length=length,
                    )
                    beyond[v, b, slot] = True
                    slot += 1
    arrays["num_interactions"] = np.count_nonzero(arrays["interactions"], axis=-1).astype(np.int32)
    if float32:
        for name in ("tau", "theta_t", "phi_t", "theta_r", "phi_r", "vertices"):
            arrays[name] = arrays[name].astype(np.float32)
        arrays["a_baseband"] = arrays["a_baseband"].astype(np.complex64)
    return MirrorScene(
        geom=geom,
        arrays=arrays,
        object_names=OBJECT_NAMES,
        vs=list(expected.values()),
        path_type=path_type,
        los_visible=los_visible,
        ground_bounce_visible=ground_visible,
        beyond_period=beyond,
    )


def expected_planes() -> dict[str, tuple[np.ndarray, float, int]]:
    """Return ``{name: (normal, offset, object)}`` with ``normal . x = offset``."""
    return {name: (n, float(np.dot(n, p0)), obj) for name, (p0, n, obj) in PLANES.items()}
