"""Physically consistent synthetic RF-camera datasets for viewer tests.

The scene is one 10 m box ``[-5, 5] x [-5, 5] x [0, 10]`` standing on the
ground plane ``z = 0`` at the world origin. Ring views look at the box centre
``(0, 0, 5)`` and two base stations illuminate it. Each (view, BS) link carries
two plane waves: the direct line-of-sight (LoS) wave and its specular ground
reflection. This is the smallest geometry whose per-view aperture already
contains two interfering paths with a known, closed-form geometry, so the
viewer's image, delay and path-resynthesis code paths are all exercised.

The path ground truth is built FIRST (delays, arrival directions, reflection
vertices and baseband coefficients), and the aperture CFR is then synthesised
from it with :func:`~plateau_rt.domain.rf_camera.paths.synthesize_cfr`. Path-GT
resynthesis therefore matches the stored aperture by construction, up to the
complex64 rounding of the stored dtypes. Derived per-BS images are developed
exactly like the writer does, so a viewer can reproduce them from the manifest
alone.

Sign convention: the local aperture phase of a plane wave arriving from the
UE-local unit direction ``u`` is ``exp(+j*2*pi*(u_y*y + u_z*z))`` on the
centred Sionna PlanarArray grid. With this sign
``calibrate_angular_cfr(aperture_to_angular_fft(...))`` peaks at
``(ky, kz) = (u_y, u_z)`` and the hemisphere index is ``0`` (front) when
``u_x >= 0`` and ``1`` (back) otherwise.
"""

from __future__ import annotations

import struct
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import pi, radians
from pathlib import Path
from typing import Any

import numpy as np
from rf_manifest_fixtures import write_v2_dataset, write_v3_dataset

from plateau_rt.domain.rf_camera.calibration import (
    geometric_los_source_direction_local,
    rotation_matrix,
)
from plateau_rt.domain.rf_camera.camera import (
    RFViewSpec,
    build_direction_cosine_camera_model,
    generate_ring_views,
)
from plateau_rt.domain.rf_camera.delay import SPEED_OF_LIGHT_M_S
from plateau_rt.domain.rf_camera.develop import (
    DevelopParams,
    center_frequency_products,
    delay_products,
    develop_hemisphere_image,
)
from plateau_rt.domain.rf_camera.imaging import frequency_offsets
from plateau_rt.domain.rf_camera.paths import (
    apply_path_order,
    canonical_path_order,
    synthesize_cfr,
)

CARRIER_HZ = 3.5e9
BANDWIDTH_HZ = 20e6
SPACING_LAMBDA = 0.5
PHASE_FLOOR_DB = -35.0
SOURCE_SCENE = "scene/scene.xml"
BOX_MIN_M = (-5.0, -5.0, 0.0)
BOX_MAX_M = (5.0, 5.0, 10.0)
BOX_CENTER_M = (0.0, 0.0, 5.0)
GROUND_Z_M = 0.0
RING_RADIUS_M = 20.0
UE_HEIGHT_M = 1.5
BS_POSITIONS_M = ((60.0, 5.0, 24.0), (-23.0, 50.0, 18.0))
GROUND_REFLECTION = -0.6
FAR_WALL_REFLECTION = -0.8
FAR_WALL_MARGIN_M = 100.0

CAMERA_MODEL_FILE_NAME = "camera_model.npz"

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True)
class PathTruth:
    """One valid path's expected values as stored in the path-GT arrays."""

    kind: str
    tau_s: float
    direction_local: tuple[float, float, float]
    hemisphere: str
    beyond_period: bool

    @property
    def ky(self) -> float:
        """Local ``ky/k`` of the arrival (image column coordinate)."""
        return self.direction_local[1]

    @property
    def kz(self) -> float:
        """Local ``kz/k`` of the arrival (image row coordinate)."""
        return self.direction_local[2]


@dataclass(frozen=True)
class PairTruth:
    """Expected values for one (view, BS) link."""

    view_index: int
    bs_index: int
    view_id: str
    bs_id: str
    bs_direction_local: tuple[float, float, float]
    bs_hemisphere: str
    has_los: bool
    paths: tuple[PathTruth, ...]
    expected_peak_pixel: tuple[int, int] | None
    beyond_period_path_indices: tuple[int, ...]


@dataclass(frozen=True)
class FixtureTruth:
    """Frozen expected values of one synthetic viewer dataset."""

    root: Path
    schema_version: int
    view_ids: tuple[str, ...]
    bs_ids: tuple[str, ...]
    num_paths: int
    frequency_offsets_hz: tuple[float, ...]
    unambiguous_delay_s: float
    ky_over_k: tuple[float, ...]
    kz_over_k: tuple[float, ...]
    far_wall_x_m: float | None
    pairs: tuple[PairTruth, ...]

    def pair(self, view_index: int, bs_index: int) -> PairTruth:
        """Return the ``(view_index, bs_index)`` pair; ``KeyError`` when absent."""
        for pair_truth in self.pairs:
            if pair_truth.view_index == view_index and pair_truth.bs_index == bs_index:
                return pair_truth
        raise KeyError(f"no pair for view_index={view_index}, bs_index={bs_index}")


@dataclass(frozen=True)
class _Candidate:
    """One path candidate before occlusion testing."""

    kind: str
    image_source: np.ndarray
    factor: float
    segments: tuple[tuple[np.ndarray, np.ndarray], ...]
    departure_point: np.ndarray
    reflection_point: np.ndarray | None
    object_name: str | None


@dataclass(frozen=True)
class _PathInfo:
    """Per-slot truth recorded while the path arrays are filled."""

    kind: str
    tau_s: float
    direction_local: tuple[float, float, float]
    beyond_period: bool


def nearest_pixel(
    ky: float,
    kz: float,
    ky_over_k: Sequence[float],
    kz_over_k: Sequence[float],
) -> tuple[int, int]:
    """Return ``(row, col)`` = ``(argmin|kz_over_k - kz|, argmin|ky_over_k - ky|)``."""
    ky_axis = np.asarray(ky_over_k, dtype=np.float64)
    kz_axis = np.asarray(kz_over_k, dtype=np.float64)
    row = int(np.argmin(np.abs(kz_axis - kz)))
    col = int(np.argmin(np.abs(ky_axis - ky)))
    return row, col


def chebyshev_distance(a: tuple[int, int], b: tuple[int, int]) -> int:
    """Return the Chebyshev (max-coordinate) distance between two pixels."""
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _png_chunk(tag: bytes, data: bytes) -> bytes:
    """Return one PNG chunk with its big-endian length and CRC-32."""
    crc = zlib.crc32(tag + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)


def encode_png(pixels: np.ndarray) -> bytes:
    """Encode uint8 ``[H, W]`` (grayscale) or ``[H, W, 4]`` (RGBA) as a PNG."""
    array = np.asarray(pixels)
    if array.dtype != np.uint8:
        raise ValueError(f"encode_png expects uint8 pixels, got {array.dtype}")
    if array.ndim == 2:
        colour_type = 0
    elif array.ndim == 3 and array.shape[2] == 4:
        colour_type = 6
    else:
        raise ValueError(f"encode_png expects [H, W] or [H, W, 4], got {array.shape}")
    height, width = int(array.shape[0]), int(array.shape[1])
    raw = b"".join(b"\x00" + array[row].tobytes() for row in range(height))
    ihdr = struct.pack(">IIBBBBB", width, height, 8, colour_type, 0, 0, 0)
    return (
        _PNG_SIGNATURE
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(raw, 9))
        + _png_chunk(b"IEND", b"")
    )


def _resolve_bs_positions(num_bs: int) -> list[tuple[float, float, float]]:
    """Return ``num_bs`` BS positions (the first two are fixed, extras deterministic)."""
    positions = [
        (float(position[0]), float(position[1]), float(position[2])) for position in BS_POSITIONS_M
    ]
    index = len(positions)
    while len(positions) < num_bs:
        angle = radians(200.0 + 45.0 * (index - 2))
        positions.append((40.0 * float(np.cos(angle)), 40.0 * float(np.sin(angle)), 20.0))
        index += 1
    return positions[:num_bs]


def _orientation_from_rotation(rotation: np.ndarray) -> tuple[float, float, float]:
    """Recover Sionna Euler angles from a ``world_from_local`` rotation matrix."""
    beta = float(np.arcsin(np.clip(-rotation[2, 0], -1.0, 1.0)))
    if abs(rotation[2, 0]) < 1.0 - 1e-12:
        alpha = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        gamma = float(np.arctan2(rotation[2, 1], rotation[2, 2]))
    else:
        alpha = 0.0
        gamma = float(np.arctan2(-rotation[1, 2], rotation[1, 1]))
    return alpha, beta, gamma


def _resolve_views(
    view_poses: Sequence[tuple[Sequence[float], Any]] | None,
    num_views: int,
) -> list[RFViewSpec]:
    """Return ring views or one :class:`RFViewSpec` per supplied pose."""
    if view_poses is None:
        return generate_ring_views(
            target=BOX_CENTER_M,
            radius_m=RING_RADIUS_M,
            ue_height_m=UE_HEIGHT_M,
            num_views=num_views,
        )
    views: list[RFViewSpec] = []
    for index, (position, rotation) in enumerate(view_poses):
        matrix = np.asarray(rotation, dtype=np.float64)
        if matrix.shape != (3, 3):
            raise ValueError(f"view_poses[{index}] rotation must be 3x3, got {matrix.shape}")
        orthonormal = np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-9)
        if not orthonormal or not np.isclose(np.linalg.det(matrix), 1.0, atol=1e-9):
            raise ValueError(f"view_poses[{index}] rotation must be orthonormal with det +1")
        orientation = _orientation_from_rotation(matrix)
        if not np.allclose(rotation_matrix(orientation), matrix, atol=1e-9):
            raise ValueError(f"view_poses[{index}] rotation could not be recovered")
        pos = (float(position[0]), float(position[1]), float(position[2]))
        forward = matrix[:, 0]
        look_at = (
            float(pos[0] + forward[0]),
            float(pos[1] + forward[1]),
            float(pos[2] + forward[2]),
        )
        views.append(
            RFViewSpec(
                view_id=f"ue_{index:06d}",
                position=pos,
                look_at=look_at,
                orientation=orientation,
            )
        )
    return views


def _segment_intersects_box(
    start: np.ndarray,
    end: np.ndarray,
    box_min: tuple[float, float, float],
    box_max: tuple[float, float, float],
) -> bool:
    """Return True when the closed segment ``[start, end]`` touches the closed box."""
    t_min, t_max = 0.0, 1.0
    for axis in range(3):
        delta = float(end[axis] - start[axis])
        if abs(delta) < 1e-15:
            if start[axis] < box_min[axis] or start[axis] > box_max[axis]:
                return False
            continue
        t1 = (box_min[axis] - start[axis]) / delta
        t2 = (box_max[axis] - start[axis]) / delta
        if t1 > t2:
            t1, t2 = t2, t1
        t_min = max(t_min, t1)
        t_max = min(t_max, t2)
        if t_min > t_max:
            return False
    return True


def _candidate_slots(
    ue: np.ndarray,
    bs: np.ndarray,
    *,
    beyond_period: bool,
    far_wall_x_m: float | None,
) -> list[_Candidate | None]:
    """Build the candidate paths of one link in slot order los, ground, far_wall."""
    slots: list[_Candidate | None] = [
        _Candidate("los", bs.copy(), 1.0, ((ue, bs),), ue, None, None)
    ]
    if ue[2] > 0.0 and bs[2] > 0.0:
        image = np.array([bs[0], bs[1], -bs[2]], dtype=np.float64)
        t = float(ue[2] / (ue[2] + bs[2]))
        ground = ue + t * (image - ue)
        ground[2] = GROUND_Z_M
        slots.append(
            _Candidate(
                "ground",
                image,
                GROUND_REFLECTION,
                ((ue, ground), (ground, bs)),
                ground,
                ground.copy(),
                "ground",
            )
        )
    else:
        slots.append(None)
    if beyond_period:
        assert far_wall_x_m is not None
        if ue[0] > far_wall_x_m and bs[0] > far_wall_x_m:
            image = np.array([2.0 * far_wall_x_m - bs[0], bs[1], bs[2]], dtype=np.float64)
            t = float((far_wall_x_m - ue[0]) / (image[0] - ue[0]))
            wall = ue + t * (image - ue)
            wall[0] = far_wall_x_m
            slots.append(
                _Candidate(
                    "far_wall",
                    image,
                    FAR_WALL_REFLECTION,
                    ((ue, wall), (wall, bs)),
                    wall,
                    wall.copy(),
                    "far_wall",
                )
            )
        else:
            slots.append(None)
    return slots


def _is_blocked(candidate: _Candidate) -> bool:
    """Return True when any segment of ``candidate`` touches the occluding box."""
    return any(
        _segment_intersects_box(start, end, BOX_MIN_M, BOX_MAX_M)
        for start, end in candidate.segments
    )


def _world_angles(vector: np.ndarray) -> tuple[float, float]:
    """Return Sionna ``(theta, phi)`` of a world-frame unit vector."""
    theta = float(np.arccos(np.clip(vector[2], -1.0, 1.0)))
    phi = float(np.arctan2(vector[1], vector[0]))
    return theta, phi


def _emit_candidate(
    arrays: dict[str, np.ndarray],
    indices: tuple[int, int, int],
    candidate: _Candidate,
    ue: np.ndarray,
    rotation: np.ndarray,
    bs: np.ndarray,
    *,
    name_to_index: Mapping[str, int],
    unambiguous_delay_s: float,
) -> _PathInfo:
    """Fill the stored path arrays for one valid candidate and return its truth."""
    view_index, bs_index, path_index = indices
    offset = candidate.image_source - ue
    length = float(np.linalg.norm(offset))
    tau32 = np.float32(length / SPEED_OF_LIGHT_M_S)
    if candidate.kind == "far_wall" and float(tau32) < unambiguous_delay_s:
        raise RuntimeError("far-wall path is shorter than the unambiguous delay")
    arrival_world = offset / length
    arrival_local = rotation.T @ arrival_world
    departure_world = candidate.departure_point - bs
    departure_world = departure_world / np.linalg.norm(departure_world)

    wavelength = SPEED_OF_LIGHT_M_S / CARRIER_HZ
    amp = (
        candidate.factor
        * wavelength
        / (4.0 * pi * length)
        * np.exp(-1j * 2.0 * pi * CARRIER_HZ * float(tau32))
    )

    rows = arrays["a_baseband"].shape[3]
    cols = arrays["a_baseband"].shape[4]
    y_col = SPACING_LAMBDA * (np.arange(cols) - (cols - 1) / 2.0)
    z_row = SPACING_LAMBDA * ((rows - 1) / 2.0 - np.arange(rows))
    phase = np.exp(
        1j * 2.0 * pi * (arrival_local[1] * y_col[None, :] + arrival_local[2] * z_row[:, None])
    )
    hemisphere = 0 if arrival_local[0] >= 0.0 else 1
    arrays["a_baseband"][view_index, bs_index, hemisphere, :, :, path_index] = (amp * phase).astype(
        np.complex64
    )

    theta_t, phi_t = _world_angles(departure_world)
    theta_r, phi_r = _world_angles(arrival_world)
    arrays["valid"][view_index, bs_index, path_index] = True
    arrays["tau"][view_index, bs_index, path_index] = tau32
    arrays["theta_t"][view_index, bs_index, path_index] = np.float32(theta_t)
    arrays["phi_t"][view_index, bs_index, path_index] = np.float32(phi_t)
    arrays["theta_r"][view_index, bs_index, path_index] = np.float32(theta_r)
    arrays["phi_r"][view_index, bs_index, path_index] = np.float32(phi_r)
    if candidate.kind == "los":
        arrays["interactions"][view_index, bs_index, path_index, 0] = 0
        arrays["num_interactions"][view_index, bs_index, path_index] = 0
    else:
        arrays["interactions"][view_index, bs_index, path_index, 0] = 1
        arrays["num_interactions"][view_index, bs_index, path_index] = 1
        arrays["primitives"][view_index, bs_index, path_index, 0] = 0
    if candidate.reflection_point is not None:
        arrays["vertices"][view_index, bs_index, path_index, 0, :] = candidate.reflection_point
    if candidate.object_name is not None:
        arrays["object_index"][view_index, bs_index, path_index, 0] = name_to_index[
            candidate.object_name
        ]

    return _PathInfo(
        kind=candidate.kind,
        tau_s=float(tau32),
        direction_local=(
            float(arrival_local[0]),
            float(arrival_local[1]),
            float(arrival_local[2]),
        ),
        beyond_period=bool(float(tau32) >= unambiguous_delay_s),
    )


def _new_path_arrays(
    num_views: int, num_bs: int, num_paths: int, rows: int, cols: int
) -> dict[str, np.ndarray]:
    """Allocate the stored path-GT arrays with the exporter's keys and dtypes."""
    return {
        "valid": np.zeros((num_views, num_bs, num_paths), dtype=bool),
        "tau": np.full((num_views, num_bs, num_paths), -1.0, dtype=np.float32),
        "a_baseband": np.zeros((num_views, num_bs, 2, rows, cols, num_paths), dtype=np.complex64),
        "theta_t": np.zeros((num_views, num_bs, num_paths), dtype=np.float32),
        "phi_t": np.zeros((num_views, num_bs, num_paths), dtype=np.float32),
        "theta_r": np.zeros((num_views, num_bs, num_paths), dtype=np.float32),
        "phi_r": np.zeros((num_views, num_bs, num_paths), dtype=np.float32),
        "interactions": np.zeros((num_views, num_bs, num_paths, 1), dtype=np.uint32),
        "primitives": np.full((num_views, num_bs, num_paths, 1), 0xFFFFFFFF, dtype=np.uint32),
        "vertices": np.zeros((num_views, num_bs, num_paths, 1, 3), dtype=np.float32),
        "object_index": np.full((num_views, num_bs, num_paths, 1), -1, dtype=np.int32),
        "num_interactions": np.zeros((num_views, num_bs, num_paths), dtype=np.int32),
    }


def _apply_ordering(arrays: dict[str, np.ndarray], order: np.ndarray) -> None:
    """Reorder every stored path array in place along its path axis."""
    for name in (
        "valid",
        "tau",
        "theta_t",
        "phi_t",
        "theta_r",
        "phi_r",
        "num_interactions",
    ):
        arrays[name] = apply_path_order(arrays[name], order, path_axis=2)
    arrays["a_baseband"] = apply_path_order(arrays["a_baseband"], order, path_axis=5)
    for name in ("interactions", "primitives", "object_index"):
        arrays[name] = apply_path_order(arrays[name], order, path_axis=2)
    arrays["vertices"] = apply_path_order(arrays["vertices"], order, path_axis=2)


def _develop_artifacts(
    aperture_bs: np.ndarray,
    params: DevelopParams,
    valid_mask: np.ndarray,
    frequency_offsets_hz: np.ndarray,
    num_frequency_bins: int,
) -> dict[str, np.ndarray | bytes]:
    """Reproduce the writer's per-BS derived artifacts (front hemisphere)."""
    developed = develop_hemisphere_image(aperture_bs[0], params)
    center = center_frequency_products(
        developed.image, valid_mask, PHASE_FLOOR_DB, freq_bin=num_frequency_bins // 2
    )
    delays = delay_products(developed.image, valid_mask, frequency_offsets_hz)

    power = np.asarray(center.center_power, dtype=np.float64)
    peak = max(float(np.max(power[valid_mask])), 1e-30)
    db = 10.0 * np.log10(np.maximum(power, 1e-30) / peak)
    scaled = np.clip((db + 60.0) / 60.0, 0.0, 1.0)
    pixels = np.round(scaled * 255.0).astype(np.uint8)
    pixels[~valid_mask] = 0

    return {
        "angular_cfr_center": center.center_cfr,
        "angular_power_center": center.center_power,
        "phase_valid_mask": center.phase_valid,
        "dominant_delay_s": delays.dominant_delay_s,
        "dominant_delay_power": delays.dominant_delay_power,
        "debug_power_png": encode_png(np.flipud(pixels)),
    }


def write_rf_dataset(
    root: Path,
    *,
    num_views: int = 3,
    num_bs: int = 2,
    rows: int = 8,
    cols: int = 8,
    num_bins: int = 16,
    fft: int = 32,
    path_gt: bool = True,
    beyond_period: bool = False,
    schema_version: int = 3,
    view_poses: Sequence[tuple[Sequence[float], Any]] | None = None,
) -> FixtureTruth:
    """Build the physically consistent dataset under ``root`` and return its truth."""
    if schema_version not in (2, 3):
        raise ValueError(f"schema_version must be 2 or 3, got {schema_version}")
    if num_views < 1:
        raise ValueError("num_views must be >= 1")
    if num_bs < 1:
        raise ValueError("num_bs must be >= 1")
    if rows < 1 or cols < 1:
        raise ValueError("rows and cols must be >= 1")
    if num_bins < 2:
        raise ValueError("num_bins must be >= 2")
    if fft < max(rows, cols):
        raise ValueError("fft must not be smaller than rows/cols")
    if view_poses is not None and len(view_poses) == 0:
        raise ValueError("view_poses must not be empty")

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if schema_version == 2:
        num_bs = 1
    bs_positions = _resolve_bs_positions(num_bs)
    views = _resolve_views(view_poses, num_views)
    num_views = len(views)

    offsets = frequency_offsets(BANDWIDTH_HZ, num_bins)
    unambiguous_delay_s = num_bins / BANDWIDTH_HZ
    far_wall_x_m = (
        -(SPEED_OF_LIGHT_M_S * unambiguous_delay_s + FAR_WALL_MARGIN_M) if beyond_period else None
    )
    object_names = sorted(["box", "ground"] + (["far_wall"] if beyond_period else []))
    name_to_index = {name: index for index, name in enumerate(object_names)}
    num_paths = 2 + int(beyond_period)

    arrays = _new_path_arrays(num_views, num_bs, num_paths, rows, cols)
    slot_infos: list[list[list[_PathInfo | None]]] = []
    for view_index, view in enumerate(views):
        ue = np.asarray(view.position, dtype=np.float64)
        rotation = rotation_matrix(view.orientation)
        view_infos: list[list[_PathInfo | None]] = []
        for bs_index, bs_position in enumerate(bs_positions):
            bs = np.asarray(bs_position, dtype=np.float64)
            slots = _candidate_slots(
                ue,
                bs,
                beyond_period=beyond_period,
                far_wall_x_m=far_wall_x_m,
            )
            infos: list[_PathInfo | None] = []
            for path_index, candidate in enumerate(slots):
                if candidate is None or _is_blocked(candidate):
                    infos.append(None)
                    continue
                infos.append(
                    _emit_candidate(
                        arrays,
                        (view_index, bs_index, path_index),
                        candidate,
                        ue,
                        rotation,
                        bs,
                        name_to_index=name_to_index,
                        unambiguous_delay_s=unambiguous_delay_s,
                    )
                )
            view_infos.append(infos)
        slot_infos.append(view_infos)

    power = np.sum(np.abs(arrays["a_baseband"].astype(np.complex128)) ** 2, axis=(2, 3, 4))
    order = canonical_path_order(arrays["tau"], power, arrays["valid"])
    _apply_ordering(arrays, order)

    ordered_infos: list[list[list[_PathInfo | None]]] = []
    for view_index in range(num_views):
        view_infos = []
        for bs_index in range(num_bs):
            view_infos.append(
                [
                    slot_infos[view_index][bs_index][int(order[view_index, bs_index, slot])]
                    for slot in range(num_paths)
                ]
            )
        ordered_infos.append(view_infos)

    aperture = np.zeros((num_views, num_bs, 2, rows, cols, num_bins), dtype=np.complex64)
    for view_index in range(num_views):
        for bs_index in range(num_bs):
            aperture[view_index, bs_index] = synthesize_cfr(
                arrays["a_baseband"][view_index, bs_index],
                arrays["tau"][view_index, bs_index],
                offsets,
            ).astype(np.complex64)

    params = DevelopParams(
        fft_rows=fft,
        fft_cols=fft,
        rx_rows=rows,
        rx_cols=cols,
        horizontal_spacing_lambda=SPACING_LAMBDA,
        vertical_spacing_lambda=SPACING_LAMBDA,
        phase_floor_db=PHASE_FLOOR_DB,
    )
    valid_mask = build_direction_cosine_camera_model(
        fft_rows=fft,
        fft_cols=fft,
        horizontal_spacing_lambda=SPACING_LAMBDA,
        vertical_spacing_lambda=SPACING_LAMBDA,
    )["valid_mask"]

    def derive(
        view_index: int, bs_index: int, aperture_bs: np.ndarray
    ) -> dict[str, np.ndarray | bytes]:
        return _develop_artifacts(aperture_bs, params, valid_mask, offsets, num_bins)

    if schema_version == 3:
        write_v3_dataset(
            root,
            views=views,
            bs_positions=bs_positions,
            bs_look_at=BOX_CENTER_M,
            rows=rows,
            cols=cols,
            bins=num_bins,
            seed=0,
            source_scene=SOURCE_SCENE,
            carrier_hz=CARRIER_HZ,
            bandwidth_hz=BANDWIDTH_HZ,
            fft_rows=fft,
            fft_cols=fft,
            apertures=aperture,
            derive=derive,
            path_gt_arrays=arrays,
            path_gt_object_names=object_names,
            write_path_gt=path_gt,
        )
    else:
        write_v2_dataset(
            root,
            views=views,
            bs_position=bs_positions[0],
            bs_look_at=BOX_CENTER_M,
            rows=rows,
            cols=cols,
            bins=num_bins,
            seed=0,
            source_scene=SOURCE_SCENE,
            carrier_hz=CARRIER_HZ,
            bandwidth_hz=BANDWIDTH_HZ,
            fft_rows=fft,
            fft_cols=fft,
            apertures=aperture[:, 0],
            derive=derive,
            path_gt_arrays=arrays,
            path_gt_object_names=object_names,
            write_path_gt=path_gt,
            write_path_schema=path_gt,
        )

    with np.load(root / CAMERA_MODEL_FILE_NAME) as model:
        ky_over_k = tuple(float(value) for value in model["ky_over_k"])
        kz_over_k = tuple(float(value) for value in model["kz_over_k"])

    pairs: list[PairTruth] = []
    for view_index, view in enumerate(views):
        for bs_index, bs_position in enumerate(bs_positions):
            bs_local = geometric_los_source_direction_local(
                tx_position=bs_position,
                ue_position=view.position,
                ue_orientation=view.orientation,
            )
            bs_direction = (
                float(bs_local[0]),
                float(bs_local[1]),
                float(bs_local[2]),
            )
            paths = tuple(
                PathTruth(
                    kind=info.kind,
                    tau_s=info.tau_s,
                    direction_local=info.direction_local,
                    hemisphere="front" if info.direction_local[0] >= 0.0 else "back",
                    beyond_period=info.beyond_period,
                )
                for info in ordered_infos[view_index][bs_index]
                if info is not None
            )
            has_los = any(path.kind == "los" for path in paths)
            expected_peak_pixel = (
                nearest_pixel(bs_direction[1], bs_direction[2], ky_over_k, kz_over_k)
                if has_los
                else None
            )
            pairs.append(
                PairTruth(
                    view_index=view_index,
                    bs_index=bs_index,
                    view_id=view.view_id,
                    bs_id=f"bs_{bs_index:03d}",
                    bs_direction_local=bs_direction,
                    bs_hemisphere="front" if bs_direction[0] >= 0.0 else "back",
                    has_los=has_los,
                    paths=paths,
                    expected_peak_pixel=expected_peak_pixel,
                    beyond_period_path_indices=tuple(
                        index for index, path in enumerate(paths) if path.beyond_period
                    ),
                )
            )

    return FixtureTruth(
        root=root,
        schema_version=schema_version,
        view_ids=tuple(view.view_id for view in views),
        bs_ids=tuple(f"bs_{index:03d}" for index in range(num_bs)),
        num_paths=num_paths,
        frequency_offsets_hz=tuple(float(value) for value in offsets),
        unambiguous_delay_s=unambiguous_delay_s,
        ky_over_k=ky_over_k,
        kz_over_k=kz_over_k,
        far_wall_x_m=far_wall_x_m,
        pairs=tuple(pairs),
    )
