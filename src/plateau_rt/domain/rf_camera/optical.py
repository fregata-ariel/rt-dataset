"""Pinhole-camera math for optical reference renders (NumPy only).

This module provides the camera model needed to render optical reference
images that line up pixel-for-pixel with every RF-camera view. The renders are
optical reference artifacts -- useful for debugging and for a parallel optical
Gaussian-Splatting dataset -- and are **not** an RF training target.

Camera-local frame
------------------
RF views and :class:`sionna.rt.Camera` share the same camera-local frame:
**x = forward, y = left, z = up**, with ``world_from_local = rotation_matrix(
view.orientation)`` (see :mod:`plateau_rt.domain.rf_camera.calibration`).
The receive aperture lies in the local y-z plane.

Pixel convention
----------------
The following convention was verified against Mitsuba's perspective sensor as
built by Sionna (``fov_axis='x'``). Pixel coordinates are continuous: pixel
``(c, r)`` has its centre at ``(c + 0.5, r + 0.5)`` and the image spans
``[0, W] x [0, H]``. The centre of pixel ``(row r, col c)`` maps to the
camera-local direction proportional to ``(1, -right, up)`` with

``right = (c + 0.5 - W / 2) / fx``, ``up = (H / 2 - (r + 0.5)) / fy``,
``fx = fy = (W / 2) / tan(fov_x / 2)``.

**Row 0 is the top of the image (+z). Column 0 is the left (+y).** The
horizontal FOV ``fov_x`` fixes the shared focal length of the square pixels.

OpenGL/NeRF convention
----------------------
``transforms.json`` uses the OpenGL/NeRF camera convention: the camera looks
along **-Z**, **+Y is up** and **+X is right**. :func:`camera_to_world_opengl`
maps that convention onto the RF camera-local axes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import atan, degrees, isfinite, radians, tan
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

# Columns are the OpenGL/NeRF camera axes expressed in the RF camera-local
# frame (x forward, y left, z up): +X right, +Y up, +Z back.
_RF_LOCAL_AXES_OF_GL = np.array(
    [
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)

_SRGB_LINEAR_CUTOFF = 0.0031308


@dataclass(frozen=True)
class PinholeIntrinsics:
    """Intrinsics of a pinhole camera with square pixels."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        if self.width < 1:
            raise ValueError("width must be >= 1")
        if self.height < 1:
            raise ValueError("height must be >= 1")
        if not isfinite(self.fx) or self.fx <= 0:
            raise ValueError("fx must be finite and > 0")
        if not isfinite(self.fy) or self.fy <= 0:
            raise ValueError("fy must be finite and > 0")
        if not isfinite(self.cx):
            raise ValueError("cx must be finite")
        if not isfinite(self.cy):
            raise ValueError("cy must be finite")

    @classmethod
    def from_horizontal_fov(cls, width: int, height: int, fov_x_deg: float) -> PinholeIntrinsics:
        """Build intrinsics from an image size and a horizontal field of view."""
        if width < 1:
            raise ValueError("width must be >= 1")
        if height < 1:
            raise ValueError("height must be >= 1")
        if not 0.0 < fov_x_deg < 180.0:
            raise ValueError("fov_x_deg must be in (0, 180)")

        focal = (width / 2.0) / tan(radians(fov_x_deg) / 2.0)
        return cls(
            width=int(width),
            height=int(height),
            fx=float(focal),
            fy=float(focal),
            cx=float(width / 2.0),
            cy=float(height / 2.0),
        )

    @property
    def fov_x_deg(self) -> float:
        """Horizontal field of view in degrees."""
        return float(degrees(2.0 * atan(self.width / (2.0 * self.fx))))

    @property
    def fov_y_deg(self) -> float:
        """Vertical field of view in degrees."""
        return float(degrees(2.0 * atan(self.height / (2.0 * self.fy))))

    def matrix(self) -> np.ndarray:
        """Return the OpenCV 3x3 intrinsic matrix ``K`` as float64."""
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )


def pinhole_ray_directions_local(intrinsics: PinholeIntrinsics) -> np.ndarray:
    """Return unit camera-local ray directions for every pixel centre.

    The result has shape ``[H, W, 3]`` in the camera-local frame (x forward,
    y left, z up); ``[r, c]`` indexes the output.
    """
    cols = np.arange(intrinsics.width, dtype=np.float64)[None, :]
    rows = np.arange(intrinsics.height, dtype=np.float64)[:, None]

    right = (cols + 0.5 - intrinsics.cx) / intrinsics.fx
    up = (intrinsics.cy - (rows + 0.5)) / intrinsics.fy

    directions = np.empty((intrinsics.height, intrinsics.width, 3), dtype=np.float64)
    directions[..., 0] = 1.0
    directions[..., 1] = -right
    directions[..., 2] = up
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    return directions


def project_local_points(
    points_local: np.ndarray, intrinsics: PinholeIntrinsics
) -> tuple[np.ndarray, np.ndarray]:
    """Project camera-local points to continuous pixel coordinates.

    ``points_local`` has shape ``[..., 3]`` (x, y, z camera-local). Returns
    ``(col, row)`` as float64 arrays of shape ``points_local.shape[:-1]``.
    Points with ``x <= 0`` are behind or on the camera plane and yield NaN for
    both ``col`` and ``row``. This is the exact inverse of
    :func:`pinhole_ray_directions_local`.
    """
    points = np.asarray(points_local, dtype=np.float64)
    if points.ndim < 1 or points.shape[-1] != 3:
        raise ValueError("points_local must have shape [..., 3]")

    x = points[..., 0]
    y = points[..., 1]
    z = points[..., 2]

    col = np.full(x.shape, np.nan, dtype=np.float64)
    row = np.full(x.shape, np.nan, dtype=np.float64)
    valid = x > 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        col[valid] = intrinsics.cx + intrinsics.fx * (-y[valid] / x[valid])
        row[valid] = intrinsics.cy - intrinsics.fy * (z[valid] / x[valid])
    return col, row


def local_to_world_rays(
    directions_local: np.ndarray, world_from_local: np.ndarray, position: ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """Map camera-local ray directions to world-space rays.

    ``directions_local`` has shape ``[..., 3]``, ``world_from_local`` is the
    3x3 rotation ``R`` and ``position`` is a length-3 array-like camera origin.
    Returns ``(origins, directions_world)``, both float64 with shape
    ``directions_local.shape``. The directions are a pure rotation of unit
    vectors, so they are not renormalised.
    """
    directions = np.asarray(directions_local, dtype=np.float64)
    if directions.ndim < 1 or directions.shape[-1] != 3:
        raise ValueError("directions_local must have shape [..., 3]")

    rotation = np.asarray(world_from_local, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError("world_from_local must be a 3x3 matrix")

    origin = np.asarray(position, dtype=np.float64)
    if origin.shape != (3,):
        raise ValueError("position must have shape (3,)")

    directions_world = directions @ rotation.T
    origins = np.broadcast_to(origin, directions.shape).copy()
    return origins, directions_world


def camera_to_world_opengl(world_from_local: np.ndarray, position: ArrayLike) -> np.ndarray:
    """Return a 4x4 camera-to-world matrix in the OpenGL/NeRF convention.

    With ``R = world_from_local`` the 3x3 block maps the GL axes (-Z forward,
    +Y up, +X right) onto the RF camera-local frame; the translation column is
    ``position`` and the last row is ``(0, 0, 0, 1)``.
    """
    rotation = np.asarray(world_from_local, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError("world_from_local must be a 3x3 matrix")

    origin = np.asarray(position, dtype=np.float64)
    if origin.shape != (3,):
        raise ValueError("position must have shape (3,)")

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation @ _RF_LOCAL_AXES_OF_GL
    matrix[:3, 3] = origin
    return matrix


def z_depth_from_range(range_m: np.ndarray, directions_local: np.ndarray) -> np.ndarray:
    """Convert ray ranges to optical-axis (z) depths.

    For unit camera-local directions ``directions_local[..., 0]`` is the cosine
    to the +x optical axis, so ``range_m * directions_local[..., 0]`` is the
    distance along that axis. NaN ranges stay NaN.
    """
    directions = np.asarray(directions_local, dtype=np.float64)
    if directions.ndim < 1 or directions.shape[-1] != 3:
        raise ValueError("directions_local must have shape [..., 3]")

    ranges = np.asarray(range_m, dtype=np.float64)
    return ranges * directions[..., 0]


def linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    """Apply the IEC 61966-2-1 linear-to-sRGB transfer function elementwise."""
    channels = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0)
    low = 12.92 * channels
    high = 1.055 * channels ** (1.0 / 2.4) - 0.055
    return np.where(channels <= _SRGB_LINEAR_CUTOFF, low, high)


def to_rgba8(rgb_linear: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """Convert linear RGB and coverage to an 8-bit RGBA image.

    ``rgb_linear`` has shape ``[..., 3]`` and ``alpha`` has shape
    ``rgb_linear.shape[:-1]``. NaN values become 0 and the result is clipped to
    ``[0, 255]`` before conversion. The output is a uint8 array of shape
    ``[..., 4]``.
    """
    rgb = np.asarray(rgb_linear, dtype=np.float64)
    if rgb.ndim < 1 or rgb.shape[-1] != 3:
        raise ValueError("rgb_linear must have shape [..., 3]")

    coverage = np.asarray(alpha, dtype=np.float64)
    if coverage.shape != rgb.shape[:-1]:
        raise ValueError("alpha must have shape rgb_linear.shape[:-1]")

    rgb_clean = np.nan_to_num(rgb, nan=0.0)
    alpha_clean = np.nan_to_num(coverage, nan=0.0)

    colour = np.clip(np.round(255.0 * linear_to_srgb(rgb_clean)), 0.0, 255.0)
    opacity = np.clip(np.round(255.0 * np.clip(alpha_clean, 0.0, 1.0)), 0.0, 255.0)

    output = np.empty(rgb.shape[:-1] + (4,), dtype=np.uint8)
    output[..., :3] = colour.astype(np.uint8)
    output[..., 3] = opacity.astype(np.uint8)
    return output


def nerf_transforms(
    intrinsics: PinholeIntrinsics, frames: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Build a nerfstudio/NeRF-style ``transforms.json`` dictionary.

    Each input frame mapping has ``file_path`` (str), ``camera_to_world`` (4x4
    array-like) and an optional ``depth_file_path`` (str). Frames keep their
    input order and every number in the result is a plain Python type so that
    ``json.dumps`` works without a custom encoder.
    """
    width = int(intrinsics.width)
    height = int(intrinsics.height)
    fl_x = float(intrinsics.fx)

    result: dict[str, Any] = {
        "camera_model": "OPENCV",
        "w": width,
        "h": height,
        "fl_x": fl_x,
        "fl_y": float(intrinsics.fy),
        "cx": float(intrinsics.cx),
        "cy": float(intrinsics.cy),
        "camera_angle_x": float(2.0 * atan(width / (2.0 * fl_x))),
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
        "frames": [],
    }

    for frame in frames:
        camera_to_world = np.asarray(frame["camera_to_world"], dtype=np.float64)
        if camera_to_world.shape != (4, 4):
            raise ValueError("camera_to_world must be a 4x4 matrix")

        entry: dict[str, Any] = {
            "file_path": str(frame["file_path"]),
            "transform_matrix": np.asarray(camera_to_world, dtype=np.float64).tolist(),
        }
        depth_file_path = frame.get("depth_file_path")
        if depth_file_path is not None:
            entry["depth_file_path"] = str(depth_file_path)
        result["frames"].append(entry)

    return result
