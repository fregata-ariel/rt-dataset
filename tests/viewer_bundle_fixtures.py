"""Analytic viewer-bundle fixtures: optical, scene, bundle, archive and broken data.

Provides :func:`cast_box_ground`, :func:`write_optical`, :func:`write_scene`,
:func:`write_observed`, :func:`write_partial`, :func:`write_bundle_dir`,
:func:`write_fixture_bundle`, :func:`make_archive`, :func:`make_malicious_archive`
and :func:`write_broken_dataset` for viewer tests, plus a CLI::

    PYTHONPATH=src python tests/viewer_bundle_fixtures.py OUT [--archive zip|tar|tar.gz|dir]
        [--schema 2|3] [--broken CASE] [--malicious CASE]
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import re
import shutil
import tarfile
import tempfile
import warnings
import zipfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from viewer_fixtures import (
    BOX_MAX_M,
    BOX_MIN_M,
    GROUND_Z_M,
    FixtureTruth,
    encode_png,
    write_rf_dataset,
)

from plateau_rt.application.rf_camera_observe import observe_dataset
from plateau_rt.application.rf_camera_partial import build_partial_dataset
from plateau_rt.application.rf_dataset_manifest import load_rf_dataset_manifest
from plateau_rt.domain.ground import GROUND_ITU_MATERIAL, ground_plane_mesh
from plateau_rt.domain.rf_camera.impairments import ImpairmentConfig, NoiseSpec
from plateau_rt.domain.rf_camera.optical import (
    PinholeIntrinsics,
    camera_to_world_opengl,
    local_to_world_rays,
    nerf_transforms,
    pinhole_ray_directions_local,
    to_rgba8,
    z_depth_from_range,
)

GROUND_SIZE_M = 400.0
OPTICAL_SIZE_PX = 32
OPTICAL_FOV_X_DEG = 90.0
OPTICAL_LEFT_RGB = (1.0, 0.1, 0.1)
OPTICAL_RIGHT_RGB = (0.1, 0.1, 1.0)
OPTICAL_GROUND_SHADE = 0.35
OPTICAL_RENDERER = "analytic box-and-ground ray caster (tests/viewer_bundle_fixtures.py)"
SCENE_XML_NAME = "scene.xml"
BOX_MATERIAL = "itu_concrete"
MEMBER_KINDS = ("rf_dataset", "rf_partial", "tomo_run", "scene")
ARCHIVE_FORMATS = ("zip", "tar", "tar.gz")
MALICIOUS_CASES = (
    "dotdot",
    "absolute",
    "symlink",
    "hardlink",
    "device",
    "duplicate",
    "too_many_files",
    "bomb",
)
MALICIOUS_MEMBER_NAMES: dict[str, str | None] = {
    "dotdot": "../evil.txt",
    "absolute": "/tmp/evil.txt",
    "symlink": "link",
    "hardlink": "hardlink",
    "device": "dev/null",
    "duplicate": "dup.txt",
    "too_many_files": None,
    "bomb": "zeros.bin",
}
BROKEN_CASES = (
    "bad_schema_version",
    "missing_artifact",
    "bs_order_mismatch",
    "duplicate_view_id",
    "bad_axis_order",
    "bad_position",
)
BROKEN_CASE_MESSAGES: dict[str, str] = {
    "bad_schema_version": "unsupported 'schema_version'",
    "missing_artifact": "missing required key 'aperture_cfr'",
    "bs_order_mismatch": "do not match base station order",
    "duplicate_view_id": "duplicate view_id",
    "bad_axis_order": "'axis_order'",
    "bad_position": "'position_m'",
}

_MEMBER_ID_RE = re.compile(r"[A-Za-z0-9_.-]{1,64}\Z")
_MEMBER_KEYS = ("id", "kind", "path", "for", "source")
_ALLOWED_MEMBER_KEYS = frozenset(_MEMBER_KEYS)
_DIRECTORY_KINDS = ("rf_dataset", "rf_partial", "tomo_run")
_MARKER_FILE = {
    "rf_dataset": "dataset_manifest.json",
    "rf_partial": "partial_manifest.json",
    "tomo_run": "run_manifest.json",
}
_BOX_MIN = np.asarray(BOX_MIN_M, dtype=np.float64)
_BOX_MAX = np.asarray(BOX_MAX_M, dtype=np.float64)
_TIE_EPS = 1e-12
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)


@dataclass(frozen=True)
class BundleFixture:
    """Paths and truth of one canonical valid viewer bundle."""

    root: Path
    bundle_json: Path
    dataset_dir: Path
    scene_xml: Path
    partial_dirs: tuple[Path, ...]
    observation_names: tuple[str, ...]
    truth: FixtureTruth


def cast_box_ground(origins: np.ndarray, directions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Intersect rays [N,3] with the closed box and the finite ground square."""
    origins = np.asarray(origins, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    count = origins.shape[0]
    box_t = np.full(count, np.inf, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        t0 = (_BOX_MIN - origins) / directions
        t1 = (_BOX_MAX - origins) / directions
        t_min = np.minimum(t0, t1)
        t_max = np.maximum(t0, t1)
        parallel = directions == 0.0
        inside = (origins >= _BOX_MIN) & (origins <= _BOX_MAX)
        t_min = np.where(parallel & ~inside, np.inf, t_min)
        t_max = np.where(parallel & ~inside, -np.inf, t_max)
        entry = np.max(t_min, axis=1)
        exit_ = np.min(t_max, axis=1)
        valid = (entry <= exit_) & (exit_ > 1e-9)
        hit_t = np.where(entry > 1e-9, entry, exit_)
        box_t = np.where(valid, hit_t, np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_ground = (float(GROUND_Z_M) - origins[:, 2]) / directions[:, 2]
        half = float(GROUND_SIZE_M) / 2.0
        hit_x = origins[:, 0] + t_ground * directions[:, 0]
        hit_y = origins[:, 1] + t_ground * directions[:, 1]
    ground_ok = (
        (directions[:, 2] < 0.0)
        & (t_ground > 1e-9)
        & (np.abs(hit_x) <= half)
        & (np.abs(hit_y) <= half)
    )
    ground_t = np.where(ground_ok, t_ground, np.inf)
    range_m = np.minimum(box_t, ground_t)
    kind = np.zeros(count, dtype=np.int8)
    hit = np.isfinite(range_m)
    range_m = np.where(hit, range_m, np.nan)
    box_wins = hit & (box_t <= ground_t + _TIE_EPS)
    ground_wins = hit & ~box_wins
    kind[box_wins] = 1
    kind[ground_wins] = 2
    return range_m, kind


def _optical_rgb(left_mask: np.ndarray, kind: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Return linear RGB for a hit pattern (red left, blue right, dark ground)."""
    left = np.asarray(left_mask, dtype=bool).reshape(shape)
    kind = np.asarray(kind, dtype=np.int8).reshape(shape)
    rgb = np.empty(shape + (3,), dtype=np.float64)
    rgb[left] = OPTICAL_LEFT_RGB
    rgb[~left] = OPTICAL_RIGHT_RGB
    rgb[kind == 2] *= float(OPTICAL_GROUND_SHADE)
    rgb[kind == 0] = 0.0
    return rgb


def write_optical(dataset_dir: Path, truth: FixtureTruth) -> Path:
    """Render analytic optical references for every view; return transforms path."""
    dataset_dir = Path(dataset_dir)
    dataset = load_rf_dataset_manifest(dataset_dir)
    if tuple(dataset.view_ids) != tuple(truth.view_ids):
        raise ValueError(
            f"dataset view_ids {list(dataset.view_ids)} != truth {list(truth.view_ids)}"
        )
    manifest: Any = dataset.raw
    manifest_path = dataset.manifest_path
    intrinsics = PinholeIntrinsics.from_horizontal_fov(
        int(OPTICAL_SIZE_PX), int(OPTICAL_SIZE_PX), float(OPTICAL_FOV_X_DEG)
    )
    pinhole_dirs_local = pinhole_ray_directions_local(intrinsics)
    height, width = pinhole_dirs_local.shape[:2]
    with np.load(dataset.camera_model_path) as model:
        valid_mask = np.asarray(model["valid_mask"], dtype=bool)
        ray_dirs_local = np.asarray(model["ray_directions_local"], dtype=np.float64)
    fft_rows, fft_cols = valid_mask.shape
    frames: list[dict[str, Any]] = []
    for dataset_view in dataset.views:
        view_index = dataset_view.index
        view_id = dataset_view.view_id
        view = manifest["views"][view_index]
        pose = json.loads(dataset_view.pose_path.read_text(encoding="utf-8"))
        rotation = np.asarray(pose["world_from_local_rotation"], dtype=np.float64)
        position = np.asarray(pose["position_m"], dtype=np.float64)
        optical_dir = dataset_dir / "views" / view_id / "optical"
        optical_dir.mkdir(parents=True, exist_ok=True)

        origins, directions = local_to_world_rays(pinhole_dirs_local, rotation, position)
        range_flat, kind_flat = cast_box_ground(origins.reshape(-1, 3), directions.reshape(-1, 3))
        hit = (kind_flat > 0).reshape(height, width)
        range_m = range_flat.reshape(height, width)
        rgb = _optical_rgb(pinhole_dirs_local[..., 1] > 0, kind_flat, (height, width))
        rgba = to_rgba8(rgb, hit.astype(float))
        rgba_path = optical_dir / "pinhole_rgba.png"
        depth_path = optical_dir / "pinhole_depth_m.npy"
        range_path = optical_dir / "pinhole_range_m.npy"
        rgba_path.write_bytes(encode_png(rgba))
        np.save(
            depth_path,
            np.where(hit, z_depth_from_range(range_m, pinhole_dirs_local), 0.0).astype(np.float32),
        )
        np.save(range_path, range_m.astype(np.float32))
        frames.append(
            {
                "file_path": str(rgba_path.relative_to(dataset_dir)),
                "depth_file_path": str(depth_path.relative_to(dataset_dir)),
                "camera_to_world": camera_to_world_opengl(rotation, position),
            }
        )

        valid_dirs = ray_dirs_local[valid_mask]
        origins_h, directions_h = local_to_world_rays(valid_dirs, rotation, position)
        range_v, kind_v = cast_box_ground(origins_h, directions_h)
        hit_grid = np.zeros((fft_rows, fft_cols), dtype=bool)
        hit_grid[valid_mask] = kind_v > 0
        rgb_grid = np.zeros((fft_rows, fft_cols, 3), dtype=np.float64)
        rgb_grid[valid_mask] = _optical_rgb(
            valid_dirs[..., 1] > 0, kind_v, (int(np.count_nonzero(valid_mask)),)
        ).reshape(-1, 3)
        range_grid = np.full((fft_rows, fft_cols), np.nan, dtype=np.float32)
        range_grid[valid_mask] = range_v.astype(np.float32)
        rgba_hemi = to_rgba8(rgb_grid, hit_grid.astype(float))
        hemi_rgba_path = optical_dir / "hemisphere_rgba.png"
        hemi_range_path = optical_dir / "hemisphere_range_m.npy"
        hemi_rgba_path.write_bytes(encode_png(np.flipud(rgba_hemi)))
        np.save(hemi_range_path, range_grid)

        view.setdefault("artifacts", {})
        view["artifacts"]["optical_pinhole_rgba"] = str(rgba_path.relative_to(dataset_dir))
        view["artifacts"]["optical_pinhole_depth_m"] = str(depth_path.relative_to(dataset_dir))
        view["artifacts"]["optical_pinhole_range_m"] = str(range_path.relative_to(dataset_dir))
        view["artifacts"]["optical_hemisphere_rgba"] = str(hemi_rgba_path.relative_to(dataset_dir))
        view["artifacts"]["optical_hemisphere_range_m"] = str(
            hemi_range_path.relative_to(dataset_dir)
        )

    transforms = nerf_transforms(intrinsics, frames)
    transforms_path = dataset_dir / "transforms.json"
    transforms_path.write_text(json.dumps(transforms, indent=2), encoding="utf-8")
    manifest["optical_reference"] = {
        "purpose": (
            "Reference optical renders for debugging and for a parallel optical "
            "Gaussian-Splatting dataset; NOT an RF training target."
        ),
        "renderer": OPTICAL_RENDERER,
        "spp": 1,
        "seed": 0,
        "pinhole": {
            "width": intrinsics.width,
            "height": intrinsics.height,
            "fov_x_deg": float(OPTICAL_FOV_X_DEG),
            "fx": intrinsics.fx,
            "fy": intrinsics.fy,
            "cx": intrinsics.cx,
            "cy": intrinsics.cy,
            "image_orientation": (
                "standard photo orientation: row 0 = top (+z), column 0 = camera left (+y)"
            ),
            "depth_definition": (
                "z-depth along the optical axis (local +x), metres; 0.0 where no "
                "hit (nerfstudio convention for missing depth)"
            ),
            "range_definition": (
                "distance from the camera position to the hit point, metres; NaN where no hit"
            ),
            "transforms": "transforms.json",
        },
        "hemisphere": {
            "grid": (
                "RF direction-cosine grid of camera_model.npz (front hemisphere), "
                "pixel-aligned with the RF arrays: row = kz index increasing "
                "upward in kz, column = ky index increasing toward +ky"
            ),
            "png_orientation": (
                "hemisphere_rgba.png rows are flipped (np.flipud) so +kz is at "
                "the top, keeping columns in +ky order, to match the RF debug "
                "PNGs (matplotlib origin='lower'); the underlying .npy array is "
                "NOT flipped and keeps the RF grid row/column order. Because +ky "
                "is the camera's LEFT, the RF/hemisphere images are left-right "
                "mirrored relative to the pinhole photo."
            ),
            "range_definition": (
                "distance from the camera position to the hit point along each "
                "direction-cosine ray, metres; NaN where no hit or outside "
                "valid_mask"
            ),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return transforms_path


def _write_ply(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    """Write a trimesh-style binary little-endian PLY file."""
    verts = np.asarray(vertices, dtype="<f4")
    face_idx = np.asarray(faces, dtype="<i4")
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(f"vertices must have shape [N, 3], got {verts.shape}")
    if face_idx.ndim != 2 or face_idx.shape[1] != 3:
        raise ValueError(f"faces must have shape [M, 3], got {face_idx.shape}")
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {verts.shape[0]}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        f"element face {face_idx.shape[0]}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    )
    face_struct = np.empty(face_idx.shape[0], dtype=[("n", "u1"), ("i", "<i4", (3,))])
    face_struct["n"] = 3
    face_struct["i"] = face_idx
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(header.encode("ascii"))
        handle.write(np.ascontiguousarray(verts).tobytes())
        handle.write(face_struct.tobytes())


def write_scene(root: Path) -> Path:
    """Write box.ply, ground.ply and scene.xml under ``root``; return XML path."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    box_vertices = np.array(
        [
            [BOX_MIN_M[0], BOX_MIN_M[1], BOX_MIN_M[2]],
            [BOX_MAX_M[0], BOX_MIN_M[1], BOX_MIN_M[2]],
            [BOX_MAX_M[0], BOX_MAX_M[1], BOX_MIN_M[2]],
            [BOX_MIN_M[0], BOX_MAX_M[1], BOX_MIN_M[2]],
            [BOX_MIN_M[0], BOX_MIN_M[1], BOX_MAX_M[2]],
            [BOX_MAX_M[0], BOX_MIN_M[1], BOX_MAX_M[2]],
            [BOX_MAX_M[0], BOX_MAX_M[1], BOX_MAX_M[2]],
            [BOX_MIN_M[0], BOX_MAX_M[1], BOX_MAX_M[2]],
        ],
        dtype=np.float64,
    )
    box_faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [2, 3, 7],
            [2, 7, 6],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ],
        dtype=np.int64,
    )
    ground_vertices, ground_faces = ground_plane_mesh(float(GROUND_SIZE_M), float(GROUND_Z_M))
    _write_ply(root / "box.ply", box_vertices, box_faces)
    _write_ply(root / "ground.ply", ground_vertices, ground_faces)
    box_mat = f"mat-{BOX_MATERIAL}"
    ground_mat = f"mat-{GROUND_ITU_MATERIAL}"
    xml_text = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<scene version="3.0.0">\n'
        f'    <bsdf type="diffuse" id="{box_mat}">\n'
        '        <rgb name="reflectance" value="0.5, 0.5, 0.5"/>\n'
        "    </bsdf>\n"
        f'    <bsdf type="diffuse" id="{ground_mat}">\n'
        '        <rgb name="reflectance" value="0.5, 0.5, 0.5"/>\n'
        "    </bsdf>\n"
        '    <shape type="ply" id="box">\n'
        '        <string name="filename" value="box.ply"/>\n'
        f'        <ref id="{box_mat}" name="bsdf"/>\n'
        "    </shape>\n"
        '    <shape type="ply" id="ground">\n'
        '        <string name="filename" value="ground.ply"/>\n'
        f'        <ref id="{ground_mat}" name="bsdf"/>\n'
        "    </shape>\n"
        "</scene>\n"
    )
    xml_path = root / SCENE_XML_NAME
    xml_path.write_text(xml_text, encoding="utf-8")
    return xml_path


def write_observed(
    dataset_dir: Path,
    name: str = "obs",
    *,
    seed: int = 0,
    snr_db: float | None = 30.0,
    **impairments: Any,
) -> Path:
    """Apply an observation variant to ``dataset_dir``; return the manifest path."""
    return observe_dataset(
        Path(dataset_dir),
        ImpairmentConfig(**impairments),
        noise=NoiseSpec(snr_db=snr_db),
        seed=seed,
        name=name,
    )


def write_partial(dataset_dir: Path, out_dir: Path, *, summary: str, **options: Any) -> Path:
    """Build a partial dataset; return the ``partial_manifest.json`` path."""
    return build_partial_dataset(Path(dataset_dir), Path(out_dir), summary=summary, **options)


def _validate_members(root: Path, members: Sequence[Mapping[str, Any]]) -> None:
    """Validate bundle members, raising ValueError naming the member id and rule."""
    if not members or len(members) > 1024:
        raise ValueError("bundle 'members' must be non-empty with at most 1024 entries")
    seen_ids: dict[str, Mapping[str, Any]] = {}
    for position, member in enumerate(members):
        unknown = [key for key in member if key not in _ALLOWED_MEMBER_KEYS]
        label = str(member.get("id", f"#{position}"))
        if unknown:
            raise ValueError(f"member {label!r}: unknown key(s) {unknown}")
        for required in ("id", "kind", "path"):
            if required not in member:
                raise ValueError(f"member {label!r}: missing required key {required!r}")
        member_id = member["id"]
        if (
            not isinstance(member_id, str)
            or _MEMBER_ID_RE.fullmatch(member_id) is None
            or member_id in (".", "..")
        ):
            raise ValueError(f"member {label!r}: invalid id {member_id!r}")
        if member_id in seen_ids:
            raise ValueError(f"member {member_id!r}: duplicate id")
        seen_ids[member_id] = member
        if member["kind"] not in MEMBER_KINDS:
            raise ValueError(f"member {member_id!r}: invalid kind {member['kind']!r}")
        path = member["path"]
        if (
            not isinstance(path, str)
            or not path
            or path == "."
            or path.startswith("/")
            or "\\" in path
            or "\x00" in path
            or re.match(r"[A-Za-z]:", path) is not None
        ):
            raise ValueError(f"member {member_id!r}: invalid path {path!r}")
        segments = path.split("/")
        if any(seg in ("", ".", "..") for seg in segments):
            raise ValueError(f"member {member_id!r}: invalid path {path!r}")
        current = Path(root)
        for seg in segments:
            current = current / seg
            if current.is_symlink():
                raise ValueError(f"member {member_id!r}: path {path!r} contains a symlink")
            if not os.path.lexists(current):
                raise ValueError(f"member {member_id!r}: path {path!r} missing on disk")
        target = root / path
        kind = member["kind"]
        if kind in _DIRECTORY_KINDS:
            if not target.is_dir():
                raise ValueError(f"member {member_id!r}: path {path!r} is not a directory")
            marker = target / _MARKER_FILE[kind]
            if not marker.is_file() or marker.is_symlink():
                raise ValueError(
                    f"member {member_id!r}: directory path {path!r} lacks "
                    f"marker {_MARKER_FILE[kind]!r}"
                )
        else:
            if target.is_symlink() or not target.is_file():
                raise ValueError(f"member {member_id!r}: path {path!r} is not a regular file")
            if not path.endswith(".xml"):
                raise ValueError(f"member {member_id!r}: scene path {path!r} must end in .xml")
    dir_paths = [
        (str(member["id"]), str(member["path"]))
        for member in members
        if member["kind"] in _DIRECTORY_KINDS
    ]
    for index, (first_id, first) in enumerate(dir_paths):
        for second_id, second in dir_paths[index + 1 :]:
            if first == second:
                raise ValueError(
                    f"members {first_id!r} and {second_id!r}: duplicate directory path"
                )
            if first.startswith(second + "/") or second.startswith(first + "/"):
                raise ValueError(f"members {first_id!r} and {second_id!r}: nested directory paths")
    id_to_kind = {str(member["id"]): str(member["kind"]) for member in members}
    scenes_for: dict[str, str] = {}
    for member in members:
        member_id = str(member["id"])
        kind = str(member["kind"])
        if "for" in member:
            if kind != "scene":
                raise ValueError(f"member {member_id!r}: 'for' only allowed on scene members")
            target_id = member["for"]
            if target_id not in id_to_kind or id_to_kind[target_id] != "rf_dataset":
                raise ValueError(f"member {member_id!r}: 'for' names unknown or non-rf_dataset id")
            if target_id in scenes_for:
                raise ValueError(
                    f"member {member_id!r}: two scenes with the same 'for' {target_id!r}"
                )
            scenes_for[str(target_id)] = member_id
        if "source" in member:
            if kind not in ("rf_partial", "tomo_run"):
                raise ValueError(
                    f"member {member_id!r}: 'source' only allowed on rf_partial/tomo_run"
                )
            target_id = member["source"]
            if target_id not in id_to_kind or id_to_kind[target_id] != "rf_dataset":
                raise ValueError(
                    f"member {member_id!r}: 'source' names unknown or non-rf_dataset id"
                )


def write_bundle_dir(
    root: Path,
    *,
    members: Sequence[Mapping[str, str]],
    created_by: Mapping[str, Any] | None = None,
    validate: bool = True,
) -> Path:
    """Write ``<root>/bundle.json``; return its path."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    member_list = list(members)
    if validate:
        _validate_members(root, member_list)
    ordered: list[dict[str, Any]] = []
    for member in member_list:
        # Known keys in canonical order; unknown keys (only reachable with
        # validate=False) are kept after them so broken bundles can be written.
        entry = {key: member[key] for key in _MEMBER_KEYS if key in member}
        entry.update({key: value for key, value in member.items() if key not in entry})
        ordered.append(entry)
    payload: dict[str, Any] = {
        "bundle_format_version": 1,
        "members": ordered,
        "created_by": (
            dict(created_by)
            if created_by is not None
            else {"tool": "viewer_bundle_fixtures", "tool_version": 1}
        ),
    }
    bundle_json = root / "bundle.json"
    bundle_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return bundle_json


def write_fixture_bundle(root: Path, *, schema_version: int = 3) -> BundleFixture:
    """Build the canonical valid viewer bundle under ``root``."""
    root = Path(root)
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"root {root} exists and is not empty")
    dataset_dir = root / "dataset"
    truth = write_rf_dataset(dataset_dir, schema_version=schema_version)
    write_optical(dataset_dir, truth)
    observation_names: tuple[str, ...] = ()
    if schema_version == 3:
        write_observed(
            dataset_dir,
            "obs",
            seed=1,
            snr_db=30.0,
            front_to_back_db=20.0,
            element_gain_std_db=0.5,
            element_phase_std_deg=5.0,
            timing_offset_ns=3.0,
        )
        observation_names = ("obs",)
    scene_xml = write_scene(root / "scene")
    partial_p0 = root / "partials" / "p0"
    partial_p1 = root / "partials" / "p1"
    write_partial(
        dataset_dir,
        partial_p0,
        summary="none",
        view_fraction=0.67,
        element_mask_kind="checkerboard",
        subband="4:12",
        seed=0,
    )
    write_partial(dataset_dir, partial_p1, summary="delay", seed=0)
    bundle_json = write_bundle_dir(
        root,
        members=[
            {"id": "dataset", "kind": "rf_dataset", "path": "dataset"},
            {"id": "scene", "kind": "scene", "path": "scene/scene.xml", "for": "dataset"},
            {"id": "p0", "kind": "rf_partial", "path": "partials/p0", "source": "dataset"},
            {"id": "p1", "kind": "rf_partial", "path": "partials/p1"},
        ],
    )
    return BundleFixture(
        root=root,
        bundle_json=bundle_json,
        dataset_dir=dataset_dir,
        scene_xml=scene_xml,
        partial_dirs=(partial_p0, partial_p1),
        observation_names=observation_names,
        truth=truth,
    )


def _collect_tree(src_dir: Path) -> tuple[list[str], dict[str, bytes], set[str]]:
    """Collect sorted entry names, file bytes and dir names under ``src_dir``."""
    src_dir = Path(src_dir)
    dir_names: set[str] = set()
    file_data: dict[str, bytes] = {}
    for dirpath, dirnames, filenames in os.walk(src_dir, followlinks=False):
        dirnames.sort()
        for name in dirnames:
            full = Path(dirpath) / name
            if full.is_symlink():
                raise ValueError(f"symlink not allowed in archive: {full}")
            if not full.is_dir():
                raise ValueError(f"non-regular file in archive: {full}")
            rel = (Path(dirpath) / name).relative_to(src_dir).as_posix()
            dir_names.add(rel)
        for name in sorted(filenames):
            full = Path(dirpath) / name
            if full.is_symlink():
                raise ValueError(f"symlink not allowed in archive: {full}")
            if not full.is_file():
                raise ValueError(f"non-regular file in archive: {full}")
            rel = full.relative_to(src_dir).as_posix()
            file_data[rel] = full.read_bytes()
    names = sorted(dir_names | set(file_data))
    return names, file_data, dir_names


def make_archive(src_dir: Path, out_path: Path, fmt: str, *, root_name: str | None = None) -> Path:
    """Write a deterministic archive of ``src_dir``; return ``out_path``."""
    if fmt not in ARCHIVE_FORMATS:
        raise ValueError(f"unknown archive format {fmt!r}")
    src_dir = Path(src_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    names, file_data, dir_names = _collect_tree(src_dir)
    prefix = "" if root_name is None else f"{root_name}/"
    files = {f"{prefix}{name}": data for name, data in file_data.items()}
    dirs = {f"{prefix}{name}" for name in dir_names}
    if root_name is not None:
        dirs.add(root_name)
    all_names = sorted(dirs | {f"{prefix}{name}" for name in names})
    if fmt == "zip":
        with zipfile.ZipFile(
            out_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for name in all_names:
                if name in dirs:
                    info = _zip_info(name + "/", 0o040755)
                    info.external_attr |= 0x10  # MS-DOS directory flag
                    archive.writestr(info, b"")
                else:
                    archive.writestr(_zip_info(name, 0o100644), files[name])
        return out_path
    with _tar_writer(out_path, fmt) as archive:
        for name in all_names:
            if name in dirs:
                archive.addfile(_tar_entry(name, entry_type=tarfile.DIRTYPE, mode=0o755))
            else:
                archive.addfile(_tar_entry(name, files[name]), io.BytesIO(files[name]))
    return out_path


@contextmanager
def _tar_writer(out_path: Path, fmt: str) -> Iterator[tarfile.TarFile]:
    """Open a PAX tar writer; ``tar.gz`` goes through a gzip stream with mtime 0 and no name."""
    with open(out_path, "wb") as raw:
        if fmt == "tar.gz":
            with (
                gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as gz,
                tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as archive,
            ):
                yield archive
        else:
            with tarfile.open(fileobj=raw, mode="w", format=tarfile.PAX_FORMAT) as archive:
                yield archive


def _zip_info(name: str, mode: int) -> zipfile.ZipInfo:
    """Build one deterministic deflated ZipInfo with Unix ``mode`` bits."""
    info = zipfile.ZipInfo(name, date_time=_ZIP_EPOCH)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = mode << 16
    return info


def _tar_entry(
    name: str,
    data: bytes = b"",
    *,
    entry_type: bytes = tarfile.REGTYPE,
    mode: int = 0o644,
    linkname: str = "",
    devmajor: int = 0,
    devminor: int = 0,
) -> tarfile.TarInfo:
    """Build one deterministic TarInfo entry."""
    info = tarfile.TarInfo(name)
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    info.mode = mode
    info.type = entry_type
    if entry_type in (tarfile.SYMTYPE, tarfile.LNKTYPE):
        info.linkname = linkname
    elif entry_type == tarfile.CHRTYPE:
        info.devmajor = devmajor
        info.devminor = devminor
    info.size = len(data) if entry_type == tarfile.REGTYPE else 0
    return info


def make_malicious_archive(
    out_path: Path,
    case: str,
    *,
    fmt: str = "tar.gz",
    file_count: int = 2000,
    bomb_bytes: int = 8 * 1024 * 1024,
) -> Path:
    """Write a small archive that a safe extractor must reject; return ``out_path``."""
    if case not in MALICIOUS_CASES:
        raise ValueError(f"unknown malicious case {case!r}")
    if fmt not in ARCHIVE_FORMATS:
        raise ValueError(f"unknown archive format {fmt!r}")
    if fmt == "zip" and case in ("hardlink", "device"):
        raise ValueError(f"case {case!r} is not supported for zip")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok_data = b"ok\n"
    if fmt == "zip":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with zipfile.ZipFile(
                out_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
            ) as archive:

                def _add(name: str, data: bytes, mode: int = 0o100644) -> None:
                    archive.writestr(_zip_info(name, mode), data)

                _add("ok.txt", ok_data)
                if case in ("dotdot", "absolute"):
                    _add(str(MALICIOUS_MEMBER_NAMES[case]), b"evil\n")
                elif case == "symlink":
                    _add("link", b"../../etc/passwd", 0o120777)
                elif case == "duplicate":
                    _add("dup.txt", b"first\n")
                    _add("dup.txt", b"second\n")
                elif case == "too_many_files":
                    for index in range(int(file_count)):
                        _add(f"many/{index:06d}.txt", b"")
                elif case == "bomb":
                    _add("zeros.bin", b"\x00" * int(bomb_bytes))
    else:
        with _tar_writer(out_path, fmt) as archive:
            archive.addfile(_tar_entry("ok.txt", ok_data), io.BytesIO(ok_data))
            if case == "dotdot":
                archive.addfile(_tar_entry("../evil.txt", b"evil\n"), io.BytesIO(b"evil\n"))
            elif case == "absolute":
                archive.addfile(_tar_entry("/tmp/evil.txt", b"evil\n"), io.BytesIO(b"evil\n"))
            elif case == "symlink":
                archive.addfile(
                    _tar_entry(
                        "link",
                        entry_type=tarfile.SYMTYPE,
                        mode=0o777,
                        linkname="../../etc/passwd",
                    )
                )
            elif case == "hardlink":
                archive.addfile(
                    _tar_entry("hardlink", entry_type=tarfile.LNKTYPE, linkname="/etc/passwd")
                )
            elif case == "device":
                archive.addfile(
                    _tar_entry(
                        "dev/null",
                        entry_type=tarfile.CHRTYPE,
                        mode=0o666,
                        devmajor=1,
                        devminor=3,
                    )
                )
            elif case == "duplicate":
                archive.addfile(_tar_entry("dup.txt", b"first\n"), io.BytesIO(b"first\n"))
                archive.addfile(_tar_entry("dup.txt", b"second\n"), io.BytesIO(b"second\n"))
            elif case == "too_many_files":
                for index in range(int(file_count)):
                    name = f"many/{index:06d}.txt"
                    archive.addfile(_tar_entry(name, b""), io.BytesIO(b""))
            elif case == "bomb":
                data = b"\x00" * int(bomb_bytes)
                archive.addfile(_tar_entry("zeros.bin", data), io.BytesIO(data))
    return out_path


def write_broken_dataset(root: Path, case: str) -> Path:
    """Write the default v3 dataset with one manifest edit breaking the reader."""
    if case not in BROKEN_CASES:
        raise ValueError(f"unknown broken case {case!r}")
    root = Path(root)
    write_rf_dataset(root)
    manifest_path = root / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if case == "bad_schema_version":
        manifest["schema_version"] = 99
    elif case == "missing_artifact":
        del manifest["views"][0]["artifacts"]["aperture_cfr"]
    elif case == "bs_order_mismatch":
        manifest["views"][0]["bs"] = list(reversed(manifest["views"][0]["bs"]))
    elif case == "duplicate_view_id":
        manifest["views"][1]["view_id"] = manifest["views"][0]["view_id"]
    elif case == "bad_axis_order":
        manifest["raw_observation"]["axis_order"] = list(
            reversed(manifest["raw_observation"]["axis_order"])
        )
    elif case == "bad_position":
        manifest["views"][0]["position_m"] = [0.0, 0.0]
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return root


def _build_parser() -> argparse.ArgumentParser:
    """Return the CLI argument parser."""
    parser = argparse.ArgumentParser(description="Build viewer-bundle fixtures.")
    parser.add_argument("out", help="output directory")
    parser.add_argument("--archive", default=None, choices=["zip", "tar", "tar.gz", "dir"])
    parser.add_argument("--schema", default=None, type=int, choices=[2, 3])
    parser.add_argument("--broken", default=None, choices=list(BROKEN_CASES))
    parser.add_argument("--malicious", default=None, choices=list(MALICIOUS_CASES))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Build one fixture artifact; print its path and return 0."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.broken is not None and args.malicious is not None:
        parser.error("--broken cannot be combined with --malicious")
    if args.schema is not None and (args.broken is not None or args.malicious is not None):
        parser.error("--schema cannot be combined with --broken or --malicious")
    archive = args.archive
    if archive is None:
        archive = "tar.gz" if args.malicious is not None else "zip"
    if args.malicious is not None and archive == "dir":
        parser.error("--malicious cannot be combined with --archive dir")
    if args.malicious in ("hardlink", "device") and archive == "zip":
        parser.error(f"--malicious {args.malicious} is not supported for zip")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.malicious is not None:
        ext = {"zip": ".zip", "tar": ".tar", "tar.gz": ".tar.gz"}[archive]
        out_path = out_dir / f"malicious_{args.malicious}{ext}"
        try:
            make_malicious_archive(out_path, args.malicious, fmt=archive)
        except ValueError as exc:
            parser.error(str(exc))
        print(str(out_path))
        return 0
    if args.broken is not None:
        name = f"broken_{args.broken}"
    elif args.schema == 2:
        name = "bundle_v2"
    else:
        name = "bundle"
    if archive == "dir":
        bundle_path = out_dir / name
        if bundle_path.exists():
            if (bundle_path / "bundle.json").is_file():
                shutil.rmtree(bundle_path)
            else:
                parser.error(f"output directory {bundle_path} exists and is not a bundle")
        bundle_path.mkdir(parents=True, exist_ok=True)
        if args.broken is not None:
            write_broken_dataset(bundle_path / "dataset", args.broken)
            write_bundle_dir(
                bundle_path,
                members=[{"id": "dataset", "kind": "rf_dataset", "path": "dataset"}],
            )
        else:
            write_fixture_bundle(bundle_path, schema_version=int(args.schema) if args.schema else 3)
        print(str(bundle_path))
        return 0
    ext = {"zip": ".zip", "tar": ".tar", "tar.gz": ".tar.gz"}[archive]
    out_path = out_dir / f"{name}{ext}"
    with tempfile.TemporaryDirectory() as tmp:
        tmp_bundle = Path(tmp) / name
        if args.broken is not None:
            write_broken_dataset(tmp_bundle / "dataset", args.broken)
            write_bundle_dir(
                tmp_bundle,
                members=[{"id": "dataset", "kind": "rf_dataset", "path": "dataset"}],
            )
        else:
            write_fixture_bundle(tmp_bundle, schema_version=int(args.schema) if args.schema else 3)
        make_archive(tmp_bundle, out_path, archive, root_name=name)
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
