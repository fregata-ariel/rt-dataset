"""Build ``tomography_gt.npz`` (design §6.1) from a dataset's path GT and scene mesh.

Sionna-free: the path ground truth is read through the shared manifest reader
and the PLY meshes of the Mitsuba scene XML with trimesh (imported lazily).
"""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ElementTree
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from plateau_rt.application import rf_tomography_io
from plateau_rt.application.rf_dataset_manifest import (
    ManifestError,
    RFDatasetManifest,
    load_rf_dataset_manifest,
)
from plateau_rt.domain.rf_tomography import gt
from plateau_rt.domain.rf_tomography.antenna import PATTERN_KINDS
from plateau_rt.domain.rf_tomography.geometry import CaptureGeometry

__all__ = [
    "SceneMesh",
    "build_tomography_gt",
    "default_surface_roi",
    "load_path_gt",
    "load_scene_mesh",
    "resolve_scene_xml",
    "summarize_tomography_gt",
    "write_tomography_gt",
]


@dataclass(frozen=True)
class SceneMesh:
    """A Mitsuba PLY scene mesh: triangles, per-triangle object index and provenance."""

    triangles: np.ndarray  # [T,3,3] float64
    triangle_object: np.ndarray  # [T] int64 index into object_names
    object_names: tuple[str, ...]  # <shape id="..."> in XML order
    sha256: str


def load_path_gt(manifest: RFDatasetManifest) -> gt.PathGT:
    """Load the canonical path ground truth referenced by ``manifest``.

    Raises :class:`ManifestError` when the manifest has no path GT or its schema
    is not the canonical mode.
    """
    path_gt = manifest.path_geometry_gt
    if path_gt is None:
        raise ManifestError("manifest has no 'path_geometry_gt' artifact")
    schema = path_gt.load_schema()
    mode = schema.get("mode")
    if mode != "canonical":
        raise ManifestError(
            f"path GT schema mode must be 'canonical', got {mode!r}; "
            "explicit-array datasets are not supported"
        )
    object_names = tuple(str(name) for name in schema.get("object_names", ()))
    return gt.PathGT.from_arrays(path_gt.load_arrays(), object_names)


def _scene_candidates(manifest: RFDatasetManifest, source: str) -> list[Path]:
    """Return the candidate scene paths for a relative ``source`` in search order."""
    candidates = [Path.cwd() / source, manifest.root / source]
    for ancestor in manifest.root.parents:
        candidates.append(ancestor / source)
    return candidates


def resolve_scene_xml(manifest: RFDatasetManifest, override: Path | str | None = None) -> Path:
    """Resolve the scene XML of ``manifest`` (or an explicit ``override``) to an existing path."""
    if override is not None:
        path = Path(override)
        if not path.exists():
            raise FileNotFoundError(f"scene XML not found at {path}")
        return path
    source = manifest.source_scene
    if source is None:
        raise FileNotFoundError("manifest has no 'source_scene' to resolve")
    path = Path(source)
    if path.is_absolute():
        if path.exists():
            return path
        raise FileNotFoundError(f"scene XML not found at {path}")
    candidates = _scene_candidates(manifest, source)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    tried = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"scene XML {source!r} not found; tried cwd, dataset root and ancestors: {tried}"
    )


def load_scene_mesh(xml_path: Path | str) -> SceneMesh:
    """Load the ``<shape type="ply">`` meshes of a Mitsuba XML into one triangle soup."""
    import trimesh  # local import: the module must stay importable without trimesh

    location = Path(xml_path)
    root = ElementTree.parse(location).getroot()
    directory = location.parent
    triangles: list[np.ndarray] = []
    triangle_object: list[np.ndarray] = []
    object_names: list[str] = []
    digest_pairs: list[list[str]] = []
    for shape in root.iter("shape"):
        if shape.get("type") != "ply":
            continue
        shape_id = shape.get("id")
        filename = None
        for child in shape.findall("string"):
            if child.get("name") == "filename":
                filename = child.get("value")
                break
        if shape_id is None or filename is None:
            raise ValueError(f"ply shape in {location} must have an id and a filename")
        ply_path = directory / filename
        mesh = trimesh.load(ply_path, force="mesh", process=False)
        vertices = np.asarray(mesh.vertices, dtype=np.float64)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        object_index = len(object_names)
        triangles.append(vertices[faces])
        triangle_object.append(np.full(faces.shape[0], object_index, dtype=np.int64))
        object_names.append(str(shape_id))
        digest_pairs.append([str(shape_id), rf_tomography_io.sha256_file(ply_path)])

    if triangles:
        stacked = np.concatenate(triangles, axis=0)
        objects = np.concatenate(triangle_object, axis=0)
    else:
        stacked = np.empty((0, 3, 3), dtype=np.float64)
        objects = np.empty(0, dtype=np.int64)
    sha256 = _sha256_of_json(digest_pairs)
    return SceneMesh(
        triangles=stacked,
        triangle_object=objects,
        object_names=tuple(object_names),
        sha256=sha256,
    )


def _sha256_of_json(payload: Any) -> str:
    """Return the hex SHA-256 of the UTF-8 JSON encoding of ``payload``."""
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()


def default_surface_roi(
    geom: CaptureGeometry, points: np.ndarray, margin: float = 5.0
) -> tuple[np.ndarray, np.ndarray]:
    """Return the default ``(lo, hi)`` surface ROI around poses and interaction points."""
    blocks = [geom.ue_pos, geom.bs_pos]
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if pts.shape[0] > 0:
        blocks.append(pts)
    stacked = np.concatenate(blocks, axis=0)
    low = stacked.min(axis=0) - float(margin)
    high = stacked.max(axis=0) + float(margin)
    return low, high


def _str_scalar(value: Any) -> str:
    """Return a Python ``str`` from a (possibly 0-d) numpy string scalar."""
    return str(np.asarray(value).reshape(-1)[0]) if np.asarray(value).size else ""


def build_tomography_gt(
    dataset: Path | str,
    *,
    scene: Path | str | None = None,
    surfaces: bool = True,
    surface_spacing: float = gt.DEFAULT_SURFACE_SPACING_M,
    surface_margin: float = 5.0,
    surface_roi: tuple[Sequence[float], Sequence[float]] | None = None,
    los_polarization: str = "none",
    cluster_tol_m: float = gt.DEFAULT_CLUSTER_TOL_M,
    support_radius: float = gt.DEFAULT_SUPPORT_RADIUS_M,
) -> dict[str, np.ndarray]:
    """Assemble the tomography ground-truth array dict for a dataset."""
    data = rf_tomography_io.load_dataset(dataset)
    pattern = data.tx_pattern
    if pattern is None or pattern not in PATTERN_KINDS:
        raise ValueError(f"dataset tx_pattern must be one of {PATTERN_KINDS}, got {pattern!r}")
    manifest = data.manifest
    path = load_path_gt(manifest)
    result = gt.path_ground_truth(
        path,
        data.geom,
        pattern=pattern,
        los_polarization=los_polarization,
        cluster_tol_m=cluster_tol_m,
    )
    path_gt_path = manifest.path_geometry_gt
    assert path_gt_path is not None  # guaranteed by load_path_gt
    result["schema"] = np.asarray(gt.GT_SCHEMA)
    result["path_type_names"] = np.asarray(gt.PATH_TYPE_NAMES)
    result["path_object_names"] = np.asarray(path.object_names)
    result["view_ids"] = np.asarray(manifest.view_ids)
    result["bs_ids"] = np.asarray(manifest.bs_ids)
    result["source_path_gt_sha256"] = np.asarray(rf_tomography_io.sha256_file(path_gt_path.path))
    result["pattern"] = np.asarray(pattern)
    result["los_polarization"] = np.asarray(los_polarization)
    result["cluster_tol_m"] = np.asarray(cluster_tol_m, dtype=np.float64)
    result["delay_period_s"] = np.asarray(data.geom.delay_period, dtype=np.float64)

    if surfaces:
        mesh = load_scene_mesh(resolve_scene_xml(manifest, scene))
        if surface_roi is not None:
            low = np.asarray(surface_roi[0], dtype=np.float64).reshape(3)
            high = np.asarray(surface_roi[1], dtype=np.float64).reshape(3)
        else:
            low, high = default_surface_roi(data.geom, result["interaction_points"], surface_margin)
        specular_points = result["interaction_points"][
            (result["interaction_type"] & gt.INTERACTION_SPECULAR) != 0
        ]
        result.update(
            gt.surface_ground_truth(
                mesh.triangles,
                mesh.triangle_object,
                data.geom.bs_pos,
                data.geom.ue_pos,
                specular_points,
                spacing=surface_spacing,
                roi=(low, high),
                support_radius=support_radius,
            )
        )
        result["surface_object_names"] = np.asarray(mesh.object_names)
        result["source_mesh_sha256"] = np.asarray(mesh.sha256)
        result["surface_spacing_m"] = np.asarray(surface_spacing, dtype=np.float64)
        result["specular_support_radius_m"] = np.asarray(support_radius, dtype=np.float64)
    else:
        result["source_mesh_sha256"] = np.asarray("")
    return result


def _relative_artifact(out: Path, root: Path) -> str:
    """Return ``out`` relative to ``root`` as POSIX when inside it, else absolute."""
    try:
        return out.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(out.resolve())


def write_tomography_gt(
    dataset: Path | str,
    *,
    out: Path | str | None = None,
    register: bool = True,
    **kwargs: Any,
) -> Path:
    """Build, write and optionally register ``tomography_gt.npz``; return its path."""
    manifest = load_rf_dataset_manifest(dataset)
    data = build_tomography_gt(dataset, **kwargs)
    out_path = Path(out) if out is not None else manifest.root / rf_tomography_io.GT_FILE_NAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **data)
    if register:
        raw = json.loads(manifest.manifest_path.read_text(encoding="utf-8"))
        mesh_sha = _str_scalar(data["source_mesh_sha256"])
        raw["tomography_gt"] = {
            "artifact": _relative_artifact(out_path, manifest.root),
            "source_path_gt_sha256": _str_scalar(data["source_path_gt_sha256"]),
            "source_mesh_sha256": mesh_sha if mesh_sha else None,
            "schema": gt.GT_SCHEMA,
        }
        manifest.manifest_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    return out_path


def _percentile(values: np.ndarray, q: float) -> float | None:
    """Return the q-th percentile of finite ``|values|`` or None when none are finite."""
    finite = np.abs(np.asarray(values, dtype=np.float64))
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    return float(np.percentile(finite, q))


def summarize_tomography_gt(gt_arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Return a JSON-serialisable summary of a tomography ground-truth dict."""
    vs_pos = np.asarray(gt_arrays["vs_pos"])
    vs_order = np.asarray(gt_arrays["vs_order"])
    vs_bs = np.asarray(gt_arrays["vs_bs"])
    vs_spread = np.asarray(gt_arrays["vs_spread"])
    los_visible = np.asarray(gt_arrays["los_visible"], dtype=bool)
    phase = np.asarray(gt_arrays["los_phase_model_error"], dtype=np.float64)
    amp_db = np.asarray(gt_arrays["los_amp_model_error_db"], dtype=np.float64)

    summary: dict[str, Any] = {
        "num_vs": int(vs_pos.shape[0]),
        "vs_per_order": {
            int(order): int(np.count_nonzero(vs_order == order)) for order in np.unique(vs_order)
        },
        "vs_per_bs": {int(bs): int(np.count_nonzero(vs_bs == bs)) for bs in np.unique(vs_bs)},
        "max_vs_spread_m": float(np.max(vs_spread)) if vs_spread.size else 0.0,
        "los_visible": los_visible.tolist(),
        "num_los_visible": int(np.count_nonzero(los_visible)),
        "los_phase_error_deg_p50": _percentile(np.degrees(phase), 50.0),
        "los_phase_error_deg_p90": _percentile(np.degrees(phase), 90.0),
        "los_amp_error_db_p50": _percentile(amp_db, 50.0),
        "los_amp_error_db_p90": _percentile(amp_db, 90.0),
        "num_planes": int(np.asarray(gt_arrays["plane_normal"]).shape[0]),
        "num_interactions": int(np.asarray(gt_arrays["interaction_points"]).shape[0]),
        "num_beyond_period": int(np.count_nonzero(np.asarray(gt_arrays["beyond_period"]))),
    }
    if "surface_samples" in gt_arrays:
        summary["num_surface_samples"] = int(np.asarray(gt_arrays["surface_samples"]).shape[0])
        summary["num_surface_observable"] = int(
            np.count_nonzero(np.asarray(gt_arrays["surface_observable"]))
        )
        summary["num_surface_specular_support"] = int(
            np.count_nonzero(np.asarray(gt_arrays["surface_specular_support"]))
        )
    return summary
