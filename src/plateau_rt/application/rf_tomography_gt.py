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
from plateau_rt.domain.rf_tomography import gt, metrics
from plateau_rt.domain.rf_tomography.antenna import PATTERN_KINDS
from plateau_rt.domain.rf_tomography.geometry import CaptureGeometry, VoxelGrid

__all__ = [
    "DETECTION_GATES_M",
    "LOS_STRATA",
    "SURFACE_STRATA",
    "VS_DYNAMIC_RANGE_DB",
    "VS_STRATA_KINDS",
    "SceneMesh",
    "build_tomography_gt",
    "default_surface_roi",
    "load_path_gt",
    "load_scene_mesh",
    "load_tomography_gt",
    "resolve_scene_xml",
    "score_planes",
    "score_surface_map",
    "summarize_tomography_gt",
    "vs_detectable",
    "vs_recall_strata",
    "vs_strata",
    "write_tomography_gt",
]


VS_DYNAMIC_RANGE_DB: float = 30.0
DETECTION_GATES_M: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0)
SURFACE_STRATA: tuple[str, ...] = ("all", "observable", "specular")
LOS_STRATA: tuple[str, ...] = ("los_visible", "los_blocked")
VS_STRATA_KINDS: tuple[str, ...] = ("mechanism", "order", "los")


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
    vs_keys = ("path_power", "path_type", "vs_power", "vs_visibility", "vs_bs")
    if all(key in gt_arrays for key in vs_keys):
        summary["num_vs_detectable"] = int(np.count_nonzero(vs_detectable(gt_arrays)))
        if all(key in gt_arrays for key in ("vs_path_type", "los_visible")):
            mechanism = vs_strata(gt_arrays)["mechanism"]
            summary["vs_per_mechanism"] = {
                str(label): int(np.count_nonzero(mechanism == label))
                for label in sorted(set(mechanism.tolist()))
            }
    return summary


def load_tomography_gt(path: Path | str) -> dict[str, np.ndarray]:
    """Load a ``tomography_gt.npz`` file into a plain array dict."""
    location = Path(path)
    try:
        with np.load(location, allow_pickle=False) as payload:
            arrays = {name: np.asarray(payload[name]) for name in payload.files}
    except OSError as error:
        raise ValueError(f"tomography GT could not be loaded from {location}: {error}") from error
    if "schema" not in arrays or str(arrays["schema"]) != gt.GT_SCHEMA:
        raise ValueError(f"{location} has no valid tomography GT schema {gt.GT_SCHEMA!r}")
    return arrays


def _vs_shapes(
    gt_arrays: Mapping[str, np.ndarray],
) -> tuple[int, int, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(V, B, M, path_type, path_power, vs_power, vs_visibility, vs_bs)``."""
    try:
        path_type = np.asarray(gt_arrays["path_type"])
        path_power = np.asarray(gt_arrays["path_power"], dtype=np.float64)
        vs_power = np.asarray(gt_arrays["vs_power"], dtype=np.float64)
        vs_visibility = np.asarray(gt_arrays["vs_visibility"], dtype=bool)
        vs_bs = np.asarray(gt_arrays["vs_bs"]).reshape(-1)
    except (TypeError, ValueError) as error:
        raise ValueError("tomography GT VS arrays have invalid dtypes") from error
    if path_type.ndim != 3 or path_power.shape != path_type.shape:
        raise ValueError("path_type and path_power must share a [V, B, P] shape")
    num_views, num_bs = path_type.shape[0], path_type.shape[1]
    num_vs = int(vs_bs.shape[0])
    if vs_power.shape != (num_vs, num_views) or vs_visibility.shape != (num_vs, num_views):
        raise ValueError("vs_power and vs_visibility must have shape [M, V]")
    if num_vs > 0 and (
        not np.all(np.isfinite(vs_power))
        or np.any(vs_power < 0.0)
        or np.any(vs_bs < 0)
        or np.any(vs_bs >= num_bs)
    ):
        raise ValueError("vs_power must be finite and >= 0 with vs_bs in [0, B)")
    return num_views, num_bs, num_vs, path_type, path_power, vs_power, vs_visibility, vs_bs


def vs_detectable(
    gt_arrays: Mapping[str, np.ndarray],
    dynamic_range_db: float = VS_DYNAMIC_RANGE_DB,
    captures: np.ndarray | None = None,
) -> np.ndarray:
    """Flag virtual sources within the dynamic range of their capture's peak."""
    num_views, num_bs, num_vs, path_type, path_power, vs_power, vs_visibility, vs_bs = _vs_shapes(
        gt_arrays
    )
    if num_vs == 0:
        return np.zeros((0,), dtype=bool)
    if captures is None:
        keep = np.ones((num_views, num_bs), dtype=bool)
    else:
        try:
            keep = np.asarray(captures, dtype=bool)
        except (TypeError, ValueError) as error:
            raise ValueError("captures must have shape [V, B]") from error
        if keep.shape != (num_views, num_bs):
            raise ValueError(f"captures must have shape [{num_views}, {num_bs}], got {keep.shape}")
    table = np.zeros((num_vs, num_views * num_bs), dtype=np.float64)
    view_index = np.arange(num_views)
    table[np.arange(num_vs)[:, None], view_index[None, :] * num_bs + vs_bs[:, None]] = np.where(
        vs_visibility, vs_power, 0.0
    )
    masked = np.where(path_type >= 0, path_power, -np.inf)
    best = np.max(masked, axis=-1)
    reference = np.where(np.isfinite(best), best, 0.0).reshape(-1)
    flat_keep = keep.reshape(-1).astype(np.float64)
    table = table * flat_keep[None, :]
    reference = reference * flat_keep
    return metrics.detectable_mask(table, dynamic_range_db, reference=reference)


def vs_strata(gt_arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Label every virtual source by mechanism, order and LoS visibility."""
    num_views, _, num_vs, _, _, vs_power, vs_visibility, vs_bs = _vs_shapes(gt_arrays)
    try:
        vs_path_type = np.asarray(gt_arrays["vs_path_type"]).reshape(num_vs, num_views)
        vs_order = np.asarray(gt_arrays["vs_order"]).reshape(num_vs)
        los_visible = np.asarray(gt_arrays["los_visible"], dtype=bool)
    except (TypeError, ValueError) as error:
        raise ValueError("tomography GT VS label arrays have invalid shapes") from error
    if los_visible.ndim != 2 or los_visible.shape[0] != num_views:
        raise ValueError(f"los_visible must have shape [{num_views}, B]")
    if num_vs > 0 and np.any(vs_bs >= los_visible.shape[1]):
        raise ValueError("vs_bs must index los_visible columns")
    mechanism: list[str] = []
    for m in range(num_vs):
        seen = np.flatnonzero(vs_visibility[m])
        if seen.shape[0] == 0:
            mechanism.append("none")
            continue
        ranked = seen[np.argmax(vs_power[m, seen])]
        kind = int(vs_path_type[m, int(ranked)])
        mechanism.append(gt.PATH_TYPE_NAMES[kind] if kind >= 0 else "none")
    order: list[str] = []
    for value in vs_order.tolist():
        rank = int(value)
        if rank < 0:
            raise ValueError("vs_order must be >= 0")
        order.append(str(rank) if rank <= 2 else "3+")
    los: list[str] = []
    for m in range(num_vs):
        base = int(vs_bs[m])
        visible = bool(np.any(vs_visibility[m] & los_visible[:, base]))
        los.append("los_visible" if visible else "los_blocked")
    return {
        "mechanism": np.asarray(mechanism),
        "order": np.asarray(order),
        "los": np.asarray(los),
    }


def vs_recall_strata(
    detections: np.ndarray,
    gt_arrays: Mapping[str, np.ndarray],
    *,
    detectable: np.ndarray | None = None,
    gates: Sequence[float] = DETECTION_GATES_M,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Stratified VS recall of ``detections`` at every gate."""
    try:
        det = np.asarray(detections, dtype=np.float64).reshape(-1, 3)
    except (TypeError, ValueError) as error:
        raise ValueError("detections must have shape [N, 3]") from error
    gate_list = [float(gate) for gate in gates]
    for gate in gate_list:
        if not np.isfinite(gate) or gate <= 0.0:
            raise ValueError("gates must be finite and > 0")
    positions = np.asarray(gt_arrays["vs_pos"], dtype=np.float64).reshape(-1, 3)
    num_vs = int(positions.shape[0])
    if detectable is None:
        selected = vs_detectable(gt_arrays)
    else:
        try:
            selected = np.asarray(detectable, dtype=bool)
        except (TypeError, ValueError) as error:
            raise ValueError("detectable must have shape [M]") from error
        if selected.shape != (num_vs,):
            raise ValueError(f"detectable must have shape [{num_vs}], got {selected.shape}")
    strata = vs_strata(gt_arrays)
    gt_points = positions[selected]
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for kind in VS_STRATA_KINDS:
        scoped = strata[kind][selected]
        per_gate = [
            metrics.stratified_recall(metrics.match(det, gt_points, gate), scoped)
            for gate in gate_list
        ]
        per_label: dict[str, dict[str, Any]] = {}
        for label in sorted(set(scoped.tolist())):
            per_label[label] = {
                "num_gt": int(per_gate[0][label]["num_gt"]) if per_gate else 0,
                "recall": {
                    f"{gate}": float(per_gate[i][label]["recall"])
                    for i, gate in enumerate(gate_list)
                },
            }
        out[kind] = per_label
    return out


def score_surface_map(
    gt_arrays: Mapping[str, np.ndarray],
    density: np.ndarray,
    grid: VoxelGrid,
    *,
    rel_threshold: float = metrics.MAP_REL_THRESHOLD,
    thresholds: Sequence[float] = metrics.SURFACE_THRESHOLDS_M,
) -> dict[str, Any]:
    """Score a BV-space map against the mesh surface samples per stratum."""
    for key in ("surface_samples", "surface_observable", "surface_specular_support"):
        if key not in gt_arrays:
            raise KeyError(f"gt_arrays is missing {key!r}")
    try:
        samples = np.asarray(gt_arrays["surface_samples"], dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("surface_samples must be a real [S, 3] array") from error
    if samples.size == 0:
        samples = np.zeros((0, 3), dtype=np.float64)
    if samples.ndim != 2 or samples.shape[1] != 3:
        raise ValueError("surface_samples must have shape [S, 3]")
    if samples.shape[0] > 0 and not np.all(np.isfinite(samples)):
        raise ValueError("surface_samples must contain only finite values")
    count = samples.shape[0]
    masks: dict[str, np.ndarray] = {"all": np.ones((count,), dtype=bool)}
    for key, full in (
        ("observable", "surface_observable"),
        ("specular", "surface_specular_support"),
    ):
        try:
            masks[key] = np.asarray(gt_arrays[full], dtype=bool).reshape(count)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{full} must have shape [{count}]") from error
    gate_list = [float(item) for item in thresholds]
    pred, pred_w = metrics.map_point_cloud(density, grid, rel_threshold)  # validates density
    raw = np.asarray(density, dtype=np.float64)
    finite = np.isfinite(raw)
    energy_pos = np.asarray(grid.centers()[np.flatnonzero(finite)], dtype=np.float64)
    low = float(np.min(raw[finite])) if bool(np.any(finite)) else 0.0
    energy_w = np.asarray(raw[finite] - low, dtype=np.float64)
    origin = np.asarray(grid.origin, dtype=np.float64)
    spacing = float(grid.spacing)
    shape = np.asarray(grid.shape, dtype=np.float64)
    lo = origin - spacing / 2.0
    hi = origin + (shape - 1.0) * spacing + spacing / 2.0
    in_box = (
        np.all((samples >= lo) & (samples <= hi), axis=1)
        if count > 0
        else np.zeros((0,), dtype=bool)
    )
    strata: dict[str, Any] = {}
    for stratum in SURFACE_STRATA:
        selected = masks[stratum]
        ref = samples[selected]
        recall_mask = in_box[selected]
        report = metrics.surface_report(
            pred,
            ref,
            gate_list,
            pred_weight=pred_w,
            recall_mask=recall_mask,
            energy_pos=energy_pos,
            energy_weight=energy_w,
        )
        strata[stratum] = {
            "num_ref": int(np.count_nonzero(recall_mask)),
            "num_pred": int(pred.shape[0]),
            "prf": {
                f"{gate}": {
                    "precision": float(score.precision),
                    "recall": float(score.recall),
                    "f": float(score.f_score),
                }
                for gate, score in zip(gate_list, report.scores, strict=True)
            },
            "chamfer": {
                "accuracy": float(report.chamfer.accuracy),
                "completeness": float(report.chamfer.completeness),
                "chamfer": float(report.chamfer.chamfer),
            },
            "energy_within": {
                f"{gate}": float(value)
                for gate, value in zip(gate_list, report.energy, strict=True)
            },
        }
    return {
        "rel_threshold": float(rel_threshold),
        "box": [lo.tolist(), hi.tolist()],
        "strata": strata,
    }


def score_planes(
    gt_arrays: Mapping[str, np.ndarray],
    est_normal: np.ndarray,
    est_offset: np.ndarray,
    *,
    max_angle_deg: float = metrics.PLANE_MAX_ANGLE_DEG,
    max_offset_m: float = metrics.PLANE_MAX_OFFSET_M,
) -> dict[str, Any]:
    """Match estimated planes against the GT reflection planes."""
    gt_normal = np.asarray(gt_arrays["plane_normal"], dtype=np.float64)
    gt_offset = np.asarray(gt_arrays["plane_offset"], dtype=np.float64)
    num_gt = int(gt_normal.shape[0])
    points = np.asarray(gt_arrays["interaction_points"], dtype=np.float64).reshape(-1, 3)
    planes = np.asarray(gt_arrays["interaction_plane"]).reshape(-1)
    if points.shape[0] != planes.shape[0]:
        raise ValueError("interaction_points and interaction_plane must share their length")
    anchors = np.zeros((num_gt, 3), dtype=np.float64)
    for plane in range(num_gt):
        selected = points[planes == plane]
        if selected.shape[0] > 0:
            anchors[plane] = np.mean(selected, axis=0)
    matching = metrics.match_planes(
        est_normal,
        est_offset,
        gt_normal,
        gt_offset,
        max_angle_deg=max_angle_deg,
        max_offset_m=max_offset_m,
        gt_anchor=anchors,
    )
    matches = [
        {
            "est": int(est),
            "gt": int(gt_idx),
            "angle_deg": float(np.degrees(angle)),
            "offset_m": float(offset),
        }
        for est, gt_idx, angle, offset in zip(
            matching.est_idx, matching.gt_idx, matching.angle, matching.offset, strict=True
        )
    ]
    return {
        "num_est": int(matching.num_est),
        "num_gt": int(matching.num_gt),
        "tp": int(matching.tp),
        "fp": int(matching.fp),
        "fn": int(matching.fn),
        **matching.summary(),
        "matches": matches,
    }
