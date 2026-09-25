"""Dataset loading, ground truth and output files of the tomography benchmark; Sionna-free."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from plateau_rt.application.rf_dataset_manifest import (
    ManifestError,
    RFDatasetManifest,
    load_rf_dataset_manifest,
)
from plateau_rt.domain.rf_tomography.geometry import CaptureGeometry

RESULT_SCHEMA: str = "rf_tomo_result/1"
RUN_MANIFEST_SCHEMA: str = "rf_tomo_run/1"
RESULTS_FILE: str = "results.jsonl"
RUN_MANIFEST_FILE: str = "run_manifest.json"
RECON_DIR: str = "recon"
GT_FILE_NAME: str = "tomography_gt.npz"
STAGES: tuple[str, ...] = ("E1", "ROI", "E2", "planned")
STATUSES: tuple[str, ...] = ("ok", "n/a", "error")
RESULT_KEYS: tuple[str, ...] = (
    "schema",
    "scene",
    "realization",
    "config",
    "node",
    "subset",
    "sync",
    "column",
    "lattices",
    "budget",
    "track",
    "space",
    "strategy",
    "stage",
    "solver",
    "status",
    "reason",
    "n_iter",
    "hyper",
    "n_detections",
    "metrics",
    "gauge_errors",
    "ill_posed",
    "runtime_s",
    "recon",
)
RUN_MANIFEST_KEYS: tuple[str, ...] = (
    "schema",
    "created_utc",
    "options",
    "versions",
    "dataset",
    "suite",
    "grid",
    "tracks",
    "configs",
    "strategies",
    "seeds",
    "noise",
    "gt",
    "results",
    "runtime_s",
)


@dataclass(frozen=True)
class TomographyDataset:
    """A dataset loaded for the benchmark (clean data, geometry and provenance)."""

    manifest: RFDatasetManifest
    name: str
    y_clean: np.ndarray
    geom: CaptureGeometry
    los_visible: np.ndarray
    los_visible_source: str
    tx_pattern: str | None
    target: np.ndarray
    hashes: Mapping[str, str]


def sha256_file(path: Path | str) -> str:
    """Return the hex SHA-256 of the file at ``path``."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_geometry(manifest: RFDatasetManifest) -> CaptureGeometry:
    """Build the capture geometry of ``manifest`` from its poses and config."""
    config = manifest.config
    try:
        spacing_h = float(config["horizontal_spacing_lambda"])
        spacing_v = float(config["vertical_spacing_lambda"])
    except KeyError as error:
        raise ValueError(f"manifest config is missing {error}") from error
    if spacing_h != spacing_v:
        raise ValueError(
            f"horizontal and vertical spacing must match, got {spacing_h} and {spacing_v}"
        )
    orientations: list[Any] = []
    for view in manifest.views:
        if view.orientation_rad is None:
            raise ValueError(f"view {view.view_id!r} has no orientation_rad")
        orientations.append(view.orientation_rad)
    geom = CaptureGeometry.from_orientations(
        ue_pos=[view.position_m for view in manifest.views],
        ue_orientations=orientations,
        bs_pos=[bs.position_m for bs in manifest.base_stations],
        f_c=manifest.carrier_frequency_hz,
        bandwidth=float(config["bandwidth_hz"]),
        num_bins=manifest.num_frequency_bins,
        aperture_shape=(manifest.rx_rows, manifest.rx_cols),
        spacing_lambda=spacing_h,
        bs_look_at=np.array([bs.look_at_m for bs in manifest.base_stations]),
    )
    manifest_offsets = np.asarray(manifest.frequency_offsets_hz, dtype=np.float64)
    mismatch = float(np.max(np.abs(geom.freq_offsets - manifest_offsets)))
    if mismatch > 1e-6 * geom.delta_f:
        raise ValueError(f"frequency offsets mismatch: max deviation {mismatch}")
    return geom


def los_visibility(manifest: RFDatasetManifest) -> tuple[np.ndarray, str]:
    """Return the ``[V, B]`` LoS visibility and its source (``path_gt`` or ``assumed``)."""
    shape = (manifest.num_views, manifest.num_bs)
    try:
        path_gt = manifest.path_geometry_gt
        if path_gt is None:
            raise ManifestError("no path geometry ground truth")
        if path_gt.array_axes("valid") != ("view", "bs", "path"):
            raise ManifestError("unexpected axes for 'valid'")
        if path_gt.array_axes("num_interactions") != ("view", "bs", "path"):
            raise ManifestError("unexpected axes for 'num_interactions'")
        arrays = path_gt.load_arrays()
        valid = np.asarray(arrays["valid"], dtype=bool)
        num_interactions = np.asarray(arrays["num_interactions"])
        visible = np.any(valid & (num_interactions == 0), axis=-1)
        if visible.shape != shape:
            raise ManifestError(f"unexpected visibility shape {visible.shape}")
        return np.asarray(visible, dtype=bool), "path_gt"
    except (ManifestError, OSError, KeyError, ValueError):
        return np.ones(shape, dtype=bool), "assumed"


def load_dataset(path: Path | str) -> TomographyDataset:
    """Load the dataset at ``path`` (directory or manifest) for the benchmark."""
    manifest = load_rf_dataset_manifest(path)
    y_clean = np.stack([manifest.load_aperture_cfr(view) for view in manifest.views]).astype(
        np.complex128, copy=False
    )
    look_ats = [view.look_at_m for view in manifest.views if view.look_at_m is not None]
    if look_ats:
        target = np.mean(np.asarray(look_ats, dtype=np.float64), axis=0)
    else:
        target = np.mean(
            np.asarray([bs.look_at_m for bs in manifest.base_stations], dtype=np.float64),
            axis=0,
        )
    hashes: dict[str, str] = {"manifest": sha256_file(manifest.manifest_path)}
    for view in manifest.views:
        hashes[f"aperture_cfr/{view.view_id}"] = sha256_file(view.aperture_cfr_path)
    geom = build_geometry(manifest)
    los_visible, source = los_visibility(manifest)
    raw_pattern = manifest.config.get("tx_pattern")
    tx_pattern = None if raw_pattern is None else str(raw_pattern)
    return TomographyDataset(
        manifest=manifest,
        name=manifest.root.name,
        y_clean=np.asarray(y_clean, dtype=np.complex128),
        geom=geom,
        los_visible=np.asarray(los_visible, dtype=bool),
        los_visible_source=source,
        tx_pattern=tx_pattern,
        target=np.asarray(target, dtype=np.float64),
        hashes=hashes,
    )


@dataclass(frozen=True)
class GroundTruth:
    """Point ground truth used for scoring (design §6.1 keys)."""

    points_pos: np.ndarray | None
    points_space: str
    points_rho: np.ndarray | None
    vs_pos: np.ndarray | None
    path: Path | None
    sha256: str | None

    def positions(self, space: str) -> np.ndarray | None:
        """Return the GT points scored in ``space`` (points_pos if its space matches)."""
        if self.points_pos is not None and self.points_space == space:
            return self.points_pos
        if space == "vs" and self.vs_pos is not None:
            return self.vs_pos
        return None


def write_ground_truth(
    path: Path | str,
    *,
    points_pos: np.ndarray | None = None,
    points_rho: np.ndarray | None = None,
    points_space: str = "bv",
    vs_pos: np.ndarray | None = None,
) -> Path:
    """Write point ground truth to ``path`` as npz and return the path."""
    out = Path(path)
    if points_space not in ("bv", "vs"):
        raise ValueError(f"points_space must be 'bv' or 'vs', got {points_space!r}")
    payload: dict[str, np.ndarray] = {}
    if points_pos is not None:
        pos = np.asarray(points_pos, dtype=np.float64)
        if pos.ndim != 2 or pos.shape[1] != 3 or not np.all(np.isfinite(pos)):
            raise ValueError("points_pos must be a finite [K, 3] array")
        payload["points_pos"] = pos
    if points_rho is not None:
        rho = np.asarray(points_rho, dtype=np.complex128)
        if rho.ndim != 1:
            raise ValueError("points_rho must be a [K] array")
        if "points_pos" in payload and rho.shape[0] != payload["points_pos"].shape[0]:
            raise ValueError("points_rho length must match points_pos")
        payload["points_rho"] = rho
    payload["points_space"] = np.asarray(points_space)
    if vs_pos is not None:
        vs = np.asarray(vs_pos, dtype=np.float64)
        if vs.ndim != 2 or vs.shape[1] != 3 or not np.all(np.isfinite(vs)):
            raise ValueError("vs_pos must be a finite [M, 3] array")
        payload["vs_pos"] = vs
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **payload)
    return out


def load_ground_truth(path: Path | str) -> GroundTruth:
    """Load and validate the ground truth stored at ``path``."""
    location = Path(path)
    try:
        with np.load(location, allow_pickle=False) as data:
            arrays = {name: np.asarray(data[name]) for name in data.files}
    except OSError as error:
        raise ValueError(f"ground truth could not be loaded from {location}: {error}") from error
    points_pos: np.ndarray | None = None
    if "points_pos" in arrays:
        pos = np.asarray(arrays["points_pos"], dtype=np.float64)
        if pos.ndim != 2 or pos.shape[1] != 3 or not np.all(np.isfinite(pos)):
            raise ValueError("points_pos must be a finite [K, 3] array")
        points_pos = pos
    points_rho: np.ndarray | None = None
    if "points_rho" in arrays:
        rho = np.asarray(arrays["points_rho"], dtype=np.complex128)
        if rho.ndim != 1:
            raise ValueError("points_rho must be a [K] array")
        if points_pos is not None and rho.shape[0] != points_pos.shape[0]:
            raise ValueError("points_rho length must match points_pos")
        points_rho = rho
    points_space = "bv"
    if "points_space" in arrays:
        points_space = str(np.asarray(arrays["points_space"]).reshape(()))
        if points_space not in ("bv", "vs"):
            raise ValueError(f"points_space must be 'bv' or 'vs', got {points_space!r}")
    vs_pos: np.ndarray | None = None
    if "vs_pos" in arrays:
        vs = np.asarray(arrays["vs_pos"], dtype=np.float64)
        if vs.ndim != 2 or vs.shape[1] != 3 or not np.all(np.isfinite(vs)):
            raise ValueError("vs_pos must be a finite [M, 3] array")
        vs_pos = vs
    return GroundTruth(
        points_pos=points_pos,
        points_space=points_space,
        points_rho=points_rho,
        vs_pos=vs_pos,
        path=location,
        sha256=sha256_file(location),
    )


def find_ground_truth(
    dataset: TomographyDataset, override: Path | str | None = None
) -> GroundTruth | None:
    """Locate the scoring ground truth for ``dataset`` (override, manifest key, else None)."""
    if override is not None:
        return load_ground_truth(override)
    raw = dataset.manifest.raw
    entry = raw.get("tomography_gt") if isinstance(raw, Mapping) else None
    if isinstance(entry, Mapping) and isinstance(entry.get("artifact"), str):
        return load_ground_truth(dataset.manifest.root / str(entry["artifact"]))
    return None


def to_jsonable(value: Any) -> Any:
    """Convert ``value`` to JSON-serializable Python (non-finite floats become None)."""
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, int):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [to_jsonable(item) for item in value]
    return str(value)


def _sanitize_component(component: str) -> str:
    """Replace path separators and spaces inside one path component by underscores."""
    return str(component).replace("/", "_").replace("\\", "_").replace(" ", "_")


def recon_relpath(
    scene: str,
    config: str,
    stage: str,
    solver: str,
    track: str,
    space: str,
    strategy: str,
    realization: int,
) -> str:
    """Return the POSIX recon path of one result row."""
    parts = [_sanitize_component(part) for part in (scene, config, track, space, strategy)]
    scene_c, config_c, track_c, space_c, strategy_c = parts
    solver_c = _sanitize_component(solver)
    stage_c = _sanitize_component(stage)
    leaf = f"{track_c}.{space_c}.{strategy_c}.r{int(realization):03d}.npz"
    return f"{RECON_DIR}/{scene_c}/{config_c}/{stage_c}-{solver_c}/{leaf}"


def write_recon(out_dir: Path | str, relpath: str, arrays: Mapping[str, Any]) -> Path:
    """Write ``arrays`` compressed to ``out_dir / relpath`` and return the path."""
    out = Path(out_dir) / relpath
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **dict(arrays))
    return out


def _is_int(value: Any) -> bool:
    """Return True for genuine ints (bools excluded)."""
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def validate_result_row(row: Mapping[str, Any]) -> None:
    """Validate one result row mapping, raising ``ValueError`` on any violation."""
    if not isinstance(row, Mapping):
        raise ValueError("result row must be a mapping")
    if tuple(row) != RESULT_KEYS:
        raise ValueError(f"result row keys must equal RESULT_KEYS in order, got {tuple(row)}")
    if row["schema"] != RESULT_SCHEMA:
        raise ValueError(f"result row schema must be {RESULT_SCHEMA!r}")
    if row["stage"] not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    if row["status"] not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}")
    if row["space"] not in ("bv", "vs"):
        raise ValueError("space must be 'bv' or 'vs'")
    if row["status"] == "ok":
        if row["reason"] is not None:
            raise ValueError("reason must be None when status is 'ok'")
    elif not isinstance(row["reason"], str):
        raise ValueError("reason must be a str unless status is 'ok'")
    for key in ("realization", "n_detections"):
        if not _is_int(row[key]) or int(row[key]) < 0:
            raise ValueError(f"{key} must be an int >= 0")
    if row["n_iter"] is not None and (not _is_int(row["n_iter"])):
        raise ValueError("n_iter must be an int or None")
    if not isinstance(row["lattices"], list) or any(
        not isinstance(item, str) for item in row["lattices"]
    ):
        raise ValueError("lattices must be a list of str")
    hyper = row["hyper"]
    if not isinstance(hyper, Mapping):
        raise ValueError("hyper must be a dict")
    for key, item in hyper.items():
        if not isinstance(key, str):
            raise ValueError("hyper keys must be str")
        if not isinstance(item, (int, float)) or isinstance(item, bool):
            raise ValueError("hyper values must be floats")
    for key in ("metrics", "gauge_errors", "ill_posed"):
        if row[key] is not None and not isinstance(row[key], Mapping):
            raise ValueError(f"{key} must be a dict or None")
    runtime = row["runtime_s"]
    if (
        not isinstance(runtime, (int, float))
        or isinstance(runtime, bool)
        or not math.isfinite(float(runtime))
        or float(runtime) < 0.0
    ):
        raise ValueError("runtime_s must be a finite float >= 0")
    recon = row["recon"]
    if row["status"] == "ok" and row["stage"] != "planned":
        if not isinstance(recon, str):
            raise ValueError("recon must be a str for ok non-planned rows")
    elif recon is not None:
        raise ValueError("recon must be None unless status is 'ok' and stage is not 'planned'")


def write_result_row(handle: TextIO, row: Mapping[str, Any]) -> None:
    """Validate ``row`` and append it to ``handle`` as one JSON line."""
    validate_result_row(row)
    handle.write(json.dumps(to_jsonable(row), allow_nan=False) + "\n")
    handle.flush()


def read_results(path: Path | str) -> list[dict[str, Any]]:
    """Read the JSON-lines results file at ``path``."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def validate_run_manifest(payload: Mapping[str, Any]) -> None:
    """Validate a run-manifest mapping, raising ``ValueError`` on any violation."""
    if not isinstance(payload, Mapping):
        raise ValueError("run manifest must be a mapping")
    if tuple(sorted(payload)) != tuple(sorted(RUN_MANIFEST_KEYS)):
        raise ValueError(
            f"run manifest keys must be {sorted(RUN_MANIFEST_KEYS)}, got {sorted(payload)}"
        )
    if payload["schema"] != RUN_MANIFEST_SCHEMA:
        raise ValueError(f"run manifest schema must be {RUN_MANIFEST_SCHEMA!r}")
    results = payload["results"]
    if not isinstance(results, Mapping):
        raise ValueError("run manifest 'results' must be a mapping")
    if tuple(sorted(results)) != ("path", "rows", "sha256", "status_counts"):
        raise ValueError("run manifest 'results' must hold path, sha256, rows, status_counts")


def write_run_manifest(path: Path | str, payload: Mapping[str, Any]) -> None:
    """Validate ``payload`` and write it to ``path`` as indented JSON."""
    validate_run_manifest(payload)
    location = Path(path)
    location.parent.mkdir(parents=True, exist_ok=True)
    location.write_text(
        json.dumps(to_jsonable(payload), indent=2, allow_nan=False), encoding="utf-8"
    )


def software_versions() -> dict[str, str | None]:
    """Return the software versions recorded in the run manifest."""
    try:
        import scipy

        scipy_version: str | None = scipy.__version__
    except Exception:  # pragma: no cover - scipy is always installed here
        scipy_version = None
    try:
        plateau_version: str | None = importlib.metadata.version("plateau-sionna-dataset")
    except Exception:
        plateau_version = None
    git_commit: str | None = None
    try:
        repo_root = Path(__file__).parents[3]
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            git_commit = completed.stdout.strip()
    except Exception:
        git_commit = None
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy_version,
        "plateau_rt": plateau_version,
        "git_commit": git_commit,
    }
