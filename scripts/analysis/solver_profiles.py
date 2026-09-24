"""Solver-profile benchmark for the RF-camera multi-view generator.

Measures how Sionna PathSolver settings change the RF-camera observation on
two scenes: the mock one-box scene and Sionna's ``simple_street_canyon``.
Runs in the GPU image and writes ``solver_profiles.json`` plus
``solver_profiles.md`` to the output directory.

Two profile families are compared step-wise (never mixed, except the single
labelled material-effect pair): ``as_loaded`` (materials untouched) and
``scattering_s0.3`` (every radio material set to S=0.3, restored afterwards).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import drjit as dr
import numpy as np
from sionna.rt import Receiver, Transmitter, load_scene

from plateau_rt.adapters.sionna.rf_camera_dataset import RFMultiViewConfig
from plateau_rt.adapters.sionna.rf_patterns import HEMISPHERE_SPLIT_PATTERN
from plateau_rt.adapters.sionna.rf_tracing import (
    aperture_cfrs,
    configure_rf_camera_arrays,
    path_attributes,
    trace_paths,
)
from plateau_rt.application.solver_profile_report import (
    CAVEATS,
    FAMILY_AS_LOADED,
    FAMILY_SCATTERING_S03,
    ComparisonSpec,
    as_loaded_vs_complete_order,
    attach_noise_floor,
    compare_apertures,
    comparison_specs,
    json_safe,
    noise_floor_for,
    noise_floor_matrix_for,
    scene_markdown,
)
from plateau_rt.domain.rf_camera.camera import (
    RFViewSpec,
    generate_ring_views,
    look_at_orientation,
)
from plateau_rt.domain.rf_camera.imaging import frequency_offsets
from plateau_rt.domain.rf_camera.solver_metrics import (
    count_paths_by_type,
    hemisphere_energy,
)

REFERENCE_PROFILE = "complete"
DIFFUSE_SCATTERING = 0.3

# Scene B geometry (measured from simple_street_canyon object bboxes):
# buildings flank the street along x in two rows (north row y in [9.6, 38.2],
# south row y in [-36.5, -8.6]); the street gap is y in (-8.6, 9.6).
# The BS sits in the street above ground but below the rooftops (>= 21.8 m);
# 8 UEs at 1.5 m form a line along the street centre (y = 0.5) and look at a
# point on the north facade (no-name-1 south wall at y = 9.572).
STREET_BS_POSITION = (-50.0, 0.5, 6.0)
STREET_BS_LOOK_AT = (10.0, 0.5, 5.0)
STREET_FACADE_TARGET = (0.0, 9.572, 10.0)
STREET_UE_Y = 0.5
STREET_UE_HEIGHT = 1.5
STREET_UE_XS = (-40.0, -28.5, -17.0, -5.7, 5.7, 17.0, 28.5, 40.0)

# Scene A ring settings mirror the Makefile rf-camera-multiview-mock target.
MOCK_TARGET = (5.0, 5.0, 5.0)
MOCK_RADIUS_M = 30.0
MOCK_UE_HEIGHT_M = 1.5
MOCK_NUM_VIEWS = 8


@dataclass(frozen=True)
class SolverProfile:
    """One PathSolver configuration under test."""

    name: str
    max_depth: int
    synthetic_array: bool
    los: bool
    specular_reflection: bool
    refraction: bool
    diffraction: bool
    diffuse_reflection: bool
    scattering_coefficient: float | None = None
    seed_offset: int = 0
    family: str = FAMILY_AS_LOADED


def default_profiles() -> list[SolverProfile]:
    """Return the benchmark profile set in two material families."""
    s03 = FAMILY_SCATTERING_S03
    # ``los=True, specular_reflection=True, refraction=True`` is the shared
    # "current" mechanism base; written out explicitly (no ** splat).
    return [
        SolverProfile(
            name="los_only",
            max_depth=5,
            synthetic_array=True,
            los=True,
            specular_reflection=False,
            refraction=False,
            diffraction=False,
            diffuse_reflection=False,
        ),
        SolverProfile(
            name="specular",
            max_depth=5,
            synthetic_array=True,
            los=True,
            specular_reflection=True,
            refraction=False,
            diffraction=False,
            diffuse_reflection=False,
        ),
        SolverProfile(
            name="current",
            max_depth=5,
            synthetic_array=True,
            los=True,
            specular_reflection=True,
            refraction=True,
            diffraction=False,
            diffuse_reflection=False,
        ),
        SolverProfile(
            name="current_depth1",
            max_depth=1,
            synthetic_array=True,
            los=True,
            specular_reflection=True,
            refraction=True,
            diffraction=False,
            diffuse_reflection=False,
        ),
        SolverProfile(
            name="current_depth3",
            max_depth=3,
            synthetic_array=True,
            los=True,
            specular_reflection=True,
            refraction=True,
            diffraction=False,
            diffuse_reflection=False,
        ),
        SolverProfile(
            name="current_explicit_array",
            max_depth=5,
            synthetic_array=False,
            los=True,
            specular_reflection=True,
            refraction=True,
            diffraction=False,
            diffuse_reflection=False,
        ),
        SolverProfile(
            name="plus_diffraction",
            max_depth=5,
            synthetic_array=True,
            los=True,
            specular_reflection=True,
            refraction=True,
            diffraction=True,
            diffuse_reflection=False,
        ),
        SolverProfile(
            name="complete",
            max_depth=5,
            synthetic_array=True,
            los=True,
            specular_reflection=True,
            refraction=True,
            diffraction=True,
            diffuse_reflection=True,
        ),
        SolverProfile(
            name="complete_seed2",
            max_depth=5,
            synthetic_array=True,
            los=True,
            specular_reflection=True,
            refraction=True,
            diffraction=True,
            diffuse_reflection=True,
            seed_offset=1,
        ),
        SolverProfile(
            name="current_s03",
            max_depth=5,
            synthetic_array=True,
            los=True,
            specular_reflection=True,
            refraction=True,
            diffraction=False,
            diffuse_reflection=False,
            scattering_coefficient=DIFFUSE_SCATTERING,
            family=s03,
        ),
        SolverProfile(
            name="plus_diffuse_s03",
            max_depth=5,
            synthetic_array=True,
            los=True,
            specular_reflection=True,
            refraction=True,
            diffraction=False,
            diffuse_reflection=True,
            scattering_coefficient=DIFFUSE_SCATTERING,
            family=s03,
        ),
        SolverProfile(
            name="complete_s03",
            max_depth=5,
            synthetic_array=True,
            los=True,
            specular_reflection=True,
            refraction=True,
            diffraction=True,
            diffuse_reflection=True,
            scattering_coefficient=DIFFUSE_SCATTERING,
            family=s03,
        ),
        SolverProfile(
            name="plus_diffuse_s03_seed2",
            max_depth=5,
            synthetic_array=True,
            los=True,
            specular_reflection=True,
            refraction=True,
            diffraction=False,
            diffuse_reflection=True,
            scattering_coefficient=DIFFUSE_SCATTERING,
            seed_offset=1,
            family=s03,
        ),
        SolverProfile(
            name="complete_s03_seed2",
            max_depth=5,
            synthetic_array=True,
            los=True,
            specular_reflection=True,
            refraction=True,
            diffraction=True,
            diffuse_reflection=True,
            scattering_coefficient=DIFFUSE_SCATTERING,
            seed_offset=1,
            family=s03,
        ),
    ]


def setup_scene(
    xml_path: str | None,
    tx_position: tuple[float, float, float],
    tx_look_at: tuple[float, float, float],
    views: list[RFViewSpec],
    cfg: RFMultiViewConfig,
) -> Any:
    """Build the RF-camera scene, mirroring ``RFMultiViewDataset.run``.

    ``xml_path=None`` loads Sionna's ``simple_street_canyon`` example scene.
    """
    if xml_path is None:
        from sionna.rt.scene import simple_street_canyon

        scene = load_scene(simple_street_canyon)
    else:
        scene = load_scene(str(xml_path))
    scene.frequency = cfg.carrier_frequency_hz
    configure_rf_camera_arrays(
        scene,
        rx_rows=cfg.rx_rows,
        rx_cols=cfg.rx_cols,
        vertical_spacing_lambda=cfg.vertical_spacing_lambda,
        horizontal_spacing_lambda=cfg.horizontal_spacing_lambda,
        tx_pattern=cfg.tx_pattern,
        rx_pattern=HEMISPHERE_SPLIT_PATTERN,
        polarization=cfg.polarization,
    )
    scene.add(
        Transmitter(
            name="rf_camera_bs_000",
            position=list(tx_position),
            look_at=list(tx_look_at),
        )
    )
    for view in views:
        scene.add(
            Receiver(
                name=view.view_id,
                position=list(view.position),
                orientation=list(view.orientation),
            )
        )
    return scene


def scene_geometry(scene: Any) -> dict[str, Any]:
    """Return the scene bbox plus one bbox per Sionna object (best effort)."""
    payload: dict[str, Any] = {}
    try:
        bbox = scene.mi_scene.bbox()
        payload["scene_bbox"] = {
            "min": [float(v) for v in bbox.min],
            "max": [float(v) for v in bbox.max],
        }
    except Exception as exc:
        payload["scene_bbox_error"] = str(exc)
    objects = []
    for name in sorted(getattr(scene, "objects", {}).keys()):
        entry: dict[str, Any] = {"name": name}
        try:
            obb = scene.objects[name].mi_mesh.bbox()
            entry["bbox_min"] = [float(v) for v in obb.min]
            entry["bbox_max"] = [float(v) for v in obb.max]
        except Exception as exc:
            entry["bbox_error"] = str(exc)
        objects.append(entry)
    payload["objects"] = objects
    return payload


def check_containment(
    points: dict[str, tuple[float, float, float]], geometry: dict[str, Any]
) -> dict[str, str | None]:
    """Map each point to the name of the building bbox containing it (if any)."""
    boxes = [
        (obj["name"], np.array(obj["bbox_min"]), np.array(obj["bbox_max"]))
        for obj in geometry.get("objects", [])
        if "bbox_min" in obj and obj["name"] != "floor"
    ]
    result: dict[str, str | None] = {}
    for label, point in points.items():
        hit: str | None = None
        p = np.array(point, dtype=float)
        for name, bmin, bmax in boxes:
            if bool(np.all(p >= bmin) and np.all(p <= bmax)):
                hit = name
                break
        result[label] = hit
    return result


def _apply_scattering(scene: Any, value: float) -> dict[str, Any]:
    previous: dict[str, Any] = {}
    for name, mat in scene.radio_materials.items():
        previous[name] = mat.scattering_coefficient
        mat.scattering_coefficient = value
    return previous


def _restore_scattering(scene: Any, previous: dict[str, Any]) -> None:
    for name, value in previous.items():
        scene.radio_materials[name].scattering_coefficient = value


def _sync() -> None:
    dr.sync_thread()


def _materialise_paths(paths: Any) -> None:
    """Force evaluation of the path tensors this benchmark uses."""
    for tensor in (
        getattr(paths, "valid", None),
        getattr(paths, "tau", None),
        getattr(paths, "a", None),
    ):
        if tensor is None:
            continue
        try:
            dr.eval(tensor)
        except Exception:
            for part in ("real", "imag"):
                try:
                    dr.eval(getattr(tensor, part))
                except Exception:
                    pass


def _pattern_counts(scene: Any) -> tuple[int, int]:
    num_rx = len(scene.rx_array.antenna_pattern.patterns)
    num_tx = len(scene.tx_array.antenna_pattern.patterns)
    return int(num_rx), int(num_tx)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _min(values: list[float]) -> float | None:
    if not values:
        return None
    return float(np.min(np.asarray(values, dtype=np.float64)))


def run_profile_repeats(
    scene: Any,
    profile: SolverProfile,
    cfg: RFMultiViewConfig,
    num_views: int,
    frequency_offsets_hz: np.ndarray,
    samples_per_src: int | None,
    repeats: int,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """Trace one profile ``repeats`` times; run 1 is an untimed warm-up.

    Times three segments separately, each closed by a sync: (a) ``trace_s``
    (trace until the path tensors are materialised and synchronised),
    (b) ``export_s`` (path attributes plus path counts), (c) ``cfr_s``
    (aperture CFRs plus sync). Reports the median and min over runs 2..K.
    The apertures of the LAST run feed the image comparisons, and the
    same-seed last-vs-second-to-last difference is the repeatability floor.
    """
    num_rx_patterns, num_tx_patterns = _pattern_counts(scene)
    per_repeat: list[dict[str, Any]] = []
    apertures_runs: list[np.ndarray | None] = []
    for _rep in range(repeats):
        previous: dict[str, Any] = {}
        if profile.scattering_coefficient is not None:
            previous = _apply_scattering(scene, profile.scattering_coefficient)
        try:
            start = time.perf_counter()
            paths = trace_paths(
                scene,
                max_depth=profile.max_depth,
                synthetic_array=profile.synthetic_array,
                seed=cfg.seed + profile.seed_offset,
                los=profile.los,
                specular_reflection=profile.specular_reflection,
                diffuse_reflection=profile.diffuse_reflection,
                refraction=profile.refraction,
                diffraction=profile.diffraction,
                samples_per_src=samples_per_src,
            )
            _materialise_paths(paths)
            _sync()
            after_trace = time.perf_counter()
            exported = path_attributes(paths, ("valid", "interactions"))
            if exported["valid"].ndim == 5:
                counts = count_paths_by_type(
                    exported["valid"],
                    exported["interactions"],
                    num_rx_patterns=num_rx_patterns,
                    num_tx_patterns=num_tx_patterns,
                )
            else:
                counts = count_paths_by_type(exported["valid"], exported["interactions"])
            _sync()
            after_export = time.perf_counter()
            apertures: np.ndarray | None = None
            cfr_error: str | None = None
            try:
                apertures = aperture_cfrs(
                    paths,
                    frequency_offsets_hz,
                    num_rx=num_views,
                    rx_rows=cfg.rx_rows,
                    rx_cols=cfg.rx_cols,
                )
            except Exception as exc:
                cfr_error = f"{type(exc).__name__}: {exc}"
            _sync()
            after_cfr = time.perf_counter()
        finally:
            if previous:
                _restore_scattering(scene, previous)
        per_repeat.append(
            {
                "trace_s": after_trace - start,
                "export_s": after_export - after_trace,
                "cfr_s": after_cfr - after_export,
                "runtime_s": after_cfr - start,
                "path_counts": counts,
                "cfr_error": cfr_error,
            }
        )
        apertures_runs.append(apertures)
    timed = per_repeat[1:] if len(per_repeat) > 1 else per_repeat
    trace_vals = [r["trace_s"] for r in timed]
    export_vals = [r["export_s"] for r in timed]
    cfr_vals = [r["cfr_s"] for r in timed]
    runtime_vals = [r["runtime_s"] for r in timed]
    last_apertures = apertures_runs[-1] if apertures_runs else None
    repeat_diff: dict[str, Any] | None = None
    if len(apertures_runs) >= 2 and apertures_runs[-1] is not None:
        prev_ap = apertures_runs[-2]
        if prev_ap is not None and prev_ap.shape == apertures_runs[-1].shape:
            repeat_diff = compare_apertures(apertures_runs[-1], prev_ap)
    metrics: dict[str, Any] = {
        "settings": asdict(profile),
        "family": profile.family,
        "path_counts": per_repeat[-1]["path_counts"] if per_repeat else None,
        "trace_s_median": _median(trace_vals),
        "trace_s_min": _min(trace_vals),
        "export_s_median": _median(export_vals),
        "export_s_min": _min(export_vals),
        "cfr_s_median": _median(cfr_vals),
        "cfr_s_min": _min(cfr_vals),
        "runtime_s_median": _median(runtime_vals),
        "runtime_s_min": _min(runtime_vals),
        "repeats": per_repeat,
        "num_repeats": repeats,
        "repeat_diff": repeat_diff,
        "seed": cfg.seed + profile.seed_offset,
        "seed_offset": profile.seed_offset,
        "samples_per_src": samples_per_src,
        "scattering_coefficient": profile.scattering_coefficient,
        "num_rx_patterns": num_rx_patterns,
        "num_tx_patterns": num_tx_patterns,
    }
    if last_apertures is not None:
        metrics["hemisphere_energy"] = hemisphere_energy(last_apertures)
        metrics["aperture_shape"] = list(last_apertures.shape)
    if per_repeat and per_repeat[-1]["cfr_error"] is not None:
        metrics["cfr_error"] = per_repeat[-1]["cfr_error"]
    return last_apertures, metrics


def _sionna_version() -> str:
    try:
        from importlib.metadata import version

        for dist in ("sionna-rt", "sionna"):
            try:
                return version(dist)
            except Exception:
                continue
    except Exception:
        pass
    try:
        import sionna

        return str(getattr(sionna, "__version__", "unknown"))
    except Exception:
        return "unknown"


def _matrix_max(matrix: Any) -> float | None:
    try:
        arr = np.asarray(matrix, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if arr.size == 0:
        return None
    try:
        with np.errstate(invalid="ignore"):
            return float(np.nanmax(arr))
    except ValueError:
        return None


def build_comparisons(
    apertures_by_profile: dict[str, np.ndarray | None],
    metrics_by_profile: dict[str, dict[str, Any]],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[tuple[str, str], Any],
    dict[str, Any],
    float | None,
]:
    """Build step-wise comparison stats with per-cell noise floors.

    Returns ``(comparisons, seed_matrices, repeat_matrices,
    repeat_floor_overall)``. ``seed_matrices`` maps ``(test, baseline)`` to
    the seed-noise ``rel_diff_matrix``; ``repeat_matrices`` maps profile
    names to their same-seed ``rel_diff_matrix``. Each non-noise comparison
    stores the per-cell ``noise_floor_matrix``, ``resolved_matrix``,
    ``n_resolved_cells``, ``resolved_max`` and the display-only scalar
    ``noise_floor``. Noise comparisons are measurements (``effect ==
    "noise floor"``, ``resolved is None``). ``repeat_floor_overall`` is
    information only and is never used as a floor.
    """
    specs = comparison_specs()
    wanted = {(s.test, s.baseline) for s in specs}
    seed_matrices: dict[tuple[str, str], Any] = {}
    for spec in specs:
        if not spec.is_noise:
            continue
        test_ap = apertures_by_profile.get(spec.test)
        base_ap = apertures_by_profile.get(spec.baseline)
        if test_ap is None or base_ap is None:
            continue
        if test_ap.shape != base_ap.shape:
            continue
        try:
            seed_matrices[(spec.test, spec.baseline)] = np.asarray(
                compare_apertures(test_ap, base_ap)["rel_diff_matrix"], dtype=np.float64
            )
        except Exception:
            continue
    repeat_matrices: dict[str, Any] = {}
    for name, metrics in metrics_by_profile.items():
        repeat = metrics.get("repeat_diff")
        if isinstance(repeat, dict) and "rel_diff_matrix" in repeat:
            try:
                repeat_matrices[name] = np.asarray(repeat["rel_diff_matrix"], dtype=np.float64)
            except (TypeError, ValueError):
                continue
    overall_repeat: float | None = None
    for matrix in repeat_matrices.values():
        candidate = _matrix_max(matrix)
        if candidate is not None and not (isinstance(candidate, float) and np.isnan(candidate)):
            if overall_repeat is None or candidate > overall_repeat:
                overall_repeat = candidate
    comparisons: dict[str, dict[str, Any]] = {}
    for spec in specs:
        if (spec.test, spec.baseline) not in wanted:
            continue
        test_ap = apertures_by_profile.get(spec.test)
        base_ap = apertures_by_profile.get(spec.baseline)
        if test_ap is None or base_ap is None:
            continue
        if test_ap.shape != base_ap.shape:
            comparisons[f"{spec.test}__vs__{spec.baseline}"] = {
                "test": spec.test,
                "baseline": spec.baseline,
                "question": spec.question,
                "is_noise": spec.is_noise,
                "error": "shape mismatch or missing apertures",
            }
            continue
        stats = compare_apertures(test_ap, base_ap)
        if spec.is_noise:
            repeat_max: float | None = None
            for profile in (spec.test, spec.baseline):
                if profile in repeat_matrices:
                    candidate = _matrix_max(repeat_matrices[profile])
                    if candidate is not None and not (
                        isinstance(candidate, float) and np.isnan(candidate)
                    ):
                        if repeat_max is None or candidate > repeat_max:
                            repeat_max = candidate
            stats["repeat_max"] = repeat_max
            comparisons[f"{spec.test}__vs__{spec.baseline}"] = {
                "test": spec.test,
                "baseline": spec.baseline,
                "question": spec.question,
                "is_noise": spec.is_noise,
                **attach_noise_floor(stats, None, is_noise=True),
            }
            # Preserve the repeat max for the noise sentence.
            comparisons[f"{spec.test}__vs__{spec.baseline}"]["repeat_max"] = repeat_max
            continue
        floor = noise_floor_for(spec, seed_matrices, repeat_matrices)
        comparisons[f"{spec.test}__vs__{spec.baseline}"] = {
            "test": spec.test,
            "baseline": spec.baseline,
            "question": spec.question,
            "is_noise": spec.is_noise,
            **attach_noise_floor(stats, floor),
        }
    return comparisons, seed_matrices, repeat_matrices, overall_repeat


def build_secondary_vs_complete(
    apertures_by_profile: dict[str, np.ndarray | None],
    seed_matrices: dict[tuple[str, str], Any] | None = None,
    repeat_matrices: dict[str, Any] | None = None,
    comparisons: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build the secondary each-as_loaded-profile-vs-complete table.

    Each entry uses the per-cell floor logic: the test profile's seed
    source plus ``complete``'s seed source, plus the same-seed repeat
    matrices of both. The legacy ``comparisons`` argument is accepted for
    backwards compatibility but ignored.
    """
    _ = comparisons
    ref = apertures_by_profile.get(REFERENCE_PROFILE)
    if ref is None:
        return {}
    table: dict[str, dict[str, Any]] = {}
    for name in as_loaded_vs_complete_order():
        if name == REFERENCE_PROFILE:
            continue
        test_ap = apertures_by_profile.get(name)
        if test_ap is None or test_ap.shape != ref.shape:
            continue
        floor = noise_floor_matrix_for(name, REFERENCE_PROFILE, seed_matrices, repeat_matrices)
        table[name] = attach_noise_floor(compare_apertures(test_ap, ref), floor)
    return table


def street_canyon_views() -> list[RFViewSpec]:
    """Return 8 UE views on a line along the street, facing the north facade."""
    views = []
    for index, x in enumerate(STREET_UE_XS):
        position = (float(x), STREET_UE_Y, STREET_UE_HEIGHT)
        views.append(
            RFViewSpec(
                view_id=f"ue_{index:06d}",
                position=position,
                look_at=STREET_FACADE_TARGET,
                orientation=look_at_orientation(position, STREET_FACADE_TARGET),
            )
        )
    return views


def run_scene(
    scene_key: str,
    xml_path: str | None,
    tx_position: tuple[float, float, float],
    tx_look_at: tuple[float, float, float],
    views: list[RFViewSpec],
    cfg: RFMultiViewConfig,
    profiles: list[SolverProfile],
    frequency_offsets_hz: np.ndarray,
    samples_per_src: int | None,
    repeats: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray | None], dict[str, dict[str, Any]]]:
    """Set up one scene and run every profile ``repeats`` times (errors recorded)."""
    print(f"=== solver profiles: scene {scene_key} ===")
    scene = setup_scene(xml_path, tx_position, tx_look_at, views, cfg)
    geometry = scene_geometry(scene)
    print(f"scene bbox: {geometry.get('scene_bbox')}")
    for obj in geometry.get("objects", []):
        print(
            f"  object {obj['name']}: {obj.get('bbox_min')} .. {obj.get('bbox_max')}"
            f"{' (' + obj['bbox_error'] + ')' if 'bbox_error' in obj else ''}"
        )
    points = {"bs": tx_position, **{v.view_id: v.position for v in views}}
    containment = check_containment(points, geometry)
    for label, hit in containment.items():
        print(f"  containment {label}: {'inside ' + hit if hit else 'clear'}")

    apertures_by_profile: dict[str, np.ndarray | None] = {}
    metrics_by_profile: dict[str, dict[str, Any]] = {}
    for profile in profiles:
        print(f"-- profile {profile.name} x{repeats} ...")
        try:
            apertures, metrics = run_profile_repeats(
                scene,
                profile,
                cfg,
                len(views),
                frequency_offsets_hz,
                samples_per_src,
                repeats,
            )
        except Exception as exc:
            print(f"   ERROR: {type(exc).__name__}: {exc}")
            apertures, metrics = (
                None,
                {"settings": asdict(profile), "error": f"{type(exc).__name__}: {exc}"},
            )
        apertures_by_profile[profile.name] = apertures
        metrics_by_profile[profile.name] = metrics
        print(f"   metrics: {json.dumps(metrics, default=str)[:400]}")
    comparisons, seed_matrices, repeat_matrices, repeat_floor = build_comparisons(
        apertures_by_profile, metrics_by_profile
    )
    seed_maxima = {
        f"{test}__vs__{baseline}": _matrix_max(matrix)
        for (test, baseline), matrix in seed_matrices.items()
    }
    secondary = build_secondary_vs_complete(apertures_by_profile, seed_matrices, repeat_matrices)
    scene_payload = {
        "xml": xml_path,
        "geometry": geometry,
        "containment": containment,
        "tx_position": list(tx_position),
        "tx_look_at": list(tx_look_at),
        "views": [
            {
                "view_id": v.view_id,
                "position": list(v.position),
                "look_at": list(v.look_at),
                "orientation": list(v.orientation),
            }
            for v in views
        ],
        "profiles": metrics_by_profile,
        "comparisons": comparisons,
        "secondary_vs_complete": secondary,
        "seed_maxima": seed_maxima,
        "repeat_floor_overall": repeat_floor,
        "repeats": repeats,
    }
    return scene_payload, apertures_by_profile, metrics_by_profile


def _counts_cell(metrics: dict[str, Any]) -> str:
    counts = metrics.get("path_counts")
    if not counts:
        return metrics.get("error", "n/a")
    order = ("total", "los", "specular", "diffuse", "refraction", "diffraction")
    body = ", ".join(f"{k}={counts.get(k, '?')}" for k in order)
    links = counts.get("num_links", "?")
    mean = counts.get("mean_paths_per_link", "?")
    maxlink = counts.get("max_paths_per_link", "?")
    mean_str = f"{mean:.3g}" if isinstance(mean, (int, float)) else str(mean)
    return f"{body} (links={links}, mean={mean_str}, max={maxlink})"


def _timing_cell(metrics: dict[str, Any]) -> str:
    parts = []
    for key in ("trace_s_median", "export_s_median", "cfr_s_median"):
        value = metrics.get(key)
        parts.append("n/a" if value is None else f"{value:.3g}")
    return "trace/export/cfr median=" + "/".join(parts)


def write_markdown(
    path: Path,
    scenes: dict[str, dict[str, Any]],
    cfg: RFMultiViewConfig,
    samples_per_src: int | None,
    repeats: int,
) -> None:
    """Write one markdown section per scene plus data-derived interpretations."""
    lines = [
        "# Solver-profile benchmark",
        "",
        f"RFMultiViewConfig defaults: carrier={cfg.carrier_frequency_hz / 1e9:.2f} GHz, "
        f"bandwidth={cfg.bandwidth_hz / 1e6:.0f} MHz, bins={cfg.num_frequency_bins}, "
        f"aperture={cfg.rx_rows}x{cfg.rx_cols}, seed={cfg.seed}, "
        f"samples_per_src={samples_per_src}, repeats={repeats} "
        "(run 1 is a warm-up; medians over runs 2..K).",
        "",
    ]
    for scene_key, payload in scenes.items():
        lines += [
            f"## Scene: {scene_key}",
            "",
            f"BS={payload['tx_position']} look_at={payload['tx_look_at']}, "
            f"views={len(payload['views'])}.",
            "",
            "| profile | family | valid paths by type | E_front | E_back | timings s |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for name, metrics in payload["profiles"].items():
            energy = metrics.get("hemisphere_energy", {})
            lines.append(
                f"| {name} | {metrics.get('family', 'n/a')} | {_counts_cell(metrics)} "
                f"| {energy.get('front', 'n/a')} | {energy.get('back', 'n/a')} "
                f"| {_timing_cell(metrics)} |"
            )
        lines += [""]
        comparisons = payload.get("comparisons", {})
        entries: list[tuple[ComparisonSpec, dict[str, Any]]] = []
        for spec in comparison_specs():
            entry = comparisons.get(f"{spec.test}__vs__{spec.baseline}")
            if entry is not None and "max_diff" in entry:
                entries.append((spec, entry))
        lines.append(scene_markdown(scene_key, entries))
        secondary = payload.get("secondary_vs_complete", {})
        if secondary:
            lines += [
                "### Secondary: each as_loaded profile vs complete",
                "",
                "| profile | resolved max | cells | aggregate front/back/total |",
                "| --- | --- | --- | --- |",
            ]
            for name, stats in secondary.items():
                agg = stats.get("aggregate", {})

                def _cell(value: Any) -> str:
                    if value is None:
                        return "n/a"
                    try:
                        number = float(value)  # type: ignore[arg-type]
                    except (TypeError, ValueError):
                        return "n/a"
                    if number != number:  # NaN
                        return "nan"
                    if number == float("inf"):
                        return "inf"
                    if number == float("-inf"):
                        return "-inf"
                    return f"{number:.3g}"

                n_cells = stats.get("n_resolved_cells", 0)
                n_views = stats.get("num_views", 0)
                lines.append(
                    f"| {name} | {_cell(stats.get('resolved_max'))} "
                    f"| {n_cells}/{int(n_views) * 2} "
                    f"| {_cell(agg.get('front'))}/{_cell(agg.get('back'))}/"
                    f"{_cell(agg.get('total'))} |"
                )
            lines += [""]
    lines += [
        "## Caveats",
        "",
        *[f"- {caveat}" for caveat in CAVEATS],
        "- Timings are medians over runs 2..K (run 1 is a JIT warm-up); "
        "trace_s ends after the path tensors are materialised and synchronised, "
        "export_s covers path_attributes plus path counts, cfr_s covers the "
        "aperture CFRs plus sync.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/generated/analysis/solver_profiles"))
    parser.add_argument(
        "--mock-xml", type=str, default="data/generated/mock_results/mock_building.city.xml"
    )
    parser.add_argument("--scenes", type=str, default="mock,street_canyon")
    parser.add_argument(
        "--profiles", type=str, default="", help="comma-separated profile names (default: all)"
    )
    parser.add_argument("--samples-per-src", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.repeats < 2:
        raise SystemExit("--repeats must be >= 2 (run 1 is a warm-up)")
    cfg = RFMultiViewConfig()
    cfg.validate()
    all_profiles = default_profiles()
    wanted = {p.strip() for p in args.scenes.split(",") if p.strip()}
    if args.profiles:
        keep = {p.strip() for p in args.profiles.split(",") if p.strip()}
        profiles = [p for p in all_profiles if p.name in keep]
        unknown = keep - {p.name for p in all_profiles}
        if unknown:
            raise SystemExit(f"unknown profiles: {sorted(unknown)}")
    else:
        profiles = all_profiles
    if REFERENCE_PROFILE not in {p.name for p in profiles}:
        print(
            f"WARNING: reference profile '{REFERENCE_PROFILE}' not selected; "
            "secondary vs-complete table will be n/a."
        )
    if "mock" in wanted and not Path(args.mock_xml).exists():
        raise SystemExit(f"mock scene XML not found: {args.mock_xml}; run `make build-mock` first")

    offsets = frequency_offsets(cfg.bandwidth_hz, cfg.num_frequency_bins)
    scenes: dict[str, dict[str, Any]] = {}
    if "mock" in wanted:
        mock_views = generate_ring_views(
            target=MOCK_TARGET,
            radius_m=MOCK_RADIUS_M,
            ue_height_m=MOCK_UE_HEIGHT_M,
            num_views=MOCK_NUM_VIEWS,
        )
        payload, _, _ = run_scene(
            "mock",
            args.mock_xml,
            cfg.tx_position,
            cfg.tx_look_at,
            mock_views,
            cfg,
            profiles,
            offsets,
            args.samples_per_src,
            args.repeats,
        )
        scenes["mock"] = payload
    if "street_canyon" in wanted:
        payload, _, _ = run_scene(
            "street_canyon",
            None,
            STREET_BS_POSITION,
            STREET_BS_LOOK_AT,
            street_canyon_views(),
            cfg,
            profiles,
            offsets,
            args.samples_per_src,
            args.repeats,
        )
        scenes["street_canyon"] = payload
    unknown_scenes = wanted - {"mock", "street_canyon"}
    if unknown_scenes:
        raise SystemExit(f"unknown scenes: {sorted(unknown_scenes)}")

    args.out.mkdir(parents=True, exist_ok=True)
    document = {
        "config": {
            **asdict(cfg),
            "samples_per_src": args.samples_per_src,
            "repeats": args.repeats,
        },
        "sionna_version": _sionna_version(),
        "reference_profile": REFERENCE_PROFILE,
        "scenes": scenes,
    }
    json_path = args.out / "solver_profiles.json"
    json_path.write_text(
        json.dumps(json_safe(document), indent=2, allow_nan=False), encoding="utf-8"
    )
    md_path = args.out / "solver_profiles.md"
    write_markdown(md_path, scenes, cfg, args.samples_per_src, args.repeats)
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")


if __name__ == "__main__":
    main()
