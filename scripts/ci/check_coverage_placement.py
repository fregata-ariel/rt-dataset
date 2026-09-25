"""CI: verify a coverage-map UE placement dataset (#16).

Reuses :mod:`check_mock_outputs` for the shared RF-camera / optical checks and
additionally validates the manifest ``placement`` section: the saved radio map
and its sha256 digests, the reproducibility from the saved map
(``replan_from_manifest``), the recorded poses, the candidate-cell invariants
and the seed-sensitivity contract.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_mock_outputs as cmo  # noqa: E402

from plateau_rt.application.rf_dataset_manifest import (  # noqa: E402
    ManifestError,
    load_rf_dataset_manifest,
)
from plateau_rt.application.rf_tomography_io import los_visibility  # noqa: E402
from plateau_rt.application.ue_placement import (  # noqa: E402
    PLACEMENT_DIR,
    RADIO_MAP_INDOOR_MASK_FILE,
    RADIO_MAP_LOS_MASK_FILE,
    RADIO_MAP_PATH_GAIN_FILE,
    building_exclusion_mask,
    file_sha256,
    load_radio_map,
    replan_from_manifest,
)
from plateau_rt.domain.rf_camera.placement import (  # noqa: E402
    CoveragePlacementSettings,
    candidate_cells,
    path_gain_db,
    views_from_placement_record,
)

# The manifest keys that must stay equal when only --placement-seed changes.
SEED_INVARIANT_KEYS = (
    "schema_version",
    "mode",
    "source_scene",
    "config",
    "frequency_offsets_hz",
    "absolute_frequencies_hz",
    "delay_resolution_s",
    "unambiguous_delay_s",
    "base_stations",
    "raw_observation",
    "camera_model",
    "path_geometry_gt",
    "path_schema",
)
PLACEMENT_INVARIANT_KEYS = (
    "grid",
    "threshold_db",
    "candidate_count",
    "cell_counts",
    "exclusion",
    "rng",
    "multi_bs",
    "los",
    "sampler_version",
)


def _db(value: float) -> float:
    """Return ``10*log10(value)`` or ``-inf`` for a non-positive value."""
    return 10.0 * math.log10(value) if value > 0.0 else float("-inf")


def _format_db(value: float) -> str:
    """Format a dB value, using ``-inf`` for non-finite values."""
    return "-inf" if not math.isfinite(value) else f"{value:.2f}"


def _aggregated_gain_db(path_gain: np.ndarray, aggregation: str, iy: int, ix: int) -> float:
    """Aggregate the linear path gain of one cell in dB, matching the domain."""
    gain = np.asarray(path_gain, dtype=np.float64)
    gain_db = path_gain_db(gain)
    if aggregation == "max":
        return float(np.max(gain_db[:, iy, ix]))
    if aggregation == "sum":
        column = gain[:, iy, ix]
        total = float(np.sum(np.where(np.isfinite(column) & (column > 0.0), column, 0.0)))
        return 10.0 * math.log10(total) if total > 0.0 else float("-inf")
    return float(np.min(gain_db[:, iy, ix]))


def _check_radio_map_hashes(c: cmo.Checker, dataset_dir: Path, placement: Mapping) -> None:
    """Check the recorded sha256 digests of both radio-map artifacts."""
    placement_dir = dataset_dir / PLACEMENT_DIR
    try:
        metadata_path = dataset_dir / placement["radio_map"]["metadata"]
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (KeyError, OSError, json.JSONDecodeError) as exc:
        c.check(False, f"radio_map.json readable: {exc}")
        return
    manifest_hashes = placement["radio_map"].get("sha256", {})
    metadata_hashes = metadata.get("sha256", {})
    checks = [
        ("path_gain", RADIO_MAP_PATH_GAIN_FILE),
        ("indoor_mask", RADIO_MAP_INDOOR_MASK_FILE),
    ]
    if "los_mask" in manifest_hashes:
        checks.append(("los_mask", RADIO_MAP_LOS_MASK_FILE))
    for key, file_name in checks:
        artifact = placement_dir / file_name
        actual = file_sha256(artifact)
        c.check(
            actual == manifest_hashes.get(key),
            f"{file_name} sha256 matches manifest placement.radio_map",
        )
        c.check(
            actual == metadata_hashes.get(key),
            f"{file_name} sha256 matches placement/radio_map.json",
        )


def _check_replan(c: cmo.Checker, dataset_dir: Path, placement: Mapping) -> None:
    """Check that the saved map + seed reproduces the recorded placement."""
    try:
        replanned = replan_from_manifest(dataset_dir)
    except (ManifestError, ValueError, OSError) as exc:
        c.check(False, f"replan_from_manifest: {exc}")
        return
    c.check(
        replanned.views == views_from_placement_record(placement),
        "replan_from_manifest views == views_from_placement_record",
    )
    c.check(
        replanned.to_record()["views"] == placement["views"],
        "replan_from_manifest to_record views == manifest placement views",
    )


def _check_poses(c: cmo.Checker, dataset_dir: Path, manifest, placement: Mapping) -> None:
    """Check pose.json against the placement record and the manifest views."""
    for entry in placement["views"]:
        view_id = entry["view_id"]
        pose_path = dataset_dir / "views" / view_id / "pose.json"
        try:
            pose = json.loads(pose_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            c.check(False, f"{view_id} pose.json readable: {exc}")
            continue
        c.check(
            pose["position_m"] == entry["position_m"],
            f"{view_id} pose position_m == placement record",
        )
        c.check(
            pose["look_at_m"] == entry["look_at_m"],
            f"{view_id} pose look_at_m == placement record",
        )
        c.check(
            pose["orientation_rad"] == entry["orientation_rad"],
            f"{view_id} pose orientation_rad == placement record",
        )
        try:
            manifest_view = manifest.view(view_id)
        except ManifestError as exc:
            c.check(False, f"{view_id} manifest view: {exc}")
            continue
        c.check(
            list(manifest_view.position_m) == entry["position_m"],
            f"{view_id} manifest position_m == placement record",
        )
        c.check(
            list(manifest_view.look_at_m or []) == entry["look_at_m"],
            f"{view_id} manifest look_at_m == placement record",
        )
        c.check(
            list(manifest_view.orientation_rad or []) == entry["orientation_rad"],
            f"{view_id} manifest orientation_rad == placement record",
        )


def _check_cells(c: cmo.Checker, dataset_dir: Path, manifest, placement: Mapping) -> None:
    """Recompute candidates from the saved map and validate the chosen cells."""
    saved = load_radio_map(dataset_dir / placement["radio_map"]["metadata"])
    settings = CoveragePlacementSettings.from_dict(placement["settings"])
    clearance = float(placement["exclusion"]["building_clearance_m"])
    exclusion = building_exclusion_mask(saved.indoor_mask, saved.grid, clearance_m=clearance)
    bs_positions = [list(bs.position_m) for bs in manifest.base_stations]
    try:
        candidates = candidate_cells(
            saved.path_gain,
            saved.grid,
            threshold=settings.threshold,
            aggregation=settings.aggregation,
            exclusion_mask=exclusion,
            bs_positions=bs_positions,
            min_bs_distance_m=settings.min_bs_distance_m,
            los_mask=saved.los_mask,
        )
    except ValueError as exc:
        c.check(False, f"candidate_cells from saved map: {exc}")
        return

    centers = saved.grid.cell_centers()
    cell_x = float(saved.grid.cell_size_m[0])
    cell_y = float(saved.grid.cell_size_m[1])
    gain_db_all = path_gain_db(saved.path_gain)
    thresholds = list(placement["threshold_db"])
    positions: list[tuple[float, float]] = []
    for entry in placement["views"]:
        view_id = entry["view_id"]
        iy, ix = int(entry["cell_index"][0]), int(entry["cell_index"][1])
        c.check(bool(candidates.mask[iy, ix]), f"{view_id} cell ({iy},{ix}) is a candidate")
        if settings.aggregation == "all":
            for bs_index, threshold in enumerate(thresholds):
                c.check(
                    float(gain_db_all[bs_index, iy, ix]) >= threshold - 1e-9,
                    f"{view_id} BS{bs_index} gain >= threshold {threshold:.2f} dB",
                )
        elif settings.aggregation == "any":
            passes_any = False
            for bs_index, threshold in enumerate(thresholds):
                if threshold is None:
                    continue
                if float(gain_db_all[bs_index, iy, ix]) >= float(threshold) - 1e-9:
                    passes_any = True
                    break
            c.check(passes_any, f"{view_id} passes at least one BS threshold (any)")
        else:
            gain_db = _aggregated_gain_db(saved.path_gain, settings.aggregation, iy, ix)
            c.check(
                gain_db >= thresholds[0] - 1e-9,
                f"{view_id} aggregated gain {gain_db:.2f} dB >= threshold {thresholds[0]:.2f} dB",
            )
        c.check(not bool(saved.indoor_mask[iy, ix]), f"{view_id} cell is not indoor")
        c.check(not bool(exclusion[iy, ix]), f"{view_id} cell is not within clearance")
        center = centers[iy, ix]
        position = entry["position_m"]
        inside = (
            abs(float(position[0]) - float(center[0])) <= 0.5 * cell_x + 1e-9
            and abs(float(position[1]) - float(center[1])) <= 0.5 * cell_y + 1e-9
            and abs(float(position[2]) - float(center[2])) <= 1e-9
        )
        c.check(inside, f"{view_id} position lies inside cell ({iy},{ix})")
        for bs_index, bs_position in enumerate(bs_positions):
            distance = float(np.linalg.norm(center - np.asarray(bs_position, dtype=np.float64)))
            c.check(
                distance >= settings.min_bs_distance_m - 1e-9,
                f"{view_id} distance to BS{bs_index} {distance:.3f} m "
                f">= {settings.min_bs_distance_m} m",
            )
        positions.append((float(position[0]), float(position[1])))

    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            spacing = math.hypot(
                positions[i][0] - positions[j][0], positions[i][1] - positions[j][1]
            )
            c.check(
                spacing >= settings.min_spacing_m - 1e-9,
                f"views {i}/{j} position spacing {spacing:.3f} m >= {settings.min_spacing_m} m",
            )


def _check_los(c: cmo.Checker, dataset_dir: Path, manifest, placement: Mapping) -> None:
    """Check the per-view LoS flags against the saved LoS mask, the quota and the path GT.

    Without jitter a UE sits on its cell centre, so the geometric LoS flag of
    its cell must match the traced LoS path (``los_visibility`` of the path GT).
    """
    los_section = placement.get("los")
    if los_section is None:
        return
    saved = load_radio_map(dataset_dir / placement["radio_map"]["metadata"])
    if saved.los_mask is None:
        c.check(False, "placement records a LoS section but the saved map has no los_mask")
        return
    for entry in placement["views"]:
        view_id = entry["view_id"]
        iy, ix = int(entry["cell_index"][0]), int(entry["cell_index"][1])
        expected = [bool(v) for v in saved.los_mask[:, iy, ix]]
        c.check(entry.get("los_bs") == expected, f"{view_id} los_bs == saved los_mask")
    settings = placement["settings"]
    if float(settings.get("jitter_fraction", 0.0)) == 0.0:
        visible, source = los_visibility(manifest)
        c.check(source == "path_gt", f"LoS visibility from the path GT (source={source})")
        row_of = {view.view_id: row for row, view in enumerate(manifest.views)}
        for entry in placement["views"]:
            traced = [bool(v) for v in visible[row_of[entry["view_id"]]]]
            c.check(
                entry.get("los_bs") == traced,
                f"{entry['view_id']} geometric los_bs {entry.get('los_bs')} == traced LoS {traced}",
            )
    los_fraction = settings.get("los_fraction")
    if los_fraction is not None:
        num_views = int(settings["num_views"])
        wanted = int(math.floor(float(los_fraction) * num_views + 0.5))
        got = sum(1 for entry in placement["views"] if entry.get("los") is True)
        c.check(
            got == wanted,
            f"LoS view count {got} == floor(los_fraction*num_views + 0.5)={wanted}",
        )


def _check_orientation(c: cmo.Checker, manifest, placement: Mapping) -> None:
    """For ``face_bs``, check the faced BS is in the front hemisphere."""
    settings = placement["settings"]
    if settings.get("orientation_policy") != "face_bs":
        return
    for entry in placement["views"]:
        view_id = entry["view_id"]
        facing = entry.get("facing_bs_index")
        if facing is None:
            c.check(False, f"{view_id} face_bs placement records a facing BS")
            continue
        try:
            manifest_view = manifest.view(view_id)
        except ManifestError as exc:
            c.check(False, f"{view_id} manifest view: {exc}")
            continue
        c.check(
            manifest_view.bs[int(facing)].bs_in_front_hemisphere is True,
            f"{view_id} faced BS {int(facing)} is in the front hemisphere",
        )


def _check_same_seed(
    c: cmo.Checker, dataset_dir: Path, placement: Mapping, rerun_dir: Path
) -> None:
    """Check a same-seed rerun reproduces the placement and the pose files."""
    try:
        rerun = load_rf_dataset_manifest(rerun_dir)
    except (ManifestError, OSError) as exc:
        c.check(False, f"same-seed rerun manifest: {exc}")
        return
    rerun_placement = rerun.placement
    c.check(
        isinstance(rerun_placement, Mapping) and rerun_placement.get("views") == placement["views"],
        "same-seed rerun placement views identical",
    )
    for entry in placement["views"]:
        view_id = entry["view_id"]
        first = (dataset_dir / "views" / view_id / "pose.json").read_bytes()
        second = (rerun_dir / "views" / view_id / "pose.json").read_bytes()
        c.check(first == second, f"same-seed rerun {view_id} pose.json byte-identical")


def _check_other_seed(
    c: cmo.Checker, dataset_dir: Path, placement: Mapping, other_dir: Path
) -> None:
    """Check a different seed changes only the placement and its seed."""
    try:
        other = load_rf_dataset_manifest(other_dir)
        other_raw = json.loads((other_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
        main_raw = json.loads((dataset_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    except (ManifestError, OSError, json.JSONDecodeError) as exc:
        c.check(False, f"other-seed rerun manifest: {exc}")
        return
    other_placement = other.placement
    if not isinstance(other_placement, Mapping):
        c.check(False, "other-seed rerun has a placement section")
        return
    c.check(
        other_placement.get("placement_seed") != placement.get("placement_seed"),
        "other-seed rerun has a different placement_seed",
    )
    changed = any(
        first["position_m"] != second["position_m"]
        for first, second in zip(placement["views"], other_placement["views"])
    )
    c.check(changed, "other-seed rerun changes at least one view position")
    for key in SEED_INVARIANT_KEYS:
        c.check(
            main_raw.get(key) == other_raw.get(key),
            f"manifest {key} equal across placement seeds",
        )
    for key in PLACEMENT_INVARIANT_KEYS:
        c.check(
            placement.get(key) == other_placement.get(key),
            f"placement {key} equal across placement seeds",
        )
    c.check(
        placement["radio_map"].get("sha256") == other_placement["radio_map"].get("sha256"),
        "placement radio_map.sha256 equal across placement seeds",
    )
    first_settings = dict(placement["settings"])
    second_settings = dict(other_placement["settings"])
    first_seed = first_settings.pop("placement_seed", None)
    second_seed = second_settings.pop("placement_seed", None)
    c.check(
        first_seed != second_seed and first_settings == second_settings,
        "placement settings differ only in placement_seed",
    )


def _print_view_table(manifest, placement: Mapping) -> None:
    """Print one row per view with gains and per-BS hemisphere energies."""
    print("--- per-view coverage placement ---")
    for entry in placement["views"]:
        view_id = entry["view_id"]
        gain_db = entry.get("path_gain_db")
        gain_text = "-inf" if gain_db is None else f"{float(gain_db):.2f}"
        bs_text = ""
        try:
            manifest_view = manifest.view(view_id)
        except ManifestError:
            manifest_view = None
        if manifest_view is not None:
            parts = []
            for bs_entry in manifest_view.bs:
                front = _format_db(_db(float(bs_entry.hemisphere_energy.get("front", 0.0))))
                back = _format_db(_db(float(bs_entry.hemisphere_energy.get("back", 0.0))))
                parts.append(f"{bs_entry.bs_id}[front={front} back={back}]")
            bs_text = " ".join(parts)
        print(
            f"  {view_id} cell={entry['cell_index']} position={entry['position_m']} "
            f"gain_db={gain_text} facing_bs={entry.get('facing_bs_index')} {bs_text}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path, help="coverage placement dataset directory")
    parser.add_argument("--num-views", type=int, required=True)
    parser.add_argument("--num-bs", type=int, required=True)
    parser.add_argument("--same-seed-rerun", type=Path, default=None)
    parser.add_argument("--other-seed", type=Path, default=None)
    parser.add_argument(
        "--mock-box",
        action="store_true",
        help="also run the mock-building box / optical checks",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    c = cmo.Checker(dataset_dir)

    if args.mock_box:
        cmo.check_multiview(c, dataset_dir, args.num_views, args.num_bs)
        cmo.check_optical(c, dataset_dir)

    print("--- coverage placement manifest (#16) ---")
    try:
        manifest = load_rf_dataset_manifest(dataset_dir)
    except (ManifestError, OSError) as exc:
        c.check(False, f"dataset manifest readable: {exc}")
        _finish(c)
        return
    placement = manifest.placement
    if not isinstance(placement, Mapping):
        c.check(False, "manifest has a placement section")
        _finish(c)
        return
    c.check(placement.get("method") == "coverage", "placement method == 'coverage'")
    views = placement.get("views")
    c.check(isinstance(views, list) and len(views) == args.num_views, "placement views count")
    c.check(
        int(placement.get("candidate_count", -1)) >= args.num_views,
        "candidate_count >= num_views",
    )
    counts = placement.get("cell_counts", {})
    if isinstance(counts, Mapping) and "cells" in counts:
        total = sum(int(value) for key, value in counts.items() if key != "cells")
        c.check(total == int(counts["cells"]), "cell_counts values sum to cells")
    else:
        c.check(False, "placement has cell_counts with a 'cells' entry")
    if not isinstance(views, list) or not views:
        _finish(c)
        return

    _check_radio_map_hashes(c, dataset_dir, placement)
    _check_replan(c, dataset_dir, placement)
    _check_poses(c, dataset_dir, manifest, placement)
    _check_cells(c, dataset_dir, manifest, placement)
    _check_los(c, dataset_dir, manifest, placement)
    _check_orientation(c, manifest, placement)
    if args.same_seed_rerun is not None:
        _check_same_seed(c, dataset_dir, placement, args.same_seed_rerun)
    if args.other_seed is not None:
        _check_other_seed(c, dataset_dir, placement, args.other_seed)
    _print_view_table(manifest, placement)
    _finish(c)


def _finish(c: cmo.Checker) -> None:
    """Print the failure summary and exit non-zero when any check failed."""
    if c.failures:
        print(f"\n❌ {len(c.failures)} check(s) failed:")
        for failure in c.failures:
            print(f"  - {failure}")
        sys.exit(1)
    print("\n✅ Coverage placement outputs look sane")


if __name__ == "__main__":
    main()
