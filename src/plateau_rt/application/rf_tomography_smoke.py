"""Heavy CI tomography smoke (design §6.7, §8 T22); Sionna-free."""

from __future__ import annotations

import dataclasses
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from plateau_rt.application import rf_tomography_benchmark as bench
from plateau_rt.application import rf_tomography_gt as gt_app
from plateau_rt.application import rf_tomography_io as tio
from plateau_rt.domain.ground import GROUND_PLANE_ID
from plateau_rt.domain.rf_tomography.configs import CONFIGS

SMOKE_SCHEMA = "rf_tomo_smoke/1"
SMOKE_MANIFEST_FILE = "smoke_manifest.json"
REPORT_FILE = "report.md"
SUMMARY_FILE = "summary.json"
SMOKE_NUM_BINS = 32
SMOKE_SIGMA_T = 100e-9
SMOKE_WORKERS = 8
SMOKE_MAX_RUNTIME_S = 600.0
SMOKE_STRATEGIES = ("none", "los", "xcorr", "self_cal", "varpro", "oracle")
EXCLUDED_STRATEGIES = {
    "blind": "blind_tau_search takes ~400 s per call at 32 bins on the smoke "
    "grid (Phase 1: 6800-7600 s at 64 bins); covered by the T11 unit tests"
}
GATE_FACTOR = 0.6 * math.sqrt(3.0)
LOS_AMP_GATE_DB = 1.0
GAUGE_DELAY_GATE_NS = 1.0
GAUGE_PHASE_GATE_DEG = 10.0
SYNC_RTOL = 1e-10
SYNC_DET_ATOL_M = 1e-6
LOS_CONFIGS = ("I-S", "ID-S", "IDP-S")
IMAGE_CONFIGS = ("ID-S", "IDP-S")
NEGATIVE_CONFIG = "IDP-N"
VS_HALF_XY_M = 12.0
VS_Z_MARGIN_M = 4.0
L2_BS_POSITIONS = ((-70.0, 5.0, 25.0), (60.0, -40.0, 20.0))
L2_BUILDING_BS = 1
L2_BUILDING_CENTER = (-16.0, -38.0, 22.0)
L2_BUILDING_HALF = (20.0, 12.0, 14.0)
EXPECTED_FAILURES = {
    "mirror:ID-S": (
        "32-bin sub-band (25 MHz): the ground image shares the LoS delay cell (0.9 m vs 12 m) "
        "2.4 angular cells away and the unwindowed |c|^2 kernel of the ID GLRT is pulled by the "
        "LoS sidelobes and the LoS x ground cross term; clean-data argmax on a 0.25 m grid is "
        "6.3 m off at 32 bins and 1.5 m at 128 bins (IDP-S 0.25 / 0.0 m); T22 notes"
    ),
    "building:ID-S": (
        "the detectable building images of the 8-ring-view ci profile are each seen by one view "
        "only; the joint ID GLRT over all views is pulled by the other views' clutter: clean-data "
        "argmax 6.2 m (32 bins) / 3.2 m (128 bins) off (IDP-S 1.1 / 0.8 m); T22 notes"
    ),
}


@dataclass(frozen=True)
class SmokeRun:
    """One smoke output directory: space, BS subset, grid and job selection."""

    name: str
    kind: str
    space: str
    bs: tuple[int, ...] | None
    grid_center: tuple[float, float, float] | None
    grid_half_size: tuple[float, float, float] | None
    configs: tuple[str, ...] | None
    strategies: tuple[str, ...] = SMOKE_STRATEGIES


@dataclass(frozen=True)
class Check:
    """One smoke gate: status, measured value, gate and detail."""

    name: str
    status: str
    measured: Any
    gate: float | None
    detail: str = ""


def smoke_suite() -> bench.Suite:
    """Return the heavy-smoke suite (smoke grid, N-track 100 ns timing)."""
    return dataclasses.replace(bench.SUITES["smoke"], name="heavy-smoke", sigma_t=SMOKE_SIGMA_T)


def smoke_plan(bs_pos: np.ndarray) -> tuple[SmokeRun, ...]:
    """Plan the smoke runs for the BS positions ``bs_pos`` ``[B, 3]``."""
    positions = np.asarray(bs_pos, dtype=np.float64).reshape(-1, 3)
    runs: list[SmokeRun] = [SmokeRun("bv", "bv", "bv", None, None, None, None)]
    for b in range(positions.shape[0]):
        x_b, y_b, z_b = (float(positions[b, 0]), float(positions[b, 1]), float(positions[b, 2]))
        runs.append(
            SmokeRun(
                f"vs_bs_{b:03d}",
                "bs",
                "vs",
                (b,),
                (x_b, y_b, 0.0),
                (VS_HALF_XY_M, VS_HALF_XY_M, abs(z_b) + VS_Z_MARGIN_M),
                None,
            )
        )
    if positions.shape == (2, 3) and np.allclose(positions, L2_BS_POSITIONS, atol=1e-6):
        runs.append(
            SmokeRun(
                "vs_building",
                "building",
                "vs",
                (L2_BUILDING_BS,),
                L2_BUILDING_CENTER,
                L2_BUILDING_HALF,
                ("ID-S", "IDP-S"),
                ("none",),
            )
        )
    return tuple(runs)


def run_smoke(
    dataset: Path | str,
    out_dir: Path | str,
    *,
    workers: int = SMOKE_WORKERS,
    num_bins: int = SMOKE_NUM_BINS,
    plan: Sequence[SmokeRun] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run every smoke plan entry and write the smoke manifest."""
    data = tio.load_dataset(dataset)
    if tio.find_ground_truth(data) is None:
        raise ValueError("dataset has no tomography_gt; run rf-tomo-gt first")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / SMOKE_MANIFEST_FILE).exists() and not overwrite:
        raise FileExistsError(f"smoke manifest exists in {out}")
    runs = smoke_plan(data.geom.bs_pos) if plan is None else tuple(plan)
    suite = smoke_suite()
    entries: list[dict[str, Any]] = []
    total = 0.0
    for run in runs:
        started = time.perf_counter()
        result = bench.run_benchmark(
            data,
            out / run.name,
            suite,
            configs=run.configs,
            strategies=run.strategies,
            spaces=[run.space],
            grid_center=run.grid_center,
            grid_half_size=run.grid_half_size,
            bs=run.bs,
            num_bins=num_bins,
            overwrite=overwrite,
            workers=workers,
        )
        runtime = time.perf_counter() - started
        total += runtime
        counts = {"ok": 0, "n/a": 0, "error": 0}
        for row in result.rows:
            counts[str(row["status"])] += 1
        entries.append(
            {
                "name": run.name,
                "kind": run.kind,
                "space": run.space,
                "bs": None if run.bs is None else [int(v) for v in run.bs],
                "grid_center": None if run.grid_center is None else list(run.grid_center),
                "grid_half_size": None if run.grid_half_size is None else list(run.grid_half_size),
                "configs": None if run.configs is None else list(run.configs),
                "strategies": list(run.strategies),
                "runtime_s": float(runtime),
                "status_counts": counts,
            }
        )
    manifest = {
        "schema": SMOKE_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset),
        "num_bins": int(num_bins),
        "suite": dataclasses.asdict(suite),
        "workers": int(workers),
        "excluded_strategies": dict(EXCLUDED_STRATEGIES),
        "runs": entries,
        "runtime_s": float(total),
    }
    (out / SMOKE_MANIFEST_FILE).write_text(
        json.dumps(tio.to_jsonable(manifest), indent=2), encoding="utf-8"
    )
    return manifest


def classify(name: str, passed: bool, expected: Mapping[str, str] = EXPECTED_FAILURES) -> str:
    """Classify a boolean outcome with the expected-failure table."""
    if passed:
        return "xpass" if name in expected else "pass"
    return "xfail" if name in expected else "fail"


def exit_code(checks: Sequence[Check]) -> int:
    """Return 1 when any check failed, else 0."""
    return 1 if any(check.status == "fail" for check in checks) else 0


def min_distance(detections: np.ndarray, target: np.ndarray) -> float:
    """Return the minimum Euclidean distance of ``detections`` to ``target``."""
    det = np.asarray(detections, dtype=np.float64).reshape(-1, 3)
    point = np.asarray(target, dtype=np.float64).reshape(3)
    if det.shape[0] == 0:
        return math.inf
    return float(np.min(np.linalg.norm(det - point, axis=1)))


def _check(
    name: str, passed: bool, measured: Any, gate: float | None, expected: Mapping[str, str]
) -> Check:
    """Build a Check whose status and detail follow the expected-failure table."""
    return Check(
        name, classify(name, bool(passed), expected), measured, gate, expected.get(name, "")
    )


def check_localisation(
    name: str,
    detections: np.ndarray,
    target: np.ndarray,
    gate: float,
    expected: Mapping[str, str] = EXPECTED_FAILURES,
) -> Check:
    """Check that the closest detection to ``target`` is within ``gate`` (inclusive)."""
    distance = min_distance(detections, target)
    return _check(name, distance <= gate, {"distance_m": distance}, float(gate), expected)


def check_negative_control(
    name: str,
    d_none: float,
    d_los: float,
    d_oracle: float,
    gate: float,
    expected: Mapping[str, str] = EXPECTED_FAILURES,
) -> Check:
    """Check the negative control: only the uncorrected N map misses the LoS."""
    passed = d_none > gate and d_los <= gate and d_oracle <= gate
    measured = {"d_none": float(d_none), "d_los": float(d_los), "d_oracle": float(d_oracle)}
    return _check(name, passed, measured, float(gate), expected)


def check_los_amplitude(
    name: str,
    payload: Mapping[str, Any] | None,
    los_visible: np.ndarray,
    expected: Mapping[str, str] = EXPECTED_FAILURES,
) -> Check:
    """Check the fitted LoS magnitude against the path GT on the LoS-visible captures."""
    fit = payload.get("los_fit") if isinstance(payload, Mapping) else None
    fit = fit if isinstance(fit, Mapping) else {}
    worst = fit.get("amp_error_db_max")
    passed = (
        fit.get("fitted") == np.asarray(los_visible, dtype=bool).tolist()
        and worst is not None
        and float(worst) <= LOS_AMP_GATE_DB
    )
    measured = {"amp_error_db_max": worst, "num_fitted": fit.get("num_fitted")}
    return _check(name, passed, measured, LOS_AMP_GATE_DB, expected)


def check_gauge_los(
    name: str,
    payload: Mapping[str, Any] | None,
    los_visible: np.ndarray,
    expected: Mapping[str, str] = EXPECTED_FAILURES,
) -> Check:
    """Check per-capture LoS-visible gauge errors (delay < 1 ns, phase < 10 deg + |eps|)."""
    payload = payload if isinstance(payload, Mapping) else {}
    entry = payload.get("by_los", {}).get("los_visible", {})
    fit = payload.get("los_fit") if isinstance(payload.get("los_fit"), Mapping) else {}
    num = int(entry.get("num", 0))
    passed = num >= 1 and fit.get("fitted") == np.asarray(los_visible, dtype=bool).tolist()
    delay_max = excess_max = None
    if "delay_ns" in entry:
        delays = [float(value) for value in entry["delay_ns"]]
        delay_max = max(delays, default=None)
        passed = passed and all(value < GAUGE_DELAY_GATE_NS for value in delays)
    if "phase_deg" in entry:
        phases = [abs(float(value)) for value in entry["phase_deg"]]
        raw_eps = entry.get("los_model_error_deg") or [None] * len(phases)
        eps = [0.0 if value is None else abs(float(value)) for value in raw_eps]
        excess = [phase - slack for phase, slack in zip(phases, eps, strict=True)]
        excess_max = max(excess, default=None)
        passed = passed and all(value < GAUGE_PHASE_GATE_DEG for value in excess)
    measured = {"num": num, "delay_max_ns": delay_max, "phase_excess_max_deg": excess_max}
    return _check(name, passed, measured, GAUGE_DELAY_GATE_NS, expected)


def check_sync_pairs(
    name: str,
    pairs: Sequence[tuple[Mapping[str, np.ndarray], Mapping[str, np.ndarray]]],
    expected: Mapping[str, str] = EXPECTED_FAILURES,
) -> Check:
    """Check that paired S/N recon maps and detections agree (I_S == I_N)."""
    passed = len(pairs) >= 1
    max_rel = max_det = 0.0
    for first, second in pairs:
        for key in ("map", "detections"):
            if key not in first:
                continue
            a = np.asarray(first[key], dtype=np.float64)
            b = np.asarray(second.get(key, np.zeros((0,))), dtype=np.float64)
            if a.shape != b.shape:
                passed = False
                continue
            gap = float(np.max(np.abs(a - b))) if a.size else 0.0
            if key == "map":
                scale = float(np.max(np.abs(a))) if a.size else 0.0
                max_rel = max(
                    max_rel, gap / scale if scale > 0.0 else (0.0 if gap == 0 else math.inf)
                )
                passed = passed and gap <= SYNC_RTOL * scale
            else:
                max_det = max(max_det, gap)
                passed = passed and gap <= SYNC_DET_ATOL_M
    measured = {"pairs": len(pairs), "max_rel": max_rel, "max_det_m": max_det}
    return _check(name, passed, measured, None, expected)


def _load_recon(run_dir: Path, relpath: str) -> dict[str, np.ndarray]:
    """Load one recon npz into a plain array dict."""
    with np.load(run_dir / relpath, allow_pickle=False) as payload:
        return {name: np.asarray(payload[name]) for name in payload.files}


def _is_int(value: Any) -> bool:
    """Return True for genuine ints (bools excluded)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _e1_row(rows: Sequence[Mapping[str, Any]], config: str, strategy: str) -> Mapping | None:
    """Return the ok E1 row of ``config`` / ``strategy`` with a recon, or None."""
    for row in rows:
        if (
            row["config"] == config
            and row["strategy"] == strategy
            and row["stage"] == "E1"
            and row["status"] == "ok"
            and row["recon"] is not None
        ):
            return row
    return None


def _detections(run_dir: Path, row: Mapping[str, Any] | None) -> np.ndarray:
    """Return the detections of ``row`` (empty without a row)."""
    if row is None:
        return np.zeros((0, 3), dtype=np.float64)
    return _load_recon(run_dir, str(row["recon"]))["detections"]


def _los_chains(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], Mapping[str, Any]]:
    """Return the first row with a gauge payload of every ``los`` chain, keyed by config/track."""
    chains: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        if row["strategy"] == "los" and row.get("gauge_errors") is not None:
            chains.setdefault((str(row["config"]), str(row["track"])), row)
    return dict(sorted(chains.items()))


def _schema_check(
    name: str,
    rows: Sequence[Mapping[str, Any]],
    run_manifest: Mapping[str, Any],
    run_dir: Path,
    expected: Mapping[str, str],
) -> Check:
    """Check row schemas, statuses, finite recon arrays and the run manifest of one run."""
    problems: list[str] = []
    counts = {"rows": len(rows), "ok": 0, "n/a": 0, "error": 0}
    for index, row in enumerate(rows):
        try:
            if list(row) != list(tio.RESULT_KEYS):
                raise ValueError("keys out of order")
            tio.validate_result_row(row)
        except ValueError as error:
            problems.append(f"row {index}: {error}")
            continue
        status = str(row["status"])
        counts[status] += 1
        if status == "error":
            problems.append(f"row {index} error: {row['config']}/{row['strategy']}")
        if status != "ok":
            continue
        if row["stage"] == "E2" and not all(
            _is_int(row[key]) for key in ("n_iter", "n_forward", "n_adjoint")
        ):
            problems.append(f"row {index}: E2 counts are not ints")
        if row["recon"] is None:
            continue
        if not (run_dir / str(row["recon"])).exists():
            problems.append(f"row {index}: missing recon")
            continue
        arrays = _load_recon(run_dir, str(row["recon"]))
        keys = ["map", "density", "detections", "scores", "centers"]
        if row["strategy"] != "none":
            keys.append("gauges")
        for key in keys:
            if key in arrays and not bool(np.all(np.isfinite(arrays[key]))):
                problems.append(f"row {index}: non-finite {key}")
    try:
        tio.validate_run_manifest(run_manifest)
        if list(run_manifest) != list(tio.RUN_MANIFEST_KEYS):
            raise ValueError("run manifest keys out of order")
    except ValueError as error:
        problems.append(str(error))
    suite = run_manifest.get("suite")
    if not isinstance(suite, Mapping) or (
        suite.get("realizations"),
        suite.get("e2_iterations"),
    ) != (
        1,
        10,
    ):
        problems.append("suite is not R = 1 with 10 E2 iterations")
    return _check(name, not problems, {**counts, "problems": problems[:20]}, None, expected)


def _object_names(gt_arrays: Mapping[str, np.ndarray]) -> list[str]:
    """Return the path-GT object names indexed by ``vs_objects``."""
    return [str(value) for value in np.asarray(gt_arrays["path_object_names"]).tolist()]


def ground_image_target(gt_arrays: Mapping[str, np.ndarray], bs: int) -> np.ndarray | None:
    """Return the first-order ground-image VS of BS ``bs`` (None when the GT has none)."""
    names = _object_names(gt_arrays)
    for m in range(np.asarray(gt_arrays["vs_pos"]).shape[0]):
        if int(gt_arrays["vs_bs"][m]) != bs or int(gt_arrays["vs_order"][m]) != 1:
            continue
        objects = [int(o) for o in np.asarray(gt_arrays["vs_objects"][m]) if o >= 0]
        if objects and all(names[o] == GROUND_PLANE_ID for o in objects):
            return np.asarray(gt_arrays["vs_pos"][m], dtype=np.float64)
    return None


def building_targets(
    gt_arrays: Mapping[str, np.ndarray], bs: int, lower: np.ndarray, upper: np.ndarray
) -> np.ndarray:
    """Return the detectable building VS ``[K, 3]`` of BS ``bs`` inside ``[lower, upper]``.

    A building VS has order >= 1 and at least one interaction object that is not
    the ground plane.
    """
    names = _object_names(gt_arrays)
    detectable = gt_app.vs_detectable(gt_arrays)
    vs_pos = np.asarray(gt_arrays["vs_pos"], dtype=np.float64)
    keep = []
    for m in range(vs_pos.shape[0]):
        objects = [int(o) for o in np.asarray(gt_arrays["vs_objects"][m]) if o >= 0]
        keep.append(
            int(gt_arrays["vs_bs"][m]) == bs
            and bool(detectable[m])
            and int(gt_arrays["vs_order"][m]) >= 1
            and any(names[o] != GROUND_PLANE_ID for o in objects)
            and bool(np.all(vs_pos[m] >= lower - 1e-9) and np.all(vs_pos[m] <= upper + 1e-9))
        )
    return vs_pos[np.asarray(keep, dtype=bool)].reshape(-1, 3)


@dataclass(frozen=True)
class _RunOutput:
    """One finished smoke run read back from disk."""

    spec: Mapping[str, Any]
    run_dir: Path
    rows: list[dict[str, Any]]
    manifest: dict[str, Any]

    @property
    def gate(self) -> float:
        """Localisation gate ``GATE_FACTOR * spacing`` of the run grid."""
        return float(GATE_FACTOR * self.manifest["grid"]["spacing"])

    @property
    def los_visible(self) -> np.ndarray:
        """LoS visibility ``[V, B]`` of the (selected) captures of the run."""
        return np.asarray(self.manifest["dataset"]["los_visible"], dtype=bool)

    @property
    def grid_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """First and last voxel centres of the run grid."""
        grid = self.manifest["grid"]
        origin = np.asarray(grid["origin"], dtype=np.float64)
        return origin, origin + (np.asarray(grid["shape"]) - 1.0) * float(grid["spacing"])


def _read_runs(out: Path, manifest: Mapping[str, Any]) -> list[_RunOutput]:
    """Read every run of a smoke manifest."""
    runs = []
    for spec in manifest["runs"]:
        run_dir = out / str(spec["name"])
        rows = tio.read_results(run_dir / tio.RESULTS_FILE)
        run_manifest = json.loads((run_dir / tio.RUN_MANIFEST_FILE).read_text(encoding="utf-8"))
        runs.append(_RunOutput(spec, run_dir, rows, run_manifest))
    return runs


def evaluate_smoke(
    out_dir: Path | str,
    *,
    dataset: Path | str | None = None,
    max_runtime_s: float = SMOKE_MAX_RUNTIME_S,
    expected: Mapping[str, str] = EXPECTED_FAILURES,
) -> list[Check]:
    """Evaluate a smoke output directory into the ordered §6.7 checks."""
    out = Path(out_dir)
    manifest = json.loads((out / SMOKE_MANIFEST_FILE).read_text(encoding="utf-8"))
    data = tio.load_dataset(dataset if dataset is not None else manifest["dataset"])
    gt = tio.find_ground_truth(data)
    if gt is None:
        raise ValueError("dataset has no tomography_gt; run rf-tomo-gt first")
    bs_pos = np.asarray(data.geom.bs_pos, dtype=np.float64)
    runs = _read_runs(out, manifest)
    bv = next((run for run in runs if run.spec["kind"] == "bv"), None)
    per_bs = [run for run in runs if run.spec["kind"] == "bs"]
    checks = [
        _schema_check(f"schema:{run.spec['name']}", run.rows, run.manifest, run.run_dir, expected)
        for run in runs
    ]

    bv_rows = [] if bv is None else bv.rows
    ran = {str(row["config"]) for row in bv_rows}
    ok_e1 = {
        str(row["config"]) for row in bv_rows if row["stage"] == "E1" and row["status"] == "ok"
    }
    missing = sorted(set(CONFIGS) - ran)
    no_ok_e1 = sorted(
        name for name, cfg in CONFIGS.items() if cfg.e1 is not None and name not in ok_e1
    )
    checks.append(
        _check(
            "all_nodes",
            not missing and not no_ok_e1,
            {"missing": missing, "no_ok_e1": no_ok_e1},
            None,
            expected,
        )
    )

    for run in per_bs:
        b = int(run.spec["bs"][0])
        for cfg in LOS_CONFIGS:
            row = _e1_row(run.rows, cfg, "none")
            checks.append(
                check_localisation(
                    f"los_e1:{cfg}:bs_{b:03d}",
                    _detections(run.run_dir, row),
                    bs_pos[b],
                    run.gate,
                    expected,
                )
            )

    los_chains = {} if bv is None else _los_chains(bv.rows)
    for (cfg, track), row in los_chains.items():
        checks.append(
            check_los_amplitude(
                f"los_amplitude:{cfg}:{track}", row["gauge_errors"], bv.los_visible, expected
            )
        )

    for run in per_bs:
        if [int(b) for b in run.spec["bs"]] != [0]:
            continue
        target = ground_image_target(gt.arrays, 0)
        for cfg in IMAGE_CONFIGS:
            name = f"mirror:{cfg}"
            if target is None:
                checks.append(_check(name, False, {"distance_m": None}, run.gate, expected))
                continue
            row = _e1_row(run.rows, cfg, "none")
            checks.append(
                check_localisation(name, _detections(run.run_dir, row), target, run.gate, expected)
            )

    for run in runs:
        if run.spec["kind"] != "building":
            continue
        targets = building_targets(gt.arrays, int(run.spec["bs"][0]), *run.grid_bounds)
        for cfg in IMAGE_CONFIGS:
            det = _detections(run.run_dir, _e1_row(run.rows, cfg, "none"))
            distance = min((min_distance(det, point) for point in targets), default=math.inf)
            checks.append(
                _check(
                    f"building:{cfg}",
                    distance <= run.gate,
                    {"distance_m": distance, "num_targets": int(targets.shape[0])},
                    run.gate,
                    expected,
                )
            )

    for run in runs:
        keyed: dict[str, dict[tuple[str, str, str], Mapping[str, Any]]] = {"I-S": {}, "I-N": {}}
        for row in run.rows:
            if row["config"] in keyed and row["strategy"] == "none" and row["status"] == "ok":
                if row["recon"] is not None:
                    keyed[row["config"]][(row["space"], row["stage"], row["solver"])] = row
        if not keyed["I-S"] or not keyed["I-N"]:
            continue
        pairs = [
            (
                _load_recon(run.run_dir, str(row["recon"])),
                _load_recon(run.run_dir, str(keyed["I-N"][key]["recon"])),
            )
            for key, row in keyed["I-S"].items()
            if key in keyed["I-N"]
        ]
        checks.append(check_sync_pairs(f"sync_invariance:{run.spec['name']}", pairs, expected))

    for (cfg, track), row in los_chains.items():
        checks.append(
            check_gauge_los(
                f"gauge_los:{cfg}:{track}", row["gauge_errors"], bv.los_visible, expected
            )
        )

    for run in per_bs:
        b = int(run.spec["bs"][0])
        distances = [
            min_distance(
                _detections(run.run_dir, _e1_row(run.rows, NEGATIVE_CONFIG, strategy)), bs_pos[b]
            )
            for strategy in ("none", "los", "oracle")
        ]
        checks.append(
            check_negative_control(f"negative_control:bs_{b:03d}", *distances, run.gate, expected)
        )

    runtime = float(manifest["runtime_s"])
    checks.append(
        _check("runtime", runtime <= max_runtime_s, runtime, float(max_runtime_s), expected)
    )
    return checks


def _format_point(point: Sequence[float] | None) -> str:
    """Format an optional xyz triple for the report."""
    if point is None:
        return "-"
    return f"({float(point[0]):.1f}, {float(point[1]):.1f}, {float(point[2]):.1f})"


def _format_value(value: Any) -> str:
    """Format an optional finite number with two decimals ("-" otherwise)."""
    if value is None or not np.isfinite(float(value)):
        return "-"
    return f"{float(value):.2f}"


def write_smoke_report(out_dir: Path | str, checks: Sequence[Check]) -> tuple[Path, Path]:
    """Write ``summary.json`` and ``report.md`` for a smoke output directory."""
    out = Path(out_dir)
    manifest = json.loads((out / SMOKE_MANIFEST_FILE).read_text(encoding="utf-8"))
    data = tio.load_dataset(manifest["dataset"])
    bs_pos = np.asarray(data.geom.bs_pos, dtype=np.float64)
    counts = {"pass": 0, "fail": 0, "xfail": 0, "xpass": 0}
    for check in checks:
        counts[check.status] += 1
    summary = {
        "schema": SMOKE_SCHEMA,
        "smoke_manifest": manifest,
        "checks": [dataclasses.asdict(check) for check in checks],
        "counts": counts,
        "ok": exit_code(checks) == 0,
    }
    summary_path = out / SUMMARY_FILE
    summary_path.write_text(json.dumps(tio.to_jsonable(summary), indent=2), encoding="utf-8")

    suite = manifest["suite"]
    lines = [
        "# Tomography heavy smoke (T22)",
        "",
        f"- dataset: {manifest['dataset']}",
        f"- num_bins: {manifest['num_bins']} (central sub-band)",
        f"- sigma_t: {suite['sigma_t']} s, snr_db: {suite['snr_db']}, "
        f"e2_iterations: {suite['e2_iterations']}, realizations: {suite['realizations']}",
        f"- workers: {manifest['workers']}",
        *(f"- excluded strategy {k}: {v}" for k, v in manifest["excluded_strategies"].items()),
        f"- runtime_s: {float(manifest['runtime_s']):.1f}",
        f"- result: {counts}",
        "",
        "## Runs",
        "",
        "| name | kind | space | bs | grid center | half size | ok | n/a | error | time_s |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for run in manifest["runs"]:
        bs = "all" if run["bs"] is None else ",".join(str(v) for v in run["bs"])
        status = run["status_counts"]
        lines.append(
            f"| {run['name']} | {run['kind']} | {run['space']} | {bs} | "
            f"{_format_point(run['grid_center'])} | {_format_point(run['grid_half_size'])} | "
            f"{status['ok']} | {status['n/a']} | {status['error']} | {run['runtime_s']:.1f} |"
        )
    lines += [
        "",
        "## Checks",
        "",
        "| check | status | measured | gate |",
        "| --- | --- | --- | --- |",
    ]
    for check in checks:
        measured = json.dumps(tio.to_jsonable(check.measured), sort_keys=True)
        gate = "-" if check.gate is None else f"{check.gate:g}"
        lines.append(f"| {check.name} | {check.status.upper()} | {measured} | {gate} |")
    lines += ["", "## Expected failures", ""]
    flagged = [check for check in checks if check.status in ("xfail", "xpass")]
    lines += [f"- {c.name} ({c.status.upper()}): {c.detail}" for c in flagged] or ["- none"]

    runs = _read_runs(out, manifest)
    lines += [
        "",
        "## VS localisation (E1, strategy none)",
        "",
        "Distances in m from the closest detection to the BS (LoS VS) and to its ground image.",
        "",
        "| run | config | track | d_los | d_image |",
        "| --- | --- | --- | --- | --- |",
    ]
    for run in runs:
        if run.spec["kind"] not in ("bs", "building"):
            continue
        b = int(run.spec["bs"][0])
        image = bs_pos[b] * np.array([1.0, 1.0, -1.0])
        for row in run.rows:
            if row["stage"] == "E1" and row["strategy"] == "none" and row["status"] == "ok":
                det = _detections(run.run_dir, row)
                lines.append(
                    f"| {run.spec['name']} | {row['config']} | {row['track']} | "
                    f"{_format_value(min_distance(det, bs_pos[b]))} | "
                    f"{_format_value(min_distance(det, image))} |"
                )
    lines += [
        "",
        "## Gauge errors by LoS visibility",
        "",
        "Per stratum: captures / phase max (deg) / delay max (ns), after the stratum's own "
        "best global phase.",
        "",
        "| config | track | strategy | LoS visible | LoS blocked |",
        "| --- | --- | --- | --- | --- |",
    ]
    for run in runs:
        if run.spec["kind"] != "bv":
            continue
        for row in run.rows:
            if row["stage"] != "E1" or row.get("gauge_errors") is None:
                continue
            cells = []
            for stratum in gt_app.LOS_STRATA:
                entry = row["gauge_errors"].get("by_los", {}).get(stratum, {})
                cells.append(
                    f"{entry.get('num', '-')} / {_format_value(entry.get('phase_max_deg'))} / "
                    f"{_format_value(entry.get('delay_max_ns'))}"
                )
            lines.append(
                f"| {row['config']} | {row['track']} | {row['strategy']} | "
                + " | ".join(cells)
                + " |"
            )
    lines.append("")
    report_path = out / REPORT_FILE
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path, report_path
