"""Sionna-free comparison specs, statistics and text for solver profiles.

This module must never import sionna, mitsuba or drjit so it stays usable in
CPU-only unit tests and post-processing. All Sionna work (scene setup,
tracing, timing, material changes, I/O) lives in ``scripts/analysis``.

Noise-floor method statement: for a non-noise comparison ``(test, baseline)``
the noise-floor MATRIX ``[num_views, 2]`` is the element-wise max of the
matrices that exist among:

- the seed-noise matrix of ``SEED_NOISE_SOURCE[test]`` (the
  ``rel_diff_matrix`` of that seed-noise comparison);
- the seed-noise matrix of ``SEED_NOISE_SOURCE[baseline]``;
- the same-seed repeat matrix of ``test`` (its
  ``repeat_diff.rel_diff_matrix``);
- the same-seed repeat matrix of ``baseline``.

NaN entries are treated as missing (ignored in the element-wise max); an inf
entry in any source stays inf in the floor. When no source matrix exists the
floor is None and the comparison is ``unresolved (no floor)``. The
scene-wide repeat max is never used as a floor; it is reported in the JSON
as ``repeat_floor_overall`` for information only.

A cell ``(v, h)`` is resolved when ``diff[v, h] > RESOLVED_FACTOR *
floor[v, h]``. An inf diff over a finite floor is resolved; an inf floor
cell never resolves; NaN in either side never resolves. ``resolved`` is
true when any cell is resolved. ``resolved_max`` is the max diff over the
resolved cells (0.0 when none) and ``noise_floor`` is the scalar max of the
floor matrix, for display only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from plateau_rt.domain.rf_camera.solver_metrics import relative_cfr_difference

# A cell counts as resolved when its diff exceeds this factor times the
# floor cell.
RESOLVED_FACTOR = 2.0

# Effect classes on the resolved max per-view-hemisphere relative difference.
EFFECT_SMALL_BELOW = 0.05
EFFECT_MODERATE_BELOW = 0.3

FAMILY_AS_LOADED = "as_loaded"
FAMILY_SCATTERING_S03 = "scattering_s0.3"

# Every profile the benchmark may produce, mapped to its material family.
PROFILE_FAMILY: dict[str, str] = {
    "los_only": FAMILY_AS_LOADED,
    "specular": FAMILY_AS_LOADED,
    "current": FAMILY_AS_LOADED,
    "current_depth1": FAMILY_AS_LOADED,
    "current_depth3": FAMILY_AS_LOADED,
    "current_explicit_array": FAMILY_AS_LOADED,
    "plus_diffraction": FAMILY_AS_LOADED,
    "complete": FAMILY_AS_LOADED,
    "complete_seed2": FAMILY_AS_LOADED,
    "current_s03": FAMILY_SCATTERING_S03,
    "plus_diffuse_s03": FAMILY_SCATTERING_S03,
    "complete_s03": FAMILY_SCATTERING_S03,
    "plus_diffuse_s03_seed2": FAMILY_SCATTERING_S03,
    "complete_s03_seed2": FAMILY_SCATTERING_S03,
}

# Each profile maps to the seed-noise comparison covering its stochastic
# mechanisms. Sionna samples the interaction type per bounce with the seeded
# sampler, so specular+refraction+diffraction profiles are seed-dependent
# too and map to the complete seed pair; only the s03 diffuse/diffraction
# profiles map to their own s03 seed pairs.
SEED_NOISE_SOURCE: dict[str, tuple[str, str]] = {
    "los_only": ("complete_seed2", "complete"),
    "specular": ("complete_seed2", "complete"),
    "current": ("complete_seed2", "complete"),
    "current_depth1": ("complete_seed2", "complete"),
    "current_depth3": ("complete_seed2", "complete"),
    "current_explicit_array": ("complete_seed2", "complete"),
    "plus_diffraction": ("complete_seed2", "complete"),
    "complete": ("complete_seed2", "complete"),
    "complete_seed2": ("complete_seed2", "complete"),
    "current_s03": ("complete_seed2", "complete"),
    "plus_diffuse_s03": ("plus_diffuse_s03_seed2", "plus_diffuse_s03"),
    "plus_diffuse_s03_seed2": ("plus_diffuse_s03_seed2", "plus_diffuse_s03"),
    "complete_s03": ("complete_s03_seed2", "complete_s03"),
    "complete_s03_seed2": ("complete_s03_seed2", "complete_s03"),
}

CAVEATS: tuple[str, ...] = (
    "Sionna GPU tracing is not bit-reproducible; compare with tolerances, "
    "never with exact equality.",
    "ITU radio materials default to scattering_coefficient=0, so "
    "diffuse_reflection=True alone adds no energy.",
    "The scattering_s0.3 family sets S=0.3 on every radio material "
    "(restored afterwards); S>0 also scales down specular reflections.",
    "Method: each non-noise comparison uses a per-cell noise-floor matrix "
    "[num_views, 2], the element-wise max of the seed-noise matrix of "
    "SEED_NOISE_SOURCE[test], the seed-noise matrix of "
    "SEED_NOISE_SOURCE[baseline], and the same-seed repeat matrices of "
    "test/baseline where they exist; a cell is resolved when diff > 2x its "
    "floor cell; the scene-wide repeat max is information only, never a floor.",
)


@dataclass(frozen=True)
class ComparisonSpec:
    """One step-wise ``(test, baseline, question)`` comparison."""

    test: str
    baseline: str
    question: str
    is_noise: bool = False


def family_of(profile: str) -> str:
    """Return the material family of ``profile`` (``unknown`` if unlisted)."""
    return PROFILE_FAMILY.get(profile, "unknown")


def comparison_specs() -> list[ComparisonSpec]:
    """Return the step-wise within-family comparison list.

    The only allowed cross-family pair is ``current_s03`` vs ``current``,
    labelled as a material effect.
    """
    return [
        ComparisonSpec("specular", "los_only", "specular reflection"),
        ComparisonSpec("current", "specular", "refraction"),
        ComparisonSpec("current_depth1", "current", "depth truncation 1 vs 5"),
        ComparisonSpec("current_depth3", "current", "depth truncation 3 vs 5"),
        ComparisonSpec("plus_diffraction", "current", "diffraction"),
        ComparisonSpec("complete", "plus_diffraction", "diffuse with the loaded materials"),
        ComparisonSpec("current_explicit_array", "current", "synthetic vs explicit array"),
        ComparisonSpec(
            "current_s03",
            "current",
            "material change only (S=0.3 weakens specular)",
        ),
        ComparisonSpec("plus_diffuse_s03", "current_s03", "diffuse scattering at S=0.3"),
        ComparisonSpec("complete_s03", "plus_diffuse_s03", "diffraction under S=0.3"),
        ComparisonSpec(
            "complete_seed2",
            "complete",
            "seed noise floor (as_loaded complete)",
            is_noise=True,
        ),
        ComparisonSpec(
            "complete_s03_seed2",
            "complete_s03",
            "seed noise floor (s03 complete)",
            is_noise=True,
        ),
        ComparisonSpec(
            "plus_diffuse_s03_seed2",
            "plus_diffuse_s03",
            "seed noise floor (s03 diffuse)",
            is_noise=True,
        ),
    ]


def as_loaded_vs_complete_order() -> list[str]:
    """Return as_loaded profile names for the secondary vs-complete table."""
    return [name for name, fam in PROFILE_FAMILY.items() if fam == FAMILY_AS_LOADED]


def _fmt(value: float) -> str:
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    if isinstance(value, float) and math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{float(value):.3g}"


def json_safe(obj: Any) -> Any:
    """Recursively convert ``obj`` to strict-JSON-safe Python types.

    Non-finite floats (Python and NumPy) become the strings ``"inf"``,
    ``"-inf"`` and ``"nan"``; NumPy scalars become Python scalars and
    NumPy arrays become nested lists. All other values pass through
    (dict keys are stringified when needed by the caller).
    """
    if isinstance(obj, np.ndarray):
        return [json_safe(v) for v in obj.tolist()]
    if isinstance(obj, (np.floating, np.integer, np.bool_)):
        return json_safe(obj.item())
    if isinstance(obj, float):
        if math.isnan(obj):
            return "nan"
        if math.isinf(obj):
            return "inf" if obj > 0 else "-inf"
        return obj
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def _is_scalar_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and np.ndim(value) == 0


def cell_is_resolved(diff_val: float, floor_val: float | None) -> bool:
    """Return whether one cell ``diff_val`` resolves over ``floor_val``.

    An inf floor cell never resolves; NaN on either side never resolves;
    a None/NaN floor never resolves. An inf diff over a finite floor is
    resolved.
    """
    if floor_val is None:
        return False
    try:
        d = float(diff_val)
        f = float(floor_val)
    except (TypeError, ValueError):
        return False
    if math.isnan(d) or math.isnan(f):
        return False
    if math.isinf(f):
        return False
    if math.isinf(d):
        return math.isfinite(f)
    return d > RESOLVED_FACTOR * f


def resolve_cells(diff_matrix: Any, floor_matrix: Any | None) -> tuple[np.ndarray, int, float]:
    """Resolve ``diff_matrix`` against ``floor_matrix`` per cell.

    Returns ``(resolved_matrix[bool], n_resolved_cells, resolved_max)``
    where ``resolved_max`` is the max diff over the resolved cells, or 0.0
    when none resolves (or the floor is None).
    """
    diff = np.asarray(diff_matrix, dtype=np.float64)
    if floor_matrix is None:
        resolved = np.zeros_like(diff, dtype=bool)
        return resolved, 0, 0.0
    floor = np.asarray(floor_matrix, dtype=np.float64)
    if floor.shape != diff.shape:
        raise ValueError(f"shape mismatch: diff={diff.shape}, floor={floor.shape}")
    with np.errstate(invalid="ignore"):
        finite_floor = np.isfinite(floor)
        finite_diff = np.isfinite(diff)
        inf_diff = np.isposinf(diff)
        above = diff > RESOLVED_FACTOR * floor
        # inf floor cells never resolve; NaN comparisons are False already.
        resolved = (above | (inf_diff & finite_floor)) & ~np.isnan(diff) & ~np.isnan(floor)
        resolved = resolved & ~np.isinf(floor)
        # Guard: negative-inf diff never resolves.
        resolved = resolved & (finite_diff | inf_diff)
    n_resolved = int(np.sum(resolved))
    if n_resolved:
        resolved_max = float(np.max(diff[resolved]))
    else:
        resolved_max = 0.0
    return resolved, n_resolved, resolved_max


def _classify_magnitude(resolved_max: float) -> str:
    if float(resolved_max) < EFFECT_SMALL_BELOW:
        return "small"
    if float(resolved_max) < EFFECT_MODERATE_BELOW:
        return "moderate"
    return "large"


def is_resolved(diff: Any, noise_floor: Any | None) -> bool:
    """Return whether any cell of ``diff`` resolves over ``noise_floor``.

    Accepts scalars (single-cell rule) or ``[num_views, 2]`` matrices
    (per-cell rule: any cell with ``diff > RESOLVED_FACTOR * floor``).
    A None floor never resolves. An inf floor cell never resolves; an inf
    diff over a finite floor resolves.
    """
    if noise_floor is None:
        return False
    if _is_scalar_number(diff) and _is_scalar_number(noise_floor):
        d = float(diff)  # type: ignore[arg-type]
        f = float(noise_floor)  # type: ignore[arg-type]
        return cell_is_resolved(d, f)
    try:
        _resolved, n, _ = resolve_cells(diff, noise_floor)
    except ValueError:
        # Scalar-vs-matrix broadcast fallback: compare element-wise.
        diff_arr = np.asarray(diff, dtype=np.float64)
        floor_arr = np.asarray(noise_floor, dtype=np.float64)
        try:
            floor_b, diff_b = np.broadcast_arrays(floor_arr, diff_arr)
        except ValueError:
            return False
        _resolved, n, _ = resolve_cells(diff_b, floor_b)
    return bool(n > 0)


def classify_effect(diff: Any, noise_floor: Any | None) -> str:
    """Classify ``diff`` against ``noise_floor`` per cell.

    Returns ``"unresolved (no floor)"`` when the floor is None, ``"within
    noise"`` when no cell resolves, otherwise ``small``/``moderate``/``large``
    from ``resolved_max`` (small < 0.05, moderate < 0.3, large).
    Accepts scalars or matrices like :func:`is_resolved`.
    """
    if noise_floor is None:
        return "unresolved (no floor)"
    if _is_scalar_number(diff) and _is_scalar_number(noise_floor):
        f = float(noise_floor)  # type: ignore[arg-type]
        d = float(diff)  # type: ignore[arg-type]
        if math.isnan(f):
            return "unresolved (no floor)"
        if not cell_is_resolved(d, f):
            # NaN diff is not a detection either.
            if math.isnan(d):
                return "within noise"
            return "within noise"
        return _classify_magnitude(d)
    try:
        _resolved, n, resolved_max = resolve_cells(diff, noise_floor)
    except ValueError:
        diff_arr = np.asarray(diff, dtype=np.float64)
        floor_arr = np.asarray(noise_floor, dtype=np.float64)
        try:
            floor_b, diff_b = np.broadcast_arrays(floor_arr, diff_arr)
        except ValueError:
            return "unresolved (no floor)"
        _resolved, n, resolved_max = resolve_cells(diff_b, floor_b)
    if n == 0:
        return "within noise"
    return _classify_magnitude(resolved_max)


def compare_apertures(test: np.ndarray, baseline: np.ndarray) -> dict[str, Any]:
    """Compute per-view-per-hemisphere stats for one ``(test, baseline)`` pair.

    Both inputs are ``[view, hemisphere(2), row, col, freq]`` aperture CFRs.
    Returns a JSON-serializable dict with: ``rel_diff_matrix`` [V, 2] (F6
    semantics: inf when the reference cell is exactly zero and the test
    differs, 0.0 when both are zero), ``per_hemisphere`` (median/max/argmax
    view/n_inf over the finite values; inf entries are counted in ``n_inf``
    and excluded from the median), ``energy_ratio`` (E_test/E_baseline per
    cell; both-zero gives 1.0, zero-baseline otherwise gives inf),
    ``baseline_fraction`` (baseline hemisphere energy share within the view;
    0.0 for an all-zero view), ``aggregate`` (energy-weighted front/back/total
    relative differences) and ``max_diff`` (max over the matrix).
    """
    test_arr = np.asarray(test)
    base_arr = np.asarray(baseline)
    if test_arr.shape != base_arr.shape:
        raise ValueError(f"shape mismatch: test={test_arr.shape}, baseline={base_arr.shape}")
    if test_arr.ndim != 5 or test_arr.shape[1] != 2:
        raise ValueError(
            f"apertures must be [view, hemisphere=2, row, col, freq], got {test_arr.shape}"
        )
    num_views = test_arr.shape[0]
    rel = np.empty((num_views, 2), dtype=np.float64)
    energy_ratio = np.empty((num_views, 2), dtype=np.float64)
    baseline_fraction = np.zeros((num_views, 2), dtype=np.float64)
    for view in range(num_views):
        base_energies = np.array([float(np.sum(np.abs(base_arr[view, h]) ** 2)) for h in range(2)])
        view_energy = float(np.sum(base_energies))
        for hemi in range(2):
            rel[view, hemi] = relative_cfr_difference(test_arr[view, hemi], base_arr[view, hemi])
            test_energy = float(np.sum(np.abs(test_arr[view, hemi]) ** 2))
            base_energy = float(base_energies[hemi])
            if base_energy == 0.0:
                energy_ratio[view, hemi] = 1.0 if test_energy == 0.0 else float("inf")
            else:
                energy_ratio[view, hemi] = test_energy / base_energy
        if view_energy > 0.0:
            baseline_fraction[view] = base_energies / view_energy

    per_hemisphere: dict[str, dict[str, Any]] = {}
    for hemi, name in enumerate(("front", "back")):
        values = rel[:, hemi]
        is_inf = np.isinf(values)
        finite = values[np.isfinite(values)]
        if finite.size:
            median = float(np.median(finite))
            max_val = float(np.max(values))  # inf propagates when present
            if np.isinf(max_val):
                argmax = int(np.argmax(is_inf))
            else:
                argmax = int(np.argmax(values))
        else:
            median = float("inf")
            max_val = float("inf")
            argmax = 0
        per_hemisphere[name] = {
            "median": median,
            "max": max_val,
            "argmax_view": argmax,
            "n_inf": int(np.sum(is_inf)),
            "n_views": int(num_views),
        }
    aggregate = {
        "front": float(relative_cfr_difference(test_arr[:, 0], base_arr[:, 0])),
        "back": float(relative_cfr_difference(test_arr[:, 1], base_arr[:, 1])),
        "total": float(relative_cfr_difference(test_arr, base_arr)),
    }
    return {
        "rel_diff_matrix": rel.tolist(),
        "per_hemisphere": per_hemisphere,
        "energy_ratio": energy_ratio.tolist(),
        "baseline_fraction": baseline_fraction.tolist(),
        "aggregate": aggregate,
        "energy_weighted_aggregate": dict(aggregate),
        "max_diff": float(np.max(rel)),
        "num_views": int(num_views),
    }


def _candidate_matrices(
    test: str,
    baseline: str,
    seed_matrices: dict[tuple[str, str], Any] | None,
    repeat_matrices: dict[str, Any] | None,
) -> list[np.ndarray]:
    """Collect the existing floor source matrices for ``(test, baseline)``."""
    out: list[np.ndarray] = []
    for profile in (test, baseline):
        key = SEED_NOISE_SOURCE.get(profile)
        if key is not None and seed_matrices is not None and key in seed_matrices:
            try:
                out.append(np.asarray(seed_matrices[key], dtype=np.float64))
            except (TypeError, ValueError):
                continue
    if repeat_matrices is not None:
        for profile in (test, baseline):
            if profile in repeat_matrices and repeat_matrices[profile] is not None:
                try:
                    out.append(np.asarray(repeat_matrices[profile], dtype=np.float64))
                except (TypeError, ValueError):
                    continue
    return out


def noise_floor_matrix_for(
    test: str,
    baseline: str,
    seed_matrices: dict[tuple[str, str], Any] | None = None,
    repeat_matrices: dict[str, Any] | None = None,
) -> np.ndarray | None:
    """Return the per-cell floor matrix for ``(test, baseline)`` or None.

    The floor is the element-wise max over the existing source matrices
    (seed-noise matrices of ``SEED_NOISE_SOURCE[test/baseline]`` plus the
    same-seed repeat matrices of test/baseline). NaN is treated as missing;
    inf stays inf. Matrices whose shape differs from the first candidate
    are ignored. Returns None when no source matrix exists.
    """
    candidates = _candidate_matrices(test, baseline, seed_matrices, repeat_matrices)
    if not candidates:
        return None
    shape = candidates[0].shape
    aligned = [c for c in candidates if c.shape == shape]
    if not aligned:
        return None
    stacked = np.stack([np.where(np.isnan(c), -np.inf, c) for c in aligned], axis=0)
    with np.errstate(invalid="ignore"):
        floor = np.max(stacked, axis=0)
    all_missing = np.all(
        np.stack([np.isnan(np.asarray(c, dtype=np.float64)) for c in aligned], axis=0), axis=0
    )
    floor = np.where(all_missing, np.nan, floor)
    return np.asarray(floor, dtype=np.float64)


def noise_floor_for(
    spec: ComparisonSpec,
    seed_matrices: dict[tuple[str, str], Any] | None = None,
    repeat_matrices: dict[str, Any] | None = None,
) -> np.ndarray | None:
    """Return the per-cell floor matrix for ``spec`` (None for noise pairs).

    Noise-floor comparisons are measurements, not effects, so they get no
    floor. Otherwise the floor follows :func:`noise_floor_matrix_for` with
    the ``SEED_NOISE_SOURCE`` mapping. Never falls back to a scene-wide
    repeat max.
    """
    if spec.is_noise:
        return None
    return noise_floor_matrix_for(spec.test, spec.baseline, seed_matrices, repeat_matrices)


def _scalar_floor_display(floor_matrix: np.ndarray | None) -> float | None:
    if floor_matrix is None:
        return None
    arr = np.asarray(floor_matrix, dtype=np.float64)
    if arr.size == 0:
        return None
    with np.errstate(invalid="ignore"):
        try:
            val = float(np.nanmax(arr))
        except ValueError:
            return None
    return val


def attach_noise_floor(
    stats: dict[str, Any], noise_floor: Any | None, *, is_noise: bool = False
) -> dict[str, Any]:
    """Return a copy of ``stats`` with per-cell floor/resolution attached.

    Stores ``noise_floor_matrix``, ``resolved_matrix`` (bool), ``resolved``
    (any cell; None for noise comparisons), ``n_resolved_cells``,
    ``resolved_max`` (max diff over resolved cells, 0.0 when none) and
    ``noise_floor`` (scalar max of the floor matrix, display only).
    ``effect`` is ``"noise floor"`` for noise comparisons, ``"unresolved
    (no floor)"`` when the floor is None, ``"within noise"`` when no cell
    resolves, else small/moderate/large from ``resolved_max``. ``noise_floor``
    may be a ``[V, 2]`` matrix, a scalar (treated as a uniform matrix), or
    None.
    """
    out = dict(stats)
    if is_noise:
        rel = np.asarray(stats.get("rel_diff_matrix", []), dtype=np.float64)
        out["noise_floor"] = None
        out["noise_floor_matrix"] = None
        out["resolved_matrix"] = np.zeros_like(rel, dtype=bool).tolist()
        out["resolved"] = None
        out["n_resolved_cells"] = 0
        out["resolved_max"] = 0.0
        out["effect"] = "noise floor"
        return out
    diff = np.asarray(stats.get("rel_diff_matrix", []), dtype=np.float64)
    floor_matrix: np.ndarray | None
    if noise_floor is None:
        floor_matrix = None
    else:
        arr = np.asarray(noise_floor, dtype=np.float64)
        if arr.ndim == 0:
            floor_matrix = np.full_like(diff, float(arr))
        else:
            floor_matrix = arr
            if arr.shape != diff.shape:
                raise ValueError(f"shape mismatch: diff={diff.shape}, floor={arr.shape}")
    if floor_matrix is None:
        out["noise_floor"] = None
        out["noise_floor_matrix"] = None
        out["resolved_matrix"] = np.zeros_like(diff, dtype=bool).tolist()
        out["resolved"] = False
        out["n_resolved_cells"] = 0
        out["resolved_max"] = 0.0
        out["effect"] = "unresolved (no floor)"
        return out
    resolved, n_resolved, resolved_max = resolve_cells(diff, floor_matrix)
    out["noise_floor_matrix"] = floor_matrix.tolist()
    out["resolved_matrix"] = resolved.tolist()
    out["n_resolved_cells"] = int(n_resolved)
    out["resolved_max"] = float(resolved_max)
    out["noise_floor"] = _scalar_floor_display(floor_matrix)
    out["resolved"] = bool(n_resolved > 0)
    out["effect"] = "within noise" if n_resolved == 0 else _classify_magnitude(resolved_max)
    return out


def _largest_resolved_cell(stats: dict[str, Any]) -> tuple[int, str, float, float, float] | None:
    """Return ``(view, hemi_name, diff, floor, ratio)`` of the top resolved cell."""
    if int(stats.get("n_resolved_cells", 0)) <= 0:
        return None
    diff = np.asarray(stats["rel_diff_matrix"], dtype=np.float64)
    resolved = np.asarray(stats["resolved_matrix"], dtype=bool)
    floor = np.asarray(stats["noise_floor_matrix"], dtype=np.float64)
    ratio = np.asarray(stats["energy_ratio"], dtype=np.float64)
    masked = np.where(resolved, np.where(np.isinf(diff), np.inf, diff), -np.inf)
    flat = int(np.argmax(masked))
    view, hemi = int(flat // 2), int(flat % 2)
    hemi_name = ("front", "back")[hemi]
    return (
        view,
        hemi_name,
        float(diff[view, hemi]),
        float(floor[view, hemi]),
        float(ratio[view, hemi]),
    )


def describe_comparison(spec: ComparisonSpec, stats: dict[str, Any]) -> str:
    """Render one data-derived sentence for ``(spec, stats)``.

    Only numbers present in ``stats`` are formatted; no external claims are
    added. ``stats`` must carry the :func:`attach_noise_floor` keys. Noise
    comparisons report medians/maxima plus the same-seed repeat max and
    never mention resolution verdicts.
    """
    front = stats["per_hemisphere"]["front"]
    back = stats["per_hemisphere"]["back"]
    if spec.is_noise or stats.get("effect") == "noise floor":
        repeat_max = stats.get("repeat_max")
        if repeat_max is None:
            repeat_str = "n/a"
        else:
            try:
                repeat_str = _fmt(float(repeat_max))
            except (TypeError, ValueError):
                repeat_str = "n/a"
        return (
            f"{spec.question} ({spec.test} vs {spec.baseline}): "
            f"front median {_fmt(float(front['median']))} / max {_fmt(float(front['max']))} "
            f"(view {int(front['argmax_view'])}), back median {_fmt(float(back['median']))} "
            f"/ max {_fmt(float(back['max']))} (view {int(back['argmax_view'])}); "
            f"same-seed repeat max {repeat_str}."
        )
    effect = str(stats.get("effect"))
    if stats.get("resolved"):
        verdict = f"resolved ({effect})"
    else:
        verdict = effect
    total_cells = int(stats.get("num_views", 0)) * 2
    n_resolved = int(stats.get("n_resolved_cells", 0))
    resolved_max = float(stats.get("resolved_max", 0.0))
    if stats.get("noise_floor") is None:
        floor_str = "n/a"
    else:
        floor_str = _fmt(float(stats["noise_floor"]))  # type: ignore[arg-type]
    base = (
        f"{spec.question} ({spec.test} vs {spec.baseline}): "
        f"front median {_fmt(float(front['median']))} / max {_fmt(float(front['max']))} "
        f"(view {int(front['argmax_view'])}), back median {_fmt(float(back['median']))} "
        f"/ max {_fmt(float(back['max']))} (view {int(back['argmax_view'])}); "
        f"max floor cell {floor_str} "
        f"-> {verdict}; {n_resolved}/{total_cells} cells resolved "
        f"(resolved max {_fmt(resolved_max)})"
    )
    top = _largest_resolved_cell(stats)
    if top is not None:
        view, hemi_name, diff_v, floor_v, ratio = top
        base += (
            f"; largest resolved cell view {view} {hemi_name} "
            f"(diff {_fmt(diff_v)}, floor {_fmt(floor_v)}, "
            f"energy ratio {_fmt(ratio)})."
        )
    else:
        base += "."
    return base


def _rank_key(stats: dict[str, Any]) -> tuple[float, int]:
    return (float(stats.get("resolved_max", 0.0)), int(stats.get("n_resolved_cells", 0)))


def interpretation_for_scene(
    scene_key: str, entries: list[tuple[ComparisonSpec, dict[str, Any]]]
) -> str:
    """Render one data-derived interpretation paragraph for ``scene_key``.

    Ranks the non-noise comparisons by ``(resolved_max, n_resolved_cells)``
    and states the ranking; any mechanism-vs-mechanism sentence (depth
    truncation vs diffraction) uses the same key. Only factual caveats are
    appended (GPU non-reproducibility, ITU S=0 default, s03 material change).
    """
    if not entries:
        return f"Scene {scene_key}: no comparisons available."
    ranked = sorted(
        [(s, st) for s, st in entries if not s.is_noise],
        key=lambda pair: _rank_key(pair[1]),
        reverse=True,
    )
    rank_clause = "; ".join(
        f"{s.test} vs {s.baseline} (resolved_max={_fmt(float(st.get('resolved_max', 0.0)))}, "
        f"n={int(st.get('n_resolved_cells', 0))}, {st.get('effect')})"
        for s, st in ranked
    )
    parts = [f"Scene {scene_key}: ranked by (resolved_max, n_resolved_cells): {rank_clause}."]
    by_key = {(s.test, s.baseline): (s, st) for s, st in entries}
    depth = by_key.get(("current_depth1", "current"))
    diffr = by_key.get(("plus_diffraction", "current"))
    if depth is not None and diffr is not None:
        depth_key = _rank_key(depth[1])
        diffr_key = _rank_key(diffr[1])
        relation = "larger than" if depth_key > diffr_key else "smaller than or equal to"
        parts.append(
            f"Depth truncation 1->5 (resolved_max={_fmt(depth_key[0])}, n={depth_key[1]}) "
            f"is {relation} diffraction "
            f"(resolved_max={_fmt(diffr_key[0])}, n={diffr_key[1]})."
        )
    for spec, stats in entries:
        parts.append(describe_comparison(spec, stats))
    parts.append(
        "Caveats: " + " ".join(CAVEATS) + " All numbers above are computed "
        "from this run's aperture CFRs."
    )
    return " ".join(parts)


def scene_markdown(scene_key: str, entries: list[tuple[ComparisonSpec, dict[str, Any]]]) -> str:
    """Render a markdown section for ``scene_key`` from computed entries only."""
    lines = ["### Step-wise comparisons", ""]
    lines.append("| test vs baseline | question | front median/max | back median/max |")
    lines.append("| --- | --- | --- | --- |")
    for spec, stats in entries:
        front = stats["per_hemisphere"]["front"]
        back = stats["per_hemisphere"]["back"]
        lines.append(
            f"| {spec.test} vs {spec.baseline} | {spec.question} "
            f"| {_fmt(float(front['median']))} / {_fmt(float(front['max']))} "
            f"| {_fmt(float(back['median']))} / {_fmt(float(back['max']))} |"
        )
    lines += ["", interpretation_for_scene(scene_key, entries), ""]
    return "\n".join(lines)
