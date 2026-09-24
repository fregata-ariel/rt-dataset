"""Unit tests for plateau_rt.application.solver_profile_report (NumPy only)."""

import json

import numpy as np

from plateau_rt.application.solver_profile_report import (
    SEED_NOISE_SOURCE,
    attach_noise_floor,
    classify_effect,
    compare_apertures,
    comparison_specs,
    describe_comparison,
    family_of,
    interpretation_for_scene,
    is_resolved,
    json_safe,
    noise_floor_for,
    scene_markdown,
)


def _spec_by_key(test: str, baseline: str):
    for spec in comparison_specs():
        if spec.test == test and spec.baseline == baseline:
            return spec
    raise AssertionError(f"missing spec {test} vs {baseline}")


def _fake_stats(
    front_max: float,
    back_max: float,
    noise_floor: float,
    front_median: float = 0.0,
    back_median: float = 0.0,
) -> dict:
    stats = {
        "rel_diff_matrix": [[0.0, 0.0], [front_max, back_max]],
        "per_hemisphere": {
            "front": {
                "median": front_median,
                "max": front_max,
                "argmax_view": 1,
                "n_inf": 0,
                "n_views": 2,
            },
            "back": {
                "median": back_median,
                "max": back_max,
                "argmax_view": 1,
                "n_inf": 0,
                "n_views": 2,
            },
        },
        "energy_ratio": [[1.0, 1.0], [1.0, 1.0]],
        "baseline_fraction": [[0.5, 0.5], [0.5, 0.5]],
        "aggregate": {"front": front_max, "back": back_max, "total": front_max},
        "max_diff": float(max(front_max, back_max)),
        "num_views": 2,
    }
    return attach_noise_floor(stats, noise_floor)


def _matrix_stats(diff: list[list[float]]) -> dict:
    arr = np.asarray(diff, dtype=np.float64)
    num_views = arr.shape[0]
    front_vals = arr[:, 0]
    back_vals = arr[:, 1]
    return {
        "rel_diff_matrix": arr.tolist(),
        "per_hemisphere": {
            "front": {
                "median": float(np.median(front_vals[np.isfinite(front_vals)]))
                if np.any(np.isfinite(front_vals))
                else float("inf"),
                "max": float(np.max(front_vals)),
                "argmax_view": int(np.argmax(front_vals)),
                "n_inf": int(np.sum(np.isinf(front_vals))),
                "n_views": int(num_views),
            },
            "back": {
                "median": float(np.median(back_vals[np.isfinite(back_vals)]))
                if np.any(np.isfinite(back_vals))
                else float("inf"),
                "max": float(np.max(back_vals)),
                "argmax_view": int(np.argmax(back_vals)),
                "n_inf": int(np.sum(np.isinf(back_vals))),
                "n_views": int(num_views),
            },
        },
        "energy_ratio": np.ones_like(arr).tolist(),
        "baseline_fraction": (np.full_like(arr, 0.5)).tolist(),
        "aggregate": {"front": 0.0, "back": 0.0, "total": 0.0},
        "max_diff": float(np.max(arr)),
        "num_views": int(num_views),
    }


def test_ranking_follows_computed_max():
    diffr = _spec_by_key("plus_diffraction", "current")
    depth = _spec_by_key("current_depth1", "current")
    entries = [
        (diffr, _fake_stats(0.9, 0.8, 0.01)),
        (depth, _fake_stats(0.1, 0.05, 0.01)),
    ]
    text = interpretation_for_scene("mock", entries)
    assert text.index("plus_diffraction") < text.index("current_depth1")
    swapped = [
        (diffr, _fake_stats(0.1, 0.05, 0.01)),
        (depth, _fake_stats(0.9, 0.8, 0.01)),
    ]
    flipped = interpretation_for_scene("mock", swapped)
    assert flipped.index("current_depth1") < flipped.index("plus_diffraction")


def test_resolved_labels_follow_noise_floor():
    spec = _spec_by_key("plus_diffraction", "current")
    below = _fake_stats(0.1, 0.05, 0.1)
    assert below["resolved"] is False
    assert below["effect"] == "within noise"
    assert "within noise" in describe_comparison(spec, below)
    assert is_resolved(0.1, 0.1) is False
    above = _fake_stats(0.5, 0.4, 0.1)
    assert above["resolved"] is True
    assert above["effect"] != "within noise"
    assert "resolved" in describe_comparison(spec, above)
    assert is_resolved(0.5, 0.1) is True
    assert classify_effect(0.03, 0.0) == "small"
    assert classify_effect(0.2, 0.0) == "moderate"
    assert classify_effect(0.9, 0.0) == "large"


def test_explicit_array_baseline_and_family_purity():
    specs = comparison_specs()
    keys = [(s.test, s.baseline) for s in specs]
    assert ("current_explicit_array", "current") in keys
    cross = [(s.test, s.baseline) for s in specs if family_of(s.test) != family_of(s.baseline)]
    assert cross == [("current_s03", "current")]


def test_per_view_hemisphere_stats_hand_built():
    baseline = np.ones((3, 2, 1, 1, 2), dtype=np.complex128)
    baseline[1, 1] = 0.0
    baseline[2, 1] = 0.0
    test = baseline.copy()
    test[1, 0] = 2.0  # front rel. diff 1.0
    test[1, 1] = 5.0  # zero reference, nonzero test -> inf
    stats = compare_apertures(test, baseline)
    rel = np.asarray(stats["rel_diff_matrix"])
    assert rel.shape == (3, 2)
    assert rel[0, 0] == 0.0 and rel[0, 1] == 0.0
    assert rel[1, 0] == float(abs(2.0 - 1.0))
    assert np.isinf(rel[1, 1])
    assert rel[2, 0] == 0.0 and rel[2, 1] == 0.0
    front = stats["per_hemisphere"]["front"]
    assert front["max"] == max(float(v) for v in rel[:, 0])
    assert front["argmax_view"] == 1
    assert front["n_inf"] == 0
    assert front["median"] == float(np.median([rel[0, 0], rel[1, 0], rel[2, 0]]))
    back = stats["per_hemisphere"]["back"]
    assert back["n_inf"] == 1
    assert np.isinf(back["max"])
    assert back["argmax_view"] == 1
    assert back["median"] == 0.0  # inf excluded from the median
    energy_ratio = np.asarray(stats["energy_ratio"])
    assert energy_ratio[0, 0] == 1.0
    assert np.isinf(energy_ratio[1, 1])
    assert energy_ratio[2, 1] == 1.0  # both zero -> 1.0
    fraction = np.asarray(stats["baseline_fraction"])
    assert fraction[1, 0] == 1.0 and fraction[1, 1] == 0.0


def test_markdown_only_reports_computed_numbers():
    diffr = _spec_by_key("plus_diffraction", "current")
    depth = _spec_by_key("current_depth1", "current")
    stats = _fake_stats(0.456, 0.321, 0.01)
    other = _fake_stats(0.111, 0.099, 0.01)
    entries = [(diffr, stats), (depth, other)]
    md = scene_markdown("mock", entries)
    assert f"{0.456:.3g}" in md
    assert "much closer" not in md
    assert "only modestly" not in md
    assert "plus_diffraction vs current" in md


def test_per_cell_resolution_max_vs_max_is_wrong():
    floor = [[0.1, 0.0], [0.0, 0.9]]
    # Cell (0, front): 1.0 > 2*0.1 -> resolved, even though the scalar
    # max-vs-max rule says 1.0 <= 2*0.9 (within noise).
    stats = attach_noise_floor(_matrix_stats([[1.0, 0.0], [0.0, 0.0]]), floor)
    assert stats["resolved"] is True
    assert stats["effect"] != "within noise"
    assert stats["n_resolved_cells"] == 1
    assert stats["resolved_max"] == 1.0
    assert is_resolved([[1.0, 0.0], [0.0, 0.0]], floor) is True
    # Cell (1, back): 1.0 <= 2*0.9 -> within noise.
    quiet = attach_noise_floor(_matrix_stats([[0.0, 0.0], [0.0, 1.0]]), floor)
    assert quiet["resolved"] is False
    assert quiet["effect"] == "within noise"
    assert quiet["n_resolved_cells"] == 0
    assert quiet["resolved_max"] == 0.0
    assert is_resolved([[0.0, 0.0], [0.0, 1.0]], floor) is False


def test_per_cell_resolution_inf_semantics():
    # inf diff over a finite floor resolves.
    stats = attach_noise_floor(
        _matrix_stats([[float("inf"), 0.0], [0.0, 0.0]]),
        [[0.5, 0.0], [0.0, 0.0]],
    )
    assert stats["resolved"] is True
    assert stats["n_resolved_cells"] == 1
    assert np.isinf(stats["resolved_max"])
    assert stats["effect"] == "large"
    # An inf floor cell never resolves, even for an inf diff.
    quiet = attach_noise_floor(
        _matrix_stats([[float("inf"), 0.0], [0.0, 0.0]]),
        [[float("inf"), 0.0], [0.0, 0.0]],
    )
    assert quiet["resolved"] is False
    assert quiet["effect"] == "within noise"
    assert quiet["n_resolved_cells"] == 0


def test_floor_source_uses_seed_mapping_not_scene_wide_max():
    seed_matrices = {
        ("complete_seed2", "complete"): np.full((2, 2), 0.01),
        ("complete_s03_seed2", "complete_s03"): np.full((2, 2), 5.0),
        ("plus_diffuse_s03_seed2", "plus_diffuse_s03"): np.full((2, 2), 7.0),
    }
    assert SEED_NOISE_SOURCE["current_s03"] == ("complete_seed2", "complete")
    # current_s03 vs current must use the complete seed matrix, not s03.
    floor = noise_floor_for(_spec_by_key("current_s03", "current"), seed_matrices, {})
    assert floor is not None
    assert np.allclose(np.asarray(floor), 0.01)
    # plus_diffuse_s03 vs current_s03 includes the plus_diffuse seed matrix.
    floor2 = noise_floor_for(_spec_by_key("plus_diffuse_s03", "current_s03"), seed_matrices, {})
    assert floor2 is not None
    assert np.allclose(np.asarray(floor2), 7.0)
    # An unrelated huge repeat matrix must not leak into another floor.
    repeat = {
        "plus_diffuse_s03_seed2": np.full((2, 2), 100.0),
        "specular": np.full((2, 2), 0.02),
        "los_only": np.full((2, 2), 0.03),
    }
    floor3 = noise_floor_for(_spec_by_key("specular", "los_only"), seed_matrices, repeat)
    assert floor3 is not None
    assert np.allclose(np.asarray(floor3), 0.03)


def test_noise_comparison_is_measurement_not_effect():
    spec = _spec_by_key("complete_seed2", "complete")
    assert spec.is_noise
    stats = attach_noise_floor(_matrix_stats([[0.5, 0.1], [0.2, 0.3]]), None, is_noise=True)
    assert stats["effect"] == "noise floor"
    assert stats["resolved"] is None
    sentence = describe_comparison(spec, stats)
    assert "resolved" not in sentence
    assert "within noise" not in sentence


def test_ranking_breaks_inf_ties_by_cell_count():
    diffr = _spec_by_key("plus_diffraction", "current")
    depth = _spec_by_key("current_depth1", "current")
    few = attach_noise_floor(
        _matrix_stats([[float("inf"), 0.0], [0.0, 0.0]]),
        [[0.1, 0.0], [0.0, 0.0]],
    )
    many = attach_noise_floor(
        _matrix_stats([[float("inf"), float("inf")], [0.0, 0.0]]),
        [[0.1, 0.1], [0.0, 0.0]],
    )
    assert np.isinf(few["resolved_max"]) and np.isinf(many["resolved_max"])
    assert many["n_resolved_cells"] == 2
    assert few["n_resolved_cells"] == 1
    # List the fewer-cell entry first; the ranking must still put `many` first.
    text = interpretation_for_scene("mock", [(diffr, few), (depth, many)])
    assert text.index("current_depth1") < text.index("plus_diffraction")


def test_json_safe_converts_non_finite():
    payload = json_safe({"a": float("inf"), "b": [np.float64("nan")], "c": np.array([1.0, np.inf])})
    text = json.dumps(payload, allow_nan=False)
    assert '"inf"' in text
    assert '"nan"' in text
    assert payload == {"a": "inf", "b": ["nan"], "c": [1.0, "inf"]}
