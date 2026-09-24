"""Tests for partial/summary RF-camera observations (CPU only, no Sionna)."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner
from rf_manifest_fixtures import write_v2_dataset, write_v3_dataset

from plateau_rt.application.rf_camera_partial import build_partial_dataset
from plateau_rt.application.rf_dataset_manifest import load_rf_dataset_manifest
from plateau_rt.cli.main import cli
from plateau_rt.domain.rf_camera.delay import circular_delay_error_s
from plateau_rt.domain.rf_camera.imaging import frequency_offsets, uniform_frequency_spacing
from plateau_rt.domain.rf_camera.partial import (
    apply_element_mask,
    element_mask,
    element_power,
    hemisphere_total_power,
    parse_subband,
    select_subband,
    select_views,
    view_dominant_delay,
)

BANDWIDTH_HZ = 80e6
NUM_BINS = 8
ROWS = COLS = 4
NUM_VIEWS = 4
NUM_BS = 2
DELAY_RESOLUTION_S = 1.0 / BANDWIDTH_HZ
TAU_BIN = 3
TAU_S = TAU_BIN * DELAY_RESOLUTION_S
BACK_CFR = 0.5 + 0.25j


def _domain_offsets() -> np.ndarray:
    return frequency_offsets(BANDWIDTH_HZ, NUM_BINS).astype(np.float64)


def _plane_wave(
    rows: int,
    cols: int,
    offsets: np.ndarray,
    delay_s: float,
    amplitude: complex = 1.0,
    sign: np.ndarray | None = None,
) -> np.ndarray:
    """Complex plane wave ``[rows, cols, freq]`` with delay ``delay_s``."""
    base = complex(amplitude) * np.exp(-2j * np.pi * np.asarray(offsets) * delay_s)
    cfr = np.broadcast_to(base, (rows, cols, len(np.asarray(offsets)))).copy()
    if sign is not None:
        cfr = cfr * np.asarray(sign)[:, :, None]
    return cfr.astype(np.complex128)


def _synthetic_5d() -> tuple[np.ndarray, np.ndarray]:
    """One-BS 5-D CFR: front plane wave on an exact bin, constant back CFR."""
    offsets = _domain_offsets()
    front = _plane_wave(ROWS, COLS, offsets, TAU_S)
    back = np.full((ROWS, COLS, NUM_BINS), BACK_CFR, dtype=np.complex128)
    return np.stack([np.stack([front, back])]).astype(np.complex64), offsets


def _fixture_offsets(root: Path) -> np.ndarray:
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    return np.asarray(manifest["frequency_offsets_hz"], dtype=np.float64)


def _overwrite_view_cfrs(root: Path, manifest: dict, front_per_bs: list[np.ndarray]) -> None:
    """Replace every view's aperture CFR with ``[bs, hemi=front, ...]`` built from parts."""
    for view in manifest["views"]:
        view_id = view["view_id"]
        path = root / "views" / view_id / "rf" / "aperture_cfr.npy"
        current = np.load(path)
        num_bs, _, rows, cols, bins = current.shape
        assert num_bs == len(front_per_bs)
        rebuilt = np.zeros_like(current)
        for bs_index, front in enumerate(front_per_bs):
            assert front.shape == (rows, cols, bins)
            rebuilt[bs_index, 0] = front
        np.save(path, rebuilt.astype(np.complex64))


def _snapshot_tree(root: Path) -> tuple[bytes, dict[str, bytes]]:
    manifest = (root / "dataset_manifest.json").read_bytes()
    payloads = {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*.npy"))}
    return manifest, payloads


def _assert_tree_unchanged(root: Path, before: tuple[bytes, dict[str, bytes]]) -> None:
    manifest, payloads = before
    assert (root / "dataset_manifest.json").read_bytes() == manifest
    now = {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*.npy"))}
    assert set(now) == set(payloads)
    for key, value in payloads.items():
        assert now[key] == value


# ---------------------------------------------------------------------------
# Domain: view selection / masks
# ---------------------------------------------------------------------------


def test_select_views_deterministic_sorted_unique_count() -> None:
    first = select_views(NUM_VIEWS, 0.5, seed=7)
    second = select_views(NUM_VIEWS, 0.5, seed=7)
    np.testing.assert_array_equal(first, second)
    assert first.tolist() == sorted(first.tolist())
    assert len(set(first.tolist())) == len(first)
    assert len(first) == 2
    np.testing.assert_array_equal(select_views(NUM_VIEWS, 1.0, seed=0), [0, 1, 2, 3])
    assert len(select_views(8, 0.01, seed=0)) == 1  # at least one view
    assert len(select_views(1, 0.5, seed=0)) == 1
    variants = {tuple(select_views(NUM_VIEWS, 0.5, seed=s).tolist()) for s in range(10)}
    assert len(variants) > 1  # different seeds usually differ
    with pytest.raises(ValueError):
        select_views(NUM_VIEWS, 0.0, seed=0)
    with pytest.raises(ValueError):
        select_views(NUM_VIEWS, 1.5, seed=0)
    with pytest.raises(ValueError):
        select_views(0, 0.5, seed=0)


def test_select_views_rounds_half_up() -> None:
    # 0.5 * 5 = 2.5 rounds half up to 3 (banker's round() would give 2).
    assert len(select_views(5, 0.5, 0)) == 3
    assert len(select_views(5, 0.5, 1)) == 3


def test_select_views_stream_independent_from_mask() -> None:
    for seed in range(10):
        views = select_views(16, 0.5, seed)
        kept_flat = np.flatnonzero(element_mask(4, 4, "random", fraction=0.5, seed=seed))
        assert views.tolist() != kept_flat.tolist()


def test_element_mask_exact_patterns() -> None:
    rows, cols = 4, 5
    np.testing.assert_array_equal(
        element_mask(rows, cols, "none"), np.ones((rows, cols), dtype=bool)
    )
    expected_rows = np.zeros((rows, cols), dtype=bool)
    expected_rows[0::2, :] = True
    np.testing.assert_array_equal(element_mask(rows, cols, "every_other_row"), expected_rows)
    expected_cols = np.zeros((rows, cols), dtype=bool)
    expected_cols[:, 0::2] = True
    np.testing.assert_array_equal(element_mask(rows, cols, "every_other_col"), expected_cols)
    rr, cc = np.indices((rows, cols))
    np.testing.assert_array_equal(element_mask(rows, cols, "checkerboard"), (rr + cc) % 2 == 0)

    total = rows * cols
    mask = element_mask(rows, cols, "random", fraction=0.5, seed=3)
    assert mask.dtype == bool
    assert int(mask.sum()) == 10  # round-half-up(0.5 * 20)
    assert total == 20
    np.testing.assert_array_equal(mask, element_mask(rows, cols, "random", fraction=0.5, seed=3))
    assert int(element_mask(rows, cols, "random", fraction=0.01, seed=0).sum()) == 1
    with pytest.raises(ValueError):
        element_mask(rows, cols, "bogus")
    with pytest.raises(ValueError):
        element_mask(rows, cols, "random", fraction=0.0)
    with pytest.raises(ValueError):
        element_mask(rows, cols, "random", fraction=2.0)
    with pytest.raises(ValueError):
        element_mask(0, cols, "none")


def test_element_mask_random_rounds_half_up() -> None:
    # 0.5 * 5 = 2.5 rounds half up to 3 (banker's round() would give 2).
    assert int(element_mask(1, 5, "random", fraction=0.5, seed=0).sum()) == 3


# ---------------------------------------------------------------------------
# Domain: 5-D masking / subband / power
# ---------------------------------------------------------------------------


def test_apply_element_mask_5d_broadcasts_over_bs_hemi_freq() -> None:
    aperture, _ = _synthetic_5d()
    assert aperture.shape == (1, 2, ROWS, COLS, NUM_BINS)
    mask = element_mask(ROWS, COLS, "checkerboard")
    out = apply_element_mask(aperture, mask)
    assert out.shape == aperture.shape
    assert out.dtype == aperture.dtype
    np.testing.assert_array_equal(out[:, :, mask, :], aperture[:, :, mask, :])
    np.testing.assert_array_equal(out[:, :, ~mask, :], np.zeros_like(out[:, :, ~mask, :]))
    # The two BS slices of a duplicated input stay identical after masking.
    doubled = np.concatenate([aperture, aperture], axis=0)
    doubled_out = apply_element_mask(doubled, mask)
    np.testing.assert_array_equal(doubled_out[0], doubled_out[1])
    with pytest.raises(ValueError):
        apply_element_mask(aperture, np.ones((ROWS + 1, COLS), dtype=bool))
    with pytest.raises(ValueError):
        apply_element_mask(aperture[0], mask)


def test_domain_functions_reject_4d_arrays() -> None:
    aperture, offsets = _synthetic_5d()
    four_d = aperture[0]
    assert four_d.ndim == 4
    mask = element_mask(ROWS, COLS, "checkerboard")
    with pytest.raises(ValueError):
        apply_element_mask(four_d, mask)
    with pytest.raises(ValueError):
        select_subband(four_d, offsets, 0, 4)
    with pytest.raises(ValueError):
        element_power(four_d)
    with pytest.raises(ValueError):
        hemisphere_total_power(four_d)


def test_parse_and_select_subband_5d() -> None:
    assert parse_subband(None, NUM_BINS) == (0, NUM_BINS)
    assert parse_subband("2:6", NUM_BINS) == (2, 6)
    assert parse_subband("0:8", NUM_BINS) == (0, NUM_BINS)
    for bad in ["", "2", "6:2", "2:2", "-1:3", "0:9", "a:b", "1:2:3", "3:2"]:
        with pytest.raises(ValueError):
            parse_subband(bad, NUM_BINS)

    aperture, offsets = _synthetic_5d()
    sub_cfr, sub_offsets = select_subband(aperture, offsets, 2, 6)
    assert sub_cfr.shape == (1, 2, ROWS, COLS, 4)
    np.testing.assert_array_equal(sub_cfr, aperture[..., 2:6])
    np.testing.assert_array_equal(sub_offsets, offsets[2:6])
    with pytest.raises(ValueError):
        select_subband(aperture, offsets, 6, 2)
    with pytest.raises(ValueError):
        select_subband(aperture, offsets, 0, NUM_BINS + 1)
    with pytest.raises(ValueError):
        select_subband(aperture, offsets[:4], 0, 4)


def test_element_and_hemisphere_power_hand_constants() -> None:
    # Hand-computed literals (not the fixture formula): all-ones CFR.
    cfr = np.ones((1, 2, 2, 3, 4), dtype=np.complex64)
    power = element_power(cfr)
    assert power.dtype == np.float32
    assert power.shape == (1, 2, 2, 3)
    np.testing.assert_array_equal(power, np.full((1, 2, 2, 3), 4.0, dtype=np.float32))
    total = hemisphere_total_power(cfr)
    assert total.dtype == np.float64
    assert total.shape == (1, 2)
    np.testing.assert_array_equal(total, np.full((1, 2), 24.0, dtype=np.float64))

    # Complex constant 0.5+0.25j: |c|^2 = 0.3125 per bin.
    cfr2 = np.full((2, 2, 1, 2, 8), 0.5 + 0.25j, dtype=np.complex64)
    np.testing.assert_allclose(element_power(cfr2), np.full((2, 2, 1, 2), 2.5), rtol=1e-6)
    np.testing.assert_allclose(hemisphere_total_power(cfr2), np.full((2, 2), 5.0), rtol=1e-6)


# ---------------------------------------------------------------------------
# Domain: dominant delay
# ---------------------------------------------------------------------------


def test_view_dominant_delay_recovers_tau_exact() -> None:
    aperture, offsets = _synthetic_5d()
    result = view_dominant_delay(aperture[0, 0], offsets)
    assert set(result) == {
        "delay_s",
        "power",
        "delay_resolution_s",
        "unambiguous_delay_s",
        "valid",
    }
    assert result["valid"] is True
    assert result["delay_s"] == pytest.approx(TAU_S, abs=1e-12)
    assert result["delay_resolution_s"] == pytest.approx(DELAY_RESOLUTION_S, rel=1e-9)
    assert result["unambiguous_delay_s"] == pytest.approx(NUM_BINS / BANDWIDTH_HZ, rel=1e-9)
    assert result["power"] > 0.0

    masked = view_dominant_delay(aperture[0, 0], offsets, element_mask(ROWS, COLS, "none"))
    assert masked["delay_s"] == pytest.approx(TAU_S, abs=1e-12)

    masked_result = view_dominant_delay(
        aperture[0, 0], offsets, element_mask(ROWS, COLS, "checkerboard")
    )
    assert masked_result["delay_s"] == pytest.approx(TAU_S, abs=1e-12)


def test_view_dominant_delay_incoherent_not_coherent() -> None:
    """A checkerboard-signed path cancels coherently but wins incoherently."""
    offsets = _domain_offsets()
    resolution = 1.0 / BANDWIDTH_HZ
    tau_a, tau_b = 3 * resolution, 5 * resolution
    rr, cc = np.indices((ROWS, COLS))
    sign_a = ((-1) ** (rr + cc)).astype(np.float64)  # sums to exactly zero
    front = _plane_wave(ROWS, COLS, offsets, tau_a, 1.0, sign_a) + _plane_wave(
        ROWS, COLS, offsets, tau_b, 0.5
    )
    result = view_dominant_delay(front, offsets)
    assert result["valid"] is True
    # Incoherent: 16 * 1.0 at bin 3 beats 16 * 0.25 at bin 5. Coherent would give bin 5.
    assert result["delay_s"] == pytest.approx(tau_a, abs=1e-12)


def test_view_dominant_delay_respects_element_mask() -> None:
    offsets = _domain_offsets()
    resolution = 1.0 / BANDWIDTH_HZ
    tau_a, tau_b = 2 * resolution, 6 * resolution
    mask = element_mask(ROWS, COLS, "checkerboard")
    front_a_only_dropped = _plane_wave(ROWS, COLS, offsets, tau_a, 1.0)
    front_a_only_dropped[mask] = 0.0
    front_b_only_kept = _plane_wave(ROWS, COLS, offsets, tau_b, 0.3)
    front_b_only_kept[~mask] = 0.0
    front = front_a_only_dropped + front_b_only_kept
    with_mask = view_dominant_delay(front, offsets, mask)
    assert with_mask["delay_s"] == pytest.approx(tau_b, abs=1e-12)
    without_mask = view_dominant_delay(front, offsets)
    assert without_mask["delay_s"] == pytest.approx(tau_a, abs=1e-12)


@pytest.mark.parametrize("start,stop", [(0, 4), (4, 8)])
def test_view_dominant_delay_offcentre_subbands_exact(start: int, stop: int) -> None:
    offsets = _domain_offsets()
    delta_f = uniform_frequency_spacing(offsets)
    n_sub = stop - start
    resolution = 1.0 / (n_sub * delta_f)
    tau = 2 * resolution  # exactly on the subband delay grid
    front = _plane_wave(ROWS, COLS, offsets, tau)
    sub = front[:, :, start:stop]
    sub_offsets = offsets[start:stop]
    result = view_dominant_delay(sub, sub_offsets)
    assert result["delay_s"] == pytest.approx(tau, abs=1e-12)
    assert result["delay_resolution_s"] == pytest.approx(resolution, rel=1e-9)
    assert result["unambiguous_delay_s"] == pytest.approx(1.0 / delta_f, rel=1e-9)


def test_view_dominant_delay_odd_length_subband_exact() -> None:
    offsets = _domain_offsets()
    delta_f = uniform_frequency_spacing(offsets)
    n_sub = 5
    resolution = 1.0 / (n_sub * delta_f)
    tau = 3 * resolution
    front = _plane_wave(ROWS, COLS, offsets, tau)
    result = view_dominant_delay(front[:, :, 1:6], offsets[1:6])
    assert result["delay_s"] == pytest.approx(tau, abs=1e-12)
    assert result["delay_resolution_s"] == pytest.approx(resolution, rel=1e-9)
    assert result["unambiguous_delay_s"] == pytest.approx(1.0 / delta_f, rel=1e-9)


def test_view_dominant_delay_offgrid_error_within_half_bin() -> None:
    offsets = _domain_offsets()
    delta_f = uniform_frequency_spacing(offsets)
    n_sub = 4
    resolution = 1.0 / (n_sub * delta_f)
    period = 1.0 / delta_f
    tau = 2 * resolution + 0.3 * resolution  # 0.3 bins off grid
    front = _plane_wave(ROWS, COLS, offsets, tau)
    result = view_dominant_delay(front[:, :, 0:4], offsets[0:4])
    error = circular_delay_error_s(result["delay_s"], tau % period, period)
    assert error <= 0.5 * resolution + 1e-12


def test_view_dominant_delay_zero_power_is_invalid() -> None:
    offsets = _domain_offsets()
    front = np.zeros((ROWS, COLS, NUM_BINS), dtype=np.complex64)
    result = view_dominant_delay(front, offsets)
    assert result["power"] == 0.0
    assert result["valid"] is False
    assert math.isnan(result["delay_s"])


def test_view_dominant_delay_errors() -> None:
    offsets = _domain_offsets()
    front = _plane_wave(ROWS, COLS, offsets, TAU_S)
    with pytest.raises(ValueError):
        view_dominant_delay(front[:, :, :1], offsets[:1])
    with pytest.raises(ValueError):
        view_dominant_delay(front, offsets, np.zeros((ROWS, COLS), dtype=bool))
    with pytest.raises(ValueError):
        view_dominant_delay(front, offsets, np.ones((ROWS + 1, COLS), dtype=bool))
    with pytest.raises(ValueError):
        view_dominant_delay(front, offsets[:4])


# ---------------------------------------------------------------------------
# Build: round trips on v3 (primary) and v2
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("summary", ["none", "power", "delay"])
def test_build_partial_dataset_round_trip_v3(tmp_path: Path, summary: str) -> None:
    src = tmp_path / "src"
    write_v3_dataset(src, num_views=NUM_VIEWS, num_bs=NUM_BS, rows=ROWS, cols=COLS, bins=NUM_BINS)
    dataset = load_rf_dataset_manifest(src)
    full_offsets = np.asarray(dataset.frequency_offsets_hz)
    out = tmp_path / "out"
    manifest_path = build_partial_dataset(
        src,
        out,
        view_fraction=0.5,
        element_mask_kind="checkerboard",
        subband="2:6",
        summary=summary,
        seed=7,
    )
    partial = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert partial["schema_version"] == 1
    assert partial["mode"] == "rf_camera_partial_observation"
    assert partial["source_manifest_schema_version"] == 3
    assert partial["options"] == {
        "view_fraction": 0.5,
        "element_mask_kind": "checkerboard",
        "mask_fraction": 0.5,
        "subband": "2:6",
        "summary": summary,
        "seed": 7,
    }
    expected_indices = select_views(NUM_VIEWS, 0.5, seed=7).tolist()
    assert partial["kept_view_indices"] == expected_indices
    assert partial["kept_view_ids"] == [dataset.views[i].view_id for i in expected_indices]

    expected_mask = element_mask(ROWS, COLS, "checkerboard", fraction=0.5, seed=7)
    saved_mask = np.load(out / "element_mask.npy")
    np.testing.assert_array_equal(saved_mask, expected_mask)
    assert partial["element_mask"]["kept_count"] == int(expected_mask.sum())
    assert partial["element_mask"]["total"] == ROWS * COLS
    assert partial["subband"]["start"] == 2
    assert partial["subband"]["stop"] == 6
    assert partial["subband"]["num_bins"] == 4
    np.testing.assert_allclose(partial["subband"]["frequency_offsets_hz"], full_offsets[2:6])
    assert partial["summary"]["kind"] == summary
    assert "files" not in partial["summary"]
    if summary == "none":
        assert partial["raw_observation_axis_order"] == [
            "bs",
            "hemisphere",
            "row",
            "col",
            "frequency_offset",
        ]
    else:
        assert "raw_observation_axis_order" not in partial

    assert len(partial["views"]) == len(expected_indices)
    for entry, source_index in zip(partial["views"], expected_indices):
        view_obj = dataset.views[source_index]
        assert entry["source_index"] == source_index
        assert entry["view_id"] == view_obj.view_id
        assert entry["position_m"] == list(view_obj.position_m)
        full = dataset.load_aperture_cfr(view_obj)
        reproduced, _ = select_subband(full, full_offsets, 2, 6)
        reproduced = apply_element_mask(reproduced, expected_mask)
        rf_dir = out / "views" / entry["view_id"] / "rf"
        assert len(entry["bs"]) == NUM_BS
        for bs_pos, bs_entry in enumerate(entry["bs"]):
            source_bs = view_obj.bs[bs_pos]
            assert bs_entry["bs_id"] == source_bs.bs_id
            assert bs_entry["bs_index"] == source_bs.bs_index
            assert bs_entry["bs_direction_local"] == list(source_bs.bs_direction_local)
            assert bs_entry["bs_in_front_hemisphere"] == source_bs.bs_in_front_hemisphere
        if summary == "none":
            saved = np.load(rf_dir / "aperture_cfr.npy")
            assert saved.dtype == np.complex64
            assert saved.shape == (NUM_BS, 2, ROWS, COLS, 4)
            np.testing.assert_array_equal(saved, reproduced.astype(np.complex64))
            assert entry["artifacts"] == {
                "aperture_cfr": f"views/{entry['view_id']}/rf/aperture_cfr.npy"
            }
            for bs_entry in entry["bs"]:
                assert bs_entry["artifacts"] == {}
        elif summary == "power":
            np.testing.assert_array_equal(
                np.load(rf_dir / "element_power.npy"), element_power(reproduced)
            )
            np.testing.assert_array_equal(
                np.load(rf_dir / "hemisphere_power.npy"),
                hemisphere_total_power(reproduced),
            )
            assert np.load(rf_dir / "hemisphere_power.npy").shape == (NUM_BS, 2)
        else:
            assert entry["artifacts"] == {}
            for bs_pos, bs_entry in enumerate(entry["bs"]):
                saved = json.loads((rf_dir / bs_entry["bs_id"] / "dominant_delay.json").read_text())
                expected = view_dominant_delay(
                    reproduced[bs_pos, 0], full_offsets[2:6], expected_mask
                )
                assert saved["valid"] == expected["valid"]
                assert saved["power"] == pytest.approx(expected["power"])
                if expected["valid"]:
                    assert saved["delay_s"] == pytest.approx(expected["delay_s"], abs=1e-12)
                else:
                    assert saved["delay_s"] is None


def test_build_partial_v2_yields_5d_bs000(tmp_path: Path) -> None:
    src = tmp_path / "src"
    manifest = write_v2_dataset(src, num_views=NUM_VIEWS, rows=ROWS, cols=COLS, bins=NUM_BINS)
    out = tmp_path / "out"
    manifest_path = build_partial_dataset(src, out, summary="none", seed=0)
    partial = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert partial["source_manifest_schema_version"] == 2
    assert partial["base_stations"] == [
        {
            "bs_id": "bs_000",
            "index": 0,
            "position_m": list(manifest["config"]["tx_position"]),
            "look_at_m": list(manifest["config"]["tx_look_at"]),
        }
    ]
    for entry in partial["views"]:
        assert len(entry["bs"]) == 1
        assert entry["bs"][0]["bs_id"] == "bs_000"
        assert entry["bs"][0]["bs_index"] == 0
        saved = np.load(out / "views" / entry["view_id"] / "rf" / "aperture_cfr.npy")
        assert saved.dtype == np.complex64
        assert saved.shape == (1, 2, ROWS, COLS, NUM_BINS)
    gt = partial["path_geometry_gt"]
    assert gt is not None
    assert gt["axis_order"] == ["view", "bs", "path"]
    assert gt["path_schema"] is None


def test_build_partial_bs_axis_not_mixed(tmp_path: Path) -> None:
    src = tmp_path / "src"
    manifest = write_v3_dataset(src, num_views=2, num_bs=2, rows=ROWS, cols=COLS, bins=NUM_BINS)
    offsets = _fixture_offsets(src)
    delta_f = uniform_frequency_spacing(offsets)
    resolution = 1.0 / (NUM_BINS * delta_f)
    front_bs0 = _plane_wave(ROWS, COLS, offsets, 2 * resolution)
    front_bs1 = _plane_wave(ROWS, COLS, offsets, 5 * resolution)
    _overwrite_view_cfrs(src, manifest, [front_bs0, front_bs1])
    out = tmp_path / "out"
    build_partial_dataset(src, out, summary="delay", seed=0)
    partial = json.loads((out / "partial_manifest.json").read_text(encoding="utf-8"))
    assert len(partial["views"]) == 2
    for entry in partial["views"]:
        rf_dir = out / "views" / entry["view_id"] / "rf"
        delay0 = json.loads((rf_dir / "bs_000" / "dominant_delay.json").read_text())
        delay1 = json.loads((rf_dir / "bs_001" / "dominant_delay.json").read_text())
        assert delay0["delay_s"] == pytest.approx(2 * resolution, abs=1e-12)
        assert delay1["delay_s"] == pytest.approx(5 * resolution, abs=1e-12)


def test_build_partial_records_path_gt_and_bs_flags_v3(tmp_path: Path) -> None:
    src = tmp_path / "src"
    write_v3_dataset(src, num_views=2, num_bs=2, rows=ROWS, cols=COLS, bins=NUM_BINS)
    dataset = load_rf_dataset_manifest(src)
    out = tmp_path / "out"
    build_partial_dataset(src, out, summary="none", seed=0)
    partial = json.loads((out / "partial_manifest.json").read_text(encoding="utf-8"))
    gt = partial["path_geometry_gt"]
    assert gt is not None
    assert gt["axis_order"] == ["view", "bs", "path"]
    assert gt["view_index"] == "source_index"
    assert not Path(gt["artifact"]).is_absolute()
    assert (out / gt["artifact"]).exists()
    assert not Path(gt["path_schema"]).is_absolute()
    assert (out / gt["path_schema"]).resolve() == dataset.path_geometry_gt.schema_path.resolve()
    for entry, source_index in zip(partial["views"], partial["kept_view_indices"]):
        for bs_pos, bs_entry in enumerate(entry["bs"]):
            source_bs = dataset.views[source_index].bs[bs_pos]
            assert bs_entry["bs_in_front_hemisphere"] == source_bs.bs_in_front_hemisphere
            assert bs_entry["bs_direction_local"] == list(source_bs.bs_direction_local)


def test_build_partial_summary_axis_orders(tmp_path: Path) -> None:
    src = tmp_path / "src"
    write_v3_dataset(src, num_views=2, num_bs=2, rows=ROWS, cols=COLS, bins=NUM_BINS)
    for summary, expected in [
        ("none", {"aperture_cfr": ["bs", "hemisphere", "row", "col", "frequency_offset"]}),
        (
            "power",
            {
                "element_power": ["bs", "hemisphere", "row", "col"],
                "hemisphere_power": ["bs", "hemisphere"],
            },
        ),
        ("delay", {}),
    ]:
        out = tmp_path / f"out-{summary}"
        build_partial_dataset(src, out, summary=summary, seed=0)
        partial = json.loads((out / "partial_manifest.json").read_text(encoding="utf-8"))
        assert partial["summary"] == {"kind": summary, "axis_order": expected}


def test_build_partial_delay_requires_two_bins(tmp_path: Path) -> None:
    src = tmp_path / "src"
    write_v3_dataset(src, num_views=2, num_bs=2, rows=ROWS, cols=COLS, bins=NUM_BINS)
    with pytest.raises(ValueError):
        build_partial_dataset(tmp_path / "src", tmp_path / "out", summary="delay", subband="2:3")
    assert not (tmp_path / "out").exists()


def test_build_partial_single_bin_source_power_summary(tmp_path: Path) -> None:
    src = tmp_path / "src"
    write_v3_dataset(src, num_views=2, num_bs=NUM_BS, rows=ROWS, cols=COLS, bins=1)
    out = tmp_path / "out"
    manifest_path = build_partial_dataset(src, out, summary="power", seed=0)
    partial = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert partial["subband"]["num_bins"] == 1
    assert partial["subband"]["delay_resolution_s"] is None
    assert partial["subband"]["unambiguous_delay_s"] is None
    view_id = partial["views"][0]["view_id"]
    assert (out / "views" / view_id / "rf" / "element_power.npy").exists()


def test_build_partial_delay_zero_front_writes_null(tmp_path: Path) -> None:
    src = tmp_path / "src"
    manifest = write_v3_dataset(src, num_views=1, num_bs=2, rows=ROWS, cols=COLS, bins=NUM_BINS)
    offsets = _fixture_offsets(src)
    delta_f = uniform_frequency_spacing(offsets)
    resolution = 1.0 / (NUM_BINS * delta_f)
    front_bs0 = _plane_wave(ROWS, COLS, offsets, 2 * resolution)
    front_bs1 = np.zeros((ROWS, COLS, NUM_BINS), dtype=np.complex128)
    _overwrite_view_cfrs(src, manifest, [front_bs0, front_bs1])
    out = tmp_path / "out"
    build_partial_dataset(src, out, summary="delay", seed=0)
    view_id = manifest["views"][0]["view_id"]
    payload1 = json.loads(
        (out / "views" / view_id / "rf" / "bs_001" / "dominant_delay.json").read_text()
    )
    assert payload1 == {
        "delay_s": None,
        "power": 0.0,
        "delay_resolution_s": pytest.approx(1.0 / (NUM_BINS * delta_f)),
        "unambiguous_delay_s": pytest.approx(1.0 / delta_f),
        "valid": False,
    }
    # Strict JSON: no NaN literals.
    raw = (out / "views" / view_id / "rf" / "bs_001" / "dominant_delay.json").read_text()
    assert "NaN" not in raw


# ---------------------------------------------------------------------------
# Portable manifest
# ---------------------------------------------------------------------------


def test_build_partial_manifest_paths_relative_and_config_copied(tmp_path: Path) -> None:
    src = tmp_path / "src"
    source_manifest = write_v3_dataset(
        src, num_views=2, num_bs=2, rows=ROWS, cols=COLS, bins=NUM_BINS
    )
    out = tmp_path / "out"
    build_partial_dataset(src, out, summary="power", seed=1)
    partial = json.loads((out / "partial_manifest.json").read_text(encoding="utf-8"))

    assert not Path(partial["source_dataset"]).is_absolute()
    assert (out / partial["source_dataset"]).resolve() == src.resolve()
    assert not Path(partial["source_manifest"]).is_absolute()
    assert (out / partial["source_manifest"]).resolve() == (src / "dataset_manifest.json").resolve()
    assert not Path(partial["camera_model_source"]).is_absolute()
    assert (out / partial["camera_model_source"]).exists()
    assert Path(partial["source_dataset"]).as_posix() == partial["source_dataset"]
    assert (
        os.path.relpath(src.resolve(), out.resolve()).replace(os.sep, "/")
        == partial["source_dataset"]
    )

    assert partial["config"] == source_manifest["config"]
    assert partial["carrier_frequency_hz"] == source_manifest["config"]["carrier_frequency_hz"]
    assert partial["hemispheres"] == ["front", "back"]
    assert [bs["bs_id"] for bs in partial["base_stations"]] == ["bs_000", "bs_001"]

    subband = partial["subband"]
    dataset = load_rf_dataset_manifest(src)
    np.testing.assert_allclose(
        subband["absolute_frequencies_hz"], np.asarray(dataset.absolute_frequencies_hz)
    )
    delta_f = uniform_frequency_spacing(np.asarray(dataset.frequency_offsets_hz))
    assert subband["delay_resolution_s"] == pytest.approx(1.0 / (NUM_BINS * delta_f))
    assert subband["unambiguous_delay_s"] == pytest.approx(1.0 / delta_f)

    def _collect_paths(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key in (
                    "artifact",
                    "file",
                    "aperture_cfr",
                    "element_power",
                    "hemisphere_power",
                    "dominant_delay",
                ):
                    yield value
                else:
                    yield from _collect_paths(value)
        elif isinstance(node, list):
            for value in node:
                yield from _collect_paths(value)

    checked = 0
    for rel in _collect_paths({"views": partial["views"], "mask": partial["element_mask"]}):
        assert not Path(rel).is_absolute()
        assert (out / rel).exists()
        checked += 1
    assert checked > 0


# ---------------------------------------------------------------------------
# out_dir must never overwrite the source
# ---------------------------------------------------------------------------


def _write_v3_source(root: Path) -> Path:
    write_v3_dataset(root, num_views=2, num_bs=2, rows=ROWS, cols=COLS, bins=NUM_BINS)
    return root


def test_out_dir_equal_to_dataset_dir_is_rejected(tmp_path: Path) -> None:
    src = _write_v3_source(tmp_path / "src")
    before = _snapshot_tree(src)
    with pytest.raises(ValueError):
        build_partial_dataset(src, src, subband="2:6", element_mask_kind="checkerboard")
    _assert_tree_unchanged(src, before)


def test_out_dir_inside_dataset_dir_is_rejected(tmp_path: Path) -> None:
    src = _write_v3_source(tmp_path / "src")
    before = _snapshot_tree(src)
    with pytest.raises(ValueError):
        build_partial_dataset(src, src / "partial")
    assert not (src / "partial").exists()
    _assert_tree_unchanged(src, before)


def test_out_dir_containing_dataset_dir_is_rejected(tmp_path: Path) -> None:
    src = _write_v3_source(tmp_path / "src")
    before = _snapshot_tree(src)
    with pytest.raises(ValueError):
        build_partial_dataset(src, tmp_path)
    _assert_tree_unchanged(src, before)


def test_out_dir_symlink_to_dataset_dir_is_rejected(tmp_path: Path) -> None:
    src = _write_v3_source(tmp_path / "src")
    link = tmp_path / "link"
    link.symlink_to(src, target_is_directory=True)
    before = _snapshot_tree(src)
    with pytest.raises(ValueError):
        build_partial_dataset(src, link)
    _assert_tree_unchanged(src, before)


# ---------------------------------------------------------------------------
# Overwrite handling
# ---------------------------------------------------------------------------


def test_rerun_into_existing_output_requires_overwrite(tmp_path: Path) -> None:
    src = _write_v3_source(tmp_path / "src")
    out = tmp_path / "out"
    build_partial_dataset(src, out, summary="none", seed=0)
    with pytest.raises(FileExistsError):
        build_partial_dataset(src, out, summary="power", seed=0)


def test_overwrite_removes_stale_views_and_summaries(tmp_path: Path) -> None:
    src = _write_v3_source(tmp_path / "src")
    out = tmp_path / "out"
    build_partial_dataset(
        src, out, summary="none", view_fraction=1.0, seed=0, element_mask_kind="none"
    )
    first_views = {
        entry["view_id"]
        for entry in json.loads((out / "partial_manifest.json").read_text())["views"]
    }
    assert len(first_views) == 2
    build_partial_dataset(
        src,
        out,
        summary="delay",
        view_fraction=0.5,
        seed=7,
        element_mask_kind="checkerboard",
        subband="2:6",
        overwrite=True,
    )
    partial = json.loads((out / "partial_manifest.json").read_text(encoding="utf-8"))
    kept = set(partial["kept_view_ids"])
    assert kept <= first_views
    leftovers = sorted(
        p.relative_to(out).as_posix() for p in (out / "views").rglob("*") if p.is_file()
    )
    assert leftovers, "expected delay files in the overwritten output"
    assert all("/bs_00" in name and name.endswith("dominant_delay.json") for name in leftovers), (
        leftovers
    )
    assert sorted((out / "views").iterdir())[0].is_dir()
    assert {p.name for p in (out / "views").iterdir()} == kept
    assert not list(out.rglob("aperture_cfr.npy"))
    assert not list(out.rglob("element_power.npy"))


def test_overwrite_refuses_directory_with_foreign_entries(tmp_path: Path) -> None:
    src = _write_v3_source(tmp_path / "src")
    out = tmp_path / "out"
    out.mkdir()
    stray = out / "notes.txt"
    stray.write_text("unrelated", encoding="utf-8")
    with pytest.raises(FileExistsError):
        build_partial_dataset(src, out, summary="none", seed=0, overwrite=True)
    assert stray.exists()
    assert not (out / "partial_manifest.json").exists()


def test_failed_validation_creates_nothing(tmp_path: Path) -> None:
    src = _write_v3_source(tmp_path / "src")
    out = tmp_path / "fresh-out"
    with pytest.raises(ValueError):
        build_partial_dataset(src, out, subband="6:2")
    assert not out.exists()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_success_on_v3(tmp_path: Path) -> None:
    src = tmp_path / "src"
    write_v3_dataset(src, num_views=2, num_bs=2, rows=ROWS, cols=COLS, bins=NUM_BINS)
    out = tmp_path / "out"
    result = CliRunner().invoke(
        cli,
        [
            "rf-camera-partial",
            str(src),
            str(out),
            "--view-fraction",
            "0.5",
            "--element-mask",
            "checkerboard",
            "--summary",
            "delay",
        ],
    )
    assert result.exit_code == 0, result.output
    assert (out / "partial_manifest.json").exists()


def test_cli_same_dir_is_clean_error(tmp_path: Path) -> None:
    src = _write_v3_source(tmp_path / "src")
    before = _snapshot_tree(src)
    result = CliRunner().invoke(cli, ["rf-camera-partial", str(src), str(src)])
    assert result.exit_code != 0
    assert "must not equal" in result.output or "nested" in result.output
    assert "Traceback" not in result.output
    assert isinstance(result.exception, SystemExit)
    _assert_tree_unchanged(src, before)


def test_cli_schema1_is_clean_error(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "dataset_manifest.json").write_text(
        json.dumps({"schema_version": 1, "views": []}), encoding="utf-8"
    )
    result = CliRunner().invoke(cli, ["rf-camera-partial", str(src), str(tmp_path / "out")])
    assert result.exit_code != 0
    assert "unsupported" in result.output
    assert "Traceback" not in result.output
    assert isinstance(result.exception, SystemExit)


def test_cli_nonempty_out_without_overwrite_is_clean_error(tmp_path: Path) -> None:
    src = _write_v3_source(tmp_path / "src")
    out = tmp_path / "out"
    out.mkdir()
    (out / "existing.npy").touch()
    result = CliRunner().invoke(cli, ["rf-camera-partial", str(src), str(out)])
    assert result.exit_code != 0
    assert "not empty" in result.output
    assert "Traceback" not in result.output
    assert isinstance(result.exception, SystemExit)
