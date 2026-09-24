"""CPU-only tests for :mod:`plateau_rt.application.rf_camera_observe` and its CLI."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest
from click.testing import CliRunner
from rf_manifest_fixtures import write_v2_dataset, write_v3_dataset

from plateau_rt.application.rf_camera_observe import (
    link_rng,
    observe_dataset,
    ue_array_rng,
)
from plateau_rt.application.rf_dataset_manifest import ManifestError, load_rf_dataset_manifest
from plateau_rt.cli.main import cli
from plateau_rt.domain.rf_camera.impairments import (
    ImpairmentConfig,
    NoiseSpec,
    apply_impairments,
    draw_element_errors,
)

NAME = "observed"


def _manifest_path(root):
    return root / "dataset_manifest.json"


def _read_manifest(root):
    return json.loads(_manifest_path(root).read_text(encoding="utf-8"))


def _gt_path(root, view_id, bs_id, name=NAME):
    return root / "views" / view_id / "rf" / bs_id / "observed" / name / "impairment_gt.json"


def _read_gt(root, view_id, bs_id, name=NAME):
    return json.loads(_gt_path(root, view_id, bs_id, name).read_text(encoding="utf-8"))


def _pair_path(root, view_id, bs_id, name=NAME):
    return root / "views" / view_id / "rf" / bs_id / "observed" / name / "aperture_cfr.npy"


def _isotropic_pair(shape, power, rng):
    """A pair whose isotropic front+back has mean power ``power`` (all in front)."""
    sample = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
    sample = sample / np.sqrt(np.mean(np.abs(sample) ** 2)) * np.sqrt(power)
    front = sample.astype(np.complex64)
    back = np.zeros(shape, dtype=np.complex64)
    return np.stack([front, back])


def _patch_ap(aperture_path, aperture):
    np.save(aperture_path, aperture.astype(np.complex64))


def _snapshot_files(root):
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file())


def test_dataset_wide_noise_floor_and_reference(tmp_path):
    root = tmp_path / "ds"
    rows = cols = 8
    bins = 64
    write_v3_dataset(root, num_views=2, num_bs=2, rows=rows, cols=cols, bins=bins, seed=0)
    manifest = _read_manifest(root)
    view0, view1 = manifest["views"][0], manifest["views"][1]
    rng = np.random.default_rng(123)

    _patch_ap(
        root / view0["artifacts"]["aperture_cfr"],
        np.stack(
            [
                _isotropic_pair((rows, cols, bins), 1.0, rng),
                _isotropic_pair((rows, cols, bins), 1e-12, rng),
            ]
        ),
    )
    _patch_ap(
        root / view1["artifacts"]["aperture_cfr"],
        np.stack(
            [
                _isotropic_pair((rows, cols, bins), 1e-12, rng),
                _isotropic_pair((rows, cols, bins), 0.01, rng),
            ]
        ),
    )

    observe_dataset(
        root,
        ImpairmentConfig(front_to_back_db=0.0),
        noise=NoiseSpec(snr_db=10.0),
        seed=0,
    )
    written = _read_manifest(root)
    noise = written["observations"][NAME]["noise"]
    assert noise["mode"] == "snr_relative_to_reference"
    assert noise["reference"] == "max_pair_isotropic_mean_power"
    assert noise["reference_power"] == pytest.approx(1.0, rel=1e-4)
    assert noise["reference_pair"] == {"view_id": view0["view_id"], "bs_id": "bs_000"}
    variance = noise["noise_variance"]
    assert variance == pytest.approx(noise["reference_power"] / 10.0, rel=1e-9)

    strong = _read_gt(root, view0["view_id"], "bs_000")
    weak = _read_gt(root, view1["view_id"], "bs_001")
    for gt in (strong, weak):
        assert gt["noise_variance"] == pytest.approx(variance, rel=1e-9)
    assert strong["expected_snr_db"] == pytest.approx(10.0, abs=1e-4)
    assert weak["expected_snr_db"] == pytest.approx(-10.0, abs=1e-4)
    assert strong["achieved_snr_db"] == pytest.approx(strong["expected_snr_db"], abs=1.0)
    assert weak["achieved_snr_db"] == pytest.approx(weak["expected_snr_db"], abs=1.0)


def test_zero_signal_pair_still_gets_noise(tmp_path):
    root = tmp_path / "ds"
    rows = cols = 8
    bins = 64
    write_v3_dataset(root, num_views=1, num_bs=2, rows=rows, cols=cols, bins=bins, seed=1)
    manifest = _read_manifest(root)
    view = manifest["views"][0]
    aperture_path = root / view["artifacts"]["aperture_cfr"]
    aperture = np.load(aperture_path)
    aperture[0] = 0.0
    np.save(aperture_path, aperture)

    observe_dataset(root, ImpairmentConfig(front_to_back_db=0.0), noise=NoiseSpec(snr_db=10.0))

    written = _read_manifest(root)
    noise = written["observations"][NAME]["noise"]
    gt = _read_gt(root, view["view_id"], "bs_000")
    observed = np.load(_pair_path(root, view["view_id"], "bs_000"))

    assert not np.all(observed == 0.0)
    measured = float(np.mean(np.abs(observed.astype(np.complex128)) ** 2))
    assert measured == pytest.approx(noise["noise_variance"], rel=0.2)
    assert gt["signal_power"] == 0.0
    assert gt["expected_snr_db"] is None
    assert gt["achieved_snr_db"] is None

    manifest_text = _manifest_path(root).read_text(encoding="utf-8")
    gt_text = _gt_path(root, view["view_id"], "bs_000").read_text(encoding="utf-8")
    for text in (manifest_text, gt_text):
        assert "Infinity" not in text
        assert "NaN" not in text


def test_per_pair_seeding_matches_direct_calls(tmp_path):
    root = tmp_path / "ds"
    rows, cols, bins = 4, 4, 8
    write_v3_dataset(root, num_views=2, num_bs=2, rows=rows, cols=cols, bins=bins, seed=2)
    config = ImpairmentConfig(
        front_to_back_db=10.0,
        timing_offset_std_ns=2.0,
        element_gain_std_db=0.5,
        element_phase_std_deg=5.0,
        random_common_phase=True,
    )
    seed = 11
    observe_dataset(root, config, noise=NoiseSpec(snr_db=20.0), seed=seed)

    dataset = load_rf_dataset_manifest(root)
    written = _read_manifest(root)
    variance = written["observations"][NAME]["noise"]["noise_variance"]
    freqs = dataset.frequency_offsets_hz

    per_view_errors = {}
    for view in dataset.views:
        cfr = dataset.load_aperture_cfr(view)
        errors = draw_element_errors(config, rows, cols, ue_array_rng(seed, view.index))
        per_view_errors[view.view_id] = errors
        for entry in view.bs:
            direct, direct_gt = apply_impairments(
                cfr[entry.bs_index][[0, 1]],
                freqs,
                config,
                link_rng(seed, view.index, entry.bs_index),
                element_errors=errors,
                noise_variance=variance,
            )
            observed = np.load(_pair_path(root, view.view_id, entry.bs_id))
            assert np.array_equal(observed, direct)

            loaded_gt = _read_gt(root, view.view_id, entry.bs_id)
            for key in ("view_id", "bs_id", "observation", "ideal_isotropic_power"):
                loaded_gt.pop(key)
            assert loaded_gt == direct_gt

    view0 = dataset.views[0]
    bs0 = _read_gt(root, view0.view_id, "bs_000")
    bs1 = _read_gt(root, view0.view_id, "bs_001")
    assert bs0["element_gain_db"] == bs1["element_gain_db"]
    assert bs0["element_phase_rad"] == bs1["element_phase_rad"]
    assert bs0["timing_offset_s"] != bs1["timing_offset_s"]
    assert bs0["common_phase_rad"] != bs1["common_phase_rad"]

    view1 = _read_gt(root, dataset.views[1].view_id, "bs_000")
    assert view1["element_gain_db"] != bs0["element_gain_db"]

    assert ue_array_rng(3, 0).standard_normal() != link_rng(3, 0, 0).standard_normal()

    first = np.load(_pair_path(root, view0.view_id, "bs_000"))
    observe_dataset(root, config, noise=NoiseSpec(snr_db=20.0), seed=seed + 1)
    second = np.load(_pair_path(root, view0.view_id, "bs_000"))
    assert not np.array_equal(first, second)


def test_observe_is_idempotent(tmp_path):
    root = tmp_path / "ds"
    write_v3_dataset(root, num_views=2, num_bs=2, rows=3, cols=3, bins=6, seed=3)
    config = ImpairmentConfig(front_to_back_db=15.0, timing_offset_ns=4.0)

    observe_dataset(root, config, noise=NoiseSpec(snr_db=15.0), seed=4)
    manifest_first = _manifest_path(root).read_text(encoding="utf-8")
    arrays_first = {}
    gts_first = {}
    for view, bs in load_rf_dataset_manifest(root).pairs():
        arrays_first[(view.view_id, bs.bs_id)] = np.load(
            _pair_path(root, view.view_id, bs.bs_id)
        ).copy()
        gts_first[(view.view_id, bs.bs_id)] = _gt_path(root, view.view_id, bs.bs_id).read_text(
            encoding="utf-8"
        )

    observe_dataset(root, config, noise=NoiseSpec(snr_db=15.0), seed=4)
    assert _manifest_path(root).read_text(encoding="utf-8") == manifest_first
    for view, bs in load_rf_dataset_manifest(root).pairs():
        assert np.array_equal(
            np.load(_pair_path(root, view.view_id, bs.bs_id)),
            arrays_first[(view.view_id, bs.bs_id)],
        )
        assert (
            _gt_path(root, view.view_id, bs.bs_id).read_text(encoding="utf-8")
            == gts_first[(view.view_id, bs.bs_id)]
        )


def test_named_variants_coexist(tmp_path):
    root = tmp_path / "ds"
    write_v3_dataset(root, num_views=2, num_bs=2, rows=3, cols=3, bins=6, seed=5)
    dataset = load_rf_dataset_manifest(root)

    observe_dataset(root, ImpairmentConfig(), noise=NoiseSpec(snr_db=20.0), name="snr20")
    first_files = {
        (view.view_id, bs.bs_id): (
            np.load(_pair_path(root, view.view_id, bs.bs_id, "snr20")).copy(),
            _gt_path(root, view.view_id, bs.bs_id, "snr20").read_text(encoding="utf-8"),
        )
        for view, bs in dataset.pairs()
    }

    observe_dataset(root, ImpairmentConfig(), noise=NoiseSpec(snr_db=0.0), name="snr0")
    written = _read_manifest(root)
    assert set(written["observations"]) == {"snr20", "snr0"}
    for dataset_view in written["views"]:
        assert dataset_view["bs"]
        for entry in dataset_view["bs"]:
            keys = entry["artifacts"]
            assert "observed.snr20.aperture_cfr" in keys
            assert "observed.snr0.aperture_cfr" in keys

    for view, bs in dataset.pairs():
        array, gt_text = first_files[(view.view_id, bs.bs_id)]
        assert np.array_equal(np.load(_pair_path(root, view.view_id, bs.bs_id, "snr20")), array)
        assert (
            _gt_path(root, view.view_id, bs.bs_id, "snr20").read_text(encoding="utf-8") == gt_text
        )


def test_invalid_name_writes_nothing(tmp_path):
    root = tmp_path / "ds"
    write_v3_dataset(root, num_views=1, num_bs=2, rows=2, cols=2, bins=4, seed=6)
    before = _manifest_path(root).read_text(encoding="utf-8")
    files_before = _snapshot_files(root)
    for bad_name in ("../x", "", "a/b"):
        with pytest.raises(ValueError):
            observe_dataset(root, ImpairmentConfig(), name=bad_name)
    assert _manifest_path(root).read_text(encoding="utf-8") == before
    assert _snapshot_files(root) == files_before


def test_schema_v2_is_rejected(tmp_path):
    root = tmp_path / "ds"
    write_v2_dataset(root, num_views=2, rows=2, cols=2, bins=4, seed=7)
    before = _manifest_path(root).read_text(encoding="utf-8")
    files_before = _snapshot_files(root)
    with pytest.raises(ManifestError):
        observe_dataset(root, ImpairmentConfig(), noise=NoiseSpec(snr_db=10.0))
    assert _manifest_path(root).read_text(encoding="utf-8") == before
    assert _snapshot_files(root) == files_before


def test_manifest_stays_readable_and_original_parts_unchanged(tmp_path):
    root = tmp_path / "ds"
    write_v3_dataset(root, num_views=2, num_bs=2, rows=3, cols=3, bins=6, seed=8)
    original = _read_manifest(root)

    observe_dataset(root, ImpairmentConfig(front_to_back_db=12.0), noise=NoiseSpec(snr_db=12.0))
    rewritten = _read_manifest(root)

    dataset = load_rf_dataset_manifest(root)
    assert dataset.num_views == 2
    assert dataset.num_bs == 2

    stripped = copy.deepcopy(rewritten)
    stripped.pop("observations", None)
    expected = copy.deepcopy(original)
    for view in stripped["views"]:
        for entry in view["bs"]:
            for key in list(entry.get("artifacts", {})):
                if key.startswith("observed."):
                    del entry["artifacts"][key]
    assert stripped == expected


def test_cli_flag_conflicts_and_normal_run(tmp_path):
    root = tmp_path / "ds"
    write_v3_dataset(root, num_views=2, num_bs=2, rows=3, cols=3, bins=6, seed=9)
    runner = CliRunner()

    conflict = runner.invoke(
        cli,
        [
            "rf-camera-observe",
            str(root),
            "--random-common-phase",
            "--common-phase-deg",
            "30",
        ],
    )
    assert conflict.exit_code != 0
    assert "Error" in conflict.output or "error" in conflict.output

    both_noise = runner.invoke(
        cli,
        ["rf-camera-observe", str(root), "--snr-db", "10", "--noise-variance", "1e-3"],
    )
    assert both_noise.exit_code != 0

    result = runner.invoke(cli, ["rf-camera-observe", str(root), "--snr-db", "20"])
    assert result.exit_code == 0, result.output
    assert result.output.count("expected_snr_db=") == 4


def test_cli_rejects_schema_v2_and_reports_no_noise(tmp_path):
    runner = CliRunner()
    v2_root = tmp_path / "v2"
    write_v2_dataset(v2_root, num_views=1, rows=2, cols=2, bins=4, seed=10)
    rejected = runner.invoke(cli, ["rf-camera-observe", str(v2_root), "--snr-db", "10"])
    assert rejected.exit_code == 1
    assert "schema v3" in rejected.output
    assert rejected.exception is None or isinstance(rejected.exception, SystemExit)

    v3_root = tmp_path / "v3"
    write_v3_dataset(v3_root, num_views=1, num_bs=2, rows=2, cols=2, bins=4, seed=11)
    result = runner.invoke(cli, ["rf-camera-observe", str(v3_root), "--noise-variance", "0"])
    assert result.exit_code == 0, result.output
    assert result.output.count("no noise") == 2
