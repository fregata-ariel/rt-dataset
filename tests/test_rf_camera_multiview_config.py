"""Multi-BS config validation, zero-energy delay, and CLI count checks."""

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from click.testing import CliRunner

from plateau_rt.adapters.sionna.rf_camera_dataset import (
    RFMultiViewConfig,
    RFMultiViewDataset,
)
from plateau_rt.cli.main import cli
from plateau_rt.domain.rf_camera.camera import (
    build_direction_cosine_camera_model,
    generate_ring_views,
)
from plateau_rt.domain.rf_camera.imaging import frequency_offsets


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tx_positions": (-50.0, -50.0, 30.0)},
        {"tx_positions": ((1.0, 2.0),)},
        {"tx_positions": ((1.0, 2.0, 3.0, 4.0),)},
        {"tx_positions": ((float("nan"), 0.0, 0.0),)},
        {"tx_positions": ((float("inf"), 0.0, 0.0),)},
        {
            "tx_positions": ((0.0, 0.0, 30.0),),
            "tx_look_ats": ((0.0, 0.0),),
        },
        {"tx_look_at": (0.0, 0.0)},
    ],
)
def test_validate_rejects_bad_bs_vectors(kwargs):
    with pytest.raises(ValueError):
        RFMultiViewConfig(**kwargs).validate()


def test_validate_accepts_valid_two_bs_config():
    cfg = RFMultiViewConfig(
        tx_positions=((0.0, 0.0, 30.0), (1, 2, 3)),
        tx_look_at=(5.0, 5.0, 5.0),
    )
    cfg.validate()
    stations = cfg.resolve_base_stations()
    assert [s[0] for s in stations] == ["bs_000", "bs_001"]
    assert stations[1][1] == (1.0, 2.0, 3.0)
    assert all(isinstance(v, float) for v in stations[1][1])
    assert stations[0][2] == (5.0, 5.0, 5.0)

    per_bs = RFMultiViewConfig(
        tx_positions=((0.0, 0.0, 30.0), (10.0, 0.0, 30.0)),
        tx_look_at=(5.0, 5.0, 5.0),
        tx_look_ats=((0.0, 0.0, 0.0), (1.0, 2.0, 3.0)),
    )
    per_bs.validate()
    resolved = per_bs.resolve_base_stations()
    assert resolved[0][2] == (0.0, 0.0, 0.0)
    assert resolved[1][2] == (1.0, 2.0, 3.0)


def _make_dataset(tmp_path: Path):
    cfg = RFMultiViewConfig(rx_rows=4, rx_cols=4, fft_rows=16, fft_cols=16, num_frequency_bins=8)
    view = generate_ring_views(target=(0, 0, 0), radius_m=10, ue_height_m=1.5, num_views=1)[0]
    ds = RFMultiViewDataset(Path("unused.xml"), views=[view], config=cfg)
    valid_mask = build_direction_cosine_camera_model(
        fft_rows=16,
        fft_cols=16,
        horizontal_spacing_lambda=0.5,
        vertical_spacing_lambda=0.5,
    )["valid_mask"]
    freqs = frequency_offsets(cfg.bandwidth_hz, 8)
    return ds, view, valid_mask, freqs


def test_write_view_bs_zero_aperture_gives_nan_delay(tmp_path):
    ds, view, valid_mask, freqs = _make_dataset(tmp_path)
    aperture = np.zeros((2, 4, 4, 8), dtype=complex)
    entry = ds._write_view_bs(
        tmp_path,
        view,
        aperture,
        bs_id="bs_000",
        bs_position=(-10.0, 0.0, 1.5),
        frequency_offsets_hz=freqs,
        valid_ray_mask=valid_mask,
    )
    delay = np.load(tmp_path / entry["artifacts"]["dominant_delay_s"])
    power = np.load(tmp_path / entry["artifacts"]["dominant_delay_power"])
    assert np.isnan(delay).all()
    assert (power == 0.0).all()


def test_write_view_bs_broadside_wave_zero_delay(tmp_path):
    ds, view, valid_mask, freqs = _make_dataset(tmp_path)
    aperture = np.zeros((2, 4, 4, 8), dtype=complex)
    aperture[0] = 1.0
    entry = ds._write_view_bs(
        tmp_path,
        view,
        aperture,
        bs_id="bs_000",
        bs_position=(-10.0, 0.0, 1.5),
        frequency_offsets_hz=freqs,
        valid_ray_mask=valid_mask,
    )
    delay = np.load(tmp_path / entry["artifacts"]["dominant_delay_s"])
    power = np.load(tmp_path / entry["artifacts"]["dominant_delay_power"])
    assert (np.isfinite(delay) == (power > 0)).all()
    assert np.isnan(delay[~valid_mask]).all()
    peak = np.unravel_index(int(np.argmax(power)), power.shape)
    assert delay[peak] == 0.0


def test_cli_bs_look_at_count_mismatch(tmp_path):
    xml_file = tmp_path / "empty.xml"
    xml_file.write_text("<xml/>")
    out_dir = tmp_path / "out"
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "rf-camera-multiview",
            str(xml_file),
            str(out_dir),
            "--bs-position",
            "0",
            "0",
            "30",
            "--bs-position",
            "1",
            "1",
            "30",
            "--bs-look-at",
            "0",
            "0",
            "0",
        ],
    )
    assert result.exit_code == 2
    assert "--bs-look-at" in result.output


def test_tx_power_dbm_default_and_validation():
    """tx_power_dbm defaults to 44 dBm and rejects non-finite values."""
    assert RFMultiViewConfig().tx_power_dbm == 44.0
    for bad in (float("inf"), float("nan"), True):
        with pytest.raises(ValueError):
            RFMultiViewConfig(tx_power_dbm=bad).validate()
    assert "tx_power_dbm" in asdict(RFMultiViewConfig())


def test_mechanism_and_los_free_defaults_and_validation():
    """Mechanism flags default to (True, True, False, False) and reject non-bools."""
    config = RFMultiViewConfig()
    assert (config.specular_reflection, config.refraction, config.diffraction) == (
        True,
        True,
        False,
    )
    assert config.los_free_trace is False
    assert {"specular_reflection", "refraction", "diffraction", "los_free_trace"} <= set(
        asdict(config)
    )
    config.validate()
    bad_kwargs: list[dict[str, Any]] = [
        {"specular_reflection": 1},
        {"refraction": 1},
        {"diffraction": 0},
        {"los_free_trace": "yes"},
    ]
    for kwargs in bad_kwargs:
        with pytest.raises(ValueError):
            RFMultiViewConfig(**kwargs).validate()


def test_write_view_with_placement_and_los_free(tmp_path):
    ds, view, valid_mask, freqs = _make_dataset(tmp_path)
    aperture = (
        np.arange(2 * 2 * 4 * 4 * 8, dtype=np.float64).reshape(2, 2, 4, 4, 8)
        + 1j * np.ones((2, 2, 4, 4, 8))
    ).astype(np.complex64)
    placement = {"bank_index": 0, "source": "ring"}
    entry = ds._write_view(
        tmp_path,
        view,
        aperture,
        base_stations=[("bs_000", (-10.0, 0.0, 1.5), (0.0, 0.0, 0.0))],
        frequency_offsets_hz=freqs,
        valid_ray_mask=valid_mask,
        placement=placement,
        aperture_cfr_los_free=0.5 * aperture,
    )
    assert list(entry) == [
        "view_id",
        "position_m",
        "look_at_m",
        "orientation_rad",
        "placement",
        "artifacts",
        "bs",
    ]
    assert entry["placement"] == placement
    assert list(entry["artifacts"]) == ["pose", "aperture_cfr", "aperture_cfr_los_free"]
    loaded = np.load(tmp_path / entry["artifacts"]["aperture_cfr_los_free"])
    assert loaded.dtype == np.complex64
    assert np.array_equal(loaded, (0.5 * aperture).astype(np.complex64))

    plain = ds._write_view(
        tmp_path / "plain",
        view,
        aperture,
        base_stations=[("bs_000", (-10.0, 0.0, 1.5), (0.0, 0.0, 0.0))],
        frequency_offsets_hz=freqs,
        valid_ray_mask=valid_mask,
    )
    assert list(plain) == [
        "view_id",
        "position_m",
        "look_at_m",
        "orientation_rad",
        "artifacts",
        "bs",
    ]
    assert list(plain["artifacts"]) == ["pose", "aperture_cfr"]
    assert not (
        tmp_path / "plain" / "views" / view.view_id / "rf" / "aperture_cfr_los_free.npy"
    ).exists()


def test_dataset_rejects_mismatched_view_placements(tmp_path):
    cfg = RFMultiViewConfig(rx_rows=4, rx_cols=4, fft_rows=16, fft_cols=16, num_frequency_bins=8)
    view = generate_ring_views(target=(0, 0, 0), radius_m=10, ue_height_m=1.5, num_views=1)[0]
    with pytest.raises(ValueError):
        RFMultiViewDataset(Path("unused.xml"), views=[view], config=cfg, view_placements=[{}, {}])
