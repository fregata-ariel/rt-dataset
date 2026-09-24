"""Multi-BS config validation, zero-energy delay, and CLI count checks."""

from pathlib import Path

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
