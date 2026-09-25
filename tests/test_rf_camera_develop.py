"""Golden and property tests for the Sionna-free RF-camera development."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from rf_manifest_fixtures import write_v2_dataset, write_v3_dataset

from plateau_rt.application.rf_camera_develop import develop_params_from_manifest
from plateau_rt.application.rf_dataset_manifest import (
    ManifestError,
    load_rf_dataset_manifest,
)
from plateau_rt.domain.rf_camera.delay import circular_delay_error_s
from plateau_rt.domain.rf_camera.develop import (
    DevelopParams,
    angle_delay_power,
    center_frequency_products,
    delay_products,
    develop_hemisphere_image,
    raw_spectrum_energy,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "rf_camera_develop"
OUTPUT_NAMES = (
    "angular_cfr_center",
    "angular_power_center",
    "phase_valid_mask",
    "dominant_delay_s",
    "dominant_delay_power",
)


def _load_fixture() -> tuple[DevelopParams, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Return ``(params, aperture, offsets_hz, valid_mask, meta)`` of the golden inputs."""
    meta = json.loads((FIXTURE_DIR / "inputs.json").read_text(encoding="utf-8"))
    config = meta["config"]
    params = DevelopParams(
        fft_rows=config["fft_rows"],
        fft_cols=config["fft_cols"],
        rx_rows=config["rx_rows"],
        rx_cols=config["rx_cols"],
        horizontal_spacing_lambda=config["horizontal_spacing_lambda"],
        vertical_spacing_lambda=config["vertical_spacing_lambda"],
        phase_floor_db=config["phase_floor_db"],
    )
    with np.load(FIXTURE_DIR / "inputs.npz") as data:
        aperture = data["aperture_cfr_bs"]
        offsets_hz = data["frequency_offsets_hz"]
        valid_mask = data["valid_mask"]
    return params, aperture, offsets_hz, valid_mask, meta


def _dominant_delay_bin_width(offsets_hz: np.ndarray) -> float:
    """Return the delay bin width ``1 / (N * delta_f)`` of the frequency grid."""
    offsets = np.asarray(offsets_hz, dtype=np.float64)
    delta_f = float((offsets[-1] - offsets[0]) / (offsets.size - 1))
    return 1.0 / (offsets.size * delta_f)


def test_domain_development_matches_golden_bytes():
    params, aperture, offsets_hz, valid_mask, _ = _load_fixture()
    developed = develop_hemisphere_image(aperture[0], params)
    center = center_frequency_products(
        developed.image, valid_mask, params.phase_floor_db, freq_bin=None
    )
    delays = delay_products(developed.image, valid_mask, offsets_hz)
    actuals = {
        "angular_cfr_center": center.center_cfr,
        "angular_power_center": center.center_power,
        "phase_valid_mask": center.phase_valid,
        "dominant_delay_s": delays.dominant_delay_s,
        "dominant_delay_power": delays.dominant_delay_power,
    }
    for name in OUTPUT_NAMES:
        golden = np.load(FIXTURE_DIR / f"{name}.npy")
        actual = actuals[name]
        assert actual.dtype == golden.dtype, name
        assert actual.shape == golden.shape, name
        assert actual.tobytes() == golden.tobytes(), name

    golden_delay = np.load(FIXTURE_DIR / "dominant_delay_s.npy")
    delay = actuals["dominant_delay_s"]
    assert np.array_equal(np.isnan(delay), np.isnan(golden_delay))
    assert np.isnan(delay).any()
    assert np.isfinite(delay).any()


def test_writer_matches_golden():
    pytest.importorskip("sionna.rt")
    spec = importlib.util.spec_from_file_location(
        "_rf_camera_make_golden", FIXTURE_DIR / "make_golden.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.check_golden(FIXTURE_DIR) == 0


def test_center_products_explicit_bin():
    params, aperture, _offsets, valid_mask, _ = _load_fixture()
    image = develop_hemisphere_image(aperture[0], params).image

    default = center_frequency_products(image, valid_mask, params.phase_floor_db)
    explicit = center_frequency_products(image, valid_mask, params.phase_floor_db, freq_bin=16 // 2)
    assert default.center_cfr.tobytes() == explicit.center_cfr.tobytes()
    assert default.center_power.tobytes() == explicit.center_power.tobytes()
    assert np.array_equal(default.phase_valid, explicit.phase_valid)

    first = center_frequency_products(image, valid_mask, params.phase_floor_db, freq_bin=0)
    assert np.array_equal(first.center_cfr, image[:, :, 0].astype(np.complex64))

    with pytest.raises(ValueError):
        center_frequency_products(image, valid_mask, params.phase_floor_db, freq_bin=-1)
    with pytest.raises(ValueError):
        center_frequency_products(image, valid_mask, params.phase_floor_db, freq_bin=16)


@pytest.mark.parametrize("hemisphere", [0, 1])
def test_raw_spectrum_energy_parseval(hemisphere):
    params, aperture, _offsets, _valid, _ = _load_fixture()
    hemi = aperture[hemisphere]
    expected = float(np.sum(np.abs(hemi.astype(np.complex128)) ** 2))
    energy = raw_spectrum_energy(hemi, params)
    assert abs(energy - expected) / expected <= 1e-6


def test_raw_spectrum_energy_respects_fft_size():
    params, aperture, _offsets, _valid, _ = _load_fixture()
    larger = DevelopParams(
        fft_rows=20,
        fft_cols=24,
        rx_rows=params.rx_rows,
        rx_cols=params.rx_cols,
        horizontal_spacing_lambda=params.horizontal_spacing_lambda,
        vertical_spacing_lambda=params.vertical_spacing_lambda,
        phase_floor_db=params.phase_floor_db,
    )
    hemi = aperture[0]
    expected = float(np.sum(np.abs(hemi.astype(np.complex128)) ** 2))
    assert abs(raw_spectrum_energy(hemi, larger) - expected) / expected <= 1e-6


def test_angle_delay_power_parseval():
    params, aperture, offsets_hz, valid_mask, _ = _load_fixture()
    num_bins = offsets_hz.size
    bin_width = _dominant_delay_bin_width(offsets_hz)

    for hemisphere in (0, 1):
        image = develop_hemisphere_image(aperture[hemisphere], params).image
        power, delay_s = angle_delay_power(image, offsets_hz)

        expected_mean = float(np.sum(np.abs(image) ** 2)) / num_bins
        assert abs(float(power.sum()) - expected_mean) / expected_mean <= 1e-6

        expected_per_pixel = (np.abs(image) ** 2).sum(-1) / num_bins
        atol = 1e-12 * float(np.max(np.abs(image) ** 2))
        assert np.allclose(power.sum(-1), expected_per_pixel, rtol=1e-6, atol=atol)

        assert delay_s.shape == (num_bins,)
        assert delay_s[0] == 0.0
        assert delay_s[1] == pytest.approx(bin_width, rel=1e-6)

        dominant = delay_products(image, valid_mask, offsets_hz)
        observed = valid_mask & (power.max(-1) > 0.0)
        assert np.array_equal(
            dominant.dominant_delay_power[observed],
            power.max(-1).astype(np.float32)[observed],
        )


def test_plane_waves_land_on_expected_pixels():
    params, aperture, offsets_hz, valid_mask, meta = _load_fixture()
    for hemisphere_index, hemisphere in enumerate(("front", "back")):
        developed = develop_hemisphere_image(aperture[hemisphere_index], params)
        image = developed.image
        power = np.sum(np.abs(image) ** 2, axis=-1)
        row, col = np.unravel_index(int(np.argmax(power)), power.shape)

        waves = [wave for wave in meta["plane_waves"] if wave["hemisphere"] == hemisphere]
        strongest = max(waves, key=lambda wave: abs(complex(*wave["amplitude"])))
        step_y = float(developed.ky_over_k[1] - developed.ky_over_k[0])
        step_z = float(developed.kz_over_k[1] - developed.kz_over_k[0])
        assert abs(developed.ky_over_k[col] - strongest["ky_over_k"]) <= step_y
        assert abs(developed.kz_over_k[row] - strongest["kz_over_k"]) <= step_z

        if hemisphere == "front":
            delays = delay_products(image, valid_mask, offsets_hz)
            bin_width = _dominant_delay_bin_width(offsets_hz)
            period = len(offsets_hz) * bin_width
            error = circular_delay_error_s(
                float(delays.dominant_delay_s[row, col]),
                float(strongest["delay_s"]),
                period,
            )
            assert error <= bin_width + 1e-15


def test_develop_validation():
    with pytest.raises(ValueError):
        DevelopParams(4, 32, 8, 8, 0.5, 0.5, -35.0)
    with pytest.raises(ValueError):
        DevelopParams(32, 32, 8, 8, 0.0, 0.5, -35.0)
    with pytest.raises(ValueError):
        DevelopParams(32, 32, 8, 8, 0.5, 0.5, float("nan"))

    params, aperture, offsets_hz, valid_mask, _ = _load_fixture()
    with pytest.raises(ValueError):
        develop_hemisphere_image(np.zeros((8, 8), dtype=np.complex128), params)
    with pytest.raises(ValueError):
        develop_hemisphere_image(np.zeros((7, 8, 4), dtype=np.complex128), params)
    with pytest.raises(ValueError):
        center_frequency_products(np.zeros((8, 8, 4), dtype=np.complex128), valid_mask, -35.0)
    with pytest.raises(ValueError):
        delay_products(np.zeros((8, 8, 4), dtype=np.complex128), valid_mask, offsets_hz)
    with pytest.raises(ValueError):
        raw_spectrum_energy(np.zeros((8, 8), dtype=np.complex128), params)


def test_develop_params_from_manifest_v3_v2(tmp_path: Path):
    for writer in (write_v3_dataset, write_v2_dataset):
        root = tmp_path / writer.__name__
        writer(root)
        dataset = load_rf_dataset_manifest(root / "dataset_manifest.json")
        params = develop_params_from_manifest(dataset)
        assert params == DevelopParams(16, 16, 2, 3, 0.5, 0.5, -35.0)

        aperture = dataset.load_aperture_cfr(dataset.views[0])
        developed = develop_hemisphere_image(aperture[0, 0], params)
        assert developed.image.shape == (16, 16, dataset.num_frequency_bins)


def _rewrite_manifest_config(root: Path, config: dict) -> Path:
    """Rewrite ``dataset_manifest.json`` under ``root`` with ``config``."""
    path = root / "dataset_manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["config"] = config
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_develop_params_from_manifest_missing_key(tmp_path: Path):
    keys = (
        "fft_rows",
        "fft_cols",
        "rx_rows",
        "rx_cols",
        "horizontal_spacing_lambda",
        "vertical_spacing_lambda",
        "phase_floor_db",
    )
    reader_required = {"rx_rows", "rx_cols"}
    for key in keys:
        root = tmp_path / f"missing_{key}"
        write_v3_dataset(root)
        path = root / "dataset_manifest.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        del data["config"][key]
        path.write_text(json.dumps(data), encoding="utf-8")

        if key in reader_required:
            with pytest.raises(ManifestError) as excinfo:
                load_rf_dataset_manifest(path)
            assert key in str(excinfo.value)
            continue

        dataset = load_rf_dataset_manifest(path)
        with pytest.raises(ManifestError) as excinfo:
            develop_params_from_manifest(dataset)
        assert key in str(excinfo.value)


def test_develop_params_from_manifest_bad_types(tmp_path: Path):
    root = tmp_path / "bad_types"
    write_v3_dataset(root)
    base = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))["config"]

    for key, value in (("fft_rows", "32"), ("phase_floor_db", True)):
        config = dict(base)
        config[key] = value
        path = _rewrite_manifest_config(root, config)
        dataset = load_rf_dataset_manifest(path)
        with pytest.raises(ManifestError) as excinfo:
            develop_params_from_manifest(dataset)
        assert key in str(excinfo.value)

    config = dict(base)
    config["fft_rows"] = 1
    path = _rewrite_manifest_config(root, config)
    dataset = load_rf_dataset_manifest(path)
    with pytest.raises(ManifestError):
        develop_params_from_manifest(dataset)
