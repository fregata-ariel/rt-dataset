"""Acceptance tests for the physically consistent viewer fixtures."""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import time
import zipfile
import zlib
from pathlib import Path

import numpy as np
import pytest
import viewer_fixtures
from rf_manifest_fixtures import write_v2_dataset, write_v3_dataset
from viewer_fixtures import (
    CARRIER_HZ,
    encode_png,
    write_rf_dataset,
)

from plateau_rt.application.rf_camera_develop import develop_params_from_manifest
from plateau_rt.application.rf_dataset_manifest import load_rf_dataset_manifest
from plateau_rt.domain.rf_camera.calibration import (
    calibrate_angular_cfr,
    geometric_los_source_direction_local,
    rotation_matrix,
)
from plateau_rt.domain.rf_camera.develop import (
    center_frequency_products,
    delay_products,
    develop_hemisphere_image,
)
from plateau_rt.domain.rf_camera.image_sources import virtual_source_positions
from plateau_rt.domain.rf_camera.imaging import aperture_to_angular_fft
from plateau_rt.domain.rf_camera.paths import (
    PATH_GT_MODE_CANONICAL,
    build_path_schema,
    quantise_tau,
    synthesize_cfr,
)
from plateau_rt.domain.rf_camera.solver_metrics import hemisphere_energy

REPO_ROOT = Path(__file__).resolve().parents[1]
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(scope="module")
def default_truth(tmp_path_factory: pytest.TempPathFactory):
    return write_rf_dataset(tmp_path_factory.mktemp("v3_default"))


@pytest.fixture(scope="module")
def beyond_truth(tmp_path_factory: pytest.TempPathFactory):
    return write_rf_dataset(tmp_path_factory.mktemp("v3_beyond"), beyond_period=True)


@pytest.fixture(scope="module")
def v2_truth(tmp_path_factory: pytest.TempPathFactory):
    return write_rf_dataset(tmp_path_factory.mktemp("v2_default"), schema_version=2)


def _camera_axes(dataset):
    with np.load(dataset.camera_model_path) as model:
        return model["ky_over_k"], model["kz_over_k"], model["valid_mask"]


def _all_referenced_files(dataset):
    paths = [dataset.camera_model_path]
    for view in dataset.views:
        paths.extend(view.artifacts.values())
        for entry in view.bs:
            paths.extend(entry.artifacts.values())
    if dataset.path_geometry_gt is not None:
        paths.append(dataset.path_geometry_gt.path)
        if dataset.path_geometry_gt.schema_path is not None:
            paths.append(dataset.path_geometry_gt.schema_path)
    return paths


def _relative_files(root: Path):
    return sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file())


def _total_size(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _decode_png(data: bytes) -> np.ndarray:
    assert data[:8] == PNG_SIGNATURE
    position = 8
    width = height = colour_type = 0
    idat = b""
    while position < len(data):
        length = struct.unpack(">I", data[position : position + 4])[0]
        tag = data[position + 4 : position + 8]
        chunk = data[position + 8 : position + 8 + length]
        position += 12 + length
        if tag == b"IHDR":
            width, height, _depth, colour_type = struct.unpack(">IIBB", chunk[:10])
        elif tag == b"IDAT":
            idat += chunk
        elif tag == b"IEND":
            break
    raw = zlib.decompress(idat)
    channels = 1 if colour_type == 0 else 4
    stride = width * channels
    shape = (height, width) if channels == 1 else (height, width, 4)
    out = np.zeros(shape, dtype=np.uint8)
    for row in range(height):
        line = raw[row * (stride + 1) : (row + 1) * (stride + 1)]
        assert line[0] == 0
        out[row] = np.frombuffer(line[1:], dtype=np.uint8).reshape(
            (width,) if channels == 1 else (width, 4)
        )
    return out


# 1. Loadable + files exist.
def test_dataset_files_and_manifest(default_truth, beyond_truth, v2_truth):
    for truth in (default_truth, beyond_truth, v2_truth):
        dataset = load_rf_dataset_manifest(truth.root)
        assert dataset.schema_version == truth.schema_version
        assert dataset.num_views == len(truth.view_ids) == 3
        expected_bs = 1 if truth.schema_version == 2 else 2
        assert dataset.num_bs == expected_bs
        assert dataset.view_ids == truth.view_ids
        assert dataset.bs_ids == truth.bs_ids

        for path in _all_referenced_files(dataset):
            assert path.exists(), path
        for view in dataset.views:
            loaded = dataset.load_aperture_cfr(view)
            assert loaded.shape == dataset.aperture_cfr_shape
            for entry in view.bs:
                png = entry.artifacts["debug_power_png"].read_bytes()
                assert png.startswith(PNG_SIGNATURE)
                width, height = struct.unpack(">II", png[16:24])
                assert width == height == 32
        if truth.schema_version == 3:
            assert dataset.path_geometry_gt is not None
            assert dataset.path_geometry_gt.schema_path is not None


def test_no_path_gt_omits_artifacts(tmp_path: Path):
    truth = write_rf_dataset(tmp_path, path_gt=False)
    dataset = load_rf_dataset_manifest(truth.root)
    assert dataset.path_geometry_gt is None
    assert not (tmp_path / "path_geometry_gt.npz").exists()
    assert not (tmp_path / "path_schema.json").exists()
    assert "path_geometry_gt" not in dataset.raw
    assert "path_schema" not in dataset.raw


def test_v2_forces_single_bs(tmp_path: Path):
    truth = write_rf_dataset(tmp_path, num_bs=2, schema_version=2)
    dataset = load_rf_dataset_manifest(truth.root)
    assert dataset.num_bs == 1
    assert dataset.bs_ids == ("bs_000",)
    assert truth.bs_ids == ("bs_000",)


# 2. Front peak.
@pytest.mark.parametrize("variant", ["default", "v2"])
def test_front_hemisphere_peak(default_truth, v2_truth, variant):
    truth = default_truth if variant == "default" else v2_truth
    dataset = load_rf_dataset_manifest(truth.root)
    ky, kz, valid_mask = _camera_axes(dataset)
    checked = 0
    for pair in truth.pairs:
        entry = dataset.views[pair.view_index].bs[pair.bs_index]
        if not entry.bs_in_front_hemisphere or not pair.has_los:
            continue
        checked += 1
        power = np.load(entry.artifacts["angular_power_center"])
        peak = np.unravel_index(int(np.argmax(np.where(valid_mask, power, -np.inf))), power.shape)
        direction = entry.bs_direction_local
        expected = viewer_fixtures.nearest_pixel(direction[1], direction[2], ky, kz)
        assert expected == pair.expected_peak_pixel
        assert viewer_fixtures.chebyshev_distance((int(peak[0]), int(peak[1])), expected) <= 1
    assert checked >= 1
    if variant == "default":
        assert checked == 4


# 3. Back hemisphere.
def test_back_hemisphere_peak(default_truth):
    truth = default_truth
    dataset = load_rf_dataset_manifest(truth.root)
    ky, kz, valid_mask = _camera_axes(dataset)
    params = develop_params_from_manifest(dataset)
    back_pairs = set()
    for pair in truth.pairs:
        entry = dataset.views[pair.view_index].bs[pair.bs_index]
        if not pair.bs_hemisphere == "back":
            continue
        back_pairs.add((pair.view_index, pair.bs_index))
        assert entry.hemisphere_energy["back"] > entry.hemisphere_energy["front"]
        aperture = dataset.load_aperture_cfr(dataset.views[pair.view_index])
        image = develop_hemisphere_image(aperture[pair.bs_index, 1], params).image[
            :, :, dataset.num_frequency_bins // 2
        ]
        peak = np.unravel_index(
            int(np.argmax(np.where(valid_mask, np.abs(image) ** 2, -np.inf))), image.shape
        )
        expected = viewer_fixtures.nearest_pixel(
            entry.bs_direction_local[1], entry.bs_direction_local[2], ky, kz
        )
        assert viewer_fixtures.chebyshev_distance((int(peak[0]), int(peak[1])), expected) <= 1
    assert len(back_pairs) >= 1
    assert back_pairs == {(0, 0), (1, 1)}


# 4. Resynthesis.
@pytest.mark.parametrize("variant", ["default", "beyond", "v2"])
def test_path_gt_resynthesis(default_truth, beyond_truth, v2_truth, variant):
    truth = {"default": default_truth, "beyond": beyond_truth, "v2": v2_truth}[variant]
    dataset = load_rf_dataset_manifest(truth.root)
    arrays = dataset.path_geometry_gt.load_arrays()
    a_baseband = arrays["a_baseband"]
    tau = arrays["tau"]
    for view_index, view in enumerate(dataset.views):
        stored = dataset.load_aperture_cfr(view)
        for bs_index in range(dataset.num_bs):
            synthesized = synthesize_cfr(
                a_baseband[view_index, bs_index],
                tau[view_index, bs_index],
                dataset.frequency_offsets_hz,
            )
            reference = stored[bs_index]
            peak = float(np.max(np.abs(reference)))
            if peak == 0.0:
                assert float(np.max(np.abs(synthesized))) <= 1e-12
                continue
            error = float(np.max(np.abs(synthesized - reference)) / peak)
            assert error <= 1e-6

    script = REPO_ROOT / "scripts" / "ci" / "check_path_gt_resynthesis.py"
    result = subprocess.run(
        [sys.executable, str(script), str(truth.root)],
        capture_output=True,
        text=True,
        env=os.environ,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "✅" in result.stdout


# 5. Supplied view poses.
def test_supplied_view_poses(tmp_path: Path):
    first = ((12.0, -7.0, 1.5), rotation_matrix((0.4, -0.1, 0.05)))
    second = ((-9.0, 14.0, 2.0), rotation_matrix((-2.0, 0.2, 0.0)))
    truth = write_rf_dataset(tmp_path, view_poses=[first, second])
    dataset = load_rf_dataset_manifest(truth.root)
    for index, (position, rotation) in enumerate((first, second)):
        view_id = f"ue_{index:06d}"
        payload = json.loads((tmp_path / "views" / view_id / "pose.json").read_text())
        assert np.allclose(payload["position_m"], position, atol=1e-12)
        assert np.allclose(payload["world_from_local_rotation"], rotation, atol=1e-12)
        assert np.allclose(dataset.views[index].position_m, position, atol=1e-12)

    with pytest.raises(ValueError):
        write_rf_dataset(tmp_path / "bad", view_poses=[(first[0], 2.0 * np.eye(3))])


# 6. Byte determinism.
def test_byte_determinism(tmp_path: Path):
    for kwargs in ({}, {"beyond_period": True}):
        first = write_rf_dataset(tmp_path / "a", **kwargs)
        second = write_rf_dataset(tmp_path / "b", **kwargs)
        assert _relative_files(first.root) == _relative_files(second.root)
        for relative in _relative_files(first.root):
            assert (first.root / relative).read_bytes() == (second.root / relative).read_bytes()
        for path in first.root.rglob("*.npz"):
            with zipfile.ZipFile(path) as archive:
                assert all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in archive.infolist())


# 7. Size and speed.
def test_size_and_speed(default_truth, beyond_truth, tmp_path: Path):
    assert _total_size(default_truth.root) <= 3 * 1024 * 1024
    assert _total_size(beyond_truth.root) <= 3 * 1024 * 1024
    start = time.perf_counter()
    write_rf_dataset(tmp_path)
    assert time.perf_counter() - start < 1.5


# 8. Sionna-free imports.
def test_imports_are_sionna_free():
    tests_dir = str(Path(__file__).resolve().parent)
    code = (
        "import sys; sys.path.insert(0, {tests!r}); import tempfile; "
        "import viewer_fixtures as vf; vf.write_rf_dataset(tempfile.mkdtemp()); "
        "print(','.join(m for m in ('sionna','mitsuba','drjit','matplotlib') "
        "if m in sys.modules))"
    ).format(tests=tests_dir)
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=os.environ
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""


# 9. Sign convention of the single-path aperture.
def test_single_path_sign_convention(default_truth):
    truth = default_truth
    dataset = load_rf_dataset_manifest(truth.root)
    arrays = dataset.path_geometry_gt.load_arrays()
    a_baseband = arrays["a_baseband"]
    ky, kz = truth.ky_over_k, truth.kz_over_k
    for pair in truth.pairs:
        for path_index, path in enumerate(pair.paths):
            hemisphere = 0 if path.direction_local[0] >= 0.0 else 1
            assert path.hemisphere == ("front" if hemisphere == 0 else "back")
            other = a_baseband[pair.view_index, pair.bs_index, 1 - hemisphere, :, :, path_index]
            assert np.all(other == 0)
            single = a_baseband[pair.view_index, pair.bs_index, hemisphere, :, :, path_index][
                :, :, None
            ].astype(np.complex64)
            calibration = calibrate_angular_cfr(
                aperture_to_angular_fft(single, fft_rows=32, fft_cols=32),
                aperture_rows=8,
                aperture_cols=8,
                horizontal_spacing_lambda=0.5,
                vertical_spacing_lambda=0.5,
            )
            peak = np.unravel_index(
                int(np.argmax(np.abs(calibration.cfr[:, :, 0]))),
                calibration.cfr[:, :, 0].shape,
            )
            assert peak == viewer_fixtures.nearest_pixel(path.ky, path.kz, ky, kz)


# 10. Geometry and virtual sources.
def test_path_geometry(default_truth):
    truth = default_truth
    dataset = load_rf_dataset_manifest(truth.root)
    arrays = dataset.path_geometry_gt.load_arrays()
    tau = arrays["tau"]
    theta_r = arrays["theta_r"]
    phi_r = arrays["phi_r"]
    object_names = dataset.path_geometry_gt.load_schema()["object_names"]
    ground_index = object_names.index("ground")

    for pair in truth.pairs:
        ue = np.asarray(dataset.views[pair.view_index].position_m, dtype=np.float64)
        bs = np.asarray(dataset.base_stations[pair.bs_index].position_m, dtype=np.float64)
        mirrored = np.array([bs[0], bs[1], -bs[2]], dtype=np.float64)
        for path_index, path in enumerate(pair.paths):
            u = [pair.view_index, pair.bs_index, path_index]
            if path.kind == "los":
                expected = np.linalg.norm(bs - ue) / viewer_fixtures.SPEED_OF_LIGHT_M_S
                assert abs(float(tau[tuple(u)]) - expected) <= 1e-6 * expected
                assert int(arrays["num_interactions"][tuple(u)]) == 0
                assert np.all(arrays["interactions"][tuple(u)] == 0)
                source = bs
            else:
                expected = np.linalg.norm(mirrored - ue) / viewer_fixtures.SPEED_OF_LIGHT_M_S
                assert abs(float(tau[tuple(u)]) - expected) <= 1e-6 * expected
                assert int(arrays["interactions"][tuple(u)][0]) == 1
                vertex = arrays["vertices"][tuple(u)][0].astype(np.float64)
                assert vertex[2] == 0.0
                line = mirrored - ue
                offset = vertex - ue
                distance = np.linalg.norm(offset - np.dot(offset, line) / np.dot(line, line) * line)
                assert distance < 1e-3
                assert int(arrays["object_index"][tuple(u)][0]) == ground_index
                source = mirrored
            # Free-space amplitude times the reflection factor on every element.
            factor = 1.0 if path.kind == "los" else viewer_fixtures.GROUND_REFLECTION
            wavelength = viewer_fixtures.SPEED_OF_LIGHT_M_S / CARRIER_HZ
            length = np.linalg.norm(source - ue)
            hemisphere = 0 if path.hemisphere == "front" else 1
            coefficient = arrays["a_baseband"][
                pair.view_index, pair.bs_index, hemisphere, :, :, path_index
            ]
            np.testing.assert_allclose(
                np.abs(coefficient), abs(factor) * wavelength / (4.0 * np.pi * length), rtol=1e-5
            )
            virtual = virtual_source_positions(
                ue,
                float(tau[tuple(u)]),
                float(theta_r[tuple(u)]),
                float(phi_r[tuple(u)]),
            )
            assert np.linalg.norm(np.asarray(virtual) - source) < 1e-2


# 11. Canonical order and schema.
def test_canonical_order_and_schema(default_truth):
    truth = default_truth
    dataset = load_rf_dataset_manifest(truth.root)
    arrays = dataset.path_geometry_gt.load_arrays()
    valid = arrays["valid"]
    tau = arrays["tau"]
    assert truth.num_paths == tau.shape[-1]
    for view_index in range(dataset.num_views):
        for bs_index in range(dataset.num_bs):
            mask = valid[view_index, bs_index]
            count = int(np.count_nonzero(mask))
            assert np.all(mask[:count]) and not np.any(mask[count:])
            if count > 1:
                assert np.all(np.diff(tau[view_index, bs_index, :count]) >= 0)
                assert np.all(np.diff(quantise_tau(tau[view_index, bs_index, :count])) >= 0)
            pair = truth.pair(view_index, bs_index)
            assert len(pair.paths) == count
            for path_index, path in enumerate(pair.paths):
                assert path.tau_s == float(tau[view_index, bs_index, path_index])

    object_names = dataset.path_geometry_gt.load_schema()["object_names"]
    expected = build_path_schema(
        arrays,
        mode=PATH_GT_MODE_CANONICAL,
        object_names=object_names,
        carrier_frequency_hz=CARRIER_HZ,
        bs_ids=dataset.bs_ids,
        view_ids=dataset.view_ids,
    )
    on_disk = json.loads(dataset.path_geometry_gt.schema_path.read_text())
    assert on_disk == expected


# 12. Beyond-period far wall.
def test_beyond_period(default_truth, beyond_truth):
    default_dataset = load_rf_dataset_manifest(default_truth.root)
    default_arrays = default_dataset.path_geometry_gt.load_arrays()
    assert not np.any(
        default_arrays["valid"] & (default_arrays["tau"] >= default_truth.unambiguous_delay_s)
    )

    dataset = load_rf_dataset_manifest(beyond_truth.root)
    arrays = dataset.path_geometry_gt.load_arrays()
    valid = arrays["valid"]
    tau = arrays["tau"]
    assert beyond_truth.far_wall_x_m is not None
    found = 0
    for pair in beyond_truth.pairs:
        stored_indices = tuple(
            int(path_index)
            for path_index in np.nonzero(
                valid[pair.view_index, pair.bs_index]
                & (tau[pair.view_index, pair.bs_index] >= dataset.unambiguous_delay_s)
            )[0]
        )
        assert stored_indices == pair.beyond_period_path_indices
        if stored_indices:
            found += 1
            for path_index in stored_indices:
                assert pair.paths[path_index].kind == "far_wall"
                vertex = arrays["vertices"][pair.view_index, pair.bs_index, path_index][0]
                assert vertex[0] == beyond_truth.far_wall_x_m
    assert found >= 1


# 13. Manifest values like the writer.
def test_manifest_values_match_writer(default_truth):
    truth = default_truth
    dataset = load_rf_dataset_manifest(truth.root)
    _, _, valid_mask = _camera_axes(dataset)
    params = develop_params_from_manifest(dataset)
    for view_index, view in enumerate(dataset.views):
        aperture = dataset.load_aperture_cfr(view)
        for bs_index, entry in enumerate(view.bs):
            assert entry.hemisphere_energy == pytest.approx(
                hemisphere_energy(aperture[bs_index]), rel=1e-6
            )
            expected_direction = geometric_los_source_direction_local(
                tx_position=dataset.base_stations[bs_index].position_m,
                ue_position=view.position_m,
                ue_orientation=view.orientation_rad,
            )
            assert np.allclose(entry.bs_direction_local, expected_direction, atol=1e-12)
            assert entry.bs_in_front_hemisphere == (entry.bs_direction_local[0] >= 0.0)

            developed = develop_hemisphere_image(aperture[bs_index, 0], params)
            center = center_frequency_products(
                developed.image,
                valid_mask,
                params.phase_floor_db,
                freq_bin=dataset.num_frequency_bins // 2,
            )
            delays = delay_products(developed.image, valid_mask, dataset.frequency_offsets_hz)
            np.testing.assert_array_equal(
                np.load(entry.artifacts["angular_cfr_center"]), center.center_cfr
            )
            np.testing.assert_array_equal(
                np.load(entry.artifacts["angular_power_center"]), center.center_power
            )
            np.testing.assert_array_equal(
                np.load(entry.artifacts["phase_valid_mask"]), center.phase_valid
            )
            np.testing.assert_array_equal(
                np.load(entry.artifacts["dominant_delay_s"]), delays.dominant_delay_s
            )
            np.testing.assert_array_equal(
                np.load(entry.artifacts["dominant_delay_power"]),
                delays.dominant_delay_power,
            )


# 14. Occlusion by the box.
def test_occlusion(tmp_path: Path):
    truth = write_rf_dataset(
        tmp_path,
        num_views=1,
        num_bs=1,
        view_poses=[((-20.0, 0.0, 1.5), rotation_matrix((0.0, 0.0, 0.0)))],
    )
    pair = truth.pair(0, 0)
    assert pair.has_los is False
    assert pair.paths == ()
    assert pair.expected_peak_pixel is None
    dataset = load_rf_dataset_manifest(truth.root)
    arrays = dataset.path_geometry_gt.load_arrays()
    assert not np.any(arrays["valid"])
    aperture = dataset.load_aperture_cfr(dataset.views[0])
    assert np.count_nonzero(aperture) == 0


# 15. Validation errors.
@pytest.mark.parametrize(
    "kwargs",
    [
        {"schema_version": 4},
        {"num_views": 0},
        {"rows": 8, "fft": 4},
    ],
)
def test_validation_errors(tmp_path: Path, kwargs):
    with pytest.raises(ValueError):
        write_rf_dataset(tmp_path, **kwargs)


# 16. PNG round trip.
def test_encode_png_round_trip():
    grey = np.arange(20, dtype=np.uint8).reshape(4, 5)
    encoded = encode_png(grey)
    assert encoded.startswith(PNG_SIGNATURE)
    assert np.array_equal(_decode_png(encoded), grey)

    rgba = np.arange(3 * 3 * 4, dtype=np.uint8).reshape(3, 3, 4)
    assert np.array_equal(_decode_png(encode_png(rgba)), rgba)
    with pytest.raises(ValueError):
        encode_png(np.zeros((2, 2, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        encode_png(np.zeros((2, 2), dtype=np.float32))


# 17. rf_manifest_fixtures back-compat.
def test_rf_manifest_fixtures_back_compat(tmp_path: Path):
    v3_root = tmp_path / "v3"
    v2_root = tmp_path / "v2"
    write_v3_dataset(v3_root)
    write_v2_dataset(v2_root)
    # Placeholder mode writes the .npy artifacts only (no debug PNG, no stray files).
    assert not list(tmp_path.rglob("*.png*"))
    assert load_rf_dataset_manifest(v3_root).num_bs == 2
    assert load_rf_dataset_manifest(v2_root).num_bs == 1
    for root in (v3_root, v2_root):
        for name in ("camera_model.npz", "path_geometry_gt.npz"):
            with zipfile.ZipFile(root / name) as archive:
                assert all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in archive.infolist())
