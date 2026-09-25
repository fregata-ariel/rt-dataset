"""Unit tests for the tomography resynthesis validation module (T21, design §8)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
from rf_tomography_gt_fixtures import (
    OBJECT_NAMES,
    build_mirror_scene,
    v_pol_factor,
    write_mirror_dataset,
)

from plateau_rt.application import rf_tomography_gt as app
from plateau_rt.application.rf_tomography_resynthesis import (
    direct_path_summary,
    operator_amplitudes,
    polarization_factor,
    resynthesis_report,
    unmodelled_cfr,
)
from plateau_rt.domain.rf_camera.paths import synthesize_cfr
from plateau_rt.domain.rf_tomography.antenna import bs_pattern
from plateau_rt.domain.rf_tomography.geometry import CaptureGeometry
from plateau_rt.domain.rf_tomography.gt import PATH_TYPE_NAMES, PathGT, path_ground_truth

FIXTURES = Path(__file__).parent / "fixtures" / "rf_tomography"


def _path(scene) -> PathGT:
    """Return the scene's :class:`PathGT`."""
    return PathGT.from_arrays(scene.arrays, scene.object_names)


def _match(vs_pos: np.ndarray, vs_bs: np.ndarray, bs: int, pos: np.ndarray) -> int:
    """Return the VS index on ``bs`` nearest to ``pos``."""
    candidates = np.nonzero(vs_bs == bs)[0]
    distances = np.linalg.norm(vs_pos[candidates] - pos, axis=1)
    return int(candidates[int(np.argmin(distances))])


def _load_check_module():
    """Load ``scripts/ci/check_tomography_resynthesis.py`` as a module."""
    script = (
        Path(__file__).resolve().parent.parent
        / "scripts"
        / "ci"
        / "check_tomography_resynthesis.py"
    )
    spec = importlib.util.spec_from_file_location("check_tomography_resynthesis", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sionna_los() -> tuple[PathGT, CaptureGeometry]:
    """Build the real Sionna LoS fixture as ``(PathGT, CaptureGeometry)``."""
    fixture = np.load(FIXTURES / "sionna_mock_los.npz")
    geom = CaptureGeometry.from_orientations(
        ue_pos=fixture["ue_pos"],
        ue_orientations=fixture["ue_orientation"],
        bs_pos=fixture["bs_pos"],
        f_c=float(fixture["f_c"]),
        bandwidth=100e6,
        num_bins=64,
        bs_look_at=fixture["bs_look_at"],
    )
    departure = fixture["ue_pos"] - fixture["bs_pos"]
    departure = departure / np.linalg.norm(departure, axis=1, keepdims=True)
    arrival = -departure
    num_views = fixture["ue_pos"].shape[0]
    a_baseband = np.zeros((num_views, 1, 2, 8, 8, 1), dtype=np.complex128)
    for v in range(num_views):
        a_baseband[v, 0, :, :, :, 0] = fixture["aperture_cfr"][v, :, :, :, 8]
    arrays = {
        "valid": np.ones((num_views, 1, 1), dtype=bool),
        "tau": fixture["path_tau"].reshape(num_views, 1, 1),
        "a_baseband": a_baseband,
        "interactions": np.zeros((num_views, 1, 1, 1), dtype=np.uint32),
        "object_index": np.full((num_views, 1, 1, 1), -1, dtype=np.int32),
        "vertices": np.zeros((num_views, 1, 1, 1, 3)),
        "theta_t": np.arccos(departure[:, 2]).reshape(num_views, 1, 1),
        "phi_t": np.arctan2(departure[:, 1], departure[:, 0]).reshape(num_views, 1, 1),
        "theta_r": np.arccos(arrival[:, 2]).reshape(num_views, 1, 1),
        "phi_r": np.arctan2(arrival[:, 1], arrival[:, 0]).reshape(num_views, 1, 1),
    }
    return PathGT.from_arrays(arrays, ("ground_plane",)), geom


def _flip_rows(root: Path, view_index: int = 3) -> str:
    """Reverse the element rows of one view's stored aperture; return its view id."""
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    view_id = str(manifest["views"][view_index]["view_id"])
    path = root / "views" / view_id / "rf" / "aperture_cfr.npy"
    aperture = np.load(path)
    np.save(path, aperture[:, :, ::-1, :, :])
    return view_id


def test_operator_amplitudes_mirror_scene() -> None:
    scene = build_mirror_scene()
    path = _path(scene)
    geom = scene.geom
    arrays = path_ground_truth(path, geom, pattern="tr38901")
    amps = operator_amplitudes(
        path, geom, arrays["path_vs"], arrays["vs_pos"], arrays["vs_bs"], pattern="tr38901"
    )
    vs_order = np.asarray(arrays["vs_order"])
    vs_pos = np.asarray(arrays["vs_pos"], dtype=np.float64)
    vs_bs = np.asarray(arrays["vs_bs"], dtype=np.int64)
    path_vs = np.asarray(arrays["path_vs"])

    for v in range(path.num_views):
        for b in range(path.num_bs):
            for p in range(path.num_paths):
                if not path.valid[v, b, p]:
                    continue
                m = int(path_vs[v, b, p])
                if m < 0 or int(vs_order[m]) > 1:
                    continue
                assert abs(amps.gain_ratio_db[v, b, p]) < 1e-9

    order_two = next(expected for expected in scene.vs if expected.order == 2)
    m_two = _match(vs_pos, vs_bs, order_two.bs, order_two.pos)
    assert int(vs_order[m_two]) == 2
    s = vs_pos[m_two]
    bs = geom.bs_pos[order_two.bs]
    normal = (s - bs) / np.linalg.norm(s - bs)
    max_ratio = 0.0
    for v in order_two.beta:
        p = int(np.nonzero(path_vs[v, order_two.bs] == m_two)[0][0])
        point = geom.ue_pos[v]
        d = (point - s) / np.linalg.norm(point - s)
        d_hh = d - 2.0 * float(d @ normal) * normal
        vertex = np.asarray(scene.arrays["vertices"])[v, order_two.bs, p, 0]
        d_true = (vertex - bs) / np.linalg.norm(vertex - bs)
        ratio = (
            bs_pattern(d_true[None], geom.bs_rot[order_two.bs], kind="tr38901")[0]
            / (bs_pattern(d_hh[None], geom.bs_rot[order_two.bs], kind="tr38901")[0])
        )
        expected_db = 20.0 * np.log10(abs(ratio))
        assert abs(amps.gain_ratio_db[v, order_two.bs, p] - expected_db) < 1e-9
        max_ratio = max(max_ratio, abs(amps.gain_ratio_db[v, order_two.bs, p]))
    assert max_ratio > 0.01

    for expected in scene.vs:
        if expected.order > 1:
            continue
        m = _match(vs_pos, vs_bs, expected.bs, expected.pos)
        assert np.linalg.norm(vs_pos[m] - expected.pos) < 1e-6
        for v, beta in expected.beta.items():
            assert abs(amps.amps[m, v, expected.bs] - beta) <= 1e-9 * abs(beta)

    copied = {name: np.array(value, copy=True) for name, value in scene.arrays.items()}
    source, target = (0, 1, 1), (0, 1, 5)
    for name in ("valid", "tau", "theta_t", "phi_t", "theta_r", "phi_r"):
        copied[name][target] = scene.arrays[name][source]
    copied["a_baseband"][0, 1, :, :, :, 5] = scene.arrays["a_baseband"][0, 1, :, :, :, 1]
    for name in ("interactions", "object_index", "primitives", "vertices"):
        copied[name][0, 1, 5] = scene.arrays[name][0, 1, 1]
    copied["num_interactions"][0, 1, 5] = scene.arrays["num_interactions"][0, 1, 1]
    duplicate_path = PathGT.from_arrays(copied, OBJECT_NAMES)
    duplicate_arrays = path_ground_truth(duplicate_path, geom, pattern="tr38901")
    duplicate_vs = np.asarray(duplicate_arrays["path_vs"])
    duplicate_amps = operator_amplitudes(
        duplicate_path,
        geom,
        duplicate_arrays["path_vs"],
        duplicate_arrays["vs_pos"],
        duplicate_arrays["vs_bs"],
        pattern="tr38901",
    )
    m_dup = int(duplicate_vs[0, 1, 1])
    ground = next(vs for vs in scene.vs if vs.bs == 1 and vs.planes == ("ground",))
    beta = ground.beta[0]
    assert int(duplicate_amps.num_members[m_dup, 0]) == 2
    assert abs(duplicate_amps.amps[m_dup, 0, 1] - 2.0 * beta) <= 1e-9 * abs(2.0 * beta)


def test_unmodelled_cfr() -> None:
    scene = build_mirror_scene()
    path = _path(scene)
    freq = scene.geom.freq_offsets
    result = unmodelled_cfr(path, freq)
    assert result.shape == (scene.geom.num_views, scene.geom.num_bs, 2, 4, 4, freq.size)
    for v in range(path.num_views):
        for b in range(path.num_bs):
            if (v, b) in ((0, 0), (2, 0)):
                continue
            assert bool(np.all(result[v, b] == 0.0))

    for v in (0, 2):
        p = 5
        baseband = scene.arrays["a_baseband"][v, 0, :, :, :, p][..., None]
        tau = np.array([scene.arrays["tau"][v, 0, p]])
        expected = synthesize_cfr(baseband, tau, freq)
        np.testing.assert_allclose(result[v, 0], expected, rtol=1e-12, atol=1e-15)


def test_l0f_float64_is_exact(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    scene = write_mirror_dataset(root, float32=False)
    report = resynthesis_report(root)
    assert report["ok"] is True
    assert report["failures"] == []
    assert report["gt_source"] == "built"
    assert report["l0f"]["nmse"]["max"] < 1e-18
    assert report["l0f"]["nmse_aligned"]["max"] < 1e-18
    assert report["l0f"]["physical_nmse_aligned"]["max"] > 1e-6
    assert report["controls"]["row_flip_nmse_aligned"]["p50"] > 0.1
    assert report["direct_path"]["pattern_db"]["max"] < 1e-9
    assert report["direct_path"]["num_los"] == 11

    offsets = np.asarray(scene.geom.freq_offsets, dtype=np.float64)
    diffuse = synthesize_cfr(
        scene.arrays["a_baseband"][0, 0, :, :, :, 5][..., None],
        np.array([scene.arrays["tau"][0, 0, 5]]),
        offsets,
    )
    total = synthesize_cfr(
        scene.arrays["a_baseband"][0, 0],
        scene.arrays["tau"][0, 0][None, None, None, :],
        offsets,
    )
    expected_fraction = float(np.sum(np.abs(diffuse) ** 2) / np.sum(np.abs(total) ** 2))
    actual = float(report["l0f"]["unmodelled_fraction"]["max"])
    assert abs(actual - expected_fraction) <= 0.01 * expected_fraction


def test_l0f_float32_passes(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    write_mirror_dataset(root, float32=True)
    report = resynthesis_report(root)
    assert report["ok"] is True
    assert report["l0f"]["nmse"]["max"] < 1e-6


def test_departure_gate_catches_offset_los_source(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    write_mirror_dataset(root, float32=True)
    report = resynthesis_report(root)
    for order in ("0", "1"):
        assert report["l0f"]["gain_ratio_db_by_order"][order]["max"] < 1e-3

    arrays = app.build_tomography_gt(root, surfaces=False)
    order0 = np.asarray(arrays["vs_order"]) == 0
    arrays["vs_pos"] = np.array(arrays["vs_pos"], copy=True)
    arrays["vs_pos"][order0] += np.array([0.0, 0.0, 1e-5])
    shifted = tmp_path / "shifted_gt.npz"
    np.savez_compressed(shifted, **arrays)
    bad = resynthesis_report(root, gt_file=shifted)
    assert any("order-0 departure" in failure for failure in bad["failures"])
    assert bad["l0f"]["nmse"]["max"] < 1e-6  # the re-referenced amplitudes hide it from the NMSE


def test_l0f_detects_row_flip(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    write_mirror_dataset(root, float32=True)
    view_id = _flip_rows(root, 3)
    assert view_id == "ue_000003"
    report = resynthesis_report(root)
    assert report["ok"] is False
    assert any(failure.startswith("l0f:") and view_id in failure for failure in report["failures"])


def test_l0f_detects_swapped_bs(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    write_mirror_dataset(root, float32=True)
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    view_id = str(manifest["views"][1]["view_id"])
    path = root / "views" / view_id / "rf" / "aperture_cfr.npy"
    aperture = np.load(path)
    aperture[[0, 1]] = aperture[[1, 0]]
    np.save(path, aperture)
    report = resynthesis_report(root)
    assert report["ok"] is False
    assert any(failure.startswith("l0f:") for failure in report["failures"])


def test_direct_path_needs_vpol_data(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    write_mirror_dataset(root, los_polarization=False)
    report = resynthesis_report(root)
    assert report["ok"] is False
    assert any(failure.startswith("direct path:") for failure in report["failures"])
    assert not any(failure.startswith("l0f:") for failure in report["failures"])


def test_direct_path_wrong_pattern(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    write_mirror_dataset(root, float32=True)
    manifest_path = root / "dataset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["config"]["tx_pattern"] = "iso"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report = resynthesis_report(root)
    assert any(failure.startswith("direct path:") for failure in report["failures"])
    assert report["direct_path"]["iso_control_db"] is None


def test_direct_path_sionna_fixture() -> None:
    path, geom = _sionna_los()
    summary = direct_path_summary(path, geom, pattern="tr38901")
    assert summary["pattern_db"]["max"] < 0.01
    assert 0.01 < summary["scalar_db"]["max"] < 0.5
    assert summary["eps_los_deg_none"]["max"] < 0.1
    assert summary["iso_control_db"]["max"] > 0.5
    assert summary["rel_error_vv"]["max"] < summary["rel_error_none"]["max"]


def test_polarization_summary(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    scene = write_mirror_dataset(root, float32=False, los_polarization=True)
    report = resynthesis_report(root)
    polarization = report["polarization"]
    assert polarization["los"]["n"] == 11

    injected = []
    for v in range(scene.geom.num_views):
        for b in range(scene.geom.num_bs):
            if not scene.los_visible[v, b]:
                continue
            d_dep = scene.geom.ue_pos[v] - scene.geom.bs_pos[b]
            d_dep = d_dep / np.linalg.norm(d_dep)
            injected.append(v_pol_factor(scene.geom, v, b, d_dep, -d_dep))
    assert abs(polarization["los"]["min"] - min(injected)) < 1e-9
    assert polarization["los"]["negative"] == 0

    arrays = path_ground_truth(_path(scene), scene.geom, pattern="tr38901")
    order = np.asarray(arrays["vs_order"])
    visibility = np.asarray(arrays["vs_visibility"], dtype=bool)
    vs_type = np.asarray(arrays["vs_path_type"])
    specular = PATH_TYPE_NAMES.index("specular")
    expected_pairs = sum(
        1
        for m in range(order.shape[0])
        if int(order[m]) == 1
        for v in range(visibility.shape[1])
        if visibility[m, v] and int(vs_type[m, v]) == specular
    )
    assert polarization["first_order"]["n"] == expected_pairs

    d_dep = scene.geom.ue_pos[0] - scene.geom.bs_pos[0]
    d_dep = d_dep / np.linalg.norm(d_dep)
    value = polarization_factor(scene.geom.bs_pos[0][None], scene.geom, 0, 0, pattern="tr38901")[0]
    assert abs(float(value) - v_pol_factor(scene.geom, 0, 0, d_dep, -d_dep)) < 1e-12


def test_registered_and_stale_gt(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    write_mirror_dataset(root, float32=True)
    npz = app.write_tomography_gt(root, surfaces=False)
    registered = resynthesis_report(root)
    assert registered["gt_source"] == "registered"
    assert registered["ok"] is True
    from_file = resynthesis_report(root, gt_file=npz)
    assert from_file["gt_source"] == "file"

    with np.load(root / "path_geometry_gt.npz", allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    arrays["a_baseband"][0, 0, :, :, :, 0] = arrays["a_baseband"][0, 0, :, :, :, 0] * 1.001
    np.savez_compressed(root / "path_geometry_gt.npz", **arrays)
    stale = resynthesis_report(root)
    assert any(failure.startswith("stale tomography GT") for failure in stale["failures"])


def test_check_script(tmp_path: Path, capsys) -> None:
    check = _load_check_module()
    root = tmp_path / "dataset"
    write_mirror_dataset(root, float32=True)
    report_path = tmp_path / "r" / "report.json"
    assert check.main([str(root), "--report", str(report_path)]) == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["schema"] == "rf_tomo_resynthesis/1"
    capsys.readouterr()

    flipped = tmp_path / "flipped"
    write_mirror_dataset(flipped, float32=True)
    _flip_rows(flipped, 3)
    assert check.main([str(flipped)]) == 1
    assert "FAIL: l0f" in capsys.readouterr().err


def test_report_is_json_serialisable(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    write_mirror_dataset(root, float32=True)
    report = resynthesis_report(root)
    json.dumps(report, allow_nan=False)
