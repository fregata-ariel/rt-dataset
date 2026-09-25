"""Acceptance tests for the viewer-bundle fixtures (CPU only, no Sionna)."""

from __future__ import annotations

import json
import os
import re
import stat
import struct
import subprocess
import sys
import tarfile
import time
import warnings
import zipfile
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import viewer_bundle_fixtures as vbf
from viewer_fixtures import BOX_CENTER_M, BOX_MAX_M, BOX_MIN_M, GROUND_Z_M, write_rf_dataset

from plateau_rt.application.optical_reference import render_optical_references
from plateau_rt.application.rf_dataset_manifest import ManifestError, load_rf_dataset_manifest
from plateau_rt.domain.rf_camera.optical import (
    PinholeIntrinsics as _Intrinsics,
)
from plateau_rt.domain.rf_camera.optical import (
    local_to_world_rays,
    pinhole_ray_directions_local,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
OPTICAL_KEYS = (
    "optical_pinhole_rgba",
    "optical_pinhole_depth_m",
    "optical_pinhole_range_m",
    "optical_hemisphere_rgba",
    "optical_hemisphere_range_m",
)


@pytest.fixture(scope="module")
def bundle_v3(tmp_path_factory: pytest.TempPathFactory):
    """Canonical v3 bundle (never mutated; copy it when a test needs to modify)."""
    return vbf.write_fixture_bundle(tmp_path_factory.mktemp("bundle_v3"))


@pytest.fixture(scope="module")
def bundle_v2(tmp_path_factory: pytest.TempPathFactory):
    """Canonical v2 bundle (never mutated; copy it when a test needs to modify)."""
    return vbf.write_fixture_bundle(tmp_path_factory.mktemp("bundle_v2"), schema_version=2)


def _decode_png(data: bytes) -> np.ndarray:
    """Decode a stdlib-written PNG, checking signature, CRCs and filter bytes."""
    assert data[:8] == PNG_SIGNATURE
    position = 8
    width = height = 0
    colour_type = bit_depth = interlace = -1
    idat = b""
    while position < len(data):
        (length,) = struct.unpack(">I", data[position : position + 4])
        tag = data[position + 4 : position + 8]
        chunk = data[position + 8 : position + 8 + length]
        (stored_crc,) = struct.unpack(">I", data[position + 8 + length : position + 12 + length])
        assert zlib.crc32(tag + chunk) & 0xFFFFFFFF == stored_crc
        position += 12 + length
        if tag == b"IHDR":
            width, height, bit_depth, colour_type, _, _, interlace = struct.unpack(
                ">IIBBBBB", chunk
            )
        elif tag == b"IDAT":
            idat += chunk
        elif tag == b"IEND":
            break
    assert bit_depth == 8
    assert colour_type in (0, 6)
    assert interlace == 0
    raw = zlib.decompress(idat)
    channels = 1 if colour_type == 0 else 4
    stride = width * channels
    shape = (height, width) if channels == 1 else (height, width, 4)
    out = np.zeros(shape, dtype=np.uint8)
    for row in range(height):
        line = raw[row * (stride + 1) : (row + 1) * (stride + 1)]
        assert line[0] == 0
        out[row] = np.frombuffer(line[1:], dtype=np.uint8).reshape(shape[1:])
    return out


def _camera_model(dataset):
    with np.load(dataset.camera_model_path) as model:
        return (
            np.asarray(model["valid_mask"], dtype=bool),
            np.asarray(model["ray_directions_local"], dtype=np.float64),
        )


def _relative_files(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())


def _total_size(root: Path) -> int:
    return sum(p.stat().st_size for p in root.rglob("*") if p.is_file())


# AC1 optical ---------------------------------------------------------------


@pytest.mark.parametrize("which", ["v3", "v2"])
def test_optical_artifacts_exist_and_load(bundle_v3, bundle_v2, which):
    """Optical artifacts exist, load, and match transforms.json and the manifest."""
    bundle = bundle_v3 if which == "v3" else bundle_v2
    dataset = load_rf_dataset_manifest(bundle.dataset_dir)
    assert dataset.view_ids == bundle.truth.view_ids
    transforms = json.loads((bundle.dataset_dir / "transforms.json").read_text())
    assert len(transforms["frames"]) == len(dataset.views)
    valid_mask, _ = _camera_model(dataset)
    for index, view in enumerate(dataset.views):
        raw_artifacts = dataset.raw["views"][index]["artifacts"]
        for key in OPTICAL_KEYS:
            assert view.artifact(key) == bundle.dataset_dir / raw_artifacts[key]
            assert view.artifact(key).is_file()
        frame = transforms["frames"][index]
        assert frame["file_path"] == raw_artifacts["optical_pinhole_rgba"]
        assert frame["depth_file_path"] == raw_artifacts["optical_pinhole_depth_m"]
        rgba = _decode_png(
            (bundle.dataset_dir / raw_artifacts["optical_pinhole_rgba"]).read_bytes()
        )
        assert rgba.shape == (32, 32, 4)
        hemi = _decode_png(
            (bundle.dataset_dir / raw_artifacts["optical_hemisphere_rgba"]).read_bytes()
        )
        assert hemi.shape == valid_mask.shape + (4,)
    assert dataset.raw["optical_reference"]["renderer"] == vbf.OPTICAL_RENDERER


def test_hemisphere_png_alpha(bundle_v3):
    """Flipped-back hemisphere alpha is 0 outside valid_mask and matches range."""
    dataset = load_rf_dataset_manifest(bundle_v3.dataset_dir)
    valid_mask, _ = _camera_model(dataset)
    for index, view in enumerate(dataset.views):
        raw_artifacts = dataset.raw["views"][index]["artifacts"]
        png = _decode_png(
            (bundle_v3.dataset_dir / raw_artifacts["optical_hemisphere_rgba"]).read_bytes()
        )
        alpha = np.flipud(png)[..., 3]
        assert alpha[~valid_mask].tolist() == [0] * int(np.count_nonzero(~valid_mask))
        assert set(np.unique(alpha).tolist()) <= {0, 255}
        ranges = np.load(bundle_v3.dataset_dir / raw_artifacts["optical_hemisphere_range_m"])
        np.testing.assert_array_equal(alpha == 255, np.isfinite(ranges))
        hit = alpha == 255
        assert not np.array_equal(hit, np.flipud(hit))
        assert hit.any()


def test_optical_colours_mirrored(bundle_v3):
    """Opaque pinhole/hemisphere pixels are red on the left, blue on the right."""
    dataset = load_rf_dataset_manifest(bundle_v3.dataset_dir)
    valid_mask, ray_dirs = _camera_model(dataset)
    width = vbf.OPTICAL_SIZE_PX
    for index, view in enumerate(dataset.views):
        raw_artifacts = dataset.raw["views"][index]["artifacts"]
        png = _decode_png(
            (bundle_v3.dataset_dir / raw_artifacts["optical_pinhole_rgba"]).read_bytes()
        )
        opaque = png[..., 3] == 255
        assert opaque[:, : width // 2].any() and opaque[:, width // 2 :].any()
        left = png[:, : width // 2][opaque[:, : width // 2]]
        right = png[:, width // 2 :][opaque[:, width // 2 :]]
        assert (left[:, 0].astype(int) > left[:, 2].astype(int)).all()
        assert (right[:, 2].astype(int) > right[:, 0].astype(int)).all()
        hemi = np.flipud(
            _decode_png(
                (bundle_v3.dataset_dir / raw_artifacts["optical_hemisphere_rgba"]).read_bytes()
            )
        )
        hopaque = hemi[..., 3] == 255
        assert hopaque[ray_dirs[..., 1] > 0].any() and hopaque[ray_dirs[..., 1] <= 0].any()
        for row, col in zip(*np.nonzero(hopaque)):
            pixel = hemi[row, col]
            if ray_dirs[row, col, 1] > 0:
                assert int(pixel[0]) > int(pixel[2])
            else:
                assert int(pixel[2]) > int(pixel[0])


def _on_ground_or_box(point: np.ndarray, box_min: np.ndarray, box_max: np.ndarray) -> bool:
    """Return True when ``point`` lies on the ground plane or on a box face (1e-4 m)."""
    if abs(point[2] - GROUND_Z_M) < 1e-4:
        return True
    inside = bool(np.all((point >= box_min - 1e-4) & (point <= box_max + 1e-4)))
    on_face = bool(np.any((np.abs(point - box_min) < 1e-4) | (np.abs(point - box_max) < 1e-4)))
    return inside and on_face


def test_optical_geometry(bundle_v3):
    """Hit points lie on the box or ground; depth matches range; centres hit box."""
    dataset = load_rf_dataset_manifest(bundle_v3.dataset_dir)
    valid_mask, ray_dirs = _camera_model(dataset)
    intrinsics = _Intrinsics.from_horizontal_fov(32, 32, 90.0)
    pinhole_dirs = pinhole_ray_directions_local(intrinsics)
    box_min = np.asarray(BOX_MIN_M, dtype=np.float64)
    box_max = np.asarray(BOX_MAX_M, dtype=np.float64)
    for index, view in enumerate(dataset.views):
        raw_artifacts = dataset.raw["views"][index]["artifacts"]
        pose = json.loads(view.pose_path.read_text())
        rotation = np.asarray(pose["world_from_local_rotation"], dtype=np.float64)
        position = np.asarray(pose["position_m"], dtype=np.float64)
        ranges = np.load(bundle_v3.dataset_dir / raw_artifacts["optical_pinhole_range_m"])
        depths = np.load(bundle_v3.dataset_dir / raw_artifacts["optical_pinhole_depth_m"])
        _, dirs_world = local_to_world_rays(pinhole_dirs, rotation, position)
        hit = np.isfinite(ranges)
        np.testing.assert_allclose(depths[hit], ranges[hit] * pinhole_dirs[..., 0][hit], rtol=1e-5)
        assert (depths[~hit] == 0.0).all()
        points = position + ranges[..., None] * dirs_world
        for row, col in zip(*np.nonzero(hit)):
            point = points[row, col]
            assert _on_ground_or_box(point, box_min, box_max)
        centre = hit[15:17, 15:17]
        assert centre.all()
        hemi_ranges = np.load(bundle_v3.dataset_dir / raw_artifacts["optical_hemisphere_range_m"])
        hhit = np.isfinite(hemi_ranges)
        assert (hhit <= valid_mask).all()
        _, hdirs_world = local_to_world_rays(ray_dirs, rotation, position)
        hpoints = position + np.nan_to_num(hemi_ranges)[..., None] * hdirs_world
        for row, col in zip(*np.nonzero(hhit)):
            point = hpoints[row, col]
            assert _on_ground_or_box(point, box_min, box_max)


class _FakeRenderer:
    """Renderer stub replaying the analytic box-and-ground caster."""

    def render(self, origins, directions, spp=64, seed=0):
        origins = np.asarray(origins, dtype=np.float64)
        directions = np.asarray(directions, dtype=np.float64)
        range_m, kind = vbf.cast_box_ground(origins, directions)
        return SimpleNamespace(
            rgb=np.zeros((origins.shape[0], 3), dtype=np.float64),
            hit=kind > 0,
            range_m=range_m,
        )


def test_optical_matches_writer(tmp_path: Path):
    """Analytic writer matches render_optical_references driven by a fake renderer."""
    dir_a = tmp_path / "a"
    dir_b = tmp_path / "b"
    truth_a = write_rf_dataset(dir_a)
    write_rf_dataset(dir_b)
    vbf.write_optical(dir_a, truth_a)
    render_optical_references(
        dir_b, renderer=_FakeRenderer(), width=32, height=32, fov_x_deg=90.0, spp=1, seed=0
    )
    raw_a = json.loads((dir_a / "dataset_manifest.json").read_text())
    raw_b = json.loads((dir_b / "dataset_manifest.json").read_text())
    for view_a, view_b in zip(raw_a["views"], raw_b["views"]):
        assert view_a["artifacts"] == view_b["artifacts"]
    ref_a = dict(raw_a["optical_reference"])
    ref_b = dict(raw_b["optical_reference"])
    for ref in (ref_a, ref_b):
        ref.pop("renderer")
        ref.pop("spp")
        ref.pop("seed")
    assert ref_a == ref_b
    assert json.loads((dir_a / "transforms.json").read_text()) == json.loads(
        (dir_b / "transforms.json").read_text()
    )
    import matplotlib.image as mpimg

    for view_a, view_b in zip(raw_a["views"], raw_b["views"]):
        for key in (
            "optical_pinhole_depth_m",
            "optical_pinhole_range_m",
            "optical_hemisphere_range_m",
        ):
            array_a = np.load(dir_a / view_a["artifacts"][key])
            array_b = np.load(dir_b / view_b["artifacts"][key])
            assert array_a.dtype == array_b.dtype
            np.testing.assert_array_equal(array_a, array_b)
        for key in ("optical_pinhole_rgba", "optical_hemisphere_rgba"):
            alpha_a = _decode_png((dir_a / view_a["artifacts"][key]).read_bytes())[..., 3]
            alpha_b = np.round(
                np.asarray(mpimg.imread(dir_b / view_b["artifacts"][key]))[..., 3] * 255
            ).astype(np.uint8)
            np.testing.assert_array_equal(alpha_a, alpha_b)
    assert len(truth_a.view_ids) == 3


def test_cast_box_ground():
    """Hand-computed rays hit box, ground, miss, and stay warning-free."""
    range_m, kind = vbf.cast_box_ground(np.array([[20.0, 0.0, 5.0]]), np.array([[-1.0, 0.0, 0.0]]))
    assert range_m[0] == pytest.approx(15.0)
    assert kind[0] == 1
    range_m, kind = vbf.cast_box_ground(np.array([[20.0, 0.0, 5.0]]), np.array([[0.0, 0.0, -1.0]]))
    assert range_m[0] == pytest.approx(5.0)
    assert kind[0] == 2
    range_m, kind = vbf.cast_box_ground(np.array([[20.0, 0.0, 5.0]]), np.array([[0.0, 0.0, 1.0]]))
    assert np.isnan(range_m[0]) and kind[0] == 0
    direction = np.asarray([1.0, 0.0, -0.01])
    direction = direction / np.linalg.norm(direction)
    range_m, kind = vbf.cast_box_ground(np.array([[20.0, 0.0, 5.0]]), direction[None, :])
    assert np.isnan(range_m[0]) and kind[0] == 0
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        range_m, kind = vbf.cast_box_ground(
            np.array([[20.0, 0.0, 5.0], [0.0, 0.0, 5.0]]),
            np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        )
    assert range_m.shape == (2,)


# AC2 observed / partial -----------------------------------------------------


def test_observed_readable(bundle_v3):
    """The obs variant reads the way existing observe tests read it."""
    dataset = load_rf_dataset_manifest(bundle_v3.dataset_dir)
    artifact_keys = dataset.raw["observations"]["obs"]["artifact_keys"]
    assert set(artifact_keys) >= {"aperture_cfr", "impairment_gt"}
    front_index = list(dataset.hemispheres).index("front")
    for view in dataset.views:
        ideal = dataset.load_aperture_cfr(view)
        for entry in view.bs:
            path = entry.artifact("observed.obs.aperture_cfr")
            assert path.is_file()
            observed = np.load(path)
            assert observed.dtype == np.complex64
            assert observed.shape == (8, 8, 16)
            gt = json.loads(entry.artifact("observed.obs.impairment_gt").read_text())
            assert gt["view_id"] == view.view_id
            assert gt["bs_id"] == entry.bs_id
            assert gt["observation"] == "obs"
            assert not np.array_equal(observed, ideal[entry.bs_index, front_index])


def test_observed_requires_v3(bundle_v2):
    """write_observed on a v2 dataset raises ManifestError."""
    with pytest.raises(ManifestError):
        vbf.write_observed(bundle_v2.dataset_dir)


@pytest.mark.parametrize("which,partial", [("v3", "p0"), ("v3", "p1"), ("v2", "p0")])
def test_partial_readable(bundle_v3, bundle_v2, which, partial):
    """Partials link back into their dataset and carry the writer's payloads."""
    bundle = bundle_v3 if which == "v3" else bundle_v2
    index = 0 if partial == "p0" else 1
    partial_dir = bundle.partial_dirs[index]
    manifest = json.loads((partial_dir / "partial_manifest.json").read_text())
    assert (partial_dir / manifest["source_dataset"]).resolve() == bundle.dataset_dir.resolve()
    for key in ("source_manifest", "camera_model_source"):
        assert (partial_dir / manifest[key]).is_file()
    gt = manifest["path_geometry_gt"]
    assert (partial_dir / gt["artifact"]).is_file()
    assert (partial_dir / gt["artifact"]).resolve().is_relative_to(bundle.dataset_dir.resolve())
    dataset = load_rf_dataset_manifest(bundle.dataset_dir)
    if partial == "p0":
        assert len(manifest["views"]) == 2
        assert manifest["subband"]["start"] == 4
        assert manifest["subband"]["stop"] == 12
        assert manifest["subband"]["num_bins"] == 8
        saved = np.load(partial_dir / manifest["views"][0]["artifacts"]["aperture_cfr"])
        assert saved.shape == (dataset.num_bs, 2, 8, 8, 8)
        assert (partial_dir / manifest["element_mask"]["file"]).is_file()
        np.load(partial_dir / manifest["element_mask"]["file"])
    else:
        assert manifest["summary"]["kind"] == "delay"
        for entry in manifest["views"]:
            for bs_entry in entry["bs"]:
                payload = json.loads(
                    (partial_dir / bs_entry["artifacts"]["dominant_delay"]).read_text()
                )
                assert set(payload) == {
                    "delay_s",
                    "power",
                    "delay_resolution_s",
                    "unambiguous_delay_s",
                    "valid",
                }


# AC3 archives ---------------------------------------------------------------


@pytest.mark.parametrize("fmt", vbf.ARCHIVE_FORMATS)
def test_archive_bytes_deterministic(tmp_path: Path, fmt):
    """Same arguments give identical bytes across directories and across runs."""
    first = tmp_path / "first"
    second = tmp_path / "second"
    vbf.write_fixture_bundle(first)
    vbf.write_fixture_bundle(second)
    ext = {"zip": ".zip", "tar": ".tar", "tar.gz": ".tar.gz"}[fmt]
    out_a = tmp_path / f"a{ext}"
    out_b = tmp_path / f"b{ext}"
    vbf.make_archive(first, out_a, fmt, root_name="bundle")
    vbf.make_archive(second, out_b, fmt, root_name="bundle")
    assert out_a.read_bytes() == out_b.read_bytes()
    vbf.make_archive(first, out_a, fmt, root_name="bundle")
    assert out_a.read_bytes() == out_b.read_bytes()


def test_archive_formats_extract_identically(bundle_v3, tmp_path: Path):
    """zip/tar/tar.gz extract to the same files with the same bytes."""
    outputs = {}
    for fmt in vbf.ARCHIVE_FORMATS:
        ext = {"zip": ".zip", "tar": ".tar", "tar.gz": ".tar.gz"}[fmt]
        out = tmp_path / f"bundle{ext}"
        vbf.make_archive(bundle_v3.root, out, fmt, root_name="bundle")
        outputs[fmt] = out
    extracted = {}
    (tmp_path / "zip_out").mkdir()
    with zipfile.ZipFile(outputs["zip"]) as archive:
        archive.extractall(tmp_path / "zip_out")
    extracted["zip"] = tmp_path / "zip_out"
    for fmt in ("tar", "tar.gz"):
        target = tmp_path / f"{fmt}_out"
        target.mkdir()
        with tarfile.open(outputs[fmt]) as archive:
            archive.extractall(target, filter="data")
        extracted[fmt] = target
    references = None
    for fmt, root in extracted.items():
        files = _relative_files(root)
        assert files[0].startswith("bundle/")
        payloads = {name: (root / name).read_bytes() for name in files}
        if references is None:
            references = payloads
        else:
            assert sorted(payloads) == sorted(references)
            for name, data in payloads.items():
                assert data == references[name]
    assert references is not None
    source = {
        f"bundle/{name}": (bundle_v3.root / name).read_bytes()
        for name in _relative_files(bundle_v3.root)
    }
    assert sorted(references) == sorted(source)
    for name, data in source.items():
        assert references[name] == data


def test_archive_metadata_fixed(bundle_v3, tmp_path: Path):
    """Archive entries carry fixed metadata and sorted unique names."""
    out_zip = tmp_path / "b.zip"
    out_tar = tmp_path / "b.tar"
    out_gz = tmp_path / "b.tar.gz"
    vbf.make_archive(bundle_v3.root, out_zip, "zip", root_name="bundle")
    vbf.make_archive(bundle_v3.root, out_tar, "tar", root_name="bundle")
    vbf.make_archive(bundle_v3.root, out_gz, "tar.gz", root_name="bundle")
    with zipfile.ZipFile(out_zip) as archive:
        names = [info.filename for info in archive.infolist()]
        assert names == sorted(names, key=lambda n: n.rstrip("/"))
        assert len(set(names)) == len(names)
        assert all(name.startswith("bundle/") for name in names)
        for info in archive.infolist():
            assert info.date_time == (1980, 1, 1, 0, 0, 0)
            assert info.create_system == 3
            if info.is_dir():
                assert info.external_attr == (0o040755 << 16) | 0x10
            else:
                assert info.external_attr == 0o100644 << 16
    with tarfile.open(out_tar) as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        assert names == sorted(names, key=lambda n: n.rstrip("/"))
        assert len(set(names)) == len(names)
        assert all(name == "bundle" or name.startswith("bundle/") for name in names)
        for member in members:
            assert member.mtime == 0
            assert member.uid == 0 and member.gid == 0
            assert member.uname == "" and member.gname == ""
            assert member.mode in (0o644, 0o755)
            if member.isdir():
                assert member.mode == 0o755
            else:
                assert member.mode == 0o644
    raw = out_gz.read_bytes()
    assert raw[:2] == b"\x1f\x8b"
    assert raw[4:8] == b"\x00\x00\x00\x00"


def test_make_archive_rejects_symlink_and_bad_format(tmp_path: Path):
    """Symlinks and unknown formats are rejected."""
    src = tmp_path / "src"
    (src / "sub").mkdir(parents=True)
    (src / "sub" / "file.txt").write_text("hi")
    (src / "link").symlink_to(src / "sub" / "file.txt")
    with pytest.raises(ValueError):
        vbf.make_archive(src, tmp_path / "out.zip", "zip")
    (src / "link").unlink()
    with pytest.raises(ValueError):
        vbf.make_archive(src, tmp_path / "out.rar", "rar")


# AC4 broken -----------------------------------------------------------------


@pytest.mark.parametrize("case", vbf.BROKEN_CASES)
def test_broken_dataset(tmp_path: Path, case):
    """Each broken case raises ManifestError with the documented message."""
    root = vbf.write_broken_dataset(tmp_path / "dataset", case)
    with pytest.raises(ManifestError, match=re.escape(vbf.BROKEN_CASE_MESSAGES[case])):
        load_rf_dataset_manifest(root)


def test_broken_dataset_control_and_unknown(tmp_path: Path):
    """Unmodified output loads; unknown cases raise ValueError."""
    truth = write_rf_dataset(tmp_path / "good")
    load_rf_dataset_manifest(truth.root)
    with pytest.raises(ValueError):
        vbf.write_broken_dataset(tmp_path / "bad", "nope")


# AC5 malicious --------------------------------------------------------------


_MALICIOUS_PAIRS = [(case, fmt) for case in vbf.MALICIOUS_CASES for fmt in ("tar", "tar.gz")] + [
    (case, "zip") for case in vbf.MALICIOUS_CASES if case not in ("hardlink", "device")
]


@pytest.mark.parametrize("case,fmt", _MALICIOUS_PAIRS)
def test_malicious_archive_shape(tmp_path: Path, case, fmt):
    """Malicious archives have the intended entry shape."""
    out = tmp_path / f"mal.{fmt}"
    count = 50 if case == "too_many_files" else 2000
    bombs = 1_000_000 if case == "bomb" else 8 * 1024 * 1024
    vbf.make_malicious_archive(out, case, fmt=fmt, file_count=count, bomb_bytes=bombs)
    repeat = tmp_path / f"rep.{fmt}"
    vbf.make_malicious_archive(repeat, case, fmt=fmt, file_count=count, bomb_bytes=bombs)
    assert out.read_bytes() == repeat.read_bytes()
    if fmt == "zip":
        with zipfile.ZipFile(out) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            assert "ok.txt" in names
            assert archive.read("ok.txt") == b"ok\n"
            if case == "dotdot":
                assert "../evil.txt" in names
            elif case == "absolute":
                assert any(name.startswith("/") for name in names)
            elif case == "symlink":
                info = next(i for i in infos if i.filename == "link")
                assert stat.S_ISLNK(info.external_attr >> 16)
                assert archive.read("link") == b"../../etc/passwd"
            elif case == "duplicate":
                dups = [i for i in infos if i.filename == "dup.txt"]
                assert len(dups) == 2
                assert archive.open(dups[0]).read() == b"first\n"
                assert archive.open(dups[1]).read() == b"second\n"
            elif case == "too_many_files":
                assert len(infos) == count + 1
            elif case == "bomb":
                info = next(i for i in infos if i.filename == "zeros.bin")
                assert info.file_size == bombs
                assert out.stat().st_size < bombs / 100
    else:
        with tarfile.open(out) as archive:
            members = archive.getmembers()
            names = [member.name for member in members]
            assert "ok.txt" in names
            ok_file = archive.extractfile("ok.txt")
            assert ok_file is not None and ok_file.read() == b"ok\n"
            if case == "dotdot":
                assert "../evil.txt" in names
            elif case == "absolute":
                assert any(name.startswith("/") for name in names)
            elif case == "symlink":
                member = archive.getmember("link")
                assert member.issym()
                assert member.linkname == "../../etc/passwd"
            elif case == "hardlink":
                member = archive.getmember("hardlink")
                assert member.islnk()
                assert member.linkname == "/etc/passwd"
            elif case == "device":
                member = archive.getmember("dev/null")
                assert member.ischr()
                assert member.devmajor == 1 and member.devminor == 3
            elif case == "duplicate":
                dups = [m for m in members if m.name == "dup.txt"]
                assert len(dups) == 2
                first_file = archive.extractfile(dups[0])
                second_file = archive.extractfile(dups[1])
                assert first_file is not None and second_file is not None
                assert first_file.read() == b"first\n"
                assert second_file.read() == b"second\n"
            elif case == "too_many_files":
                assert len(members) == count + 1
            elif case == "bomb":
                member = archive.getmember("zeros.bin")
                assert member.size == bombs
                if fmt == "tar.gz":
                    assert out.stat().st_size < bombs / 100
                else:
                    assert out.stat().st_size >= bombs
    if case == "too_many_files" and count == 50:
        default_out = tmp_path / "default.tar.gz"
        vbf.make_malicious_archive(default_out, "too_many_files")
        with tarfile.open(default_out) as archive:
            assert len(archive.getmembers()) == 2000 + 1


def test_malicious_unsupported(tmp_path: Path):
    """Unsupported malicious combinations raise ValueError."""
    with pytest.raises(ValueError):
        vbf.make_malicious_archive(tmp_path / "h.zip", "hardlink", fmt="zip")
    with pytest.raises(ValueError):
        vbf.make_malicious_archive(tmp_path / "d.zip", "device", fmt="zip")
    with pytest.raises(ValueError):
        vbf.make_malicious_archive(tmp_path / "x.tar.gz", "nope")
    with pytest.raises(ValueError):
        vbf.make_malicious_archive(tmp_path / "x.rar", "dotdot", fmt="rar")


# AC6 CLI --------------------------------------------------------------------


def test_cli_script_zip(tmp_path: Path):
    """The CLI script builds a <= 5 MB zip with a single top-level bundle/."""
    out = tmp_path / "out"
    out.mkdir()
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    first = subprocess.run(
        [sys.executable, "tests/viewer_bundle_fixtures.py", str(out), "--archive", "zip"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert first.returncode == 0, first.stderr
    assert first.stdout.strip() == str(out / "bundle.zip")
    assert (out / "bundle.zip").stat().st_size <= 5_000_000
    extract = tmp_path / "extract"
    extract.mkdir()
    with zipfile.ZipFile(out / "bundle.zip") as archive:
        archive.extractall(extract)
    top = [p for p in extract.iterdir()]
    assert len(top) == 1 and top[0].name == "bundle"
    assert (top[0] / "bundle.json").is_file()
    assert _total_size(top[0]) <= 5_000_000
    first_bytes = (out / "bundle.zip").read_bytes()
    second = subprocess.run(
        [sys.executable, "tests/viewer_bundle_fixtures.py", str(out), "--archive", "zip"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert second.returncode == 0, second.stderr
    assert (out / "bundle.zip").read_bytes() == first_bytes
    assert second.stdout.strip() == str(out / "bundle.zip")


def test_cli_modes(tmp_path: Path, capsys: pytest.CaptureFixture):
    """In-process CLI modes: dir, v2, broken, malicious, and reruns."""
    out = tmp_path / "out"
    out.mkdir()
    assert vbf.main([str(out), "--archive", "dir"]) == 0
    printed = Path(capsys.readouterr().out.strip())
    assert printed == out / "bundle"
    assert _total_size(printed) <= 5_000_000
    load_rf_dataset_manifest(printed / "dataset")

    assert vbf.main([str(out), "--schema", "2", "--archive", "dir"]) == 0
    printed = Path(capsys.readouterr().out.strip())
    assert printed == out / "bundle_v2"
    dataset = load_rf_dataset_manifest(printed / "dataset")
    assert dataset.schema_version == 2
    assert "observations" not in dataset.raw

    assert vbf.main([str(out), "--broken", "bs_order_mismatch", "--archive", "dir"]) == 0
    printed = Path(capsys.readouterr().out.strip())
    assert printed == out / "broken_bs_order_mismatch"
    with pytest.raises(ManifestError):
        load_rf_dataset_manifest(printed / "dataset")
    assert len(json.loads((printed / "bundle.json").read_text())["members"]) == 1

    assert vbf.main([str(out), "--broken", "missing_artifact", "--archive", "tar.gz"]) == 0
    printed = Path(capsys.readouterr().out.strip())
    assert printed.is_file()

    assert vbf.main([str(out), "--malicious", "symlink"]) == 0
    printed = Path(capsys.readouterr().out.strip())
    assert printed.suffixes[-2:] == [".tar", ".gz"]
    with tarfile.open(printed) as archive:
        assert archive.getmember("link").issym()

    assert vbf.main([str(out), "--archive", "dir"]) == 0
    capsys.readouterr()


def test_cli_errors(tmp_path: Path):
    """Invalid CLI combinations exit with code 2."""
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(SystemExit) as exc:
        vbf.main([str(out), "--malicious", "hardlink", "--archive", "zip"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        vbf.main([str(out), "--broken", "dotdot", "--malicious", "dotdot"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        vbf.main([str(out), "--schema", "2", "--broken", "dotdot"])
    assert exc.value.code == 2
    with pytest.raises(SystemExit) as exc:
        vbf.main([str(out), "--malicious", "dotdot", "--archive", "dir"])
    assert exc.value.code == 2
    (out / "bundle").mkdir()
    (out / "bundle" / "notes.txt").write_text("not a bundle")
    with pytest.raises(SystemExit) as exc:
        vbf.main([str(out), "--archive", "dir"])
    assert exc.value.code == 2


# AC7 sionna-free ------------------------------------------------------------


def test_helper_is_sionna_free(tmp_path: Path):
    """Importing and running the helper never pulls heavy renderer modules."""
    tests_dir = str(Path(__file__).resolve().parent)
    code = (
        "import importlib.abc, io, sys, tempfile\n"
        "from contextlib import redirect_stdout\n"
        "blocked = {'sionna', 'mitsuba', 'drjit', 'matplotlib', 'PIL'}\n"
        "class _Blocker(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in blocked:\n"
        "            raise ImportError(name)\n"
        "        return None\n"
        "sys.meta_path.insert(0, _Blocker())\n"
        f"sys.path.insert(0, {tests_dir!r})\n"
        "import viewer_bundle_fixtures as vbf\n"
        "tmp = tempfile.mkdtemp()\n"
        "import pathlib\n"
        "vbf.write_fixture_bundle(pathlib.Path(tmp) / 'b3')\n"
        "vbf.write_fixture_bundle(pathlib.Path(tmp) / 'b2', schema_version=2)\n"
        "for fmt in vbf.ARCHIVE_FORMATS:\n"
        "    vbf.make_archive(pathlib.Path(tmp) / 'b3', pathlib.Path(tmp) / f'a.{fmt}', fmt)\n"
        "for case in vbf.MALICIOUS_CASES:\n"
        "    try:\n"
        "        vbf.make_malicious_archive(pathlib.Path(tmp) / f'm-{case}.tar.gz', case)\n"
        "    except ValueError:\n"
        "        pass\n"
        "vbf.write_broken_dataset(pathlib.Path(tmp) / 'broken', 'bad_position')\n"
        "with redirect_stdout(io.StringIO()):\n"
        "    vbf.main([tmp + '/cli', '--archive', 'zip'])\n"
        "found = [m.split('.')[0] for m in sys.modules if m.split('.')[0] in blocked]\n"
        "print(','.join(sorted(found)))"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ""


# Other tests ----------------------------------------------------------------


def _read_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse a trimesh-style binary PLY written by the fixtures."""
    raw = path.read_bytes()
    header_end = raw.index(b"end_header\n") + len(b"end_header\n")
    lines = raw[:header_end].decode("ascii").splitlines()
    assert lines[0] == "ply"
    assert lines[1] == "format binary_little_endian 1.0"
    vertex_count = int(lines[2].split()[2])
    assert lines[2] == f"element vertex {vertex_count}"
    assert lines[3] == "property float x"
    assert lines[4] == "property float y"
    assert lines[5] == "property float z"
    face_count = int(lines[6].split()[2])
    assert lines[6] == f"element face {face_count}"
    assert lines[7] == "property list uchar int vertex_indices"
    assert lines[8] == "end_header"
    assert len(lines) == 9
    body = raw[header_end:]
    vertices = np.frombuffer(body[: vertex_count * 12], dtype="<f4").reshape(vertex_count, 3)
    faces = np.frombuffer(
        body[vertex_count * 12 :],
        dtype=[("n", "u1"), ("i", "<i4", (3,))],
    )
    assert len(faces) == face_count
    assert (faces["n"] == 3).all()
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces["i"], dtype=np.int64)


def test_scene_files(bundle_v3):
    """PLY meshes and scene XML match the box/ground geometry."""
    scene_dir = bundle_v3.root / "scene"
    box_vertices, box_faces = _read_ply(scene_dir / "box.ply")
    ground_vertices, ground_faces = _read_ply(scene_dir / "ground.ply")
    assert box_vertices.shape == (8, 3)
    assert box_faces.shape == (12, 3)
    np.testing.assert_allclose(box_vertices.min(axis=0), np.asarray(BOX_MIN_M), atol=1e-6)
    np.testing.assert_allclose(box_vertices.max(axis=0), np.asarray(BOX_MAX_M), atol=1e-6)
    centre = np.asarray(BOX_CENTER_M, dtype=np.float64)
    for face in box_faces:
        points = box_vertices[face]
        normal = np.cross(points[1] - points[0], points[2] - points[0])
        assert float(np.dot(normal, points.mean(axis=0) - centre)) > 0
    assert ground_vertices.shape == (4, 3)
    assert ground_faces.shape == (2, 3)
    assert (ground_vertices[:, 2] == 0.0).all()
    assert float(np.max(np.abs(ground_vertices[:, :2]))) == pytest.approx(200.0)
    for face in ground_faces:
        points = ground_vertices[face]
        normal = np.cross(points[1] - points[0], points[2] - points[0])
        assert normal[2] > 0
    import xml.etree.ElementTree as ET

    tree = ET.parse(scene_dir / "scene.xml")
    assert tree.getroot().tag == "scene"
    shapes = tree.getroot().findall("shape")
    dataset = load_rf_dataset_manifest(bundle_v3.dataset_dir)
    assert dataset.path_geometry_gt is not None
    assert dataset.path_geometry_gt.schema_path is not None
    object_names = dataset.path_geometry_gt.load_schema()["object_names"]
    assert sorted(s.get("id") or "" for s in shapes) == sorted(object_names)
    bsdf_ids = {node.get("id") for node in tree.getroot().findall("bsdf")}
    for shape in shapes:
        string_node = shape.find("string")
        ref_node = shape.find("ref")
        assert string_node is not None and ref_node is not None
        filename = string_node.get("value")
        assert filename is not None and (scene_dir / filename).is_file()
        assert ref_node.get("id") in bsdf_ids


def test_bundle_json_contents(bundle_v3, bundle_v2):
    """bundle.json equals the canonical dict and ends with a newline."""
    expected = {
        "bundle_format_version": 1,
        "members": [
            {"id": "dataset", "kind": "rf_dataset", "path": "dataset"},
            {"id": "scene", "kind": "scene", "path": "scene/scene.xml", "for": "dataset"},
            {"id": "p0", "kind": "rf_partial", "path": "partials/p0", "source": "dataset"},
            {"id": "p1", "kind": "rf_partial", "path": "partials/p1"},
        ],
        "created_by": {"tool": "viewer_bundle_fixtures", "tool_version": 1},
    }
    for bundle in (bundle_v3, bundle_v2):
        text = bundle.bundle_json.read_text()
        assert text.endswith("\n")
        assert json.loads(text) == expected


def _mini_bundle_root(root: Path) -> Path:
    """Create a tiny valid bundle layout (markers only) for validation tests."""
    (root / "dataset").mkdir(parents=True)
    (root / "dataset" / "dataset_manifest.json").write_text("{}")
    (root / "partials" / "p0").mkdir(parents=True)
    (root / "partials" / "p0" / "partial_manifest.json").write_text("{}")
    (root / "runs" / "r0").mkdir(parents=True)
    (root / "runs" / "r0" / "run_manifest.json").write_text("{}")
    (root / "scene").mkdir(parents=True)
    (root / "scene" / "scene.xml").write_text("<scene/>")
    (root / "scene" / "other.xml").write_text("<scene/>")
    (root / "notes.txt").write_text("notes")
    (root / "emptydir").mkdir()
    (root / "dataset" / "nested").mkdir()
    (root / "dataset" / "nested" / "partial_manifest.json").write_text("{}")
    link = root / "link"
    if not link.exists() and not link.is_symlink():
        link.symlink_to(root / "dataset", target_is_directory=True)
    return root


def _invalid_member_lists() -> list[tuple[list[dict], str]]:
    """One invalid member list per validation rule, with the expected message fragment."""
    good_partial = {"id": "p0", "kind": "rf_partial", "path": "partials/p0", "source": "dataset"}
    good_dataset = {"id": "dataset", "kind": "rf_dataset", "path": "dataset"}
    good_scene = {"id": "scene", "kind": "scene", "path": "scene/scene.xml", "for": "dataset"}
    many = [{"id": f"m{i}", "kind": "rf_dataset", "path": "dataset"} for i in range(1025)]

    def bad_path(path: str) -> tuple[list[dict], str]:
        return [{"id": "x", "kind": "rf_dataset", "path": path}], "invalid path"

    return [
        ([], "non-empty"),
        (many, "at most 1024"),
        ([{"id": "a", "kind": "rf_dataset", "path": "dataset", "sorce": "dataset"}], "unknown key"),
        ([{"kind": "rf_dataset", "path": "dataset"}], "missing required key 'id'"),
        ([{"id": "bad id!", "kind": "rf_dataset", "path": "dataset"}], "invalid id"),
        ([{"id": ".", "kind": "rf_dataset", "path": "dataset"}], "invalid id"),
        ([{"id": "..", "kind": "rf_dataset", "path": "dataset"}], "invalid id"),
        ([{"id": "x" * 65, "kind": "rf_dataset", "path": "dataset"}], "invalid id"),
        ([good_dataset, dict(good_dataset, kind="rf_partial")], "duplicate id"),
        ([{"id": "x", "kind": "placement", "path": "dataset"}], "invalid kind"),
        bad_path(""),
        bad_path("."),
        bad_path("/tmp/x"),
        bad_path("a\\b"),
        bad_path("C:/dataset"),
        bad_path("dataset/"),
        bad_path("./dataset"),
        bad_path("dataset/../dataset"),
        ([{"id": "x", "kind": "rf_dataset", "path": "missing"}], "missing on disk"),
        ([{"id": "x", "kind": "rf_dataset", "path": "link"}], "symlink"),
        ([{"id": "x", "kind": "rf_dataset", "path": "scene/scene.xml"}], "not a directory"),
        ([{"id": "x", "kind": "rf_partial", "path": "emptydir"}], "lacks marker"),
        ([{"id": "x", "kind": "scene", "path": "dataset"}], "not a regular file"),
        ([{"id": "x", "kind": "scene", "path": "notes.txt"}], "must end in .xml"),
        (
            [good_dataset, {"id": "dup", "kind": "rf_dataset", "path": "dataset"}],
            "duplicate directory path",
        ),
        (
            [good_dataset, {"id": "n", "kind": "rf_partial", "path": "dataset/nested"}],
            "nested directory paths",
        ),
        ([good_dataset, dict(good_partial, **{"for": "dataset"})], "'for' only allowed"),
        ([dict(good_dataset, source="dataset")], "'source' only allowed"),
        ([good_dataset, dict(good_scene, **{"for": "nope"})], "'for' names unknown"),
        ([good_dataset, dict(good_scene, **{"for": "scene"})], "'for' names unknown"),
        ([good_dataset, dict(good_partial, source="nope")], "'source' names unknown"),
        ([good_dataset, dict(good_partial, source="p0")], "'source' names unknown"),
        (
            [
                good_dataset,
                good_scene,
                {"id": "s2", "kind": "scene", "path": "scene/other.xml", "for": "dataset"},
            ],
            "two scenes",
        ),
    ]


@pytest.mark.parametrize("members,match", _invalid_member_lists())
def test_bundle_json_validation(tmp_path: Path, members, match):
    """Each invalid member list fails for its own rule, but writes with validate=False."""
    root = _mini_bundle_root(tmp_path / "root")
    with pytest.raises(ValueError, match=re.escape(match)):
        vbf.write_bundle_dir(root, members=members, validate=True)
    path = vbf.write_bundle_dir(root, members=members, validate=False)
    assert json.loads(path.read_text())["members"] == members


def test_bundle_json_accepts_valid_members(tmp_path: Path):
    """A valid layout with every member kind and link is accepted."""
    root = _mini_bundle_root(tmp_path / "root")
    members = [
        {"id": "dataset", "kind": "rf_dataset", "path": "dataset"},
        {"id": "scene", "kind": "scene", "path": "scene/scene.xml", "for": "dataset"},
        {"id": "p0", "kind": "rf_partial", "path": "partials/p0", "source": "dataset"},
        {"id": "r0", "kind": "tomo_run", "path": "runs/r0", "source": "dataset"},
        {"id": "loose", "kind": "scene", "path": "scene/other.xml"},
    ]
    path = vbf.write_bundle_dir(root, members=members, created_by={"tool": "t"})
    assert json.loads(path.read_text()) == {
        "bundle_format_version": 1,
        "members": members,
        "created_by": {"tool": "t"},
    }


def test_write_fixture_bundle_refuses_non_empty_root(tmp_path: Path):
    """A non-empty root is refused."""
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "notes.txt").write_text("hi")
    with pytest.raises(ValueError):
        vbf.write_fixture_bundle(root)


def test_bundle_speed(tmp_path: Path):
    """The canonical bundle builds in under 3 s."""
    start = time.perf_counter()
    vbf.write_fixture_bundle(tmp_path / "bundle")
    assert time.perf_counter() - start < 3.0
