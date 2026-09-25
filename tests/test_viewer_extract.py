"""Tests for the safe archive extractor (CPU only, no Sionna)."""

from __future__ import annotations

import gzip
import io
import os
import stat
import struct
import tarfile
import zipfile
from pathlib import Path
from typing import Any

import pytest
import viewer_bundle_fixtures as vbf

from plateau_rt.viewer import extract
from plateau_rt.viewer.extract import (
    ExtractLimits,
    UnsafeArchiveError,
    detect_format,
    safe_extract,
)

SMALL = ExtractLimits(max_files=100, max_extracted_bytes=1 * 1024 * 1024)
BIG = ExtractLimits(max_files=10_000, max_extracted_bytes=1 << 30)


@pytest.fixture(scope="module")
def bundle_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the canonical viewer bundle once for the whole module."""
    root = tmp_path_factory.mktemp("viewer_bundle") / "src"
    vbf.write_fixture_bundle(root)
    return root


def _write_tar(path: Path, entries: list[dict[str, Any]]) -> Path:
    """Write a tar with the given entry specs."""
    with tarfile.open(path, "w") as archive:
        for entry in entries:
            entry_type = entry.get("type", tarfile.REGTYPE)
            info = tarfile.TarInfo(entry["name"])
            info.mtime = 0
            info.type = entry_type
            info.mode = entry.get("mode", 0o644)
            if "linkname" in entry:
                info.linkname = entry["linkname"]
            data = entry.get("data", b"")
            if entry_type == tarfile.REGTYPE:
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
            else:
                archive.addfile(info)
    return path


def _write_zip(path: Path, entries: list[dict[str, Any]]) -> Path:
    """Write a zip with the given entry specs."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for entry in entries:
            info = zipfile.ZipInfo(entry["name"])
            info.create_system = entry.get("create_system", 3)
            info.external_attr = entry.get("external_attr", 0o100644 << 16)
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, entry.get("data", b""))
    return path


EXPECTED_REASONS = {
    "dotdot": "dotdot",
    "absolute": "absolute",
    "symlink": "symlink",
    "hardlink": "hardlink",
    "device": "device",
    "duplicate": "duplicate",
    "too_many_files": "too_many_files",
    "bomb": "too_large",
}


@pytest.mark.parametrize("case", vbf.MALICIOUS_CASES)
@pytest.mark.parametrize("fmt", ["tar.gz", "tar", "zip"])
def test_malicious_archives_are_rejected(tmp_path: Path, case: str, fmt: str) -> None:
    if fmt == "zip" and case in ("hardlink", "device"):
        pytest.skip("zip does not support hardlink/device entries")
    archive = vbf.make_malicious_archive(tmp_path / "evil", case, fmt=fmt)
    dest = tmp_path / "out"
    with pytest.raises(UnsafeArchiveError) as info:
        safe_extract(archive, dest, SMALL)
    err = info.value
    assert err.reason == EXPECTED_REASONS[case]
    if case == "too_many_files" and fmt == "zip":
        assert err.member is None  # rejected from the central-directory count before opening
    elif case == "too_many_files":
        assert err.member is not None and err.member.startswith("many/")
    else:
        assert err.member == vbf.MALICIOUS_MEMBER_NAMES[case]
    assert not dest.exists()


def test_copy_limited_stops_at_budget() -> None:
    sink = io.BytesIO()
    with pytest.raises(UnsafeArchiveError) as info:
        extract._copy_limited(io.BytesIO(b"\x00" * (8 << 20)), sink, 1 << 20, "zeros.bin")
    assert info.value.reason == "too_large"
    assert len(sink.getvalue()) <= 1 << 20
    data = b"\x00" * (3 << 20)
    sink2 = io.BytesIO()
    assert extract._copy_limited(io.BytesIO(data), sink2, len(data), "zeros.bin") == len(data)


def test_header_sizes_are_not_trusted(tmp_path: Path) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("big.bin", b"\x00" * 10)
    data = bytearray(buffer.getvalue())
    local = data.find(b"PK\x03\x04")
    central = data.find(b"PK\x01\x02")
    assert local != -1 and central != -1
    struct.pack_into("<I", data, local + 22, 0x7FFFFFFF)
    struct.pack_into("<I", data, central + 24, 0x7FFFFFFF)
    archive = tmp_path / "big.zip"
    archive.write_bytes(data)
    dest = tmp_path / "out"
    report = safe_extract(archive, dest, SMALL)
    assert report.total_bytes == 10
    assert (dest / "big.bin").read_bytes() == b"\x00" * 10


def test_too_large_boundary(tmp_path: Path) -> None:
    archive = _write_zip(tmp_path / "one.zip", [{"name": "f.bin", "data": b"x" * 1000}])
    exact = tmp_path / "exact"
    report = safe_extract(archive, exact, ExtractLimits(max_files=10, max_extracted_bytes=1000))
    assert report.total_bytes == 1000
    tight = tmp_path / "tight"
    with pytest.raises(UnsafeArchiveError) as info:
        safe_extract(archive, tight, ExtractLimits(max_files=10, max_extracted_bytes=999))
    assert info.value.reason == "too_large"
    assert not tight.exists()


def test_zip_and_targz_of_same_bundle_extract_identically(
    tmp_path: Path, bundle_root: Path
) -> None:
    source_files = sorted(
        path.relative_to(bundle_root).as_posix()
        for path in bundle_root.rglob("*")
        if path.is_file()
    )
    source_bytes = {name: (bundle_root / name).read_bytes() for name in source_files}
    expected = tuple(sorted(f"bundle/{name}" for name in source_files))
    seen = []
    for fmt in ("zip", "tar", "tar.gz"):
        archive = vbf.make_archive(
            bundle_root, tmp_path / f"bundle_{fmt.replace('.', '_')}.arc", fmt, root_name="bundle"
        )
        dest = tmp_path / f"out_{fmt.replace('.', '_')}"
        report = safe_extract(archive, dest, BIG)
        assert report.format == fmt
        assert report.files == expected
        assert report.root == dest / "bundle"
        assert (report.root / "bundle.json").is_file()
        for name in source_files:
            assert (dest / "bundle" / name).read_bytes() == source_bytes[name]
        seen.append(report)
    assert seen[0].files == seen[1].files == seen[2].files
    flat = vbf.make_archive(bundle_root, tmp_path / "flat.tar", "tar")
    flat_dest = tmp_path / "out_flat"
    flat_report = safe_extract(flat, flat_dest, BIG)
    assert flat_report.root == flat_dest
    assert flat_report.files == tuple(sorted(source_files))


def _small_src(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    (src / "a.txt").write_bytes(b"hello")
    return src


@pytest.mark.parametrize(
    ("fmt", "misleading"),
    [("zip", "thing.tar.gz"), ("tar", "thing.zip"), ("tar.gz", "thing.tar")],
)
def test_detect_format_ignores_extension(tmp_path: Path, fmt: str, misleading: str) -> None:
    src = _small_src(tmp_path)
    archive = vbf.make_archive(src, tmp_path / misleading, fmt)
    assert detect_format(archive) == fmt


def test_unknown_formats_rejected(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.write_bytes(b"")
    random_bytes = tmp_path / "random"
    random_bytes.write_bytes(b"\x01\x02\x03" * 200)
    bz2 = tmp_path / "b.tar.bz2"
    with tarfile.open(bz2, "w:bz2") as archive:
        info = tarfile.TarInfo("a")
        info.size = 1
        archive.addfile(info, io.BytesIO(b"x"))
    gz = tmp_path / "g.gz"
    gz.write_bytes(gzip.compress(b"hello" * 200))
    full = _write_zip(tmp_path / "full.zip", [{"name": "a.txt", "data": b"hello"}])
    payload = full.read_bytes()
    truncated = tmp_path / "truncated.zip"
    truncated.write_bytes(payload[: len(payload) // 2])
    cases = {
        "empty": empty,
        "random": random_bytes,
        "bz2": bz2,
        "gzip-nontar": gz,
        "truncated": truncated,
    }
    for name in ("empty", "random", "bz2", "gzip-nontar"):
        with pytest.raises(UnsafeArchiveError) as info:
            detect_format(cases[name])
        assert info.value.reason == "unknown_format", name
    for name, path in cases.items():
        dest = tmp_path / f"out_{name}"
        with pytest.raises(UnsafeArchiveError) as info:
            safe_extract(path, dest, SMALL)
        assert info.value.reason == "unknown_format", name
        assert not dest.exists(), name


def test_bad_names(tmp_path: Path) -> None:
    tar_backslash = _write_tar(tmp_path / "back.tar", [{"name": "a\\b.txt", "data": b"x"}])
    with pytest.raises(UnsafeArchiveError) as info:
        safe_extract(tar_backslash, tmp_path / "o1", SMALL)
    assert info.value.reason == "bad_name"
    zip_backslash = _write_zip(tmp_path / "back.zip", [{"name": "a\\b.txt", "data": b"x"}])
    with pytest.raises(UnsafeArchiveError) as info:
        safe_extract(zip_backslash, tmp_path / "o2", SMALL)
    assert info.value.reason == "bad_name"
    nul = _write_zip(tmp_path / "nul.zip", [{"name": "a_txt", "data": b"x"}])
    nul.write_bytes(nul.read_bytes().replace(b"a_txt", b"a\x00txt"))
    with pytest.raises(UnsafeArchiveError) as info:
        safe_extract(nul, tmp_path / "o3", SMALL)
    assert info.value.reason == "bad_name"
    empty_name = _write_zip(tmp_path / "empty_name.zip", [{"name": "", "data": b"x"}])
    with pytest.raises(UnsafeArchiveError) as info:
        safe_extract(empty_name, tmp_path / "o4", SMALL)
    assert info.value.reason == "bad_name"
    long_name = _write_tar(tmp_path / "long.tar", [{"name": "a" * 256, "data": b"x"}])
    with pytest.raises(UnsafeArchiveError) as info:
        safe_extract(long_name, tmp_path / "o5", SMALL)
    assert info.value.reason == "bad_name"


def test_absolute_windows_drive(tmp_path: Path) -> None:
    archive = _write_tar(tmp_path / "drive.tar", [{"name": "C:/x.txt", "data": b"x"}])
    with pytest.raises(UnsafeArchiveError) as info:
        safe_extract(archive, tmp_path / "out", SMALL)
    assert info.value.reason == "absolute"


@pytest.mark.parametrize("name", ["a/../../b", "a/.."])
def test_dotdot_is_rejected(tmp_path: Path, name: str) -> None:
    archive = _write_tar(tmp_path / "dotdot.tar", [{"name": name, "data": b"x"}])
    with pytest.raises(UnsafeArchiveError) as info:
        safe_extract(archive, tmp_path / "out", SMALL)
    assert info.value.reason == "dotdot"


def test_dot_components_are_normalized(tmp_path: Path) -> None:
    archive = _write_tar(
        tmp_path / "norm.tar",
        [
            {"name": "./", "type": tarfile.DIRTYPE},
            {"name": "./d/", "type": tarfile.DIRTYPE},
            {"name": "./d/x.txt", "data": b"x"},
        ],
    )
    dest = tmp_path / "out"
    report = safe_extract(archive, dest, SMALL)
    assert report.files == ("d/x.txt",)
    assert report.root == dest / "d"


@pytest.mark.parametrize(
    ("entries", "member"),
    [
        ([{"name": "README.txt"}, {"name": "readme.TXT"}], "readme.TXT"),
        ([{"name": "a", "data": b"x"}, {"name": "a/b.txt", "data": b"y"}], "a/b.txt"),
        ([{"name": "a/b.txt", "data": b"y"}, {"name": "a", "data": b"x"}], "a"),
        ([{"name": "A/x.txt", "data": b"x"}, {"name": "a/y.txt", "data": b"y"}], "a/y.txt"),
        ([{"name": "\u00e9.txt"}, {"name": "e\u0301.txt"}], "e\u0301.txt"),
    ],
)
def test_duplicate_entries(tmp_path: Path, entries: Any, member: str) -> None:
    archive = _write_tar(tmp_path / "dup.tar", entries)
    with pytest.raises(UnsafeArchiveError) as info:
        safe_extract(archive, tmp_path / "out", SMALL)
    assert info.value.reason == "duplicate"
    assert info.value.member == member


def test_explicit_dir_then_file_is_fine(tmp_path: Path) -> None:
    archive = _write_tar(
        tmp_path / "ok.tar",
        [
            {"name": "d/", "type": tarfile.DIRTYPE},
            {"name": "d/x.txt", "data": b"x"},
        ],
    )
    report = safe_extract(archive, tmp_path / "out", SMALL)
    assert report.files == ("d/x.txt",)


def test_zip_fifo_is_device(tmp_path: Path) -> None:
    archive = _write_zip(
        tmp_path / "fifo.zip",
        [{"name": "pipe", "external_attr": (stat.S_IFIFO | 0o644) << 16, "create_system": 3}],
    )
    with pytest.raises(UnsafeArchiveError) as info:
        safe_extract(archive, tmp_path / "out", SMALL)
    assert info.value.reason == "device"


@pytest.mark.parametrize("entry_type", [tarfile.FIFOTYPE, tarfile.BLKTYPE])
def test_tar_special_files_are_device(tmp_path: Path, entry_type: bytes) -> None:
    archive = _write_tar(tmp_path / "special.tar", [{"name": "special", "type": entry_type}])
    with pytest.raises(UnsafeArchiveError) as info:
        safe_extract(archive, tmp_path / "out", SMALL)
    assert info.value.reason == "device"


def test_modes_are_forced(tmp_path: Path) -> None:
    archive = _write_tar(
        tmp_path / "modes.tar",
        [
            {"name": "f.txt", "data": b"x", "mode": 0o4777},
            {"name": "d", "type": tarfile.DIRTYPE, "mode": 0o700},
        ],
    )
    dest = tmp_path / "out"
    safe_extract(archive, dest, SMALL)
    assert stat.S_IMODE(os.stat(dest / "f.txt").st_mode) == 0o644
    assert stat.S_IMODE(os.stat(dest / "d").st_mode) == 0o755
    for path in (dest / "f.txt", dest / "d"):
        assert os.path.islink(path) is False
        mode = os.lstat(path).st_mode
        assert stat.S_ISREG(mode) or stat.S_ISDIR(mode)


@pytest.mark.parametrize("fmt", ["tar", "zip"])
def test_max_files_boundary(tmp_path: Path, fmt: str) -> None:
    entries = [{"name": f"f{index}.txt", "data": b"x"} for index in range(3)]
    writer = _write_tar if fmt == "tar" else _write_zip
    archive = writer(tmp_path / "three.arc", entries)
    report = safe_extract(
        archive, tmp_path / "out", ExtractLimits(max_files=3, max_extracted_bytes=1 << 20)
    )
    assert report.entry_count == 3
    with pytest.raises(UnsafeArchiveError) as info:
        safe_extract(
            archive, tmp_path / "out2", ExtractLimits(max_files=2, max_extracted_bytes=1 << 20)
        )
    assert info.value.reason == "too_many_files"
    assert not (tmp_path / "out2").exists()
    if fmt == "zip":
        return
    with_dot = _write_tar(
        tmp_path / "three_dot.tar", [{"name": "./", "type": tarfile.DIRTYPE}, *entries]
    )
    report_dot = safe_extract(
        with_dot, tmp_path / "out3", ExtractLimits(max_files=3, max_extracted_bytes=1 << 20)
    )
    assert report_dot.entry_count == 3


def test_existing_dest_rules(tmp_path: Path) -> None:
    archive = _write_zip(tmp_path / "a.zip", [{"name": "a.txt", "data": b"x"}])
    empty = tmp_path / "empty"
    empty.mkdir()
    report = safe_extract(archive, empty, SMALL)
    assert report.dest == empty.absolute()
    nonempty = tmp_path / "nonempty"
    nonempty.mkdir()
    (nonempty / "keep.txt").write_bytes(b"keep")
    with pytest.raises(ValueError):
        safe_extract(archive, nonempty, SMALL)
    assert (nonempty / "keep.txt").read_bytes() == b"keep"
    existing_file = tmp_path / "afile"
    existing_file.write_bytes(b"data")
    with pytest.raises(ValueError):
        safe_extract(archive, existing_file, SMALL)
    assert existing_file.read_bytes() == b"data"


def test_unexpected_write_error_is_reraised(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = _write_zip(
        tmp_path / "two.zip",
        [{"name": "a.txt", "data": b"x"}, {"name": "b.txt", "data": b"y"}],
    )
    calls = {"count": 0}
    real = extract._copy_limited

    def flaky(src: Any, dst: Any, budget: int, member: str) -> int:
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError(28, "No space left on device")
        return real(src, dst, budget, member)

    monkeypatch.setattr(extract, "_copy_limited", flaky)
    dest = tmp_path / "out"
    with pytest.raises(OSError) as info:
        safe_extract(archive, dest, SMALL)
    assert info.value.errno == 28
    assert not dest.exists()


def test_root_rule(tmp_path: Path) -> None:
    single_file = _write_zip(tmp_path / "one.zip", [{"name": "only.txt", "data": b"x"}])
    report = safe_extract(single_file, tmp_path / "r1", SMALL)
    assert report.root == report.dest
    dir_and_hidden = _write_zip(
        tmp_path / "hidden.zip",
        [{"name": "d/x.txt", "data": b"x"}, {"name": ".DS_Store", "data": b""}],
    )
    report = safe_extract(dir_and_hidden, tmp_path / "r2", SMALL)
    assert report.root == report.dest
    single_dir = _write_zip(tmp_path / "dir.zip", [{"name": "d/x.txt", "data": b"x"}])
    report = safe_extract(single_dir, tmp_path / "r3", SMALL)
    assert report.root == report.dest / "d"


def test_api_validation() -> None:
    with pytest.raises(ValueError):
        UnsafeArchiveError("nope")
    with pytest.raises(ValueError):
        ExtractLimits(0, 1)
    with pytest.raises(ValueError):
        ExtractLimits(1, -1)
    with pytest.raises(ValueError):
        ExtractLimits(True, 1)
    assert extract.REJECT_REASONS == (
        "dotdot",
        "absolute",
        "symlink",
        "hardlink",
        "device",
        "duplicate",
        "too_many_files",
        "too_large",
        "bad_name",
        "unknown_format",
    )


def test_zip_entry_count_checked_per_entry_without_end_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(extract, "_zip_declared_entries", lambda path: None)
    archive = vbf.make_malicious_archive(tmp_path / "many.zip", "too_many_files", fmt="zip")
    dest = tmp_path / "out"
    with pytest.raises(UnsafeArchiveError) as info:
        safe_extract(archive, dest, SMALL)
    assert info.value.reason == "too_many_files"
    assert info.value.member == "many/000099.txt"  # ok.txt + many/000000..098 fill 100
    assert not dest.exists()


def test_repeated_root_entry_is_duplicate(tmp_path: Path) -> None:
    archive = _write_tar(
        tmp_path / "dots.tar",
        [
            {"name": "./", "type": tarfile.DIRTYPE},
            {"name": "a.txt", "data": b"x"},
            {"name": ".", "type": tarfile.DIRTYPE},
        ],
    )
    dest = tmp_path / "out"
    with pytest.raises(UnsafeArchiveError) as info:
        safe_extract(archive, dest, SMALL)
    assert info.value.reason == "duplicate"
    assert not dest.exists()


def test_missing_archive_is_not_an_unsafe_archive(tmp_path: Path) -> None:
    dest = tmp_path / "out"
    with pytest.raises(FileNotFoundError):
        safe_extract(tmp_path / "missing.zip", dest, SMALL)
    assert not dest.exists()
