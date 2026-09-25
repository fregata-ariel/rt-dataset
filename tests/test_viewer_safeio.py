"""Tests for the viewer safe loaders: path containment, npy/npz/XML bounds."""

from __future__ import annotations

import ast
import io
import os
import zipfile
from pathlib import Path

import numpy as np
import pytest

import plateau_rt.viewer
import plateau_rt.viewer.safeio as safeio
from plateau_rt.viewer.safeio import (
    UnsafeArrayError,
    UnsafePathError,
    UnsafeXmlError,
    check_npz,
    load_npy,
    load_npz,
    parse_xml,
    read_bytes,
    read_npy_header,
    resolve_inside,
)


def test_resolve_inside_relative_and_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    (root / "a").mkdir(parents=True)
    assert resolve_inside(root, "a/b.npy") == Path(os.path.realpath(root / "a/b.npy"))
    assert resolve_inside(root, "") == Path(os.path.realpath(root))
    assert resolve_inside(root, ".") == Path(os.path.realpath(root))


@pytest.mark.parametrize(
    "bad",
    ["/etc/passwd", "\\x", "C:x", "../x", "a/../../x", "a/../b", "a\\..\\b", "a\x00b"],
)
def test_resolve_inside_rejects(tmp_path: Path, bad: str) -> None:
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(UnsafePathError):
        resolve_inside(root, bad)


def test_resolve_inside_rejects_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "file").write_bytes(b"x")
    os.symlink(outside, root / "link")
    with pytest.raises(UnsafePathError):
        resolve_inside(root, "link/file")


def test_resolve_inside_accepts_symlink_inside(tmp_path: Path) -> None:
    root = tmp_path / "root"
    inner = root / "inner"
    inner.mkdir(parents=True)
    (inner / "file").write_bytes(b"x")
    os.symlink(inner, root / "link")
    assert resolve_inside(root, "link/file") == Path(os.path.realpath(inner / "file"))


def test_load_npy_values_and_header(tmp_path: Path) -> None:
    array = np.arange(6, dtype=np.float32).reshape(2, 3)
    path = tmp_path / "a.npy"
    np.save(path, array)
    info = read_npy_header(path)
    assert info.shape == (2, 3)
    assert info.dtype == np.dtype("<f4")
    assert info.nbytes == 24
    loaded = load_npy(path)
    assert loaded.dtype == np.float32
    np.testing.assert_array_equal(loaded, array)


def test_load_npy_mmap(tmp_path: Path) -> None:
    array = np.arange(10, dtype=np.float64)
    path = tmp_path / "a.npy"
    np.save(path, array)
    loaded = load_npy(path, mmap=True)
    assert isinstance(loaded, np.memmap)
    np.testing.assert_array_equal(loaded, array)


def test_load_npy_rejects_oversized_declared_header(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "big.npy"
    with open(path, "wb") as handle:
        np.lib.format.write_array_header_1_0(
            handle, {"shape": (100000, 100000), "fortran_order": False, "descr": "<f8"}
        )
        handle.write(b"\x00" * 16)
    calls = {"count": 0}

    def fake_load(*args: object, **kwargs: object) -> object:
        calls["count"] += 1
        raise AssertionError("np.load must not be called")

    monkeypatch.setattr(safeio.np, "load", fake_load)
    with pytest.raises(UnsafeArrayError) as excinfo:
        load_npy(path)
    assert "(100000, 100000)" in str(excinfo.value)
    assert calls["count"] == 0


def test_load_npy_rejects_object_dtype(tmp_path: Path) -> None:
    path = tmp_path / "obj.npy"
    np.save(path, np.array([{"a": 1}], dtype=object), allow_pickle=True)
    with pytest.raises(UnsafeArrayError) as excinfo:
        load_npy(path)
    assert "object" in str(excinfo.value)


def test_load_npy_respects_max_bytes(tmp_path: Path) -> None:
    array = np.zeros(100, dtype=np.float32)
    path = tmp_path / "a.npy"
    np.save(path, array)
    size = read_npy_header(path).nbytes
    with pytest.raises(UnsafeArrayError):
        load_npy(path, max_bytes=size - 1)
    loaded = load_npy(path, max_bytes=size)
    np.testing.assert_array_equal(loaded, array)


def test_load_npy_rejects_truncated(tmp_path: Path) -> None:
    array = np.arange(100, dtype=np.float32)
    path = tmp_path / "a.npy"
    np.save(path, array)
    data = path.read_bytes()
    path.write_bytes(data[:-4])
    with pytest.raises(UnsafeArrayError) as excinfo:
        load_npy(path)
    assert "size mismatch" in str(excinfo.value)


def test_load_npy_rejects_bad_version_and_non_npy(tmp_path: Path) -> None:
    version = tmp_path / "v3.npy"
    version.write_bytes(b"\x93NUMPY\x03\x00" + b"\x00" * 16)
    with pytest.raises(UnsafeArrayError):
        load_npy(version)
    other = tmp_path / "not.npy"
    other.write_bytes(b"hello world")
    with pytest.raises(UnsafeArrayError):
        load_npy(other)


def _fake_npy(shape: tuple[int, ...], data: bytes = b"") -> bytes:
    """Return a valid .npy magic+header declaring ``shape`` plus ``data``."""
    buffer = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        buffer, {"shape": shape, "fortran_order": False, "descr": "<f8"}
    )
    buffer.write(data)
    return buffer.getvalue()


def test_check_npz_and_load_npz_roundtrip(tmp_path: Path) -> None:
    a = np.arange(6, dtype=np.int32).reshape(2, 3)
    b = np.array([1.5], dtype=np.float32)
    stored = tmp_path / "a.npz"
    np.savez(stored, a=a, b=b)
    infos = check_npz(stored)
    assert infos["a"].shape == (2, 3)
    assert infos["a"].dtype == np.int32
    assert infos["b"].dtype == np.float32
    loaded = load_npz(stored)
    np.testing.assert_array_equal(loaded["a"], a)
    np.testing.assert_array_equal(loaded["b"], b)
    compressed = tmp_path / "c.npz"
    np.savez_compressed(compressed, a=np.zeros((10, 10), dtype=np.uint8))
    assert check_npz(compressed)["a"].shape == (10, 10)


def test_check_npz_rejects_big_declared_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "big.npz"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as archive:
        archive.writestr("a.npy", _fake_npy((100000, 100000), b"\x00" * 16))

    def fake_load(*args: object, **kwargs: object) -> object:
        raise AssertionError("np.load must not be called")

    monkeypatch.setattr(safeio.np, "load", fake_load)
    with pytest.raises(UnsafeArrayError) as excinfo:
        check_npz(path)
    assert "(100000, 100000)" in str(excinfo.value)


def test_check_npz_rejects_object_member(tmp_path: Path) -> None:
    path = tmp_path / "obj.npz"
    np.savez(path, a=np.array([1, 2], dtype=object))
    with pytest.raises(UnsafeArrayError) as excinfo:
        check_npz(path)
    assert "object" in str(excinfo.value)


def test_check_npz_rejects_by_file_size_before_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "zeros.npz"
    np.savez_compressed(path, a=np.zeros(2 * 1024 * 1024, dtype=np.uint8))

    def fail_open(*args: object, **kwargs: object) -> object:
        raise AssertionError("member must not be opened")

    monkeypatch.setattr(zipfile.ZipFile, "open", fail_open)
    with pytest.raises(UnsafeArrayError) as excinfo:
        check_npz(path, max_member_bytes=1024 * 1024)
    assert "a.npy" in str(excinfo.value)


def test_check_npz_rejects_non_npy_member(tmp_path: Path) -> None:
    path = tmp_path / "bad.npz"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as archive:
        archive.writestr("a.txt", b"nope")
    with pytest.raises(UnsafeArrayError):
        check_npz(path)


def test_load_npz_keys(tmp_path: Path) -> None:
    stored = tmp_path / "a.npz"
    np.savez(stored, a=np.array([1], dtype=np.int32), b=np.array([2], dtype=np.int32))
    selected = load_npz(stored, ["a"])
    assert set(selected) == {"a"}
    with pytest.raises(KeyError):
        load_npz(stored, ["zz"])


def test_parse_xml_normal(tmp_path: Path) -> None:
    path = tmp_path / "scene.xml"
    path.write_text(
        '<scene version="3.0.0"><string name="filename" value="box.ply"/></scene>',
        encoding="utf-8",
    )
    root = parse_xml(path)
    assert root.tag == "scene"
    child = root.find("string")
    assert child is not None
    assert child.get("value") == "box.ply"


def test_parse_xml_billion_laughs(tmp_path: Path) -> None:
    path = tmp_path / "laughs.xml"
    path.write_text(
        '<?xml version="1.0"?>\n'
        "<!DOCTYPE lolz [\n"
        '  <!ENTITY lol "lol">\n'
        '  <!ENTITY lol2 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">\n'
        '  <!ENTITY lol3 "&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;&lol2;">\n'
        "]>\n"
        "<lolz>&lol3;</lolz>\n",
        encoding="utf-8",
    )
    with pytest.raises(UnsafeXmlError):
        parse_xml(path)


def test_parse_xml_external_entity(tmp_path: Path) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("TOP_SECRET_MARKER_XYZ", encoding="utf-8")
    path = tmp_path / "xxe.xml"
    path.write_text(
        '<?xml version="1.0"?>\n'
        '<!DOCTYPE foo [<!ENTITY xxe SYSTEM "file://' + str(secret) + '">]>\n'
        "<foo>&xxe;</foo>\n",
        encoding="utf-8",
    )
    with pytest.raises(UnsafeXmlError) as excinfo:
        parse_xml(path)
    assert "TOP_SECRET_MARKER_XYZ" not in str(excinfo.value)


def test_parse_xml_malformed_and_too_large(tmp_path: Path) -> None:
    malformed = tmp_path / "bad.xml"
    malformed.write_text("<scene><unclosed></scene>", encoding="utf-8")
    with pytest.raises(UnsafeXmlError):
        parse_xml(malformed)
    large = tmp_path / "large.xml"
    large.write_text("<scene/>", encoding="utf-8")
    with pytest.raises(UnsafeXmlError):
        parse_xml(large, max_bytes=2)


def test_read_bytes_bound(tmp_path: Path) -> None:
    path = tmp_path / "data.bin"
    path.write_bytes(b"abcdef")
    assert read_bytes(path, max_bytes=6) == b"abcdef"
    with pytest.raises(ValueError):
        read_bytes(path, max_bytes=5)


def forbidden_uses(source: str) -> list[str]:
    """Return forbidden numpy/xml/lxml uses found in Python ``source``."""
    tree = ast.parse(source)
    problems: list[str] = []
    numpy_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if name == "numpy" or name.startswith("numpy."):
                    numpy_names.add(alias.asname or name.split(".")[0])
                if name == "lxml" or name.startswith("lxml."):
                    problems.append(f"import {name}")
                if name == "xml" or name.startswith("xml."):
                    problems.append(f"import {name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "lxml" or module.startswith("lxml."):
                problems.append(f"from {module} import ...")
            if module == "xml" or module.startswith("xml."):
                problems.append(f"from {module} import ...")
            if module == "numpy":
                for alias in node.names:
                    if alias.name == "load":
                        problems.append("from numpy import load")
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        chain = _attribute_chain(node)
        if chain is None:
            continue
        if (
            node.attr == "load"
            and isinstance(node.value, ast.Name)
            and node.value.id in numpy_names
        ):
            problems.append(f"{node.value.id}.load")
        if any(chain[i : i + 2] == ["xml", "etree"] for i in range(len(chain) - 1)):
            problems.append(".".join(chain))
        if (
            len(chain) >= 4
            and chain[0] in numpy_names
            and chain[1:3] == ["lib", "format"]
            and chain[3].startswith("read_array")
        ):
            problems.append(".".join(chain))
    return problems


def _attribute_chain(node: ast.Attribute) -> list[str] | None:
    """Return the dotted name of an attribute access, or None."""
    parts: list[str] = []
    current: ast.expr = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if not isinstance(current, ast.Name):
        return None
    parts.append(current.id)
    parts.reverse()
    return parts


def test_forbidden_uses_detects_and_ignores() -> None:
    assert forbidden_uses("import numpy as xp\nxp.load('a')") == ["xp.load"]
    assert forbidden_uses("import numpy\nnumpy.load('a')") == ["numpy.load"]
    assert forbidden_uses("from numpy import load") == ["from numpy import load"]
    assert forbidden_uses("import xml.etree.ElementTree as ET") == ["import xml.etree.ElementTree"]
    assert forbidden_uses("from xml.etree import ElementTree") == ["from xml.etree import ..."]
    assert forbidden_uses("from xml.dom import minidom") == ["from xml.dom import ..."]
    sneaky = forbidden_uses("from plateau_rt.viewer import safeio\nsafeio.xml.etree.parse('a')")
    assert "safeio.xml.etree.parse" in sneaky
    assert forbidden_uses("import numpy as np\nnp.save('a', 1)\njson.load(f)") == []


def test_viewer_modules_avoid_forbidden_uses() -> None:
    viewer_dir = Path(plateau_rt.viewer.__file__).resolve().parent
    offenders: dict[str, list[str]] = {}
    for path in sorted(viewer_dir.rglob("*.py")):
        if path == viewer_dir / "safeio.py":
            continue
        bad = forbidden_uses(path.read_text(encoding="utf-8"))
        if bad:
            offenders[str(path.relative_to(viewer_dir))] = bad
    assert offenders == {}


def test_safeio_is_caught_by_the_guard() -> None:
    safety = Path(plateau_rt.viewer.__file__).resolve().parent / "safeio.py"
    reports = forbidden_uses(safety.read_text(encoding="utf-8"))
    assert any("load" in report for report in reports)
