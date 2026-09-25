"""Safe path containment and bounded loaders for external data.

This is the only viewer module allowed to call ``np.load`` or import ``xml.etree``: every path
that comes from stored data must first go through :func:`resolve_inside`, and every array/XML
read is size-checked before any data is materialised.
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePath
from typing import Any

import defusedxml
import defusedxml.ElementTree
import numpy as np

from plateau_rt.viewer.settings import DEFAULT_MAX_ARRAY_BYTES

DEFAULT_MAX_XML_BYTES = 256 * 1024 * 1024
NPY_HEADER_SLACK_BYTES = 65536
MAX_NPZ_MEMBERS = 4096

_DRIVE_RE = re.compile(r"[A-Za-z]:")

# The element type returned by parse_xml (other modules must not import xml.etree themselves).
XmlElement = xml.etree.ElementTree.Element


class UnsafePathError(ValueError):
    """A path is absolute, contains ``..`` or leaves the allowed root."""


class UnsafeArrayError(ValueError):
    """An ``.npy`` / ``.npz`` file was rejected (size, dtype or format)."""


class UnsafeXmlError(ValueError):
    """An XML file was rejected (entities, DTD, external refs, malformed or too large)."""


@dataclass(frozen=True)
class NpyInfo:
    """The declared header of one ``.npy`` array."""

    shape: tuple[int, ...]
    dtype: np.dtype
    fortran_order: bool
    nbytes: int


def resolve_inside(root: Path | str, relpath: str | PurePath) -> Path:
    """Resolve ``relpath`` inside ``root``; reject absolute, ``..`` or escaping paths."""
    text = str(relpath)
    if "\x00" in text:
        raise UnsafePathError(f"path contains NUL: {text!r}")
    if (
        os.path.isabs(text)
        or text.startswith("/")
        or text.startswith("\\")
        or _DRIVE_RE.match(text) is not None
    ):
        raise UnsafePathError(f"path is absolute: {text!r}")
    for segment in text.replace("\\", "/").split("/"):
        if segment == "..":
            raise UnsafePathError(f"path leaves the root: {text!r}")
    root_real = os.path.realpath(root)
    candidate = os.path.realpath(os.path.join(root, text))
    if candidate != root_real and not candidate.startswith(root_real + os.sep):
        raise UnsafePathError(f"path leaves the root: {text!r}")
    return Path(candidate)


def _read_header(fileobj: Any) -> tuple[NpyInfo, int]:
    """Read a ``.npy`` magic and header, returning ``(NpyInfo, header length)``."""
    try:
        major, minor = np.lib.format.read_magic(fileobj)
    except Exception as exc:
        raise UnsafeArrayError(f"not a valid .npy header: {exc}") from exc
    if (major, minor) not in ((1, 0), (2, 0)):
        raise UnsafeArrayError(f"unsupported .npy version {major}.{minor}")
    try:
        if major == 1:
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(fileobj)
        else:
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(fileobj)
    except Exception as exc:
        raise UnsafeArrayError(f"not a valid .npy header: {exc}") from exc
    nbytes = 1
    for dim in shape:
        nbytes *= int(dim)
    nbytes *= int(dtype.itemsize)
    info = NpyInfo(
        shape=tuple(int(dim) for dim in shape),
        dtype=dtype,
        fortran_order=bool(fortran_order),
        nbytes=nbytes,
    )
    return info, int(fileobj.tell())


def read_npy_header(path: Path | str) -> NpyInfo:
    """Read and validate only the ``.npy`` header of ``path``."""
    with open(path, "rb") as handle:
        info, _ = _read_header(handle)
    return info


def load_npy(
    path: Path | str,
    *,
    max_bytes: int = DEFAULT_MAX_ARRAY_BYTES,
    mmap: bool = False,
) -> np.ndarray:
    """Size-check a ``.npy`` header, then load it (optionally memory-mapped)."""
    with open(path, "rb") as handle:
        info, header_len = _read_header(handle)
    if info.dtype.hasobject:
        raise UnsafeArrayError(f"object dtype {info.dtype} is not allowed")
    if info.nbytes > max_bytes:
        raise UnsafeArrayError(
            f"array shape {info.shape} dtype {info.dtype} declares {info.nbytes} bytes "
            f"> max_bytes {max_bytes}"
        )
    file_size = os.path.getsize(path)
    if header_len + info.nbytes != file_size:
        raise UnsafeArrayError(
            f"size mismatch: header {header_len} + nbytes {info.nbytes} != file size {file_size}"
        )
    return np.load(path, allow_pickle=False, mmap_mode="r" if mmap else None)


def check_npz(
    path: Path | str,
    *,
    max_member_bytes: int = DEFAULT_MAX_ARRAY_BYTES,
) -> dict[str, NpyInfo]:
    """Validate every member of an ``.npz`` archive without reading array data."""
    try:
        archive = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        raise UnsafeArrayError(f"not a valid zip archive: {exc}") from exc
    with archive:
        infos = archive.infolist()
        if len(infos) > MAX_NPZ_MEMBERS:
            raise UnsafeArrayError(f"too many npz members: {len(infos)}")
        result: dict[str, NpyInfo] = {}
        for info in infos:
            name = info.filename
            if not name.endswith(".npy") or "/" in name:
                raise UnsafeArrayError(f"invalid npz member name {name!r}")
            if info.flag_bits & 0x1:
                raise UnsafeArrayError(f"encrypted npz member {name!r}")
            if info.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                raise UnsafeArrayError(f"unsupported compression for npz member {name!r}")
            if info.file_size > max_member_bytes + NPY_HEADER_SLACK_BYTES:
                raise UnsafeArrayError(
                    f"member {name!r} file_size {info.file_size} exceeds "
                    f"max_member_bytes {max_member_bytes}"
                )
            with archive.open(info) as member:
                member_info, header_len = _read_header(member)
            if member_info.dtype.hasobject:
                raise UnsafeArrayError(f"member {name!r} has object dtype {member_info.dtype}")
            if member_info.nbytes > max_member_bytes:
                raise UnsafeArrayError(
                    f"member {name!r} shape {member_info.shape} dtype {member_info.dtype} "
                    f"declares {member_info.nbytes} bytes > max_member_bytes {max_member_bytes}"
                )
            if header_len + member_info.nbytes != info.file_size:
                raise UnsafeArrayError(
                    f"member {name!r} size mismatch: header {header_len} + "
                    f"nbytes {member_info.nbytes} != file size {info.file_size}"
                )
            result[name[:-4]] = member_info
    return result


def load_npz(
    path: Path | str,
    keys: Sequence[str] | None = None,
    *,
    max_member_bytes: int = DEFAULT_MAX_ARRAY_BYTES,
) -> dict[str, np.ndarray]:
    """Validate an ``.npz`` archive and fully load the requested members."""
    infos = check_npz(path, max_member_bytes=max_member_bytes)
    selected = list(infos) if keys is None else list(keys)
    result: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as data:
        for key in selected:
            if key not in infos:
                raise KeyError(key)
            result[key] = data[key]
    return result


def parse_xml(
    path: Path | str,
    *,
    max_bytes: int = DEFAULT_MAX_XML_BYTES,
) -> xml.etree.ElementTree.Element:
    """Size-check then parse XML with DTDs, entities and external refs forbidden."""
    size = os.path.getsize(path)
    if size > max_bytes:
        raise UnsafeXmlError(f"{path}: {size} bytes exceeds {max_bytes}")
    try:
        tree = defusedxml.ElementTree.parse(
            path, forbid_dtd=True, forbid_entities=True, forbid_external=True
        )
    except defusedxml.DefusedXmlException as exc:
        raise UnsafeXmlError(f"{path}: unsafe XML: {exc}") from exc
    except xml.etree.ElementTree.ParseError as exc:
        raise UnsafeXmlError(f"{path}: malformed XML: {exc}") from exc
    root = tree.getroot()
    if root is None:
        raise UnsafeXmlError(f"{path}: empty XML document")
    return root


def read_bytes(path: Path | str, *, max_bytes: int) -> bytes:
    """Return the bytes of ``path`` after checking its size against ``max_bytes``."""
    size = os.path.getsize(path)
    if size > max_bytes:
        raise ValueError(f"{path}: {size} bytes exceeds {max_bytes}")
    with open(path, "rb") as handle:
        return handle.read()
