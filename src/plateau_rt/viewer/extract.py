"""Safe archive extraction for viewer bundles (Sionna-free).

Only member streams are read (``TarFile.extractfile`` / ``ZipFile.open``) and every output file is
written by this module, so a malicious archive can never escape ``dest``.
"""

from __future__ import annotations

import contextlib
import gzip
import os
import re
import shutil
import stat
import tarfile
import unicodedata
import zipfile
import zlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, cast

ARCHIVE_FORMATS: tuple[str, ...] = ("zip", "tar", "tar.gz")
REJECT_REASONS: tuple[str, ...] = (
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
FILE_MODE = 0o644
DIR_MODE = 0o755

_CHUNK_BYTES = 1 << 20
_USTAR_OFFSET = 257
_USTAR_MARKER = b"ustar"
_MAX_COMPONENT_BYTES = 255
_MAX_PATH_BYTES = 4096
_DRIVE_RE = re.compile(r"^[A-Za-z]:")
_READ_ERRORS: tuple[type[BaseException], ...] = (
    tarfile.TarError,
    zipfile.BadZipFile,
    zlib.error,
    gzip.BadGzipFile,
    EOFError,
    NotImplementedError,
    RuntimeError,
)


class UnsafeArchiveError(Exception):
    """An archive was rejected; ``reason`` is one of REJECT_REASONS."""

    def __init__(self, reason: str, member: str | None = None) -> None:
        if reason not in REJECT_REASONS:
            raise ValueError(f"unknown reject reason {reason!r}")
        self.reason = reason
        self.member = member
        if member is None:
            message = f"unsafe archive ({reason})"
        else:
            message = f"unsafe archive ({reason}): {member!r}"
        super().__init__(message)


@dataclass(frozen=True)
class ExtractLimits:
    """Hard caps enforced while extracting an archive."""

    max_files: int
    max_extracted_bytes: int

    def __post_init__(self) -> None:
        for name, value in (
            ("max_files", self.max_files),
            ("max_extracted_bytes", self.max_extracted_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")


@dataclass(frozen=True)
class ExtractReport:
    """Outcome of a successful :func:`safe_extract`."""

    format: str
    dest: Path
    root: Path
    files: tuple[str, ...]
    total_bytes: int
    entry_count: int


def detect_format(path: Path) -> str:
    """Return "zip", "tar" or "tar.gz" from the leading bytes; raise on anything else."""
    with open(path, "rb") as handle:
        head = handle.read(512)
    if head[:4] in (b"PK\x03\x04", b"PK\x05\x06"):
        return "zip"
    if head[:2] == b"\x1f\x8b":
        try:
            with gzip.open(path, "rb") as gz:
                block = gz.read(512)
        except (OSError, EOFError, zlib.error) as exc:
            raise UnsafeArchiveError("unknown_format", None) from exc
        if len(block) >= _USTAR_OFFSET + len(_USTAR_MARKER) and (
            block[_USTAR_OFFSET : _USTAR_OFFSET + 5] == _USTAR_MARKER
        ):
            return "tar.gz"
        raise UnsafeArchiveError("unknown_format", None)
    if len(head) >= _USTAR_OFFSET + len(_USTAR_MARKER) and (
        head[_USTAR_OFFSET : _USTAR_OFFSET + 5] == _USTAR_MARKER
    ):
        return "tar"
    raise UnsafeArchiveError("unknown_format", None)


def safe_extract(archive_path: Path, dest: Path, limits: ExtractLimits) -> ExtractReport:
    """Extract ``archive_path`` into ``dest`` safely; on rejection remove ``dest`` and raise."""
    archive_path = Path(archive_path)
    fmt = detect_format(archive_path)
    dest = Path(dest).absolute()
    adopted = False
    if os.path.lexists(dest):
        if os.path.islink(dest) or not os.path.isdir(dest):
            raise ValueError(f"destination {dest} is not a directory")
        if any(dest.iterdir()):
            raise ValueError(f"destination {dest} is not empty")
        adopted = True
    try:
        if not adopted:
            os.mkdir(dest, DIR_MODE)
        return _extract(archive_path, dest, limits, fmt)
    except BaseException:
        shutil.rmtree(dest, ignore_errors=True)
        raise


def _copy_limited(src: BinaryIO, dst: BinaryIO, budget: int, member: str) -> int:
    """Copy ``src`` to ``dst`` writing at most ``budget`` bytes; return bytes written."""
    written = 0
    while True:
        try:
            chunk = src.read(min(_CHUNK_BYTES, budget - written + 1))
        except _READ_ERRORS as exc:
            raise UnsafeArchiveError("unknown_format", member) from exc
        if not chunk:
            return written
        if written + len(chunk) > budget:
            raise UnsafeArchiveError("too_large", member)
        dst.write(chunk)
        written += len(chunk)


def _extract(archive_path: Path, dest: Path, limits: ExtractLimits, fmt: str) -> ExtractReport:
    """Run the checks and writes for an already-detected archive."""
    index = _PathIndex()
    files: list[str] = []
    entry_count = 0
    total_bytes = 0
    seen_root_entry = False
    entries = _iter_entries(archive_path, fmt, limits.max_files)
    with contextlib.closing(entries):
        for raw, kind, opener in entries:
            _check_name(raw)
            if _is_absolute(raw):
                raise UnsafeArchiveError("absolute", raw)
            if _has_dotdot(raw):
                raise UnsafeArchiveError("dotdot", raw)
            normalized = _normalize(raw)
            if not normalized:
                if kind != "dir":
                    raise UnsafeArchiveError("bad_name", raw)
                if seen_root_entry:  # a second "./" entry: uncounted entries must not repeat
                    raise UnsafeArchiveError("duplicate", raw)
                seen_root_entry = True
                continue
            if kind in ("symlink", "hardlink", "device"):
                raise UnsafeArchiveError(kind, raw)
            index.check(normalized, kind, raw)
            entry_count += 1
            if entry_count > limits.max_files:
                raise UnsafeArchiveError("too_many_files", raw)
            target = dest / normalized
            norm_target = os.path.normpath(str(target))
            if os.path.commonpath([str(dest), norm_target]) != str(dest):
                raise UnsafeArchiveError("dotdot", raw)
            current = dest
            for part in normalized.split("/")[:-1]:
                current = current / part
                _ensure_dir(current, raw)
            if kind == "dir":
                _ensure_dir(target, raw)
            else:
                total_bytes += _write_file(
                    target, opener, limits.max_extracted_bytes - total_bytes, raw
                )
                files.append(normalized)
            index.add(normalized, kind)
    root = _pick_root(dest)
    return ExtractReport(
        format=fmt,
        dest=dest,
        root=root,
        files=tuple(sorted(files)),
        total_bytes=total_bytes,
        entry_count=entry_count,
    )


def _write_file(target: Path, opener: Callable[[], BinaryIO | None], budget: int, raw: str) -> int:
    """Open ``target`` exclusively and copy the member stream into it."""
    try:
        fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, FILE_MODE)
    except FileExistsError as exc:
        raise UnsafeArchiveError("duplicate", raw) from exc
    with os.fdopen(fd, "wb") as dst:  # owns and closes fd exactly once
        try:
            src = opener()
        except _READ_ERRORS as exc:
            raise UnsafeArchiveError("unknown_format", raw) from exc
        if src is None:
            raise UnsafeArchiveError("unknown_format", raw)
        with contextlib.closing(src):
            written = _copy_limited(src, dst, budget, raw)
        os.fchmod(dst.fileno(), FILE_MODE)
    return written


def _pick_root(dest: Path) -> Path:
    """Return the bundle root: descend at most once into a single top-level directory."""
    entries = list(dest.iterdir())
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return dest


def _ensure_dir(path: Path, member: str) -> None:
    """Create ``path`` (mode DIR_MODE) if missing; reject an existing non-directory."""
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        os.mkdir(path, DIR_MODE)
        os.chmod(path, DIR_MODE)
        return
    if not stat.S_ISDIR(info.st_mode):
        raise UnsafeArchiveError("duplicate", member)


def _zip_declared_entries(archive_path: Path) -> int | None:
    """Return the entry count from the zip end-of-central-directory record, if readable."""
    end_record = getattr(zipfile, "_EndRecData", None)
    if end_record is None:
        return None
    try:
        with open(archive_path, "rb") as handle:
            record = end_record(handle)
    except _READ_ERRORS + (OSError,):
        return None
    return None if record is None else int(record[zipfile._ECD_ENTRIES_TOTAL])


def _iter_entries(
    archive_path: Path, fmt: str, max_files: int
) -> Iterator[tuple[str, str, Callable[[], BinaryIO | None]]]:
    """Yield ``(raw_name, kind, opener)`` for every entry, in archive order."""
    if fmt == "zip":
        # ZipFile loads the whole central directory into memory: bound it before opening.
        # The per-entry count below still applies (this record is only a cheap early check).
        declared = _zip_declared_entries(archive_path)
        if declared is not None and declared > max_files + 1:
            raise UnsafeArchiveError("too_many_files", None)
        try:
            archive = zipfile.ZipFile(archive_path)
        except (zipfile.BadZipFile, OSError) as exc:
            raise UnsafeArchiveError("unknown_format", None) from exc
        with archive:
            try:
                infos = archive.infolist()
            except _READ_ERRORS as exc:
                raise UnsafeArchiveError("unknown_format", None) from exc
            for info in infos:
                raw = info.orig_filename
                if info.flag_bits & 0x1:
                    raise UnsafeArchiveError("unknown_format", raw)
                kind = _classify_zip(info)

                def zip_opener(info: zipfile.ZipInfo = info) -> BinaryIO:
                    return cast(BinaryIO, archive.open(info))

                yield raw, kind, zip_opener
        return
    mode = "r|gz" if fmt == "tar.gz" else "r|"
    try:
        tar = tarfile.open(archive_path, mode=mode)
    except _READ_ERRORS as exc:
        raise UnsafeArchiveError("unknown_format", None) from exc
    with tar:
        while True:
            try:
                member = tar.next()
            except _READ_ERRORS as exc:
                raise UnsafeArchiveError("unknown_format", None) from exc
            if member is None:
                return
            kind = _classify_tar(member)

            def tar_opener(member: tarfile.TarInfo = member) -> BinaryIO | None:
                stream = tar.extractfile(member)
                return None if stream is None else cast(BinaryIO, stream)

            yield member.name, kind, tar_opener


def _classify_tar(member: tarfile.TarInfo) -> str:
    """Map a tar member type to file / dir / symlink / hardlink / device."""
    if member.isreg():
        return "file"
    if member.isdir():
        return "dir"
    if member.issym():
        return "symlink"
    if member.islnk():
        return "hardlink"
    return "device"


def _classify_zip(info: zipfile.ZipInfo) -> str:
    """Map a zip entry (Unix external attributes) to file / dir / symlink / device."""
    if info.is_dir():
        return "dir"
    mode = (info.external_attr >> 16) if info.create_system == 3 else 0
    if stat.S_ISLNK(mode):
        return "symlink"
    if stat.S_ISCHR(mode) or stat.S_ISBLK(mode) or stat.S_ISFIFO(mode) or stat.S_ISSOCK(mode):
        return "device"
    if stat.S_ISDIR(mode):
        return "dir"
    return "file"


def _check_name(name: str) -> None:
    """Reject empty, backslash/NUL, non-UTF-8 or over-long entry names."""
    if not name or "\\" in name or "\x00" in name:
        raise UnsafeArchiveError("bad_name", name)
    try:
        name.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise UnsafeArchiveError("bad_name", name) from exc
    parts = [part for part in name.split("/") if part not in ("", ".")]
    if any(len(part.encode("utf-8")) > _MAX_COMPONENT_BYTES for part in parts):
        raise UnsafeArchiveError("bad_name", name)
    if len("/".join(parts).encode("utf-8")) > _MAX_PATH_BYTES:
        raise UnsafeArchiveError("bad_name", name)


def _is_absolute(name: str) -> bool:
    """Return True for POSIX-absolute or Windows drive-prefixed names."""
    return name.startswith("/") or _DRIVE_RE.match(name) is not None


def _has_dotdot(name: str) -> bool:
    """Return True when any ``/``-separated component is ``..``."""
    return any(part == ".." for part in name.split("/"))


def _normalize(name: str) -> str:
    """Drop empty and ``.`` components and join the rest with ``/``."""
    return "/".join(part for part in name.split("/") if part not in ("", "."))


def _path_key(path: str) -> str:
    """Return the case-insensitive, NFC-normalized key of a path."""
    return unicodedata.normalize("NFC", path).casefold()


def _ancestors(path: str) -> list[str]:
    """Return every proper ancestor path of ``path``, nearest first."""
    parts = path.split("/")
    return ["/".join(parts[:index]) for index in range(1, len(parts))]


class _PathIndex:
    """Track accepted paths (explicit and implicit) to reject duplicates early."""

    def __init__(self) -> None:
        self._explicit: set[str] = set()
        self._paths: dict[str, tuple[str, str]] = {}

    def check(self, path: str, kind: str, member: str) -> None:
        """Raise ``duplicate`` when ``path`` collides with a registered path."""
        key = _path_key(path)
        if key in self._explicit:
            raise UnsafeArchiveError("duplicate", member)
        existing = self._paths.get(key)
        if existing is not None and existing[0] != path:
            raise UnsafeArchiveError("duplicate", member)
        if kind == "file" and existing is not None and existing[1] == "dir":
            raise UnsafeArchiveError("duplicate", member)
        for ancestor in _ancestors(path):
            parent = self._paths.get(_path_key(ancestor))
            if parent is None:
                continue
            if parent[1] == "file" or parent[0] != ancestor:
                raise UnsafeArchiveError("duplicate", member)

    def add(self, path: str, kind: str) -> None:
        """Register ``path`` and its implicit parent directories."""
        self._explicit.add(_path_key(path))
        self._paths[_path_key(path)] = (path, kind)
        for ancestor in _ancestors(path):
            self._paths.setdefault(_path_key(ancestor), (ancestor, "dir"))
