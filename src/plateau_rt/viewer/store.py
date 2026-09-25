"""Content-addressed bundle store and SQLite index (Sionna-free).

Layout documented in ``docs/viewer.md`` ("Store format").
"""

from __future__ import annotations

import contextlib
import datetime
import errno
import fcntl
import hashlib
import json
import os
import re
import shutil
import sqlite3
import stat
import unicodedata
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from plateau_rt.viewer import VIEWER_VERSION
from plateau_rt.viewer.extract import (
    DIR_MODE,
    FILE_MODE,
    ExtractLimits,
    UnsafeArchiveError,
    _copy_limited,
    _pick_root,
)
from plateau_rt.viewer.settings import ViewerSettings

STORE_SCHEMA_VERSION = 1
BUNDLE_STATUS_READY = "ready"
RAW_FILE_MODE = 0o444
RAW_DIR_MODE = 0o555
_DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")
_LOCK_TIMEOUT_S = 30.0
_CHUNK_BYTES = 1 << 20


class StoreVersionError(RuntimeError):
    """The store's schema is newer than this viewer (or unreadable); not opened."""

    def __init__(
        self, found: int | None, supported: int, written_by: str | None, path: Path
    ) -> None:
        """Record version details and build the error message."""
        self.found = found
        self.supported = supported
        self.written_by = written_by
        self.path = path
        if found is None:
            message = f"store {path} has no readable store_schema_version"
        else:
            message = (
                f"store {path} has schema version {found}; it was created by viewer "
                f"{written_by or 'unknown'} or later, "
                f"this viewer ({VIEWER_VERSION}) supports up to {supported}"
            )
        super().__init__(message)


@dataclass(frozen=True)
class FileRecord:
    """One content-addressed file inside a bundle."""

    digest: str
    relpath: str
    size: int
    sha256: str


@dataclass(frozen=True)
class BundleRecord:
    """One stored bundle with its index metadata."""

    digest: str
    name: str
    members: tuple[dict[str, Any], ...]
    created_at: str
    total_bytes: int
    file_count: int
    status: str
    error: str | None
    validated_with: str | None


@dataclass(frozen=True)
class StoreUsage:
    """Byte counts for the store layout and free disk space."""

    raw_bytes: int
    derived_bytes: int
    staging_bytes: int
    index_bytes: int
    free_bytes: int


def _normalize_digest(value: str) -> str:
    """Lowercase ``value`` and require 64 hex chars, else raise ValueError."""
    lowered = value.lower()
    if _DIGEST_RE.match(lowered) is None:
        raise ValueError(f"invalid digest {value!r}")
    return lowered


def _make_writable_and_remove(path: Path) -> None:
    """Chmod a tree writable (no symlink following) and remove it."""
    if not os.path.lexists(path):
        return
    if os.path.islink(path):
        os.unlink(path)
        return
    _chmod_tree(path, 0o755, 0o644)
    shutil.rmtree(path, ignore_errors=False)


def _chmod_tree(root: Path | str, dir_mode: int, file_mode: int) -> None:
    """Chmod dirs/files below ``root`` without following symlinks."""
    stack: list[str] = [os.fspath(root)]
    while stack:
        current = stack.pop()
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode):
            continue
        if stat.S_ISDIR(info.st_mode):
            try:
                os.chmod(current, dir_mode)
            except FileNotFoundError:
                continue
            try:
                with os.scandir(current) as it:
                    children = [entry.path for entry in it]
            except FileNotFoundError:
                continue
            stack.extend(children)
        elif stat.S_ISREG(info.st_mode):
            try:
                os.chmod(current, file_mode)
            except FileNotFoundError:
                continue


def _tree_bytes(root: Path | str) -> int:
    """Sum ``st_size`` of regular files below ``root`` without following symlinks."""
    total = 0
    stack: list[str] = [os.fspath(root)]
    while stack:
        current = stack.pop()
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode):
            continue
        if stat.S_ISDIR(info.st_mode):
            try:
                with os.scandir(current) as it:
                    children = [entry.path for entry in it]
            except FileNotFoundError:
                continue
            stack.extend(children)
        elif stat.S_ISREG(info.st_mode):
            total += info.st_size
    return total


def compute_digest(root: Path) -> tuple[str, list[FileRecord]]:
    """Hash every regular file below ``root`` into a content digest."""
    base = Path(root)
    if os.path.islink(base) or not base.is_dir():
        raise ValueError(f"not a directory: {root}")
    base_str = os.fspath(base.absolute())
    collected: list[tuple[str, int, str]] = []
    stack: list[tuple[str, str]] = [(base_str, "")]
    while stack:
        dir_abs, prefix = stack.pop()
        with os.scandir(dir_abs) as it:
            entries = list(it)
        for entry in entries:
            name = entry.name
            rel = f"{prefix}/{name}" if prefix else name
            try:
                rel.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise ValueError(f"unencodable name: {rel!r}") from exc
            if entry.is_symlink():
                raise ValueError(f"symlink not allowed: {rel}")
            if entry.is_dir(follow_symlinks=False):
                stack.append((entry.path, rel))
            elif entry.is_file(follow_symlinks=False):
                info = entry.stat(follow_symlinks=False)
                digest_h = hashlib.sha256()
                with open(entry.path, "rb") as handle:
                    while True:
                        chunk = handle.read(_CHUNK_BYTES)
                        if not chunk:
                            break
                        digest_h.update(chunk)
                collected.append((rel, info.st_size, digest_h.hexdigest()))
            else:
                raise ValueError(f"non-regular entry: {rel}")
    collected.sort(key=lambda item: item[0].encode("utf-8"))
    text = "".join(f"{rel}\t{size}\t{sha}\n" for rel, size, sha in collected)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    records = [
        FileRecord(digest=digest, relpath=rel, size=size, sha256=sha)
        for rel, size, sha in collected
    ]
    return digest, records


def _migrate_to_1(conn: sqlite3.Connection, data_dir: Path) -> None:
    """Create the version-1 schema."""
    conn.execute("CREATE TABLE store_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute(
        "CREATE TABLE bundles (digest TEXT PRIMARY KEY, name TEXT NOT NULL, "
        "members_json TEXT NOT NULL, created_at TEXT NOT NULL, "
        "total_bytes INTEGER NOT NULL, file_count INTEGER NOT NULL, "
        "status TEXT NOT NULL, error TEXT, validated_with TEXT)"
    )
    conn.execute(
        "CREATE TABLE files (digest TEXT NOT NULL, relpath TEXT NOT NULL, "
        "size INTEGER NOT NULL, sha256 TEXT NOT NULL, PRIMARY KEY (digest, relpath))"
    )
    conn.execute("CREATE INDEX files_sha256 ON files (sha256)")
    conn.execute(
        "CREATE TABLE derived (digest TEXT NOT NULL, member TEXT NOT NULL, "
        "deriver TEXT NOT NULL, version INTEGER NOT NULL, params_key TEXT NOT NULL, "
        "links_key TEXT NOT NULL, status TEXT NOT NULL, error TEXT, "
        "updated_at TEXT NOT NULL, "
        "PRIMARY KEY (digest, member, deriver, version, params_key, links_key))"
    )
    conn.execute(
        "CREATE TABLE jobs (job_id TEXT PRIMARY KEY, kind TEXT NOT NULL, digest TEXT, "
        "member TEXT, deriver TEXT, params_key TEXT, status TEXT NOT NULL, stage TEXT, "
        "done_bytes INTEGER, total_bytes INTEGER, error TEXT, "
        "created_at TEXT NOT NULL, updated_at TEXT NOT NULL)"
    )
    conn.execute("CREATE INDEX jobs_digest ON jobs (digest)")
    conn.execute("CREATE INDEX jobs_status ON jobs (status)")


MIGRATIONS: dict[int, Callable[[sqlite3.Connection, Path], None]] = {1: _migrate_to_1}


def _read_schema_version(conn: sqlite3.Connection, data_dir: Path, target: int) -> int:
    """Return the stored schema version (0 when no store_meta table)."""
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'store_meta'"
    ).fetchone()
    if row is None:
        return 0
    value_row = conn.execute(
        "SELECT value FROM store_meta WHERE key = 'store_schema_version'"
    ).fetchone()
    if value_row is None:
        raise StoreVersionError(None, target, None, data_dir)
    value = value_row[0]
    if not isinstance(value, str) or re.fullmatch(r"[0-9]+", value) is None:
        raise StoreVersionError(None, target, None, data_dir)
    return int(value)


class Store:
    """Content-addressed bundle store with an SQLite index."""

    settings: ViewerSettings
    data_dir: Path
    staging_dir: Path
    bundles_dir: Path
    index_path: Path
    lock_path: Path
    schema_version: int

    def __init__(self, settings: ViewerSettings) -> None:
        """Open (creating and migrating) the store under ``settings.data_dir``."""
        self.settings = settings
        self.data_dir = Path(os.path.normpath(Path(settings.data_dir).absolute()))
        self.staging_dir = self.data_dir / "staging"
        self.bundles_dir = self.data_dir / "bundles"
        self.index_path = self.data_dir / "index.sqlite"
        self.lock_path = self.data_dir / "store.lock"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        self.bundles_dir.mkdir(parents=True, exist_ok=True)
        target = max(MIGRATIONS)
        with self._lock():
            with self.connect() as conn:
                mode_row = conn.execute("PRAGMA journal_mode=WAL").fetchone()
                if mode_row is None or str(mode_row[0]).lower() != "wal":
                    raise RuntimeError(f"cannot enable WAL mode on {self.index_path}")
                version = _read_schema_version(conn, self.data_dir, target)
                if version > target:
                    by_row = conn.execute(
                        "SELECT value FROM store_meta WHERE key = 'migrated_by_viewer'"
                    ).fetchone()
                    written = by_row[0] if by_row is not None else None
                    raise StoreVersionError(version, target, written, self.data_dir)
                for v in range(version + 1, target + 1):
                    conn.execute("BEGIN IMMEDIATE")
                    try:
                        MIGRATIONS[v](conn, self.data_dir)
                        conn.execute(
                            "INSERT INTO store_meta(key, value) VALUES (?, ?) "
                            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                            ("store_schema_version", str(v)),
                        )
                        conn.execute(
                            "INSERT INTO store_meta(key, value) VALUES (?, ?) "
                            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                            ("migrated_by_viewer", VIEWER_VERSION),
                        )
                        conn.execute(
                            "INSERT OR IGNORE INTO store_meta(key, value) VALUES (?, ?)",
                            ("created_by_viewer", VIEWER_VERSION),
                        )
                        conn.execute("COMMIT")
                    except BaseException:
                        try:
                            conn.execute("ROLLBACK")
                        except sqlite3.Error:
                            pass
                        raise
        self.schema_version = target

    @contextlib.contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        """Yield a fresh autocommit SQLite connection to the index."""
        conn = sqlite3.connect(str(self.index_path), timeout=_LOCK_TIMEOUT_S, isolation_level=None)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA busy_timeout = 30000")
            yield conn
        finally:
            conn.close()

    @contextlib.contextmanager
    def _lock(self) -> Iterator[None]:
        """Hold an exclusive flock on ``store.lock`` with a fresh fd."""
        fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield None
        finally:
            os.close(fd)

    def new_staging(self) -> Path:
        """Create and return an empty staging directory."""
        while True:
            path = self.staging_dir / uuid.uuid4().hex
            try:
                os.mkdir(path, 0o755)
                return path
            except FileExistsError:
                continue

    def discard_staging(self, path: Path) -> None:
        """Remove the top-level staging dir containing ``path``."""
        _, top = self._staging_paths(path)
        if not os.path.lexists(top):
            return
        _make_writable_and_remove(top)

    def _staging_paths(self, path: Path) -> tuple[Path, Path]:
        """Return ``(normalised path, top-level staging dir)``; path must be strictly inside."""
        norm = Path(os.path.normpath(Path(path).absolute()))
        try:
            rel = norm.relative_to(self.staging_dir)
        except ValueError as exc:
            raise ValueError(f"path is not inside staging: {path}") from exc
        if not rel.parts:
            raise ValueError(f"path is not inside staging: {path}")
        # No symlink anywhere on the way: the real path must be the lexical one.
        expected = os.path.join(os.path.realpath(self.staging_dir), *rel.parts)
        if os.path.realpath(norm) != expected:
            raise ValueError(f"path goes through a symlink: {path}")
        return norm, self.staging_dir / rel.parts[0]

    def stage_from_directory(self, src: Path, limits: ExtractLimits) -> Path:
        """Copy a server directory into staging and return its bundle root."""
        src_path = Path(src)
        if os.path.islink(src_path) or not src_path.is_dir():
            raise ValueError(f"not a directory: {src}")
        staging = self.new_staging()
        try:
            self._copy_server_tree(src_path.absolute(), staging.absolute(), limits)
        except BaseException:
            _make_writable_and_remove(staging)
            raise
        return _pick_root(staging)

    def _copy_server_tree(self, src_abs: Path, staging_abs: Path, limits: ExtractLimits) -> int:
        """Copy ``src_abs`` into ``staging_abs`` enforcing extraction limits."""
        total = 0
        count = 0
        stack: list[tuple[str, str, str]] = [(os.fspath(src_abs), os.fspath(staging_abs), "")]
        while stack:
            src_dir, dst_dir, prefix = stack.pop()
            with os.scandir(src_dir) as it:
                entries = sorted(list(it), key=lambda e: e.name)
            seen: set[str] = set()
            for entry in entries:
                name = entry.name
                try:
                    name.encode("utf-8")
                except UnicodeEncodeError as exc:
                    rel_bad = f"{prefix}/{name}" if prefix else name
                    raise UnsafeArchiveError("bad_name", rel_bad) from exc
                rel = f"{prefix}/{name}" if prefix else name
                key = unicodedata.normalize("NFC", name).casefold()
                if key in seen:
                    raise UnsafeArchiveError("duplicate", rel)
                seen.add(key)
                if entry.is_symlink():
                    raise UnsafeArchiveError("symlink", rel)
                if entry.is_dir(follow_symlinks=False):
                    count += 1
                    if count > limits.max_files:
                        raise UnsafeArchiveError("too_many_files", rel)
                    dst = os.path.join(dst_dir, name)
                    os.mkdir(dst, DIR_MODE)
                    os.chmod(dst, DIR_MODE)
                    stack.append((entry.path, dst, rel))
                elif entry.is_file(follow_symlinks=False):
                    count += 1
                    if count > limits.max_files:
                        raise UnsafeArchiveError("too_many_files", rel)
                    dst = os.path.join(dst_dir, name)
                    try:
                        src_fd = os.open(entry.path, os.O_RDONLY | os.O_NOFOLLOW)
                    except OSError as exc:
                        if exc.errno == errno.ELOOP:
                            raise UnsafeArchiveError("symlink", rel) from exc
                        raise
                    with os.fdopen(src_fd, "rb") as src_f:
                        info = os.fstat(src_f.fileno())
                        if not stat.S_ISREG(info.st_mode):
                            raise UnsafeArchiveError("device", rel)
                        dst_fd = os.open(dst, os.O_WRONLY | os.O_CREAT | os.O_EXCL, FILE_MODE)
                        with os.fdopen(dst_fd, "wb") as dst_f:
                            written = _copy_limited(
                                src_f,
                                dst_f,  # type: ignore[arg-type]
                                limits.max_extracted_bytes - total,
                                rel,
                            )
                        os.chmod(dst, FILE_MODE)
                    total += written
                else:
                    raise UnsafeArchiveError("device", rel)
        return total

    def commit(
        self,
        staging: Path,
        *,
        name: str,
        members: Sequence[Mapping[str, Any]],
        validated_with: str | None = None,
    ) -> tuple[str, bool]:
        """Store a staged bundle root and index it."""
        absolute, top = self._staging_paths(staging)
        if not absolute.is_dir():
            raise ValueError(f"not a directory: {staging}")
        digest, records = compute_digest(absolute)
        if not records:
            raise ValueError("empty bundle")
        members_json = json.dumps(
            list(members), sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )
        total_bytes = sum(record.size for record in records)
        with self._lock():
            with self.connect() as conn:
                existing = conn.execute(
                    "SELECT digest FROM bundles WHERE digest = ?", (digest,)
                ).fetchone()
                if existing is not None:
                    self.discard_staging(absolute)
                    return digest, False
                bundle_path = self.bundles_dir / digest
                if os.path.lexists(bundle_path):
                    _make_writable_and_remove(bundle_path)
                created_at = datetime.datetime.now(datetime.timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%S.%fZ"
                )
                self._freeze_tree(absolute)
                os.mkdir(bundle_path, 0o755)
                try:
                    os.mkdir(bundle_path / "derived", 0o755)
                    raw_path = bundle_path / "raw"
                    os.rename(absolute, raw_path)
                    os.chmod(raw_path, RAW_DIR_MODE)
                    conn.execute("BEGIN IMMEDIATE")
                    conn.execute(
                        "INSERT INTO bundles(digest, name, members_json, created_at, "
                        "total_bytes, file_count, status, error, validated_with) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            digest,
                            name,
                            members_json,
                            created_at,
                            total_bytes,
                            len(records),
                            BUNDLE_STATUS_READY,
                            None,
                            validated_with,
                        ),
                    )
                    conn.executemany(
                        "INSERT INTO files(digest, relpath, size, sha256) VALUES (?, ?, ?, ?)",
                        [
                            (record.digest, record.relpath, record.size, record.sha256)
                            for record in records
                        ],
                    )
                    conn.execute("COMMIT")
                except BaseException:
                    try:
                        if conn.in_transaction:
                            conn.execute("ROLLBACK")
                    except sqlite3.Error:
                        pass
                    _make_writable_and_remove(bundle_path)
                    raise
                if os.path.lexists(top):
                    _make_writable_and_remove(top)
                return digest, True

    def _freeze_tree(self, staging: Path) -> None:
        """Chmod staged files/dirs read-only (except ``staging`` itself)."""
        base = os.fspath(staging)
        dirs: list[str] = []
        stack: list[str] = [base]
        while stack:
            current = stack.pop()
            with os.scandir(current) as it:
                children = list(it)
            for entry in children:
                if entry.is_symlink():
                    continue
                if entry.is_dir(follow_symlinks=False):
                    dirs.append(entry.path)
                    stack.append(entry.path)
                elif entry.is_file(follow_symlinks=False):
                    os.chmod(entry.path, RAW_FILE_MODE)
        for directory in dirs:
            os.chmod(directory, RAW_DIR_MODE)

    def bundle_dir(self, digest: str) -> Path:
        """Return the bundle directory for ``digest``."""
        return self.bundles_dir / _normalize_digest(digest)

    def raw_dir(self, digest: str) -> Path:
        """Return the raw directory for ``digest``."""
        return self.bundles_dir / _normalize_digest(digest) / "raw"

    def derived_dir(self, digest: str) -> Path:
        """Return the derived directory for ``digest``."""
        return self.bundles_dir / _normalize_digest(digest) / "derived"

    def get(self, digest: str) -> BundleRecord | None:
        """Return the bundle row for ``digest``, or None."""
        key = _normalize_digest(digest)
        with self.connect() as conn:
            row = conn.execute(
                "SELECT digest, name, members_json, created_at, total_bytes, "
                "file_count, status, error, validated_with FROM bundles WHERE digest = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        return BundleRecord(
            digest=row["digest"],
            name=row["name"],
            members=tuple(json.loads(row["members_json"])),
            created_at=row["created_at"],
            total_bytes=row["total_bytes"],
            file_count=row["file_count"],
            status=row["status"],
            error=row["error"],
            validated_with=row["validated_with"],
        )

    def list(self) -> list[BundleRecord]:
        """List all bundles ordered by creation time."""
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT digest, name, members_json, created_at, total_bytes, "
                "file_count, status, error, validated_with FROM bundles "
                "ORDER BY created_at, digest"
            ).fetchall()
        result: list[BundleRecord] = []
        for row in rows:
            result.append(
                BundleRecord(
                    digest=row["digest"],
                    name=row["name"],
                    members=tuple(json.loads(row["members_json"])),
                    created_at=row["created_at"],
                    total_bytes=row["total_bytes"],
                    file_count=row["file_count"],
                    status=row["status"],
                    error=row["error"],
                    validated_with=row["validated_with"],
                )
            )
        return result

    def find_by_file_sha256(self, sha256: str) -> list[FileRecord]:
        """Return every file row with the given sha256."""
        key = _normalize_digest(sha256)
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT digest, relpath, size, sha256 FROM files "
                "WHERE sha256 = ? ORDER BY digest, relpath",
                (key,),
            ).fetchall()
        found: list[FileRecord] = []
        for row in rows:
            found.append(
                FileRecord(
                    digest=row["digest"],
                    relpath=row["relpath"],
                    size=row["size"],
                    sha256=row["sha256"],
                )
            )
        return found

    def delete(self, digest: str) -> bool:
        """Delete a bundle's rows and directory."""
        key = _normalize_digest(digest)
        with self._lock():
            with self.connect() as conn:
                row = conn.execute("SELECT digest FROM bundles WHERE digest = ?", (key,)).fetchone()
                bundle_path = self.bundles_dir / key
                if row is None and not os.path.lexists(bundle_path):
                    return False
                conn.execute("BEGIN IMMEDIATE")
                try:
                    conn.execute("DELETE FROM bundles WHERE digest = ?", (key,))
                    conn.execute("DELETE FROM files WHERE digest = ?", (key,))
                    conn.execute("DELETE FROM derived WHERE digest = ?", (key,))
                    conn.execute("DELETE FROM jobs WHERE digest = ?", (key,))
                    conn.execute("COMMIT")
                except BaseException:
                    try:
                        if conn.in_transaction:
                            conn.execute("ROLLBACK")
                    except sqlite3.Error:
                        pass
                    raise
                if os.path.lexists(bundle_path):
                    target = self.staging_dir / ("deleting-" + uuid.uuid4().hex)
                    os.rename(bundle_path, target)
                    _chmod_tree(target, 0o755, 0o644)
                    shutil.rmtree(target, ignore_errors=False)
                return True

    def disk_usage(self) -> StoreUsage:
        """Measure raw, derived, staging, index and free bytes."""
        raw_bytes = 0
        derived_bytes = 0
        if self.bundles_dir.is_dir() and not os.path.islink(self.bundles_dir):
            try:
                with os.scandir(self.bundles_dir) as it:
                    children = [entry.path for entry in it]
            except FileNotFoundError:
                children = []
            for child in children:
                if os.path.islink(child) or not os.path.isdir(child):
                    continue
                raw_bytes += _tree_bytes(os.path.join(child, "raw"))
                derived_bytes += _tree_bytes(os.path.join(child, "derived"))
        staging_bytes = _tree_bytes(self.staging_dir) if os.path.lexists(self.staging_dir) else 0
        index_bytes = 0
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(str(self.index_path) + suffix)
            try:
                info = os.lstat(candidate)
            except FileNotFoundError:
                continue
            if stat.S_ISREG(info.st_mode):
                index_bytes += info.st_size
        free_bytes = shutil.disk_usage(self.data_dir).free
        return StoreUsage(
            raw_bytes=raw_bytes,
            derived_bytes=derived_bytes,
            staging_bytes=staging_bytes,
            index_bytes=index_bytes,
            free_bytes=free_bytes,
        )
